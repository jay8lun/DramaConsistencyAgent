"""CLI 入口：B 通道全自动流水线（第 5 章）。

**2026-10-31：C2 浏览器接管通道已整体移除。** 原因是平台侧权限，不是代码缺陷：
点「生成」会被「升级会员即可解锁 · Seedance 2.0 mini 和 720P」的订阅墙拦住，
积分从未被扣过一次，生成态从未出现。完整代码留在 `_archive_c2_2026-10-31/`。

当前实现到 **S3**：S0 项目壳 → S1 剧本结构化 → S2 定妆照 + 自检 → S3 首帧 + 判定 + 定向重取。
S4（图生视频）/ S5（合成）/ S6（交付）在 M1-S3d 补上——**不在这里留假实现**，
未实现的阶段会明确打印，不会静默跳过（9.4 第 15 条：禁止虚构数据）。

用法：
    python run.py --mock --project-id demo01
    python run.py --mock --project-id demo01 --force --creative "深夜电梯里的第三个人"
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from config import ensure_dirs, settings
from models import Character, Scene, Script
from pipeline import phase as P
from pipeline import prompt_builder as PB
from pipeline import ref_desc as RD
from pipeline import report as R
from pipeline import s1_script as S1
from pipeline import s2_anchor as S2
from pipeline import s3_keyframe as S3
from pipeline import s4_video as S4
from pipeline import s5_compose as S5
from pipeline.cost import CostLedger
from pipeline.project import ProjectCtx, create_or_load, write_json
from pipeline.state import StateStore
from providers.mock import MockImage, MockJudge, MockText, MockVideo

#: 尚未实现的阶段。列出来是为了让「跑到哪一步停了」在终端上可见。
#: 当前 S0~S6 全部打通，此列表为空；保留常量是为了后续加阶段时不必改打印逻辑。
PENDING_STAGES: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="run.py",
        description="DramaConsistencyAgent 流水线入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # `--project` 是 `--project-id` 的别名：7.6 的运行示例写的是
    # `python run.py --project <id>`，只留 --project-id 会让照着文档抄的人报错，
    # 而这种错看着像"参数写错了"，实际是文档与实现不一致。
    ap.add_argument("--project-id", "--project", dest="project_id", default=None,
                    help="项目 ID（字母/数字/下划线/短横，≤32 字符）。"
                         "**不给则启动 Web 服务**（容器入口）。")
    ap.add_argument("--mock", action="store_true",
                    help="使用本地占位供应商（零成本、秒级），用于验证链路而不花钱")
    ap.add_argument("--allow-paid", action="store_true",
                    help="显式授权付费调用。mock 模式自动授权（成本为 0）；"
                         "接真实 provider 后必须手动加此参数，否则每次调用都会抛 PaidCallBlocked")
    ap.add_argument("--budget", type=float, default=None,
                    help=f"本次运行的预算上限（元），默认 BUDGET_CAP={settings.budget_cap}")
    # 2026-10-31：C2（浏览器接管）通道整体移除，只保留 B 通道。
    # 参数仍然保留但只接受 api，让照旧命令抄的人看到一句能懂的说明，
    # 而不是 argparse 的 "invalid choice: 'web'"。
    ap.add_argument("--channel", choices=("api",), default="api",
                    help="只有 api 一种（B 通道全自动）。C2 浏览器通道已于 2026-10-31 移除")
    ap.add_argument("--mode", choices=("api", "web"), default=None,
                    help="运行模式：api = B 通道全自动（容器内可跑）。"
                         "web 已于 2026-10-31 移除，传入会明确报错")
    ap.add_argument("--force", action="store_true",
                    help="已存在项目时重建 project.json（会覆盖 created_at，谨慎）")
    ap.add_argument("--creative", default="深夜加班回家的女人，在电梯里发现楼层按钮多按了一层。",
                    help="创意文本（S1 的输入）")
    ap.add_argument("--shots", type=int, default=6, help="镜头数，默认 6（20~30 秒短剧）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只做供应商装配检查 + 预算预估，不发起任何调用（不花钱）")
    # ---- 视频阶段的人工确认（2026-10-29 新增）----
    # 末帧判定不达标的镜头停在 awaiting_review，不自动重跑：
    # 一条 10 秒视频约 ¥2，而「脸还能不能看」这件事人看一眼只要 5 秒。
    ap.add_argument("--review", action="store_true",
                    help="列出待人工确认的镜头（只读，不改状态）")
    ap.add_argument("--decide", metavar="SHOT_ID", default=None,
                    help="裁决一个待确认镜头，必须配合 --accept / --degrade / --reject 之一")
    ap.add_argument("--accept", action="store_true", help="放行（记 done）")
    ap.add_argument("--degrade", action="store_true", help="放行但标红（记 degraded，报告里列出）")
    ap.add_argument("--reject", action="store_true", help="驳回，下次跑流水线时重出这一条（约 ¥2）")
    ap.add_argument("--note", default="", help="裁决理由，写进状态日志与报告")

    # ---- 分步确认（2026-10-31）：三个停点 ----
    # 默认行为从"一口气跑完"改成"跑到停点就停"。这不是加了限制，是补上原本
    # 缺失的语义：S1 出完让人改剧本、S2a 出完让人看脸，这两件事都发生在
    # 花钱的大头（六镜视频）之前，停下来才是最省的。
    ap.add_argument("--stop-after", choices=("s1", "s2a"), default=None,
                    help="跑到哪个阶段就停（默认停在第一个未确认的停点）")
    ap.add_argument("--confirm", action="store_true",
                    help="确认当前停点，继续跑下一段。停点不确认不会往下走")
    ap.add_argument("--auto", action="store_true",
                    help="跳过所有停点，一路跑到成片（老行为；只在确信剧本和脸都没问题时用）")
    ap.add_argument("--role-source", choices=("brief", "image"), default=None,
                    help="角色从哪来：brief=程序按创意生成 / image=用 --ref 给的参考图")
    ap.add_argument("--ref", action="append", default=None, metavar="PATH",
                    help="角色参考图（可重复，按 char_01、char_02 顺序绑定）。用一次即视为 role-source=image")
    ap.add_argument("--regen-front", metavar="CHAR_ID", default=None,
                    help="只重出某个角色的正脸（停 2 不满意时用），配合 --feedback 说明哪里不对")
    ap.add_argument("--feedback", default="", metavar="TEXT",
                    help="配合 --regen-front：哪里不满意。拼进提示词，不覆盖原设定")
    ap.add_argument("--update-card", action="store_true",
                    help="配合 --feedback：把这条意见同时写进角色卡（会影响所有含该角色的镜头）")
    ap.add_argument("--rerun-shot", metavar="SHOT_ID", default=None,
                    help="只重跑某一镜（删首帧与视频、清判定、退回 pending）。"
                         "改了某一镜的分镜描述后必须用它，否则 S3 见 done 会直接跳过")
    ap.add_argument("--rerun-reason", default="", metavar="TEXT",
                    help="配合 --rerun-shot：这次为什么要重出。既留档到 reruns.json，"
                         "也作为修正指令拼进首帧提示词——只留档不生效等于原样再来一次")
    return ap


def build_providers(use_mock: bool):
    """按 `--mock` 装配四个供应商，返回 (text, image, video, judge)。

    **这是全项目唯一决定「花不花钱」的地方。** 之前这里是四处硬编码的 `MockImage()`，
    真实调用根本无从谈起；集中成一个函数后，换供应商只改这里 + `.env`。

    mock 链路的成本恒为 0（各 provider 的 `is_mock=True` 让 cost.py 不记账），
    真实链路每次调用都要 `allow_paid=True`，否则抛 PaidCallBlocked（9.4 护栏）。
    """
    if use_mock:
        return MockText(), MockImage(), MockVideo(), MockJudge()

    # 延迟导入：mock 模式下不 import httpx / openai 的客户端，
    # 让 `--mock` 在缺少依赖或断网时也能跑（14.7「测试必须离线可跑」）。
    from providers.dashscope_video import DashScopeVideo
    from providers.llm import ChatProvider
    from providers.vlm import VLMJudge

    # 出图两套实现，按 IMAGE_PROVIDER 选。**必须在装配处显式打印选了哪套**：
    # 两家的单价差 10 倍（¥0.20 vs ¥0.02），装错了预算预估会静默失真。
    if settings.image_provider == "relay":
        from providers.relay_image import RelayImage

        image = RelayImage()
    elif settings.image_provider == "dashscope":
        from providers.dashscope_image import DashScopeImage

        image = DashScopeImage()
    else:
        raise SystemExit(
            f"IMAGE_PROVIDER={settings.image_provider!r} 不认识，只支持 relay / dashscope"
        )

    return (
        ChatProvider(
            settings.text_api,
            name="chat-text",
            # 关思考：实测 193s→28s、9891→1299 token，产出无差别（见 config.py 注释）
            extra_body={"thinking": {"type": "disabled"}} if settings.text_disable_thinking else None,
        ),
        image,
        DashScopeVideo(),
        VLMJudge(settings.judge_vlm),
    )


def estimate_cost(shots: int, chars: int = 2) -> dict[str, float]:
    """按当前配置粗估一次运行的花费，用于在开跑前把上限摆到台面上。

    口径与 12.2 一致，但用的是 config 里的**实测单价**而不是计划书的旧估值：
      S2 定妆照 chars×3 张（正/侧/背各一张）
      S3 首帧 shots×2 张（平均 2 次尝试）
      S4 视频 shots×5 秒 ×1.2（**只留技术重试余量**）
      VLM 判定 chars×1（定妆照自检）+ shots×2（每镜 2 次）

    **视频余量为什么从 1.8 降到 1.2（2026-10-29 改）**：1.8 是「末帧不合格会自动重出」
    那个年代的口径——那种设计下每镜平均要出 1.8 条视频。现在末帧 <75 改为停在
    `awaiting_review` 等人裁决、**不自动重跑**（S4 规格），视频多花钱的唯一来源只剩
    技术失败重试（超时/5xx），实测概率很低。继续按 1.8 估会把六镜算成 ¥10.80 视频费，
    而真实一次过只要 ¥6.00——**过高的估值会让预算闸挡住本来跑得起的运行**，
    这不是保守，是失真。

    判定按「每镜 2 次」估，它同时覆盖两种真实情况：多角色镜头（2 人各判 1 次），
    或单角色镜头（1 次判定 + 1 次质量重试）。多角色且还要重试时会低估，
    但出图那一项的余量通常盖得住；真要跑 3 人以上同框的镜头，请调高 BUDGET_CAP。
    """
    n_img = chars * 3 + shots * 2
    n_sec = shots * 5 * 1.2
    n_judge = chars + shots * 2
    img = n_img * settings.price_image
    vid = n_sec * settings.price_video_per_sec
    judge = n_judge * settings.price_vlm_per_call
    return {"images": n_img, "image_cost": img, "seconds": n_sec, "video_cost": vid,
            "judges": n_judge, "judge_cost": judge, "total": img + vid + judge}


def stage_s0(args: argparse.Namespace) -> tuple[ProjectCtx, bool]:
    """S0 · 初始化：无外部调用，不会失败（第 6 章）。"""
    budget = settings.budget_cap if args.budget is None else args.budget
    return create_or_load(
        args.project_id, budget_cap=budget, channel=args.channel,
        mock=args.mock, force=args.force,
    )


def _load_characters(ctx: ProjectCtx) -> list[Character]:
    raw = json.loads(ctx.characters_file.read_text(encoding="utf-8"))
    return [Character(**c) for c in raw]


def stage_s1(ctx: ProjectCtx, args: argparse.Namespace, ledger: CostLedger,
             text_provider, judge_provider=None) -> Script:
    """S1b · 剧本结构化 + 图生文反推角色卡。

    已有 script.json 就复用——重跑一次要多花一次模型调用。

    **顺序：先读图，再写剧本**（2026-11-02 前移）。原来反推在 S1 之后，只补得上
    外貌字段，补不上已经写进分镜的称谓与动作——S1 不知道画面里是 3 头身卡通生物，
    照样会写「男生凑近女生」「修长的手指」。现在把图先读成事实喂给 S1，它才有
    可能照着这个人写戏。反推结果在 S1 之后仍然覆盖一遍外貌（命中缓存，不重复花钱），
    防止 S1 顺手改写。
    """
    refs = _resolve_refs(ctx)

    if ctx.script_file.exists() and not args.force:
        script = S1.load_script(ctx)
        print(f"[S1] 复用已有剧本：{script.title}（{len(script.shots)} 镜）")
        # 复用也要过一遍反推：停 1 才上传参考图是常见路径（先看剧本再决定给谁一张脸）。
        # 此时 script.json 已存在，只走复用分支的话，新上传的图永远不会被"读"进角色卡，
        # 而 S2a 会拿旧卡出图——用户会看到"我明明传了图，出来还是那个人"。
        # **这一支不读图喂 S1**：剧本已经写好了，读了也没有能改它的地方，纯属白花钱。
        return _apply_ref_cards(ctx, script, ledger=ledger, judge=judge_provider,
                                paid_ok=args.paid_ok)

    ref_obs: list[str] = []
    if refs and judge_provider is not None:
        cards = RD.observe(ctx, refs, judge=judge_provider, ledger=ledger,
                           allow_paid=args.paid_ok,
                           warn=lambda m: print(f"     ⚠ {m}", file=sys.stderr))
        ref_obs = RD.describe_lines(cards)
        got = sum(1 for c in cards if c.appearance)
        print(f"[S1] 先读参考图：{got}/{len(refs)} 张读到内容"
              f"（{'、'.join(sorted({c.subject_type for c in cards}))}），据此写剧本")

    result = S1.run(ctx, text_provider, creative=args.creative, shot_count=args.shots,
                    ledger=ledger, allow_paid=args.paid_ok,
                    ref_observations=ref_obs or None)
    script = result.script
    fix = f"，修复 {result.attempts - 1} 次" if result.repaired else ""
    print(f"[S1] 产出剧本：{script.title}（{len(script.shots)} 镜{fix}）")
    print(f"     角色 {len(script.characters)} 个 / 场景 {len(script.scenes)} 个")

    # 用户给了参考图 → 让角色卡的文字也来自图，而不是由模型凭空写一份再跟图打架。
    return _apply_ref_cards(ctx, script, ledger=ledger, judge=judge_provider,
                            paid_ok=args.paid_ok)


def _apply_ref_cards(ctx: ProjectCtx, script: Script, *, ledger: CostLedger,
                     judge, paid_ok: bool) -> Script:
    """有参考图就让角色卡的文字来自图（计划书 4.2 的图生文入口）。

    **幂等**：已是 `inferred` 的角色跳过（见 `describe_all(skip_inferred=...)`），
    所以「停 1 手改角色卡 → 再点继续」不会被缓存里的旧卡盖回去。
    换图后想重新反推，上传接口会把该角色的 source 重置回 brief。
    """
    refs = _resolve_refs(ctx)
    if not refs or judge is None:
        return script
    chars, notes = RD.describe_all(
        ctx, refs, list(script.characters), judge=judge, ledger=ledger,
        allow_paid=paid_ok,
        warn=lambda m: print(f"     ⚠ {m}", file=sys.stderr),
    )
    if not notes:
        return script
    print("[S1] 图生文反推（source=inferred）：角色卡按参考图改写")
    for n in notes:
        print(f"     {n}")
    script = script.model_copy(update={"characters": chars})
    # 两处都写：characters.json 是 S2/S3 的出图依据，script.json 是报告与停 1 的展示。
    # 只写一处的话，会出现"报告里一套设定、实际出图用另一套"，而且各自自洽。
    _write_character_list(ctx, chars)
    return script


def _write_character_list(ctx: ProjectCtx, chars: list[Character]) -> None:
    """整份角色列表写回两处（`characters.json` 与 `script.json`）。"""
    write_json(ctx.characters_file, [c.model_dump() for c in chars])
    if ctx.script_file.exists():
        data = json.loads(ctx.script_file.read_text(encoding="utf-8"))
        data["characters"] = [c.model_dump() for c in chars]
        write_json(ctx.script_file, data)


def stage_s2(ctx: ProjectCtx, script: Script, ledger: CostLedger, *,
             allow_paid: bool, image_provider, judge_provider) -> list[Character]:
    """S2 · 定妆照 + 自检。全部 ok 就复用，不再重复出图。"""
    status = S2.load_status(ctx)
    if status and all(v.get("status") == "ok" for v in status.values()) and ctx.characters_file.exists():
        chars = _load_characters(ctx)
        if all(c.anchor_path and Path(c.anchor_path).exists() for c in chars):
            print(f"[S2] 复用已有定妆照（{len(chars)} 角色，自检均 ok）")
            return chars

    chars, report = S2.run(
        ctx, list(script.characters), style=script.style,
        image_provider=image_provider, judge_provider=judge_provider,
        ledger=ledger, allow_paid=allow_paid,
    )
    degraded = [cid for cid, r in report.items() if r.status != "ok"]
    print(f"[S2] 三视图 {len(chars) * 3} 张，自检 "
          + ("全部通过" if not degraded else f"降级 {degraded}（后续判定结论要标不可信）"))
    return chars


def _resolve_refs(ctx: ProjectCtx) -> list[Path]:
    """把 `project.json` 里记的参考图相对路径解析成存在的绝对路径。

    **顺序就是绑定顺序**：ref_images[0] 给 char_01，ref_images[1] 给 char_02。
    不做"让模型自己判断这张图是谁"——猜错不报错，只会把 A 的长相长到 B 脸上，
    而三张定妆照各自看起来都正常。
    """
    return [p for p in (ctx.root / rel for rel in P.load(ctx).ref_images) if p.exists()]


def _write_character(ctx: ProjectCtx, char: Character) -> None:
    """把改过的角色卡写回两处：`characters.json` 与 `script.json`。

    两处都要写。只写一处的话，S3 从 `characters.json` 读锚点、报告从 `script.json`
    读描述，于是「报告里写的是新设定，实际出图用的是旧设定」——而且对得上号，
    因为两份文件各自都自洽。
    """
    chars = _load_characters(ctx)
    chars = [char if c.char_id == char.char_id else c for c in chars]
    write_json(ctx.characters_file, [c.model_dump() for c in chars])
    if ctx.script_file.exists():
        data = json.loads(ctx.script_file.read_text(encoding="utf-8"))
        data["characters"] = [c.model_dump() for c in chars]
        write_json(ctx.script_file, data)


def stage_s2a(ctx: ProjectCtx, script: Script, ledger: CostLedger, *,
              allow_paid: bool, image_provider) -> dict[str, Path]:
    """S2a · 每个角色只出**正脸一张**，出完停下来等人看。

    为什么不再一次出三视图（2026-10-31 改）：三视图一次性出的时候，
    没人评价过正脸本身——自检只比"侧背像不像正脸"，一张丑脸只要侧背照着画
    就能拿高分。拆成两段后，正脸单独停下来给人看，这个洞就补上了。
    """
    refs = _resolve_refs(ctx)
    chars = _load_characters(ctx) if ctx.characters_file.exists() else list(script.characters)
    print(f"[S2a] 出正脸（{len(chars)} 个角色，每人一张）"
          + (f"，参考图 {len(refs)} 张" if refs else "，无参考图（按角色卡文生图）"))
    out: dict[str, Path] = {}
    for i, char in enumerate(chars):
        ref = refs[i] if i < len(refs) else None
        front = S2.generate_front(
            ctx, char, style=script.style, image_provider=image_provider,
            ledger=ledger, allow_paid=allow_paid,
            references=[ref] if ref else None,
        )
        out[char.char_id] = front
        kb = front.stat().st_size // 1024 if front.exists() else 0
        print(f"     {char.char_id} {char.name:<8} ✓ {front.name}  {kb}KB"
              + (f"   ← 参考 {ref.name}" if ref else ""))
    return out


def stage_s2b(ctx: ProjectCtx, script: Script, ledger: CostLedger, *,
              allow_paid: bool, image_provider, judge_provider) -> list[Character]:
    """S2b · 出侧背 + 自检。**正脸已由人确认，这一段只重试侧背。**

    重试代价从这里开始不同：`S2.run()` 失败一轮要重出 3 张（含正脸），
    这里失败一轮只重出 2 张，而且不会把你刚认可的那张脸换掉。
    """
    chars = _load_characters(ctx)
    print("[S2b] 出侧背（以刚确认的那张正脸为参考）")
    updated, report = S2.run_views(
        ctx, chars, style=script.style, image_provider=image_provider,
        judge_provider=judge_provider, ledger=ledger, allow_paid=allow_paid,
    )
    for cid, r in report.items():
        flag = "✓" if r.status == "ok" else "△"
        print(f"     {cid}  {flag} {r.status:<9} 自检最低 {r.min_score} 分"
              f"（{r.attempts} 轮）" + (f"  ← {r.reason}" if r.reason else ""))
    return updated


def run_regen_front(ctx: ProjectCtx, args: argparse.Namespace, *,
                    ledger: CostLedger, image_provider) -> int:
    """停 2 的「重出」：带意见重出**一张**正脸，并作废该角色已出的侧背。"""
    script = S1.load_script(ctx)
    chars = _load_characters(ctx)
    target = next((c for c in chars if c.char_id == args.regen_front), None)
    if target is None:
        print(f"✗ 没有角色 {args.regen_front}；本项目有 "
              f"{[c.char_id for c in chars]}", file=sys.stderr)
        return 2

    fb = (args.feedback or "").strip()
    removed = S2.clear_views(ctx, target.char_id)
    if removed:
        print(f"     作废旧侧背 {len(removed)} 张：正脸一换，它们就属于另一个人了。"
              "不删的话三张图各自都正常，自检也可能过")

    if args.update_card and fb:
        target = target.model_copy(update={"appearance": f"{target.appearance}；{fb}"})
        _write_character(ctx, target)
        print(f"     已写入角色卡：appearance +「{fb}」")
        print("     ⚠ 这会影响所有含该角色的镜头；当前若是停 2 阶段，尚无镜头被作废")
    elif fb:
        print(f"     意见只作用于这一张：{fb}")

    refs = _resolve_refs(ctx)
    idx = [c.char_id for c in chars].index(target.char_id)
    ref = refs[idx] if idx < len(refs) else None
    front = S2.generate_front(
        ctx, target, style=script.style, image_provider=image_provider,
        ledger=ledger, allow_paid=args.paid_ok, feedback=fb,
        references=[ref] if ref else None,
    )
    # 重出之后又回到「等人看」——阶段绝不能留在 anchor_review 之外，
    # 否则下一次 --confirm 会以为这张脸已经确认过。
    P.save(ctx, phase="anchor_review", note=f"重出 {target.char_id} 正脸"
           + (f"：{fb}" if fb else ""))
    kb = front.stat().st_size // 1024 if front.exists() else 0
    print(f"[S2a] 已重出 {front.name}  {kb}KB   （第 "
          f"{S2.load_status(ctx).get(target.char_id, {}).get('front_attempts', '?')} 次）")
    print("     看完点「这张可以」再往下；侧背还没出，因为正脸刚换。")
    return 0


RERUNS_FILE = "reruns.json"


def read_reruns(ctx: ProjectCtx) -> list[dict]:
    """读重出记录。文件不存在就是空列表——不是错误。"""
    f = Path(ctx.root) / RERUNS_FILE
    if not f.exists():
        return []
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        # 记录坏了不能让界面打不开：这是"历史备注"，不是流水线依据。
        return []


def _append_rerun(ctx: ProjectCtx, shot_id: str, reason: str) -> None:
    """追加一条重出记录。**只追加不覆盖**：同一镜被重出三次要看得到三次的理由，
    只留最后一次就无法回答"为什么这一镜越修越偏"。"""
    rows = read_reruns(ctx)
    rows.append({"shot_id": shot_id, "reason": reason,
                 "ts": datetime.now().isoformat(timespec="seconds")})
    try:
        write_json(Path(ctx.root) / RERUNS_FILE, rows)
    except OSError as exc:
        # 留档失败不该让已经付过钱的重出白跑，但必须说清楚。
        print(f"     ⚠ 重出记录写盘失败（不影响本次出图）：{exc}", file=sys.stderr)


def run_single_shot(ctx: ProjectCtx, args: argparse.Namespace, *, ledger: CostLedger,
                    store: StateStore, image_provider, video_provider,
                    judge_provider) -> int:
    """只重跑一镜：首帧 → 判定 → 视频 → 末帧判定。不碰其他镜，不合成。

    **为什么需要它**（2026-11-02）：改动某一镜的分镜描述后，`S3.run` 见状态 `done`
    会直接跳过，改文字不会让画面变——人只看到"保存成功"，图还是旧的，且不报错。
    而整段重跑会把已经跑好的五镜再买一遍。
    """
    sid = args.rerun_shot
    reason = (getattr(args, "rerun_reason", "") or "").strip()
    script = S1.load_script(ctx)
    shot = next((s for s in script.shots if s.shot_id == sid), None)
    if shot is None:
        print(f"✗ 没有镜头 {sid}；本项目有 "
              f"{[s.shot_id for s in script.shots]}", file=sys.stderr)
        return 2
    chars = _load_characters(ctx)
    scenes = {s.scene_id: s for s in script.scenes}

    removed = []
    for p in (ctx.keyframe_of(sid), ctx.video_of(sid)):
        if p.exists():
            p.unlink()
            removed.append(p.name)
    store.clear_qa(ctx.project_id, sid)
    store.clear_video_qa(ctx.project_id, sid)
    store.reset_shot(ctx.project_id, sid, "单镜重跑：删产物、清判定、重新生成")
    print(f"[rerun] {sid}：删掉 {removed or '（无旧产物）'}，状态打回 pending")
    print(f"[rerun] 出场 {shot.character_ids}｜动作：{shot.action}")
    if reason:
        # 理由**同时做两件事**：留档（报告与界面能回答"这镜为什么重出过"），
        # 以及当作提示词增量发给模型。只留档不生效的话，重出等于原样再来一次，
        # 用户写的"背景太亮"就成了一段没人读的备注。
        _append_rerun(ctx, sid, reason)
        print(f"[rerun] 理由：{reason}（已作为修正指令拼进首帧提示词）")

    untrusted = S2.degraded_chars(ctx)
    kf = S3.run(ctx, shot, characters=chars, scene=scenes[shot.scene_id], style=script.style,
                image_provider=image_provider, judge_provider=judge_provider, store=store,
                ledger=ledger, allow_paid=args.paid_ok, untrusted_chars=untrusted,
                initial_delta=reason or None)
    print(f"[rerun] 首帧 {kf.status}  score={kf.score}")

    vid = S4.run(ctx, shot, characters=chars, provider=video_provider, store=store,
                 ledger=ledger, allow_paid=args.paid_ok, judge_provider=judge_provider)
    print(f"[rerun] 视频 {vid.status}  末帧={vid.video_score}")

    # 成片是拿旧素材合成的，重跑一镜之后它就过期了。**删掉**而不是留着：
    # 留着的话界面继续显示"已完成"，而那一镜的内容已经不是成片里的样子。
    final = Path(ctx.final_video)
    if final.exists():
        final.unlink()
        print(f"[rerun] 删掉旧成片 {final.name}（素材变了，需要重新合成）")
        if P.load(ctx).phase == "done":
            P.save(ctx, phase="running", note="单镜重跑，成片待重新合成")
    return 0


def stage_s3(
    ctx: ProjectCtx, script: Script, chars: list[Character],
    store: StateStore, ledger: CostLedger, *, allow_paid: bool,
    image_provider, judge_provider, should_stop=None,
) -> None:
    """S3 · 逐镜：首帧 → 判定 → 带增量重取。"""
    scenes: dict[str, Scene] = {s.scene_id: s for s in script.scenes}

    # 开跑前先扫一遍"动作里提到的人不在场上"。放在这里而不是逐镜拦：
    # 逐镜的话前几镜的钱已经花了，而且改完这一镜还得再跑一次才发现下一镜同样有坑。
    conflicts = PB.cast_conflicts(script)
    if conflicts:
        print("✗ 有镜头在动作里写了场上没有的人，模型会替他凭空造一个（且不受参考图约束）：",
              file=sys.stderr)
        for line in conflicts:
            print(f"     {line}", file=sys.stderr)
        print("   改法：把「男生/女孩」这类称谓换成角色名，或把该角色加进这一镜的 "
              "character_ids。改完再跑（本步未产生任何费用）。", file=sys.stderr)
        raise PB.CastConflict(f"{len(conflicts)} 镜的动作与出场角色不一致")

    store.ensure_shots(ctx.project_id, [s.shot_id for s in script.shots])

    # 启动时重整一次：上次被 kill 留下的中间态要按文件系统重新对齐（7.3）
    report = store.reconcile(
        ctx.project_id,
        {s.shot_id: {"keyframe": ctx.keyframe_of(s.shot_id),
                     "video": ctx.video_of(s.shot_id)} for s in script.shots},
    )
    if report.changed:
        print(f"[S3] 状态重整：{report.summary()}")

    untrusted = S2.degraded_chars(ctx)
    for i, shot in enumerate(script.shots):
        _check_stop(should_stop, ctx, "[S3] 生成首帧", i, len(script.shots))
        result = S3.run(
            ctx, shot, characters=chars, scene=scenes[shot.scene_id], style=script.style,
            image_provider=image_provider, judge_provider=judge_provider, store=store,
            ledger=ledger, allow_paid=allow_paid, untrusted_chars=untrusted,
        )
        score = "-" if result.score is None else str(result.score)
        print(f"     {shot.shot_id}  {result.status:<9} score={score:<3} "
              f"质量重试={result.quality_used} 技术重试={result.tech_used}"
              + (f"  ← {result.note}" if result.note else ""))


def stage_s4(
    ctx: ProjectCtx, script: Script, chars: list[Character],
    store: StateStore, ledger: CostLedger, *, allow_paid: bool, video_provider,
    judge_provider, should_stop=None,
) -> list[str]:
    """S4 · 图生视频 + 末帧判定。degraded 镜头也要出片（7.2），只是状态保持 degraded。

    返回**待人工确认**的镜头 id 列表——它们视频已出但判定不达标，S5 不能往下走。
    """
    print("[S4] 图生视频")
    pending: list[str] = []
    for i, shot in enumerate(script.shots):
        # 停止点放在镜头之间。**这里有实质意义**：每镜视频约 ¥1，六镜就是 ¥6，
        # 用户看着前两镜发现风格全错时，最该停的正是这个循环。
        _check_stop(should_stop, ctx, "[S4] 出片", i, len(script.shots))
        r = S4.run(ctx, shot, characters=chars, provider=video_provider,
                   store=store, ledger=ledger, allow_paid=allow_paid,
                   judge_provider=judge_provider)
        mark = {"done": "✓", "degraded": "△", "failed": "✗", "awaiting_review": "⏸"}.get(r.status, "·")
        if r.status in ("done", "degraded", "failed", "awaiting_review"):
            print(f"     {shot.shot_id}  {mark} {r.status:<15} 技术重试={r.tech_used}"
                  + (f"  ← {r.note}" if r.note else ""))
        if r.status == "awaiting_review":
            pending.append(shot.shot_id)
    return pending


def stage_review(ctx: ProjectCtx, store: StateStore) -> int:
    """`--review`：列出待人工确认的镜头。只读，不动任何状态。"""
    rows = store.awaiting_review(ctx.project_id)
    if not rows:
        print(f"[审核] {ctx.project_id} 没有待确认的镜头")
        return 0
    print(f"[审核] {ctx.project_id} 有 {len(rows)} 个镜头待确认")
    print("-" * 66)
    for r in rows:
        frame = r.get("last_frame") or "（无末帧）"
        print(f"  {r['shot_id']}  末帧判定 {r.get('video_score')} 分"
              f"（首帧判定 {r.get('score')} 分）")
        if r.get("video_diff"):
            print(f"    差异：{r['video_diff']}")
        print(f"    末帧：{frame}")
        print(f"    视频：{ctx.video_of(r['shot_id'])}")
        print()
    print("处理方式（三选一）：")
    print(f"  python run.py --project-id {ctx.project_id} --decide <shot_id> --accept --note \"理由\"")
    print(f"  python run.py --project-id {ctx.project_id} --decide <shot_id> --degrade --note \"理由\"")
    print(f"  python run.py --project-id {ctx.project_id} --decide <shot_id> --reject --note \"哪里不对\"")
    print("    accept  放行（记 done）")
    print("    degrade 放行但标红（记 degraded，报告里会列出）")
    print("    reject  驳回，回去重出这一条（下次跑流水线时执行）")
    return 0


def stage_decide(
    ctx: ProjectCtx, store: StateStore, shot_id: str, *,
    accept: bool, degrade: bool, reject: bool, note: str,
) -> int:
    """`--decide`：人工裁决一个待确认镜头。三选一，必须明确给。"""
    picked = [k for k, v in (("accept", accept), ("degrade", degrade), ("reject", reject)) if v]
    if len(picked) != 1:
        print("✗ 必须且只能给一个：--accept / --degrade / --reject", file=sys.stderr)
        return 2
    st = store.get(ctx.project_id, shot_id)
    if st is None:
        print(f"✗ 镜头 {shot_id} 不在项目 {ctx.project_id} 里", file=sys.stderr)
        return 2
    if st.status != "awaiting_review":
        print(f"✗ {shot_id} 当前状态是 {st.status}，只有 awaiting_review 可以被裁决",
              file=sys.stderr)
        return 2

    decision = picked[0]
    target = {"accept": "done", "degrade": "degraded", "reject": "videoing"}[decision]
    label = {"accept": "放行", "degrade": "放行但标红", "reject": "驳回重出"}[decision]
    reason = f"人工{label}" + (f"：{note}" if note else "")
    store.transition(ctx.project_id, shot_id, target, reason=reason)

    # 裁决留痕：判定分数保留，只补人工决定，不覆盖机器判的分数。
    qa = store.load_video_qa(ctx.project_id, shot_id) or {"score": -1}
    store.save_video_qa(
        ctx.project_id, shot_id, qa,
        human_decision={"accept": "accept", "degrade": "accept_degraded", "reject": "reject"}[decision],
        human_note=note,
    )
    print(f"[审核] {shot_id} → {target}（{reason}）")
    if decision == "reject":
        print("     下次跑流水线会重出这一条视频（约 ¥2/条）")
    return 0


def stage_s5(ctx: ProjectCtx, script: Script) -> S5.ComposeResult:
    """S5 · 合成。ffmpeg 缺失时输出无损片段包，不静默失败。"""
    from providers.tts import default_tts

    tts = default_tts()
    result = S5.run(ctx, script, tts=tts)
    if result.final:
        print(f"[S5] 成片：{result.final.name}  字幕 {result.cue_count} 条 / "
              f"{result.total_duration:g}s")
        print(f"     配音：{result.spoken} 句有台词 / {result.silent} 镜补静音"
              f"（音源 {tts.name}）")
        if result.tempo_adjusted:
            detail = "，".join(f"{sid} {t:g}×" for sid, t in result.tempo_adjusted)
            print(f"     台词超长已加速：{detail}")
        if result.truncated:
            print(f"     ⚠ 台词被截断：{result.truncated}（应回 S1 缩短或拆镜）")
    else:
        print(f"[S5] 未生成成片（{result.status}）："
              f"{result.notes[-1] if result.notes else '原因未知'}")
        if result.package_dir:
            print(f"     无损片段包：{result.package_dir}（{len(result.segments)} 段）")
    for note in result.notes:
        print(f"     · {note}")
    return result


def stage_s6(
    ctx: ProjectCtx, script: Script, store: StateStore, ledger: CostLedger,
    compose: S5.ComposeResult, anchors_status: dict,
) -> None:
    """S6 · 交付：report.html + costs.json。"""
    costs = R.write_costs(ctx, ledger)
    html = R.build(ctx, script, store=store, ledger=ledger,
                   compose=compose, anchors_status=anchors_status)
    print(f"[S6] 交付：{html.name} / {costs.name}")


def _ingest_refs(ctx: ProjectCtx, paths: list[str]) -> list[str]:
    """把用户给的参考图复制进项目 `refs/`，返回相对路径（顺序即绑定顺序）。

    **复制而不是引用原路径**：原路径可能是桌面、下载目录或临时文件，
    用户之后一挪动，重跑就找不到图；而且报告里写一个 `C:\\Users\\...\\桌面\\1.png`
    对别人毫无意义。项目自包含才能复现。
    """
    refs_dir = ctx.root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    rels: list[str] = []
    for i, raw in enumerate(paths, start=1):
        src = Path(raw).expanduser()
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        if not src.exists():
            raise OSError(f"参考图不存在：{src}")
        suffix = src.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
            raise OSError(f"参考图格式不支持：{src.name}（要 png/jpg/webp）")
        dst = refs_dir / f"ref_{i:02d}{suffix}"
        shutil.copyfile(src, dst)
        rels.append(str(dst.relative_to(ctx.root)))
    return rels


def _print_stop_hint(ctx: ProjectCtx, st: P.PhaseState) -> None:
    """停点时打印"现在该干什么"。**不打印就等于静默停住**，用户会以为程序卡了。"""
    pid = ctx.project_id
    print(f"⏸ 停在「{st.label}」（phase={st.phase}）")
    if st.phase == "created":
        print("   有人物参考图：python run.py --project-id "
              f"{pid} --ref <图片路径>（可多次，按顺序绑 char_01、char_02）")
        print(f"   没有：        python run.py --project-id {pid} --confirm")
    elif st.phase == "script_review":
        print(f"   剧本与角色卡：{ctx.script_file}")
        print(f"   改完继续：    python run.py --project-id {pid} --confirm")
        print("   或在界面「② 角色与分镜」里直接改，点「保存并确认」")
    elif st.phase == "anchor_review":
        print(f"   正脸图：      {ctx.anchors_dir}")
        print(f"   不满意：      python run.py --project-id {pid} "
              "--regen-front char_01 --feedback \"脸再瘦一点\"")
        print(f"   满意：        python run.py --project-id {pid} --confirm")
    print(f"   一路跑完：    python run.py --project-id {pid} --auto --allow-paid")


class Cancelled(RuntimeError):
    """用户中途叫停。**不是故障**：已完成的镜头保持终态，没开工的留 pending。

    用异常而不是"返回一个特殊码"：叫停可能发生在两个循环里的任意一层
    （stage_s3 / stage_s4），返回值要一层层往上翻译，漏一层就变成"停了但上层
    以为跑完了"，于是继续合成成片。异常只可能漏接，不可能被误读成成功。
    """


def _check_stop(should_stop, ctx: ProjectCtx, where: str, done: int, total: int) -> None:
    """在**两镜之间**检查是否该停。

    只在镜与镜之间检查，不在镜内检查：一镜的首帧或视频已经在付费调用里，
    半途撒手会让这一镜的钱花掉、产物却没有，账面和磁盘对不上。
    代价是"点了停止要等当前这一镜跑完"（视频约 1–2 分钟），
    这个代价比"停在一笔付了钱没落地的调用上"小得多。
    """
    if should_stop is not None and should_stop():
        raise Cancelled(f"{where}：已完成 {done}/{total}，按用户要求停下")


def _run_to_end(ctx: ProjectCtx, args: argparse.Namespace, *, ledger: CostLedger,
                store: StateStore, script: Script, image_p, video_p, judge_p,
                should_stop=None) -> int:
    """S3 → S6。进到这里说明正脸已确认（停点已过）。

    `should_stop` 是 Web 端「停止」按钮传进来的回调（2026-11-02 加）。
    CLI 不传，行为与以前完全一致。
    """
    chars = _load_characters(ctx)
    stage_s3(ctx, script, chars, store, ledger, allow_paid=args.paid_ok,
             image_provider=image_p, judge_provider=judge_p, should_stop=should_stop)
    stage_s4(ctx, script, chars, store, ledger, allow_paid=args.paid_ok,
             video_provider=video_p, judge_provider=judge_p, should_stop=should_stop)

    pending = store.awaiting_review(ctx.project_id)
    if pending:
        # S5 不能往下走：待确认镜头的视频还没被人认可，合成出来就是「悄悄放行」，
        # 而 9.4 明确禁止静默放行。这里必须停，并把人要做的事说清楚。
        P.save(ctx, phase="running", note=f"{len(pending)} 镜待人工裁决，未合成")
        print()
        print(f"⏸ 有 {len(pending)} 个镜头视频已出但末帧判定不达标，等待人工确认：")
        for r in pending:
            print(f"     {r['shot_id']}  末帧 {r.get('video_score')} 分"
                  f"｜{ctx.video_of(r['shot_id'])}")
        print(f"   先看一眼：python run.py --project-id {ctx.project_id} --review")
        print(f"   再裁决  ：python run.py --project-id {ctx.project_id} "
              f"--decide {pending[0]['shot_id']} --accept|--degrade|--reject --note \"理由\"")
        print("   处理完重跑本命令即可继续合成。")
        return 0

    compose = stage_s5(ctx, script)
    stage_s6(ctx, script, store, ledger, compose, S2.load_status(ctx))
    P.advance_to(ctx, "done", note="成片已交付")
    return 0


def run_pipeline(ctx: ProjectCtx, args: argparse.Namespace, *, ledger: CostLedger,
                 store: StateStore, text_p, image_p, video_p, judge_p,
                 should_stop=None) -> int:
    """按阶段推进流水线。

    **每次运行只推进到下一个停点就返回**（`--auto` 除外）。这不是限制，
    是补上原本缺失的语义：S1 出完让人改剧本、S2a 出完让人看脸，这两件事
    都发生在花钱的大头（六镜视频）之前，停下来才是最省的。

    `--confirm` 只够过一次停点，过完继续往下跑到**下一个**停点再停。
    想一次过三关就是 `--auto`——那等于明确表示"我认了，别问我"。
    """
    while True:
        st = P.load(ctx)

        if st.phase == "done":
            print(f"[完成] {ctx.project_id} 已经跑完了（{ctx.final_video}）")
            print("   要整部重跑：换一个 project_id，或加 --force 重建项目壳")
            return 0

        if st.is_stop and not args.auto and not args.confirm:
            _print_stop_hint(ctx, st)
            return 0

        if st.phase == "created":
            script = stage_s1(ctx, args, ledger, text_p, judge_provider=judge_p)
            P.advance_to(ctx, "script_review",
                         note=f"S1 产出 {len(script.shots)} 镜，等人确认")
        elif st.phase == "script_review":
            script = S1.load_script(ctx)
            stage_s2a(ctx, script, ledger, allow_paid=args.paid_ok, image_provider=image_p)
            P.advance_to(ctx, "anchor_review", note="正脸已出，等人确认")
        elif st.phase == "anchor_review":
            script = S1.load_script(ctx)
            stage_s2b(ctx, script, ledger, allow_paid=args.paid_ok,
                      image_provider=image_p, judge_provider=judge_p)
            P.advance_to(ctx, "running", note="正脸确认，出侧背并开始生产")
            return _run_to_end(ctx, args, ledger=ledger, store=store, script=script,
                               image_p=image_p, video_p=video_p, judge_p=judge_p,
                               should_stop=should_stop)
        elif st.phase == "running":
            script = S1.load_script(ctx)
            return _run_to_end(ctx, args, ledger=ledger, store=store, script=script,
                               image_p=image_p, video_p=video_p, judge_p=judge_p,
                               should_stop=should_stop)
        else:  # failed 或未知
            print(f"✗ 项目处于 {st.phase} 状态，无法自动继续。"
                  f"检查 project.json 的 phase 字段，或重建项目。", file=sys.stderr)
            return 2

        if not args.auto:
            _print_stop_hint(ctx, P.load(ctx))
            return 0


def serve(host: str | None = None, port: int | None = None) -> int:
    """启动 Web 服务——容器内 `--mode api` 的入口（7.6）。

    为什么服务模式要挂在 `run.py` 上而不是让 Dockerfile 直接调 uvicorn：
    7.6 定的对外形态是 `drama-agent --mode api`，使用者不该为了"有没有项目"
    换成另一条完全不同的命令。**入口只有一个**，行为由参数决定。

    7.3.1 明令禁止 `uvicorn --reload`：它会在文件变动时重启进程，
    而 C2 通道的重复提交正是这么来的。这里不提供该选项，也不接受环境变量开启。
    """
    import os

    import uvicorn

    from adapters.web.api import app as web_app

    host = host or os.environ.get("HOST", "0.0.0.0")
    port = port or int(os.environ.get("PORT", "8000"))
    print(f"[serve] Web 服务：http://{host}:{port}   （--mode api：B 通道 + 界面）")
    uvicorn.run(web_app, host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 2026-10-31：C2（浏览器接管）通道已整体移除，只保留 B 通道。
    # 这里保留对 `--mode web` 的显式拒绝，而不是把取值从 choices 里删掉：
    # 照旧文档抄命令的人会得到一句能懂的说明，而不是 argparse 的原文报错。
    if args.mode == "web":
        print("✗ --mode web（C2 浏览器接管通道）已于 2026-10-31 移除，"
              "现在只保留 B 通道。", file=sys.stderr)
        print("  改用：python run.py --project-id <id> --allow-paid", file=sys.stderr)
        return 2

    # 没有项目 ID = 服务模式。7.6 里 `docker run ... --mode api` 就是这么调的：
    # 容器入口不该要求"先有一个项目"才能启动，项目是在界面里建的。
    if not args.project_id:
        return serve()

    # 付费护栏（D7 / 9.4）：真实模式必须显式授权。
    # 这里不再拦「没有真实 provider」——vlm / relay_image / dashscope_video 都已就位，
    # 拦住付钱的只有这一道显式授权，而不是「功能没实现」。
    #
    # 例外：`--review` / `--decide` 是纯本地操作（读 SQLite、改状态），不调任何模型。
    # 让它们也要 --allow-paid 是错的——人只是想看一眼待办，不该被要求先授权花钱。
    #
    # `--dry-run` 同理（2026-10-29 补）：它只装配 provider、打印预估、立刻返回，
    # 一次请求都不发。护栏防的是"误调付费接口"，而 dry-run 恰恰是不调的那一个，
    # 让它也要 --allow-paid 会导致「想先看看六镜要花多少钱，必须先授权花钱」的荒谬顺序。
    local_only = bool(args.review or args.decide or args.dry_run)
    if not args.mock and not args.allow_paid and not local_only:
        print("✗ 真实模式必须显式加 --allow-paid（付费护栏，见 D7 与 9.4）", file=sys.stderr)
        print("  不加此参数，每一次调用都会抛 PaidCallBlocked。", file=sys.stderr)
        print("  零成本验证请用：python run.py --mock --project-id <id>", file=sys.stderr)
        return 2

    ensure_dirs()
    args.paid_ok = bool(args.mock or args.allow_paid)

    try:
        text_p, image_p, video_p, judge_p = build_providers(args.mock)
    except Exception as exc:  # noqa: BLE001  配置缺失/依赖缺失，在这里就要说清楚
        print(f"✗ 供应商装配失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("  检查 .env 的 IMAGE_* / VIDEO_* / VLM_* 是否齐全。", file=sys.stderr)
        return 2

    try:
        ctx, created = stage_s0(args)
    except ValueError as exc:
        print(f"✗ S0 失败：{exc}", file=sys.stderr)
        return 2

    print("=" * 66)
    print(f"[S0] 初始化{'（新建）' if created else '（复用已有项目）'}  {ctx.root}")
    print(f"     预算 ¥{ctx.budget_cap:.2f} / 通道 {ctx.channel} / "
          f"{'mock（零成本）' if ctx.mock else 'real'}")

    # 人工确认是**纯本地操作**：不调模型、不花钱、不装配供应商。
    # 所以放在算账之前——用户可能只想看一眼待办，不想看到一串预估和 mock 装配提示。
    if args.review or args.decide:
        store = StateStore()
        try:
            if args.review:
                return stage_review(ctx, store)
            return stage_decide(
                ctx, store, args.decide,
                accept=args.accept, degrade=args.degrade, reject=args.reject,
                note=args.note or "",
            )
        finally:
            store.close()

    # 复用项目时预算取自 project.json，`--budget` 会被静默忽略。
    # 这是刻意的（断点续跑不该让每次运行改掉预算），但**必须提示**，
    # 否则用户以为调了预算、实际还在用旧值，护栏就变成了心理安慰。
    if args.budget is not None and abs(args.budget - ctx.budget_cap) > 1e-9:
        print(f"     ⚠ --budget {args.budget:g} 未生效：项目已存在，预算沿用 project.json 的 "
              f"¥{ctx.budget_cap:.2f}。要改预算：直接改该文件，或用 --force 重建（会重置 created_at）。",
              file=sys.stderr)

    # 真实模式先算账再动手：把上限摆到台面上，超了当场停，而不是烧到一半才报 BudgetExceeded
    if not ctx.mock:
        est = estimate_cost(args.shots)
        print(f"     预估 ¥{est['total']:.2f} = 出图 {est['images']}张 ×¥{settings.price_image}"
              f"(¥{est['image_cost']:.2f}) + 视频 {est['seconds']:.0f}秒 ×¥{settings.price_video_per_sec}"
              f"(¥{est['video_cost']:.2f}) + 判定 {est['judges']}次 ×¥{settings.price_vlm_per_call}"
              f"(¥{est['judge_cost']:.2f})")
        if est["total"] > ctx.budget_cap:
            print(f"     ✗ 预估 ¥{est['total']:.2f} 超过预算上限 ¥{ctx.budget_cap:.2f}，已中止。"
                  f"用 --budget 调高或 --shots 调小。", file=sys.stderr)
            return 2

    if args.dry_run:
        names = f"{text_p.name} / {image_p.name} / {video_p.name} / {judge_p.name}"
        print(f"     [dry-run] 供应商：{names}")
        print("     [dry-run] 装配与预算检查通过，未发起任何调用。")
        return 0

    # ---- 停 0：角色来源与参考图（纯落盘，不调模型）----
    try:
        if args.ref:
            rels = _ingest_refs(ctx, args.ref)
            P.save(ctx, ref_images=rels, role_source="image")
            print(f"     参考图 {len(rels)} 张入库，按顺序绑 char_01、char_02…")
        elif args.role_source:
            P.save(ctx, role_source=args.role_source)
    except (OSError, P.PhaseError) as exc:
        print(f"✗ 参考图入库失败：{exc}", file=sys.stderr)
        return 2

    store = StateStore()
    ledger = CostLedger(ctx.project_id, cap=ctx.budget_cap)
    code = 0

    try:
        if args.regen_front:
            code = run_regen_front(ctx, args, ledger=ledger, image_provider=image_p)
        elif args.rerun_shot:
            code = run_single_shot(ctx, args, ledger=ledger, store=store,
                                   image_provider=image_p, video_provider=video_p,
                                   judge_provider=judge_p)
        else:
            code = run_pipeline(ctx, args, ledger=ledger, store=store, text_p=text_p,
                                image_p=image_p, video_p=video_p, judge_p=judge_p)
    except S1.ScriptParseError as exc:
        P.save(ctx, phase="failed", note=f"S1 失败：{exc}")
        print(f"✗ S1 失败：{exc}", file=sys.stderr)
        code = 1
    except P.PhaseError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        code = 2
    except Exception as exc:  # noqa: BLE001 —— CLI 顶层兜底，不能把栈糊在用户脸上
        try:
            P.save(ctx, phase="failed", note=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001 —— 连失败都记不下时，别再抛第二层异常
            pass
        print(f"✗ 流水线中断：{type(exc).__name__}: {exc}", file=sys.stderr)
        code = 1

    summary = store.summary(ctx.project_id)
    cost = ledger.summary()
    print("-" * 66)
    print(f"[汇总] 镜头 {summary['total']} 个：" + " / ".join(
        f"{k}={v}" for k, v in summary["counts"].items() if v))
    # 实扣与失败估算必须分开报（2026-10-29 修）：此前只打 `spent`，而 `spent` 是
    # COALESCE(actual, estimated)——它把**失败调用**的估算价也算了进去。
    # 六镜那次因此报出 ¥9.39，而真实扣费 ¥5.39，虚高 74%。失败调用（参数不兼容的 400）
    # 平台一分不扣，但本地日志里 actual_cost 为 NULL，于是被 estimated 顶上了。
    # 护栏仍然用 `spent`（保守，宁可早停），但**给人看的数必须是实扣**。
    print(f"       实扣 ¥{cost['actual_total']:.2f}"
          f"（{cost['calls']} 次调用，失败 {cost['failed_calls']} 次）"
          f" / 预算上限 ¥{cost['cap']:.2f}")
    est_failed = round(cost["estimated_total"] - cost["actual_total"], 4)
    if cost["failed_calls"]:
        print(f"       ⚠ 失败调用 {cost['failed_calls']} 次，估算 ¥{est_failed:.2f} 未扣费"
              f"（4xx 类拒绝平台不收费，故不计入预算占用）")

    if PENDING_STAGES:
        print(f"       未实现：{' / '.join(PENDING_STAGES)}")
    print("=" * 66)

    store.close()
    ledger.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
