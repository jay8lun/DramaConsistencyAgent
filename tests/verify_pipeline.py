"""M1-S3c 自检：s1_script / s2_anchor / s3_keyframe。

用可编排的假 provider 覆盖正常与异常路径（真实 API 一律不碰）：

  [1] S1 用 MockText 产出合法剧本并落盘
  [2] S1 结构不过时带错误重问，修好后成功
  [3] S1 连续失败 → 抛错（不降级：剧本错了后面全错）
  [4] S2 自检通过 → anchors 三视图齐全、characters.json 回填 anchor_path
  [5] S2 自检不过 → 重出 2 次后 degraded，且状态落盘
  [6] S3 高分 → 直接进 videoing
  [7] S3 中分 → 带增量重取，第 2 次提示词 ≠ 第 1 次且含增量
  [8] S3 低分 → degraded（不是 failed）
  [9] S3 技术失败 2 次后成功 → tech_attempts=2，仍能通过
  [10] S3 技术失败 3 次 → failed，且 quality 计数未被污染
  [11] S3 续跑：judging 且有落库分数 → 不重调 VLM
  [12] 锚点降级的角色 → 判定结论被标记为不可信
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Character, Scene, Script, Shot, VisualStyle  # noqa: E402
from pipeline import judge as J  # noqa: E402
from pipeline import s1_script as S1  # noqa: E402
from pipeline import s2_anchor as S2  # noqa: E402
from pipeline import s3_keyframe as S3  # noqa: E402
from pipeline.cost import CostLedger  # noqa: E402
from pipeline.project import ProjectCtx  # noqa: E402
from pipeline.state import StateStore, now_iso  # noqa: E402
from providers.base import ImageProvider, ProviderError, TextProvider, VisionJudge  # noqa: E402
from providers.mock import MOCK_SCRIPT, MockText, write_solid_png  # noqa: E402

PASS = 0
FAIL = 0
FAILED_NAMES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILED_NAMES.append(name)
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------- 假 provider


class ScriptedText(TextProvider):
    """按顺序返回预设文本；用完后重复最后一条。"""

    name = "scripted-text"          # 故意不带 mock 前缀：零成本靠 is_mock 标记，不靠命名
    is_mock = True

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, system, user, *, json_mode=False) -> str:
        self.calls += 1
        return self.outputs[min(self.calls - 1, len(self.outputs) - 1)]


class RecordingImage(ImageProvider):
    """记录每次收到的提示词，落一张真 PNG。可配置前 N 次抛错。"""

    name = "rec-image"              # 同上，验证 is_mock 而非名称前缀在起作用
    is_mock = True
    unit_price = 0.0

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.prompts: list[str] = []
        self.refs: list[int] = []

    def generate(self, prompt, *, out, references=None, size="1024x1792", allow_paid=False) -> Path:
        self.prompts.append(prompt)
        self.refs.append(len(references or []))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ProviderError("模拟上游 503")
        return write_solid_png(Path(out))


class ScriptedJudge(VisionJudge):
    """按顺序返回预设的原始判定 dict；用完后重复最后一条。"""

    name = "scripted-judge"
    is_mock = True

    def __init__(self, raws: list[dict]):
        self.raws = list(raws)
        self.calls = 0

    def compare(self, image, reference, elements, *, allow_paid: bool = False) -> dict:
        self.calls += 1
        return self.raws[min(self.calls - 1, len(self.raws) - 1)]


def raw(score: int, fails: tuple[str, ...] = (), diff: str = "") -> dict:
    return {
        "score": score,
        "checks": [
            {"element": e, "matched": e not in fails,
             "expected": "黑色齐肩直发" if e == "发型" else "见角色卡",
             "actual": "红棕色高马尾" if e in fails else "一致"}
            for e in J.ELEMENTS
        ],
        "diff": diff,
    }


def new_project(tmp: Path, name: str) -> ProjectCtx:
    root = tmp / name
    for sub in ("anchors", "keyframes", "shots", "inbox"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return ProjectCtx(project_id=name, root=root, budget_cap=10.0,
                      channel="api", mock=True, created_at=now_iso())


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="m1s3c_"))
    ctx = new_project(tmp, "p1")
    store = StateStore(tmp / "state.db")
    ledger = CostLedger("p1", cap=10.0, db_path=tmp / "state.db")

    # ---------------------------------------------------------------- S1
    print("=" * 64)
    print("[1] S1b：MockText → 合法剧本 + 落盘")
    print("=" * 64)
    res = S1.run(ctx, MockText(), creative="深夜电梯里的第三个人", shot_count=6)
    script = res.script
    check("产出 Script 且通过 4.7 全部校验", isinstance(script, Script))
    check("镜数 = 6 且 index 连续", len(script.shots) == 6 and [s.index for s in script.shots] == list(range(1, 7)))
    check("script.json 已落盘", ctx.script_file.exists())
    check("characters.json 单独落盘（6 章要求分开）", ctx.characters_file.exists())
    chars_on_disk = json.loads(ctx.characters_file.read_text(encoding="utf-8"))
    check("characters.json 是 2 张角色卡", len(chars_on_disk) == 2)
    check("首次即通过，未触发修复", res.attempts == 1 and res.repaired is False)
    check("空镜分支被覆盖（第 6 镜 environment_only）",
          any(s.environment_only for s in script.shots))

    print()
    print("=" * 64)
    print("[2][3] S1b：结构不过带错误重问 / 连续失败抛错")
    print("=" * 64)
    broken = json.dumps({"title": "x", "shots": []}, ensure_ascii=False)
    ctx2 = new_project(tmp, "p2")
    text = ScriptedText([broken, json.dumps(MOCK_SCRIPT, ensure_ascii=False)])
    res2 = S1.run(ctx2, text, creative="测试修复")
    check("第 1 次不合法 → 第 2 次成功", res2.attempts == 2 and res2.repaired is True)
    check("确实调用了两次模型", text.calls == 2)

    ctx3 = new_project(tmp, "p3")
    bad = ScriptedText([broken])
    try:
        S1.run(ctx3, bad, creative="永远修不好", max_repair=3)
        check("连续失败必须抛错", False, "没有抛错")
    except S1.ScriptParseError as exc:
        check("连续失败必须抛错（不降级）", "3 次" in str(exc), str(exc)[:80])
    check("重问次数 = max_repair", bad.calls == 3, f"{bad.calls}")

    # ---------------------------------------------------------------- S2
    print()
    print("=" * 64)
    print("[4] S2：三视图生成 + 自检通过")
    print("=" * 64)
    chars = list(script.characters)
    img = RecordingImage()
    ok_judge = ScriptedJudge([raw(92)])
    chars, s2res = S2.run(ctx, chars, style=script.style, image_provider=img,
                          judge_provider=ok_judge, ledger=ledger, allow_paid=True)
    check("每个角色 3 张视图", all(len(r.paths) == 3 for r in s2res.values()),
          f"{ {k: len(v.paths) for k, v in s2res.items()} }")
    check("文件真的存在且可读", all(Path(p).exists() for r in s2res.values() for p in r.paths.values()))
    check("状态都是 ok", all(r.status == "ok" for r in s2res.values()))
    check("侧/背图带了参考图（正面）", any(n == 1 for n in img.refs[1:]), f"{img.refs}")
    check("character.anchor_path 已回填", all(c.anchor_path for c in chars))
    check("anchor_views 有 3 条", all(len(c.anchor_views) == 3 for c in chars))
    check("characters.json 被回写", "anchor_path" in ctx.characters_file.read_text(encoding="utf-8"))
    check("自检状态落盘 anchors/_status.json", (ctx.anchors_dir / "_status.json").exists())

    print()
    print("=" * 64)
    print("[5] S2：自检不过 → 重出后 degraded 且继续")
    print("=" * 64)
    ctx4 = new_project(tmp, "p4")
    img4 = RecordingImage()
    bad_judge = ScriptedJudge([raw(70, fails=("发型",), diff="发型不符")])
    _, s2bad = S2.run(ctx4, chars, style=script.style, image_provider=img4,
                      judge_provider=bad_judge, ledger=ledger, allow_paid=True,
                      max_retry=2)
    check("重出 2 次后标 degraded（不是失败退出）",
          all(r.status == "degraded" for r in s2bad.values()), f"{[r.status for r in s2bad.values()]}")
    check("attempts = 3（首次 + 2 次重出）", all(r.attempts == 3 for r in s2bad.values()))
    check("重出确实跑了 3 轮图（2 角色 × 3 视图 × 3 轮 = 18 次）", len(img4.prompts) == 18,
          f"{len(img4.prompts)}")
    status = json.loads((ctx4.anchors_dir / "_status.json").read_text(encoding="utf-8"))
    check("degraded 被写入状态文件（否则 S3 不知道自己用了错锚点）",
          all(v["status"] == "degraded" for v in status.values()), f"{status}")
    check("degraded_chars() 能取到", S2.degraded_chars(ctx4) == {"char_01", "char_02"})

    # ---------------------------------------------------------------- S3
    print()
    print("=" * 64)
    print("[6] S3：高分直接通过")
    print("=" * 64)
    store.ensure_shots(ctx.project_id, [s.shot_id for s in script.shots])
    shot = script.shots[0]
    scene = next(s for s in script.scenes if s.scene_id == shot.scene_id)
    img6 = RecordingImage()
    judge6 = ScriptedJudge([raw(88)])
    r6 = S3.run(ctx, shot, characters=chars, scene=scene, style=script.style,
                image_provider=img6, judge_provider=judge6, store=store, ledger=ledger,
                allow_paid=True)
    check("状态 = videoing", r6.status == "videoing", r6.status)
    check("分数 88", r6.score == 88, f"{r6.score}")
    check("只生成 1 次、只判 1 次", len(img6.prompts) == 1 and judge6.calls == 1)
    check("生成时带了锚点参考图", img6.refs[0] >= 1, f"{img6.refs}")
    check("质量重试计数为 0", r6.quality_used == 0, f"{r6.quality_used}")
    check("首帧文件已落盘", ctx.keyframe_of(shot.shot_id).exists())

    print()
    print("=" * 64)
    print("[7] S3：中分带增量重取（第 2 次提示词 ≠ 第 1 次）")
    print("=" * 64)
    shot2 = script.shots[1]
    scene2 = next(s for s in script.scenes if s.scene_id == shot2.scene_id)
    img7 = RecordingImage()
    judge7 = ScriptedJudge([raw(70, fails=("发型",), diff="发型长度不符，实际扎起"), raw(90)])
    r7 = S3.run(ctx, shot2, characters=chars, scene=scene2, style=script.style,
                image_provider=img7, judge_provider=judge7, store=store, ledger=ledger,
                allow_paid=True)
    check("最终通过", r7.status == "videoing", r7.status)
    check("生成 2 次、判定 2 次", len(img7.prompts) == 2 and judge7.calls == 2)
    check("第 2 次提示词 ≠ 第 1 次（不是原样重跑）", img7.prompts[0] != img7.prompts[1])
    check("第 2 次提示词含增量指令", "修正要求" in img7.prompts[1] and "发型" in img7.prompts[1],
          img7.prompts[1][-60:])
    check("增量文本完整保留（含括号）", "（差异：发型长度不符，实际扎起）" in img7.prompts[1],
          img7.prompts[1][-60:])
    check("quality_attempts = 1", r7.quality_used == 1, f"{r7.quality_used}")
    check("状态库 attempts 与结果一致",
          store.get(ctx.project_id, shot2.shot_id).quality_attempts == 1)

    print()
    print("=" * 64)
    print("[8] S3：低分 → degraded（不是 failed）")
    print("=" * 64)
    shot3 = script.shots[2]
    scene3 = next(s for s in script.scenes if s.scene_id == shot3.scene_id)
    img8 = RecordingImage()
    judge8 = ScriptedJudge([raw(40, fails=("发型", "脸型"), diff="整体不像")])
    r8 = S3.run(ctx, shot3, characters=chars, scene=scene3, style=script.style,
                image_provider=img8, judge_provider=judge8, store=store, ledger=ledger,
                allow_paid=True)
    check("状态 = degraded", r8.status == "degraded", r8.status)
    check("<60 不浪费重取（只生成 1 次）", len(img8.prompts) == 1, f"{len(img8.prompts)}")

    print()
    print("=" * 64)
    print("[9][10] S3：技术重试（原样重发，不计质量重试）")
    print("=" * 64)
    shot4 = script.shots[3]
    scene4 = next(s for s in script.scenes if s.scene_id == shot4.scene_id)
    img9 = RecordingImage(fail_times=2)
    r9 = S3.run(ctx, shot4, characters=chars, scene=scene4, style=script.style,
                image_provider=img9, judge_provider=ScriptedJudge([raw(91)]),
                store=store, ledger=ledger, allow_paid=True)
    check("失败 2 次后成功 → videoing", r9.status == "videoing", r9.status)
    check("tech_attempts = 2", store.get(ctx.project_id, shot4.shot_id).tech_attempts == 2,
          f"{store.get(ctx.project_id, shot4.shot_id).tech_attempts}")
    check("技术重试不污染质量重试计数",
          store.get(ctx.project_id, shot4.shot_id).quality_attempts == 0)

    shot5 = script.shots[4]
    scene5 = next(s for s in script.scenes if s.scene_id == shot5.scene_id)
    img10 = RecordingImage(fail_times=99)
    r10 = S3.run(ctx, shot5, characters=chars, scene=scene5, style=script.style,
                 image_provider=img10, judge_provider=ScriptedJudge([raw(90)]),
                 store=store, ledger=ledger, allow_paid=True)
    check("技术失败 3 次 → failed", r10.status == "failed", r10.status)
    check("技术重试恰好 3 次（不多不少）", len(img10.prompts) == 3, f"{len(img10.prompts)}")
    check("failed 是终态，不参与合成", store.get(ctx.project_id, shot5.shot_id).is_terminal)

    print()
    print("=" * 64)
    print("[11] S3 空镜：没有角色参考图 → 跳过判定（不判 0 分、不标红）")
    print("=" * 64)
    shot6 = script.shots[5]
    scene6 = next(s for s in script.scenes if s.scene_id == shot6.scene_id)
    img11 = RecordingImage()
    judge11 = ScriptedJudge([raw(0)])
    r11 = S3.run(ctx, shot6, characters=chars, scene=scene6, style=script.style,
                 image_provider=img11, judge_provider=judge11, store=store, ledger=ledger,
                 allow_paid=True)
    check("空镜照常生成首帧", ctx.keyframe_of(shot6.shot_id).exists())
    check("空镜进 videoing 而不是 degraded", r11.status == "videoing", r11.status)
    check("空镜不调 VLM（无参考图可比）", judge11.calls == 0, f"{judge11.calls}")
    check("空镜不编造分数", r11.score is None, f"{r11.score}")
    check("说明里写明跳过判定的原因", "空镜" in (r11.note or ""), r11.note)

    print()
    print("=" * 64)
    print("[12] S3 续跑：judging 且有落库分数 → 不重调 VLM")
    print("=" * 64)
    ctx6 = new_project(tmp, "p6")
    store6 = StateStore(tmp / "state.db")
    ledger6 = CostLedger("p6", cap=10.0, db_path=tmp / "state.db")
    resume_shot = script.shots[0]
    resume_scene = next(s for s in script.scenes if s.scene_id == resume_shot.scene_id)
    store6.ensure_shots("p6", [resume_shot.shot_id])
    img12 = RecordingImage()
    judge12 = ScriptedJudge([raw(87)])
    first = S3.run(ctx6, resume_shot, characters=chars, scene=resume_scene, style=script.style,
                   image_provider=img12, judge_provider=judge12, store=store6, ledger=ledger6,
                   allow_paid=True)
    check("首跑通过进 videoing", first.status == "videoing", first.status)
    # 人为把状态打回 judging（模拟「判定写完、迁移前被 kill」）
    store6._force("p6", resume_shot.shot_id, "judging", "模拟进程中断")
    calls_before = judge12.calls
    prompts_before = len(img12.prompts)
    r12 = S3.run(ctx6, resume_shot, characters=chars, scene=resume_scene, style=script.style,
                 image_provider=img12, judge_provider=judge12, store=store6, ledger=ledger6,
                 allow_paid=True)
    check("续跑不重调 VLM（判定费用不白烧）",
          judge12.calls == calls_before, f"{calls_before}→{judge12.calls}")
    check("续跑不重新生成首帧",
          len(img12.prompts) == prompts_before, f"{prompts_before}→{len(img12.prompts)}")
    check("续跑后仍进 videoing", r12.status == "videoing", r12.status)
    check("续跑说明里写明复用了判定", "续跑" in (r12.note or ""), r12.note)

    print()
    print("=" * 64)
    print("[13] 锚点降级的角色 → 判定结论标为不可信")
    print("=" * 64)
    ctx5 = new_project(tmp, "p5")
    store5 = StateStore(tmp / "state.db")
    ledger5 = CostLedger("p5", cap=10.0, db_path=tmp / "state.db")
    store5.ensure_shots("p5", [shot.shot_id])
    r12 = S3.run(ctx5, shot, characters=chars, scene=scene,
                 style=script.style, image_provider=RecordingImage(),
                 judge_provider=ScriptedJudge([raw(90)]), store=store5, ledger=ledger5,
                 allow_paid=True, untrusted_chars={"char_01"})
    qa12 = store5.load_qa("p5", shot.shot_id)
    check("结果带 anchor_untrusted 标记", r12.anchor_untrusted is True)
    check("判定记录里写明锚点不可信（报告能读到）",
          "锚点自检降级" in (qa12 or {}).get("diff", ""), f"{(qa12 or {}).get('diff')}")

    print()
    print("=" * 64)
    print("[14] 记账：mock 链路零成本")
    print("=" * 64)
    check("mock 供应商产生 ¥0 记账（不触发预算封顶）",
          ledger.spent() == 0.0, f"spent={ledger.spent()}")
    check("但每次调用都留了审计记录（没有记录等于没发生）",
          len(ledger.entries()) >= 4, f"{len(ledger.entries())}")

    print()
    print("=" * 64)
    print("[15] S3 判定：每个角色各判一次，取最差")
    print("=" * 64)
    # 2026-10-29 真跑发现的缺陷：之前 `judge_reference` 只返回第一个角色的锚点，
    # 多角色镜头的第二个人从来没被判定过（ds2 第一镜多出一个剧本没要求的人，仍判 95）。
    dual = shot.model_copy(update={"character_ids": [c.char_id for c in chars]})

    refs_single = S3.judge_references(shot, chars)
    check("单人镜头：只有一张判定参考图", len(refs_single) == 1, str(refs_single))
    refs_dual = S3.judge_references(dual, chars)
    check("双人镜头：每个角色各一张", len(refs_dual) == 2, str([c for c, _ in refs_dual]))
    check("顺序与 character_ids 一致（出图挂图与判定比对指向同一人）",
          [cid for cid, _ in refs_dual] == list(dual.character_ids),
          str([cid for cid, _ in refs_dual]))

    class PerCharJudge(VisionJudge):
        """按「参考图文件名 → 预设序列」返回判定，用完重复最后一条。

        为什么不用 ScriptedJudge：它按全局调用顺序取，而判定现在是
        「每轮 × 每个角色」，全局顺序会把第二个角色的返回值串到第一个角色头上。
        """

        name = "per-char-judge"
        is_mock = True

        def __init__(self, plan: dict[str, list[dict]]):
            self.plan = plan
            self.seen: dict[str, int] = {}
            self.calls = 0

        def compare(self, image, reference, elements, *, allow_paid: bool = False) -> dict:
            self.calls += 1
            key = Path(reference).stem
            seq = self.plan.get(key) or [raw(95)]
            i = self.seen.get(key, 0)
            self.seen[key] = i + 1
            return seq[min(i, len(seq) - 1)]

    # A. 取最差：char_01 = 95、char_02 = 70 → 分数必须是 70，不是平均 82.5
    ctx7 = new_project(tmp, "p7")
    store7 = StateStore(tmp / "state.db")
    ledger7 = CostLedger("p7", cap=10.0, db_path=tmp / "state.db")
    store7.ensure_shots("p7", [dual.shot_id])
    worst_judge = PerCharJudge({
        Path(chars[0].anchor_path).stem: [raw(95)],
        Path(chars[1].anchor_path).stem: [raw(70, fails=("发型",), diff="第二位发型不符")],
    })
    r15 = S3.run(ctx7, dual, characters=chars, scene=scene, style=script.style,
                 image_provider=RecordingImage(), judge_provider=worst_judge,
                 store=store7, ledger=ledger7, allow_paid=True, max_quality_retry=0)
    check("双人镜头每个角色各判一次（共 2 次）", worst_judge.calls == 2, f"calls={worst_judge.calls}")
    check("分数取最差而非平均（95 与 70 → 70）", r15.score == 70, f"score={r15.score}")
    check("第二个人不合格 → 降级，不会带着坏图进视频",
          r15.status == "degraded", r15.status)
    qa7 = store7.load_qa("p7", dual.shot_id) or {}
    check("判定记录写明逐角色分数（报告能读出谁最差）",
          "char_01=95" in qa7.get("diff", "") and "char_02=70" in qa7.get("diff", ""),
          qa7.get("diff", "")[:120])
    store7.close()
    ledger7.close()

    # B. 第二个人只是暂时不像 → 重取一次后两人都过，恢复 videoing
    ctx8 = new_project(tmp, "p8")
    store8 = StateStore(tmp / "state.db")
    ledger8 = CostLedger("p8", cap=10.0, db_path=tmp / "state.db")
    store8.ensure_shots("p8", [dual.shot_id])
    fixable_judge = PerCharJudge({
        Path(chars[0].anchor_path).stem: [raw(95)],
        # 第一次 70（触发带增量重取），第二次 95
        Path(chars[1].anchor_path).stem: [raw(70, fails=("发型",), diff="第二位发型不符"), raw(95)],
    })
    r16 = S3.run(ctx8, dual, characters=chars, scene=scene, style=script.style,
                 image_provider=RecordingImage(), judge_provider=fixable_judge,
                 store=store8, ledger=ledger8, allow_paid=True)
    check("重取一轮 = 4 次判定（2 角色 × 2 轮）", fixable_judge.calls == 4, f"calls={fixable_judge.calls}")
    check("第二个人修好后整镜通过", r16.status == "videoing", r16.status)
    check("最终分取最后一轮的最差值（95）", r16.score == 95, f"score={r16.score}")
    store8.close()
    ledger8.close()

    store.close()
    store5.close()
    store6.close()
    ledger.close()
    ledger5.close()
    ledger6.close()

    print()
    print("=" * 64)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    if FAILED_NAMES:
        print("失败项：")
        for n in FAILED_NAMES:
            print("  -", n)
    print("=" * 64)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
