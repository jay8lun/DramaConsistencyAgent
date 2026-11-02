"""S2 · 定妆照生成 + 自检（第 6 章）。

产出 `anchors/{char_id}_{front,side,back}.png`，并做一次自检：三视图是不是同一人、
服装是否一致、是否真是正/侧/背三个视角。

**为什么这一步也要自检**：定妆照是全程锚点。正面对而侧面不像，后面每一镜都拿它比，
错误会被复制到全片。这一步自检约 ¥0.25，视频阶段返工一次 ¥5 起——
在便宜的地方修问题，是成本纪律（6 章原话）。

自检不合格重出，最多 2 次；仍不合格标 `degraded` 但继续（6 章）。
**degraded 必须落盘**（`anchors/_status.json`）：锚点不可信时，后续 S3 的判定结果
也跟着不可信，这个事实不能只活在内存里——否则报告里只会写「S2 降级」，
没人知道 S3 的那些「通过」是拿一张错的参考图比出来的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from config import settings
from models import Character, VisualStyle
from pipeline import judge as J
from pipeline.cost import CostLedger
from pipeline.project import ProjectCtx, write_json
from pipeline.state import now_iso
from providers.base import ImageProvider, VisionJudge

VIEWS: tuple[str, ...] = ("front", "side", "back")

#: 三视图的取景说明。**不写「正面」「侧面」两个字就够**——要写清身体朝向，
#: 否则模型会把「侧面」画成「四分之三侧脸」，自检必然判不过。
VIEW_NOTES: dict[str, str] = {
    "front": "正面全身定妆照，身体正对镜头，双臂自然下垂，面部完整可见",
    "side": "正侧面全身定妆照，身体完全朝向右方，侧脸轮廓与耳廓清晰可见",
    "back": "背面全身定妆照，背对镜头，后脑发型与服装背面完整可见",
}

STATUS_FILE = "_status.json"

#: 有参考图时必须写进提示词的那一段（2026-11-02 补）。
#:
#: **为什么非加不可**：`references=` 只是把图喂给模型，提示词里一个字不提它，
#: 模型就把"图"和"文字角色卡"当成两个独立线索各自采纳一部分——2026-11-02
#: 的现场结果是：参考图是黄色卡通吉祥物，角色卡写"23-27 岁人类男性、肤色白皙"，
#: 生成图既是写实真人（听文字的）、脖子上又有一片黄（听图的）。
#: 两边都没满足，得到一个谁都没要求过的第三种东西。
#:
#: 所以这段的核心不是"提到参考图"，而是**定优先级**：冲突时听图的。
#: 只写"参考参考图"没有用，模型会继续折中；必须写明"以参考图为准"。
REF_LOCK = (
    "严格以随附参考图中的人物为唯一造型依据：脸型、发型与发色、肤色、体型、"
    "服装与配色全部照参考图；参考图里非人类的特征（例如身体或皮肤的固有色）"
    "也必须原样保留，不得替换成人类肤色；画风同样照参考图（参考图是卡通或 3D "
    "渲染就保持该画风），不要改成真人实拍。"
    "**下面的文字描述若与参考图冲突，一律以参考图为准。**"
)


@dataclass
class AnchorResult:
    char_id: str
    status: str  # ok / degraded
    attempts: int
    min_score: int | None
    paths: dict[str, str] = field(default_factory=dict)
    cost: float = 0.0
    reason: str = ""


def build_view_prompt(
    char: Character, style: VisualStyle, view: str, *, feedback: str = "",
    has_ref: bool = False,
) -> str:
    """单张定妆照的提示词。与镜头提示词同源（都用 appearance/wardrobe），
    但**不带动作与场景**——定妆照是纯角色参考，混入剧情会让后续比对失真。

    `feedback` 是停 2 里人填的那句"哪里不满意"（2026-10-31 加）。

    **为什么拼在最后而不是替换原句**：反馈是"增量修正"，原角色卡仍然是主体。
    如果拿反馈直接顶掉 appearance，用户填一句"脸再瘦点"就会把发型、肤色、
    服装全丢掉——生成结果会变成另一个人，而界面上看不出任何异常。

    `has_ref`（2026-11-02 加）：这一次是不是带了参考图。带了就要把 `REF_LOCK`
    插进去定优先级；没带就绝不能出现"以参考图为准"这种指向
    不存在东西的句子——文生图时它会变成一句无约束的空话，白白稀释有效描述。

    主体句走 `Character.subject_phrase()`：`age_range` 是人可编辑的自由文本，
    直接当形容词拼会拼出"女不是真实人无年龄段的豆包"。
    """
    parts = [
        style.style_prompt.strip(),
        char.subject_phrase(),
        char.appearance,
        f"穿着{char.wardrobe}",
        VIEW_NOTES[view],
        "纯色背景，影棚平光，单人，画幅 9:16",
        "ultra detailed, 高细节",
    ]
    if has_ref:
        # 插在风格词之后、人物描述之前：让它先立规矩，后面的 appearance
        # 才是在"参考图已定调"的前提下补充细节，而不是与图并列的另一个版本。
        parts.insert(1, REF_LOCK)
    fb = (feedback or "").strip()
    if fb:
        # 用「在此基础上调整」而不是「改为」：后者会被模型当成整体重写。
        parts.append(f"在上述设定基础上调整：{fb}")
    return "，".join(p for p in parts if p)


def _image_estimate(provider: ImageProvider) -> float:
    """估算单张图成本。mock 一律 0——否则零成本的验证链路会被预算封顶拦下。"""
    if getattr(provider, "is_mock", False):
        return 0.0
    return float(getattr(provider, "unit_price", 0.0) or settings.price_image)


def _judge_estimate(provider: VisionJudge) -> float:
    if getattr(provider, "is_mock", False):
        return 0.0
    return float(settings.price_vlm_per_call)


def _selfcheck(
    ctx: ProjectCtx,
    char: Character,
    front: Path,
    view_path: Path,
    *,
    judge_provider: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool,
) -> int:
    """拿侧/背视图与正面对比，返回分数（0~100）。"""
    with ledger.paid(
        _judge_estimate(judge_provider),
        stage="anchor_check",
        provider=judge_provider.name,
        operation="compare",
        allow_paid=allow_paid,
    ) as slot:
        raw = judge_provider.compare(view_path, front, list(J.ELEMENTS), allow_paid=allow_paid)
        slot.charge(_judge_estimate(judge_provider))
    # max_retries=0：自检不做质量重取，只给一个分数，重出由外层控制。
    return J.judge(raw, characters=[char], retries_used=0, max_retries=0).score


def update_status(ctx: ProjectCtx, char_id: str, **fields: object) -> dict:
    """读-改-写 `_status.json` 的单个角色条目。

    **为什么要"改"而不是"整表覆盖"**：停 2 阶段先写 `front_status` 与
    `feedback_log`，侧背出完后 `run()` 又要写 `status`/`min_score`。
    整表覆盖会让前一批字段凭空消失——而"这张脸是人确认过的"这个事实一旦丢了，
    下游就会把一张没人看过的图当锚点用，且不报错。
    """
    data = load_status(ctx)
    cur = dict(data.get(char_id) or {})
    cur.update(fields)
    data[char_id] = cur
    write_json(ctx.anchors_dir / STATUS_FILE, data)
    return cur


def confirm_front(ctx: ProjectCtx, char_id: str, *, note: str = "") -> dict:
    """人认可了这张正脸（停 2 的「这张可以」）。"""
    return update_status(
        ctx, char_id, front_status="confirmed", front_confirmed_at=now_iso(),
        front_note=note,
    )


def generate_front(
    ctx: ProjectCtx,
    char: Character,
    *,
    style: VisualStyle,
    image_provider: ImageProvider,
    ledger: CostLedger,
    allow_paid: bool = False,
    feedback: str = "",
    references: list[Path] | None = None,
) -> Path:
    """只出正脸一张（停 2 之前那一步）。**不出侧背、也不跑自检。**

    不跑自检的理由有两层：

    1. 自检的实现是「拿侧背与正脸比」，此刻侧背还不存在，无从比；
    2. 就算能比，它也只约束「侧背像不像正脸」，**对正脸本身零约束**。
       一张丑脸只要侧背照着它画，自检照样给高分。2026-10-31 之前
       三视图是一次性出的，所以这个盲区一直没暴露——现在拆成两段，
       正脸单独停下来给人看，正好把这个洞补上。

    `references` 用于「用户给了参考图」这条入口（停 0 选"我有参考图"）：
    把用户原图当参考图传进去，生成的脸才会像他给的那个人。
    """
    front = ctx.anchor_of(char.char_id, "front")
    use_ref = bool(references)
    prompt = build_view_prompt(char, style, "front", feedback=feedback, has_ref=use_ref)
    with ledger.paid(
        _image_estimate(image_provider),
        stage="anchor",
        provider=image_provider.name,
        operation="generate_front",
        allow_paid=allow_paid,
    ) as slot:
        image_provider.generate(
            prompt, out=front, references=references or None, allow_paid=allow_paid,
        )
        slot.charge(_image_estimate(image_provider))

    prev = load_status(ctx).get(char.char_id) or {}
    log = list(prev.get("feedback_log") or [])
    fb = (feedback or "").strip()
    if fb:
        log.append({"at": now_iso(), "text": fb})
    update_status(
        ctx, char.char_id,
        front_status="pending",
        front_path=str(front),
        front_attempts=int(prev.get("front_attempts") or 0) + 1,
        feedback_log=log,
        front_prompt=prompt,
    )
    return front


def generate_views(
    ctx: ProjectCtx,
    char: Character,
    *,
    style: VisualStyle,
    image_provider: ImageProvider,
    ledger: CostLedger,
    allow_paid: bool = False,
    front: Path | None = None,
) -> dict[str, Path]:
    """出侧背（停 2 之后那一步），**以已确认的那张正脸为参考图**。

    正脸换了而侧背还是旧的，是本项目最隐蔽的一类错：三张图各自都正常、
    自检也可能过，但侧脸属于另一个人。所以这里强制要求 `front` 存在，
    并且在重出正脸时由调用方负责清掉旧侧背（`clear_views`）。
    """
    front = Path(front or ctx.anchor_of(char.char_id, "front"))
    if not front.exists():
        raise FileNotFoundError(
            f"{char.char_id} 没有正脸图，侧背必须先有正脸：{front}"
        )
    paths: dict[str, Path] = {"front": front}
    for view in ("side", "back"):
        out = ctx.anchor_of(char.char_id, view)
        with ledger.paid(
            _image_estimate(image_provider),
            stage="anchor",
            provider=image_provider.name,
            operation=f"generate_{view}",
            allow_paid=allow_paid,
        ) as slot:
            image_provider.generate(
                # 侧背同样是"以参考图为准"：这里的参考图就是刚被确认的正脸。
                # 不写这句的话，侧脸只靠 appearance 文字重新描述一遍，
                # 等于让模型自由发挥——而自检只能事后打分，发现不了"两句话描述不同"。
                build_view_prompt(char, style, view, has_ref=True),
                out=out,
                references=[front],
                allow_paid=allow_paid,
            )
            slot.charge(_image_estimate(image_provider))
        paths[view] = out
    return paths


def clear_views(ctx: ProjectCtx, char_id: str) -> list[str]:
    """删掉该角色的侧背图（正脸重出时必须调）。

    返回被删的路径，便于审计。**不做"忘了删"的容错**：调用方必须在重出正脸时
    显式调用它，而不是指望下游自己发现图不对——下游没有这个判断能力。
    """
    removed: list[str] = []
    for view in ("side", "back"):
        p = ctx.anchor_of(char_id, view)
        if p.exists():
            p.unlink()
            removed.append(str(p))
    if removed:
        update_status(ctx, char_id, views_done=False, status="pending")
    return removed


def run(
    ctx: ProjectCtx,
    characters: Sequence[Character],
    *,
    style: VisualStyle,
    image_provider: ImageProvider,
    judge_provider: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool = False,
    max_retry: int = 2,
) -> tuple[list[Character], dict[str, AnchorResult]]:
    """为每个角色出三视图并自检。返回更新过 `anchor_path` 的角色卡与自检报告。"""
    updated: list[Character] = []
    results: dict[str, AnchorResult] = {}

    for char in characters:
        cost = 0.0
        best_score: int | None = None
        last_reason = ""
        paths: dict[str, str] = {}

        for attempt in range(1, max_retry + 2):  # 首次 + 最多 2 次重出
            # 正面先出，侧/背以正面为参考——这是保证「同一个人」的唯一手段。
            front = generate_front(
                ctx, char, style=style, image_provider=image_provider,
                ledger=ledger, allow_paid=allow_paid,
            )
            view_paths = generate_views(
                ctx, char, style=style, image_provider=image_provider,
                ledger=ledger, allow_paid=allow_paid, front=front,
            )
            paths = {k: str(v) for k, v in view_paths.items()}
            cost += 3 * _image_estimate(image_provider)

            scores = [
                _selfcheck(ctx, char, front, Path(paths[v]),
                           judge_provider=judge_provider, ledger=ledger, allow_paid=allow_paid)
                for v in ("side", "back")
            ]
            best_score = min(scores)
            if best_score >= J.PASS_LINE:
                break
            last_reason = f"自检 {scores} 未达 {J.PASS_LINE:g}（第 {attempt} 次）"

        status = "ok" if best_score is not None and best_score >= J.PASS_LINE else "degraded"
        results[char.char_id] = AnchorResult(
            char_id=char.char_id,
            status=status,
            attempts=attempt,
            min_score=best_score,
            paths=paths,
            cost=round(cost, 4),
            reason="" if status == "ok" else f"重出 {max_retry} 次仍不合格：{last_reason}，降级继续",
        )
        updated.append(
            char.model_copy(update={
                "anchor_path": paths.get("front"),
                "anchor_views": [paths[v] for v in VIEWS if v in paths],
            })
        )

    write_json(ctx.characters_file, [c.model_dump() for c in updated])
    for cid, r in results.items():
        # **合并写，不整表覆盖**：停 2 阶段已经落盘的 front_status / feedback_log
        # 会被整表覆盖吃掉，而「这张脸是人确认过的」这个事实一旦丢了，
        # 下游就会把没人看过的图当锚点用，且不报错。
        update_status(ctx, cid, status=r.status, attempts=r.attempts,
                      min_score=r.min_score, reason=r.reason, views_done=True)
    return updated, results


def run_views(
    ctx: ProjectCtx,
    characters: Sequence[Character],
    *,
    style: VisualStyle,
    image_provider: ImageProvider,
    judge_provider: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool = False,
    max_retry: int = 2,
) -> tuple[list[Character], dict[str, AnchorResult]]:
    """停 2 之后那一段：只出侧背 + 自检，**正脸已由人确认，不再重出**。

    这跟 `run()` 的差别不只是"少一张图"，而是**重试的代价不一样**：

    ```
    run()      失败一轮 = 正脸 + 侧背 + 背 = 3 张 = ¥0.60
    run_views  失败一轮 = 侧背 + 背       = 2 张 = ¥0.40
    ```

    更关键的是最后一轮：`run()` 最坏可能重出一张人没看过的正脸，
    而这里重试多少次，正脸始终是原来那张。
    """
    updated: list[Character] = []
    results: dict[str, AnchorResult] = {}

    for char in characters:
        front = ctx.anchor_of(char.char_id, "front")
        if not front.exists():
            raise FileNotFoundError(
                f"{char.char_id} 没有正脸图，不能出侧背。"
                f"先跑 S2a（python run.py --project-id ... --stop-after s2a）"
            )

        best_score: int | None = None
        last_reason = ""
        cost = 0.0
        paths: dict[str, str] = {}

        for attempt in range(1, max_retry + 2):
            view_paths = generate_views(
                ctx, char, style=style, image_provider=image_provider,
                ledger=ledger, allow_paid=allow_paid, front=front,
            )
            paths = {k: str(v) for k, v in view_paths.items()}
            cost += 2 * _image_estimate(image_provider)
            scores = [
                _selfcheck(ctx, char, front, Path(paths[v]),
                           judge_provider=judge_provider, ledger=ledger, allow_paid=allow_paid)
                for v in ("side", "back")
            ]
            best_score = min(scores)
            if best_score >= J.PASS_LINE:
                break
            last_reason = f"自检 {scores} 未达 {J.PASS_LINE:g}（第 {attempt} 次）"

        status = "ok" if best_score is not None and best_score >= J.PASS_LINE else "degraded"
        results[char.char_id] = AnchorResult(
            char_id=char.char_id, status=status, attempts=attempt, min_score=best_score,
            paths=paths, cost=round(cost, 4),
            reason="" if status == "ok" else f"重出 {max_retry} 次仍不合格：{last_reason}，降级继续",
        )
        update_status(
            ctx, char.char_id, status=status, views_attempts=attempt,
            min_score=best_score, reason="" if status == "ok" else last_reason,
            views_done=True,
        )
        updated.append(
            char.model_copy(update={
                "anchor_path": str(front),
                "anchor_views": [paths[v] for v in VIEWS if v in paths],
            })
        )

    write_json(ctx.characters_file, [c.model_dump() for c in updated])
    return updated, results


def load_status(ctx: ProjectCtx) -> dict:
    """读自检状态。S3 用它判断「这一镜的参考锚点是否可信」。"""
    path = ctx.anchors_dir / STATUS_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def degraded_chars(ctx: ProjectCtx) -> set[str]:
    """自检降级的角色 id。S3 判定时必须把这些镜头的结论标为「参考锚点不可信」。

    **只认 `degraded`/`failed`，不认 `pending`。** 停 2 阶段会先写
    `front_status=pending`，此时 `status` 字段还没写（侧背没出、自检没跑）。
    如果沿用原来的 `status != "ok"`，这些"正常在等确认"的角色会被判成降级——
    结论是「参考锚点不可信」，而真实原因是「流程还没走到那一步」。
    """
    return {
        cid for cid, s in load_status(ctx).items()
        if s.get("status") in ("degraded", "failed")
    }
