"""S3 · 首帧生成 + 判定 + 定向重取（第 6 章 S3 / 7.1 状态机）。

单镜闭环：生成首帧 → VLM 要素级判定 → 通过进视频 / 60~75 带增量重取 / <60 或超限降级。

三条最容易写错、也最容易在报告里被发现的规则：

1. **两种重试分开计数，各自的上限也不同**（7.1）：
   技术重试（超时/5xx/空响应）3 次，**原样重发**；质量重试（60~75）2 次，**必须带增量**。
   混用一个计数器，会让「平均重试次数」这个要拿去讲的数据直接失真。
2. **两种参考图不是同一张**（4.2 / 9.2 第 7 条）：
   - 生成参考 = `anchor_path`（重制过的干净定妆照，脏元素少）
   - 判定参考 = `truth_image` 优先，为空才退到 `anchor_path`（原图才是真实身份）
   混用会让判定拿一张已经漂移过的图当标准答案，等于放水。
3. **镜头里有几个人就判几次，取最差**（2026-10-29 补）：
   多角色镜头只判第一个人，等于第二个人完全没人看。ds2 第一镜就是这么漏掉
   一个剧本没要求的陌生人的（判定 95）。见 `_judge_all`。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from config import settings
from models import Character, Scene, Shot, VisualStyle
from pipeline import judge as J
from pipeline.cost import CostLedger
from pipeline.project import ProjectCtx
from pipeline.prompt_builder import assemble_image_prompt
from pipeline.state import StateStore, artifact_ok
from providers.base import ImageProvider, ProviderError, VisionJudge

#: 7.1：keyframing 的技术失败重试 3 次
MAX_TECH_RETRY = 3


@dataclass
class KeyframeResult:
    shot_id: str
    status: str  # videoing / degraded / failed / done / degraded(跳过)
    keyframe: Path | None = None
    score: int | None = None
    quality_used: int = 0
    tech_used: int = 0
    delta: str | None = None
    judge_reference: str | None = None
    #: 参考锚点是否来自 S2 自检降级的角色。为真时判定结论不可信——
    #: 这是 S2 degraded 会污染全片的那个隐患的落点，必须一路传到报告。
    anchor_untrusted: bool = False
    cost: float = 0.0
    note: str = ""


# ---------------------------------------------------------------- 参考图


def generate_references(shot: Shot, characters: Sequence[Character]) -> list[Path]:
    """生成用参考图：干净的定妆照优先，没有才退回原图。"""
    by_id = {c.char_id: c for c in characters}
    refs: list[Path] = []
    for cid in shot.character_ids:
        c = by_id.get(cid)
        if c is None:
            continue
        for candidate in (c.anchor_path, c.truth_image):
            if candidate and Path(candidate).exists():
                refs.append(Path(candidate))
                break
    return refs


def judge_reference(shot: Shot, characters: Sequence[Character]) -> Path | None:
    """判定用参考图（单张，取第一个有锚点的角色）。

    保留这个函数是因为 `KeyframeResult.judge_reference` 与落库记录里存的是单个路径，
    报告页按这个字段展示「这一镜是拿谁比的」。**它不再参与判定决策**——
    真正的判定走 `judge_references`（多角色，见下）。
    """
    refs = judge_references(shot, characters)
    return refs[0][1] if refs else None


def judge_references(
    shot: Shot, characters: Sequence[Character]
) -> list[tuple[str, Path]]:
    """判定用参考图（**每个角色一张**），返回 [(char_id, 路径)]。

    为什么必须逐个角色判（2026-10-29 真跑发现）：

    `shot.character_ids` 有两个角色的镜头，出图时两张定妆照都会挂上（provider 支持多图），
    但判定只拿第一个角色的锚点比了一次。结果第二个人长什么样、穿什么，**完全没被检查**。

    ds2 那次撞上了：第一镜的 `character_ids` 只有 char_01，画面右缘却站着一个剧本
    没要求的戴帽男性（场景描述里「镜面映出模糊人影」被模型实体化了）。判定给了 95 分——
    因为被判的只有 char_01，多出来的人不在任何一次比对里。

    顺序按 `shot.character_ids`，与 `generate_references` 保持一致：
    同一个下标在「出图挂图」和「判定比对」里指的是同一个人。
    """
    by_id = {c.char_id: c for c in characters}
    out: list[tuple[str, Path]] = []
    for cid in shot.character_ids:
        c = by_id.get(cid)
        if c is None:
            continue
        for candidate in (c.truth_image, c.anchor_path):
            if candidate and Path(candidate).exists():
                out.append((cid, Path(candidate)))
                break
    return out


def _image_estimate(provider: ImageProvider) -> float:
    if getattr(provider, "is_mock", False):
        return 0.0
    return float(getattr(provider, "unit_price", 0.0) or settings.price_image)


def _judge_estimate(provider: VisionJudge) -> float:
    if getattr(provider, "is_mock", False):
        return 0.0
    return float(settings.price_vlm_per_call)


# ---------------------------------------------------------------- 单步动作


def _generate_once(
    ctx: ProjectCtx,
    shot: Shot,
    *,
    characters: Sequence[Character],
    scene: Scene,
    style: VisualStyle,
    provider: ImageProvider,
    ledger: CostLedger,
    allow_paid: bool,
    delta: str | None,
) -> tuple[Path, float, list[str]]:
    """生成一张首帧。返回 (路径, 花费, 截断告警)。技术异常直接抛给调用方。"""
    # 先算好参考图，提示词要知道"这一次到底带没带图"——不带图却写
    # 「以参考图为准」，是指向不存在东西的空话，还占长度预算。
    refs = generate_references(shot, characters) or None
    prompt = assemble_image_prompt(shot, characters, scene, style, delta=delta,
                                   has_ref=bool(refs))
    out = ctx.keyframe_of(shot.shot_id)
    est = _image_estimate(provider)

    with ledger.paid(
        est,
        stage="keyframe",
        provider=provider.name,
        operation="generate",
        allow_paid=allow_paid,
        shot_id=shot.shot_id,
    ) as slot:
        provider.generate(
            prompt.text,
            out=out,
            references=refs,
            allow_paid=allow_paid,
        )
        slot.charge(est)

    # 非空 ≠ 可用：损坏的 png 会让 S5 合成时才炸，而那时已经花完全部的钱。
    if not artifact_ok(out, kind="image"):
        raise ProviderError(f"首帧产物不可用（形态校验失败）：{out}")

    warnings: list[str] = []
    if prompt.truncated:
        warnings.append(f"提示词被截断，删掉 {len(prompt.dropped)} 段修饰词")
    if prompt.still_over:
        warnings.append("提示词删完仍超长，需要人工压缩")
    if prompt.violations:
        warnings.append(f"视频提示词前瞻检查发现外貌描述 {list(prompt.violations)}")
    return out, est, warnings


def _judge_once(
    *,
    keyframe: Path,
    reference: Path | None,
    characters: Sequence[Character],
    shot_id: str,
    provider: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool,
) -> dict:
    """调一次 VLM。异常不向上抛——判定失败只能降级，不能让整镜消失。"""
    if reference is None:
        return {}
    est = _judge_estimate(provider)
    with ledger.paid(
        est,
        stage="judging",
        provider=provider.name,
        operation="compare",
        allow_paid=allow_paid,
        shot_id=shot_id,
    ) as slot:
        raw = provider.compare(keyframe, reference, list(J.ELEMENTS), allow_paid=allow_paid)
        slot.charge(est)
    return raw


def _judge_all(
    *,
    keyframe: Path,
    refs: Sequence[tuple[str, Path]],
    characters: Sequence[Character],
    shot_id: str,
    provider: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool,
    retries_used: int,
    max_retries: int,
) -> tuple[J.Decision, dict[str, int]]:
    """对每个角色各判一次，返回（最差的那个决定, 各角色分数）。

    **取最差而不是平均**：一个角色漂了就是这一镜不能要，平均分会被另一个
    正常的角色抬过通过线，等于用「另一个人没问题」掩盖「这个人错了」。

    每个角色的决定都用 `characters=[该角色]` 单独归一——因为
    `judge._expected_for` 的回落逻辑读的是 `characters[0].appearance`，
    传整个角色列表会让 char_02 的失败要素按 char_01 的角色卡去写增量，
    结果「改成了另一个人」。
    """
    by_id = {c.char_id: c for c in characters}
    per_char: list[tuple[str, J.Decision]] = []

    for cid, ref in refs:
        try:
            raw = _judge_once(
                keyframe=keyframe, reference=ref, characters=characters, shot_id=shot_id,
                provider=provider, ledger=ledger, allow_paid=allow_paid,
            )
        except Exception as exc:  # noqa: BLE001
            # 判定调用失败：拿不到分数就不能假装通过（9.2 第 4 条）。
            raw = {"score": 0, "checks": [], "diff": f"判定调用失败：{exc}"}
        char = by_id.get(cid)
        decision = J.judge(
            raw, characters=[char] if char else (),
            retries_used=retries_used, max_retries=max_retries,
        )
        per_char.append((cid, decision))

    scores = {cid: d.score for cid, d in per_char}
    worst_cid, worst = min(per_char, key=lambda p: p[1].score)

    if len(per_char) > 1:
        detail = "、".join(f"{cid}={d.score}" for cid, d in per_char)
        name = by_id[worst_cid].name if worst_cid in by_id else worst_cid
        qa = worst.qa
        if qa is not None:
            # 把「谁最差、其他人多少分」写进 diff，报告页能看出这一镜登过几个角色。
            # 逐角色分数不进 qa_payload —— save_qa 只落固定列，多传的键会被静默丢弃，
            # 所以走 diff 这个已有字段。
            note = f"多角色判定：{detail}"
            qa = qa.model_copy(
                update={"diff": f"{qa.diff}；{note}" if qa.diff else note}
            )
        # Decision 是 frozen dataclass，只能重建不能就地改。
        worst = replace(
            worst,
            reason=f"多角色取最差（{detail}），最差为 {name}｜{worst.reason}",
            qa=qa,
        )

    return worst, scores


# ---------------------------------------------------------------- 主流程


def run(
    ctx: ProjectCtx,
    shot: Shot,
    *,
    characters: Sequence[Character],
    scene: Scene,
    style: VisualStyle,
    image_provider: ImageProvider,
    judge_provider: VisionJudge,
    store: StateStore,
    ledger: CostLedger,
    allow_paid: bool = False,
    max_quality_retry: int | None = None,
    max_tech_retry: int = MAX_TECH_RETRY,
    untrusted_chars: set[str] | None = None,
    initial_delta: str | None = None,
) -> KeyframeResult:
    """跑完一镜的首帧阶段，状态机停在 `videoing` / `degraded` / `failed`。

    `initial_delta`（2026-11-02 加）：人工要求重出时写的那句"哪里不对"。
    它只在**状态库里的 delta 为空**时生效——续跑时库里已有的增量是上一轮
    判定给出的、针对具体画面的修正，比人手写的一句更具体，不能被顶掉。
    """
    pid = ctx.project_id
    sid = shot.shot_id
    max_quality = settings.max_retry_per_shot if max_quality_retry is None else max_quality_retry
    untrusted = bool(untrusted_chars and (set(shot.character_ids) & untrusted_chars))

    st = store.get(pid, sid)
    if st is None:
        raise ValueError(f"{sid} 未在状态库登记，S1 之后必须先 ensure_shots")

    if st.status in ("done", "degraded"):
        return KeyframeResult(shot_id=sid, status=st.status, cost=st.cost,
                              note="终态，跳过")
    if st.status in ("videoing", "video_judging", "awaiting_review"):
        # 时间上都已经过了首帧阶段（videoing 及之后）。**必须一起跳过**：
        # 2026-10-29 加了 video_judging / awaiting_review 两条状态后，这里漏改会让
        # 续跑时对 awaiting_review 的镜头再跑一遍首帧 → 非法迁移 awaiting_review → judging。
        return KeyframeResult(shot_id=sid, status=st.status,
                              keyframe=ctx.keyframe_of(sid) if ctx.keyframe_of(sid).exists() else None,
                              score=st.score, quality_used=st.quality_attempts,
                              tech_used=st.tech_attempts, cost=st.cost, note="已过首帧阶段，跳过")
    if st.status == "pending":
        st = store.transition(pid, sid, "keyframing", reason="开始生成首帧")

    delta: str | None = store.load_qa(pid, sid).get("delta") if store.load_qa(pid, sid) else None
    if not delta:
        delta = (initial_delta or "").strip() or None
    # 每个角色一张判定参考图。判定阶段逐个比，取最差（见 _judge_all）。
    refs = judge_references(shot, characters)
    ref = refs[0][1] if refs else None
    tech_used = 0
    score: int | None = None
    keyframe: Path | None = None

    # 续跑：judging 且分数已落库 → 直接按下过的判定决策，不重调 VLM。
    if st.status == "judging" and st.score is not None:
        record = store.load_qa(pid, sid)
        if record:
            decision = J.rebuild(record, characters=characters,
                                 retries_used=st.quality_attempts, max_retries=max_quality)
            keyframe = ctx.keyframe_of(sid) if ctx.keyframe_of(sid).exists() else None
            return _apply(pid, sid, decision, store=store, ctx=ctx, shot=shot,
                          keyframe=keyframe, tech_used=st.tech_attempts,
                          untrusted=untrusted, note="续跑：复用已落库的判定")

    while True:
        # ---------- 1. 生成（技术重试：原样重发，不带增量）----------
        generated = False
        for attempt in range(1, max_tech_retry + 1):
            try:
                keyframe, cost, warnings = _generate_once(
                    ctx, shot, characters=characters, scene=scene, style=style,
                    provider=image_provider, ledger=ledger, allow_paid=allow_paid, delta=delta,
                )
                store.add_cost(pid, sid, cost)
                technical_note = "；".join(warnings)
                generated = True
                break
            except Exception as exc:  # noqa: BLE001 —— 技术失败一律重试，不吞掉原因
                tech_used += 1
                store.mark_tech_attempt(pid, sid,
                                        reason=f"首帧技术失败（{attempt}/{max_tech_retry}）：{exc}",
                                        cost_delta=_image_estimate(image_provider))
                if attempt == max_tech_retry:
                    store.transition(pid, sid, "failed",
                                     reason=f"首帧连续 {max_tech_retry} 次技术失败：{exc}")
                    return KeyframeResult(shot_id=sid, status="failed", tech_used=tech_used,
                                          delta=delta, judge_reference=str(ref) if ref else None,
                                          anchor_untrusted=untrusted,
                                          cost=store.get(pid, sid).cost,
                                          note="技术重试耗尽")

        if not generated or keyframe is None:
            store.transition(pid, sid, "failed", reason="首帧未产出")
            return KeyframeResult(shot_id=sid, status="failed", tech_used=tech_used,
                                  cost=store.get(pid, sid).cost, note="首帧未产出")

        # ---------- 2. 判定 ----------
        # 空镜（environment_only）没有角色，`compare` 没有参考图可比。
        # 强行判只会得 0 分 → 每个空镜都被标红；造个假分数又违反 9.4 第 15 条。
        # 所以承认「这一镜不判定」，直接进视频（状态机为此留了 keyframing → videoing）。
        if shot.environment_only or not refs:
            st = store.transition(
                pid, sid, "videoing",
                reason="空镜：无角色参考图，跳过要素判定" if shot.environment_only
                else "缺少可用参考图（anchor/truth_image 均不存在），跳过判定并如实记录",
            )
            return KeyframeResult(
                shot_id=sid, status="videoing", keyframe=keyframe, score=None,
                quality_used=st.quality_attempts, tech_used=tech_used,
                anchor_untrusted=untrusted, cost=st.cost,
                note=("空镜，未做要素判定" if shot.environment_only
                      else "缺参考图，未做要素判定——该镜质量未被验证"),
            )

        st = store.transition(pid, sid, "judging",
                              reason=f"首帧已生成{f'（{technical_note}）' if technical_note else ''}")
        decision, _per_char_scores = _judge_all(
            keyframe=keyframe, refs=refs, characters=characters, shot_id=sid,
            provider=judge_provider, ledger=ledger, allow_paid=allow_paid,
            retries_used=st.quality_attempts, max_retries=max_quality,
        )
        score = decision.score

        qa_payload = decision.qa.model_dump() if decision.qa else {"score": score}
        # 注：逐角色分数不进 qa_payload —— `StateStore.save_qa` 只落固定列
        # （score/passed/degraded/diff/checks_json/delta/attempts），多传的键会被静默丢弃。
        # 所以「谁最差、其他人多少分」由 `_judge_all` 并进 diff 文本，走已有字段落库。
        if untrusted:
            qa_payload["diff"] = (qa_payload.get("diff") or "") + "；⚠ 参考锚点自检降级，判定结论不可信"
        store.save_qa(pid, sid, qa_payload, delta=decision.prompt_delta)
        store.set_score(pid, sid, score)

        applied = _apply(pid, sid, decision, store=store, ctx=ctx, shot=shot,
                         keyframe=keyframe, tech_used=tech_used,
                         untrusted=untrusted, note=technical_note)
        if applied.status != "keyframing":
            return applied

        # 质量重试：状态已回到 keyframing，下一轮必须带增量（7.1 硬规则第 6 条）
        delta = decision.prompt_delta
        if not delta:
            store.transition(pid, sid, "degraded",
                             reason="需要重取但增量生成为空，拒绝原样重跑")
            return KeyframeResult(shot_id=sid, status="degraded", keyframe=keyframe, score=score,
                                  quality_used=store.get(pid, sid).quality_attempts,
                                  tech_used=tech_used, anchor_untrusted=untrusted,
                                  cost=store.get(pid, sid).cost, note="增量为空，拒绝原样重跑")


def _apply(
    project_id: str,
    shot_id: str,
    decision: J.Decision,
    *,
    store: StateStore,
    ctx: ProjectCtx,
    shot: Shot,
    keyframe: Path | None,
    tech_used: int,
    untrusted: bool,
    note: str,
) -> KeyframeResult:
    """把判定动作落到状态机上。三种出口一一对应 7.1 的三个分支。"""
    if decision.action is J.Action.PASS:
        st = store.transition(project_id, shot_id, "videoing",
                              reason=f"{decision.score} 分通过：{decision.reason}")
        return KeyframeResult(shot_id=shot_id, status="videoing", keyframe=keyframe,
                              score=decision.score, quality_used=st.quality_attempts,
                              tech_used=tech_used, anchor_untrusted=untrusted,
                              cost=st.cost, note=note)

    if decision.action is J.Action.RETRY:
        st = store.transition(project_id, shot_id, "keyframing", quality_retry=True,
                              reason=f"{decision.score} 分，带增量重取：{decision.prompt_delta}")
        return KeyframeResult(shot_id=shot_id, status="keyframing", keyframe=keyframe,
                              score=decision.score, quality_used=st.quality_attempts,
                              tech_used=tech_used, delta=decision.prompt_delta,
                              anchor_untrusted=untrusted, cost=st.cost, note=note)

    st = store.transition(project_id, shot_id, "degraded", reason=decision.reason)
    return KeyframeResult(shot_id=shot_id, status="degraded", keyframe=keyframe,
                          score=decision.score, quality_used=st.quality_attempts,
                          tech_used=tech_used, anchor_untrusted=untrusted,
                          cost=st.cost, note=note)
