"""S4 · 图生视频 + 末帧回比 + 人工确认（第 6 章 S4 / 9.1 硬规则第 1 条）。

**只允许 i2v**：`VideoProvider.generate` 的第一个参数就是 `keyframe: Path`，
签名里没有「纯文本出片」的分支。这不是约定，是 D1（画面一致性）的代码落点——
没有首帧就没有锚点，文生视频等于每镜重新抽卡。

三种终态的处置不同，容易写错：
- `videoing` → 出片成功 → 末帧判定 → `done` / `awaiting_review`；技术失败 → `failed`（7.1）
- `degraded` → **仍然要出片**（7.2「degraded 进合成」），但状态**保持 degraded**。
  `degraded` 是终态、不再迁出；成片里包含它，报告里标红。
  如果为了「出片成功」把它改成 done，那份「哪一镜是凑合的」记录就没了。
- `awaiting_review` → **新增（2026-10-29）**：末帧判定不达标时停下来等人看，
  不自动重跑。

## 为什么加末帧判定

计划书 S4 原本没有判定，只有「API 失败重试 2 次」。这意味着流水线默认
「首帧对了，视频就一定对」——而 15.2 单独验过这条假设并不总成立：
i2v 在中途崩脸、换装、丢人，都是真实发生过的失败模式，而它们**不会被任何现有环节发现**，
坏画面直接进合成。

## 为什么不达标就自动重跑，而是停下等人

一条 10 秒视频（720P）约 ¥2，重跑一次就是再花一次钱；而「脸是不是还能看」
这件事，VLM 给的是一个连续分数，阈值卡在临界点时它自己也未必比人准。
拿 ¥2 去赌一个自己都拿不准的判断，不如让人看一眼——**人看一眼是 5 秒，重跑是 2 分钟加 ¥2**。

所以这里的规则是：**VLM 只做「要不要打扰人」的筛子，最终放不放行由人定**。
`awaiting_review` 是一个真实停靠态（不是进程残留），重启不会清掉它。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import find_bin, settings
from models import Character, Shot
from pipeline import judge as J
from pipeline.cost import CostLedger
from pipeline.project import ProjectCtx
from pipeline.prompt_builder import assemble_video_prompt, snap_duration
from pipeline.state import StateStore, artifact_ok
from providers.base import (
    NonRetryableProviderError,
    ProviderError,
    VideoProvider,
    VisionJudge,
)

#: 7.1：videoing 的技术失败重试 2 次（不是 3 次，keyframing 才是 3 次）
MAX_TECH_RETRY = 2

#: 抽末帧时从「时长 - 该值」处取。留 0.15s 是因为最后一帧在部分编码里是空帧/黑帧。
LAST_FRAME_BACKOFF_SEC = 0.15


@dataclass
class VideoResult:
    shot_id: str
    status: str
    video: Path | None = None
    tech_used: int = 0
    cost: float = 0.0
    prompt: str = ""
    note: str = ""
    #: 末帧判定分数。None = 没判（空镜 / 判定失败 / 降级镜头跳过）
    video_score: int | None = None


def _estimate(provider: VideoProvider, duration: float) -> float:
    """整段视频的估算价。mock 一律 0。"""
    if getattr(provider, "is_mock", False):
        return 0.0
    per_sec = float(getattr(provider, "price_per_second", 0.0) or settings.price_video_per_sec)
    return round(per_sec * duration, 4)


def _extract_last_frame(video: Path, out: Path, duration: float) -> Path:
    """用 ffmpeg 抽末帧。失败抛 ProviderError（由调用方决定是否阻塞整镜）。"""
    ff = find_bin("ffmpeg")
    if not ff:
        raise ProviderError("找不到 ffmpeg，无法抽末帧")
    ss = max(0.0, duration - LAST_FRAME_BACKOFF_SEC)
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{ss:.3f}", "-i", str(video),
           "-frames:v", "1", "-q:v", "2", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError as exc:
        raise ProviderError(f"抽末帧失败：{exc.stderr[:200]!r}") from exc
    if not artifact_ok(out, kind="image"):
        raise ProviderError(f"末帧不可用：{out}")
    return out


def _judge_last_frame(
    *,
    ctx: ProjectCtx,
    sid: str,
    keyframe: Path,
    video: Path,
    duration: float,
    characters: Sequence[Character],
    provider: VisionJudge | None,
    ledger: CostLedger,
    allow_paid: bool,
) -> tuple[int | None, dict]:
    """末帧 vs 首帧。返回 (分数, 落库用 qa dict)；判定不可用时返回 (None, {...})。

    比的是**首帧**，不是定妆照：这一步要回答的是「动起来之后漂了没有」，
    而首帧是这一镜的起点，用它当基准才能把「视频阶段引入的差异」单独隔离出来。
    拿定妆照比会把「首帧本来就有的角度/景别差异」也算进视频头上。
    """
    import config as _config

    if provider is None:
        # 没配判定器（例如只跑 S4 的单测）：不抽帧、不判定，明确记录「未验证」。
        return None, {"score": -1, "diff": "未配置判定器，跳过末帧判定", "last_frame": None}

    last = ctx.root / "shots" / f".{sid}_lastframe.jpg"
    try:
        _extract_last_frame(video, last, duration)
    except Exception as exc:  # noqa: BLE001
        return None, {"score": -1, "diff": f"抽末帧失败：{exc}", "last_frame": None}

    est = 0.0 if getattr(provider, "is_mock", False) else float(_config.settings.price_vlm_per_call)
    try:
        with ledger.paid(
            est, stage="video_judging", provider=provider.name, operation="compare",
            allow_paid=allow_paid, shot_id=sid,
        ) as slot:
            raw = provider.compare(last, keyframe, list(J.ELEMENTS), allow_paid=allow_paid)
            slot.charge(est)
    except Exception as exc:  # noqa: BLE001
        return None, {"score": -1, "diff": f"末帧判定调用失败：{exc}", "last_frame": str(last)}

    # 用 judge() 归一（校验要素齐全、算 diff），但**不采纳它的 action**：
    # 这里的出口不是「重试/降级」，而是「达标 → done」「不达标 → 等人看」。
    decision = J.judge(raw, characters=characters, retries_used=0, max_retries=0)
    qa = decision.qa.model_dump() if decision.qa else {"score": decision.score}
    qa["last_frame"] = str(last)
    return decision.score, qa


def _finish_video(
    ctx: ProjectCtx,
    sid: str,
    *,
    keyframe: Path,
    out: Path,
    seconds: float,
    characters: Sequence[Character],
    judge_provider: VisionJudge | None,
    store: StateStore,
    ledger: CostLedger,
    allow_paid: bool,
    tech_used: int,
    cost: float,
    prompt: str,
    notes: list[str],
) -> VideoResult:
    """视频已落盘之后的收尾：末帧判定 → done / awaiting_review。

    单独提出来，是因为它有**两个入口**：刚出完片（videoing）、以及续跑时视频已在
    （video_judging）。两条路都不该重出视频。
    """
    pid = ctx.project_id
    st = store.get(pid, sid)

    if st is not None and st.status == "degraded":
        # 降级镜头已标记过，再判一次只会多花钱，且它的结论进不了合成决策。
        return VideoResult(sid, "degraded", out, tech_used=tech_used, cost=cost,
                           prompt=prompt, note="；".join(notes))

    if st is not None and st.status == "videoing":
        store.transition(pid, sid, "video_judging", reason=f"视频已出（{seconds:g}s），做末帧判定")

    score, qa = _judge_last_frame(
        ctx=ctx, sid=sid, keyframe=keyframe, video=out, duration=seconds,
        characters=characters, provider=judge_provider, ledger=ledger, allow_paid=allow_paid,
    )
    store.save_video_qa(pid, sid, qa)

    if score is None:
        # 判定不可用（抽帧失败/调用失败/未配判定器）：不能假装通过，也不能因为
        # 一个额外检查挂了就把已经花钱做出来的视频丢掉。放行 + 明确留痕。
        reason = f"末帧未判定：{qa.get('diff', '')}"
        store.transition(pid, sid, "done", reason=reason)
        return VideoResult(sid, "done", out, tech_used=tech_used, cost=cost, prompt=prompt,
                           note="；".join([*notes, f"⚠ {reason}（该镜视频未经一致性验证）"]))

    if score >= J.PASS_LINE:
        store.transition(pid, sid, "done", reason=f"末帧判定 {score} ≥ {J.PASS_LINE:g}，通过")
        return VideoResult(sid, "done", out, tech_used=tech_used, cost=cost, prompt=prompt,
                           video_score=score, note="；".join([*notes, f"末帧 {score} 分"]))

    # 不达标：**停下等人看**，不自动重跑（理由见模块 docstring）
    store.transition(
        pid, sid, "awaiting_review",
        reason=f"末帧判定 {score} < {J.PASS_LINE:g}，等待人工确认（不自动重跑）",
    )
    return VideoResult(sid, "awaiting_review", out, tech_used=tech_used, cost=cost, prompt=prompt,
                       video_score=score, note="；".join([*notes, f"末帧 {score} 分，待人工确认"]))


def run(
    ctx: ProjectCtx,
    shot: Shot,
    *,
    characters: Sequence[Character],
    provider: VideoProvider,
    store: StateStore,
    ledger: CostLedger,
    allow_paid: bool = False,
    max_tech_retry: int = MAX_TECH_RETRY,
    judge_provider: VisionJudge | None = None,
) -> VideoResult:
    pid, sid = ctx.project_id, shot.shot_id
    st = store.get(pid, sid)
    if st is None:
        raise ValueError(f"{sid} 未在状态库登记")

    out = ctx.video_of(sid)

    if st.status == "done":
        return VideoResult(sid, "done", out if out.exists() else None, note="已完成，跳过")
    if st.status == "failed":
        return VideoResult(sid, "failed", note="failed 终态，不进合成")
    if st.status == "awaiting_review":
        return VideoResult(sid, "awaiting_review", out if out.exists() else None,
                           note="等待人工确认（--review 查看 / --decide 处理）")
    if st.status == "degraded" and artifact_ok(out, kind="video"):
        return VideoResult(sid, "degraded", out, note="降级镜头已出片，跳过")

    keyframe = ctx.keyframe_of(sid)

    # 续跑：视频已在、判定没跑完 → 只补判定，绝不重出（一条约 ¥2）
    if st.status == "video_judging" and artifact_ok(out, kind="video"):
        if not artifact_ok(keyframe, kind="image"):
            store.transition(pid, sid, "failed", reason="补判定时缺首帧，无法比对")
            return VideoResult(sid, "failed", out, note="缺首帧，末帧无法比对")
        snap0 = snap_duration(shot.duration)
        return _finish_video(
            ctx, sid, keyframe=keyframe, out=out, seconds=snap0.seconds,
            characters=characters, judge_provider=judge_provider, store=store,
            ledger=ledger, allow_paid=allow_paid, tech_used=0, cost=0.0,
            prompt=shot.video_prompt or "", notes=["续跑：视频已存在，只补判定"],
        )

    if st.status not in ("videoing", "degraded"):
        # 还没过首帧阶段（pending/keyframing/judging）：S4 不越权替他生成首帧
        return VideoResult(sid, st.status, note=f"尚未到出片阶段（{st.status}），跳过")

    if not artifact_ok(keyframe, kind="image"):
        # 没有首帧就不能 i2v（9.1 第 1 条）。videoing 状态没有别的出路，只能 failed。
        store.transition(pid, sid, "failed", reason="出片阶段缺可用首帧，无法 i2v")
        return VideoResult(sid, "failed", note="缺首帧，拒绝文生视频")

    snap = snap_duration(shot.duration)
    prompt = shot.video_prompt or assemble_video_prompt(shot, characters).text
    # 8.2 的第 2 条禁止是可执行断言：视频提示词里出现外貌描述就是数据有问题。
    violations = assemble_video_prompt(shot, characters).violations

    est = _estimate(provider, snap.seconds)
    tech_used = 0
    last_error: Exception | None = None

    for attempt in range(1, max_tech_retry + 1):
        try:
            with ledger.paid(
                est, stage="video", provider=provider.name, operation="generate",
                allow_paid=allow_paid, shot_id=sid,
            ) as slot:
                provider.generate(
                    keyframe, prompt, duration=snap.seconds, out=out, allow_paid=allow_paid,
                )
                slot.charge(est)
            if not artifact_ok(out, kind="video"):
                raise ProviderError(f"视频产物不可用（形态校验失败）：{out}")

            store.add_cost(pid, sid, est)
            notes = [snap.note] if snap.needs_split else []
            if violations:
                notes.append(f"视频提示词含外貌描述 {list(violations)}，需回 S1 修 action")
            # 出片只是「有视频了」；状态怎么走由末帧判定决定（见 _finish_video）。
            # degraded 镜头在 _finish_video 里直接放行，不额外判定。
            return _finish_video(
                ctx, sid, keyframe=keyframe, out=out, seconds=snap.seconds,
                characters=characters, judge_provider=judge_provider, store=store,
                ledger=ledger, allow_paid=allow_paid, tech_used=tech_used, cost=est,
                prompt=prompt, notes=notes,
            )
        except NonRetryableProviderError as exc:
            # **不重试**（2026-10-29）：参数不兼容、鉴权失败、额度耗尽这类错误，
            # 重发一次和重发一百次结果相同。原来它们混在下面的 `except Exception` 里
            # 一起重试，六镜那次在 shot_05 上白等了约 6 分钟。
            # 记一次技术尝试（用于报告里区分"试过但没救"和"根本没试"），然后立刻放弃这一镜。
            tech_used += 1
            last_error = exc
            store.mark_tech_attempt(
                pid, sid, reason=f"不可重试错误，立即失败（未重试）：{exc}",
                cost_delta=0.0,
            )
            break
        except Exception as exc:  # noqa: BLE001
            tech_used += 1
            last_error = exc
            # 失败不扣费，因此**不往镜头成本里加钱**（cost_delta=0）。
            # 原实现写的是 cost_delta=est，导致 shot_05 那三次「10 秒不被支持」的
            # 400 拒绝被记成 ¥2.0×3，报告里这一镜显示 ¥7.02，而整部片子实扣只有
            # ¥6.40——明细比总数还大，账自己打自己。估算额在 costs 表里仍可查。
            store.mark_tech_attempt(pid, sid, reason=f"出片技术失败（{attempt}/{max_tech_retry}）：{exc}",
                                    cost_delta=0.0)
            if attempt == max_tech_retry:
                break

    if st.status == "videoing":
        store.transition(pid, sid, "failed",
                         reason=f"出片连续 {max_tech_retry} 次技术失败：{last_error}")
    return VideoResult(sid, "failed" if st.status == "videoing" else "degraded",
                       tech_used=tech_used, cost=0.0, prompt=prompt,
                       note=f"技术重试耗尽：{last_error}")
