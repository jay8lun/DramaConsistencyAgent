"""项目阶段机：三个停点的落点（停 0 / 停 1 / 停 2）。

**为什么单独一个模块，而不是让 CLI 和 Web 各自判断**

三个停点会被三条入口共用（CLI、Web 接口、离线断言）。判断"现在该跑哪一段"
这件事只要有第二份实现，两边就会漂——而漂的方向一定是最坏的那个：
Web 以为该跑 S2 了，CLI 以为还在等确认，于是一个把用户改过的剧本覆盖掉，
另一个对着旧首帧跑视频。**阶段只有一个事实源。**

**落在 `project.json` 而不是内存**

`RunState`（`adapters/web/api.py`）是进程内字典，服务一重启就丢。2026-10-31 已经
因为这件事栽过一次（片子跑完了、报告页却打不开）。停点比那更要紧：它决定
"要不要花钱"，丢了就会在用户没确认的时候自己往下跑。

**旧项目不迁移**

`six01` / `bv2` 是在引入阶段机之前跑完的，`project.json` 里没有 `phase` 字段。
这里用 `infer()` 按产物推断一次，让它们照常工作（推断结果写回，之后就有字段了）。
不做批量迁移脚本：那是不可逆的批量写盘，为了省一次推断不值得。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.project import ProjectCtx, write_json
from pipeline.state import now_iso

#: 阶段取值。**顺序即进度顺序**，`save()` 用它做合法性校验。
PHASES: tuple[str, ...] = (
    "created",         # 刚建项目：还没问「角色有没有参考图」（停 0 的落点）
    "script_review",   # 停 1：剧本与角色卡已出，等人改完放行
    "anchor_review",   # 停 2：正脸已出，等人看脸（可带意见重出）
    "running",         # 侧背 → 首帧 → 视频 → 合成，一段到底
    "done",
    "failed",
)

LABELS: dict[str, str] = {
    "created": "等你选角色来源",
    "script_review": "等你确认角色与分镜",
    "anchor_review": "等你确认定妆照",
    "running": "生产中",
    "done": "已完成",
    "failed": "中断",
    # `stopped` **不在 PHASES 里**，只用于展示（2026-11-02）。
    # 叫停不是一个落盘阶段：磁盘上的 phase 仍是 running（片子确实没跑完），
    # "停过"这件事只有进程内知道。把它塞进 PHASES 会让转移表多出一堆
    # running↔stopped 的边，而那些边对"该不该花钱"没有任何影响。
    "stopped": "已停止（可继续）",
}

#: 允许的转移。**不含的转移一律拒绝**，不打印警告然后照做——
#: "表里没有但我还是走了"是状态机最常见的腐烂方式。
TRANSITIONS: dict[str, set[str]] = {
    "created": {"script_review", "failed"},
    # 从 script_review 回 created：用户改完又发现角色来源选错了（比如一开始
    # 选了"让程序生成"，后来找到参考图）。允许回去重选，代价只是重跑 S1。
    "script_review": {"anchor_review", "created", "failed"},
    # 回 script_review：看脸之后发现"这个人根本不对"，那是剧本写错了角色，
    # 不是图的问题。允许退回去改卡。
    "anchor_review": {"running", "script_review", "failed"},
    # 回 anchor_review：重出正脸。这是本轮新增的能力（带意见重出），必须允许。
    "running": {"done", "failed", "anchor_review"},
    "done": {"running", "anchor_review"},
    "failed": {"created", "script_review", "anchor_review", "running"},
}

#: 需人工放行的阶段。`resume()` 碰到这些阶段必须先拿到确认。
STOPS: frozenset[str] = frozenset({"created", "script_review", "anchor_review"})


class PhaseError(RuntimeError):
    """阶段转移不合法，或调用方在没有确认的情况下想越过停点。"""


@dataclass
class PhaseState:
    phase: str = "created"
    #: 角色从哪来：brief=程序按创意生成 / image=用户给了参考图
    role_source: str = "brief"
    #: 参考图相对路径（相对项目根），按 char_01、char_02 的顺序绑定
    ref_images: list[str] = field(default_factory=list)
    updated_at: str = ""
    note: str = ""

    @property
    def is_stop(self) -> bool:
        return self.phase in STOPS

    @property
    def label(self) -> str:
        return LABELS.get(self.phase, self.phase)


# ---------------------------------------------------------------- 读写


def _read_raw(ctx: ProjectCtx) -> dict[str, Any]:
    path = ctx.project_file
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 坏掉的 project.json 不该让整个项目打不开：按空处理，由 infer() 兜底。
        return {}


def load(ctx: ProjectCtx) -> PhaseState:
    """读阶段状态。没有 `phase` 字段的旧项目走 `infer()`。"""
    data = _read_raw(ctx)
    phase = str(data.get("phase") or "").strip()
    if phase not in PHASES:
        phase = infer(ctx)
    refs = data.get("ref_images") or []
    return PhaseState(
        phase=phase,
        role_source=str(data.get("role_source") or ("image" if refs else "brief")),
        ref_images=[str(x) for x in refs],
        updated_at=str(data.get("phase_updated_at") or ""),
        note=str(data.get("phase_note") or ""),
    )


def infer(ctx: ProjectCtx) -> str:
    """按磁盘产物推断阶段。**只在旧项目上跑**（没有 phase 字段时）。

    判据刻意从"最靠后"往"最靠前"问：有 final 就是 done，有单条视频就是 running，
    有正脸图就是 anchor_review，有剧本就是 script_review。
    反过来先问剧本，会把跑完的项目判成"等人改剧本"。
    """
    root: Path = ctx.root
    if (root / "final.mp4").exists():
        return "done"
    if any(ctx.shots_dir.glob("*.mp4")):
        return "running"
    if any(ctx.anchors_dir.glob("*_front.png")):
        return "anchor_review"
    if ctx.script_file.exists():
        return "script_review"
    return "created"


def save(
    ctx: ProjectCtx,
    *,
    phase: str | None = None,
    role_source: str | None = None,
    ref_images: list[str] | None = None,
    note: str | None = None,
    force: bool = False,
) -> PhaseState:
    """写阶段状态。`force=True` 跳转移校验（只在 `infer()` 写回时用）。"""
    cur = load(ctx)
    if phase is not None and phase != cur.phase:
        if phase not in PHASES:
            raise PhaseError(f"未知阶段：{phase!r}（可选 {list(PHASES)}）")
        if not force and phase not in TRANSITIONS.get(cur.phase, set()):
            raise PhaseError(
                f"不允许从 {cur.phase} 直接到 {phase}。"
                f"当前阶段可去：{sorted(TRANSITIONS.get(cur.phase, set())) or '无'}"
            )
        cur.phase = phase
    if role_source is not None:
        if role_source not in ("brief", "image"):
            raise PhaseError(f"role_source 只能是 brief / image，收到 {role_source!r}")
        cur.role_source = role_source
    if ref_images is not None:
        cur.ref_images = [str(x) for x in ref_images]
    if note is not None:
        cur.note = note
    cur.updated_at = now_iso()

    data = _read_raw(ctx)
    data.update({
        "phase": cur.phase,
        "role_source": cur.role_source,
        "ref_images": cur.ref_images,
        "phase_updated_at": cur.updated_at,
        "phase_note": cur.note,
    })
    write_json(ctx.project_file, data)
    return cur


# ---------------------------------------------------------------- 停点守卫


def require_confirmed(ctx: ProjectCtx, *, confirm: bool, action: str) -> PhaseState:
    """要越过停点必须显式确认。

    **不做"没确认就默认放行"**：那等于把人工闸门做成装饰。
    也不做"没确认就静默跳过"：用户会以为跑了，实际什么都没发生。
    两条都不行，只能明确报错并说清下一步该点哪里。
    """
    st = load(ctx)
    if st.is_stop and not confirm:
        raise PhaseError(
            f"项目 {ctx.project_id} 正停在「{st.label}」（phase={st.phase}），"
            f"要{action}必须先确认。"
            f"命令行加 --confirm；界面点「确认并继续」。"
        )
    return st


def advance_to(ctx: ProjectCtx, target: str, *, note: str = "") -> PhaseState:
    """转移并记一句 note（写进 project.json，界面和报告都能看到为什么走到这）。"""
    return save(ctx, phase=target, note=note)
