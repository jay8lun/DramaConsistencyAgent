"""Web 层：3 个接口 + 单页 5 视图（计划书 5.1 / 7.5）。

**三条硬约束，写在最前面，改这个文件之前先读：**

1. **禁止 `BackgroundTasks`**（7.5）。流水线跑在一个独立的 `threading.Thread` 里，
   理由很具体：`BackgroundTasks` 绑定在请求生命周期上，响应发出去之后它的
   取消语义、异常可见性都不受我们控制；而这条流水线要跑 15~30 分钟、要能断点续跑、
   出错必须留下痕迹。用线程，状态自己管。

2. **禁止 `uvicorn --reload`**（7.3.1）。reload 会在文件变动时重启 worker，
   而重启意味着"在途的那一镜"被砍掉重做——B 通道只是多花钱，C2 通道是重复提交。
   开发期改前端请直接刷新浏览器。

3. **状态只存 SQLite**，进程里那点内存（`RUNS`）只是"这一轮跑到哪了"的
   临时标记，丢了不影响正确性：重启后 `reconcile` 会按文件系统把状态摆正（7.3）。

三个接口与视图的对应关系：首页 → POST /projects，进度视图 → 轮询 GET /status，
报告视图 → GET /artifacts。操作台的人工裁决走 POST /decide。
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

import config as C
import run as R
from models import Character, Script
from pipeline import phase as P
from pipeline import project as PR
from pipeline import ref_desc as RD
from pipeline.cost import CostLedger
from pipeline.state import StateStore

#: 参考图最短边的平台限制。**只从 provider 那边取一次**，Web 层不另写一个数字——
#: 两处各写一个值，改了一处忘另一处，就会出现"上传时说没问题、出图时 400"。
from providers.dashscope_image import REF_MIN_SIDE

app = FastAPI(title="DramaConsistencyAgent", docs_url="/api/docs")

ROOT = C.PROJECTS_DIR


# ---------------------------------------------------------------- 运行态
class RunState:
    """一次运行的内存标记。**不是事实来源**——事实在 SQLite 和文件系统里。"""

    __slots__ = ("phase", "error", "started_at", "finished_at", "pid_alive", "mock",
                 "busy", "busy_char", "busy_shot", "stopped", "cancel")

    def __init__(self) -> None:
        self.phase: str = "idle"        # idle / running / stopped / done / failed / awaiting_review
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.pid_alive: bool = False
        self.mock: bool = False
        #: 有一张正脸正在重出。**与 phase 分开**：重出只是一张图，不是整条流水线在跑。
        #: 借 phase="running" 表达会有两个后果——status 报"生产中"（页面被甩去进度页），
        #: 以及"重出会不会被误当成流水线在跑"取决于读代码的人记不记得这层借用。
        self.busy: bool = False
        #: 正在重出哪个角色。只暴露 busy 不够：界面上有两张卡，全禁掉会让人
        #: 以为"两个人的图都在换"，而实际只动了一个。
        self.busy_char: str | None = None
        #: 正在重跑哪个镜头（复用 busy，但要知道是哪一镜，界面才能只禁那一行）。
        self.busy_shot: str | None = None
        #: 用户点了「停止」。**只在本次进程内有效**：进程重启后没人记得停过，
        #: 而磁盘上的 phase 仍是 running（片子确实没跑完），语义正好对上——
        #: 重启后本字段为 False，界面显示"已中断，点继续接着跑"。
        self.stopped: bool = False
        #: 叫停信号。用 Event 而不是 bool 是因为它跨线程：请求线程置位，
        #: 流水线线程在镜与镜之间读。bool 也能读，但 Event 让"等待/超时"以后可加。
        self.cancel: threading.Event = threading.Event()


RUNS: dict[str, RunState] = {}
_LOCK = threading.Lock()


def _state(pid: str) -> RunState:
    with _LOCK:
        return RUNS.setdefault(pid, RunState())



def _load_existing(pid: str) -> Any:
    """读一个**已存在**的项目；不存在就 404，绝不顺手创建。

    `create_or_load` 的语义是「没有就建」——直接拿它写 GET 接口会出两个问题：
    ① 查询一个拼错的 id 会真的在磁盘上建出一个空项目；
    ② 前端轮询路径写错时，后端不报错，只是永远返回空状态，问题被藏起来。
    **读接口不能有副作用。**
    """
    PR.validate_project_id(pid)
    if not (C.project_dir(pid) / "project.json").exists():
        raise HTTPException(status_code=404, detail=f"项目 {pid} 不存在")
    ctx, _ = PR.create_or_load(pid, budget_cap=C.settings.budget_cap,
                               mock=False, channel="api", force=False)
    return ctx


# ---------------------------------------------------------------- 请求体


class CreateBody(BaseModel):
    creative: str = Field(min_length=2, max_length=500)
    shots: int = Field(default=6, ge=1, le=12)
    budget: float = Field(default=10.0, gt=0, le=1000)
    mock: bool = False
    channel: str = Field(default="api")
    #: 停 0 的答案：brief=程序按创意生成 / image=用户会给参考图
    role_source: str = Field(default="brief", pattern="^(brief|image)$")


class RunBody(BaseModel):
    # 付费授权必须显式传（D7 / 9.4）。前端是一个勾选框，默认不勾。
    allow_paid: bool = False
    #: 越过当前停点。**停点没有这一项就不会往下走**——默认 False 是刻意的：
    #: 让"忘了传"变成"停在原地"，而不是"悄悄花了钱"。
    confirm: bool = False


class RegenFrontBody(BaseModel):
    """停 2 的「重出」：只重出一张正脸。"""

    char_id: str = Field(pattern=r"^char_\d{2}$")
    feedback: str = Field(default="", max_length=300)
    #: 是否把这条意见同时写进角色卡。**默认 False**——写卡会影响所有含该角色的镜头，
    #: 而"这张不行，再来一张"和"人物设定不对"是两件不同的事，不能混在一个默认里。
    update_card: bool = False
    allow_paid: bool = False


class AnchorConfirmBody(BaseModel):
    """停 2 的「这张可以」。逐角色确认，不是一次性全放行。"""

    char_id: str = Field(pattern=r"^char_\d{2}$")
    note: str = ""


class ScriptBody(BaseModel):
    """整份 script.json 回传。**不做字段级补丁**：局部更新在嵌套结构上极易漏改，
    而漏改的表现是"界面上改了、存盘的没改"，比直接报错难查得多。"""

    script: dict[str, Any]


class RefBody(BaseModel):
    """参考图：base64 直传。不引 multipart 依赖，也让前端无需 FormData。"""

    filename: str = Field(default="ref.png", max_length=120)
    data_base64: str = Field(min_length=16)


class DecideBody(BaseModel):
    shot_id: str
    decision: str = Field(pattern="^(accept|degrade|reject)$")
    note: str = ""


class AttributeBody(BaseModel):
    """C2 归因（10.4）。取值限定为四个，**不许自由发挥**——它直接决定重试策略。"""

    shot_id: str
    reason: str = Field(pattern="^(face_drift|wrong_action|blurry|model_limit)$")
    note: str = ""


ATTRIBUTION_LABEL = {
    "face_drift": "脸漂了（降低动作幅度，重填参数）",
    "wrong_action": "动作不对（改一句动作描述）",
    "blurry": "画面糊了（删掉一半形容词）",
    "model_limit": "这样也行（标 degraded 归档）",
}


# ---------------------------------------------------------------- 流水线执行


def _execute(pid: str, *, allow_paid: bool, confirm: bool = False) -> None:
    """在独立线程里推进流水线**一段**（到下一个停点为止）。所有异常都必须落进 RunState.error。

    **不再是一口气跑完**（2026-10-31 改）。`run_pipeline` 按 `project.json` 里的
    phase 决定跑哪一段：S1→停、S2a→停、S2b 之后才一路到成片。
    前端每次点「确认并继续」就调一次本函数，`confirm=True`。
    """
    st = _state(pid)
    st.phase, st.error, st.started_at = "running", None, time.time()
    store = None
    ctx = None
    try:
        ctx, _ = PR.create_or_load(pid, budget_cap=C.settings.budget_cap,
                                   mock=_state(pid).mock, channel="api", force=False)
        req_file = ctx.root / "request.json"
        req = json.loads(req_file.read_text(encoding="utf-8")) if req_file.exists() else {}
        args = SimpleNamespace(
            project_id=pid,
            creative=req.get("creative", ""),
            shots=int(req.get("shots", 6)),
            force=False,
            paid_ok=bool(allow_paid or ctx.mock),
            mock=bool(ctx.mock),
            confirm=confirm,
            auto=False,
            stop_after=None,
            regen_front=None,
            feedback="",
            update_card=False,
            role_source=None,
            ref=None,
        )
        text_p, image_p, video_p, judge_p = R.build_providers(ctx.mock)
        store = StateStore()
        ledger = CostLedger(pid, cap=ctx.budget_cap)

        code = R.run_pipeline(ctx, args, ledger=ledger, store=store, text_p=text_p,
                              image_p=image_p, video_p=video_p, judge_p=judge_p,
                              should_stop=st.cancel.is_set)
        st.phase = _resolve_phase(ctx, store, code)
    except R.Cancelled as exc:
        # **叫停不是失败。** 磁盘上的 phase 保持 running：片子确实没跑完，
        # 改了这一点的话，"继续"会以为该从停 2 重新开始，把已经出好的首帧再买一遍。
        # 只把内存态标成 stopped，界面据此显示"已停止，可继续"。
        st.phase = "stopped"
        st.stopped = True
        st.error = None
        if ctx is not None:
            try:
                P.save(ctx, note=f"用户停止：{exc}")
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001 —— 后台线程必须吞掉一切并留痕
        st.phase = "failed"
        st.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        if ctx is not None:
            try:
                P.save(ctx, phase="failed", note=f"{type(exc).__name__}: {exc}")
            except Exception:  # noqa: BLE001
                pass
    finally:
        st.finished_at = time.time()
        st.pid_alive = False
        if store is not None:
            store.close()


def _resolve_phase(ctx: Any, store: StateStore, code: int) -> str:
    """跑完之后对外报什么阶段。**以磁盘为准，内存只做覆盖。**"""
    if code != 0:
        return "failed"
    ph = P.load(ctx)
    if ph.phase == "running" and store.awaiting_review(ctx.project_id):
        return "awaiting_review"
    return ph.phase


# ---------------------------------------------------------------- 接口


@app.post("/api/projects")
def create_project(body: CreateBody) -> dict[str, Any]:
    """POST /api/projects：建项目并把创意写进 project.json（不调任何模型）。

    **建完不直接开跑**（2026-10-31 改）：先停在「停 0 · 角色有参考图吗」。
    以前是建完就一路跑到底，用户在写下角色卡之前没有任何插话机会——
    而角色卡是后面每一张图的依据，改它等于全片重来。
    """
    pid = f"web{int(time.time())}"
    ctx, _ = PR.create_or_load(pid, budget_cap=body.budget, mock=body.mock,
                              channel=body.channel, force=True)
    opts = {"creative": body.creative, "shots": body.shots,
            "mock": body.mock, "budget": body.budget, "channel": body.channel,
            "role_source": body.role_source}
    PR.write_json(ctx.root / "request.json", opts)
    P.save(ctx, phase="created", role_source=body.role_source, force=True)
    st = _state(pid)
    st.mock = body.mock
    return {"project_id": pid, "project": str(ctx.root), "request": opts,
            "phase": "created", "phase_label": P.LABELS["created"]}


@app.post("/api/projects/{pid}/run")
def run_project(pid: str, body: RunBody) -> dict[str, Any]:
    """推进一段流水线。**同一项目同时只允许一个线程**，重复点会被告知正在跑。

    `confirm=true` 才越过当前停点。不带它调这个接口会得到 409 + 当前停在哪，
    而不是"看起来成功了但什么都没发生"。
    """
    st = _state(pid)
    if st.phase == "running":
        raise HTTPException(status_code=409, detail="该项目正在运行中，请等待完成")
    ctx = _load_existing(pid)
    req_file = ctx.root / "request.json"
    if req_file.exists():
        opts = json.loads(req_file.read_text(encoding="utf-8"))
        RUNS[pid].mock = bool(opts.get("mock"))
    ph = P.load(ctx)
    if ph.is_stop and not body.confirm:
        raise HTTPException(
            status_code=409,
            detail=f"项目停在「{ph.label}」（phase={ph.phase}），要继续必须 confirm=true",
        )
    st.phase = "running"
    st.error = None
    st.pid_alive = True
    # 上一轮叫停留下的信号必须清掉，否则"点了继续 → 下一镜之前立刻又停"。
    # 这个坑只要踩一次就会表现得像"继续按钮坏了"。
    st.cancel.clear()
    st.stopped = False
    threading.Thread(target=_execute, args=(pid,),
                     kwargs={"allow_paid": body.allow_paid, "confirm": body.confirm},
                     daemon=True, name=f"pipeline-{pid}").start()
    return {"project_id": pid, "phase": "running", "confirm": body.confirm}


def _characters_brief(ctx: Any) -> list[dict[str, Any]]:
    """角色卡 + 正脸状态，给停 1（改卡）和停 2（看脸）共用。

    **`front_status` 与 `status` 是两个不同的东西**，不能合并：
    `front_status` 是"这张脸人确认了没"，`status` 是"侧背自检过没过"。
    停 2 时只有前者有意义——侧背还没出，后者压根不存在。
    """
    if not ctx.characters_file.exists():
        return []
    try:
        chars = json.loads(ctx.characters_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    try:
        s2 = json.loads((ctx.anchors_dir / "_status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        s2 = {}
    out: list[dict[str, Any]] = []
    for c in chars:
        cid = str(c.get("char_id", ""))
        stt = s2.get(cid) or {}
        front = ctx.anchor_of(cid, "front")
        has_front = front.exists()
        out.append({
            "char_id": cid,
            "name": c.get("name", ""),
            "gender": c.get("gender", ""),
            "age_range": c.get("age_range", ""),
            "appearance": c.get("appearance", ""),
            "wardrobe": c.get("wardrobe", ""),
            "personality": c.get("personality", ""),
            "source": c.get("source", "brief"),
            "has_front": has_front,
            # 前端加 ?t= 破坏缓存，否则重出之后浏览器还是显示旧图
            "front_url": (f"/files/{ctx.project_id}/anchors/{front.name}" if has_front else None),
            "front_status": stt.get("front_status", ""),
            "front_attempts": int(stt.get("front_attempts") or 0),
            "feedback_log": stt.get("feedback_log") or [],
            "views_done": bool(stt.get("views_done")),
        })
    return out


@app.get("/api/projects/{pid}/status")
def project_status(pid: str) -> dict[str, Any]:
    """GET /api/projects/{id}/status：全量状态（各镜 status/score/cost + 当前阶段）。

    前端每 2 秒轮询这个接口（5.1：不用 WebSocket）。
    """
    store = StateStore()
    try:
        # **只有"项目不存在"才 404。** "存在但一镜还没跑"是正常状态，不是错误。
        #
        # 2026-10-31 踩到：这里原本 `if not states: raise 404`，结果刚建好、流水线
        # 还没写出第一条镜头记录时，接口直接 404——而 404 会把同一响应里的
        # `phase` 和 `error` 一起吞掉，界面上只剩一句"没有镜头记录"，
        # **最该看的失败原因反而拿不到**。用户看到的是"跑完了啥也没有"。
        if not (C.project_dir(pid) / "project.json").exists():
            raise HTTPException(status_code=404, detail=f"项目 {pid} 不存在")
        states = store.all(pid)
        summary = store.summary(pid)
        # 上限要取**这个项目**的 budget_cap，不是全局 settings。
        # 2026-10-31 修：这里原本写死 `C.settings.budget_cap`，于是界面上
        # "上限"永远是 ¥100，而项目实际预算是 9.5——用户按这个数字判断
        # "还能再跑几镜"会错得离谱，而且这个错误看起来完全正常。
        cap = float(C.settings.budget_cap)
        try:
            proj = json.loads((C.project_dir(pid) / "project.json").read_text(encoding="utf-8"))
            cap = float(proj.get("budget_cap", cap))
        except (OSError, ValueError):
            pass
        ledger = CostLedger(pid, cap=cap)
        cs = ledger.summary()
        pending = store.awaiting_review(pid)
        qa = {s.shot_id: store.load_qa(pid, s.shot_id) for s in states}
        vqa = {s.shot_id: store.load_video_qa(pid, s.shot_id) for s in states}
        run = _state(pid)
        ctx, _ = PR.create_or_load(pid, budget_cap=cap, mock=False, channel="api", force=False)
        ph = P.load(ctx)
        # **阶段以磁盘为准，内存只覆盖「正在跑」和「刚出错」。**
        # RunState 是进程内字典，服务一重启就丢；而 phase 决定"要不要花钱"，
        # 它丢了就会在用户没确认的时候自己往下跑。（2026-10-31 的报告页打不开
        # 是同一类问题，只是那次丢的是"能不能看报告"，这次丢的是"该不该花钱"。）
        if run.phase == "running":
            phase = "running"
        elif run.phase == "stopped":
            # 用户叫停。磁盘仍是 running（片子没跑完），但"有没有线程在跑"
            # 是**进程内才知道**的事实，界面靠它区分"还在生产"和"停在这里等你"。
            phase = "stopped"
        elif run.phase == "failed" and run.error:
            phase = "failed"
        elif pending:
            phase = "awaiting_review"
        else:
            phase = ph.phase
        reruns = R.read_reruns(ctx)
        last_reason = {}
        for r in reruns:                       # 只取每镜最后一条，界面一行放得下
            last_reason[r.get("shot_id", "")] = r.get("reason", "")
        return {
            "project_id": pid,
            "phase": phase,
            "phase_label": P.LABELS.get(phase, phase),
            "role_source": ph.role_source,
            "ref_count": len(ph.ref_images),
            "mock": ctx.mock,
            "error": run.error,
            # 重出状态要能从磁盘之外读到：刷新页面后，界面上唯一能说明
            # "这张正在换"的证据就是这个字段。只靠前端本地变量，换个标签就丢。
            "regenerating": bool(run.busy),
            "regenerating_char": run.busy_char,
            # 有没有一个线程真的在跑这一段。**不能只看 phase**：
            # 服务重启后磁盘 phase 仍是 running，而进程里一个线程都没有，
            # 界面会一直显示"生产中"并永远轮询下去，人只能干等。
            "alive": run.phase == "running",
            # 点了停止、但当前这一镜还没跑完。界面据此把按钮改成"正在停止…"，
            # 而不是让人以为没点上，反复点。
            "stopping": bool(run.cancel.is_set()) and run.phase == "running",
            "rerunning": run.busy_shot,
            "elapsed": (round((run.finished_at or time.time()) - run.started_at, 1)
                        if run.started_at else None),
            "counts": summary["counts"],
            "total": summary["total"],
            "cost": {"spent": cs["actual_total"], "estimated": cs["estimated_total"],
                     "cap": cs["cap"], "calls": cs["calls"],
                     "failed_calls": cs["failed_calls"],
                     "by_stage": cs["by_stage"]},
            "awaiting_review": [{"shot_id": r["shot_id"], "video_score": r.get("video_score")}
                                for r in pending],
            "shots": [
                {
                    "shot_id": s.shot_id,
                    "status": s.status,
                    "score": s.score,
                    "tech_attempts": s.tech_attempts,
                    "quality_attempts": s.quality_attempts,
                    "cost": round(s.cost, 4),
                    "diff": (qa.get(s.shot_id) or {}).get("diff", ""),
                    "video_score": (vqa.get(s.shot_id) or {}).get("score"),
                    "video_diff": (vqa.get(s.shot_id) or {}).get("diff", ""),
                    "human_decision": (vqa.get(s.shot_id) or {}).get("human_decision", ""),
                    # 这一镜最近一次重出的理由（reruns.json 里的最后一条）。
                    # 界面上要能回答"这镜我上次说哪里不对"，只靠记忆会重复踩同一个坑。
                    "rerun_reason": last_reason.get(s.shot_id, ""),
                }
                for s in sorted(states, key=lambda x: x.shot_id)
            ],
            "characters": _characters_brief(ctx),
        }
    finally:
        store.close()


@app.get("/api/projects/{pid}/artifacts")
def project_artifacts(pid: str) -> dict[str, Any]:
    """GET /api/projects/{id}/artifacts：产物清单 + 可直接访问的静态 URL。"""
    ctx = _load_existing(pid)

    def url(p: Path) -> str:
        return f"/files/{pid}/{p.name}" if p.parent == ctx.root else \
               f"/files/{pid}/{p.parent.name}/{p.name}"

    def listing(sub: str, pattern: str) -> list[dict[str, str]]:
        d = ctx.root / sub
        if not d.exists():
            return []
        return [{"name": p.stem, "url": f"/files/{pid}/{sub}/{p.name}"}
                for p in sorted(d.glob(pattern)) if not p.name.startswith(".")]

    final = ctx.root / "final.mp4"
    return {
        "project_id": pid,
        "final": ({"url": f"/files/{pid}/final.mp4",
                   "size_mb": round(final.stat().st_size / 1048576, 2)}
                  if final.exists() else None),
        "report": (f"/files/{pid}/report.html"
                   if (ctx.root / "report.html").exists() else None),
        "costs": (f"/files/{pid}/costs.json"
                  if (ctx.root / "costs.json").exists() else None),
        "anchors": listing("anchors", "*.png"),
        "keyframes": listing("keyframes", "*.png"),
        "shots": listing("shots", "*.mp4"),
        "audio": (f"/files/{pid}/audio.m4a" if (ctx.root / "audio.m4a").exists() else None),
        "subtitles": (f"/files/{pid}/subtitles.srt"
                      if (ctx.root / "subtitles.srt").exists() else None),
    }


@app.post("/api/projects/{pid}/decide")
def decide(pid: str, body: DecideBody) -> dict[str, Any]:
    """人工裁决一个 `awaiting_review` 镜头（对应 CLI 的 --decide）。"""
    with _LOCK:
        rs = RUNS.get(pid)
    store = StateStore()
    try:
        st = store.get(pid, body.shot_id)
        if st is None:
            raise HTTPException(status_code=404, detail=f"{body.shot_id} 不在项目 {pid} 里")
        if st.status != "awaiting_review":
            raise HTTPException(status_code=409,
                                detail=f"{body.shot_id} 当前是 {st.status}，"
                                       "只有 awaiting_review 可以裁决")
        target = {"accept": "done", "degrade": "degraded", "reject": "videoing"}[body.decision]
        label = {"accept": "放行", "degrade": "放行但标红", "reject": "驳回重出"}[body.decision]
        reason = f"人工{label}" + (f"：{body.note}" if body.note else "")
        store.transition(pid, body.shot_id, target, reason=reason)
        vqa = store.load_video_qa(pid, body.shot_id) or {}
        store.save_video_qa(pid, body.shot_id, {
            **vqa,
            "human_decision": "accept" if body.decision == "accept" else
                              ("accept_degraded" if body.decision == "degrade" else "reject"),
            "human_note": body.note,
        })
        if rs is not None:
            rs.phase = "idle" if body.decision != "reject" else "idle"
        return {"project_id": pid, "shot_id": body.shot_id,
                "status": target, "reason": reason}
    finally:
        store.close()


@app.post("/api/projects/{pid}/attribute")
def attribute(pid: str, body: AttributeBody) -> dict[str, Any]:
    """C2 归因（10.4）：把人在单镜页点的那个按钮转成结构化记录。

    四种症状对应四种**相反**的处理，所以归因必须先落记录、再决定动作：
    脸漂要降动作幅度、画面糊要删形容词——记错了，重试就会往反方向使劲。
    """
    _load_existing(pid)   # 只校验存在，不读内容
    store = StateStore()
    try:
        st = store.get(pid, body.shot_id)
        if st is None:
            raise HTTPException(status_code=404, detail=f"{body.shot_id} 不在项目 {pid} 里")
        label = ATTRIBUTION_LABEL[body.reason]
        note = f"{label}" + (f"：{body.note}" if body.note else "")
        store.log_note(pid, body.shot_id, "c2_attribution", note)
        return {"project_id": pid, "shot_id": body.shot_id,
                "reason": body.reason, "label": label,
                "next": "已记录。reject 重试会走 submitter 护栏，不绕开 12/30 上限"}
    finally:
        store.close()


@app.get("/api/projects/{pid}/script")
def project_script(pid: str) -> dict[str, Any]:
    """视图 2（角色与分镜确认）用的剧本原文。

    直接回传 `script.json` 的解析结果，**不做字段裁剪**——前端要能编辑，
    裁剪会让"界面上看到的"与"存盘的"不一致。
    """
    ctx = _load_existing(pid)
    if not ctx.script_file.exists():
        raise HTTPException(status_code=404, detail="剧本还没生成（S1 未完成）")
    import json
    return json.loads(ctx.script_file.read_text(encoding="utf-8"))


@app.put("/api/projects/{pid}/script")
def save_script(pid: str, body: ScriptBody) -> dict[str, Any]:
    """停 1 的「保存」：整份回写剧本与角色卡。

    **强类型校验后才落盘。** 不校验等于让前端随手写的 JSON 直接变成 S2/S3 的输入，
    而结构错误不会在写入时报错——它会在二十秒后某张图生成失败时才暴露，
    那时人已经忘了自己改过什么。

    **改长相/服装会作废该角色的旧锚点**（变更传播）。只改了第 3 镜的动作则不动锚点：
    这两件事的代价差一个数量级，不能一律处理。
    """
    ctx = _load_existing(pid)
    ph = P.load(ctx)
    # 停 2 也允许改：看过脸才发现"这个人根本不对"是最常见的路径，
    # 挡住它等于逼用户重建项目。改完若作废了锚点，会自动退回停 1（见下）。
    if ph.phase not in ("created", "script_review", "anchor_review"):
        raise HTTPException(
            status_code=409,
            detail=f"当前阶段「{ph.label}」不允许改剧本"
                   "（只在停 1 或停 2 看脸时改）",
        )
    try:
        script = Script(**body.script)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"剧本结构不合法：{exc}") from exc

    old: dict[str, dict] = {}
    if ctx.characters_file.exists():
        try:
            old = {c["char_id"]: c
                   for c in json.loads(ctx.characters_file.read_text(encoding="utf-8"))}
        except (OSError, json.JSONDecodeError, KeyError):
            old = {}

    # 人点了「保存并确认」= 这些卡已经过目。`low_confidence_fields` 是给
    # **还没看的人**的提示，核对完不清掉的话黄框会永远挂着，下次再看就分不清
    # "我核对过了"和"我还没看"。原始置信度仍在 refs/desc_*.json 里，可追溯。
    acked = [c.char_id for c in script.characters if c.low_confidence_fields]
    for c in script.characters:
        c.low_confidence_fields = []

    PR.write_json(ctx.script_file, script.model_dump())
    PR.write_json(ctx.characters_file, [c.model_dump() for c in script.characters])

    removed: list[str] = []
    for c in script.characters:
        o = old.get(c.char_id)
        if not o:
            continue
        changed = [k for k in ("appearance", "wardrobe", "age_range", "gender")
                   if (o.get(k) or "") != getattr(c, k)]
        if not changed:
            continue
        for view in ("front", "side", "back"):
            p = ctx.anchor_of(c.char_id, view)
            if p.exists():
                p.unlink()
                removed.append(p.name)
        R.S2.update_status(ctx, c.char_id, front_status="", status="pending",
                           views_done=False, invalidated=changed)

    # 作废了锚点就把阶段退回停 1。**不退回的后果很具体**：停在 anchor_review
    # 而正脸图已经删了，下一次 --confirm 会直接去出侧背，`generate_views`
    # 找不到正脸就抛错——错在"没图"，而真实原因是"设定改了、该重出正脸"。
    if removed and ph.phase == "anchor_review":
        P.save(ctx, phase="script_review",
               note=f"角色设定变更，作废 {len(removed)} 张锚点，退回停 1 重出")

    return {"project_id": pid, "shots": len(script.shots),
            "characters": len(script.characters), "invalidated": removed,
            "confirmed_marks": acked}


def _regen_worker(pid: str, body: RegenFrontBody) -> None:
    """后台重出一张正脸。出错写进 RunState.error（前端轮询就能看到）。

    **开跑第一件事是清掉上一轮的 error**（2026-11-02 修）。原来只在成功时清，
    失败留下的旧 error 会一直挂着；而前端轮询用 `if(d.error)` 判断，是硬判断——
    于是"上一次失败了"会让"这一次刚开始"立刻被读成"这一次也失败了"。
    旧账不清，就永远只有第一次失败是真的。

    `st.busy` 在跑的时候置真：`regenerate_front` 靠它拦重复提交
    （连点两下「重出」会真的扣两次钱）。**不用 `st.phase="running"`**——
    status 接口见 `run.phase == "running"` 就对外报"生产中"，于是重出一张脸的
    二十秒里，界面会说自己正在跑整条流水线，用户一刷新就被甩到进度页。
    一张图的重出不该冒充一次流水线运行。
    """
    st = _state(pid)
    st.busy, st.busy_char, st.error = True, body.char_id, None
    try:
        ctx = _load_existing(pid)
        _, image_p, _, _ = R.build_providers(ctx.mock)
        args = SimpleNamespace(
            project_id=pid, creative="", shots=6, force=False,
            paid_ok=bool(body.allow_paid or ctx.mock), mock=bool(ctx.mock),
            confirm=False, auto=False, stop_after=None,
            regen_front=body.char_id, feedback=body.feedback,
            update_card=body.update_card, role_source=None, ref=None,
        )
        ledger = CostLedger(pid, cap=ctx.budget_cap)
        try:
            R.run_regen_front(ctx, args, ledger=ledger, image_provider=image_p)
        finally:
            ledger.close()
        st.error = None
    except Exception as exc:  # noqa: BLE001 —— 后台线程必须吞掉一切并留痕
        st.error = f"重出失败：{type(exc).__name__}: {exc}"
    finally:
        st.busy = False
        st.busy_char = None


@app.post("/api/projects/{pid}/anchor/confirm")
def confirm_anchor(pid: str, body: AnchorConfirmBody) -> dict[str, Any]:
    """停 2 的「这张可以」：逐角色确认，**不全确认就不解锁下一步**。

    逐张而不是一次性：两个角色很可能一个行、一个不行。一次性放行会逼用户
    要么全收、要么全部重出，而重出一张 ¥0.20、重出六张 ¥1.20。
    """
    ctx = _load_existing(pid)
    ph = P.load(ctx)
    if ph.phase != "anchor_review":
        raise HTTPException(status_code=409,
                            detail=f"当前阶段「{ph.label}」不在看脸环节")
    front = ctx.anchor_of(body.char_id, "front")
    if not front.exists():
        raise HTTPException(status_code=404,
                            detail=f"{body.char_id} 还没有正脸图，先出图再确认")

    if _state(pid).busy:
        # 重出进行中不许确认。20 秒里人完全来得及点「这张可以」，
        # 而那张图下一秒就被覆盖——确认记录留下，图却不是被确认的那张。
        raise HTTPException(status_code=409, detail="这张正在重出，等新图出来再确认")

    R.S2.confirm_front(ctx, body.char_id, note=body.note)
    chars = R._load_characters(ctx)
    st = R.S2.load_status(ctx)
    done = [c.char_id for c in chars
            if (st.get(c.char_id) or {}).get("front_status") == "confirmed"]
    return {"project_id": pid, "char_id": body.char_id, "confirmed": True,
            "confirmed_ids": done, "all_confirmed": len(done) == len(chars)}


@app.post("/api/projects/{pid}/anchor/regenerate")
def regenerate_front(pid: str, body: RegenFrontBody) -> dict[str, Any]:
    """停 2 的「重出」：只重出一张正脸，可选把意见写进角色卡。

    走后台线程：出图约 20 秒，同步返回会让前端一直转圈；而前端本来就每 2 秒
    轮询一次 status，交给它更自然——重出完成后 `front_attempts` 会 +1，界面自己更新。

    **只在停 2 允许**（2026-11-02 收紧）。原来还放行 running / done，那是错的：
    这两个阶段的首帧、视频、成片都是拿**旧脸**生成的，而重出只换掉锚点图、
    不动任何下游产物。结果就是「界面显示新脸、片子里是旧脸」，而且全程不报错——
    `S3.run` 见状态是 done/videoing 直接跳过，根本不会拿新锚点重算。
    要么重出时连带作废受影响的镜头（那是"单角色重跑"，另一件事），
    要么别在这里假装能换。选后者：明确拒绝比静默不一致诚实。
    """
    ctx = _load_existing(pid)
    ph = P.load(ctx)
    if ph.phase in ("running", "done"):
        raise HTTPException(
            status_code=409,
            detail=("片子已经开跑，重出正脸不会让已生成的首帧和视频跟着变——"
                    "改了脸，成片还是旧脸。要换人请新建项目重跑；"
                    "只想改某一镜请到「生产进度」里单独处理。"),
        )
    if ph.phase != "anchor_review":
        raise HTTPException(status_code=409,
                            detail=f"当前阶段「{ph.label}」还没有正脸可重出")
    if _state(pid).busy or _state(pid).phase == "running":
        raise HTTPException(status_code=409, detail="流水线或上一次重出还在跑，等它停了再重出")
    if not (ctx.mock or body.allow_paid):
        # 402 而不是 400（2026-11-02 改）：400 会被前端当成"参数写错了"，
        # 而这里的真因是"这一笔还没授权花钱"，是可以由人补一次确认就通过的。
        # 状态码分开，前端才能自动兜一层确认，而不是扔一句人看不懂的报错。
        #
        # 文案也改了：原文说「需要先勾选付费确认」，但界面上**从来没有那个勾选框**
        # （付确认一直是 window.confirm）。要求人去做一个不存在的动作，
        # 等于把人卡死在停 2——2026-11-02 用户就卡在这。
        raise HTTPException(
            status_code=402,
            detail="重出会真实出图（约 ¥0.20），需要在弹窗里勾选「我确认本次重出会真实扣费」",
        )

    threading.Thread(target=_regen_worker, args=(pid, body),
                     daemon=True, name=f"regen-{pid}").start()
    return {"project_id": pid, "char_id": body.char_id,
            "phase": ph.phase, "status": "regenerating",
            "feedback": body.feedback, "update_card": body.update_card}


@app.post("/api/projects/{pid}/refs")
def upload_ref(pid: str, body: RefBody) -> dict[str, Any]:
    """停 0 的「我有参考图」：上传一张人物参考图（base64）。

    只在停 0 / 停 1 允许：再往后（正脸已出）加图，会与已生成的图冲突，
    而冲突的表现是"片子里的脸一半像一半不像"，不会报错。
    """
    ctx = _load_existing(pid)
    ph = P.load(ctx)
    # `failed` 也放开（2026-10-31）：项目因为"参考图太小"这类原因中断后，
    # 人第一件事就是换一张图重来。把 failed 挡在外面，会出现
    # "出图失败 → 想换图 → 界面说当前阶段不允许改图" 的死结。
    if ph.phase not in ("created", "script_review", "failed"):
        raise HTTPException(status_code=409,
                            detail="参考图只在停 0 / 停 1（或中断后）能改（再往后改会和已出的图冲突）")

    raw = body.data_base64.strip()
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[-1]
    try:
        blob = base64.b64decode(raw, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"图片 base64 解不开：{exc}") from exc
    if len(blob) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="单张参考图不能超过 8MB（base64 后更大）")

    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        suffix = ".png"
    elif blob[:3] == b"\xff\xd8\xff":
        suffix = ".jpg"
    elif blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        suffix = ".webp"
    else:
        raise HTTPException(status_code=400, detail="只接受 PNG / JPEG / WEBP（按文件头判断，不看扩展名）")

    refs_dir = ctx.root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    idx = len(list(refs_dir.glob("ref_*"))) + 1
    dst = refs_dir / f"ref_{idx:02d}{suffix}"
    dst.write_bytes(blob)

    # 上传这一刻就把尺寸问题说出来（2026-10-31 新增）。
    # 起因：用一张 121×246 的图当参考，S2a 出图时平台回 400
    #   InvalidParameter: Error validating image ... resolution must ...
    # 报错里只有一个 OSS 签名 URL，既看不出是哪张图、也看不出原图多大，
    # 而且是在"写剧本 + 跑完 S1"之后才炸的。现在挪到上传时提示，零成本、当场可改。
    width = height = 0
    warning = ""
    try:
        with Image.open(io.BytesIO(blob)) as im:
            width, height = im.size
        if min(width, height) < REF_MIN_SIDE:
            warning = (f"这张图只有 {width}×{height}，最短边低于平台要求的 "
                       f"{REF_MIN_SIDE}px。会自动放大后使用，但放大换不回细节，"
                       "生成的脸可能不太像这张图——想要像就换一张更大的原图")
    except Exception:  # noqa: BLE001 —— 解不出尺寸不影响上传（provider 那边还有兜底）
        pass

    rels = list(ph.ref_images)
    rel = str(dst.relative_to(ctx.root))
    if rel not in rels:
        rels.append(rel)
    P.save(ctx, ref_images=rels, role_source="image")
    # 换了图就把对应角色的卡打回 brief，下次「继续」时重新反推。
    # 不打回的话：角色是 inferred → `describe_all(skip_inferred=True)` 跳过 →
    # 新图永远不会被读进角色卡，而界面上看起来"图明明传上去了"。
    reset = _reset_inferred_at(ctx, len(rels) - 1)
    return {"project_id": pid, "ref": rel, "count": len(rels),
            "width": width, "height": height,
            "too_small": bool(warning), "warning": warning,
            "reset_character": reset}


def _reset_inferred_at(ctx: Any, index: int) -> str | None:
    """把第 `index` 个角色的卡打回 `brief`（只是让它重新被反推，不清内容）。

    内容保留：万一反推又失败，角色卡还是能用的旧值，而不是一片空白。
    """
    if not ctx.characters_file.exists():
        return None
    try:
        raw = json.loads(ctx.characters_file.read_text(encoding="utf-8"))
        chars = [Character(**c) for c in raw]
    except Exception:  # noqa: BLE001 —— 卡坏了不该让上传失败
        return None
    if index >= len(chars) or chars[index].source != "inferred":
        return None
    chars[index] = chars[index].model_copy(
        update={"source": "brief", "low_confidence_fields": []})
    R._write_character_list(ctx, chars)
    # 缓存也要删：不删的话下次反推直接命中旧缓存，等于什么也没变。
    RD.cache_path(ctx, index).unlink(missing_ok=True)
    return chars[index].char_id


class RerunShotBody(BaseModel):
    """重跑某一镜。**理由不是可选的装饰**：它既留档，也作为修正指令进提示词。"""

    reason: str = Field(default="", max_length=300)
    allow_paid: bool = False


@app.get("/api/projects/{pid}/timeline")
def project_timeline(pid: str, limit: int = 120) -> dict[str, Any]:
    """「这个项目干了啥」：梗概 + 逐镜流程 + 状态变更时间线。

    **单独一个接口，不塞进 status**：status 每 2 秒被轮询一次，而这份数据里有
    上百条历史日志，塞进去等于让轮询背着全量日志跑。它只在打开项目时取一次。

    时间线取 `state_log`（每次状态迁移都留一行）。**倒序取、正序还**：
    只取最近 N 条才不会把老项目的完整历史一次捞出来，而人读流程是从头读的。
    """
    ctx = _load_existing(pid)
    ph = P.load(ctx)
    raw = {}
    try:
        raw = json.loads(ctx.project_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    req = {}
    rf = ctx.root / "request.json"
    if rf.exists():
        try:
            req = json.loads(rf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            req = {}

    title, logline, shot_n = "", "", 0
    try:
        script = R.S1.load_script(ctx)
        title, logline = script.title, script.logline
        shot_n = len(script.shots)
    except Exception:  # noqa: BLE001 —— 剧本还没写出来是正常状态（停 0）
        title = ""

    store = StateStore()
    try:
        rows = store.log_entries(pid, limit=limit)
    finally:
        store.close()

    events = [
        {"ts": r.get("ts", ""), "shot_id": r.get("shot_id", ""),
         "from": r.get("from_status") or "", "to": r.get("to_status", ""),
         "reason": r.get("reason") or ""}
        for r in reversed(rows)
    ]
    return {
        "project_id": pid,
        "title": title,
        "logline": logline,
        "creative": req.get("creative", ""),
        "created_at": raw.get("created_at", ""),
        "updated_at": ph.updated_at,
        "phase": ph.phase,
        "phase_label": P.LABELS.get(ph.phase, ph.phase),
        "phase_note": ph.note,
        "role_source": ph.role_source,
        "ref_count": len(ph.ref_images),
        "mock": bool(raw.get("mock")),
        "shots": shot_n,
        "events": events,
        # 产物直接给相对路径，前端拼 /files/{pid}/... 就能打开
        "artifacts": {
            "final": (ctx.root / "final.mp4").exists(),
            "report": (ctx.root / "report.html").exists(),
            "reruns": len(R.read_reruns(ctx)),
        },
    }


@app.post("/api/projects/{pid}/stop")
def stop_project(pid: str) -> dict[str, Any]:
    """叫停正在跑的流水线。**不是杀进程**，是在镜与镜之间不再开新的。

    为什么不 `kill`：一镜的视频请求已经在平台侧提交并计费，把进程打死只会让
    "钱花了、文件没落地、账面没有这笔"，下一次续跑要么重复付费、要么以为没跑过。
    等这一镜收尾再停，代价最多两分钟，换来账目和产物一致。
    """
    _load_existing(pid)
    st = _state(pid)
    if st.phase != "running":
        raise HTTPException(status_code=409,
                            detail="现在没有在跑的流水线（停止只在生产中有效）")
    if st.cancel.is_set():
        return {"project_id": pid, "stopping": True,
                "detail": "已经收到停止请求，正在等当前这一镜收尾"}
    st.cancel.set()
    return {"project_id": pid, "stopping": True,
            "detail": "已请求停止：当前这一镜会跑完（约 1–2 分钟），之后不再开新镜"}


def _rerun_shot_worker(pid: str, sid: str, reason: str, allow_paid: bool) -> None:
    """后台重跑一镜：首帧 → 判定 → 视频 → 末帧判定。不合成成片。"""
    st = _state(pid)
    st.busy, st.busy_shot, st.error = True, sid, None
    try:
        ctx = _load_existing(pid)
        _, image_p, video_p, judge_p = R.build_providers(ctx.mock)
        args = SimpleNamespace(
            project_id=pid, creative="", shots=6, force=False,
            paid_ok=bool(allow_paid or ctx.mock), mock=bool(ctx.mock),
            confirm=True, auto=True, stop_after=None,
            regen_front=None, feedback="", update_card=False,
            role_source=None, ref=None,
            rerun_shot=sid, rerun_reason=reason,
        )
        ledger = CostLedger(pid, cap=ctx.budget_cap)
        store = StateStore()
        try:
            R.run_single_shot(ctx, args, ledger=ledger, store=store,
                              image_provider=image_p, video_provider=video_p,
                              judge_provider=judge_p)
        finally:
            store.close()
            ledger.close()
    except Exception as exc:  # noqa: BLE001 —— 后台线程必须吞掉一切并留痕
        st.error = f"重跑 {sid} 失败：{type(exc).__name__}: {exc}"
    finally:
        st.busy, st.busy_shot = False, None
        st.finished_at = time.time()


@app.post("/api/projects/{pid}/shots/{sid}/rerun")
def rerun_shot(pid: str, sid: str, body: RerunShotBody) -> dict[str, Any]:
    """只重出某一镜（用户看着成片觉得这一镜不对时用）。

    **必须视频已经出过**才算"重出"：还没跑的镜头走正常的「继续」，
    否则界面上会出现两条都能让同一镜开跑的路径，账目上分不清是谁花的钱。
    """
    ctx = _load_existing(pid)
    st = _state(pid)
    if st.phase == "running":
        raise HTTPException(status_code=409,
                            detail="流水线正在跑。先点「停止」，等当前这一镜收尾，再重跑单镜")
    if st.busy:
        raise HTTPException(status_code=409, detail="上一笔重跑还没结束，等它回来")
    ph = P.load(ctx)
    if ph.phase not in ("running", "done"):
        raise HTTPException(status_code=409,
                            detail=f"当前阶段「{ph.label}」还没有出过视频，没有可重出的镜头")
    script = R.S1.load_script(ctx)
    if not any(s.shot_id == sid for s in script.shots):
        raise HTTPException(status_code=404,
                            detail=f"没有 {sid}；本项目有 {[s.shot_id for s in script.shots]}")
    if not (ctx.mock or body.allow_paid):
        # 402 而不是 400：这笔没过授权闸门，前端可以补一次确认后重发。
        raise HTTPException(status_code=402,
                            detail="重跑一镜会真实出图与出片（约 ¥1.20），"
                                   "需要在弹窗里勾选「我确认本次重跑会真实扣费」")
    # 预算预检。**必须在开跑前**：等首帧和判定都花完、视频那一步才被 budget 闸拦下，
    # 会留下一镜"首帧已换、视频还是旧的"中间态——要收拾它得再花一次钱。
    # 用 409 而不是 402：这不是授权问题，是项目额度不够，前端不该去弹"再确认一次"。
    if not ctx.mock:
        est = (float(C.settings.price_image)
               + float(C.settings.price_video_per_sec) * 5
               + float(C.settings.price_vlm_per_call) * 2)
        _led = CostLedger(pid, cap=ctx.budget_cap)
        try:
            left = _led.remaining()
        finally:
            _led.close()
        if left < est:
            raise HTTPException(
                status_code=409,
                detail=f"预算不够重跑一镜：还剩 ¥{left:.2f}，约需 ¥{est:.2f}。"
                       f"把 project.json 的 budget_cap 调高再试（只改这个数，不动已花的账）",
            )
    threading.Thread(target=_rerun_shot_worker, args=(pid, sid, body.reason, body.allow_paid),
                     daemon=True, name=f"rerun-{pid}-{sid}").start()
    return {"project_id": pid, "shot_id": sid, "status": "rerunning", "reason": body.reason}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str) -> dict[str, Any]:
    """删掉一个项目的**全部产物与账目**。不可恢复。

    **先删目录、后清数据库**（顺序不能反）：目录删失败很常见（Windows 上文件被
    播放器占着、被杀进程留下的半截文件），这时数据库里的账还在，项目仍然"存在"，
    用户能看到失败原因再试一次。反过来的话——行删了、目录没删——界面上项目消失，
    磁盘上留一堆再也不会被引用的视频，账也查不到了。
    """
    PR.validate_project_id(pid)
    st = _state(pid)
    if st.phase == "running":
        raise HTTPException(status_code=409, detail="流水线正在跑，先点「停止」再删")
    if st.busy:
        raise HTTPException(status_code=409, detail="有重跑正在进行，等它结束再删")
    root = C.project_dir(pid)
    if not (root / "project.json").exists():
        raise HTTPException(status_code=404, detail=f"项目 {pid} 不存在")
    files = [p for p in root.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    try:
        shutil.rmtree(root)
    except OSError as exc:
        raise HTTPException(status_code=500,
                            detail=f"删除目录失败（数据库未动，可重试）：{exc}") from exc
    store = StateStore()
    try:
        removed = store.purge(pid)
    finally:
        store.close()
    RUNS.pop(pid, None)
    return {"project_id": pid, "deleted": True, "files": len(files),
            "bytes": size, "db_rows": removed}


@app.get("/api/projects")
def list_projects() -> dict[str, Any]:
    """首页列表：只列有 request.json 或 script.json 的项目目录。"""
    root = C.PROJECTS_DIR
    items = []
    if root.exists():
        for d in sorted(root.iterdir(), reverse=True):
            if not d.is_dir() or not (d / "project.json").exists():
                continue
            final = d / "final.mp4"
            items.append({
                "project_id": d.name,
                "has_final": final.exists(),
                "created_at": (d / "project.json").stat().st_mtime,
            })
    return {"projects": items[:20]}


# ---------------------------------------------------------------- 静态与页面

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@app.get("/")
def index() -> FileResponse:
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail=f"缺少前端页面：{page}")
    # **不让浏览器缓存首页**（2026-11-02）。没有构建流水线，改完前端就是直接改
    # index.html；Chrome 对 HTML 有启发式缓存，普通 F5 可能连请求都不发，
    # 于是"代码改了、界面没变"，人会去怀疑后端逻辑。
    # 这一条踩过一次：徽章明明在文件里，页面上就是没有。
    return FileResponse(page, media_type="text/html; charset=utf-8",
                        headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/files/{pid}/{rest:path}")
def artifact_file(pid: str, rest: str) -> FileResponse:
    """产物静态服务。**必须挡住路径穿越**：`..` 一律拒绝。"""
    PR.validate_project_id(pid)
    if ".." in rest or rest.startswith("/"):
        raise HTTPException(status_code=400, detail="非法路径")
    base = C.PROJECTS_DIR.resolve()
    target = (base / pid / rest).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="非法路径")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target)
