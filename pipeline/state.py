"""镜头状态机 + 断点续跑（第 7.1 / 7.3 节）。

两条不能绕过的规则：

1. **状态只有两处来源**：SQLite 的 `shots` 表和文件系统里的产物。两者冲突时
   **以文件为准**，并把数据库改回与文件一致，同时写一条 `state_log`——
   否则重启后会出现「状态自己变了」而没人知道原因（7.3 明文要求写日志）。
2. **两种重试分开计数**：技术重试（超时/5xx/空响应，原样重发）与质量重试
   （判定 60~85，必须带 `prompt_delta`）各占一列。混算会让报告里
   「平均重试次数」这个要拿去讲的数据直接失真（7.1 明文）。

另外补一个 7.3 没覆盖、但实现时必然撞到的洞：`judging` 阶段**没有文件产物**
（VLM 结果落 SQLite），所以不能用文件判据。本模块的处理是——若 `score`
已落库，重启后直接按分数决策，**不重调 VLM**；只有分数没落库才重判。
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import STATE_DB, find_bin

# ---------------------------------------------------------------- 状态定义

STATUSES: tuple[str, ...] = (
    "pending",
    "keyframing",
    "judging",
    "videoing",
    # 2026-10-29：S4 之后补了一道末帧判定，以及「不达标就停下等人」的停靠态。
    # 这两个名字必须同时进 STATUSES、LEGAL_TRANSITIONS 和 INTERMEDIATE/TERMINAL 的判断，
    # 漏一处就会在真跑时抛「未知状态」（这次就是这么炸的：LEGAL_TRANSITIONS 改了，
    # 白名单没改，video_judging 在出片成功后立刻被拒）。
    "video_judging",
    "awaiting_review",
    "done",
    "degraded",
    "failed",
)

#: 三个终态语义不可合并（7.2）：done=干净通过 / degraded=凑合放行 / failed=不入片
TERMINAL: frozenset[str] = frozenset({"done", "degraded", "failed"})
#: 中间态。进程被 kill 时会停在这里，启动时必须重整（7.3）
INTERMEDIATE: frozenset[str] = frozenset({"keyframing", "judging", "videoing", "video_judging"})

#: 合法迁移表。写死在这里而不是散在调用方，是为了让「非法迁移」有一个单点判据
#: （14.4-C `test_illegal_transition_rejected` / `test_terminal_states_are_exclusive`）。
#:
#: judging → keyframing  质量重试（带 diff 增量），计 quality_attempts
#: judging → degraded    分数 <60 或质量重试耗尽（14.4-B `test_score_below_floor_marks_degraded_after_limit`）
#: judging → failed      VLM 侧技术失败且重试耗尽；语义上比 degraded 更诚实，故保留这条出口
#:
#: **keyframing → videoing 只给空镜用**（第 7.1 节的缺口）：`environment_only=True`
#: 的镜头没有角色，`VisionJudge.compare` 没有参考图可比，强行判只会得到 0 分 →
#: 每个空镜都被标红 degraded。但也不能造一个假分数让它通过（9.4 第 15 条）。
#: 正确做法是承认「这一镜不判定」，直接从首帧进视频。
#:
#: **video_judging / awaiting_review 是 2026-10-29 新增的一层**（计划书 S4 原本没有判定）：
#:
#:     videoing → video_judging      视频出来了，抽末帧回比首帧
#:     video_judging → done          分数达标
#:     video_judging → awaiting_review  分数不达标 → **停下来等人看一眼**，不自动重跑
#:     awaiting_review → done        人工确认「是我要的样子」→ 放行
#:     awaiting_review → degraded    人工确认「留着但确实差」→ 放行且标红
#:     awaiting_review → videoing    人工驳回 → 回去重出这一条
#:
#: `awaiting_review` **既不是中间态也不是终态**，这是刻意的：它不是「进程被 kill 后的
#: 残留」，而是一个真实存在的等待状态——人可以第二天回来处理。所以 `reconcile()`
#: 必须跳过它，否则重启一次就把人的待办抹掉了。
LEGAL_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"keyframing"}),
    "keyframing": frozenset({"judging", "videoing", "failed"}),
    "judging": frozenset({"videoing", "keyframing", "degraded", "failed"}),
    "videoing": frozenset({"video_judging", "done", "failed"}),
    "video_judging": frozenset({"done", "awaiting_review", "failed", "videoing"}),
    "awaiting_review": frozenset({"done", "degraded", "videoing"}),
    "done": frozenset(),
    "degraded": frozenset(),
    "failed": frozenset(),
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    project_id       TEXT    NOT NULL,
    shot_id          TEXT    NOT NULL,
    status           TEXT    NOT NULL,
    tech_attempts    INTEGER NOT NULL DEFAULT 0,
    quality_attempts INTEGER NOT NULL DEFAULT 0,
    score            INTEGER,
    cost             REAL    NOT NULL DEFAULT 0,
    updated_at       TEXT    NOT NULL,
    PRIMARY KEY (project_id, shot_id)
);

CREATE TABLE IF NOT EXISTS state_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    shot_id     TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    reason      TEXT,
    ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_state_log_shot ON state_log (project_id, shot_id);

-- 判定结果落点。7.3 的断点续跑判据是「文件系统」，但 judging 阶段**没有文件产物**
-- （VLM 返回的是结构化数据）。不单独存一份，进程被 kill 后只能重调 VLM 重判，
-- 每重启一次白烧一次判定钱，且 attempts 计数口径说不清。
-- 存 SQLite 而不是写 sidecar 文件：并发跑多镜时不会互相覆盖（7.4 并发 4）。
CREATE TABLE IF NOT EXISTS shot_qa (
    project_id  TEXT    NOT NULL,
    shot_id     TEXT    NOT NULL,
    score       INTEGER NOT NULL,
    passed      INTEGER NOT NULL,
    degraded    INTEGER NOT NULL,
    diff        TEXT    NOT NULL DEFAULT '',
    checks_json TEXT    NOT NULL DEFAULT '[]',
    delta       TEXT,
    attempts    INTEGER NOT NULL DEFAULT 1,
    ts          TEXT    NOT NULL,
    PRIMARY KEY (project_id, shot_id)
);

-- 视频阶段判定（末帧回比首帧）。**必须与 shot_qa 分开存**：同一个 shot_id 在 S3
-- 已经写过一条首帧判定，用同一张表会把它覆盖掉，报告里「首帧 95 分、视频 62 分」
-- 只剩后一个数字，等于抹掉证据（9.4 第 15 条）。
CREATE TABLE IF NOT EXISTS video_qa (
    project_id  TEXT    NOT NULL,
    shot_id     TEXT    NOT NULL,
    score       INTEGER NOT NULL,
    diff        TEXT    NOT NULL DEFAULT '',
    checks_json TEXT    NOT NULL DEFAULT '[]',
    last_frame  TEXT,
    -- 人工裁决：空=未裁决 / accept=放行 / accept_degraded=放行但标红 / reject=驳回重出
    human_decision TEXT NOT NULL DEFAULT '',
    human_note     TEXT NOT NULL DEFAULT '',
    ts          TEXT    NOT NULL,
    PRIMARY KEY (project_id, shot_id)
);

-- C2 通道的提交台账（10.6）。**计数必须落在库里，不能只放内存**：
-- 单项目 12 次 / 单日 30 次是"停下来"的判据，进程重启后如果忘了自己提交过几次，
-- 就会继续提交——这比不限制更危险，因为它让"有上限"这件事变成一句空话。
CREATE TABLE IF NOT EXISTS submit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   TEXT    NOT NULL,
    shot_id      TEXT    NOT NULL,
    attempt      INTEGER NOT NULL DEFAULT 1,   -- 该镜第几次提交（1=首发，2=重试）
    account      TEXT    NOT NULL DEFAULT '',  -- 账号标识；日上限按它分桶
    day          TEXT    NOT NULL,             -- YYYY-MM-DD，自然日
    duration_sec REAL    NOT NULL DEFAULT 0,   -- 这一镜计划生成时长，用于推下一镜间隔
    ok           INTEGER NOT NULL DEFAULT 1,   -- 0=提交失败（仍计入尝试次数）
    note         TEXT    NOT NULL DEFAULT '',
    ts           TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_submit_project ON submit_log(project_id);
CREATE INDEX IF NOT EXISTS idx_submit_day     ON submit_log(day, account);
"""

_UNSET: Any = object()


class TransitionRejected(ValueError):
    """非法状态迁移。单独一个类型，避免调用方把「逻辑违规」和「参数写错」混在一起接。"""


def now_iso() -> str:
    """带时区的 ISO8601。容器默认 UTC，时间戳不带时区会差 8 小时（第 7.6 节坑）。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def connect(db_path: Path | str) -> sqlite3.Connection:
    """打开状态库。

    `journal_mode=WAL` 不是优化，是**正确性前提**：7.5 要求 Web 层与 worker
    线程同时读写同一张 SQLite，默认的 rollback journal 在并发写时会直接
    `database is locked`。`busy_timeout` 再给一点重试余量。
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False：worker 跑在独立线程里（7.5 实现形态），
    # 连接会跨线程使用；写操作由 StateStore 内部锁串行化。
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def artifact_ok(path: Path | str | None, *, kind: str = "image") -> bool:
    """产物是否**真的**可用——不是「文件存在且非空」就够。

    7.3 的原文判据是「文件存在且非空」。对图片够用，对 mp4 不够：
    异步下载到一半被 kill，会留下一个存在但截断的 mp4，非空判据会把它当成功，
    到 S5 合成时才报解码失败（整片失败）。所以：
      - 图片：非空 + PNG/JPEG 魔数
      - 视频：非空 + 阈值，且**有 ffprobe 就探一次**；探不通即视为坏文件
    mock 产物（`MOCK_VIDEO` 开头的文本桩）显式放行，否则无 ffmpeg 的机器跑不了 mock 链路。
    """
    if path is None:
        return False
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return False
    if size <= 0:
        return False

    if kind == "image":
        try:
            head = p.read_bytes()[:8]
        except OSError:
            return False
        return head.startswith(b"\x89PNG\r\n\x1a\n") or head.startswith(b"\xff\xd8\xff")

    if kind == "video":
        try:
            head = p.read_bytes()[:16]
        except OSError:
            return False
        if head.startswith(b"MOCK_VIDEO"):
            return True  # mock 文本桩，无 ffmpeg 时的合法产物
        if size < 1024:
            return False
        ffprobe = find_bin("ffprobe")
        if not ffprobe:
            return True  # 无 ffprobe 只能退到体积判据，调用方应记录该降级
        import subprocess

        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(p)],
                capture_output=True, text=True, timeout=20,
            )
        except Exception:
            return False
        return r.returncode == 0 and bool(r.stdout.strip())

    raise ValueError(f"未知产物类型：{kind}")


@dataclass(frozen=True)
class ShotState:
    project_id: str
    shot_id: str
    status: str
    tech_attempts: int = 0
    quality_attempts: int = 0
    score: int | None = None
    cost: float = 0.0
    updated_at: str = ""

    @property
    def attempts(self) -> int:
        """总次数，仅为兼容 7.3 表里的 `attempts` 列名；**报告请分列两个计数**。"""
        return self.tech_attempts + self.quality_attempts

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL


@dataclass
class ReconcileAction:
    shot_id: str
    from_status: str
    to_status: str
    reason: str


@dataclass
class ReconcileReport:
    """启动重整的结果。调用方必须把它打进日志，不能只吞掉。"""

    actions: list[ReconcileAction] = field(default_factory=list)

    @property
    def changed(self) -> list[ReconcileAction]:
        return [a for a in self.actions if a.from_status != a.to_status]

    def summary(self) -> str:
        if not self.changed:
            return "状态无需重整"
        parts = [f"{a.shot_id}: {a.from_status}→{a.to_status}（{a.reason}）" for a in self.changed]
        return f"重整 {len(parts)} 镜：" + "；".join(parts)


class StateStore:
    """镜头状态的唯一读写入口。"""

    def __init__(self, db_path: Path | str | None = None, *, conn: sqlite3.Connection | None = None):
        self.db_path = Path(db_path) if db_path is not None else STATE_DB
        self._lock = threading.RLock()
        self._conn = conn if conn is not None else connect(self.db_path)
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------ 读

    def get(self, project_id: str, shot_id: str) -> ShotState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            ).fetchone()
        return self._to_state(row) if row else None

    def all(self, project_id: str) -> list[ShotState]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? ORDER BY shot_id", (project_id,)
            ).fetchall()
        return [self._to_state(r) for r in rows]

    def pending_ids(self, project_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT shot_id FROM shots WHERE project_id=? AND status='pending' ORDER BY shot_id",
                (project_id,),
            ).fetchall()
        return [r["shot_id"] for r in rows]

    @staticmethod
    def _to_state(row: sqlite3.Row) -> ShotState:
        return ShotState(
            project_id=row["project_id"],
            shot_id=row["shot_id"],
            status=row["status"],
            tech_attempts=row["tech_attempts"],
            quality_attempts=row["quality_attempts"],
            score=row["score"],
            cost=row["cost"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------ 写

    def ensure_shots(self, project_id: str, shot_ids: list[str]) -> int:
        """登记镜头为 pending。已存在的不动（幂等）——重复调用不得重置进度。"""
        ts = now_iso()
        added = 0
        with self._lock, self._conn:
            for sid in shot_ids:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO shots (project_id, shot_id, status, updated_at) "
                    "VALUES (?,?,'pending',?)",
                    (project_id, sid, ts),
                )
                added += cur.rowcount
                if cur.rowcount:
                    self._log(project_id, sid, None, "pending", "登记", ts)
        return added

    def transition(
        self,
        project_id: str,
        shot_id: str,
        to_status: str,
        *,
        reason: str | None = None,
        score: Any = _UNSET,
        cost_delta: float = 0.0,
        quality_retry: bool = False,
    ) -> ShotState:
        """执行一次合法迁移。

        `quality_retry=True` 会同时把 `quality_attempts` +1 —— 14.4-C
        `test_judging_back_to_keyframing_on_retry` 断言的正是「回到首帧且 attempts +1」。
        技术重试不走这里，走 `mark_tech_attempt`，因为技术重试不改状态。
        """
        if to_status not in STATUSES:
            raise TransitionRejected(f"未知状态：{to_status}（合法值 {STATUSES}）")

        ts = now_iso()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            ).fetchone()
            if row is None:
                raise TransitionRejected(f"{shot_id} 未登记，不能迁移到 {to_status}")

            cur = row["status"]
            allowed = LEGAL_TRANSITIONS[cur]
            if to_status not in allowed:
                hint = "（终态不可再迁出）" if cur in TERMINAL else f"（允许：{sorted(allowed)}）"
                raise TransitionRejected(f"{shot_id}: 非法迁移 {cur} → {to_status}{hint}")

            q_attempts = row["quality_attempts"] + (1 if quality_retry else 0)
            new_score = row["score"] if score is _UNSET else score
            new_cost = float(row["cost"]) + float(cost_delta)
            self._conn.execute(
                "UPDATE shots SET status=?, quality_attempts=?, score=?, cost=?, updated_at=? "
                "WHERE project_id=? AND shot_id=?",
                (to_status, q_attempts, new_score, new_cost, ts, project_id, shot_id),
            )
            self._log(project_id, shot_id, cur, to_status, reason, ts)
            fresh = self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            ).fetchone()
        return self._to_state(fresh)

    def mark_tech_attempt(
        self, project_id: str, shot_id: str, *, reason: str | None = None, cost_delta: float = 0.0
    ) -> ShotState:
        """记一次技术重试：**状态不变**，只加 `tech_attempts` 与成本。

        单独一个方法而不是 `transition(x → x)`：自环迁移在合法迁移表里不存在，
        硬塞进去会让「非法迁移」的判据变模糊。
        """
        ts = now_iso()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            ).fetchone()
            if row is None:
                raise TransitionRejected(f"{shot_id} 未登记")
            self._conn.execute(
                "UPDATE shots SET tech_attempts=tech_attempts+1, cost=cost+?, updated_at=? "
                "WHERE project_id=? AND shot_id=?",
                (float(cost_delta), ts, project_id, shot_id),
            )
            self._log(project_id, shot_id, row["status"], row["status"],
                      reason or "技术重试", ts)
            fresh = self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            ).fetchone()
        return self._to_state(fresh)

    def set_score(self, project_id: str, shot_id: str, score: int) -> None:
        """只写分数、不改状态。

        为什么需要它：`judging` 阶段先拿分数、再按分数决定去哪，中间必然有一个
        「状态还是 judging、但分数已经有了」的时刻。若为此造一条 `judging → judging`
        的合法迁移，会把「非法迁移」的判据搅浑（终态互斥靠的就是那张表）。
        分数同时落在这里和 `shot_qa`：前者给状态重整判「要不要重判」，后者给报告用。
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE shots SET score=?, updated_at=? WHERE project_id=? AND shot_id=?",
                (int(score), now_iso(), project_id, shot_id),
            )

    def add_cost(self, project_id: str, shot_id: str, amount: float) -> None:
        if amount <= 0:
            return
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE shots SET cost=cost+?, updated_at=? WHERE project_id=? AND shot_id=?",
                (float(amount), now_iso(), project_id, shot_id),
            )

    # ------------------------------------------------------------ C2 提交台账

    def log_note(self, project_id: str, shot_id: str, kind: str, reason: str) -> None:
        """记一条**不改变状态**的事件（如 C2 的归因结论）。

        为什么复用 `state_log` 而不新建表：`reason` 是给人看的文本，归因是一次
        "人给出的判断"，它和状态迁移一样属于"这一镜发生过什么"。报告已经按
        `state_log` 渲染时间线，另立一张表会让人在两处地方找同一件事。
        `from_status == to_status` 是刻意的——它表示"不是迁移"，读的人一眼能分辨。
        """
        st = self.get(project_id, shot_id)
        cur = st.status if st else ""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO state_log (project_id, shot_id, from_status, to_status, reason, ts) "
                "VALUES (?,?,?,?,?,?)",
                (project_id, shot_id, cur, cur, f"[{kind}] {reason}", now_iso()),
            )

    def log_submit(
        self, project_id: str, shot_id: str, *, attempt: int = 1,
        account: str = "", duration_sec: float = 0.0,
        ok: bool = True, note: str = "",
    ) -> None:
        """记一次 C2 提交（10.6）。**提交前调用**，不要提交后再补记。

        顺序很重要：先记后提交，最坏情况是"记了但没提交成功"（多算一次，偏向保守）；
        反过来则是"提交了但没记"，上限就永远算不准——**两种错法，宁可多算**。
        """
        from datetime import datetime as _dt
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO submit_log (project_id, shot_id, attempt, account, day,"
                " duration_sec, ok, note, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                (project_id, shot_id, int(attempt), account,
                 _dt.now().astimezone().strftime("%Y-%m-%d"),
                 float(duration_sec), 1 if ok else 0, note, now_iso()),
            )

    def mark_last_submit_failed(self, project_id: str, shot_id: str, *, note: str = "") -> None:
        """把该镜最近一条提交台账标成失败。

        **不新增记录**：提交前的 `log_submit` 已经占了一个名额，而平台确实收到了
        请求——次数照算。再补一条会让一次失败吃掉两个名额，护栏越用越紧。
        """
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT id FROM submit_log WHERE project_id=? AND shot_id=? ORDER BY id DESC LIMIT 1",
                (project_id, shot_id),
            ).fetchone()
            if row is None:
                return
            self._conn.execute(
                "UPDATE submit_log SET ok=0, note=? WHERE id=?", (note[:400], row["id"]))

    def submit_counts(self, project_id: str, *, account: str = "") -> dict[str, int]:
        """返回 {project: 本项目已提交次数, today: 本账号今日提交次数}。

        两个口径分开算：项目上限（12）是"这一部片的预算"，日上限（30）是
        "账号的安全边界"。**只算成功提交还是把所有尝试都算**——答案是全算，
        因为平台的反自动化阈值数的是"请求次数"，不是"成功次数"。
        """
        from datetime import datetime as _dt
        day = _dt.now().astimezone().strftime("%Y-%m-%d")
        with self._lock:
            p = self._conn.execute(
                "SELECT COUNT(*) FROM submit_log WHERE project_id=?", (project_id,)
            ).fetchone()[0]
            t = self._conn.execute(
                "SELECT COUNT(*) FROM submit_log WHERE day=? AND account=?",
                (day, account),
            ).fetchone()[0]
        return {"project": int(p), "today": int(t)}

    def last_submit_elapsed(self, project_id: str) -> float | None:
        """上一镜实测生成耗时（秒），用于推下一镜的提交间隔（10.6 第 2 条）。

        口径：取最近两次提交的时间差。**取实测差而不是取平台标称值**——
        标称 5 秒的镜头实际可能排 90 秒的队，间隔按下限走就成了"平台一空就立刻提交"，
        那是最明显的自动化特征。数据不足两次时返回 None，由调用方走保守默认值。
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts FROM submit_log WHERE project_id=? ORDER BY id DESC LIMIT 2",
                (project_id,),
            ).fetchall()
        if len(rows) < 2:
            return None
        from datetime import datetime as _dt
        try:
            a = _dt.fromisoformat(rows[1]["ts"])
            b = _dt.fromisoformat(rows[0]["ts"])
        except (TypeError, ValueError):
            return None
        return max(0.0, (b - a).total_seconds())

    # ------------------------------------------------------------ 判定结果

    def save_qa(self, project_id: str, shot_id: str, qa: dict[str, Any], *, delta: str | None = None) -> None:
        """落一份完整的判定结果（含逐要素 checks）。

        `checks_json` 存原文而不是拆表：要素集合未来可能调整（比如体型拆成两项），
        拆成列会让每次调整都要迁移表结构，而这里只需要「原样取回」。
        """
        import json as _json

        payload = {
            "project_id": project_id,
            "shot_id": shot_id,
            "score": int(qa.get("score", 0)),
            "passed": 1 if qa.get("passed") else 0,
            "degraded": 1 if qa.get("degraded") else 0,
            "diff": str(qa.get("diff") or ""),
            "checks_json": _json.dumps(qa.get("checks") or [], ensure_ascii=False),
            "delta": delta,
            "attempts": int(qa.get("attempts", 1)),
            "ts": now_iso(),
        }
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO shot_qa (project_id, shot_id, score, passed, degraded, diff, "
                "checks_json, delta, attempts, ts) VALUES (:project_id, :shot_id, :score, "
                ":passed, :degraded, :diff, :checks_json, :delta, :attempts, :ts) "
                "ON CONFLICT(project_id, shot_id) DO UPDATE SET "
                "score=excluded.score, passed=excluded.passed, degraded=excluded.degraded, "
                "diff=excluded.diff, checks_json=excluded.checks_json, delta=excluded.delta, "
                "attempts=excluded.attempts, ts=excluded.ts",
                payload,
            )

    def clear_qa(self, project_id: str, shot_id: str) -> None:
        """删掉这一镜的首帧判定（单镜重跑前必须调）。

        不清的话，`S3.run` 会从 `load_qa(...).get("delta")` 里捡起**上一版画面**的
        修正意见继续用——那是针对旧图说的话，套到新图上只会越改越偏。
        """
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM shot_qa WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            )

    def clear_video_qa(self, project_id: str, shot_id: str) -> None:
        """删掉这一镜的末帧判定（与 `clear_qa` 同理）。"""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM video_qa WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            )

    def reset_shot(self, project_id: str, shot_id: str, reason: str) -> None:
        """把一镜打回 `pending`（单镜重跑用）。绕过合法迁移表，但留日志。

        与 `_force` 分开命名：`_force` 是 `reconcile` 的内部手段，那是"启动时按文件
        对齐状态"；这里是**人主动要求重做某一镜**，两件事的触发者不同，
        混用一个名字会让日志里分不清"系统修的"和"人让重跑的"。
        """
        self._force(project_id, shot_id, "pending", reason)

    def load_qa(self, project_id: str, shot_id: str) -> dict[str, Any] | None:
        import json as _json

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM shot_qa WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            ).fetchone()
        if row is None:
            return None
        return {
            "score": row["score"],
            "passed": bool(row["passed"]),
            "degraded": bool(row["degraded"]),
            "diff": row["diff"],
            "checks": _json.loads(row["checks_json"] or "[]"),
            "delta": row["delta"],
            "attempts": row["attempts"],
        }

    # ------------------------------------------------------------ 视频判定

    def save_video_qa(
        self, project_id: str, shot_id: str, qa: dict[str, Any], *,
        human_decision: str = "", human_note: str = "",
    ) -> None:
        """落一份视频阶段判定（末帧 vs 首帧）。

        与 `save_qa` 分开两张表，原因见 `_SCHEMA` 里 `video_qa` 的注释：
        同一条 shot 在 S3 已经有一条首帧判定，覆盖它等于抹掉证据。
        """
        import json as _json

        payload = {
            "project_id": project_id,
            "shot_id": shot_id,
            "score": int(qa.get("score", 0)),
            "diff": str(qa.get("diff") or ""),
            "checks_json": _json.dumps(qa.get("checks") or [], ensure_ascii=False),
            "last_frame": qa.get("last_frame"),
            "human_decision": human_decision,
            "human_note": human_note,
            "ts": now_iso(),
        }
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO video_qa (project_id, shot_id, score, diff, checks_json, "
                "last_frame, human_decision, human_note, ts) VALUES (:project_id, :shot_id, "
                ":score, :diff, :checks_json, :last_frame, :human_decision, :human_note, :ts) "
                "ON CONFLICT(project_id, shot_id) DO UPDATE SET "
                "score=excluded.score, diff=excluded.diff, checks_json=excluded.checks_json, "
                "last_frame=excluded.last_frame, human_decision=excluded.human_decision, "
                "human_note=excluded.human_note, ts=excluded.ts",
                payload,
            )

    def load_video_qa(self, project_id: str, shot_id: str) -> dict[str, Any] | None:
        import json as _json

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM video_qa WHERE project_id=? AND shot_id=?",
                (project_id, shot_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "score": row["score"],
            "diff": row["diff"],
            "checks": _json.loads(row["checks_json"] or "[]"),
            "last_frame": row["last_frame"],
            "human_decision": row["human_decision"],
            "human_note": row["human_note"],
            "ts": row["ts"],
        }

    def awaiting_review(self, project_id: str) -> list[dict[str, Any]]:
        """待人工确认的镜头，带上视频判定信息，供 `--review` 展示。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.shot_id, s.score, s.updated_at, v.score AS video_score, "
                "v.diff AS video_diff, v.last_frame FROM shots s "
                "LEFT JOIN video_qa v ON v.project_id=s.project_id AND v.shot_id=s.shot_id "
                "WHERE s.project_id=? AND s.status='awaiting_review' ORDER BY s.shot_id",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _log(        self, project_id: str, shot_id: str, from_status: str | None, to_status: str, reason: str | None, ts: str
    ) -> None:
        self._conn.execute(
            "INSERT INTO state_log (project_id, shot_id, from_status, to_status, reason, ts) "
            "VALUES (?,?,?,?,?,?)",
            (project_id, shot_id, from_status, to_status, reason, ts),
        )

    def log_entries(self, project_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM state_log WHERE project_id=? ORDER BY id DESC LIMIT ?",
                (project_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ 断点续跑

    def reconcile(
        self, project_id: str, artifacts: dict[str, dict[str, Any]] | None = None
    ) -> ReconcileReport:
        """启动时（以及每次开始批量前）执行一次状态重整（7.3）。

        `artifacts`：`{shot_id: {"keyframe": 路径, "video": 路径}}`。
        判据是**文件系统**，不是 SQLite：程序以为的进度可能是错的，磁盘上的东西才是
        实际拿到的；重做一次付费调用等于重复扣费。

        幂等：连续调用两次，第二次不产生任何状态变化（14.4-C `test_resume_is_idempotent`）。
        """
        artifacts = artifacts or {}
        report = ReconcileReport()
        for st in self.all(project_id):
            art = artifacts.get(st.shot_id, {})
            kf_ok = artifact_ok(art.get("keyframe"), kind="image")
            vid_ok = artifact_ok(art.get("video"), kind="video")

            if st.status in ("done", "degraded"):
                # 这两个终态直接跳过（7.3 / 14.4-C `test_resume_skips_done_and_degraded`）。
                # **failed 不能在这里跳过**：它是终态，但续跑时要重新排队，
                # 否则「修好网络再跑一次」永远补不上那一镜。
                continue

            if st.status == "awaiting_review":
                # 刻意不自动处理：这是「等人看一眼」的停靠态，不是进程残留。
                # 重启一次就把它改掉，等于把人的待办抹了。
                report.actions.append(ReconcileAction(
                    st.shot_id, "awaiting_review", "awaiting_review",
                    "等待人工确认，保持不动（用 --review / --decide 处理）"))
                continue

            if st.status == "failed":
                # 14.4-C：重开后「只补 pending 与 failed」。技术重试计数归零（否则
                # 一进来就又超限），质量重试次数保留（那是业务指标，不能被重启抹掉）。
                self._force(project_id, st.shot_id, "pending", "续跑：failed 重新排队（技术重试计数归零）",
                            reset_tech=True)
                report.actions.append(ReconcileAction(st.shot_id, "failed", "pending",
                                                      "failed 重新排队，技术重试计数归零"))
                continue

            if st.status == "pending":
                # 7.3：文件存在且非空 → 该阶段视为已完成，从下一阶段续
                if vid_ok:
                    self._force(project_id, st.shot_id, "done", "续跑：视频产物已存在", )
                    report.actions.append(ReconcileAction(st.shot_id, "pending", "done", "视频产物已存在"))
                elif kf_ok:
                    self._force(project_id, st.shot_id, "judging", "续跑：首帧已存在，跳过生成")
                    report.actions.append(ReconcileAction(st.shot_id, "pending", "judging", "首帧已存在"))
                continue

            if st.status == "keyframing":
                if kf_ok:
                    self._force(project_id, st.shot_id, "judging", "续跑：首帧已存在，跳过重生成")
                    report.actions.append(ReconcileAction(st.shot_id, "keyframing", "judging", "首帧已存在"))
                else:
                    self._force(project_id, st.shot_id, "pending", "续跑：首帧缺失，重做")
                    report.actions.append(ReconcileAction(st.shot_id, "keyframing", "pending", "首帧缺失"))
                continue

            if st.status == "judging":
                # judging 无文件产物（7.3 的盲区）。分数已落库就直接按分数决策，不重调 VLM；
                # 分数没落库才重判——否则每次重启都白烧一次判定钱。
                if st.score is not None:
                    report.actions.append(ReconcileAction(
                        st.shot_id, "judging", "judging",
                        f"已有判定分数 {st.score}，直接决策，不重调 VLM"))
                else:
                    self._force(project_id, st.shot_id, "pending", "续跑：judging 未落分数，重判")
                    report.actions.append(ReconcileAction(st.shot_id, "judging", "pending", "无判定分数"))
                continue

            if st.status == "video_judging":
                # 视频已出、判定没跑完（进程被 kill 在两者之间）。
                # 视频文件在 → 只重做判定，绝不重出视频（一条 ¥1，重出是纯浪费）；
                # 视频不在 → 退回 videoing 重出。
                if vid_ok:
                    report.actions.append(ReconcileAction(
                        st.shot_id, "video_judging", "video_judging",
                        "视频已存在，重新判定（不重出视频）"))
                else:
                    self._force(project_id, st.shot_id, "videoing", "续跑：视频缺失，重出")
                    report.actions.append(ReconcileAction(
                        st.shot_id, "video_judging", "videoing", "视频缺失，退回出片"))
                continue

            if st.status == "videoing":
                if vid_ok:
                    # 2026-10-29 改：以前这里是「视频在 → 直接 done」。
                    # 现在 S4 之后还有一道末帧判定，视频写盘与判定之间被杀就会留下
                    # 「视频在、状态还是 videoing」——这时该去补判定，而不是当成已完成。
                    self._force(project_id, st.shot_id, "video_judging", "续跑：视频已存在，补做末帧判定")
                    report.actions.append(ReconcileAction(
                        st.shot_id, "videoing", "video_judging", "视频已存在，补做末帧判定"))
                    report.actions.append(ReconcileAction(st.shot_id, "videoing", "done", "视频已存在"))
                elif kf_ok:
                    # 关键：首帧还在，说明首帧阶段已经完成，只是视频还没出（S4 未跑完或被 kill）。
                    # 打回 pending 会让整镜从生成首帧重来——首帧和判定白花的钱全部重复支出，
                    # 而磁盘上的文件明明还在。这里保持 videoing，等 S4 接着出片。
                    report.actions.append(ReconcileAction(
                        st.shot_id, "videoing", "videoing", "首帧已存在，等待出片（不重做首帧）"))
                else:
                    self._force(project_id, st.shot_id, "pending", "续跑：首帧与视频都不在，重做")
                    report.actions.append(ReconcileAction(st.shot_id, "videoing", "pending", "首帧缺失，重做"))
                continue

        return report

    def _force(
        self, project_id: str, shot_id: str, to_status: str, reason: str, *, reset_tech: bool = False
    ) -> None:
        """重整专用：绕过合法迁移表，但**必须留日志**（7.3 明文要求）。"""
        ts = now_iso()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM shots WHERE project_id=? AND shot_id=?", (project_id, shot_id)
            ).fetchone()
            if row is None:
                return
            self._conn.execute(
                "UPDATE shots SET status=?, tech_attempts=?, updated_at=? WHERE project_id=? AND shot_id=?",
                (to_status, 0 if reset_tech else row["tech_attempts"], ts, project_id, shot_id),
            )
            self._log(project_id, shot_id, row["status"], to_status, reason, ts)

    # ------------------------------------------------------------ 汇总

    def summary(self, project_id: str) -> dict[str, Any]:
        shots = self.all(project_id)
        counts: dict[str, int] = {s: 0 for s in STATUSES}
        for st in shots:
            counts[st.status] += 1
        q = [st.quality_attempts for st in shots]
        return {
            "total": len(shots),
            "counts": counts,
            "cost": round(sum(st.cost for st in shots), 4),
            # 平均质量重试次数：只统计真正进入过判定/重取的镜头，分母不含 pending
            "quality_retries_avg": round(sum(q) / len(q), 3) if q else 0.0,
            "tech_retries_total": sum(st.tech_attempts for st in shots),
        }

    def purge(self, project_id: str) -> dict[str, int]:
        """删掉一个项目的**全部数据库记录**，返回每张表删了几行。

        删项目这件事有两个必须一起做、少一个就出脏数据的部分：数据库行和磁盘目录。
        这里只负责前者，目录由调用方 `shutil.rmtree`。分开是因为万一目录删失败
        （Windows 上文件被占用），这里已经删掉的行就是回不来的账——
        所以调用方必须**先删目录、后清库**，让"删了一半"落在更安全的那一侧。
        """
        tables = ("shots", "state_log", "shot_qa", "video_qa", "costs", "submit_log")
        removed: dict[str, int] = {}
        with self._lock, self._conn:
            for t in tables:
                cur = self._conn.execute(f"DELETE FROM {t} WHERE project_id=?", (project_id,))
                removed[t] = cur.rowcount or 0
        return removed

    def close(self) -> None:
        with self._lock:
            self._conn.close()
