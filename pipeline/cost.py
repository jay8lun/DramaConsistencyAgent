"""记账 + 预算封顶（第 12.3 节 / D7）。

三条硬规则，对应 14.4-D 的四个用例：

1. **超预算必须先抛异常，再谈调用**（`test_budget_cap_blocks_before_call`）。
   顺序反过来就等于钱已经花掉了再报警。
2. **估算价与实际价分开记**（`test_estimate_and_actual_both_recorded`）。
   只记实际价，事后无法校准单价；只记估算价，永远不知道真实偏差。
3. **失败也要记账**（`test_failed_call_still_charged` / `test_paid_call_requires_audit_entry`）。
   调用超时不代表平台没扣钱。`paid()` 上下文管理器保证「出口必有一条审计记录」，
   没有记录等于没发生（9.4 第 15 条：禁止虚构数据）。
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from config import STATE_DB, settings
from pipeline.state import connect, now_iso
from providers.base import PaidCallBlocked

_SCHEMA = """
CREATE TABLE IF NOT EXISTS costs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     TEXT    NOT NULL,
    shot_id        TEXT,
    stage          TEXT    NOT NULL,
    provider       TEXT    NOT NULL,
    operation      TEXT    NOT NULL,
    estimated_cost REAL    NOT NULL DEFAULT 0,
    actual_cost    REAL,
    success        INTEGER NOT NULL DEFAULT 1,
    error          TEXT,
    ts             TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_costs_project ON costs (project_id, id);
"""


class BudgetExceeded(RuntimeError):
    """超出 `BUDGET_CAP`。必须**在调用前**抛（12.3）。"""


@dataclass
class CallSlot:
    """`paid()` 交给调用方的可变槽位：填 actual / success，出口统一落库。"""

    actual_cost: float | None = None
    success: bool = False
    error: str | None = None
    extra: dict[str, Any] | None = None

    def charge(self, amount: float) -> None:
        self.actual_cost = float(amount)
        self.success = True


class CostLedger:
    """单个项目的账本。跨项目共用同一个 SQLite 文件，但账按 `project_id` 隔离。"""

    def __init__(
        self,
        project_id: str,
        *,
        cap: float | None = None,
        db_path: Path | str | None = None,
        conn: sqlite3.Connection | None = None,
    ):
        self.project_id = project_id
        self.cap = float(settings.budget_cap if cap is None else cap)
        self.db_path = Path(db_path) if db_path is not None else STATE_DB
        self._lock = threading.RLock()
        self._conn = conn if conn is not None else connect(self.db_path)
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------ 读

    def spent(self) -> float:
        """已花费：**只算成功的调用**，成功时优先用实际价、为空退到估算价。

        **失败调用不计入（2026-10-29 改）**，原实现是
        `SUM(COALESCE(actual_cost, estimated_cost))` 不分成功与否，注释理由是
        「失败调用可能拿不到账单，但额度确实被扣了」。这个保守假设有个致命后果：
        **护栏会越用越紧，最后把自己卡死**——六镜那次真花 ¥5.39，而两条参数错误的
        失败请求（平台当场拒绝、一分不扣）贡献了 ¥4.00 估算，`spent()` 变成 ¥9.39；
        接着补拍 shot_05 需要再花 ¥1.00，被判定「9.39 + 1.00 > 9.50」直接拒绝。
        **账户里明明还有 4.61 元，程序说没钱了。**

        失败的调用绝大多数是 4xx（参数不兼容、额度不足、鉴权失败），平台不扣费；
        真正「扣了钱但返回失败」的情况（如超时后服务端仍完成任务）无法在本地区分，
        这类差额会体现在下一次成功调用的实际账单里。**宁可承认这个不确定性，
        也不能让护栏基于一笔虚构的支出拒绝真实的工作。**
        失败调用的估算额仍在 `summary()['estimated_total']` 里可查。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(CASE WHEN success=1 "
                "THEN COALESCE(actual_cost, estimated_cost) ELSE 0 END), 0.0) AS s "
                "FROM costs WHERE project_id=?",
                (self.project_id,),
            ).fetchone()
        return round(float(row["s"]), 6)

    def remaining(self) -> float:
        return round(self.cap - self.spent(), 6)

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM costs WHERE project_id=? ORDER BY id", (self.project_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict[str, Any]:
        rows = self.entries()
        est = round(sum(r["estimated_cost"] for r in rows), 4)
        act = round(sum(r["actual_cost"] or 0.0 for r in rows), 4)
        failed = sum(1 for r in rows if not r["success"])
        by_stage: dict[str, float] = {}
        for r in rows:
            # 只累计成功的调用（与 spent() 同口径，理由见那里）。失败调用的估算价
            # 若混进来，报告里每一阶段的成本都会虚高。
            if not r["success"]:
                continue
            by_stage[r["stage"]] = round(
                by_stage.get(r["stage"], 0.0) + (r["actual_cost"] or r["estimated_cost"]), 4
            )
        return {
            "calls": len(rows),
            "failed_calls": failed,
            "estimated_total": est,
            "actual_total": act,
            "spent": self.spent(),
            "cap": self.cap,
            "remaining": self.remaining(),
            "by_stage": by_stage,
        }

    # ------------------------------------------------------------ 护栏

    def check_before_call(self, estimated: float, *, what: str = "") -> None:
        """调用前的唯一闸门。**必须在发起任何网络/浏览器动作之前调用。**

        12.3：「超限立即抛异常并落盘当前进度，不得跑完这一镜再说」。
        """
        spent = self.spent()
        projected = spent + max(0.0, float(estimated))
        if projected > self.cap:
            raise BudgetExceeded(
                f"预算封顶：已花 ¥{spent:.4f} + 本次预估 ¥{estimated:.4f} "
                f"= ¥{projected:.4f} > BUDGET_CAP ¥{self.cap:.2f}"
                + (f"（{what}）" if what else "")
            )

    # ------------------------------------------------------------ 写

    def record(
        self,
        *,
        stage: str,
        provider: str,
        operation: str,
        estimated: float,
        actual: float | None = None,
        success: bool = True,
        error: str | None = None,
        shot_id: str | None = None,
    ) -> int:
        ts = now_iso()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO costs (project_id, shot_id, stage, provider, operation, "
                "estimated_cost, actual_cost, success, error, ts) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self.project_id, shot_id, stage, provider, operation,
                 float(estimated), None if actual is None else float(actual),
                 1 if success else 0, error, ts),
            )
            return int(cur.lastrowid or 0)

    @contextlib.contextmanager
    def paid(
        self,
        estimated: float,
        *,
        stage: str,
        provider: str,
        operation: str,
        allow_paid: bool = False,
        shot_id: str | None = None,
    ) -> Iterator[CallSlot]:
        """付费调用的唯一入口：先查授权、再查预算、出口必落审计记录。

        用法：
            with ledger.paid(0.08, stage="judging", provider="vlm",
                             operation="compare", allow_paid=True, shot_id="shot_01") as slot:
                result = judge.compare(...)
                slot.charge(0.08)

        即使 `judge.compare` 抛异常，`finally` 也会写一条 `success=0` 的记录——
        这就是 `test_paid_call_requires_audit_entry` 要的「没有记录等于没发生」。
        """
        if not allow_paid:
            raise PaidCallBlocked(
                f"{provider}.{operation} 是付费调用，必须显式 allow_paid=True（第 9.4 节护栏）"
            )
        self.check_before_call(estimated, what=f"{provider}.{operation}")

        slot = CallSlot()
        try:
            yield slot
        except BaseException as exc:  # noqa: BLE001 —— 审计记录必须覆盖所有出口
            slot.success = False
            slot.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.record(
                stage=stage,
                provider=provider,
                operation=operation,
                estimated=estimated,
                actual=slot.actual_cost,
                success=slot.success,
                error=slot.error,
                shot_id=shot_id,
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
