"""M1-S3a 自检：状态机 + 断点续跑 + 记账封顶。

覆盖范围对齐第 14.4 节：
  - C 组 `pipeline/state.py` 的 8 条用例
  - D 组 `pipeline/cost.py` 的 4 条用例
  - 7.3 未给用例、但实现必须处理的两种情形（首帧已存在 → 不重复扣费；
    截断的 mp4 → 不能当成功）

全部在临时目录里跑，不碰真实 `data/state.db`，可重复执行。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cost import BudgetExceeded, CostLedger  # noqa: E402
from pipeline.state import (  # noqa: E402
    StateStore,
    TransitionRejected,
    artifact_ok,
)
from providers.base import PaidCallBlocked  # noqa: E402
from providers.mock import write_solid_png  # noqa: E402

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


def expect_raises(name: str, fn, exc_type=Exception, contains: str = "") -> None:
    try:
        fn()
    except exc_type as exc:  # noqa: PERF203
        ok = (contains in str(exc)) if contains else True
        check(name, ok, f"异常信息不含「{contains}」：{exc}")
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"抛了 {type(exc).__name__}，期望 {exc_type.__name__}：{exc}")
    else:
        check(name, False, f"未抛 {exc_type.__name__}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="m1s3a_"))
    db = tmp / "state.db"
    proj = "p_test"
    arts = tmp / "artifacts"
    arts.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("[1] 状态机合法迁移（14.4-C 前 3 条）")
    print("=" * 64)
    store = StateStore(db)
    store.ensure_shots(proj, ["shot_01", "shot_02", "shot_03", "shot_04", "shot_05"])

    st = store.transition(proj, "shot_01", "keyframing")
    check("test_pending_to_keyframing", st.status == "keyframing", f"实际 {st.status}")

    st = store.transition(proj, "shot_01", "judging", score=88)
    check("test_keyframing_to_judging", st.status == "judging" and st.score == 88,
          f"实际 {st.status} score={st.score}")

    st = store.transition(proj, "shot_01", "videoing")
    check("test_judging_to_videoing_on_pass", st.status == "videoing", f"实际 {st.status}")

    print()
    print("=" * 64)
    print("[2] 质量重试：judging → keyframing 且 attempts +1（14.4-C）")
    print("=" * 64)
    store.transition(proj, "shot_02", "keyframing")
    store.transition(proj, "shot_02", "judging", score=70)
    before = store.get(proj, "shot_02")
    st = store.transition(proj, "shot_02", "keyframing", reason="质量重试，带 diff 增量", quality_retry=True)
    check("test_judging_back_to_keyframing_on_retry",
          st.status == "keyframing" and st.quality_attempts == before.quality_attempts + 1,
          f"{st.status} quality {before.quality_attempts}→{st.quality_attempts}")

    print()
    print("=" * 64)
    print("[3] 非法迁移与终态互斥（14.4-C）")
    print("=" * 64)
    store.transition(proj, "shot_03", "keyframing")
    store.transition(proj, "shot_03", "judging", score=92)
    store.transition(proj, "shot_03", "videoing")
    store.transition(proj, "shot_03", "done")
    expect_raises("test_illegal_transition_rejected", lambda: store.transition(proj, "shot_03", "keyframing"),
                  TransitionRejected, "非法迁移")
    expect_raises("test_terminal_states_are_exclusive（done→degraded）",
                  lambda: store.transition(proj, "shot_03", "degraded"), TransitionRejected, "终态")
    store.transition(proj, "shot_04", "keyframing")
    store.transition(proj, "shot_04", "failed")
    expect_raises("test_terminal_states_are_exclusive（failed→done）",
                  lambda: store.transition(proj, "shot_04", "done"), TransitionRejected, "终态")

    print()
    print("=" * 64)
    print("[4] 技术重试与技术/质量分开计数（7.1）")
    print("=" * 64)
    store.mark_tech_attempt(proj, "shot_05", reason="首帧接口 503", cost_delta=0.25)
    store.mark_tech_attempt(proj, "shot_05", reason="首帧接口 503", cost_delta=0.25)
    s5 = store.get(proj, "shot_05")
    check("技术重试累加到 tech_attempts 且不改状态",
          s5.tech_attempts == 2 and s5.status == "pending" and abs(s5.cost - 0.5) < 1e-9,
          f"tech={s5.tech_attempts} status={s5.status} cost={s5.cost}")
    check("质量重试不计入技术重试", s5.quality_attempts == 0, f"quality={s5.quality_attempts}")
    check("attempts 属性 = 两者之和", s5.attempts == 2, f"attempts={s5.attempts}")

    print()
    print("=" * 64)
    print("[5] 断点续跑：文件为准（7.3）")
    print("=" * 64)
    kf = arts / "shot_05.png"
    write_solid_png(kf)
    vid = arts / "shot_05.mp4"
    vid.write_bytes(b"MOCK_VIDEO duration=5 keyframe=shot_05.png")

    # shot_01 停在 videoing，视频不存在但首帧在 → 保持 videoing，不重做首帧
    rep = store.reconcile(proj, {"shot_01": {"keyframe": kf, "video": arts / "nope.mp4"}})
    changed = {a.shot_id: (a.from_status, a.to_status) for a in rep.changed}
    check("videoing 且首帧在、视频缺 → 保持 videoing（不重复扣首帧的钱）",
          store.get(proj, "shot_01").status == "videoing", store.get(proj, "shot_01").status)
    # 首帧和视频都没有 → 只能重做
    rep = store.reconcile(proj, {"shot_01": {"keyframe": arts / "none.png",
                                             "video": arts / "nope.mp4"}})
    check("videoing 且首帧也没有 → pending 重做",
          store.get(proj, "shot_01").status == "pending", store.get(proj, "shot_01").status)

    # shot_04 是 failed → 重新排队，技术计数归零、质量计数保留
    s4_before = store.get(proj, "shot_04")
    rep = store.reconcile(proj, {})
    s4_after = store.get(proj, "shot_04")
    check("test_resume_skips_done_and_degraded（failed 重新排队）",
          s4_after.status == "pending" and s4_after.tech_attempts == 0
          and s4_after.quality_attempts == s4_before.quality_attempts,
          f"{s4_before.status}→{s4_after.status} tech={s4_after.tech_attempts}")
    check("done 不被重整动过",
          store.get(proj, "shot_03").status == "done", store.get(proj, "shot_03").status)

    # shot_05：pending + 首帧已存在 → 直接进 judging，不重复扣费
    rep = store.reconcile(proj, {"shot_05": {"keyframe": kf}})
    check("pending 但首帧已存在 → judging（文件为准，不重生成）",
          store.get(proj, "shot_05").status == "judging", store.get(proj, "shot_05").status)

    # judging 但分数没落库 → 回 pending 重判
    rep = store.reconcile(proj, {"shot_05": {"keyframe": kf}})
    act = [a for a in rep.actions if a.shot_id == "shot_05"]
    check("judging 无分数 → 回 pending 重判",
          store.get(proj, "shot_05").status == "pending", store.get(proj, "shot_05").status)

    # 造一个「judging 且已有分数」的场景：pending→keyframing→judging(score)
    store.transition(proj, "shot_05", "keyframing")
    store.transition(proj, "shot_05", "judging", score=64)
    rep = store.reconcile(proj, {"shot_05": {"keyframe": kf}})
    check("judging 且有分数 → 保持 judging，不重调 VLM",
          store.get(proj, "shot_05").status == "judging",
          f"实际 {store.get(proj, 'shot_05').status}")

    print()
    print("=" * 64)
    print("[6] 续跑幂等（14.4-C 最后一条）")
    print("=" * 64)
    snap1 = [(s.shot_id, s.status, s.tech_attempts, s.quality_attempts) for s in store.all(proj)]
    rep2 = store.reconcile(proj, {"shot_05": {"keyframe": kf}})
    snap2 = [(s.shot_id, s.status, s.tech_attempts, s.quality_attempts) for s in store.all(proj)]
    rep3 = store.reconcile(proj, {"shot_05": {"keyframe": kf}})
    snap3 = [(s.shot_id, s.status, s.tech_attempts, s.quality_attempts) for s in store.all(proj)]
    check("test_resume_is_idempotent（两次重整状态不变）",
          snap2 == snap3, f"{snap2} != {snap3}")
    check("第二次重整无状态变化", len(rep3.changed) == 0, f"{[a.shot_id for a in rep3.changed]}")

    print()
    print("=" * 64)
    print("[7] 产物判据：非空 ≠ 可用（7.3 的洞）")
    print("=" * 64)
    good_png = arts / "ok.png"
    write_solid_png(good_png)
    bad_png = arts / "trunc.png"
    bad_png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)
    check("正常 PNG → True", artifact_ok(good_png, kind="image"))
    check("截断 PNG（魔数对但内容不全）→ 仍按图片魔数放行（图片无轻量校验手段）",
          artifact_ok(bad_png, kind="image") is True)
    tiny_mp4 = arts / "tiny.mp4"
    tiny_mp4.write_bytes(b"\x00" * 100)
    check("100 字节的 mp4 → False（下载中断的典型形态）", artifact_ok(tiny_mp4, kind="video") is False)
    check("mock 视频桩 → True（无 ffmpeg 时也要能跑 mock 链路）",
          artifact_ok(vid, kind="video") is True)
    check("不存在的文件 → False", artifact_ok(arts / "nope.png", kind="image") is False)

    print()
    print("=" * 64)
    print("[8] 记账与预算封顶（14.4-D 四条）")
    print("=" * 64)
    ledger = CostLedger(proj, cap=1.0, db_path=db)

    called = {"n": 0}

    def fake_provider_call():
        called["n"] += 1
        return "never"

    ledger.record(stage="keyframe", provider="mock-image", operation="generate",
                  estimated=0.25, actual=0.25, shot_id="shot_01")
    ledger.record(stage="judging", provider="mock-vlm", operation="compare",
                  estimated=0.08, actual=0.08, shot_id="shot_01")
    check("估算价与实际价都落库",
          all(r["estimated_cost"] > 0 and r["actual_cost"] is not None for r in ledger.entries()),
          f"{ledger.entries()}")
    check("spent 累计正确", abs(ledger.spent() - 0.33) < 1e-9, f"spent={ledger.spent()}")

    # 失败调用**不计入 spent**（2026-10-29 改口径）。
    # 原用例叫 test_failed_call_still_charged，断言「失败仍计入花费」。
    # 那个假设在六镜真跑时把护栏卡死了：两条 param 错误的 400 拒绝（平台不扣费）
    # 贡献了 ¥4.00 估算，spent 虚报 ¥9.39，补拍一镜需要 ¥1.00 时被判定超预算直接拒绝，
    # 而账户里还有 ¥4.61。**护栏不能基于虚构支出拒绝真实工作。**
    # 失败调用的估算额仍然落库（estimated_cost），只是不占预算。
    ledger.record(stage="video", provider="mock-video", operation="generate",
                  estimated=1.5, actual=None, success=False, error="400 参数不兼容", shot_id="shot_02")
    check("失败调用不计入 spent（4xx 拒绝平台不扣费）",
          abs(ledger.spent() - 0.33) < 1e-9, f"spent={ledger.spent()}")
    check("失败调用的估算额仍可查（审计不丢）",
          any(r["success"] == 0 and r["estimated_cost"] == 1.5 for r in ledger.entries()))

    # 补一条成功调用把预算吃满，再验封顶（原来靠失败调用占满，口径变了要另找路径）
    ledger.record(stage="video", provider="mock-video", operation="generate",
                  estimated=0.6, actual=0.6, shot_id="shot_03")
    check("成功调用照常计入 spent", abs(ledger.spent() - 0.93) < 1e-9, f"spent={ledger.spent()}")

    # 超预算：先抛异常，再断言 provider 未被调用
    reached = False
    try:
        ledger.check_before_call(0.5, what="test")
        fake_provider_call()
        reached = True
    except BudgetExceeded:
        pass
    check("test_budget_cap_blocks_before_call（超预算先抛、provider 未被调用）",
          (not reached) and called["n"] == 0, f"reached={reached} called={called['n']}")

    # paid() 上下文：未授权 → PaidCallBlocked，且不留记录
    before_n = len(ledger.entries())
    expect_raises("未 allow_paid 抛 PaidCallBlocked", lambda: _use_paid(ledger, allow=False), PaidCallBlocked)
    check("未授权的调用不留记录（它根本没发生）", len(ledger.entries()) == before_n)

    # paid() 出口必落一条审计记录（成功）
    locked = CostLedger("p_locked", cap=100.0, db_path=db)
    with locked.paid(0.08, stage="judging", provider="vlm", operation="compare",
                     allow_paid=True, shot_id="shot_01") as slot:
        slot.charge(0.075)
    rows = locked.entries()
    check("test_paid_call_requires_audit_entry（成功调用有记录）",
          len(rows) == 1 and rows[0]["actual_cost"] == 0.075 and rows[0]["success"] == 1,
          f"{rows}")

    # paid() 内抛异常 → 依然留记录（success=0）
    def boom():
        with locked.paid(0.08, stage="keyframe", provider="img", operation="generate",
                         allow_paid=True, shot_id="shot_02") as slot:
            slot.charge(0.25)
            raise RuntimeError("上游 503")

    try:
        boom()
    except RuntimeError:
        pass
    rows = locked.entries()
    check("paid() 内异常仍落审计记录（success=0 且保留实际扣费）",
          len(rows) == 2 and rows[1]["success"] == 0 and rows[1]["actual_cost"] == 0.25
          and "503" in (rows[1]["error"] or ""),
          f"{rows[-1]}")

    print()
    print("=" * 64)
    print("[9] WAL 与日志（7.5 并发前提 / 7.3 写日志要求）")
    print("=" * 64)
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    check("journal_mode = wal（多线程同写不会 database is locked）", str(mode).lower() == "wal", f"实际 {mode}")
    logs = store.log_entries(proj)
    check("每次迁移都留了 state_log", len(logs) > 10, f"{len(logs)} 条")
    check("重整也留了日志（reason 含「续跑」）",
          any("续跑" in (r["reason"] or "") for r in logs), "无续跑日志")

    store.close()
    ledger.close()
    locked.close()

    print()
    print("=" * 64)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    if FAILED_NAMES:
        print("失败项：")
        for n in FAILED_NAMES:
            print("  -", n)
    print("=" * 64)
    return 0 if FAIL == 0 else 1


def _use_paid(ledger: CostLedger, *, allow: bool) -> None:
    with ledger.paid(0.08, stage="judging", provider="vlm", operation="compare", allow_paid=allow):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
