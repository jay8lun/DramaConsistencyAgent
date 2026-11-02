"""视频阶段判定 + 人工确认的自检（M24）。全部离线，用 mock provider，不花钱。

验的是四件容易做错的事：

1. **不达标就停下，不自动重跑**。一条 10 秒视频约 ¥2，而重跑需要 2 分钟。
   「脸还能不能看」这种事，人看一眼 5 秒，模型只是筛子。
2. **重启不清待办**。`awaiting_review` 既不是中间态也不是终态，`reconcile()`
   必须跳过它——否则「第二天回来看」这件事就不成立。
3. **续跑不重出视频**。视频已在、判定没跑完时，只补判定。
4. **两条判定记录不互相覆盖**。S3 的首帧判定在 `shot_qa`，S4 的末帧判定在 `video_qa`；
   用同一张表会把「首帧 95 分」抹掉，报告里只剩一个数字。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Character, Scene, Shot, VisualStyle  # noqa: E402
from pipeline import s4_video as S4  # noqa: E402
from pipeline.cost import CostLedger  # noqa: E402
from pipeline.project import ProjectCtx  # noqa: E402
from pipeline.state import LEGAL_TRANSITIONS, StateStore, now_iso  # noqa: E402
from providers.base import VisionJudge  # noqa: E402
from providers.mock import MockVideo, write_solid_png  # noqa: E402

PASS, FAIL = 0, 0
FAILED: list[str] = []


def ok(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  {extra}" if extra else ""))
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL  {name}  {extra}")


class CountingVideo(MockVideo):
    """记录调用次数：用来断言「没有偷偷重出」。"""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, keyframe, prompt, *, duration, out, allow_paid=False) -> Path:
        self.calls += 1
        return super().generate(keyframe, prompt, duration=duration, out=out, allow_paid=allow_paid)


class FixedJudge(VisionJudge):
    name = "fixed-judge"
    is_mock = True

    def __init__(self, score: int, diff: str = ""):
        self.score = score
        self.diff = diff
        self.calls = 0

    def compare(self, image, reference, elements, *, allow_paid: bool = False) -> dict:
        self.calls += 1
        return {
            "score": self.score,
            "checks": [{"element": e, "matched": True, "expected": "见角色卡", "actual": "一致"}
                       for e in elements],
            "diff": self.diff,
        }


def new_project(tmp: Path, name: str) -> ProjectCtx:
    root = tmp / name
    for sub in ("anchors", "keyframes", "shots", "inbox"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return ProjectCtx(project_id=name, root=root, budget_cap=50.0,
                      channel="api", mock=True, created_at=now_iso())


def make_shot(sid: str = "shot_01") -> Shot:
    return Shot(
        shot_id=sid, index=1, scene_id="scene_01", character_ids=["char_01"],
        camera="中景，固定机位", action="她微微低头", emotion="紧张",
        duration=5.0, dialogue=None, environment_only=False,
    )


def make_char(ctx: ProjectCtx) -> Character:
    anchor = ctx.anchors_dir / "char_01_front.png"
    write_solid_png(anchor, 720, 1280, (60, 70, 90))
    return Character(
        char_id="char_01", name="林晚", age_range="26-30岁", gender="女",
        appearance="黑色齐肩直发，鹅蛋脸，眉眼平直，肤色偏白，体型偏瘦",
        wardrobe="深灰色羊毛大衣", source="image",
        truth_image=str(anchor), anchor_path=str(anchor),
    )


def prep_videoing(ctx: ProjectCtx, store: StateStore, sid: str = "shot_01") -> None:
    """把一镜推到「首帧已过、等出片」的状态。"""
    store.ensure_shots(ctx.project_id, [sid])
    store.transition(ctx.project_id, sid, "keyframing", reason="test")
    store.transition(ctx.project_id, sid, "judging", reason="test")
    store.transition(ctx.project_id, sid, "videoing", reason="test")
    write_solid_png(ctx.keyframe_of(sid), 720, 1280, (90, 90, 90))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="m24_"))
    print("=" * 66)
    print("视频判定 + 人工确认自检")
    print("=" * 66)

    # ---------------- 1. 状态机 ----------------
    print("\n[1] 状态机：新增两条通路")
    ok("videoing → video_judging 合法", "video_judging" in LEGAL_TRANSITIONS["videoing"])
    ok("video_judging → awaiting_review 合法", "awaiting_review" in LEGAL_TRANSITIONS["video_judging"])
    ok("awaiting_review → done / degraded / videoing 三个出口齐备",
       LEGAL_TRANSITIONS["awaiting_review"] == frozenset({"done", "degraded", "videoing"}),
       str(sorted(LEGAL_TRANSITIONS["awaiting_review"])))
    ok("awaiting_review 不能直接进 failed（人工裁决不存在「失败」这个选项）",
       "failed" not in LEGAL_TRANSITIONS["awaiting_review"])

    # ---------------- 2. 判定不达标 → 停下等人 ----------------
    print("\n[2] 末帧判定不达标 → awaiting_review，且不自动重跑")
    ctx = new_project(tmp, "p1")
    store = StateStore(tmp / "state.db")
    ledger = CostLedger("p1", cap=50.0, db_path=tmp / "state.db")
    char = make_char(ctx)
    shot = make_shot()
    scene = Scene(scene_id="scene_01", name="电梯", description="狭窄电梯厢",
                  lighting="冷白顶灯", time_of_day="深夜")
    style = VisualStyle(style_prompt="写实都市悬疑", negative_prompt="", aspect_ratio="9:16")

    prep_videoing(ctx, store)
    video = CountingVideo()
    low = FixedJudge(55, diff="人物面部在中段开始漂移")
    r = S4.run(ctx, shot, characters=[char], provider=video, store=store, ledger=ledger,
               allow_paid=True, judge_provider=low)
    ok("低分 → 状态 awaiting_review", r.status == "awaiting_review", r.status)
    ok("视频只出了 1 次（没有自动重跑）", video.calls == 1, f"calls={video.calls}")
    ok("判定调用 1 次", low.calls == 1, f"calls={low.calls}")
    ok("结果里带末帧分数", r.video_score == 55, f"{r.video_score}")
    vqa = store.load_video_qa("p1", "shot_01") or {}
    ok("视频判定落库（score=55）", vqa.get("score") == 55, str(vqa.get("score")))
    ok("落库带末帧路径（人能直接去看）", bool(vqa.get("last_frame")), str(vqa.get("last_frame")))
    ok("状态日志写明「等待人工确认」",
       any("人工确认" in (e.get("reason") or "") for e in store.log_entries("p1")),
       "已记录")

    # ---------------- 3. 重启不清待办 ----------------
    print("\n[3] 重启（reconcile）不冲掉 awaiting_review")
    rep = store.reconcile("p1", {"shot_01": {"keyframe": str(ctx.keyframe_of("shot_01")),
                                             "video": str(ctx.video_of("shot_01"))}})
    ok("reconcile 后状态仍是 awaiting_review",
       store.get("p1", "shot_01").status == "awaiting_review",
       store.get("p1", "shot_01").status)
    ok("reconcile 明确写了「保持不动」",
       any("保持不动" in (a.reason or "") for a in rep.actions),
       str([a.reason for a in rep.actions][:3]))

    # 再跑一次 S4：不该重出视频，也不该重复判定
    r2 = S4.run(ctx, shot, characters=[char], provider=video, store=store, ledger=ledger,
                allow_paid=True, judge_provider=low)
    ok("待确认状态再跑 S4 → 直接跳过，不重出也不重判",
       r2.status == "awaiting_review" and video.calls == 1 and low.calls == 1,
       f"video={video.calls} judge={low.calls}")

    # ---------------- 4. 人工三选一 ----------------
    print("\n[4] 人工裁决三个出口")
    store.transition("p1", "shot_01", "done", reason="人工放行：是我要的效果")
    ok("accept → done", store.get("p1", "shot_01").status == "done")

    ctx2 = new_project(tmp, "p2")
    store2 = StateStore(tmp / "state.db")
    ledger2 = CostLedger("p2", cap=50.0, db_path=tmp / "state.db")
    char2 = make_char(ctx2)
    prep_videoing(ctx2, store2)
    v2 = CountingVideo()
    S4.run(ctx2, make_shot(), characters=[char2], provider=v2, store=store2, ledger=ledger2,
           allow_paid=True, judge_provider=FixedJudge(50))
    store2.transition("p2", "shot_01", "degraded", reason="人工放行但标红：能看但有瑕疵")
    ok("degrade → degraded（报告里会列出）", store2.get("p2", "shot_01").status == "degraded")

    ctx3 = new_project(tmp, "p3")
    store3 = StateStore(tmp / "state.db")
    ledger3 = CostLedger("p3", cap=50.0, db_path=tmp / "state.db")
    char3 = make_char(ctx3)
    prep_videoing(ctx3, store3)
    v3 = CountingVideo()
    S4.run(ctx3, make_shot(), characters=[char3], provider=v3, store=store3, ledger=ledger3,
           allow_paid=True, judge_provider=FixedJudge(50))
    store3.transition("p3", "shot_01", "videoing", reason="人工驳回：脸崩了，重出")
    ok("reject → 回到 videoing（下次跑流水线会重出）",
       store3.get("p3", "shot_01").status == "videoing")
    r3 = S4.run(ctx3, make_shot(), characters=[char3], provider=v3, store=store3,
                ledger=ledger3, allow_paid=True, judge_provider=FixedJudge(50))
    ok("驳回后确实重出了视频（calls 2）", v3.calls == 2, f"calls={v3.calls}")
    ok("重出后仍不达标 → 再次停下等人", r3.status == "awaiting_review", r3.status)

    # ---------------- 5. 续跑只补判定 ----------------
    print("\n[5] 视频已在、判定未跑完 → 只补判定，不重出（一条约 ¥2）")
    ctx4 = new_project(tmp, "p4")
    store4 = StateStore(tmp / "state.db")
    ledger4 = CostLedger("p4", cap=50.0, db_path=tmp / "state.db")
    char4 = make_char(ctx4)
    prep_videoing(ctx4, store4)
    v4 = CountingVideo()
    S4.run(ctx4, make_shot(), characters=[char4], provider=v4, store=store4, ledger=ledger4,
           allow_paid=True, judge_provider=FixedJudge(50))
    # 模拟「判定前被 kill」：状态退回 video_judging，视频文件还在
    store4._force("p4", "shot_01", "video_judging", "模拟中断")
    judge4 = FixedJudge(92)
    r4 = S4.run(ctx4, make_shot(), characters=[char4], provider=v4, store=store4,
                ledger=ledger4, allow_paid=True, judge_provider=judge4)
    ok("续跑没有重出视频（calls 仍是 1）", v4.calls == 1, f"calls={v4.calls}")
    ok("续跑补做了判定", judge4.calls == 1, f"calls={judge4.calls}")
    ok("补判后达标 → done", r4.status == "done", r4.status)

    # ---------------- 6. 两条判定记录不互相覆盖 ----------------
    print("\n[6] 首帧判定与视频判定分开存")
    store4.save_qa("p4", "shot_01", {"score": 95, "passed": True, "checks": [], "diff": ""})
    store4.save_video_qa("p4", "shot_01", {"score": 88, "diff": "轻微漂移", "checks": []})
    ok("shot_qa 仍是首帧分数 95", (store4.load_qa("p4", "shot_01") or {}).get("score") == 95)
    ok("video_qa 是视频分数 88", (store4.load_video_qa("p4", "shot_01") or {}).get("score") == 88)

    # ---------------- 7. 不达标时 S5 不合成 ----------------
    print("\n[7] 有待确认镜头时不得合成（禁止静默放行）")
    rows = store.awaiting_review("p1")
    ok("accept 之后待办清单为空", rows == [], str(rows))
    rows3 = store3.awaiting_review("p3")
    ok("驳回重出仍不达标的镜头留在待办里", len(rows3) == 1 and rows3[0]["shot_id"] == "shot_01",
       str([r["shot_id"] for r in rows3]))

    for s in (store, store2, store3, store4):
        s.close()
    for lg in (ledger, ledger2, ledger3, ledger4):
        lg.close()

    print()
    print("=" * 66)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    if FAILED:
        print("失败项：")
        for n in FAILED:
            print("  -", n)
    print("=" * 66)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
