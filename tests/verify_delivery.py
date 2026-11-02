"""M1-S3d 自检：s4_video / s5_compose / report。

覆盖第 14.4 节 G 组（s5 的 2 条）+ 交付链路的诚实性要求：
  [1] 字幕时间轴按镜长累加（14.4-G 第 1 条）
  [2] 无台词镜头仍推进时间轴（不做这条断言，第 3 条字幕就会提前）
  [3] 拼接顺序严格按镜号（14.4-G 第 2 条）
  [4] 无 ffmpeg → 无损片段包 + 报告写明原因（6 章失败处理）
  [5] 无可用片段 → 明确失败，不产出空成片
  [6] S4 videoing → done
  [7] S4 degraded 出片后仍标 degraded（不伪装成 done，7.2）
  [8] S4 缺首帧 → failed（拒绝文生视频，9.1 第 1 条）
  [9] S4 技术重试恰好 2 次后 failed
  [10] 报告里的数字由记录算出：降级镜头标红、均分分母只算已判定镜
  [11] costs.json 估算/实际分开，实际价未知标 actual_unknown
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Script, Shot  # noqa: E402
from pipeline import report as R  # noqa: E402
from pipeline import s4_video as S4  # noqa: E402
from pipeline import s5_compose as S5  # noqa: E402
from pipeline.cost import CostLedger  # noqa: E402
from pipeline.project import ProjectCtx  # noqa: E402
from pipeline.state import StateStore, now_iso  # noqa: E402
from providers.tts import MockTTS  # noqa: E402

#: 显式传占位 TTS，不依赖 `S5.run(tts=None)` 的默认值。
#: 默认值是 EdgeTTS（要联网），14.7 要求测试断网可跑。当前两处调用都恰好在
#: 音轨生成之前返回，所以不会真的联网——但那是靠执行顺序侥幸成立的，
#: 一旦将来调整 run() 的步骤顺序就会静默变成联网测试。显式传参把这条锁死。
_NO_TTS = MockTTS()
from providers.base import ProviderError, VideoProvider  # noqa: E402

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


class StubVideo(VideoProvider):
    """写一个 mock 视频桩（无 ffmpeg 时的合法产物形态）。"""

    name = "stub-video"
    is_mock = True

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0

    def generate(self, keyframe, prompt, *, duration, out, allow_paid=False) -> Path:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ProviderError("模拟出片 502")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(f"MOCK_VIDEO duration={duration}".encode())
        return Path(out)


def new_project(tmp: Path, name: str) -> ProjectCtx:
    root = tmp / name
    for sub in ("anchors", "keyframes", "shots", "inbox"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return ProjectCtx(project_id=name, root=root, budget_cap=10.0,
                      channel="api", mock=True, created_at=now_iso())


def mk_shot(i: int, *, duration: float = 5.0, dialogue: str | None = None,
            chars: bool = True) -> Shot:
    return Shot(
        shot_id=f"shot_{i:02d}", index=i, scene_id="scene_01", duration=duration,
        camera="中景，固定机位", action="缓缓转头",
        character_ids=["char_01"] if chars else [],
        environment_only=not chars, dialogue=dialogue, emotion="平静",
    )


def mk_script(shots: list[Shot]) -> Script:
    from models import Character, Scene, VisualStyle

    return Script(
        title="测试片", logline="测试",
        style=VisualStyle(style_prompt="冷色调"),
        characters=[Character(char_id="char_01", name="甲", age_range="26-30岁", gender="女",
                              appearance="黑色齐肩直发，鹅蛋脸，细长眼，白皙肤色，偏瘦",
                              wardrobe="深灰大衣", source="brief")],
        scenes=[Scene(scene_id="scene_01", name="电梯", description="狭窄",
                      lighting="冷白", time_of_day="深夜")],
        shots=shots,
    )


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="m1s3d_"))

    # ---------------------------------------------------------------- SRT
    print("=" * 64)
    print("[1][2] 字幕时间轴按镜长累加（14.4-G 第 1 条）")
    print("=" * 64)
    shots = [
        mk_shot(1, dialogue="第一句"),
        mk_shot(2, dialogue=None),          # 无台词，仍要占 5s
        mk_shot(3, dialogue="第三句"),
    ]
    srt, n = S5.build_srt(shots)
    check("字幕条数只算有台词的镜", n == 2, f"{n}")
    check("第 1 条从 00:00:00,000 开始", "00:00:00,000 --> 00:00:05,000" in srt, srt)
    check("第 2 条起始 = 前两镜总时长 10s（无台词镜也计入）",
          "00:00:10,000 --> 00:00:15,000" in srt, srt)
    check("SRT 序号连续从 1 开始", srt.startswith("1\n") and "\n2\n" in srt, srt)

    # 这一条测的是**字幕时间轴的跨镜累加**，不是 Schema 的档位校验（那个在 models 用例里测）。
    # 当前视频模型只有 5 秒一档（`config.PLATFORM_TIERS` 从 .env 的 VIDEO_SUPPORTED_SECONDS 读，
    # 2026-10-29 从 (5,10) 收窄为 (5,)），所以这里用 `model_construct` 跳过校验造一个 10 秒镜头：
    # 10 秒只是把"下一镜起点 = 前面所有镜总时长"这个偏移放大到一眼能看出来。
    long_shot = Shot.model_construct(
        shot_id="shot_01", index=1, scene_id="scene_01", duration=10.0,
        camera="中景，固定机位", action="缓缓转头", character_ids=["char_01"],
        environment_only=False, dialogue="长镜", emotion="平静",
        qa=None, attempt_log=[], keyframe_path=None, video_path=None,
        image_prompt=None, video_prompt=None,
    )
    srt2, n2 = S5.build_srt([long_shot, mk_shot(2, dialogue="下一句")])
    check("10s 档位同样正确", "00:00:10,000 --> 00:00:15,000" in srt2, srt2)
    check("毫秒不进位成 1000", ",999" not in srt2 and ",000" in srt2)

    # ---------------------------------------------------------------- 拼接顺序
    print()
    print("=" * 64)
    print("[3] 拼接顺序严格按镜号（14.4-G 第 2 条）")
    print("=" * 64)
    ctx = new_project(tmp, "p1")
    # 故意乱序创建，且 filenames 与创建顺序无关
    for i in (4, 1, 3, 2):
        (ctx.shots_dir / f"shot_{i:02d}.mp4").write_bytes(b"MOCK_VIDEO x")
    order_shots = [mk_shot(3), mk_shot(1), mk_shot(4), mk_shot(2)]  # 输入也乱序
    included, paths = S5.collect_segments(ctx, order_shots)
    check("返回顺序 = index 升序", [s.index for s in included] == [1, 2, 3, 4],
          f"{[s.index for s in included]}")
    check("路径顺序与镜号一致", [p.stem for p in paths] == ["shot_01", "shot_02", "shot_03", "shot_04"],
          f"{[p.stem for p in paths]}")
    check("缺失的镜头不参与合成", len(included) == 4)

    # ---------------------------------------------------------------- 无 ffmpeg
    print()
    print("=" * 64)
    print("[4] 无 ffmpeg → 无损片段包 + 明确上报（不静默失败）")
    print("=" * 64)
    script4 = mk_script([mk_shot(1, dialogue="你好"), mk_shot(2, dialogue="再见")])
    for s in script4.shots:
        (ctx.shots_dir / f"{s.shot_id}.mp4").write_bytes(f"MOCK_VIDEO {s.shot_id}".encode())
    res4 = S5.run(ctx, script4, ffmpeg_ok=False, tts=_NO_TTS)
    check("状态 = degraded（不是 ok）", res4.status == "degraded", res4.status)
    check("不产出成片", res4.final is None)
    check("仍产出 srt（字幕可单独使用）", res4.srt is not None and res4.srt.exists())
    check("输出无损片段包", res4.package_dir is not None and res4.package_dir.exists())
    check("包内片段数与可用镜数一致", len(list(res4.package_dir.glob("*.mp4"))) == 2,
          f"{list(res4.package_dir.glob('*.mp4'))}")
    check("包内有 concat 清单", (res4.package_dir / "concat_list.txt").exists())
    check("包内有手工合成说明且写明原因",
          "ffmpeg" in (res4.package_dir / "ASSEMBLE.md").read_text(encoding="utf-8"))
    check("notes 里写明原因（报告要读到）",
          any("ffmpeg" in n for n in res4.notes), f"{res4.notes}")

    print()
    print("=" * 64)
    print("[5] 无可用片段 → 明确失败")
    print("=" * 64)
    ctx5 = new_project(tmp, "p5")
    res5 = S5.run(ctx5, mk_script([mk_shot(1)]), ffmpeg_ok=True, tts=_NO_TTS)
    check("状态 degraded 且无成片", res5.status == "degraded" and res5.final is None)
    check("原因写明没有片段", any("没有任何可用片段" in n for n in res5.notes), f"{res5.notes}")

    # ---------------------------------------------------------------- S4
    print()
    print("=" * 64)
    print("[6][7] S4：videoing → done；degraded 出片后仍是 degraded")
    print("=" * 64)
    ctx6 = new_project(tmp, "p6")
    store6 = StateStore(tmp / "state.db")
    ledger6 = CostLedger("p6", cap=10.0, db_path=tmp / "state.db")
    s_a, s_b = mk_shot(1), mk_shot(2)
    store6.ensure_shots("p6", ["shot_01", "shot_02"])
    for sid in ("shot_01", "shot_02"):
        (ctx6.keyframes_dir / f"{sid}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
        )
    # shot_01 走到 videoing
    store6.transition("p6", "shot_01", "keyframing")
    store6.transition("p6", "shot_01", "judging", score=90)
    store6.transition("p6", "shot_01", "videoing")
    # shot_02 直接降级（模拟重试耗尽）
    store6.transition("p6", "shot_02", "keyframing")
    store6.transition("p6", "shot_02", "judging", score=55)
    store6.transition("p6", "shot_02", "degraded")

    v = StubVideo()
    r_a = S4.run(ctx6, s_a, characters=script4.characters, provider=v, store=store6,
                 ledger=ledger6, allow_paid=True)
    check("videoing 出片成功 → done", r_a.status == "done", r_a.status)
    check("视频文件落盘", ctx6.video_of("shot_01").exists())
    r_b = S4.run(ctx6, s_b, characters=script4.characters, provider=v, store=store6,
                 ledger=ledger6, allow_paid=True)
    check("degraded 也出片（7.2 要进合成）", ctx6.video_of("shot_02").exists())
    check("但状态保持 degraded，不伪装成 done", r_b.status == "degraded", r_b.status)
    check("状态库里的 shot_02 仍是 degraded",
          store6.get("p6", "shot_02").status == "degraded", store6.get("p6", "shot_02").status)

    print()
    print("=" * 64)
    print("[8][9] S4：缺首帧 → failed；技术失败 2 次 → failed")
    print("=" * 64)
    ctx7 = new_project(tmp, "p7")
    store7 = StateStore(tmp / "state.db")
    ledger7 = CostLedger("p7", cap=10.0, db_path=tmp / "state.db")
    store7.ensure_shots("p7", ["shot_01", "shot_02"])
    store7.transition("p7", "shot_01", "keyframing")
    store7.transition("p7", "shot_01", "judging", score=90)
    store7.transition("p7", "shot_01", "videoing")
    # 不创建 keyframe → 缺首帧
    r8 = S4.run(ctx7, mk_shot(1), characters=script4.characters, provider=StubVideo(),
                store=store7, ledger=ledger7, allow_paid=True)
    check("缺首帧 → failed，不偷偷文生视频", r8.status == "failed", r8.status)
    check("拒因写明缺首帧", "缺首帧" in r8.note, r8.note)

    (ctx7.keyframes_dir / "shot_02.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    store7.transition("p7", "shot_02", "keyframing")
    store7.transition("p7", "shot_02", "judging", score=88)
    store7.transition("p7", "shot_02", "videoing")
    failing = StubVideo(fail_times=99)
    r9 = S4.run(ctx7, mk_shot(2), characters=script4.characters, provider=failing,
                store=store7, ledger=ledger7, allow_paid=True)
    check("技术失败 2 次 → failed", r9.status == "failed", r9.status)
    check("重试次数恰好 2（7.1 对 videoing 的额度）", failing.calls == 2, f"{failing.calls}")
    check("状态库落到 failed", store7.get("p7", "shot_02").status == "failed")

    # ---------------------------------------------------------------- 报告
    print()
    print("=" * 64)
    print("[10][11] 报告：数字由记录算出、降级如实标注、成本不混淆估算与实际")
    print("=" * 64)
    store6.save_qa("p6", "shot_01", {"score": 90, "passed": True, "degraded": False,
                                     "diff": "", "checks": [
                                         {"element": e, "matched": True, "expected": "x", "actual": "y"}
                                         for e in ("发型", "脸型", "眉眼", "肤色", "体型")]})
    store6.save_qa("p6", "shot_02", {"score": 55, "passed": False, "degraded": True,
                                     "diff": "发型不符", "checks": [
                                         {"element": "发型", "matched": False, "expected": "黑色齐肩直发",
                                          "actual": "红棕色马尾"}]})
    script6 = mk_script([s_a, s_b])
    ledger6.record(stage="video", provider="stub-video", operation="generate",
                   estimated=0.5, actual=0.5, shot_id="shot_01")
    ledger6.record(stage="video", provider="stub-video", operation="generate",
                   estimated=0.5, actual=None, success=False, error="502", shot_id="shot_02")

    costs_path = R.write_costs(ctx6, ledger6)
    payload = json.loads(costs_path.read_text(encoding="utf-8"))
    check("costs.json 有逐条记录", len(payload["entries"]) >= 2, f"{len(payload['entries'])}")
    unknown = [e for e in payload["entries"] if e["actual_cost"] is None]
    check("实际价缺失的调用被标 actual_unknown", unknown and unknown[0]["actual_unknown"] is True,
          f"{unknown[:1]}")
    check("spent 用估算价兜底（失败也可能扣费）", payload["spent"] > 0, f"{payload['spent']}")

    html_path = R.build(ctx6, script6, store=store6, ledger=ledger6)
    html = html_path.read_text(encoding="utf-8")
    check("report.html 已生成", html_path.exists() and "<html" in html)
    check("降级镜头出现在报告里并被标红",
          "degraded" in html and "#b3261e" in html, "未见降级标记")
    check("失败要素被列出", "发型" in html, "未列出未通过要素")
    check("报告含成本明细表", "成本明细" in html)
    check("报告含「需要知道的问题」段（不美化）", "需要知道的问题" in html)

    metrics = R.compute_metrics(ctx6, script6, store6, ledger6)
    check("一致性均分由分数算出（(90+55)/2=72.5）", metrics.score_avg == 72.5, f"{metrics.score_avg}")
    check("均分分母只算已判定镜", metrics.scored == 2, f"{metrics.scored}")
    check("降级镜被单独列出", metrics.degraded_shots == ["shot_02"], f"{metrics.degraded_shots}")

    store6.close()
    store7.close()
    ledger6.close()
    ledger7.close()

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
