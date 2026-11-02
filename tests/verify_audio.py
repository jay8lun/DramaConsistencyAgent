"""S5 音轨编排自检。

只测一件事：**每段音频是否严格等于对应镜长**。
这条不成立，后面所有镜头都会累积错位，而且错得越到后面越离谱，
在成片里表现为「第 4 镜开始口型对不上声音」——那种问题靠肉眼看片子很难定位。

用真实 ffmpeg 生成可控时长的音频，不依赖任何网络与 TTS 服务。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import find_bin  # noqa: E402
from models import Character, Scene, Script, Shot, VisualStyle  # noqa: E402
from pipeline import s5_compose as S5  # noqa: E402
from providers.base import TTSProvider  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


class StubTTS(TTSProvider):
    """可控 TTS：合成指定时长的静音，用来模拟「台词多长」。"""

    name = "stub-tts"
    is_mock = True
    price_per_call = 0.0

    def __init__(self, seconds: float = 1.5, fail: bool = False):
        self.seconds = seconds
        self.fail = fail
        self.calls: list[str] = []

    def synthesize(self, text, *, voice, out, allow_paid=False) -> Path:
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("模拟 TTS 服务不可用")
        exe = find_bin("ffmpeg")
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [exe, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
             "-t", f"{self.seconds:.3f}", "-c:a", "aac", "-b:a", "64k", str(out)],
            capture_output=True, check=True,
        )
        return out


def make_shot(idx: int, *, dialogue: str | None, duration: float = 5.0) -> Shot:
    return Shot(
        shot_id=f"shot_{idx:02d}", index=idx, scene_id="scene_01",
        duration=duration, camera="中景", action="站立",
        character_ids=["char_01"], dialogue=dialogue,
    )


def main() -> int:
    if not find_bin("ffmpeg"):
        print("SKIP: 没有 ffmpeg，音轨编排无法验证")
        return 0

    tmp = Path(tempfile.mkdtemp(prefix="s5_audio_"))

    print("=" * 64)
    print("[1] 有台词 / 无台词混排：每段必须等于镜长")
    print("=" * 64)
    shots = [
        make_shot(1, dialogue="我按的是十七楼。"),
        make_shot(2, dialogue=None),
        make_shot(3, dialogue="你也是这一层？"),
    ]
    tts = StubTTS(seconds=1.5)
    track, plan = S5.build_audio_track(shots, tts, out_root=tmp)
    check("生成音轨文件", track is not None and track.exists(), str(track))
    for s in shots:
        p = tmp / "voice" / f"{s.shot_id}.m4a"
        d = S5.media_duration(p) if p.exists() else None
        check(f"{s.shot_id} 音频段 = {s.duration:g}s",
              d is not None and abs(d - s.duration) < 0.1, f"实际 {d}")
    check("只有 2 镜调用了 TTS（无台词的镜不调用）", len(tts.calls) == 2, str(tts.calls))
    check("统计：2 句有台词 / 1 镜静音", plan.spoken == 2 and plan.silent == 1,
          f"spoken={plan.spoken} silent={plan.silent}")
    total_d = S5.media_duration(track) if track else None
    check("总音轨时长 = 视频总时长 15s",
          total_d is not None and abs(total_d - 15.0) < 0.3, f"实际 {total_d}")

    print()
    print("=" * 64)
    print("[2] 台词超长：先加速，加速不够才截断，且截断必须被记录")
    print("=" * 64)
    tmp2 = Path(tempfile.mkdtemp(prefix="s5_long_"))
    # 镜长 5s，配音 9s → 需要 1.8× 加速（未超 MAX_TEMPO=2.0）
    shots2 = [make_shot(1, dialogue="很长的一句台词" * 3)]
    _, plan2 = S5.build_audio_track(shots2, StubTTS(seconds=9.0), out_root=tmp2)
    check("超长台词触发加速", len(plan2.tempo_adjusted) == 1, str(plan2.tempo_adjusted))
    if plan2.tempo_adjusted:
        sid, tempo = plan2.tempo_adjusted[0]
        check(f"加速倍数约 1.8（实际 {tempo}）", 1.7 <= tempo <= 1.9, str(tempo))
    check("加速足够时不记为截断", plan2.truncated == [], str(plan2.truncated))
    d2 = S5.media_duration(tmp2 / "voice" / "shot_01.m4a")
    check("加速后仍严格等于 5s", d2 is not None and abs(d2 - 5.0) < 0.1, f"实际 {d2}")

    tmp3 = Path(tempfile.mkdtemp(prefix="s5_trunc_"))
    # 镜长 5s，配音 20s → 即使 2.0× 加速也只到 10s，必然截断
    _, plan3 = S5.build_audio_track(shots2, StubTTS(seconds=20.0), out_root=tmp3)
    check("加速上限 2.0×", plan3.tempo_adjusted and plan3.tempo_adjusted[0][1] == 2.0,
          str(plan3.tempo_adjusted))
    check("仍超长时记为截断（不静默吞掉）", plan3.truncated == ["shot_01"], str(plan3.truncated))
    check("截断写进 notes 且提示回 S1 改剧本",
          any("截断" in n and "S1" in n for n in plan3.notes), str(plan3.notes))

    print()
    print("=" * 64)
    print("[3] TTS 失败：补静音继续，不能炸掉整条 S5")
    print("=" * 64)
    tmp4 = Path(tempfile.mkdtemp(prefix="s5_fail_"))
    _, plan4 = S5.build_audio_track(shots, StubTTS(fail=True), out_root=tmp4)
    check("TTS 失败不抛异常（返回 plan 而非崩溃）", plan4 is not None)
    check("失败的镜记为 failed", len(plan4.failed) == 2, str(plan4.failed))
    check("失败的镜全部补静音", plan4.silent == 3, f"silent={plan4.silent}")
    check("失败原因写进 notes", any("配音失败" in n for n in plan4.notes), str(plan4.notes))
    d4 = S5.media_duration(tmp4 / "voice" / "shot_01.m4a")
    check("补静音后仍等于镜长（音画不会错位）",
          d4 is not None and abs(d4 - 5.0) < 0.1, f"实际 {d4}")

    print()
    print("=" * 64)
    print("[4] 空镜（environment_only）必须补静音，且不调用 TTS")
    print("=" * 64)
    tmp5 = Path(tempfile.mkdtemp(prefix="s5_empty_"))
    empty_shot = Shot(
        shot_id="shot_09", index=1, scene_id="scene_01", duration=5.0,
        camera="远景", action="空镜", character_ids=[], environment_only=True,
    )
    tts5 = StubTTS(seconds=1.5)
    _, plan5 = S5.build_audio_track([empty_shot], tts5, out_root=tmp5)
    check("空镜不调用 TTS", len(tts5.calls) == 0, str(tts5.calls))
    check("空镜记为静音", plan5.silent == 1, str(plan5.silent))

    print()
    print("=" * 64)
    print("[5] media_duration：无 ffprobe 时用 ffmpeg -i 解析")
    print("=" * 64)
    d = S5.media_duration(tmp / "voice" / "shot_01.m4a")
    check("能读出时长（无 ffprobe 也能工作）", d is not None and d > 0, str(d))
    check("不存在的文件返回 None", S5.media_duration(tmp / "nope.mp4") is None)

    print()
    print("=" * 64)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    print("=" * 64)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
