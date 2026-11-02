"""S5 · 合成（第 6 章 S5）。

步骤：TTS 逐句配音 → 字幕生成与对齐 → 片段拼接 → BGM 混音 → 导出 9:16。

**字幕时间轴按镜长累加**，不是按语音时长——14.4-G 的断言就是
「第 N 条字幕起始时间 = 前 N-1 镜总时长」。语音快了慢了都靠字幕兜住，
反过来（按语音排时间轴）会让画面和台词错位。

**ffmpeg 缺失或失败时不静默失败**（6 章明文）：输出「无损片段包」
（原始片段 + concat 清单 + 手工合成命令），并在报告里写明原因。
无声无字幕地交一个半成品，比交不出东西更糟——使用者会以为成片就是这样。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from config import find_bin
from models import Script, Shot

#: 字幕样式。Windows 上必须显式指定字体，否则中文全是方块（7.6 已知坑）。
DEFAULT_FONT = "Microsoft YaHei"
#: 常见中文字体的候选路径。找不到就退到 ffmpeg 的 fontconfig 默认，并记一条告警。
FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)

#: 字体文件 → libass 认得**字体族名**。这两件事不一样：
#: `find_font()` 找到的是文件路径，而 `force_style` 里写的是族名，libass 按族名去
#: fontconfig 里查。容器里只装了 Noto Sans CJK，却硬写 `Microsoft YaHei`，
#: libass 查不到就退回默认字体——**如果能找到一个含汉字的默认字体就看不出问题，
#: 找不到就是一片方块，而且 ffmpeg 照样返回 0**。这就是 7.6 那条坑的真正成因。
_FONT_FAMILY_BY_FILE: tuple[tuple[str, str], ...] = (
    ("msyhbd", "Microsoft YaHei"),
    ("msyh", "Microsoft YaHei"),
    ("simhei", "SimHei"),
    ("NotoSansCJK", "Noto Sans CJK SC"),
    ("NotoSerifCJK", "Noto Serif CJK SC"),
)  # type: ignore[assignment]


SEG_DIR = "segments"


@dataclass
class ComposeResult:
    status: str  # ok / degraded
    final: Path | None = None
    srt: Path | None = None
    audio: Path | None = None
    segments: list[Path] = field(default_factory=list)
    package_dir: Path | None = None
    notes: list[str] = field(default_factory=list)
    cue_count: int = 0
    total_duration: float = 0.0
    spoken: int = 0
    silent: int = 0
    tempo_adjusted: list[tuple[str, float]] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- 字幕


def _ts(seconds: float) -> str:
    """SRT 时间戳 `HH:MM:SS,mmm`。用整数毫秒算，避免浮点累积出 00:00:04,999。"""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(shots: Sequence[Shot]) -> tuple[str, int]:
    """按镜长累加生成 SRT。返回 (srt 文本, 字幕条数)。

    **时间轴按 `shot.duration` 累加**，与台词长短无关：一镜一句台词是 4.7 的刻意约束，
    所以「镜」就是时间轴的刻度。没有台词的镜头只推进游标、不产生字幕条目——
    但它的时长必须计入，否则后面每条字幕都会提前。
    """
    lines: list[str] = []
    cursor = 0.0
    count = 0
    for shot in sorted(shots, key=lambda s: s.index):
        start = cursor
        cursor += float(shot.duration)
        text = (shot.dialogue or "").strip()
        if not text:
            continue
        count += 1
        lines.append(str(count))
        lines.append(f"{_ts(start)} --> {_ts(cursor)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines), count


# ---------------------------------------------------------------- ffmpeg


def find_ffmpeg() -> str | None:
    """先看 `FFMPEG_DIR` 配置，再退到 PATH。见 `config.find_bin` 的说明。"""
    return find_bin("ffmpeg")


def find_font() -> str | None:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def font_family() -> str:
    """当前环境**实际可用**的中文字体族名（写进 ffmpeg 的 `force_style`）。

    三档优先级：
      1. 显式配置（.env 的 `SUBTITLE_FONT`）——换字体不用改代码
      2. 按 `find_font()` 找到的文件推断族名
      3. 退回 `DEFAULT_FONT`（Windows 默认，也是历史行为）

    为什么不能一直用 `DEFAULT_FONT`：容器里没有 Microsoft YaHei，libass 按族名
    查不到会退回默认字体，**找不到含汉字的就烧成方块，而 ffmpeg 返回码仍是 0**。
    这类失败不报错、只在成片里显形，所以族名必须跟实际装了的字体对齐。
    """
    override = (os.environ.get("SUBTITLE_FONT") or "").strip()
    if override:
        return override
    found = find_font()
    if found:
        name = Path(found).name
        for fragment, family in _FONT_FAMILY_BY_FILE:
            if fragment.lower() in name.lower():
                return family
    return DEFAULT_FONT



def _run_ffmpeg(args: list[str], *, timeout: int = 300) -> tuple[bool, str]:
    exe = find_ffmpeg()
    if not exe:
        return False, "ffmpeg 不在 PATH 中"
    try:
        proc = subprocess.run([exe, *args], capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg 超时（>{timeout}s）"
    except OSError as exc:
        return False, f"ffmpeg 启动失败：{exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return False, " | ".join(tail) or f"退出码 {proc.returncode}"
    return True, ""


def _filter_escape(path: Path) -> str:
    """ffmpeg 滤镜参数里的 Windows 路径要转义：`C:/a.srt` → `C\\:/a.srt`。

    这是 7.6「中文/路径相关缺字与找不到文件」那类坑的根源之一：
    不转义时 ffmpeg 会把盘符后的冒号当参数分隔符，报找不到文件。
    """
    return str(path).replace("\\", "/").replace(":", r"\:")


def media_duration(path: Path) -> float | None:
    """读媒体时长。**本机没有 ffprobe**（imageio-ffmpeg 那份包不带），
    所以用 `ffmpeg -i` 解析 stderr 的 `Duration:` 行。

    这不是凑合：ffmpeg 对未知输入返回非零码但照样打印元信息，解析它得到的时长
    与 ffprobe 同源。拿不到就返回 None，调用方必须按「未知」处理而不是当 0。
    """
    exe = find_ffmpeg()
    if not exe:
        return None
    proc = subprocess.run([exe, "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


# ---------------------------------------------------------------- 音轨


#: 台词超长时允许的最大加速倍数。再高就听不清了，不如如实标出来让人改剧本。
MAX_TEMPO = 2.0


@dataclass
class AudioPlan:
    """音轨编排结果。逐镜保留，便于报告里定位「哪一镜的配音有问题」。"""

    total: float = 0.0
    spoken: int = 0          # 真正有台词的镜数
    silent: int = 0          # 补静音的镜数（含空镜）
    tempo_adjusted: list[tuple[str, float]] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_audio_track(
    shots: Sequence[Shot],
    tts,
    *,
    out_root: Path,
    gender_by_char: dict[str, str] | None = None,
) -> tuple[Path | None, AudioPlan]:
    """逐镜合成配音并拼成一条与视频等长的音轨。

    **每段严格等于 `shot.duration`**：多段音频靠 concat 拼接，只要有一段的时长
    与对应镜头不一致，后面所有镜头都会音画错位——而且是累积错位，越到后面越离谱。
    所以补静音（`apad`）与压缩（`atempo`）都不是可选项。

    台词超长时的处置顺序：① 加速到最多 `MAX_TEMPO` 倍 → ② 仍超则截断并记入
    `truncated`。**截断是丢字，属于事故**，它的正确修法在 S1 改剧本，这里只负责
    如实上报，不静默吞掉。
    """
    plan = AudioPlan()
    voice_dir = out_root / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    pieces: list[Path] = []

    for shot in sorted(shots, key=lambda s: s.index):
        target = float(shot.duration)
        plan.total += target
        seg = voice_dir / f"{shot.shot_id}.m4a"
        text = (shot.dialogue or "").strip()

        raw: Path | None = None
        if text:
            try:
                raw = tts.synthesize(
                    text,
                    voice=_voice_of(shot, gender_by_char or {}),
                    out=voice_dir / f"{shot.shot_id}_raw.m4a",
                    allow_paid=True,  # 免费引擎价格为 0；付费引擎由账本另计
                )
            except Exception as exc:  # noqa: BLE001 - 任何 TTS 失败都不该炸整条 S5
                plan.failed.append((shot.shot_id, f"{type(exc).__name__}: {exc}"))
                plan.notes.append(f"{shot.shot_id} 配音失败，已补静音：{exc}")

        if raw is None:
            ok, err = _ffmpeg_silence(seg, target)
            if not ok:
                return None, plan
            plan.silent += 1
            pieces.append(seg)
            continue

        plan.spoken += 1
        got = media_duration(raw) or 0.0
        tempo = 1.0
        if got > target + 0.05 and got > 0:
            tempo = min(got / target, MAX_TEMPO)
            plan.tempo_adjusted.append((shot.shot_id, round(tempo, 2)))

        ok, err = _ffmpeg_fit(seg, raw, target=target, tempo=tempo)
        if not ok:
            plan.notes.append(f"{shot.shot_id} 音频规整失败（{err}），已补静音")
            plan.silent += 1
            ok2, _ = _ffmpeg_silence(seg, target)
            if not ok2:
                return None, plan
        elif tempo > 1.0 and got / tempo > target + 0.05:
            plan.truncated.append(shot.shot_id)
            plan.notes.append(
                f"{shot.shot_id} 台词超出镜长，已加速 {tempo:.1f}× 仍不够，尾部被截断"
                f"（应回 S1 缩短台词或拆镜）"
            )
        pieces.append(seg)

    if not pieces:
        return None, plan

    listing = voice_dir / "audio_list.txt"
    listing.write_text("\n".join(f"file '{p.as_posix()}'" for p in pieces) + "\n",
                       encoding="utf-8")
    track = out_root / "audio.m4a"
    ok, err = _run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                           "-c", "copy", str(track)])
    if not ok:
        plan.notes.append(f"音轨拼接失败：{err}")
        return None, plan
    return track, plan


def _voice_of(shot: Shot, gender_by_char: dict[str, str]) -> str:
    """按镜头里第一个角色的性别选音色。

    一镜多人时只取第一个——这是已知局限：真要对上每个人的声线，得把台词拆成
    逐句带 speaker 的结构，那是 4.6 `Shot.dialogue` 单字符串之外的改动，超出 A.5
    的授权范围。当前按「一镜一句台词」的既有约束，第一顺位角色就是说话人。
    """
    from providers.tts import voice_for

    if not shot.character_ids:
        return voice_for(None)
    return voice_for(gender_by_char.get(shot.character_ids[0]))


def _ffmpeg_silence(out: Path, duration: float) -> tuple[bool, str]:
    args = ["-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", f"{duration:.3f}", "-c:a", "aac", "-b:a", "64k", str(out)]
    return _run_ffmpeg(args)


def _ffmpeg_fit(out: Path, src: Path, *, target: float, tempo: float) -> tuple[bool, str]:
    """把一段配音规整到目标时长：先加速（如需），再补静音，最后按目标时长切齐。"""
    chain = []
    if tempo > 1.0:
        chain.append(f"atempo={tempo:.4f}")
    chain.append("apad")
    args = ["-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
            "-af", ",".join(chain), "-t", f"{target:.3f}",
            "-ar", "44100", "-ac", "1", "-c:a", "aac", "-b:a", "64k", str(out)]
    return _run_ffmpeg(args)


# ---------------------------------------------------------------- 主流程


def collect_segments(ctx, shots: Sequence[Shot]) -> tuple[list[Shot], list[Path]]:
    """按镜号收集可合成片段。顺序严格按 mirror index（14.4-G 第 2 条：防串镜）。"""
    included: list[Shot] = []
    paths: list[Path] = []
    for shot in sorted(shots, key=lambda s: s.index):
        p = ctx.video_of(shot.shot_id)
        if p.exists() and p.stat().st_size > 0:
            included.append(shot)
            paths.append(p)
    return included, paths


def write_lost_package(ctx, segments: list[Path], *, reason: str) -> Path:
    """「无损片段包」：把可用片段原样拷出 + 生成 concat 清单与手工合成说明。

    6 章 S5 的失败处理明文要求这个产物。名字叫「无损」是因为它不重编码、
    不降质——只是把拼接这一步交给人或下一次运行。
    """
    pkg = ctx.root / SEG_DIR
    pkg.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for p in segments:
        dst = pkg / p.name
        if not dst.exists() or dst.stat().st_size != p.stat().st_size:
            shutil.copy2(p, dst)
        copied.append(dst)

    listing = pkg / "concat_list.txt"
    listing.write_text(
        "\n".join(f"file '{c.as_posix()}'" for c in copied) + "\n", encoding="utf-8"
    )
    (pkg / "ASSEMBLE.md").write_text(
        "# 无损片段包\n\n"
        f"合成未完成，原因：{reason}\n\n"
        f"共 {len(copied)} 段，按镜号顺序排列，未重编码、未降质。\n\n"
        "## 手工合成\n\n"
        "装好 ffmpeg（含 libx264 与 fontconfig）后执行：\n\n"
        "```bash\n"
        f'ffmpeg -f concat -safe 0 -i "{listing.name}" -c copy ../final.mp4\n'
        "```\n\n"
        "需要烧字幕时（必须指定字体族名，否则中文是方块）：\n\n"
        "```bash\n"
        "ffmpeg -f concat -safe 0 -i concat_list.txt "
        f"-vf \"subtitles=../subtitles.srt:force_style='FontName={font_family()}'\" "
        "-c:v libx264 -c:a copy ../final.mp4\n"
        "```\n",
        encoding="utf-8",
    )
    return pkg


def run(
    ctx,
    script: Script,
    *,
    ffmpeg_ok: bool | None = None,
    burn_subtitles: bool = True,
    tts=None,
) -> ComposeResult:
    """S5 主流程。返回结果里 `status=degraded` 时必须把 notes 原样写进报告。

    步骤顺序（第 6 章 S5）：TTS 逐句配音 → 字幕生成与对齐 → 片段拼接 → BGM 混音
    → 导出 9:16。当前 **BGM 未实现**（没有素材，也没有定方案），音轨只有人声。
    """
    notes: list[str] = []
    shots, segments = collect_segments(ctx, script.shots)

    if not segments:
        return ComposeResult(status="degraded", notes=["没有任何可用片段，未生成成片"])

    excluded = [s.shot_id for s in script.shots if s not in shots]
    if excluded:
        notes.append(f"未参与合成：{excluded}（failed 或视频缺失）")

    srt_text, cue_count = build_srt(shots)
    srt_path = ctx.root / "subtitles.srt"
    srt_path.write_text(srt_text, encoding="utf-8")
    total = sum(float(s.duration) for s in shots)

    has_ffmpeg = find_ffmpeg() is not None if ffmpeg_ok is None else ffmpeg_ok
    if not has_ffmpeg:
        reason = "ffmpeg 不可用（未配置 FFMPEG_DIR 且不在 PATH 中），无法拼接与烧字幕"
        pkg = write_lost_package(ctx, segments, reason=reason)
        return ComposeResult(
            status="degraded", final=None, srt=srt_path, segments=segments,
            package_dir=pkg, cue_count=cue_count, total_duration=total,
            notes=[*notes, reason, f"已输出无损片段包：{pkg}"],
        )

    # ---------- 音轨：逐镜配音，每段严格等于镜长 ----------
    if tts is None:
        from providers.tts import default_tts

        tts = default_tts()
    gender_by_char = {c.char_id: (c.gender or "") for c in script.characters}
    audio_track, plan = build_audio_track(
        shots, tts, out_root=ctx.root, gender_by_char=gender_by_char
    )
    notes.extend(plan.notes)
    if audio_track is None:
        notes.append("音轨生成失败，成片将无配音（字幕仍会烧录）")

    listing = ctx.root / "concat_list.txt"
    listing.write_text("\n".join(f"file '{p.as_posix()}'" for p in segments) + "\n",
                       encoding="utf-8")
    final = ctx.final_video

    # 第一步：无损拼接。失败说明片段参数不一致（分辨率/编码不同），
    # 那属于上游问题，不该靠重编码掩盖——重编码会让画面再掉一次质量。
    ok, err = _run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                           "-c", "copy", str(final)])
    if not ok:
        reason = f"片段拼接失败：{err}"
        pkg = write_lost_package(ctx, segments, reason=reason)
        return ComposeResult(
            status="degraded", final=None, srt=srt_path, segments=segments,
            package_dir=pkg, cue_count=cue_count, total_duration=total,
            notes=[*notes, reason, f"已输出无损片段包：{pkg}"],
            audio=audio_track, spoken=plan.spoken, silent=plan.silent,
            tempo_adjusted=plan.tempo_adjusted, truncated=plan.truncated,
        )

    # 第二步：烧字幕 + 混入音轨，一次完成（避免重复重编码）。
    need_reencode = bool(burn_subtitles and cue_count)
    burned = final.with_name("final_muxed.mp4")
    args = ["-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio_track is not None:
        args += ["-i", str(audio_track)]

    if need_reencode:
        font = find_font()
        if not font:
            notes.append("未找到中文字体文件，字幕可能显示为方块（7.6 已知坑）")
        # 族名跟着实际字体走：Windows 上是 Microsoft YaHei，容器里是 Noto Sans CJK SC
        family = font_family()
        style = f"FontName={family},FontSize=16,Outline=1,Shadow=0"
        args += ["-vf", f"subtitles='{_filter_escape(srt_path)}':force_style='{style}'",
                 "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        args += ["-c:v", "copy"]

    if audio_track is not None:
        args += ["-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "128k"]
    args += ["-shortest", str(burned)]

    ok2, err2 = _run_ffmpeg(args)
    if ok2:
        burned.replace(final)
    elif need_reencode:
        # 烧字幕 + 混音失败没有到「交付不了」的程度：无字幕成片 + 独立 srt 仍可用。
        # 但音轨也一起丢了——这一点必须写明，不能只说「字幕失败」。
        notes.append(
            f"烧字幕与混音一并失败，成片无字幕也无配音"
            f"（srt 与 audio.m4a 已单独输出）：{err2}"
        )
        burned.unlink(missing_ok=True)
    else:
        notes.append(f"混入音轨失败，成片无配音（audio.m4a 已单独输出）：{err2}")
        burned.unlink(missing_ok=True)

    if not final.exists():
        reason = "成片未生成"
        pkg = write_lost_package(ctx, segments, reason=reason)
        return ComposeResult(status="degraded", final=None, srt=srt_path, segments=segments,
                             package_dir=pkg, cue_count=cue_count, total_duration=total,
                             notes=[*notes, reason], audio=audio_track)

    return ComposeResult(
        status="degraded" if notes else "ok",
        final=final, srt=srt_path, audio=audio_track, segments=segments,
        cue_count=cue_count, total_duration=total, notes=notes,
        spoken=plan.spoken, silent=plan.silent,
        tempo_adjusted=plan.tempo_adjusted, truncated=plan.truncated,
    )
