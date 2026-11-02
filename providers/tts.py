"""语音合成 provider。

**为什么单独一个文件而不是塞进 llm.py**：`TextProvider` 的契约是「文本进、文本出」，
TTS 是「文本进、音频文件出」。失败模式（网络/音色/时长）、计费口径（按句/按字）、
产物形态全不同，混在一起会让 mock 与预算判断都变形。

当前状态（2026-10-28 实测）：
  - `EdgeTTS` 在本机**不可用**：连续 5 次 `NoAudioReceived`（0/5），已排除网络、
    系统时间、客户端版本号三个常见原因（详见 `tests/fixtures/m0_eval/` 之外的诊断记录）。
    服务端能建连并回 `turn.start`，但随即 `CLOSED` 不给音频。
  - 因此默认走 `MockTTS`：用 ffmpeg 生成**与估算语速等长的静音**，目的是先把
    「音轨编排 → 与视频合并」这条通路打通并验证。它不产生人声，但时长是合理的，
    足以暴露音画错位、超长台词、空镜补静音这几类真问题。

换引擎时只改这一个文件：`pipeline/s5_compose.py` 只依赖 `TTSProvider` 抽象。
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

from config import find_bin
from providers.base import ProviderError, TTSProvider

#: 中文播报语速的经验值（字/秒）。只用于占位音与超长判定，不用于字幕时间轴——
#: 字幕按镜长累加（14.4-G 的断言），语音快了慢了都靠字幕兜住。
CHARS_PER_SECOND = 4.5

#: 音色映射。角色 `gender` 字段 → edge-tts 音色名。
#: 只映射到已核实存在的音色（`edge-tts --list-voices` 实测，zh-CN 共 8 个）。
VOICE_BY_GENDER: dict[str, str] = {
    "女": "zh-CN-XiaoxiaoNeural",
    "男": "zh-CN-YunxiNeural",
}
DEFAULT_VOICE = "zh-CN-XiaoxiaoNeural"


def voice_for(gender: str | None) -> str:
    """按性别选音色。未知性别退到默认女声——不是随便选的：
    M0 数据集的 15 个角色里女性占多数，退到女声比退到男声更少出错。"""
    return VOICE_BY_GENDER.get((gender or "").strip(), DEFAULT_VOICE)


def estimate_seconds(text: str) -> float:
    """按字数估算播报时长。标点算半个字（会停顿但不出声）。

    这个估算有两个用途：① 占位音的长度；② 判定台词是否超出镜长。
    真实引擎接上后，① 会被真实时长取代，② 仍然需要——因为真实时长只有合成完才知道，
    而「超长」这件事必须在合成**之后**才能判断，代码里的用法是「合成后回填」。
    """
    chars = sum(0.5 if c in "，。！？、；：,.!?;:" else 1.0 for c in text if not c.isspace())
    return max(0.8, chars / CHARS_PER_SECOND)


class MockTTS(TTSProvider):
    """占位语音：生成与估算语速等长的静音音频。

    **不是空文件**。空文件会让 amix/concat 报错或产生 0 秒音轨，从而掩盖真正的
    编排问题。等长静音能让「6 段音频拼起来正好等于视频时长」这条断言真正生效。
    """

    name = "mock-tts"
    is_mock = True
    price_per_call = 0.0

    def synthesize(
        self, text: str, *, voice: str, out: Path, allow_paid: bool = False
    ) -> Path:
        ffmpeg = find_bin("ffmpeg")
        if not ffmpeg:
            raise ProviderError("MockTTS 需要 ffmpeg 生成占位音轨（当前未找到 ffmpeg）")
        dur = estimate_seconds(text)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
            "-t", f"{dur:.3f}", "-c:a", "aac", "-b:a", "64k", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise ProviderError(f"占位音轨生成失败：{(proc.stderr or '').strip()[-200:]}")
        return out


class EdgeTTS(TTSProvider):
    """edge-tts（免费）。

    **必须打一个版本号补丁才能用**（2026-10-28 实测）：
    库内硬编码的 `Sec-MS-GEC-Version` 是 `2-143.0.3650.75`，用它会 0/5 全失败；
    改成 `2-155.0.4283.45` 后 4/4 成功。失败形态极具误导性——`NoAudioReceived`
    看起来像参数错，实际是服务端把不认的版本号静默拒绝，连上了却不给音频。

    端点不需要换：外派单提到的新端点 `api.msedgeservices.com` 实测并非必需。
    """

    name = "edge-tts"
    is_mock = False
    price_per_call = 0.0

    def __init__(self) -> None:
        self._patched = False

    def _patch_version(self) -> None:
        """把库常量改成配置值。幂等，只改内存，不碰 site-packages。

        改的是 `edge_tts.communicate` 模块里的名字而不是 `constants`：后者在
        导入时已被 `from .constants import SEC_MS_GEC_VERSION` 复制成局部引用，
        只改 constants 不会生效——这个坑第一次就踩了。
        """
        from config import settings

        import edge_tts.communicate as ec

        want = settings.tts_sec_version
        if ec.SEC_MS_GEC_VERSION != want:
            ec.SEC_MS_GEC_VERSION = want
        self._patched = True

    def synthesize(
        self, text: str, *, voice: str, out: Path, allow_paid: bool = False
    ) -> Path:
        try:
            import asyncio

            import edge_tts
        except ImportError as exc:  # pragma: no cover - 环境缺包
            raise ProviderError(f"edge-tts 未安装：{exc}") from exc

        self._patch_version()
        out.parent.mkdir(parents=True, exist_ok=True)

        async def _run() -> None:
            await edge_tts.Communicate(text, voice).save(str(out))

        try:
            asyncio.run(_run())
        except RuntimeError as exc:
            # 已经在事件循环里（FastAPI 等）时 asyncio.run 会拒绝。如实报，
            # 不偷偷起线程——那会让「为什么没声音」变得无法定位。
            raise ProviderError(f"edge-tts 无法在已有事件循环中同步调用：{exc}") from exc
        except Exception as exc:
            from config import settings

            raise ProviderError(
                f"edge-tts 合成失败：{type(exc).__name__}: {exc}"
                f"（若为 NoAudioReceived，多半是 TTS_SEC_VERSION={settings.tts_sec_version} 过期）"
            ) from exc

        if not out.exists() or out.stat().st_size == 0:
            raise ProviderError(f"edge-tts 返回空音频：{out}")
        return out


def default_tts() -> TTSProvider:
    """当前环境下的默认 TTS：edge-tts（已验证可用，0 成本）。

    不用 MockTTS 兜底的原因：MockTTS 是等长静音，与「TTS 失败后补静音」在成片里
    完全一样，但报告里一个写「2 句人声」、一个写「配音失败」——自动降级会让
    「这片子到底有没有人声」变成一个需要翻日志才能回答的问题。
    """
    return EdgeTTS()
