"""视频 provider：DashScope 异步任务制的图生视频（i2v）。

**不带厂商名**（A.8 第 2 条）：base_url、key、model 来自 `settings.video_api`，
换供应商只改 `.env`。当前指向阿里百炼，模型 wan2.2-i2v-flash。

实现 `VideoProvider` 契约（A.3），与 `MockVideo` 可互换。

## 调用形态与图像完全不同

不是 OpenAI 兼容，是 DashScope 原生的**异步任务制**：

    创建  POST {base}/api/v1/services/aigc/video-generation/video-synthesis
          Header: X-DashScope-Async: enable
          → 返回 task_id（HTTP 200，1.3s）
    轮询  GET  {base}/api/v1/tasks/{task_id}
          → task_status: PENDING / RUNNING / SUCCEEDED / FAILED
    下载  output.video_url

实测耗时：创建 1.3s，PENDING→RUNNING 11.6s，SUCCEEDED 44~105s。

## 两个刻意的决定

**1. 首帧在这里裁成 9:16，不留给上游。**
i2v 的输出比例**跟随输入首帧**——这是接口语义，不是我们的偏好。所以「保证首帧是
9:16」属于 provider 履约的一部分：契约里没有 ratio 参数，输出比例的唯一控制点
就是这张输入图。上游传什么比例都行，这里统一归一化到 720×1280。

**2. `audio=False` 且 `prompt_extend=False`，理由都不是省钱。**
- `audio=False`：本项目的声音由 edge-tts 那条链路单独产出（S5 已实现），
  模型再生成一遍人声会形成双音轨，而且无法控制音色/断句/与字幕的对齐。
  视频只出画面，声音单独负责，两件事各自可控。
- `prompt_extend=False`：开着时模型会用 LLM 改写提示词。但 8.2 的视频提示词规格
  是「**只写运动，禁止重复描述外貌**」，一改写，LLM 会自己补外貌描述，
  等于绕过了规格，还拉长生成时间。
"""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import httpx
from PIL import Image

import config
from providers.base import (
    NonRetryableProviderError,
    PaidCallBlocked,
    ProviderError,
    VideoProvider,
)

#: 创建任务的接口路径（DashScope 原生，非 OpenAI 兼容）
CREATE_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/"

#: 首帧统一到 9:16 竖屏。i2v 输出跟随此比例。
FRAME_W, FRAME_H = 720, 1280

#: 轮询参数。实测 44~105s 完成，给 900s 上限足够容纳排队。
POLL_INTERVAL_SEC = 10
POLL_TIMEOUT_SEC = 900

TIMEOUT = httpx.Timeout(180.0, connect=90.0)


class DashScopeVideo(VideoProvider):
    """图生视频 provider（DashScope 异步任务制）。"""

    name = "dashscope-video"
    is_mock = False

    #: 该模型只支持固定 5 秒。10 秒档需要换 wan2.6/2.7（支持 2~15s 任意整数），
    #: 换模型时只改 .env 的 VIDEO_MODEL，并同步这个集合。
    SUPPORTED_DURATIONS: tuple[int, ...] = (5,)

    def __init__(self) -> None:
        cfg = config.settings.video_api
        if not (cfg.api_key and cfg.base_url and cfg.model):
            raise ProviderError(
                "video_api 未配置完整：需要 VIDEO_API_KEY / VIDEO_BASE_URL / VIDEO_MODEL"
            )
        self._cfg = cfg
        self.price_per_second = config.settings.price_video_per_sec
        self.resolution = config.settings.video_resolution_level

    # ------------------------------------------------------------------
    def _normalize_frame(self, keyframe: Path, tmp: Path) -> Path:
        """把首帧裁成严格 9:16 并缩到 720×1280。

        输出比例跟随输入图，所以这一步决定了成片画幅，不能省。
        """
        im = Image.open(keyframe).convert("RGB")
        w, h = im.size
        target = FRAME_W / FRAME_H
        if w / h > target:  # 太宽 → 裁两侧
            nw = int(h * target)
            left = (w - nw) // 2
            im = im.crop((left, 0, left + nw, h))
        else:               # 太高 → 裁上下
            nh = int(w / target)
            top = (h - nh) // 2
            im = im.crop((0, top, w, top + nh))
        im = im.resize((FRAME_W, FRAME_H), Image.LANCZOS)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        im.save(tmp, "JPEG", quality=90)
        return tmp

    # ------------------------------------------------------------------
    def generate(
        self,
        keyframe: Path,
        prompt: str,
        *,
        duration: float,
        out: Path,
        allow_paid: bool = False,
    ) -> Path:
        if not allow_paid:
            raise PaidCallBlocked(
                f"{self.name}：视频生成是付费调用（¥{self.price_per_second}/秒），"
                "必须显式传 allow_paid=True"
            )
        keyframe = Path(keyframe)
        if not keyframe.exists():
            raise ProviderError(f"首帧不存在：{keyframe}")
        if not prompt or not prompt.strip():
            raise ProviderError("prompt 不能为空")

        secs = int(round(duration))
        if secs not in self.SUPPORTED_DURATIONS:
            # 不可重试：参数不在模型能力范围内，重发一百次也是同样结果。
            raise NonRetryableProviderError(
                f"{self.name}：模型 {self._cfg.model} 只支持 "
                f"{self.SUPPORTED_DURATIONS} 秒，收到 {duration}。"
                "要出更长的镜头，请把 .env 的 VIDEO_MODEL 换成 wan2.6-i2v / wan2.7-i2v"
                "（支持 2~15s），并把 VIDEO_SUPPORTED_SECONDS 一起改"
            )

        frame = self._normalize_frame(keyframe, out.parent / f".{out.stem}_firstframe.jpg")
        b64 = base64.b64encode(frame.read_bytes()).decode()

        body = {
            "model": self._cfg.model,
            "input": {
                "prompt": prompt,
                # Base64 直传。实测被接受，因此 S3→S4 之间不需要图床或对象存储。
                "img_url": f"data:image/jpeg;base64,{b64}",
            },
            "parameters": {
                "resolution": self.resolution,
                "duration": secs,
                "prompt_extend": False,   # 见模块 docstring
                "watermark": False,
            },
        }

        task_id = self._create_task(body)
        video_url = self._wait(task_id)
        return self._download(video_url, out)

    # ------------------------------------------------------------------
    def _create_task(self, body: dict) -> str:
        url = self._cfg.base_url.rstrip("/") + CREATE_PATH
        headers = {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=TIMEOUT)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"{self.name} 创建任务连接失败：{exc}") from exc
        if resp.status_code != 200:
            raise ProviderError(
                f"{self.name} 创建任务 HTTP {resp.status_code}: {resp.text[:400]}"
            )
        task_id = (resp.json().get("output") or {}).get("task_id")
        if not task_id:
            raise ProviderError(f"{self.name} 未返回 task_id：{resp.text[:300]}")
        return task_id

    def _wait(self, task_id: str) -> str:
        url = self._cfg.base_url.rstrip("/") + TASK_PATH + task_id
        headers = {"Authorization": f"Bearer {self._cfg.api_key}"}
        deadline = time.time() + POLL_TIMEOUT_SEC
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL_SEC)
            try:
                data = httpx.get(url, headers=headers, timeout=90.0).json()
            except Exception:  # noqa: BLE001  单次轮询失败不该让任务报废，下一轮再查
                continue
            output = data.get("output") or {}
            status = output.get("task_status") or "?"
            if status == "SUCCEEDED":
                video_url = output.get("video_url")
                if not video_url:
                    results = output.get("results") or []
                    if results:
                        video_url = results[0].get("url")
                if not video_url:
                    raise ProviderError(f"{self.name} 任务成功但没有 video_url：{str(data)[:300]}")
                return video_url
            if status in ("FAILED", "CANCELED", "UNKNOWN"):
                raise ProviderError(f"{self.name} 任务终态 {status}：{str(data)[:400]}")
        raise ProviderError(
            f"{self.name} 轮询超时 {POLL_TIMEOUT_SEC}s（task_id={task_id}，可手工查询）"
        )

    def _download(self, video_url: str, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(video_url, timeout=300.0, follow_redirects=True)
        if resp.status_code != 200:
            raise ProviderError(f"{self.name} 下载视频 HTTP {resp.status_code}")
        out.write_bytes(resp.content)
        return out
