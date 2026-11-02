"""供应商抽象层。

所有可能花钱的调用都必须经过这里，并且默认拒绝执行。
要花钱，必须显式传 allow_paid=True —— 这不是洁癖，是预算护栏。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class PaidCallBlocked(PermissionError):
    """未显式授权就尝试产生费用的调用。"""


class ProviderError(RuntimeError):
    """上游服务返回了不可用的结果。**默认视为可重试**（超时、5xx、返回空）。"""


class NonRetryableProviderError(ProviderError):
    """**再重试一万次结果也一样**的错误。

    2026-10-29 加。起因：六镜跑到 `shot_05` 时，S1 给它写了 `duration=10`，而
    `wan2.2-i2v-flash` 只支持 5 秒。provider 立刻拒绝并给出清楚的原因，但 S4
    把所有 `ProviderError` 一律当网络抖动重试 2 次——**白等约 6 分钟**，最后还是 failed。

    这类错误的共同点：**请求本身不合法，与网络、负载、时间都无关**。
      · 参数超出模型能力（时长档位、分辨率、图片数量）
      · 鉴权失败、余额不足、配额耗尽
      · 输入文件损坏/格式不支持
      · 内容审核拒绝

    区分它们的意义不是"少等 6 分钟"，而是**让整批立刻停下来**：额度耗尽时继续
    跑下一镜，只会把"没钱了"伪装成"六镜全部失败"。**能被重试解决的叫抖动，
    不能被重试解决的叫状态——两者混在一起，人就看不出到底发生了什么。**
    """


class ImageProvider(ABC):
    #: 是否为占位实现（mock）。**预算判定必须用这个属性，不要靠类名/名称前缀猜。**
    #: 猜错的代价是双向的：把真实 provider 当 mock → 漏记费用、预算封顶失效；
    #: 把 mock 当真实 → 零成本的验证链路被预算拦住，M1 根本跑不完。
    is_mock: bool = False

    name: str = "image-base"
    unit_price: float = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        out: Path,
        references: list[Path] | None = None,
        size: str = "1024x1792",
        allow_paid: bool = False,
    ) -> Path:
        """生成一张图并返回落盘路径。"""


class VideoProvider(ABC):
    #: 是否为占位实现（mock）。**预算判定必须用这个属性，不要靠类名/名称前缀猜。**
    #: 猜错的代价是双向的：把真实 provider 当 mock → 漏记费用、预算封顶失效；
    #: 把 mock 当真实 → 零成本的验证链路被预算拦住，M1 根本跑不完。
    is_mock: bool = False

    name: str = "video-base"
    price_per_second: float = 0.0

    @abstractmethod
    def generate(
        self,
        keyframe: Path,
        prompt: str,
        *,
        duration: float,
        out: Path,
        allow_paid: bool = False,
    ) -> Path:
        """首帧图 + 动作描述 → 视频片段。只做图生视频，不做文生视频。"""


class TextProvider(ABC):
    #: 是否为占位实现（mock）。**预算判定必须用这个属性，不要靠类名/名称前缀猜。**
    #: 猜错的代价是双向的：把真实 provider 当 mock → 漏记费用、预算封顶失效；
    #: 把 mock 当真实 → 零成本的验证链路被预算拦住，M1 根本跑不完。
    is_mock: bool = False

    name: str = "text-base"

    @abstractmethod
    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """返回模型原始文本。"""


class VisionJudge(ABC):
    #: 是否为占位实现（mock）。**预算判定必须用这个属性，不要靠类名/名称前缀猜。**
    #: 猜错的代价是双向的：把真实 provider 当 mock → 漏记费用、预算封顶失效；
    #: 把 mock 当真实 → 零成本的验证链路被预算拦住，M1 根本跑不完。
    is_mock: bool = False

    name: str = "vlm-base"

    @abstractmethod
    def compare(
        self,
        image: Path,
        reference: Path,
        elements: list[str],
        *,
        allow_paid: bool = False,
    ) -> dict:
        """比对两张图在指定要素上是否一致，返回结构化结果。

        `allow_paid` 与另外两个付费接口（`ImageProvider.generate` /
        `VideoProvider.generate`）保持一致。

        **这一条是 2026-10-28 真实跑 6 镜时补上的**：A.3 原契约里 `compare` 没有
        `allow_paid`，但 `VLMJudge` 实现时自己加了（因为是付费调用），
        结果 pipeline 调用时没传，S2 的自检第一次真跑就被 PaidCallBlocked 拦住。
        契约缺参数 → 实现自己补 → 调用方不知道，这是典型的接口漂移。
        """


class TTSProvider(ABC):
    """语音合成。单独一个接口，不塞进 TextProvider。

    理由很具体：`TextProvider.complete(system, user) -> str` 的语义是「给文本、拿文本」，
    TTS 是「给文本、拿音频文件」，两者的失败模式、计费口径、产物形态全不同。
    塞进同一个接口会让 mock 和预算判断都变得别扭。
    """

    #: 是否为占位实现（mock）。预算判定用它，不靠名字前缀猜。
    is_mock: bool = False

    name: str = "tts-base"
    #: 每次调用约价（元/句），预算护栏用。免费引擎填 0。
    price_per_call: float = 0.0

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        out: Path,
        allow_paid: bool = False,
    ) -> Path:
        """把一句台词合成为音频文件并返回落盘路径。

        只合成一句：一镜一句台词是 4.7 的刻意约束，也是字幕时间轴的刻度。
        多句合成一个文件会让「哪句对应哪一镜」无法回溯。
        """
