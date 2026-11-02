"""OpenAI 兼容协议的文本模型适配（DeepSeek / Qwen 通用）。"""

from __future__ import annotations

from config import ModelConfig
from providers.base import ProviderError, TextProvider


class ChatProvider(TextProvider):
    def __init__(
        self,
        config: ModelConfig,
        name: str = "chat",
        *,
        timeout: float = 300.0,
        extra_body: dict | None = None,
    ) -> None:
        if not config.usable:
            raise ProviderError(f"{name} 缺少 api_key / base_url / model 配置")
        from openai import OpenAI

        self.name = name
        self._config = config
        self._extra = extra_body or {}
        # 超时 300s。**原值 120s 是错的**，2026-10-28 真实跑 6 镜时正是死在这里：
        # 豆包 2.1 pro 写 6 镜剧本要输出 2000+ token 的 JSON，加上模型侧排队，
        # 单次就可能超过 120s。S1 重试 3 次 × 120s = 6 分钟全部白等，最后报
        # "Request timed out"——而模型其实只是还在写。
        # 教训：文本 provider 的超时不能按「短对话」估，S1 是这个项目里输出最长的一次调用。
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        kwargs: dict = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.7,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # extra_body 由 SDK 合并进请求体顶层。
        # 注意：**不能自己往 HTTP body 里塞一个叫 `extra_body` 的字段**——
        # 那是 SDK 的参数名，直接发 HTTP 时它会被服务端当作未知字段忽略，
        # 开关看起来传了、实际没生效（2026-10-28 踩过一次）。
        if self._extra:
            kwargs["extra_body"] = self._extra

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # 网络、鉴权、限流都在这
            raise ProviderError(f"{self.name} 调用失败：{exc}") from exc

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ProviderError(f"{self.name} 返回空内容")
        return content
