"""视觉判定 provider：一致性比对（OpenAI 兼容协议）。

**厂商无关是硬要求**（计划书 A.8 第 2 条）：base_url / model / api_key 全部来自
`config.settings.judge_vlm`，本文件不出现任何厂商名。换厂商只改 `.env`，不动代码。

**职责边界**（A.3 语义 3）：本 provider 只产出「原始比对结果 dict」。
分数归一（85 / 60~85 / <60 的业务含义）与转 `QAResult` 是 `pipeline/judge.py` 的事，
这里只做解析与结构对齐，不做任何业务判断。

**付费护栏**：`compare()` 的 `allow_paid` 默认 `False`，未显式授权抛 `PaidCallBlocked`
（9.4 节护栏的代码落点）。`providers/base.py` 的 `VisionJudge` 抽象签名里没有这个参数，
这里用「关键字参数 + 默认关闭」的方式补上——不破坏抽象契约，同时让护栏有落点。

**不依赖 Pillow**：本地图直接按原始字节 Base64 内联发出（OpenAI 兼容协议的
`image_url.url` 支持 `data:image/png;base64,...`）。判定要看脸，缩放会丢细节；
体积超限直接报错，不静默降级。
"""

from __future__ import annotations

import base64
import json
import re
import struct
import time
from pathlib import Path
from typing import Any

from config import ModelConfig
from providers.base import PaidCallBlocked, ProviderError, VisionJudge

# 五项外观要素的中文说明（8.1 角色卡 appearance 必填五项）。
# 判定 prompt 直接引用这张表，避免每次手写提示词时口径漂移。
ELEMENT_HINTS: dict[str, str] = {
    "发型": "长度、样式、颜色",
    "脸型": "面部轮廓形状",
    "眉眼": "眉形、眼型、眼距",
    "肤色": "面部与颈部肤色",
    "体型": "肩宽与身形比例",
}

# 判定 prompt 的版本号。改动提示词必须改这个号，否则 M0 的数字不可比。
JUDGE_VERSION = "m0-judge-v1"

# 评分分档里**故意不出现** 85 / 60 这两条业务线：
# 15.1 第四节要验的正是「这两条线画得对不对」，若在 prompt 里把阈值告诉模型，
# 测到的就是「模型有没有照抄我给的线」，而不是「线画得对不对」。分档只描述
# 视觉一致程度，与任何通过/不通过的阈值无关；线的位置留给 M0-S3 用数据定。
_SYSTEM_PROMPT = """你是一个影视短剧的角色一致性审核员。

任务：给出同一角色的「参考定妆照」和「待判画面」，判断二者在五项外观要素上是否一致，并给出可执行的修改建议。

五项要素及含义：
{element_hints}

评分标准（只反映视觉一致程度，与任何通过/不通过的阈值无关，不要考虑用途）：
- 90~100：五项要素全部一致，差异仅来自拍摄角度、光线、表情或景别
- 70~89：大部分要素一致，一到两项有可见差异
- 40~69：多项要素明显不同
- 10~39：大部分要素不同
- 0~9：明显不是同一个人

判定要求：
1. 逐项独立判断，不要因为其他项一致就默认本项一致。
2. matched 为 true 表示该项在两张图上视觉上一致（允许角度、光线带来的轻微差异）。
3. expected 写参考图里该项的客观描述，只写看得见的特征（如「黑色齐肩直发」），不要写「帅气」「有气质」这类主观词。
4. actual 写待判画面里该项的客观描述。
5. diff 只写最重要的一处偏差，用「什么不一致：参考图是…，待判画面是…」的句式，便于直接改写成生成提示词；完全一致时写空字符串。
6. 只输出 JSON，不要输出任何解释性文字。

输出格式：
{{"score": 0, "checks": [{{"element": "发型", "matched": true, "expected": "", "actual": ""}}], "diff": ""}}"""

_USER_TEXT_NORMAL = "图1是参考定妆照，图2是待判画面。请逐项比对上述五项要素，按系统提示的 JSON 格式输出。"
_USER_TEXT_SWAPPED = "图1是待判画面，图2是参考定妆照。请逐项比对上述五项要素，按系统提示的 JSON 格式输出。"

# 单图原始字节上限。多数 OpenAI 兼容网关的请求体上限在 10MB 量级，留一点余量；
# 超限时直接报错，不静默压缩（压缩会丢判定细节）。
MAX_IMAGE_BYTES = 8 * 1024 * 1024

# 单次请求超时（秒）。
#
# **这个值必须大于「最坏情况下的思考耗时」，否则会引发静默空转。**
# 2026-10-28 实测：D13 这一对（两个高度相似的少女角色）开了深度思考后耗时 189 秒、
# reasoning_tokens=11,195。原先设 180 秒，导致每次都在 180 秒被掐断 → 判定为可重试
# → 换下一组参数重发 → 再次超时 → 4 组参数走完约 12~14 分钟，最后报「调用失败」。
# 表面症状是「某一对特别慢」，实际是一组固定参数把每次都快超时的请求反复重发。
#
# 取 420 秒：给最坏情况留 2 倍余量。思考 token 越多耗时越长，而难题恰恰最需要思考，
# 把超时设小等于在最有价值的样本上必然失败。
TIMEOUT_SEC = 420.0


def _is_retryable(exc: Exception) -> bool:
    """判断异常是否值得退避重试（限流 / 瞬时服务端错误）。

    参数类错误（400 鉴权、404 模型名不存在）重试没有意义，直接换下一种参数组合。
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if status is None:
        # 没有状态码时按网络瞬时错误处理（超时、连接重置）
        return True
    return status == 429 or 500 <= int(status) < 600


def _is_timeout(exc: Exception) -> bool:
    """判断异常是否为超时。

    超时和「参数不对」是两类完全不同的失败：前者说明模型还在思考（思考 token 多的
    难题最慢），换参数重发没有意义；后者才该换参数组合。必须分开处理，否则会把
    一次慢请求放大成四倍慢。
    """
    name = type(exc).__name__
    if "Timeout" in name or "timeout" in name:
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def _backoff_seconds(attempt: int) -> float:
    """退避时长：3s / 6s / 12s…，上限 20s。"""
    return min(20.0, 3.0 * (2 ** max(0, attempt - 1)))


def _element_hints_block(elements: list[str]) -> str:
    lines = []
    for el in elements:
        hint = ELEMENT_HINTS.get(el)
        lines.append(f"- {el}：{hint}" if hint else f"- {el}")
    return "\n".join(lines)


def _image_info(data: bytes) -> tuple[str, int, int]:
    """返回 (mime, 宽, 高)。只解析 PNG / JPEG 文件头，不引入 Pillow。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return "image/png", int(w), int(h)
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return "image/jpeg", int(w), int(h)
            seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
            i += 2 + seg_len
    return "image/png", 0, 0


def _encode_image(path: Path) -> tuple[str, dict[str, Any]]:
    """本地图片 → data URL（Base64 内联）。返回 (data_url, 记账信息)。"""
    data = path.read_bytes()
    mime, w, h = _image_info(data)
    if len(data) > MAX_IMAGE_BYTES:
        raise ProviderError(
            f"图片过大：{path.name} {len(data) / 1048576:.1f}MB 超过 "
            f"{MAX_IMAGE_BYTES / 1048576:.0f}MB 上限（不静默压缩，避免丢判定细节）"
        )
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}", {"bytes": len(data), "width": w, "height": h}


def build_judge_messages(
    reference: Path,
    image: Path,
    elements: list[str],
    *,
    swap: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构造判定请求的 messages + 记账信息。**不发请求**，纯函数，便于离线自检。

    `swap=True` 用于消融实验：把参考图与待判图的位置对调，检验模型有没有位置偏好
    （VLM 常见「第一张图是基准」的系统性偏差，会直接污染 FP/FN）。
    """
    ref_url, ref_info = _encode_image(Path(reference))
    img_url, img_info = _encode_image(Path(image))

    first, second = (ref_url, img_url)
    if swap:
        first, second = img_url, ref_url
    user_text = _USER_TEXT_SWAPPED if swap else _USER_TEXT_NORMAL

    system_text = _SYSTEM_PROMPT.format(element_hints=_element_hints_block(elements))
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": first}},
                {"type": "image_url", "image_url": {"url": second}},
            ],
        },
    ]
    meta = {
        "reference": ref_info,
        "image": img_info,
        "payload_bytes": len(ref_url) + len(img_url) + len(system_text) + len(user_text),
    }
    return messages, meta


_DESC_SYSTEM = """你是角色设定师。用户给你一张人物参考图，你要把它转写成**可直接用于文生图的文字角色卡**。

只输出一个 JSON 对象，不要解释、不要 markdown 代码块。

铁律（违反任何一条，这张卡就会把后面的定妆照和每一镜都带偏）：

1. **只写看得见的。** 看不清的字段留空字符串 ""，不要按"通常这样"补。
2. **不许按人类去补。** 如果图里是卡通角色、吉祥物、动物或非人生物，就照它本来的样子写：
   体表固有色、头身比、肢体形态、材质（毛绒/塑料/金属）都写进去。
   绝不能写"白皙肤色""人类男性"这类与图不符的默认值——2026-11-02 就是这么
   把一个黄色卡通吉祥物写成了"23-27 岁人类男性、肤色白皙"，出图得到一张
   写实真人配一片黄脖子。
3. appearance 是一个字符串，按顺序写：头部/发型的形状与颜色 → 脸型与五官 →
   肤色或体表颜色 → 体型与头身比。
4. wardrobe 写死颜色与材质（例：亮黄色连帽卫衣，棉质）。图里没有服装就写 ""。
5. confidence 是每个字段 0~1 的自评：图里清楚 → 0.8 以上；只能靠猜 → 0.4 以下。
   **宁可标低**，人会在下一步核对，标高了没人看。

JSON 结构（键名逐字照抄）：
{
  "subject_type": "human | cartoon | animal | other",
  "gender": "男 | 女 | 不确定",
  "age_range": "如 26-30岁；看不出就写 不确定",
  "appearance": "头部与发型、脸型五官、肤色或体表色、体型头身比",
  "wardrobe": "服装，颜色与材质",
  "style_hint": "这张图的画风：写实摄影 / 3D 卡通渲染 / 2D 动画 / 手绘 / 其他",
  "distinctive": "最不该被改掉的一个特征（一句话）",
  "confidence": {"gender": 0.9, "age_range": 0.5, "appearance": 0.8, "wardrobe": 0.7}
}"""

_DESC_USER = "把这张参考图转写成文字角色卡。"


def build_describe_messages(image: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """构造「图生文反推角色卡」的 messages。**不发请求**，纯函数，便于离线自检。

    与 `build_judge_messages` 分开而不是塞一个函数加参数：两者输入数量不同
    （判定是两张图 + 要素列表，反推是一张图），合并只会让调用方每次都传一半用不上的参数。
    """
    img_url, img_info = _encode_image(Path(image))
    messages = [
        {"role": "system", "content": _DESC_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _DESC_USER},
                {"type": "image_url", "image_url": {"url": img_url}},
            ],
        },
    ]
    meta = {"image": img_info,
            "payload_bytes": len(img_url) + len(_DESC_SYSTEM) + len(_DESC_USER)}
    return messages, meta


def _extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON 对象。

    真实模型经常加围栏（```json ... ```）或前后废话，所以不能直接 json.loads。
    """
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 扫描第一个括号配平的 JSON 对象（正确处理字符串里的花括号与转义）
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escaped = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break
        start = s.find("{", start + 1)
    raise ProviderError(f"模型返回不是合法 JSON：{text[:200]}")


def _normalize(data: dict, elements: list[str], model: str, usage: dict | None) -> dict:
    """把模型原始 JSON 对齐成固定结构。

    这里只做**类型对齐与缺项标记**，不做业务判断：
    - score 转 int 并夹到 0~100（防模型返回 120 这种脏值）
    - checks 按输入的要素列表补齐，模型漏掉的要素 matched 记 None（表示「没判」，
      与「判为不一致」严格区分，否则要素级准确率的分母会被污染）
    """
    try:
        score = int(round(float(data.get("score"))))
    except (TypeError, ValueError):
        score = 0
    score = max(0, min(100, score))

    raw_checks = data.get("checks")
    by_element: dict[str, dict] = {}
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            el = str(item.get("element", "")).strip()
            if not el:
                continue
            matched = item.get("matched")
            if isinstance(matched, str):
                matched = matched.strip().lower() in ("true", "yes", "1", "一致", "相同")
            elif matched is None:
                matched = None
            else:
                matched = bool(matched)
            by_element[el] = {
                "element": el,
                "matched": matched,
                "expected": str(item.get("expected", "")).strip(),
                "actual": str(item.get("actual", "")).strip(),
            }

    checks = []
    missing = []
    for el in elements:
        if el in by_element:
            checks.append(by_element[el])
        else:
            missing.append(el)
            checks.append({"element": el, "matched": None, "expected": "", "actual": ""})

    diff = data.get("diff")
    diff = "" if diff is None else str(diff).strip()

    return {
        "score": score,
        "checks": checks,
        "diff": diff,
        "model": model,
        "judge_version": JUDGE_VERSION,
        "missing_elements": missing,
        "usage": usage or {},
    }


class VLMJudge(VisionJudge):
    """OpenAI 兼容协议的视觉判定器。

    `compare()` 的签名与 A.3 的抽象契约一致（image / reference / elements → dict），
    只多了一个带默认值的 `allow_paid` 关键字参数，用来落付费护栏。
    """

    name = "vlm-judge"

    def __init__(self, config: ModelConfig, name: str = "vlm-judge") -> None:
        if not config.usable:
            raise ProviderError(f"{name} 缺少 api_key / base_url / model 配置")
        from openai import OpenAI  # 延迟导入，与 providers/llm.py 保持一致

        self.name = name
        self._config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=TIMEOUT_SEC)

    # ---------- 内部：调 API ----------

    def _chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        json_mode: bool,
        *,
        enable_thinking: bool = True,
    ) -> tuple[str, dict]:
        """按候选参数组合依次尝试，直到拿到一次成功响应。

        组合顺序：优先 json_object + 低温 → 去掉 json_object → 提高温度 → 最小参数集。
        有些 OpenAI 兼容网关不支持 `response_format` 或要求 temperature > 0，
        逐级降级比一次失败就报错更实用。

        `enable_thinking` 控制模型的「深度思考」：开 = 允许模型先推理再作答，
        关 = 强制直答。**默认开**——实测（2026-10-27）同一对图，关掉思考后模型把
        「四分之三侧视导致一只眼露出面积变小」误判成眉眼不一致，直接制造假阴性；
        开思考后判为一致、diff 为空。判定准确率是 M0 的唯一产出，值得多花的 150 秒。

        思考开关通过 `extra_body` 传（OpenAI SDK 不认识 `thinking` 这个顶层参数，
        但火山方舟的 OpenAI 兼容网关会读它）。网关若不认这个字段会忽略，不影响调用。
        """
        base: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            # 开思考时思考 token 计入输出，实测单次总输出约 7000 token，
            # 给足上限避免被截断导致 JSON 不完整。
            "max_tokens": 1200 if not enable_thinking else 9000,
        }
        if not enable_thinking:
            base["extra_body"] = {"thinking": {"type": "disabled"}}

        variants: list[dict[str, Any]] = []
        if json_mode:
            variants.append({**base, "temperature": temperature, "response_format": {"type": "json_object"}})
        variants.append({**base, "temperature": temperature})
        if temperature != 0.7:
            variants.append({**base, "temperature": 0.7})
        variants.append({"model": self._config.model, "messages": messages, "max_tokens": base["max_tokens"]})

        last_error: Exception | None = None
        for kwargs in variants:
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                last_error = exc
                # 超时**不是参数问题**，是模型还在思考。换参数重发等于把同一件事再做
                # 四遍——2026-10-28 实测 D13 因此空转 14 分钟。这里只原地重试一次，
                # 二次仍超时就放弃这一对并明确报错，让调用方知道是「太慢」而不是「参数错」。
                if _is_timeout(exc):
                    try:
                        resp = self._client.chat.completions.create(**kwargs)
                    except Exception as exc2:
                        raise ProviderError(
                            f"{self.name} 超时：连续两次 {TIMEOUT_SEC:.0f}s 未返回"
                            f"（该样本思考量可能过大，可考虑 --no-thinking 或换模型）：{exc2}"
                        ) from exc2
                else:
                    # 限流 / 瞬时 5xx 值得退避后换下一组参数；参数错误（400/404）直接换下一组。
                    if _is_retryable(exc):
                        time.sleep(_backoff_seconds(1))
                    continue
            content = resp.choices[0].message.content
            if not content or not content.strip():
                last_error = ProviderError(f"{self.name} 返回空内容")
                continue
            usage = {}
            if getattr(resp, "usage", None):
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
                    "total_tokens": getattr(resp.usage, "total_tokens", 0),
                }
            # 记下服务端 request id：对账/排障时唯一能反查单次调用的凭证。
            # 控制台用量统计有延迟或口径差异时，靠这个 id 才能证明"这次调用真的发生了"。
            usage["request_id"] = getattr(resp, "id", "") or ""
            usage["response_model"] = getattr(resp, "model", "") or ""
            return content, usage
        raise ProviderError(f"{self.name} 调用失败：{last_error}") from last_error

    # ---------- 接口实现 ----------

    def compare(
        self,
        image: Path,
        reference: Path,
        elements: list[str],
        *,
        allow_paid: bool = False,
        swap: bool = False,
        enable_thinking: bool = True,
    ) -> dict:
        """比对两张图在指定要素上是否一致，返回结构化结果。

        参数顺序与 A.3 契约一致：`image` 是待判侧，`reference` 是参考侧。
        """
        if not allow_paid:
            raise PaidCallBlocked(
                f"{self.name}：VLM 判定是付费调用，必须显式传 allow_paid=True"
            )

        image = Path(image)
        reference = Path(reference)
        for p in (image, reference):
            if not p.exists():
                raise ProviderError(f"待判定图片不存在：{p}")

        elements = list(elements)
        if not elements:
            raise ProviderError("elements 不能为空：没有要素就没有比对依据")

        messages, meta = build_judge_messages(reference, image, elements, swap=swap)
        content, usage = self._chat(
            messages, temperature=0.0, json_mode=True, enable_thinking=enable_thinking
        )
        result = _normalize(_extract_json(content), elements, self._config.model, usage)
        result["payload_bytes"] = meta["payload_bytes"]
        result["reference_size"] = (meta["reference"]["width"], meta["reference"]["height"])
        result["image_size"] = (meta["image"]["width"], meta["image"]["height"])
        result["swapped"] = swap
        result["thinking"] = enable_thinking
        return result

    def describe(self, image: Path, *, allow_paid: bool = False) -> dict:
        """把一张人物参考图反推成文字角色卡（计划书 4.2 的「图生文」入口）。

        这是 `compare` 之外的第二种用法，**不是判定**：不出一致性分数，
        只出一份可被人改写的角色卡原件。放在 provider 层是因为它同样
        "读图 + 调同一个视觉模型"，而放 pipeline 层就要自己拼 messages、
        自己处理鉴权——那是 provider 的活。

        `enable_thinking=False`：反推是**观察**，不是推理。开思考会让单次
        等 100 秒以上，而多花的那些 token 只用来反复确认"这到底是不是人"。

        返回 `{"card": 原始 dict, "usage":..., "meta":...}`；
        字段归一、低置信判定、落盘缓存在 `pipeline/ref_desc.py`，不在这里。
        """
        if not allow_paid:
            raise PaidCallBlocked(
                f"{self.name}：图生文反推是付费调用，必须显式传 allow_paid=True"
            )
        image = Path(image)
        if not image.exists():
            raise ProviderError(f"参考图不存在：{image}")

        messages, meta = build_describe_messages(image)
        content, usage = self._chat(messages, temperature=0.2, json_mode=True,
                                    enable_thinking=False)
        return {"card": _extract_json(content), "usage": usage, "meta": meta}
