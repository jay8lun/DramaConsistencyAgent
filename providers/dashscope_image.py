"""出图 provider：DashScope 原生的「多模态生成」接口（文生图 / 图生图 / 多图参考）。

**不带厂商名**（A.8 第 2 条）：base_url、key、model 全部来自 `settings.image_native_api`，
换供应商只改 `.env`，这个文件不用动。当前指向阿里百炼，模型 wan2.7-image。

实现 `ImageProvider` 契约（A.3），与 `MockImage` / `RelayImage` 可互换。

## 为什么需要这个文件（2026-10-29）

出图原本只有中转站一家（`relay_image.py`）。10-29 那天它整站故障：

    GET  /v1/models            200  ✅
    POST /v1/images/generations 503  ❌ 2~4 秒快速失败
    POST /v1/images/edits       503  ❌

关键证据是**连 prompt 都不传也返回 503**——请求根本没走到参数校验层，
说明是上游挂了，不是模型名或参数的问题。4 个模型、两种尺寸、有/无 prompt，
共 10 次请求全部 503。出图一旦单点，整条流水线停摆，所以加了这条独立来源。

## 与 OpenAI 兼容接口的三个结构差异

1. **不是 `/images/generations`**，而是
   `POST {base}/api/v1/services/aigc/multimodal-generation/generation`。
2. **输入是 messages 结构**，不是扁平的 prompt 字段：
   `input.messages[0].content = [{"image": data_url}, ..., {"text": prompt}]`
   —— 参考图就是这个数组里的 `image` 项，**可以有多个**。
3. **返回是 OSS 临时 URL**（带 Expires 签名），不是 b64，需要再下载一次。

## 实测数据（2026-10-29）

| 场景 | 耗时 | 输入 token | 输出 | 结果 |
|---|---|---|---|---|
| 文生图 1024×1792 | 17.8~18.6s | 80 | 1024×1792 | 写实、无水印 |
| 单参考图换场景 | 7.6s | 9,573 | 1024×1792 | VLM 判 95 分、五要素全 PASS |
| **双参考图（两个角色）** | 8.1s | 18,996 | 1024×1792 | 两人脸与服装均独立保持 |

**尺寸是精确的**：传 `1024*1792` 就得到 1024×1792（比例 0.5714）。
中转站会把 1024×1792 吸附成 948×1659，这里不会——少一次后端缩放。

## 多参考图：这是本文件相对 relay 的实质增量

`relay_image.py` 的 `_edit` 只取第一张参考图，并在日志里留痕（多图字段名未验证）。
本接口的多图是**同一个 content 数组里放多个 image 项**，实测两个角色各自
的脸与服装都保住了（灰大衣 + 酒红双面呢，没有互相污染）。
所以 S3 的多角色镜头终于可以把所有角色的定妆照都挂上，而不是靠提示词蒙。

## 不做的事

**不做裁剪。** 与 relay 一致：生成什么比例就交付什么，9:16 的裁切由 S4 在喂给
i2v 之前统一做（`dashscope_video.py` 里 `_normalize_frame`）。
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

import config
from providers.base import ImageProvider, PaidCallBlocked, ProviderError

#: 原生接口路径。注意与 OpenAI 兼容端点（`/compatible-mode/v1/...`）是两套东西，
#: 即使同一个 key 都能用，路径与请求体结构也完全不同。
GENERATE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"

#: 重试次数与退避。这里**不用 relay 那套 12 个状态码**，因为实测该接口很干净：
#: 3 次请求 0 次抖动。只保留最常见的瞬时错误，宁可快速失败也不长时间卡住。
MAX_ATTEMPTS = 3
RETRY_BASE_SEC = 5.0
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: 参考图上传前的压缩参数。
#: 实测：定妆照 2 MB PNG → 103 KB JPEG(1200px/q92) 后，身份保真判分不变（95 分）。
#: 不压也能用，但 base64 会把体积再放大 1/3 塞进 JSON body，
#: 大 body 只是白白拖慢上传，对生成质量没有贡献。
REF_MAX_SIDE = 1200
REF_JPEG_QUALITY = 92

#: **平台硬要求：参考图最短边不小于 512px。**
#: 2026-10-31 实测：用 121×246 的人物图当参考，接口直接 400——
#:   InvalidParameter: Error validating image ... resolution must ...
#: 注意它是在**上传到 OSS 之后**才校验的，报错里只有一个签名 URL，
#: 看不出是哪张图、原图多大，排查很费劲。所以这里在发送前就自己兜住。
REF_MIN_SIDE = 512

#: 为了满足最短边，长边最多放宽到这里；再大就别传了（体积换不来质量）。
REF_HARD_MAX = 2048


def _target_size(w: int, h: int) -> tuple[int, int]:
    """参考图的目标尺寸：**先保证最短边 ≥ REF_MIN_SIDE，再压低长边**。

    分两种情况，不能合成一个 scale：

    - **小图**（短边 < 512）：必须放大。放大后长边可能顶到 REF_HARD_MAX，
      到这里为止就是上限——再大只是白撑 body，换不来细节。
    - **大图**（短边已够）：不放大，只按 REF_MAX_SIDE 压体积（原有行为）。
      如果不分开写，3000×4000 这种图会被"放宽到 2048"的规则误伤，
      body 从 ~200KB 涨到 ~1.3MB，纯属变慢。
    """
    short, long_ = min(w, h), max(w, h)
    if short < REF_MIN_SIDE:
        scale = REF_MIN_SIDE / short
        if long_ * scale > REF_HARD_MAX:
            scale = REF_HARD_MAX / long_
    else:
        scale = min(1.0, REF_MAX_SIDE / long_)
    return max(1, round(w * scale)), max(1, round(h * scale))

#: 单次请求超时。实测 7.6~18.6 秒，给 600s 余量以容纳排队。
TIMEOUT = httpx.Timeout(600.0, connect=90.0)

#: 下载生成图（OSS 签名 URL）的超时。文件 2.5~3.5 MB。
DOWNLOAD_TIMEOUT = 300.0


def _warn(msg: str) -> None:
    """打一行诊断信息，**永远不抛**。

    为什么需要这个：`print(..., file=sys.stderr)` 在下面两种环境里会抛
    `OSError: [Errno 22] Invalid argument`（Windows，实测）：

    - 服务用管道启动且下游已退出（`python run.py | head -40`，head 读满即退）
    - 进程被托管、stdout/stderr 句柄已失效（服务管理器 / 脱离终端运行）

    stderr 无缓冲，写不进去当场抛；而 stdout 有缓冲，同样的故障要等到
    进程退出才暴露——所以"日志看起来正常、流水线突然 OSError"这种组合很迷惑人。

    诊断信息的价值远低于一次付费调用，所以这里一律吞掉写失败。
    """
    try:
        print(f"⚠ {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 —— 见 docstring：不许因为打日志失败中断出图
        pass


def _prepare_ref(ref: Path) -> tuple[str, dict[str, int | bool]]:
    """参考图 → (`data:image/jpeg;base64,...`, 尺寸信息)。

    压成 JPEG 再传：实测不影响身份保真（判分 95），但把 body 从 ~4.5 MB 降到 ~140 KB。

    **小图会被放大**（`_target_size`）。这不是"优化画质"——放大换不回丢失的细节，
    模型看到的仍然是一张糊脸。它只是把"平台 400 直接失败"换成"能跑，但脸可能不像"，
    并在返回的尺寸信息里如实标 `upscaled`，让调用方把这句话说出来。
    不这么做的话，用户只会看到一条带 OSS 签名的 400，不知道是自己那张图太小。

    压图失败（损坏 / 非位图）时退回原文件直传——宁可慢，也不能因为压图失败把这一镜判死。
    """
    try:
        with Image.open(ref) as im:
            im = im.convert("RGB")
            w, h = im.size
            nw, nh = _target_size(w, h)
            if min(nw, nh) < REF_MIN_SIDE:
                # 宽高比太极端（例如 100×5000）：放到长边上限也凑不出 512 的短边。
                # 这种情况放大是假的，直接说清原因，别让平台回一条看不懂的 400。
                raise ProviderError(
                    f"参考图 {ref.name} 是 {w}×{h}，宽高比过于极端："
                    f"最短边最多只能放大到 {min(nw, nh)}px，低于平台要求的 "
                    f"{REF_MIN_SIDE}px。请裁成接近 9:16 的构图，或换一张更方的图"
                )
            if (nw, nh) != (w, h):
                im = im.resize((nw, nh), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=REF_JPEG_QUALITY, optimize=True)
        info: dict[str, int | bool] = {
            "w": w, "h": h, "out_w": nw, "out_h": nh,
            "upscaled": nw > w or nh > h,
        }
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), info
    except ProviderError:
        raise
    except Exception:  # noqa: BLE001
        raw = ref.read_bytes()
        mime = "image/png" if ref.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(raw).decode(), {}


def _encode_ref(ref: Path) -> str:
    """只要 data URL 时的薄封装（保持旧签名，离线断言与 relay 那边不受影响）。"""
    return _prepare_ref(ref)[0]


def _native_size(size: str) -> str:
    """契约里的 `1024x1792` → 本接口要的 `1024*1792`。

    契约用 `WxH`（与 OpenAI 兼容接口一致），DashScope 原生用 `W*H`。
    这一层转换放在 provider 内部，上游不需要知道供应商的写法差异。
    """
    return size.replace("x", "*").replace("X", "*")


class DashScopeImage(ImageProvider):
    """多模态生成接口的出图 provider（文生图 + 图生图 + 多图参考）。"""

    name = "dashscope-image"
    is_mock = False

    def __init__(self) -> None:
        cfg = config.settings.image_native_api
        if not (cfg.api_key and cfg.base_url and cfg.model):
            raise ProviderError(
                "image_native_api 未配置完整：需要 IMAGE_NATIVE_API_KEY / "
                "IMAGE_NATIVE_BASE_URL / IMAGE_NATIVE_MODEL"
            )
        self._cfg = cfg
        self.unit_price = config.settings.price_image

    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._cfg.api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        prompt: str,
        *,
        out: Path,
        references: list[Path] | None = None,
        size: str = "1024x1792",
        allow_paid: bool = False,
    ) -> Path:
        if not allow_paid:
            raise PaidCallBlocked(
                f"{self.name}：出图是付费调用（¥{self.unit_price}/张），必须显式传 allow_paid=True"
            )
        if not prompt or not prompt.strip():
            raise ProviderError("prompt 不能为空")

        refs = [Path(p) for p in (references or []) if p]
        for p in refs:
            if not p.exists():
                raise ProviderError(f"参考图不存在：{p}")

        # 参考图与文本放在同一个 content 数组里，顺序即「图1、图2…」的指代顺序。
        # 提示词里说「图1中的女性」能对上号，这一点实测有效。
        content: list[dict] = []
        for p in refs:
            uri, info = _prepare_ref(p)
            content.append({"image": uri})
            if info.get("upscaled"):
                # 必须说出来。放大能跑通，但脸会糊、生成的可能是"另一个人"，
                # 而这件事从成片上分不出来（画面正常、不报错）——正是本项目一直在防的失败形态。
                #
                # **但这一行绝不能把流水线搞挂**（2026-10-31 踩到）：服务如果用
                # `python run.py | head -40` 这类管道启动，`head` 读满就退出、管道读端关闭，
                # 之后往 stderr 写会抛 `OSError: [Errno 22] Invalid argument`
                # （stderr 无缓冲，当场炸；stdout 有缓冲，所以日志看起来一切正常）。
                # 那次的现场是：S1 正常、S2a 报 OSError、积分一分没扣——纯粹被一行告警打断。
                _warn(f"参考图 {p.name} 只有 {info['w']}×{info['h']}，"
                      f"已放大到 {info['out_w']}×{info['out_h']} 才满足平台最短边 {REF_MIN_SIDE}px。"
                      "放大换不回细节，生成的脸可能不太像这张图；要像就换一张更大的原图")
        content.append({"text": prompt})

        body = {
            "model": self._cfg.model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "size": _native_size(size),
                "n": 1,
                # 不留水印。本项目成片用 ffmpeg 自己烧字幕，画面里不能有第三方标记。
                "watermark": False,
            },
        }
        return self._post_and_save(body, out)

    # ------------------------------------------------------------------
    def _post_and_save(self, body: dict, out: Path) -> Path:
        url = self._cfg.base_url.rstrip("/") + GENERATE_PATH
        last: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = httpx.post(url, headers=self._headers(), json=body, timeout=TIMEOUT)
                if resp.status_code == 200:
                    return self._save(resp.json(), out)
                if resp.status_code not in RETRYABLE_STATUS:
                    # 4xx（鉴权 / 参数 / 模型未开通）重试无意义，直接把原文抛出去
                    raise ProviderError(
                        f"{self.name} 出图 HTTP {resp.status_code}: {resp.text[:400]}"
                    )
                last = ProviderError(
                    f"{self.name} 出图 HTTP {resp.status_code}（上游抖动，可重试）"
                )
            except ProviderError:
                raise
            except Exception as exc:  # noqa: BLE001  连接层错误才重试
                last = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_SEC * attempt)
        raise ProviderError(f"{self.name} 出图连续 {MAX_ATTEMPTS} 次失败：{last}")

    # ------------------------------------------------------------------
    def _save(self, payload: dict, out: Path) -> Path:
        """从 `output.choices[0].message.content` 里取图片 URL 并下载。"""
        image_url = self._extract_url(payload)
        if not image_url:
            raise ProviderError(f"{self.name} 响应里没有图片：{str(payload)[:300]}")

        resp = httpx.get(image_url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
        if resp.status_code != 200:
            raise ProviderError(f"{self.name} 下载生成图失败 HTTP {resp.status_code}")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return out

    @staticmethod
    def _extract_url(payload: dict) -> str:
        """遍历 content 数组找图片项。

        结构（实测）：
            output.choices[0].message.content = [{"image": "https://...", "type": "image"}]
        个别情况下同一数组里还会有文本项，所以按 type / 有无 image 字段判断，
        而不是硬取 `[0]`。
        """
        choices = ((payload.get("output") or {}).get("choices") or [])
        if not choices:
            return ""
        content = ((choices[0].get("message") or {}).get("content")) or []
        if isinstance(content, dict):          # 少数返回会把 content 给成对象
            content = [content]
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("image"):
                return item["image"]
        return ""
