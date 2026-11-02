"""出图 provider：OpenAI 兼容的图像生成 / 图像编辑接口。

**不带厂商名**（A.8 第 2 条）：base_url、key、model 全部来自 `settings.image_api`，
换供应商只改 `.env`，这个文件不用动。当前指向一个中转站，模型 gpt-image-2.5-sunburst。

它实现的是 `ImageProvider` 契约（A.3），与 `MockImage` 可互换。

## 两个接口，按有没有参考图分流

- **无参考图** → `POST {base}/images/generations`，纯文本出图（S2 定妆照的第一张）
- **有参考图** → `POST {base}/images/edits`，multipart 上传 + 文本指令
  （S3 首帧必须挂定妆照，这是全片一致性的唯一来源）

实测依据（2026-10-28）：
- generations：HTTP 200，20.9~67.4s，返回 b64_json（2.4~2.6 MB PNG）
- edits：HTTP 200，94.7s，`usage.input_images: 1`，**脸保得住**（VLM 判 93/95）
- `size` 传 1024x1792 会被等比吸附成 948x1659（比例 0.5714 不变，像素略减）
- 该中转站连通性间歇：同一时刻连发三次 `/v1/models` 结果是 ConnectError / 200 / 200，
  所以连接层错误必须重试

## 不做的事

**不做裁剪。** 生成什么比例就交付什么比例，9:16 的裁切由 S4 在喂给 i2v 之前统一做
（`prompt_builder` 里已有画幅约定）。provider 的职责边界是「把图生成出来并落盘」，
把裁切塞进来会让「同一张图被不同阶段要求不同画幅」时无从下手。
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path

import httpx

import config
from providers.base import ImageProvider, PaidCallBlocked, ProviderError

#: 连接层错误的重试次数。3 次是实测够用的值：中转站的抖动通常是单次握手失败，
#: 隔 5/10 秒再试就能过。不要盲目拉高——长时间卡在重试里比失败更难排查。
MAX_ATTEMPTS = 3
RETRY_BASE_SEC = 5.0

#: **可重试**的 HTTP 状态码。这些不是参数错误，是网关/上游抖动，重发有意义。
#:
#: 524 是 Cloudflare 的「源站响应超时」——2026-10-29 真实把流水线打断过一次：
#: 参考图出图（edits）单次要 ~95s，而网关在 100s 左右就放弃，于是返回 524，
#: 图其实可能已经生成完。当成硬失败会让整条流水线白跑。
#: 429 也要重试（限流），但要走退避，不能立刻重发。
RETRYABLE_STATUS = frozenset(
    {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
)

#: 参考图上传前的压缩上限（长边像素）与 JPEG 质量。
#: edits 是 multipart 上传，2 MB PNG 会让「上传 + 生成」更容易撞上网关超时。
#: 实测压到 ~300 KB JPEG 后，同一套提示词的判定分没有变化（93~95），
#: 但请求完成时间明显更稳。
REF_MAX_SIDE = 1600
REF_JPEG_QUALITY = 92

#: 单次请求超时。实测 edits 最慢 94.7s，给 420s 的余量；
#: 连接超时单独放宽到 90s，因为失败案例里多数卡在 TLS 握手。
TIMEOUT = httpx.Timeout(420.0, connect=90.0)


def _prepare_ref(ref: Path) -> tuple[bytes, str, str]:
    """把参考图压成 JPEG 再上传，返回 (字节, mime, 文件名)。

    为什么要压：定妆照是 2 MB 级 PNG，而 edits 是 multipart 上传 + 模型生成
    串在一次请求里，网关（Cloudflare）在 100s 左右就会甩 524。压缩不改变画面内容，
    只减少传输与解码开销，让整次请求更快收尾。

    压缩失败（文件损坏 / 非位图）时**原样上传**并保留原 mime——宁可慢，
    也不能因为压图失败就把这一镜判死。
    """
    try:
        import io

        from PIL import Image

        with Image.open(ref) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = REF_MAX_SIDE / max(w, h)
            if scale < 1:
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=REF_JPEG_QUALITY, optimize=True)
        return buf.getvalue(), "image/jpeg", ref.stem + ".jpg"
    except Exception:  # noqa: BLE001
        return ref.read_bytes(), (mimetypes.guess_type(ref.name)[0] or "image/jpeg"), ref.name


class RelayImage(ImageProvider):
    """OpenAI 兼容的图像生成 provider（中转站 / 官方 / 任何自建兼容服务都能用）。"""

    name = "relay-image"
    is_mock = False

    def __init__(self) -> None:
        cfg = config.settings.image_api
        if not (cfg.api_key and cfg.base_url and cfg.model):
            raise ProviderError(
                "image_api 未配置完整：需要 IMAGE_API_KEY / IMAGE_BASE_URL / IMAGE_MODEL"
            )
        self._cfg = cfg
        self.unit_price = config.settings.price_image

    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._cfg.api_key}"}

    def _save(self, payload: dict, out: Path) -> Path:
        """把响应里的 b64_json 或 url 落盘。两种都可能出现，按存在的那个处理。"""
        item = (payload.get("data") or [{}])[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        if item.get("b64_json"):
            out.write_bytes(base64.b64decode(item["b64_json"]))
            return out
        url = item.get("url")
        if url:
            resp = httpx.get(url, timeout=180.0, follow_redirects=True)
            if resp.status_code != 200:
                raise ProviderError(f"下载生成图失败 HTTP {resp.status_code}")
            out.write_bytes(resp.content)
            return out
        raise ProviderError(f"响应里既没有 b64_json 也没有 url：{str(payload)[:200]}")

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        out: Path,
        references: list[Path] | None = None,
        size: str = "1024x1792",
        allow_paid: bool = False,
    ) -> Path:
        """生成一张图并落盘。

        `references` 非空时走 edits（挂参考图保人脸），否则走 generations。
        """
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

        if refs:
            return self._edit(prompt, out=out, refs=refs, size=size)
        return self._text2image(prompt, out=out, size=size)

    # ------------------------------------------------------------------
    def _text2image(self, prompt: str, *, out: Path, size: str) -> Path:
        url = self._cfg.base_url.rstrip("/") + "/images/generations"
        body = {
            "model": self._cfg.model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
        }
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = httpx.post(url, headers=self._headers(), json=body, timeout=TIMEOUT)
                if resp.status_code == 200:
                    return self._save(resp.json(), out)
                if resp.status_code not in RETRYABLE_STATUS:
                    # 4xx（参数/鉴权错）重试无意义，直接报出原文
                    raise ProviderError(
                        f"{self.name} 出图 HTTP {resp.status_code}: {resp.text[:400]}"
                    )
                last = ProviderError(f"{self.name} 出图 HTTP {resp.status_code}（网关抖动，可重试）")
            except ProviderError:
                raise
            except Exception as exc:  # noqa: BLE001  连接层错误才重试
                last = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_SEC * attempt)
        raise ProviderError(f"{self.name} 出图连续 {MAX_ATTEMPTS} 次失败：{last}")

    # ------------------------------------------------------------------
    def _edit(self, prompt: str, *, out: Path, refs: list[Path], size: str) -> Path:
        """挂参考图生成。

        multipart 字段名用 `image`（单数）。实测该字段名被接受，
        `usage.input_images` 回显 1，说明图确实进了模型。

        多张参考图目前只取第一张：契约允许 `references` 是列表，但当前接口的单图
        字段名已验证、多图字段名（`image[]`）未验证。**宁可只用第一张并留痕，
        也不要用未验证的字段名静默丢图**——S3 目前只挂一张定妆照，不受影响。
        """
        if len(refs) > 1:
            # 不静默：让调用方在日志里看见「第二张没被用」
            print(f"[{self.name}] 警告：收到 {len(refs)} 张参考图，当前只使用第一张 {refs[0].name}")

        ref = refs[0]
        payload, mime, fname = _prepare_ref(ref)
        url = self._cfg.base_url.rstrip("/") + "/images/edits"
        data = {"model": self._cfg.model, "prompt": prompt, "size": size, "n": "1"}

        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = httpx.post(
                    url,
                    headers=self._headers(),
                    files={"image": (fname, payload, mime)},
                    data=data,
                    timeout=TIMEOUT,
                )
                if resp.status_code == 200:
                    return self._save(resp.json(), out)
                if resp.status_code not in RETRYABLE_STATUS:
                    raise ProviderError(
                        f"{self.name} 参考图出图 HTTP {resp.status_code}: {resp.text[:400]}"
                    )
                last = ProviderError(
                    f"{self.name} 参考图出图 HTTP {resp.status_code}（网关抖动，可重试）"
                )
            except ProviderError:
                raise
            except Exception as exc:  # noqa: BLE001
                last = exc
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_SEC * attempt)
        raise ProviderError(f"{self.name} 参考图出图连续 {MAX_ATTEMPTS} 次失败：{last}")
