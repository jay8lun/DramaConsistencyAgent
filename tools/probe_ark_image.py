"""方舟图像生成接口探针（M15-S4）。

目的有三个，按重要性排序：

1. **验「参考图」的语义**：seedream 的输入图到底是「图生图」（改风格，脸会变）
   还是「主体参考」（还是这个人，换场景）。这决定 S3 能不能成立。
2. 验 Key 能不能用、模型有没有开通。报错信息会直接说是哪一层的问题。
3. 看清返回结构（URL 还是 base64、有没有 seed、usage 长什么样）。

**为什么不用 SDK 反查字段名**：`arkruntime` 在镜像源里不存在，官方源拉取超时。
而方舟对参数错误返回的是 400 + 明确的字段提示，且**参数校验失败不计费**
（见 T-017 第 17 条：审核失败不计费）。所以直接打一次比翻文档快。

用法：
    python scripts/probe_ark_image.py              # dry-run，只打印将要发送的内容
    python scripts/probe_ark_image.py --go          # 真发（最小请求，约 ¥0.30）
    python scripts/probe_ark_image.py --go --with-ref   # 带参考图，验语义（这才是重点）
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402  openai 的传递依赖，环境里已有

import config  # noqa: E402

PROBE_DIR = config.ASSETS / "probe"

#: 出图模型。
#:
#: **必须带日期后缀。** T-017 给的是 `doubao-seedream-5-0-pro`，实测报
#: `InvalidEndpointOrModel.NotFound`（HTTP 404）——文档里的短名不能直接调用。
#: 从 `GET /api/v3/models` 拿到的真实 ID 是 `doubao-seedream-5-0-pro-260628`，
#: 其 `task_type` 为 `['ImageToImage', 'TextToImage']`，即图生图是明确能力。
IMAGE_MODEL = "doubao-seedream-5-0-pro-260628"

#: 15.2 要的竖屏尺寸。1024×1792 = 183.5 万像素 < 261 万，落在 ¥0.30 档；
#: 若写成 2K 竖屏会越过 261 万线，单价跳到 ¥0.60（T-017 第 16 条）。
SIZE = "1024x1792"

#: 探测用的提示词。刻意写成「主体不变、只换环境」，这样出来的图能直接判断
#: 参考图是保 ID 还是只做风格迁移。
PROMPT_PLAIN = (
    "一位 25 岁亚洲女性，黑色齐肩直发，鹅蛋脸，深棕色眼睛，浅麦色皮肤，中等体型，"
    "深灰色羊毛大衣，内搭白色高领毛衣，站在雨夜街头的中景，暖黄路灯从右后方照射，"
    "电影感，超精细画质"
)
PROMPT_WITH_REF = (
    "保持这个人完全相同的面部特征、五官比例和发型不变，"
    "把她放到雨夜街头的咖啡馆门口，中景，暖黄路灯从右后方照射，电影感，超精细画质"
)


def _mask(s: str, keep: int = 6) -> str:
    return f"{s[:keep]}...{s[-4:]} (len={len(s)})" if len(s) > 12 else "<短值>"


def build_payload(*, model: str, with_ref: bool, ref_image: str | None) -> dict:
    """组装请求体。

    字段名刻意只用 OpenAI 图像生成的标准字段 + 方舟文档里出现过的两个。
    `image` 是 seedream 系列常见的参考图字段名；若服务端不认，报错会把合法字段列出来。
    """
    body: dict = {
        "model": model,
        "prompt": PROMPT_WITH_REF if with_ref else PROMPT_PLAIN,
        "size": SIZE,
        "response_format": "url",
        "watermark": False,
    }
    if with_ref and ref_image:
        # 先按「单值」传。若服务端要求数组，报错会说 expected array。
        body["image"] = ref_image
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="真发请求（会花钱）")
    ap.add_argument("--with-ref", action="store_true", help="带参考图，验参考图语义")
    ap.add_argument("--ref", default="", help="参考图路径，默认用 M0 的定妆照样本")
    args = ap.parse_args()

    settings = config.settings
    # T-017 第 2 条：方舟的图像、视频、文本共用同一个 ARK_API_KEY。
    # 本地 DOUBAO_API_KEY 是空的，而 VLM_API_KEY 是有效的方舟 Key（M0 用它跑过 30 对），
    # 所以这里做回退，而不是去 .env 里复制一份 —— 同一凭据存两处，改的时候必然漏一处。
    api_key = (settings.ark.api_key or settings.judge_vlm.api_key or "").strip()
    base_url = (settings.ark.base_url or settings.judge_vlm.base_url or "").strip().rstrip("/")
    model = (settings.image_model or IMAGE_MODEL).strip()

    print("=" * 70)
    print("方舟图像接口探针")
    print("=" * 70)
    print(f"base_url : {base_url}")
    print(f"api_key  : {_mask(api_key)}" + ("  [回退自 VLM_API_KEY]" if not settings.ark.api_key else ""))
    print(f"model    : {model}")
    print(f"size     : {SIZE}  （{1024*1792/10000:.1f} 万像素）")

    if not api_key or not base_url:
        print("\n[FATAL] .env 里 VLM_API_KEY / VLM_BASE_URL 为空，先填上。")
        return 2

    # ---- 参考图准备 ----
    ref_image = None
    if args.with_ref:
        ref_path = Path(args.ref) if args.ref else None
        if ref_path is None:
            # 默认找一个 M0 阶段的真实人脸图当参考
            cands = sorted(config.ASSETS.glob("m0*/**/*.png")) + sorted(
                config.PROJECTS_DIR.glob("*/anchors/*front*.png")
            )
            ref_path = cands[0] if cands else None
        if ref_path is None or not ref_path.exists():
            print("\n[FATAL] 找不到参考图，用 --ref 指定一个。")
            return 2
        raw = ref_path.read_bytes()
        b64 = base64.b64encode(raw).decode()
        mime = "png" if ref_path.suffix.lower() == ".png" else "jpeg"
        ref_image = f"data:image/{mime};base64,{b64}"
        print(f"参考图   : {ref_path}  ({len(raw)/1024:.0f} KB → base64 {len(ref_image)/1024:.0f} KB)")

    payload = build_payload(model=model, with_ref=args.with_ref, ref_image=ref_image)
    shown = dict(payload)
    if "image" in shown:
        shown["image"] = f"<data url {len(shown['image'])} chars>"

    print("\n--- 请求体 ---")
    print(json.dumps(shown, ensure_ascii=False, indent=2))

    url = f"{base_url}/images/generations"
    print(f"\nPOST {url}")

    if not args.go:
        print("\n[dry-run] 未发送。确认无误后加 --go。")
        print("预计花费：¥0.30（1024x1792 落在 ≤261 万像素档）")
        return 0

    # ---- 真发 ----
    t0 = time.time()
    try:
        resp = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[NETWORK ERROR] {type(exc).__name__}: {exc}")
        return 3

    dt = time.time() - t0
    print(f"\n--- 响应 HTTP {resp.status_code}  耗时 {dt:.1f}s ---")

    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        print(resp.text[:2000])
        return 4

    # base64 太长了，打印时截断
    def _trim(obj):
        if isinstance(obj, dict):
            return {k: _trim(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_trim(v) for v in obj]
        if isinstance(obj, str) and len(obj) > 200:
            return obj[:120] + f"...<{len(obj)} chars>"
        return obj

    print(json.dumps(_trim(data), ensure_ascii=False, indent=2)[:4000])

    if resp.status_code != 200:
        print("\n[失败] 但参数校验/审核失败不计费。对照上表判断是哪一层：")
        print("  401 → Key 无效（本地是 ark- 前缀旧格式，需控制台重新生成）")
        print("  403 → 模型未开通")
        print("  400 + 字段提示 → 按提示改字段名，改完重跑")
        return 5

    # ---- 成功，落盘 ----
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    item = (data.get("data") or [{}])[0]
    stamp = time.strftime("%H%M%S")
    out = PROBE_DIR / f"probe_{'ref' if args.with_ref else 'plain'}_{stamp}.png"

    if item.get("url"):
        print(f"\n返回的是 URL：{item['url'][:120]}...")
        try:
            img = httpx.get(item["url"], timeout=120.0)
            out.write_bytes(img.content)
            print(f"已下载 → {out}  ({len(img.content)/1024:.0f} KB)")
        except Exception as exc:  # noqa: BLE001
            print(f"[下载失败] {exc}")
            return 6
    elif item.get("b64_json"):
        out.write_bytes(base64.b64decode(item["b64_json"]))
        print(f"返回的是 base64，已落盘 → {out}")
    else:
        print("[异常] 200 但既没有 url 也没有 b64_json，看上面的响应体。")
        return 7

    print("\n" + "=" * 70)
    print("关键结论要回答的两件事：")
    print("  1. 返回的是 url 还是 base64 —— 决定 i2v 首帧能不能直接透传（T-017 第 8 条要求公网 URL）")
    print("  2. 带 --with-ref 时脸还是不是同一个人 —— 决定 S3 的参考图模式能不能成立")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
