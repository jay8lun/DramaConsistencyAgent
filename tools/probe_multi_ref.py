"""探针：验证中转站 /images/edits 的多参考图字段名（`image[]`）。

## 为什么需要这个探针

`RelayImage._edit` 目前只发第一张参考图，并在收到多张时打印警告。
原因不是不想发，是**多图字段名未经验证**——OpenAI 兼容实现里多图有两种写法：

- `files=[("image[]", a), ("image[]", b)]`   ← 官方 images/edits 的数组写法
- `files={"image": (a,), "image2": (b,)}`    ← 某些网关的自定义写法

猜错的后果不是报错，而是**静默只用了第一张**（`input_images` 回显 1），
然后你以为两个角色都挂了参考图，其实只有第一个。

## 判据

看响应里的 `usage.input_images`：
- 2  → 字段名正确，两张都进了模型
- 1  → 只认第一张，`image[]` 不被识别
- 缺失 → 该网关不回显，只能靠看图判断

用法：
    python scripts/probe_multi_ref.py                # dry-run，不发请求
    python scripts/probe_multi_ref.py --go           # 真发（约 ¥0.02）
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

REFS = [
    Path("assets/projects/real2/anchors/char_01_front.png"),
    Path("assets/projects/real2/anchors/char_02_front.png"),
]

PROMPT = (
    "把图1中的女性和图2中的女性一起放进同一个画面：两人站在电梯轿厢里，"
    "图1的女性穿深灰色长大衣，图2的女性穿酒红色双排扣大衣，"
    "两人都保持各自完全相同的面部特征、发型和服装不变，"
    "电梯内冷白灯光，竖构图中景，电影感，超精细画质"
)

OUT = Path("assets/probe/multi_ref.png")


def main() -> int:
    ap = argparse.ArgumentParser(description="验证多参考图字段名")
    ap.add_argument("--go", action="store_true", help="真发请求（约 ¥0.02）")
    ap.add_argument("--field", default="image[]", help="多图字段名，默认 image[]")
    args = ap.parse_args()

    cfg = config.settings.image_api
    for p in REFS:
        if not p.exists():
            print(f"✗ 参考图不存在：{p}", file=sys.stderr)
            return 2

    print("=" * 66)
    print("多参考图字段名探针")
    print("=" * 66)
    print(f"base_url : {cfg.base_url}")
    print(f"model    : {cfg.model}")
    print(f"字段名   : {args.field}")
    for p in REFS:
        print(f"参考图   : {p}  ({p.stat().st_size/1024:.0f} KB)")
    print(f"提示词   : {PROMPT[:60]}…")

    from providers.relay_image import _prepare_ref

    payloads = []
    for p in REFS:
        raw, mime, name = _prepare_ref(p)
        payloads.append((args.field, (name, raw, mime)))
        print(f"  压缩后 {p.name}: {len(raw)/1024:.0f} KB → {name}")

    if not args.go:
        print("\n[dry-run] 请求可构造。加 --go 真发（约 ¥0.02）。")
        return 0

    import httpx

    url = cfg.base_url.rstrip("/") + "/images/edits"
    t0 = time.time()
    resp = httpx.post(
        url,
        headers={"Authorization": f"Bearer {cfg.api_key}"},
        files=payloads,
        data={"model": cfg.model, "prompt": PROMPT, "size": "1024x1792", "n": "1"},
        timeout=httpx.Timeout(420.0, connect=90.0),
    )
    dt = time.time() - t0
    print(f"\nHTTP {resp.status_code}  耗时 {dt:.0f}s")

    if resp.status_code != 200:
        print(resp.text[:800])
        return 1

    j = resp.json()
    usage = j.get("usage") or {}
    print(f"usage    : {json.dumps(usage, ensure_ascii=False)}")
    n_in = usage.get("input_images")
    if n_in == 2:
        print("✓ 判定：两张参考图都被接受，字段名 `%s` 可用" % args.field)
    elif n_in == 1:
        print("✗ 判定：只认第一张，`%s` 未被识别" % args.field)
    else:
        print("? 判定：网关未回显 input_images，只能看图确认")

    item = (j.get("data") or [{}])[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if item.get("b64_json"):
        OUT.write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        OUT.write_bytes(httpx.get(item["url"], timeout=180).content)
    else:
        print("✗ 响应里既无 b64 也无 url")
        return 1
    print(f"图已保存 → {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
