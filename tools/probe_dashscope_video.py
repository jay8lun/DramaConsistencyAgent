"""阿里百炼视频 i2v 接口探针（M15-S7）。

验四件事：
1. `wan2.2-i2v-flash` 是否开通、免费额度认不认 i2v
2. `img_url` 传 Base64 能不能被接受（能就不用为 i2v 单独搭图床）
3. 输出分辨率是不是 720×1280（9:16 竖屏）
4. **跑出来的视频里脸还在不在**——这才是 15.2 的全部意义

调用形态是 DashScope 异步任务制，不是 OpenAI 兼容：
    创建 POST /api/v1/services/aigc/video-generation/video-synthesis
         Header: X-DashScope-Async: enable
    轮询 GET  /api/v1/tasks/{task_id}

费用：wan2.2-i2v-flash 720P ¥0.2/秒 × 5 秒 = ¥1.00/条。

用法：
    python scripts/probe_dashscope_video.py              # dry-run
    python scripts/probe_dashscope_video.py --go         # 真跑
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from PIL import Image  # noqa: E402

PROBE_DIR = Path(__file__).resolve().parent.parent / "assets" / "probe"

BASE = "https://dashscope.aliyuncs.com"
CREATE = BASE + "/api/v1/services/aigc/video-generation/video-synthesis"
TASK = BASE + "/api/v1/tasks/"

#: 选它而不是 wan2.7-i2v 的理由：
#:   ① 720P ¥0.2/秒 vs ¥0.6/秒，差 3 倍
#:   ② 输入模态只有 Image、输出只有 Video —— 纯图生视频不出音频，
#:      与本项目「视频只出画面、声音由 TTS 单独负责」的架构一致
#:   ③ 描述里明确写「稳定保持图片文字、人像和商品一致性」，正是 15.2 要验的
MODEL = "wan2.2-i2v-flash"

#: 视频提示词必须按计划书 8.2 的规则：**只写运动，禁止重复描述外貌**。
#: 外貌已经由首帧图固定住了，再写一遍会让模型在运动和长相之间打架。
PROMPT = "她缓缓转头看向镜头，雨水顺着发梢滑落，镜头缓慢推近，光影在脸上流动"

TARGET_W, TARGET_H = 720, 1280  # 9:16 竖屏


def prepare_frame(src_path: Path, out_path: Path, quality: int = 88) -> tuple[bytes, str]:
    """把首帧图裁成严格 9:16 并缩到 720×1280，返回 (jpeg bytes, base64)。"""
    im = Image.open(src_path).convert("RGB")
    w, h = im.size
    target_ar = TARGET_W / TARGET_H
    if w / h > target_ar:          # 太宽 → 裁两侧
        nw = int(h * target_ar)
        left = (w - nw) // 2
        im = im.crop((left, 0, left + nw, h))
    else:                           # 太高 → 裁上下
        nh = int(w / target_ar)
        top = (h - nh) // 2
        im = im.crop((0, top, w, top + nh))
    im = im.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    im.save(out_path, "JPEG", quality=quality)
    raw = out_path.read_bytes()
    return raw, base64.b64encode(raw).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--key", default="", help="百炼 API Key，默认读环境变量 DASHSCOPE_API_KEY")
    ap.add_argument("--frame", default="", help="首帧图路径")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--duration", type=int, default=5)
    ap.add_argument("--resolution", default="720P")
    ap.add_argument("--timeout", type=int, default=900, help="轮询最长等待秒数")
    args = ap.parse_args()

    import os

    key = (args.key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        print("[FATAL] 没有 API Key。用 --key 传，或设 DASHSCOPE_API_KEY。")
        return 2

    src = Path(args.frame) if args.frame else PROBE_DIR / "relay_1024x1536.png"
    if not src.exists():
        print(f"[FATAL] 首帧图不存在：{src}")
        return 2

    frame_path = PROBE_DIR / "i2v_first_frame_720x1280.jpg"
    raw, b64 = prepare_frame(src, frame_path)

    print("=" * 70)
    print("百炼视频 i2v 探针")
    print("=" * 70)
    print(f"model      : {args.model}")
    print(f"resolution : {args.resolution}   duration: {args.duration}s")
    print(f"首帧源图   : {src}")
    print(f"处理后     : {frame_path}  {TARGET_W}x{TARGET_H}  {len(raw)/1024:.0f} KB (jpeg q88)")
    print(f"base64     : {len(b64)/1024:.0f} KB")
    print(f"提示词     : {PROMPT}")

    body = {
        "model": args.model,
        "input": {
            "prompt": PROMPT,
            # Base64 用 data URL 形式。若服务端要求裸 base64，报错信息会说。
            "img_url": f"data:image/jpeg;base64,{b64}",
        },
        "parameters": {
            "resolution": args.resolution,
            "duration": args.duration,
            # prompt_extend 必须关：开着时 LLM 会改写提示词，而 8.2 规定只写运动、
            # 禁止重复外貌，被改写等于绕过规格。
            "prompt_extend": False,
            "watermark": False,
        },
    }
    shown = json.loads(json.dumps(body))
    shown["input"]["img_url"] = f"<data url {len(body['input']['img_url'])} chars>"
    print("\n--- 请求体 ---")
    print(json.dumps(shown, ensure_ascii=False, indent=2))

    if not args.go:
        print("\n[dry-run] 未发送。预计 ¥1.00（720P ¥0.2/s × 5s）")
        return 0

    hdr = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    t0 = time.time()
    r = httpx.post(CREATE, headers=hdr, json=body, timeout=180)
    print(f"\n--- 创建任务 HTTP {r.status_code}  {time.time()-t0:.1f}s ---")
    try:
        j = r.json()
    except Exception:  # noqa: BLE001
        print(r.text[:1000])
        return 4
    print(json.dumps(j, ensure_ascii=False, indent=1)[:1200])

    task_id = (j.get("output") or {}).get("task_id")
    if not task_id:
        print("\n[失败] 没有拿到 task_id。常见原因：模型未开通 / img_url 格式不对 / 余额不足。")
        return 5
    print(f"\ntask_id = {task_id}\n开始轮询（最长 {args.timeout}s）…")

    deadline = time.time() + args.timeout
    last = ""
    while time.time() < deadline:
        time.sleep(10)
        try:
            q = httpx.get(TASK + task_id, headers={"Authorization": f"Bearer {key}"}, timeout=90)
            qj = q.json()
        except Exception as exc:  # noqa: BLE001
            print(f"  轮询异常 {type(exc).__name__}，继续")
            continue
        out = qj.get("output") or {}
        status = out.get("task_status") or qj.get("task_status") or "?"
        if status != last:
            print(f"  [{time.time()-t0:6.1f}s] {status}")
            last = status
        if status in ("SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"):
            print("\n--- 终态响应 ---")
            print(json.dumps(qj, ensure_ascii=False, indent=1)[:2000])
            if status == "SUCCEEDED":
                vurl = None
                res = out.get("results") or out.get("result") or []
                if isinstance(res, list) and res:
                    vurl = res[0].get("url") or res[0].get("video_url")
                if not vurl:
                    vurl = out.get("video_url")
                if vurl:
                    mp4 = PROBE_DIR / "i2v_result.mp4"
                    v = httpx.get(vurl, timeout=300)
                    mp4.write_bytes(v.content)
                    print(f"\n已下载 → {mp4}  ({len(v.content)/1024:.0f} KB)")
                    print("下一步：抽帧看脸 / 用 ffprobe 看分辨率")
                    return 0
                print("\n[异常] 成功但没有 video url，看上面的响应体")
                return 6
            return 7
    print("\n[超时] 轮询到时限仍未终态。任务可能还在跑，用 task_id 手查：")
    print(f"  GET {TASK}{task_id}")
    return 8


if __name__ == "__main__":
    raise SystemExit(main())
