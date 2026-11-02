"""15.2 三景别验证：补齐全景与近景（M15-S9）。

中景已经在 M15-S7 单条验证通过（VLM 93/95 分）。这一步补另两个景别，
因为**全景是 i2v 最容易崩脸的情况**——脸只占画面一小块，模型没有足够的
像素去维持身份，而中景能过不代表全景能过。

每个景别走同一条链：
    中转站 /images/edits（挂定妆照保脸）→ 裁 9:16 720×1280
    → 百炼 wan2.2-i2v-flash 5s → 抽末帧 → 豆包 VLM 判定

费用：
    出图 2 × ¥0.02 = ¥0.04
    视频 2 × 5s   = 走 50 秒免费额度（还剩 45 秒，不额外扣）
    判定 2 × ¥0.09 = ¥0.18

用法：
    python scripts/m152_triple.py            # dry-run
    python scripts/m152_triple.py --go
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import httpx  # noqa: E402

import config  # noqa: E402
from providers.vlm import VLMJudge  # noqa: E402
from pipeline.judge import ELEMENTS, judge as normalize  # noqa: E402
from probe_dashscope_video import CREATE, TASK, prepare_frame  # noqa: E402

# ---- 出图（中转站）----
# **Key 只从 .env 读，不写在这里。** 原先硬编码在这个脚本里，属于泄露风险：
# 脚本会进版本库，而 .env 在 .gitignore 里。
RELAY_BASE = config.settings.image_api.base_url
RELAY_KEY = config.settings.image_api.api_key
RELAY_MODEL = config.settings.image_api.model
RELAY_PRICE = config.settings.price_image

# ---- 方舟（出图，中转站不可用时的回退） ----
#: 2026-10-28 21:27 实测中转站 api.x5m5x.com 全面不可达（GET /v1/models 三次分别
#: ConnectTimeout / ReadTimeout / ConnectTimeout），连最轻的请求都握手超时，
#: 判断为其服务端故障而非本端问题。方舟这条链路已开通、走赠送额度，
#: 且参考图（image 字段传 data URL）在 M15-S4 已验证可用，作为回退。
ARK_BASE = "https://ark.cn-beijing.volces.com/api/v3"
ARK_MODEL = "doubao-seedream-5-0-pro-260628"

# ---- 百炼（视频） ---- 同样只从 .env 读
DASHSCOPE_KEY = config.settings.video_api.api_key
VIDEO_MODEL = config.settings.video_api.model

OUT = ROOT / "assets" / "probe" / "m152"

#: 参考图 = 已确认过能保脸的那张（中景，站雨夜街头）。
#: 三张首帧都挂同一张作参考，这才是"同一张定妆照派生出不同景别"的真实场景。
REF_IMAGE = ROOT / "assets" / "probe" / "relay_1024x1536.png"

#: 8.1 图像提示词规格：整句写全画面，缺一项模型就瞎猜。
#: 景别必须显式写，而且要用模型能听懂的说法（"全身远景"比"全景"有效）。
SCENES = [
    {
        "key": "wide",
        "name": "全景",
        "image_prompt": (
            "保持这个人完全相同的面部特征、五官比例和发型不变。"
            "全身远景，她独自站在雨夜空旷的街口，人物在画面中占三分之一高度，"
            "深灰色羊毛大衣的白领口清晰可见，远处店铺霓虹在湿滑地面拖出倒影，"
            "暖黄路灯从左前方照射，竖构图，电影感，超精细画质"
        ),
        "video_prompt": "她缓缓向前走了两步，雨水落在地面溅起细小水花，镜头缓慢后拉",
    },
    {
        "key": "closeup",
        "name": "近景",
        "image_prompt": (
            "保持这个人完全相同的面部特征、五官比例和发型不变。"
            "面部特写，肩膀以上，她侧头看向镜头，发丝被雨水打湿贴在脸颊，"
            "皮肤质感清晰可见，背景是虚化的暖黄路灯光斑，竖构图，电影感，超精细画质"
        ),
        "video_prompt": "她轻轻眨了下眼，嘴角微微上扬，镜头极缓慢推近",
    },
]


def ark_generate(prompt: str, out_path: Path, ref: Path, attempts: int = 3) -> Path:
    """方舟 seedream 出图，挂参考图保脸（image 字段传 data URL）。

    M15-S4 已验证：`image` 字段名正确，响应 `usage.input_images: 1`，
    返回的是方舟 TOS 上的公网 URL（可直接下载）。
    """
    key = (config.settings.ark.api_key or config.settings.judge_vlm.api_key or "").strip()
    assert key, "方舟 Key 为空"
    b64 = base64.b64encode(ref.read_bytes()).decode()
    body = {
        "model": ARK_MODEL,
        "prompt": prompt,
        "size": "1024x1792",          # 183.5 万像素，落在 ¥0.30 档，不吃 0.60 溢价
        "response_format": "url",
        "watermark": False,
        "image": f"data:image/jpeg;base64,{b64}",
    }
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            r = httpx.post(ARK_BASE + "/images/generations",
                           headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                           json=body, timeout=httpx.Timeout(300.0, connect=60.0))
            if r.status_code != 200:
                raise RuntimeError(f"方舟出图 HTTP {r.status_code}: {r.text[:300]}")
            it = (r.json().get("data") or [{}])[0]
            if it.get("url"):
                out_path.write_bytes(httpx.get(it["url"], timeout=180).content)
            elif it.get("b64_json"):
                out_path.write_bytes(base64.b64decode(it["b64_json"]))
            else:
                raise RuntimeError("方舟返回既无 url 也无 b64")
            return out_path
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts:
                print(f"     第 {i} 次失败（{type(exc).__name__}: {str(exc)[:60]}），{5*i}s 后重试")
                time.sleep(5 * i)
    raise RuntimeError(f"方舟出图连续 {attempts} 次失败：{last}")


def prepare_ref(src: Path, out_path: Path, quality: int = 92) -> Path:
    """把参考图压成 JPEG 再上传。

    原图是 2.4 MB 的 PNG，走 multipart 上传时偶发被网关重置（见 relay_edit 注释）。
    参考图只用来提供人脸特征，1024 长边的 JPEG q92 完全够，体积能降到 1/5。
    """
    from PIL import Image

    im = Image.open(src).convert("RGB")
    im.save(out_path, "JPEG", quality=quality)
    return out_path


def relay_edit(prompt: str, out_path: Path, ref: Path, attempts: int = 3) -> Path:
    """走中转站 /images/edits，挂参考图保人脸。

    带重试：该中转站连通性间歇（同一时刻连发三次 GET /v1/models，
    结果可能是 ConnectError / 200 / 200），大请求更容易在等响应时被重置。
    连接层错误（RemoteProtocolError / ReadError / handshake timeout）重试即可。
    """
    last: Exception | None = None
    for i in range(1, attempts + 1):
        try:
            r = httpx.post(
                RELAY_BASE + "/images/edits",
                headers={"Authorization": f"Bearer {RELAY_KEY}"},
                files={"image": ("ref.jpg", ref.read_bytes(), "image/jpeg")},
                data={"model": RELAY_MODEL, "prompt": prompt, "size": "1024x1792", "n": "1"},
                timeout=httpx.Timeout(420.0, connect=90.0),
            )
            if r.status_code != 200:
                raise RuntimeError(f"出图失败 HTTP {r.status_code}: {r.text[:400]}")
            item = (r.json().get("data") or [{}])[0]
            if item.get("b64_json"):
                out_path.write_bytes(base64.b64decode(item["b64_json"]))
            elif item.get("url"):
                out_path.write_bytes(httpx.get(item["url"], timeout=180).content)
            else:
                raise RuntimeError("出图返回既无 b64 也无 url")
            return out_path
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i < attempts:
                print(f"     第 {i} 次失败（{type(exc).__name__}: {str(exc)[:60]}），{5*i}s 后重试")
                time.sleep(5 * i)
    raise RuntimeError(f"出图连续 {attempts} 次失败：{last}")


def make_video(first_frame: Path, prompt: str, out_mp4: Path, timeout_s: int = 420) -> dict:
    """创建 i2v 任务并轮询到终态，下载 mp4。"""
    body = {
        "model": VIDEO_MODEL,
        "input": {"prompt": prompt,
                  "img_url": f"data:image/jpeg;base64,{base64.b64encode(first_frame.read_bytes()).decode()}"},
        "parameters": {"resolution": "720P", "duration": 5,
                       "prompt_extend": False, "watermark": False},
    }
    hdr = {"Authorization": f"Bearer {DASHSCOPE_KEY}",
           "Content-Type": "application/json", "X-DashScope-Async": "enable"}
    r = httpx.post(CREATE, headers=hdr, json=body, timeout=180)
    j = r.json()
    task_id = (j.get("output") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"创建任务失败 HTTP {r.status_code}: {json.dumps(j, ensure_ascii=False)[:400]}")
    print(f"    task_id = {task_id}")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        time.sleep(10)
        try:
            q = httpx.get(TASK + task_id,
                          headers={"Authorization": f"Bearer {DASHSCOPE_KEY}"},
                          timeout=90).json()
        except Exception:  # noqa: BLE001  轮询中断不该让整个任务失败，下一轮再查
            continue
        status = (q.get("output") or {}).get("task_status", "?")
        if status == "SUCCEEDED":
            vurl = (q.get("output") or {}).get("video_url")
            out_mp4.write_bytes(httpx.get(vurl, timeout=300).content)
            print(f"    SUCCEEDED in {time.time()-t0:.0f}s → {out_mp4.name} "
                  f"({out_mp4.stat().st_size/1024:.0f} KB)")
            return q
        if status in ("FAILED", "CANCELED"):
            raise RuntimeError(f"任务终态 {status}: {json.dumps(q, ensure_ascii=False)[:500]}")
    raise RuntimeError(f"轮询超时 {timeout_s}s（task_id={task_id}，可手工查）")


def grab_last_frame(mp4: Path, out_jpg: Path) -> Path:
    """抽最后一帧。末帧是「漂移累积到最大」的时刻，比中段更能暴露问题。"""
    ff = config.find_bin("ffmpeg")
    dur = 5.0
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(max(0.0, dur - 0.3)), "-i", str(mp4),
                    "-frames:v", "1", "-q:v", "2", str(out_jpg)], check=True)
    return out_jpg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--only", default="", help="只跑某个景别：wide / closeup")
    ap.add_argument("--provider", default="ark", choices=["ark", "relay"],
                    help="出图供应商：ark=方舟 seedream（默认，走赠送额度）；relay=中转站 gpt-image")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    scenes = [s for s in SCENES if not args.only or s["key"] == args.only]
    est = len(scenes) * RELAY_PRICE
    print("=" * 74)
    print("15.2 三景别验证 · 补全景与近景")
    print("=" * 74)
    print(f"参考图    : {REF_IMAGE}")
    print(f"出图      : {ARK_MODEL if args.provider=='ark' else RELAY_MODEL} @ "
          f"{'方舟（赠送额度）' if args.provider=='ark' else f'中转站 ¥{RELAY_PRICE}/张'}")
    print(f"视频      : {VIDEO_MODEL} @ 百炼 720P 无声 ¥0.2/秒（走免费额度）")
    print(f"本次景别  : {[s['name'] for s in scenes]}")
    print(f"预计扣费  : ¥{est:.2f}（出图）+ 视频走免费额度")
    if not args.go:
        print("\n[dry-run] 未执行。加 --go 开跑。")
        return 0

    judge = VLMJudge(config.settings.judge_vlm)
    results = []

    ref_jpg = OUT / "ref.jpg"
    prepare_ref(REF_IMAGE, ref_jpg)
    print(f"\n参考图压缩 : {REF_IMAGE.name} ({REF_IMAGE.stat().st_size/1024:.0f} KB)"
          f" → {ref_jpg.name} ({ref_jpg.stat().st_size/1024:.0f} KB)")

    for i, sc in enumerate(scenes, 1):
        print(f"\n{'='*74}\n[{i}/{len(scenes)}] {sc['name']} ({sc['key']})\n{'='*74}")
        raw_png = OUT / f"{sc['key']}_raw.png"
        frame_jpg = OUT / f"{sc['key']}_firstframe.jpg"
        mp4 = OUT / f"{sc['key']}.mp4"
        last_jpg = OUT / f"{sc['key']}_lastframe.jpg"

        print("  ① 出图（挂定妆照参考）…")
        t = time.time()
        gen = ark_generate if args.provider == "ark" else relay_edit
        gen(sc["image_prompt"], raw_png, ref_jpg)
        print(f"     {raw_png.name}  {raw_png.stat().st_size/1024:.0f} KB  耗时 {time.time()-t:.0f}s")

        print("  ② 裁 9:16 → 720×1280")
        _, b64len = prepare_frame(raw_png, frame_jpg)
        print(f"     {frame_jpg.name}  {frame_jpg.stat().st_size/1024:.0f} KB  base64 {len(b64len)/1024:.0f} KB")

        print("  ③ i2v 生成…")
        make_video(frame_jpg, sc["video_prompt"], mp4)

        print("  ④ 抽末帧")
        grab_last_frame(mp4, last_jpg)
        print(f"     {last_jpg.name}")

        print("  ⑤ VLM 判定（末帧 vs 首帧）…")
        t = time.time()
        raw = judge.compare(last_jpg, frame_jpg, list(ELEMENTS), allow_paid=True)
        d = normalize(raw, max_retries=2)
        print(f"     score={d.score}  action={d.action.value}  ({time.time()-t:.0f}s)")
        for c in d.qa.checks:
            print(f"       {c.element}: {'PASS' if c.matched else 'FAIL'}")

        results.append({"key": sc["key"], "name": sc["name"], "score": d.score,
                        "action": d.action.value,
                        "fails": [c.element for c in d.qa.checks if not c.matched],
                        "diff": d.qa.diff or ""})

    print(f"\n{'='*74}\n汇总\n{'='*74}")
    print(f"{'景别':<8}{'分数':<8}{'结论':<10}{'未过要素'}")
    for r in results:
        print(f"{r['name']:<8}{r['score']:<8}{r['action']:<10}{','.join(r['fails']) or '-'}")
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细已存 {OUT/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
