"""两个正式 provider 的契约自检（M16-S1）。全部离线，不发起任何网络请求。

验的是「provider 有没有守住 A.3 契约」和「几条护栏有没有生效」，
**不验模型效果**——效果由 15.2 的三景别验证负责（见 assets/probe/m152/）。

重点覆盖三件容易出错的事：
1. `is_mock` 必须为 False。猜错的代价是双向的（见 providers/base.py 注释）：
   真实 provider 被当 mock → 漏记费用、预算封顶失效。
2. `allow_paid=False` 必须抛 `PaidCallBlocked`。这是「付费默认关闭」（D7）的落点。
3. 首帧归一化必须真的把任意比例变成 720×1280。i2v 的输出比例跟随首帧，
   这一步错了，成片画幅就错了，而且是在花完钱之后才发现。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import config  # noqa: E402
from providers.base import PaidCallBlocked, ProviderError  # noqa: E402
from providers.dashscope_image import DashScopeImage, _encode_ref, _native_size  # noqa: E402
from providers.dashscope_video import FRAME_H, FRAME_W, DashScopeVideo  # noqa: E402
from providers.relay_image import RelayImage  # noqa: E402

PASS, FAIL = 0, 0
FAILED: list[str] = []


def ok(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f"  {extra}" if extra else ""))
    else:
        FAIL += 1
        FAILED.append(name)
        print(f"  FAIL  {name}  {extra}")


def raises(name: str, fn, exc_type, *, contains: str = "") -> None:
    try:
        fn()
    except exc_type as exc:
        hit = (contains in str(exc)) if contains else True
        ok(name, hit, f"抛 {exc_type.__name__}；{'含' if hit else '缺'}「{contains}」")
    except Exception as exc:  # noqa: BLE001
        ok(name, False, f"抛错类型不对：{type(exc).__name__}: {exc}")
    else:
        ok(name, False, "没有抛错")


def main() -> int:
    print("=" * 66)
    print("providers 契约自检")
    print("=" * 66)

    # ---------------- RelayImage ----------------
    print("\n[1] RelayImage · 出图")
    # 2026-11-02：`.env` 只保留 4 个在跑的模型，中转站那套改为注释停用（IMAGE_PROVIDER=dashscope）。
    # 这里仍然要验 RelayImage 的契约——它没被删，只是没被装配——所以缺配置时
    # 临时塞一份**假配置**把它构造出来。下面这几条断言全部在走到网络之前就抛错，
    # 用假 key 不会发出任何真实请求；真实 key 早就不需要了。
    # **不能直接 `RelayImage()`**：那会在 `.env` 干净的情况下当场 ProviderError，
    # 把整个脚本炸掉，看起来像"改个配置改坏了代码"。
    from config import ModelConfig as _MC

    origin_img_cfg = config.settings.image_api
    relay_from_env = config.settings.image_api.usable
    if not relay_from_env:
        object.__setattr__(config.settings, "image_api", _MC(
            api_key="placeholder-key", base_url="https://invalid.invalid/v1", model="placeholder"))
        print("      （中转站未启用，用占位配置验契约；所有断言都在发请求之前抛错）")
    try:
        img = RelayImage()
        ok("is_mock 为 False（防止被当占位实现漏记账）", img.is_mock is False, f"is_mock={img.is_mock}")
        ok("unit_price 与 config.price_image 一致",
           abs(img.unit_price - config.settings.price_image) < 1e-9,
           f"{img.unit_price}")
        ok("name 不含厂商名以外的硬编码", img.name == "relay-image", img.name)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "a.png"
            raises("不传 allow_paid → PaidCallBlocked", lambda: img.generate("x", out=out),
                   PaidCallBlocked, contains="allow_paid")
            raises("空 prompt → ProviderError", lambda: img.generate("   ", out=out, allow_paid=True),
                   ProviderError, contains="prompt")
            raises("参考图不存在 → ProviderError",
                   lambda: img.generate("x", out=out, references=[Path(td) / "nope.png"], allow_paid=True),
                   ProviderError, contains="参考图不存在")
    finally:
        if not relay_from_env:
            object.__setattr__(config.settings, "image_api", origin_img_cfg)

    # ---------------- DashScopeVideo ----------------
    print("\n[2] DashScopeVideo · 视频")
    vid = DashScopeVideo()
    ok("is_mock 为 False", vid.is_mock is False, f"is_mock={vid.is_mock}")
    ok("price_per_second 与 config 一致",
       abs(vid.price_per_second - config.settings.price_video_per_sec) < 1e-9,
       f"{vid.price_per_second}")
    ok("resolution 来自配置", vid.resolution == config.settings.video_resolution_level, vid.resolution)
    ok("支持的时长档位为 (5,)", vid.SUPPORTED_DURATIONS == (5,), str(vid.SUPPORTED_DURATIONS))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 造一张 1000×1000 的正方形首帧，验证归一化真的会裁
        sq = td / "square.png"
        Image.new("RGB", (1000, 1000), (90, 80, 70)).save(sq)
        out = td / "v.mp4"

        raises("不传 allow_paid → PaidCallBlocked",
               lambda: vid.generate(sq, "x", duration=5, out=out), PaidCallBlocked, contains="allow_paid")
        raises("首帧不存在 → ProviderError",
               lambda: vid.generate(td / "nope.png", "x", duration=5, out=out, allow_paid=True),
               ProviderError, contains="首帧不存在")
        raises("空 prompt → ProviderError",
               lambda: vid.generate(sq, "  ", duration=5, out=out, allow_paid=True),
               ProviderError, contains="prompt")
        raises("duration=10 → ProviderError 并提示换模型",
               lambda: vid.generate(sq, "x", duration=10, out=out, allow_paid=True),
               ProviderError, contains="VIDEO_MODEL")

        # 归一化：正方形 1000×1000 → 必须变成 720×1280
        norm = vid._normalize_frame(sq, td / "norm.jpg")
        with Image.open(norm) as im:
            size = im.size
        ok("首帧归一化到 720×1280（i2v 输出比例跟随首帧）",
           size == (FRAME_W, FRAME_H), f"{size[0]}x{size[1]}")

        # 竖长图（比 9:16 还窄）→ 裁上下，不能拉伸
        tall = td / "tall.png"
        Image.new("RGB", (600, 1600), (30, 60, 90)).save(tall)
        n2 = vid._normalize_frame(tall, td / "norm2.jpg")
        with Image.open(n2) as im:
            s2 = im.size
        ok("竖长图也归一化到 720×1280（裁切而非拉伸）",
           s2 == (FRAME_W, FRAME_H), f"{s2[0]}x{s2[1]}")

    # ---------------- DashScopeImage ----------------
    # 2026-10-29 新增：中转站整站故障后补的第二条出图路径。
    # 这里只验契约与两个易错的数据转换，不发起真实生成。
    print("\n[3] DashScopeImage · 出图（原生多模态生成）")
    dimg = DashScopeImage()
    ok("is_mock 为 False", dimg.is_mock is False, f"is_mock={dimg.is_mock}")
    ok("unit_price 与 config.price_image 一致",
       abs(dimg.unit_price - config.settings.price_image) < 1e-9, f"{dimg.unit_price}")
    ok("name 固定为 dashscope-image", dimg.name == "dashscope-image", dimg.name)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out = td / "a.png"
        raises("不传 allow_paid → PaidCallBlocked",
               lambda: dimg.generate("x", out=out), PaidCallBlocked, contains="allow_paid")
        raises("空 prompt → ProviderError",
               lambda: dimg.generate("   ", out=out, allow_paid=True), ProviderError, contains="prompt")
        raises("参考图不存在 → ProviderError",
               lambda: dimg.generate("x", out=out, references=[td / "nope.png"], allow_paid=True),
               ProviderError, contains="参考图不存在")

        # 契约用 WxH，DashScope 原生要 W*H。转换错了就是 400，却要花一次往返才发现。
        ok("尺寸写法转换 1024x1792 → 1024*1792",
           _native_size("1024x1792") == "1024*1792", _native_size("1024x1792"))
        ok("尺寸转换对多档位都成立",
           _native_size("1024x1536") == "1024*1536" and _native_size("720X1280") == "720*1280")

        # 参考图编码：必须是 data URL，且压过（大图不许原样塞进 JSON body）
        big = td / "big.png"
        Image.new("RGB", (2400, 4200), (120, 30, 30)).save(big)
        data_url = _encode_ref(big)
        ok("参考图编码成 data:image/jpeg;base64", data_url.startswith("data:image/jpeg;base64,"),
           data_url[:32] + "...")
        import base64 as _b64

        decoded = _b64.b64decode(data_url.split(",", 1)[1])
        ok("参考图被压小（2400×4200 PNG → JPEG ≤ 400KB）", len(decoded) <= 400 * 1024,
           f"{len(decoded) // 1024}KB / 原 {big.stat().st_size // 1024}KB")
        import io as _io

        with Image.open(_io.BytesIO(decoded)) as im2:
            long_side = max(im2.size)
        ok("参考图长边被限到 1200", long_side == 1200, str(long_side))

        # ---------------- 参考图短边下限（2026-10-31 新增）----------------
        # 平台校验「最短边 ≥ 512」，实测 121×246 的图直接 400：
        #   InvalidParameter: Error validating image ... resolution must ...
        # 而且报错里只有一个 OSS 签名 URL，看不出是哪张图、多大。
        # 这几条断言守的就是"别再让小图原样发出去"。
        from providers.dashscope_image import REF_MIN_SIDE as _MIN  # noqa: PLC0415
        from providers.dashscope_image import _prepare_ref  # noqa: PLC0415

        small = td / "small.png"
        Image.new("RGB", (121, 246), (200, 180, 170)).save(small)
        _, info_small = _prepare_ref(small)
        ok("小图被标记为放大（否则就是原样发出去等 400）",
           bool(info_small.get("upscaled")), str(info_small))
        ok(f"放大后最短边 ≥ {_MIN}",
           min(int(info_small["out_w"]), int(info_small["out_h"])) >= _MIN, str(info_small))

        _, info_big = _prepare_ref(big)
        ok("大图不放大，只压体积（放大只会白拖慢上传）",
           not info_big.get("upscaled")
           and max(int(info_big["out_w"]), int(info_big["out_h"])) == 1200, str(info_big))

        # 极端宽高比：放到长边上限也凑不出短边。这时**必须提前报错**，
        # 而不是把 41×2048 发出去，让平台回一条看不出原因的 400。
        strip = td / "strip.png"
        Image.new("RGB", (100, 5000), (10, 10, 10)).save(strip)
        raises("极端宽高比提前报错（不硬发）",
               lambda: _prepare_ref(strip), ProviderError, contains="宽高比")

        # 告警本身绝不能抛（2026-10-31 实测事故）：服务用 `| head -40` 启动，
        # head 读满退出后管道关闭，线程里 print 到 stderr 抛
        # `OSError: [Errno 22] Invalid argument`，一次付费出图被一行告警打断。
        from providers.dashscope_image import _warn  # noqa: PLC0415

        class _BrokenStderr:
            def write(self, *_a):
                raise OSError(22, "Invalid argument")

            def flush(self):
                raise OSError(22, "Invalid argument")

        _orig = sys.stderr
        sys.stderr = _BrokenStderr()  # type: ignore[assignment]
        try:
            _warn("测试：stderr 写不进去")
            warn_ok, warn_detail = True, ""
        except Exception as exc:  # noqa: BLE001
            warn_ok, warn_detail = False, repr(exc)
        finally:
            sys.stderr = _orig
        ok("stderr 不可写时告警不抛（诊断信息不许中断付费流程）", warn_ok, warn_detail)

    # 响应解析：content 数组里先文本后图片也要能取到图（实测结构并不保证顺序）
    ok("能从 content 数组里取到图片 URL（图片不在第一位）",
       DashScopeImage._extract_url({"output": {"choices": [{"message": {"content": [
           {"type": "text", "text": "here"}, {"type": "image", "image": "u"}]}}]}}) == "u")
    ok("content 里没有图片项时返回空串",
       DashScopeImage._extract_url({"output": {"choices": [{"message": {"content": [
           {"type": "text", "text": "x"}]}}]}}) == "")
    ok("空响应不炸，返回空串", DashScopeImage._extract_url({}) == "")

    # ---------------- 配置缺失时的行为 ----------------
    print("\n[4] 配置缺失")
    origin_img = config.settings.image_api
    origin_native = config.settings.image_native_api
    origin_vid = config.settings.video_api
    try:
        from config import ModelConfig

        object.__setattr__(config.settings, "image_api", ModelConfig(api_key="", base_url="", model=""))
        raises("image_api 空 → 实例化报错并指明缺哪三个变量",
               RelayImage, ProviderError, contains="IMAGE_API_KEY")

        object.__setattr__(config.settings, "image_native_api",
                           ModelConfig(api_key="", base_url="", model=""))
        raises("image_native_api 空 → 实例化报错并指明缺哪三个变量",
               DashScopeImage, ProviderError, contains="IMAGE_NATIVE_API_KEY")

        object.__setattr__(config.settings, "video_api", ModelConfig(api_key="", base_url="", model=""))
        raises("video_api 空 → 实例化报错并指明缺哪三个变量",
               DashScopeVideo, ProviderError, contains="VIDEO_API_KEY")
    finally:
        object.__setattr__(config.settings, "image_api", origin_img)
        object.__setattr__(config.settings, "image_native_api", origin_native)
        object.__setattr__(config.settings, "video_api", origin_vid)

    print("\n" + "=" * 66)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    if FAILED:
        print("失败项：")
        for n in FAILED:
            print("  -", n)
    print("=" * 66)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
