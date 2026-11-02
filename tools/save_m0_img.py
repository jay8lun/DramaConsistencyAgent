"""M0-S1 出图落盘工具（纯标准库，不依赖 Pillow）。

用法：
    # 自动取「下载」目录里最新的图片，落盘为 01_a.png
    python scripts/save_m0_img.py 01_a

    # 指定源文件
    python scripts/save_m0_img.py 01_b --from "C:/Users/xxx/Downloads/jimeng-123.png"

    # 覆盖已存在的目标
    python scripts/save_m0_img.py 01_b --from ... --force

    # 查看 30 张图的到位情况与尺寸一致性
    python scripts/save_m0_img.py --status
"""

import argparse
import re
import struct
import sys
from pathlib import Path

# pairs/ 与本脚本同级目录的兄弟：tests/m0_eval/pairs
PAIRS_DIR = Path(__file__).resolve().parent.parent / "tests" / "m0_eval" / "pairs"
ID_PATTERN = re.compile(r"^(\d{2})_([ab])$")
EXPECTED_IDS = [f"{i:02d}_{s}" for i in range(1, 16) for s in ("a", "b")]
# 不硬编码目标像素：只要 9:16 且 30 张一致即可。以下几档是见过的值，仅用于提示。
KNOWN_SIZES = {(1080, 1920), (1440, 2560), (1024, 1792)}


def image_size(path: Path):
    """读宽高：PNG 读 IHDR，JPEG 扫 SOF 段。失败返回 None。"""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return w, h
            f.seek(0)
            data = f.read()
    except OSError:
        return None

    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(data)
    while i < n - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h, w = struct.unpack(">HH", data[i + 5 : i + 9])
            return w, h
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg_len = struct.unpack(">H", data[i + 2 : i + 4])[0]
        i += 2 + seg_len
    return None


def download_dirs():
    """候选下载目录，返回存在的那些。D:/jay 是本机实测在用的浏览器下载目录。"""
    home = Path.home()
    cands = (
        Path("D:/jay"),
        home / "Downloads",
        home / "OneDrive" / "Downloads",
        Path("D:/Downloads"),
        Path("D:/下载"),
    )
    seen, out = set(), []
    for c in cands:
        try:
            if c.is_dir() and c.resolve() not in seen:
                seen.add(c.resolve())
                out.append(c)
        except OSError:
            continue
    return out


def find_downloads():
    """兼容旧调用：返回第一个可用的下载目录。"""
    dirs = download_dirs()
    return dirs[0] if dirs else None


def latest_image():
    """跨所有候选目录找 mtime 最新的图片。"""
    exts = {".png", ".jpg", ".jpeg"}
    files = []
    for d in download_dirs():
        try:
            files += [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts]
        except OSError:
            continue
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def check_ratio(w, h):
    """返回宽高比提示。9:16 = 0.5625，容差 1%。"""
    ratio = w / h
    if abs(ratio - 9 / 16) / (9 / 16) <= 0.01:
        return True, f"{w}x{h}（9:16，正解）"
    return False, f"{w}x{h}（比例 {ratio:.3f}，不是 9:16！）"


def do_save(args):
    img_id = args.id
    if not ID_PATTERN.match(img_id):
        print(f"[错误] id 必须形如 01_a / 15_b，收到：{img_id}")
        return 1

    src = Path(args.source) if args.source else latest_image()
    if src is None:
        print("[错误] 找不到源文件。用 --from 指定，或检查下载目录。")
        return 1
    if not src.is_file():
        print(f"[错误] 源文件不存在：{src}")
        return 1

    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    dst = PAIRS_DIR / f"{img_id}.png"

    if dst.exists() and not args.force:
        print(f"[错误] 目标已存在：{dst.name}。要覆盖加 --force")
        return 1

    # 先复制成 .png：即梦可能下 jpg，内容不变，只是改名
    data = src.read_bytes()
    dst.write_bytes(data)

    size = image_size(dst)
    print(f"源   {src.name}")
    print(f"落盘 {dst}")
    if size:
        ok, msg = check_ratio(*size)
        print(f"尺寸 {msg}")
        if not ok:
            print("     -> 这张图比例不对，建议重出（30 张必须同一比例）")
        elif size not in KNOWN_SIZES:
            print("     -> 这个档位之前没见过，只要后面 29 张都用同一尺寸就没问题")
    else:
        print("尺寸 无法解析（可能不是 png/jpg）")
    print("提示 出完一组后用 --status 检查 30 张尺寸是否一致")
    return 0


def do_status(_args):
    if not PAIRS_DIR.is_dir():
        print(f"[错误] 目录不存在：{PAIRS_DIR}")
        return 1

    present, missing, sizes = 0, [], {}
    for img_id in EXPECTED_IDS:
        p = PAIRS_DIR / f"{img_id}.png"
        if not p.exists():
            missing.append(img_id)
            continue
        present += 1
        size = image_size(p)
        sizes[img_id] = size

    print(f"目录 {PAIRS_DIR}")
    print(f"到位 {present}/30")
    if missing:
        print(f"缺   {' '.join(missing)}")

    uniq = {}
    for img_id, size in sizes.items():
        uniq.setdefault(size, []).append(img_id)
    if len(uniq) > 1:
        print("尺寸不一致：")
        for size, ids in uniq.items():
            label = "x".join(map(str, size)) if size else "无法解析"
            print(f"  {label}: {' '.join(ids)}")
    elif uniq:
        size = next(iter(uniq))
        label = "x".join(map(str, size)) if size else "无法解析"
        print(f"尺寸 全部一致：{label}")

    if present == 30:
        print("30 张齐了，可以去填 labels.json 的 elements 了")
    return 0


def main():
    ap = argparse.ArgumentParser(description="M0-S1 出图落盘工具")
    ap.add_argument("id", nargs="?", help="形如 01_a / 15_b")
    ap.add_argument("--from", dest="source", help="源文件路径，省略则取下载目录最新图片")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的目标文件")
    ap.add_argument("--status", action="store_true", help="查看 30 张到位情况")
    args = ap.parse_args()

    if args.status:
        return do_status(args)
    if not args.id:
        ap.print_help()
        return 1
    return do_save(args)


if __name__ == "__main__":
    sys.exit(main())
