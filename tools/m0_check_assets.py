#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0-S1 资产验收：30 张图 + labels.json 真值表。

只读校验，不改任何文件。只用标准库（不依赖 Pillow），
PNG/JPEG 尺寸通过直接读文件头解析。

用法：
    python scripts/m0_check_assets.py
    python scripts/m0_check_assets.py --verbose
"""

import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "tests" / "m0_eval"
PAIRS_DIR = EVAL_DIR / "pairs"
LABELS = EVAL_DIR / "labels.json"

ELEMENT_KEYS = ["发型", "脸型", "眉眼", "肤色", "体型"]


def image_size(path):
    """读 PNG / JPEG 文件头拿宽高，不引入 Pillow。"""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"\xff\xd8":
                f.seek(2)
                while True:
                    b = f.read(1)
                    while b and b != b"\xff":
                        b = f.read(1)
                    if not b:
                        return None
                    marker = f.read(1)
                    while marker == b"\xff":
                        marker = f.read(1)
                    if not marker:
                        return None
                    if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6,
                                     0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        f.read(3)
                        h, w = struct.unpack(">HH", f.read(4))
                        return int(w), int(h)
                    seg_len = struct.unpack(">H", f.read(2))[0]
                    f.seek(seg_len - 2, 1)
    except Exception:
        return None
    return None


def main():
    verbose = "--verbose" in sys.argv
    ok = True
    problems = []

    # --- 1. labels.json 结构 ---
    print("[1] labels.json")
    if not LABELS.exists():
        print("    FAIL 文件不存在")
        return 1
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    same = [x for x in labels if x["is_same_person"]]
    diff = [x for x in labels if not x["is_same_person"]]
    nulls = [x for x in labels
             if any(v is None for v in x.get("elements", {}).values())]
    missing_key = [x["pair_id"] for x in labels
                   if set(x.get("elements", {})) != set(ELEMENT_KEYS)]

    print(f"    条数 {len(labels)}  同人 {len(same)}  异人 {len(diff)}")
    if len(labels) != 30:
        problems.append(f"labels 条数应为 30，实际 {len(labels)}")
    if len(same) != 15 or len(diff) != 15:
        problems.append(f"同人/异人应各 15，实际 {len(same)}/{len(diff)}")
    if nulls:
        problems.append(f"{len(nulls)} 条 elements 含 null: "
                        f"{[x['pair_id'] for x in nulls]}")
    if missing_key:
        problems.append(f"{len(missing_key)} 条 elements 字段不全: {missing_key}")
    print(f"    未填 elements: {len(nulls)}  <- 硬验收要求 0")

    # --- 2. 图片齐全性 ---
    print("\n[2] pairs 目录")
    expected = [f"{i:02d}_{s}.png" for i in range(1, 16) for s in ("a", "b")]
    actual = sorted(p.name for p in PAIRS_DIR.glob("*") if p.is_file()) if PAIRS_DIR.exists() else []
    missing = [f for f in expected if f not in actual]
    extra = [f for f in actual if f not in expected]
    print(f"    到位 {len(expected) - len(missing)}/{len(expected)}")
    if missing:
        problems.append(f"缺 {len(missing)} 张: {missing}")
    if extra:
        problems.append(f"多余/命名不符 {len(extra)} 个: {extra}")

    # --- 3. 尺寸一致性 ---
    print("\n[3] 尺寸")
    sizes = {}
    unreadable = []
    for name in expected:
        p = PAIRS_DIR / name
        if not p.exists():
            continue
        s = image_size(p)
        if s is None:
            unreadable.append(name)
        else:
            sizes.setdefault(s, []).append(name)
    if unreadable:
        problems.append(f"无法解析宽高: {unreadable}")
    if sizes:
        for s, names in sizes.items():
            flag = "" if len(names) > 1 or len(sizes) == 1 else ""
            print(f"    {s[0]}x{s[1]}  共 {len(names)} 张{flag}")
        if len(sizes) > 1:
            detail = {f"{w}x{h}": len(v) for (w, h), v in sizes.items()}
            problems.append(f"尺寸不一致，出现 {len(sizes)} 种: {detail}")
        else:
            w, h = next(iter(sizes))
            if abs(w / h - 9 / 16) > 0.02:
                problems.append(f"比例 {w}:{h} 偏离 9:16")
        if verbose:
            for (w, h), names in sizes.items():
                print(f"      {w}x{h}: {', '.join(names)}")

    # --- 4. labels 引用的图片必须存在 ---
    print("\n[4] labels 与图片对应")
    dangling = []
    for item in labels:
        for side in ("a", "b"):
            f = f"{item[side]}.png"
            if f not in actual:
                dangling.append(f"{item['pair_id']}:{f}")
    if dangling:
        problems.append(f"{len(dangling)} 处引用缺图: {dangling}")
    else:
        print("    30 对引用的 30 张图全部存在")

    # --- 5. 结论 ---
    print("\n" + "=" * 46)
    if problems:
        print("M0-S1 未通过，共 %d 项问题：" % len(problems))
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        return 1
    print("M0-S1 通过：30 张图齐全、尺寸一致、labels 全部填好")
    print("可以进入 M0-S2（VLM 判定）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
