#!/usr/bin/env python
"""一键跑全部离线断言：**零花费、断网可跑**。

用法：
    python tests/run_all.py              # 全部离线项（约 2 分钟）
    python tests/run_all.py -k phase     # 只跑文件名含 phase 的脚本
    python tests/run_all.py --list       # 只列会跑哪些

为什么不是 pytest：这套断言是**按阶段组织的独立脚本**，每个都能单独跑、
自带用例计数、不依赖 fixture 机制。转成 pytest 要重写全部 500 多项，
而收益只是"能用 pytest 命令"，不划算。所以 `requirements.txt` 里没有 pytest。

每个脚本自己打印 `结果：N 通过 / M 失败` 并以退出码表示成败；本文件只做汇总。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULT = re.compile(r"(\d+)\s*通过\s*/\s*(\d+)\s*失败")
ONLY_NUM = re.compile(r"(\d+)\s*通过")


def discover(pattern: str) -> list[Path]:
    files = sorted(HERE.glob("verify_*.py"))
    if pattern:
        files = [f for f in files if pattern in f.name]
    return files


def run_one(script: Path) -> tuple[int, int, bool, float]:
    """跑一个脚本，返回 (通过数, 失败数, 是否成功, 秒)。"""
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,                 # 断言脚本用 parents[1] 定位项目根，这里保持一致
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
    )
    cost = time.time() - t0
    out = proc.stdout + proc.stderr

    ok = n = fail = 0
    hits = RESULT.findall(out)
    if hits:
        # 大多数脚本结尾是「结果：N 通过 / M 失败」
        n, fail = (int(x) for x in hits[-1])
        ok = n - fail
    elif "PASS" in out or "FAIL" in out:
        # 少数脚本（verify_models）只逐条打 PASS/FAIL，结尾给「全部通过」不给总数。
        # 这时按行数点，比"算 1 项"准确。
        ok = sum(1 for line in out.splitlines() if line.strip().startswith("PASS"))
        fail = sum(1 for line in out.splitlines() if line.strip().startswith("FAIL"))
        n = ok + fail
    elif proc.returncode == 0:
        ok = n = 1          # 没有计数行但退出码 0：算 1 项通过
    else:
        fail = 1

    return ok, fail, proc.returncode == 0 and fail == 0, cost


def run_ui_check() -> tuple[bool, str]:
    """前端结构检查。它不是 verify_*，没有计数行，只能看输出里的告警标记。"""
    script = HERE / "check_ui.py"
    if not script.exists():
        return True, "跳过（无 check_ui.py）"
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
    )
    out = proc.stdout + proc.stderr
    bad = []
    if "★不配对★" in out:
        bad.append("标签不配对")
    if re.search(r"未定义函数:\s*\[", out):
        bad.append("存在未定义函数")
    if re.search(r"引用了但不存在的 id:\s*\[", out):
        bad.append("存在孤儿 id")
    return (not bad), ("、".join(bad) if bad else "无孤儿 id、无未定义函数")


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("-k", default="", help="只跑文件名含该片段的脚本")
    ap.add_argument("--list", action="store_true", help="只列出脚本，不执行")
    args = ap.parse_args()

    files = discover(args.k)
    if not files:
        print("没有匹配的脚本。")
        return 1
    if args.list:
        for f in files:
            print(" ", f.name)
        print(f"\n共 {len(files)} 个。")
        return 0

    print("=" * 72)
    print(f"离线断言 · {len(files)} 个脚本 · 不联网、零花费")
    print("=" * 72)

    rows, total_ok, total_fail, failed = [], 0, 0, []
    for f in files:
        try:
            ok, fail, good, cost = run_one(f)
        except subprocess.TimeoutExpired:
            ok, fail, good, cost = 0, 1, False, 1800.0
            print(f"  超时  {f.name}")
        rows.append((f.name, ok, fail, cost, good))
        total_ok += ok
        total_fail += fail
        if not good:
            failed.append(f.name)
        mark = "通过" if good else "失败"
        print(f"  {mark}  {f.name:26} {ok:4} 项  {cost:6.1f}s")

    ui_ok, ui_msg = run_ui_check()
    if not ui_ok:
        failed.append("check_ui.py")
    print(f"  {'通过' if ui_ok else '失败'}  {'check_ui.py':26} {'':4}     {ui_msg}")

    print("-" * 72)
    print(f"合计 {total_ok} 项通过 / {total_fail} 项失败"
          f"（{len(rows)} 个脚本 + 前端结构检查）")
    if failed:
        print("失败：", "、".join(failed))
        print("单独复现：python tests/" + failed[0])
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
