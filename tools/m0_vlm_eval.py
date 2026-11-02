#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M0-S2：对 30 对图跑 VLM 判定，落下原始结果。

判定方式与生产一致（15.1 第三节）：
- 参考侧 = a 图（定妆照 / 角色正面照）
- 待判侧 = b 图（异人组是另一角色的正面照，同人组是同角色的不同视角照）
- 要素列表 = 固定 5 项：发型、脸型、眉眼、肤色、体型
- 输出 = score + checks[]（逐要素）+ diff

**这个脚本不读 labels.json 的真值来构造任何提示词**——真值只用于 M0-S3 事后比对。
如果判定时能看见真值，准确率就是假的。

用法：
    python scripts/m0_vlm_eval.py --dry-run          # 只构造请求、不发网络调用
    python scripts/m0_vlm_eval.py --allow-paid        # 真跑（花钱，需授权）
    python scripts/m0_vlm_eval.py --allow-paid --limit 5
    python scripts/m0_vlm_eval.py --allow-paid --swap  # 消融：交换两图位置

产出：
    tests/m0_eval/raw_results.json    30 条原始判定结果
"""

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from providers.vlm import ELEMENT_HINTS, JUDGE_VERSION, VLMJudge  # noqa: E402

EVAL_DIR = ROOT / "tests" / "m0_eval"
PAIRS_DIR = EVAL_DIR / "pairs"
LABELS = EVAL_DIR / "labels.json"
RAW_RESULTS = EVAL_DIR / "raw_results.json"

ELEMENTS = list(ELEMENT_HINTS.keys())


def load_pairs():
    """从 labels.json 取「哪两张图组成一对」。

    只取 a / b / pair_id 三个字段，**不取 is_same_person 和 elements 真值**。
    """
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    return [{"pair_id": x["pair_id"], "a": x["a"], "b": x["b"]} for x in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-paid", action="store_true",
                    help="确认授权付费调用。不带这个开关一律只做 dry-run。")
    ap.add_argument("--dry-run", action="store_true",
                    help="只构造请求并打印，不发网络调用（默认行为）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 对，0 = 全跑")
    ap.add_argument("--swap", action="store_true",
                    help="交换参考图与待判图的位置，用于检验位置偏好")
    ap.add_argument("--resume", action="store_true",
                    help="跳过 raw_results.json 里已有的 pair_id（断点续跑）")
    ap.add_argument("--workers", type=int, default=0,
                    help="并发数，0 = 用 .env 的 VLM_CONCURRENCY（默认 4）")
    ap.add_argument("--no-thinking", action="store_true",
                    help="关闭模型深度思考。默认开启（实测关掉会产生假阴性，不建议）")
    args = ap.parse_args()

    dry_run = args.dry_run or not args.allow_paid

    pairs = load_pairs()
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"待跑对数：{len(pairs)}  要素：{ELEMENTS}")

    # --- 干跑：只验证资产与请求构造 ---
    if dry_run:
        from providers.vlm import build_judge_messages
        missing = []
        for p in pairs:
            for side in ("a", "b"):
                f = PAIRS_DIR / f"{p[side]}.png"
                if not f.exists():
                    missing.append(str(f))
        if missing:
            print(f"FAIL 缺图 {len(missing)} 个：{missing[:5]}")
            return 1

        sample = pairs[0]
        msgs, meta = build_judge_messages(
            PAIRS_DIR / f"{sample['a']}.png",
            PAIRS_DIR / f"{sample['b']}.png",
            ELEMENTS,
            swap=args.swap,
        )
        user_content = msgs[1]["content"]
        text_block = next(c for c in user_content if c["type"] == "text")
        img_blocks = [c for c in user_content if c["type"] == "image_url"]
        print(f"\n[干跑] 样本 {sample['pair_id']}")
        print(f"  system prompt 长度：{len(msgs[0]['content'])} 字符")
        print(f"  user text：{text_block['text']}")
        print(f"  图片块数：{len(img_blocks)}")
        print(f"  参考图尺寸：{meta['reference']['width']}x{meta['reference']['height']}"
              f"  待判图尺寸：{meta['image']['width']}x{meta['image']['height']}")
        print(f"  请求体估算：{meta['payload_bytes'] / 1048576:.2f} MB")
        print(f"  judge 版本：{JUDGE_VERSION}  swap={args.swap}")
        print("\n干跑通过：资产齐全，请求可构造。去掉 --dry-run 并加 --allow-paid 才会真的花钱。")
        return 0

    # --- 真跑 ---
    if not settings.judge_vlm.usable:
        print("FAIL 未配置 VLM 凭据。请在项目根目录 .env 里设置：")
        print("     VLM_API_KEY=<你的 key>")
        print("     VLM_BASE_URL=<OpenAI 兼容端点>")
        print("     VLM_MODEL=<模型名>")
        return 2

    judge = VLMJudge(settings.judge_vlm)
    workers = args.workers or settings.judge_vlm_concurrency
    thinking = settings.judge_vlm_thinking and not args.no_thinking
    print(f"VLM：model={settings.judge_vlm.model} base_url={settings.judge_vlm.base_url}")
    print(f"并发={workers}  深度思考={'开' if thinking else '关'}")

    done: set[str] = set()
    existing: list[dict] = []
    if args.resume and RAW_RESULTS.exists():
        existing = json.loads(RAW_RESULTS.read_text(encoding="utf-8"))
        done = {r["pair_id"] for r in existing}
        print(f"断点续跑：已有 {len(done)} 条结果")

    results = list(existing)
    ok = fail = 0
    t0 = time.time()
    write_lock = threading.Lock()
    tok_in = 0
    tok_out = 0

    def judge_one(p: dict) -> dict:
        """判定一对图。异常在调用方捕获。"""
        img = PAIRS_DIR / f"{p['b']}.png"
        ref = PAIRS_DIR / f"{p['a']}.png"
        r = judge.compare(
            img, ref, ELEMENTS,
            allow_paid=True,
            swap=args.swap,
            enable_thinking=thinking,
        )
        return {
            "pair_id": p["pair_id"],
            "reference": f"{p['a']}.png",
            "image": f"{p['b']}.png",
            **r,
        }

    todo = [p for p in pairs if p["pair_id"] not in done]
    if not todo:
        print("没有待跑的对（配合 --resume 使用）。")
        return 0

    def save_locked() -> None:
        """每跑完一对就落盘：中断时已完成的部分不丢（每次调用都要花钱，不能白跑）。"""
        with write_lock:
            RAW_RESULTS.write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print(f"开始跑 {len(todo)} 对...\n")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(judge_one, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            p = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:
                fail += 1
                print(f"  [{i:>2}/{len(todo)}] {p['pair_id']} FAIL {exc}")
                continue
            with write_lock:
                results.append(r)
                tok_in += r.get("usage", {}).get("prompt_tokens", 0)
                tok_out += r.get("usage", {}).get("completion_tokens", 0)
            ok += 1
            save_locked()
            print(f"  [{i:>2}/{len(todo)}] {p['pair_id']} score={r['score']:>3} "
                  f"matched={sum(1 for c in r['checks'] if c['matched'] is True)}/5 "
                  f"({time.time() - t0:.0f}s)")

    # 汇总：按 pair_id 排序落盘，保证同一份输入每次跑出的文件顺序一致
    results.sort(key=lambda r: r["pair_id"])
    save_locked()

    elapsed = time.time() - t0
    print(f"\n完成：成功 {ok}  失败 {fail}  累计 {len(results)} 条 -> {RAW_RESULTS}")
    print(f"耗时 {elapsed:.0f}s（并发 {workers}）")
    print(f"token 累计：输入 {tok_in}  输出 {tok_out}")
    cost = (tok_in + tok_out) / 1_000_000 * 3.0  # 粗估：按 3 元/百万 token 混合价
    print(f"成本粗估（按 3 元/百万 token 混合价）：约 ¥{cost:.2f}")
    if ok:
        print("下一步：M0-S3 跑 scripts/m0_calc.py 算两条分布 + FP/FN + 门槛达成。")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
