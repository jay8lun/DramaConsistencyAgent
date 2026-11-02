#!/usr/bin/env python
"""补跑单对判定（绕过 _chat 的多参数组重试循环）。

背景：D13 用正常流程跑了 11 分钟仍未落盘。原因是 `_chat` 会对失败请求依次尝试
4 组参数（json_object+temp0 → temp0 → temp0.7 → 最小参数集），每组开深度思考约
150 秒，全部失败就是 10 分钟。本脚本只发**一次**请求，把原始响应/错误直接打出来，
便于判断到底是「返回空内容」「JSON 不合法」还是「超时」。

用法：
    python scripts/rejudge_one.py D13
    python scripts/rejudge_one.py D13 --model doubao-seed-2-1-pro-260628
    python scripts/rejudge_one.py D13 --no-thinking
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from providers.vlm import (  # noqa: E402
    ELEMENT_HINTS,
    _extract_json,
    _normalize,
    build_judge_messages,
)

PAIRS = ROOT / "tests" / "m0_eval" / "pairs"
LABELS = ROOT / "tests" / "m0_eval" / "labels.json"
RAW = ROOT / "tests" / "m0_eval" / "raw_results.json"
ELEMENTS = list(ELEMENT_HINTS.keys())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pair_id")
    ap.add_argument("--model", default="", help="覆盖 .env 里的模型名")
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--save", action="store_true", help="成功则合并进 raw_results.json")
    args = ap.parse_args()

    labels = {x["pair_id"]: x for x in json.loads(LABELS.read_text(encoding="utf-8"))}
    lab = labels[args.pair_id]
    model = args.model or settings.judge_vlm.model

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.judge_vlm.api_key,
        base_url=settings.judge_vlm.base_url,
        timeout=420.0,
    )
    msgs, meta = build_judge_messages(
        PAIRS / f"{lab['a']}.png", PAIRS / f"{lab['b']}.png", ELEMENTS
    )
    print(f"pair={args.pair_id}  {lab['a']} vs {lab['b']}")
    print(f"model={model}  thinking={'off' if args.no_thinking else 'on'}  "
          f"请求体 {meta['payload_bytes'] / 1048576:.2f} MB")

    kwargs = dict(model=model, messages=msgs, max_tokens=9000 if not args.no_thinking else 1200,
                  temperature=0.0, response_format={"type": "json_object"})
    if args.no_thinking:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAIL {time.time() - t0:.0f}s  {type(exc).__name__}")
        print(str(exc)[:800])
        return 1

    content = resp.choices[0].message.content or ""
    print(f"\nOK {time.time() - t0:.0f}s  finish={resp.choices[0].finish_reason}  "
          f"usage={resp.usage}")
    print(f"content 长度={len(content)}")
    print(f"content 头部: {content[:300]!r}")

    try:
        obj = _extract_json(content)
    except Exception as exc:  # noqa: BLE001
        print(f"\nJSON 解析失败: {exc}")
        print(f"完整内容:\n{content}")
        return 1

    usage = {
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        "total_tokens": getattr(resp.usage, "total_tokens", 0),
        "request_id": getattr(resp, "id", "") or "",
        "response_model": getattr(resp, "model", "") or "",
    }
    result = _normalize(obj, ELEMENTS, model, usage)
    result.update({
        "pair_id": args.pair_id,
        "reference": f"{lab['a']}.png",
        "image": f"{lab['b']}.png",
        "payload_bytes": meta["payload_bytes"],
        "reference_size": (meta["reference"]["width"], meta["reference"]["height"]),
        "image_size": (meta["image"]["width"], meta["image"]["height"]),
        "swapped": False,
        "thinking": not args.no_thinking,
    })

    matched = sum(1 for c in result["checks"] if c["matched"] is True)
    print(f"\nscore={result['score']}  matched={matched}/{len(ELEMENTS)}")
    print(f"diff: {result['diff']}")
    for c in result["checks"]:
        print(f"  {c['element']}: {'✓' if c['matched'] else '✗'}  "
              f"{c['expected'][:36]} | {c['actual'][:36]}")

    if args.save:
        data = json.loads(RAW.read_text(encoding="utf-8"))
        data = [x for x in data if x["pair_id"] != args.pair_id]
        data.append(result)
        data.sort(key=lambda x: x["pair_id"])
        RAW.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已合并进 {RAW.name}，现有 {len(data)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
