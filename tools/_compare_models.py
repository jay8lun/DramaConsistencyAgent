#!/usr/bin/env python
"""临时工具：验证替代判定模型（turbo）的质量是否与原模型（pro）可比。

背景：doubao-seed-2-1-pro 在 M0-S2 跑到 23/30 时命中 429 SetLimitExceeded。
方舟的报错提示"去关闭安心体验模式"是通用话术，实际是 per-model 免费额度耗尽。
本脚本拿已判定过的 pair（pro 的结果已在 raw_results.json 里）重跑一遍，
逐条对比 score / matched / diff，判断能不能换模型继续。

用法：python scripts/_compare_models.py
"""

import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from providers.vlm import (  # noqa: E402
    ELEMENT_HINTS,
    JUDGE_VERSION,
    _SYSTEM_PROMPT,
    _extract_json,
    _normalize,
)

PAIRS = ROOT / "tests" / "m0_eval" / "pairs"
LABELS = ROOT / "tests" / "m0_eval" / "labels.json"
RAW = ROOT / "tests" / "m0_eval" / "raw_results.json"

ELEMENTS = list(ELEMENT_HINTS.keys())
CANDIDATE = "doubao-seed-2-1-turbo-260628"

# 抽样：1 对同人 + 2 对异人（其中一对是"脸型与体型"型差异，另一对是多要素差异）
SAMPLE = ["S01", "D01", "D12"]


def data_url(path: Path) -> str:
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b}"


def main() -> int:
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.judge_vlm.api_key,
        base_url=settings.judge_vlm.base_url,
        timeout=300.0,
    )
    labels = {x["pair_id"]: x for x in json.loads(LABELS.read_text(encoding="utf-8"))}
    pro = {x["pair_id"]: x for x in json.loads(RAW.read_text(encoding="utf-8"))}

    sys_prompt = _SYSTEM_PROMPT.format(
        element_hints="\n".join(f"- {e}：{ELEMENT_HINTS[e]}" for e in ELEMENTS)
    )

    print(f"候选模型：{CANDIDATE}")
    print(f"对照基准：{next(iter(pro.values()))['model']}（judge {JUDGE_VERSION}）\n")

    agree_score = 0
    agree_matched = 0
    n = 0
    for pid in SAMPLE:
        lab = labels[pid]
        msgs = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图1是参考定妆照，图2是待判画面。"
                                             "请逐项比对上述五项要素，按系统提示的 JSON 格式输出。"},
                    {"type": "image_url", "image_url": {"url": data_url(PAIRS / f"{lab['a']}.png")}},
                    {"type": "image_url", "image_url": {"url": data_url(PAIRS / f"{lab['b']}.png")}},
                ],
            },
        ]
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=CANDIDATE,
                messages=msgs,
                max_tokens=9000,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            print(f"  {pid}: FAIL {str(exc)[:200]}")
            continue

        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        }
        out = _normalize(
            _extract_json(resp.choices[0].message.content or ""),
            ELEMENTS,
            CANDIDATE,
            usage,
        )
        m_new = sum(1 for c in out["checks"] if c["matched"] is True)
        p = pro[pid]
        m_old = sum(1 for c in p["checks"] if c["matched"] is True)
        n += 1
        if abs(out["score"] - p["score"]) <= 10:
            agree_score += 1
        if m_new == m_old:
            agree_matched += 1
        print(f"  {pid}: turbo score={out['score']:>3} matched={m_new}/5 "
              f"({time.time() - t0:.0f}s tok {usage['prompt_tokens']}/{usage['completion_tokens']})"
              f"  |  pro score={p['score']:>3} matched={m_old}/5")
        if out["diff"]:
            print(f"       turbo diff: {out['diff'][:110]}")
        if p.get("diff"):
            print(f"       pro   diff: {p['diff'][:110]}")

    print(f"\n一致性：score(±10 内) {agree_score}/{n}   matched(全同) {agree_matched}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
