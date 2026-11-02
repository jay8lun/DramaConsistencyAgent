#!/usr/bin/env python
"""探测账号里哪些模型可用、哪些还有额度。

用途：M0-S2 连续遇到 429 SetLimitExceeded（per-model 免费额度耗尽），
需要一个确定性的答案——到底还有哪个模型能拿来跑判定。

做法：对 models.list() 返回的每个模型发一次**最小文本请求**（max_tokens=8），
按返回的错误分类：
    OK        可调用且有额度
    LIMIT     额度耗尽（429 SetLimitExceeded）
    NOTOPEN   账号没开通（404 ModelNotOpen）—— 去控制台点开通即可
    NOEXIST   模型 ID 不存在 / 无权限（404 InvalidEndpointOrModel）
    OTHER     其他错误

成本：每个失败请求不计费，成功的最多几 token，全表扫一遍可以忽略。

用法：python scripts/probe_models.py [--only 关键字]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402


def classify(exc: Exception) -> str:
    s = str(exc)
    if "SetLimitExceeded" in s:
        return "LIMIT"
    if "ModelNotOpen" in s:
        return "NOTOPEN"
    if "InvalidEndpointOrModel" in s or "does not exist" in s:
        return "NOEXIST"
    return "OTHER"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只探模型 ID 含该关键字的")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(
        api_key=settings.judge_vlm.api_key,
        base_url=settings.judge_vlm.base_url,
        timeout=60.0,
    )

    ids = sorted(m.id for m in client.models.list().data)
    if args.only:
        ids = [i for i in ids if args.only in i]

    print(f"探测 {len(ids)} 个模型（每次 max_tokens=8）\n")
    buckets: dict[str, list[str]] = {}
    for mid in ids:
        try:
            client.chat.completions.create(
                model=mid,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=8,
            )
            tag = "OK"
        except Exception as exc:  # noqa: BLE001
            tag = classify(exc)
        buckets.setdefault(tag, []).append(mid)
        if tag == "OK":
            print(f"  [OK] {mid}")

    print("\n" + "=" * 60)
    for tag in ("OK", "LIMIT", "NOTOPEN", "OTHER", "NOEXIST"):
        items = buckets.get(tag, [])
        print(f"\n{tag}（{len(items)} 个）")
        if tag in ("OK", "LIMIT", "NOTOPEN", "OTHER"):
            for i in items:
                print("   ", i)
        else:
            for i in items[:8]:
                print("   ", i)
            if len(items) > 8:
                print(f"    ... 另有 {len(items) - 8} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
