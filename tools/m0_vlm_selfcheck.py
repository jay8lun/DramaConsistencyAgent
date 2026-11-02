#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""providers/vlm.py 的离线自检：不联网、不花钱、不依赖真实 API。

覆盖三条容易出错又不会被「跑通一次」暴露的逻辑：
1. 付费护栏：allow_paid 未授权时必须抛 PaidCallBlocked
2. JSON 抽取：模型输出常带 ```json 围栏或前后废话，必须能抠出对象
3. 归一化：模型漏判某要素时 matched 必须是 None（表示"没判"），
   不能被当成 False 污染要素级准确率的分母

用法：
    python scripts/m0_vlm_selfcheck.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from providers.base import PaidCallBlocked  # noqa: E402
from providers.vlm import (  # noqa: E402
    ELEMENT_HINTS,
    JUDGE_VERSION,
    VLMJudge,
    _extract_json,
    _normalize,
)

ELEMENTS = list(ELEMENT_HINTS.keys())
failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


print("[1] 付费护栏")
try:
    VLMJudge.__new__(VLMJudge).compare(
        Path("x.png"), Path("y.png"), ELEMENTS, allow_paid=False
    )
    check("allow_paid=False 抛 PaidCallBlocked", False, "竟然没抛")
except PaidCallBlocked:
    check("allow_paid=False 抛 PaidCallBlocked", True)
except Exception as exc:
    check("allow_paid=False 抛 PaidCallBlocked", False, f"抛了 {type(exc).__name__}")

print("\n[2] JSON 抽取")
cases = [
    ('{"score": 88, "checks": []}', 88, "裸 JSON"),
    ('```json\n{"score": 77, "checks": []}\n```', 77, "带围栏"),
    ('好的，结果如下：\n{"score": 66, "checks": []}\n希望有帮助', 66, "前后带废话"),
    ('{"note": "含 } 的字符串", "score": 55}', 55, "字符串里有花括号"),
]
for text, expect, label in cases:
    try:
        obj = _extract_json(text)
        check(f"抽取成功：{label}", obj.get("score") == expect, f"得到 {obj}")
    except Exception as exc:
        check(f"抽取成功：{label}", False, f"{type(exc).__name__}: {exc}")

try:
    _extract_json("完全不是 JSON")
    check("非 JSON 输入应报错", False, "竟然没报错")
except Exception:
    check("非 JSON 输入应报错", True)

print("\n[3] 归一化")
raw = {
    "score": 118,  # 越界
    "checks": [
        {"element": "发型", "matched": True, "expected": "黑色齐肩直发", "actual": "黑色齐肩直发"},
        {"element": "脸型", "matched": "false", "expected": "鹅蛋脸", "actual": "方脸"},
    ],
    "diff": "脸型不一致：参考图是鹅蛋脸，待判画面是方脸",
}
res = _normalize(raw, ELEMENTS, "test-model", {"total_tokens": 100})
check("score 越界被夹到 100", res["score"] == 100, f"得到 {res['score']}")
check("字符串 'false' 被解析为 False",
      res["checks"][1]["matched"] is False, f"得到 {res['checks'][1]['matched']}")
check("漏判的要素 matched 为 None",
      all(c["matched"] is None for c in res["checks"][2:]),
      f"得到 {[c['matched'] for c in res['checks'][2:]]}")
check("漏判要素被记入 missing_elements",
      res["missing_elements"] == ["眉眼", "肤色", "体型"],
      f"得到 {res['missing_elements']}")
check("checks 数量等于要素数", len(res["checks"]) == 5, f"得到 {len(res['checks'])}")
check("diff 保留原文", res["diff"].startswith("脸型不一致"), f"得到 {res['diff']!r}")
check("judge 版本被记录", res["judge_version"] == JUDGE_VERSION)

print("\n[4] 请求构造（不发送）")
from providers.vlm import build_judge_messages  # noqa: E402
pairs = ROOT / "tests" / "m0_eval" / "pairs"
ref = pairs / "01_a.png"
img = pairs / "02_a.png"
if ref.exists() and img.exists():
    msgs, meta = build_judge_messages(ref, img, ELEMENTS)
    blocks = msgs[1]["content"]
    img_blocks = [b for b in blocks if b["type"] == "image_url"]
    check("发出 2 个图片块", len(img_blocks) == 2, f"得到 {len(img_blocks)}")
    check("图片以 data URL 内联",
          all(b["image_url"]["url"].startswith("data:image/") for b in img_blocks))
    check("系统提示含五项要素说明",
          all(el in msgs[0]["content"] for el in ELEMENTS))
    check("系统提示不泄露业务阈值 85/60",
          "85" not in msgs[0]["content"] and "60~85" not in msgs[0]["content"])
    print(f"  INFO  请求体 {meta['payload_bytes'] / 1048576:.2f} MB"
          f"  参考图 {meta['reference']['width']}x{meta['reference']['height']}")
else:
    print("  SKIP  pairs 缺图，跳过")

print("\n" + "=" * 46)
if failures:
    print(f"自检未通过，{len(failures)} 项失败：{failures}")
    sys.exit(1)
print("自检全部通过")
