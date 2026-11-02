#!/usr/bin/env python
"""调用审计：把「到底怎么调用模型」的每一项事实打出来，用于对账。

回答的问题：
  1. 请求发往哪个 host / 哪个 region
  2. 用的是哪个 model ID（模型名直连 还是 接入点 ep-xxx）
  3. 用的是哪个 API Key（只打印指纹，不打印明文）
  4. 方舟侧把这次调用记在哪个账号名下（错误响应里的 account id）
  5. 每次调用的 request id（对账凭证，可在控制台/工单里查）

用法：python scripts/audit_call.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ENV_FILE, settings  # noqa: E402


def fingerprint(secret: str) -> str:
    """只暴露前缀与哈希，避免把密钥打进日志。"""
    if not secret:
        return "(空)"
    h = hashlib.sha256(secret.encode()).hexdigest()[:12]
    return f"{secret[:12]}...{secret[-4:]}  sha256[:12]={h}  长度={len(secret)}"


def main() -> int:
    print("=" * 66)
    print("一、配置来源")
    print("=" * 66)
    print(f".env 路径        : {ENV_FILE}")
    print(f"文件是否存在     : {ENV_FILE.exists()}")
    print(f"api_key 指纹     : {fingerprint(settings.judge_vlm.api_key)}")
    print(f"base_url         : {settings.judge_vlm.base_url}")
    print(f"model            : {settings.judge_vlm.model}")
    print(f"thinking 开关    : {settings.judge_vlm_thinking}")
    print(f"并发             : {settings.judge_vlm_concurrency}")

    base = settings.judge_vlm.base_url.rstrip("/")
    print()
    print("=" * 66)
    print("二、实际请求落点")
    print("=" * 66)
    print(f"完整 URL         : POST {base}/chat/completions")
    print(f"协议             : OpenAI Chat Completions（OpenAI 兼容模式）")
    host = base.split("//")[-1].split("/")[0]
    print(f"host             : {host}")
    print(f"region 推断      : {'cn-beijing（华北2·北京）' if 'cn-beijing' in host else '非北京，请核对'}")
    model = settings.judge_vlm.model
    print(f"model 字段类型   : "
          f"{'推理接入点 ID（ep-xxx）' if model.startswith('ep-') else '模型名直连（非接入点）'}")

    print()
    print("=" * 66)
    print("三、实测：发一次最小调用，看方舟怎么回")
    print("=" * 66)
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.judge_vlm.api_key,
        base_url=settings.judge_vlm.base_url,
        timeout=60.0,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
        )
        print("结果: 成功")
        print(f"  返回 id      : {getattr(resp, 'id', '?')}  ← 这是方舟的 request id")
        print(f"  model 回显   : {getattr(resp, 'model', '?')}")
        print(f"  usage        : {resp.usage}")
    except Exception as exc:  # noqa: BLE001
        print("结果: 失败（但这恰好是证据——失败响应里带着归属信息）")
        body = getattr(exc, "body", None)
        text = json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else str(exc)
        print(f"  异常类型     : {type(exc).__name__}")
        print(f"  HTTP 状态    : {getattr(exc, 'status_code', '?')}")
        print(f"  响应体       : {text[:600]}")
        # 从响应里抠出 request id 与 account id
        import re

        rid = re.search(r"Request id:\s*([0-9a-f]+)", text)
        acc = re.search(r"account \[(\d+)\]", text)
        if rid:
            print(f"\n  → request id : {rid.group(1)}")
            print("     （凭证：可在方舟控制台/工单里用这个 id 反查这次调用）")
        if acc:
            print(f"  → 账号 ID    : {acc.group(1)}")
            print("     （凭证：方舟把这次调用记在这个账号名下；"
                  "若与你控制台登录的账号 ID 不同，说明看错账号了）")

    print()
    print("=" * 66)
    print("四、历史调用凭证（从 raw_results.json 统计）")
    print("=" * 66)
    raw = ROOT / "tests" / "fixtures" / "m0_eval" / "raw_results.json"
    backup = ROOT / "tests" / "fixtures" / "m0_eval" / "raw_results_pro_23.json"
    for path in (raw, backup):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        models = sorted({x["model"] for x in data})
        ti = sum(x["usage"].get("prompt_tokens", 0) for x in data)
        to = sum(x["usage"].get("completion_tokens", 0) for x in data)
        print(f"  {path.name}: {len(data)} 条  模型={models}  token={ti + to:,}")
        # 检查是否存了 request id
        has_rid = any("request_id" in x for x in data)
        print(f"    是否保存了 request id: {'是' if has_rid else '否（当前版本未采集）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
