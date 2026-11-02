"""Web 层离线验证（13.5 六条验收 + 7.5 两条硬约束）。

**全部离线、零花费**：不启动 uvicorn、不调任何外部 API，用 TestClient 直接打 app。

为什么不用真实服务器测：`uvicorn` 起来之后要占端口、要清理，而这里验的是
「接口有没有按契约返回」「该拦的有没有拦住」——这些在被调用方就验完了。
真机的启动一遍由 `tests/verify_delivery.py` 之外的**手工验收**覆盖（见文档 13.5）。

跑法：`python tests/verify_web.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from adapters.web.api import app  # noqa: E402

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  {detail}" if detail else ""))


client = TestClient(app)

# ---------------------------------------------------------------- 1 静态与页面
print("=" * 64)
print("[1] 单页与五个视图（13.5：5 个视图均可访问）")
print("=" * 64)
r = client.get("/")
check("GET / 返回 200", r.status_code == 200, f"HTTP {r.status_code}")
html = r.text if r.status_code == 200 else ""
for v, label in (("home", "首页"), ("confirm", "角色与分镜"), ("progress", "生产进度"),
                 ("console", "操作台"), ("report", "报告")):
    check(f"视图「{label}」在页面里", f'id="v-{v}"' in html)
check("页面无 npm/构建产物引用（不出现 /src/ 或 .jsx）",
      "/src/" not in html and ".jsx" not in html)
check("前端未依赖 CDN（可离线双击）", "cdn." not in html and "unpkg" not in html)

# ---------------------------------------------------------------- 2 三个必备接口
print()
print("=" * 64)
print("[2] 三个接口都在（5.1）")
print("=" * 64)
paths = {(r.path, tuple(sorted(getattr(r, "methods", []) or []))) for r in app.routes}
check("POST /api/projects", ("/api/projects", ("POST",)) in paths)
check("GET  /api/projects/{pid}/status", ("/api/projects/{pid}/status", ("GET",)) in paths)
check("GET  /api/projects/{pid}/artifacts", ("/api/projects/{pid}/artifacts", ("GET",)) in paths)
check("GET  /api/projects/{pid}/script（视图 2 需要）",
      ("/api/projects/{pid}/script", ("GET",)) in paths)
check("POST /api/projects/{pid}/decide（操作台裁决）",
      ("/api/projects/{pid}/decide", ("POST",)) in paths)

# ---------------------------------------------------------------- 3 不存在的项目
print()
print("=" * 64)
print("[3] 不存在的项目返回 404，不抛 500")
print("=" * 64)
r = client.get("/api/projects/nosuchproj/status")
check("status 未知名 → 404", r.status_code == 404, f"HTTP {r.status_code}")
r = client.get("/api/projects/nosuchproj/artifacts")
check("artifacts 未知名 → 404", r.status_code == 404, f"HTTP {r.status_code}")
r = client.post("/api/projects/nosuchproj/run", json={"allow_paid": False})
check("run 未知名 → 404", r.status_code == 404, f"HTTP {r.status_code}")

# ---------------------------------------------------------------- 3b 空项目
print()
print("=" * 64)
print("[3b] 建好但还没跑的项目：200 + 空结构，不吞错误（2026-10-31）")
print("=" * 64)
import json as _json  # noqa: E402
import shutil as _shutil  # noqa: E402

from config import project_dir as _pdir  # noqa: E402

_tmp = "tmpwebempty"
_root = _pdir(_tmp)
try:
    _root.mkdir(parents=True, exist_ok=True)
    (_root / "project.json").write_text(_json.dumps({
        "project_id": _tmp, "budget_cap": 9.5, "channel": "api",
        "mock": True, "created_at": "2026-10-31T00:00:00+08:00",
    }), encoding="utf-8")
    r = client.get(f"/api/projects/{_tmp}/status")
    d = r.json() if r.status_code == 200 else {}
    # 这里原本返回 404。后果是：刚建好、流水线还没写第一条镜头记录时，
    # 界面只剩"没有镜头记录"——而同一响应里的 phase 与 error 一起被丢了，
    # 最该看的失败原因反而拿不到（用户看到的是"跑完了啥也没有"）。
    check("空项目 status → 200（不是 404）", r.status_code == 200, f"HTTP {r.status_code}")
    check("空项目返回 total=0 与空 shots",
          d.get("total") == 0 and d.get("shots") == [])
    check("空项目仍带 phase / error 字段", "phase" in d and "error" in d,
          "404 会把这两个字段一起吞掉")
    check("上限取项目 budget_cap（9.5），不是全局默认",
          d.get("cost", {}).get("cap") == 9.5,
          f"cap={d.get('cost', {}).get('cap')}")
finally:
    _shutil.rmtree(_root, ignore_errors=True)

# ---------------------------------------------------------------- 4 路径穿越
print()
print("=" * 64)
print("[4] 静态服务挡住路径穿越（安全）")
print("=" * 64)
for evil in ("..%2F..%2F.env", "%2e%2e/%2e%2e/.env"):
    r = client.get(f"/files/six01/{evil}")
    check(f"拒绝 {evil}", r.status_code in (400, 404), f"HTTP {r.status_code}")
r = client.get("/files/../.env")
check("拒绝项目名穿越", r.status_code in (400, 404, 422), f"HTTP {r.status_code}")

# ---------------------------------------------------------------- 5 付费默认关闭
print()
print("=" * 64)
print("[5] 付费默认关闭（D7 / 9.4）")
print("=" * 64)
from adapters.web.api import CreateBody, DecideBody, RunBody  # noqa: E402

check("RunBody.allow_paid 默认 False", RunBody().allow_paid is False)
check("CreateBody.mock 默认 False", CreateBody(creative="测试").mock is False)
try:
    DecideBody(shot_id="shot_01", decision="pass")
    check("裁决只接受 accept/degrade/reject", False, "非法值没被拒")
except Exception:
    check("裁决只接受 accept/degrade/reject", True)

# ---------------------------------------------------------------- 6 7.5 两条硬约束
print()
print("=" * 64)
print("[6] 7.5 硬约束：不用 BackgroundTasks、不提供 reload")
print("=" * 64)
src_path = Path(__file__).resolve().parents[1] / "adapters" / "web" / "api.py"
text = src_path.read_text(encoding="utf-8")
# 用 AST 判"有没有真的用到"，而不是字符串包含——注释里为了说明为什么禁用
# `BackgroundTasks` 会写到这个词，字符串搜索会把自己写的注释当成违规（第一版就这么误报了）。
import ast  # noqa: E402

_tree = ast.parse(text)
_used = {n.id for n in ast.walk(_tree) if isinstance(n, ast.Name)}
_used |= {n.attr for n in ast.walk(_tree) if isinstance(n, ast.Attribute)}
_imported = {a.name for n in ast.walk(_tree) if isinstance(n, ast.ImportFrom)
             for a in n.names}
check("源码里没有使用 BackgroundTasks（AST 级）",
      "BackgroundTasks" not in _used and "BackgroundTasks" not in _imported)
check("用独立线程跑流水线", "threading.Thread" in text)
check("没有把 --reload 写进任何启动命令", "--reload" not in text.replace("禁止 `uvicorn --reload`", ""))

# ---------------------------------------------------------------- 7 已有项目的读取
print()
print("=" * 64)
print("[7] 对已有真实项目（six01）的读取契约")
print("=" * 64)
r = client.get("/api/projects/six01/status")
if r.status_code == 200:
    d = r.json()
    need = {"project_id", "phase", "counts", "total", "cost", "awaiting_review", "shots"}
    check("status 字段齐全", need <= set(d), f"缺 {need - set(d)}" if not need <= set(d) else "")
    check("cost 里有 spent/cap（实扣与上限分开）",
          {"spent", "cap"} <= set(d.get("cost", {})))
    if d.get("shots"):
        s = d["shots"][0]
        check("单镜字段含 status/score/video_score/cost",
              {"shot_id", "status", "score", "video_score", "cost"} <= set(s))
    r2 = client.get("/api/projects/six01/artifacts")
    a = r2.json()
    check("artifacts 有 final/report/shots 三段",
          {"final", "report", "shots"} <= set(a))
    check("shots 数量与镜头数一致", len(a["shots"]) == d["total"],
          f"{len(a['shots'])} vs {d['total']}")
    # RunState 是进程内字典，测试进程里本来是空的；全靠按数据库推断。
    # 推断失效的后果很具体：片子跑完了，"报告"页点不开。
    check("重启后 phase 仍能推断为 done（报告页可点）",
          d.get("phase") == "done", f"phase={d.get('phase')}")
else:
    check("six01 存在（本机应有真实项目）", False, f"HTTP {r.status_code}")

print()
print("=" * 64)
print(f"结果：{passed} 通过 / {failed} 失败")
print("=" * 64)
sys.exit(1 if failed else 0)
