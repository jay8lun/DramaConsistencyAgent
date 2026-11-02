"""分步确认（三个停点）的离线验证。

覆盖四组：阶段机、S2 两段式、Web 接口契约、前端结构。
**全部离线、零花费**：mock provider + TestClient，不启动 uvicorn、不调外部 API。

跑法：`python tests/verify_phase.py`
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import config as C  # noqa: E402
from adapters.web.api import app  # noqa: E402
from models import Character, VisualStyle  # noqa: E402
from pipeline import phase as P  # noqa: E402
from pipeline import s2_anchor as S2  # noqa: E402
from pipeline.cost import CostLedger  # noqa: E402
from pipeline.project import create_or_load, write_json  # noqa: E402

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  {detail}" if detail else ""))


def raises(fn) -> tuple[bool, str]:
    try:
        fn()
        return False, "没抛异常"
    except Exception as exc:  # noqa: BLE001
        return True, f"{type(exc).__name__}: {exc}"


class CountingImage:
    """假出图器：只记调用，不联网。`is_mock=True` 让记账按 0 元走。"""

    is_mock = True
    name = "counting-image"
    unit_price = 0.0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt, *, out, references=None, size="1024x1792", allow_paid=False):
        self.calls.append({
            "prompt": prompt,
            "refs": [str(r) for r in (references or [])],
        })
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
        return p


class SilentJudge:
    is_mock = True
    name = "silent-judge"

    def compare(self, image, reference, elements, *, allow_paid: bool = False):
        return {"score": 95, "elements": {e: True for e in elements}, "diff": ""}


STYLE = VisualStyle(style_prompt="电影感，冷色调", negative_prompt="", aspect_ratio="9:16")
CHAR = Character(char_id="char_01", name="林晚", age_range="26-30岁", gender="女",
                 appearance="黑色齐肩直发，鹅蛋脸，细长眼", wardrobe="深灰羊毛大衣",
                 source="brief")

TMP_IDS = ["vphaseA", "vphaseB", "vphaseWeb", "vphaseReal", "vphaseDel", "vphaseTl"]


def rm_project(pid: str) -> None:
    """删临时项目。Windows 上句柄可能还没释放，重试几次。

    `ignore_errors=True` 单独用不够：它会静默跳过被占用的目录，
    留下一堆 `web<时间戳>` 垃圾，下次跑测试时它们会出现在"已有项目"列表里。
    """
    target = C.project_dir(pid)
    for _ in range(6):
        shutil.rmtree(target, ignore_errors=True)
        if not target.exists():
            return
        time.sleep(0.3)


def clean() -> None:
    for pid in TMP_IDS:
        rm_project(pid)


def wait_phase(client: TestClient, pid: str, targets: set[str], timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    d: dict = {}
    while time.time() < deadline:
        d = client.get(f"/api/projects/{pid}/status").json()
        if d.get("phase") in targets:
            return d
        time.sleep(0.4)
    return d


def main() -> int:
    clean()
    print("一、阶段机：转移合法性与落盘")
    ctx, _ = create_or_load(TMP_IDS[0], budget_cap=2.0, mock=True)
    st = P.load(ctx)
    check("新项目推断为 created", st.phase == "created", st.phase)
    check("默认来源是 brief", st.role_source == "brief")
    check("created 是停点", st.is_stop)

    ok, msg = raises(lambda: P.save(ctx, phase="running"))
    check("created 不能直接跳到 running", ok, msg.split("：")[-1][:40])

    st = P.save(ctx, phase="script_review")
    check("created → script_review 允许", st.phase == "script_review")
    raw = json.loads(ctx.project_file.read_text(encoding="utf-8"))
    check("写回保留 project_id", raw.get("project_id") == TMP_IDS[0])
    check("写回带时间戳", bool(raw.get("phase_updated_at")))

    ok, _ = raises(lambda: P.save(ctx, role_source="guess"))
    check("role_source 只收 brief/image", ok)

    ok, msg = raises(lambda: P.require_confirmed(ctx, confirm=False, action="出图"))
    check("停点未确认时拒绝继续", ok and "confirm" in msg.lower(), msg.split("，")[0][:40])
    ok2, _ = raises(lambda: P.require_confirmed(ctx, confirm=True, action="出图"))
    check("确认后放行", not ok2)

    print("\n二、旧项目按产物推断（不迁移）")
    ctx2, _ = create_or_load(TMP_IDS[1], budget_cap=2.0, mock=True)
    check("空项目 → created", P.infer(ctx2) == "created", P.infer(ctx2))
    write_json(ctx2.script_file, {"title": "t"})
    check("有剧本 → script_review", P.infer(ctx2) == "script_review", P.infer(ctx2))
    (ctx2.anchors_dir / "char_01_front.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    check("有正脸 → anchor_review", P.infer(ctx2) == "anchor_review", P.infer(ctx2))
    (ctx2.shots_dir / "shot_01.mp4").write_bytes(b"0")
    check("有单条视频 → running", P.infer(ctx2) == "running", P.infer(ctx2))
    (ctx2.root / "final.mp4").write_bytes(b"0")
    check("有成片 → done", P.infer(ctx2) == "done", P.infer(ctx2))

    print("\n三、S2 两段式：正脸单独出、侧背以正脸为参考")
    img = CountingImage()
    ledger = CostLedger(TMP_IDS[0], cap=2.0)
    front = S2.generate_front(ctx, CHAR, style=STYLE, image_provider=img,
                              ledger=ledger, allow_paid=True)
    check("generate_front 只出 1 张", len(img.calls) == 1, f"{len(img.calls)} 张")
    check("正脸文件已落盘", front.exists())
    stt = S2.load_status(ctx).get("char_01") or {}
    check("写 front_status=pending", stt.get("front_status") == "pending")
    check("front_attempts=1", stt.get("front_attempts") == 1)

    front = S2.generate_front(ctx, CHAR, style=STYLE, image_provider=img, ledger=ledger,
                              allow_paid=True, feedback="脸再瘦一点")
    stt = S2.load_status(ctx)["char_01"]
    check("重出后 attempts=2", stt.get("front_attempts") == 2, stt.get("front_attempts"))
    check("意见落进 feedback_log",
          [x.get("text") for x in stt.get("feedback_log") or []] == ["脸再瘦一点"])
    check("意见进了提示词", "脸再瘦一点" in (stt.get("front_prompt") or ""))
    check("原设定没被覆盖", "黑色齐肩直发" in (stt.get("front_prompt") or ""))
    # 2026-11-02：参考图必须写进提示词，而且必须定优先级（"以参考图为准"）。
    # 只把图喂进去、文字里不提，模型会把两者各采纳一部分——卡通参考图 + 人类角色卡
    # 出来的既不是卡通也不是真人，还带着一片解释不了的黄脖子。
    check("不带参考图时不出现「参考图为准」（避免指向不存在的东西）",
          "参考图为准" not in S2.build_view_prompt(CHAR, STYLE, "front"))
    check("带参考图时提示词要求以参考图为准",
          "以参考图为准" in S2.build_view_prompt(CHAR, STYLE, "front", has_ref=True))
    img.calls.clear()
    S2.generate_front(ctx, CHAR, style=STYLE, image_provider=img, ledger=ledger,
                      allow_paid=True, references=[front])
    check("真带图出图时，提示词里确实有这段约束",
          "以参考图为准" in (img.calls[0]["prompt"] if img.calls else ""))

    img.calls.clear()
    views = S2.generate_views(ctx, CHAR, style=STYLE, image_provider=img,
                              ledger=ledger, allow_paid=True, front=front)
    check("generate_views 出 2 张", len(img.calls) == 2, f"{len(img.calls)} 张")
    check("侧背都以那张正脸为参考",
          all(c["refs"] and c["refs"][0].endswith("char_01_front.png") for c in img.calls))
    check("三视图齐了", all(Path(v).exists() for v in views.values()))

    removed = S2.clear_views(ctx, "char_01")
    check("clear_views 删掉侧背 2 张", len(removed) == 2, f"{len(removed)} 张")
    check("正脸没被误删", ctx.anchor_of("char_01", "front").exists())
    check("侧背已不存在", not ctx.anchor_of("char_01", "side").exists())

    ctx3 = create_or_load(TMP_IDS[1], budget_cap=2.0, mock=True)[0]
    # 第二节为了测 infer() 在这里造过一张假正脸，先清掉再验"没有正脸就不能出侧背"
    for p in list(ctx3.anchors_dir.glob("*_front.png")):
        p.unlink()
    ok, msg = raises(lambda: S2.generate_views(ctx3, CHAR, style=STYLE, image_provider=img,
                                               ledger=ledger, allow_paid=True))
    check("没有正脸时出侧背被拒绝", ok and "正脸" in msg, msg[:40])

    S2.update_status(ctx, "char_01", front_status="pending", status="pending")
    check("pending 不算降级", "char_01" not in S2.degraded_chars(ctx))
    S2.update_status(ctx, "char_01", status="degraded")
    check("degraded 才算降级", "char_01" in S2.degraded_chars(ctx))

    print("\n四、Web 接口：停点必须显式确认")
    client = TestClient(app)
    r = client.post("/api/projects", json={"creative": "验证用的短剧创意文本",
                                           "shots": 2, "budget": 2.0,
                                           "mock": True, "role_source": "brief"})
    pid = r.json()["project_id"]
    check("建项目返回 phase=created", r.json().get("phase") == "created")
    d = client.get(f"/api/projects/{pid}/status").json()
    check("status 报 created", d.get("phase") == "created", d.get("phase"))
    check("status 带中文阶段名", bool(d.get("phase_label")), d.get("phase_label"))
    check("status 带 mock 标记", d.get("mock") is True)

    r = client.post(f"/api/projects/{pid}/run", json={"confirm": False})
    check("不带 confirm 被 409 拦", r.status_code == 409, f"HTTP {r.status_code}")

    r = client.post(f"/api/projects/{pid}/run", json={"confirm": True})
    check("带 confirm 开跑", r.status_code == 200)
    d = wait_phase(client, pid, {"script_review"})
    check("S1 后停在 script_review", d.get("phase") == "script_review", d.get("phase"))
    check("角色卡已产出", len(d.get("characters") or []) >= 2)

    sc = client.get(f"/api/projects/{pid}/script").json()
    bad = json.loads(json.dumps(sc))
    bad["shots"][0]["duration"] = 7
    r = client.put(f"/api/projects/{pid}/script", json={"script": bad})
    check("非法时长被 422 拦（不写盘）", r.status_code == 422, f"HTTP {r.status_code}")

    good = json.loads(json.dumps(sc))
    good["shots"][0]["action"] = (good["shots"][0].get("action") or "") + "（改）"
    r = client.put(f"/api/projects/{pid}/script", json={"script": good})
    check("改动作保存成功", r.status_code == 200)
    check("改动作不作废任何图", r.json().get("invalidated") == [])

    r = client.post(f"/api/projects/{pid}/run", json={"confirm": True})
    d = wait_phase(client, pid, {"anchor_review"})
    check("S2a 后停在 anchor_review", d.get("phase") == "anchor_review", d.get("phase"))
    chars = d.get("characters") or []
    check("每个角色都出了正脸", chars and all(c.get("has_front") for c in chars))
    check("侧背还没出",
          not any(C.project_dir(pid).glob("anchors/*_side.png")))
    check("正脸状态是待确认", all(c.get("front_status") == "pending" for c in chars))

    r = client.post(f"/api/projects/{pid}/anchor/confirm", json={"char_id": chars[0]["char_id"]})
    check("逐张确认成功", r.json().get("confirmed") is True)
    check("只确认一张时未解锁", r.json().get("all_confirmed") is False)
    r = client.post(f"/api/projects/{pid}/anchor/confirm", json={"char_id": chars[1]["char_id"]})
    check("两张都确认后解锁", r.json().get("all_confirmed") is True)

    # 402 必须在 **anchor_review 阶段**测。放到改长相之后测，会先撞上
    # "退回停 1、当前阶段没有正脸可重出"的 409——那是另一个正确的分支，
    # 拿它当 402 的证据等于什么都没验。
    #
    # 但 `pid` 这个项目是 mock 的，而 mock 下 `ctx.mock` 为真、永不拦付费。
    # 所以真要验授权闸，必须另起一个 **mock=False** 的项目（不跑流水线，
    # 手工把阶段摆到 anchor_review 并放一张正脸），否则这条断言永远是绿的假绿。
    real_id = TMP_IDS[3]
    rm_project(real_id)
    rctx, _ = create_or_load(real_id, budget_cap=2.0, mock=False)
    P.save(rctx, phase="script_review")
    P.save(rctx, phase="anchor_review")
    rf = rctx.anchor_of("char_01", "front")
    rf.parent.mkdir(parents=True, exist_ok=True)
    rf.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    r = client.post(f"/api/projects/{real_id}/anchor/regenerate",
                    json={"char_id": "char_01", "feedback": "没有 allow_paid"})
    # 402 而不是 400（2026-11-02）：400 是"参数写错了"，402 是"这笔还没授权"。
    # 前端要靠这个区别判断"能不能自动兜一次确认后重发"。
    check("真实模式重出返回 402（未授权，可补确认重发）",
          r.status_code == 402, f"HTTP {r.status_code}")
    detail = str(r.json().get("detail", ""))
    check("402 的文案指向弹窗里的勾选项（不是让用户去找不存在的控件）",
          "勾选" in detail and "弹窗" in detail, detail[:60])
    # **不测"带 allow_paid 就放行"**：这个项目是 mock=False，放行等于真的去
    # dashscope 出图（¥0.20/张）。而拿 mock 项目测放行，后台线程会与后续
    # "改长相作废"抢同一批文件，得到一个时灵时不灵的断言。
    # 放行那条路由用户真跑时验证更可靠；这里只锁死"没授权必须被拦且说得清"。

    # 接口级端到端：在 **mock 项目**上真走一次「重出」（不花钱、毫秒级返回）。
    # 2026-11-02 用户就是卡在这条路上：接口 402 + 前端传的授权状态是丢的。
    # 只验 402 不够——还得证明"授权之后那张图真的换掉了"。
    row0 = next(c for c in client.get(f"/api/projects/{pid}/status").json()["characters"]
                if c["char_id"] == chars[0]["char_id"])
    r = client.post(f"/api/projects/{pid}/anchor/regenerate",
                    json={"char_id": chars[0]["char_id"], "feedback": "换一张",
                          "allow_paid": True})
    check("mock 下重出接口放行",
          r.status_code == 200 and r.json().get("status") == "regenerating",
          f"HTTP {r.status_code} {r.text[:50]}")
    target = row0["front_attempts"] + 1
    cur, phases = row0, set()
    deadline = time.time() + 30
    while time.time() < deadline:
        dd = client.get(f"/api/projects/{pid}/status").json()
        phases.add(dd.get("phase"))
        cur = next(c for c in dd["characters"] if c["char_id"] == chars[0]["char_id"])
        if cur["front_attempts"] >= target:
            break
        time.sleep(0.2)
    check("重出真的换了图（front_attempts +1）",
          cur["front_attempts"] == target, f"{row0['front_attempts']} → {cur['front_attempts']}")
    # 重出只是一张图，不该让 status 报"生产中"——否则用户一刷新就被甩去进度页。
    check("重出期间不冒充流水线（phase 始终 anchor_review）",
          phases == {"anchor_review"}, str(sorted(phases)))
    sd = client.get(f"/api/projects/{pid}/status").json()
    check("status 暴露 regenerating / regenerating_char（刷新也能看出在换脸）",
          "regenerating" in sd and "regenerating_char" in sd,
          f"regenerating={sd.get('regenerating')}")

    # 片子开跑之后不许重出：首帧、视频、成片都是拿旧脸做的，
    # 换锚点不会让它们跟着变，而 S3 见状态是 done 会直接跳过。
    # 结果就是"界面新脸、片子旧脸"，且全程不报错。
    P.save(rctx, phase="running")
    r = client.post(f"/api/projects/{real_id}/anchor/regenerate",
                    json={"char_id": "char_01", "allow_paid": True})
    check("开跑后重出被 409 拒", r.status_code == 409, f"HTTP {r.status_code}")
    check("拒绝理由说清「成片还是旧脸」",
          "旧脸" in str(r.json().get("detail", "")), str(r.json().get("detail", ""))[:40])
    P.save(rctx, phase="anchor_review")

    sc2 = client.get(f"/api/projects/{pid}/script").json()
    sc2["characters"][0]["appearance"] = (sc2["characters"][0].get("appearance") or "") + "，右眉一道疤"
    r = client.put(f"/api/projects/{pid}/script", json={"script": sc2})
    check("改长相作废该角色正脸", len(r.json().get("invalidated") or []) == 1)
    d = client.get(f"/api/projects/{pid}/status").json()
    check("作废后退回 script_review", d.get("phase") == "script_review", d.get("phase"))
    check("其他角色正脸还在",
          [c["has_front"] for c in d["characters"]].count(True) == len(d["characters"]) - 1)

    r = client.post(f"/api/projects/{pid}/refs",
                    json={"filename": "x.png", "data_base64": "aGVsbG8gd29ybGQhIQ=="})
    check("非图片参考图被 400 拦", r.status_code == 400, f"HTTP {r.status_code}")

    # 前端就是 index.html，没有构建步骤。浏览器对 HTML 有启发式缓存，
    # 普通 F5 可能连请求都不发——2026-11-02 因此误判过"代码改了界面没变"。
    r = client.get("/")
    check("首页禁止缓存（改前端后一次刷新即生效）",
          "no-store" in (r.headers.get("cache-control") or ""),
          r.headers.get("cache-control", ""))

    print("\n五、前端结构")
    h = (Path(__file__).resolve().parents[1] / "web" / "index.html").read_text(encoding="utf-8")
    for v in ("home", "confirm", "anchor", "progress", "console", "report"):
        check(f"有视图 v-{v}", f'id="v-{v}"' in h)
    check("有停 0 面板", 'id="stop0"' in h)
    check("有停 1 面板", 'id="stop1"' in h)
    check("停 0 两个选项都在", 'pickSource(\'image\')' in h and 'pickSource(\'brief\')' in h)
    check("有重出意见输入框", 'id="fb-text"' in h)
    check("有「同时更新角色设定」勾选", 'id="fb-card"' in h)
    check("重出按钮带价格提示", "重出 ¥0.20" in h)
    # 2026-11-02：重出必须有自己的付费勾选。原来它读全局 S.allowPaid，
    # 而那个变量刷新即丢、界面上又没有输入口，于是真实模式重出必定失败。
    check("重出弹窗有付费勾选（不再靠内存里的全局开关）", 'id="fb-paid"' in h)
    check("勾选与按钮联动", "function syncRegenBtn" in h and "$('fb-paid').onchange = syncRegenBtn" in h)
    check("前端能识别 402 并兜一次确认", "e.status === 402" in h)
    # 反推最常没把握的就是这两项（卡通角色没有性别年龄），而它们原来只能看不能改，
    # 于是"请核对"这句话落不到任何能改的地方。
    check("性别/年龄段可编辑", 'data-k="gender"' in h and 'data-k="age_range"' in h)
    check("反推来源有徽章 + 低置信有标黄",
          "来自参考图" in h and "low_confidence_fields" in h and "#e6a23c" in h)

    print("\n六、停止 / 重跑单镜 / 删除项目")
    # 停止：只在"有线程在跑"时才有意义。没在跑还给 200 的话，
    # 界面上的停止按钮会一直点得下去，人以为停了、其实什么都没发生。
    r = client.post(f"/api/projects/{pid}/stop")
    check("没在跑时停止返回 409", r.status_code == 409, f"HTTP {r.status_code}")

    d = client.get(f"/api/projects/{pid}/status").json()
    # alive 是"进程内有没有线程在跑"。只看 phase 不够：服务重启后磁盘 phase
    # 仍是 running，界面会永远显示"生产中"，人只能干等。
    check("status 有 alive（区分「在生产」与「磁盘上是 running」）", "alive" in d)
    check("status 有 stopping（点了停止但当前镜还没收尾）", "stopping" in d)

    # 重跑单镜：阶段还没到出片时必须拒。否则界面上会出现两条都能让同一镜
    # 开跑的路径（继续 / 重跑），账目上分不清是谁花的钱。
    r = client.post(f"/api/projects/{pid}/shots/shot_01/rerun",
                    json={"reason": "x", "allow_paid": True})
    check("还没出过视频时重跑单镜被 409 拒", r.status_code == 409, f"HTTP {r.status_code}")

    # 402 关税授权：用一个 mock=False 的壳（复制合法 script.json，不跑流水线）。
    # 用 mock 项目测不出这条——mock 下 `ctx.mock` 为真，永不拦付费，
    # 断言会永远绿，等于没验。
    del_id = TMP_IDS[4]
    rm_project(del_id)
    dctx, _ = create_or_load(del_id, budget_cap=2.0, mock=False)
    # 剧本从**刚跑完 S1 的那个项目**拷：TMP_IDS[0] 只是阶段机的壳，没有 script.json，
    # 拿它做源会直接 FileNotFoundError——而这正是第一次跑时的实际报错。
    src_ctx, _ = create_or_load(pid, budget_cap=2.0, mock=True)
    shutil.copy(src_ctx.script_file, dctx.script_file)
    P.save(dctx, phase="script_review", force=True)
    P.save(dctx, phase="running", force=True)
    r = client.post(f"/api/projects/{del_id}/shots/shot_01/rerun",
                    json={"reason": "没授权"})
    check("重跑单镜未授权返回 402", r.status_code == 402, f"HTTP {r.status_code}")
    check("402 文案说清价格与勾选项",
          "¥" in str(r.json().get("detail", "")) and "勾选" in str(r.json().get("detail", "")),
          str(r.json().get("detail", ""))[:60])

    # 删除：先删目录、后清库。重建一个带产物的项目，验证两边都清干净。
    r = client.delete("/api/projects/no_such_project")
    check("删不存在的项目返回 404", r.status_code == 404, f"HTTP {r.status_code}")
    rm_project(del_id)
    dctx, _ = create_or_load(del_id, budget_cap=2.0, mock=True)
    shutil.copy(src_ctx.script_file, dctx.script_file)
    (dctx.root / "final.mp4").write_bytes(b"0" * 1024)
    P.save(dctx, phase="script_review", force=True)
    P.save(dctx, phase="running", force=True)
    r = client.delete(f"/api/projects/{del_id}")
    check("删除项目成功", r.status_code == 200 and r.json().get("deleted") is True,
          f"HTTP {r.status_code}")
    check("目录已删掉", not C.project_dir(del_id).exists())
    check("返回了删掉多少文件（界面要能告诉人删了什么）",
          int(r.json().get("files") or 0) >= 2, str(r.json().get("files")))
    from pipeline.state import StateStore  # noqa: E402  局部导入，避免头部再挂一个依赖
    _st = StateStore()
    try:
        _left = _st.all(del_id)
    finally:
        _st.close()
    check("数据库记录一起清掉（不留孤儿账）", _left == [], f"{len(_left)} 条残留")

    print("\n七、前端结构（停止 / 重跑 / 删除）")
    check("进度页有停止按钮", 'id="btn-stop"' in h)
    check("进度页有继续按钮", 'id="btn-continue"' in h)
    check("镜头表有重跑入口", "openShotRerun(" in h and "重跑" in h)
    check("重出单镜有自己的弹窗与付费勾选",
          'id="modal-shot"' in h and 'id="shot-paid"' in h)
    check("首页有删除按钮", "deleteProject(" in h)
    check("删除要手打项目名确认（不是点一下就删）",
          "输入不一致，已取消" in h)
    check("停止后能继续（前端读 alive 而不是只看 phase）",
          "d.alive" in h and "stopped" in h)

    print("\n八、项目卡（打开前先看这个项目干了啥）")
    r = client.get(f"/api/projects/no_such/timeline")
    check("时间线：项目不存在返回 404", r.status_code == 404, f"HTTP {r.status_code}")
    r = client.get(f"/api/projects/{pid}/timeline")
    check("时间线可读", r.status_code == 200, f"HTTP {r.status_code}")
    t = r.json()
    for k in ("title", "logline", "creative", "created_at", "phase", "phase_label",
              "role_source", "shots", "events", "artifacts"):
        check(f"时间线含 {k}", k in t)
    check("时间线自带阶段中文名（弹窗不用再查字典）",
          bool(t.get("phase_label")), str(t.get("phase_label")))
    # `pid` 这个项目停在停 1，**一条 shot 记录都还没登记**，事件自然为空。
    # 直接断言"必须有事件"会在正确的代码上报假红。所以另起一个项目，
    # 用它登记两镜——`ensure_shots` 本身就会写一条"登记"迁移，正好当样本。
    tl_id = TMP_IDS[5]
    rm_project(tl_id)
    tlctx, _ = create_or_load(tl_id, budget_cap=2.0, mock=True)
    shutil.copy(src_ctx.script_file, tlctx.script_file)
    _tls = StateStore()
    try:
        _tls.ensure_shots(tl_id, ["shot_01", "shot_02"])
    finally:
        _tls.close()
    t2 = client.get(f"/api/projects/{tl_id}/timeline").json()
    rm_project(tl_id)
    evs = t2.get("events") or []
    check("有状态迁移事件", len(evs) >= 2, f"{len(evs)} 条")
    # 正序而不是倒序：人读流程是从头读的，SQL 里是 ORDER BY id DESC 取最近 N 条，
    # 反转这一步漏掉的话，时间线会从上到下倒着走，看起来像"越跑越倒退"。
    check("事件按时间正序（SQL 取倒序、这里翻过来）",
          all(evs[i]["ts"] <= evs[i + 1]["ts"] for i in range(len(evs) - 1)),
          f"{evs[0]['ts']} → {evs[-1]['ts']}" if evs else "")
    check("事件带镜号/去向/原因三个字段",
          all({"ts", "shot_id", "to", "reason"} <= set(e) for e in evs[:3]) if evs else False)

    print("\n九、前端：项目卡与重出记录")
    check("首页「打开」走项目卡而不是直接进项目",
          "openProjectCard(" in h and 'onclick="openProject(' not in h)
    check("项目卡有「进入项目」按钮", 'id="btn-pj-open"' in h)
    check("项目卡含逐镜表与时间线容器",
          'id="pj-shots"' in h and 'id="pj-timeline"' in h)
    check("项目卡宽版可滚动（内容比正脸弹窗长得多）",
          ".modal-box.wide" in h and "max-height:88vh" in h)
    # 新窗口打开：单页应用的状态全在内存里，URL 不带项目 id 就没法被另一个标签页打开。
    check("有「新窗口打开」与「在本页打开」两个入口",
          'id="btn-pj-open"' in h and 'id="btn-pj-here"' in h)
    check("新窗口打开时 URL 带 ?p=（否则新标签页是空白首页）",
          "window.open(" in h and "?p=${encodeURIComponent(pid)}" in h)
    check("进入项目会把 id 写进地址栏", "setUrlProject(pid)" in h)
    check("带 ?p= 直接进项目（刷新/收藏/分享才回得去）",
          "URLSearchParams(location.search).get('p')" in h)
    check("回首页抹掉 ?p=（不然刷新又被拽回项目）",
          "b.dataset.v === 'home'" in h and "setUrlProject(null)" in h)
    check("?p= 指向已删除的项目时回列表而不是空白报错页",
          "e.status === 404" in h and "可能已被删除" in h)

    print("\n清理临时项目 ……")
    clean()
    rm_project(pid)
    print("   剩余项目：", sorted(p.name for p in C.PROJECTS_DIR.iterdir() if p.is_dir()))
    print()
    print("=" * 64)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
