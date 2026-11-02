"""图生文反推角色卡的离线验证（计划书 4.2 入口 A）。

覆盖四组：解析规则、覆盖规则（宁缺勿错）、缓存与失效、接进 S1 的端到端。
**全部离线、零花费**：MockJudge + MockText，不发任何请求。

跑法：`python tests/verify_ref_desc.py`
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import config as C  # noqa: E402
import run as R  # noqa: E402
from adapters.web.api import app  # noqa: E402
from models import Character, VisualStyle  # noqa: E402
from pipeline import phase as P  # noqa: E402
from pipeline import ref_desc as RD  # noqa: E402
from pipeline import s1_script as S1  # noqa: E402
from pipeline.cost import CostLedger  # noqa: E402
from pipeline.project import create_or_load, write_json  # noqa: E402
from providers.mock import MockJudge, MockText  # noqa: E402

client = TestClient(app)

PID = "vrefdesc"

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


class CountingJudge:
    """包一层 MockJudge，记调用次数——验缓存必须能数清"到底调没调模型"。"""

    is_mock = True
    name = "counting-vlm"

    def __init__(self) -> None:
        self.calls = 0
        self._inner = MockJudge()

    def describe(self, image, *, allow_paid: bool = False):
        self.calls += 1
        return self._inner.describe(image, allow_paid=allow_paid)

    def compare(self, image, reference, elements, *, allow_paid: bool = False):
        return self._inner.compare(image, reference, elements, allow_paid=allow_paid)


STYLE = VisualStyle(style_prompt="电影感，冷色调", negative_prompt="", aspect_ratio="9:16")
CHAR = Character(char_id="char_01", name="林晚", age_range="26-30岁", gender="女",
                 appearance="黑色齐肩直发，鹅蛋脸，细长眼，白皙肤色，偏瘦",
                 wardrobe="深灰色羊毛大衣", source="brief")
CHAR2 = Character(char_id="char_02", name="陈默", age_range="31-35岁", gender="男",
                  appearance="黑色极短寸头，长方脸", wardrobe="藏青色工装夹克",
                  source="brief")

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def clean() -> None:
    target = C.project_dir(PID)
    shutil.rmtree(target, ignore_errors=True)


def main() -> int:
    clean()
    print("=" * 64)
    print("一、parse_desc：只做类型对齐，不替模型翻译")
    print("=" * 64)

    raw = dict(MockJudge.MOCK_CARD)
    card = RD.parse_desc(raw, image="refs/ref_01.png")
    check("appearance 原样保留", card.appearance == raw["appearance"], card.appearance[:20])
    check("subject_type=cartoon", card.subject_type == "cartoon")
    check("gender「不确定」→ 空（不冒充事实）", card.gender == "", f"'{card.gender}'")
    check("age_range「不确定」→ 空", card.age_range == "", f"'{card.age_range}'")
    check("wardrobe 空就是空", card.wardrobe == "")
    check("非人角色强制把 gender/age_range 标低置信",
          {"gender", "age_range"} <= set(card.low_confidence), str(card.low_confidence))
    check("confidence 0.1 的 wardrobe 也进低置信", "wardrobe" in card.low_confidence)
    check("confidence 0.9 的 appearance 不进低置信", "appearance" not in card.low_confidence)

    ok, msg = raises(lambda: RD.parse_desc({"appearance": "  "}))
    check("appearance 为空 → 抛错（空卡不许覆盖原卡）", ok and "appearance" in msg, msg[:40])
    ok, msg = raises(lambda: RD.parse_desc("not a dict"))
    check("非对象 → 抛错", ok, msg[:40])
    check("gender 含「女」就取女", RD.parse_desc({"appearance": "x", "gender": "女性"}).gender == "女")
    check("未知 subject_type 归 other",
          RD.parse_desc({"appearance": "x", "subject_type": "robot"}).subject_type == "other")
    check("confidence 非数字被忽略",
          RD.parse_desc({"appearance": "x", "confidence": {"gender": "高"}}).confidence == {})

    print()
    print("=" * 64)
    print("二、apply_to_characters：宁缺勿错")
    print("=" * 64)

    chars, notes = RD.apply_to_characters([CHAR, CHAR2], [card, None])
    c1, c2 = chars
    check("第 1 个角色的 appearance 被图覆盖",
          c1.appearance == raw["appearance"], c1.appearance[:18])
    check("source 标成 inferred", c1.source == "inferred", c1.source)
    check("truth_image 指向原图", (c1.truth_image or "").endswith("ref_01.png"), c1.truth_image or "")
    check("低置信字段带到角色卡", "age_range" in c1.low_confidence_fields)
    check("gender 没判出来时保留原值（字段必填，清空会让模型自己补）",
          c1.gender == "女", c1.gender)
    check("age_range 没判出来时保留原值", c1.age_range == "26-30岁", c1.age_range)
    check("wardrobe 反推为空 → **置空**，不保留原卡那件大衣",
          c1.wardrobe == "", f"'{c1.wardrobe}'")
    check("name 不被改（名字图里读不出）", c1.name == "林晚", c1.name)
    check("没图可对的角色原样不动",
          c2.appearance == CHAR2.appearance and c2.source == "brief")
    check("摘要说清来源与低置信", bool(notes) and "ref_01" in notes[0], notes[0] if notes else "")

    print()
    print("=" * 64)
    print("三、缓存：不重复花钱，换图必失效")
    print("=" * 64)

    ctx, _ = create_or_load(PID, budget_cap=2.0, mock=True)
    P.save(ctx, phase="created", role_source="image")
    refs_dir = ctx.root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    img = refs_dir / "ref_01.png"
    img.write_bytes(PNG)

    judge = CountingJudge()
    ledger = CostLedger(PID, cap=2.0)
    try:
        d1 = RD.describe_ref(ctx, 0, img, judge=judge, ledger=ledger, allow_paid=True)
        check("首次反推调用模型一次", judge.calls == 1, f"{judge.calls} 次")
        check("缓存落盘", RD.cache_path(ctx, 0).exists())
        d2 = RD.describe_ref(ctx, 0, img, judge=judge, ledger=ledger, allow_paid=True)
        check("同样一张图第二次走缓存，不再调模型",
              judge.calls == 1 and d2.appearance == d1.appearance, f"{judge.calls} 次")

        # 换图：内容变了但文件名不变。只按 mtime 判断的话，同一秒内重传会漏掉，
        # 表现就是"脸还是上一张的"且没有任何提示。
        img.write_bytes(PNG + b"changed")
        RD.describe_ref(ctx, 0, img, judge=judge, ledger=ledger, allow_paid=True)
        check("换图后指纹变化 → 重新反推", judge.calls == 2, f"{judge.calls} 次")

        ledger2 = CostLedger("vrefdesc_nopay", cap=2.0)
        try:
            ok, msg = raises(lambda: RD.describe_ref(
                ctx, 1, img, judge=judge, ledger=ledger2, allow_paid=False))
            check("没授权付费 → 拒绝反推", ok and "allow_paid" in msg, msg[:50])
        finally:
            ledger2.close()
    finally:
        ledger.close()

    print()
    print("=" * 64)
    print("四、端到端：接进 S1，剧本的名字 + 图的长相")
    print("=" * 64)

    ctx, _ = create_or_load(PID, budget_cap=2.0, mock=True, force=True)
    P.save(ctx, phase="created", role_source="image", force=True)
    refs_dir = ctx.root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    img = refs_dir / "ref_01.png"
    img.write_bytes(PNG)
    P.save(ctx, ref_images=["refs/ref_01.png"], role_source="image")

    args = Namespace(project_id=PID, creative="电梯里的十七楼", shots=6, force=False,
                     paid_ok=True, mock=True, confirm=False, auto=False, stop_after=None,
                     regen_front=None, feedback="", update_card=False, role_source=None,
                     ref=None)
    ledger = CostLedger(PID, cap=2.0)
    try:
        script = R.stage_s1(ctx, args, ledger, MockText(), judge_provider=MockJudge())
    finally:
        ledger.close()

    saved = json.loads(ctx.characters_file.read_text(encoding="utf-8"))
    s1 = saved[0]
    check("characters.json 第一张卡来自反推", s1["source"] == "inferred", s1["source"])
    check("appearance 是图里那套（不是剧本原写的人类女性）",
          s1["appearance"] == MockJudge.MOCK_CARD["appearance"], s1["appearance"][:18])
    # 两处都要写：只写 characters.json 的话，报告页（读 script.json）会显示旧设定，
    # 而且两份文件各自自洽，查不出来。
    on_disk = json.loads(ctx.script_file.read_text(encoding="utf-8"))
    check("script.json 同步更新（报告与停 1 才不会各说一套）",
          on_disk["characters"][0]["appearance"] == MockJudge.MOCK_CARD["appearance"])
    check("分镜没被反推影响", len(script.shots) == 6, f"{len(script.shots)} 镜")
    check("第二张卡没有对应图，保持 brief", saved[1]["source"] == "brief", saved[1]["source"])

    # 反推失败必须不阻断：可选增强拖垮主流程是最坏的设计。
    class BrokenJudge:
        is_mock = True
        name = "broken-vlm"

        def describe(self, image, *, allow_paid: bool = False):
            raise RuntimeError("模拟视觉模型挂了")

        def compare(self, image, reference, elements, *, allow_paid: bool = False):
            raise RuntimeError("不该被调用")

    ctx, _ = create_or_load(PID, budget_cap=2.0, mock=True, force=True)
    P.save(ctx, phase="created", role_source="image", force=True)
    refs_dir = ctx.root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "ref_01.png").write_bytes(PNG)
    P.save(ctx, ref_images=["refs/ref_01.png"], role_source="image")
    # 必须清掉这三样，否则这个场景什么也没验：
    # - 留着 script.json，stage_s1 会走"复用已有剧本"直接 return，根本不进反推；
    # - 留着 desc 缓存，BrokenJudge 不会被调用，覆盖照样成功（那是缓存救的，不是容错）。
    for f in (ctx.script_file, ctx.characters_file, RD.cache_path(ctx, 0)):
        Path(f).unlink(missing_ok=True)
    args.force = False
    ledger = CostLedger(PID, cap=2.0)
    try:
        script = R.stage_s1(ctx, args, ledger, MockText(), judge_provider=BrokenJudge())
        ok = True
    except Exception as exc:  # noqa: BLE001
        ok, script = False, None
        print(f"     反推异常没被吞：{exc}")
    finally:
        ledger.close()
    check("反推挂掉时 S1 照常出剧本（不阻断）",
          ok and script is not None and len(script.shots) == 6)
    kept = json.loads(ctx.characters_file.read_text(encoding="utf-8"))[0]
    check("反推失败的角色卡保持 brief（没有半覆盖）", kept["source"] == "brief", kept["source"])

    print()
    print("=" * 64)
    print("四之二、保存即核对：低置信标记要能消掉")
    print("=" * 64)

    # 黄框是给"还没看的人"的提示。人点了「保存并确认」还不清，
    # 下次再看就分不清"我核对过了"和"我还没看"。
    ctx, _ = create_or_load(PID, budget_cap=2.0, mock=True, force=True)
    P.save(ctx, phase="created", role_source="image", force=True)
    mark = CHAR.model_copy(update={
        "appearance": "亮黄色圆头", "source": "inferred",
        "truth_image": "refs/ref_01.png",
        "low_confidence_fields": ["gender", "age_range"],
    })
    # 用 mock 剧本做底：它的 scenes/shots 引用自洽，直接换角色即可，
    # 不必手搓一份能过 check_references 的剧本。
    payload = S1.parse(MockText().complete("s", "u", json_mode=True)).model_dump()
    payload["characters"] = [mark.model_dump(), CHAR2.model_dump()]
    write_json(ctx.characters_file, payload["characters"])
    write_json(ctx.script_file, payload)
    sc = client.get(f"/api/projects/{PID}/script").json()
    r = client.put(f"/api/projects/{PID}/script", json={"script": sc})
    check("保存成功", r.status_code == 200, f"HTTP {r.status_code}")
    check("返回里报出被清掉标记的角色",
          r.json().get("confirmed_marks") == ["char_01"], str(r.json().get("confirmed_marks")))
    after = json.loads(ctx.characters_file.read_text(encoding="utf-8"))
    check("落盘的角色卡没有低置信标记了",
          all(not c["low_confidence_fields"] for c in after),
          str([c["low_confidence_fields"] for c in after]))
    check("来源仍是 inferred（清标记不等于改来源）",
          after[0]["source"] == "inferred", after[0]["source"])

    print()
    print("=" * 64)
    print("五、换图重推：上传接口把 inferred 打回 brief")
    print("=" * 64)

    # 不重置的话：角色是 inferred → describe_all 跳过 → 新图永远进不了角色卡，
    # 而界面上"图明明传上去了"。这是比原 bug 更难查的一类：钱花了、图传了、没生效。
    ctx, _ = create_or_load(PID, budget_cap=2.0, mock=True, force=True)
    P.save(ctx, phase="created", role_source="image", force=True)
    inferred = CHAR.model_copy(update={
        "appearance": "旧卡外观", "source": "inferred",
        "truth_image": "refs/ref_01.png", "low_confidence_fields": ["age_range"],
    })
    write_json(ctx.characters_file, [inferred.model_dump(), CHAR2.model_dump()])
    refs_dir = ctx.root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", (600, 800), (250, 220, 60)).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    r = client.post(f"/api/projects/{PID}/refs",
                    json={"filename": "ref.png", "data_base64": b64})
    check("真图上传成功", r.status_code == 200, f"HTTP {r.status_code} {r.text[:60]}")
    back = json.loads(ctx.characters_file.read_text(encoding="utf-8"))[0]
    check("该角色被打回 brief（下次继续会重新反推）", back["source"] == "brief", back["source"])
    check("内容不清空（反推再失败也还有旧卡可用）",
          back["appearance"] == "旧卡外观", back["appearance"])
    check("低置信标记清掉（卡已经不作数了）", back["low_confidence_fields"] == [])
    check("重复上传不会重复记账（rel 去重）", r.json().get("count") == 1, r.json().get("count"))

    print()
    print("=" * 64)
    print("六、先读图再写剧本（S1 前移）")
    print("=" * 64)
    # 2026-11-02：S1 原来只拿创意文本编人物，写出来的卡是"23-27 岁人类男性"，
    # 分镜动作也按人类的习惯写；反推只在之后覆盖外貌，改不了已写进分镜的称谓与动作。
    human = RD.DescribedCard(
        image="ref_01.png", subject_type="human", gender="女", age_range="18-22岁",
        appearance="深棕色盘发，鹅蛋脸，白皙肤色，体型纤细",
        wardrobe="黑色长裙，白色蕾丝领口", style_hint="写实摄影",
        distinctive="左眉尾一颗小痣", low_confidence=[])
    cartoon = RD.DescribedCard(
        image="ref_02.png", subject_type="cartoon", gender="", age_range="",
        appearance="明黄色圆形身体，约三头身，短粗四肢，无脖颈",
        wardrobe="深紫色长袍", style_hint="3D卡通渲染",
        distinctive="头顶一对小尖角", low_confidence=["gender", "age_range"])
    lines = RD.describe_lines([human, cartoon])
    check("一张图一行", len(lines) == 2, str(len(lines)))
    check("主体类型译成中文（不把 cartoon 原样丢给模型）",
          "卡通形象" in lines[1] and "cartoon" not in lines[1], lines[1][:40])
    check("画风、外貌、服装、显著特征都进事实清单",
          all(k in lines[1] for k in ("3D卡通渲染", "明黄色", "深紫色长袍", "小尖角")), lines[1])
    check("低置信字段点名提醒别硬填", "不要硬填" in lines[1])
    check("读不出来的图不编内容", "读取失败" in RD.describe_lines([RD.DescribedCard(image="x.png")])[0])

    _sys, user = S1.build_messages("某创意", shot_count=3, ref_observations=lines)
    check("事实清单进了 user 消息", all(l in user for l in lines))
    check("标明是既成事实", "既成事实" in user)
    check("要求原样抄回外貌与服装", "原样抄回" in user)
    check("非人角色性别年龄写「无」", '都写 "无"' in user)
    check("禁止人类化身体动作", "三头身" in user and "修长的手指" in user)
    check("禁止新增角色", "不要新增角色" in user)
    # 这两句原来互斥：加了事实清单就不该再说"角色由你设计"，否则模型先按创意编一遍
    check("有事实清单时不再说「角色由你设计」", "角色由你设计" not in user)
    _sys2, user2 = S1.build_messages("某创意", shot_count=3)
    check("没图时仍是老行为（由模型设计角色）", "角色由你设计" in user2 and "既成事实" not in user2)

    # 非人类角色拼进提示词时不该出现「奶龙（无，无）」
    from models import Character as _C  # noqa: E402
    loong = _C(char_id="char_02", name="奶龙", gender="无", age_range="无",
               appearance="明黄色", wardrobe="紫袍", source="inferred")
    check("gender/age 都是「无」时只念名字", loong.subject_phrase() == "奶龙", loong.subject_phrase())
    human_c = _C(char_id="char_01", name="豆包", gender="女", age_range="18-22岁",
                 appearance="x", wardrobe="y", source="brief")
    check("正常年龄仍是形容词结构", human_c.subject_phrase() == "女18-22岁的豆包")

    clean()
    print()
    print("=" * 64)
    print(f"结果：{passed} 通过 / {failed} 失败")
    print("=" * 64)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
