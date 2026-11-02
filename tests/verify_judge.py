"""M1-S3b 自检：prompt_builder.py + judge.py。

对齐第 14.4 节：
  - B 组 `pipeline/judge.py` 的 6 条用例
  - E 组 `pipeline/prompt_builder.py` 的 4 条用例

另外额外验三件计划书要求、但没有单独用例的事：
  - `judge.ELEMENTS` 与 `providers/vlm.ELEMENT_HINTS` 的键一致（防止两处漂移）
  - 重取时 diff 增量不被截断逻辑吞掉（7.1 硬规则第 6 条 + 运行时炸点）
  - 三条分数线的边界值（PASS、PASS-1、FLOOR、FLOOR-1，全部从常量推导）

全部离线，不发任何网络请求。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import Character, Scene, Shot, VisualStyle  # noqa: E402
from pipeline import judge as J  # noqa: E402
from pipeline.prompt_builder import (  # noqa: E402
    CastConflict,
    Prompt,
    assemble_image_prompt,
    assemble_video_prompt,
    cast_conflicts,
    find_appearance_leaks,
    snap_duration,
    truncate,
)

PASS = 0
FAIL = 0
FAILED_NAMES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILED_NAMES.append(name)
        print(f"  FAIL  {name}  {detail}")


# ---------------------------------------------------------------- 固定装置

STYLE = VisualStyle(style_prompt="国风水墨，冷色调，电影感")
CHAR = Character(
    char_id="char_01", name="林晚", age_range="26-30岁", gender="女",
    appearance="黑色齐肩直发，鹅蛋脸，细长眼，白皙肤色，偏瘦",
    wardrobe="深灰色羊毛大衣",
    source="brief",
)
SCENE = Scene(
    scene_id="scene_01", name="老旧电梯", description="狭窄电梯间，金属壁有划痕",
    lighting="顶部冷白灯，侧向硬光", time_of_day="深夜",
)
SHOT = Shot(
    shot_id="shot_01", index=1, scene_id="scene_01", duration=5.0,
    camera="中景，缓慢推近", action="缓缓转头看向左侧",
    character_ids=["char_01"], dialogue="你终于来了。", emotion="警惕",
)

APPEARANCE_SEGS = ["黑色齐肩直发", "鹅蛋脸", "细长眼", "白皙肤色", "偏瘦"]


def ok_checks(**override) -> list[dict]:
    base = [{"element": e, "matched": True, "expected": "见角色卡", "actual": "一致"} for e in J.ELEMENTS]
    for item in base:
        if item["element"] in override:
            item["matched"] = override[item["element"]]
    return base


def main() -> int:
    print("=" * 64)
    print("[E] prompt_builder：图像提示词（14.4-E 第 1 条）")
    print("=" * 64)
    img = assemble_image_prompt(SHOT, [CHAR], SCENE, STYLE)
    missing = [s for s in APPEARANCE_SEGS if s not in img.text]
    check("test_image_prompt_contains_appearance",
          not missing, f"缺 {missing}；实际：{img.text}")
    check("图像提示词含动作", SHOT.action in img.text)
    check("图像提示词含服装", CHAR.wardrobe in img.text)
    check("图像提示词含场景光线", "顶部冷白灯" in img.text)
    check("negative 单独给出（8.3 两种格式都出）", img.negative == STYLE.negative_prompt)
    check("flat() 把负面词并进一条", "negative:" in img.flat())

    print()
    print("=" * 64)
    print("[E] prompt_builder：视频提示词只写运动（14.4-E 第 2 条）")
    print("=" * 64)
    vid = assemble_video_prompt(SHOT, [CHAR])
    leaks = find_appearance_leaks(vid.text, [CHAR])
    check("test_video_prompt_must_not_describe_appearance",
          not leaks, f"泄漏 {leaks}；实际：{vid.text}")
    check("视频提示词不含任何 appearance 段",
          not any(s in vid.text for s in APPEARANCE_SEGS), vid.text)
    check("视频提示词不含服装", CHAR.wardrobe not in vid.text, vid.text)
    check("视频提示词不含场景描述", "电梯间" not in vid.text, vid.text)
    check("视频提示词含运动与运镜", SHOT.action in vid.text and "推近" in vid.text, vid.text)
    check("景别（中景）不进视频提示词——它是构图，首帧已定", "中景" not in vid.text, vid.text)

    dirty = SHOT.model_copy(update={"action": "黑色长发被风扬起，缓缓转头"})
    dvid = assemble_video_prompt(dirty, [CHAR])
    check("action 里写了外貌 → 记进 violations 而不是静默放过",
          bool(dvid.violations) and "黑色" in "".join(dvid.violations), f"{dvid.violations}")
    check("violations 不阻断生成（不抛异常、仍返回提示词）", isinstance(dvid, Prompt) and bool(dvid.text))

    legal = SHOT.model_copy(update={"action": "她脱下外套，转身走向门口"})
    lvid = assemble_video_prompt(legal, [CHAR])
    check("合法动作「脱下外套」不被误判为外貌泄漏",
          not lvid.violations, f"{lvid.violations}")

    print()
    print("=" * 64)
    print("[E] prompt_builder：时长按档位取整（14.4-E 第 3 条）")
    print("=" * 64)
    s7 = snap_duration(7.0)
    # 期望值从当前档位表算，不写死 10：2026-10-29 档位从 (5,10) 收窄为 (5,)，
    # 7 秒现在应向上取到 5（唯一档位）。用例验的是「结果必须落在档位表内」，
    # 不是「必须等于 10」。
    from config import PLATFORM_TIERS
    _snapped_7 = min((x for x in PLATFORM_TIERS if x >= 7.0), default=max(PLATFORM_TIERS))
    check("test_duration_snaps_to_platform_tiers（7s → 档位值）",
          s7.seconds in PLATFORM_TIERS and s7.seconds == _snapped_7, f"得到 {s7.seconds}")
    check("返回的是「实际采用值」而非原值", s7.seconds != s7.requested, f"{s7.seconds} vs {s7.requested}")
    check("已是档位值时不改动", snap_duration(5.0).seconds == 5.0)
    check("超过最大档位标记 needs_split", snap_duration(12.0).needs_split is True)
    check("12s 仍返回档位值（不发明新档位）",
          snap_duration(12.0).seconds in PLATFORM_TIERS, f"得到 {snap_duration(12.0).seconds}")
    check("3s 向上取到 5s（向下截会切掉动作）", snap_duration(3.0).seconds == 5.0)

    print()
    print("=" * 64)
    print("[E] prompt_builder：超长截断保留主体与动作（14.4-E 第 4 条）")
    print("=" * 64)
    long_shot = SHOT.model_copy(update={"camera": "中景，缓慢推近，" + "极其克制的运镜，" * 20})
    long_prompt = assemble_image_prompt(long_shot, [CHAR], SCENE, STYLE, max_chars=120)
    must = [SHOT.action, *APPEARANCE_SEGS, CHAR.wardrobe]
    missing = [k for k in must if k not in long_prompt.text]
    check("test_over_length_prompt_truncated_keeping_subject_and_action",
          long_prompt.truncated and len(long_prompt.text) <= 120 and not missing,
          f"len={len(long_prompt.text)} missing={missing}")
    check("截掉的是修饰词（有 dropped 记录）", bool(long_prompt.dropped), f"{long_prompt.dropped}")
    check("未删到无法收敛（still_over=False）", long_prompt.still_over is False)

    # 运行时炸点：diff 增量不能被截断逻辑误删（7.1 硬规则第 6 条）
    delta = "发型改为黑色齐肩直发"
    dprompt = assemble_image_prompt(long_shot, [CHAR], SCENE, STYLE, delta=delta, max_chars=120)
    check("重取时 diff 增量在截断后仍保留",
          delta in dprompt.text, f"增量被吞；实际：{dprompt.text}")
    check("增量与原始提示词可区分（第 2 次 ≠ 第 1 次）",
          dprompt.text != long_prompt.text)
    check("截断函数的 protect 参数确实生效",
          delta in truncate("a，" * 50 + delta, protect=(delta,), max_chars=40).text)

    # 真实增量里天然带「，」：`发型改为X（差异：Y，Z）`。若按「，」切分就会切成两半，
    # 得到「（差异：发型长度不符」这种残缺指令——第一版实测出现过，这里钉住。
    real_delta = "发型改为黑色齐肩直发（差异：发型长度不符，实际扎起）"
    rp = assemble_image_prompt(long_shot, [CHAR], SCENE, STYLE, delta=real_delta, max_chars=150)
    check("增量内部含「，」时不被从中间切断",
          real_delta in rp.text, f"实际：{rp.text}")
    check("括号成对（没有留下半个指令）",
          rp.text.count("（") == rp.text.count("）"), rp.text)

    # 2026-11-02：镜头提示词也必须显式要求"以角色参考图为准"。
    # 只把定妆照当 references 传进去、文字里不提，模型会把图和角色卡各采纳一半。
    ref_prompt = assemble_image_prompt(SHOT, [CHAR], SCENE, STYLE, has_ref=True)
    check("带参考图时镜头提示词要求以参考图为准",
          "以参考图为准" in ref_prompt.text, ref_prompt.text[:60])
    check("不带参考图时不写这句（不指向不存在的东西）",
          "参考图为准" not in assemble_image_prompt(SHOT, [CHAR], SCENE, STYLE).text)
    # 这句是指令，不是修饰词：超长截断时必须活下来，否则等于"带了图却不要求照着画"。
    long_ref = assemble_image_prompt(long_shot, [CHAR], SCENE, STYLE,
                                     has_ref=True, max_chars=120)
    check("截断后「以参考图为准」仍在",
          long_ref.truncated and "以参考图为准" in long_ref.text,
          f"truncated={long_ref.truncated} len={len(long_ref.text)}")

    # 2026-11-02：age_range 是人可编辑的自由文本。卡通角色没有年龄，人就会填
    # "不是真实人无年龄段"，当形容词拼出来是"女不是真实人无年龄段的豆包"——
    # 一句模型看不懂的话，而且会被抄进每一镜的提示词。
    check("常规年龄仍用模型最熟的形容词结构",
          CHAR.subject_phrase() == "女26-30岁的林晚", CHAR.subject_phrase())
    weird = CHAR.model_copy(update={"age_range": "不是真实人无年龄段"})
    check("非常规年龄写法退成括注",
          weird.subject_phrase() == "林晚（女，不是真实人无年龄段）", weird.subject_phrase())
    weird_shot = SHOT.model_copy(update={"character_ids": [weird.char_id]})
    wp = assemble_image_prompt(weird_shot, [weird], SCENE, STYLE)
    check("镜头提示词里也不会拼成病句",
          "不是真实人无年龄段的林晚" not in wp.text, wp.text[:70])
    check("空年龄退成括注（不留「女的林晚」这种半句）",
          CHAR.model_copy(update={"age_range": ""}).subject_phrase() == "林晚（女）",
          CHAR.model_copy(update={"age_range": ""}).subject_phrase())

    # 动作句里点到场上没有的人 → 模型替他凭空造一个，且不受任何参考图约束。
    # 2026-11-02 的实录：只列了豆包的镜头，动作写"凑近男生的手臂"，出来一条写实人类胳膊；
    # 而自检比对的是脸和服装，"多出一个人"它根本看不到。
    male = CHAR.model_copy(update={"char_id": "char_02", "name": "陈默", "gender": "男"})
    bad_shot = SHOT.model_copy(update={"action": "凑近男生的手臂", "character_ids": ["char_01"]})
    blocked, msg = False, ""
    try:
        assemble_image_prompt(bad_shot, [CHAR], SCENE, STYLE)
    except CastConflict as exc:
        blocked, msg = True, str(exc)
    check("动作提到场外的人 → 拼提示词时直接拦", blocked and "男生" in msg, msg[:60])
    check("拒绝理由说清「会画成没有参考图约束的人」",
          blocked and "参考图" in msg, msg[:90])
    good_shot = bad_shot.model_copy(update={"character_ids": ["char_01", "char_02"]})
    check("把对方加进 character_ids 就放行",
          "男生的手臂" in assemble_image_prompt(good_shot, [CHAR, male], SCENE, STYLE).text)
    check("场上有同性角色时不拦（称谓能被理解成在指他）",
          "男孩走过" in assemble_image_prompt(
              SHOT.model_copy(update={"action": "男孩走过",
                                      "character_ids": ["char_01", "char_02"]}),
              [CHAR, male], SCENE, STYLE).text)
    empty_ok = True
    try:
        assemble_image_prompt(
            SHOT.model_copy(update={"action": "走廊空无一人", "character_ids": [],
                                    "environment_only": True}), [CHAR], SCENE, STYLE)
    except CastConflict:
        empty_ok = False
    check("空镜不参与这条校验", empty_ok)
    fake_script = SimpleNamespace(shots=[bad_shot], characters=[CHAR])
    problems = cast_conflicts(fake_script)
    check("全片预检一次列出冲突（含镜头号与场上角色）",
          len(problems) == 1 and "shot_01" in problems[0] and "林晚" in problems[0],
          problems[0] if problems else "空")

    print()
    print("=" * 64)
    print("[B] judge：三条分数线（14.4-B 前 3 条）")
    print("=" * 64)
    d = J.judge({"score": 88, "checks": ok_checks(), "diff": ""}, characters=[CHAR])
    check("test_score_above_threshold_passes（88 → pass）",
          d.action is J.Action.PASS and d.prompt_delta is None and d.qa.passed,
          f"{d.action} delta={d.prompt_delta}")
    check("pass 不进入重试（无增量）", d.prompt_delta is None)

    d = J.judge({"score": 70, "checks": ok_checks(眉眼=False), "diff": "眉眼偏窄，实际眼距更开"},
                characters=[CHAR], retries_used=0, max_retries=2)
    check("test_score_mid_range_triggers_targeted_retry（70 → retry 且带 diff）",
          d.action is J.Action.RETRY and bool(d.qa.diff), f"{d.action} diff={d.qa.diff}")
    check("retry 必带 prompt_delta（7.1 硬规则第 6 条）", bool(d.prompt_delta), f"{d.prompt_delta}")

    d = J.judge({"score": 50, "checks": ok_checks(发型=False, 脸型=False), "diff": "整体不像"},
                characters=[CHAR], retries_used=2, max_retries=2)
    check("test_score_below_floor_marks_degraded_after_limit（50 且已到上限 → degraded，不是 failed）",
          d.action is J.Action.DEGRADED and d.qa.degraded and not d.qa.passed,
          f"{d.action}")

    print()
    print("=" * 64)
    print("[B] judge：diff → 增量、重试预算、脏数据兜底")
    print("=" * 64)
    d = J.judge({"score": 72, "checks": ok_checks(发型=False), "diff": "发型长度不符，实际偏长且扎起"},
                characters=[CHAR])
    delta = d.prompt_delta or ""
    check("test_diff_is_translated_to_prompt_delta（语义关键词命中）",
          "发型" in delta and "黑色齐肩直发" in delta, f"增量：{delta}")
    check("增量不是全文照抄 diff（是改写，不是复制）", delta != d.qa.diff, f"{delta}")
    check("增量带上原始差异说明", "发型长度不符" in delta, delta)

    check("test_retry_budget_is_enforced（第 3 次请求被拒）",
          J.can_retry(retries_used=1, max_retries=2) is True
          and J.can_retry(retries_used=2, max_retries=2) is False)
    d = J.judge({"score": 70, "checks": ok_checks(肤色=False), "diff": "肤色偏暗"},
                characters=[CHAR], retries_used=2, max_retries=2)
    check("超限返回明确信号而非静默继续",
          d.action is J.Action.DEGRADED and "用尽" in d.reason, f"{d.action} {d.reason}")

    d = J.judge("这不是 JSON，只是一段废话", characters=[CHAR])
    check("test_malformed_vlm_json_falls_back_safely（非 JSON → 降级不抛异常）",
          d.action is J.Action.DEGRADED and d.raw_text and "废话" in d.raw_text,
          f"{d.action} raw={d.raw_text!r}")

    d = J.judge({"score": 90, "diff": ""}, characters=[CHAR])
    check("缺 checks → 即便 90 分也降级（分数无从校验）",
          d.action is J.Action.DEGRADED and "checks" in d.reason, f"{d.action} {d.reason}")

    d = J.judge({"score": 90, "checks": ok_checks()[:3], "diff": ""}, characters=[CHAR])
    check("部分要素缺失 → 补齐为未判定并记问题，不静默丢",
          len(d.qa.checks) == 5 and "缺少要素" in d.qa.diff, f"{len(d.qa.checks)} {d.qa.diff}")

    d = J.judge({"score": 87, "checks": [{"element": "发型", "matched": None,
                                          "expected": "黑色齐肩直发", "actual": ""}] + ok_checks()[1:]},
                characters=[CHAR])
    check("matched=null（漏判）按不一致处理并留痕",
          "漏判" in d.qa.diff and J.failing_elements(d.qa) == ["发型"],
          f"{d.qa.diff} / {J.failing_elements(d.qa)}")

    d = J.judge({"score": "八十八", "checks": ok_checks()}, characters=[CHAR])
    check("score 非数字 → 降级且不留未捕获异常", d.action is J.Action.DEGRADED, d.reason)

    fenced = "```json\n" + json.dumps({"score": 92, "checks": ok_checks(), "diff": ""}, ensure_ascii=False) + "\n```"
    d = J.judge(fenced, characters=[CHAR])
    check("```json 包裹的返回能解析（网关常见形态）", d.action is J.Action.PASS, f"{d.action} {d.reason}")

    # expected 是占位文本时必须回落到角色卡原文，否则增量会变成「发型改为见角色卡」
    placeholders = [{"element": e, "matched": e != "发型", "expected": "见角色卡", "actual": "不一致"}
                    for e in J.ELEMENTS]
    d = J.judge({"score": 74, "checks": placeholders, "diff": "发型不符"}, characters=[CHAR])
    check("expected 是占位「见角色卡」时回落角色卡原文",
          "发型" in (d.prompt_delta or "") and "黑色齐肩直发" in (d.prompt_delta or ""),
          f"{d.prompt_delta}")

    print()
    print("=" * 64)
    print("[B] judge：边界值与一致性")
    print("=" * 64)
    # 边界值从常量推导，不写死——换判定模型后线会变（85→75），
    # 写死会让自检在阈值变更时给出假失败。
    _p, _f = int(J.PASS_LINE), int(J.FLOOR_LINE)
    check(f"{_p} 分 → pass（含边界）", J.decide(_p, retries_used=0, max_retries=2)[0] is J.Action.PASS)
    check(f"{_p - 1} 分 → retry", J.decide(_p - 1, retries_used=0, max_retries=2)[0] is J.Action.RETRY)
    check(f"{_f} 分 → retry（含边界）", J.decide(_f, retries_used=0, max_retries=2)[0] is J.Action.RETRY)
    check(f"{_f - 1} 分 → degraded", J.decide(_f - 1, retries_used=0, max_retries=2)[0] is J.Action.DEGRADED)
    check(f"{_p - 1} 分但预算已尽 → degraded",
          J.decide(_p - 1, retries_used=2, max_retries=2)[0] is J.Action.DEGRADED)

    from providers.vlm import ELEMENT_HINTS
    check("judge.ELEMENTS 与 providers/vlm.ELEMENT_HINTS 键一致（防止两处漂移）",
          tuple(ELEMENT_HINTS) == J.ELEMENTS, f"{tuple(ELEMENT_HINTS)} vs {J.ELEMENTS}")

    print()
    print("=" * 64)
    print(f"结果：{PASS} 通过 / {FAIL} 失败")
    if FAILED_NAMES:
        print("失败项：")
        for n in FAILED_NAMES:
            print("  -", n)
    print("=" * 64)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
