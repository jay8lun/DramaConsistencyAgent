"""models.py 数据契约验证：A.5 的 7 处补丁 + 4.7 的 7 条校验。

对应 14.4-A 的 6 个用例，但这里用独立脚本先跑一遍，方便单步验收；
正式用例在 M1-S3 写成 pytest。

用法：python tests/verify_models.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
import models as M  # noqa: E402
from pydantic import ValidationError  # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def expect_error(name, fn, keyword=""):
    """断言 fn() 抛 ValidationError，且错误信息含 keyword。"""
    try:
        fn()
    except ValidationError as exc:
        msg = str(exc)
        ok = keyword in msg
        check(name, ok, f"抛错；含「{keyword}」" if ok else f"抛错但信息不含「{keyword}」")
        return
    check(name, False, "竟然通过了")


BASE_CHAR = dict(
    char_id="char_01", name="测试", age_range="26-30岁", gender="女",
    appearance="黑色齐肩直发，鹅蛋脸，杏眼", wardrobe="白色棉质衬衫",
)


def char(**kw):
    d = dict(BASE_CHAR)
    d.setdefault("source", "brief")
    d.update(kw)
    return M.Character(**d)


def shot(**kw):
    d = dict(shot_id="shot_01", index=1, scene_id="scene_01", camera="中景", action="转身",
             character_ids=["char_01"])
    d.update(kw)
    return M.Shot(**d)


def script(shots, characters=None):
    return M.Script(
        title="t", logline="l", style=M.VisualStyle(style_prompt="3D 渲染"),
        characters=characters or [char()],
        scenes=[M.Scene(scene_id="scene_01", name="教室", description="空教室",
                        lighting="冷白顶灯", time_of_day="夜晚")],
        shots=shots,
    )


print("=" * 56)
print("M1-S1 · A.5 的 7 处补丁")
print("=" * 56)
for n, ok in [
    ("Character.source", "source" in M.Character.model_fields),
    ("Character.truth_image", "truth_image" in M.Character.model_fields),
    ("Character.low_confidence_fields", "low_confidence_fields" in M.Character.model_fields),
    ("AttemptLog 类存在", hasattr(M, "AttemptLog")),
    ("Shot.attempt_log", "attempt_log" in M.Shot.model_fields),
    ("Shot.video_prompt", "video_prompt" in M.Shot.model_fields),
    ("Shot.environment_only", "environment_only" in M.Shot.model_fields),
]:
    check(n, ok)

print("\n" + "=" * 56)
print("M1-S2 · 4.7 的校验 + config.PLATFORM_TIERS")
print("=" * 56)

check("config.PLATFORM_TIERS 存在", hasattr(config, "PLATFORM_TIERS"),
      f"= {getattr(config, 'PLATFORM_TIERS', None)}")
# 档位表由 .env 的 VIDEO_SUPPORTED_SECONDS 决定（2026-10-29 起默认只有 5），
# **不能把具体档位写进断言**——否则换视频模型时测试会假失败，而它想验的是
# 「Schema 会照着配置表校验」，不是「配置表里有 10」。
_tiers = tuple(getattr(config, "PLATFORM_TIERS", ()))
check("档位表非空且都已排序", bool(_tiers) and list(_tiers) == sorted(_tiers), f"= {_tiers}")

# 4.7 #6 · duration 必须落在档位
print("\n-- duration 档位（14.4-A test_shot_duration_must_fit_platform_tier）--")
# 遍历**当前配置的**档位（而不是写死 5 和 10），并检验非档位值一律被拒。
for v in _tiers:
    try:
        s = shot(duration=v)
        check(f"duration={v:g} 通过（当前档位）", s.duration == v)
    except ValidationError as exc:
        check(f"duration={v:g} 通过（当前档位）", False, str(exc.errors()[0]["msg"])[:50])
for bad in (3.0, 7.0, 9.0):
    if any(abs(bad - x) < 1e-6 for x in _tiers):
        continue
    expect_error(f"duration={bad:g} 抛错（不在当前档位）", lambda b=bad: shot(duration=b), "不在平台档位")

# 4.7 #1 · shots 非空
print("\n-- shots 非空 --")
expect_error("空分镜表抛错", lambda: script([]), "分镜表不能为空")

# 4.7 #2 · 角色 vs 空镜互斥
print("\n-- 角色 / 空镜互斥（14.4-A test_shot_rejects_empty_character_refs）--")
expect_error("character_ids=[] 且 environment_only=False → 抛错",
             lambda: script([shot(character_ids=[], environment_only=False)]),
             "environment_only")
try:
    script([shot(character_ids=[], environment_only=True)])
    check("同一分镜置 environment_only=True → 通过", True)
except ValidationError as exc:
    check("同一分镜置 environment_only=True → 通过", False, str(exc.errors()[0]["msg"])[:50])
expect_error("两者都有 → 抛错（互斥）",
             lambda: script([shot(character_ids=["char_01"], environment_only=True)]),
             "互斥")

# 4.7 #3 · 角色引用存在
print("\n-- 引用完整性 --")
expect_error("引用不存在的 char_id → 抛错",
             lambda: script([shot(character_ids=["char_99"])]), "不存在的角色")
expect_error("引用不存在的 scene_id → 抛错",
             lambda: script([shot(scene_id="scene_99")]), "不存在的场景")

# 4.7 #5 · index 连续
print("\n-- index 连续（14.4-A test_index_must_be_continuous）--")
expect_error("index 跳号 1,2,4 → 抛错",
             lambda: script([
                 shot(shot_id="shot_01", index=1), shot(shot_id="shot_02", index=2),
                 shot(shot_id="shot_03", index=4)]), "连续递增")
expect_error("index 从 0 起 → 抛错",
             lambda: script([shot(index=0)]), "连续递增")

# 4.7 #7 · source=image 必须带 truth_image
print("\n-- source / truth_image（14.4-A test_image_source_requires_truth_image）--")
expect_error("source=image 且 truth_image=None → 抛错",
             lambda: char(source="image"), "truth_image")
c = char(source="brief")
check("source=brief 允许 truth_image 为空", c.truth_image is None)
c2 = char(source="inferred", low_confidence_fields=["发型", "肤色"])
check("source=inferred 可带低置信字段", c2.low_confidence_fields == ["发型", "肤色"])
expect_error("source=photo（非法枚举）→ 抛错",
             lambda: char(source="photo"), "source")
expect_error("不传 source → 抛错", lambda: M.Character(**BASE_CHAR), "source")

print("\n" + "=" * 56)
if fails:
    print(f"失败 {len(fails)} 项：")
    for f in fails:
        print("   -", f)
    sys.exit(1)
print("全部通过")
