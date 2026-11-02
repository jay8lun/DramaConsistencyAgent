"""提示词组装与平台适配（第 8 章）。

**两类提示词语法几乎相反，必须由两个独立函数生成**（8.2）：
- 图像：写全画面——主体 / 服装 / 动作 / 场景 / 光线 / 构图，缺一项模型就瞎猜
- 视频：只写运动——写外貌会让模型重新构图，脸开始漂

时长取整与超长截断也在这里（8.3），**不在模型层做**（4.7.1）：模型层只接受档位值，
否则会出现「Schema 以为已合规，提示词里实际写的是 7s」。

平台参数一律标注「待验证」（8.3 硬要求：不得装作知道）。本模块没有任何网络调用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from config import PLATFORM_TIERS, PROMPT_MAX_CHARS
from models import Character, Scene, Shot, VisualStyle

#: 分句符。提示词按「，」分句，截断的粒度就是句——不切词，切词会切出半个词。
_SEP = "，"
_SPLIT_RE = re.compile(r"[，,、;；]")
#: 掩码哨兵：受保护串内部的分隔符临时换成它，拼回后还原（见 `truncate`）。
_MASK = "\u0001"


def _mask(text: str, guards: Sequence[str]) -> str:
    """把受保护串内部的分句符换成哨兵，让整串在切分时保持为一个整体。"""
    out = text
    for k in guards:
        if k and _SEP in k and k in out:
            out = out.replace(k, k.replace(_SEP, _MASK))
    return out

#: 运镜动词。camera 字段常写成「中景，缓慢推近」，景别属于构图（首帧已定），
#: 只有运动部分该进视频提示词（8.2 第 2 条禁止）。
_MOTION_HINTS = ("推", "拉", "摇", "移", "跟", "升", "降", "旋", "固定", "手持", "晃", "甩")

#: 外貌泄漏检测用的封闭词表。视频提示词里出现这些词就说明写错了。
#: 用词表而不是「全部 appearance 分词」，是因为：「黑色齐肩直发」整段不会出现在
#: 动作描述里，但「黑色长发」会——只比对整段等于漏检。
#:
#: **刻意只收「颜色 + 毛发」**，不收服装名词：动作可以是「她脱下外套」，
#: 那是运动而不是外貌描述，把「外套」列进词表会把合法镜头判成违规。
#: 服装的整段复制粘贴由 `appearance_terms` 逐段比对兜住。
_LEAK_LEXICON: tuple[str, ...] = (
    "黑色", "白色", "红色", "金色", "银色", "棕色", "灰色", "蓝色", "绿色", "黄色", "粉色", "紫色",
    "红棕", "深灰", "浅灰", "米白", "藏青",
    "头发", "长发", "短发", "发型", "马尾", "刘海", "寸头", "丸子头",
)


@dataclass(frozen=True)
class Prompt:
    """一条可直接投喂平台的提示词，连同它的负面词与截断痕迹。

    `truncated` / `dropped` / `still_over` 必须能被上层读到并写进报告：
    静默截断会让人以为提示词完整，实际少了内容（9.5「效果不达标必须如实记录」）。
    """

    text: str
    negative: str = ""
    truncated: bool = False
    dropped: tuple[str, ...] = ()
    still_over: bool = False
    #: 视频提示词里检出外貌描述的位置。非空即说明上游数据有问题，
    #: **不在这里自动删词**——删了会让人以为提示词是对的，应回 S1 修 action。
    violations: tuple[str, ...] = ()

    def __str__(self) -> str:  # 方便直接 print / 断言 "关键词 in prompt"
        return self.text

    def flat(self) -> str:
        """单串格式：负面词塞进同一条 prompt（8.3「两种格式都出」）。"""
        return f"{self.text}\nnegative: {self.negative}" if self.negative else self.text


@dataclass(frozen=True)
class Snapped:
    seconds: float
    requested: float
    note: str = ""
    needs_split: bool = False


# ---------------------------------------------------------------- 时长档位


def snap_duration(value: float, *, tiers: Sequence[float] = PLATFORM_TIERS) -> Snapped:
    """把任意秒数取整到平台档位（8.3），并返回**实际采用值**。

    取整方向定为**向上**：向下截会切掉动作（分镜写 7s 通常因为动作需要 7s），
    向上的代价只是多几秒素材，而素材多出来可以剪掉，少了补不回来。
    超过最大档位不取整——那属于「超时拆镜」，交给上层处理，这里只标记 `needs_split`。
    """
    ordered = sorted(tiers)
    if not ordered:
        raise ValueError("平台档位表为空，无法取整")

    for t in ordered:
        if abs(value - t) < 1e-6:
            return Snapped(seconds=t, requested=value, note="已是档位值")
    for t in ordered:
        if value < t:
            return Snapped(seconds=t, requested=value, note=f"{value:g}s 向上取整到 {t:g}s")
    top = ordered[-1]
    return Snapped(
        seconds=top,
        requested=value,
        note=f"{value:g}s 超过最大档位 {top:g}s，需拆镜（先按 {top:g}s 生成）",
        needs_split=True,
    )


# ---------------------------------------------------------------- 术语工具

#: 有角色参考图时必须写进镜头提示词的优先级声明（2026-11-02 补）。
#:
#: 与 `s2_anchor.REF_LOCK` 是同一件事的两个场景：定妆照拿用户原图当参考，
#: 镜头拿定妆照当参考。**两处都缺这句话**，模型就把"参考图"和"文字角色卡"
#: 当成两个独立线索，各采纳一部分——2026-11-02 的现场结果：参考图是黄色卡通
#: 吉祥物、角色卡写"人类男性、肤色白皙"，出来一张写实真人 + 一片黄脖子。
#: 两边都没满足，得到一个谁都没要求过的第三种东西。
#:
#: 关键不是"提到参考图"，而是**定优先级**：冲突时听图的。只说"参考参考图"，
#: 模型会继续折中。
CAST_LOCK = (
    "画面人物的长相以随附的角色定妆照为准（多位角色按出场顺序对应），"
    "与本段文字冲突时以参考图为准：脸型、发型与发色、肤色、体型、服装配色保持一致"
)

#: 上面这句属于"绝不能丢的指令"，超长截断时按 protect 处理（与 diff 增量同级）。


#: 动作句里可能出现的"人类称谓"。它们不是角色名，提示词里没有任何东西能把它们
#: 绑到某个角色身上——**包括参考图**。2026-11-02 的现场：`shot_02` 的角色列表
#: 只有豆包，动作写"凑近男生的手臂"，模型就理直气壮地画了一条写实人类的胳膊。
#: 那一镜的钱照付、画面照出、自检照过，因为"多出一个人"从来不在判定要素里。
#:
#: 只收明确的名词，不收"他/她"：代词在这类句子里太密（"他胳膊上""她手里"），
#: 一律拦会把人烦到直接关掉校验。
PRONOUN_TERMS: dict[str, tuple[str, ...]] = {
    "男": ("男生", "男孩", "男人", "少年", "男子"),
    "女": ("女孩", "女生", "女人", "少女", "女子"),
}


class CastConflict(RuntimeError):
    """动作句里出现了场上没有对应角色的人类称谓。

    这是**业务错误**，重试不会变好——必须人去改分镜或补 `character_ids`。
    用独立异常类型而不是 `ProviderError`：后者会触发技术重试，白等三轮。
    """


def find_offcast_terms(action: str, cast: Sequence[Character]) -> list[str]:
    """找出动作句里"没人在场"的称谓。

    判据是性别而不是名字：句子里出现"男生"，而场上没有任何 `gender` 含"男"的
    角色 → 那个"男生"只能由模型凭空造，且不受任何参考图约束。
    """
    text = action or ""
    hits: list[str] = []
    for gender, words in PRONOUN_TERMS.items():
        if any(gender in (c.gender or "") for c in cast):
            continue  # 场上有这个性别的角色，称谓可以被理解为在指他
        for w in words:
            if w in text:
                hits.append(w)
    return hits


def cast_conflicts(script) -> list[str]:
    """全片扫一遍"动作句提到的人不在场上"，返回人话清单。

    放在**开跑之前**而不是逐镜生成时：逐镜检查的话，前三镜的钱已经花完，
    第四镜才报错停下；而且改一镜要重跑一次才知道下一镜也有同样问题。
    一次列全，人改一轮就能过。
    """
    problems: list[str] = []
    for shot in script.shots:
        if shot.environment_only:
            continue
        cast = [c for c in script.characters if c.char_id in shot.character_ids]
        if not cast:
            continue
        bad = find_offcast_terms(shot.action, cast)
        if bad:
            names = "、".join(c.name for c in cast)
            problems.append(f"{shot.shot_id}：动作有「{'、'.join(bad)}」，场上只有 {names}")
    return problems


def appearance_terms(characters: Iterable[Character]) -> tuple[str, ...]:
    """角色卡里所有可视特征段，用于泄漏检测与截断保护。"""
    out: list[str] = []
    for c in characters:
        for field in (c.appearance, c.wardrobe):
            for seg in _SPLIT_RE.split(field or ""):
                seg = seg.strip()
                if len(seg) >= 2:
                    out.append(seg)
    return tuple(dict.fromkeys(out))


def find_appearance_leaks(prompt: str, characters: Iterable[Character]) -> list[str]:
    """视频提示词里出现外貌/服装描述的位置（8.2 三条禁止的可执行判据）。"""
    hits: list[str] = []
    for term in appearance_terms(characters):
        if term in prompt:
            hits.append(term)
    for word in _LEAK_LEXICON:
        if word in prompt and word not in hits:
            hits.append(word)
    return hits


def _pronoun(characters: Sequence[Character]) -> str:
    if not characters:
        return "画面"
    g = characters[0].gender or ""
    if "女" in g:
        return "她"
    if "男" in g:
        return "他"
    return "人物"


def _camera_motion(camera: str) -> str:
    parts = [p.strip() for p in _SPLIT_RE.split(camera or "") if p.strip()]
    motion = [p for p in parts if any(h in p for h in _MOTION_HINTS)]
    return _SEP.join(motion) if motion else "镜头固定"


# ---------------------------------------------------------------- 截断


@dataclass(frozen=True)
class Truncation:
    text: str
    dropped: tuple[str, ...] = ()
    truncated: bool = False
    still_over: bool = False


def truncate(
    text: str,
    *,
    must_keep: Sequence[str] = (),
    protect: Sequence[str] = (),
    max_chars: int = PROMPT_MAX_CHARS,
) -> Truncation:
    """超长截断：**只删修饰词，保留主体与动作**（8.3 / 14.4-E 第 4 条）。

    实现方式按 14.4-E 固定下来的口径：不判「语义是否保留」（那要 LLM，违反 14.7），
    也不靠「删形容词白名单」（脆弱）。做法是**按句保留 + 关键词命中**：
    含 `must_keep` / `protect` 关键词的句子受保护，其余从尾部往前删。
    这样「保留了主体与动作」这件事可以直接用 `关键词 in 结果` 断言。

    `protect` 与 `must_keep` 分开：`must_keep` 是要断言的业务关键词（主体/动作/服装），
    `protect` 是**绝不能丢的指令**（diff 增量）。两者都保护，但语义不同——
    增量被截掉等于重试不带修正，直接违反 7.1 硬规则第 6 条。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正")
    if len(text) <= max_chars:
        return Truncation(text=text)

    guards = tuple(k for k in (*must_keep, *protect) if k)

    # 受保护的串必须**整体不可分**。增量里天然带「，」（`发型改为X（差异：Y，Z）`），
    # 若直接按「，」切分，它会被切成两半：前半句留在提示词里、后半句被当修饰词删掉，
    # 结果是一条残缺指令（实测得到过 `（差异：发型长度不符` 这种半个括号）。
    # 做法是把串内分隔符临时替换成哨兵字符，整串因此成为一个分句，拼回后再还原。
    pairs = tuple((k, k.replace(_SEP, _MASK)) for k in guards)

    def guarded(seg: str) -> bool:
        return any(masked in seg for _, masked in pairs)

    segs = [s.strip() for s in _SPLIT_RE.split(_mask(text, guards)) if s.strip()]
    keep = [True] * len(segs)
    dropped: list[str] = []

    def length() -> int:
        return len(_SEP.join(s for s, k in zip(segs, keep) if k).replace(_MASK, _SEP))

    # 第一轮：从尾部删。尾部是「风格补充 / 画质词」，8.3 明确这些可以先压。
    for i in range(len(segs) - 1, -1, -1):
        if length() <= max_chars:
            break
        if guarded(segs[i]):
            continue
        keep[i] = False
        dropped.append(segs[i].replace(_MASK, _SEP))

    # 第二轮：尾部删完还不够，从头部往后删未被保护的句子（通常是风格前缀）。
    if length() > max_chars:
        for i in range(len(segs)):
            if length() <= max_chars:
                break
            if not keep[i] or guarded(segs[i]):
                continue
            keep[i] = False
            dropped.append(segs[i].replace(_MASK, _SEP))

    result = _SEP.join(s for s, k in zip(segs, keep) if k).replace(_MASK, _SEP)
    return Truncation(
        text=result,
        dropped=tuple(dropped),
        truncated=True,
        # 真删不动了要如实报出来，不能悄悄返回一个超长串。
        still_over=len(result) > max_chars,
    )


# ---------------------------------------------------------------- 图像提示词


def assemble_image_prompt(
    shot: Shot,
    characters: Sequence[Character],
    scene: Scene,
    style: VisualStyle,
    *,
    delta: str | None = None,
    has_ref: bool = False,
    max_chars: int = PROMPT_MAX_CHARS,
) -> Prompt:
    """按 8.1 的模板拼图像提示词，顺序固定，便于逐项核对。

    模板：{style}, {主体}, wearing {wardrobe}, {参考图优先级}, {动作}, {场景},
    {光线}, {构图}, {风格补充}, 画质词

    `delta`（重取时的修正指令）单独成句并**受保护**：它可能比原提示词的后半段更重要，
    被截断逻辑误删就等于「重试没带增量」（7.1 硬规则第 6 条）。

    `has_ref`（2026-11-02 加）：这一次是不是会带角色参考图。带了才写 `CAST_LOCK`。
    没带却写「以参考图为准」，是指向一个不存在的东西——空镜和没定妆照的链路
    都会因此多一句无约束的空话，白白挤占长度预算。
    """
    parts: list[str] = [style.style_prompt.strip()]

    # **只拼本镜出场的角色**（2026-10-29 修，这是本轮最严重的 bug）。
    # 原实现是 `for c in characters` 遍历全部角色卡，不看 `shot.character_ids`，
    # 后果是**每一镜的提示词都写进了全部人物**：六镜短剧里 char_02（红衣女人）
    # 本该第 4 镜才出场，结果第 1 镜的林晚独自等电梯画面里就站着两个人。
    # 三个连锁后果，都不是小问题：
    #   ① 剧情穿帮——"突然发现角落有个人"的悬念在第一个镜头就没了；
    #   ② 一致性判定被污染——参考图是单人定妆照，待判画面是双人，眉眼比对
    #      时判定器会拿错人的眼睛去比（shot_02/04 卡在 75 分就是这么来的）；
    #   ③ 画面上多出的那个人不受任何角色卡约束，每次都长得不一样。
    # `cast` 为空但 `environment_only` 为假，属于数据异常（models 已在校验里拦住），
    # 这里仍按空镜降级处理，不让它拼出半句残缺提示词。
    cast = [c for c in characters if c.char_id in shot.character_ids]

    # 动作句里点到"场上没有的人"，模型就会替他造一个——而且这个人是画面里唯一
    # 不受参考图约束的对象。2026-11-02：`shot_02` 只列了豆包，动作写"凑近男生的
    # 手臂"，出来的就是一条写实人类胳膊。这类错误自检抓不到（"多出一个人"不在
    # 判定要素里），只能在拼提示词时挡。
    if cast:
        offcast = find_offcast_terms(shot.action, cast)
        if offcast:
            who = "、".join(f"{c.name}（{c.gender or '性别未填'}）" for c in cast)
            raise CastConflict(
                f"{shot.shot_id} 的动作里有「{'、'.join(offcast)}」，但这一镜的出场角色是 {who}。"
                f"模型会把「{'、'.join(offcast)}」画成一个没有参考图约束的人。"
                f"改法：把称谓换成角色名，或把该角色加进这一镜的 character_ids。"
            )

    if shot.environment_only or not cast:
        parts.append("空镜，画面中无人物")
    else:
        for c in cast:
            # 主体句统一走 `subject_phrase()`：age_range 是人可编辑的自由文本，
            # 直接当形容词拼会得到"女不是真实人无年龄段的豆包"这种句子。
            parts.append(f"{c.subject_phrase()}，{c.appearance}")
            parts.append(f"穿着{c.wardrobe}")
        if has_ref:
            # 紧跟主体之后：先说是谁，再定"长相听图的"。放到最后会被截断逻辑
            # 优先删掉（尾部先删），也离人物描述太远。
            parts.append(CAST_LOCK)

    parts.append(shot.action)
    parts.append(f"{scene.name}：{scene.description}")
    parts.append(f"光线：{scene.lighting}，{scene.time_of_day}")
    parts.append(f"构图：{shot.camera}")

    if style.aspect_ratio:
        parts.append(f"画幅 {style.aspect_ratio}")

    parts.append("ultra detailed, 高细节，电影质感")

    if delta:
        parts.append(f"修正要求：{delta}")

    raw = _SEP.join(p for p in parts if p)
    # 保护词必须是**分句级**的：appearance 整串里带「，」，分句后没有任何一句能匹配
    # 整串，等于什么都没保护（第一版就这样，截断会把主体删掉）。
    must_keep: list[str] = [shot.action, *appearance_terms(cast)]
    # 参考图优先级是"指令"不是"修饰词"，与 delta 同级保护：
    # 被截掉就等于"带了参考图却不要求照着画"，退化成原来那种图文各说一半。
    protect: list[str] = [p for p in (delta, CAST_LOCK if (has_ref and cast) else None) if p]
    tr = truncate(
        raw,
        must_keep=tuple(must_keep),
        protect=tuple(protect),
        max_chars=max_chars,
    )
    return Prompt(
        text=tr.text,
        negative=style.negative_prompt,
        truncated=tr.truncated,
        dropped=tr.dropped,
        still_over=tr.still_over,
    )


# ---------------------------------------------------------------- 视频提示词


def assemble_video_prompt(
    shot: Shot,
    characters: Sequence[Character],
    *,
    max_chars: int = PROMPT_MAX_CHARS,
) -> Prompt:
    """按 8.2 拼视频提示词：`{谁在动} + {怎么动} + {镜头怎么动}`。

    **刻意不读** `appearance` / `wardrobe` / `style` / `scene` 四个字段——
    它们已由首帧确定，再写一遍会让模型重新构图。这不是「记得不要写」，
    而是结构上拿不到：本函数的入参里就没有 style 与 scene。

    `emotion` 也不进提示词：情绪不是运动，混进去会退化成「悲伤氛围」这类
    §8.2 示例里被判为错误的写法。情绪的去处是 S5 的配音与字幕。
    """
    who = _pronoun(characters)
    motion = shot.action.strip()
    if shot.environment_only or not characters:
        base = f"{motion}，{_camera_motion(shot.camera)}"
    else:
        base = f"{who}{motion}，{_camera_motion(shot.camera)}"

    tr = truncate(base, must_keep=(shot.action,), max_chars=max_chars)
    leaks = tuple(find_appearance_leaks(tr.text, characters))
    # 检出泄漏只记录、不自动删词、也不抛异常：
    # 抛异常会让「她脱下外套」这类合法动作整镜失败；自动删词会掩盖上游数据问题。
    # 正确处置是回 S1 改 action，或由 s4 在提交前按报告口径记一条告警。
    return Prompt(
        text=tr.text,
        truncated=tr.truncated,
        dropped=tr.dropped,
        still_over=tr.still_over,
        violations=leaks,
    )


def render_negative(style: VisualStyle) -> str:
    return style.negative_prompt
