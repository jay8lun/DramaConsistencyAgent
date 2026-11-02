"""图生文反推角色卡（计划书 4.2 的入口 A / C）。

**为什么必须补这一步**（2026-11-02 的现场事故）：

用户上传了一张 3D 卡通吉祥物（奶龙）当参考图。生成出来的定妆照是
一个写实真人男性，**脖子上有一片解释不了的黄色**。链路上没有任何报错。

原因不是"参考图没传进去"——传了。是**文字角色卡和参考图各说各话**：
`role_source=image` 只被落盘、没有任何下游读取，S1 仍然凭空写卡
（"23-27 岁人类男性、肤色白皙"），而 `build_view_prompt` 从头到尾没提参考图。
模型于是把两条线索各采纳一部分：人类的画风 + 吉祥物的体色。

补 `REF_LOCK` / `CAST_LOCK`（提示词层）只能**定优先级**，压不住"文字本身是错的"
这件事——文字还在那儿，只是被判为次优。要真正消除冲突，只能让文字来自图。

所以这层的产出不是"另一份参考"，而是一张 **`source="inferred"` 的角色卡**：
- `appearance` / `wardrobe` 来自图，不再由模型凭空编；
- `truth_image` 指向用户原图（判定时优先拿它当标准答案，比再生成一遍的定妆照可靠）；
- `low_confidence_fields` 标出"这才 0.55，人是猜的"，供停 1 高亮。

**不覆盖 name**：反推读不出名字，名字由 S1 从创意里起。这也是为什么反推放在
S1 之后——顺序反过来就得先给角色起名才能写分镜。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from config import settings
from models import Character
from pipeline.cost import CostLedger
from pipeline.project import ProjectCtx, write_json
from providers.base import VisionJudge

#: 自评低于这个值就进 `low_confidence_fields`，停 1 会高亮提示人核对。
#: 0.6 不是拍脑袋：模型给"看得见"的字段普遍打 0.8~0.95，凡落到 0.5 档的
#: 基本都是"图里糊/被挡住/只能靠常识猜"，正是最该让人看一眼的那批。
LOW_CONF = 0.6

#: 会被反推覆盖的字段。**name / personality 不在内**：前者图里读不出，
#: 后者是剧情需要，不是长相。
CARD_FIELDS: tuple[str, ...] = ("gender", "age_range", "appearance", "wardrobe")


class DescParseError(RuntimeError):
    """反推结果不可用。消息里必须写清是哪一项废了。"""


@dataclass
class DescribedCard:
    """一张参考图反推出来的角色卡原件。"""

    image: str
    subject_type: str = "other"
    gender: str = ""
    age_range: str = ""
    appearance: str = ""
    wardrobe: str = ""
    style_hint: str = ""
    distinctive: str = ""
    confidence: dict[str, float] = field(default_factory=dict)
    low_confidence: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image": self.image, "subject_type": self.subject_type,
            "gender": self.gender, "age_range": self.age_range,
            "appearance": self.appearance, "wardrobe": self.wardrobe,
            "style_hint": self.style_hint, "distinctive": self.distinctive,
            "confidence": self.confidence, "low_confidence": self.low_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DescribedCard":
        return cls(
            image=str(data.get("image", "")),
            subject_type=str(data.get("subject_type", "other") or "other"),
            gender=str(data.get("gender", "") or ""),
            age_range=str(data.get("age_range", "") or ""),
            appearance=str(data.get("appearance", "") or ""),
            wardrobe=str(data.get("wardrobe", "") or ""),
            style_hint=str(data.get("style_hint", "") or ""),
            distinctive=str(data.get("distinctive", "") or ""),
            confidence={k: float(v) for k, v in (data.get("confidence") or {}).items()
                        if isinstance(v, (int, float))},
            low_confidence=[str(x) for x in (data.get("low_confidence") or [])],
            raw=data.get("raw") or {},
        )


def parse_desc(raw: object, *, image: str = "") -> DescribedCard:
    """把模型原始 JSON 归成 `DescribedCard`。**纯函数**，便于离线自检。

    只做类型对齐与"哪些字段不可信"的判定，不改写内容：
    模型说"黄色圆头、短腿"，就原样留着——不该由程序替它翻译成"卡通角色"。
    """
    if not isinstance(raw, dict):
        raise DescParseError(f"反推结果不是 JSON 对象：{type(raw).__name__}")

    appearance = str(raw.get("appearance") or "").strip()
    if not appearance:
        # 没有外观描述，这张卡就什么也没提供；继续下去等于"用空卡覆盖原卡"，
        # 反而把 S1 写的东西删掉了。
        raise DescParseError("反推结果没有 appearance（外观描述为空）")

    subject = str(raw.get("subject_type") or "other").strip().lower()
    if subject not in ("human", "cartoon", "animal", "other"):
        subject = "other"

    gender = str(raw.get("gender") or "").strip()
    # 归一成"男"/"女"/""。原文常是"女性""男性""female"，直接塞进提示词会拼出
    # "女性26-30岁的林晚"这种半通不通的句子，而提示词是要给模型看的。
    if "男" in gender:
        gender = "男"
    elif "女" in gender:
        gender = "女"
    else:
        gender = ""  # 有"不确定"/看不出 → 当没判出来，由调用方保留原值

    age = str(raw.get("age_range") or "").strip()
    if age in ("", "不确定", "看不清", "unknown", "n/a", "N/A"):
        age = ""

    conf: dict[str, float] = {}
    for k, v in (raw.get("confidence") or {}).items():
        if isinstance(v, (int, float)):
            conf[str(k)] = max(0.0, min(1.0, float(v)))

    low = [f for f in CARD_FIELDS if conf.get(f, 1.0) < LOW_CONF]
    # 非人角色没有"性别/年龄"这回事，模型硬填一个也基本是错的。
    # 无条件标出来让人删，比让它在提示词里冒充事实好。
    if subject != "human":
        for f in ("gender", "age_range"):
            if f not in low:
                low.append(f)

    return DescribedCard(
        image=image, subject_type=subject, gender=gender, age_range=age,
        appearance=appearance,
        wardrobe=str(raw.get("wardrobe") or "").strip(),
        style_hint=str(raw.get("style_hint") or "").strip(),
        distinctive=str(raw.get("distinctive") or "").strip(),
        confidence=conf, low_confidence=low, raw=dict(raw),
    )


def _fingerprint(image: Path) -> str:
    """图内容指纹。用 sha1 而不是 mtime：同一秒内换图（拖拽重传）mtime 可能不变，
    而"换了图却复用旧卡"的表现是——脸还是上一张的，且没有任何提示。"""
    h = hashlib.sha1()
    with open(image, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _price(judge: VisionJudge) -> float:
    """单次反推的估算价。mock 一律 0——否则零成本验证会被预算封顶拦在门外。"""
    if getattr(judge, "is_mock", False):
        return 0.0
    return float(settings.price_vlm_per_call)


def cache_path(ctx: ProjectCtx, index: int) -> Path:
    return ctx.root / "refs" / f"desc_{index + 1:02d}.json"


def load_cached(ctx: ProjectCtx, index: int, image: Path) -> DescribedCard | None:
    """读缓存。指纹对不上就当没有——重传过图，旧卡已经不对应这张脸了。"""
    p = cache_path(ctx, index)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("fingerprint") != _fingerprint(image):
        return None
    card = DescribedCard.from_dict(data)
    return card if card.appearance else None


def describe_ref(
    ctx: ProjectCtx,
    index: int,
    image: Path,
    *,
    judge: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool = False,
) -> DescribedCard:
    """反推一张图并落盘缓存。**失败会抛**，由调用方决定是否阻断（S1 不阻断）。"""
    image = Path(image)
    if not image.exists():
        raise DescParseError(f"参考图不存在：{image}")
    cached = load_cached(ctx, index, image)
    if cached is not None:
        return cached

    with ledger.paid(
        _price(judge),
        stage="ref_desc",
        provider=getattr(judge, "name", "vlm-judge"),
        operation="describe",
        allow_paid=allow_paid,
    ) as slot:
        data = judge.describe(image, allow_paid=allow_paid)
        slot.charge(_price(judge))

    card = parse_desc(data.get("card"), image=str(image))
    write_json(cache_path(ctx, index), {**card.to_dict(),
                                        "fingerprint": _fingerprint(image)})
    return card


def observe(
    ctx: ProjectCtx,
    images: Sequence[Path],
    *,
    judge: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool = False,
    warn=None,
) -> list[DescribedCard]:
    """S1 之前先把参考图读成「人物观察」（2026-11-02 加）。

    **为什么要在写剧本之前读**：S1 原来只拿创意文本去编人物，写出来的角色卡是
    「23-27 岁人类男性」这种凭空设定，分镜动作也照人类的习惯写；而反推只覆盖
    `appearance/wardrobe`，改不了已经写进分镜的称谓与动作。先把图读成事实再交给
    S1，它才有可能「照着这个人」写戏。

    **失败不阻断**：读不出来就退回「纯文字编」，那是老行为，至少还能出片。
    """
    cards: list[DescribedCard] = []
    for i, img in enumerate(images):
        try:
            cards.append(describe_ref(ctx, i, Path(img), judge=judge,
                                      ledger=ledger, allow_paid=allow_paid))
        except Exception as exc:  # noqa: BLE001 —— 可选增强，不许拖垮主流程
            (warn or (lambda m: None))(
                f"{Path(img).name} 读取失败（这一张没有事实可给剧本，只能按创意编）：{exc}")
            cards.append(DescribedCard(image=str(img)))
    return cards


#: 主体类型 → 给模型看的中文。**不用英文原值**：`subject_type=cartoon` 出现在
#: 中文提示词里，模型容易当成一个待填的字段去"补全"，反而照人类的模板写。
_SUBJECT_CN = {"human": "人", "cartoon": "卡通形象", "animal": "动物", "other": "其他"}


def describe_lines(cards: Sequence[DescribedCard]) -> list[str]:
    """把观察翻成给剧本模型看的**既成事实**清单。

    只给事实、不给字段名（不写 `appearance=...`）：写成字段样式，模型会以为这是
    一份要它填的 schema，进而按人类模板补全；写成「外貌=…」的陈述句，它才会当
    既成事实引用。
    """
    out: list[str] = []
    for i, c in enumerate(cards, 1):
        head = f"- 第 {i} 张"
        if not c.appearance:
            out.append(f"{head}：读取失败，没有可用信息——按创意写一个角色，"
                       "但不要把它写成与这张图一样的人。")
            continue
        bits = [f"主体={_SUBJECT_CN.get(c.subject_type, '其他')}"]
        if c.style_hint:
            bits.append(f"画风={c.style_hint}")
        bits.append(f"外貌={c.appearance}")
        if c.wardrobe:
            bits.append(f"服装={c.wardrobe}")
        if c.distinctive:
            bits.append(f"显著特征={c.distinctive}")
        if c.low_confidence:
            bits.append(f"（{'/'.join(c.low_confidence)} 图上判断不准，不要硬填）")
        out.append(head + "：" + "；".join(bits))
    return out


def apply_to_characters(
    characters: Sequence[Character],
    cards: Sequence[DescribedCard | None],
) -> tuple[list[Character], list[str]]:
    """把反推卡覆盖到角色上（按顺序，第 i 张图 → 第 i 个角色）。

    返回 (新角色列表, 人话摘要)。

    三条覆盖规则都偏向"**宁缺勿错**"：
    1. `appearance` 必替 —— 这是本功能存在的唯一理由；
    2. `wardrobe` 反推为空就置空，**不保留原卡的服装** —— 原卡的"深灰羊毛大衣"
       正是这场事故的来源（图和卡各说一半），留着等于没修；
    3. `gender` / `age_range` 没判出来时保留原值 —— 字段必填，清空会让模型
       自己补一个（可能又是"男性"），不如留着并标进低置信字段让人改。
    """
    out: list[Character] = []
    notes: list[str] = []
    for i, char in enumerate(characters):
        card = cards[i] if i < len(cards) else None
        if card is None:
            out.append(char)
            continue
        gender = card.gender or char.gender
        age = card.age_range or char.age_range
        out.append(char.model_copy(update={
            "gender": gender,
            "age_range": age,
            "appearance": card.appearance,
            "wardrobe": card.wardrobe,      # 空就是空，见规则 2
            "source": "inferred",
            "truth_image": card.image,
            "low_confidence_fields": list(card.low_confidence),
        }))
        low = "/".join(card.low_confidence) or "无"
        notes.append(f"{char.char_id} {char.name} ← {Path(card.image).name}"
                     f"（{card.subject_type}，画风 {card.style_hint or '未知'}，"
                     f"低置信 {low}）")
    return out, notes


def describe_all(
    ctx: ProjectCtx,
    images: Sequence[Path],
    characters: Sequence[Character],
    *,
    judge: VisionJudge,
    ledger: CostLedger,
    allow_paid: bool = False,
    warn=None,
    skip_inferred: bool = True,
) -> tuple[list[Character], list[str]]:
    """按上传顺序逐张反推，再覆盖到角色上。

    **失败不阻断**（2026-11-02 定）：反推挂了，S1 的角色卡仍然可用——
    只是退回"文字与图可能打架"的老状态，而那至少是能出图的。
    让整条流水线因为一次可选增强而停在停 1 之前，才是真的坏事。

    `skip_inferred`（默认开）：已经是 `inferred` 的角色**不再覆盖**。
    不加这条的话，人 在停 1 把"明黄色"改准确、再点继续，就会被缓存里的旧卡
    原样盖回去——界面上刚改完的文字自己变回原样，而且没有任何提示。
    想重新反推（换了图）时，上传接口会把该角色的 source 重置回 brief。
    """
    if not images:
        return list(characters), []
    if len(images) > len(characters):
        msg = (f"参考图 {len(images)} 张，角色只有 {len(characters)} 个："
               f"多出的 {len(images) - len(characters)} 张不会参与（按顺序绑定）")
        (warn or (lambda m: None))(msg)

    cards: list[DescribedCard | None] = []
    skipped: list[str] = []
    for i, img in enumerate(images[: len(characters)]):
        if skip_inferred and characters[i].source == "inferred":
            # 已经按图反推并被（可能）人工核对过，原样保留。
            skipped.append(characters[i].char_id)
            cards.append(None)
            continue
        try:
            cards.append(describe_ref(ctx, i, img, judge=judge, ledger=ledger,
                                      allow_paid=allow_paid))
        except Exception as exc:  # noqa: BLE001 —— 可选增强，不许拖垮主流程
            (warn or (lambda m: None))(f"{Path(img).name} 反推失败（沿用原角色卡）：{exc}")
            cards.append(None)
    chars, notes = apply_to_characters(characters, cards)
    if skipped:
        notes.append(f"{'/'.join(skipped)} 已是 inferred，保留现值（不覆盖人工修改）")
    return chars, notes
