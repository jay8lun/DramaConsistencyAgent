"""VLM 要素级判定 + diff 转提示词增量（第 7.1 节 / D3b）。

**判定逻辑只存在于这里**：s3 与 C2 的归因回路都必须调用本模块，不得各写一份
（第 5 章模块划分硬要求）。

本模块**不发起任何网络调用**——`VisionJudge.compare` 的原始返回值由调用方传进来。
这样判定规则可以完全离线单测（14.7：35 个用例断网可跑），也保证「判定口径」与
「某一家 VLM 的协议」解耦，换厂商不影响决策逻辑。

三条分数线（7.1。**数值随判定模型走**，见 config.JUDGE_PASS_LINE 的注释）：
    ≥75            → pass
    60~75 且未超限  → retry（**必须带 prompt_delta**，7.1 硬规则第 6 条）
    <60 或超限      → degraded（不是 failed，7.2 语义不可合并）

当前值是 qwen3-vl-plus 的刻度：同人组 P5=85、异人组 P95=65，75 取空档正中。
豆包时代这两条是 85/60（同人 P5=90.5）——换模型必须重标，别把 75 当通用常量。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence

from config import JUDGE_FLOOR_LINE, JUDGE_PASS_LINE, PROMPT_MAX_CHARS
from models import Character, ElementCheck, QAResult
from pipeline.prompt_builder import truncate

#: 判定的五个要素。**必须是业务层的单一来源**，不能由某家 provider 决定——
#: 换 VLM 时要素不能跟着变，否则 M0 得出的准确率无法外推到生产。
#: 与 `providers/vlm.ELEMENT_HINTS` 的一致性由 `scripts/_verify_judge.py` 断言。
ELEMENTS: tuple[str, ...] = ("发型", "脸型", "眉眼", "肤色", "体型")

PASS_LINE: float = JUDGE_PASS_LINE
FLOOR_LINE: float = JUDGE_FLOOR_LINE

#: 增量文本的字符上限。增量要短——它会被拼进提示词，太长会挤掉原描述。
DELTA_MAX_CHARS: int = 240


class Action(str, Enum):
    """判定后的动作。三个值直接对应状态机的三个出口（7.1）。"""

    PASS = "pass"
    RETRY = "retry"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class Decision:
    action: Action
    score: int
    reason: str
    qa: QAResult | None = None
    prompt_delta: str | None = None
    #: 解析失败时的原始返回文本。必须留存——否则「模型返回了什么」永久丢失，
    #: 事后只能靠猜（9.4 第 15 条禁止虚构数据）。
    raw_text: str | None = None


# ---------------------------------------------------------------- 分数决策


def decide(score: int, *, retries_used: int, max_retries: int) -> tuple[Action, str]:
    """按 7.1 的三条线给出动作。纯函数，不碰任何对象，便于穷举测试。"""
    if score >= PASS_LINE:
        return Action.PASS, f"{score} ≥ {PASS_LINE:g}，直接通过"
    if score < FLOOR_LINE:
        # <60 不给重取机会：差这么远通常不是「局部要素偏了」，而是锚点/场景整体错了，
        # 带一条小增量改不动，重取等于换张图重抽（7.1 明文）。
        return Action.DEGRADED, f"{score} < {FLOOR_LINE:g}，差异过大，重取无意义，降级放行"
    if retries_used >= max_retries:
        return (
            Action.DEGRADED,
            f"{score} 落在 {FLOOR_LINE:g}~{PASS_LINE:g}，但质量重试已用尽"
            f"（{retries_used}/{max_retries}），降级放行",
        )
    return (
        Action.RETRY,
        f"{score} 落在 {FLOOR_LINE:g}~{PASS_LINE:g}，第 {retries_used + 1}/{max_retries} 次质量重试",
    )


def can_retry(*, retries_used: int, max_retries: int) -> bool:
    """预算是否还允许再一次重取。**调用方必须在发起生成前查这个**，
    否则会出现「先调用、再发现超限」——钱已经花了（14.4-B `test_retry_budget_is_enforced`）。"""
    return retries_used < max_retries


# ---------------------------------------------------------------- 原始结果归一


def _as_mapping(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """把 provider 的原始返回转成 dict。解析不了就返回 (None, 原文)。"""
    if isinstance(raw, dict):
        return raw, None
    if isinstance(raw, str):
        text = raw.strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # 有些网关会把 JSON 包在 ```json ... ``` 里，剥一层再试一次。
            if "```" in text:
                inner = text.split("```")
                for chunk in inner:
                    c = chunk.strip().removeprefix("json").strip()
                    try:
                        obj = json.loads(c)
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    return None, text
            else:
                return None, text
        if isinstance(obj, dict):
            return obj, text
        return None, text
    return None, None if raw is None else str(raw)


def coerce_checks(raw: Any) -> tuple[list[ElementCheck], list[str]]:
    """把 provider 的 `checks` 归一成 `ElementCheck` 列表，同时返回问题清单。

    两个必须处理的脏数据形态：
    1. `matched` 为 `null`（模型漏判）——`ElementCheck.matched` 是 bool，只能落成 False，
       但**必须记进问题清单**，否则「漏判」会被当成「判为不一致」混进统计。
    2. 要素缺失/多出——缺失的补一条「未判定」，不静默丢。
    """
    problems: list[str] = []
    if not isinstance(raw, list):
        return [], ["checks 缺失或不是列表"]

    seen: dict[str, ElementCheck] = {}
    for item in raw:
        if not isinstance(item, dict):
            problems.append(f"checks 里有非对象项：{item!r}")
            continue
        el = str(item.get("element", "")).strip()
        if el not in ELEMENTS:
            problems.append(f"未知要素 {el!r}，已忽略")
            continue
        matched = item.get("matched")
        if matched is None:
            problems.append(f"{el} 的 matched 为空（模型漏判），按不一致处理")
            matched = False
        seen[el] = ElementCheck(
            element=el,
            matched=bool(matched),
            expected=str(item.get("expected") or ""),
            actual=str(item.get("actual") or ""),
        )

    for el in ELEMENTS:
        if el not in seen:
            problems.append(f"缺少要素 {el}，补为未判定")
            seen[el] = ElementCheck(element=el, matched=False, expected="未判定", actual="未返回")

    return [seen[el] for el in ELEMENTS], problems


# ---------------------------------------------------------------- diff 转增量

#: 「不能用」的期望值。`providers/mock.MockJudge` 返回的就是「见角色卡」，
#: 真实模型也常回「参考图」这类占位——直接拼进增量会得到「发型改为见角色卡」，
#: 一句无法执行的指令，等于这条重试白跑（还照扣钱）。
_PLACEHOLDER_EXPECTED = frozenset(
    {"", "见角色卡", "未判定", "未返回", "参考图", "一致", "不符", "无", "-", "--", "同上"}
)


def _expected_for(element: str, expected: str, characters: Sequence[Character]) -> str:
    """取该要素的目标描述：优先用 VLM 回显的期望值，占位/过短则回落到角色卡原文。

    回落用**位置**对应（8.1 规定 appearance 必填五项且顺序固定：
    发型、脸型、眉眼、肤色、体型）。段数不是 5 说明角色卡没按规范写，
    此时整串回落到增量里，宁可长一点也不能给出一条空指令。
    """
    exp = (expected or "").strip()
    if len(exp) >= 3 and exp not in _PLACEHOLDER_EXPECTED and "见角色卡" not in exp:
        return exp
    if not characters:
        return ""
    appearance = (characters[0].appearance or "").strip()
    segs = [s.strip() for s in re.split(r"[，,、;；]", appearance) if s.strip()]
    if len(segs) == len(ELEMENTS):
        return segs[ELEMENTS.index(element)]
    return appearance


def build_prompt_delta(
    diff: str,
    checks: Iterable[ElementCheck],
    characters: Sequence[Character] = (),
    *,
    max_chars: int = DELTA_MAX_CHARS,
) -> str:
    """把 `diff` 翻译成可执行的提示词增量（D3b）。

    **为什么不用 LLM 翻译**：
    1. 判定已经给出了要素级的 `expected`——那是角色卡里的原文，比模型再写一遍更准；
    2. 每镜多一次 LLM 调用就是多一条失败路径（LLM 挂了 → 增量拿不到 → 违反
       「重试必须带增量」的硬规则），而 12.2 的成本表里**没有**这一项；
    3. 规则式输出可离线断言，LLM 输出只能靠「语义相似」判断，与 14.7 冲突。
    LLM 改写留作可选优化（需要时在调用方包一层），不影响这里的确定性。

    输出形如 `发型改为黑色齐肩直发；脸型改为鹅蛋脸（差异：发型长度不符）`，
    这样「改什么、改成什么」都能被关键词断言命中（14.4-B 第 4 条）。
    """
    fails = [c for c in checks if not c.matched]

    clauses: list[str] = []
    for c in fails:
        exp = _expected_for(c.element, c.expected, characters)
        clauses.append(f"{c.element}改为{exp}" if exp else f"{c.element}按参考图重绘")

    if not clauses:
        # 分数不达标却没有失败要素：说明 checks 与 score 不自洽。
        # 此时仍**必须返回非空增量**，否则重试变成原样重跑（7.1 硬规则第 6 条）。
        clauses.append(diff.strip() if diff.strip() else "整体外观按角色卡重绘")

    delta = "；".join(clauses)
    if diff.strip() and diff.strip() not in delta:
        delta = f"{delta}（差异：{diff.strip()}）"

    tr = truncate(delta, protect=tuple(clauses), max_chars=max_chars)
    return tr.text


# ---------------------------------------------------------------- 主入口


def judge(
    raw: Any,
    *,
    characters: Sequence[Character] = (),
    retries_used: int = 0,
    max_retries: int = 2,
) -> Decision:
    """把 VLM 原始返回 + 重试状态，归一成一个可执行的决定。

    `raw` 可以是 dict、JSON 字符串、或任何解析不了的东西——**永不抛未捕获异常**
    （14.4-B `test_malformed_vlm_json_falls_back_safely`）。解析不了就走降级，
    并保留原文。
    """
    data, raw_text = _as_mapping(raw)

    if data is None:
        return Decision(
            action=Action.DEGRADED,
            score=0,
            reason="VLM 返回无法解析为 JSON，按降级处理（不入 pass 也不静默重试）",
            qa=QAResult(score=0, passed=False, checks=[], diff="VLM 原始返回无法解析",
                        attempts=retries_used + 1, degraded=True),
            raw_text=raw_text,
        )

    score_raw = data.get("score")
    try:
        score = int(round(float(score_raw)))
    except (TypeError, ValueError):
        return Decision(
            action=Action.DEGRADED,
            score=0,
            reason=f"VLM 返回的 score 非法（{score_raw!r}），按降级处理",
            qa=QAResult(score=0, passed=False, checks=[], diff=f"score 非法：{score_raw!r}",
                        attempts=retries_used + 1, degraded=True),
            raw_text=raw_text if raw_text is not None else json.dumps(data, ensure_ascii=False),
        )
    score = max(0, min(100, score))

    checks, problems = coerce_checks(data.get("checks"))
    diff = str(data.get("diff") or "").strip()
    if problems:
        # 问题写进 diff 而不是丢弃：报告要能看到「这一镜的判定本身是残缺的」。
        diff = "；".join([diff] if diff else []) + ("；" if diff else "") + "；".join(problems)

    if not checks:
        # checks 整个缺失：分数无从校验，不能当成通过。走降级（14.4-B 最后一条）。
        return Decision(
            action=Action.DEGRADED,
            score=score,
            reason="VLM 返回缺 checks，无法做要素级归因，按降级处理",
            qa=QAResult(score=score, passed=False, checks=[], diff=diff or "缺 checks",
                        attempts=retries_used + 1, degraded=True),
            raw_text=raw_text if raw_text is not None else json.dumps(data, ensure_ascii=False),
        )

    action, reason = decide(score, retries_used=retries_used, max_retries=max_retries)

    delta: str | None = None
    if action is Action.RETRY:
        delta = build_prompt_delta(diff, checks, characters)
        if not delta:
            # 兜底：宁可降级也不能让「带增量的重试」退化成原样重跑（7.1 硬规则第 6 条）。
            action = Action.DEGRADED
            reason += "；但增量生成为空，拒绝原样重跑，改为降级放行"

    qa = QAResult(
        score=score,
        passed=action is Action.PASS,
        checks=checks,
        diff=diff,
        attempts=retries_used + 1,
        degraded=action is Action.DEGRADED,
    )
    return Decision(action=action, score=score, reason=reason, qa=qa, prompt_delta=delta,
                    raw_text=raw_text)


def failing_elements(qa: QAResult | None) -> list[str]:
    """报告用：哪些要素没过。体型是 M0-a 实测最弱项（异人组仅 66.7%），需要单独可查。"""
    if qa is None:
        return []
    return [c.element for c in qa.checks if not c.matched]


def rebuild(
    record: dict[str, Any],
    *,
    characters: Sequence[Character] = (),
    retries_used: int = 0,
    max_retries: int = 2,
) -> Decision:
    """从 `state.load_qa()` 的记录重建决定，**不调用 VLM**。

    存在的理由：`judging` 阶段没有文件产物，进程被 kill 后如果重判，每重启一次
    都白烧一次判定费用（实测单次约 150 秒 / ¥0.08）。判定结果已落 `shot_qa`，
    续跑时按原分数重新走一遍决策即可。
    """
    score = int(record.get("score") or 0)
    checks, _problems = coerce_checks(record.get("checks") or [])
    diff = str(record.get("diff") or "")
    action, reason = decide(score, retries_used=retries_used, max_retries=max_retries)

    delta: str | None = None
    if action is Action.RETRY:
        # 优先用当时存的增量：它是「当次」的修正指令，重建时可能因角色卡变动而不一致。
        delta = record.get("delta") or build_prompt_delta(diff, checks, characters)
        if not delta:
            action = Action.DEGRADED
            reason += "；重建时增量取不到，拒绝原样重跑，改为降级放行"

    return Decision(
        action=action,
        score=score,
        reason=f"续跑（判定结果已落库，未重调 VLM）：{reason}",
        qa=QAResult(
            score=score,
            passed=action is Action.PASS,
            checks=checks,
            diff=diff,
            attempts=int(record.get("attempts") or retries_used + 1),
            degraded=action is Action.DEGRADED,
        ),
        prompt_delta=delta,
    )
