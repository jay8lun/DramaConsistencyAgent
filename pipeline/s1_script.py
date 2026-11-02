"""S1b · 剧本结构化（第 6 章）。

把「创意 + 已确认的角色卡」变成一份通过 4.7 全部校验的 `script.json`。

**S1a 角色闸门（三入口 + 人工确认）不在这里**——那是唯一的强制人工介入点，
属 M2 范围。本模块假定角色卡已经存在：要么调用方传入（S1a 的产物），
要么让模型一并生成（M1 的零成本链路走这条）。

失败处理按 6 章：结构校验不过 → 把错误信息带回重问，最多 3 次。
**3 次仍不过就抛错，不降级**——剧本错了后面每一镜都是错的，不能凑合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from config import PLATFORM_TIERS, settings
from models import Character, Script, VisualStyle
from pipeline.cost import CostLedger
from pipeline.project import ProjectCtx, write_json
from providers.base import TextProvider

MAX_REPAIR = 3

SYSTEM_PROMPT = """你是短剧分镜师，负责把一个创意拆成可直接投喂视频模型的分镜表。

只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块。
**字段名必须逐字照抄下面的结构：不许改名（不要写成 costume / location / camera_movement /
environment 之类的同义词），不许把标量换成对象，不许省略任何键。** 这个 JSON 会被程序直接
反序列化成强类型对象，键名不对整份产出作废。

{
  "title": "片名",
  "logline": "一句话故事",
  "style": {
    "style_prompt": "风格词，全片统一",
    "negative_prompt": "负面词，逗号分隔",
    "aspect_ratio": "9:16"
  },
  "characters": [
    {
      "char_id": "char_01",
      "name": "角色名",
      "age_range": "26-30岁",
      "gender": "女",
      "appearance": "**一个字符串，不是对象**：发型（长度+样式+颜色），脸型，眉眼，肤色，体型",
      "wardrobe": "服装，写死颜色与材质",
      "personality": "性格，一句话",
      "source": "brief"
    }
  ],
  "scenes": [
    {
      "scene_id": "scene_01",
      "name": "场景名",
      "description": "环境描述",
      "lighting": "光线描述",
      "time_of_day": "深夜"
    }
  ],
  "shots": [
    {
      "shot_id": "shot_01",
      "index": 1,
      "scene_id": "scene_01",
      "duration": 5,
      "camera": "景别与运镜，如「中景，缓慢推近」",
      "action": "角色动作，一句话",
      "character_ids": ["char_01"],
      "dialogue": null,
      "emotion": "情绪词"
    }
  ]
}

以上每个键都是必填（source 固定写 "brief"）。纯场景空镜的镜头：character_ids 写成 []，
并加一个 "environment_only": true。

命名与取值是硬约束，违反即视为不合格：
- char_id 形如 char_01、scene_id 形如 scene_01、shot_id 形如 shot_01（两位数字，从 01 开始连续）
- shots 的 index 必须从 1 连续递增，不允许跳号
- duration 只能取 {DURATION_TIERS}。这是**当前视频模型的支持边界**，非任意秒数——
  写别的秒数会被 provider 当场拒绝，那一镜直接失败、白花钱
- 每一镜**必须二选一**：给出非空的 character_ids，或 environment_only=true（纯场景空镜），
  两者互斥，不允许既有角色又标 environment_only
- dialogue 一镜最多一句，没有就写 null
- 角色卡 appearance 是一个字符串，按顺序写全五项：发型（长度+样式+颜色）、脸型、眉眼、肤色、体型。
  **禁止主观形容词**（帅气、有气质、温柔、成熟）。记忆点只给一个，其余保持标准，
  特征越极端跨镜头越容易走形。
- 服装写死颜色与材质，不写「穿西装」，写「深灰色羊毛大衣，内搭白色高领毛衣」
- **动作句里必须用角色卡上的名字称呼人物**（「豆包抬起手」），不许用「男生/女孩/男人/
  女人/少年/少女」这类称谓。这不是文风要求：
  ① 称谓和角色名之间没有任何绑定，程序不会替你翻译——「男生」在提示词里就是四个
     不含约束的字，模型不知道它指的是谁；
  ② 模型一旦把「男生」当成另一个人，画面里就会凭空多出一个**不受参考图约束**的真人。
     2026-11-02 的实录：镜头只列了豆包，动作写「凑近男生的手臂」，出来一条写实人类胳膊，
     而自检只比对脸和服装，「多出一个人」它根本看不到。
  需要对方的身体入画（伸手抓他的手臂、被抱住）时，把对方也写进 character_ids。

分镜动作必须落在模型能力边界内，以下五类**禁止出现**（重试也救不回来，只能在分镜阶段避开）：
1. 大幅转身、快速奔跑 → 改「停住脚步」「肩膀轻微下沉」
2. 手部精细动作（拆信、系扣、打字特写）→ 改「视线落点」「低头」
3. 两人以上同框对视 → 拆成单人镜头交替
4. 剧烈镜头运动（甩镜、快速环绕）→ 改「缓慢推近」「固定机位」
5. 大特写面部（超过画面 60%）→ 改用近景（胸口以上）

全片风格必须统一：同一部片子里所有镜头使用同一个 style_prompt。"""


class ScriptParseError(RuntimeError):
    """模型产出无法构成合法 Script。消息里必须带上具体是哪条校验失败。"""


@dataclass
class ScriptResult:
    script: Script
    attempts: int
    repaired: bool


def build_messages(
    creative: str,
    *,
    shot_count: int = 6,
    characters: list[Character] | None = None,
    style: VisualStyle | None = None,
    repair_hint: str | None = None,
    ref_observations: list[str] | None = None,
) -> tuple[str, str]:
    """拼 system / user 两条消息。`repair_hint` 用于把上一轮的错误带回去重问。

    `ref_observations`（2026-11-02 加）：从参考图读出来的既成事实（`ref_desc.describe_lines`
    的产出）。**它必须出现在「角色由你设计」之前**，否则模型先按创意把人编完，
    再把这几行当成补充说明——那正是原来「凭空写一个 23-27 岁人类男性」的来源。
    """
    lines = [
        f"创意：{creative}",
        f"镜头数：{shot_count}",
    ]
    if ref_observations:
        lines.append("")
        lines.append("已从参考图读到的人物事实（**这是既成事实，不是建议**）：")
        lines.extend(ref_observations)
        lines.append("")
        lines.append(
            "这几个人物必须**按上面的顺序**依次对应 char_01、char_02…，你只负责为它们起名、"
            "写性格、写分镜，**不要重新设计长相**："
        )
        lines.append("- appearance 与 wardrobe **原样抄回上面的「外貌」「服装」**，"
                     "不要改写，也不要按人的常识去补全（图上没有的细节不要加）")
        lines.append("- 主体不是「人」的角色，gender 与 age_range 都写 \"无\""
                     "——卡通形象没有性别年龄，硬填一个就是错的")
        lines.append("- 动作必须符合它的体型与物种：三头身卡通生物不要写「修长的手指」"
                     "「手臂肌肉」「解开纽扣」这类只有人类做得到的动作")
        lines.append("- 除这几个人物之外不要新增角色。剧情需要一个图里没有的人时，"
                     "把他写成画外，或只用已列出的角色")
    elif characters:
        lines.append("角色卡已确定，必须原样使用（不要再生成角色，也不要改名改外貌）：")
        lines.append(json.dumps([c.model_dump() for c in characters], ensure_ascii=False, indent=2))
    else:
        lines.append("角色由你设计：2 个足够，每个角色给出完整角色卡。")
    if style:
        lines.append(f"风格固定为：{json.dumps(style.model_dump(), ensure_ascii=False)}")
    if repair_hint:
        lines.append("")
        lines.append("上一版没有通过结构校验，错误如下，请逐条修正后重新输出完整 JSON：")
        lines.append(repair_hint)
    # 档位从 config 注入（2026-10-29）。写成纯文本 replace 而不是 str.format：
    # SYSTEM_PROMPT 里有大段 JSON 示例（花括号成对出现），用 format 会把它们当占位符炸掉。
    tiers = "、".join(f"{t:g} 秒" for t in PLATFORM_TIERS)
    if len(PLATFORM_TIERS) == 1:
        tiers += "（本模型只有这一档）"
    system = SYSTEM_PROMPT.replace("{DURATION_TIERS}", tiers)
    return system, "\n".join(lines)


def parse(raw: str, *, characters: list[Character] | None = None) -> Script:
    """把模型原始文本解析成 Script。任何不符合 4.7 的地方都要带原因抛出。"""
    text = (raw or "").strip()
    if not text:
        raise ScriptParseError("模型返回空内容")

    # 网关常在 JSON 外面包一层 ```json ... ```，剥掉再试。
    if text.startswith("```"):
        chunks = text.split("```")
        text = max(chunks, key=len).removeprefix("json").strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ScriptParseError(f"返回的不是合法 JSON：{exc.msg}（位置 {exc.pos}）") from exc
    if not isinstance(payload, dict):
        raise ScriptParseError(f"返回的顶层类型是 {type(payload).__name__}，要求是 JSON 对象")

    if characters:
        # 角色卡以已确认的为准，覆盖模型自己编的那份。
        payload["characters"] = [c.model_dump() for c in characters]

    try:
        return Script(**payload)
    except ValidationError as exc:
        details = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            details.append(f"- {loc or '(根)'}：{err.get('msg')}")
        raise ScriptParseError("结构校验未通过：\n" + "\n".join(details)) from exc


def run(
    ctx: ProjectCtx,
    text_provider: TextProvider,
    *,
    creative: str,
    characters: list[Character] | None = None,
    style: VisualStyle | None = None,
    shot_count: int = 6,
    max_repair: int = MAX_REPAIR,
    ledger: CostLedger | None = None,
    allow_paid: bool = False,
    ref_observations: list[str] | None = None,
) -> ScriptResult:
    """S1b 主流程：调用 → 解析 → 失败带错误重问，最多 `max_repair` 次。

    落盘两份：`script.json`（整部）与 `characters.json`（单独一份）。
    6 章要求分开落盘，目的是让角色确认闸门有独立落点——改角色不必重跑分镜。

    `ledger` 传入时每次模型调用都落一条审计记录。文本调用也是付费调用，
    「没有记录等于没发生」（9.4 第 15 条），不能因为单价低就跳过记账。
    """
    hint: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, max_repair + 1):
        system, user = build_messages(
            creative, shot_count=shot_count, characters=characters, style=style,
            repair_hint=hint, ref_observations=ref_observations,
        )
        if ledger is not None:
            with ledger.paid(
                settings.price_llm_per_call,
                stage="script",
                provider=text_provider.name,
                operation="complete",
                allow_paid=allow_paid,
            ) as slot:
                raw = text_provider.complete(system, user, json_mode=True)
                slot.charge(settings.price_llm_per_call)
        else:
            raw = text_provider.complete(system, user, json_mode=True)
        try:
            script = parse(raw, characters=characters)
        except ScriptParseError as exc:
            last_error = exc
            hint = str(exc)
            continue

        write_json(ctx.script_file, script.model_dump())
        write_json(ctx.characters_file, [c.model_dump() for c in script.characters])
        return ScriptResult(script=script, attempts=attempt, repaired=attempt > 1)

    raise ScriptParseError(
        f"S1b 连续 {max_repair} 次未产出合法剧本，最后一次错误：{last_error}"
    )


def load_script(ctx: ProjectCtx) -> Script:
    """从磁盘读回剧本（断点续跑与 S3 起跑都用它，避免重复调用模型）。"""
    path = ctx.script_file
    if not path.exists():
        raise FileNotFoundError(f"剧本不存在：{path}（S1 未完成）")
    return Script(**json.loads(Path(path).read_text(encoding="utf-8")))
