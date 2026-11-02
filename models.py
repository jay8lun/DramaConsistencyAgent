"""数据契约：全系统唯一的 Schema 定义。

设计原则：字段只增不减。
下游任何模块需要的输入，都必须在这里有落点，否则就会出现
「S3 要用角色特征做参考图，但分镜里没有 char_id」这类返工。
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# 时长档位从 config 读，**不在本文件写死**（第 4.7.1 节第 2 条）。
# R1 未实测，档位表可能变；写死意味着改一条配置要动 Schema 和测试两处。
from config import PLATFORM_TIERS

#: 这些取值等于"没有这一项"。非人类角色按提示词规则写 `无`，拼进提示词时应当整个
#: 省掉，而不是输出 `奶龙（无，无）`——那句话读起来像占位符没填，还白占长度预算。
_EMPTY_BITS = frozenset({"无", "未知", "-", "—", "N/A", "n/a"})

#: "26-30岁" / "18-22岁" 这类真年龄写法。只有长得像年龄时才允许当形容词用。
#: 放在模块级而不是 Character 的类属性上：pydantic 会把带下划线的类属性
#: 当成 private attribute，访问到的是包装对象而不是正则本身。
_AGE_LIKE = re.compile(r"^\s*\d+\s*[-~—至到]\s*\d+\s*岁?\s*$|^\s*\d+\s*岁\s*$")


class VisualStyle(BaseModel):
    """全片统一的视觉风格，会注入每一个出图提示词。"""

    style_prompt: str = Field(description="风格词，如「国风水墨，冷色调，电影感」")
    negative_prompt: str = "低质量, 模糊, 变形, 多余手指, 水印, 文字"
    aspect_ratio: str = "9:16"


class Character(BaseModel):
    """角色卡 = 定妆照的唯一依据，字段越具体，三视图越稳。"""

    char_id: str = Field(pattern=r"^char_\d{2}$")
    name: str
    age_range: str
    gender: str
    appearance: str = Field(description="发型/脸型/五官/体型的完整描述，禁止只写「好看」")
    wardrobe: str = Field(description="服装与配饰，跨镜头必须一致")
    personality: str = ""

    # ---- 来源与锚点（第 4.2 节的三入口设计）----
    # source 不带默认值：三条入口（用户上传图 / 文字简介 / 图生文反推）必须显式声明来自
    # 哪一条，否则跨项目复用角色卡时无法判断可信度。给默认值等于让调用方「不选也算选」。
    source: Literal["image", "brief", "inferred"] = Field(
        description="角色卡来源：image=用户上传原图 / brief=文字简介 / inferred=图生文反推"
    )
    # 身份真相图（只有入口 A「用户上传原图」才有）。VLM 判定时**优先用它**，
    # 为空才退到 anchor_path —— 因为 anchor_path 是重制图，本身已漂移一次，
    # 拿它当标准答案等于放宽标准（第 4.2 节明文）。
    truth_image: str | None = Field(default=None, description="身份真相图，入口 A 上传的原图")
    # 图生文反推时不确定的字段名，供第 6 章 S1a 的人工确认环节高亮显示。
    # 这是「唯一的强制人工介入点」的落点，不能省。
    low_confidence_fields: list[str] = Field(
        default_factory=list, description="图生文反推的低置信字段，需人工确认"
    )

    anchor_path: str | None = Field(default=None, description="定妆照主图，S2 产出")
    anchor_views: list[str] = Field(default_factory=list, description="三视图，S2 产出")

    #: "26-30岁" / "18-22岁" 这类真年龄写法。只有长得像年龄时才允许当形容词用。
    def subject_phrase(self) -> str:
        """角色在提示词里怎么被"念"出来。

        常规是 `女26-30岁的林晚`——中文提示词里模型最熟的写法。
        但 `age_range` 是**人可编辑的自由文本**：卡通角色没有年龄，人就可能填
        "不是真实人无年龄段"，用形容词拼出来就是
        **"女不是真实人无年龄段的豆包"**，一句模型看不懂的话，
        而且它会被抄进每一镜的提示词里。

        所以只有"长得像年龄"时才用形容词结构，其余一律退成括注：
        `豆包（女，不是真实人无年龄段）`。括注对任何内容都不产生语法依赖。

        放在 Character 上而不是各调用处：定妆照和镜头提示词是两条独立的拼接路径，
        两处各写一份判断，早晚会漂成两种念法。
        """
        gender = (self.gender or "").strip()
        age = (self.age_range or "").strip()
        if gender and age and _AGE_LIKE.match(age):
            return f"{gender}{age}的{self.name}"
        # 非人类角色按新规则写 `gender="无" / age_range="无"`（`s1_script` 的硬约束），
        # 直接拼会得到 `奶龙（无，无）`——拼得对，但读起来像占位符没填，
        # 而且白占提示词长度。没有性别年龄这件事，**不写**比写"无"更准确。
        bits = [b for b in (gender, age) if b and b not in _EMPTY_BITS]
        return f"{self.name}（{'，'.join(bits)}）" if bits else self.name

    @model_validator(mode="after")
    def check_source_consistency(self) -> "Character":
        """入口 A（用户上传原图）必须带 truth_image（第 4.7 节第 7 条）。

        这条放在 Character 上而不是 Script 上：它是**角色自身的不变量**，
        与剧本无关。放在 Script 里意味着单建一张角色卡时约束失效，
        问题会推迟到组装剧本才暴露，报错位置离原因太远。
        """
        if self.source == "image" and not self.truth_image:
            raise ValueError(
                f"{self.char_id} 的 source=image，但没有 truth_image；"
                "入口 A 必须带用户上传的原图，否则判定时只能退到已漂移过的 anchor_path"
            )
        return self

    def reference_for_shot(self) -> str:
        """给 VLM 判定用的期望描述，只取可见要素。"""
        return f"{self.gender}，{self.age_range}，{self.appearance}，穿着{self.wardrobe}"


class Scene(BaseModel):
    """场景卡：光线和时段必须写死，否则跨镜头色温会漂。"""

    scene_id: str = Field(pattern=r"^scene_\d{2}$")
    name: str
    description: str
    lighting: str
    time_of_day: str


class ElementCheck(BaseModel):
    """VLM 单要素比对结果。要素化是为了能定位到具体问题，而不是一个主观分。"""

    element: str
    matched: bool
    expected: str
    actual: str


class QAResult(BaseModel):
    """L3 判定结果。degraded=True 表示重试耗尽后降级放行，报告里要标红。"""

    score: int = Field(ge=0, le=100)
    passed: bool
    checks: list[ElementCheck] = Field(default_factory=list)
    diff: str = ""
    attempts: int = 1
    degraded: bool = False


class AttemptLog(BaseModel):
    """单次尝试的日志（第 4.5 节）。

    这是**最终报告与简历数据的唯一来源**：平均重试次数、失败原因分布、成本明细
    全部从这里统计，禁止在别处另建计数（4.5 明文）。

    为什么必须独立成类而不是塞进 QAResult：QAResult 描述「这一次判得怎么样」，
    AttemptLog 描述「这一次是怎么失败的、改了什么」。第 9.4 节第 15 条禁止虚构数据，
    而 C2 通道的归因回路（第 10.4 节 fail_reason 命名规范）只有在这里才有落点。
    """

    attempt: int = Field(ge=1, description="第几次尝试，从 1 开始")
    stage: Literal["keyframe", "video"] = Field(description="失败发生在哪个阶段")
    score: int | None = Field(default=None, description="本次判定的分数，无判定则为空")
    fail_reason: str | None = Field(default=None, description="失败原因，命名规范见 10.4")
    prompt_delta: str | None = Field(default=None, description="本次重试相对上次改了什么")
    cost: float = Field(default=0.0, ge=0.0, description="本次调用的实际花费（元）")
    ts: str = Field(description="ISO8601 时间戳")


class Shot(BaseModel):
    """分镜：一个可独立生成、独立重试、独立替换的最小单位。"""

    shot_id: str = Field(pattern=r"^shot_\d{2}$")
    index: int
    scene_id: str
    duration: float = Field(
        default=5.0,
        validate_default=True,
        description="必须是平台档位值，取自 config.PLATFORM_TIERS",
    )
    camera: str = Field(description="景别 + 运镜，如「中景，缓慢推近」")
    action: str = Field(description="画面里发生什么，一句具体动作")
    character_ids: list[str] = Field(default_factory=list)
    # 空镜/纯场景镜头（环境交代、道具特写、纯风景）必须显式置 True。
    # **与 character_ids 严格互斥**：两者都有 = 语义矛盾（既说没角色又列了角色），
    # 两者都无 = 没说清这是什么镜头。校验在 Script.check_references 里做。
    environment_only: bool = Field(
        default=False, description="纯场景镜头标记；为 True 时 character_ids 必须为空"
    )
    dialogue: str | None = None
    emotion: str = ""

    image_prompt: str | None = Field(default=None, description="S3 组装出来的完整提示词")
    # 视频提示词必须与图像提示词**分开存**。第 8.2 节的核心规则是：图像提示词写全画面，
    # 视频提示词只写运动、绝不复述外貌，两者语法几乎相反。合并成一个字段，
    # 这条规则将从「做不到」退化为「没地方做」——所以这个字段是结构性的，不是可选项。
    video_prompt: str | None = Field(default=None, description="S4 组装，只写运动，见 8.2")
    keyframe_path: str | None = None
    video_path: str | None = None
    qa: QAResult | None = None
    attempt_log: list[AttemptLog] = Field(
        default_factory=list, description="逐次尝试记录，报告与成本统计的唯一来源"
    )

    @field_validator("duration")
    @classmethod
    def duration_must_fit_platform_tier(cls, v: float) -> float:
        """duration 必须直接落在平台档位上，**模型层不做取整**（第 4.7.1 节第 1 条）。

        传 7.0 直接抛错而不是静默取整——档位是平台能力边界（R1），不是审美选择。
        取整是 `pipeline/prompt_builder.py` 的职责（14.4-E
        `test_duration_snaps_to_platform_tiers`）。两层都取整会让模型层
        误以为已合规，而提示词里实际写的是 7s。

        用容差比较而非 `in`：档位是 float，7.0 与 7 在序列里 `in` 判定不稳，
        且外部可能传 5.0000001 这种浮点误差值。
        """
        tiers = sorted(PLATFORM_TIERS)
        if not any(abs(v - t) < 1e-6 for t in tiers):
            raise ValueError(
                f"duration={v} 不在平台档位 {tiers} 内。"
                "模型层只接受档位值，取整请交给 prompt_builder"
            )
        return v


class Script(BaseModel):
    """一部片子的完整计划。"""

    title: str
    logline: str
    style: VisualStyle
    characters: list[Character]
    scenes: list[Scene]
    shots: list[Shot]

    @model_validator(mode="after")
    def check_references(self) -> "Script":
        """引用完整性校验：这是真正的结构校验，不是数个数。"""
        char_ids = {c.char_id for c in self.characters}
        scene_ids = {s.scene_id for s in self.scenes}

        if not self.shots:
            raise ValueError("分镜表不能为空")

        for shot in self.shots:
            missing_chars = set(shot.character_ids) - char_ids
            if missing_chars:
                raise ValueError(f"{shot.shot_id} 引用了不存在的角色：{sorted(missing_chars)}")
            if shot.scene_id not in scene_ids:
                raise ValueError(f"{shot.shot_id} 引用了不存在的场景：{shot.scene_id}")

            # 角色镜头 vs 纯场景镜头：**互斥且必须二选一**。
            # 只加 environment_only 字段而不改这条校验，空镜照样进不来——所以这处逻辑
            # 是和字段一起改的，不是可选的后续优化。
            has_chars = bool(shot.character_ids)
            if shot.environment_only and has_chars:
                raise ValueError(
                    f"{shot.shot_id} 标了 environment_only=True，却列了角色 "
                    f"{shot.character_ids}；两者互斥，请二选一"
                )
            if not shot.environment_only and not has_chars:
                raise ValueError(
                    f"{shot.shot_id} 既没角色也没标 environment_only；"
                    "纯场景镜头必须显式置 environment_only=True"
                )

        indexes = [s.index for s in self.shots]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError(f"分镜 index 必须从 1 连续递增，当前：{indexes}")

        return self

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.shots)

    def shots_of(self, char_id: str) -> list[Shot]:
        return [s for s in self.shots if char_id in s.character_ids]


class CostRecord(BaseModel):
    """单次外部调用记账。估算价与实际价分开记，才能事后校准。"""

    provider: str
    operation: str
    estimated_cost: float = 0.0
    actual_cost: float = 0.0
    success: bool = True
    error: str | None = None
