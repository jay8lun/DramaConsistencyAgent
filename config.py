"""集中配置：所有密钥从环境变量读，代码里不出现任何 key。

本项目独立发布，**只读项目根目录下的 .env**，不向上级目录查找、不依赖任何外部
仓库的配置。首次使用请复制 `.env.example` 为 `.env` 并填入密钥。
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)

# 产物根目录。**多项目隔离**：一部片子一个目录，互不干扰。
#   assets/projects/{project_id}/
#       script.json  characters.json  project.json
#       anchors/     定妆照（正脸 / 侧视 / 背视）
#       keyframes/   每镜首帧
#       shots/       每镜视频
#       voice/       配音
#       refs/        用户上传的参考图与反推结果
#
# 2026-11-02 删除了早期扁平布局的 5 个常量（assets/{scripts,anchors,keyframes,videos,output}）：
# 迁移到多项目隔离后它们一直空着，但 ensure_dirs() 每次启动都重建，
# 结果看起来像"这些目录里本该有东西、只是丢了"。全项目除本文件外无任何引用，
# 删掉不影响启动与运行。早期项目（assets/_archive/）按旧布局存在，只作历史存档、不再运行。
ASSETS = ROOT / "assets"
PROJECTS_DIR = ASSETS / "projects"
DATA_DIR = ROOT / "data"
# 状态库。**状态不放内存**（第 7.5 节）：Web 与 worker 都通过它读写进度。
STATE_DB = DATA_DIR / "state.db"


def project_dir(project_id: str) -> Path:
    return PROJECTS_DIR / project_id

# 平台支持的视频时长档位（秒）。**这是平台能力边界，不是审美选择。**
#
# 第 8.3 节：时长档位 5s / 10s，非任意秒数；分镜 duration 必须直接是档位值，
# 超时拆镜。把任意秒数取整到档位是 `pipeline/prompt_builder.py` 的职责
# （14.4-E `test_duration_snaps_to_platform_tiers`），**不是模型层的职责**。
# 两层都做取整会导致「模型层以为已合规，实际提示词里写的是 7s」。
#
# 为什么是模块级常量而不是写死在 validator 里（第 4.7.1 节第 2 条）：
# R1 尚未实测（10s 是否真支持 9s），档位表可能要改。写成常量，改档位只动这一行；
# 写死在 validator 里，改档位要同时改 Schema 和测试。
#
# **2026-10-29 改为从 .env 读，默认只有 5 秒。** 教训：这里原本写死 `(5.0, 10.0)`，
# 而当前视频模型 `wan2.2-i2v-flash` **只支持 5 秒**。S1 的提示词照着这个表告诉模型
# 「duration 只能取 5 或 10」，模型给 shot_05 写了 10 秒 → provider 拒绝 →
# S4 把它当网络抖动重试 2 次 → 白等 6 分钟、那一镜直接 failed。
# **档位表必须反映「当前这个模型支持什么」，不是「行业里常见什么」。**
# 换模型（如 wan2.6-i2v 支持 2~15s）时，改 `.env` 的 VIDEO_SUPPORTED_SECONDS 即可。
def _parse_tiers(raw: str) -> tuple[float, ...]:
    vals = tuple(sorted(float(x) for x in raw.replace("，", ",").split(",") if x.strip()))
    return vals or (5.0,)


PLATFORM_TIERS: tuple[float, ...] = _parse_tiers(os.getenv("VIDEO_SUPPORTED_SECONDS", "5"))

# 判定阈值（第 7.1 节）：≥85 通过 / 60~85 带增量重取 / <60 降级。
# 与档位同理，不写死在 judge.py 里——换判定模型必须重新标定这两条线。
#
# **这两个数字属于「当前判定模型」，不是项目的通用常量。**
# 2026-10-29 判定器从豆包 turbo 换成 qwen3-vl-plus，同一批 30 对图重标：
#   豆包  同人 P5=90.5  异人 P95=62.5  → 85 有余量，可用
#   qwen  同人 P5=85.0  异人 P95=65.0  → 同人组下移到 85，85 的余量归零
# qwen 实测分布：同人 [85,85,95×13]；异人 [15×8,25,25,35,35,45,65,65]
# 取 75 = 空档（65~85）正中，两侧各留 10 分。M0 上 FP=0/15、FN=0/15。
#
# FLOOR 维持 60 不动：它是「放弃重试线」，而重试一次只要 ¥0.029，
# 误判成 degraded 却要让一张坏首帧进视频（¥1.00）。宁可多试，不可早弃。
JUDGE_PASS_LINE: float = float(os.getenv("JUDGE_PASS_LINE", "75"))
JUDGE_FLOOR_LINE: float = float(os.getenv("JUDGE_FLOOR_LINE", "60"))

# 提示词字符上限。**待验证**：8.3 只写「部分平台输入框限制字符数」，没给具体值，
# 本项目没有实测过任何平台的输入框上限。先取 800 作保守值，实测后按平台覆盖。
PROMPT_MAX_CHARS: int = int(os.getenv("PROMPT_MAX_CHARS", "800"))


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    base_url: str
    model: str

    @property
    def usable(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


@dataclass(frozen=True)
class Settings:
    script_llm: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        )
    )
    # 判定用的视觉模型。厂商无关（A.8 第 2 条）：变量名不带任何厂商名，
    # 换厂商只改 .env。默认走阿里云百炼的 OpenAI 兼容端点，改 .env 即可换成
    # 火山方舟 / 智谱等同样提供 OpenAI 兼容协议的端点。
    judge_vlm: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("VLM_API_KEY", ""),
            base_url=os.getenv("VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model=os.getenv("VLM_MODEL", "qwen-vl-max"),
        )
    )
    # 判定时是否开启模型的「深度思考」。默认开启，理由是实测：
    # 同一对图（01_a / 01_b，四分之三侧面），关思考时模型把「侧视导致的一只眼
    # 露出面积变小」判成眉眼不一致（假阴性），开思考时判为一致且 diff 为空。
    # 判定准确率是 M0 的唯一产出，宁可慢也不要用错误的快。
    judge_vlm_thinking: bool = field(
        default_factory=lambda: os.getenv("VLM_THINKING", "1").strip() not in ("0", "false", "False")
    )
    # 判定并发数。开启思考时单次判定约 150~170 秒，串行 30 对要 80 分钟以上；
    # 并发把总时长压到十几分钟。QPS 上限随模型与账号档位变化，默认保守取 4。
    judge_vlm_concurrency: int = field(
        default_factory=lambda: max(1, int(os.getenv("VLM_CONCURRENCY", "4")))
    )
    ark: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("DOUBAO_API_KEY", ""),
            base_url=os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
            # DOUBAO_MODEL 通常形如 ep-xxxxxxxx，是推理接入点 ID 而非模型名，
            # 绑死了某一个模型，与出图/出片所用的模型无关。此处仅作「方舟已配置」的标记。
            model=os.getenv("DOUBAO_MODEL", ""),
        )
    )

    # ============================================================
    # TODO 模型待定 —— 尚未选定供应商与型号，以下三项默认留空。
    #
    # 出图与出片是两个不同的模型，必须在平台上**分别开通**、分别计费。
    # 候选（价格与能力待实测，见计划书 R1~R4）：
    #   出图：火山 Seedream / 阿里通义万相 / 即梦
    #   出片：火山 Seedance / 可灵 Kling / 海螺
    #
    # 用法：开通后把模型名填进 .env 的 ARK_IMAGE_MODEL / ARK_VIDEO_MODEL，
    #      不要写死在代码里 —— 换供应商时只改 .env，不动代码。
    # 注意：代码在 provider 层做适配，所以这里的变量名带 ARK_ 前缀只是沿用
    #      现有 .env 的命名，不代表最终绑定火山。
    # ============================================================
    image_model: str = field(default_factory=lambda: os.getenv("ARK_IMAGE_MODEL", "").strip())
    video_model: str = field(default_factory=lambda: os.getenv("ARK_VIDEO_MODEL", "").strip())
    video_resolution: str = field(default_factory=lambda: os.getenv("ARK_VIDEO_RESOLUTION", "").strip())

    # ============================================================
    # 出图 / 视频 的供应商配置（2026-10-28 定案，取代上面三个待定字段）
    #
    # 变量名**不带厂商名**（A.8 第 2 条）：换供应商只改 .env，不动代码。
    # 目前的实际映射（记录在此，代码里不出现厂商名）：
    #   image_api → 中转站，模型 gpt-image-2.5-sunburst，¥0.02/张（OpenAI 兼容协议）
    #   video_api → 阿里百炼，模型 wan2.2-i2v-flash，720P 无声 ¥0.2/秒
    #   实际用哪套出图由下面的 image_provider 决定，两套可以同时配着，切换只改一个变量。
    #
    # 为什么分成两套而不是一套：两者是不同的服务、不同的鉴权、不同的计费口径，
    # 而且实测中转站会整段不可达（约 25 分钟），出图需要有独立的回退来源。
    # ============================================================
    image_api: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("IMAGE_API_KEY", "").strip(),
            base_url=os.getenv("IMAGE_BASE_URL", "").strip(),
            model=os.getenv("IMAGE_MODEL", "").strip(),
        )
    )
    video_api: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            api_key=os.getenv("VIDEO_API_KEY", "").strip(),
            base_url=os.getenv("VIDEO_BASE_URL", "https://dashscope.aliyuncs.com").strip(),
            model=os.getenv("VIDEO_MODEL", "wan2.2-i2v-flash").strip(),
        )
    )
    # 出图的第二套：DashScope 原生「多模态生成」接口。
    #
    # 为什么要有第二套（2026-10-29）：中转站整站故障，`/images/generations` 与
    # `/images/edits` 全部 503，连不传 prompt 都 503（说明没走到参数校验层）。
    # 出图一旦单点，S2/S3 直接停摆，整条流水线跑不动。判定与视频都已经有第二家，
    # 只有出图没有，这是当时唯一的结构性缺口。
    #
    # 默认复用 video_api 那把 key（同一个百炼账号，一个 key 通用），
    # 所以 .env 里通常只需要设 IMAGE_NATIVE_MODEL，不用再贴一次密钥。
    image_native_api: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            api_key=(os.getenv("IMAGE_NATIVE_API_KEY", "").strip()
                     or os.getenv("VIDEO_API_KEY", "").strip()),
            base_url=os.getenv("IMAGE_NATIVE_BASE_URL", "https://dashscope.aliyuncs.com").strip(),
            model=os.getenv("IMAGE_NATIVE_MODEL", "wan2.7-image").strip(),
        )
    )
    #: 出图用哪一套：`relay` = OpenAI 兼容（中转站）；`dashscope` = 原生多模态生成。
    #:
    #: 默认 dashscope，因为 2026-10-29 实测中转站出图不可用，而原生接口三个场景
    #: 全部通过（文生图 18s / 单图保脸 7.6s 判 95 分 / 双图参考 8.1s）。
    #: **换回 relay 时记得同步改 PRICE_IMAGE**：relay ¥0.02/张，dashscope ¥0.20/张，
    #: 差 10 倍，护栏的分母错了就等于没有护栏。
    image_provider: str = field(
        default_factory=lambda: os.getenv("IMAGE_PROVIDER", "dashscope").strip().lower()
    )
    # 视频分辨率档位。wan2.2-i2v-flash 支持 480P/720P/1080P，
    # 720P 是「人脸够 VLM 判、又不付 1080P 溢价」的平衡点。
    video_resolution_level: str = field(
        default_factory=lambda: os.getenv("VIDEO_RESOLUTION", "720P").strip()
    )

    # S1 剧本结构化用的文本模型。
    #
    # **不要回退到 VLM_*。** 2026-10-29 真实踩到：判定的 `VLM_*` 从方舟换到百炼之后，
    # 这个字段原本写着「api_key 回退到 VLM_API_KEY、base_url 回退到 VLM_BASE_URL」，
    # 于是 S1 拿着豆包的模型名 `doubao-seed-2-1-pro-260628` 去请求百炼端点，
    # 直接 404 model_not_found，整条流水线在第一步就停。
    # 教训：**两个不同厂商的配置不能互相回退**。模型名与端点是绑定的，
    # 回退只搬了 key/base 却没搬 model，必然错配。现在只认 TEXT_*（或显式的 DOUBAO_API_KEY）。
    text_api: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            api_key=(os.getenv("TEXT_API_KEY", "").strip()
                     or os.getenv("DOUBAO_API_KEY", "").strip()),
            base_url=(os.getenv("TEXT_BASE_URL", "").strip()
                      or "https://ark.cn-beijing.volces.com/api/v3"),
            model=os.getenv("TEXT_MODEL", "").strip() or "doubao-seed-2-1-pro-260628",
        )
    )
    #: S1 是否关闭文本模型的深度思考。
    #:
    #: 实测（2026-10-28，doubao-seed-2-1-pro 写 6 镜剧本，同一 prompt）：
    #:   开：193 秒，completion 9,891 token（其中 reasoning 8,290，占 84%），JSON 合法
    #:   关： 28 秒，completion 1,299 token（reasoning 0），              JSON 合法
    #: **快 7 倍、省 87% token，产出质量无差别。**
    #: 写剧本是「按格式填内容」，不需要推理链；而这个差距直接决定 S1 会不会撞超时——
    #: 193 秒离 300 秒的 provider 超时只有 1.5 倍余量，一旦排队就崩。
    text_disable_thinking: bool = field(
        default_factory=lambda: os.getenv("TEXT_DISABLE_THINKING", "1").strip()
        not in ("0", "false", "False")
    )

    @property
    def ark_ready(self) -> bool:
        """方舟是否具备发起调用的条件（凭据层面）。

        判据是 key 与 base_url 齐备，**不能用 `ark.usable`**：后者要求 model 非空，
        而 ark.model 是接入点 ID，拿它判断会让配置看起来「不可用」。
        注意：本属性只说明凭据齐了，不代表出图/出片的模型已开通。
        """
        return bool(self.ark.api_key and self.ark.base_url)

    # 预算护栏：任何一次运行的总花费不超过这个数（元）。默认 100 取自计划书 12.3 原文
    # 「`BUDGET_CAP` 默认 100 元，可配」。骨架里曾写 50，无任何依据，已对齐。
    #
    # **这个数的语义是「最多愿意损失多少」，不是「够跑几部」。**
    # 依据 D7 与 12.3 的原话：设它是为了防「某镜陷入重试循环、无人看管时烧光预算」，
    # 它是保险丝，不是预算表。所以它跟单价无关，只跟心理承受线有关。
    #
    # 换算参考（**两个口径差 3.3 倍，A.6 校准前都无法确认**）：
    #   按 12.1 的 ¥0.3/秒  → 单部 ≈ ¥22.2 → 100 元够 4.5 部
    #   按 PRICE_VIDEO_PER_SEC 的 0.994 → 单部 ≈ ¥59.6 → 100 元够 1.7 部
    # 现在的 0.994 **没有出处**（骨架注释只写「火山的量级参考」），却比计划书高 3.3 倍。
    # 我保留它不动，理由是护栏宁可高估单价：高估的后果是提前停（可恢复），
    # 低估的后果是超支（不可恢复）。定出片模型后按 A.6 校准，同时回填 12.1。
    budget_cap: float = field(default_factory=lambda: float(os.getenv("BUDGET_CAP", "100")))
    # 每个镜头最多重试几次
    max_retry_per_shot: int = field(default_factory=lambda: int(os.getenv("MAX_RETRY_PER_SHOT", "2")))

    # 单价（元），用于估算与封顶。**这两个数是护栏的分母，错了护栏就等于没有。**
    #
    # 2026-10-28 按已落地的真实供应商校准（不再是"量级参考"）：
    #
    # 出图单价。**必须与 image_provider 同步**，两家差 10 倍：
    #   dashscope（wan2.7-image，默认）¥0.20/张
    #   relay（gpt-image-2.5-sunburst）¥0.02/张
    #   方舟 seedream-5-0-pro ¥0.30/张（且当前未开通）
    # 默认值取 0.20 而不是 0.02：默认 provider 是 dashscope，
    # 默认值若偏小十倍，护栏在「忘了配 .env」的情况下会静默失效——护栏宁可高估。
    price_image: float = field(default_factory=lambda: float(os.getenv("PRICE_IMAGE", "0.20")))
    # 视频：阿里百炼 wan2.2-i2v-flash，720P 无声 **¥0.2/秒**（平台价目接口返回）。
    #   5 秒一条 = ¥1.00。
    #   两个已作废的旧值，都错在不同方向：
    #     计划书 12.1 的 ¥0.3/秒 —— 那是 480p 文生视频档位，不是 720P i2v；
    #     config 旧值 0.994   —— 对应方舟 Seedance 2.0 720p 的官方价表示例，
    #                            而方舟 i2v 实际区间 5.44~12.10 元/条（1.09~2.42 元/秒），
    #                            0.994 仍偏低 2.4 倍。
    price_video_per_sec: float = field(default_factory=lambda: float(os.getenv("PRICE_VIDEO_PER_SEC", "0.2")))

    # VLM 判定单价（元/次）。A.6 要求定模型后按实际报价校准。
    #
    # 2026-10-28 按 M0-a 完整 30 对实测校准（取代 10-27 的 20 对中间值）：
    #   输入 3,082 + 输出 7,744 ≈ 10,826 token/次，**输出占 72%**（思考 token 计入输出）
    #   30 对合计 324,786 token
    #   按 3/9 元每百万 → ¥0.079/次；按 8/8 元每百万 → ¥0.087/次
    #
    # 配置值取 0.08（区间中值）。**预算护栏请按 0.09 的保守口径核对**，
    # 因为思考 token 随样本难度波动很大（实测单次 reasoning_tokens 从 2,939 到 12,989）。
    #
    # 生产阶段含义：单部 6 镜 × 2 次判定 = 12 次 ≈ 13 万 token ≈ ¥0.95。
    # 钱不是问题，**额度是**：单模型 50 万免费额度只够 1.5 部。
    price_vlm_per_call: float = field(default_factory=lambda: float(os.getenv("PRICE_VLM_PER_CALL", "0.01")))

    # ffmpeg / ffprobe 所在目录（可选）。留空则从系统 PATH 找。
    #
    # **为什么要有这个配置项**：7.6 要求 ffmpeg 必须带 libx264 与 fontconfig，
    # 但很多机器（含本机）没有 winget/scoop/choco，装完还得改系统 PATH —— 改 PATH
    # 会影响机器上其他程序，属越界操作。给一个目录配置项，就能做到「项目自带依赖、
    # 删目录即回滚」。填的是解压后 ffmpeg 的 `bin` 目录。
    ffmpeg_dir: str = field(default_factory=lambda: os.getenv("FFMPEG_DIR", ""))

    # edge-tts 的 `Sec-MS-GEC-Version`。**这个值会随微软升级服务端而失效，失效后
    # 表现为清一色 `NoAudioReceived`（不是报错，是「连上了但没音频」，极难猜）。**
    #
    # 2026-10-28 实测：edge-tts 7.3.1 库内硬编码 `2-143.0.3650.75`，用它 0/5 全失败；
    # 改成 `2-155.0.4283.45` 后 4/4 成功（两个音色、长短句都试过）。
    # 端点 `speech.platform.bing.com` **不用换**——外派单提到的新端点
    # `api.msedgeservices.com` 经实测并非必需，根因只是这个版本号常量过期。
    #
    # 下次再失效时：查 edge-tts 的 issue 或本机 Edge 的版本号（`msedge.exe` 的
    # ProductVersion，取前四段），拼成 `2-<版本>` 填进 .env 的 `TTS_SEC_VERSION`。
    tts_sec_version: str = field(
        default_factory=lambda: os.getenv("TTS_SEC_VERSION", "2-155.0.4283.45")
    )

    # 剧本结构化（S1）的文本调用单价。**默认 0**：12.1 的单价表没单列这一项，
    # 也没核实过具体报价，先记 0 而不是编一个数——账本里仍会留下审计记录（调用次数、
    # provider、成败），只是不参与预算累计。等 A.6 定文案模型后再校准。
    # 量级参考：一次 S1 约 3~5k token，按主流文案模型价格约 ¥0.005/部，可忽略。
    price_llm_per_call: float = field(
        default_factory=lambda: float(os.getenv("PRICE_LLM_PER_CALL", "0"))
    )


def ensure_dirs() -> None:
    """只建真正在用的两个目录。

    2026-11-02 起不再创建 assets/ 下那 5 个空的扁平目录（已随常量一起删除）。
    少建目录不影响任何阶段：每个项目的子目录由 `pipeline/project.py` 在需要时按项目建。
    """
    for d in (PROJECTS_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def find_bin(name: str) -> str | None:
    """查找外部可执行文件：先看 `FFMPEG_DIR`，再退到系统 PATH。

    两处（`s5_compose.find_ffmpeg`、`state` 的视频探测）都走这里，避免各写一份
    查找逻辑后行为漂移——尤其在 Windows 上要补 `.exe` 后缀，漏写就会误判「不可用」。

    **必须支持带版本号的文件名**：`imageio-ffmpeg` 打的包叫
    `ffmpeg-win-x86_64-v7.1.exe`，不是 `ffmpeg.exe`。只做精确匹配会把它判成不可用，
    而它恰恰是能力完整的那份（gyan.dev essentials build，含 libx264 + libass +
    fontconfig）。前缀匹配用 `glob(f"{name}*{suffix}")`：`ffmpeg` 不会误匹配到
    `ffprobe`，两者前缀不同。

    返回绝对路径或 None。**返回 None 不等于失败**：调用方有降级路径
    （S5 输出无损片段包、状态机退到体积判据），必须如实记录而不是静默跳过。
    """
    if settings.ffmpeg_dir:
        d = Path(settings.ffmpeg_dir)
        suffix = ".exe" if os.name == "nt" else ""
        exact = d / f"{name}{suffix}"
        if exact.is_file():
            return str(exact)
        for cand in sorted(d.glob(f"{name}*{suffix}")):
            if cand.is_file():
                return str(cand)
    return shutil.which(name)


settings = Settings()
