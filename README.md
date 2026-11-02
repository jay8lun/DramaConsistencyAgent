# DramaConsistencyAgent

输入一段短剧创意，输出一条**人物长相前后一致**的成片。

它解决的问题很具体：多镜头、多张图分别生成时，同一个人往往在第二镜就换了张脸。
这里把「像不像同一个人」变成可打分的判定，判不过就自动带着差异重画，最后连成片、
配字幕，并出一份能逐镜对比的报告。

流程：剧本结构化 → 定妆照 → 每镜首帧 → 一致性判定与定向重取 → 图生视频 → 合成 → 报告。

> 本仓库只保留了 API 全自动这一条通道。原先还有一条浏览器接管通道（C2），
> 因为平台要求开通会员才能解锁对应清晰度、而账号积分从始至终没被扣过一次，已整体移除。
> 结项时保留了什么、删了什么，看 [`docs/交付清单.md`](docs/交付清单.md)；
> 各阶段用哪个模型、单价多少，看 [`docs/模型清单.md`](docs/模型清单.md)。

---

## 准备三样东西

| 需要 | 说明 |
|---|---|
| **Python 3.11+** | 开发与实际运行都在 3.12 |
| **ffmpeg** | 拼接与烧字幕用，要带 libx264、libass 和中文字体。装好后在 `.env` 的 `FFMPEG_DIR` 填路径，留空则从 PATH 找 |
| **两个平台的密钥** | 火山方舟（写剧本）＋ 阿里百炼（看图判定、出图、出片） |

```bash
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

打开 `.env`，按注释填 4 个模型的位置。整个项目只需要这 4 个模型，其余参数都有默认值，
不用改。文件名与用途在注释里写清了，照着填即可。

## 跑起来

```bash
python run.py --mode api
```

浏览器打开 <http://127.0.0.1:8000>，在界面里建项目、填创意、传参考图，按提示点确认。

服务默认监听 `0.0.0.0:8000`。**访问时要用 `127.0.0.1` 或 `localhost`**——`0.0.0.0`
是「监听所有网卡」的写法，本身不是一个可访问地址，填进浏览器会打不开。

想先看一遍完整流程又不花钱：

```bash
python run.py --mock --project-id demo01 --auto
```

`--mock` 用假实现替换全部外部调用，流程、目录结构、报告都和真跑一样，只是图是色块、视频几秒钟。

## 界面上有三个停点

流水线不会一口气跑到尾，它在三个地方停下来等你，因为后面每一步都比前面贵十倍。

| 停点 | 你要做的 | 为什么停在这 |
|---|---|---|
| **① 角色来源** | 有人物参考图就传（按顺序绑 `char_01`、`char_02`）；没有就让模型自行设计 | 后面所有的脸都从这一步长出来，参考图决定上限 |
| **② 角色卡与分镜** | 改外貌、服装，改每镜的动作和出场人物 | 这一步只动文字、不花钱，但决定了六张首帧画什么 |
| **③ 正脸定妆照** | 看正脸像不像你要的人，不满意就写意见重出一张 | 侧背、首帧、视频全部以这张脸为锚，这里歪了后面全歪 |

三步都点了「确认」才继续。停点进度写在项目目录的 `project.json` 里，关掉网页、
重启服务都不会丢。

## 花多少钱

| 项目 | 单价 |
|---|---|
| 写剧本（文本模型） | ¥0.01 – 0.02 / 次 |
| 看图反推、一致性判定 | ¥0.01 / 次 |
| 定妆照、每镜首帧 | ¥0.20 / 张 |
| 5 秒视频（720P） | ¥1.00 / 条 |
| 配音（edge-tts） | 免费 |

一部六镜的片子实际落在 **¥8 – 12**，其中视频占 73%。单镜重跑约 ¥1.20。

两道闸：`.env` 里的 `BUDGET_CAP` 是硬上限，超了会在调用前报错停住，不会烧到一半才发现；
命令行必须显式带 `--allow-paid`（界面上是勾选付费确认）才允许真实调用。

## 已经跑出来的三部片

| 项目 | 实扣 | 说明 |
|---|---|---|
| `web1793586789` | ¥12.27 | 主样例。唯一走完「参考图反推 → 三个停点 → 单镜重出 → 报告」全流程的 |
| `bv2` | ¥8.56 | 只有文字创意、没有参考图，六镜一次过，可用来对照纯文生图的效果上限 |
| `six01` | ¥6.40 | 首次完整跑通，最早期基线 |

每部片的目录里都有 `final.mp4`（成片）、`report.html`（逐镜对比墙 + 成本明细）、
`script.json`、`characters.json`，以及 `anchors/`（定妆照）、`keyframes/`（首帧）、
`shots/`（单镜视频）、`costs.json`（逐笔账）。直接双击 `report.html` 就能看。

这三部片的产物不进仓库（体积以百 MB 计），克隆下来是干净的代码库，跑一部就有了。

## 日常操作的四个场景

**跑到一半想停。** 点「停止」。它不会强杀进程，而是等**当前这一镜跑完**再停——已经花钱
出的图、已经提交的视频任务都保住，账面也有记录。停下后点「继续生成」从原地接着跑。

**某一镜不满意。** 在进度页点该镜的「重跑」，写下理由（例如「白布应该在奶龙手上」）。
理由会拼进提示词，只重做这一镜，其它镜不动。重跑后旧的 `final.mp4` 会被删掉、阶段退回
「进行中」，需要再点一次「继续生成」重新合成——合成不花钱，只是重新拼接。

**剧本或角色卡要改。** 停点 ② 里直接改，改完再确认。改完分镜描述后如果要重跑已有镜头，
用「重跑」而不是「继续生成」，否则已完成的镜头会被跳过、你以为改了其实没生效。

**想删掉一个项目。** 首页项目卡上的「删除」，需要手动输入项目名才执行。它会连磁盘目录
一起删，不可恢复。

## 命令行

界面之外，同一套能力也能用命令行驱动，适合放进脚本或定时任务。

```bash
# 建项目，停在「角色有参考图吗」
python run.py --project-id mydrama --shots 6 --allow-paid --budget 9.5

# 有参考图就传（按顺序绑 char_01、char_02）
python run.py --project-id mydrama --ref 桌面/她.png

# 确认 → 写剧本 → 停在下个确认点；再确认 → 出正脸
python run.py --project-id mydrama --confirm

# 正脸不满意：带意见重出一张，只动这一张
python run.py --project-id mydrama --regen-front char_01 --feedback "脸再瘦一点，眼睛别这么凶"
#   加 --update-card 才会同时改角色卡（会影响所有含该角色的镜头）

# 重跑某一镜
python run.py --project-id mydrama --rerun-shot shot_03 --rerun-reason "白布应该在奶龙手上"

# 不想被问：一次跑完
python run.py --project-id mydrama --auto --allow-paid

# 只看要花多少，不调任何接口
python run.py --project-id mydrama --shots 6 --dry-run
```

不加 `--auto` 时，每条命令跑一个阶段就停，并打印下一步该敲什么。

## 用 Docker 部署

镜像里已经装好 ffmpeg 与中文字体（Noto Sans CJK）。

```bash
docker pull python:3.12-slim
docker build -t drama-agent:1.0 .
```

运行时两个挂载卷不能省——`assets/projects/` 放成片与中间产物，`data/` 放状态库，
不挂的话容器一删，成片和断点记录一起没了：

```bash
docker run --rm -p 8000:8000 \
  -v "$(pwd)/assets/projects:/app/assets/projects" \
  -v "$(pwd)/data:/app/data" \
  --env-file .env \
  drama-agent:1.0 --mode api
```

Windows 上请在 **PowerShell** 里跑，不要用 Git Bash——它会把 `$(pwd)` 的路径改写成
`...\assets\projects;C` 这种脏值，Docker 认不出就静默退回匿名卷：

```powershell
$root = "D:\path\to\DramaConsistencyAgent"
docker run -d --name drama-api -p 8000:8000 `
  -v "${root}\assets\projects:/app/assets/projects" `
  -v "${root}\data:/app/data" `
  --env-file "${root}\.env" `
  drama-agent:1.0 --mode api
```

挂对了的判据：`docker exec drama-api ls /app/assets/projects` 能看到你的项目目录名。

「页面能开但项目列表是空的」几乎都是这一种原因：从 Docker Desktop 界面点运行按钮不会带
端口映射也不带挂载，`docker ps` 里 `PORTS` 只显示 `8000/tcp`，那不是发布。必须走命令行。

网络受限时（`docker pull` 卡在某个 `Pulling fs layer` 超过两分钟、或 `docker build` 卡在
`apt-get update` 超过三分钟，就是断了而不是慢）：

```bash
docker pull docker.1ms.run/library/python:3.12-slim
docker tag docker.1ms.run/library/python:3.12-slim python:3.12-slim
```

镜像内 apt 与 pip 用的是官方源。要换源请临时改一份 `Dockerfile`，构建完还原——
不要把国内源提交进仓库，否则海外用户反而变慢。

## 自检：确认它还能跑

```bash
python tests/run_all.py                   # 586 项，约 30 秒，最后给一行汇总
python tests/run_all.py -k phase          # 只跑文件名含 phase 的
python tests/verify_docker.py --runtime   # 追加真容器验证（构建镜像 + 全 mock 跑一遍）
```

全是离线自检，不联网、不花钱。它们不是 pytest，而是按阶段拆成的 12 个独立脚本，
每个自带计数和退出码，可以单独跑也可以汇总——所以 `requirements.txt` 里没有 pytest。

## 目录

**完整的目录树和每个文件的职责，见 [`docs/交付清单.md` 第一节「最终目录」](docs/交付清单.md)。**

一句话版：`run.py` 是唯一入口；`pipeline/` 按 s1 → s6 顺序执行；`providers/` 是四类供应商
（换源只改 `.env`）；`adapters/web/` + `web/` 是界面；`tests/` 是零花费离线自检；
`tools/` 是一次性脚本。

不进仓库的（`.gitignore` 已排除）：`assets/projects/`（每部片的产物）、
`assets/_archive/`（早期项目存档）、`data/state.db`（账目与镜头状态）、`.env`（密钥）。

**建议的阅读顺序**：本文件（怎么用）→ [`docs/模型清单.md`](docs/模型清单.md)（谁在哪个阶段、
多少钱）→ [`docs/交付清单.md`](docs/交付清单.md)（保留了什么、已知限制）→ `tests/`（怎么验证）。

## 已知限制

1. **剧本仍可能不完全遵守参考图。** 参考图会先被读成一份「人物事实」再交给剧本模型，
   目前能写对（性别年龄留白、动作写「抬起戴白手套的手」），但这仍是提示词约束、不是结构强制。
   所以每镜开跑前那道「动作提到的人不在场上」的预检不能撤。
2. **重出单镜不能无中生有。** 理由会拼进提示词，但模型只能画这一镜 `character_ids` 里的角色。
   要让新角色进这一镜，得先把他加进出场名单，否则结果往往是「原来的没了、新的也没来」。

## 许可

代码以 [MIT 许可证](LICENSE) 发布：可自由使用、修改、商用，保留版权声明即可。

`tests/fixtures/m0_eval/` 下的评测图片仅用于标定判定阈值，版权不在本许可证覆盖范围内。
