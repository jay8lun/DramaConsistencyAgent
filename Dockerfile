# DramaConsistencyAgent —— 运行镜像
#
#   容器内跑：全自动流水线（调 API）+ Web 界面 + S5 合成
#   启动：python run.py --mode api（不带 --project-id 时直接进 Web 模式）
#
# 镜像里**不装任何浏览器**：唯一需要浏览器的那条通道（接管本机 Chrome）已因平台
# 会员墙于 2026-10-31 移除，原因写在 README「只有一条通道」。这条约束也顺带保证
# 镜像里没有多余的自动化依赖。

FROM python:3.12-slim

# 为什么是 3.12 而不是 3.11：本地 venv 实测就是 3.12.15（requirements.txt 第 2 行），
# 镜像与开发环境同小版本，能挡掉一类"本地能跑、容器里报错"的差异。
# 项目要求是 3.11+（Pydantic v2 / `X | None` 语法），3.12 满足。
#
# ⚠ 构建卡住 → 开 VPN。
#   典型卡点两处：① 这行 FROM 的拉取（Docker Hub，国内常卡在 Pulling fs layer）
#                 ② 下面的 apt-get update（官方 Debian 源实测 54 KB/s）
#   判断标准：同一处停 2~3 分钟不动就是断了，不是慢。
#   国内网络下的绕法见 README「用容器跑 B 通道」末尾。

# ---------------------------------------------------------------- 7.6 四条硬要求
# 1. CJK 字体：不装 fonts-noto-cjk，ffmpeg 烧字幕会全是方块。
#    **这个坑构建时不报错，只在成片里出现**，所以必须在 S5 阶段就验一次。
# 2. ffmpeg：用 apt 装，不要用 pip 的静态包（缺 libx264 与字体渲染支持）。
# 3. 绝不装 chromium / playwright：容器只跑 --mode api；装了会引诱后来者把 C2 塞进来。
# 4. 挂载卷：assets/projects 与 data 在 `docker run -v` 时指定，这里声明 VOLUME
#    只是防止有人忘了挂、把产物写进容器层然后随容器一起消失。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 源一律用官方：`docker build -t drama-agent:1.0 .` 就是全部步骤（7.6「一次构建成功，
# 无需额外手动步骤」）。**不做"国内加速参数"** —— 那条分支在官方源能跑通前无法验证，
# 留着等于给后来者一个没被跑过的选项；网络受限时的绕法写在 README，不进镜像定义。
#
# 两个 cache mount 是为**弱网**准备的（实测过一次教训：下到 500MB 时代理断一下，
# apt 那层整体作废、已下载的包全丢，等于从零再来）。挂上 /var/cache/apt 与
# /var/lib/apt 后，重试时已下载的 .deb 与索引都还在，只需补下缺的那几个。
# 它们是**构建期缓存，不进镜像**，所以不影响产物大小与可移植性。
# `Acquire::Retries=10` 让 apt 自己对瞬时断连重试，而不是整层失败。
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && apt-get -o Acquire::Retries=10 install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-cjk \
        tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    # 装完字体不刷新缓存，ffmpeg 仍然找不到它——这个坑同样只在成片里显形
    && fc-cache -fv

WORKDIR /app

# 先装依赖再拷代码：只改业务代码时，这一层走缓存，重建从几分钟降到几秒
COPY requirements.txt ./
# pip 的缓存同样挂到构建缓存：弱网下断了重来只补缺的 wheel。
# 缓存目录不进镜像（这是 mount，不是 COPY），所以不再需要 --no-cache-dir。
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.txt

COPY . .

VOLUME ["/app/assets/projects", "/app/data"]

EXPOSE 8000

# 密钥只走 --env-file 传入，**禁止 COPY .env**（.dockerignore 已排除）
# 不带 --project-id 时 run.py 会启动 Web 服务，这就是容器入口的工作方式
ENTRYPOINT ["python", "run.py"]
CMD ["--mode", "api"]
