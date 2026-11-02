"""容器化离线验证（13.7 七条验收里不依赖 Docker 的那五条 + 7.6 的四条硬要求）。

**全部离线、零花费、不需要 Docker 守护进程**：只读四个文件（Dockerfile /
.dockerignore / README.md / run.py）并对 run.py 做两次子进程调用。

为什么把「验 Dockerfile」写成断言：13.7 第 3 条（中文字幕不能是方块）和第 6 条
（镜像里不能有 Chromium）**都在构建时不报错**——前者表现为成片里一堆方块，
后者表现为"以后有人顺手把 C2 塞进容器"。这两类错只能靠读文件当场拦住，
等真跑一遍才发现就太晚了。

不在这里验的两条（要真 Docker，见 13.7 第 2/4 条）：
  - `docker run --mode api` 后能跑一次 mock 全链路
  - 挂载卷后容器删除、产物仍在宿主机
放 `_verify_docker.py --runtime` 里，默认跳过。

跑法：
    python tests/verify_docker.py              # 只跑离线项（默认）
    python tests/verify_docker.py --runtime    # 追加真容器验证（会有 build/run 开销）
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

passed = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{title}")


DOCKERFILE = ROOT / "Dockerfile"
DOCKERIGNORE = ROOT / ".dockerignore"
README = ROOT / "README.md"
RUNPY = ROOT / "run.py"


# ============================================================ 一、7.6 四条硬要求
section("一、7.6 四条硬要求（缺一条就会在成片或安全上出问题）")

check("Dockerfile 存在", DOCKERFILE.exists())
df = DOCKERFILE.read_text(encoding="utf-8") if DOCKERFILE.exists() else ""
# 去掉注释再匹配：注释里大段解释了"为什么装字体"，直接搜关键词会把说明当实现
df_code = "\n".join(
    line for line in df.splitlines() if not line.lstrip().startswith("#")
)

# 基础镜像的小版本必须和本地 venv 一致（当前 3.12.*）。
# 不一致的后果不是构建失败，而是"本地跑得好、容器里某个依赖行为不同"——这类差异
# 排查成本极高，所以在这里用运行本脚本的解释器版本做锚点。
_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
check(f"基础镜像与本地 venv 同小版本（python:{_ver}-slim）",
      f"FROM python:{_ver}-slim" in df_code,
      f"本地解释器 {_ver}.{sys.version_info.micro}")

check("Dockerfile 里没有未验证的换源分支（不留没跑过的选项）",
      "build-arg" not in df_code and "APT_MIRROR" not in df_code)

check("① 装 CJK 字体（fonts-noto-cjk）",
      "fonts-noto-cjk" in df_code,
      "缺它 → ffmpeg 烧字幕全是方块，构建时不报错")
check("① 装字体后刷新缓存（fc-cache -fv）",
      "fc-cache" in df_code,
      "不刷缓存，ffmpeg 仍然找不到字体")

check("② 用 apt 装 ffmpeg",
      # 允许 `apt-get -o Acquire::Retries=10 install` 这种带选项的写法
      re.search(r"apt-get[^\n]{0,120}install[\s\S]{0,400}ffmpeg", df_code) is not None)
check("② 没有用 pip 装 ffmpeg 替代品",
      "pip install" not in df_code.split("ffmpeg")[0].replace("pip install --no-cache-dir -r requirements.txt", "")
      and "imageio-ffmpeg" not in df_code)

check("③ 镜像内不装 Chromium", "chromium" not in df_code.lower())
check("③ 镜像内不装 Playwright", "playwright" not in df_code.lower())
check("③ 镜像内不装 Selenium", "selenium" not in df_code.lower())

check("④ VOLUME 声明 assets/projects", "/app/assets/projects" in df_code)
check("④ VOLUME 声明 data", "/app/data" in df_code)
# 两个卷必须在同一条 VOLUME 里声明，否则后一条会覆盖前一条（Docker 的坑）
m_vol = re.search(r"VOLUME\s*\[([^\]]*)\]", df_code)
check("④ 两个卷写在同一条 VOLUME 里（否则后者覆盖前者）",
      bool(m_vol) and "assets/projects" in m_vol.group(1) and "/app/data" in m_vol.group(1))


# ============================================================ 二、7.6 三个坑
section("二、7.6「另外三个容易踩的坑」")

check("时区 TZ=Asia/Shanghai（否则日志与 costs.json 差 8 小时）",
      re.search(r"TZ=Asia/Shanghai", df_code) is not None)
check("入口是 python run.py", 'ENTRYPOINT ["python", "run.py"]' in df_code)
check("默认 CMD 是 --mode api", '"--mode", "api"' in df_code or "--mode api" in df_code)
check("密钥不用 COPY 进镜像", not re.search(r"COPY\s+\.env", df_code))
check("ENTRYPOINT 用 exec 形式（能收到 SIGTERM，Ctrl+C 停得掉）",
      re.search(r'ENTRYPOINT\s*\[', df_code) is not None)


# ============================================================ 三、.dockerignore
section("三、.dockerignore（13.7 第 5 条：查不到任何密钥）")

check(".dockerignore 存在", DOCKERIGNORE.exists())
di = DOCKERIGNORE.read_text(encoding="utf-8") if DOCKERIGNORE.exists() else ""
di_lines = {ln.strip() for ln in di.splitlines()}

check("排除 .env", ".env" in di_lines)
check("排除 .env 的备份文件（.env.bak*）", any(x.startswith(".env.") for x in di_lines))
check("排除 assets/projects（产物不进镜像）",
      any(x.rstrip("/") == "assets/projects" for x in di_lines))
check("排除 data（SQLite 不进镜像）", any(x.rstrip("/") == "data" for x in di_lines))
check("排除 __pycache__", any("__pycache__" in x for x in di_lines))
check("排除 .git", any(x.rstrip("/") == ".git" for x in di_lines))
# 反证：没有任何一条 !xxx 把 .env 又加回来
check("没有 !.env 之类的反向规则", not any(x.startswith("!.env") for x in di_lines))


# ============================================================ 四、README 第一屏
section("四、README 第一屏（13.7 第 7 条）")

check("README 存在", README.exists())
rd = README.read_text(encoding="utf-8") if README.exists() else ""
first_screen = "\n".join(rd.splitlines()[:40])

check("第一屏出现 --mode api", "--mode api" in first_screen)
# 2026-10-31：C2 通道移除后，第一屏的任务从「说明两条通道各在哪跑」变成
# 「说清只剩一条，且另一条为什么没了」。断言跟着事实改，不是放宽。
check("第一屏标明只有一条通道",
      "只有一条通道" in first_screen or "只保留了" in first_screen)
check("第一屏说明 C2 已移除",
      "C2" in first_screen and "移除" in first_screen)
check("第一屏给出移除原因（会员/解锁/订阅）",
      any(k in first_screen for k in ("会员", "解锁", "订阅")))
check("第一屏给出积分证据（说明不是代码问题）",
      "积分" in first_screen)
check("README 不再教人跑 --mode web",
      "--mode web --project-id" not in rd and "run.py --mode web" not in rd)


# ============================================================ 五、run.py 只做分发
section("五、run.py 是分发点，不是第二套入口（7.6 实现约束）")

check("有 --mode 参数且仍接受 web 取值（好给出可读报错，而非 argparse 原文）",
      re.search(r'"--mode",\s*choices=\("api",\s*"web"\)', RUNPY.read_text(encoding="utf-8")) is not None)
# 2026-10-31：C2 通道移除后，--channel 只剩 api。这条断言改成守"移除是干净的"，
# 而不是守一个已经不存在的映射。
check("--channel 只接受 api（c2 已移除）",
      re.search(r'"--channel",\s*choices=\("api",\)', RUNPY.read_text(encoding="utf-8")) is not None)
check("run.py 不再 import channels",
      "from channels" not in RUNPY.read_text(encoding="utf-8"))
check("run_c2_mode 已删除",
      "def run_c2_mode" not in RUNPY.read_text(encoding="utf-8"))

# 真跑两次子进程，验的是**行为**而不是源码写法
py = sys.executable

r = subprocess.run([py, "run.py", "--mode", "web"], cwd=ROOT,
                   capture_output=True, text=True, timeout=120)
check("--mode web 不带项目 ID → 非零退出", r.returncode != 0, f"exit={r.returncode}")
check("报错信息点名 --project-id（不是无声失败）",
      "--project-id" in (r.stdout + r.stderr), (r.stdout + r.stderr).strip().splitlines()[-1][:60]
      if (r.stdout + r.stderr).strip() else "")

# --mode api 不带项目 ID 应进入服务模式：用 --help 级别的探针避免真起服务占端口。
# 做法：读源码确认 serve 分支存在，再验"不会因为缺 project_id 报错退出"。
src = RUNPY.read_text(encoding="utf-8")
check("--mode api 不带项目 ID 时走 serve 分支", "def serve(" in src and "serve(" in src.split("if not args.project_id")[-1][:400])


# ============================================================ 六、入口一致性
section("六、参数与文档一致（7.6 的示例命令能不能照抄跑通）")

r2 = subprocess.run([py, "run.py", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
help_text = r2.stdout + r2.stderr
check("--project 是 --project-id 的别名（7.6 示例写的是 --project）",
      "--project-id" in help_text and "--project PROJECT_ID" in help_text)


# ============================================================ 七、可选：真容器
if "--runtime" in sys.argv:
    section("七、真容器验证（13.7 第 2/3/4/5/6 条，需要 Docker 守护进程 + 已构建镜像）")

    import shutil

    def sh(cmd: list[str], timeout: int = 300, **kw: object) -> tuple[int, str]:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout, **kw)  # type: ignore[arg-type]
        return p.returncode, (p.stdout + p.stderr)

    IMG = "drama-agent:1.0"

    rc, out = sh(["docker", "version", "--format", "{{.Server.Version}}"], 30)
    has_docker = rc == 0
    check("Docker 守护进程可用", has_docker, out.strip()[:40])

    if has_docker:
        rc, out = sh(["docker", "image", "inspect", IMG, "--format", "{{.Id}}"], 30)
        check(f"镜像 {IMG} 已构建", rc == 0, out.strip()[:20])

    if has_docker and rc == 0:
        # ---------------------------------------------------------- 13.7 第 6 条
        rc, out = sh(["docker", "run", "--rm", "--entrypoint", "sh", IMG, "-c",
                      "which chromium chromium-browser google-chrome playwright 2>/dev/null; echo '---';"
                      "pip list 2>/dev/null | grep -i -E 'playwright|selenium'; echo done"], 180)
        after = out.split("---")[-1].lower() if "---" in out else ""
        check("镜像内没有浏览器与 playwright/selenium",
              "done" in after and "playwright" not in after and "selenium" not in after,
              out.strip().splitlines()[0][:50] if out.strip() else "")

        # ---------------------------------------------------------- 13.7 第 5 条
        rc, out = sh(["docker", "run", "--rm", "--entrypoint", "sh", IMG, "-c",
                      "ls -a /app | grep -c '^\\.env$' || true"], 120)
        check("镜像内没有 .env（密钥不过镜像）", out.strip() in ("0", ""), out.strip()[:10])

        # ---------------------------------------------------------- 7.6 时区
        rc, out = sh(["docker", "run", "--rm", "--entrypoint", "sh", IMG, "-c", "date +%Z"], 120)
        check("容器内时区是 CST（差 8 小时会让 costs.json 时间戳错位）", "CST" in out, out.strip()[:12])

        # ---------------------------------------------------------- 13.7 第 3 条
        # 判定"不是方块"不能靠肉眼看，用一个可证伪的判据：
        #   **渲染两个字形完全不同的汉字（永 / 一），若字体缺字，两者都会画成同一个
        #     缺字方框，导出图像素完全相同**。像素不同 ⇒ 字体真的有这两个字形。
        # 这条同时验证了 `font_family()` 在容器里返回的是实际装了的族名——
        # 硬写 Microsoft YaHei 的地方在这里会暴露。
        tmp = ROOT / "data" / ".fonttest"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "a.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n永\n", encoding="utf-8")
        (tmp / "b.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n一\n", encoding="utf-8")

        rc, out = sh(["docker", "run", "--rm", "-v", f"{tmp}:/t", "--entrypoint", "sh", IMG, "-c",
                      'F=$(python -c "from pipeline.s5_compose import font_family; '
                      'print(font_family())"); echo "family=$F"; '
                      'for n in a b; do ffmpeg -loglevel error -f lavfi -i color=c=black:s=480x270:d=1 '
                      '-vf "subtitles=/t/$n.srt:force_style=\'FontName=$F\'" '
                      '-frames:v 1 -y /t/$n.png; done; md5sum /t/a.png /t/b.png'], 240)

        family = next((ln.split("=", 1)[1].strip() for ln in out.splitlines()
                       if ln.startswith("family=")), "")
        check("容器内推断出的字体族名不是 Windows 的 Microsoft YaHei",
              bool(family) and "YaHei" not in family, family or "(空)")

        hashes = re.findall(r"^([0-9a-f]{32})\s+/t/([ab])\.png", out, re.M)
        h = {name: hsh for hsh, name in hashes}
        check("「永」与「一」渲染结果不同 ⇒ 不是缺字方框",
              len(h) == 2 and h.get("a") != h.get("b"),
              f"a={h.get('a', '')[:8]} b={h.get('b', '')[:8]}")

        # ---------------------------------------------------------- 13.7 第 2/4 条
        # 真跑一次 mock 全链路（零 API 花费），产物必须落在**宿主机**挂载目录里
        host_assets = ROOT / "data" / ".ctest_assets"
        host_data = ROOT / "data" / ".ctest_data"
        for d in (host_assets, host_data):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

        rc, out = sh(["docker", "run", "--rm",
                      "-v", f"{host_assets}:/app/assets/projects",
                      "-v", f"{host_data}:/app/data",
                      IMG, "--mock", "--project-id", "cver01", "--shots", "2"], 900)
        check("容器内 mock 全链路跑通（退出码 0）", rc == 0,
              (out.strip().splitlines()[-1][:70] if out.strip() else ""))

        final = host_assets / "cver01" / "final.mp4"
        check("成片落在宿主机挂载目录（容器删掉也还在）",
              final.exists(), f"{final.stat().st_size // 1024} KB" if final.exists() else "不存在")

        srt = host_assets / "cver01" / "subtitles.srt"
        srt_text = srt.read_text(encoding="utf-8", errors="replace") if srt.exists() else ""
        check("字幕文件含中文且不是乱码", any("\u4e00" <= ch <= "\u9fff" for ch in srt_text),
              srt_text.strip().splitlines()[-1][:30] if srt_text.strip() else "空")

        state_db = host_data / "state.db"
        check("SQLite 状态落在宿主机 data/（断点续跑不丢）", state_db.exists(),
              f"{state_db.stat().st_size // 1024} KB" if state_db.exists() else "不存在")

        check("成片不是空壳（>20KB，含真实编码）",
              final.exists() and final.stat().st_size > 20 * 1024,
              f"{final.stat().st_size // 1024} KB" if final.exists() else "-")

print()
print("=" * 64)
print(f"结果：{passed} 通过 / {failed} 失败")
print("=" * 64)
sys.exit(1 if failed else 0)
