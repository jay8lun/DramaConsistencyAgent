"""项目上下文（S0 的产物落点）。

第 6 章 S0：输入创意/角色来源/风格/预算，产出 `project.json`。
本期只落「项目壳」——目录、`project.json`、状态库登记；剧本与角色卡属于 S1。

为什么单独一个模块而不是塞进 `run.py`：`pipeline/s1~s5` 都要拿项目路径，
若定义在 CLI 里，stage 模块就得反向 import CLI，形成环。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from config import project_dir
from pipeline.state import now_iso

#: 只允许字母数字下划线短横。用正则而不是 `Path` 拼接来防目录穿越：
#: `--project-id ../..` 这种输入如果不拦，S0 会往项目外写文件。
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")

PROJECT_FILE = "project.json"
SCRIPT_FILE = "script.json"
CHARACTERS_FILE = "characters.json"

#: S0 就要建齐的目录。与第 5 章目标态一致，避免各 stage 自己 mkdir 出现路径分歧。
SUBDIRS: tuple[str, ...] = ("anchors", "keyframes", "shots", "inbox")


@dataclass(frozen=True)
class ProjectCtx:
    project_id: str
    root: Path
    budget_cap: float
    channel: str
    mock: bool
    created_at: str

    # ---- 产物路径 ----
    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def script_file(self) -> Path:
        return self.root / SCRIPT_FILE

    @property
    def characters_file(self) -> Path:
        return self.root / CHARACTERS_FILE

    @property
    def anchors_dir(self) -> Path:
        return self.root / "anchors"

    @property
    def keyframes_dir(self) -> Path:
        return self.root / "keyframes"

    @property
    def shots_dir(self) -> Path:
        return self.root / "shots"

    @property
    def inbox_dir(self) -> Path:
        return self.root / "inbox"

    @property
    def final_video(self) -> Path:
        return self.root / "final.mp4"

    @property
    def report_file(self) -> Path:
        return self.root / "report.html"

    @property
    def costs_file(self) -> Path:
        return self.root / "costs.json"

    def keyframe_of(self, shot_id: str) -> Path:
        return self.keyframes_dir / f"{shot_id}.png"

    def video_of(self, shot_id: str) -> Path:
        return self.shots_dir / f"{shot_id}.mp4"

    def anchor_of(self, char_id: str, view: str) -> Path:
        return self.anchors_dir / f"{char_id}_{view}.png"

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "created_at": self.created_at,
            "budget_cap": self.budget_cap,
            "channel": self.channel,
            "mock": self.mock,
        }


def write_json(path: Path, data: object) -> Path:
    """统一落盘入口：缩进 2 空格、保留中文。所有 `.json` 产物都走这里，
    避免各模块自己 `json.dump` 导致编码/缩进不一致（中文变 \\uXXXX 会让报告不可读）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_project_id(project_id: str) -> str:
    if not PROJECT_ID_RE.match(project_id or ""):
        raise ValueError(
            f"非法 project_id：{project_id!r}。只允许字母/数字/下划线/短横，"
            "以字母或数字开头，最长 32 字符（防止目录穿越）"
        )
    return project_id


def create_or_load(
    project_id: str,
    *,
    budget_cap: float,
    channel: str = "api",
    mock: bool = False,
    force: bool = False,
) -> tuple[ProjectCtx, bool]:
    """S0：建项目或复用已有项目。

    返回 `(ctx, created)`。`created=False` 表示复用——
    **复用时不覆盖 `created_at`**，否则「项目什么时候开始的」这个事实会被每次运行改掉。
    """
    validate_project_id(project_id)
    root = project_dir(project_id)
    pf = root / PROJECT_FILE

    if pf.exists() and not force:
        data = json.loads(pf.read_text(encoding="utf-8"))
        ctx = ProjectCtx(
            project_id=data["project_id"],
            root=root,
            budget_cap=float(data.get("budget_cap", budget_cap)),
            channel=data.get("channel", channel),
            mock=bool(data.get("mock", mock)),
            created_at=data.get("created_at", ""),
        )
        _ensure_subdirs(root)
        return ctx, False

    created_at = now_iso()
    ctx = ProjectCtx(
        project_id=project_id,
        root=root,
        budget_cap=float(budget_cap),
        channel=channel,
        mock=mock,
        created_at=created_at,
    )
    _ensure_subdirs(root)
    pf.write_text(json.dumps(ctx.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return ctx, True


def _ensure_subdirs(root: Path) -> None:
    for name in SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
