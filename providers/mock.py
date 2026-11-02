"""本地占位供应商：零成本，秒级返回。

存在的意义不是「假数据糊弄」，而是让链路 bug 和 API bug 分离开。
先证明管道通，再证明模型好用。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from providers.base import ImageProvider, TextProvider, VideoProvider, VisionJudge


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_solid_png(path: Path, width: int = 720, height: int = 1280, rgb: tuple[int, int, int] = (38, 42, 52)) -> Path:
    """用标准库手写一张纯色 PNG，避免为占位图引入 Pillow。"""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class MockImage(ImageProvider):
    name = "mock-image"
    is_mock = True
    unit_price = 0.0

    def generate(self, prompt, *, out, references=None, size="1024x1792", allow_paid=False) -> Path:
        # 有参考图时用不同的底色，肉眼就能看出「参考图有没有被用上」
        rgb = (60, 90, 120) if references else (38, 42, 52)
        return write_solid_png(out, rgb=rgb)


class MockVideo(VideoProvider):
    name = "mock-video"
    is_mock = True
    price_per_second = 0.0

    def generate(self, keyframe, prompt, *, duration, out, allow_paid=False) -> Path:
        """没有 ffmpeg 时留一个文本桩；有 ffmpeg 时生成真实可播的黑屏片段。

        查找走 `config.find_bin`，不直接 `shutil.which`：本机的 ffmpeg 不在 PATH 里，
        而是通过 `FFMPEG_DIR` 指定的（见 .env 的说明）。用 which 会让「装了但不在
        PATH」被误判成「没装」，白白退回文本桩。
        """
        import subprocess

        from config import find_bin

        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = find_bin("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg, "-y", "-loop", "1", "-i", str(keyframe),
                "-t", str(duration), "-r", "24",
                "-vf", f"scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        else:
            out.write_text(f"MOCK_VIDEO duration={duration} keyframe={keyframe.name}", encoding="utf-8")
        return out


class MockText(TextProvider):
    """返回一份固定的、通过 4.7 全部校验的剧本 JSON。

    原先这里是直接 `raise ProviderError("MockText 不产出内容")`。改掉它的原因很具体：
    M1 的验收标准是「`--mock` 跑通 S0~S6」，而 S1 必须产出一份结构化的 `script.json`；
    没有文本 provider 就永远走不到 S2 之后，mock 链路无法自证。

    固定的这份数据不是随手编的，它被约束成：
      - 6 镜、index 连续、duration 全为档位值 5.0
      - 2 角色 / 2 场景，引用自洽（`Script.check_references` 全过）
      - 第 6 镜是 `environment_only=True` 的空镜，专门覆盖「纯场景镜头」分支
      - 动作全部落在 9.3 的能力边界内（无大幅转身、无手部特写、无两人同框对视）
    真实 provider 不受影响：它只在这条零成本的验证链路上被调用。
    """

    name = "mock-text"
    is_mock = True

    def complete(self, system, user, *, json_mode=False) -> str:
        import json

        return json.dumps(MOCK_SCRIPT, ensure_ascii=False)


#: 占位剧本。字段名与 `models.Script` 一一对应，改 Schema 时这里必须同步，
#: 否则 `--mock` 会在 S1 的解析步骤直接失败（这是好事：能立刻发现契约变更）。
MOCK_SCRIPT: dict = {
    "title": "电梯里的第三个人",
    "logline": "深夜加班回家的女人，在电梯里发现楼层按钮多按了一层。",
    "style": {
        "style_prompt": "国风水墨，冷色调，电影感",
        "negative_prompt": "低质量, 模糊, 变形, 多余手指, 水印, 文字",
        "aspect_ratio": "9:16",
    },
    "characters": [
        {
            "char_id": "char_01",
            "name": "林晚",
            "age_range": "26-30岁",
            "gender": "女",
            "appearance": "黑色齐肩直发，鹅蛋脸，细长眼，白皙肤色，偏瘦",
            "wardrobe": "深灰色羊毛大衣，内搭白色高领毛衣",
            "personality": "克制，习惯独处",
            "source": "brief",
        },
        {
            "char_id": "char_02",
            "name": "陈默",
            "age_range": "30-35岁",
            "gender": "男",
            "appearance": "黑色极短寸头，长方脸，浓眉，小麦色皮肤，肩背宽厚",
            "wardrobe": "藏青色工装夹克，深色牛仔裤",
            "personality": "沉默，观察力强",
            "source": "brief",
        },
    ],
    "scenes": [
        {
            "scene_id": "scene_01",
            "name": "老旧电梯",
            "description": "狭窄电梯间，金属壁有划痕，楼层指示灯泛黄",
            "lighting": "顶部冷白灯，侧向硬光",
            "time_of_day": "深夜",
        },
        {
            "scene_id": "scene_02",
            "name": "楼道尽头",
            "description": "长走廊，墙面斑驳，尽头有一扇半开的铁门",
            "lighting": "走廊灯明暗不均，地面反光",
            "time_of_day": "深夜",
        },
    ],
    "shots": [
        {
            "shot_id": "shot_01", "index": 1, "scene_id": "scene_01", "duration": 5.0,
            "camera": "中景，缓慢推近", "action": "缓缓转头看向楼层按钮",
            "character_ids": ["char_01"], "dialogue": None, "emotion": "疲惫",
        },
        {
            "shot_id": "shot_02", "index": 2, "scene_id": "scene_01", "duration": 5.0,
            "camera": "近景，固定机位", "action": "视线落在最高的按钮上",
            "character_ids": ["char_01"], "dialogue": "我按的是十七楼。", "emotion": "警觉",
        },
        {
            "shot_id": "shot_03", "index": 3, "scene_id": "scene_01", "duration": 5.0,
            "camera": "中景，缓慢横移", "action": "肩膀轻微下沉，向电梯角落退半步",
            "character_ids": ["char_01"], "dialogue": None, "emotion": "不安",
        },
        {
            "shot_id": "shot_04", "index": 4, "scene_id": "scene_01", "duration": 5.0,
            "camera": "近景，缓慢推近", "action": "停住脚步，抬头看向电梯门方向",
            "character_ids": ["char_02"], "dialogue": "你也是这一层？", "emotion": "平静",
        },
        {
            "shot_id": "shot_05", "index": 5, "scene_id": "scene_01", "duration": 5.0,
            "camera": "特写，固定机位", "action": "睫毛轻颤，视线偏向一侧",
            "character_ids": ["char_01"], "dialogue": None, "emotion": "恐惧",
        },
        {
            "shot_id": "shot_06", "index": 6, "scene_id": "scene_02", "duration": 5.0,
            "camera": "远景，固定机位", "action": "走廊尽头铁门缓慢晃动",
            "character_ids": [], "environment_only": True, "dialogue": None, "emotion": "悬疑",
        },
    ],
}


class MockJudge(VisionJudge):
    name = "mock-vlm"
    is_mock = True

    #: 反推用的假卡片。**刻意与 mock 剧本的角色卡完全不同**（剧本里 char_01 是
    #: 黑色直发的人类女性、穿深灰大衣；这里是明黄色的卡通角色、没有服装）。
    #: 两边长得像的话，就分不清"覆盖生效了"和"根本没调用反推"——
    #: 而这条链路的核心断言恰恰是"文字真的来自图了"。
    MOCK_CARD = {
        "subject_type": "cartoon",
        "gender": "不确定",
        "age_range": "不确定",
        "appearance": "亮黄色圆头，短吻，黑色圆眼，明黄色体表带光泽，短腿两足站立",
        "wardrobe": "",
        "style_hint": "3D 卡通渲染",
        "distinctive": "整体明黄色",
        "confidence": {"gender": 0.3, "age_range": 0.2,
                       "appearance": 0.9, "wardrobe": 0.1},
    }

    def describe(self, image, *, allow_paid: bool = False):
        """占位实现不收钱，`allow_paid` 只为与 VLMJudge 签名一致。"""
        return {"card": dict(self.MOCK_CARD), "usage": {}, "meta": {"image": {}}}

    def compare(self, image, reference, elements, *, allow_paid: bool = False):
        # 占位实现不收钱，所以 allow_paid 不影响行为；参数存在只为与 VLMJudge 签名一致，
        # 这样 pipeline 里同一行调用在 mock 与真实之间可以原样复用。
        return {
            "score": 92,
            "checks": [{"element": e, "matched": True, "expected": "见角色卡", "actual": "一致"} for e in elements],
            "diff": "",
        }
