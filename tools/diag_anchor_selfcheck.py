"""复现 S2 定妆照自检的判定明细（取证用，只读项目状态）。

背景：2026-10-28 真实跑 real2 时，char_01 的三视图自检连跑 3 轮仍未通过，
但**分数从未打印、也从未落盘**（`anchors/_status.json` 因流水线中断没写成功）。
于是「判定认为不是同一个人」这个说法一直停留在推断，没有证据。

本脚本只做一件事：把已有的三视图重新送进判定，把 score / 五项 matched /
expected / actual / diff 全部打出来并存档。**不生成任何图，不改任何项目状态。**

用法：
    python scripts/diag_anchor_selfcheck.py --project real2
    python scripts/diag_anchor_selfcheck.py --project real2 --judge qwen
    python scripts/diag_anchor_selfcheck.py --project real2 --judge both

判据说明：`--judge ark` 用的是产品配置里的豆包（与线上完全一致，但账号欠费会 403）；
`--judge qwen` 用百炼 qwen3-vl-plus（免费额度）。**两者的分数不可直接比**——
73/60 两条线是 M0 用豆包校准的，换模型等于换了一把尺子。qwen 的价值在于
"要素级结论"（哪一项被判不一致），而不是分数本身。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from config import ModelConfig  # noqa: E402
from pipeline import judge as J  # noqa: E402


def build_judge(which: str):
    """按名字造一个判定器。ark 用产品配置，qwen 用百炼的 OpenAI 兼容端点。"""
    from providers.vlm import VLMJudge

    if which == "ark":
        return VLMJudge(config.settings.judge_vlm, name="diag-ark")

    if which == "qwen":
        cfg = ModelConfig(
            api_key=config.settings.video_api.api_key,
            # 注意：video_api.base_url 是原生 DashScope 端点（不带 /compatible-mode/v1），
            # 判定要走 OpenAI 兼容协议，必须显式换成兼容端点。
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3-vl-plus",
        )
        return VLMJudge(cfg, name="diag-qwen")

    raise SystemExit(f"未知 judge：{which}")


def run_one(judge, char_id: str, anchors: Path, views: dict[str, Path]) -> dict:
    front = views["front"]
    out: dict = {"char_id": char_id, "judge": judge.name, "pairs": {}}

    for view in ("side", "back"):
        pair = f"{view}_vs_front"
        try:
            raw = judge.compare(views[view], front, list(J.ELEMENTS), allow_paid=True)
        except Exception as exc:  # noqa: BLE001 —— 取证脚本，失败要原样记录
            out["pairs"][pair] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"    {pair:<16} 调用失败：{type(exc).__name__}: {str(exc)[:90]}")
            continue

        checks = raw.get("checks") or []
        fails = [c.get("element") for c in checks if not c.get("matched")]
        out["pairs"][pair] = {
            "score": raw.get("score"),
            "fails": fails,
            "checks": checks,
            "diff": raw.get("diff") or "",
            "usage": raw.get("usage") or {},
        }
        failtxt = "、".join(fails) if fails else "无"
        print(f"    {pair:<16} score={raw.get('score')}  不一致要素：{failtxt}")
        for c in checks:
            mark = "PASS" if c.get("matched") else "FAIL"
            print(f"        {c.get('element'):<4} {mark}  expected={str(c.get('expected'))[:34]}")
            print(f"                  actual={str(c.get('actual'))[:40]}")
        if raw.get("diff"):
            print(f"        diff = {str(raw.get('diff'))[:110]}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="复现 S2 三视图自检的判定明细")
    ap.add_argument("--project", default="real2", help="项目 id（assets/projects/<id>）")
    ap.add_argument("--judge", default="qwen", choices=("ark", "qwen", "both"))
    ap.add_argument("--chars", default="", help="只跑指定角色，逗号分隔；默认全部")
    args = ap.parse_args()

    anchors = config.ASSETS / "projects" / args.project / "anchors"
    if not anchors.is_dir():
        print(f"✗ 找不到 {anchors}", file=sys.stderr)
        return 2

    wanted = {c.strip() for c in args.chars.split(",") if c.strip()}
    char_ids = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1]
                       for p in anchors.glob("*_front.png")})
    if wanted:
        char_ids = [c for c in char_ids if c in wanted]

    if not char_ids:
        print("✗ 没有找到三视图（需要 *_front.png）", file=sys.stderr)
        return 2

    judges = ["ark", "qwen"] if args.judge == "both" else [args.judge]
    report: list[dict] = []

    print("=" * 74)
    print(f"S2 自检判定明细复现 · 项目 {args.project}")
    print(f"要素全表：{list(J.ELEMENTS)}   通过线 {J.PASS_LINE:g}  降级线 {J.FLOOR_LINE:g}")
    print("=" * 74)

    for which in judges:
        judge = build_judge(which)
        if which == "qwen":
            print(f"\n[{judge.name}] 模型 {judge._config.model}（分数与豆包不可直接比）")
        else:
            print(f"\n[{judge.name}] 模型 {judge._config.model}（与线上一致）")

        for cid in char_ids:
            views = {v: anchors / f"{cid}_{v}.png" for v in ("front", "side", "back")}
            if not all(p.exists() for p in views.values()):
                print(f"  {cid}: 三视图不全，跳过")
                continue
            print(f"  {cid}:")
            report.append(run_one(judge, cid, anchors, views))

    out = config.ASSETS / "probe" / f"anchor_selfcheck_{args.project}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n明细已存档 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
