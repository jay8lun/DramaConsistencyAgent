#!/usr/bin/env python
"""M0-S3：算两条分数分布 + FP/FN + 冗余项检查 + 对照预注册门槛，出第一版报告。

严格按计划书 15.1 第四、五节实现：
- 第四节：产出不是「一个准确率」，是**两条分数分布**；核心问题是「三条线画得对不对」
  （≥PASS / FLOOR~PASS / <FLOOR），所以必须看两条分布是否可分。
- 第五节：4 项门槛**预注册**，跑之前定死，跑完不许改。

**数据流向是单向的**：raw_results.json 是判定结果（不含真值），labels.json 是真值。
本脚本是唯一把两者放在一起的地方，所以要素级准确率 / FP / FN 只有在这里才算得出来。
判定脚本 `m0_vlm_eval.py` 全程不读真值，判定时看得见真值的话准确率就是假的。

用法：
    python tools/m0_calc.py
    python tools/m0_calc.py --out tests/fixtures/m0_eval/report.md
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "tests" / "fixtures" / "m0_eval"
LABELS = EVAL_DIR / "labels.json"
RAW_RESULTS = EVAL_DIR / "raw_results.json"
DEFAULT_REPORT = EVAL_DIR / "report.md"

# 7.1 的业务阈值线。**不写死**——换判定模型后这两条线要重标，
# 写死就会出现「报告按 85 算、生产按 75 判」的两套口径。
from config import JUDGE_FLOOR_LINE, JUDGE_PASS_LINE  # noqa: E402

PASS_LINE = int(JUDGE_PASS_LINE)
FAIL_LINE = int(JUDGE_FLOOR_LINE)

# 15.1 第五节预注册门槛（跑之前定死，不许改）
THRESHOLDS = {
    "异人识别准确率(1-FP)": (0.95, "FP 是质量事故：跑偏的脸被判成合格，直接进成片"),
    "同人识别准确率(1-FN)": (0.80, "FN 只是多花钱：重试一次再来"),
    "要素级matched准确率": (0.85, "决定 diff 能不能定位到具体要素"),
    "diff可转提示词率": (0.70, "转不了，L3 闭环就断在最后一步"),
}

ELEMENTS = ["发型", "脸型", "眉眼", "肤色", "体型"]

# 已登记的数据集缺陷（出处：tests/fixtures/m0_eval/PROMPTS.md 第十一节偏差表）。
# 这些对的「画面」与「真值」本身有已知偏差，判定结果无论对错都不能直接算成
# 模型的判定能力——所以报告里要**同时给两组数**：
#   全样本（含缺陷对）：反映真实流水线会遇到的情况
#   剔除缺陷对（干净样本）：反映判定器本身的能力
# 只报干净样本会掩盖生产风险（生产里真实镜头也会有遮挡）；
# 只报全样本会让一个已知的数据问题拖累整体数字。两组并排给，读的人才不会误判。
KNOWN_DEFECTS = {
    "S13": "13_b 双马尾只看到一条（另一条被头部挡住）——真实遮挡，非造型变化。"
           "真值仍记发型=true，但 VLM 会据「只见一条马尾」判不一致（PROMPTS.md 已预判）",
}


def percentile(values: list[float], pct: float) -> float:
    """线性插值分位数。样本量小（15），用最简单可解释的实现。

    不引入 numpy：venv 里有，但 requirements.txt 不含它（计划书要求 M0 阶段
    只用标准库 statistics），而且手写实现让报告里的数字可追溯。
    """
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    raw = json.loads(RAW_RESULTS.read_text(encoding="utf-8"))
    truth = {x["pair_id"]: x for x in labels}
    results = {x["pair_id"]: x for x in raw}
    return truth, results, labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_REPORT), help="报告输出路径")
    args = ap.parse_args()

    if not RAW_RESULTS.exists():
        print(f"FAIL 缺少 {RAW_RESULTS}，先跑 scripts/m0_vlm_eval.py")
        return 1

    truth, results, labels = load()
    if not results:
        print(f"FAIL {RAW_RESULTS.name} 里没有判定结果，先跑 scripts/m0_vlm_eval.py")
        return 1

    missing = [x["pair_id"] for x in labels if x["pair_id"] not in results]
    same_pairs = [x for x in labels if x["is_same_person"]]
    diff_pairs = [x for x in labels if not x["is_same_person"]]

    # ---------- 一、两条分数分布（15.1 第四节核心） ----------
    same_scores = [results[x["pair_id"]]["score"] for x in same_pairs if x["pair_id"] in results]
    diff_scores = [results[x["pair_id"]]["score"] for x in diff_pairs if x["pair_id"] in results]

    same_p5 = percentile(same_scores, 0.05) if same_scores else float("nan")
    same_min = min(same_scores) if same_scores else float("nan")
    diff_p95 = percentile(diff_scores, 0.95) if diff_scores else float("nan")
    diff_max = max(diff_scores) if diff_scores else float("nan")

    # 可分性：同人组 P5 是否严格高于异人组 P95
    separable = (same_p5 > diff_p95) if same_scores and diff_scores else False
    # 越线样本 = 落进「会被业务判错」或「会落进 60~85 模糊带」的样本。
    # 两个方向都要看：
    #   同人组 < 85  → 会被判成不同人（FN，多花钱）
    #   异人组 ≥ 60  → 进入模糊带或以上（≥85 是 FP 质量事故；60~85 会触发人工复核）
    overlap = []
    for x in labels:
        if x["pair_id"] not in results:
            continue
        s = results[x["pair_id"]]["score"]
        if x["is_same_person"] and s < PASS_LINE:
            overlap.append((x["pair_id"], s, f"同人组 {s} < {PASS_LINE} 通过线 → 会被判为不同人（FN）"))
        elif (not x["is_same_person"]) and s >= PASS_LINE:
            overlap.append((x["pair_id"], s, f"异人组 {s} ≥ {PASS_LINE} → 质量事故（FP）"))
        elif (not x["is_same_person"]) and s >= FAIL_LINE:
            overlap.append((x["pair_id"], s, f"异人组 {s} 落进 {FAIL_LINE}~{PASS_LINE} 模糊带 → 会触发人工复核"))

    # ---------- 二、FP / FN 分开统计（15.1 第四节第 4 条） ----------
    # 判定口径：用业务线（score>=PASS_LINE 判同一人），这才是生产里真正用的判据
    fp = [(x["pair_id"], results[x["pair_id"]]["score"])
          for x in diff_pairs
          if x["pair_id"] in results and results[x["pair_id"]]["score"] >= PASS_LINE]
    fn = [(x["pair_id"], results[x["pair_id"]]["score"])
          for x in same_pairs
          if x["pair_id"] in results and results[x["pair_id"]]["score"] < PASS_LINE]

    n_diff = len(diff_scores)
    n_same = len(same_scores)
    fp_rate = len(fp) / n_diff if n_diff else float("nan")
    fn_rate = len(fn) / n_same if n_same else float("nan")
    diff_acc = 1 - fp_rate if n_diff else float("nan")
    same_acc = 1 - fn_rate if n_same else float("nan")

    # ---------- 二之二、剔除已登记缺陷后的对照（干净样本） ----------
    # 缺陷对（如 S13 的双马尾遮挡）不该算成模型能力问题，但也不能从报告里藏掉——
    # 所以单独算一份「干净样本」数字，与全样本并排呈现。
    clean_fn = [(x["pair_id"], results[x["pair_id"]]["score"])
                for x in same_pairs
                if x["pair_id"] in results
                and x["pair_id"] not in KNOWN_DEFECTS
                and results[x["pair_id"]]["score"] < PASS_LINE]
    n_same_clean = n_same - sum(1 for p in KNOWN_DEFECTS if p in results and p in {x["pair_id"] for x in same_pairs})
    fn_rate_clean = len(clean_fn) / n_same_clean if n_same_clean else float("nan")
    same_acc_clean = 1 - fn_rate_clean if n_same_clean else float("nan")

    # ---------- 三、要素级 matched 准确率 ----------
    # 总体准确率（对全部已判定样本）与分组准确率分开算。
    # 分组要分开，因为「同人组全对」会掩盖「异人组把不同的人判成一致」——
    # 后者才是质量事故（FP）。合并算一个平均数会同时掩盖两个方向的问题。
    elem_total = 0
    elem_correct = 0
    elem_detail = {}
    for el in ELEMENTS:
        t = c = 0
        for x in labels:
            pid = x["pair_id"]
            if pid not in results:
                continue
            vlm_matched = next(
                (ch["element"] and ch["matched"] for ch in results[pid]["checks"]
                 if ch["element"] == el), None
            )
            if vlm_matched is None:
                continue  # 模型漏判，不计入分母（否则污染统计）
            truth_val = x["elements"].get(el)
            if truth_val is None:
                continue
            t += 1
            if bool(vlm_matched) == bool(truth_val):
                c += 1
        elem_detail[el] = (c, t)
        elem_total += t
        elem_correct += c
    elem_acc = elem_correct / elem_total if elem_total else float("nan")

    # ---------- 四、冗余项检查（15.1 第三节顺带验） ----------
    # 判据只看**异人组**：某要素在异人组里几乎恒判 True，说明它没有区分力，
    # checks 是装饰、diff 定位不到真问题，D3 的「要素级判定」卖点要重写。
    # 用同人组算没有意义——同人组本来就该全 True。
    # 注意：这里统计的是「VLM 判成一致的次数」，**不是判定正确率**。
    # 异人组里真值大多为 False，若 VLM 也恒判 False，正确率会很高但恰恰说明
    # 它有区分力；反过来恒判 True 才是冗余。用正确率当冗余判据会完全反掉。
    redundant = []
    elem_diff_detail = {}   # 异人组判定正确率（用于表格）
    elem_diff_true = {}     # 异人组 VLM 判 True 的比率（用于冗余判据）
    for el in ELEMENTS:
        t = c = n_true = 0
        for x in diff_pairs:
            pid = x["pair_id"]
            if pid not in results:
                continue
            vlm_matched = next(
                (ch["matched"] for ch in results[pid]["checks"] if ch["element"] == el), None
            )
            if vlm_matched is None:
                continue
            truth_val = x["elements"].get(el)
            if truth_val is None:
                continue
            t += 1
            if bool(vlm_matched) == bool(truth_val):
                c += 1
            if bool(vlm_matched):
                n_true += 1
        elem_diff_detail[el] = (c, t)
        elem_diff_true[el] = (n_true, t)
        if t == 0:
            continue
        rate = n_true / t
        if rate >= 0.95:
            redundant.append((el, rate, n_true, t))
    redundant_desc = "无" if not redundant else "、".join(
        f"{el}（异人组被判一致 {c}/{t} = {rate:.0%}）" for el, rate, c, t in redundant
    )

    # ---------- 五、门槛对照（预注册，跑完不许改） ----------
    # diff 可转提示词率需要人工标注，M0-a 阶段由人工抽 10 条填写
    # tests/fixtures/m0_eval/diff_review.json；文件不存在则该项记为「未评估」，不算不达标。
    diff_review_path = EVAL_DIR / "diff_review.json"
    diff_review = []
    if diff_review_path.exists():
        raw_review = json.loads(diff_review_path.read_text(encoding="utf-8"))
        # 兼容两种写法：模板给的 {"records": [...]} 与裸数组 [...]
        if isinstance(raw_review, dict):
            diff_review = raw_review.get("records", [])
        else:
            diff_review = raw_review
    convertible = [d for d in diff_review if d.get("convertible") is True]
    undecidable = [d for d in diff_review if d.get("convertible") is None]
    diff_rate = len(convertible) / len(diff_review) if diff_review and not undecidable else float("nan")

    metrics = {
        "异人识别准确率(1-FP)": diff_acc,
        "同人识别准确率(1-FN)": same_acc,
        "要素级matched准确率": elem_acc,
        "diff可转提示词率": diff_rate,
    }
    gate_lines = []
    gate_pass = True
    gate_unevaluated = 0
    for name, (threshold, why) in THRESHOLDS.items():
        v = metrics[name]
        if v != v:  # NaN
            gate_unevaluated += 1
            gate_lines.append(f"  - {name}：未评估（门槛 ≥{threshold:.0%}）")
            continue
        ok = v >= threshold
        gate_pass = gate_pass and ok
        gate_lines.append(
            f"  - {name}：{v:.1%}（门槛 ≥{threshold:.0%}）{'✓ 达标' if ok else '✗ 未达标'}"
        )

    # 结论口径：有未评估项时不能说「全部达标」——那会把「没测」说成「过了」，
    # 正是 15.1 第五节要防的「自我说服」。分三种状态：
    #   全部达标 / 存在未达标项 / 有 N 项未评估（结论待补）
    if gate_unevaluated and gate_pass:
        gate_verdict = f"已评估项全部达标，但有 {gate_unevaluated} 项未评估（结论待补）"
    elif gate_unevaluated:
        gate_verdict = f"存在未达标项，且有 {gate_unevaluated} 项未评估（结论待补）"
    else:
        gate_verdict = "全部达标" if gate_pass else "存在未达标项"

    # ---------- 六、终端输出 ----------
    print(f"=== M0-S3 第一版数字（M0-a，不做最终决策）===")
    print(f"判定模型：{next(iter(results.values()))['model']}  "
          f"judge版本：{next(iter(results.values()))['judge_version']}")
    if missing:
        print(f"⚠ 有 {len(missing)} 对未判定：{missing}（下面统计只覆盖已判定的）")
    print(f"\n[一] 两条分数分布")
    print(f"  同人组 n={n_same}：min={same_min:.0f}  P5={same_p5:.1f}  "
          f"max={max(same_scores) if same_scores else float('nan'):.0f}  "
          f"均值={statistics.mean(same_scores) if same_scores else float('nan'):.1f}")
    print(f"  异人组 n={n_diff}：min={min(diff_scores) if diff_scores else float('nan'):.0f}  "
          f"P95={diff_p95:.1f}  max={diff_max:.0f}  "
          f"均值={statistics.mean(diff_scores) if diff_scores else float('nan'):.1f}")
    print(f"  可分性：{'是' if separable else '否'}"
          f"（同人 P5={same_p5:.1f} vs 异人 P95={diff_p95:.1f}）")
    if overlap:
        print(f"  越线样本 {len(overlap)} 条：")
        for pid, s, why in overlap:
            print(f"    {pid} score={s} —— {why}")
    else:
        print(f"  越线样本：无（同人组全部 ≥{PASS_LINE}，异人组全部 <{PASS_LINE}）")

    print(f"\n[二] 误判方向（分开统计）")
    print(f"  FP（异人判同人，质量事故）：{len(fp)}/{n_diff} = {fp_rate:.1%}"
          f"  {[f'{p}({s})' for p, s in fp]}")
    print(f"  FN（同人判异人，只多花钱）：{len(fn)}/{n_same} = {fn_rate:.1%}"
          f"  {[f'{p}({s})' for p, s in fn]}")
    if KNOWN_DEFECTS and n_same_clean:
        print(f"    ↳ 剔除已登记缺陷 {list(KNOWN_DEFECTS)} 后："
              f"FN {len(clean_fn)}/{n_same_clean} = {fn_rate_clean:.1%}（同人识别 {same_acc_clean:.1%}）")

    print(f"\n[三] 要素级 matched 准确率：{elem_acc:.1%}（{elem_correct}/{elem_total}）")
    for el in ELEMENTS:
        c, t = elem_detail[el]
        if t:
            print(f"  {el}：{c}/{t} = {c / t:.1%}  （异人组 {elem_diff_detail[el][0]}/{elem_diff_detail[el][1]}）")

    print(f"\n[四] 冗余项检查（异人组里恒为 True 的要素）：{redundant_desc}")
    if n_diff == 0:
        print("  （异人组尚无已判定样本，本项无法评估）")

    print(f"\n[五] 预注册门槛（15.1 第五节，跑之前定死）")
    for line in gate_lines:
        print(line)
    if not diff_review:
        print("  提示：diff 可转提示词率需人工抽 10 条标注，填 "
              "tests/fixtures/m0_eval/diff_review.json 后重跑本脚本")

    print(f"\n[六] 结论：{gate_verdict}")
    if not separable and n_diff and n_same:
        print(f"  ⚠ 两条分布不可分 → {PASS_LINE}/{FAIL_LINE} 两条线是拍脑袋的，必须重新标定"
              "（或改为「要素命中数 ≥ 4/5 才通过」的计数判据）")
    if missing:
        print(f"  ⚠ 还有 {len(missing)} 对未判定，以上数字只是部分结果")

    # ---------- 七、写报告 ----------
    report = []
    report.append("# M0 判定能力验证 · 第一版报告（M0-a）\n")
    report.append("> **这是 M0-a 的第一版数字，不做最终决策。**")
    report.append("> 最终决策要用 M0-b（出图 API 定下来后用真实链路复跑）的数字。")
    report.append("> 出处：计划书 15.1 第六节。\n")
    meta = next(iter(results.values()))
    report.append(f"- 判定模型：`{meta['model']}`（深度思考："
                  f"{'开' if meta.get('thinking', True) else '关'}）")
    report.append(f"- 判定 prompt 版本：`{meta['judge_version']}`")
    report.append(f"- 数据集：`tests/fixtures/m0_eval/pairs/` 30 张 AI 出图（同人 15 + 异人 15）")
    report.append(f"- 已判定：{len(results)}/{len(labels)} 对\n")

    report.append("## 一、两条分数分布（15.1 第四节）\n")
    report.append("| 组 | n | min | P5 | 均值 | P95 | max |")
    report.append("|---|---|---|---|---|---|---|")
    report.append(f"| 同人组 | {n_same} | {same_min:.0f} | {same_p5:.1f} | "
                  f"{statistics.mean(same_scores) if same_scores else 0:.1f} | - | "
                  f"{max(same_scores) if same_scores else 0:.0f} |")
    report.append(f"| 异人组 | {n_diff} | {min(diff_scores) if diff_scores else 0:.0f} | - | "
                  f"{statistics.mean(diff_scores) if diff_scores else 0:.1f} | {diff_p95:.1f} | "
                  f"{diff_max:.0f} |")
    report.append("")
    report.append(f"**可分性：{'是' if separable else '否'}**"
                  f"（同人组 P5={same_p5:.1f}，异人组 P95={diff_p95:.1f}）")
    if not separable:
        report.append("")
        report.append("> 两条分布重叠 → 计划书 7.1 的 `≥85 / 60~85 / <60` 三条线"
                      "缺乏数据支撑，必须重新标定，或改用「要素命中数 ≥ 4/5」的计数判据。")
    if overlap:
        report.append("")
        report.append("越线样本：")
        for pid, s, why in overlap:
            report.append(f"- `{pid}` score={s} —— {why}")

    report.append("\n## 二、误判方向（FP / FN 分开统计）\n")
    report.append("| 方向 | 数量 | 比例 | 样本 | 性质 |")
    report.append("|---|---|---|---|---|")
    report.append(f"| FP（异人判同人） | {len(fp)}/{n_diff} | {fp_rate:.1%} | "
                  f"{', '.join(f'`{p}`({s})' for p, s in fp) or '无'} | 质量事故：跑偏的脸进成片 |")
    report.append(f"| FN（同人判异人） | {len(fn)}/{n_same} | {fn_rate:.1%} | "
                  f"{', '.join(f'`{p}`({s})' for p, s in fn) or '无'} | 只多花钱：重试一次 |")
    if KNOWN_DEFECTS:
        report.append("")
        report.append("**剔除已登记缺陷后的对照**（缺陷对不算模型能力，但也不藏掉）：\n")
        report.append("| 口径 | 同人组 FN | 同人识别准确率 |")
        report.append("|---|---|---|")
        report.append(f"| 全样本（含缺陷对） | {len(fn)}/{n_same} | {same_acc:.1%} |")
        report.append(f"| 剔除缺陷对 | {len(clean_fn)}/{n_same_clean} | "
                      f"{same_acc_clean:.1%} |")
        report.append("")
        for pid, why in KNOWN_DEFECTS.items():
            report.append(f"- `{pid}`：{why}")

    report.append("\n## 三、要素级 matched 准确率\n")
    report.append(f"总体：**{elem_acc:.1%}**（{elem_correct}/{elem_total}）\n")
    report.append("| 要素 | 全部样本 | 异人组 |")
    report.append("|---|---|---|")
    for el in ELEMENTS:
        c, t = elem_detail[el]
        dc, dt = elem_diff_detail[el]
        all_txt = f"{c}/{t} = {c / t:.1%}" if t else "0/0 = -"
        diff_txt = f"{dc}/{dt} = {dc / dt:.1%}" if dt else "0/0 = -"
        report.append(f"| {el} | {all_txt} | {diff_txt} |")

    report.append("\n## 四、冗余项检查\n")
    report.append(f"{redundant_desc}")
    report.append("")
    report.append("> 检查口径：某要素**在异人组里**几乎恒判 True，说明它没有区分力，"
                  "`checks` 是装饰、`diff` 定位不到真问题，D3 的「要素级判定」卖点要重写。")
    report.append("> 用同人组算没有意义——同人组本来就该全 True。"
                  "所以只看异人组一列。")

    report.append("\n## 五、预注册门槛对照（15.1 第五节）\n")
    report.append("| 指标 | 实测 | 门槛 | 结果 | 为什么是这个方向 |")
    report.append("|---|---|---|---|---|")
    for name, (threshold, why) in THRESHOLDS.items():
        v = metrics[name]
        if v != v:
            report.append(f"| {name} | 未评估 | ≥{threshold:.0%} | 待人工标注 | {why} |")
        else:
            ok = v >= threshold
            report.append(f"| {name} | {v:.1%} | ≥{threshold:.0%} | "
                          f"{'✓ 达标' if ok else '✗ 未达标'} | {why} |")

    report.append("\n## 六、结论与下一步\n")
    report.append(f"**门槛结论：{gate_verdict}**\n")
    if missing:
        report.append(f"- ⚠ **还有 {len(missing)} 对未判定**（{', '.join(missing)}），"
                      "上面所有统计只覆盖已判定样本，跑完才作数。")
    if not diff_review:
        report.append("- **diff 可转提示词率未评估**：需人工抽 10 条 diff 标注「能否直接"
                      "改写成生成提示词」，结果填入 `tests/fixtures/m0_eval/diff_review.json` 后重跑本脚本。")
    if not separable and n_diff and n_same:
        report.append("- **两条分布不可分**：按 15.1 第五节预写的处理顺序，"
                      "FP 超标时依次尝试：① 收紧判定 prompt（要求逐项给出差异证据）"
                      "→ ② 参考侧改三视图多图比对 → ③ 换 VLM → ④ 降级人审 + 机器预筛。")
    if redundant:
        report.append(f"- **存在冗余要素**：{redundant_desc}，需重做要素拆分或换参考图策略。")
    if missing:
        report.append(f"- ⚠ 有 {len(missing)} 对未判定：{', '.join(missing)}")

    # 对「diff 可转率」过高的自我质疑。
    # 这条不是客气话：判定 prompt 里强制了「什么不一致：参考图是…，待判画面是…」的句式，
    # 只要模型按格式作答，这段文字天然就能改写成提示词增量——所以这个指标近似恒真，
    # 70% 门槛几乎不可能不过。把它当成「格式合规性检查」而不是「闭环可用性证明」。
    if metrics["diff可转提示词率"] == metrics["diff可转提示词率"] and \
            metrics["diff可转提示词率"] >= 0.95:
        report.append(
            f"- ⚠ **diff 可转率 {metrics['diff可转提示词率']:.0%} 这个数字信息量很低**："
            "判定 prompt 强制了「什么不一致：参考图是…，待判画面是…」的句式，"
            "模型只要按格式作答，几乎必然可转。本项实际验的是**格式合规性**，"
            "不是**闭环可用性**。真正的验证在 M4——把 diff 转成 prompt_delta 后重新生成，"
            "看首帧是否真的被修正。**不要拿这一项当 D3 成立的证据。**")

    report.append("- **M0-a 数字仅供参考，不做最终决策**；等 A.6 出图 API 定下后，"
                  "用真实链路复跑 M0-b，用那个数字做最终决策。")

    out = Path(args.out)
    out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n报告已写：{out}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
