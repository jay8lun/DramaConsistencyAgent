"""S6 · 交付：report.html + costs.json（第 6 章 S6 / 9.5 数据诚实性）。

**所有数字都从记录算出来，禁止手填**（9.5 第 15 条）：
- 分数、重试次数 ← `shots` 表 + `state_log`
- 成本 ← `costs` 表（CostRecord：估算价与实际价分开）
- 降级 / 失败 ← 状态机终态，不美化（9.5 第 14 条）

`AttemptLog` 在这一层的等价物是 `state_log`：每次迁移/技术重试都有一行
（stage 由 from_state 推断、reason 为失败原因、ts 为时间戳）。不把
`Shot.attempt_log` 逐镜写回 `script.json`，是因为那份文件是全片共享的，
并发跑镜时逐镜回写会互相覆盖（7.4 并发 4）。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from models import Script
from pipeline.cost import CostLedger
from pipeline.project import ProjectCtx, write_json
from pipeline.state import StateStore

#: 状态配色：降级标红、失败置灰（9.5 第 14 条：不得隐藏 degraded）
STATUS_STYLE = {
    "done": "#137333",
    "degraded": "#b3261e",
    "failed": "#5f6368",
    "videoing": "#8a6d00",
    # 待人工确认：橙色。它不是失败，也不是通过——不能和 done 同色，
    # 否则报告里看不出「这一镜还没人认可」。
    "awaiting_review": "#b45309",
}

STAGE_LABEL = {
    "script": "剧本结构化",
    "anchor": "定妆照出图",
    "anchor_check": "定妆照自检",
    "keyframe": "首帧出图",
    "judging": "一致性判定",
    "video": "图生视频",
}


@dataclass
class Metrics:
    total: int = 0
    counts: dict[str, int] = None  # type: ignore[assignment]
    scored: int = 0
    score_avg: float | None = None
    score_min: int | None = None
    retry_avg: float | None = None
    tech_total: int = 0
    cost_total: float = 0.0
    cost_estimated: float = 0.0
    failed_calls: int = 0
    degraded_shots: list[str] = None  # type: ignore[assignment]
    failed_shots: list[str] = None  # type: ignore[assignment]
    unjudged_shots: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.counts = self.counts or {}
        self.degraded_shots = self.degraded_shots or []
        self.failed_shots = self.failed_shots or []
        self.unjudged_shots = self.unjudged_shots or []


def compute_metrics(ctx: ProjectCtx, script: Script, store: StateStore,
                    ledger: CostLedger) -> Metrics:
    states = {s.shot_id: s for s in store.all(ctx.project_id)}
    m = Metrics(total=len(script.shots))
    m.counts = dict(Counter(s.status for s in states.values()))

    scored = [s.score for s in states.values() if s.score is not None]
    m.scored = len(scored)
    m.score_avg = round(sum(scored) / len(scored), 1) if scored else None
    m.score_min = min(scored) if scored else None
    m.tech_total = sum(s.tech_attempts for s in states.values())

    # 平均质量重试：分母只算「真的进入过判定」的镜头。把未判定的空镜算进分母
    # 会把平均数压低，看起来像是重试更少（9.5 第 15 条要的是真实值，不是好看的值）。
    judged = [s for s in states.values() if s.score is not None]
    m.retry_avg = round(sum(s.quality_attempts for s in judged) / len(judged), 2) if judged else None

    m.degraded_shots = [s.shot_id for s in states.values() if s.status == "degraded"]
    m.failed_shots = [s.shot_id for s in states.values() if s.status == "failed"]
    m.unjudged_shots = [s.shot_id for s in states.values() if s.score is None and s.status != "failed"]

    cost = ledger.summary()
    # 报告首页那个"总花费"必须是**实扣**（2026-10-29 修）：原本取 `spent`，
    # 而 `spent` = COALESCE(actual, estimated)，会把失败调用的估算价也算进去
    # （六镜那次显示 ¥9.39，真实扣费 ¥5.39）。`spent` 只用于预算闸，不进报告。
    m.cost_total = cost["actual_total"]
    m.cost_estimated = cost["estimated_total"]
    m.failed_calls = cost["failed_calls"]
    return m


def write_costs(ctx: ProjectCtx, ledger: CostLedger) -> Path:
    """costs.json：一次调用一条。**估计值与实际值分开**（9.5 第 12 条）。

    实际价拿不到时写 `null` 并标 `actual_unknown: true`，
    不拿估算值冒充实际值。
    """
    rows = []
    for r in ledger.entries():
        actual = r["actual_cost"]
        rows.append({
            "ts": r["ts"],
            "stage": r["stage"],
            "provider": r["provider"],
            "operation": r["operation"],
            "shot_id": r["shot_id"],
            "estimated_cost": r["estimated_cost"],
            "actual_cost": actual,
            "actual_unknown": actual is None,
            "success": bool(r["success"]),
            "error": r["error"],
        })
    payload = {
        "project_id": ctx.project_id,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "currency": "CNY",
        "calls": len(rows),
        "estimated_total": round(sum(r["estimated_cost"] for r in rows), 4),
        "actual_total": round(sum(r["actual_cost"] or 0.0 for r in rows), 4),
        "spent": ledger.spent(),
        "budget_cap": ledger.cap,
        "entries": rows,
    }
    return write_json(ctx.costs_file, payload)


def _cell(text: str, color: str | None = None, *, bold: bool = False) -> str:
    style = ""
    if color:
        style += f"color:{color};"
    if bold:
        style += "font-weight:600;"
    return f'<td style="{style}">{text}</td>' if style else f"<td>{text}</td>"


def build(
    ctx: ProjectCtx,
    script: Script,
    *,
    store: StateStore,
    ledger: CostLedger,
    compose: Any | None = None,
    anchors_status: dict[str, Any] | None = None,
) -> Path:
    """生成 report.html。返回值是文件路径。"""
    m = compute_metrics(ctx, script, store, ledger)
    states = {s.shot_id: s for s in store.all(ctx.project_id)}
    qa = {s.shot_id: store.load_qa(ctx.project_id, s.shot_id) for s in script.shots}
    logs = store.log_entries(ctx.project_id, limit=400)

    notes: list[str] = []
    if anchors_status:
        for cid, st in anchors_status.items():
            if st.get("status") != "ok":
                notes.append(f"角色 {cid} 的定妆照自检未通过（{st.get('reason', '')}），"
                             "其相关镜头的判定结论不可信")
    if m.unjudged_shots:
        notes.append(f"未做要素判定的镜头：{m.unjudged_shots}（空镜或缺参考图）"
                     "——这些镜头的画面质量没有被验证")
    if m.degraded_shots:
        notes.append(f"降级放行：{m.degraded_shots}（重试耗尽或分数过低，已进成片但质量不达标）")
    pending_review = [s.shot_id for s in states.values() if s.status == "awaiting_review"]
    if pending_review:
        # 待人工确认必须单列：它既不是通过也不是失败，混进任何一边都是掩盖。
        notes.append(f"待人工确认：{pending_review}（视频已出但末帧判定未达标，"
                     "尚未进成片，用 --review / --decide 处理）")
    if m.failed_shots:
        notes.append(f"失败未入片：{m.failed_shots}")
    if compose is not None:
        notes.extend(getattr(compose, "notes", []) or [])
        # 配音情况必须进报告：终端里那几行刷过去就没了，而「这片子哪几镜没声音」
        # 是使用者拿到成片后第一个会问的问题。
        spoken = getattr(compose, "spoken", 0) or 0
        silent = getattr(compose, "silent", 0) or 0
        if spoken or silent:
            notes.append(f"配音：{spoken} 句人声 / {silent} 镜补静音")
        if getattr(compose, "audio", None) is None:
            notes.append("成片无音轨：TTS 全程失败或未启用")
        truncated = getattr(compose, "truncated", None) or []
        if truncated:
            notes.append(f"台词被截断：{truncated}——应在 S1 缩短台词或拆镜，"
                         "这不是合成阶段能修的")
        tempo = getattr(compose, "tempo_adjusted", None) or []
        if tempo:
            detail = "，".join(f"{sid} {t:g}×" for sid, t in tempo)
            notes.append(f"台词超出镜长已加速：{detail}")
    if m.failed_calls:
        notes.append(f"有 {m.failed_calls} 次调用失败（已计入成本，失败也可能扣费）")

    # ---- 人工重出记录（2026-11-02 加）----
    # 不加这一节会有两个问题：① 同一镜重出过三次，报告里看起来和一次过一样；
    # ② 账上多出来的那几笔钱找不到出处——人只会看到"总价比上次高了"，
    # 却没有任何地方写着"因为你在 12:10 重出了 shot_04"。
    # 直接读文件而不是 import run：run 已经 import 了本模块，反向 import 会成环。
    reruns: list[dict] = []
    rf = Path(ctx.root) / "reruns.json"
    if rf.exists():
        try:
            raw_rr = json.loads(rf.read_text(encoding="utf-8"))
            reruns = raw_rr if isinstance(raw_rr, list) else []
        except (OSError, json.JSONDecodeError):
            reruns = []      # 记录坏了不能让整份报告打不开
    if reruns:
        ids = sorted({str(r.get("shot_id", "")) for r in reruns})
        notes.append(f"人工重出：{len(reruns)} 次（{', '.join(ids)}）"
                     "——重出会重新出图与出片并再次计费，成本差额来自这里")
    rerun_rows = "".join(
        f"<tr><td>{_cell(str(r.get('shot_id', '')))}</td>"
        f"<td>{_cell(str(r.get('reason') or '（未填理由）'))}</td>"
        f"<td>{_cell(str(r.get('ts', '')).replace('T', ' ')[:16])}</td></tr>"
        for r in reruns
    )
    rerun_section = (
        f"""
<h2>人工重出记录</h2>
<div class="sub">下列镜头被人工要求重出过。填写的理由同时作为修正指令拼进了首帧提示词，
不是只做备注；每次重出都会重新出图、出片并再次计费。</div>
<table><thead><tr><th>镜号</th><th>重出理由</th><th>时间</th></tr></thead>
<tbody>{rerun_rows}</tbody></table>
"""
        if reruns else ""
    )

    avg_txt = "—" if m.score_avg is None else f"{m.score_avg}"
    retry_txt = "—" if m.retry_avg is None else f"{m.retry_avg}"
    cost_state = "已达上限" if m.cost_total >= ledger.cap else "未超限"

    # ---- 角色一致性对比墙（13.3，2026-10-29 补）----
    # 这一节原来完全不存在：`report.html` 里 `<img>` 数是 0，13.3 要求的
    # 「同一角色跨所有镜头并排展示」从未实现。而它恰恰是这个项目**唯一能让人
    # 用眼睛验证 D3 的东西**——分数表格说得再好，也得亲眼确认"这是不是同一个人"。
    # 布局：每个角色一条横向胶片条，头一张是定妆照（基准），后面按镜序排首帧，
    # 每张下方标镜号与该镜得分。缺少首帧的镜头直接不画，不留空框。
    def _src(p: Any) -> str:
        path = Path(str(p))
        try:
            return path.resolve().relative_to(ctx.root.resolve()).as_posix()
        except (ValueError, OSError):
            return path.as_posix()   # 不在项目目录内（如外部 truth_image）就用绝对路径

    wall_blocks: list[str] = []
    # **定妆照路径在 characters.json，不在 script.json**（第 6 章 S1b：两者分开落盘，
    # 角色确认闸门要有独立落点）。报告如果只用 script.characters，读到的 anchor_path
    # 全是空，对比墙会一张图都渲染不出来——2026-10-29 就踩了这个，`<img>` 数是 0。
    # 正确做法是以 characters.json 为准补全，读不到才退回 script 里的角色。
    anchors_by_id: dict[str, str] = {}
    try:
        if ctx.characters_file.exists():
            raw = json.loads(ctx.characters_file.read_text(encoding="utf-8"))
            for item in (raw.get("characters") if isinstance(raw, dict) else raw) or []:
                cid = item.get("char_id")
                if cid and item.get("anchor_path"):
                    anchors_by_id[cid] = item["anchor_path"]
    except (json.JSONDecodeError, OSError):
        anchors_by_id = {}

    for ch in script.characters:
        anchor = anchors_by_id.get(ch.char_id) or ch.anchor_path
        if not anchor or not Path(anchor).exists():
            continue
        figs = [
            f'<figure><img src="{_src(anchor)}" alt="定妆照">'
            f'<figcaption>定妆照<small>基准</small></figcaption></figure>'
        ]
        for shot in sorted(script.shots, key=lambda s: s.index):
            if ch.char_id not in shot.character_ids:
                continue
            kf = ctx.keyframe_of(shot.shot_id)
            if not kf.exists():
                continue
            st = states.get(shot.shot_id)
            score = "未判定" if (st is None or st.score is None) else str(st.score)
            tone = ""
            if st is not None and st.score is not None:
                tone = ' class="low"' if st.score < 80 else ""
            figs.append(
                f'<figure><img src="{_src(kf)}" alt="{shot.shot_id}">'
                f'<figcaption>{shot.shot_id}<small{tone}>{score}</small></figcaption></figure>'
            )
        wall_blocks.append(
            f'<div class="wall"><div class="who">{ch.name}'
            f'<span>{ch.char_id} · 出场 {len(figs) - 1} 镜</span></div>'
            f'<div class="strip">{"".join(figs)}</div></div>'
        )
    wall_html = "".join(wall_blocks) or '<p class="sub">没有可展示的首帧图。</p>'

    rows = []
    for shot in sorted(script.shots, key=lambda s: s.index):
        st = states.get(shot.shot_id)
        if st is None:
            continue
        color = STATUS_STYLE.get(st.status)
        q = qa.get(shot.shot_id) or {}
        fails = [c["element"] for c in (q.get("checks") or []) if not c.get("matched")]
        rows.append(
            "<tr>"
            + _cell(shot.shot_id)
            + _cell(f'<span style="color:{color};font-weight:600">{st.status}</span>')
            + _cell("—" if st.score is None else str(st.score))
            + _cell("、".join(fails) if fails else ("未判定" if st.score is None else "—"))
            + _cell(str(st.tech_attempts))
            + _cell(str(st.quality_attempts))
            + _cell(f"¥{st.cost:.4f}")
            + _cell((q.get("diff") or "")[:80])
            + "</tr>"
        )

    log_rows = "".join(
        f"<tr>{_cell(r['shot_id'])}{_cell(r['from_status'] or '—')}"
        f"{_cell(r['to_status'])}{_cell((r['reason'] or '')[:90])}"
        f"{_cell(r['ts'])}</tr>"
        for r in reversed(logs[:60])
    )

    cost_rows = []
    by_stage: dict[str, list[dict]] = {}
    for e in ledger.entries():
        by_stage.setdefault(e["stage"], []).append(e)
    for stage, entries in by_stage.items():
        est = sum(x["estimated_cost"] for x in entries)
        act = sum(x["actual_cost"] or 0.0 for x in entries)
        cost_rows.append(
            "<tr>" + _cell(STAGE_LABEL.get(stage, stage)) + _cell(str(len(entries)))
            + _cell(f"¥{est:.4f}") + _cell(f"¥{act:.4f}") + "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{script.title} · 一致性报告</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
      margin:0;padding:24px;background:#f7f7f8;color:#1f1f1f;line-height:1.6}}
 .wrap{{max-width:1080px;margin:0 auto}}
 h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px;color:#333}}
 .sub{{color:#5f6368;font-size:13px;margin-bottom:18px}}
 .cards{{display:flex;flex-wrap:wrap;gap:10px}}
 .card{{background:#fff;border:1px solid #e3e3e6;border-radius:8px;padding:12px 16px;min-width:120px}}
 .card .k{{font-size:12px;color:#5f6368}} .card .v{{font-size:20px;font-weight:600}}
 table{{width:100%;border-collapse:collapse;background:#fff;font-size:13px;
        border:1px solid #e3e3e6;border-radius:8px;overflow:hidden}}
 th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #eee;vertical-align:top}}
 th{{background:#fafafa;font-weight:600;color:#444;font-size:12px}}
 tr:last-child td{{border-bottom:none}}
 .notes{{background:#fff;border:1px solid #e3e3e6;border-radius:8px;padding:14px 18px}}
 .notes li{{margin:4px 0}}
 .warn{{background:#fff4f3;border-color:#f3c9c4}}
 code{{background:#f2f2f4;padding:1px 5px;border-radius:4px;font-size:12px}}
 .wall{{background:#fff;border:1px solid #e3e3e6;border-radius:8px;padding:12px 14px;margin-bottom:12px}}
 .who{{font-size:14px;font-weight:600;margin-bottom:8px}}
 .who span{{font-weight:400;font-size:12px;color:#5f6368;margin-left:8px}}
 .strip{{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}}
 .strip figure{{margin:0;flex:0 0 auto;width:104px}}
 .strip img{{width:104px;height:185px;object-fit:cover;border-radius:6px;
             border:1px solid #e3e3e6;display:block;background:#eee}}
 .strip figcaption{{font-size:11px;color:#5f6368;text-align:center;margin-top:4px;line-height:1.4}}
 .strip figcaption small{{display:block;font-weight:600;color:#1a7f37}}
 .strip figcaption small.low{{color:#b3261e}}
</style></head><body><div class="wrap">
<h1>{script.title}</h1>
<div class="sub">{script.logline}<br>
项目 <code>{ctx.project_id}</code> · 生成于 {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')}
· 通道 {ctx.channel} · 供应商 {'mock（零成本占位）' if ctx.mock else 'real'}</div>

<div class="cards">
 <div class="card"><div class="k">镜头数</div><div class="v">{m.total}</div></div>
 <div class="card"><div class="k">干净通过</div><div class="v" style="color:{STATUS_STYLE['done']}">{m.counts.get('done', 0)}</div></div>
 <div class="card"><div class="k">降级放行</div><div class="v" style="color:{STATUS_STYLE['degraded']}">{m.counts.get('degraded', 0)}</div></div>
 <div class="card"><div class="k">失败未入片</div><div class="v" style="color:{STATUS_STYLE['failed']}">{m.counts.get('failed', 0)}</div></div>
 <div class="card"><div class="k">一致性均分</div><div class="v">{avg_txt}</div></div>
 <div class="card"><div class="k">平均质量重试</div><div class="v">{retry_txt}</div></div>
 <div class="card"><div class="k">单部成本</div><div class="v">¥{m.cost_total:.4f}</div></div>
</div>

<h2>角色一致性对比墙</h2>
<div class="sub">同一角色跨所有镜头并排。第一张是定妆照（判定基准），后面按镜序排首帧，
图下是该镜与定妆照比对得到的分数；<span style="color:#b3261e;font-weight:600">红色分数</span>
表示低于 80（接近通过线 75，需人工过目）。</div>
{wall_html}

<h2>逐镜结果</h2>
<table><thead><tr>
 <th>镜号</th><th>状态</th><th>分数</th><th>未通过要素</th>
 <th>技术重试</th><th>质量重试</th><th>成本</th><th>差异描述</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
{rerun_section}
<h2>成本明细（按阶段）</h2>
<table><thead><tr><th>阶段</th><th>调用次数</th><th>估算</th><th>实际</th></tr></thead>
<tbody>{''.join(cost_rows)}</tbody>
<tfoot><tr><th>合计</th><th>{len(ledger.entries())}</th>
<th>¥{m.cost_estimated:.4f}</th><th>¥{m.cost_total:.4f}</th></tr></tfoot></table>
<div class="sub">预算上限 ¥{ledger.cap:.2f}（{cost_state}）。实际价为空的调用在
<code>costs.json</code> 里标 <code>actual_unknown</code>，不以估算值冒充。</div>

<h2>需要知道的问题</h2>
<div class="notes{' warn' if m.degraded_shots or m.failed_shots else ''}">
{'<ul>' + ''.join(f'<li>{n}</li>' for n in notes) + '</ul>' if notes else '<p>无</p>'}
</div>

<h2>状态变更日志（最近 60 条）</h2>
<table><thead><tr><th>镜号</th><th>从</th><th>到</th><th>原因</th><th>时间</th></tr></thead>
<tbody>{log_rows}</tbody></table>
</div></body></html>
"""
    ctx.report_file.write_text(html, encoding="utf-8")
    return ctx.report_file
