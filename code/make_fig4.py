#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Figure 4 —— 精确缓存 vs 近似视图（分域净增量为主图）
================================================================================

为什么把分域增量提到主图：
    `agent 29.2%  vs  2wiki 3.2%  vs  musique 2.9%  vs  hotpotqa 0.1%`
    这是全文最强的单一发现。它同时回答两个质疑：
      (a)「这不就是结果缓存吗」—— 净增量为正且可观
      (b)「MuSiQue 有构造偏差」—— 支撑视图的收益恰恰不在 MuSiQue 上
    审稿人扫图的时间比读表多，这个对比必须看得见。

两个子图：
    (a) 主图：分域净增量（视图 @0.85 命中但归一化缓存未命中的成本占比）
              按增量降序排列，agent 高亮
    (b) 副图：全域堆叠条 —— 缓存基线 + 净增量，四个保证水平
              展示「保证越严，增量越小，0.90 时机制退化」

用法
----
    python make_fig4.py --run runs/main_v10
    python make_fig4.py --run runs/main_v10 --out paper/figures --style vldb

产出
----
    <out>/cache_vs_view.pdf   论文用（矢量）
    <out>/cache_vs_view.png   预览用
    <out>/fig4_data.csv       图中每个数值，便于核对
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ------------------------------------------------------------------ 样式 ----
# VLDB 双栏：单栏宽 3.33 in，双栏宽 7.0 in。Fig 4 建议跨双栏。
STYLE = {
    "vldb": dict(figsize=(7.0, 2.6), fontsize=8, dpi=600),
    "slide": dict(figsize=(10.0, 4.0), fontsize=13, dpi=150),
}

# 工作负载显示名与配色。agent 用强调色，其余灰阶，控制组用斜纹。
DOMAIN_META = {
    "agent":           dict(label="W1\nAgent", color="#C1432E", hatch=None, emph=True),
    "replay:musique":  dict(label="W2\nMuSiQue", color="#8C8C8C", hatch=None, emph=False),
    "replay:2wiki":    dict(label="W3\n2Wiki", color="#8C8C8C", hatch=None, emph=False),
    "replay:hotpotqa": dict(label="W4\nHotpot*", color="#CFCFCF", hatch="//", emph=False),
}
ORDER_FALLBACK = ["agent", "replay:2wiki", "replay:musique", "replay:hotpotqa"]


def load(run: Path) -> dict:
    p = run / "stages" / "s23" / "cache_vs_view.json"
    if not p.exists():
        sys.exit(f"找不到 {p}\n先跑 s23：--stages s23")
    d = json.loads(p.read_text(encoding="utf-8"))
    if "baselines" not in d:
        sys.exit(f"{p} 里没有 baselines，s23 可能失败了（检查 error 字段）")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="Figure 4: cache vs approximate views")
    ap.add_argument("--run", default="runs/main_v10")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--style", default="vldb", choices=list(STYLE))
    ap.add_argument("--usetex", action="store_true",
                    help="用系统 LaTeX 渲染文字（需装 texlive），论文终版建议开")
    ap.add_argument("--target", default="0.85", help="主图用哪个保证水平")
    a = ap.parse_args()

    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = load(run)
    st = STYLE[a.style]

    # 默认不启用 usetex（多数环境没装 LaTeX）。启用时百分号要转义，
    # 所以百分号统一用 PCT 变量控制。
    PCT = r"\%" if a.usetex else "%"
    if a.usetex:
        plt.rcParams["text.usetex"] = True
    plt.rcParams.update({
        "font.size": st["fontsize"],
        "font.family": "serif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ---------------- 取数 ----------------
    base_norm = d["baselines"]["normalized_cache"]["hit_cost"]
    base_exact = d["baselines"]["exact_cache"]["hit_cost"]
    inc_main = (d.get("increment") or {}).get(a.target)
    if not inc_main:
        sys.exit(f"保证水平 {a.target} 在 increment 里不存在，"
                 f"可选：{sorted((d.get('increment') or {}))}")

    # 分域增量（按成本），降序
    bysrc = inc_main.get("by_source") or {}
    srcs = [s for s in ORDER_FALLBACK if s in bysrc] + \
           [s for s in bysrc if s not in ORDER_FALLBACK]
    srcs.sort(key=lambda s: -bysrc[s]["cost_frac"])
    vals = [bysrc[s]["cost_frac"] for s in srcs]
    metas = [DOMAIN_META.get(s, dict(label=s, color="#8C8C8C", hatch=None, emph=False))
             for s in srcs]

    # 全域各档：缓存基线 + 净增量
    tgts = sorted((d.get("views") or {}))
    view_cost, incr_cost, feasible = [], [], []
    for t in tgts:
        v = (d["views"] or {}).get(t)
        i = (d.get("increment") or {}).get(t)
        if v and i:
            view_cost.append(v["hit_cost"]); incr_cost.append(i["cost_frac_of_eval"])
            feasible.append(True)
        else:
            view_cost.append(0.0); incr_cost.append(0.0); feasible.append(False)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=st["figsize"], gridspec_kw={"width_ratios": [1.15, 1.0]})

    # =============== (a) 主图：分域净增量 ===============
    x = np.arange(len(srcs))
    bars = ax1.bar(x, [v * 100 for v in vals], width=0.62,
                   color=[m["color"] for m in metas],
                   hatch=[m["hatch"] for m in metas],
                   edgecolor="white", linewidth=0.8)
    # agent 描边强调
    for b, m in zip(bars, metas):
        if m["emph"]:
            b.set_edgecolor("#6E1E12"); b.set_linewidth(1.1)

    for xi, v in zip(x, vals):
        ax1.text(xi, v * 100 + max(vals) * 100 * 0.035, f"{v*100:.1f}",
                 ha="center", va="bottom", fontsize=st["fontsize"] - 0.5,
                 fontweight="bold" if v == max(vals) else "normal")

    ax1.set_xticks(x)
    ax1.set_xticklabels([m["label"] for m in metas], fontsize=st["fontsize"] - 1)
    ax1.set_ylabel(f"Net gain over cache\n({PCT} of retrieval cost)")
    ax1.set_ylim(0, max(vals) * 100 * 1.22)
    ax1.set_title(f"(a) Where approximate matching pays off "
                  f"(guarantee {a.target})", fontsize=st["fontsize"], pad=6)
    ax1.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.text(0.0, -0.30, "*negative control: no sub-task structure",
             transform=ax1.transAxes, fontsize=st["fontsize"] - 2,
             color="#666666", style="italic")

    # 用一条注记把结论写在图里
    if len(vals) >= 2 and vals[1] > 0:
        mult = r"$\times$" if a.usetex else "\u00d7"
        ax1.annotate(f"{vals[0]/vals[1]:.0f}{mult} the next workload",
                     xy=(0.32, vals[0] * 100 * 0.55),
                     xytext=(1.05, vals[0] * 100 * 0.62),
                     fontsize=st["fontsize"] - 1, color="#6E1E12", va="center",
                     arrowprops=dict(arrowstyle="->", color="#6E1E12", lw=0.7,
                                     shrinkA=0, shrinkB=2))

    # =============== (b) 副图：全域堆叠 ===============
    xb = np.arange(len(tgts))
    cache_part = [min(base_norm, vc) for vc in view_cost]
    ax2.bar(xb, [c * 100 for c in cache_part], width=0.58,
            color="#B8C6D9", edgecolor="white", linewidth=0.8,
            label="Reachable by exact cache")
    ax2.bar(xb, [i * 100 for i in incr_cost], width=0.58,
            bottom=[c * 100 for c in cache_part],
            color="#C1432E", edgecolor="white", linewidth=0.8,
            label="Net gain from approximate matching")
    ax2.axhline(base_norm * 100, ls="--", lw=0.8, color="#444444", zorder=0)
    ax2.text(len(tgts) - 0.45, base_norm * 100 + 0.5,
             f"cache {base_norm*100:.1f}", fontsize=st["fontsize"] - 1.5,
             ha="right", color="#444444")

    for xi, (vc, ic, ok) in enumerate(zip(view_cost, incr_cost, feasible)):
        if not ok:
            ax2.text(xi, 1.0, "no feasible\nthreshold", ha="center", va="bottom",
                     fontsize=st["fontsize"] - 2, color="#888888", style="italic")
        else:
            ax2.text(xi, vc * 100 + 0.6, f"{vc*100:.1f}", ha="center", va="bottom",
                     fontsize=st["fontsize"] - 0.5)

    ax2.set_xticks(xb)
    ax2.set_xticklabels([(f"$\\geq${t}" if a.usetex else f"\u2265 {t}") for t in tgts])
    ax2.set_xlabel("Recall guarantee")
    ax2.set_ylabel(f"Retrieval cost served ({PCT})")
    ax2.set_ylim(0, max(view_cost) * 100 * 1.28)
    ax2.set_title("(b) The guarantee--coverage trade-off",
                  fontsize=st["fontsize"], pad=6)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.legend(loc="upper right", fontsize=st["fontsize"] - 1.5,
               handlelength=1.2, borderpad=0.3)

    fig.tight_layout(pad=0.6, w_pad=2.0)
    fig.subplots_adjust(bottom=0.26)
    for ext in ("pdf", "png"):
        p = out / f"cache_vs_view.{ext}"
        fig.savefig(p, dpi=st["dpi"], bbox_inches="tight")
        print(f"  -> {p}")
    plt.close(fig)

    # ---------------- 导出图中数值，便于核对 ----------------
    csvp = out / "fig4_data.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "key", "value", "note"])
        w.writerow(["a", "target", a.target, "guarantee level for panel (a)"])
        for s, v in zip(srcs, vals):
            w.writerow(["a", s, round(v, 4), "net increment, cost-weighted"])
        w.writerow(["b", "exact_cache", round(base_exact, 4), ""])
        w.writerow(["b", "normalized_cache", round(base_norm, 4), ""])
        for t, vc, ic, ok in zip(tgts, view_cost, incr_cost, feasible):
            w.writerow(["b", f"view@{t}", round(vc, 4),
                        "" if ok else "no feasible threshold"])
            w.writerow(["b", f"increment@{t}", round(ic, 4), ""])
    print(f"  -> {csvp}")

    print()
    print(f"  主图 (a) 保证 {a.target} 的分域净增量：")
    for s, v in zip(srcs, vals):
        print(f"     {s:<20} {v:>7.1%}")
    print(f"  副图 (b) 归一化缓存基线 {base_norm:.1%}；"
          f"视图 @{a.target} {dict(zip(tgts, view_cost))[a.target]:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
