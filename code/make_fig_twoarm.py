#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_fig_twoarm.py — §6 的中心图：两个判官配置的校准曲线并排。

两条曲线来自同一批对、同一个提示词、同一个呈现顺序、同一个温度，
唯一的差别是推理预算。它们在中段交叉：think1 在 bin2–bin6 更宽松、
在恒等带（bin9）更严，于是同一个 0.85 保证落在相差 0.056 的两个阈值上。

数据源（两份 s8/calib.json，由 phase1_v13.py 的 s8 产出）：
    runs/main_v10/stages/s8/calib.json         -> 主臂（已发表口径）
    runs/main_v12_think/stages/s8/calib.json   -> 推理预算放大后的臂

用法：
    python3 make_fig_twoarm.py --a <calib_A.json> --b <calib_B.json> \
        --out figures/calibration_twoarm.pdf

视觉上刻意沿用 figures/calibration.pdf 的语言（同样的砖红、同样的保证
水平横线、同样的 τ* 竖线标注），但身份不靠颜色区分 —— 实线圆点 vs
虚线方块 + 图例，黑白打印同样可读。
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HOUSE_RED = "#c1432e"      # 从现有 calibration.pdf 里取的
ARM_B = "#1f4e79"          # 深蓝，灰度下比砖红深，和实线/虚线一起双重编码
GUIDE = "#9a9a9a"
GRID = "#d9d9d9"


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    pb = d["per_bin"]
    xs, ps, los, his = [], [], [], []
    for k in sorted(pb, key=int):
        b = pb[k]
        lo_r, hi_r = b["range"]
        xs.append((lo_r + hi_r) / 2.0)
        ps.append(b["p"])
        los.append(b["p"] - b["lo"])
        his.append(b["hi"] - b["p"])
    return d, xs, ps, los, his


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="主臂 calib.json")
    ap.add_argument("--b", required=True, help="对比臂 calib.json")
    ap.add_argument("--label-a", default="primary judge")
    ap.add_argument("--label-b", default="$+$ reasoning budget")
    ap.add_argument("--target", default="0.85",
                    help="在图上标哪个保证水平的 τ*")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.6,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    dA, xa, pa, loa, hia = load(a.a)
    dB, xb, pb_, lob, hib = load(a.b)

    fig, ax = plt.subplots(figsize=(3.33, 2.55))

    # 保证水平横线：和现有图一样，recessive
    for g in (0.80, 0.85, 0.90):
        ax.axhline(g, color=GUIDE, lw=0.55, alpha=0.75, zorder=1)
    ax.text(1.018, 0.85, "0.85", color=GUIDE, fontsize=6.0,
            va="center", ha="left", zorder=2)
    ax.text(1.018, 0.80, "0.80", color=GUIDE, fontsize=6.0,
            va="center", ha="left", zorder=2)
    ax.text(1.018, 0.90, "0.90", color=GUIDE, fontsize=6.0,
            va="center", ha="left", zorder=2)
    ax.text(1.018, 0.955, "target", color=GUIDE, fontsize=6.0,
            va="center", ha="left", zorder=2, style="italic")

    ax.errorbar(xa, pa, yerr=[loa, hia], color=HOUSE_RED, marker="o", ms=3.4,
                mew=0, lw=1.3, ls="-", capsize=1.6, elinewidth=0.7,
                label=a.label_a, zorder=5)
    ax.errorbar(xb, pb_, yerr=[lob, hib], color=ARM_B, marker="s", ms=3.0,
                mew=0, lw=1.3, ls="--", dashes=(3.4, 1.8), capsize=1.6,
                elinewidth=0.7, label=a.label_b, zorder=4)

    tA = dA["tau_multi"][a.target]["tau_star"]
    tB = dB["tau_multi"][a.target]["tau_star"]
    for t, col, ls_ in ((tA, HOUSE_RED, (0, (5, 2))), (tB, ARM_B, (0, (1.4, 1.4)))):
        ax.axvline(t, color=col, lw=0.85, ls=ls_, alpha=0.9, zorder=3)

    # τ* 标注：两个阈值分得够开（0.883 vs 0.940），分别贴在竖线旁
    ax.annotate(r"$\tau^\star=%.3f$" % tB, xy=(tB, 0.055),
                xytext=(tB - 0.012, 0.055), color=ARM_B, fontsize=6.6,
                rotation=90, ha="right", va="bottom", zorder=6)
    ax.annotate(r"$\tau^\star=%.3f$" % tA, xy=(tA, 0.055),
                xytext=(tA - 0.012, 0.055), color=HOUSE_RED, fontsize=6.6,
                rotation=90, ha="right", va="bottom", zorder=6)


    ax.set_xlabel("Query cosine similarity")
    ax.set_ylabel("Substitutability\n(conditioned on self-coverage)")
    ax.set_xlim(0.50, 1.005)
    ax.set_ylim(-0.03, 1.06)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, color=GRID, lw=0.45, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.95,
                    borderpad=0.35, handlelength=2.0, labelspacing=0.28,
                    bbox_to_anchor=(0.012, 1.0), bbox_transform=ax.transAxes)
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_edgecolor("#cccccc")

    fig.tight_layout(pad=0.25)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.015)
    png = a.out.rsplit(".", 1)[0] + ".png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.015)
    print("  -> %s" % a.out)
    print("  -> %s" % png)
    print("  臂A τ*@%s = %.4f   臂B τ*@%s = %.4f   差 %.4f"
          % (a.target, tA, a.target, tB, tA - tB))
    print("  恒等带 bin9: A %.4f  B %.4f" % (pa[-1], pb_[-1]))
    for lvl in ("0.80", "0.85", "0.90"):
        ta = dA["tau_multi"].get(lvl, {}).get("tau_star")
        tb = dB["tau_multi"].get(lvl, {}).get("tau_star")
        fa = dA["tau_multi"].get(lvl, {}).get("pass_frac")
        fb = dB["tau_multi"].get(lvl, {}).get("pass_frac")
        print("  保证 %s: τ* %s / %s   放行 %s / %s"
              % (lvl,
                 "%.4f" % ta if ta else ta, "%.4f" % tb if tb else tb,
                 "%.4f" % fa if fa else fa, "%.4f" % fb if fb else fb))


if __name__ == "__main__":
    main()
