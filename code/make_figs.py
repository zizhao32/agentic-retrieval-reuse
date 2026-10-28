#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
生成论文图 1 / 2 / 3（图 4 用 make_fig4.py）
================================================================================
    Fig 1  意图分布：频次 log-log + 头部覆盖累积 + 粒度敏感性
    Fig 2  规模律：五条工作负载曲线 + 第二个 agent + BCP
    Fig 3  校准曲线：逐层可替代率 + Wilson 区间 + 多档 τ*

尺寸按 PVLDB 双栏排版：单栏 3.33in，跨栏 7.0in。
用法：
    python make_figs.py --run runs/main_v10 --out matview-vldb2027/figures \
        [--agent-b runs/agent_b] [--bcp runs/phase2_B]
================================================================================
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL, DBL, DPI = 3.33, 7.0, 600
STYLE = {"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
         "xtick.major.width": 0.6, "ytick.major.width": 0.6,
         "xtick.major.size": 2.5, "ytick.major.size": 2.5,
         "legend.frameon": False, "axes.spines.top": False,
         "axes.spines.right": False, "axes.grid": True,
         "grid.alpha": 0.25, "grid.linewidth": 0.5}

SRC_STYLE = {
    "agent":            dict(label="W1 Agent (Tongyi)", c="#C1432E", ls="-",  lw=1.6, m="o"),
    "agent-qwen32b":    dict(label="W1b Agent (Qwen3-32B)", c="#E08A7B", ls="--", lw=1.3, m="s"),
    "replay:2wiki":     dict(label="W3 2WikiMultihop", c="#4A6FA5", ls="-",  lw=1.1, m="^"),
    "replay:musique":   dict(label="W2 MuSiQue", c="#8C8C8C", ls=":",  lw=1.1, m="v"),
    "replay:hotpotqa":  dict(label="W4 HotpotQA (control)", c="#333333", ls="-.", lw=1.3, m="x"),
    "agent-bcp":        dict(label="W5 BrowseComp-Plus", c="#5B8C5A", ls="--", lw=1.3, m="d"),
}


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def fig1(run: Path, out: Path) -> bool:
    R = rj(run / "stages/s10/reuse.json")
    G = rj(run / "stages/s9/granularity.json")
    if not R:
        print("  跳过 Fig 1：缺 s10/reuse.json"); return False
    comp = R.get("components") or {}
    freq = np.array(comp.get("freq") or [], float)
    cost = np.array(comp.get("cost") or [], float)
    if freq.size == 0:
        print("  跳过 Fig 1：reuse.json 里没有 freq"); return False
    fig, ax = plt.subplots(1, 3, figsize=(DBL, 1.95))

    f = np.sort(freq)[::-1]
    ax[0].loglog(np.arange(1, len(f) + 1), f, lw=1.2, color="#C1432E")
    ax[0].set_xlabel("Intent rank"); ax[0].set_ylabel("Retrieval calls")
    ax[0].set_title("(a) Intent frequency", fontsize=8, pad=4)

    c = np.sort(cost)[::-1] if cost.size else f
    cum = np.cumsum(c) / c.sum()
    x = np.arange(1, len(c) + 1) / len(c)
    ax[1].plot(x * 100, cum * 100, lw=1.4, color="#C1432E")
    k = max(int(0.05 * len(c)) - 1, 0)
    ax[1].axvline(5, ls="--", lw=0.8, c="#444")
    ax[1].plot([5], [cum[k] * 100], "o", ms=4, color="#C1432E")
    ax[1].annotate(f"top 5%: {cum[k]*100:.0f}%", (5, cum[k] * 100),
                   xytext=(22, cum[k] * 100 - 18), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[1].set_xlabel("Intents (%, by cost)"); ax[1].set_ylabel("Cumulative cost (%)")
    ax[1].set_title("(b) Head coverage", fontsize=8, pad=4)

    if G:
        t = sorted(G, key=float)
        ax[2].plot([float(x) for x in t], [G[x]["n_components"] for x in t],
                   "o-", ms=3, lw=1.2, color="#4A6FA5", label="components")
        ax[2].plot([float(x) for x in t], [G[x]["n_cliques"] for x in t],
                   "s--", ms=3, lw=1.0, color="#8C8C8C", label="cliques")
        ax[2].set_xlabel(r"Clustering threshold $\tau$"); ax[2].set_ylabel("Intents")
        ax[2].legend(fontsize=6.5)
    ax[2].set_title("(c) Granularity sensitivity", fontsize=8, pad=4)
    for a in ax:
        a.set_axisbelow(True)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    for e in ("pdf", "png"):
        fig.savefig(out / f"figure1.{e}", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  -> {out/'figure1.pdf'}"); return True


def fig2(run: Path, out: Path, extra) -> bool:
    S = rj(run / "stages/s13/scale.json")
    if not S:
        print("  跳过 Fig 2：缺 s13/scale.json"); return False
    curves = {}
    for s, rows in (S.get("by_source") or {}).items():
        curves[s] = rows
    for tag, p in extra.items():
        E = rj(Path(p) / "stages/s13/scale.json")
        if E:
            for s, rows in (E.get("by_source") or {}).items():
                curves.setdefault(s, rows)
        elif p:
            print(f"  （{tag} 缺 s13/scale.json，跳过该线）")
    fits = dict(S.get("fits") or {})
    for p in extra.values():
        E = rj(Path(p) / "stages/s13/scale.json")
        if E:
            for k, v in (E.get("fits") or {}).items():
                fits.setdefault(k, v)

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    order = [s for s in SRC_STYLE if s in curves]
    # 末端标注容易互相压住（曲线常在相近的 x/y 收尾），按序号错开
    dy = {s: (8 if i % 2 == 0 else -10) for i, s in enumerate(order)}
    for s in order:
        rows = curves[s]
        st = SRC_STYLE[s]
        x = [r["n_tasks"] for r in rows]
        y = [r["head_cov_cost"] * 100 for r in rows]
        e = [r.get("sd_cost", 0) * 100 for r in rows]
        ax.errorbar(x, y, yerr=e, marker=st["m"], ms=3, lw=st["lw"], ls=st["ls"],
                    color=st["c"], capsize=1.5, elinewidth=0.6, label=st["label"])
        f = fits.get(s)
        if f and abs(f.get("slope", 0)) > 1e-3:
            ax.annotate(f"{f['slope']:+.3f}/dec", (x[-1], y[-1]), xytext=(4, dy[s]),
                        textcoords="offset points", fontsize=6, color=st["c"],
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=st["c"],
                                        shrinkA=0, shrinkB=1))
    ax.set_xscale("log")
    ax.set_xlim(right=ax.get_xlim()[1] * 2.2)     # 给末端标注留出空间
    ax.set_xlabel("Tasks in workload (log scale)")
    ax.set_ylabel("Head coverage, cost-weighted (%)")
    ax.legend(fontsize=6, loc="upper left", ncol=1, handlelength=2.0)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    for e in ("pdf", "png"):
        fig.savefig(out / f"scaling.{e}", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  -> {out/'scaling.pdf'}（{len(order)} 条线）"); return True


def fig3(run: Path, out: Path) -> bool:
    C = rj(run / "stages/s8/calib.json")
    if not C or not C.get("per_bin"):
        print("  跳过 Fig 3：缺 s8/calib.json"); return False
    pb = C["per_bin"]
    ks = [b for b in sorted(pb, key=lambda b: pb[b].get("range", [0])[0])
          if pb[b].get("range") and pb[b].get("p") is not None]
    if not ks:
        print("  跳过 Fig 3：per_bin 缺 range/p"); return False
    xm = [(pb[b]["range"][0] + pb[b]["range"][1]) / 2 for b in ks]
    p = [pb[b]["p"] for b in ks]
    # 小样本层的加权 Wilson 可能给出 lo>p 或 hi<p，夹到非负（matplotlib 会拒负值）
    lo = [max(0.0, pb[b]["p"] - pb[b].get("lo", pb[b]["p"])) for b in ks]
    hi = [max(0.0, pb[b].get("hi", pb[b]["p"]) - pb[b]["p"]) for b in ks]
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ax.errorbar(xm, p, yerr=[lo, hi], fmt="o-", ms=3.5, lw=1.2, capsize=2,
                color="#C1432E", elinewidth=0.7)
    tm = C.get("tau_multi") or {}
    cols = {"0.80": "#8C8C8C", "0.85": "#333333", "0.90": "#4A6FA5"}
    for g, col in cols.items():
        t = (tm.get(g) or {}).get("tau_star")
        if t:
            deg = (tm.get(g) or {}).get("degenerate")
            ax.axvline(t, ls=":" if deg else "--", lw=0.9, color=col)
            ax.annotate((r"$\tau^\star_{%s}$" % g) + ("*" if deg else ""),
                        (t, 0.04), rotation=90, fontsize=6, color=col,
                        ha="right", va="bottom")
        if t:
            ax.axhline(float(g), lw=0.5, color=col, alpha=0.35)
    ax.set_xlabel("Query cosine similarity")
    ax.set_ylabel("Substitutability\n(conditioned on self-coverage)")
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    for e in ("pdf", "png"):
        fig.savefig(out / f"calibration.{e}", dpi=DPI, bbox_inches="tight")
    plt.close(fig); print(f"  -> {out/'calibration.pdf'}  (* 表示该档已退化为精确匹配)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main_v10")
    ap.add_argument("--out", default="matview-vldb2027/figures")
    ap.add_argument("--agent-b", default="runs/agent_b")
    ap.add_argument("--bcp", default="runs/phase2_B")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    run = Path(a.run)
    extra = {k: v for k, v in (("agent_b", a.agent_b), ("bcp", a.bcp))
             if v and Path(v).exists()}
    ok = sum([fig1(run, out), fig2(run, out, extra), fig3(run, out)])
    print(f"\n生成 {ok}/3。图 4 另跑：python make_fig4.py --run {run} --out {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
