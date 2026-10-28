"""Fig. pipeline: per-stratum substitutable rate under four judge configurations and
agreement (Cohen's kappa) with the primary configuration.
Input: runs/phase3/e5/compare.json (by_slice). Usage: python make_fig_pipeline.py compare.json out.pdf
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))["by_slice"]
bins = [k for k in d if k.startswith("bin")]
bins.sort(key=lambda k: int(k.split()[0][3:]))
labels = [k.split()[1].strip("[)").split(",")[0][:4] for k in bins]
arms = [("main", "Primary", "#2a78d6", "o"), ("think1", "Reasoning", "#eb6834", "s"),
        ("xfam_llama33", "Cross-family", "#1baf7a", "^"), ("k3", "Shuffled maj.", "#eda100", "D")]
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5})
fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 1.55), constrained_layout=True)
x = range(len(bins))
for key, name, c, m in arms:
    a.plot(x, [d[k][key]["pos_rate"] for k in bins], color=c, marker=m, ms=3, lw=1.2, label=name)
    if key != "main":
        b.plot(x, [d[k][key]["vs_main"]["kappa"] for k in bins], color=c, marker=m, ms=3, lw=1.2)
for ax, t in ((a, "(a) Substitutable rate"), (b, "(b) $\\kappa$ vs. primary")):
    ax.set_xticks(list(x)[::2]); ax.set_xticklabels(labels[::2])
    ax.set_xlabel("Cosine stratum (lower edge)")
    ax.set_title(t, fontsize=7)
    ax.grid(axis="y", lw=0.3, color="#dddddd")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
a.set_ylim(0, 1.02)
b.axhline(0, color="#888888", lw=0.5)
b.set_ylim(-0.2, 0.9)
a.legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.5)
fig.savefig(sys.argv[2])
