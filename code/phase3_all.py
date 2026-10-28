#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Phase 3 —— 失效机制、视图时效、可复用诊断工具、BCP 续跑
================================================================================

    H  判官失效机制探查（离线）
       已知：判官在高相似度层重测翻转 46.6%，人在同一批条目上翻转 0/12。
       未知：为什么翻转。H 用 l1.jsonl 里已有的字段找可预测翻转的信号，
       并量化「对这些样本弃判」能把翻转率压到多低、代价是损失多少覆盖。
       注意：s6 打乱文档用的是无种子的全局 random.shuffle，**顺序无法离线复现**，
       所以「位置偏好」这一类假设在此做不了，本脚本不做。

    I  视图时效性（离线）
       C3（视图维护）整块没做。这里用最便宜的切入点：在 N 个任务时建的视图，
       服务后续第 1、2、3… 批任务时命中率怎么变？以及「冻结视图」与
       「持续重建」差多少。这是系统构建者真正关心、而论文目前一个字没有的数。

    J  可复用的判官诊断工具（离线）
       把「分层重测 + 对边际稳健的一致性系数 + 弃判曲线」做成可直接用在
       别人数据上的工具，并用我们自己的数据自检。VLDB 的 workload
       characterization track 看重 reusable artifacts。

    B  BrowseComp-Plus 续跑（B1/B2 已完成，跑 B3/B4）

用法
----
    R=/mnt/data/user_workspace/zhangzizhao/matview
    PY=$R/envs/main/bin/python

    $PY phase3_all.py --config $R/config.json --exp H,I,J      # 离线，约 10 分钟
    $PY phase3_all.py --config $R/config.json --exp B --pilot 50
    setsid nohup $PY phase3_all.py --config $R/config.json --exp B > $R/logs/p3B.out 2>&1 &

    # 诊断工具用在任意数据上（generic 模式）
    $PY phase3_all.py --diagnose-input mydata.jsonl --diagnose-out out/
      输入每行需含：item_id, stratum, run1, run2[, human]

断点续跑、日志与 phase2 共用一套（runs/phase2）。
================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("p2x", str(_HERE / "phase2_experiments.py"))
p2x = _ilu.module_from_spec(_spec)
sys.modules["p2x"] = p2x
_spec.loader.exec_module(p2x)

Phase2, RunData = p2x.Phase2, p2x.RunData
rjsonl, wjson, rjson, append_jsonl = p2x.rjsonl, p2x.wjson, p2x.rjson, p2x.append_jsonl
wilson, fmtf, fmtp = p2x.wilson, p2x.fmtf, p2x.fmtp


def agree(a, b) -> dict:
    """一致性：原始一致率、κ、PABAK、AC1。后两者对边际不平衡稳健。"""
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return {"n": 0}
    po = float((a == b).mean())
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    k = (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else None
    pi = (pa + pb) / 2
    peg = 2 * pi * (1 - pi)
    return {"n": n, "po": po, "flip_rate": 1 - po, "kappa": k,
            "pabak": 2 * po - 1,
            "ac1": ((po - peg) / (1 - peg) if abs(1 - peg) > 1e-12 else None),
            "pos_run1": pa, "pos_run2": pb}


def flip_ci(rows) -> tuple:
    f = [int(r["_flip"]) for r in rows]
    if not f:
        return (None, None, None)
    lo, hi = wilson(sum(f), len(f))
    return (float(np.mean(f)), lo, hi)


# ==============================================================================
#  H：失效机制探查
# ==============================================================================

CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9.]+)')


def exp_H(p2: Phase2) -> dict:
    def run():
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        a, log = p2.a, p2.log
        md = RunData(a.main_run, log)
        rows = [r for r in md.l1 if r.get("cov_ij_shuf") is not None]
        calls = {c["qid"]: c for c in md.calls}
        for r in rows:
            r["_flip"] = int(r["cov_ij"] != r["cov_ij_shuf"])
            cj = calls.get(r["qid_j"], {})
            r["_n_facts"] = len(r.get("facts_ij") or [])
            r["_n_chunks"] = len(cj.get("retrieved_chunks") or [])
            r["_verdict"] = (r.get("verdict_ij") or "?").upper()
            m = CONF_RE.search(r.get("raw") or "")
            r["_conf"] = float(m.group(1)) if m else None
            r["_tok"] = r.get("tokens")
            r["_trunc"] = int(bool(r.get("truncated")))
            r["_selfcov"] = int(bool(r.get("self_cov_i")) and bool(r.get("self_cov_j")))
        hi = [r for r in rows if (r.get("bin") or 0) >= a.h_min_bin]
        log.info(f"  双次判定 {len(rows)}，其中 bin>={a.h_min_bin} 的 {len(hi)}"
                 f"（整体翻转 {np.mean([r['_flip'] for r in rows]):.1%}，"
                 f"高层 {np.mean([r['_flip'] for r in hi]):.1%}）")
        cov = {k: sum(1 for r in hi if r.get(k) is not None) / max(len(hi), 1)
               for k in ("_n_facts", "_conf", "_tok")}
        log.info(f"  字段可用率：facts {cov['_n_facts']:.0%}  "
                 f"confidence {cov['_conf']:.0%}  tokens {cov['_tok']:.0%}")

        out = {"n_all": len(rows), "n_high": len(hi), "min_bin": a.h_min_bin,
               "field_coverage": cov, "groups": {}}

        def group(name, keyfn, order=None):
            g = defaultdict(list)
            for r in hi:
                k = keyfn(r)
                if k is not None:
                    g[k].append(r)
            ks = order or sorted(g)
            ent = {}
            log.info(f"  [{name}]")
            for k in ks:
                if k not in g or len(g[k]) < 30:
                    continue
                f, lo, up = flip_ci(g[k])
                ent[str(k)] = {"n": len(g[k]), "flip": f, "ci": [lo, up]}
                log.info(f"     {str(k):<22} n={len(g[k]):>6}  翻转 {f:.1%}  "
                         f"CI[{lo:.1%},{up:.1%}]")
            out["groups"][name] = ent
            return ent

        # 判官自己的三分类裁决
        group("verdict_ij", lambda r: r["_verdict"],
              ["SUFFICIENT", "PARTIAL", "INSUFFICIENT"])
        # 所需事实数（证据分散度代理）
        group("n_required_facts", lambda r: (f"{r['_n_facts']}" if r["_n_facts"] <= 3
                                             else "4+") if r["_n_facts"] else None)
        # 自报置信度
        if cov["_conf"] > 0.3:
            group("confidence", lambda r: (None if r["_conf"] is None else
                                           ("<0.6" if r["_conf"] < 0.6 else
                                            "0.6-0.8" if r["_conf"] < 0.8 else ">=0.8")))
        else:
            log.info("  [confidence] 可用率过低，跳过")
        # 检索结果重叠度
        def q4(vals, v):
            if v is None:
                return None
            qs = np.percentile(vals, [25, 50, 75])
            return "Q1" if v <= qs[0] else "Q2" if v <= qs[1] else "Q3" if v <= qs[2] else "Q4"
        jv = [r.get("jaccard") for r in hi if r.get("jaccard") is not None]
        if len(jv) > 100:
            group("jaccard_quartile", lambda r: q4(jv, r.get("jaccard")),
                  ["Q1", "Q2", "Q3", "Q4"])
        tv = [r["_tok"] for r in hi if r["_tok"] is not None]
        if len(tv) > 100:
            group("tokens_quartile", lambda r: q4(tv, r["_tok"]), ["Q1", "Q2", "Q3", "Q4"])
        group("truncated", lambda r: "yes" if r["_trunc"] else "no", ["no", "yes"])

        # ---------- 哪个信号最能预测翻转 ----------
        feats, names = [], []
        def add(nm, fn):
            v = [fn(r) for r in hi]
            if all(x is not None for x in v) and len(set(v)) > 1:
                feats.append(v); names.append(nm)
        add("cos", lambda r: r["cos"])
        add("n_facts", lambda r: r["_n_facts"])
        add("jaccard", lambda r: r.get("jaccard") or 0.0)
        add("rbo", lambda r: r.get("rbo") or 0.0)
        add("tokens", lambda r: r["_tok"] or 0)
        add("is_partial", lambda r: float(r["_verdict"] == "PARTIAL"))
        add("is_insufficient", lambda r: float(r["_verdict"] == "INSUFFICIENT"))
        add("same_text", lambda r: float(bool(r.get("same_text"))))
        if feats:
            X = np.array(feats, float).T
            y = np.array([r["_flip"] for r in hi], int)
            Xs = StandardScaler().fit_transform(X)
            lr = LogisticRegression(max_iter=2000).fit(Xs, y)
            coef = sorted(zip(names, lr.coef_[0]), key=lambda x: -abs(x[1]))
            out["logit_coef"] = {k: float(v) for k, v in coef}
            log.info("  [翻转的预测因子（标准化系数，绝对值降序）]")
            for k, v in coef:
                log.info(f"     {k:<18}{v:>+8.3f}")

        # ---------- 弃判曲线：对可疑样本弃判，能把翻转压到多低 ----------
        log.info("  [弃判曲线] 对判定器最可疑的一部分弃判后，剩余样本的翻转率")
        curves = {}
        def curve(nm, score_fn, higher_is_worse=True):
            """弃判曲线。

            注意：n_facts、is_partial 这类预测子取值离散且大量并列，用分位数切会让
            实际弃判比例与名义比例严重不符（n_facts 在 quantile(0.9) 处切，名义弃 10%
            实际弃 66%）。这里改为按排名精确弃判 top-k 条，并列内随机打破，
            同时报告**真实**保留比例。
            """
            sc = [score_fn(r) for r in hi]
            if any(s is None for s in sc):
                return
            sc = np.array(sc, float)
            if not higher_is_worse:
                sc = -sc
            n = len(sc)
            rng = np.random.default_rng(0)
            order = np.lexsort((rng.random(n), -sc))     # 分数降序，并列随机
            fl = np.array([r["_flip"] for r in hi], int)
            pts = []
            for frac in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.66):
                k = int(round(frac * n))
                keep_idx = order[k:]
                if len(keep_idx) < 50:
                    continue
                f = fl[keep_idx]
                lo, up = wilson(int(f.sum()), len(f))
                pts.append({"abstain_frac": frac, "n_kept": int(len(keep_idx)),
                            "kept_frac": float(len(keep_idx) / n),
                            "flip": float(f.mean()), "ci": [lo, up]})
            curves[nm] = pts
            log.info(f"     {nm}: " + "  ".join(
                f"弃{p['abstain_frac']:.0%}(留{p['kept_frac']:.0%})->翻转{p['flip']:.1%}"
                for p in pts))

        def rule(nm, mask_fn):
            """按一条确定性规则弃判（如「只保留单事实查询」），报告真实代价。"""
            keep = np.array([bool(mask_fn(r)) for r in hi])
            if keep.sum() < 50:
                return
            fl = np.array([r["_flip"] for r in hi], int)[keep]
            lo, up = wilson(int(fl.sum()), len(fl))
            base = float(np.mean([r["_flip"] for r in hi]))
            curves[f"rule:{nm}"] = [{"kept_frac": float(keep.mean()),
                                     "n_kept": int(keep.sum()),
                                     "flip": float(fl.mean()), "ci": [lo, up],
                                     "flip_baseline": base}]
            log.info(f"     规则「{nm}」：保留 {keep.mean():.1%}（{keep.sum()} 条），"
                     f"翻转 {base:.1%} -> {fl.mean():.1%} CI[{lo:.1%},{up:.1%}]")
        curve("n_facts", lambda r: float(r["_n_facts"]))
        # 二值/离散预测子用确定性规则更诚实
        rule("只保留判官裁决为 SUFFICIENT 的", lambda r: r["_verdict"] == "SUFFICIENT")
        rule("只保留单事实查询 (n_facts<=1)", lambda r: r["_n_facts"] <= 1)
        rule("两者都满足", lambda r: r["_verdict"] == "SUFFICIENT" and r["_n_facts"] <= 1)
        if cov["_conf"] > 0.3:
            curve("low_confidence", lambda r: (None if r["_conf"] is None else -r["_conf"]))
        curve("low_jaccard", lambda r: -(r.get("jaccard") or 0.0))
        out["abstention_curves"] = curves

        # ---------- 结论 ----------
        v = out["groups"].get("verdict_ij", {})
        concl = []
        if "PARTIAL" in v and "SUFFICIENT" in v:
            d = v["PARTIAL"]["flip"] - v["SUFFICIENT"]["flip"]
            concl.append(f"判官自报 PARTIAL 的样本翻转率比 SUFFICIENT 高 {d:+.1%}"
                         + ("，可直接用作弃判信号" if d > 0.15 else "，区分力有限"))
        base = float(np.mean([r["_flip"] for r in hi]))
        for nm, pts in curves.items():
            if nm.startswith("rule:"):
                q = pts[0]
                concl.append(f"{nm[5:]}：保留 {q['kept_frac']:.0%} 的对，"
                             f"翻转率 {base:.1%} -> {q['flip']:.1%}")
        p20 = next((p for p in curves.get("n_facts", []) 
                    if abs(p["abstain_frac"] - 0.2) < 1e-9), None)
        if p20:
            concl.append(f"按 n_facts 排名弃判 20%（真实保留 {p20['kept_frac']:.0%}）后，"
                         f"翻转率 {base:.1%} -> {p20['flip']:.1%}")
        out["conclusions"] = concl
        for c in concl:
            log.info(f"  ★ {c}")
        wjson(p2.root / "H" / "h_mechanism.json", out)
        return {"summary": "；".join(concl) or "无显著信号"}
    return p2.run_step("H", "判官失效机制探查（离线）", run)


# ==============================================================================
#  I：视图时效性
# ==============================================================================

def exp_I(p2: Phase2) -> dict:
    def run():
        a, log = p2.a, p2.log
        md = RunData(a.main_run, log)
        calls = [c for c in md.calls if c["qid"] in md.qidx]
        E = md.E
        I_, J_, COS = md.pairs(min_cos=0.80)
        by_task = defaultdict(list)
        for k, c in enumerate(calls):
            by_task[c["task_id"]].append(k)
        tasks = sorted(by_task)
        W = a.i_window
        gs = [g for g in (0.80, 0.85) if md.tau(g) is not None]
        log.info(f"  任务 {len(tasks)}，建视图用前 {a.i_build} 个，"
                 f"之后每 {W} 个任务为一批，保证 {gs}，种子 {a.i_seeds}")

        pts = []
        for seed in range(a.i_seeds):
            order = list(tasks)
            random.Random(7000 + seed).shuffle(order)
            build = order[:a.i_build]
            rest = order[a.i_build:]
            nb = min(a.i_batches, len(rest) // W)
            for g in gs:
                tau = md.tau(g)
                # 冻结视图：只用最初 build 的任务
                def make(hist_tasks):
                    h = [k for t in hist_tasks for k in by_task[t]]
                    hidx = np.array([md.qidx[calls[k]["qid"]] for k in h], np.int64)
                    inh = np.zeros(len(E), bool); inh[hidx] = True
                    loc = -np.ones(len(E), np.int64); loc[hidx] = np.arange(len(hidx))
                    sel = inh[I_] & inh[J_] & (COS >= tau)
                    nc, lab = p2x._components(len(hidx), loc[I_[sel]], loc[J_[sel]])
                    if nc == 0:
                        return np.zeros((0, E.shape[1]), np.float32)
                    sz = np.bincount(lab, minlength=nc)
                    cent = p2x._centroids(E[hidx], lab, nc)
                    return cent[sz >= 2]
                frozen = make(build)
                log.info(f"  seed={seed} g={g:.2f}: 冻结视图 {len(frozen)} 个")
                for b in range(nb):
                    ev = [k for t in rest[b * W:(b + 1) * W] for k in by_task[t]]
                    if not ev:
                        continue
                    Q = E[np.array([md.qidx[calls[k]["qid"]] for k in ev], np.int64)]
                    cost = np.array([calls[k].get("cost", 1.0) for k in ev], float)
                    row = {"seed": seed, "g": g, "batch": b + 1,
                           "age_tasks": b * W, "n_eval": len(ev)}
                    bf, _ = p2x._max_sim(Q, frozen)
                    hf = bf >= tau
                    row["frozen_hit_cost"] = float(cost[hf].sum() / cost.sum())
                    # 持续重建：用到本批为止见过的全部任务
                    if a.i_refresh:
                        cur = make(build + rest[:b * W])
                        br, _ = p2x._max_sim(Q, cur)
                        hr = br >= tau
                        row["refreshed_hit_cost"] = float(cost[hr].sum() / cost.sum())
                        row["n_views_refreshed"] = int(len(cur))
                    row["n_views_frozen"] = int(len(frozen))
                    pts.append(row)
                    log.info(f"     批 {b+1} (龄 {row['age_tasks']} 任务)  冻结 "
                             f"{fmtp(row['frozen_hit_cost'])}"
                             + (f"  持续重建 {fmtp(row.get('refreshed_hit_cost'))}"
                                if a.i_refresh else ""))
                    p2.beat(f"I seed={seed} g={g} batch={b+1}")

        curve = []
        for g in gs:
            for b in sorted({r["batch"] for r in pts if r["g"] == g}):
                rr = [r for r in pts if r["g"] == g and r["batch"] == b]
                def ms(k):
                    v = [r[k] for r in rr if r.get(k) is not None]
                    return (float(np.mean(v)), float(np.std(v))) if v else (None, None)
                fm, fs = ms("frozen_hit_cost")
                rm, rs = ms("refreshed_hit_cost")
                curve.append({"g": g, "batch": b, "age_tasks": rr[0]["age_tasks"],
                              "frozen": fm, "frozen_sd": fs,
                              "refreshed": rm, "refreshed_sd": rs,
                              "n_seeds": len(rr)})
        out = {"build_tasks": a.i_build, "window": W, "curve": curve, "points": pts}
        # 衰减幅度
        for g in gs:
            c = [x for x in curve if x["g"] == g]
            if len(c) >= 2 and c[0]["frozen"] and c[-1]["frozen"]:
                out[f"frozen_decay_{g:.2f}"] = c[-1]["frozen"] - c[0]["frozen"]
                if c[-1].get("refreshed"):
                    out[f"refresh_gain_{g:.2f}"] = c[-1]["refreshed"] - c[-1]["frozen"]
        wjson(p2.root / "I" / "i_staleness.json", out)
        k = "frozen_decay_0.85"
        s = (f"g=0.85 冻结视图命中从首批到末批变化 {out[k]:+.1%}"
             + (f"，持续重建在末批高出 {out.get('refresh_gain_0.85', 0):+.1%}"
                if "refresh_gain_0.85" in out else "")) if k in out else "数据不足"
        log.info(f"  ★ {s}")
        return {"summary": s}
    return p2.run_step("I", "视图时效性：冻结 vs 持续重建（离线）", run)


# ==============================================================================
#  J：可复用的判官诊断工具
# ==============================================================================

DIAG_DOC = """判官诊断工具 —— 输入格式
每行一个 JSON，字段：
  item_id   条目标识
  stratum   难度分层（如相似度区间名）；诊断按此分层报告
  run1      第一次判定，0/1
  run2      第二次判定（同样的证据，只改呈现顺序等无关因素），0/1
  human     可选，人工判定 0/1
输出：每层的原始一致率、κ、PABAK、AC1，以及聚合值掩盖了哪一层。
"""


def diagnose(rows, out_dir: Path, log=None) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[str(r.get("stratum", "all"))].append(r)
    per, worst = {}, None
    for s in sorted(by):
        g = by[s]
        st = agree([int(r["run1"]) for r in g], [int(r["run2"]) for r in g])
        hu = [r for r in g if r.get("human") is not None]
        if len(hu) >= 20:
            st["vs_human"] = agree([int(r["run1"]) for r in hu],
                                   [int(r["human"]) for r in hu])
        per[s] = st
        if st["n"] >= 50 and (worst is None or st["pabak"] < per[worst]["pabak"]):
            worst = s
    allr = agree([int(r["run1"]) for r in rows], [int(r["run2"]) for r in rows])
    rep = {"n": len(rows), "aggregate": allr, "by_stratum": per, "worst_stratum": worst}
    if worst:
        rep["aggregate_hides"] = {
            "aggregate_pabak": allr["pabak"], "worst_pabak": per[worst]["pabak"],
            "gap": allr["pabak"] - per[worst]["pabak"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    wjson(out_dir / "diagnostic.json", rep)
    L = ["# 判官重测诊断", "", f"条目 {len(rows)}；聚合 PABAK "
         f"{allr['pabak']:.3f}（一致率 {allr['po']:.1%}）", "",
         "| 分层 | n | 一致率 | κ | PABAK | AC1 |", "|---|---|---|---|---|---|"]
    for s, st in per.items():
        L.append(f"| {s} | {st['n']} | {st['po']:.1%} | {fmtf(st['kappa'],3)} | "
                 f"{fmtf(st['pabak'],3)} | {fmtf(st['ac1'],3)} |")
    if worst:
        L += ["", f"**最差分层 `{worst}`：PABAK {per[worst]['pabak']:.3f}，"
                  f"比聚合值低 {allr['pabak']-per[worst]['pabak']:.3f}。**",
              "", "聚合一致率会被容易的分层稀释；若阈值落在最差分层附近，"
                  "该分层的重测一致性才是仪器可信度的上界。"]
    (out_dir / "diagnostic.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "INPUT_FORMAT.txt").write_text(DIAG_DOC, encoding="utf-8")
    if log:
        log.info(f"  聚合 PABAK {allr['pabak']:.3f}；最差分层 {worst} "
                 f"PABAK {per[worst]['pabak']:.3f}" if worst else "")
    return rep


def exp_J(p2: Phase2) -> dict:
    def run():
        log = p2.log
        md = RunData(p2.a.main_run, log)
        rows = [r for r in md.l1 if r.get("cov_ij_shuf") is not None]
        per_bin_range = {str(b): v.get("range")
                         for b, v in (md.calib.get("per_bin") or {}).items()}
        generic = [{"item_id": r.get("pair_id", f"{r['qid_i'][:8]}_{r['qid_j'][:8]}"),
                    "stratum": (f"cos[{per_bin_range.get(str(r.get('bin')), ['?','?'])[0]},"
                                f"{per_bin_range.get(str(r.get('bin')), ['?','?'])[1]})"
                                if per_bin_range.get(str(r.get("bin"))) else str(r.get("bin"))),
                    "run1": int(r["cov_ij"]), "run2": int(r["cov_ij_shuf"])}
                   for r in rows]
        d = p2.root / "J"
        p2x.wjsonl(d / "judge_retest_generic.jsonl", generic)
        rep = diagnose(generic, d, log)
        log.info(f"  已导出通用格式 {len(generic)} 条 -> {d/'judge_retest_generic.jsonl'}")
        log.info(f"  诊断报告 {d/'diagnostic.md'}；输入格式说明 {d/'INPUT_FORMAT.txt'}")
        log.info("  用在别人的数据上：phase3_all.py --diagnose-input X.jsonl --diagnose-out Y/")
        return {"summary": f"聚合 PABAK {rep['aggregate']['pabak']:.3f}，"
                           f"最差分层 {rep['worst_stratum']}"}
    return p2.run_step("J", "可复用的判官诊断工具（离线）", run)


# ==============================================================================
#  B：BCP 续跑
# ==============================================================================

def kill_stale(log) -> None:
    for pat in ("vllm serve", "EngineCore", "VLLM::"):
        with contextlib.suppress(Exception):
            subprocess.run(["pkill", "-9", "-f", pat], timeout=30)
    time.sleep(8)
    with contextlib.suppress(FileNotFoundError):
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        log.info(f"  清理后显存 {r.stdout.strip().splitlines()}")
        free = [int(x.split(",")[1].strip().split()[0]) for x in r.stdout.strip().splitlines()]
        if free and max(free) < 40000:
            raise RuntimeError(f"清理后最大空闲仅 {max(free)} MiB，"
                               f"可能有他人任务，请先手动确认 nvidia-smi")


def exp_B(p2: Phase2) -> dict:
    p2.log.info("清理残留推理进程")
    kill_stale(p2.log)
    return p2x.exp_B(p2)


# ==============================================================================
EXP = {"H": exp_H, "I": exp_I, "J": exp_J, "B": exp_B}
ORDER = ["H", "I", "J", "B"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3")
    ap.add_argument("--config")
    ap.add_argument("--exp", default="H,I,J")
    ap.add_argument("--phase1", default="phase1_v13.py")
    ap.add_argument("--main-run", default="runs/main_v10")
    ap.add_argument("--distractor-run", default="runs/distractor")
    ap.add_argument("--out-root", default="runs/phase2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--pilot", type=int, default=None)
    ap.add_argument("--agent", default="tongyi", choices=["tongyi", "qwen32b"])
    # H
    ap.add_argument("--h-min-bin", type=int, default=6, help="机制探查聚焦的最低层")
    # I
    ap.add_argument("--i-build", type=int, default=8000, help="建视图用的任务数")
    ap.add_argument("--i-window", type=int, default=1500, help="每批任务数")
    ap.add_argument("--i-batches", type=int, default=6)
    ap.add_argument("--i-seeds", type=int, default=2)
    ap.add_argument("--no-i-refresh", dest="i_refresh", action="store_false",
                    help="不跑「持续重建」对照（快一些）")
    # generic 诊断模式
    ap.add_argument("--diagnose-input")
    ap.add_argument("--diagnose-out", default="diagnostic_out")
    # 供 p2x.exp_B 复用
    ap.add_argument("--a-temperature", type=float, default=None)
    ap.add_argument("--b-max-queries", type=int, default=None)
    ap.add_argument("--b-max-docs", type=int, default=None)
    ap.add_argument("--b-shard", type=int, default=100000)
    ap.add_argument("--b-skip-downstream", action="store_true")
    a = ap.parse_args()

    if a.diagnose_input:
        rows = rjsonl(Path(a.diagnose_input))
        need = {"item_id", "stratum", "run1", "run2"}
        miss = need - set(rows[0]) if rows else need
        if miss:
            print(f"输入缺字段 {sorted(miss)}\n\n{DIAG_DOC}")
            return 2
        rep = diagnose(rows, Path(a.diagnose_out))
        print(json.dumps({"n": rep["n"], "aggregate_pabak": rep["aggregate"]["pabak"],
                          "worst_stratum": rep["worst_stratum"]},
                         ensure_ascii=False, indent=2))
        print(f"-> {a.diagnose_out}/diagnostic.md")
        return 0

    if not a.config:
        print("需要 --config（或用 --diagnose-input 进入通用诊断模式）")
        return 2
    if a.distractor_run and not Path(a.distractor_run).exists():
        a.distractor_run = None
    exps = ORDER if a.exp == "all" else [e.strip().upper() for e in a.exp.split(",")]
    bad = [e for e in exps if e not in EXP]
    if bad:
        print(f"未知实验 {bad}，可选 {ORDER}")
        return 2
    exps = [e for e in ORDER if e in exps]

    p2 = Phase2(a)
    p2.log.info(f"Phase 3 {exps}")
    t0 = time.time()
    for e in exps:
        try:
            EXP[e](p2)
        except Exception as ex:
            p2.failures.append((e, f"{type(ex).__name__}: {ex}"))
            p2.stop_servers(); p2.save_state()
            p2.log.error(f"{e} 失败: {type(ex).__name__}: {ex}")
            if a.stop_on_error:
                break
    p2x.write_report(p2)
    p2.log.info(f"总耗时 {(time.time()-t0)/3600:.2f} h")
    return 1 if p2.failures else 0


if __name__ == "__main__":
    sys.exit(main())
