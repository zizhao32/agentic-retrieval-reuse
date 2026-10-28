#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
补洞 —— 给已写进论文的三个说法补上它们缺的证据
================================================================================
不是新发现，是让现有主张站得住。审稿人挑的往往就是这类。

  K1  §5 缓存对照的误差棒
      净增量 8.6pp、视图命中 21.7% 现在只基于**一个** train/test 切分、一个种子，
      而 C 有 Wilson 区间、D/I 有多种子 sd、A 有配对自助法 —— 偏偏最核心那张表没有。
      做法：精确复刻 s23 的计算，先用原种子验证能复现已发表的数，再跑多种子给 ±。

  K2  §4 头部覆盖的不确定性
      28.7% / 23.3% 同样是单点。s13 在各采样比例上有跨种子 sd，但 100% 那点 sd=0。
      做法：对任务做 m-out-of-n 子采样（m=0.8n，无放回），按 sqrt(m/n) 修正得到全样本 sd。
      不用有放回自助：重复任务会产生重复 qid，与按 qid 建图的聚类冲突。

  K3  §7 成本模型敏感性
      现在写的是「按次与按成本相差不到一个百分点，故结论不依赖成本常数」——
      这句推不出来：两者接近只说明在当前常数下恰好接近。
      做法：把三个常数各扫一个数量级，看分域净增量的排序与头部覆盖是否翻转。
      要么补上这个扫描，要么把那句话改弱。

用法
----
    python fill_gaps.py --config $R/config.json --exp K1,K2,K3
    python fill_gaps.py --config $R/config.json --exp K1 --k1-seeds 5
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
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
rjsonl, wjson, rjson = p2x.rjsonl, p2x.wjson, p2x.rjson
fmtf, fmtp = p2x.fmtf, p2x.fmtp


def ms(v):
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (None, None)


# ==============================================================================
#  复刻 s23：一个种子一次完整的「缓存 vs 视图」评估
# ==============================================================================

def cache_vs_view(P, cfg, md, seed: int, targets, log=None) -> dict:
    """与 phase1_v13 的 s23 逐步对齐：同样的特征、同样的阈值搜索、同样的切分方式。
    额外返回每条评估查询的成本分量与命中掩码，供 K3 重新加权（避免重算）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    calls = md.calls
    qidx, E = md.qidx, md.E
    cmap = {c["qid"]: c for c in calls}
    l1 = [r for r in md.l1]

    # ---- 判定器：cos + 实体 Jaccard + 模板 Jaccard，按分层权重加权 ----
    X, y, W_ = [], [], []
    for r in l1:
        ci, cj = cmap.get(r["qid_i"]), cmap.get(r["qid_j"])
        if not ci or not cj:
            continue
        X.append([r["cos"],
                  P._jac(P._entity_set(ci["query"]), P._entity_set(cj["query"])),
                  P._jac(P._template_set(ci["query"]), P._template_set(cj["query"]))])
        y.append(r["substitutable"]); W_.append(r.get("w", 1.0))
    X = np.array(X, float); y = np.array(y, int); W_ = np.array(W_, float)
    sk = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sk.transform(X), y, sample_weight=W_)
    sc_all = lr.predict_proba(sk.transform(X))[:, 1]
    thr = {}
    for tg in targets:
        best = None
        for t in np.quantile(sc_all, np.linspace(0.0, 0.998, 300)):
            m = sc_all >= t
            if int(m.sum()) < 50:
                continue
            _, lo, _, _ = P.wilson_w(y[m], W_[m])
            if lo >= tg and (best is None or t < best):
                best = float(t)
        thr[tg] = best

    # ---- 按任务切分（唯一随种子变化的地方）----
    tasks = sorted({c["task_id"] for c in calls})
    rnd = random.Random(seed); rnd.shuffle(tasks)
    ncut = int(len(tasks) * cfg.view_eval_task_frac)
    build_t = set(tasks[:ncut])
    build = [c for c in calls if c["task_id"] in build_t and c["qid"] in qidx]
    evalq = [c for c in calls if c["task_id"] not in build_t and c["qid"] in qidx]

    exact = {c["query"] for c in build}
    normed = {p2x.norm_text(c["query"]) for c in build}
    hit_exact = np.array([c["query"] in exact for c in evalq])
    hit_norm = np.array([p2x.norm_text(c["query"]) in normed for c in evalq])

    tau = md.calib.get("tau_star") or 0.90
    bset = {c["qid"] for c in build}
    dsu = P._DSU([c["qid"] for c in build])
    for p in md.pairs_raw:
        if p["cos"] >= tau and p["qid_i"] in bset and p["qid_j"] in bset:
            dsu.union(p["qid_i"], p["qid_j"])
    grp = defaultdict(list)
    for c in build:
        grp[dsu.find(c["qid"])].append(c["qid"])
    views = [(k, v) for k, v in grp.items() if len(v) >= 2]
    if not views:
        return {"error": "no views", "seed": seed}
    cen = np.stack([E[[qidx[q] for q in v]].mean(0) for _, v in views])
    cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    reps = [cmap[v[0]]["query"] for _, v in views]
    rep_ent = [P._entity_set(t) for t in reps]
    rep_tpl = [P._template_set(t) for t in reps]

    ev_idx = [qidx[c["qid"]] for c in evalq]
    sims = E[ev_idx] @ cen.T
    bv = np.argmax(sims, axis=1); bs = sims[np.arange(len(ev_idx)), bv]
    feats = np.zeros((len(evalq), 3))
    for i, c in enumerate(evalq):
        j = int(bv[i])
        feats[i] = [float(bs[i]),
                    P._jac(P._entity_set(c["query"]), rep_ent[j]),
                    P._jac(P._template_set(c["query"]), rep_tpl[j])]
    sc = lr.predict_proba(sk.transform(feats))[:, 1]

    cost = np.array([c.get("cost", 1.0) for c in evalq], float)
    srcs = np.array([c["source"] for c in evalq])
    tok = np.array([c.get("context_tokens_added", 0) for c in evalq], float)
    nrr = np.array([c.get("n_reranked", 0) for c in evalq], float)

    out = {"seed": seed, "n_views": len(views), "n_eval": len(evalq), "tau": tau,
           "thresholds": {f"{k:.2f}": v for k, v in thr.items()},
           "cache_norm_cost": float(cost[hit_norm].sum() / cost.sum()),
           "cache_exact_cost": float(cost[hit_exact].sum() / cost.sum()),
           "views": {}, "increment": {}}
    masks = {}
    for tg in targets:
        t = thr.get(tg)
        if t is None:
            out["views"][f"{tg:.2f}"] = None
            continue
        hv = sc >= t
        inc = hv & (~hit_norm)
        masks[f"{tg:.2f}"] = hv
        out["views"][f"{tg:.2f}"] = {
            "hit_q": float(hv.mean()), "hit_cost": float(cost[hv].sum() / cost.sum())}
        out["increment"][f"{tg:.2f}"] = {
            "cost_frac": float(cost[inc].sum() / cost.sum()),
            "by_source": {s: float(cost[(srcs == s) & inc].sum() / max(cost[srcs == s].sum(), 1e-9))
                          for s in sorted(set(srcs.tolist()))}}
    out["_arrays"] = {"cost": cost, "tok": tok, "nrr": nrr, "srcs": srcs,
                      "hit_norm": hit_norm, "masks": masks}
    return out


def _prep(p2: Phase2):
    """共用的加载：主脚本 + 配置 + run 数据（含未压缩的 pairs，供 DSU 用）。"""
    P = p2.P
    cfg = P.Config.load(p2.a.config)
    cfg.resolve()
    md = RunData(p2.a.main_run, p2.log)
    md.pairs_raw = rjsonl(md.f("s4/pairs.jsonl"))
    return P, cfg, md


# ==============================================================================
#  K1：缓存对照的多种子误差棒
# ==============================================================================

def exp_K1(p2: Phase2) -> dict:
    def run():
        log = p2.log
        P, cfg, md = _prep(p2)
        targets = md.calib.get("report_targets") or [0.80, 0.85, 0.90, 0.95]
        published = rjson(md.f("s23/cache_vs_view.json")) if md.exists("s23/cache_vs_view.json") else None

        log.info(f"  先用原种子 {cfg.seed} 复刻 s23，核对能否复现已发表的数")
        base = cache_vs_view(P, cfg, md, cfg.seed, targets, log)
        chk = {}
        if published:
            pb = ((published.get("baselines") or {}).get("normalized_cache") or {}).get("hit_cost")
            pv = ((published.get("views") or {}).get("0.85") or {}).get("hit_cost")
            pi = ((published.get("increment") or {}).get("0.85") or {}).get("cost_frac_of_eval")
            mb, mv, mi = (base["cache_norm_cost"],
                          (base["views"].get("0.85") or {}).get("hit_cost"),
                          (base["increment"].get("0.85") or {}).get("cost_frac"))
            chk = {"published": {"cache": pb, "view": pv, "incr": pi},
                   "replicated": {"cache": mb, "view": mv, "incr": mi}}
            ok = all(a is not None and b is not None and abs(a - b) < 0.005
                     for a, b in ((pb, mb), (pv, mv), (pi, mi)))
            chk["match"] = bool(ok)
            log.info(f"    缓存 {fmtp(pb)} vs {fmtp(mb)} | 视图 {fmtp(pv)} vs {fmtp(mv)} | "
                     f"净增 {fmtp(pi)} vs {fmtp(mi)}  ->  {'一致' if ok else '★不一致'}")
            if not ok:
                log.error("    复刻与已发表结果不一致，多种子的误差棒不可用；"
                          "请先核对 s23 实现是否已改动")
        else:
            log.warning("    找不到 s23/cache_vs_view.json，跳过复刻核对")

        seeds = [cfg.seed] + [cfg.seed + 1000 * i for i in range(1, p2.a.k1_seeds)]
        runs = [base]
        for s in seeds[1:]:
            log.info(f"  种子 {s} …")
            runs.append(cache_vs_view(P, cfg, md, s, targets, log))
            p2.beat(f"K1 seed {s}")
        runs = [r for r in runs if "error" not in r]

        agg = {"n_seeds": len(runs), "seeds": [r["seed"] for r in runs],
               "replication_check": chk, "cache_norm": ms([r["cache_norm_cost"] for r in runs]),
               "cache_exact": ms([r["cache_exact_cost"] for r in runs]), "by_target": {}}
        log.info(f"  {'保证':>6}{'视图命中(成本)':>18}{'净增量':>16}{'agent 净增':>16}")
        for tg in targets:
            k = f"{tg:.2f}"
            v = ms([(r["views"].get(k) or {}).get("hit_cost") for r in runs])
            i = ms([(r["increment"].get(k) or {}).get("cost_frac") for r in runs])
            bysrc = {}
            for s in sorted({s_ for r in runs for s_ in
                             ((r["increment"].get(k) or {}).get("by_source") or {})}):
                bysrc[s] = ms([((r["increment"].get(k) or {}).get("by_source") or {}).get(s)
                               for r in runs])
            agg["by_target"][k] = {"view_hit_cost": v, "increment": i, "by_source": bysrc}
            if v[0] is not None:
                a_ = bysrc.get("agent", (None, None))
                log.info(f"  {k:>6}{f'{v[0]*100:.1f} ± {v[1]*100:.1f}':>18}"
                         f"{f'{i[0]*100:.1f} ± {i[1]*100:.1f}':>16}"
                         f"{(f'{a_[0]*100:.1f} ± {a_[1]*100:.1f}' if a_[0] is not None else '-'):>16}")
        log.info(f"  归一化缓存 {agg['cache_norm'][0]*100:.1f} ± {agg['cache_norm'][1]*100:.1f}")
        # 保留基准种子的成本分量与命中掩码，供 K3 重新加权（不必重算）
        import pickle
        (p2.root / "K").mkdir(parents=True, exist_ok=True)
        with open(p2.root / "K" / "k1_base_arrays.pkl", "wb") as f:
            pickle.dump(base["_arrays"], f)
        for r in runs:
            r.pop("_arrays", None)
        agg["per_seed"] = runs
        wjson(p2.root / "K" / "k1_cache_vs_view_seeds.json", agg)
        k85 = agg["by_target"].get("0.85", {})
        return {"summary": f"净增量 @0.85 = {k85['increment'][0]*100:.1f} ± "
                           f"{k85['increment'][1]*100:.1f} pp（{len(runs)} 个种子）"
                           f"{'；复刻一致' if chk.get('match') else ''}"}
    return p2.run_step("K1", "缓存对照的多种子误差棒", run)


# ==============================================================================
#  K2：头部覆盖的不确定性
# ==============================================================================

def exp_K2(p2: Phase2) -> dict:
    def run():
        log = p2.log
        P, cfg, md = _prep(p2)
        calls = md.calls
        pairs = md.pairs_raw
        tau = md.calib.get("tau_star") or 0.90
        by_task = defaultdict(list)
        for c in calls:
            by_task[c["task_id"]].append(c)
        tasks = sorted(by_task)
        m = int(len(tasks) * p2.a.k2_frac)
        B = p2.a.k2_boot
        log.info(f"  任务 {len(tasks)}，每次抽 {m}（{p2.a.k2_frac:.0%}，无放回）× {B} 次")
        log.info("  用 m-out-of-n 子采样而非有放回自助：重复任务会产生重复 qid，"
                 "与按 qid 建图的聚类冲突")

        full = P._head_cov(calls, pairs, tau, cfg.stop_head_frac)
        srcs = sorted({c["source"] for c in calls})
        full_src = {s: P._head_cov([c for c in calls if c["source"] == s], pairs, tau,
                                   cfg.stop_head_frac) for s in srcs}
        acc, acc_src = [], defaultdict(list)
        for b in range(B):
            keep = set(random.Random(9000 + b).sample(tasks, m))
            sub = [c for c in calls if c["task_id"] in keep]
            r = P._head_cov(sub, pairs, tau, cfg.stop_head_frac)
            if r:
                acc.append(r["head_cov_cost"])
            for s in srcs:
                rs = P._head_cov([c for c in sub if c["source"] == s], pairs, tau,
                                 cfg.stop_head_frac)
                if rs:
                    acc_src[s].append(rs["head_cov_cost"])
            if (b + 1) % 10 == 0:
                log.info(f"    {b+1}/{B}")
                p2.beat(f"K2 {b+1}/{B}")

        corr = math.sqrt(p2.a.k2_frac)      # sd_full ≈ sd_sub * sqrt(m/n)
        def rep(name, point, vals):
            if not vals:
                return None
            sd_sub = float(np.std(vals, ddof=1))
            sd = sd_sub * corr
            ci = [point - 1.96 * sd, point + 1.96 * sd]
            log.info(f"  {name:<22} {point*100:6.1f}%  ± {sd*100:.2f}  "
                     f"95%CI[{ci[0]*100:.1f}, {ci[1]*100:.1f}]  (子样本 sd {sd_sub*100:.2f})")
            return {"point": point, "sd_sub": sd_sub, "sd_corrected": sd, "ci": ci,
                    "n_resamples": len(vals)}
        out = {"tau": tau, "subsample_frac": p2.a.k2_frac, "B": B,
               "note": "sd 由 m-out-of-n 子采样经 sqrt(m/n) 修正得到，非有放回自助",
               "all": rep("全域", full["head_cov_cost"], acc),
               "by_source": {s: rep(s, full_src[s]["head_cov_cost"], acc_src[s])
                             for s in srcs if full_src.get(s)}}
        wjson(p2.root / "K" / "k2_head_coverage_ci.json", out)
        a = out["by_source"].get("agent") or out["all"]
        return {"summary": f"agent 头部覆盖 {a['point']*100:.1f}% ± {a['sd_corrected']*100:.2f}"}
    return p2.run_step("K2", "头部覆盖的不确定性", run)


# ==============================================================================
#  K3：成本模型敏感性
# ==============================================================================

def exp_K3(p2: Phase2) -> dict:
    def run():
        import pickle
        log = p2.log
        P, cfg, md = _prep(p2)
        pk = p2.root / "K" / "k1_base_arrays.pkl"
        if not pk.exists():
            raise FileNotFoundError("缺 K1 的中间产物，请先跑 --exp K1")
        A = pickle.load(open(pk, "rb"))
        cost0 = (cfg.cost_per_search_call, cfg.cost_per_context_token, cfg.cost_per_rerank_doc)
        log.info(f"  基准常数 c_call={cost0[0]} c_tok={cost0[1]} c_rerank={cost0[2]}")
        log.info("  各扫一个数量级，看分域净增量的排序与相对关系是否翻转")

        tok, nrr, srcs = A["tok"], A["nrr"], A["srcs"]
        hit_norm, masks = A["hit_norm"], A["masks"]
        hv = masks.get("0.85")
        if hv is None:
            raise RuntimeError("K1 的基准种子在 0.85 上没有可行阈值")
        inc = hv & (~hit_norm)

        def eval_cost(cc, ct, cr):
            cost = cc + ct * tok + cr * nrr
            tot = cost.sum()
            by = {s: float(cost[(srcs == s) & inc].sum() / max(cost[srcs == s].sum(), 1e-9))
                  for s in sorted(set(srcs.tolist()))}
            ag = by.get("agent")
            rep = [v for k, v in by.items() if k.startswith("replay") and v is not None]
            return {"c": [cc, ct, cr],
                    "view_hit_cost": float(cost[hv].sum() / tot),
                    "cache_cost": float(cost[hit_norm].sum() / tot),
                    "increment": float(cost[inc].sum() / tot),
                    "by_source": by,
                    "agent_over_replay": (ag / max(max(rep), 1e-9)) if ag and rep else None}

        rows = [("baseline", *cost0)]
        for f in (0.1, 10.0):
            rows += [(f"c_call x{f:g}", cost0[0] * f, cost0[1], cost0[2]),
                     (f"c_tok x{f:g}", cost0[0], cost0[1] * f, cost0[2]),
                     (f"c_rerank x{f:g}", cost0[0], cost0[1], cost0[2] * f)]
        rows += [("仅按次 (c_tok=c_rerank=0)", 1.0, 0.0, 0.0),
                 ("仅上下文 (c_call=c_rerank=0)", 0.0, cost0[1], 0.0)]
        res = []
        log.info(f"  {'设定':<28}{'缓存':>8}{'视图':>8}{'净增':>8}{'agent':>9}"
                 f"{'最高replay':>12}{'倍数':>8}")
        for name, cc, ct, cr in rows:
            r = eval_cost(cc, ct, cr); r["name"] = name; res.append(r)
            rep = [v for k, v in r["by_source"].items() if k.startswith("replay")]
            log.info(f"  {name:<28}{fmtp(r['cache_cost']):>8}{fmtp(r['view_hit_cost']):>8}"
                     f"{fmtp(r['increment']):>8}{fmtp(r['by_source'].get('agent')):>9}"
                     f"{fmtp(max(rep) if rep else None):>12}"
                     f"{fmtf(r['agent_over_replay'],1):>8}")
        ratios = [r["agent_over_replay"] for r in res if r["agent_over_replay"]]
        incs = [r["increment"] for r in res]
        # 判据应当是「排序是否保持」，而不是某个人为的倍数阈值：
        # 论文的结论是相对量（agent 域的净增量高于所有 replay 域），
        # 只要这个序在整个扫描范围内不变，结论就不依赖成本常数的取值。
        def ordered(r):
            ag = r["by_source"].get("agent")
            rep = [v for k, v in r["by_source"].items() if k.startswith("replay") and v is not None]
            return (ag is not None and rep and ag > max(rep))
        bad = [r["name"] for r in res if not ordered(r)]
        out = {"baseline_constants": cost0, "rows": res,
               "ordering_preserved": not bad, "ordering_violations": bad,
               "agent_over_replay_range": [min(ratios), max(ratios)] if ratios else None,
               "increment_range": [min(incs), max(incs)]}
        if not bad:
            out["conclusion"] = (
                "在整个扫描范围内 agent 域的净增量始终高于所有 replay 域"
                + (f"（倍数 {min(ratios):.1f}–{max(ratios):.1f}）" if ratios else "")
                + f"；净增量的绝对值随常数在 {min(incs)*100:.1f}–{max(incs)*100:.1f}pp 之间变动。"
                  "相对结论不依赖成本常数，绝对值依赖。")
        else:
            out["conclusion"] = (f"★ 分域排序在以下设定下被打破：{bad}。"
                                 "§7 那句「结论不依赖成本常数」必须删掉或限定范围。")
        log.info(f"  {out['conclusion']}")
        wjson(p2.root / "K" / "k3_cost_sensitivity.json", out)
        return {"summary": out["conclusion"][:100]}
    return p2.run_step("K3", "成本模型敏感性", run)


# ==============================================================================
EXP = {"K1": exp_K1, "K2": exp_K2, "K3": exp_K3}
ORDER = ["K1", "K2", "K3"]


def main() -> int:
    ap = argparse.ArgumentParser(description="补洞：给已有主张补证据")
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="K1,K2,K3")
    ap.add_argument("--phase1", default="phase1_v13.py")
    ap.add_argument("--main-run", default="runs/main_v10")
    ap.add_argument("--distractor-run", default=None)
    ap.add_argument("--out-root", default="runs/phase2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--k1-seeds", type=int, default=5)
    ap.add_argument("--k2-boot", type=int, default=40)
    ap.add_argument("--k2-frac", type=float, default=0.8)
    a = ap.parse_args()
    exps = [e.strip().upper() for e in a.exp.split(",")]
    exps = [e for e in ORDER if e in exps]
    if not exps:
        print(f"可选 {ORDER}"); return 2
    p2 = Phase2(a)
    p2.log.info(f"补洞 {exps}")
    t0 = time.time()
    for e in exps:
        try:
            EXP[e](p2)
        except Exception as ex:
            p2.failures.append((e, f"{type(ex).__name__}: {ex}"))
            p2.save_state()
            p2.log.error(f"{e} 失败: {type(ex).__name__}: {ex}")
            if a.stop_on_error:
                break
    p2.log.info(f"总耗时 {(time.time()-t0)/60:.1f} min")
    return 1 if p2.failures else 0


if __name__ == "__main__":
    sys.exit(main())
