#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
干扰语料实验：复用结构在真实检索难度下是否稳健
================================================================================

为什么做
--------
§7 最大的外部效度威胁：我们的语料由工作负载问题所需的支撑段落构成
（141,550 段落），检索比开放域容易得多。基础索引答得出的查询更多、
相关查询的检索结果重叠也更多，两者都抬高复用率。
审稿人会问：换成真实规模的语料，这些结论还成立吗？

实验设计
--------
    A 组（已有）  main_v10    141,550 段落（全是支撑段落）
    C 组（本脚本）distractor  原语料 + N 倍干扰段落

**关键：查询保持不变。** 我们用 main_v10 采集到的原始查询在新索引上
重新检索，而不是重跑 agent。理由是如果重跑 agent，它会看到不同文档、
走不同轨迹，复用率的变化就分不清来自语料还是来自轨迹。
同样的查询 + 更难的检索 = 受控对照，而且省掉 4 小时 agent 时间。

干扰段落的来源
--------------
用同样三个数据集的 **train split**（工作负载只用 validation），取其中
从未被查询过的问题的支撑段落。这比随机网页文本更严格：
同一领域（Wikipedia）、同样的文体、同样的粒度，只是内容无关。
一个来自开放网页的干扰语料反而更容易被嵌入模型区分开。

预期与判读
----------
自覆盖率必降、复用率必降 —— 这不是失败，是预期。
真正要看的是**净增量的分域模式是否保持**：
  agent 域远高于 replay 域  -> 结论稳健，可写进论文
  模式消失                  -> 近似匹配的收益依赖语料的易检索性，必须报告

用法
----
    # 1) 构建扩大后的语料并编码（约 30 min）
    python distractor.py build --run runs/main_v10 --out runs/distractor --scale 5

    # 2) 用原查询在新索引上重新检索（约 5 min）
    python distractor.py replay --run runs/main_v10 --out runs/distractor

    # 3) 重跑下游（s6 判官是大头，约 5-7 h）
    $R/envs/main/bin/python phase1_v13.py --config $R/config.json \
        --run-id distractor --stages s3,s4,s5,s6,s7,s8,s9,s10,s11,s13,s17,s23 --force

    # 4) 对照（秒级）
    python distractor.py compare --run-a runs/main_v10 --run-c runs/distractor
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PHASE1 = "phase1_v13.py"


def load_phase1():
    """import 主脚本，复用其 TorchRetriever / _iter_paragraphs / _chunk 等实现。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("p13", PHASE1)
    P = importlib.util.module_from_spec(spec)
    sys.modules["p13"] = P
    spec.loader.exec_module(P)
    return P


# ==============================================================================
#  build —— 构建扩大后的语料
# ==============================================================================

def cmd_build(a) -> int:
    from datasets import load_dataset
    from tqdm import tqdm
    P = load_phase1()

    run, out = Path(a.run), Path(a.out)
    cfg = P.Config.load(a.config)
    cfg.run_id = out.name
    cfg.root = str(out.parent)
    cfg.resolve()
    ctx = P.RunContext(cfg)
    log = ctx.log

    try:
        cfg._autotune = P.autotune_hardware(cfg, log)
    except Exception as e:
        log.error(f"硬件自适应失败: {e}")
        return 3

    # ---------- 1. 读原语料 ----------
    src_chunks = run / "stages" / "s1" / "chunks.jsonl"
    if not src_chunks.exists():
        sys.exit(f"找不到 {src_chunks}")
    orig = ctx.rjsonl(src_chunks)
    log.info(f"原语料 {len(orig)} 块")
    # 用于去重：支撑段落绝不能被干扰段落重复加入
    seen = {P.sha(c.get("docid", ""), c["text"][:300]) for c in orig}
    seen_text = {c["text"][:300] for c in orig}

    # ---------- 2. 抓干扰段落 ----------
    target = int(len(orig) * a.scale) - len(orig)
    log.info(f"目标：新增 {target} 块干扰段落（总规模 ×{a.scale}）")
    log.info("来源：三个数据集的 train split（工作负载只用 validation），")
    log.info("      同领域、同文体、内容无关 —— 比随机网页文本更严格的干扰")

    # train split 的候选源。与主脚本的 _DATASET_SRC 对应，但取 train。
    SRC = {
        "musique":  [("dgslibisey/MuSiQue", None, "train")],
        "hotpotqa": [("hotpotqa/hotpot_qa", "distractor", "train"),
                     ("hotpot_qa", "distractor", "train")],
        "2wiki":    [("voidful/2WikiMultihopQA", None, "train")],
    }
    names = a.datasets.split(",") if a.datasets else list(SRC)
    per_ds = target // max(len(names), 1) + 1

    dist, stats = [], {}
    for name in names:
        got = False
        for repo, sub, split in SRC.get(name, []):
            try:
                kw = {"cache_dir": cfg.hf_cache, "split": split}
                d = load_dataset(repo, sub, **kw) if sub else load_dataset(repo, **kw)
            except Exception as e:
                log.warning(f"  {name}/{split} 源 {repo} 失败: "
                            f"{type(e).__name__}: {str(e)[:110]}")
                continue
            n0, n_dup = len(dist), 0
            for r in tqdm(d, desc=f"distract-{name}", ncols=100):
                if len(dist) - n0 >= per_ds:
                    break
                for title, text in P._iter_paragraphs(r):
                    text = (text or "").strip()
                    if len(text) < 50:
                        continue
                    if text[:300] in seen_text:
                        n_dup += 1
                        continue
                    key = P.sha(title, text[:300])
                    if key in seen:
                        n_dup += 1
                        continue
                    seen.add(key); seen_text.add(text[:300])
                    dist.append({"docid": f"distract-{name}-{key}",
                                 "title": title or "",
                                 "text": (f"{title}. {text}" if title else text)})
                    if len(dist) - n0 >= per_ds:
                        break
            stats[name] = {"added": len(dist) - n0, "dup_skipped": n_dup,
                           "source": f"{repo}/{split}"}
            log.info(f"  {name}: +{len(dist)-n0} 段落（跳过重复 {n_dup}，源 {repo}）")
            got = True
            break
        if not got:
            log.error(f"  {name} 全部源失败")

    if not dist:
        sys.exit("一条干扰段落都没抓到，检查数据集可用性")

    # ---------- 3. 切块 ----------
    size, stride = cfg.chunk_tokens * 4, cfg.chunk_stride * 4
    d_ids, d_docs, d_texts = [], [], []
    for r in tqdm(dist, desc="chunk-distract", ncols=100):
        for j, c in enumerate(P._chunk(r["text"], size, stride)):
            d_ids.append(f"{r['docid']}::{j}")
            d_docs.append(r["docid"])
            d_texts.append(c)
    log.info(f"干扰切块 {len(d_ids)}")

    # ---------- 4. 编码干扰块，与原嵌入拼接 ----------
    mp = cfg.embed_model_final if cfg.use_final_embedder else cfg.embed_model_dev
    r = P.TorchRetriever(ctx, mp, cfg.retriever_gpu)
    t0 = time.time()
    d_emb = r.encode(d_texts, False, cfg.embed_batch, "distractor")
    dt = time.time() - t0

    orig_emb = np.load(run / "stages" / "s1" / "chunk_emb.npy")
    assert orig_emb.shape[0] == len(orig), \
        f"原嵌入 {orig_emb.shape[0]} 与原块数 {len(orig)} 不符"
    all_emb = np.concatenate([orig_emb, d_emb], axis=0)

    ctx.wjsonl(ctx.out("s1", "chunks.jsonl"),
               ([*orig] + [{"chunkid": i, "docid": dd, "text": t}
                           for i, dd, t in zip(d_ids, d_docs, d_texts)]))
    np.save(ctx.out("s1", "chunk_emb.npy"), all_emb)
    st = {"corpus_source": "datasets+distractor",
          "n_chunks": all_emb.shape[0],
          "n_chunks_original": len(orig),
          "n_chunks_distractor": len(d_ids),
          "scale_factor": all_emb.shape[0] / max(len(orig), 1),
          "n_docs": len(orig) + len(dist),
          "dim": int(all_emb.shape[1]), "model": mp,
          "encode_sec": dt, "distractor_sources": stats,
          "index_gb": all_emb.shape[0] * all_emb.shape[1] * 2 / 2**30}
    ctx.wjson(ctx.out("s1", "stats.json"), st)
    r.free()

    log.info(f"扩大后语料 {all_emb.shape[0]} 块 "
             f"（原 {len(orig)} + 干扰 {len(d_ids)}，×{st['scale_factor']:.2f}）")
    log.info(f"  索引 {st['index_gb']:.2f} GB，编码 {dt/60:.1f} min")
    log.info(f"  -> {ctx.out('s1', 'chunks.jsonl')}")
    print()
    print(f"  下一步：python {Path(__file__).name} replay "
          f"--run {run} --out {out}")
    return 0


# ==============================================================================
#  replay —— 用原查询在新索引上重新检索
# ==============================================================================

def cmd_replay(a) -> int:
    from tqdm import tqdm
    P = load_phase1()

    run, out = Path(a.run), Path(a.out)
    cfg = P.Config.load(a.config)
    cfg.run_id = out.name
    cfg.root = str(out.parent)
    cfg.resolve()
    ctx = P.RunContext(cfg)
    log = ctx.log
    try:
        cfg._autotune = P.autotune_hardware(cfg, log)
    except Exception as e:
        log.error(f"硬件自适应失败: {e}")
        return 3

    src = run / "stages" / "s2" / "calls.jsonl"
    if not src.exists():
        sys.exit(f"找不到 {src}")
    calls = ctx.rjsonl(src)
    log.info(f"原查询 {len(calls)} 条 / {len({c['task_id'] for c in calls})} 任务")
    log.info("保持查询不变，只换索引 —— 这样复用率的变化只能归因于语料")

    retr = P._mk_retriever(ctx)   # 读的是 out 目录下扩大后的索引
    log.info(f"新索引 {len(retr.chunk_ids)} 块")

    B = 512
    fin = []
    for s in tqdm(range(0, len(calls), B), desc="re-retrieve", ncols=100):
        b = calls[s:s + B]
        t0 = time.time()
        S, I = retr.search([c["query"] for c in b], cfg.topk, batch=B)
        lat = (time.time() - t0) * 1000 / len(b)
        for k, c in enumerate(b):
            ds = retr.docs(I[k])
            nc = dict(c)   # 保留 qid / task_id / source / query / gold_answer 等
            nc.update({
                "retrieved": [d["docid"] for d in ds],
                "retrieved_chunks": [d["chunkid"] for d in ds],
                "scores": [float(x) for x in S[k][:len(ds)]],
                "latency_ms": lat,
                "context_tokens_added": sum(len(d["text"]) for d in ds) // 4,
                "n_reranked": len(ds)})
            nc["cost"] = (cfg.cost_per_search_call
                          + cfg.cost_per_context_token * nc["context_tokens_added"]
                          + cfg.cost_per_rerank_doc * nc["n_reranked"])
            fin.append(nc)
        ctx.beat(f"re-retrieve {s+len(b)}/{len(calls)}")
    retr.free()

    # ---------- 检索结果变化了多少 ----------
    old = {c["qid"]: set(c.get("retrieved_chunks", [])) for c in calls}
    ov = [len(old[c["qid"]] & set(c["retrieved_chunks"]))
          / max(len(old[c["qid"]] | set(c["retrieved_chunks"])), 1)
          for c in fin if c["qid"] in old]
    unchanged = sum(1 for c in fin
                    if c["qid"] in old and old[c["qid"]] == set(c["retrieved_chunks"]))
    log.info(f"[检索变化] 与原检索的 Jaccard 均值 {np.mean(ov):.3f}；"
             f"完全未变 {unchanged}/{len(fin)} ({unchanged/max(len(fin),1):.1%})")
    log.info("  Jaccard 越低说明干扰语料越有效地改变了检索结果")

    qt = Counter(c["query"] for c in fin)
    bt = defaultdict(set)
    for c in fin:
        bt[c["query"]].add(c["task_id"])
    st = {"total_calls": len(fin),
          "total_tasks": len({c["task_id"] for c in fin}),
          "calls_per_task": len(fin) / max(len({c["task_id"] for c in fin}), 1),
          "replayed_from": str(run), "queries_unchanged": True,
          "retrieval_jaccard_vs_clean": float(np.mean(ov)) if ov else None,
          "retrieval_identical_frac": unchanged / max(len(fin), 1),
          "sanity": {
              "exact_dup_query_frac":
                  sum(n for q, n in qt.items() if len(bt[q]) >= 2) / max(len(fin), 1),
              "unresolved_placeholder_frac":
                  sum(1 for c in fin if re.search(r"#\d+", c["query"])) / max(len(fin), 1),
              "unique_query_frac": len(qt) / max(len(fin), 1),
              "median_query_len": float(np.median([len(c["query"]) for c in fin])),
              "with_gold_frac":
                  sum(1 for c in fin if c.get("gold_answer")) / max(len(fin), 1)}}
    # agent 统计原样带过来（查询没变）
    a_st = run / "stages" / "s2" / "stats.json"
    if a_st.exists():
        try:
            st["agent"] = ctx.rjson(a_st).get("agent", {})
        except Exception:
            pass
    ctx.wjsonl(ctx.out("s2", "calls.jsonl"), fin)
    ctx.wjson(ctx.out("s2", "stats.json"), st)
    # s0 链过去，免得重跑自检
    s0 = run / "stages" / "s0" / "preflight.json"
    if s0.exists():
        dp = ctx.out("s0", "preflight.json")
        if not dp.exists():
            os.symlink(s0.resolve(), dp)
    log.info(f"  -> {ctx.out('s2', 'calls.jsonl')}")
    print()
    print("  下一步（重跑下游，s6 判官约 5-7 h）：")
    print(f"    $R/envs/main/bin/python {PHASE1} --config $R/config.json \\")
    print(f"        --run-id {out.name} \\")
    print(f"        --stages s3,s4,s5,s6,s7,s8,s9,s10,s11,s13,s17,s23 --force")
    return 0


# ==============================================================================
#  compare —— 干净语料 vs 干扰语料
# ==============================================================================

def cmd_compare(a) -> int:
    """干净语料 vs 干扰语料。

    注意：旧版用点号拆路径，但文件名 stats.json 本身带点，
    "s1/stats.json.n_chunks" 会被错拆成 ["s1/stats","json","n_chunks"]，
    全部取不到值。这里改成 (文件, 字段路径) 两段式。
    """
    ra, rc = Path(a.run_a), Path(a.run_c)
    FILES = ["s1/stats.json", "s2/stats.json", "s6/qc.json", "s7/summary.json",
             "s8/calib.json", "s10/reuse.json", "s13/scale.json",
             "s17/view_hit.json", "s23/cache_vs_view.json"]

    def load(r):
        out = {}
        for f in FILES:
            p = r / "stages" / f
            out[f] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return out
    A, C = load(ra), load(rc)

    def get(d, f, *keys):
        cur = d.get(f, {})
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur

    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return f"{v:,}"
        if isinstance(v, float):
            return f"{v:.1%}" if 0 <= v <= 1.0 else f"{v:.4f}"
        return str(v)

    rows = [
        ("Corpus chunks",           "s1/stats.json", ("n_chunks",)),
        ("  distractor chunks",     "s1/stats.json", ("n_chunks_distractor",)),
        ("Retrieval Jaccard vs A",  "s2/stats.json", ("retrieval_jaccard_vs_clean",)),
        ("Retrieval identical",     "s2/stats.json", ("retrieval_identical_frac",)),
        None,
        ("Self-coverage rate",      "s6/qc.json", ("self_cov_rate",)),
        ("Eligibility rate",        "s6/qc.json", ("eligibility_rate",)),
        ("Cond. substitutability",  "s6/qc.json", ("substitutable_rate_conditional",)),
        ("f1_own (L2)",             "s7/summary.json", ("mean_f1_own",)),
        ("Nonzero loss frac",       "s7/summary.json", ("nonzero_loss_frac",)),
        ("P99 recall loss",         "s7/summary.json", ("p99_recall_loss_pos",)),
        None,
        ("tau*",                    "s8/calib.json", ("tau_star",)),
        ("Head cov. (cost)",        "s10/reuse.json", ("components", "head_cov_cost")),
        ("Cross-task intents",      "s10/reuse.json", ("components", "cross_intent_frac")),
        None,
        ("Normalized cache (cost)", "s23/cache_vs_view.json",
         ("baselines", "normalized_cache", "hit_cost")),
        ("View @0.85 (cost)",       "s23/cache_vs_view.json", ("views", "0.85", "hit_cost")),
        ("Net increment @0.85",     "s23/cache_vs_view.json",
         ("increment", "0.85", "cost_frac_of_eval")),
    ]
    w = 28
    line = "=" * (w + 38)
    print(); print(line)
    print(f"  {'':<{w}}{'A: clean':>16}{'C: distractor':>18}")
    print(line)
    for r in rows:
        if r is None:
            print("-" * (w + 38)); continue
        k, f, keys = r
        va, vc = get(A, f, *keys), get(C, f, *keys)
        # tau* 用小数显示
        if k == "tau*":
            sa = f"{va:.4f}" if isinstance(va, float) else "-"
            sc = f"{vc:.4f}" if isinstance(vc, float) else "-"
        else:
            sa, sc = fmt(va), fmt(vc)
        print(f"  {k:<{w}}{sa:>16}{sc:>18}")
    print(line)

    # ---------- 分域净增量：本实验的判读依据 ----------
    def inc(d):
        return get(d, "s23/cache_vs_view.json", "increment", "0.85", "by_source") or {}
    ia, ic = inc(A), inc(C)
    if ia or ic:
        print()
        print("  ★ 分域净增量（按成本，保证 0.85）")
        print(f"  {'Workload':<22}{'A: clean':>12}{'C: distractor':>16}{'delta':>10}")
        for s_ in sorted(set(ia) | set(ic)):
            va = (ia.get(s_) or {}).get("cost_frac")
            vc = (ic.get(s_) or {}).get("cost_frac")
            dd = f"{vc - va:+.1%}" if (va is not None and vc is not None) else "-"
            print(f"  {s_:<22}{fmt(va):>12}{fmt(vc):>16}{dd:>10}")
        agc = (ic.get("agent") or {}).get("cost_frac")
        rep = [v.get("cost_frac") for k_, v in ic.items()
               if k_.startswith("replay") and v.get("cost_frac") is not None]
        if agc is not None and rep:
            print(f"\n  干扰语料下 agent / 最高 replay = {agc / max(max(rep), 1e-9):.1f}x")

    # ---------- 判读：检索是否真的变了 ----------
    jac = get(C, "s2/stats.json", "retrieval_jaccard_vs_clean")
    ta, tc = get(A, "s8/calib.json", "tau_star"), get(C, "s8/calib.json", "tau_star")
    print()
    if jac is None:
        print("  ⚠ 缺检索 Jaccard（replay 的 s2/stats.json），无法判定干扰是否生效")
    elif jac > 0.9:
        print(f"  判读：检索 Jaccard {jac:.3f} > 0.9 —— 干扰段落几乎没进 top-k。")
        print("        结论仅证明对【语料规模】稳健，不能证明对【检索难度】稳健。")
        print("        原因：实体中心的事实型查询对实体不相交的干扰天然免疫。")
    else:
        print(f"  判读：检索 Jaccard {jac:.3f} —— 检索结果确实改变了。")
        if isinstance(ta, float) and isinstance(tc, float) and abs(ta - tc) < 0.01:
            print("        τ* 不变，复用结构对检索难度稳健 —— 强结论。")

    op = Path(a.out_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"A": {"run": str(ra), **A}, "C": {"run": str(rc), **C},
               "increment_by_source": {"clean": ia, "distractor": ic}},
              open(op, "w"), ensure_ascii=False, indent=2, default=str)
    print(f"\n  -> {op}")
    return 0


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="干扰语料稳健性实验")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="构建扩大后的语料并编码")
    b.add_argument("--run", default="runs/main_v10")
    b.add_argument("--out", default="runs/distractor")
    b.add_argument("--config", default=None)
    b.add_argument("--scale", type=float, default=5.0,
                   help="总语料相对原语料的倍数，默认 5（141K -> ~700K）")
    b.add_argument("--datasets", default=None,
                   help="干扰来源，逗号分隔，默认三个都用")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("replay", help="用原查询在新索引上重新检索")
    r.add_argument("--run", default="runs/main_v10")
    r.add_argument("--out", default="runs/distractor")
    r.add_argument("--config", default=None)
    r.set_defaults(func=cmd_replay)

    c = sub.add_parser("compare", help="干净 vs 干扰")
    c.add_argument("--run-a", default="runs/main_v10")
    c.add_argument("--run-c", default="runs/distractor")
    c.add_argument("--out-json", default="runs/distractor/paper/distractor.json")
    c.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
