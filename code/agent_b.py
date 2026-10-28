#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Agent 强度敏感性：第二个 agent 模型的采集与对照
================================================================================

为什么做
--------
main_v10 的 §7 里，「单一 agent 模型」是未测试的威胁之一。现有唯一对照是
多查询双臂（all vs first，差 +1.6pp），那只覆盖「查询生成方式」，
不覆盖「agent 强度」。审稿人最可能点名要的补充就是这一条。

对照设计
--------
    A 组（已有）  Tongyi-DeepResearch-30B-A3B   MoE，48 层，每 token 激活 3.3B
    B 组（本脚本）Qwen3-32B-AWQ                  dense，64 层，激活 32B

两者激活参数量差一个数量级，且 A 是专为 deep research 训练的、B 是通用模型。
如果头部复用率接近，说明结论对 agent 强度不敏感（强结论）；
如果差异大，那个差异本身就是发现（也是强结论）。**两种结果都能写。**

为什么不下载新模型
------------------
Qwen3-32B-AWQ 已在 models/ 下（判官在用），支持 tool calling，
19 GB 单卡放得下。不占额外磁盘、不需要联网。

做法
----
1) 用 phase1_v13.py 的 s2 逻辑跑 B 组 agent，只采集 agent 调用
2) 把 B 组调用并入一份新的 calls.jsonl（source 标为 agent-qwen32b）
3) 复用 main_v10 的语料与索引，重跑 s3~s13 得到 B 组的复用结构
4) 本脚本的 compare 模式直接对照两组

用法
----
    # 步骤 1：采集 B 组（需 GPU，约 2-3 h，1000 任务）
    python agent_b.py collect --run runs/main_v10 --tasks 1000

    # 步骤 2：合并并重跑下游（约 40 min，不碰 s6 判官）
    python agent_b.py merge --run runs/main_v10 --out runs/agent_b

    # 步骤 3：对照（纯离线，几秒）
    python agent_b.py compare --run-a runs/main_v10 --run-b runs/agent_b
================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# 复用主脚本的实现，避免逻辑分叉
PHASE1 = "phase1_v13.py"

AGENT_B_SOURCE = "agent-qwen32b"
AGENT_A_SOURCE = "agent"


# ==============================================================================
#  collect —— 跑 B 组 agent
# ==============================================================================

def cmd_collect(a) -> int:
    """用 main_v10 的语料与索引跑第二个 agent，只产出 agent 调用。

    直接 import 主脚本，复用 TorchRetriever / LLMPool / _agent_task /
    _extract_queries，保证 B 组和 A 组的采集逻辑完全一致 ——
    这是对照实验的前提，不能各写一套。
    """
    sys.path.insert(0, str(Path(PHASE1).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("p13", PHASE1)
    P = importlib.util.module_from_spec(spec)
    sys.modules["p13"] = P
    spec.loader.exec_module(P)

    run = Path(a.run)
    cfg = P.Config.load(a.config)
    cfg.run_id = a.run_id or f"agentb_collect"
    cfg.root = a.root
    # ---- B 组的 agent 模型 ----
    cfg.agent_path = a.model
    cfg.agent_name = Path(a.model).name
    cfg.agent_quant = a.quant
    cfg.agent_max_tasks = a.tasks
    cfg.agent_enabled = True
    cfg.resolve()

    ctx = P.RunContext(cfg)
    log = ctx.log
    log.info(f"B 组 agent: {cfg.agent_name}  ({cfg.agent_quant})")
    log.info(f"复用 {run} 的语料与索引")

    try:
        cfg._autotune = P.autotune_hardware(cfg, log)
    except Exception as e:
        log.error(f"硬件自适应失败: {e}")
        return 3

    # ---- 复用 A 组的 chunks + 嵌入，不重新编码 ----
    for src, dst in (("s1/chunks.jsonl", "s1/chunks.jsonl"),
                     ("s1/chunk_emb.npy", "s1/chunk_emb.npy"),
                     ("s1/stats.json", "s1/stats.json")):
        sp, dp = run / "stages" / src, ctx.out(*dst.split("/"))
        if not sp.exists():
            sys.exit(f"缺 {sp}，先确认 --run 指向跑完 s1 的目录")
        if not dp.exists():
            dp.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(sp.resolve(), dp)
    log.info("  已链接 A 组的语料与嵌入（不重复编码）")

    retr = P._mk_retriever(ctx)
    sm = P.ServerManager(ctx)

    def sig(s, f):
        log.warning(f"收到信号 {s}，清理退出")
        sm.stop(); sys.exit(130)
    signal.signal(signal.SIGINT, sig); signal.signal(signal.SIGTERM, sig)

    try:
        agent_gpus = [g for g in cfg.gpus if g != cfg.retriever_gpu] or list(cfg.gpus)
        log.info(f"agent 独占 GPU{agent_gpus}")
        urls = sm.ensure("agent", agent_gpus)
        pool = P.LLMPool(urls, cfg.agent_name, retries=2, timeout=900)

        # 用与 A 组**完全相同**的任务集，保证可比
        tasks = P._replay_tasks(ctx, "musique")[:cfg.agent_max_tasks]
        log.info(f"任务 {len(tasks)}（与 A 组同源同序，可逐任务配对）")

        res = P.pmap(lambda t: P._agent_task(pool, retr, t, cfg), tasks,
                     cfg.agent_concurrency, "agent-B", ctx, pool)
        nerr, top = P.summarize_errors(res, log, "agent-B")

        calls = []
        for r in res:
            if isinstance(r, dict) and not (r.get("__error__") or r.get("error")):
                for c in r.get("calls", []):
                    c["source"] = AGENT_B_SOURCE
                    calls.append(c)

        # 与 s2 相同的去重与成本计算
        seen, fin = set(), []
        for c in calls:
            c["query"] = re.sub(r"\s+", " ", str(c["query"])).strip()
            if len(c["query"]) < 5:
                continue
            qid = P.sha(c["task_id"], c["turn_idx"], c["query"])
            if qid in seen:
                continue
            seen.add(qid); c["qid"] = qid
            c["cost"] = (cfg.cost_per_search_call
                         + cfg.cost_per_context_token * c["context_tokens_added"]
                         + cfg.cost_per_rerank_doc * c["n_reranked"])
            fin.append(c)

        outp = ctx.out("s2", "agent_b_calls.jsonl")
        ctx.wjsonl(outp, fin)
        st = {"model": cfg.agent_name, "quant": cfg.agent_quant,
              "tasks": len(tasks), "errors": nerr, "top_errors": top,
              "calls": len(fin), "llm": pool.stats(),
              "multi_query_frac": float(np.mean([bool(c.get("multi_query")) for c in fin]))
                                  if fin else 0.0,
              "mean_subqueries": float(np.mean([c.get("multi_query_n", 1) for c in fin]))
                                 if fin else 0.0,
              "median_query_len": float(np.median([len(c["query"]) for c in fin]))
                                  if fin else 0.0,
              "calls_per_task": len(fin) / max(len(tasks), 1)}
        ctx.wjson(ctx.out("s2", "agent_b_stats.json"), st)
        log.info(f"B 组产出 {len(fin)} 调用 / {len(tasks)} 任务 "
                 f"({st['calls_per_task']:.2f} 次/任务)，失败 {nerr}")
        log.info(f"  多查询比例 {st['multi_query_frac']:.1%}，"
                 f"平均子查询 {st['mean_subqueries']:.2f}，"
                 f"查询中位长度 {st['median_query_len']:.0f}")
        log.info(f"  -> {outp}")
    finally:
        sm.stop(); retr.free()

    print()
    print(f"  下一步：python {Path(__file__).name} merge --run {run} "
          f"--collected {outp}")
    return 0


# ==============================================================================
#  merge —— 把 B 组并入一份新的 run，重跑下游
# ==============================================================================

def cmd_merge(a) -> int:
    """构造 run_b：A 组的 replay 调用 + B 组的 agent 调用。

    这样两个 run 的 replay 部分完全相同，**差异只来自 agent**，
    分域对比才干净。
    """
    run_a, out = Path(a.run), Path(a.out)
    (out / "stages" / "s2").mkdir(parents=True, exist_ok=True)

    calls_a = [json.loads(l) for l in
               open(run_a / "stages" / "s2" / "calls.jsonl", encoding="utf-8")
               if l.strip()]
    coll = Path(a.collected) if a.collected else \
        run_a / "stages" / "s2" / "agent_b_calls.jsonl"
    if not coll.exists():
        sys.exit(f"找不到 B 组调用 {coll}，先跑 collect")
    calls_b = [json.loads(l) for l in open(coll, encoding="utf-8") if l.strip()]

    replay = [c for c in calls_a if c["source"].startswith("replay")]
    merged = replay + calls_b
    print(f"  A 组 replay {len(replay)} + B 组 agent {len(calls_b)} = {len(merged)}")

    # 链接不变的产出，避免重复计算
    for f in ("s1/chunks.jsonl", "s1/chunk_emb.npy", "s1/stats.json"):
        sp, dp = run_a / "stages" / f, out / "stages" / f
        dp.parent.mkdir(parents=True, exist_ok=True)
        if not dp.exists():
            os.symlink(sp.resolve(), dp)
    for f in ("s0/preflight.json",):
        sp, dp = run_a / "stages" / f, out / "stages" / f
        if sp.exists():
            dp.parent.mkdir(parents=True, exist_ok=True)
            if not dp.exists():
                os.symlink(sp.resolve(), dp)

    with open(out / "stages" / "s2" / "calls.jsonl", "w", encoding="utf-8") as f:
        for c in merged:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    st = {"total_calls": len(merged),
          "total_tasks": len({c["task_id"] for c in merged}),
          "calls_per_task": len(merged) / max(len({c["task_id"] for c in merged}), 1),
          "sanity": {"unresolved_placeholder_frac": 0.0, "exact_dup_query_frac": 0.0,
                     "unique_query_frac": 0.0, "median_query_len": 0.0},
          "agent_b_merged": True}
    # sanity 按合并后重算
    qt = Counter(c["query"] for c in merged)
    bt = defaultdict(set)
    for c in merged:
        bt[c["query"]].add(c["task_id"])
    st["sanity"] = {
        "exact_dup_query_frac": sum(n for q, n in qt.items() if len(bt[q]) >= 2) / len(merged),
        "unresolved_placeholder_frac":
            sum(1 for c in merged if re.search(r"#\d+", c["query"])) / len(merged),
        "unique_query_frac": len(qt) / len(merged),
        "median_query_len": float(np.median([len(c["query"]) for c in merged])),
        "with_gold_frac": sum(1 for c in merged if c.get("gold_answer")) / len(merged)}
    json.dump(st, open(out / "stages" / "s2" / "stats.json", "w"),
              ensure_ascii=False, indent=2)
    # 复制 config，改 run_id
    cfgp = run_a / "config_effective.json"
    if cfgp.exists():
        c = json.loads(cfgp.read_text(encoding="utf-8"))
        c["run_id"] = out.name
        json.dump(c, open(out / "config_effective.json", "w"),
                  ensure_ascii=False, indent=2)
    json.dump({"created_at": time.time(), "stages": {}},
              open(out / "state.json", "w"))
    print(f"  -> {out}")
    print()
    print("  下一步（重跑下游，约 40 min，不碰 s6 判官所以很快）：")
    print(f"    $R/envs/main/bin/python {PHASE1} --config $R/config.json \\")
    print(f"        --run-id {out.name} --stages s3,s4,s5,s6,s7,s8,s9,s10,s11,s13 --force")
    print()
    print("  注：s6 会重跑判官（B 组查询是新的，必须重判）。")
    print("      若只想要复用结构而不要 τ*，可以只跑 s3,s4,s9,s10,s13，")
    print(f"      并把 {run_a.name} 的 s8/calib.json 链过去复用 τ*。")
    return 0


# ==============================================================================
#  compare —— 两组对照
# ==============================================================================

def cmd_compare(a) -> int:
    ra, rb = Path(a.run_a), Path(a.run_b)

    def load(r, name):
        p = r / "stages" / "s10" / "reuse.json"
        if not p.exists():
            sys.exit(f"缺 {p}，先跑 s10")
        return json.loads(p.read_text(encoding="utf-8"))

    A, B = load(ra, "A"), load(rb, "B")

    def agent_row(d, keys):
        for k in keys:
            if k in (d.get("by_source") or {}):
                return d["by_source"][k]
        return {}

    ga = agent_row(A, [AGENT_A_SOURCE])
    gb = agent_row(B, [AGENT_B_SOURCE, AGENT_A_SOURCE])

    def stat(r, f):
        p = r / "stages" / "s2" / f
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    sa = (stat(ra, "stats.json").get("agent") or {})
    sb = stat(ra, "agent_b_stats.json") or stat(rb, "agent_b_stats.json")

    rows = [
        ("Model", sa.get("multi_query_mode") and "Tongyi-DR-30B-A3B" or "Tongyi-DR-30B-A3B",
         sb.get("model", "Qwen3-32B-AWQ")),
        ("Architecture", "MoE, 3.3B active", "dense, 32B active"),
        ("Tasks", sa.get("tasks"), sb.get("tasks")),
        ("Retrieval calls", ga.get("n_calls"), gb.get("n_calls")),
        ("Calls per task", _r(ga.get("n_calls"), sa.get("tasks")),
         _r(gb.get("n_calls"), sb.get("tasks"))),
        ("Multi-query frac", _p(sa.get("multi_query_frac")), _p(sb.get("multi_query_frac"))),
        ("Mean subqueries", _f(sa.get("mean_subqueries")), _f(sb.get("mean_subqueries"))),
        ("Median query len", sa.get("median_query_len"), sb.get("median_query_len")),
        ("---", "---", "---"),
        ("Intents", ga.get("n_intents"), gb.get("n_intents")),
        ("Head cov. (cost)", _p(ga.get("head_cov_cost")), _p(gb.get("head_cov_cost"))),
        ("Head cov. (calls)", _p(ga.get("head_cov_calls")), _p(gb.get("head_cov_calls"))),
        ("Cross-task intents", _p(ga.get("cross_intent_frac")), _p(gb.get("cross_intent_frac"))),
        ("Tasks per intent", _f(ga.get("mean_tasks_per_intent")),
         _f(gb.get("mean_tasks_per_intent"))),
        ("Mono-text head share", _p(ga.get("head_mono_text_share")),
         _p(gb.get("head_mono_text_share"))),
    ]
    w = max(len(str(r[0])) for r in rows) + 2
    print()
    print("=" * (w + 44))
    print(f"  {'':<{w}}{'A: Tongyi (MoE)':>20}{'B: Qwen3-32B':>22}")
    print("=" * (w + 44))
    for k, va, vb in rows:
        if k == "---":
            print("-" * (w + 44)); continue
        print(f"  {k:<{w}}{str(va):>20}{str(vb):>22}")
    print("=" * (w + 44))

    # 规模律
    for tag, r in (("A", ra), ("B", rb)):
        p = r / "stages" / "s13" / "scale.json"
        if p.exists():
            f = (json.loads(p.read_text(encoding="utf-8")).get("fits") or {})
            for k in (AGENT_A_SOURCE, AGENT_B_SOURCE):
                if f.get(k):
                    d = f[k]
                    print(f"  [{tag}] {k:<16} slope {d['slope']:+.4f}/decade  "
                          f"R²={d['r2']:.3f}  current {d['current']:.1%}  "
                          f"10x {d['pred']['10x']:.1%}")

    ha, hb = ga.get("head_cov_cost"), gb.get("head_cov_cost")
    if ha and hb:
        delta = hb - ha
        print()
        print(f"  ★ 头部成本覆盖差 {delta:+.1%}（B - A）")
        if abs(delta) < 0.05:
            print("     差异 < 5pp —— 结论对 agent 强度不敏感，可直接写进 Threats。")
        else:
            print("     差异 >= 5pp —— 这个差异本身是发现，值得单独一小段：")
            print("     复用率依赖 agent 的查询生成行为，不只是工作负载的性质。")

    out = {"A": {"run": str(ra), **ga}, "B": {"run": str(rb), **gb},
           "agent_a_stats": sa, "agent_b_stats": sb,
           "delta_head_cov_cost": (hb - ha) if (ha and hb) else None}
    op = Path(a.out_json)
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=2)
    print(f"\n  -> {op}")
    return 0


def _p(v):
    return f"{v:.1%}" if isinstance(v, (int, float)) else "-"


def _f(v):
    return f"{v:.2f}" if isinstance(v, (int, float)) else "-"


def _r(a, b):
    try:
        return f"{a / b:.2f}"
    except Exception:
        return "-"


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Agent 强度敏感性对照")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="跑 B 组 agent")
    c.add_argument("--run", default="runs/main_v10", help="复用其语料与索引")
    c.add_argument("--config", default=None)
    c.add_argument("--root", default="runs")
    c.add_argument("--run-id", default=None)
    c.add_argument("--model", default="/mnt/data/user_workspace/zhangzizhao/"
                                      "matview/models/Qwen3-32B-AWQ")
    c.add_argument("--quant", default="awq_marlin")
    c.add_argument("--tasks", type=int, default=1000)
    c.set_defaults(func=cmd_collect)

    m = sub.add_parser("merge", help="并入新 run")
    m.add_argument("--run", default="runs/main_v10")
    m.add_argument("--collected", default=None)
    m.add_argument("--out", default="runs/agent_b")
    m.set_defaults(func=cmd_merge)

    p = sub.add_parser("compare", help="两组对照")
    p.add_argument("--run-a", default="runs/main_v10")
    p.add_argument("--run-b", default="runs/agent_b")
    p.add_argument("--out-json", default="runs/agent_b/paper/agent_sensitivity.json")
    p.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
