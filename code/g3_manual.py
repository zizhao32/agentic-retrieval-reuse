#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
G3（人工版）—— 用人来回答「是判官不稳，还是任务本身有歧义」
================================================================================

背景
----
E/G1 测到：同一批对、同一组文档，只打乱呈现顺序，判官在 [0.98,0.999) 层
翻转 46.6%，κ=0.044、PABAK=0.067、AC1=0.199。
没有第二个 LLM 判官时，用人做同样的重测，能回答一个更关键的问题：

  如果人在同一批样本上判得稳、且不觉得难  -> 不稳定来自判官（仪器问题）
  如果人也来回摇摆、或普遍标为难判        -> 该区域的可替代性本身没有确定答案

注意：判官的重测测的是 **cov_ij 单方向**（一条查询 + 一组文档，是否够用），
所以人工任务也是单向判断，比标注可替代性对要轻。

工作表的构成
------------
  唯一条目：从「判官翻转过」和「判官两次一致」的对里各抽一半，
            覆盖 bin7/8/9（塌陷区）与 bin2/3（对照区）。标注者不知道哪条是哪类。
  重复条目：其中若干条在表里出现两次，第二次文档顺序被打乱，两次相隔很远。
            这是人的重测一致性，直接与判官的翻转率对比。

用法
----
    # 1) 生成工作表（人读 worksheet.md，答案填 answers.tsv）
    python g3_manual.py make --run runs/main_v10 --out runs/phase2/G3m

    # 2) 标完后评分
    python g3_manual.py score --out runs/phase2/G3m

已知局限（会写进论文）
----------------------
人看到重复条目时可能认出自己标过 —— 这会**高估**人的一致性。因此人的重测
数值是上界；但「判官 46.6% vs 人 ≤1/12」这个量级差异不会被该效应解释掉。
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def rjsonl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def wjson(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, ensure_ascii=False, indent=2), encoding="utf-8")


def wilson(k, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_tail(k, n, p):
    """P(X <= k)，用于「人翻转数远少于判官翻转率」的显著性。"""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def agree(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return {}
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    k = (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else None
    pi = (pa + pb) / 2
    peg = 2 * pi * (1 - pi)
    return {"n": n, "po": po, "flip_rate": 1 - po, "kappa": k, "pabak": 2 * po - 1,
            "ac1": ((po - peg) / (1 - peg) if abs(1 - peg) > 1e-12 else None)}


# ==============================================================================
#  make
# ==============================================================================

def cmd_make(a) -> int:
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)

    l1 = [r for r in rjsonl(run / "stages/s6/l1.jsonl")
          if r.get("eligible") == 1 and r.get("ok") and r.get("cov_ij_shuf") is not None]
    calls = {c["qid"]: c for c in rjsonl(run / "stages/s2/calls.jsonl")}
    print(f"双次判定的对 {len(l1)}")

    # 分层抽样：塌陷区多抽，对照区少抽；每层里翻转/一致各半
    plan = {int(k): int(v) for k, v in (x.split(":") for x in a.plan.split(","))}
    picked = []
    for b, n in plan.items():
        pool = [r for r in l1 if r.get("bin") == b]
        fl = [r for r in pool if r["cov_ij"] != r["cov_ij_shuf"]]
        cs = [r for r in pool if r["cov_ij"] == r["cov_ij_shuf"]]
        nf = min(len(fl), n // 2)
        nc = min(len(cs), n - nf)
        sel = rnd.sample(fl, nf) + rnd.sample(cs, nc)
        for r in sel:
            r["_flipped"] = int(r["cov_ij"] != r["cov_ij_shuf"])
            # 老版本 run 可能没有 pair_id，用两端 qid 兜底
            r.setdefault("pair_id", f'{r["qid_i"][:8]}_{r["qid_j"][:8]}')
        picked += sel
        print(f"  bin{b}: 可用 翻转{len(fl)}/一致{len(cs)} -> 抽 翻转{nf}/一致{nc}")
    if len(picked) < 8:
        sys.exit("可用样本太少，检查 --plan 或 main run")

    # 需要的文档
    need = set()
    for r in picked:
        need |= set(calls[r["qid_j"]].get("retrieved_chunks", []))
    cmap = {}
    with open(run / "stages/s1/chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d["chunkid"] in need:
                cmap[d["chunkid"]] = d["text"]
            if len(cmap) == len(need):
                break
    print(f"载入文档 {len(cmap)}/{len(need)}")

    # 重复条目：优先放在塌陷区
    dup_pool = sorted(picked, key=lambda r: (-(r.get("bin") or 0), rnd.random()))
    dups = dup_pool[:a.n_dup]
    dup_ids = {id(r) for r in dups}

    items = []
    for k, r in enumerate(picked):
        ch = list(calls[r["qid_j"]].get("retrieved_chunks", []))
        items.append({"item_id": f"I{k:03d}a", "pair_id": r["pair_id"], "bin": r.get("bin"),
                      "cos": r["cos"], "presentation": "orig", "chunks": ch,
                      "query": calls[r["qid_i"]]["query"],
                      "judge_verdict": r["cov_ij"], "flipped": r["_flipped"],
                      "is_dup_of": None})
        if id(r) in dup_ids:
            sh = list(ch)
            random.Random(hash(r["pair_id"]) & 0xFFFF).shuffle(sh)
            items.append({"item_id": f"I{k:03d}b", "pair_id": r["pair_id"], "bin": r.get("bin"),
                          "cos": r["cos"], "presentation": "permuted", "chunks": sh,
                          "query": calls[r["qid_i"]]["query"],
                          "judge_verdict": r["cov_ij_shuf"], "flipped": r["_flipped"],
                          "is_dup_of": f"I{k:03d}a"})

    # 排序：先把首次出现的条目打乱，再把重复条目逐个插到距其首次出现 >= gap 的位置。
    # 随机重排在「条目少、重复多」时几乎不可能满足间隔约束，必须用确定性插入。
    firsts = [it for it in items if not it["is_dup_of"]]
    seconds = [it for it in items if it["is_dup_of"]]
    rnd.shuffle(firsts)
    rnd.shuffle(seconds)
    total = len(items)
    gap = min(a.gap, max(2, total // 3))      # 条目太少时自动放宽，但不低于 2
    if gap < a.gap:
        print(f"⚠ 条目仅 {total} 条，间隔从 {a.gap} 自动放宽到 {gap}")
    seq = list(firsts)
    for b in seconds:
        ia = next(i for i, x in enumerate(seq) if x["item_id"] == b["is_dup_of"])
        lo = min(ia + gap, len(seq))
        seq.insert(rnd.randint(lo, len(seq)), b)
    items = seq
    pos = defaultdict(list)
    for i, it in enumerate(items):
        pos[it["pair_id"]].append(i)
    mg = min((max(v) - min(v) for v in pos.values() if len(v) > 1), default=None)
    print(f"重复条目的最小间隔 {mg}（目标 {gap}）")

    # 人读的工作表
    L = ["# G3 人工重测工作表", "",
         f"共 {len(items)} 条。每条给你**一条查询**和**一组文档**，",
         "判断：这些文档里有没有**明确写出**回答该查询所需的事实？", "",
         "## 判断规则", "",
         "- 只看文档里写了什么，**不要用你自己的知识补全**。",
         "- 文档只是提到了查询里的实体、或讲了相关话题，但没说出查询问的那个属性 → **0**。",
         "- 能直接从文档中读出完整答案 → **1**。",
         "- 拿不准就填你的最佳判断，并在 `hard` 列填 1。**不要留空。**", "",
         "## 注意", "",
         "- 表里有一些条目是重复的（文档顺序不同）。**这是有意为之，请像第一次见到一样独立判断，",
         "  不要回头找自己之前怎么填的。**",
         "- 逐条顺序做，不要跳着做。", "",
         f"答案填到 `answers.tsv` 的 `label` 与 `hard` 两列。", "",
         "---", ""]
    for it in items:
        L += [f"## {it['item_id']}", "",
              f"**查询**：{it['query']}", "", "**文档**：", ""]
        for i, c in enumerate(it["chunks"][:a.topk], 1):
            txt = (cmap.get(c, "（缺失）") or "")[:a.doc_chars].replace("\n", " ")
            L += [f"{i}. {txt}", ""]
        L += ["", "---", ""]
    (out / "worksheet.md").write_text("\n".join(L), encoding="utf-8")

    with open(out / "answers.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["item_id", "label", "hard", "note"])
        for it in items:
            w.writerow([it["item_id"], "", "", ""])
    wjson(out / "key.json", {"items": items, "plan": plan, "seed": a.seed,
                             "n_unique": len(picked), "n_dup": len(dups)})

    n_dup_actual = sum(1 for it in items if it["is_dup_of"])
    print(f"\n工作表 {out/'worksheet.md'}（{len(items)} 条，其中重复 {n_dup_actual} 对）")
    print(f"答案表 {out/'answers.tsv'}   答案键 {out/'key.json'}（标完前别看）")
    print(f"预计耗时 {len(items)*2//60}–{len(items)*4//60} 小时")
    return 0


# ==============================================================================
#  score
# ==============================================================================

def cmd_score(a) -> int:
    out = Path(a.out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    items = {it["item_id"]: it for it in key["items"]}
    lab, hard = {}, {}
    with open(out / "answers.tsv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            v = (r.get("label") or "").strip()
            if v in ("0", "1"):
                lab[r["item_id"].strip()] = int(v)
                hard[r["item_id"].strip()] = 1 if (r.get("hard") or "").strip() == "1" else 0
    print(f"已标注 {len(lab)}/{len(items)}")
    if len(lab) < len(items):
        miss = [i for i in items if i not in lab]
        print(f"  未标注：{miss[:10]}{' …' if len(miss) > 10 else ''}")

    res = {"n_labeled": len(lab)}

    # ---------- 1) 人的重测一致性（重复条目）----------
    pairs = [(it["is_dup_of"], i) for i, it in items.items() if it["is_dup_of"]]
    both = [(x, y) for x, y in pairs if x in lab and y in lab]
    if both:
        h1 = [lab[x] for x, y in both]
        h2 = [lab[y] for x, y in both]
        st = agree(h1, h2)
        bins = [items[y]["bin"] for x, y in both]
        st["by_bin"] = {str(b): agree([lab[x] for (x, y), bb in zip(both, bins) if bb == b],
                                      [lab[y] for (x, y), bb in zip(both, bins) if bb == b])
                        for b in sorted(set(bins))}
        res["human_retest"] = st
        nflip = int(round(st["flip_rate"] * st["n"]))
        print(f"\n【人的重测一致性】{st['n']} 对重复条目")
        print(f"  翻转 {nflip}/{st['n']} = {st['flip_rate']:.1%}   "
              f"κ={st['kappa'] if st['kappa'] is None else round(st['kappa'],3)}  "
              f"PABAK={st['pabak']:.3f}  AC1={st['ac1'] if st['ac1'] is None else round(st['ac1'],3)}")
        for jf in (a.judge_flip,):
            p = binom_tail(nflip, st["n"], jf)
            res["binom_vs_judge"] = {"judge_flip_rate": jf, "human_flips": nflip,
                                     "n": st["n"], "p_value": p}
            print(f"  若人的真实翻转率与判官相同（{jf:.1%}），观察到 ≤{nflip} 次翻转的概率 = {p:.4f}")
            print(f"  -> {'可以拒绝「人与判官一样不稳」' if p < 0.05 else '样本不足以区分人与判官'}")

    # ---------- 2) 人 vs 判官（只用原序条目）----------
    orig = [i for i, it in items.items() if it["presentation"] == "orig" and i in lab]
    if orig:
        hv = [lab[i] for i in orig]
        jv = [items[i]["judge_verdict"] for i in orig]
        res["human_vs_judge_orig"] = agree(hv, jv)
        s = res["human_vs_judge_orig"]
        print(f"\n【人 vs 判官（原序）】n={s['n']}  一致率 {s['po']:.1%}  "
              f"κ={s['kappa'] if s['kappa'] is None else round(s['kappa'],3)}  "
              f"PABAK={s['pabak']:.3f}")
        by = {}
        for b in sorted({items[i]["bin"] for i in orig}):
            sub = [i for i in orig if items[i]["bin"] == b]
            by[str(b)] = agree([lab[i] for i in sub], [items[i]["judge_verdict"] for i in sub])
        res["human_vs_judge_by_bin"] = by
        print(f"  {'bin':>4}{'n':>5}{'一致率':>9}{'PABAK':>9}")
        for b, s2 in by.items():
            print(f"  {b:>4}{s2['n']:>5}{s2['po']:>9.1%}{s2['pabak']:>9.3f}")

    # ---------- 3) 判官翻转过的条目上，人站哪一边 ----------
    fl = [i for i in orig if items[i]["flipped"]]
    cs = [i for i in orig if not items[i]["flipped"]]
    if fl:
        side_o = sum(1 for i in fl if lab[i] == items[i]["judge_verdict"])
        lo, hi = wilson(side_o, len(fl))
        res["on_flipped_items"] = {"n": len(fl), "agree_with_original_verdict": side_o,
                                   "frac": side_o / len(fl), "ci": [lo, hi],
                                   "hard_rate": float(np.mean([hard[i] for i in fl]))}
        res["on_consistent_items"] = {
            "n": len(cs),
            "agree": sum(1 for i in cs if lab[i] == items[i]["judge_verdict"]),
            "frac": (sum(1 for i in cs if lab[i] == items[i]["judge_verdict"]) / len(cs)) if cs else None,
            "hard_rate": float(np.mean([hard[i] for i in cs])) if cs else None}
        print(f"\n【判官翻转过的条目】n={len(fl)}")
        print(f"  人与判官原序判定一致 {side_o}/{len(fl)} = {side_o/len(fl):.1%} "
              f"CI[{lo:.2f},{hi:.2f}]")
        print(f"  人标为「难判」的比例：翻转条目 {res['on_flipped_items']['hard_rate']:.1%}"
              f"  vs 一致条目 {res['on_consistent_items']['hard_rate']:.1%}"
              if cs else "")

    # ---------- 结论 ----------
    verdict = []
    hr = res.get("human_retest")
    if hr:
        if hr["flip_rate"] <= 0.15 and res.get("binom_vs_judge", {}).get("p_value", 1) < 0.05:
            verdict.append("人在同一区域判得稳 -> 不稳定来自判官，是仪器问题")
        elif hr["flip_rate"] >= 0.3:
            verdict.append("人也来回摇摆 -> 该区域的可替代性本身缺乏确定答案")
        else:
            verdict.append("人的稳定性介于两者之间，样本量不足以下强结论")
    onf = res.get("on_flipped_items")
    onc = res.get("on_consistent_items")
    if onf and onc and onc.get("hard_rate") is not None:
        if onf["hard_rate"] > onc["hard_rate"] + 0.2:
            verdict.append("判官翻转的条目，人也明显更常标为难判 -> 存在内在歧义成分")
        else:
            verdict.append("判官翻转的条目，人并不觉得更难 -> 翻转不是由内在歧义驱动")
    res["verdict"] = verdict
    print("\n★ 结论")
    for v in verdict:
        print(f"  - {v}")
    wjson(out / "g3_manual_result.json", res)
    print(f"\n-> {out/'g3_manual_result.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="G3 人工版")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--run", default="runs/main_v10")
    m.add_argument("--out", default="runs/phase2/G3m")
    m.add_argument("--plan", default="8:12,7:8,9:4,3:4,2:4",
                   help="每层抽多少条，格式 bin:n，逗号分隔（翻转/一致各半）")
    m.add_argument("--n-dup", type=int, default=12, help="其中多少条重复出现")
    m.add_argument("--gap", type=int, default=12, help="同一对两次出现至少相隔多少条")
    m.add_argument("--topk", type=int, default=10)
    m.add_argument("--doc-chars", type=int, default=700)
    m.add_argument("--seed", type=int, default=20260923)
    m.set_defaults(func=cmd_make)
    s = sub.add_parser("score")
    s.add_argument("--out", default="runs/phase2/G3m")
    s.add_argument("--judge-flip", type=float, default=0.466,
                   help="对比用的判官翻转率（默认 bin8 的 46.6%）")
    s.set_defaults(func=cmd_score)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
