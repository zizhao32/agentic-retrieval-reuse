#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dump_cases.py — 把判官分歧的案例连原文一起打出来，供人工阅读。

为什么要人读：facts_ij_n 的配对比较已经否掉了「拆得更细所以更严」
（think1 在 377 对系统性分歧上拆出 1.894 条，k1 拆出 1.976 条，更少反而更严）。
条数解释不了方向，只能看内容。

打的是「逻辑强制」切片里两个臂分歧的对：逐字相同的 query、jaccard 和 rbo
都是 1，所以 cov_ij 问的就是「query 自己的检索结果能不能覆盖自己需要的事实」。
一边说能、一边说不能，读原文就能判断谁对。

每个案例给：query 原文、检索到的段落（截断）、k1 抽出的 required_facts、
两个臂的 verdict 与 cov。注意 **think1 的 facts_ij 列表是从来源 run 拷来的旧值**
（当时的 J2 没覆写这个字段，只覆写了条数），所以事实列表只有 k1 这一份，
脚本不会假装有两份。

用法（在 projB 下）:

    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python dump_cases.py \
        --base main_v10 --arm main_v12_think \
        --n-disagree 20 --n-agree 8 --passages 5 --chars 420 \
        > cases.txt 2>&1
    wc -c cases.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path


def runs_dir():
    cwd = Path(os.getcwd())
    for p in (cwd / "runs", cwd.parent / "runs"):
        if p.is_dir():
            return p
    sys.exit("找不到 runs/，请在 projB 目录下运行")


def key(r):
    return r.get("pair_id") or "%s|%s" % (r.get("qid_i"), r.get("qid_j"))


def load_l1(p):
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[key(r)] = r
    return out


def forced(r):
    return (r.get("eligible") and r.get("same_text")
            and float(r.get("jaccard") or 0) >= 0.999
            and float(r.get("rbo") or 0) >= 0.999)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="main_v10", help="原始臂的 run")
    ap.add_argument("--arm", default="main_v12_think", help="对比臂的 run")
    ap.add_argument("--n-disagree", type=int, default=20)
    ap.add_argument("--n-agree", type=int, default=8)
    ap.add_argument("--passages", type=int, default=5, help="每个案例打几段")
    ap.add_argument("--chars", type=int, default=420, help="每段截断到多少字符")
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--no-dedup", dest="dedup_query", action="store_false",
                    help="不按查询文本去重（默认去重：同一条查询读五遍没有信息）")
    ap.set_defaults(dedup_query=True)
    a = ap.parse_args()

    rd = runs_dir()
    bp = rd / a.base / "stages" / "s6" / "l1.jsonl"
    ap_ = rd / a.arm / "stages" / "s6" / "l1.jsonl"
    for p in (bp, ap_):
        if not p.is_file():
            sys.exit("没有 %s" % p)

    sys.stderr.write("读 l1 …\n")
    base, arm = load_l1(bp), load_l1(ap_)

    shared = [k for k in arm if k in base and forced(arm[k])]
    dis = [k for k in shared
           if arm[k].get("cov_ij") == 0 and arm[k].get("cov_ji") == 0
           and base[k].get("cov_ij") == 1 and base[k].get("cov_ji") == 1]
    agr = [k for k in shared
           if arm[k].get("cov_ij") == 1 and base[k].get("cov_ij") == 1]
    # 先读 calls：按查询去重需要 query 文本
    sys.stderr.write("读 calls …\n")
    need_q = {arm[k]["qid_i"] for k in shared} | {arm[k]["qid_j"] for k in shared}
    calls = {}
    cp = rd / a.arm / "stages" / "s2" / "calls.jsonl"
    with open(cp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("qid") in need_q:
                calls[c["qid"]] = c

    def qtext(k):
        return (calls.get(arm[k]["qid_i"], {}).get("query") or "").strip()

    # 分歧对里重复查询很多（实测 18 个案例只对应 7 个不同查询），
    # 同一条查询读五遍不增加信息。默认按查询文本去重再抽样。
    def dedup(ks):
        seen, out = set(), []
        for k in ks:
            t = qtext(k).lower()
            if t and t in seen:
                continue
            seen.add(t)
            out.append(k)
        return out

    ndq_dis = len({qtext(k).lower() for k in dis if qtext(k)})
    ndq_agr = len({qtext(k).lower() for k in agr if qtext(k)})
    rng = random.Random(a.seed)
    pool_dis = dedup(rng.sample(dis, len(dis))) if a.dedup_query else dis
    pool_agr = dedup(rng.sample(agr, len(agr))) if a.dedup_query else agr
    sel = ([("分歧", k) for k in rng.sample(pool_dis,
                                            min(a.n_disagree, len(pool_dis)))]
           + [("一致", k) for k in rng.sample(pool_agr,
                                              min(a.n_agree, len(pool_agr)))])

    print("=" * 100)
    print("逻辑强制切片 %d 对；%s 判0而 %s 判1的 %d 对；两者都判1的 %d 对"
          % (len(shared), a.arm, a.base, len(dis), len(agr)))
    print("不同查询文本：分歧 %d 条（%d 对）、一致 %d 条（%d 对）"
          % (ndq_dis, len(dis), ndq_agr, len(agr)))
    print("  —— 分歧对平均每条查询重复 %.1f 次；表面的对数被重复查询放大了"
          % (len(dis) / ndq_dis if ndq_dis else 0))
    print("抽样：分歧 %d + 一致 %d%s"
          % (min(a.n_disagree, len(pool_dis)), min(a.n_agree, len(pool_agr)),
             "（已按查询去重）" if a.dedup_query else "（未去重）"))
    print("=" * 100)
    print()

    need_c = set()
    for _, k in sel:
        for q in (arm[k]["qid_i"], arm[k]["qid_j"]):
            for x in (calls.get(q, {}).get("retrieved_chunks") or [])[:a.passages]:
                need_c.add(x)
    sys.stderr.write("读 chunks（只取需要的 %d 段）…\n" % len(need_c))
    chunks = {}
    kp = rd / a.arm / "stages" / "s1" / "chunks.jsonl"
    with open(kp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c.get("chunkid") in need_c:
                chunks[c["chunkid"]] = c.get("text") or ""
                if len(chunks) == len(need_c):
                    break

    for idx, (grp, k) in enumerate(sel, 1):
        rb, ra = base[k], arm[k]
        ci = calls.get(ra["qid_i"], {})
        cj = calls.get(ra["qid_j"], {})
        print("#" * 100)
        print("案例 %d / %d   [%s]   pair=%s" % (idx, len(sel), grp, k))
        print("#" * 100)
        print("query_i : %s" % (ci.get("query") or "(缺)"))
        print("query_j : %s" % (cj.get("query") or "(缺)"))
        print("cos=%.6f  jaccard=%.3f  rbo=%.3f  bin=%s  w=%.2f  source=%s"
              % (float(ra.get("cos") or 0), float(ra.get("jaccard") or 0),
                 float(ra.get("rbo") or 0), ra.get("bin"),
                 float(ra.get("w") or 0), ci.get("source") or ci.get("dataset")))
        print()
        print("--- %s（原始臂）---" % a.base)
        print("  cov_ij=%s cov_ji=%s substitutable=%s verdict_ij=%s verdict_ji=%s"
              % (rb.get("cov_ij"), rb.get("cov_ji"), rb.get("substitutable"),
                 rb.get("verdict_ij"), rb.get("verdict_ji")))
        fl = rb.get("facts_ij")
        print("  抽出的 required_facts（%d 条）：" % (len(fl) if isinstance(fl, list) else 0))
        if isinstance(fl, list):
            for t in fl:
                print("    · %s" % t)
        print()
        print("--- %s（对比臂）---" % a.arm)
        inh = ra.get("_inherited") or {}
        print("  cov_ij=%s cov_ji=%s substitutable=%s facts_ij_n=%s votes_ij=%s votes_ji=%s"
              % (ra.get("cov_ij"), ra.get("cov_ji"), ra.get("substitutable"),
                 ra.get("facts_ij_n"), ra.get("votes_ij"), ra.get("votes_ji")))
        print("  （这个 run 的 facts_ij 列表是来源 run 的旧值，只有条数是本臂现算的）")
        if inh:
            print("  _inherited: %s" % json.dumps(
                {kk: vv for kk, vv in inh.items() if kk != "raw"},
                ensure_ascii=False)[:300])
        print()
        ids = (cj.get("retrieved_chunks") or [])[:a.passages]
        print("--- 判定 cov_ij 时给判官看的段落（query_j 的检索结果，共 %d 段，打前 %d）---"
              % (len(cj.get("retrieved_chunks") or []), len(ids)))
        for n_, x in enumerate(ids, 1):
            t = (chunks.get(x) or "(没取到正文)").replace("\n", " ")
            print("  [%d] %s" % (n_, t[:a.chars]))
        print()

    print("=" * 100)
    print("读的时候要判断的就一件事：这些段落到底有没有覆盖上面列出的 required_facts。")
    print("  覆盖了 -> k1 对，think1 在这批对上过严；")
    print("  没覆盖 -> think1 对，k1 的合格性筛选过宽，整个条件可替换率的分母被污染。")
    print("=" * 100)


if __name__ == "__main__":
    main()
