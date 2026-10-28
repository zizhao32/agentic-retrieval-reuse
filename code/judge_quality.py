#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
judge_quality.py — 不花卡。两件事：

【A】三个臂各自对 300 条人工标签的准确率。
    s5/gold_pairs.jsonl 里的 human_label 是唯一的外部真值。τ* 在 k1 下是
    0.94、在 think1 下是 0.8835，差 0.06，所以「哪个臂更准」直接决定主线数字。
    报总体 / 逐档 / 高相似区（决定阈值的那一片）的准确率、灵敏度、特异度，
    都带 Wilson 区间，再给对人的 kappa。

【B】「拆多细」这个机制的检验。
    judge_once 返回的 required_facts 决定覆盖判定有多严：事实列表越长越难
    全覆盖。facts_ij_n 在两个 run 里都是当次现算的（J2 覆写过），所以可以直接比。
    预测：在 think1 系统性判 0 的那批「同一个请求」对上，think1 拆出的事实数
    显著多于 k1，也多于它自己在判 1 的那批上拆出的 —— 若成立，判官的严格度
    是由问题分解粒度这个自由参数定的。

用法（在 projB 下）:

    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python judge_quality.py main_v10 main_v11_k3 main_v12_think \
        --gold main_v10 > quality.txt 2>&1
    cat quality.txt

--gold 指哪个 run 的 s5/gold_pairs.jsonl（三个 run 的 s5 都软链到同一份，
默认取第一个 run 的）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict


def runs_dir():
    cwd = os.getcwd()
    for p in (os.path.join(cwd, "runs"), os.path.join(os.path.dirname(cwd), "runs")):
        if os.path.isdir(p):
            return p
    sys.exit("找不到 runs/，请在 projB 目录下运行")


def wilson(k, n, z=1.96):
    """二项比例的 Wilson 区间。k 次成功 / n 次试验。"""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt_ci(k, n):
    if n == 0:
        return "      n=0        "
    p, lo, hi = wilson(k, n)
    return "%.3f [%.3f,%.3f]" % (p, lo, hi)


def key(r):
    return r.get("pair_id") or "%s|%s" % (r.get("qid_i"), r.get("qid_j"))


def load_l1(path, want=None):
    """want 是 pair_id 集合；给了就只留这些，省内存。"""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = key(r)
            if want is not None and k not in want:
                continue
            out[k] = r
    return out


def load_all(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[key(r)] = r
    return out


# ----------------------------------------------------------------- A 对人

def part_a(rd, names, gold_run):
    gp = os.path.join(rd, gold_run, "stages", "s5", "gold_pairs.jsonl")
    if not os.path.isfile(gp):
        print("!! 没有 %s，跳过 A" % gp)
        return
    gold = {}
    with open(gp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("human_label") is None:
                continue
            gold[key(r)] = r
    print("=" * 110)
    print("【A】对 %d 条人工标签的准确率" % len(gold))
    print("     human_label=1 的 %d 条，=0 的 %d 条"
          % (sum(1 for r in gold.values() if r["human_label"]),
             sum(1 for r in gold.values() if not r["human_label"])))
    print("=" * 110)

    rows = {}
    for n in names:
        p = os.path.join(rd, n, "stages", "s6", "l1.jsonl")
        if not os.path.isfile(p):
            continue
        sys.stderr.write("A: 扫 %s …\n" % n)
        rows[n] = load_l1(p, set(gold))

    hdr = ("%-16s %7s %7s  %-20s %-20s %-20s %7s"
           % ("run", "可比", "缺", "准确率", "灵敏度(真1判1)",
              "特异度(真0判0)", "kappa"))
    print(hdr)
    for n, m in rows.items():
        tp = fp = tn = fn = miss = 0
        for k, g in gold.items():
            r = m.get(k)
            s = None if r is None else r.get("substitutable")
            if s is None:
                miss += 1
                continue
            h = int(bool(g["human_label"]))
            s = int(bool(s))
            if h and s:
                tp += 1
            elif h and not s:
                fn += 1
            elif not h and s:
                fp += 1
            else:
                tn += 1
        tot = tp + fp + tn + fn
        if not tot:
            print("%-16s  (没有可比的对)" % n)
            continue
        p0 = (tp + tn) / tot
        ph1 = (tp + fn) / tot
        ps1 = (tp + fp) / tot
        pe = ph1 * ps1 + (1 - ph1) * (1 - ps1)
        kap = (p0 - pe) / (1 - pe) if pe < 1 else float("nan")
        print("%-16s %7d %7d  %-20s %-20s %-20s %7.3f"
              % (n, tot, miss, fmt_ci(tp + tn, tot), fmt_ci(tp, tp + fn),
                 fmt_ci(tn, tn + fp), kap))
    print()

    # 逐档 + 高相似区
    for label, pick in (("逐档准确率", None),
                        ("高相似区 bin>=7（决定阈值的那一片）",
                         lambda g: (g.get("bin") or 0) >= 7),
                        ("cos>=0.95", lambda g: float(g.get("cos") or 0) >= 0.95),
                        ("逐字相同", lambda g: bool(g.get("same_text")))):
        if pick is None:
            allbins = sorted({g.get("bin") for g in gold.values()
                              if g.get("bin") is not None})
            print("-" * 110)
            print(label)
            print("-" * 110)
            print("%-16s" % "run" + "".join("%11s" % ("bin%d" % b) for b in allbins))
            print("%-16s" % "  (n=)" + "".join(
                "%11d" % sum(1 for g in gold.values() if g.get("bin") == b)
                for b in allbins))
            for n, m in rows.items():
                line = "%-16s" % n
                for b in allbins:
                    ks = [k for k, g in gold.items() if g.get("bin") == b]
                    ok = tot = 0
                    for k in ks:
                        r = m.get(k)
                        s = None if r is None else r.get("substitutable")
                        if s is None:
                            continue
                        tot += 1
                        ok += int(int(bool(s)) == int(bool(gold[k]["human_label"])))
                    line += "%11s" % ("%.3f" % (ok / tot) if tot else "-")
                print(line)
            print()
            continue
        ks = [k for k, g in gold.items() if pick(g)]
        print("-" * 110)
        print("%s  (n=%d)" % (label, len(ks)))
        print("-" * 110)
        print("%-16s %7s  %-24s %7s" % ("run", "可比", "准确率", "判1比例"))
        for n, m in rows.items():
            ok = tot = one = 0
            for k in ks:
                r = m.get(k)
                s = None if r is None else r.get("substitutable")
                if s is None:
                    continue
                tot += 1
                one += int(bool(s))
                ok += int(int(bool(s)) == int(bool(gold[k]["human_label"])))
            print("%-16s %7d  %-24s %7s"
                  % (n, tot, fmt_ci(ok, tot),
                     "%.3f" % (one / tot) if tot else "-"))
        print()


# ------------------------------------------------------- B 事实分解粒度

def part_b(rd, names):
    print("=" * 110)
    print("【B】required_facts 的条数：判官把问题拆多细（main_v10 用 len(facts_ij)，重判过的 run 用 facts_ij_n）")
    print("=" * 110)
    data = {}
    for n in names:
        p = os.path.join(rd, n, "stages", "s6", "l1.jsonl")
        if not os.path.isfile(p):
            continue
        sys.stderr.write("B: 扫 %s …\n" % n)
        data[n] = load_all(p)
    if len(data) < 2:
        print("至少要两个 run 才能比")
        return

    ref = names[0] if names[0] in data else list(data)[0]
    others = [n for n in data if n != ref]

    def nfacts(r):
        """facts_ij_n 是 fix_judge 才写的字段，main_v10 没有；
        退回数 facts_ij 列表的长度。注意 main_v12_think 里的 facts_ij 是
        从来源 run 拷来的旧值（J2 当时没覆写），所以对重判过的 run 必须
        优先用 facts_ij_n，只有它缺失时才退回列表长度。"""
        v = r.get("facts_ij_n")
        if v is not None:
            return int(v)
        fl = r.get("facts_ij")
        if isinstance(fl, list):
            return len(fl)
        inh = r.get("_inherited") or {}
        fl = inh.get("facts_ij")
        return len(fl) if isinstance(fl, list) else None

    def forced(r):
        return (r.get("eligible") and r.get("same_text")
                and float(r.get("jaccard") or 0) >= 0.999
                and float(r.get("rbo") or 0) >= 0.999)

    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "   n=0"
        vals = sorted(vals)
        return ("n=%d 均值 %.3f 中位 %.0f 最大 %d"
                % (len(vals), sum(vals) / len(vals),
                   vals[len(vals) // 2], vals[-1]))

    print("切片定义：forced = 合格 + 逐字相同 + 结果集与排序都相同（答案被逻辑决定）")
    print()
    for n in [ref] + others:
        m = data[n]
        allv = [nfacts(r) for r in m.values() if r.get("eligible")]
        fv = [nfacts(r) for r in m.values() if forced(r)]
        print("%-16s 全部合格对   %s" % (n, stat(allv)))
        print("%-16s forced 切片  %s" % ("", stat(fv)))
    print()

    for n in others:
        m, base = data[n], data[ref]
        shared = [k for k in m if k in base and forced(m[k])]
        zero = [k for k in shared
                if m[k].get("cov_ij") == 0 and m[k].get("cov_ji") == 0]
        one = [k for k in shared
               if m[k].get("cov_ij") == 1 and m[k].get("cov_ji") == 1]
        print("-" * 110)
        print("%s 对 %s：forced 切片 %d 对，其中两向都判0 %d、都判1 %d"
              % (n, ref, len(shared), len(zero), len(one)))
        print("-" * 110)
        print("%-34s %-44s %s" % ("", "%s 的事实条数" % n,
                                  "%s 的事实条数" % ref))
        for nm, ks in (("两向都判 0（系统性分歧）", zero),
                       ("两向都判 1（一致）", one)):
            print("%-34s %-44s %s"
                  % (nm, stat([nfacts(m[k]) for k in ks]),
                     stat([nfacts(base[k]) for k in ks])))
        if zero and one:
            mz = [nfacts(m[k]) for k in zero if nfacts(m[k]) is not None]
            mo = [nfacts(m[k]) for k in one if nfacts(m[k]) is not None]
            if mz and mo:
                print()
                print("  >>> %s 在判0那批上拆出 %.3f 条事实，在判1那批上 %.3f 条，"
                      "差 %+.3f" % (n, sum(mz) / len(mz), sum(mo) / len(mo),
                                    sum(mz) / len(mz) - sum(mo) / len(mo)))
                print("      差值明显为正 -> 「拆得更细所以更难覆盖」的机制成立；")
                print("      差值约为 0   -> 严格度不是由分解粒度带来的，另找解释。")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gold", default=None, help="取哪个 run 的 gold_pairs.jsonl")
    ap.add_argument("--skip-b", action="store_true")
    a = ap.parse_args()
    rd = runs_dir()
    print("runs = %s" % rd)
    print()
    part_a(rd, a.runs, a.gold or a.runs[0])
    if not a.skip_b:
        part_b(rd, a.runs)


if __name__ == "__main__":
    main()
