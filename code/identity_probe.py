#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
identity_probe.py — 不花卡，用现有 l1.jsonl 区分两个假设。

背景：对一对逐字相同的 query，若两边的检索结果集也相同（jaccard=1 且 rbo=1），
那么 cov_ij 和 cov_ji 是**字面上同一个请求**：同样的 query 文本、同样的段落、
同样的顺序。所以

    P(cov_ij != cov_ji)  在这批对上 = 判官单次调用的不确定性，
                                     而不是任何实质分歧。

两个假设：
  H1 臂间系统性分歧 —— think1 对自覆盖的判断确实不同于 k1，确定性的。
     预测：本批对上 cov_ij == cov_ji 几乎总成立（方向一致性 ~1.0），
           翻转集中在「两个方向都翻」。
  H2 think1 自身不确定 —— 思考链放大采样方差。
     预测：方向一致性明显掉下来，出现大量 (1,0)/(0,1)。

若方向一致性为 c，在独立同分布的粗模型下单次翻转概率 p 满足
2p(1-p) = 1-c，脚本会把 p 解出来，并给出「两个方向都判 0」的期望占比，
和实测对比。

用法（在 projB 下）:

    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python identity_probe.py main_v10 main_v11_k3 main_v12_think \
        --config $R/config.json > probe.txt 2>&1
    cat probe.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter


def runs_dir():
    cwd = os.getcwd()
    for p in (os.path.join(cwd, "runs"), os.path.join(os.path.dirname(cwd), "runs")):
        if os.path.isdir(p):
            return p
    sys.exit("找不到 runs/，请在 projB 目录下运行")


def slice_name(r):
    """返回该记录属于哪个切片，None 表示不统计。"""
    jac = float(r.get("jaccard") or 0.0)
    rbo = float(r.get("rbo") or 0.0)
    cos = float(r.get("cos") or 0.0)
    if r.get("same_text"):
        if jac >= 0.999 and rbo >= 0.999:
            return "逐字相同 + 结果集与排序都相同（同一个请求）"
        if jac >= 0.999:
            return "逐字相同 + 结果集相同但排序不同"
        return "逐字相同 + 结果集不同"
    if cos >= 0.99:
        return "非逐字 cos>=0.99"
    if cos >= 0.95:
        return "非逐字 0.95<=cos<0.99"
    return None


ORDER = ["逐字相同 + 结果集与排序都相同（同一个请求）",
         "逐字相同 + 结果集相同但排序不同",
         "逐字相同 + 结果集不同",
         "非逐字 cos>=0.99",
         "非逐字 0.95<=cos<0.99"]


def scan(path):
    acc = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("eligible"):
                continue
            ci, cj = r.get("cov_ij"), r.get("cov_ji")
            if ci is None or cj is None:
                continue
            s = slice_name(r)
            if s is None:
                continue
            acc.setdefault(s, Counter())[(int(bool(ci)), int(bool(cj)))] += 1
    return acc


def solve_p(c):
    """由 2p(1-p) = 1-c 解较小的根 p。c 是方向一致性。"""
    d = 1.0 - c
    disc = 1.0 - 2.0 * d
    if disc < 0:
        return None          # 1-c > 0.5，模型解释不了
    return (1.0 - math.sqrt(disc)) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--config", default=None,
                    help="config.json，用来打印判官的温度等设置")
    a = ap.parse_args()
    rd = runs_dir()

    if a.config and os.path.isfile(a.config):
        try:
            cfg = json.load(open(a.config, encoding="utf-8"))
            keys = [k for k in cfg
                    if "judge" in k.lower() and
                    any(t in k.lower() for t in
                        ("temp", "top", "seed", "max_token", "retr"))]
            print("config.json 里和判官采样有关的设置：")
            for k in sorted(keys):
                print("  %-34s %s" % (k, cfg[k]))
            if not keys:
                print("  (没找到 judge_temperature 之类的键，手工确认一下)")
            print()
        except Exception as e:
            print("读 config 失败：%s\n" % e)

    data = {}
    for n in a.runs:
        p = os.path.join(rd, n, "stages", "s6", "l1.jsonl")
        if not os.path.isfile(p):
            print("!! 跳过 %s（没有 l1.jsonl）" % n)
            continue
        sys.stderr.write("扫 %s …\n" % n)
        data[n] = scan(p)

    W = max(len(x) for x in ORDER) + 2
    for s in ORDER:
        rows = [(n, d[s]) for n, d in data.items() if s in d and sum(d[s].values())]
        if not rows:
            continue
        print("=" * 108)
        print(s)
        print("=" * 108)
        print("%-16s %7s %7s %7s %7s %7s  %9s %9s  %9s"
              % ("run", "n", "(1,1)", "(1,0)", "(0,1)", "(0,0)",
                 "cov率", "方向一致", "单次翻转p"))
        for n, c in rows:
            tot = sum(c.values())
            n11, n10, n01, n00 = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
            cons = (n11 + n00) / tot
            cov = (2 * n11 + n10 + n01) / (2 * tot)
            p = solve_p(cons)
            print("%-16s %7d %7d %7d %7d %7d  %9.4f %9.4f  %9s"
                  % (n, tot, n11, n10, n01, n00, cov, cons,
                     "   n/a" if p is None else "%9.4f" % p))
        print()

    key = ORDER[0]
    print("=" * 108)
    print("怎么读")
    print("=" * 108)
    print("在「%s」这一片里，cov_ij 和 cov_ji 是字面上同一个请求。" % key)
    print("  · 方向一致 ~1.000  ->  判官是确定性的，(0,0) 那些是真的系统性判 0 ->  H1")
    print("  · 方向一致明显 <1  ->  同一请求两次给出不同答案 -> 不确定性 -> H2")
    print("「单次翻转 p」是在独立同分布粗模型下由 2p(1-p)=1-一致性 解出来的，")
    print("只是个量级参考：真实的翻转大概率不独立（难的对更容易翻），所以这个 p")
    print("是下界性质的估计，不要当成精确值写进论文。")
    print()
    print("若 H2 成立，判决性实验是把同一批对在同一个臂下重判若干次，看翻的是不是")
    print("同一批（fix_judge.py --exp J5）。H2 下每次翻的是不同的对。")


if __name__ == "__main__":
    main()
