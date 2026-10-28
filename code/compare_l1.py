#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
compare_l1.py — 把若干个 run 的 stages/s6/l1.jsonl 并排比，直接从原始判定记录
现算，不读任何 qc.json（main_v12_think 的 qc.json 是从 main_v10 拷来的，不可信）。

用法（在 projB 下）:

    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python compare_l1.py main_v10 main_v11_k3 main_v12_think

输出三块:
  1) 每个 run 一行的总览（arm / 条数 / 可替换率 未加权+代价加权 / 恒等带 / 不对称率）
  2) 恒等带与近恒等带的逐档对比 —— k3 那次就是在这里塌的
  3) bin 0..9 的逐档表，未加权与代价加权各一张

只读。l1.jsonl 流式扫，不进内存。
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

RUNS_DEFAULT = ["main_v10", "main_v11_k3", "main_v12_think"]


def find_runs_dir():
    cwd = os.getcwd()
    for p in (os.path.join(cwd, "runs"),
              os.path.join(os.path.dirname(cwd), "runs")):
        if os.path.isdir(p):
            return p
    sys.exit("找不到 runs/，请在 projB 目录下运行")


class Acc:
    """一组记录的累加器：条数、未加权和、代价加权和。"""

    def __init__(self):
        self.n = 0
        self.k = 0          # substitutable == 1 的条数
        self.w = 0.0        # 权重和
        self.wk = 0.0       # substitutable 的加权和

    def add(self, sub, w):
        self.n += 1
        self.k += 1 if sub else 0
        self.w += w
        self.wk += w if sub else 0.0

    @property
    def uw(self):
        return self.k / self.n if self.n else float("nan")

    @property
    def wt(self):
        return self.wk / self.w if self.w else float("nan")


def scan(path):
    """返回 (总览dict, 分档dict)。"""
    tot = Acc()
    bins = defaultdict(Acc)
    same_text = Acc()
    diff_text = Acc()
    cos_hi = Acc()          # cos >= 0.99 且非逐字相同
    arms = defaultdict(int)
    asym = 0                # cov_ij != cov_ji
    both = 0
    trunc = 0
    notok = 0
    bad = 0
    verdicts = defaultdict(int)
    toks = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                bad += 1
                continue
            if not r.get("eligible"):
                continue
            sub = bool(r.get("substitutable"))
            w = float(r.get("w") or 0.0)
            tot.add(sub, w)
            bins[r.get("bin")].add(sub, w)
            if r.get("same_text"):
                same_text.add(sub, w)
            else:
                diff_text.add(sub, w)
                if float(r.get("cos") or 0) >= 0.99:
                    cos_hi.add(sub, w)
            ja = r.get("judge_aggregation") or {}
            arms[ja.get("arm", "<无>")] += 1
            ci, cj = r.get("cov_ij"), r.get("cov_ji")
            if ci is not None and cj is not None:
                both += 1
                if ci != cj:
                    asym += 1
            if r.get("truncated"):
                trunc += 1
            if not r.get("ok", True):
                notok += 1
            verdicts[r.get("verdict_ij")] += 1
            toks += int(r.get("tokens") or 0)

    return {
        "tot": tot, "bins": bins, "same_text": same_text,
        "diff_text": diff_text, "cos_hi": cos_hi, "arms": dict(arms),
        "asym": asym / both if both else float("nan"),
        "trunc": trunc, "notok": notok, "bad": bad,
        "verdicts": dict(verdicts), "tokens": toks,
    }


def pct(x):
    return "  n/a " if x != x else "%6.3f" % x


def main():
    names = sys.argv[1:] or RUNS_DEFAULT
    runs_dir = find_runs_dir()
    data = {}
    for n in names:
        p = os.path.join(runs_dir, n, "stages", "s6", "l1.jsonl")
        if not os.path.isfile(p):
            print("!! 跳过 %s：没有 %s" % (n, p))
            continue
        if os.path.getsize(p) == 0:
            print("!! 跳过 %s：l1.jsonl 是空的" % n)
            continue
        sys.stderr.write("扫 %s (%.1f MB) …\n"
                         % (n, os.path.getsize(p) / 1e6))
        data[n] = scan(p)
    if not data:
        sys.exit("一个都没扫到")

    W = max(len(n) for n in data) + 2

    print()
    print("=" * 100)
    print("总览（只统计 eligible==1 的对；sub_uw 未加权，sub_w 代价加权）")
    print("=" * 100)
    print("%-*s %-10s %7s %8s %8s %8s %8s %7s %7s"
          % (W, "run", "arm", "n", "sub_uw", "sub_w", "asym", "trunc",
             "notok", "parse"))
    for n, d in data.items():
        arm = max(d["arms"], key=d["arms"].get) if d["arms"] else "?"
        if len(d["arms"]) > 1:
            arm += "(+%d)" % (len(d["arms"]) - 1)
        print("%-*s %-10s %7d %8s %8s %8s %7d %7d %7d"
              % (W, n, arm, d["tot"].n, pct(d["tot"].uw), pct(d["tot"].wt),
                 pct(d["asym"]), d["trunc"], d["notok"], d["bad"]))

    print()
    print("=" * 100)
    print("恒等带 —— k3 那次就是在这里塌的（same_text=True 的对，应当接近 1.000）")
    print("=" * 100)
    print("%-*s %18s %18s %18s"
          % (W, "run", "逐字相同 sub_uw", "cos>=.99 非逐字", "其余 sub_uw"))
    for n, d in data.items():
        print("%-*s %12s (n=%5d) %10s (n=%5d) %10s (n=%5d)"
              % (W, n,
                 pct(d["same_text"].uw), d["same_text"].n,
                 pct(d["cos_hi"].uw), d["cos_hi"].n,
                 pct(d["diff_text"].uw), d["diff_text"].n))

    allbins = sorted({b for d in data.values() for b in d["bins"]
                      if b is not None})
    for which, label in (("uw", "未加权"), ("wt", "代价加权")):
        print()
        print("=" * 100)
        print("逐档可替换率（%s）" % label)
        print("=" * 100)
        hdr = "%-*s" % (W, "run") + "".join("%9s" % ("bin%d" % b)
                                            for b in allbins)
        print(hdr)
        for n, d in data.items():
            row = "%-*s" % (W, n)
            for b in allbins:
                a = d["bins"].get(b)
                row += "%9s" % (pct(getattr(a, which)) if a and a.n
                                else "   -")
            print(row)
        print("%-*s" % (W, "  (n=)")
              + "".join("%9d" % (data[names[0]]["bins"][b].n
                                 if names[0] in data
                                 and b in data[names[0]]["bins"] else 0)
                        for b in allbins))

    print()
    print("=" * 100)
    print("verdict_ij 分布 / 总 token")
    print("=" * 100)
    for n, d in data.items():
        v = ", ".join("%s=%s" % (k, c) for k, c in
                      sorted(d["verdicts"].items(), key=lambda x: -x[1]))
        print("%-*s %s" % (W, n, v))
        print("%-*s   tokens=%d  arms=%s" % (W, "", d["tokens"], d["arms"]))


if __name__ == "__main__":
    main()
