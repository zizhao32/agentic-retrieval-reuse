#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
crosstab_l1.py — 两个 run 的 l1.jsonl 按 pair_id 对齐，逐对比判定。

用来回答三件事:
  1) 恒等带里 think1 判 0 的那些对，是不是「逻辑上不可能判 0」的那批
     （jaccard==1 ⇒ j 的结果就是 i 的结果 ⇒ cov_ij 等价于 self_cov_i == 1）；
  2) 两个臂的翻转是对称的还是单向的（0->1 多还是 1->0 多），逐档看；
  3) 两个臂之间的一致性 kappa / PABAK / AC1 —— 这是一个有出处的
     跨配置对照数，可以顶掉被删掉的那个来源不明的 kappa=0.824。

用法（在 projB 下）:

    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python crosstab_l1.py main_v10 main_v12_think

    # 把 133 条没重判、原样带过来的对排除掉（默认就排除，--keep-stale 可关）
    $R/envs/main/bin/python crosstab_l1.py main_v10 main_v12_think --keep-stale

只读。A 进内存（约 37k 条，几十 MB），B 流式扫。
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

FIELD = "substitutable"      # 主字段；同时也会报 cov_ij 单向的


def find_runs_dir():
    cwd = os.getcwd()
    for p in (os.path.join(cwd, "runs"), os.path.join(os.path.dirname(cwd), "runs")):
        if os.path.isdir(p):
            return p
    sys.exit("找不到 runs/，请在 projB 目录下运行")


def l1path(runs, name):
    p = os.path.join(runs, name, "stages", "s6", "l1.jsonl")
    if not os.path.isfile(p):
        sys.exit("没有 %s" % p)
    return p


def key(r):
    return r.get("pair_id") or "%s|%s" % (r.get("qid_i"), r.get("qid_j"))


class Conf:
    """2x2 混淆表：a=A 的标签, b=B 的标签。"""

    def __init__(self):
        self.c = defaultdict(int)

    def add(self, a, b):
        self.c[(int(bool(a)), int(bool(b)))] += 1

    @property
    def n(self):
        return sum(self.c.values())

    def cell(self, a, b):
        return self.c[(a, b)]

    @property
    def p0(self):
        return (self.cell(0, 0) + self.cell(1, 1)) / self.n if self.n else float("nan")

    def kappa(self):
        n = self.n
        if not n:
            return float("nan")
        pa1 = (self.cell(1, 0) + self.cell(1, 1)) / n
        pb1 = (self.cell(0, 1) + self.cell(1, 1)) / n
        pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
        return (self.p0 - pe) / (1 - pe) if pe < 1 else float("nan")

    def pabak(self):
        return 2 * self.p0 - 1 if self.n else float("nan")

    def ac1(self):
        n = self.n
        if not n:
            return float("nan")
        pa1 = (self.cell(1, 0) + self.cell(1, 1)) / n
        pb1 = (self.cell(0, 1) + self.cell(1, 1)) / n
        pi = (pa1 + pb1) / 2
        pe = 2 * pi * (1 - pi)
        return (self.p0 - pe) / (1 - pe) if pe < 1 else float("nan")

    def rate_a(self):
        return (self.cell(1, 0) + self.cell(1, 1)) / self.n if self.n else float("nan")

    def rate_b(self):
        return (self.cell(0, 1) + self.cell(1, 1)) / self.n if self.n else float("nan")


def f(x, w=6, d=3):
    return (" " * w) if x != x else ("%*.*f" % (w, d, x))


def row(label, c, W):
    return ("%-*s %6d  %s %s  %5d %5d %5d %5d  %s %s %s"
            % (W, label, c.n, f(c.rate_a()), f(c.rate_b()),
               c.cell(0, 0), c.cell(0, 1), c.cell(1, 0), c.cell(1, 1),
               f(c.p0), f(c.kappa()), f(c.ac1())))


HDR = "%-*s %6s  %6s %6s  %5s %5s %5s %5s  %6s %6s %6s"


def header(W):
    return HDR % (W, "切片", "n", "rateA", "rateB",
                  "0->0", "0->1", "1->0", "1->1", "p0", "kappa", "AC1")


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("-")]
    keep_stale = "--keep-stale" in sys.argv
    if len(args) != 2:
        sys.exit("用法: crosstab_l1.py <runA> <runB> [--keep-stale]")
    A, B = args
    runs = find_runs_dir()

    sys.stderr.write("读 %s …\n" % A)
    base = {}
    with open(l1path(runs, A), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("eligible"):
                continue
            base[key(r)] = r

    slices = {
        "全部": Conf(),
        "逐字相同 (same_text)": Conf(),
        "  └ 其中 jaccard>=0.999": Conf(),
        "  └ 其中 jaccard< 0.999": Conf(),
        "非逐字 cos>=0.99": Conf(),
        "非逐字 cos< 0.99": Conf(),
    }
    bins = defaultdict(Conf)
    cov = Conf()                 # 单向 cov_ij
    forced_miss = []             # 逻辑上应为 1 而 B 判 0 的对
    matched = skipped_stale = unmatched = 0

    sys.stderr.write("扫 %s …\n" % B)
    with open(l1path(runs, B), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("eligible"):
                continue
            k = key(r)
            a = base.get(k)
            if a is None:
                unmatched += 1
                continue
            if not keep_stale and not r.get("judge_aggregation"):
                skipped_stale += 1
                continue
            sa, sb = a.get(FIELD), r.get(FIELD)
            if sa is None or sb is None:
                continue
            matched += 1
            slices["全部"].add(sa, sb)
            bins[r.get("bin")].add(sa, sb)
            if a.get("cov_ij") is not None and r.get("cov_ij") is not None:
                cov.add(a["cov_ij"], r["cov_ij"])
            jac = float(r.get("jaccard") or 0.0)
            if r.get("same_text"):
                slices["逐字相同 (same_text)"].add(sa, sb)
                if jac >= 0.999:
                    slices["  └ 其中 jaccard>=0.999"].add(sa, sb)
                    if not sb:
                        forced_miss.append(r)
                else:
                    slices["  └ 其中 jaccard< 0.999"].add(sa, sb)
            elif float(r.get("cos") or 0) >= 0.99:
                slices["非逐字 cos>=0.99"].add(sa, sb)
            else:
                slices["非逐字 cos< 0.99"].add(sa, sb)

    W = max(len(s) for s in slices) + 2
    print()
    print("=" * 104)
    print("A = %s    B = %s    对齐 %d 对" % (A, B, matched))
    print("未在 A 中找到: %d    跳过未重判(无 judge_aggregation): %d%s"
          % (unmatched, skipped_stale, "  [--keep-stale 可保留]"
             if skipped_stale else ""))
    print("字段 = %s；rateA/rateB 是各自判 1 的比例" % FIELD)
    print("=" * 104)
    print(header(W))
    for name, c in slices.items():
        if c.n:
            print(row(name, c, W))

    print()
    print("=" * 104)
    print("逐档（bin）")
    print("=" * 104)
    print(header(W))
    for b in sorted(k for k in bins if k is not None):
        print(row("bin%d" % b, bins[b], W))

    print()
    print("=" * 104)
    print("单向 cov_ij（不受合取放大影响）")
    print("=" * 104)
    print(header(W))
    print(row("cov_ij 全部", cov, W))

    print()
    print("=" * 104)
    print("逻辑上应为 1 而 B 判 0 的对：same_text 且 jaccard>=0.999，共 %d 条"
          % len(forced_miss))
    print("（query 文本相同 + 检索结果集相同 ⇒ cov_ij 等价于 self_cov_i，"
          "而 self_cov_i 全为 1）")
    print("=" * 104)
    for r in forced_miss[:8]:
        print("  pair=%s bin=%s cos=%.4f jac=%.3f rbo=%.3f "
              "cov_ij=%s cov_ji=%s votes_ij=%s votes_ji=%s self_i=%s self_j=%s"
              % (str(r.get("pair_id"))[:16], r.get("bin"),
                 float(r.get("cos") or 0), float(r.get("jaccard") or 0),
                 float(r.get("rbo") or 0), r.get("cov_ij"), r.get("cov_ji"),
                 r.get("votes_ij"), r.get("votes_ji"),
                 r.get("self_cov_i"), r.get("self_cov_j")))
    if len(forced_miss) > 8:
        print("  … 还有 %d 条" % (len(forced_miss) - 8))


if __name__ == "__main__":
    main()
