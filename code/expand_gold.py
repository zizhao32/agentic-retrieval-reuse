#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
金标扩充：把评估样本量从 91 提到 200+
================================================================================

背景
----
main_v10 抽了 300 对金标，但只有 91 对通过自覆盖筛选进入评估
（合格率 31.5%，与该轮一致）。n=91 的 accuracy 置信区间宽度 0.14，
是审稿人最容易挑的统计点。

前两批（main_v7 / main_v8）各标了 299 对，合计 598 对，**已经标注过**。
把其中落在高相似度层的样本合并进来，可以把 n 提到 200+ 而不用重标。

两个必须处理的陷阱
------------------
(a) **pair_id 跨 run 不通用。** 前两批的 pair_id 属于 main_v7/v8，
    main_v10 的 l1.jsonl 里没有。必须用 (查询A, 查询B) 的文本对做匹配，
    并且要考虑 A/B 顺序互换。

(b) **标注协议不同。** 前两批用的是旧协议，首轮 accuracy 仅 0.604，
    主要错在把「大小写/语序/关键词式 vs 自然语言」判成不可替代。
    直接合并会把偏严的标注混进来，低估判官。
    本脚本用 main_v10 定稿协议的规则自动**筛出**这类可疑标注，
    默认只把它们列出来供复核，不自动改标签（--auto-fix 才改）。

用法
----
    # 1) 先看能合并多少、有多少可疑
    python expand_gold.py --dry-run

    # 2) 生成待复核清单（可疑的那些）
    python expand_gold.py --emit-review review.tsv

    # 3) 复核完（或接受自动修正）后生成合并 TSV
    python expand_gold.py --out gold_merged.tsv [--auto-fix]

    # 4) 回填评估
    $R/envs/main/bin/python phase1_v13.py --config $R/config.json \
        --run-id main_v10 --stages s15,s24,s16 --gold-tsv gold_merged.tsv --force
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ----------------------------------------------------------------- 文本归一 --
def norm(s: str) -> str:
    """查询文本归一化，用于跨 run 匹配。

    只做不改变语义的归一：小写、去标点、压空白。
    """
    s = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def pair_key(a: str, b: str) -> tuple:
    """无序对的键：A/B 互换视为同一对。"""
    na, nb = norm(a), norm(b)
    return (na, nb) if na <= nb else (nb, na)


# --------------------------------------------------- 旧协议的可疑标注检测 --
def suspicious(qa: str, qb: str, label: int) -> str | None:
    """按 main_v10 定稿协议，判断一条旧标注是否可疑。

    定稿协议的核心：**表面差异不改变检索到什么，就不改变可替代性。**
    首轮标注的系统性错误就是把下列情形判成 0。
    只检测 label==0 的情形（旧协议偏严，几乎不会错判成 1）。
    """
    if label != 0:
        return None
    na, nb = norm(qa), norm(qb)
    if na == nb:
        return "归一化后文本完全相同"
    ta, tb = na.split(), nb.split()
    if sorted(ta) == sorted(tb):
        return "仅词序不同"
    sa, sb = set(ta), set(tb)
    if not sa or not sb:
        return None
    jac = len(sa & sb) / len(sa | sb)
    # 一方是另一方的子集（多/少一个限定词或疑问词）
    if sa <= sb or sb <= sa:
        extra = (sb - sa) or (sa - sb)
        if len(extra) <= 2:
            return f"一方是另一方子集，仅多 {sorted(extra)}"
    if jac >= 0.85:
        return f"词集 Jaccard {jac:.2f}（近乎同一表述）"
    return None


# ------------------------------------------------------------------- 主流程 --
def load_labeled(tsv: Path) -> list[dict]:
    if not tsv.exists():
        print(f"  [跳过] 标注文件不存在: {tsv}")
        return []
    rows = []
    with open(tsv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            v = (r.get("human_label") or "").strip()
            if v in ("0", "1"):
                rows.append({"pair_id": (r.get("pair_id") or "").strip(),
                             "label": int(v),
                             "qa": r.get("查询A") or r.get("query_a") or "",
                             "qb": r.get("查询B") or r.get("query_b") or "",
                             "cos": r.get("cos", ""),
                             "src": tsv.name})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="合并历史金标，扩大评估样本量")
    ap.add_argument("--run", default="runs/main_v10", help="目标 run（提供判官结论）")
    ap.add_argument("--tsv", action="append", default=None,
                    help="历史标注 TSV，可重复。默认三批全收")
    ap.add_argument("--min-bin", type=int, default=5,
                    help="只合并 bin >= 此值的对（默认 5，即 cos>=0.85）")
    ap.add_argument("--out", default=None, help="输出合并后的 TSV")
    ap.add_argument("--emit-review", default=None, help="导出可疑标注供人工复核")
    ap.add_argument("--auto-fix", action="store_true",
                    help="把可疑的 0 自动改为 1（不推荐，子集规则区分不了"
                         "『多一个无关词』和『多一个改变所指的词』）")
    ap.add_argument("--review", default=None,
                    help="读回人工复核后的 TSV（由 --emit-review 导出并填好 "
                         "human_label 列），覆盖对应条目的标签")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    run = Path(a.run)
    tsvs = [Path(t) for t in (a.tsv or [
        "v7_gold_worksheet_labeled.tsv",
        "v8_gold_worksheet_labeled.tsv",
        "gold_v10_final.tsv",
    ])]

    # ---------- 1. 读目标 run 的判官结论与查询文本 ----------
    l1p = run / "stages" / "s6" / "l1.jsonl"
    cap = run / "stages" / "s2" / "calls.jsonl"
    for p in (l1p, cap):
        if not p.exists():
            sys.exit(f"找不到 {p}")
    calls = {}
    with open(cap, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                calls[c["qid"]] = c
    judged = {}          # pair_key -> dict
    with open(l1p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("eligible") != 1 or not r.get("ok"):
                continue
            ci, cj = calls.get(r["qid_i"]), calls.get(r["qid_j"])
            if not ci or not cj:
                continue
            judged[pair_key(ci["query"], cj["query"])] = {
                "pair_id": r["pair_id"], "bin": r.get("bin"), "cos": r["cos"],
                "judge": r["substitutable"], "qa": ci["query"], "qb": cj["query"]}
    print(f"  {run.name}: 自覆盖合格且解析成功的对 {len(judged)}")

    # ---------- 2. 读三批标注，按文本对匹配 ----------
    matched, unmatched, dup = [], Counter(), 0
    seen_keys = {}
    for t in tsvs:
        rows = load_labeled(t)
        hit = 0
        for r in rows:
            k = pair_key(r["qa"], r["qb"])
            j = judged.get(k)
            if not j:
                unmatched[t.name] += 1
                continue
            if k in seen_keys:
                dup += 1
                # 同一对在多批里都标过：以最新一批（列表靠后）为准
                seen_keys[k].update({"label": r["label"], "src": r["src"]})
                continue
            rec = {**j, "label": r["label"], "src": r["src"]}
            seen_keys[k] = rec
            matched.append(rec)
            hit += 1
        print(f"  {t.name:<34} 有效标注 {len(rows):>4}  匹配上 {hit:>4}")
    if dup:
        print(f"  跨批重复的对 {dup}（以最新一批的标签为准）")
    print(f"  未匹配：{dict(unmatched)}")

    # ---------- 3. 按 bin 过滤 ----------
    pool = [m for m in matched if (m["bin"] or 0) >= a.min_bin]
    print(f"\n  匹配上 {len(matched)} 对；bin>={a.min_bin} 的 {len(pool)} 对")
    bybin = Counter(m["bin"] for m in pool)
    print("  分层：" + "  ".join(f"b{b}={bybin[b]}" for b in sorted(bybin)))

    # ---------- 3.5 读回人工复核结果 ----------
    if a.review:
        rp = Path(a.review)
        if not rp.exists():
            sys.exit(f"找不到复核文件 {rp}")
        rev = {}
        with open(rp, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                v = (r.get("human_label") or "").strip()
                if v in ("0", "1"):
                    rev[pair_key(r.get("查询A", ""), r.get("查询B", ""))] = int(v)
        nchg = nsame = 0
        for m in pool:
            k = pair_key(m["qa"], m["qb"])
            if k in rev:
                if rev[k] != m["label"]:
                    m["label"] = rev[k]; m["reviewed"] = True; nchg += 1
                else:
                    m["reviewed"] = True; nsame += 1
        print(f"\n  从 {rp.name} 读回复核 {len(rev)} 条："
              f"改动 {nchg}，确认原判 {nsame}，未匹配 {len(rev)-nchg-nsame}")

    # ---------- 4. 可疑标注检测 ----------
    sus = [(m, why) for m in pool
           if not m.get("reviewed")
           and (why := suspicious(m["qa"], m["qb"], m["label"]))]
    print(f"\n  按 main_v10 定稿协议检出可疑标注 {len(sus)} 条"
          f"（全部是 label=0 但表面差异不改变检索的）")
    bysrc = Counter(m["src"] for m, _ in sus)
    for s, c in bysrc.most_common():
        print(f"     {s:<34} {c}")
    for m, why in sus[:8]:
        print(f"     [{m['src'][:12]}] cos={m['cos']:.4f} 人工=0 判官={m['judge']}  {why}")
        print(f"        A: {m['qa'][:78]}")
        print(f"        B: {m['qb'][:78]}")

    if a.emit_review:
        with open(a.emit_review, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["pair_id", "cos", "bin", "src", "prev_label",
                        "human_label", "reason", "查询A", "查询B"])
            for m, why in sus:
                w.writerow([m["pair_id"], f"{m['cos']:.4f}", m["bin"], m["src"],
                            m["label"], "", why, m["qa"], m["qb"]])
        print(f"\n  -> 待复核清单 {a.emit_review}（{len(sus)} 条，填 human_label 列）")

    # ---------- 5. 评估效果预览 ----------
    def metrics(rows):
        tp = sum(r["label"] == 1 and r["judge"] == 1 for r in rows)
        fp = sum(r["label"] == 0 and r["judge"] == 1 for r in rows)
        fn = sum(r["label"] == 1 and r["judge"] == 0 for r in rows)
        tn = sum(r["label"] == 0 and r["judge"] == 0 for r in rows)
        n = len(rows)
        if n == 0:
            return None
        import math
        acc = (tp + tn) / n
        z = 1.96; d = 1 + z * z / n
        c = (acc + z * z / (2 * n)) / d
        h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / d
        po = acc
        hp = (tp + fn) / n; jp = (tp + fp) / n
        pe = hp * jp + (1 - hp) * (1 - jp)
        kap = (po - pe) / (1 - pe) if pe < 1 else 1.0
        return dict(n=n, acc=acc, lo=max(0, c - h), hi=min(1, c + h),
                    p=tp / max(tp + fp, 1), r=tp / max(tp + fn, 1), k=kap,
                    tp=tp, fp=fp, fn=fn, tn=tn)

    v10_only = [m for m in pool if "v10" in m["src"]]
    print()
    print("  " + "=" * 66)
    for tag, rows in (("仅 main_v10（当前口径）", v10_only),
                      ("合并全部（原标签）", pool)):
        m = metrics(rows)
        if m:
            print(f"  {tag:<26} n={m['n']:>3}  acc={m['acc']:.3f} "
                  f"CI[{m['lo']:.3f},{m['hi']:.3f}]  P={m['p']:.3f} R={m['r']:.3f} "
                  f"κ={m['k']:.3f}")
    fixed = [dict(m, label=1) if any(m is s for s, _ in sus) else m for m in pool]
    m = metrics(fixed)
    if m:
        print(f"  {'合并+自动修正可疑':<26} n={m['n']:>3}  acc={m['acc']:.3f} "
              f"CI[{m['lo']:.3f},{m['hi']:.3f}]  P={m['p']:.3f} R={m['r']:.3f} "
              f"κ={m['k']:.3f}")
    print("  " + "=" * 66)
    print("  注：『自动修正』仅供参考。论文里应报人工复核后的结果，")
    print("      并在方法一节说明合并与复核的过程。")

    if a.dry_run or not a.out:
        if not a.out:
            print("\n  （未指定 --out，仅预览。加 --out gold_merged.tsv 生成合并文件）")
        return 0

    # ---------- 6. 写出合并 TSV ----------
    sus_ids = {id(m) for m, _ in sus}
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["pair_id", "cos", "human_label", "查询A", "查询B",
                    "_bin", "_src", "_flagged"])
        for m in pool:
            lab = m["label"]
            flag = "reviewed" if m.get("reviewed") else ""
            if id(m) in sus_ids:
                flag = "suspicious"
                if a.auto_fix:
                    lab = 1; flag = "auto_fixed"
            w.writerow([m["pair_id"], f"{m['cos']:.4f}", lab,
                        m["qa"], m["qb"], m["bin"], m["src"], flag])
    print(f"\n  -> {a.out}（{len(pool)} 对"
          f"{'，已自动修正 %d 条' % len(sus) if a.auto_fix else ''}）")
    print(f"\n  回填评估：")
    print(f"    $R/envs/main/bin/python phase1_v13.py --config $R/config.json \\")
    print(f"        --run-id {run.name} --stages s15,s24,s16 "
          f"--gold-tsv {a.out} --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
