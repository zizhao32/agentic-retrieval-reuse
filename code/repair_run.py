#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
repair_run.py — 就地修好一个 fix_judge.py 产出的 run，不重判。

修三件事（都是我上一版 fix_judge.py 的疏漏）:

  1) s1–s5 少链的文件。白名单漏了 s4/audit.json，s8 起手就
     FileNotFoundError。改成把来源 run 里 s1–s5 的每个文件都补上。

  2) s6/qc.json 是来源 run 的整份拷贝。里面大半是判定派生量
     （pos_consistency / strict / union / asymmetry / llm / tokens / 各种
     substitutable_rate），拷过来等于把旧臂的数字贴上新臂的标签 ——
     实测本 run 的 qc.json 和 main_v10 的每一个数值叶子都相同。
     这里按本 run 的 l1.jsonl 重算能算的，算不了的显式丢弃并记下丢了什么。

  3) 合格但重判失败、被原样带过来的记录（本 run 有 133 条）。它们带的是
     旧臂的 cov/substitutable，混在新臂的文件里。置空并标 rejudge_failed，
     旧值挪进 _inherited。

用法（在 projB 下）:

    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python repair_run.py \
        --run main_v12_think --src main_v10

    # 只看会改什么，不动文件
    $R/envs/main/bin/python repair_run.py --run main_v12_think --src main_v10 --dry-run

l1.jsonl 改写前会留 l1.jsonl.bak（已存在就不覆盖），新文件先写 .new 再原子改名。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

KEEP = ("n_sampled_pairs", "n_eligible", "eligibility_rate",
        "eligibility_by_bin", "self_cov_rate")
DROP = ("pos_consistency", "pos_consistency_n", "pos_consistency_by_bin",
        "strict", "union", "asymmetry_rate", "llm", "tokens", "n_judged",
        "effective_n", "substitutable_rate_conditional",
        "substitutable_rate_conditional_ci",
        "substitutable_rate_conditional_unweighted",
        "substitutable_rate_marginal", "same_text_direction_consistency",
        "same_text_n", "catch_fp_rate", "parse_fail_rate", "n_parse_fail",
        "truncated_share", "mode_used", "cross_kappa", "cross_n")
STALE_LABELS = ("cov_ij", "cov_ji", "substitutable", "verdict_ij", "verdict_ji")


def runs_dir():
    cwd = os.getcwd()
    for p in (Path(cwd) / "runs", Path(cwd).parent / "runs"):
        if p.is_dir():
            return p
    sys.exit("找不到 runs/，请在 projB 目录下运行")


def rjson(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def wjson(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".new"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(o, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


# ---------------------------------------------------------------- 1. 补软链

def fix_links(src: Path, run: Path, dry):
    added, missing_src = [], []
    for st in ("s1", "s2", "s3", "s4", "s5"):
        sd = src / "stages" / st
        if not sd.is_dir():
            missing_src.append(st)
            continue
        dd = run / "stages" / st
        for sp in sorted(sd.iterdir()):
            if not sp.is_file():
                continue
            dp = dd / sp.name
            if dp.exists():
                continue
            added.append("%s/%s" % (st, sp.name))
            if not dry:
                dd.mkdir(parents=True, exist_ok=True)
                os.symlink(os.path.realpath(sp), dp)
    print("【1】补软链：%d 个" % len(added))
    for a in added:
        print("      + %s" % a)
    if not added:
        print("      (没有缺的)")
    if missing_src:
        print("      ! 来源 run 没有这些阶段：%s" % ", ".join(missing_src))
    return added


# ---------------------------------------------------------- 2+3. l1 与 qc

def fix_l1_and_qc(src: Path, run: Path, dry):
    l1 = run / "stages" / "s6" / "l1.jsonl"
    if not l1.is_file():
        sys.exit("没有 %s" % l1)

    arms, n_fixed, n_total = {}, 0, 0
    good = []                      # 合格且标签有效的记录（用来重算 qc）
    tmp = str(l1) + ".new"
    fout = None if dry else open(tmp, "w", encoding="utf-8")
    try:
        with open(l1, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n_total += 1
                ja = r.get("judge_aggregation") or {}
                arms[ja.get("arm", "<无>")] = arms.get(ja.get("arm", "<无>"), 0) + 1
                # 合格、却没有本臂的 judge_aggregation == 没被重判，带的是旧标签
                if r.get("eligible") and not ja:
                    n_fixed += 1
                    r["_inherited"] = dict(r.get("_inherited") or {},
                                           **{k: r.get(k) for k in STALE_LABELS})
                    for k in STALE_LABELS:
                        r[k] = None
                    r["ok"] = False
                    r["rejudge_failed"] = True
                if r.get("eligible") and r.get("substitutable") is not None:
                    good.append({k: r.get(k) for k in
                                 ("substitutable", "cov_ij", "cov_ji",
                                  "same_text", "w", "bin")})
                if fout:
                    fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    finally:
        if fout:
            fout.close()

    print()
    print("【3】l1.jsonl：%d 条，臂分布 %s" % (n_total, arms))
    print("      合格但未重判、标签被置空的：%d 条" % n_fixed)
    if not dry:
        bak = str(l1) + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(l1, bak)
            print("      原文件已备份到 %s" % bak)
        os.replace(tmp, l1)
        print("      已改写 %s" % l1)
    else:
        print("      (dry-run，没写)")

    # ---- qc 重算
    srcqc = src / "stages" / "s6" / "qc.json"
    runqc = run / "stages" / "s6" / "qc.json"
    old = rjson(srcqc) if srcqc.is_file() else {}
    cur = rjson(runqc) if runqc.is_file() else {}
    arm_block = cur.get("judge_aggregation") or {}

    identical = [k for k in DROP
                 if k in old and k in cur and json.dumps(old[k], sort_keys=True)
                 == json.dumps(cur[k], sort_keys=True)]

    qc = {k: old[k] for k in KEEP if k in old}
    st = [r for r in good if r.get("same_text")]

    def wrate(rs, f):
        num = sum((r.get("w") or 0.0) for r in rs if f(r))
        den = sum((r.get("w") or 0.0) for r in rs)
        return (num / den) if den else None

    qc["judge_aggregation"] = arm_block
    qc["recomputed_from_l1"] = {
        "n_judged": len(good),
        "substitutable_rate_conditional_unweighted":
            mean([r["substitutable"] for r in good]),
        "substitutable_rate_conditional":
            wrate(good, lambda r: r["substitutable"]),
        "asymmetry_rate":
            (sum(1 for r in good if r.get("cov_ij") != r.get("cov_ji"))
             / len(good)) if good else None,
        "same_text_n": len(st),
        "same_text_substitutable_rate":
            mean([r["substitutable"] for r in st]),
    }
    qc["_provenance"] = {
        "inherited_from": str(src),
        "inherited_fields": [k for k in KEEP if k in old],
        "dropped_stale": sorted(k for k in DROP if k in cur),
        "dropped_because_byte_identical_to_source":
            sorted(identical),
        "repaired_by": "repair_run.py",
        "n_labels_voided": n_fixed,
        "selfcov_arm": (old.get("judge_aggregation") or {}).get("arm")
                       or "(来源 run，臂未记录)",
        "coverage_arm": arm_block.get("arm"),
        "MIXED_ARM_WARNING":
            "合格性筛选（self_cov）沿用来源 run 的判官，覆盖判定是本臂 —— "
            "两个臂混用。对逐字相同的一对，j 的检索结果就是 i 的结果，"
            "cov_ij 问的就是 self_cov_i，而筛选保证 self_cov_i=1，"
            "所以恒等带上的缺口量的是两个臂在自覆盖上的分歧，"
            "不是本臂的性质。单臂干净结果要跑 fix_judge.py --exp J4。",
        "note": "被 drop 的字段是上一个臂的判定派生量，本 run 没有重算，"
                "所以不写出来，而不是留着旧值。",
    }

    print()
    print("【2】qc.json 重算")
    print("      保留的筛选侧字段：%s" % ", ".join(qc["_provenance"]["inherited_fields"]))
    print("      丢弃的判定派生量：%d 个" % len(qc["_provenance"]["dropped_stale"]))
    print("      其中与来源 run 逐字节相同（证明是拷来的）：%d 个" % len(identical))
    for k in identical[:12]:
        print("        · %s" % k)
    if len(identical) > 12:
        print("        … 还有 %d 个" % (len(identical) - 12))
    print("      本 run 现算的数：")
    for k, v in qc["recomputed_from_l1"].items():
        print("        %-48s %s" % (k, round(v, 6) if isinstance(v, float) else v))
    if dry:
        print("      (dry-run，没写)")
    else:
        bak = str(runqc) + ".bak"
        if runqc.is_file() and not os.path.exists(bak):
            shutil.copy2(runqc, bak)
            print("      原文件已备份到 %s" % bak)
        wjson(runqc, qc)
        print("      已写 %s" % runqc)
    return qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="要修的 run（如 main_v12_think）")
    ap.add_argument("--src", required=True, help="来源 run（如 main_v10）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rd = runs_dir()
    run, src = rd / a.run, rd / a.src
    for p in (run, src):
        if not p.is_dir():
            sys.exit("没有 %s" % p)
    print("runs = %s" % rd)
    print("run  = %s" % run)
    print("src  = %s" % src)
    print("模式 = %s" % ("dry-run（只看）" if a.dry_run else "就地修改"))
    print()
    fix_links(src, run, a.dry_run)
    fix_l1_and_qc(src, run, a.dry_run)
    print()
    if a.dry_run:
        print("去掉 --dry-run 执行。")
    else:
        print("修完了。接着跑 s8：")
        print("  python phase1_v13.py --config <config> --run-id %s --stages s8"
              % a.run)


if __name__ == "__main__":
    main()
