#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
数字溯源审计 —— 论文里每个数都必须能在 run 产物里找到
================================================================================
为什么要有这个脚本：已经发现三处追不回来的数字 ——

  * \HeadCrossShare = 87.0%，但 s10/reuse.json 里全域是 null、agent 是 0.830；
    87.0 在任何产物里都找不到。这个数出现在摘要、§4.1、§8，是「复用确实是跨任务的」
    这一论点的全部证据。
  * qc.json 有 cross_kappa = 0.824 / n = 600，但现在的 config.json 里根本没有
    cross_judge_* 字段，而那段代码必须配了外部 model 才会执行 —— 也就是说
    artifact 跑不出这个数。
  * runs/phase2/B/b_browsecomp.json 不存在（B4 失败过），BCP 那组数来源待查。

EA&B 的 Reproducibility Committee 做的就是拿 artifact 重算论文里的数。
一个追不回来的数字会让审稿人怀疑全部，所以这件事必须机械地做一遍，不能靠人眼。

做法
----
1. 解析 macros.tex，取出每个 \newcommand 的值，归一成候选数值
   （"22.7\%" -> 22.7 和 0.227；"1{,}888" -> 1888；"$+0.072$" -> 0.072）。
2. 递归读入所有 run 目录下的 *.json，拍平成 (文件, 键路径, 数值)。
3. 对每个宏，在全部数值里按**它自己印出来的精度**找匹配。
4. 输出三类：
     MATCHED   —— 找到了，列出第一个来源（可用 --all-hits 看全部）
     UNMATCHED —— ★ 任何产物里都找不到，必须人工追
     SKIPPED   —— 非数值宏（模型名、区间、文字）
5. 另外交叉检查 sections/*.tex：用了但没定义的宏、定义了但没用的宏。

用法
----
    python audit_macros.py --paper matview-vldb2027 \
        --runs runs/main_v10 runs/agent_b runs/phase2 runs/main_v10_distract

    # 只看追不回来的
    python audit_macros.py --paper matview-vldb2027 --runs runs/... --only-unmatched

    # 某个宏到底出现在哪些地方
    python audit_macros.py --paper ... --runs ... --trace HeadCrossShare

退出码：有 UNMATCHED 则为 1，便于挂到提交前的检查里。
================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ------------------------------------------------------------------ 宏值归一化

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def macro_candidates(raw: str):
    """把宏的值翻成候选数值。返回 [(数值, 印出来的小数位)]，空列表表示非数值宏。

    关键点是**按印出来的精度**比较：论文写 22.7%，产物里是 0.22703…，
    只有 round(0.22703*100, 1) == 22.7 这种比法才既不漏也不滥。
    """
    s = raw
    s = s.replace("\\xspace", "").replace("\\,", "").replace("{,}", "")
    s = s.replace("\\%", "%").replace("$", "").replace("\\", "")
    s = s.replace("{", "").replace("}", "").strip()
    # 区间/范围/多值的宏（[0.78, 0.89]、74%--83%）不做单值匹配，交给人工
    if s.count("--") or ("[" in s and "," in s):
        return []
    # "22.0\pm0.3\%" 这类：拿均值去匹配，sd 另一回事。
    # 不这么处理的话，K1/R1 新加的那批带误差棒的宏会全被扔进「需人工」，
    # 而它们恰恰是最该被溯源的。
    if "pm" in s:
        s = s.split("pm")[0] + ("%" if "%" in s else "")
    nums = _NUM.findall(s)
    if len(nums) != 1:
        return []
    tok = nums[0]
    dec = len(tok.split(".")[1]) if "." in tok else 0
    v = float(tok)
    out = [(v, dec)]
    if "%" in s:                       # 22.7% 既可能存成 22.7 也可能存成 0.227
        out.append((v / 100.0, dec + 2))
    else:                              # 反过来也试一次，产物里偶有存成百分数的
        out.append((v * 100.0, max(dec - 2, 0)))
    return out


# ------------------------------------------------------------------ 产物数值索引

def walk_json(obj, path, sink, fname):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json(v, f"{path}.{k}" if path else str(k), sink, fname)
    elif isinstance(obj, list):
        # 列表只记前 200 项，避免 per_pair 这类大数组把索引撑爆
        for i, v in enumerate(obj[:200]):
            walk_json(v, f"{path}[{i}]", sink, fname)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        sink.append((fname, path, float(obj)))


def index_runs(run_dirs, log=print):
    sink, nfile = [], 0
    for rd in run_dirs:
        rp = Path(rd)
        if not rp.exists():
            log(f"  ! 找不到 {rd}，跳过")
            continue
        for jp in sorted(rp.rglob("*.json")):
            try:
                d = json.loads(jp.read_text(encoding="utf-8"))
            except Exception as e:
                log(f"  ! 读不了 {jp}: {type(e).__name__}")
                continue
            walk_json(d, "", sink, str(jp))
            nfile += 1
    log(f"  索引了 {nfile} 个 json，{len(sink)} 个数值")
    return sink


def find(cands, sink, limit=6):
    hits = []
    for fname, path, v in sink:
        for target, dec in cands:
            try:
                if round(v, dec) == round(target, dec):
                    hits.append((fname, path, v))
                    break
            except (ValueError, OverflowError):
                continue
        if len(hits) >= limit:
            break
    return hits


# ------------------------------------------------------------------ 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="论文数字溯源审计")
    ap.add_argument("--paper", default="matview-vldb2027",
                    help="论文目录，内含 macros.tex 与 sections/")
    ap.add_argument("--runs", nargs="+", required=True, help="一个或多个 run 目录")
    ap.add_argument("--only-unmatched", action="store_true")
    ap.add_argument("--all-hits", action="store_true")
    ap.add_argument("--trace", default=None, help="只追这一个宏（不带反斜杠）")
    a = ap.parse_args()

    paper = Path(a.paper)
    mac_p = paper / "macros.tex"
    if not mac_p.exists():
        print(f"找不到 {mac_p}")
        return 2

    src = mac_p.read_text(encoding="utf-8")
    macros = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*(?:%.*)?$",
                         src, re.M):
        macros[m.group(1)] = m.group(2)
    print(f"macros.tex：{len(macros)} 个宏")

    # 论文正文里实际用到哪些宏
    used = set()
    for tp in sorted((paper / "sections").glob("*.tex")) + [paper / "main.tex"]:
        if tp.exists():
            used |= set(re.findall(r"\\([A-Z][A-Za-z]+)", tp.read_text(encoding="utf-8")))

    print("索引 run 产物 …")
    sink = index_runs(a.runs)
    print()

    if a.trace:
        name = a.trace.lstrip("\\")
        if name not in macros:
            print(f"macros.tex 里没有 \\{name}")
            return 2
        cands = macro_candidates(macros[name])
        print(f"\\{name} = {macros[name]}  -> 候选数值 {cands}")
        hits = find(cands, sink, limit=40)
        if not hits:
            print("  ★ 任何产物里都找不到")
        for f, p, v in hits:
            print(f"  {v!r:>14}  {p}  ({f})")
        print()
        print("正文中出现在：")
        for tp in sorted((paper / "sections").glob("*.tex")):
            for i, ln in enumerate(tp.read_text(encoding="utf-8").splitlines(), 1):
                if f"\\{name}" in ln:
                    print(f"  {tp.name}:{i}  {ln.strip()[:100]}")
        return 0

    matched, unmatched, skipped = [], [], []
    for name in sorted(macros):
        cands = macro_candidates(macros[name])
        if not cands:
            skipped.append(name)
            continue
        hits = find(cands, sink)
        (matched if hits else unmatched).append((name, hits))

    if not a.only_unmatched:
        print(f"=== MATCHED {len(matched)} ===")
        for name, hits in matched:
            f, p, v = hits[0]
            tag = "" if name in used else "  [正文未用]"
            print(f"  \\{name:<26} = {macros[name][:22]:<24} <- {p} ({Path(f).name}){tag}")
            if a.all_hits:
                for f2, p2, v2 in hits[1:]:
                    print(f"  {'':<28}   {p2} ({Path(f2).name})")
        print()

    print(f"=== ★ UNMATCHED {len(unmatched)} —— 产物里找不到，必须人工追 ===")
    for name, _ in unmatched:
        where = []
        for tp in sorted((paper / "sections").glob("*.tex")):
            n = tp.read_text(encoding="utf-8").count(f"\\{name}")
            if n:
                where.append(f"{tp.name}x{n}")
        flag = "  ← 正文未用，可删" if not where else "  用于 " + " ".join(where)
        print(f"  \\{name:<26} = {macros[name][:30]:<32}{flag}")
    print()

    print(f"=== SKIPPED {len(skipped)}（非单值宏，人工核）===")
    print("  " + ", ".join("\\" + s for s in skipped))
    print()

    undef = sorted(u for u in used if u not in macros
                   and u not in {"S", "Pr", "Recall"})
    if undef:
        print("=== 正文用了但 macros.tex 没定义（多半是 LaTeX 命令，扫一眼）===")
        print("  " + ", ".join("\\" + u for u in undef[:40]))
        print()

    unused = sorted(n for n in macros if n not in used)
    if unused:
        print(f"=== 定义了但正文没用 {len(unused)} 个（可删，减少维护面）===")
        print("  " + ", ".join("\\" + u for u in unused))
        print()

    print(f"小结：{len(matched)} 可溯源 / {len(unmatched)} ★追不到 / "
          f"{len(skipped)} 需人工")
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
