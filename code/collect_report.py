#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
collect_report.py — 把一次 J2/重校准跑完后我需要看的东西全部抓出来，打成一份
可以直接复制给我的纯文本报告（同时落盘，太长就直接传文件）。

用法（在 projB 目录下，matview 根目录用 -R 指过去）:

    R=/mnt/data/user_workspace/zhangzizhao/matview
    $R/envs/main/bin/python collect_report.py -R $R \
        --run main_v12_think --baseline main_v12

不确定 run 名字就先列一下:

    $R/envs/main/bin/python collect_report.py -R $R --list

只跟进 J1/J3 的结果:

    $R/envs/main/bin/python collect_report.py -R $R --phase2-only

设计原则:
  * 只读，不写任何 runs/ 下的东西；
  * 对未知 schema 不做假设：小 JSON 整份打印，大 JSON 按预算裁剪（裁剪处留标记）；
  * jsonl 不整份读进内存，流式统计条数 / 低基数字段分布 / 首尾各一条；
  * 和 baseline 同路径的数值叶子做差，只打印真正变了的。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter, OrderedDict

# ---------------------------------------------------------------- 输出缓冲

class Out:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, s=""):
        self.buf.write(str(s) + "\n")

    def h1(self, s):
        self("")
        self("=" * 78)
        self(s)
        self("=" * 78)

    def h2(self, s):
        self("")
        self("--- " + s + " " + "-" * max(0, 72 - len(s)))

    def text(self):
        return self.buf.getvalue()


out = Out()

# ---------------------------------------------------------------- JSON 裁剪

MAX_STR = 300          # 单个字符串最长
MAX_LIST = 8           # 列表最多展开几项
MAX_DEPTH = 7
MAX_KEYS = 60          # 单个 dict 最多展开几个键


def prune(obj, depth=0):
    """把任意 JSON 裁成可打印的规模，裁掉的地方留下 <...> 标记。"""
    if depth >= MAX_DEPTH:
        return "<depth-capped %s>" % type(obj).__name__
    if isinstance(obj, str):
        if len(obj) > MAX_STR:
            return obj[:MAX_STR] + " …<+%d chars>" % (len(obj) - MAX_STR)
        return obj
    if isinstance(obj, dict):
        o = OrderedDict()
        for i, (k, v) in enumerate(obj.items()):
            if i >= MAX_KEYS:
                o["<...>"] = "+%d more keys" % (len(obj) - MAX_KEYS)
                break
            o[str(k)] = prune(v, depth + 1)
        return o
    if isinstance(obj, (list, tuple)):
        if len(obj) > MAX_LIST:
            head = [prune(v, depth + 1) for v in obj[:MAX_LIST]]
            head.append("<+%d more items>" % (len(obj) - MAX_LIST))
            return head
        return [prune(v, depth + 1) for v in obj]
    return obj


def dump(obj):
    return json.dumps(prune(obj), ensure_ascii=False, indent=2, default=str)


def leaves(obj, path=""):
    """展平成 {点分路径: 数值}，只保留 int/float/bool。"""
    acc = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            acc.update(leaves(v, "%s.%s" % (path, k) if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            acc.update(leaves(v, "%s[%d]" % (path, i)))
    elif isinstance(obj, bool):
        acc[path] = obj
    elif isinstance(obj, (int, float)):
        acc[path] = obj
    return acc


def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)


# ---------------------------------------------------------------- jsonl 概览

LOWCARD_MAX = 10       # 低基数字段：distinct 值不超过这个数才打印分布


def flat_scalars(obj, path="", depth=0, acc=None):
    """只取前两层的标量，用于统计分布（避免把 passage 文本也拿来计数）。"""
    if acc is None:
        acc = {}
    if depth > 2:
        return acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            flat_scalars(v, "%s.%s" % (path, k) if path else str(k),
                         depth + 1, acc)
    elif isinstance(obj, (bool, int, float)) or obj is None:
        acc[path] = obj
    elif isinstance(obj, str) and len(obj) <= 48:
        acc[path] = obj
    return acc


def summarize_jsonl(p, label):
    out.h2("%s  (%s)" % (label, human_size(p)))
    n = 0
    bad = 0
    first = last = None
    counters = {}
    numeric = {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    bad += 1
                    continue
                if first is None:
                    first = rec
                last = rec
                for k, v in flat_scalars(rec).items():
                    if isinstance(v, bool) or isinstance(v, str) or v is None:
                        c = counters.setdefault(k, Counter())
                        if len(c) <= LOWCARD_MAX + 1:
                            c[v] += 1
                    elif isinstance(v, (int, float)):
                        s = numeric.setdefault(k, [0, 0.0, None, None])
                        s[0] += 1
                        s[1] += float(v)
                        s[2] = v if s[2] is None else min(s[2], v)
                        s[3] = v if s[3] is None else max(s[3], v)
    except Exception as e:
        out("!! 读不了: %s: %s" % (type(e).__name__, e))
        return
    out("记录数 = %d   解析失败 = %d" % (n, bad))
    if first is not None:
        out("")
        out("[第一条，裁剪后]")
        out(dump(first))
    if n > 1 and last is not None:
        out("")
        out("[最后一条，裁剪后]")
        out(dump(last))
    cats = {k: c for k, c in counters.items() if 1 <= len(c) <= LOWCARD_MAX}
    if cats:
        out("")
        out("[低基数字段分布]")
        for k in sorted(cats):
            items = ", ".join("%s=%d" % (v, c)
                              for v, c in cats[k].most_common())
            out("  %-44s %s" % (k, items))
    if numeric:
        out("")
        out("[数值字段 n/mean/min/max]")
        for k in sorted(numeric):
            cnt, tot, lo, hi = numeric[k]
            out("  %-44s n=%d mean=%.4g min=%.4g max=%.4g"
                % (k, cnt, tot / max(cnt, 1), lo, hi))


def human_size(p):
    try:
        b = os.path.getsize(p)
    except OSError:
        return "?"
    for u in ("B", "KB", "MB", "GB"):
        if b < 1024 or u == "GB":
            return "%.1f%s" % (b, u)
        b /= 1024.0


# ---------------------------------------------------------------- 阶段目录

JSONL_STAGES = {"s6"}          # 这些阶段的 jsonl 走流式概览


def walk_stages(run_dir, only=None):
    st = os.path.join(run_dir, "stages")
    if not os.path.isdir(st):
        out("!! 没有 stages/ 目录: %s" % st)
        return
    names = sorted(os.listdir(st), key=stage_key)
    for name in names:
        if only and name not in only:
            continue
        d = os.path.join(st, name)
        if not os.path.isdir(d):
            continue
        link = " -> %s" % os.readlink(d) if os.path.islink(d) else ""
        out.h2("stages/%s%s" % (name, link))
        files = sorted(os.listdir(d))
        if not files:
            out("(空)")
            continue
        for fn in files:
            p = os.path.join(d, fn)
            if os.path.isdir(p):
                out("  [dir] %s/ (%d 项)" % (fn, len(os.listdir(p))))
                continue
            if fn.endswith(".jsonl"):
                if name in JSONL_STAGES or os.path.getsize(p) < 2_000_000:
                    summarize_jsonl(p, "stages/%s/%s" % (name, fn))
                else:
                    out("  %s  %s  (跳过，太大)" % (fn, human_size(p)))
                continue
            if fn.endswith(".json"):
                obj, err = load_json(p)
                out("")
                out("[stages/%s/%s  %s]" % (name, fn, human_size(p)))
                out(dump(obj) if err is None else "!! " + err)
                continue
            out("  %s  %s" % (fn, human_size(p)))


def stage_key(name):
    if name.startswith("s") and name[1:].split("_")[0].isdigit():
        return (0, int(name[1:].split("_")[0]), name)
    return (1, 0, name)


# ---------------------------------------------------------------- baseline 差

def diff_runs(new_dir, base_dir, stages):
    out.h1("与 baseline 的数值差（只列变了的叶子）")
    out("new  = %s" % new_dir)
    out("base = %s" % base_dir)
    for name in stages:
        nd = os.path.join(new_dir, "stages", name)
        bd = os.path.join(base_dir, "stages", name)
        if not (os.path.isdir(nd) and os.path.isdir(bd)):
            continue
        for fn in sorted(os.listdir(nd)):
            if not fn.endswith(".json"):
                continue
            pn, pb = os.path.join(nd, fn), os.path.join(bd, fn)
            if not os.path.isfile(pb):
                continue
            a, ea = load_json(pn)
            b, eb = load_json(pb)
            if ea or eb:
                continue
            la, lb = leaves(a), leaves(b)
            rows = []
            for k in sorted(set(la) & set(lb)):
                x, y = la[k], lb[k]
                if isinstance(x, bool) or isinstance(y, bool):
                    if x != y:
                        rows.append((k, y, x, ""))
                    continue
                if abs(float(x) - float(y)) > 1e-9:
                    d = float(x) - float(y)
                    rows.append((k, y, x, "%+.4g" % d))
            # 数组里的叶子（path 带 [i]）会把标量字段淹掉 —— s8 的 band_hi
            # 有 400 个元素，按字母序排在 tau_multi 前面，正好把要看的挤掉。
            # 所以标量先全列，数组按基名折叠成一行。
            scal = [r for r in rows if "[" not in r[0]]
            arr = defaultdict(list)
            for r in rows:
                if "[" in r[0]:
                    arr[r[0].split("[")[0]].append(r)
            only_new = sorted(set(la) - set(lb))
            if rows or only_new:
                out.h2("%s/%s" % (name, fn))
                for k, y, x, d in scal[:400]:
                    out("  %-52s base=%-14s new=%-14s %s"
                        % (k, fmt(y), fmt(x), d))
                if len(scal) > 400:
                    out("  <+%d more changed scalar leaves>" % (len(scal) - 400))
                for base_, rs in sorted(arr.items()):
                    ds = [float(x) - float(y) for _, y, x, _ in rs
                          if not isinstance(x, bool) and not isinstance(y, bool)]
                    if ds:
                        out("  %-52s %d 个元素变化，Δ 从 %+.4g 到 %+.4g"
                            % (base_ + "[…]", len(rs), min(ds), max(ds)))
                    else:
                        out("  %-52s %d 个元素变化" % (base_ + "[…]", len(rs)))
                if only_new:
                    out("  [只在 new 里有] " + ", ".join(only_new[:40])
                        + (" …+%d" % (len(only_new) - 40)
                           if len(only_new) > 40 else ""))


def fmt(v):
    if isinstance(v, float):
        return "%.6g" % v
    return str(v)


# ---------------------------------------------------------------- 日志尾巴

def tail(p, n=40, grep=None):
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        out("!! %s: %s" % (type(e).__name__, e))
        return
    if grep:
        hits = [l for l in lines if any(g in l for g in grep)]
        out("[匹配 %s 的行，共 %d 条，取后 %d]"
            % ("/".join(grep), len(hits), min(n, len(hits))))
        lines = hits
    else:
        out("[末尾 %d 行，全文 %d 行]" % (min(n, len(lines)), len(lines)))
    for l in lines[-n:]:
        out("  " + l.rstrip())


ERR_PAT = ["Traceback", "Error", "error", "ERROR", "OOM", "失败",
           "不可行", "infeasible", "WARN", "Warning", "τ*", "tau_star"]


# ---------------------------------------------------------------- 定位 runs/

SKIP_DIRS = {"envs", "node_modules", "__pycache__", "site-packages",
             ".git", "data", "corpus", "models", "checkpoints"}


def _looks_like_runs(p):
    """一个目录像不像 runs/：里面至少有一个含 stages/ 的子目录，或者有 phase2/。"""
    if not os.path.isdir(p):
        return False
    try:
        for n in os.listdir(p):
            d = os.path.join(p, n)
            if not os.path.isdir(d):
                continue
            if n == "phase2" or os.path.isdir(os.path.join(d, "stages")):
                return True
    except OSError:
        return False
    return False


def resolve_runs(root, cfg_path, explicit):
    """按 显式指定 → root/runs → cwd/runs → config.json 里的路径 → 浅搜 的顺序找。"""
    cwd = os.getcwd()
    cands = []
    if explicit:
        cands.append((os.path.abspath(explicit), "--runs"))
    cands.append((os.path.join(root, "runs"), "root/runs"))
    cands.append((os.path.join(cwd, "runs"), "cwd/runs"))
    cands.append((os.path.join(os.path.dirname(cwd), "runs"), "cwd/../runs"))

    if cfg_path and os.path.isfile(cfg_path):
        cfg, err = load_json(cfg_path)
        if err is None and isinstance(cfg, dict):
            for k, v in sorted(cfg.items()):
                if not isinstance(v, str) or not v or len(v) > 400:
                    continue
                for base in (None, root, cwd):
                    p = os.path.abspath(v if base is None
                                        else os.path.join(base, v))
                    cands.append((p, "config.json:%s" % k))
                    cands.append((os.path.join(p, "runs"),
                                  "config.json:%s/runs" % k))

    for base in (cwd, root, os.path.dirname(cwd)):
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            depth = dirpath[len(base):].count(os.sep)
            if depth >= 3:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in SKIP_DIRS]
            if "runs" in dirnames:
                cands.append((os.path.join(dirpath, "runs"), "浅搜 %s" % base))

    seen = set()
    for p, why in cands:
        if p in seen:
            continue
        seen.add(p)
        if _looks_like_runs(p):
            return p, why
    return os.path.join(root, "runs"), "都没找到，回退到 root/runs"


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-R", "--root", required=False, default=".",
                    help="matview 根目录（里面有 runs/、config.json）")
    ap.add_argument("--run", default="main_v12_think", help="新 run-id")
    ap.add_argument("--baseline", default="main_v12",
                    help="对照 run-id（原始那次全量）")
    ap.add_argument("--stages", default="s6,s8,s9,s10,s11,s13,s15,s16,s17,s23,s24",
                    help="要抓的阶段，逗号分隔；all = 全部")
    ap.add_argument("--phase2-only", action="store_true",
                    help="只抓 runs/phase2/J 下的 J1/J2/J3 结果")
    ap.add_argument("--list", action="store_true", help="列出所有 run 目录后退出")
    ap.add_argument("--runs", default=None,
                    help="直接指定 runs/ 目录（不指定就自动找）")
    ap.add_argument("--config", default=None,
                    help="config.json 路径（用来从里面读 runs 位置）")
    ap.add_argument("-o", "--outfile", default="collect_report.txt")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    cfg_path = a.config or os.path.join(root, "config.json")
    runs, why = resolve_runs(root, cfg_path, a.runs)

    if a.list:
        print("runs = %s   (来源: %s)" % (runs, why))
        print("runs/ 下的目录：")
        if os.path.isdir(runs):
            for n in sorted(os.listdir(runs)):
                p = os.path.join(runs, n)
                if os.path.isdir(p):
                    st = os.path.join(p, "stages")
                    k = len(os.listdir(st)) if os.path.isdir(st) else 0
                    print("  %-28s stages=%d" % (n, k))
        else:
            print("  !! 没有 %s" % runs)
        return 0

    out.h1("collect_report.py")
    out("root = %s" % root)
    out("runs = %s   (来源: %s)" % (runs, why))
    out("run  = %s" % a.run)
    out("base = %s" % a.baseline)
    out("cwd  = %s" % os.getcwd())
    out("python = %s" % sys.version.split()[0])

    # --- J1/J2/J3 的落盘（phase2/ 和 judge_fix_* 都扫） ----------------
    sweep = []
    if os.path.isdir(runs):
        for n in sorted(os.listdir(runs)):
            if not os.path.isdir(os.path.join(runs, n)):
                continue
            if n == "phase2" or n.startswith("judge_fix") or n.startswith("phase2_"):
                sweep.append(n)
    if not sweep:
        out.h1("没找到 phase2/ 或 judge_fix_* 目录（J1/J2 还没落盘？）")
    for name in sweep:
        base = os.path.join(runs, name)
        out.h1("runs/%s" % name)
        empty = True
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames.sort()
            rel = os.path.relpath(dirpath, base)
            for fn in sorted(filenames):
                p = os.path.join(dirpath, fn)
                tag = os.path.join(rel, fn) if rel != "." else fn
                empty = False
                if fn.endswith(".json") and os.path.getsize(p) < 400_000:
                    obj, err = load_json(p)
                    out("")
                    out("[%s/%s  %s]" % (name, tag, human_size(p)))
                    out(dump(obj) if err is None else "!! " + err)
                elif fn.endswith(".jsonl"):
                    summarize_jsonl(p, "%s/%s" % (name, tag))
                elif fn.endswith((".log", ".txt", ".err", ".out")):
                    out("")
                    out("[%s/%s  %s]" % (name, tag, human_size(p)))
                    tail(p, n=30, grep=ERR_PAT)
                else:
                    out("  %s/%s  %s" % (name, tag, human_size(p)))
        if empty:
            out("(目录是空的)")

    if a.phase2_only:
        return finish(a.outfile)

    # --- 新 run ---------------------------------------------------------
    run_dir = os.path.join(runs, a.run)
    out.h1("runs/%s" % a.run)
    if not os.path.isdir(run_dir):
        out("!! 目录不存在。--list 看看有哪些 run。")
    else:
        for meta in ("state.json", "config.snapshot.json", "meta.json"):
            p = os.path.join(run_dir, meta)
            if os.path.isfile(p):
                obj, err = load_json(p)
                out("")
                out("[%s]" % meta)
                out(dump(obj) if err is None else "!! " + err)
        only = None if a.stages == "all" else set(
            s.strip() for s in a.stages.split(",") if s.strip())
        walk_stages(run_dir, only)

        for fn in sorted(os.listdir(run_dir)):
            if fn.endswith((".log", ".txt")):
                out.h2("runs/%s/%s" % (a.run, fn))
                tail(os.path.join(run_dir, fn), n=50, grep=ERR_PAT)

    # --- 差分 ------------------------------------------------------------
    base_dir = os.path.join(runs, a.baseline)
    if os.path.isdir(base_dir) and os.path.isdir(run_dir):
        st = [s for s in (a.stages.split(",") if a.stages != "all" else [])
              if s.strip()]
        if not st:
            st = sorted(os.listdir(os.path.join(run_dir, "stages")),
                        key=stage_key)
        diff_runs(run_dir, base_dir, [s.strip() for s in st])
    else:
        out.h1("跳过差分（baseline 目录不存在：%s）" % base_dir)

    return finish(a.outfile)


def finish(outfile):
    txt = out.text()
    try:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(txt)
    except Exception as e:
        sys.stderr.write("写文件失败: %s\n" % e)
    sys.stdout.write(txt)
    n = len(txt)
    sys.stderr.write(
        "\n[collect_report] %d 字符 / 约 %d token，已写入 %s\n"
        % (n, n // 3, os.path.abspath(outfile)))
    if n > 120_000:
        sys.stderr.write(
            "[collect_report] 偏长：直接把这个文件传给 Claude，别复制粘贴。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
