#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
回应评审 —— 四个实验，针对 Stanford paper review 挑出的四个真漏洞
================================================================================
和 fill_gaps.py 同一套框架（Phase2 / run_step / state.json / 可续跑），
数字全部落盘到 runs/phase2/R/*.json。

2026-10-14 第二版的四处改动（第一版的结果已经拿到，这里是补洞）：
  * R1 加了真正的模糊匹配 L8/L9（SimHash 汉明 <=2 / <=3）。第一版的 L7 用汉明 0，
    等价于对 L4 的键再哈希一次，实测与 L4 逐项相同 —— 根本没做模糊匹配。
    L7 保留但改名为 L7_simhash_h0 并在描述里写明它不是模糊匹配。
  * R1/自检改用统一的匹配器接口（fit/hit），精确键与汉明球同一套流程；
    --selftest 现在也检查 L8/L9 有没有把「不同属性/不同人」并掉。
  * R2 的等任务量口径改成**两个 agent 的任务交集**（读 --r2-b-run），并同时
    对 A、B 两边做复刻核对。这里连错两次：第一版用自己的随机 997（A 基线 25.13%
    而已发表 25.9%）；第二版改去复刻 s20，但 s20 的 m 是「各工作负载任务数的
    最小值」=1888，对 agent 等于不裁剪（A 全量 28.7%），却仍拿 997 个任务上的
    B(17.8%) 作参照 —— 任务数没对齐，而头部覆盖本身随任务数增长，10.9pp 的
    「总差距」里混进了规模效应。交集口径把任务数与任务身份同时固定住。
    另外补了对数线性拟合与「选择性丢弃」效应，并修掉 share>1 时那句误导性结论。
  * R4 的结论改为分开判 δ 与 ε。第一版只看平均包含率就说「对 k 不敏感」，
    而那是两个反向效应抵消出来的假象。

  R1  更强的精确缓存基线（保卫 8.6 点净增量）
      审稿人：没试过结构化规范化（delexicalization / 实体归一 / W5H2 式规范键 /
      模糊哈希），这些可能吃掉一部分净增量。这条批评是对的，而且直指头号数字。
      现状更难看：s23 缓存键的 norm_text 只做「小写 + 标点转空格 + 压空白」，
      连冠词都没去；而同一份代码里算答案 F1 的 _norm **去掉了** a|an|the。
      同一篇论文两个归一化口径不一致，审稿人一眼能看出来。
      做法：把缓存键换成一条逐级加强的规范化阶梯，每级重算净增量，
      看视图相对**最强**缓存还剩多少。先用 L1 复刻已发表的 18.8%，
      复刻不过就不要信后面的数（和 K1 同一个守门逻辑）。

  R2  密度对照（修一处内部不一致，不只是补实验）
      审稿人：两 agent 对比把「查询密度」和「模型本身的差异」混在一起了，
      能不能下采样 Tongyi 的多查询调用来干净地隔离密度。
      我们其实做过（s18 的 all vs first 双臂），结果**和论文的说法相反**：
      同一个 agent 从发全部子查询降到只发第一条，头部覆盖 28.7% -> 27.1%，
      只掉 1.6 点；而跨 agent 是 8.0 点。按组内斜率，B 的 1.84 次/任务
      对 A 的 7.10 次/任务不该值 8.0 点。
      所以摘要与结论里 "consistent with query density" 比证据支持的要强。
      做法：取两个 agent 都跑完的那批任务（task_id 交集），在上面把 A 的调用
      按密度逐级下采样（含恰好等于 B 的密度），重算头部覆盖，得到「密度 -> 覆盖」
      曲线；B 在同一批任务上用它自己的候选对图和同一个 τ* 算一次作参照。
      关键一问：A 降到 B 的密度时覆盖是多少？离 B 实测有多远？
      注意必须同时固定任务数 —— 头部覆盖本身随任务数增长，不对齐就会把规模效应
      算进密度的账上。

  R3  第二遍 fact list（把「无法区分因果」变成结论）
      审稿人：收集第二遍的 required_facts 能否诊断「多事实分解」是翻转的原因
      还是共病。§6 现在写的是「我们没记第二遍，所以无法区分」。
      之前做不了的真实原因是 _fmt(shuffle=True) 用的是**没播种**的全局
      random.shuffle，原来那次的打乱顺序无法离线复现。
      做法：不复现旧顺序，而是用两个**各自播种**的新顺序重判两遍，两遍都记
      fact list。决定性的切分是：在两遍 n_facts **相同**的对里，n=1 与 n>=2
      的翻转率各是多少 —— 若此时仍有阶跃，分解是原因；若差异消失，是共病。
      这一项要判官 GPU。

  R4  k 敏感性（修一处逻辑自洽问题）
      审稿人：k=10 可能不泛化。这条不只是缺 ablation —— 我们自己把可替代性
      定义成 (k, corpus, model) 三元组的性质，然后只测了一个 k。
      做法分两层，默认只跑免费那层：
        A 层（不要 LLM）：用 s1/chunk_emb.npy + s3/query_emb.npy 重新检索
          k ∈ {5,10,20,50}，量 top-k 集合怎么变、候选对的检索 Jaccard 怎么变，
          以及**集合层面的包含损失**（q_i 的 top-k 有多少落在 q_j 的 top-k 里）
          —— 后者正是保证式里那个量的检索侧代理。标签固定在 k=10，必须明说。
        B 层（--r4-judge，要判官）：在分层抽样上按各个 k 重判自覆盖。

用法
----
    # 三个离线项（不用判官，和 K1/K2 同一台机器即可）
    python review_fixes.py --config $R/config.json --exp R1,R2,R4

    # 要判官的那项（会起 vLLM，先确认 GPU 空着）
    python review_fixes.py --config $R/config.json --exp R3

    # 全部
    python review_fixes.py --config $R/config.json --exp R1,R2,R3,R4 --r4-judge

常用开关
    --r1-seeds 5            R1 跟 K1 一样给多种子 ±
    --r2-seeds 5            R2 每个密度点的重复次数
    --r3-n 600              R3 判多少对（每对判 2 次，所以是 2x 的调用量）
    --r4-ks 5,10,20,50      R4 扫哪些 k
    --r4-judge              额外跑 R4 的 B 层（判官）
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import importlib.util as _ilu


def _load(name: str, fname: str):
    spec = _ilu.spec_from_file_location(name, str(_HERE / fname))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p2x = _load("p2x", "phase2_experiments.py")
fg = _load("fgx", "fill_gaps.py")          # 复用 K1 的 cache_vs_view 做交叉校验

Phase2, RunData = p2x.Phase2, p2x.RunData
rjsonl, wjson, rjson = p2x.rjsonl, p2x.wjson, p2x.rjson
fmtf, fmtp = p2x.fmtf, p2x.fmtp
ms = fg.ms


# ==============================================================================
#  R1 的规范化阶梯
# ==============================================================================

# 内联停用词表：容器与集群都可能没有外网，绝不依赖 nltk 下载
_STOP = set("""a an the of in on at to for from by with about into over after before
between through during without within and or but if then than that this these those
is are was were be been being am do does did doing have has had having
i me my we our you your he him his she her it its they them their what which who whom
whose when where why how all any both each few more most other some such no nor not
only own same so too very can will just should now""".split())

_ART = re.compile(r"\b(?:a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _lemma_backend():
    """优先用真词形还原器；拿不到就退回到有记录的后缀规则，并把用了哪个写进结果。
    绝不静默降级 —— 审稿人问「你们的 lemmatizer 是什么」时要答得出来。"""
    try:
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        wnl.lemmatize("tests")                      # 触发语料检查，缺就抛
        return "nltk-wordnet", lambda w: wnl.lemmatize(w)
    except Exception:
        pass
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return "spacy-en_core_web_sm", lambda w: nlp(w)[0].lemma_
    except Exception:
        pass

    def suffix(w: str) -> str:
        for a, b in (("ies", "y"), ("sses", "ss"), ("ches", "ch"), ("shes", "sh"),
                     ("xes", "x"), ("ing", ""), ("ed", ""), ("s", "")):
            if len(w) > len(a) + 2 and w.endswith(a):
                return w[: len(w) - len(a)] + b
        return w
    return "builtin-suffix-rules", suffix


LEMMA_NAME, _LEMMA = _lemma_backend()


def _tok(s: str) -> list:
    return _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip().split()


def _simhash(tokens, bits: int = 64) -> int:
    """词级 bigram shingle 上的 SimHash。"""
    if not tokens:
        return 0
    sh = [" ".join(tokens[i:i + 2]) for i in range(max(1, len(tokens) - 1))] or tokens
    v = [0] * bits
    for g in sh:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big")
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(bits):
        if v[b] > 0:
            out |= 1 << b
    return out


# ------------------------------------------------------------------ 匹配器接口
# 每一级缓存都是一个匹配器：fit(建集查询) 之后 hit(评估查询) 回答「命中否」。
# 精确键用集合；模糊哈希要在汉明球内查近邻，没法用 dict，所以统一成这个接口。

class KeySet:
    """精确键：把查询规范化成一个字符串，命中 = 该键在建集里出现过。"""

    def __init__(self, keyfn):
        self.k = keyfn
        self.s = set()

    def fit(self, queries):
        self.s = {self.k(q) for q in queries}
        return self

    def hit(self, q) -> bool:
        return self.k(q) in self.s

    def n_keys(self) -> int:
        return len(self.s)


class SimHashBall:
    """模糊哈希：命中 = 建集中存在某个键与之汉明距离 <= d。

    用分带 LSH 找候选，**不是**近似：64 位切成 4 个 16 位的带，若两个键的
    汉明距离 <= 3，则至多 3 个带里有差异位，鸽巢原理保证至少有一个带完全相同，
    因此一定会被候选集捕获。所以 d <= 3 时这个实现没有漏检。
    d >= 4 就不再成立 —— 要用更多带，这里直接拒绝。
    """

    def __init__(self, keyfn, d: int, bits: int = 64, bands: int = 4):
        if d >= bands:
            raise ValueError(f"d={d} 需要 >{d} 个带才能保证不漏检，当前 bands={bands}")
        self.k, self.d, self.bits, self.bands = keyfn, d, bits, bands
        self.w = bits // bands
        self.mask = (1 << self.w) - 1
        self.idx = [defaultdict(list) for _ in range(bands)]
        self.vals = []

    def _h(self, q) -> int:
        return _simhash([_LEMMA(w) for w in _tok(q) if w not in _STOP], self.bits)

    def fit(self, queries):
        seen = set()
        for q in queries:
            h = self._h(q)
            if h in seen:
                continue
            seen.add(h)
            i = len(self.vals)
            self.vals.append(h)
            for b in range(self.bands):
                self.idx[b][(h >> (b * self.w)) & self.mask].append(i)
        return self

    def hit(self, q) -> bool:
        h = self._h(q)
        for b in range(self.bands):
            for i in self.idx[b].get((h >> (b * self.w)) & self.mask, ()):
                if bin(h ^ self.vals[i]).count("1") <= self.d:
                    return True
        return False

    def n_keys(self) -> int:
        return len(self.vals)


def build_normalizers(P):
    """返回 [(名字, 说明, key 函数)]，逐级加强。
    L1 必须与 phase2_experiments.norm_text 完全一致 —— 它是已发表的那个基线。"""
    def L0(s): return s or ""

    def L1(s):                                      # = 已发表的 normalized cache
        return _WS.sub(" ", _PUNCT.sub(" ", (s or "").lower())).strip()

    def L2(s):                                      # 再去冠词（对齐我们自己的 _norm）
        return _WS.sub(" ", _ART.sub(" ", L1(s))).strip()

    def L3(s):                                      # 去停用词
        return " ".join(w for w in _tok(s) if w not in _STOP)

    def L4(s):                                      # 停用词 + 词形还原
        return " ".join(_LEMMA(w) for w in _tok(s) if w not in _STOP)

    def L5(s):                                      # 再做词序无关（排序后的词袋）
        return " ".join(sorted(_LEMMA(w) for w in _tok(s) if w not in _STOP))

    def L6(s):                                      # 实体/模板规范键（复用已验证的抽取器）
        e = sorted(P._entity_set(s or ""))
        t = sorted(w for w in P._template_set(s or "") if w not in _STOP)
        return "E:" + "|".join(e) + "##T:" + "|".join(t)

    def L7(s):                                      # SimHash 汉明 0
        return f"H:{_simhash([_LEMMA(w) for w in _tok(s) if w not in _STOP]):016x}"

    # 注意：这条阶梯**不是严格嵌套**的。L6 走的是实体/模板那条轴，SimHash 建在
    # 二元 shingle 上因而对词序敏感，都不是 L5 的超集。所以 R1 报「最强缓存」时取
    # 的是各级净增量的**最小值**，而不是假定最后一级最强。
    #
    # L7 的教训（第一版踩过）：汉明距离 0 的 SimHash 等价于对 L4 那个有序 token
    # 串再哈希一次，不会把任何两个不同的键并到一起，实测与 L4 逐项相同 —— 它根本
    # 不是模糊匹配。真正的模糊哈希必须在一个**汉明球**里查近邻，就是下面 L8/L9。
    # 这两级有可能把不该并的查询并掉（SimHash 在 shingle 很少时熵低），所以
    # --selftest 会把它们一起检查：并掉了「不同属性/不同人」那两对就是不可用，
    # 必须照实报告，不能当成「更强的缓存」。
    return [
        ("L0_exact", "原始字符串（精确缓存）", lambda: KeySet(L0)),
        ("L1_published", "小写+标点+空白（= 论文里的 normalized cache）",
         lambda: KeySet(L1)),
        ("L2_articles", "L1 + 去冠词", lambda: KeySet(L2)),
        ("L3_stopwords", "去停用词", lambda: KeySet(L3)),
        ("L4_lemma", f"去停用词 + 词形还原（{LEMMA_NAME}）", lambda: KeySet(L4)),
        ("L5_bagsorted", "L4 + 词序无关（排序词袋）", lambda: KeySet(L5)),
        ("L6_entity_tpl", "实体集 + 模板词集 的规范键（W5H2 式）", lambda: KeySet(L6)),
        ("L7_simhash_h0", "64 位 SimHash 汉明 0（= L4 的哈希，非模糊匹配）",
         lambda: KeySet(L7)),
        ("L8_simhash_h2", "64 位 SimHash 汉明 <=2（真模糊匹配）",
         lambda: SimHashBall(L7, 2)),
        ("L9_simhash_h3", "64 位 SimHash 汉明 <=3（真模糊匹配，更松）",
         lambda: SimHashBall(L7, 3)),
    ]


def selftest_ladder(P=None) -> int:
    """手工用例上的阶梯自检：不读数据、不要 GPU，几秒钟出结果。
    最后两对是**必须永远不合并**的 —— 不同属性、不同人。任何一级把它们并成
    同一个缓存键，那不是「更强的规范化」，而是规范化错了，R1 的结论会反过来。"""
    if P is None:
        class P:                                  # 轻量替身，避免为自检加载 phase1
            @staticmethod
            def _entity_set(s):
                return {w for w in re.findall(r"[A-Za-z0-9']+", s or "")
                        if w[:1].isupper()}
            @staticmethod
            def _template_set(s):
                return {w.lower() for w in re.findall(r"[A-Za-z']+", s or "")
                        if not w[:1].isupper()}
    cases = [
        ("只差大小写", "date of birth of Ada Lovelace", "Date of Birth of Ada Lovelace", True),
        ("只差冠词", "the father of John Adams", "father of John Adams", True),
        ("停用词", "who is the director of Alien", "director of Alien", True),
        ("词序", "children of Chiang Ching-kuo", "Chiang Ching-kuo children", True),
        ("词形", "Tony Award win", "Tony Award wins", True),
        ("★不同属性", "date of birth of Ada Lovelace",
         "place of birth of Ada Lovelace", False),
        ("★不同的人", "father of John, Count of Brienne",
         "father of John I of Brienne, Count of Eu", False),
    ]
    norms = build_normalizers(P)
    print(f"词形还原后端：{LEMMA_NAME}\n")
    print(f"{'级别':<18}" + "".join(f"{i+1:>4}" for i in range(len(cases))))
    bad = []
    for name, _, mk in norms:
        row = ""
        for lbl, a, b, should in cases:
            # 建集只放 a，再问 b 是否命中 —— 精确键与汉明球都走同一个接口
            eq = mk().fit([a]).hit(b)
            row += f"{'=' if eq else '.':>4}"
            if eq and not should:
                bad.append((name, lbl))
        print(f"{name:<18}{row}")
    print("\n列号对应：" + "，".join(f"{i+1}={c[0]}" for i, c in enumerate(cases)))
    print("= 表示该级别把这一对并成同一个缓存键")
    if bad:
        print("\n★ 以下级别把「必须不合并」的对并掉了 —— 该级别不是「更强的缓存」，"
              "而是错的：")
        for n, l in bad:
            print(f"    {n}  <-  {l}")
        print("R1 仍会跑并落盘，但这些级别的净增量**不可**当作防线报告；"
              "论文里要说明它们在手工用例上就已失效。")
        return 1
    print("\n通过：带 ★ 的两对在所有级别都没有被并掉")
    return 0


# ==============================================================================
#  R1：更强的精确缓存基线
# ==============================================================================

def _ladder_once(P, cfg, md, seed: int, targets, norms, log=None) -> dict:
    """与 fill_gaps.cache_vs_view 同构，只把缓存键换成一条阶梯。
    视图侧（判定器、阈值、DSU、质心打分）一字不改，保证可比。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    calls, qidx, E = md.calls, md.qidx, md.E
    cmap = {c["qid"]: c for c in calls}

    X, y, W_ = [], [], []
    for r in md.l1:
        ci, cj = cmap.get(r["qid_i"]), cmap.get(r["qid_j"])
        if not ci or not cj:
            continue
        X.append([r["cos"],
                  P._jac(P._entity_set(ci["query"]), P._entity_set(cj["query"])),
                  P._jac(P._template_set(ci["query"]), P._template_set(cj["query"]))])
        y.append(r["substitutable"]); W_.append(r.get("w", 1.0))
    X = np.array(X, float); y = np.array(y, int); W_ = np.array(W_, float)
    sk = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sk.transform(X), y, sample_weight=W_)
    sc_all = lr.predict_proba(sk.transform(X))[:, 1]
    thr = {}
    for tg in targets:
        best = None
        for t in np.quantile(sc_all, np.linspace(0.0, 0.998, 300)):
            m = sc_all >= t
            if int(m.sum()) < 50:
                continue
            _, lo, _, _ = P.wilson_w(y[m], W_[m])
            if lo >= tg and (best is None or t < best):
                best = float(t)
        thr[tg] = best

    tasks = sorted({c["task_id"] for c in calls})
    rnd = random.Random(seed); rnd.shuffle(tasks)
    build_t = set(tasks[: int(len(tasks) * cfg.view_eval_task_frac)])
    build = [c for c in calls if c["task_id"] in build_t and c["qid"] in qidx]
    evalq = [c for c in calls if c["task_id"] not in build_t and c["qid"] in qidx]

    cost = np.array([c.get("cost", 1.0) for c in evalq], float)
    srcs = np.array([c["source"] for c in evalq])
    tot = cost.sum()

    # ---- 缓存阶梯 ----
    bq = [c["query"] for c in build]
    cache = {}
    for name, desc, mk in norms:
        m = mk().fit(bq)
        hit = np.array([m.hit(c["query"]) for c in evalq])
        cache[name] = {"desc": desc, "hit_q": float(hit.mean()),
                       "hit_cost": float(cost[hit].sum() / tot),
                       "n_keys_build": m.n_keys(), "_hit": hit}

    # ---- 视图侧：与 s23 / K1 完全一致 ----
    tau = md.calib.get("tau_star") or 0.90
    bset = {c["qid"] for c in build}
    dsu = P._DSU([c["qid"] for c in build])
    for p in md.pairs_raw:
        if p["cos"] >= tau and p["qid_i"] in bset and p["qid_j"] in bset:
            dsu.union(p["qid_i"], p["qid_j"])
    grp = defaultdict(list)
    for c in build:
        grp[dsu.find(c["qid"])].append(c["qid"])
    views = [(k, v) for k, v in grp.items() if len(v) >= 2]
    if not views:
        return {"error": "no views", "seed": seed}
    cen = np.stack([E[[qidx[q] for q in v]].mean(0) for _, v in views])
    cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    reps = [cmap[v[0]]["query"] for _, v in views]
    rep_ent = [P._entity_set(t) for t in reps]
    rep_tpl = [P._template_set(t) for t in reps]
    ev_idx = [qidx[c["qid"]] for c in evalq]
    sims = E[ev_idx] @ cen.T
    bv = np.argmax(sims, axis=1); bs = sims[np.arange(len(ev_idx)), bv]
    feats = np.zeros((len(evalq), 3))
    for i, c in enumerate(evalq):
        j = int(bv[i])
        feats[i] = [float(bs[i]),
                    P._jac(P._entity_set(c["query"]), rep_ent[j]),
                    P._jac(P._template_set(c["query"]), rep_tpl[j])]
    sc = lr.predict_proba(sk.transform(feats))[:, 1]

    out = {"seed": seed, "n_eval": len(evalq), "n_views": len(views), "tau": tau,
           "cache": {k: {kk: vv for kk, vv in v.items() if kk != "_hit"}
                     for k, v in cache.items()},
           "views": {}, "increment_vs": {}}
    for tg in targets:
        t = thr.get(tg)
        key = f"{tg:.2f}"
        if t is None:
            out["views"][key] = None
            continue
        hv = sc >= t
        out["views"][key] = {"hit_q": float(hv.mean()),
                             "hit_cost": float(cost[hv].sum() / tot)}
        per = {}
        for name, _, _ in norms:
            hc = cache[name]["_hit"]
            inc = hv & (~hc)
            per[name] = {
                "cost_frac": float(cost[inc].sum() / tot),
                "by_source": {s: float(cost[(srcs == s) & inc].sum()
                                       / max(cost[srcs == s].sum(), 1e-9))
                              for s in sorted(set(srcs.tolist()))}}
        out["increment_vs"][key] = per
    return out


def exp_R1(p2: Phase2) -> dict:
    def run():
        log = p2.log
        P, cfg, md = fg._prep(p2)
        targets = md.calib.get("report_targets") or [0.80, 0.85, 0.90, 0.95]
        norms = build_normalizers(P)
        log.info(f"  词形还原后端：{LEMMA_NAME}")
        if LEMMA_NAME == "builtin-suffix-rules":
            log.warning("    没装 nltk/spacy，用的是内置后缀规则；"
                        "论文里必须照实写，别说成 lemmatization")

        # ---- 守门：L1 必须复刻已发表的 normalized cache ----
        log.info(f"  先用原种子 {cfg.seed} 核对 L1 是否复刻已发表的 18.8%")
        base = _ladder_once(P, cfg, md, cfg.seed, targets, norms, log)
        if "error" in base:
            raise RuntimeError(base["error"])
        ref = fg.cache_vs_view(P, cfg, md, cfg.seed, targets, log)
        ref.pop("_arrays", None)
        d_cache = abs(base["cache"]["L1_published"]["hit_cost"] - ref["cache_norm_cost"])
        d_view = abs((base["views"].get("0.85") or {}).get("hit_cost", 0)
                     - (ref["views"].get("0.85") or {}).get("hit_cost", 0))
        ok = d_cache < 1e-6 and d_view < 1e-6
        log.info(f"    L1 缓存 {fmtp(base['cache']['L1_published']['hit_cost'])} vs "
                 f"K1 口径 {fmtp(ref['cache_norm_cost'])} | "
                 f"视图@0.85 {fmtp((base['views'].get('0.85') or {}).get('hit_cost'))} vs "
                 f"{fmtp((ref['views'].get('0.85') or {}).get('hit_cost'))}  ->  "
                 f"{'一致' if ok else '★不一致'}")
        if not ok:
            raise RuntimeError(f"L1 未能复刻已发表口径（Δcache={d_cache:.2e} "
                               f"Δview={d_view:.2e}），阶梯结果不可用")

        seeds = [cfg.seed] + [cfg.seed + 1000 * i for i in range(1, p2.a.r1_seeds)]
        runs = [base]
        for s in seeds[1:]:
            log.info(f"  种子 {s} …")
            r = _ladder_once(P, cfg, md, s, targets, norms, log)
            if "error" not in r:
                runs.append(r)
            p2.beat(f"R1 seed {s}")

        key = "0.85"
        vh = ms([(r["views"].get(key) or {}).get("hit_cost") for r in runs])
        log.info("")
        log.info(f"  视图@{key} 命中(成本) {vh[0]*100:.1f} ± {vh[1]*100:.1f}"
                 f"（{len(runs)} 个种子）")
        log.info(f"  {'缓存级别':<16}{'命中(成本)':>14}{'净增量':>14}"
                 f"{'agent 净增':>14}{'replay 最高':>14}")
        agg = {"n_seeds": len(runs), "lemma_backend": LEMMA_NAME,
               "replication_check": {"delta_cache": d_cache, "delta_view": d_view,
                                     "match": ok},
               "view_hit_cost": {key: vh}, "ladder": {}}
        for name, desc, _ in norms:
            hc = ms([r["cache"][name]["hit_cost"] for r in runs])
            inc = ms([((r["increment_vs"].get(key) or {}).get(name) or {}).get("cost_frac")
                      for r in runs])
            bysrc = {}
            srcset = sorted({s for r in runs
                             for s in (((r["increment_vs"].get(key) or {}).get(name) or {})
                                       .get("by_source") or {})})
            for s in srcset:
                bysrc[s] = ms([(((r["increment_vs"].get(key) or {}).get(name) or {})
                                .get("by_source") or {}).get(s) for r in runs])
            rep = [v[0] for k_, v in bysrc.items() if k_.startswith("replay") and v[0] is not None]
            agg["ladder"][name] = {"desc": desc, "cache_hit_cost": hc,
                                   "increment": inc, "by_source": bysrc}
            ag = bysrc.get("agent", (None, None))
            cs = f"{hc[0]*100:.1f} ± {hc[1]*100:.1f}" if hc[0] is not None else "-"
            is_ = f"{inc[0]*100:.1f} ± {inc[1]*100:.1f}" if inc[0] is not None else "-"
            log.info(f"  {name:<16}{cs:>14}{is_:>14}"
                     f"{(f'{ag[0]*100:.1f}' if ag[0] is not None else '-'):>14}"
                     f"{(f'{max(rep)*100:.1f}' if rep else '-'):>14}")

        l1i = agg["ladder"]["L1_published"]["increment"][0]
        # 某一级可能没有可用值（比如该种子在 0.85 上无可行阈值），先筛掉再取最小，
        # 否则 min() 会拿 None 去比较直接崩掉
        usable = [n for n, _, _ in norms
                  if n != "L0_exact" and agg["ladder"][n]["increment"][0] is not None]
        if l1i is None or not usable:
            raise RuntimeError("0.85 上没有可用的净增量，阶梯无法比较；"
                               "先确认该 run 在 0.85 有可行阈值")
        best_name = min(usable, key=lambda n: agg["ladder"][n]["increment"][0])
        besti = agg["ladder"][best_name]["increment"][0]
        closed = l1i - besti
        # 判据：相对最强缓存，视图是否还有可观净增量，且分域格局是否还在
        bs = agg["ladder"][best_name]["by_source"]
        ag = (bs.get("agent") or (None,))[0]
        rep = [v[0] for k_, v in bs.items() if k_.startswith("replay") and v[0] is not None]
        still = (ag is not None and rep and ag > max(rep))
        agg["verdict"] = {
            "published_increment": l1i, "strongest_cache": best_name,
            "strongest_increment": besti, "points_closed": closed,
            "frac_closed": closed / max(l1i, 1e-9),
            "agent_still_highest": bool(still)}
        log.info("")
        log.info(f"  最强缓存是 {best_name}：把净增量从 {l1i*100:.1f}pp 压到 "
                 f"{besti*100:.1f}pp，吃掉 {closed*100:.1f}pp"
                 f"（{closed/max(l1i,1e-9):.0%}）")
        if still:
            log.info("  相对最强缓存，agent 域净增量仍高于所有 replay 域 —— 中心论点保住")
        else:
            log.warning("  ★ 相对最强缓存，分域格局不再成立。"
                        "§5 的核心表述必须改口径，不能只报 L1 的 8.6pp")
        (p2.root / "R").mkdir(parents=True, exist_ok=True)
        agg["per_seed"] = runs
        wjson(p2.root / "R" / "r1_cache_ladder.json", agg)
        return {"summary": f"最强缓存 {best_name} 吃掉 {closed*100:.1f}pp"
                           f"（净增量 {l1i*100:.1f} -> {besti*100:.1f}）"
                           f"{'；分域格局保住' if still else '；★分域格局被打破'}"}
    return p2.run_step("R1", "更强的精确缓存基线（规范化阶梯）", run)


# ==============================================================================
#  R2：密度对照
# ==============================================================================

def _thin(calls, frac: float, policy: str, seed: int):
    """在任务内部减少调用数，把密度压到 frac 倍。每个任务至少留 1 条。
      random  —— 任务内随机丢
      prefix  —— 只留每个任务最前面的若干条（模拟更早停的 agent）
      firstq  —— 只留每个多查询调用的第 i 条（复刻 s18 的 arm 口径）
    """
    by = defaultdict(list)
    for c in calls:
        by[c["task_id"]].append(c)
    out = []
    rnd = random.Random(seed)
    for t, cs in by.items():
        if policy == "firstq":
            keep = [c for c in cs if c.get("multi_query_idx", 0) == 0]
            out.extend(keep or cs[:1])
            continue
        n = max(1, int(round(len(cs) * frac)))
        if policy == "prefix":
            out.extend(cs[:n])
        else:
            out.extend(rnd.sample(cs, n) if n < len(cs) else cs)
    return out


def exp_R2(p2: Phase2) -> dict:
    def run():
        log = p2.log
        P, cfg, md = fg._prep(p2)
        tau = md.calib.get("tau_star") or 0.90
        pairs = md.pairs_raw
        ag = [c for c in md.calls if c["source"] == "agent"]
        by_task = defaultdict(list)
        for c in ag:
            by_task[c["task_id"]].append(c)
        all_tasks = sorted(by_task)
        log.info(f"  agent 全量：{len(ag)} 调用 / {len(all_tasks)} 任务 = "
                 f"{len(ag)/max(len(all_tasks),1):.2f} 次/任务")

        # ---- 等任务量：用两个 agent 的任务**交集**，不是随机抽样 ----
        # 这里踩过两次坑，记下来：
        #  第一版用自己的随机 997，A 基线 25.13% 而已发表是 25.9%，归因算在错基线上。
        #  第二版改去复刻 s20，但 s20 的 m 是「各工作负载任务数的最小值」= 1888，
        #  对 agent 而言等于不裁剪（A 全量 28.7%），却仍拿 997 个任务上的 B(17.8%)
        #  做参照 —— 任务数没对齐，而头部覆盖本身随任务数增长，差距里混进了规模效应。
        # 正确的对照是论文里那个跨 agent 比较：两个 agent 都跑完的那批任务。
        # 它是交集，完全确定，不含随机性。
        brun = Path(p2.a.r2_b_run) if p2.a.r2_b_run else None
        b_calls = b_pairs = None
        if brun and (brun / "stages" / "s2" / "calls.jsonl").exists():
            bc = rjsonl(brun / "stages" / "s2" / "calls.jsonl")
            b_calls = [c for c in bc if c["source"] == p2.a.r2_b_source]
            b_pairs = rjsonl(brun / "stages" / "s4" / "pairs.jsonl")
            log.info(f"  B 组（{p2.a.r2_b_source}）{len(b_calls)} 调用，"
                     f"来自 {brun}")
        else:
            log.warning(f"  找不到 B 组 run（{p2.a.r2_b_run}），"
                        "只能在 A 上给密度曲线，归因一节会跳过")

        if b_calls:
            b_by_task = defaultdict(list)
            for c in b_calls:
                b_by_task[c["task_id"]].append(c)
            common = sorted(set(by_task) & set(b_by_task))
            log.info(f"  任务交集 {len(common)}（A {len(all_tasks)} ∩ B {len(b_by_task)}）")
            if not common:
                raise RuntimeError("两个 run 的 task_id 没有交集，无法做等任务量对照；"
                                   "确认 agent_b 的 task_id 与主 run 同一套编号")
            draws = [common]
        else:
            draws = [all_tasks]

        def cov(task_ids, thin_frac=1.0, policy="random", seed=0):
            sub = [c for t in task_ids for c in by_task[t]]
            if thin_frac < 1.0 or policy == "firstq":
                sub = _thin(sub, thin_frac, policy, seed)
            r = P._head_cov(sub, pairs, tau, cfg.stop_head_frac)
            return (r["head_cov_cost"] if r else None,
                    len(sub) / max(len(task_ids), 1), len(sub))

        base = [cov(d) for d in draws]
        full_head = float(np.mean([b[0] for b in base]))
        dens0 = float(np.mean([b[1] for b in base]))
        log.info(f"  A 在交集上的头部覆盖 {fmtp(full_head)}，密度 {dens0:.2f} 次/任务")

        # B 在同一批任务上，用 B 自己的候选对图与同一个 τ*
        bhead = p2.a.r2_agent_b_head
        bdens = p2.a.r2_target_density
        if b_calls:
            bsub = [c for t in draws[0] for c in b_by_task[t]]
            rb = P._head_cov(bsub, b_pairs, tau, cfg.stop_head_frac)
            if rb:
                bhead = rb["head_cov_cost"]
                bdens = len(bsub) / max(len(draws[0]), 1)
                log.info(f"  B 在同一批任务上的头部覆盖 {fmtp(bhead)}，"
                         f"密度 {bdens:.2f} 次/任务")

        # ---- 在交集上重算 §4 那张结构对比表的全部字段 ----
        # 论文 §4 声称 cross-task intents / tasks-per-intent 也是「on the matched set」，
        # 但已发表的 A 侧其实是 1888 里随机抽 997（7,567 调用），不是 B 跑的那批任务。
        # 只改头部覆盖会让那段自相矛盾，所以整段在同一个集合上重算。
        def mono_head_share(calls_sub, prs):
            """头部意图簇中「成员查询文本完全相同」的簇所占的调用份额。"""
            qids = [c["qid"] for c in calls_sub]
            if not qids:
                return None
            dsu = P._DSU(qids); qs = set(qids)
            for p_ in prs:
                if p_["cos"] >= tau and p_["qid_i"] in qs and p_["qid_j"] in qs:
                    dsu.union(p_["qid_i"], p_["qid_j"])
            grp = defaultdict(list)
            for c in calls_sub:
                grp[dsu.find(c["qid"])].append(c)
            order = sorted(grp.items(), key=lambda kv: -len(kv[1]))
            hn = max(1, int(round(len(order) * cfg.stop_head_frac)))
            head = order[:hn]
            tot = sum(len(v) for _, v in head)
            mono = sum(len(v) for _, v in head
                       if len({c["query"] for c in v}) == 1)
            return mono / max(tot, 1)

        struct = {}
        if b_calls:
            for tag, cs, prs in (("A", [c for t in draws[0] for c in by_task[t]], pairs),
                                 ("B", [c for t in draws[0] for c in b_by_task[t]], b_pairs)):
                r = P._head_cov(cs, prs, tau, cfg.stop_head_frac) or {}
                nsub = sum(1 for c in cs if c.get("multi_query_idx", 0) > 0)
                struct[tag] = {
                    "n_tasks": r.get("n_tasks"), "n_calls": r.get("n_calls"),
                    "calls_per_task": r.get("n_calls", 0) / max(r.get("n_tasks") or 1, 1),
                    "n_intents": r.get("n_intents"),
                    "head_cov_cost": r.get("head_cov_cost"),
                    "head_cov_calls": r.get("head_cov_calls"),
                    "cross_intent_frac": r.get("cross_intent_frac"),
                    "head_cross_share": r.get("head_cross_share"),
                    "mean_tasks_per_intent": r.get("mean_tasks_per_intent"),
                    "mono_head_share": mono_head_share(cs, prs),
                    "non_first_subquery_frac": nsub / max(len(cs), 1)}
            log.info("")
            log.info("  交集上的结构对比（§4 那张表应当全部用这一列）")
            log.info(f"  {'量':<26}{'A (Tongyi)':>14}{'B (Qwen3-32B)':>16}")
            for k, lbl, f in (
                    ("n_tasks", "任务数", lambda v: f"{v:d}"),
                    ("n_calls", "检索调用", lambda v: f"{v:d}"),
                    ("calls_per_task", "每任务调用数", lambda v: f"{v:.2f}"),
                    ("n_intents", "意图数", lambda v: f"{v:d}"),
                    ("head_cov_cost", "头部覆盖(成本)", lambda v: f"{v*100:.1f}%"),
                    ("head_cov_calls", "头部覆盖(按次)", lambda v: f"{v*100:.1f}%"),
                    ("cross_intent_frac", "跨任务意图占比", lambda v: f"{v*100:.1f}%"),
                    ("mean_tasks_per_intent", "每意图任务数", lambda v: f"{v:.2f}"),
                    ("mono_head_share", "单文本头部簇", lambda v: f"{v*100:.1f}%"),
                    ("non_first_subquery_frac", "非首条子查询占比", lambda v: f"{v*100:.1f}%")):
                va, vb = struct["A"].get(k), struct["B"].get(k)
                log.info(f"  {lbl:<26}{(f(va) if va is not None else '-'):>14}"
                         f"{(f(vb) if vb is not None else '-'):>16}")

        # 复刻核对：A、B 都要对上论文已发表的等量数，否则归因不可信
        rep = {"a": None, "b": None}
        for tag, got, want in (("a", full_head, p2.a.r2_agent_a_head),
                               ("b", bhead, p2.a.r2_agent_b_head)):
            if want:
                d_ = abs(got - want)
                rep[tag] = {"published": want, "replicated": got, "delta": d_,
                            "match": bool(d_ < 0.004)}
                log.info(f"    {tag.upper()} 对已发表的 {fmtp(want)} -> "
                         f"{'一致' if rep[tag]['match'] else '★不一致'}（Δ={d_*100:.2f}pp）")
        if any(v and not v["match"] for v in rep.values()):
            log.warning("    与已发表的等量对照不一致。已知的一种可能："
                        "已发表的 A 侧是从 A 全部任务里**随机抽**到与 B 相同的任务数"
                        "（论文 §4 写的 7,567 调用 ≈ 全量 x 997/1888），"
                        "只对齐了任务数而非任务身份；本节用的是任务交集，"
                        "口径更严但数会更低。两者必须二选一，不能混着引用。")
            rep["note"] = ("A 与 B 的已发表等量数可能来自「只对齐任务数」的随机子集；"
                           "本节用任务交集，口径更严")

        target = bdens          # B 在同一批任务上的实测密度，不用命令行的缺省值
        fracs = sorted({round(x, 4) for x in
                        [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25,
                         max(0.05, min(1.0, target / max(dens0, 1e-9)))]},
                       reverse=True)
        log.info(f"  目标密度 {target}（B 的实测值）对应保留比例 "
                 f"{target/max(dens0,1e-9):.3f}")
        log.info("")
        log.info(f"  {'策略':<8}{'保留比例':>10}{'实测密度':>12}{'调用数':>10}"
                 f"{'头部覆盖(成本)':>18}")

        rows = []
        for policy in ("random", "prefix"):
            for fr in fracs:
                vals, dens, ncs = [], [], []
                nseed = p2.a.r2_seeds if policy == "random" else 1
                for d in draws:                       # 目前只有一个抽样（任务交集）
                    for s in range(nseed):
                        h, dn, nc = cov(d, fr, policy, 7000 + s)
                        if h is not None:
                            vals.append(h)
                        dens.append(dn); ncs.append(nc)
                mm, sd = ms(vals)
                row = {"policy": policy, "frac": fr, "density": float(np.mean(dens)),
                       "n_calls": int(np.mean(ncs)), "head_cov_cost": mm, "sd": sd,
                       "n_reps": len(vals)}
                rows.append(row)
                log.info(f"  {policy:<8}{fr:>10.3f}{row['density']:>12.2f}"
                         f"{row['n_calls']:>10}"
                         f"{f'{mm*100:.1f} ± {sd*100:.1f}' if mm is not None else '-':>18}")
                p2.beat(f"R2 {policy} {fr}")

        # s18 的 arm 口径：只留每个多查询调用的首条。它同时**降低密度**和
        # **选择性保留**，所以不能直接拿来当密度实验 —— 单列一行，和同密度下的
        # random 插值对比，量出「丢哪些」本身值多少。
        af = [cov(d, 1.0, "firstq", 0) for d in draws]
        arm = {"policy": "firstq(s18 arm)",
               "density": float(np.mean([a[1] for a in af])),
               "n_calls": int(np.mean([a[2] for a in af])),
               "head_cov_cost": float(np.mean([a[0] for a in af]))}
        log.info(f"  {'firstq':<8}{'-':>10}{arm['density']:>12.2f}{arm['n_calls']:>10}"
                 f"{fmtp(arm['head_cov_cost']):>18}")

        # ---- 对数线性拟合：头部覆盖 ~ a + b*log10(密度) ----
        R = [r for r in rows if r["policy"] == "random" and r["head_cov_cost"]]
        fit = None
        if len(R) >= 3:
            x = np.log10([r["density"] for r in R]); y = np.array([r["head_cov_cost"] for r in R])
            b, a = np.polyfit(x, y, 1)
            pred = a + b * x
            r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
            fit = {"intercept": float(a), "slope_per_decade": float(b), "r2": float(r2)}
            log.info("")
            log.info(f"  对数线性拟合：头部覆盖 = {a:.4f} + {b:.4f}·log10(密度)，"
                     f"R²={r2:.3f}  —— 每十倍密度 {b*100:+.1f} 个点")
            # 选择性 vs 随机：同密度下 firstq 比 random 高多少
            xi = np.argsort([r["density"] for r in R])
            dx = np.array([R[i]["density"] for i in xi]); dy = np.array([R[i]["head_cov_cost"] for i in xi])
            r_at = float(np.interp(arm["density"], dx, dy))
            arm["random_interp_same_density"] = r_at
            arm["selectivity_bonus_pp"] = (arm["head_cov_cost"] - r_at) * 100
            log.info(f"  同密度({arm['density']:.2f})下 random 插值 {fmtp(r_at)}，"
                     f"firstq 高出 {arm['selectivity_bonus_pp']:+.2f} 个点 —— "
                     f"丢哪些查询和丢多少一样重要")

        # ---- 关键一问：把 A 压到 B 的密度，覆盖落到哪 ----
        at = min((r for r in rows if r["policy"] == "random" and r["head_cov_cost"]),
                 key=lambda r: abs(r["density"] - target), default=None)
        out = {"tau": tau, "matched_tasks": len(draws[0]), "n_draws": len(draws),
               "b_run": str(brun) if brun else None,
               "replication": rep, "density_full": dens0, "density_b": bdens,
               "struct_on_intersection": struct,
               "head_cov_full": full_head, "logfit": fit, "rows": rows,
               "arm_firstq": arm, "target_density": target,
               "agent_b_head_reference": bhead, "at_target": at}
        if at and bhead:
            gap_total = full_head - bhead
            gap_dens = full_head - at["head_cov_cost"]
            share = gap_dens / max(gap_total, 1e-9)
            resid = bhead - at["head_cov_cost"]
            out["attribution"] = {
                "gap_total_pp": gap_total * 100,
                "gap_explained_by_density_pp": gap_dens * 100,
                "share_explained": share,
                "residual_b_minus_a_at_equal_density_pp": resid * 100}
            log.info("")
            log.info(f"  A 等量 {fmtp(full_head)}（密度 {dens0:.2f}） -> 压到 "
                     f"{at['density']:.2f} 次/任务时 {fmtp(at['head_cov_cost'])}；"
                     f"B 实测 {fmtp(bhead)}（密度 {target:.2f}）")
            # share>1 不是「解释得更好」，而是密度**超额**解释：等密度下 A 反而比 B 低。
            # 第一版把这种情况打成「现有表述站得住」，读起来像通过了检查，是误导。
            if share >= 1.0:
                out["verdict"] = (
                    f"密度**超额**解释：把 A 压到 B 的密度后，A 只有 "
                    f"{at['head_cov_cost']*100:.1f}%，比 B 实测的 {bhead*100:.1f}% 还低 "
                    f"{-resid*100:.1f}pp。所以查询密度足以解释全部 "
                    f"{gap_total*100:.1f}pp 的差距；等密度下 B 反而更「每查询会复用」。"
                    "论文里 'consistent with query density' 成立，且可以说得更强；"
                    "但不能说成「密度解释了 100%」，正确表述是「密度足以解释，"
                    "残差方向相反」。")
            elif share < 0.5:
                out["verdict"] = (
                    f"★ 密度只能解释跨 agent 差距的 {share:.0%}。"
                    "摘要与结论里 'consistent with query density' 必须改弱。")
            else:
                out["verdict"] = (f"密度解释了跨 agent 差距的 {share:.0%}，"
                                  f"残差 {resid*100:+.1f}pp，现有表述站得住。")
            (log.warning if share < 0.5 else log.info)(f"  {out['verdict']}")
        wjson(p2.root / "R" / "r2_density_control.json", out)
        if fit:
            return {"summary": f"头部覆盖每十倍密度 {fit['slope_per_decade']*100:+.1f}pp "
                               f"(R²={fit['r2']:.3f})；" + (out.get("verdict") or "")[:60]}
        return {"summary": "密度曲线已出"}
    return p2.run_step("R2", "密度对照（隔离 calls-per-task）", run)


# ==============================================================================
#  R3：第二遍 fact list
# ==============================================================================

def exp_R3(p2: Phase2) -> dict:
    def run():
        log = p2.log
        P, a = p2.P, p2.a
        cfg = P.Config.load(a.config)
        cfg.run_id = "phase2_R3"
        cfg.root = "runs"
        cfg.resolve()
        ctx = P.RunContext(cfg)
        cfg._autotune = P.autotune_hardware(cfg, ctx.log)
        sm = P.ServerManager(ctx)
        p2.register_server(sm)

        md = RunData(a.main_run, log)
        l1 = md.l1
        cmap_q = {c["qid"]: c for c in md.calls}
        # 段落文本：和 s6 同一个来源
        chunks = rjsonl(Path(a.main_run) / "stages" / "s1" / "chunks.jsonl")
        ctext = {c["chunkid"]: c["text"] for c in chunks}

        # 分层抽样：重点在塌陷区，另取两个稳定层作对照
        bands = [(0.95, 0.98), (0.98, 0.999), (0.999, 1.001),
                 (0.70, 0.75), (0.85, 0.90)]
        per = max(20, a.r3_n // len(bands))
        pool_items = []
        for lo, hi in bands:
            cand = [r for r in l1 if lo <= r["cos"] < hi
                    and r["qid_i"] in cmap_q and r["qid_j"] in cmap_q]
            random.Random(31).shuffle(cand)
            for r in cand[:per]:
                pool_items.append((f"[{lo},{hi})", r))
        log.info(f"  抽 {len(pool_items)} 对，每对判 2 次（两个各自播种的呈现顺序）")
        log.info("  不复现旧的打乱顺序：s6 用的是未播种的全局 random.shuffle，无法复现")

        urls = sm.ensure("judge")
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)

        def fmt_seeded(ids, seed):
            o = list(ids)
            random.Random(seed).shuffle(o)
            return "\n\n".join(f"[{i+1}] {(ctext.get(c) or '')[:cfg.judge_doc_chars]}"
                               for i, c in enumerate(o)) or "(无结果)"

        def one(it):
            band, r = it
            ci = cmap_q[r["qid_i"]]; cj = cmap_q[r["qid_j"]]
            # 判的是「q_i 能否由 q_j 的检索结果回答」，与 s6 的方向一致
            ids = cj.get("retrieved_chunks") or []
            q = ci["query"]
            o = {}
            for tag, seed in (("p1", 101), ("p2", 202)):
                j = P._judge1(pool, cfg, q, fmt_seeded(ids, seed))
                o[tag] = {"ok": j.get("ok"), "verdict": j.get("verdict"),
                          "covered": j.get("covered"),
                          "facts": j.get("facts") or [], "found": j.get("found") or []}
            return {"band": band, "qid_i": r["qid_i"], "qid_j": r["qid_j"],
                    "cos": r["cos"], "label_s6": r.get("substitutable"), **o}

        res = P.pmap(one, pool_items, cfg.judge_concurrency, "R3 双判", ctx, pool)
        res = [r for r in res if r and r["p1"]["ok"] and r["p2"]["ok"]]
        log.info(f"  有效 {len(res)}/{len(pool_items)}")

        for r in res:
            r["n1"] = len(r["p1"]["facts"]); r["n2"] = len(r["p2"]["facts"])
            r["flip"] = int(r["p1"]["covered"] != r["p2"]["covered"])
            r["nfacts_same"] = int(r["n1"] == r["n2"])

        def rate(rows):
            return (float(np.mean([x["flip"] for x in rows])), len(rows)) if rows else (None, 0)

        # 1) fact list 本身稳不稳
        same = [r for r in res if r["nfacts_same"]]
        log.info("")
        log.info(f"  两遍 n_facts 相同的比例 {len(same)/max(len(res),1):.1%}"
                 f"（{len(same)}/{len(res)}）")

        # 2) 决定性切分：只在 n_facts 两遍一致的子集里比 n=1 vs n>=2
        one_f = [r for r in same if r["n1"] == 1]
        multi = [r for r in same if r["n1"] >= 2]
        f1, n1 = rate(one_f); fm, nm = rate(multi)
        log.info(f"  在 n_facts 两遍一致的子集中：n=1 翻转 "
                 f"{f1*100 if f1 is not None else float('nan'):.1f}% (n={n1})，"
                 f"n>=2 翻转 {fm*100 if fm is not None else float('nan'):.1f}% (n={nm})")

        # 3) 对照：n_facts 两遍不一致的那批
        diff = [r for r in res if not r["nfacts_same"]]
        fd, nd = rate(diff)
        log.info(f"  n_facts 两遍不一致的子集翻转 "
                 f"{fd*100 if fd is not None else float('nan'):.1f}% (n={nd})")

        by_band = {}
        for b in sorted({r["band"] for r in res}):
            rows = [r for r in res if r["band"] == b]
            s_ = [r for r in rows if r["nfacts_same"]]
            by_band[b] = {
                "n": len(rows), "flip": rate(rows)[0],
                "nfacts_stable_frac": len(s_) / max(len(rows), 1),
                "flip_one_fact": rate([r for r in s_ if r["n1"] == 1])[0],
                "flip_multi_fact": rate([r for r in s_ if r["n1"] >= 2])[0]}
        log.info("")
        log.info(f"  {'层':<16}{'n':>6}{'翻转':>9}{'n_facts 稳定':>14}"
                 f"{'n=1 翻转':>11}{'n>=2 翻转':>11}")
        for b, v in by_band.items():
            log.info(f"  {b:<16}{v['n']:>6}{fmtp(v['flip']):>9}"
                     f"{fmtp(v['nfacts_stable_frac']):>14}"
                     f"{fmtp(v['flip_one_fact']):>11}{fmtp(v['flip_multi_fact']):>11}")

        out = {"n_judged": len(res), "bands": bands, "by_band": by_band,
               "nfacts_stable_frac": len(same) / max(len(res), 1),
               "controlled": {"flip_one_fact": f1, "n_one_fact": n1,
                              "flip_multi_fact": fm, "n_multi_fact": nm},
               "nfacts_unstable": {"flip": fd, "n": nd},
               "per_pair": res}
        if f1 is not None and fm is not None:
            gap = fm - f1
            out["verdict"] = (
                f"在两遍 n_facts 一致的子集中，n>=2 的翻转率比 n=1 高 {gap*100:.1f}pp。"
                + ("分解本身与不稳定相关，不只是共病 —— §6 可以把「无法区分」改成"
                   "「在事实数稳定的子集中仍有阶跃」。" if gap > 0.1 else
                   "阶跃在控制了事实数稳定性后基本消失 —— §6 应当明确写成共病，"
                   "而不是把 n_facts 当成因。"))
            log.info("")
            log.info(f"  {out['verdict']}")
        (p2.root / "R").mkdir(parents=True, exist_ok=True)
        wjson(p2.root / "R" / "r3_second_pass_facts.json", out)
        return {"summary": (out.get("verdict") or "双判完成")[:110]}
    return p2.run_step("R3", "第二遍 fact list（判官，需要 GPU）", run)


# ==============================================================================
#  R4：k 敏感性
# ==============================================================================

def exp_R4(p2: Phase2) -> dict:
    def run():
        log = p2.log
        P, cfg, md = fg._prep(p2)
        run_p = Path(p2.a.main_run)
        ks = sorted({int(x) for x in p2.a.r4_ks.split(",") if x.strip()})
        kmax = max(ks)
        tau = md.calib.get("tau_star") or 0.90

        CE = np.load(run_p / "stages" / "s1" / "chunk_emb.npy")
        cids = [c["chunkid"] for c in
                rjsonl(run_p / "stages" / "s1" / "chunks.jsonl")]
        log.info(f"  语料嵌入 {CE.shape}；查询嵌入 {md.E.shape}；k = {ks}")
        log.info("  只做检索侧：标签固定在已发表的 k=10，论文里必须写明这一点")

        # 需要的查询：参与候选对的那些，外加全体 agent 调用（量集合变化）
        need = set()
        cand = [p for p in md.pairs_raw if p["cos"] >= 0.70]
        for p_ in cand:
            need.add(p_["qid_i"]); need.add(p_["qid_j"])
        qids = [q for q in md.qidx if q in need]
        log.info(f"  重新检索 {len(qids)} 条查询（cos>=0.70 的候选对涉及到的）")

        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            T = torch.nn.functional.normalize(
                torch.from_numpy(CE).to(dev, dtype=torch.float16), dim=1)
            def topk(Q, k):
                q = torch.nn.functional.normalize(
                    torch.from_numpy(Q).to(dev, dtype=torch.float16), dim=1)
                I = []
                for s in range(0, q.shape[0], 512):
                    sc = q[s:s + 512] @ T.T
                    I.append(torch.topk(sc, k, dim=1).indices.cpu().numpy())
                    del sc
                return np.concatenate(I)
            log.info(f"  用 {dev} 做 matmul（不加载任何模型，只是重算 top-k）")
        except Exception as e:                           # 退回 numpy
            log.warning(f"  torch 不可用（{e}），退回 numpy，会慢")
            Cn = CE / (np.linalg.norm(CE, axis=1, keepdims=True) + 1e-9)
            def topk(Q, k):
                Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
                I = []
                for s in range(0, Qn.shape[0], 256):
                    sc = Qn[s:s + 256] @ Cn.T
                    I.append(np.argpartition(-sc, k, axis=1)[:, :k])
                return np.concatenate(I)

        Q = md.E[[md.qidx[q] for q in qids]]
        TOP = {}
        for k in ks:
            TOP[k] = topk(Q, k)
            log.info(f"    k={k} 完成")
            p2.beat(f"R4 k={k}")
        row_of = {q: i for i, q in enumerate(qids)}

        # 与已发表的 k=10 结果比对，确认重算口径一致
        chk = None
        if 10 in TOP:
            agree, n = [], 0
            for c in md.calls:
                if c["qid"] in row_of and c.get("retrieved_chunks"):
                    got = {cids[i] for i in TOP[10][row_of[c["qid"]]]}
                    had = set(c["retrieved_chunks"])
                    if had:
                        agree.append(len(got & had) / len(had)); n += 1
                if n >= 2000:
                    break
            chk = {"n": n, "mean_overlap_with_stored_top10": float(np.mean(agree))
                   if agree else None}
            log.info(f"  与落盘的 top-10 重合度 {chk['mean_overlap_with_stored_top10']:.3f}"
                     f"（n={n}）—— 低于 0.95 说明重算口径有偏差")

        # 对候选对：集合层面的包含损失，这是保证式那个量的检索侧代理
        sub = [p_ for p_ in cand if p_.get("cos", 0) >= tau
               and p_["qid_i"] in row_of and p_["qid_j"] in row_of]
        log.info(f"  τ*={tau} 之上的候选对 {len(sub)}")
        out = {"ks": ks, "tau": tau, "n_pairs": len(sub), "replication": chk,
               "note": "可替代性标签固定在 k=10；此处只量检索侧对 k 的敏感性",
               "by_k": {}}
        log.info("")
        log.info(f"  {'k':>5}{'对内 Jaccard':>14}{'包含率':>12}{'零损失占比':>14}"
                 f"{'P99 集合损失':>14}")
        for k in ks:
            jac, cont = [], []
            for p_ in sub:
                A = {cids[i] for i in TOP[k][row_of[p_["qid_i"]]]}
                B = {cids[i] for i in TOP[k][row_of[p_["qid_j"]]]}
                u = len(A | B)
                jac.append(len(A & B) / u if u else 0.0)
                cont.append(len(A & B) / len(A) if A else 0.0)
            loss = 1.0 - np.array(cont)
            v = {"pair_jaccard": float(np.mean(jac)),
                 "containment": float(np.mean(cont)),
                 "zero_loss_frac": float(np.mean(loss <= 1e-9)),
                 "p99_loss": float(np.percentile(loss, 99)),
                 "mean_loss_given_nonzero": float(loss[loss > 1e-9].mean())
                 if (loss > 1e-9).any() else 0.0}
            out["by_k"][str(k)] = v
            log.info(f"  {k:>5}{v['pair_jaccard']:>14.3f}{v['containment']:>12.3f}"
                     f"{v['zero_loss_frac']*100:>13.1f}%{v['p99_loss']:>14.3f}")

        # 只看平均包含率会得出「对 k 不敏感」，那是两个反向效应抵消出来的假象：
        # k 变大 -> 出现损失的对更多（δ 上升），但每次损失更小（ε 下降）。
        # 保证式里 δ 和 ε 是分开的两个参数，所以必须分开判。
        c10 = out["by_k"].get("10", {}).get("containment")
        cr = [out["by_k"][str(k)]["containment"] for k in ks]
        zr = [out["by_k"][str(k)]["zero_loss_frac"] for k in ks]
        er = [out["by_k"][str(k)]["mean_loss_given_nonzero"] for k in ks]
        out["containment_range"] = [min(cr), max(cr)]
        out["delta_range"] = [1 - max(zr), 1 - min(zr)]      # δ = 有损失的占比
        out["eps_range"] = [min(er), max(er)]
        if c10:
            d_cont = max(abs(x - c10) for x in cr)
            d_delta = max(zr) - min(zr)
            d_eps = max(er) - min(er)
            out["sensitivity"] = {"containment_drift": d_cont,
                                  "zero_loss_frac_drift": d_delta,
                                  "eps_drift": d_eps}
            parts = [f"平均包含率在 k ∈ {ks} 上只变动 {d_cont:.3f}（k=10 为 {c10:.3f}）"]
            if d_delta > 0.05 or d_eps > 0.05:
                parts.append(
                    f"但零损失占比从 {max(zr)*100:.1f}% 降到 {min(zr)*100:.1f}%"
                    f"（δ 随 k 上升 {d_delta*100:.1f}pp），"
                    f"非零损失均值从 {max(er):.3f} 降到 {min(er):.3f}"
                    f"（ε 随 k 下降 {d_eps:.3f}）")
                parts.append(
                    "★ 所以不能说「检索侧对 k 不敏感」—— 均值平是两个反向效应抵消。"
                    "正确表述：k 变大则失败更频繁但每次更轻；δ 与 ε 对 k 的依赖方向相反，"
                    "而 §5.6 在语料这条轴上观察到的是 δ 稳定、ε 变动，两条轴行为不同。")
            else:
                parts.append("且 δ 与 ε 同样稳定，检索侧确实对 k 不敏感")
            out["verdict"] = "；".join(parts) + "。"
            log.info("")
            for p_ in parts:
                log.info(f"  {p_}")

        if p2.a.r4_judge:
            log.info("")
            log.info("  --r4-judge：在抽样上按各 k 重判自覆盖（要起判官）")
            out["judge_tier"] = _r4_judge_tier(p2, P, cfg, md, cids, TOP, row_of, ks)

        (p2.root / "R").mkdir(parents=True, exist_ok=True)
        wjson(p2.root / "R" / "r4_k_sensitivity.json", out)
        return {"summary": (out.get("verdict") or "k 敏感性已出")[:110]}
    return p2.run_step("R4", "k 敏感性（检索侧免费；--r4-judge 加判官层）", run)


def _r4_judge_tier(p2, P, cfg, md, cids, TOP, row_of, ks) -> dict:
    """B 层：在分层抽样上，按每个 k 重判「自己的检索结果能否回答自己」。
    自覆盖率随 k 怎么变，是 k 敏感性里唯一必须动判官的量。"""
    log = p2.log
    cfg2 = P.Config.load(p2.a.config)
    cfg2.run_id = "phase2_R4j"; cfg2.root = "runs"; cfg2.resolve()
    ctx = P.RunContext(cfg2)
    cfg2._autotune = P.autotune_hardware(cfg2, ctx.log)
    sm = P.ServerManager(ctx)
    p2.register_server(sm)
    chunks = rjsonl(Path(p2.a.main_run) / "stages" / "s1" / "chunks.jsonl")
    ctext = {c["chunkid"]: c["text"] for c in chunks}
    cand = [c for c in md.calls if c["qid"] in row_of]
    random.Random(77).shuffle(cand)
    cand = cand[: p2.a.r4_judge_n]
    urls = sm.ensure("judge")
    pool = P.LLMPool(urls, cfg2.judge_name, retries=cfg2.judge_retries, timeout=300)

    res = {}
    for k in ks:
        def one(c, k=k):
            ids = [cids[i] for i in TOP[k][row_of[c["qid"]]]]
            docs = "\n\n".join(
                f"[{i+1}] {(ctext.get(x) or '')[:cfg2.judge_doc_chars]}"
                for i, x in enumerate(ids)) or "(无结果)"
            j = P._judge1(pool, cfg2, c["query"], docs)
            return j.get("covered") if j.get("ok") else None
        r = P.pmap(one, cand, cfg2.judge_concurrency, f"R4j k={k}", ctx, pool)
        v = [x for x in r if x is not None]
        res[str(k)] = {"n": len(v), "self_cov": float(np.mean(v)) if v else None}
        log.info(f"    k={k}  自覆盖 {fmtp(res[str(k)]['self_cov'])}  (n={len(v)})")
        p2.beat(f"R4j k={k}")
    return {"n_sampled": len(cand), "by_k": res}


# ==============================================================================
EXP = {"R1": exp_R1, "R2": exp_R2, "R3": exp_R3, "R4": exp_R4}
ORDER = ["R1", "R2", "R4", "R3"]          # 判官那项放最后，前三项不占 GPU


def main() -> int:
    ap = argparse.ArgumentParser(description="回应评审：四个补充实验")
    if "--selftest" in sys.argv:
        return selftest_ladder()
    ap.add_argument("--selftest", action="store_true",
                    help="只跑 R1 规范化阶梯的手工用例自检，不读数据、不要 GPU")
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="R1,R2,R4")
    ap.add_argument("--phase1", default="phase1_v13.py")
    ap.add_argument("--main-run", default="runs/main_v10")
    ap.add_argument("--distractor-run", default=None)
    ap.add_argument("--out-root", default="runs/phase2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--agent", default=None)
    # R1
    ap.add_argument("--r1-seeds", type=int, default=5)
    # R2
    ap.add_argument("--r2-seeds", type=int, default=5)
    ap.add_argument("--r2-b-run", default="runs/agent_b",
                    help="第二个 agent 的 run；取它与主 run 的 task_id 交集做等任务量对照")
    ap.add_argument("--r2-b-source", default="agent-qwen32b",
                    help="第二个 agent 在该 run 里的 source 标签")
    ap.add_argument("--r2-agent-a-head", type=float, default=0.259,
                    help="已发表的等量 A 头部覆盖，用于复刻核对；0 表示不核对")
    ap.add_argument("--r2-agent-b-head", type=float, default=0.178,
                    help="已发表的等量 B 头部覆盖，用于复刻核对与归因兜底")
    ap.add_argument("--r2-target-density", type=float, default=1.84,
                    help="B 的 calls/task；只在读不到 B 的 run 时作兜底")
    # R3
    ap.add_argument("--r3-n", type=int, default=600)
    # R4
    ap.add_argument("--r4-ks", default="5,10,20,50")
    ap.add_argument("--r4-judge", action="store_true")
    ap.add_argument("--r4-judge-n", type=int, default=800)
    a = ap.parse_args()
    for k in ("r2_agent_a_head", "r2_agent_b_head"):
        if getattr(a, k) == 0:
            setattr(a, k, None)

    exps = [e.strip().upper() for e in a.exp.split(",")]
    exps = [e for e in ORDER if e in exps]
    if not exps:
        print(f"可选 {ORDER}")
        return 2
    p2 = Phase2(a)
    p2.log.info(f"回应评审 {exps}")
    t0 = time.time()
    try:
        for e in exps:
            try:
                EXP[e](p2)
            except Exception as ex:
                import traceback
                p2.failures.append((e, f"{type(ex).__name__}: {ex}"))
                p2.save_state()
                p2.log.error(f"{e} 失败: {type(ex).__name__}: {ex}")
                p2.log.debug(traceback.format_exc())
                if a.stop_on_error:
                    break
    finally:
        # R3 / --r4-judge 会起 vLLM；不关掉会把 GPU 一直占着，
        # 下一次跑（或 phase2 的其他实验）会直接因显存不足失败
        p2.stop_servers()
    p2.log.info(f"总耗时 {(time.time()-t0)/60:.1f} min")
    if p2.failures:
        p2.log.warning(f"失败 {len(p2.failures)} 项：{[f[0] for f in p2.failures]}")
    return 1 if p2.failures else 0


if __name__ == "__main__":
    sys.exit(main())
