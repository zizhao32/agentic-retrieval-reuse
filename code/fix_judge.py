#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
修仪器 —— 让判官按构造对段落顺序不变，然后重新校准
================================================================================
为什么要做这个（评审意见第 1 条，我们选了路线 a）：
§5 的核心数字（0.85 下视图服务 21.7% 成本、净增 8.8 点）依赖判官在 [0.90,1.0)
的判定，而 §6 证明那一带 test–retest κ 只有 0.04–0.18；拿打乱顺序后的判定重新
校准，τ* 从 0.9398 跳到 0.9937 —— 跨过了退化边界。只要仪器这样，主结论就只能
写成区间，而区间两端一个「可用」一个「退化成精确缓存」，等于没有结论。

路线 a 的做法：对每一对用 K 个**各自播种**的段落顺序各判一次，取多数票，
把顺序从判定里平均掉；然后在聚合后的判定上重新校准 τ*。

但多数票不一定有用，这取决于不稳定的来源，而两种来源给出相反的预测：

  模型一（独立噪声）：每次判定 ~ 独立 Bernoulli(p)，翻转率 = 2p(1-p)。
      bin8 实测 46.6% ⟹ p(1-p)=0.233 ⟹ p ≈ 0.50。在 p=0.5 处多数票**无效**
      （K=3 的两次独立多数票之间，预测翻转率仍约 0.47）。
  模型二（系统性顺序效应）：判定由具体顺序决定（打乱使 INSUFFICIENT 更可能，
      §6 测到方向性偏差 +0.74）。此时对随机顺序取多数，收敛到「顺序分布下的
      期望判定」，两次独立多数票都在估同一个期望，一致率**大幅上升**。

先验上分不出是哪一种，所以 J1 是个**便宜的闸门**：先用约 1.5 小时判出结论，
不行就别花十小时跑全量。脚本会把两种模型的预测和实测一起打出来，
让结果无论朝哪边都可解释。

  J1  闸门：多数票到底有没有用（分层抽样，~1.5–2 h GPU）
      臂：k1 / k3 / k5 / think（开思考）/ xmodel（换模型）
      每臂做**两次独立的聚合判定**，比的是聚合判定之间的翻转率 ——
      不是 K 次单判之间的一致性（那会偏乐观，是同一次投票的内部相关）。
      判据：有没有哪个臂把 bin8 的翻转率压到 --target 以下。
      没有 -> 路线 a 失败，老实回去走路线 b，只花了两小时。

  J2  全量重判 + 重新校准（只在 J1 过关后跑，~6–10 h）
      用胜出的臂重判所有合格对，写成一个新 run 的 s6/l1.jsonl，
      s1–s5 软链主 run（和 agent_b.py 同一套做法），然后交给
      phase1_v13.py --stages s8,... 走原来那条已验证的校准路径。
      **不自己重写校准逻辑**，否则等于换了两个变量。

  J3  跨模型 κ（~20 min）
      论文里 κ=0.824 那个「第二判官」，当前 config.json 里没有 cross_judge_*，
      代码路径要求填了外部 model 才会跑 —— 也就是 artifact 复现不出来，
      而论文却据它推出「残差是任务属性而非模型属性」。这里用一个**明确记名**
      的第二模型在 n=600 上重测，并把模型名写进结果 json，让这个数可溯源。

用法
----
    R=/mnt/data/user_workspace/zhangzizhao/matview

    # 闸门：先跑这个，别跳
    python fix_judge.py --config $R/config.json --exp J1 \
        --x-judge-path $R/models/Qwen3.8-27B-FP8

    # 过关了再跑全量（--arm 填 J1 选出来的）
    python fix_judge.py --config $R/config.json --exp J2 --arm k3

    # 跨模型 κ
    python fix_judge.py --config $R/config.json --exp J3 \
        --x-judge-path $R/models/Qwen3.8-27B-FP8

J2 跑完按它打印的命令重新校准，然后比 τ* 是否还跨退化边界。
================================================================================
"""

from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _load(name: str, fname: str):
    spec = _ilu.spec_from_file_location(name, str(_HERE / fname))
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


p2x = _load("p2x", "phase2_experiments.py")
Phase2, RunData = p2x.Phase2, p2x.RunData
rjsonl, wjson, rjson = p2x.rjsonl, p2x.wjson, p2x.rjson
fmtp, fmtf = p2x.fmtp, p2x.fmtf

# 与 §6 一致的分层；前三个是塌陷区，后两个是判官稳定的对照层
BANDS = [(0.95, 0.98), (0.98, 0.999), (0.999, 1.001), (0.70, 0.75), (0.85, 0.90)]
COLLAPSE = "[0.98,0.999)"          # 主证据层，闸门就看它


# ------------------------------------------------------------------ 判定与聚合

def fmt_seeded(ids, cmap, lim, seed):
    """按给定种子打乱段落顺序。s6 用的是**未播种**的全局 random.shuffle，
    所以原来那次的顺序无法复现 —— 这也是当初没法做这个实验的原因。"""
    o = list(ids)
    random.Random(seed).shuffle(o)
    return "\n\n".join(f"[{i+1}] {(cmap.get(c) or '')[:lim]}"
                       for i, c in enumerate(o)) or "(无结果)"


def judge_once(P, pool, cfg, q, docs, prompt=None, thinking=False, max_tokens=None):
    """_judge1 把 NO_THINK 写死了，开思考必须自己发请求。其余解析沿用 P.xjson。"""
    if not thinking:
        return P._judge1(pool, cfg, q, docs, prompt)
    tmpl = prompt or P.L1_PROMPT
    kw = {"chat_template_kwargs": {"enable_thinking": True}}
    r = pool.chat([{"role": "user", "content": tmpl.format(q=q, docs=docs)}],
                  temperature=cfg.judge_temperature,
                  max_tokens=max_tokens or max(cfg.judge_max_tokens * 4, 2048), **kw)
    j = P.xjson(r["text"])
    if not j:
        return {"ok": False, "verdict": "PARSE_FAIL",
                "truncated": r["finish_reason"] == "length"}
    v = str(j.get("verdict", "")).upper()
    return {"ok": True, "verdict": v, "covered": int(v == "SUFFICIENT"),
            "facts": j.get("required_facts", [])}


def vote(verdicts):
    """奇数 K 的多数票。有解析失败就按有效的来；全失败返回 None。"""
    vs = [v for v in verdicts if v is not None]
    if not vs:
        return None
    return int(sum(vs) * 2 > len(vs))


def link_stages(src: Path, dst: Path, log, stages=("s1", "s2", "s3", "s4", "s5")):
    """把来源 run 的 s1–s5 整目录逐文件软链过去。

    原来是文件白名单，漏了 s4/audit.json，s8 起手就 FileNotFoundError。
    改成遍历目录：语料、嵌入、候选对、抽样这几个阶段本来就是整份复用的，
    少链一个文件的代价远大于多链一个。
    """
    import os
    n = 0
    for st in stages:
        sd = src / "stages" / st
        if not sd.is_dir():
            continue
        dd = dst / "stages" / st
        dd.mkdir(parents=True, exist_ok=True)
        for sp in sorted(sd.iterdir()):
            if not sp.is_file():
                continue
            dp = dd / sp.name
            if not dp.exists():
                os.symlink(sp.resolve(), dp)   # resolve：来源本身可能也是软链
                n += 1
    log.info(f"  s1–s5 已软链 {n} 个文件：{src} -> {dst}")
    return n


def recompute_qc(src: Path, dst: Path, res, arm_block, log, extra=None):
    """写 s6/qc.json。

    **不能整份拷来源 run 的 qc.json** —— 里面大半是判定派生量
    （pos_consistency / strict / union / asymmetry / llm / tokens / 各种
    substitutable_rate），拷过来就是把旧臂的数字贴上新臂的标签。上一版就是
    这么错的：main_v12_think/s6/qc.json 和 main_v10 的每一个数值叶子都相同，
    而它声称自己是 think1 的。
    """
    KEEP = ("n_sampled_pairs", "n_eligible", "eligibility_rate",
            "eligibility_by_bin", "self_cov_rate")
    DROP = ("pos_consistency", "pos_consistency_n", "pos_consistency_by_bin",
            "strict", "union", "asymmetry_rate", "llm", "tokens",
            "effective_n", "substitutable_rate_conditional", "n_judged",
            "substitutable_rate_conditional_ci",
            "substitutable_rate_conditional_unweighted",
            "substitutable_rate_marginal", "same_text_direction_consistency",
            "same_text_n", "catch_fp_rate", "parse_fail_rate", "n_parse_fail",
            "truncated_share", "mode_used", "cross_kappa", "cross_n")
    srcqc = src / "stages" / "s6" / "qc.json"
    old = rjson(srcqc) if srcqc.exists() else {}
    qc = {k: old[k] for k in KEEP if k in old}

    good = [r for r in res if r.get("substitutable") is not None]
    st = [r for r in good if r.get("same_text")]

    def wrate(rs, f):
        num = sum((r.get("w") or 0.0) for r in rs if f(r))
        den = sum((r.get("w") or 0.0) for r in rs)
        return float(num / den) if den else None

    qc["judge_aggregation"] = arm_block
    qc["recomputed_from_l1"] = {
        "n_judged": len(good),
        "substitutable_rate_conditional_unweighted":
            float(np.mean([r["substitutable"] for r in good])) if good else None,
        "substitutable_rate_conditional":
            wrate(good, lambda r: r["substitutable"]),
        "asymmetry_rate":
            (sum(1 for r in good if r.get("cov_ij") != r.get("cov_ji"))
             / len(good)) if good else None,
        "same_text_n": len(st),
        "same_text_substitutable_rate":
            float(np.mean([r["substitutable"] for r in st])) if st else None,
    }
    qc["_provenance"] = {
        "inherited_from": str(src),
        "inherited_fields": [k for k in KEEP if k in old],
        "dropped_stale": sorted(k for k in DROP if k in old),
        "note": "被 drop 的字段是上一个臂的判定派生量，本 run 没有重算，"
                "所以不写出来，而不是留着旧值。要它们必须在本 run 上重跑 s6/s7。",
    }
    if extra:
        qc["_provenance"].update(extra)
    wjson(dst / "stages" / "s6" / "qc.json", qc)
    log.info("  qc.json 已按新臂重算；丢弃的旧派生量 %d 个"
             % len(qc["_provenance"]["dropped_stale"]))
    return qc


def wait_gpu_free(need_gb: float, log, timeout: float = 300.0, poll: float = 5.0):
    """轮询等显存真正释放。单卡时这是关键路径：停掉一个 27GB 的 vLLM 后，
    进程退出到显存归还有延迟，固定 sleep 猜不准 —— 猜短了下一个模型起不来
    （上一轮 J1/J3 就是死在显存不足），猜长了白等。
    用 torch.cuda.mem_get_info，和 autotune_hardware 同一个口径。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
    except Exception:
        time.sleep(poll)
        return None
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        try:
            frees = [torch.cuda.mem_get_info(g)[0] / 2 ** 30
                     for g in range(torch.cuda.device_count())]
        except Exception:
            break
        last = max(frees) if frees else 0.0
        if last >= need_gb:
            log.info(f"  显存已释放：最大空闲 {last:.1f}GB >= 需要 {need_gb:.1f}GB"
                     f"（等了 {time.time()-t0:.0f}s）")
            return last
        time.sleep(poll)
    log.warning(f"  等了 {timeout:.0f}s 显存仍只有 {last if last is not None else -1:.1f}GB"
                f"（需要 {need_gb:.1f}GB）。下一个模型可能起不来 —— "
                "手动查: nvidia-smi; pkill -9 -f 'vllm serve'; pkill -9 -f EngineCore")
    return last


def set_gpus(cfg, spec, log):
    """指定用哪些**绝对**卡号，直接写 cfg.gpus。

    千万不要在外面设 CUDA_VISIBLE_DEVICES —— 它会造成二次映射，反而落到错的卡：
      1) autotune_hardware 看到只有一张可见卡，把 cfg.gpus 改写成相对索引 [0]
      2) ServerManager._start 给子进程设 env CUDA_VISIBLE_DEVICES=str(gpu)=“0”，
         这个赋值**覆盖**掉继承来的值，于是 “0” 又被当成绝对索引
      3) 子进程跑到物理 GPU0 上 —— 如果那张卡忙着，就是 OOM
    实测就是这么失败的（物理 GPU0 占着 64GiB，而我们以为在用空着的 GPU1）。
    写 cfg.gpus 则三处口径一致：autotune 用 torch 绝对索引读显存，
    _preflight_gpu 用 nvidia-smi 绝对索引（那条假警报也随之消失），
    _start 导出的也是绝对索引。
    """
    if not spec:
        return
    g = [int(x) for x in str(spec).replace(",", " ").split()]
    cfg.gpus = g
    if getattr(cfg, "retriever_gpu", None) not in g:
        cfg.retriever_gpu = g[0]
    log.info(f"  指定 GPU（绝对卡号）{g} —— 不要再设 CUDA_VISIBLE_DEVICES")


def port_free(port: int) -> bool:
    """端口是否可绑。残留的 vLLM 占着 8001 时，不换端口就永远起不来，
    而容器里那个进程可能在别的 PID namespace，pkill 都找不到它。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sk.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def pick_port(cfg, log, want=None):
    """选一个能用的 judge 端口并写回 cfg。--judge-port 显式指定优先。"""
    base = want or cfg.judge_base_port
    for off in range(0, 40, 2):
        if port_free(base + off):
            if off:
                log.warning(f"  端口 {base} 被占，改用 {base+off}"
                            f"（残留 vLLM 多半还在，但我们用的是另一张卡，不影响）")
            cfg.judge_base_port = base + off
            return cfg.judge_base_port
    raise RuntimeError(f"{base}..{base+38} 全被占。先 nvidia-smi / ss -ltnp 看是谁")


# ------------------------------------------------------------------ 一致性度量

def agree_stats(a, b):
    """翻转率、Cohen κ、PABAK、Gwet AC1。κ 在边际极端时会崩，所以三个一起报
    —— 这正是 §6 的 G1 得到的教训。"""
    n = len(a)
    if n == 0:
        return {}
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kap = (po - pe) / (1 - pe) if pe < 1 else 0.0
    pi = (pa1 + pb1) / 2
    pe_g = 2 * pi * (1 - pi)
    return {"n": n, "flip": 1 - po, "kappa": kap, "pabak": 2 * po - 1,
            "ac1": (po - pe_g) / (1 - pe_g) if pe_g < 1 else 1.0,
            "pos_rate_a": pa1, "pos_rate_b": pb1}


def predict_majority_flip(flip_k1: float, K: int, trials: int = 200000, seed=0):
    """独立噪声模型下，由 K=1 的实测翻转率推出 K 票多数的预测翻转率。
    2p(1-p)=flip 解出 p，再模拟两次独立的 K 票多数。
    实测明显低于这个预测 = 不稳定是系统性顺序效应，多数票有效（模型二）；
    实测贴着这个预测 = 就是独立噪声，多数票无效（模型一）。"""
    d = 1 - 2 * flip_k1
    if d < 0:
        return None
    p = 0.5 + 0.5 * math.sqrt(d)            # 取 >=0.5 的根
    rng = np.random.default_rng(seed)
    a = (rng.random((trials, K)) < p).sum(1) * 2 > K
    b = (rng.random((trials, K)) < p).sum(1) * 2 > K
    return {"implied_p": p, "pred_flip": float((a != b).mean())}


# ==============================================================================
#  J1：闸门
# ==============================================================================

def _j1_phase(p2, P, cfg, ctx, pool, items, cmapq, cmap, arm, K, inv, log):
    """跑一个臂：自然序一次 + 两次独立的 K 票多数（各自播种的打乱序）。

    J2 的结果把问题改了。既然多数票收敛到的是**打乱分布的中心**而不是自然序的值，
    那「两次 K 票多数是否一致」就不再是关键 —— 关键是自然序的判定与打乱分布的
    期望差多少。差得小，这个配置才是实质顺序无关的，才能用多数票而不丢自然序的数。
    所以每个臂同时测两样：
      stability —— 两次独立多数票之间的翻转率（顺序平均后的估计稳不稳）
      order_gap —— 自然序判定 与 多数票判定 的不一致率（系统性顺序效应有多大）
    两个都小才算可用。
    """
    def one(it):
        band, r = it
        q = cmapq[r["qid_i"]]["query"]
        ids = cmapq[r["qid_j"]]["retrieved_chunks"]
        lim = cfg.judge_doc_chars

        def call(docs):
            if arm == "think":
                return judge_once(P, pool, cfg, q, docs, thinking=True)
            if arm == "inv":
                return judge_once(P, pool, cfg, q, docs, prompt=inv)
            return judge_once(P, pool, cfg, q, docs)

        # 自然序 = 检索排名顺序，就是 s6 原来用的、也是部署时的顺序
        nat_docs = "\n\n".join(f"[{i+1}] {(cmap.get(c) or '')[:lim]}"
                               for i, c in enumerate(ids)) or "(无结果)"
        jn = call(nat_docs)
        nat = jn.get("covered") if jn.get("ok") else None

        votes = []
        sd = (abs(hash(r.get("pair_id") or f"{r['qid_i']}|{r['qid_j']}")) % 10**6) * 100
        for rep in range(2):
            vs = []
            for k in range(K):
                sd += 1
                j = call(fmt_seeded(ids, cmap, lim, sd))
                vs.append(j.get("covered") if j.get("ok") else None)
            votes.append(vote(vs))
        return {"band": band, "pair_id": r.get("pair_id"), "cos": r["cos"],
                "nat": nat, "maj": votes}

    res = P.pmap(one, items, cfg.judge_concurrency, f"J1 {arm}", ctx, pool)
    return [r for r in res if isinstance(r, dict)]


def exp_J1(p2: Phase2) -> dict:
    def run():
        log, P, a = p2.log, p2.P, p2.a
        cfg = P.Config.load(a.config)
        cfg.run_id = "judge_fix_J1"; cfg.root = "runs"; cfg.resolve()
        set_gpus(cfg, a.gpus, log)
        ctx = P.RunContext(cfg)
        cfg._autotune = P.autotune_hardware(cfg, ctx.log)

        md = RunData(a.main_run, log)
        cmapq = {c["qid"]: c for c in md.calls}
        cmap = {c["chunkid"]: c["text"] for c in
                rjsonl(Path(a.main_run) / "stages" / "s1" / "chunks.jsonl")}

        per = max(40, a.j1_n // len(BANDS))
        items = []
        for lo, hi in BANDS:
            cand = [r for r in md.l1 if lo <= r["cos"] < hi
                    and r["qid_i"] in cmapq and r["qid_j"] in cmapq
                    and (cmapq[r["qid_j"]].get("retrieved_chunks"))]
            random.Random(4242).shuffle(cand)
            for r in cand[:per]:
                items.append((f"[{lo},{hi})", r))
        log.info(f"  抽 {len(items)} 对，{len(BANDS)} 层各约 {per}")

        arms = [s.strip() for s in a.j1_arms.split(",") if s.strip()]
        KOF = {"k3": 3, "k5": 5, "think": 3, "inv": 3, "xmodel": 3}
        arms = [x for x in arms if x in KOF]
        log.info(f"  臂 {arms}；每臂每对 1（自然序）+ 2xK（打乱）次判定")
        log.info("  测两样：两次多数票之间的翻转率，以及自然序与多数票的差距 —— "
                 "后者才是 J2 暴露出来的真问题")

        inv = P._prompt_variants().get("order_inv")
        res_by_arm, xname = {}, None

        # ---- 阶段一：主判官的臂。串行，不能和第二个模型共存 ----
        main_arms = [x for x in arms if x != "xmodel"]
        if main_arms:
            sm = P.ServerManager(ctx)
            p2.register_server(sm)
            pick_port(cfg, log, a.judge_port)
            urls = sm.ensure("judge")
            pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries,
                             timeout=300)
            for arm in main_arms:
                log.info(f"  --- 臂 {arm}（{cfg.judge_name}）---")
                res_by_arm[arm] = _j1_phase(p2, P, cfg, ctx, pool, items,
                                            cmapq, cmap, arm, KOF[arm], inv, log)
                p2.beat(f"J1 {arm} done")
            # 必须先停掉，否则第二个模型要不到显存。
            # 上一版就是栽在这里：两个 27GB 判官同时起，第二个只剩 4.2GB。
            log.info("  停掉主判官，给第二个模型腾显存")
            sm.stop()
            wait_gpu_free(a.need_gb, log, timeout=a.gpu_settle)

        # ---- 阶段二：第二个模型 ----
        if "xmodel" in arms:
            if not a.x_judge_path:
                log.warning("  xmodel 臂需要 --x-judge-path，跳过")
            else:
              # 这个臂整体兜住：第二个模型可能在本环境根本起不来
              # （实测 Qwen3.8-27B-FP8 的 model_type 'qwen3_5' vLLM 0.8.5 不认，
              #  且 _start 给 judge 角色硬编码 awq_marlin 而它是 FP8）。
              # 起不来就只是少一个臂，不能让已经跑完的 k3/think 白费。
              try:
                xcfg = P.Config.load(a.config)
                xcfg.judge_path = a.x_judge_path
                xcfg.judge_name = Path(a.x_judge_path).name
                xcfg.run_id = "judge_fix_J1x"; xcfg.root = "runs"; xcfg.resolve()
                set_gpus(xcfg, a.gpus, log)
                xctx = P.RunContext(xcfg)
                xcfg._autotune = P.autotune_hardware(xcfg, xctx.log)
                xsm = P.ServerManager(xctx)
                p2.register_server(xsm)
                pick_port(xcfg, log, a.judge_port)
                xname = xcfg.judge_name
                log.info(f"  --- 臂 xmodel（{xname}）---")
                xurls = xsm.ensure("judge")
                xpool = P.LLMPool(xurls, xcfg.judge_name,
                                  retries=xcfg.judge_retries, timeout=300)
                res_by_arm["xmodel"] = _j1_phase(p2, P, xcfg, xctx, xpool, items,
                                                 cmapq, cmap, "base", 3, inv, log)
                xsm.stop()
                wait_gpu_free(a.need_gb, log, timeout=a.gpu_settle)
              except Exception as ex:
                log.error(f"  xmodel 臂起不来，跳过该臂：{type(ex).__name__}: {ex}")
                log.error("  —— 跨模型对照在本环境不可得；论文里那个来源不明的 "
                          "κ=0.824 应当删掉，而不是换个数顶上")
                res_by_arm.pop("xmodel", None)

        # ---- 汇总 ----
        out = {"n_items": len(items), "arms": list(res_by_arm), "bands": BANDS,
               "x_judge": xname, "per_band": {}, "overall": {}}
        for arm, rs in res_by_arm.items():
            ok = [r for r in rs if r["maj"] and None not in r["maj"]]
            st = agree_stats([r["maj"][0] for r in ok], [r["maj"][1] for r in ok])
            withnat = [r for r in ok if r["nat"] is not None]
            # 自然序 vs 多数票：用第一次多数票比，两次结果同分布
            gap = agree_stats([r["nat"] for r in withnat],
                              [r["maj"][0] for r in withnat])
            out["overall"][arm] = {"stability": st, "order_gap": gap}
        for band in [f"[{lo},{hi})" for lo, hi in BANDS]:
            out["per_band"][band] = {}
            for arm, rs in res_by_arm.items():
                rows = [r for r in rs if r["band"] == band]
                ok = [r for r in rows if r["maj"] and None not in r["maj"]]
                wn = [r for r in ok if r["nat"] is not None]
                out["per_band"][band][arm] = {
                    "stability": agree_stats([r["maj"][0] for r in ok],
                                             [r["maj"][1] for r in ok]),
                    "order_gap": agree_stats([r["nat"] for r in wn],
                                             [r["maj"][0] for r in wn]),
                    "pos_nat": (float(np.mean([r["nat"] for r in wn])) if wn else None),
                    "pos_maj": (float(np.mean([r["maj"][0] for r in ok])) if ok else None)}

        armlist = list(res_by_arm)
        log.info("")
        log.info("  两次多数票之间的翻转率（越低越稳）")
        log.info(f"  {'层':<16}" + "".join(f"{x:>11}" for x in armlist))
        for band in out["per_band"]:
            row = "".join(f"{fmtp((out['per_band'][band][a_]['stability'] or {}).get('flip')):>11}"
                          for a_ in armlist)
            log.info(f"  {band:<16}{row}")
        log.info("")
        log.info("  自然序 与 多数票 的不一致率（这才是 J2 暴露的问题；越低才越可用）")
        log.info(f"  {'层':<16}" + "".join(f"{x:>11}" for x in armlist))
        for band in out["per_band"]:
            row = "".join(f"{fmtp((out['per_band'][band][a_]['order_gap'] or {}).get('flip')):>11}"
                          for a_ in armlist)
            log.info(f"  {band:<16}{row}")
        log.info("")
        log.info("  同一性层的正例率：自然序 -> 多数票（原 run 自然序 0.997，K=3 全打乱 0.777）")
        for band in ("[0.999,1.001)", "[0.98,0.999)"):
            if band not in out["per_band"]:
                continue
            for a_ in armlist:
                v = out["per_band"][band][a_]
                log.info(f"    {band:<16}{a_:<9}{fmtp(v.get('pos_nat'))} -> "
                         f"{fmtp(v.get('pos_maj'))}")

        # ---- 判据：要同时稳且无系统性偏移 ----
        cb = out["per_band"].get(COLLAPSE, {})
        best, bestscore = None, 9.9
        for a_ in armlist:
            s_ = (cb.get(a_, {}).get("stability") or {}).get("flip")
            g_ = (cb.get(a_, {}).get("order_gap") or {}).get("flip")
            if s_ is None or g_ is None:
                continue
            sc = max(s_, g_)        # 短板决定可用性
            if sc < bestscore:
                best, bestscore = a_, sc
        out["gate"] = {"collapse_band": COLLAPSE, "target": a.target,
                       "best_arm": best, "worst_of_two": bestscore,
                       "passed": bool(best and bestscore <= a.target)}
        log.info("")
        if out["gate"]["passed"]:
            out["gate"]["conclusion"] = (
                f"★ 路线 a 还有戏：{best} 在 {COLLAPSE} 上「稳定性」与「自然序偏移」"
                f"的短板只有 {bestscore*100:.1f}%（目标 {a.target*100:.0f}%）。"
                f"用这个配置重跑 J2（--arm 按它的 K），自然序的数不会被挪走。")
            log.info(f"  {out['gate']['conclusion']}")
        else:
            out["gate"]["conclusion"] = (
                f"★ 路线 a 走不通：最好的臂 {best} 短板 {bestscore*100:.1f}%，"
                f"高于目标 {a.target*100:.0f}%。没有哪个配置能同时做到"
                "「顺序平均后稳定」和「不把自然序的值挪走」。"
                "改走路线 c：把呈现顺序写进操作定义，τ* 在自然序下给点估计，"
                "并把全量打乱重判（任何阈值都达不到 0.80 保证）作为失败的稳健性检验报告。")
            log.warning(f"  {out['gate']['conclusion']}")

        (p2.root / "J").mkdir(parents=True, exist_ok=True)
        out["per_pair"] = {k: v for k, v in res_by_arm.items()}
        wjson(p2.root / "J" / "j1_gate.json", out)
        return {"summary": out["gate"]["conclusion"][:130]}
    return p2.run_step("J1", "闸门：有没有配置能同时稳定且不偏移", run)


# ==============================================================================
#  J2：全量重判 + 落成新 run
# ==============================================================================

def exp_J2(p2: Phase2) -> dict:
    def run():
        import os
        log, P, a = p2.log, p2.P, p2.a
        # J1 的结论改了 J2 该怎么跑：开思考之后判官本身就接近顺序无关
        # （塌陷层自然序偏移 3%，同一性层正例率 96->93 几乎不动），
        # 所以不需要再花 K 倍的调用去平均顺序 —— 自然序判一次就够，
        # 而自然序正是部署时的顺序。顺序鲁棒性由 J1 那 500 对分层样本来证明。
        ARMS = {"k1":     dict(K=1, think=False, natural=False),
                "k3":     dict(K=3, think=False, natural=False),
                "k5":     dict(K=5, think=False, natural=False),
                "think1": dict(K=1, think=True,  natural=True),   # 推荐
                "think3": dict(K=3, think=True,  natural=False)}
        spec = ARMS.get(a.arm)
        if spec is None:
            raise ValueError(f"--arm 只能是 {sorted(ARMS)}，收到 {a.arm}")
        K, THINK, NATURAL = spec["K"], spec["think"], spec["natural"]
        log.info(f"  臂 {a.arm}: K={K} thinking={THINK} "
                 f"顺序={'自然序(检索排名)' if NATURAL else '各自播种的打乱序'}")
        gate = p2.root / "J" / "j1_gate.json"
        if gate.exists():
            g = (rjson(gate).get("gate") or {})
            if not g.get("passed"):
                raise RuntimeError(
                    "J1 闸门没过，J2 会白跑十小时。要强行跑请先删掉 "
                    f"{gate}（但先想清楚为什么）")
        else:
            log.warning("  找不到 J1 结果，建议先跑闸门")

        cfg = P.Config.load(a.config)
        cfg.run_id = a.out_run; cfg.root = "runs"; cfg.resolve()
        set_gpus(cfg, a.gpus, log)
        ctx = P.RunContext(cfg)
        cfg._autotune = P.autotune_hardware(cfg, ctx.log)
        sm = P.ServerManager(ctx)
        p2.register_server(sm)

        src = Path(a.main_run)
        dst = Path("runs") / a.out_run
        exist = dst / "stages" / "s6" / "l1.jsonl"
        if exist.exists() and not a.force:
            prev = None
            try:
                import itertools
                with open(exist, encoding="utf-8") as f:
                    for ln in itertools.islice(f, 1):
                        prev = (json.loads(ln).get("judge_aggregation") or {}).get("arm")
            except Exception:
                pass
            raise RuntimeError(
                f"{exist} 已存在（上一次的臂={prev}）。换个 --out-run，"
                f"或确认要覆盖后加 --force。不同臂的结果混进一个 run 目录，"
                f"下游 s8 起的校准会分不清口径。")
        # 软链 s1–s5：语料、嵌入、候选对、抽样全部不变，只换 s6 的判定
        link_stages(src, dst, log)

        md = RunData(a.main_run, log)
        cmapq = {c["qid"]: c for c in md.calls}
        cmap = {c["chunkid"]: c["text"] for c in
                rjsonl(src / "stages" / "s1" / "chunks.jsonl")}
        elig = [r for r in md.l1 if r.get("eligible") and r.get("ok")]
        ncall = 2 * K * len(elig)
        rate = 0.55 if THINK else 0.37        # J1 实测的单卡每次调用秒数
        log.info(f"  重判 {len(elig)} 对合格对，每对 {2*K} 次判定（双向 x K）"
                 f"= 约 {ncall} 次；单卡约 {ncall*rate/3600:.1f} h，"
                 f"两卡约 {ncall*rate/7200:.1f} h")

        urls = sm.ensure("judge")
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)

        def one(r):
            ci, cj = cmapq.get(r["qid_i"]), cmapq.get(r["qid_j"])
            if not ci or not cj:
                return None
            out = dict(r)
            base = abs(hash(r.get("pair_id") or f"{r['qid_i']}|{r['qid_j']}")) % 10**6
            lim = cfg.judge_doc_chars
            for tag, (q, ids) in (("ij", (ci["query"], cj.get("retrieved_chunks") or [])),
                                  ("ji", (cj["query"], ci.get("retrieved_chunks") or []))):
                vs, fac, vd, facts0 = [], [], [], None
                for k in range(K):
                    if NATURAL:
                        # 检索排名顺序 —— s6 原来用的、也是部署时的顺序
                        docs = "\n\n".join(f"[{i+1}] {(cmap.get(c) or '')[:lim]}"
                                            for i, c in enumerate(ids)) or "(无结果)"
                    else:
                        docs = fmt_seeded(ids, cmap, lim,
                                          base * 10 + k + (0 if tag == "ij" else 5))
                    j = judge_once(P, pool, cfg, q, docs, thinking=THINK)
                    vs.append(j.get("covered") if j.get("ok") else None)
                    vd.append(j.get("verdict"))
                    if j.get("ok"):
                        fl = j.get("facts") or []
                        fac.append(len(fl))
                        if facts0 is None:
                            facts0 = fl
                out[f"cov_{tag}"] = vote(vs)
                out[f"votes_{tag}"] = vs
                # verdict / facts 以前没覆写，留的是来源 run 的值 —— 三个 run
                # 的 verdict_ij 分布一个字不差就是这么来的。这里写本次的。
                out[f"verdicts_{tag}"] = vd
                out[f"verdict_{tag}"] = vd[0] if vd else None
                out[f"facts_{tag}"] = facts0
                if tag == "ij":
                    out["facts_ij_n"] = int(np.median(fac)) if fac else None
            # 这几个本次没有重算：tokens 没按调用累计，double_judged / raw 是
            # 旧口径的。挪进 _inherited 而不是留在原键上冒充新臂的结果。
            out["_inherited"] = {k: r.get(k) for k in
                                 ("tokens", "double_judged", "raw",
                                  "verdict_ij", "verdict_ji", "facts_ij")}
            for k_ in ("tokens", "double_judged", "raw"):
                out[k_] = None
            a_, b_ = out.get("cov_ij"), out.get("cov_ji")
            out["ok"] = a_ is not None and b_ is not None
            out["substitutable"] = int(bool(a_) and bool(b_)) if out["ok"] else None
            # 这些是旧口径的字段，聚合判定下没有意义，显式置空而不是留着旧值
            for k_ in ("cov_ij_shuf", "pos_consistent", "substitutable_strict",
                       "substitutable_union"):
                out[k_] = None
            out["judge_aggregation"] = {
                "arm": a.arm, "K": K, "thinking": THINK,
                "order": "natural" if NATURAL else "seeded_shuffle",
                "rule": "single" if K == 1 else "majority"}
            return out

        res = P.pmap(one, elig, cfg.judge_concurrency, f"J2 {a.arm}", ctx, pool)
        res = [r for r in res if isinstance(r, dict) and r.get("ok")]
        log.info(f"  有效 {len(res)}/{len(elig)}")

        # 不合格的对（self-coverage 没过）原样带过去，s8 需要完整的 l1。
        # 但「合格却这次没判成功」的那批不能原样带 —— 带的是旧臂的标签，
        # 一个文件里混两个臂。上一版 133 条就是这么混进去的。
        keep = {(r["qid_i"], r["qid_j"]) for r in res}
        rest, n_failed = [], 0
        for r0 in md.l1:
            if (r0["qid_i"], r0["qid_j"]) in keep:
                continue
            r0 = dict(r0)
            if r0.get("eligible"):
                n_failed += 1
                r0["_inherited"] = {k: r0.get(k) for k in
                                    ("cov_ij", "cov_ji", "substitutable",
                                     "verdict_ij", "verdict_ji")}
                for k_ in ("cov_ij", "cov_ji", "substitutable",
                           "verdict_ij", "verdict_ji"):
                    r0[k_] = None
                r0["ok"] = False
                r0["rejudge_failed"] = True
            rest.append(r0)
        if n_failed:
            log.warning(f"  {n_failed} 对合格对本次重判失败：标签已置空并标记 "
                        f"rejudge_failed，旧值挪到 _inherited（不混臂）")
        (dst / "stages" / "s6").mkdir(parents=True, exist_ok=True)
        outp = dst / "stages" / "s6" / "l1.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for r in res + rest:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # s7 是 L2 真值锚定。**不能软链主 run 的**：它的 recall_loss 定义在旧的
        # L1 正例集上，而这里标签全变了，软链等于拿旧口径的数去配新标签。
        # 写成 skipped 桩，让 s8 的报告按「本 run 没有 L2 锚定」处理。
        # 上一轮就是漏了这个，s8 算完 τ* 之后在写报告时崩掉。
        (dst / "stages" / "s7").mkdir(parents=True, exist_ok=True)
        wjson(dst / "stages" / "s7" / "summary.json",
              {"skipped": True,
               "reason": "L1 标签已由多数票聚合重算，旧的 L2 锚定不适用；"
                         "要 (eps,delta) 必须在本 run 上重跑 s7"})
        if not (dst / "stages" / "s7" / "l2.jsonl").exists():
            (dst / "stages" / "s7" / "l2.jsonl").write_text("", encoding="utf-8")

        # qc.json 也要有，s8 之后的阶段会读 —— 按新臂重算，不拷旧的
        arm_block = {
            "arm": a.arm, "K": K, "thinking": THINK,
            "order": "natural" if NATURAL else "seeded_shuffle",
            "rule": "single" if K == 1 else "majority",
            "note": "判定口径已变；旧的 pos_consistent / cross_kappa 不适用。"
                    "think1 口径下顺序鲁棒性由 J1 的分层样本证明，"
                    "不靠在每一对上重复采样。"}
        selfcov_arm = ((rjson(src / "stages" / "s6" / "qc.json")
                        if (src / "stages" / "s6" / "qc.json").exists() else {})
                       .get("judge_aggregation") or {}).get("arm")
        recompute_qc(src, dst, res, arm_block, log, extra={
            "selfcov_arm": selfcov_arm or "(来源 run，臂未记录)",
            "coverage_arm": a.arm,
            "MIXED_ARM_WARNING":
                "合格性筛选（self_cov）沿用来源 run 的判官，覆盖判定是本臂 —— "
                "两个臂混用。对逐字相同的一对，cov_ij 问的就是 self_cov_i，"
                "所以恒等带上的缺口量的是两个臂在自覆盖上的分歧，不是本臂的"
                "性质。要单臂干净结果，跑 --exp J4 重做筛选。",
        })

        sub = [r["substitutable"] for r in res if r["substitutable"] is not None]
        log.info("")
        log.info(f"  聚合后可替代率 {np.mean(sub):.4f}（旧口径 "
                 f"{np.mean([r['substitutable'] for r in md.l1 if r.get('substitutable') is not None]):.4f}）")
        log.info(f"  写出 {outp}")
        log.info("")
        log.info("  下一步 —— 走原来那条已验证的校准路径，别自己重算：")
        log.info(f"    python phase1_v13.py --config {a.config} --run-id {a.out_run} \\")
        log.info("        --stages s8,s9,s10,s11,s13,s15,s16,s17,s23,s24")
        log.info("  然后比新旧 τ*：原序 0.9398 / 打乱 0.9937 这个区间是否收拢成一点。")
        wjson(p2.root / "J" / "j2_requdge.json",
              {"arm": a.arm, "K": K, "n_rejudged": len(res),
               "out_run": a.out_run, "l1_path": str(outp),
               "substitutable_rate_new": float(np.mean(sub)) if sub else None})
        return {"summary": f"{a.arm} 重判 {len(res)} 对 -> {outp}，接着跑 s8 起的校准"}
    return p2.run_step(f"J2_{p2.a.arm}",
                       f"全量重判（{p2.a.arm}）", run)


# ==============================================================================
#  J3：记名的跨模型 κ
# ==============================================================================

def exp_J3(p2: Phase2) -> dict:
    def run():
        log, P, a = p2.log, p2.P, p2.a
        if not a.x_judge_path:
            raise ValueError("J3 必须给 --x-judge-path（第二个模型的路径）")
        # 先把前面实验留下的 vLLM 全停掉：判官要 27GB，两个模型共存起不来。
        # 上一版 J3 就是死在这里（"全部 2 个副本都启动失败"）。
        p2.stop_servers()
        wait_gpu_free(a.need_gb, log, timeout=a.gpu_settle)
        cfg = P.Config.load(a.config)
        cfg.judge_path = a.x_judge_path
        cfg.judge_name = Path(a.x_judge_path).name
        cfg.run_id = "judge_fix_J3"; cfg.root = "runs"; cfg.resolve()
        set_gpus(cfg, a.gpus, log)
        ctx = P.RunContext(cfg)
        cfg._autotune = P.autotune_hardware(cfg, ctx.log)
        sm = P.ServerManager(ctx)
        p2.register_server(sm)
        log.info(f"  第二判官 = {cfg.judge_name}（记名；论文里必须写出模型名）")

        md = RunData(a.main_run, log)
        cmapq = {c["qid"]: c for c in md.calls}
        cmap = {c["chunkid"]: c["text"] for c in
                rjsonl(Path(a.main_run) / "stages" / "s1" / "chunks.jsonl")}
        pool_pairs = [r for r in md.l1 if r.get("eligible") and r.get("ok")
                      and r.get("substitutable") is not None
                      and r["qid_i"] in cmapq and r["qid_j"] in cmapq]
        sub = random.Random(20260901).sample(pool_pairs,
                                             min(a.j3_n, len(pool_pairs)))
        log.info(f"  n = {len(sub)}（与论文引用的 n=600 同口径：合格对里随机抽）")

        pick_port(cfg, log, a.judge_port)
        urls = sm.ensure("judge")
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)

        def one(r):
            ci, cj = cmapq[r["qid_i"]], cmapq[r["qid_j"]]
            A = P._judge1(pool, cfg, ci["query"],
                          fmt_seeded(cj.get("retrieved_chunks") or [], cmap,
                                     cfg.judge_doc_chars, 11))
            B = P._judge1(pool, cfg, cj["query"],
                          fmt_seeded(ci.get("retrieved_chunks") or [], cmap,
                                     cfg.judge_doc_chars, 22))
            if not (A.get("ok") and B.get("ok")):
                return None
            return {"pair_id": r.get("pair_id"), "cos": r["cos"],
                    "first": r["substitutable"],
                    "second": int(bool(A.get("covered")) and bool(B.get("covered")))}

        res = [r for r in P.pmap(one, sub, cfg.judge_concurrency, "J3", ctx, pool)
               if isinstance(r, dict)]
        st = agree_stats([r["first"] for r in res], [r["second"] for r in res])
        log.info("")
        log.info(f"  跨模型一致：κ={st['kappa']:.3f}  PABAK={st['pabak']:.3f}  "
                 f"AC1={st['ac1']:.3f}  (n={st['n']})")
        log.info(f"  正例率 第一判官 {st['pos_rate_a']:.3f} / 第二判官 {st['pos_rate_b']:.3f}")
        out = {"x_judge_model": cfg.judge_name, "x_judge_path": a.x_judge_path,
               "first_judge_model": P.Config.load(a.config).judge_name,
               "stats": st, "per_pair": res,
               "note": "这个 κ 是**跨模型**的，可据此讨论误差是否为任务属性；"
                       "论文里必须写出两个模型名。旧的 0.824 来源不明且 "
                       "artifact 复现不出来。"}
        (p2.root / "J").mkdir(parents=True, exist_ok=True)
        wjson(p2.root / "J" / "j3_cross_model.json", out)
        return {"summary": f"跨模型 κ={st['kappa']:.3f} (n={st['n']}), "
                           f"第二判官 {cfg.judge_name}"}
    return p2.run_step("J3", "记名的跨模型 κ", run)


# ==============================================================================
def exp_J4(p2: Phase2) -> dict:
    """用同一个臂重做自覆盖筛选，修掉 J2 留下的混臂问题。

    为什么需要这一步：J2 只重判了覆盖，合格性（self_cov）还是来源 run 那个
    非思考判官定的。对一对逐字相同的 query，j 的检索结果就是 i 的检索结果，
    所以 cov_ij 问的恰好就是 self_cov_i —— 而 self_cov_i 被筛选保证为 1。
    于是混臂会在恒等带上留下一个逻辑上不该存在的缺口：实测
    main_v10 0.998 -> main_v12_think 0.916，缺的 8.4pp 正是两个臂在自覆盖上
    的分歧率，不是 think1 本身的性质。

    这一步重判每个 query 的自覆盖（每个 qid 一次调用，不是每对），按新的
    self_cov 重算 eligible，覆盖判定直接复用 J2 已经判好的。只会让合格集
    变小：被旧筛选挡掉的对本来就没判过覆盖，救不回来。所以结果是
    「两个臂都认为自覆盖成立」的交集，用来检验上面那个假设是对的口径；
    它**不是**一个干净的单臂 run，报数时必须这么说。
    """
    def run():
        log, P, a = p2.log, p2.P, p2.a
        ARMS = {"k1":     dict(K=1, think=False, natural=False),
                "k3":     dict(K=3, think=False, natural=False),
                "k5":     dict(K=5, think=False, natural=False),
                "think1": dict(K=1, think=True,  natural=True),
                "think3": dict(K=3, think=True,  natural=False)}
        spec = ARMS.get(a.arm)
        if spec is None:
            raise ValueError(f"--arm 只能是 {sorted(ARMS)}，收到 {a.arm}")
        THINK = spec["think"]
        src = Path(a.main_run)          # 这里要指 J2 的输出 run
        dst = Path("runs") / a.out_run
        if src.resolve() == dst.resolve():
            raise RuntimeError("--main-run 和 --out-run 不能是同一个目录")
        log.info(f"  自覆盖重判：臂 {a.arm} thinking={THINK}")
        log.info(f"  来源 {src} -> 输出 {dst}")

        md = RunData(a.main_run, log)
        cmapq = {c["qid"]: c for c in md.calls}
        # 只有「当前合格」的对能被保留，所以只需要重判这些对涉及的 query
        elig = [r for r in md.l1 if r.get("eligible")]
        qids = sorted({q for r in elig for q in (r["qid_i"], r["qid_j"])
                       if q in cmapq})
        rate = 0.55 if THINK else 0.37
        log.info(f"  {len(elig)} 对合格对涉及 {len(qids)} 个不同 query，"
                 f"每个 1 次判定；单卡约 {len(qids)*rate/3600:.1f} h，"
                 f"两卡约 {len(qids)*rate/7200:.1f} h")
        if a.dry_run:
            return {"summary": f"dry-run：{len(qids)} 次调用，不执行"}

        exist = dst / "stages" / "s6" / "l1.jsonl"
        if exist.exists() and not a.force:
            raise RuntimeError(f"{exist} 已存在，换 --out-run 或加 --force")

        cfg = P.Config.load(a.config)
        cfg.run_id = a.out_run; cfg.root = "runs"; cfg.resolve()
        set_gpus(cfg, a.gpus, log)
        ctx = P.RunContext(cfg)
        cfg._autotune = P.autotune_hardware(cfg, ctx.log)
        sm = P.ServerManager(ctx)
        p2.register_server(sm)
        link_stages(src, dst, log)

        cmap = {c["chunkid"]: c["text"] for c in
                rjsonl(src / "stages" / "s1" / "chunks.jsonl")}
        lim = cfg.judge_doc_chars
        # 自覆盖用的提示词：phase1 里可能是单独一条，名字不确定，按序试，
        # 用到哪条就打出来 —— 换了提示词就是换了口径，不能默默替换。
        pname, prompt = "L1_PROMPT", None
        for nm in ("SELFCOV_PROMPT", "SELF_COV_PROMPT", "L1_SELF_PROMPT",
                   "L1_PROMPT"):
            if getattr(P, nm, None) is not None:
                pname, prompt = nm, getattr(P, nm)
                break
        if a.selfcov_prompt:
            pname = a.selfcov_prompt
            prompt = getattr(P, pname)
        log.info(f"  自覆盖提示词用的是 P.{pname}")

        urls = sm.ensure("judge")
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries,
                         timeout=300)

        def one(qid):
            c = cmapq.get(qid)
            if not c:
                return None
            ids = c.get("retrieved_chunks") or []
            docs = "\n\n".join(f"[{i+1}] {(cmap.get(x) or '')[:lim]}"
                               for i, x in enumerate(ids)) or "(无结果)"
            j = judge_once(P, pool, cfg, c["query"], docs, prompt=prompt,
                           thinking=THINK)
            if not j.get("ok"):
                return (qid, None)
            return (qid, int(j.get("covered") or 0))

        got = [x for x in P.pmap(one, qids, cfg.judge_concurrency,
                                 f"J4 selfcov {a.arm}", ctx, pool)
               if isinstance(x, tuple)]
        sc = {q: v for q, v in got if v is not None}
        log.info(f"  判定成功 {len(sc)}/{len(qids)}；"
                 f"自覆盖通过率 {np.mean(list(sc.values())):.4f}"
                 if sc else "  全部失败")

        out_recs, kept, dropped, unknown = [], 0, 0, 0
        for r0 in md.l1:
            r0 = dict(r0)
            if not r0.get("eligible"):
                out_recs.append(r0)
                continue
            si, sj = sc.get(r0["qid_i"]), sc.get(r0["qid_j"])
            r0["self_cov_i"], r0["self_cov_j"] = si, sj
            r0["selfcov_arm"] = a.arm
            if si is None or sj is None:
                unknown += 1
                r0["eligible"] = 0
                r0["selfcov_failed"] = True
            elif si and sj:
                kept += 1
            else:
                dropped += 1
                r0["eligible"] = 0
                r0["_inherited"] = dict(r0.get("_inherited") or {},
                                        dropped_by="J4_selfcov")
            out_recs.append(r0)
        log.info(f"  重算合格性：保留 {kept}，因新筛选剔除 {dropped}，"
                 f"判定失败剔除 {unknown}")

        (dst / "stages" / "s6").mkdir(parents=True, exist_ok=True)
        outp = dst / "stages" / "s6" / "l1.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for r0 in out_recs:
                f.write(json.dumps(r0, ensure_ascii=False) + "\n")

        res = [r0 for r0 in out_recs if r0.get("eligible")
               and r0.get("substitutable") is not None]
        st = [r0 for r0 in res if r0.get("same_text")]
        band = float(np.mean([r0["substitutable"] for r0 in st])) if st else None
        log.info("")
        log.info(f"  >>> 恒等带（逐字相同，n={len(st)}）可替换率 "
                 f"{band if band is None else round(band, 4)}")
        log.info("      main_v10 单臂 0.998；main_v12_think 混臂 0.916。")
        log.info("      回到 0.99 以上 = 缺口确实是混臂造成的；"
                 "还在 0.9x 低位 = 判官对自己的判定不自洽，与顺序无关。")

        (dst / "stages" / "s7").mkdir(parents=True, exist_ok=True)
        wjson(dst / "stages" / "s7" / "summary.json",
              {"skipped": True, "reason": "合格集已变，旧 L2 锚定不适用"})
        if not (dst / "stages" / "s7" / "l2.jsonl").exists():
            (dst / "stages" / "s7" / "l2.jsonl").write_text("", encoding="utf-8")
        recompute_qc(src, dst, res, {
            "arm": a.arm, "K": spec["K"], "thinking": THINK,
            "order": "natural", "rule": "selfcov_rescreened"}, log, extra={
            "selfcov_arm": a.arm, "coverage_arm": a.arm,
            "selfcov_rescreened": True,
            "n_kept": kept, "n_dropped_by_new_selfcov": dropped,
            "n_dropped_selfcov_failed": unknown,
            "ONE_SIDED_WARNING":
                "合格集是「两个臂都认为自覆盖成立」的交集：被来源 run 的"
                "筛选挡掉的对没有被重新考虑，所以这是单向收缩，不是干净的"
                "单臂 run。要干净单臂必须在本 run 上从 s6 整个重跑。"})
        wjson(p2.root / "J" / f"j4_selfcov_{a.arm}.json",
              {"arm": a.arm, "n_qids": len(qids), "n_judged": len(sc),
               "selfcov_pass_rate": float(np.mean(list(sc.values()))) if sc else None,
               "kept": kept, "dropped": dropped, "unknown": unknown,
               "identity_band_rate": band, "out_run": a.out_run,
               "prompt_attr": pname})
        log.info("")
        log.info("  下一步（只在恒等带回到 0.99 以上时才有意义）：")
        log.info(f"    python phase1_v13.py --config {a.config} "
                 f"--run-id {a.out_run} --stages s8")
        return {"summary": f"自覆盖重筛：保留 {kept}/剔除 {dropped}，"
                           f"恒等带 {band}"}
    return p2.run_step(f"J4_{p2.a.arm}", f"自覆盖重筛（{p2.a.arm}）", run)


def exp_J5(p2: Phase2) -> dict:
    """判决性实验：同一批对、同一个臂、同样的自然序，重判 R 次。

    要区分的两个假设（见 identity_probe.py）：
      H1 臂间系统性分歧：think1 对自覆盖的判断确实不同于 k1，确定性的。
         预测：那 457 条每次都判 0，重复判定完全一致。
      H2 think1 自身不确定：思考链放大采样方差，那 457 条是掷硬币。
         预测：重复判定里翻的是不同的对，投票分布散开。

    取两组对照：
      flip 组  —— 本 run 判 0、而逻辑上应为 1 的那批（逐字相同 + 结果集排序都同）；
      agree 组 —— 同样逐字相同、但本 run 判 1 的，随机抽同样多。
    两组的重复一致性差别本身就是信息：若只有 flip 组不稳定，说明不稳定集中在
    判官自己也拿不准的那批对上，而不是全局噪声。
    """
    def run():
        log, P, a = p2.log, p2.P, p2.a
        ARMS = {"k1":     dict(K=1, think=False, natural=False),
                "k3":     dict(K=3, think=False, natural=False),
                "k5":     dict(K=5, think=False, natural=False),
                "think1": dict(K=1, think=True,  natural=True),
                "think3": dict(K=3, think=True,  natural=False)}
        spec = ARMS.get(a.arm)
        if spec is None:
            raise ValueError(f"--arm 只能是 {sorted(ARMS)}，收到 {a.arm}")
        THINK, R = spec["think"], max(2, a.j5_reps)
        src = Path(a.main_run)
        log.info(f"  重复判定：臂 {a.arm} thinking={THINK} 重复 {R} 次")
        log.info(f"  来源 {src}")

        md = RunData(a.main_run, log)
        cmapq = {c["qid"]: c for c in md.calls}

        def identical(r):
            return (r.get("eligible") and r.get("same_text")
                    and float(r.get("jaccard") or 0) >= 0.999
                    and float(r.get("rbo") or 0) >= 0.999
                    and r.get("qid_i") in cmapq and r.get("qid_j") in cmapq)

        pool_all = [r for r in md.l1 if identical(r)]
        flip = [r for r in pool_all if r.get("substitutable") == 0]
        agree = [r for r in pool_all if r.get("substitutable") == 1]
        rng = random.Random(20261016)
        n = min(a.j5_n, len(flip)) if flip else 0
        sel_flip = rng.sample(flip, n) if n else []
        sel_agree = rng.sample(agree, min(n or a.j5_n, len(agree))) if agree else []
        sel = [("flip", r) for r in sel_flip] + [("agree", r) for r in sel_agree]
        if not sel:
            raise RuntimeError("没有符合条件的对：来源 run 里找不到"
                               "「逐字相同且结果集与排序都相同」的合格对")
        ncall = len(sel) * R
        rate = 0.55 if THINK else 0.37
        log.info(f"  候选池 {len(pool_all)} 对（flip {len(flip)} / agree {len(agree)}）")
        log.info(f"  抽样 flip {len(sel_flip)} + agree {len(sel_agree)} = {len(sel)} 对，"
                 f"每对 {R} 次 = {ncall} 次；单卡约 {ncall*rate/60:.0f} 分钟")
        if a.dry_run:
            return {"summary": f"dry-run：{ncall} 次调用，不执行"}

        cfg = P.Config.load(a.config)
        cfg.run_id = f"judge_fix_J5_{a.arm}"; cfg.root = "runs"; cfg.resolve()
        set_gpus(cfg, a.gpus, log)
        ctx = P.RunContext(cfg)
        cfg._autotune = P.autotune_hardware(cfg, ctx.log)
        sm = P.ServerManager(ctx)
        p2.register_server(sm)

        cmap = {c["chunkid"]: c["text"] for c in
                rjsonl(src / "stages" / "s1" / "chunks.jsonl")}
        lim = cfg.judge_doc_chars
        urls = sm.ensure("judge")
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries,
                         timeout=300)

        def one(item):
            grp, r = item
            ci, cj = cmapq[r["qid_i"]], cmapq[r["qid_j"]]
            ids = cj.get("retrieved_chunks") or []
            # 和 J2 完全一样的自然序构造 —— 口径必须一致，否则这个实验无意义
            docs = "\n\n".join(f"[{i+1}] {(cmap.get(x) or '')[:lim]}"
                               for i, x in enumerate(ids)) or "(无结果)"
            vs, vds = [], []
            for _ in range(R):
                j = judge_once(P, pool, cfg, ci["query"], docs, thinking=THINK)
                vs.append(j.get("covered") if j.get("ok") else None)
                vds.append(j.get("verdict"))
            ok = [v for v in vs if v is not None]
            return {"group": grp, "pair_id": r.get("pair_id"),
                    "qid_i": r["qid_i"], "qid_j": r["qid_j"],
                    "bin": r.get("bin"), "cos": r.get("cos"),
                    "prev_cov_ij": r.get("cov_ij"),
                    "prev_substitutable": r.get("substitutable"),
                    "votes": vs, "verdicts": vds,
                    "n_ok": len(ok), "n_one": sum(ok),
                    "unanimous": (len(set(ok)) <= 1) if ok else None}

        res = [x for x in P.pmap(one, sel, cfg.judge_concurrency,
                                 f"J5 {a.arm}", ctx, pool)
               if isinstance(x, dict)]

        def summ(grp):
            rs = [r for r in res if r["group"] == grp and r["n_ok"] >= 2]
            if not rs:
                return None
            unan = sum(1 for r in rs if r["unanimous"])
            # 每对内部的成对不一致率，是单次翻转概率的直接估计
            dis = []
            for r in rs:
                v = [x for x in r["votes"] if x is not None]
                m = len(v)
                ones = sum(v)
                pairs = m * (m - 1) / 2
                if pairs:
                    dis.append((ones * (m - ones)) / pairs)
            hist = Counter(r["n_one"] for r in rs)
            return {"n": len(rs), "unanimous": unan,
                    "unanimous_frac": unan / len(rs),
                    "mean_pairwise_disagreement": float(np.mean(dis)) if dis else None,
                    "vote_sum_hist": {str(k): v for k, v in sorted(hist.items())},
                    "mean_rate": float(np.mean([r["n_one"] / r["n_ok"] for r in rs]))}

        out = {"arm": a.arm, "reps": R, "thinking": THINK,
               "n_pool": len(pool_all), "n_flip_pool": len(flip),
               "flip": summ("flip"), "agree": summ("agree")}
        log.info("")
        for grp in ("flip", "agree"):
            s = out[grp]
            if not s:
                continue
            log.info(f"  [{grp}] n={s['n']}  完全一致 {s['unanimous']}"
                     f"（{s['unanimous_frac']:.3f}）  "
                     f"对内不一致率 {s['mean_pairwise_disagreement']:.4f}  "
                     f"平均判 1 比例 {s['mean_rate']:.4f}")
            log.info(f"        投票和分布 {s['vote_sum_hist']}")
        log.info("")
        log.info("  怎么读：flip 组若「完全一致」接近 1.000 且几乎都判 0 -> H1，")
        log.info("          判官确定性地认为这些对不可替换，和 k1 的筛选系统性冲突。")
        log.info("          flip 组若完全一致明显 <1 -> H2，那 457 条是采样噪声，")
        log.info("          think1 在这批对上本身不稳定，不能当成修好了的仪器。")
        wjson(p2.root / "J" / f"j5_repeat_{a.arm}.json", out)
        wjson(p2.root / "J" / f"j5_pairs_{a.arm}.json", res)
        return {"summary": f"重复判定 {len(res)} 对 x{R}；"
                           f"flip 完全一致 "
                           f"{out['flip']['unanimous_frac']:.3f}"
                           if out.get("flip") else "重复判定完成"}
    return p2.run_step(f"J5_{p2.a.arm}", f"重复判定分辨 H1/H2（{p2.a.arm}）", run)


EXP = {"J1": exp_J1, "J3": exp_J3, "J2": exp_J2,
       "J4": exp_J4, "J5": exp_J5}          # J2 最贵，放最后
ORDER = ["J1", "J3", "J2"]       # J4/J5 要在 J2 之后、按需单独跑


def main() -> int:
    ap = argparse.ArgumentParser(description="修判官：顺序不变 + 重新校准")
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp", default="J1")
    ap.add_argument("--phase1", default="phase1_v13.py")
    ap.add_argument("--main-run", default="runs/main_v10")
    ap.add_argument("--distractor-run", default=None)
    ap.add_argument("--out-root", default="runs/phase2")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--agent", default=None)
    # J1
    ap.add_argument("--j1-n", type=int, default=800)
    ap.add_argument("--j1-arms", default="k3,k5,think,inv,xmodel")
    ap.add_argument("--gpu-settle", type=float, default=300.0,
                    help="等显存释放的最长秒数（轮询，够了就立刻继续）")
    ap.add_argument("--gpus", default=None,
                    help="用哪些**绝对**卡号，如 --gpus 1。"
                         "不要用 CUDA_VISIBLE_DEVICES：会二次映射到错的卡")
    ap.add_argument("--judge-port", type=int, default=None,
                    help="judge 的起始端口；默认沿用 config 的 8001，被占会自动+2 找空的")
    ap.add_argument("--need-gb", type=float, default=29.0,
                    help="下一个判官需要的显存，等到有这么多空闲才继续（27GB + 余量）")
    ap.add_argument("--target", type=float, default=0.10,
                    help="闸门：塌陷层翻转率要压到这个值以下才算路线 a 可行")
    # J2
    ap.add_argument("--arm", default="think1",
                    help="k1/k3/k5（关思考+打乱多数票）或 think1（开思考+自然序，推荐）"
                         "/think3")
    ap.add_argument("--out-run", default="main_v12_think")
    # J3
    ap.add_argument("--dry-run", action="store_true",
                    help="J4/J5：只数要判多少次并估时间，不起服务不判定")
    ap.add_argument("--j5-n", type=int, default=300,
                    help="J5：每组抽多少对（flip 组和 agree 组各一份）")
    ap.add_argument("--j5-reps", type=int, default=3,
                    help="J5：每对重复判定几次")
    ap.add_argument("--selfcov-prompt", default=None,
                    help="J4：自覆盖提示词在 phase1 里的属性名，不给就自动探测")
    ap.add_argument("--j3-n", type=int, default=600)
    ap.add_argument("--x-judge-path", default=None,
                    help="第二个判官模型的本地路径，例如 $R/models/Qwen3.8-27B-FP8")
    a = ap.parse_args()

    exps = [e.strip().upper() for e in a.exp.split(",")]
    exps = [e for e in ORDER if e in exps]
    if not exps:
        print(f"可选 {ORDER}")
        return 2
    p2 = Phase2(a)
    p2.log.info(f"修仪器 {exps}")
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
                # 每个实验结束就释放显存。只在 main 的末尾停会让前一个实验的
                # 判官一直占着卡，下一个实验直接起不来 —— 上一轮 J1/J3 就这么死的。
                p2.stop_servers()
                wait_gpu_free(a.need_gb, p2.log, timeout=a.gpu_settle)
    finally:
        p2.stop_servers()
    p2.log.info(f"总耗时 {(time.time()-t0)/60:.1f} min")
    return 1 if p2.failures else 0


if __name__ == "__main__":
    sys.exit(main())
