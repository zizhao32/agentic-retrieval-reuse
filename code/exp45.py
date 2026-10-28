#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
exp45.py —— 补实验 E4（非 MuSiQue 的 agent 工作负载）与 E5（判官的独立验证）
================================================================================
对应外部评审意见第 4、5 条。所有步骤都直接 import phase1_v13.py 复用其实现
（检索器、agent 循环、_fmt、xjson、ServerManager、pmap、s8 校准），
不另写一套逻辑 —— 否则对照就不止差一个变量。

  步骤 0  audit       判官 prompt 审计（★ 先跑，E5 依赖它；约 10 min GPU）
          main_v10 的 s6 到底用的哪个 prompt？
            - think1（main_v12_think）走 fix_judge.judge_once 的思考分支，
              代码写死 `prompt or P.L1_PROMPT` —— 用的是 base（c2a47c11…）。
            - main_v10 的 10 档 sim_bins（含 0.98 / 0.999 切点）只在 v13 里有，
              而 v13 的 judge_order_invariant 默认 True —— 若没被 config 关掉、
              也没跑 s22 对拍，main_v10 用的是 order_inv（781d9f4d…）。
          这决定了 §6「两臂只差推理预算」那句话真假。静态证据（config / 日志 /
          s22）只能推断；本步再做一个**指纹实验**：从 main_v10 的 l1 里分层抽
          ~160 对，用主判官在 temperature 0、自然序下按每个候选 prompt 重判
          ij 方向，与存档的 verdict_ij **和 facts_ij（逐字）**比对。
          用对了 prompt，facts 逐字复现率很高；用错了几乎为 0。
          结果写 runs/phase3/audit/prompt_audit.json，E5/E4 的判官调用都从这里
          取 prompt 并校验 sha256。

  E5a  跨模型族判官（Llama-3.3-70B-Instruct AWQ-INT4，约 8–10 h）
          与 main_v10 **完全相同**的输入：同一批合格对（eligible & ok）、
          同一 prompt（审计得出，sha 校验）、自然序、temperature 0、不思考、
          max_tokens 600、judge_doc_chars 1100。只换模型族。
          先跑闸门（gold 对 + 200 个决策带对）：解析失败率 / 截断率 / 金标准确率
          不过关就停，不白烧 8 小时。闸门判过的对计入全量，不重复花钱。
          产出新 run（s1–s5 软链主 run），走 phase1 的 s8 原路径重新校准。
  E5b  消除 prompt 混杂（仅当审计判定 main 的 prompt ≠ base 时自动跑）
          think_mainprompt：Qwen3-32B 开思考 + main 的 prompt（与 think1 只差 prompt）
          nothink_base    ：Qwen3-32B 不思考 + base prompt（与 main 只差 prompt）
          与已有的 main / think1 凑成 {prompt} × {推理} 的 2×2。
  E5c  盲测留出人工集（只导出 + 评估；标注要人来做）
          从 main_v10 合格对里按决策带分层抽，**排除已有金标对**（留出），
          每对拆成两个方向、全局打乱，生成离线 HTML 标注页（不含任何判官标签）。
          标完后 e5c-eval 计算标注者间一致性，以及每个判官臂对人工的准确率。

  E4   非 MuSiQue 的 agent 工作负载（Tongyi agent 跑 2WikiMultihopQA，约 5–6 h）
          collect   与 main_v10 同一 agent、同一语料与索引、同一 agent prompt，
                    只把任务源从 MuSiQue 换成 2Wiki（前 2000 题，与 replay:2wiki 同源）
          merge     新 run = main_v10 的 replay 调用 + 2Wiki agent 调用
                    （与 agent_b.py 同一做法；source 仍记为 "agent"，并加
                    agent_dataset="2wiki"，这样 s10 里所有针对 "agent" 的逻辑原样生效）
          downstream 重跑 s3,s4,s9,s10,s13；s8/calib.json 软链 main_v10（检测器固定）
          validate  在 2Wiki-agent 的候选对上用主判官（同 prompt）分层抽判，
                    检查 main_v10 标定出的 τ* 在新工作负载上是否仍达到保证水平
          report    与 main_v10 的 MuSiQue-agent 逐项对照

所有产出
--------
  runs/phase3/
    state.json                每步状态、耗时、摘要（可断点续跑）
    manifest.json             环境、模型 repo+commit、数据集指纹、脚本 sha256
    SUMMARY.md / summary.json 总报告（report 生成；summary.json 键稳定，供宏溯源）
    logs/<step>.log           每步完整日志（含 phase1 内部日志）
    logs/phase1_<run>_<stages>.log   调用 phase1 子进程的完整输出
    configs/parity_main.json  从 main_v10/config_effective.json 派生的对齐配置
    audit/                    prompt_audit.json、fingerprint.jsonl、config_parity.json
    e5/<arm>/                 partial.jsonl（逐对原始判定，断点续跑用）、gate.json、rejudge.json
    e5/compare.json           各臂一致性 / τ* / 恒等带 / 金标准确率
    e5/human/                 annotate.html、key.json、eval.json
    e4/                       collect_partial.jsonl、agent_2wiki_calls.jsonl、
                              collect_stats.json、validate_pairs.jsonl、validate.json、compare.json
  vLLM 日志在各自 run 的 logs/vllm_*_gpu*.log。

用法（在 projB 目录下，runs/ 所在处）
-----------------------------------
    R=/mnt/data/user_workspace/zhangzizhao/matview
    PY=$R/envs/main/bin/python

    $PY exp45.py download                 # 下载 Llama AWQ、校验已有模型与 2Wiki 数据
    $PY exp45.py all --smoke              # 先冒烟：全流程小规模（约 1 h），验证管线
    $PY exp45.py all                      # 正式：audit → E5a → E4 → E5b → E5c 导出 → 报告
    # 或者用 run_exp45.sh 包一层 nohup

    # 单步
    $PY exp45.py audit
    $PY exp45.py e5a [--gate-only]
    $PY exp45.py e5b
    $PY exp45.py e4-collect | e4-merge | e4-downstream | e4-validate | e4-report
    $PY exp45.py e5c-export
    $PY exp45.py e5c-eval --labels ann1.tsv ann2.tsv
    $PY exp45.py e5-report
    $PY exp45.py report

断点续跑：每步完成后记入 state.json，`all` 会跳过已完成的步；E5 的逐对判定和
E4 的逐任务采集都边跑边落盘，中断后重跑只补没做完的。要强制重做某步加 --redo。
================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import html
import importlib.util
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20261022
LOG = logging.getLogger("exp45")

# ------------------------------------------------------------------ 模型登记
# repo 已在 huggingface.co 上核实存在（2026-10-22）；第一个失败才用第二个。
XFAM_JUDGES = {
    "llama33-70b-awq": dict(
        repos=["ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
               "casperhansen/llama-3.3-70b-instruct-awq"],
        local="Llama-3.3-70B-Instruct-AWQ-INT4",
        served="Llama-3.3-70B-Instruct-AWQ",
        family="llama",
        weights_gb=38.0,      # 模型卡：加载约 35 GiB；vLLM 实测 37.07 GiB
        min_kv_gb=14.0,       # KV ≥8 GB + 激活/CUDA graph 约 6 GB。只给 8 时，卡上有残留进程
                              # 会让 autotune 放行一个起不来的配置（实测 KV 只剩 1.29 GiB）
        ctx=16384,            # 判官提示约 3–4k token，16k 足够且省 KV
        max_seqs=32,
        sec_per_call=0.9,     # 单卡粗估，只用于打印 ETA
        disk_gb=42.0,
        license="Llama 3.3 Community License（发表需注明 Built with Llama）"),
}
KNOWN_REPOS = {   # 已有模型缺失时的下载源（正常情况下 models/ 里都有）
    "Qwen3-32B-AWQ": "Qwen/Qwen3-32B-AWQ",
    "Tongyi-DeepResearch-30B-A3B": "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
}

# E5c 人工集的分层（决策带为主；恒等带只取非强制的）
HUMAN_BANDS = [("[0.80,0.90)", 0.80, 0.90), ("[0.90,0.95)", 0.90, 0.95),
               ("[0.95,0.98)", 0.95, 0.98), ("[0.98,0.999)", 0.98, 0.999),
               ("[0.999,1] non-forced", 0.999, 1.0001)]


# ==============================================================================
#  基础设施：日志 / 状态 / IO
# ==============================================================================

def setup_log(p3: Path, step: str):
    (p3 / "logs").mkdir(parents=True, exist_ok=True)
    LOG.handlers.clear(); LOG.setLevel(logging.DEBUG); LOG.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    fh = logging.FileHandler(p3 / "logs" / f"{step}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    ah = logging.FileHandler(p3 / "logs" / "exp45_all.log", encoding="utf-8")
    ah.setLevel(logging.INFO); ah.setFormatter(fmt)
    for h in (sh, fh, ah):
        LOG.addHandler(h)


def attach_matview(ctx, p3: Path, step: str):
    """RunContext 每次创建都会清空 'matview' logger 的 handler，这里补挂一个，
    让 phase1 内部的日志（pmap 进度、vLLM 启停、autotune）也进本步日志。"""
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | [p1] %(message)s", "%m-%d %H:%M:%S")
    fh = logging.FileHandler(p3 / "logs" / f"{step}.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logging.getLogger("matview").addHandler(fh)


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def wj(p, o):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(o, ensure_ascii=False, indent=2, default=_jdef), encoding="utf-8")
    tmp.replace(p)


def _jdef(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def rjl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def wjl(p, rows):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_jdef) + "\n")
    tmp.replace(p)


class Appender:
    """线程安全的 jsonl 追加器 —— 长任务边跑边落盘，断点续跑靠它。"""
    def __init__(self, p: Path):
        self.p = Path(p); self.p.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def add(self, rec):
        s = json.dumps(rec, ensure_ascii=False, default=_jdef) + "\n"
        with self.lock:
            with open(self.p, "a", encoding="utf-8") as f:
                f.write(s); f.flush()

    def load(self, key):
        """读回已有记录；同一 key 以最后一条为准（重试成功会覆盖之前的失败）。"""
        d = {}
        if self.p.exists():
            with open(self.p, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        r = json.loads(ln)
                    except Exception:
                        continue          # 被 kill 时可能留下半行
                    d[r[key]] = r
        return d


def psha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fsha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class State:
    def __init__(self, p3: Path):
        self.p = p3 / "state.json"
        self.d = rj(self.p) if self.p.exists() else {"steps": {}}

    def get(self, k):
        return self.d["steps"].get(k, {})

    def mark(self, k, **kw):
        self.d["steps"].setdefault(k, {}).update(kw)
        wj(self.p, self.d)


def run_step(a, name, fn):
    """统一的步骤包装：跳过已完成 / 记录耗时与摘要 / 失败留 traceback。"""
    st = State(a.p3)
    if st.get(name).get("status") == "done" and not a.redo:
        LOG.info(f"[{name}] 已完成（{st.get(name).get('summary','')}），跳过。要重做加 --redo")
        return st.get(name)
    setup_log(a.p3, name)
    LOG.info("=" * 78); LOG.info(f"[{name}] 开始"); LOG.info("=" * 78)
    t0 = time.time()
    st.mark(name, status="running", start=time.strftime("%F %T"), error=None, end=None)
    try:
        summ = fn(a)
    except SystemExit:
        raise
    except BaseException as e:
        tb = traceback.format_exc()
        LOG.error(tb)
        st.mark(name, status="failed", error=f"{type(e).__name__}: {e}",
                duration_h=round((time.time() - t0) / 3600, 3))
        raise
    dur = (time.time() - t0) / 3600
    st.mark(name, status="done", end=time.strftime("%F %T"), duration_h=round(dur, 3),
            summary=str(summ)[:300])
    LOG.info(f"[{name}] 完成 {dur:.2f} h — {summ}")
    return st.get(name)


# ==============================================================================
#  phase1 复用
# ==============================================================================

_P = None
_ROLE_REQ0 = None
_PARITY = None


def P1(a):
    global _P
    if _P is None:
        path = Path(a.phase1).resolve()
        if not path.exists():
            sys.exit(f"找不到 {path}（用 --phase1 指定 phase1_v13.py）")
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("p13", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["p13"] = mod
        spec.loader.exec_module(mod)
        _P = mod
        global _ROLE_REQ0
        _ROLE_REQ0 = {k: dict(v) for k, v in mod.ROLE_REQ.items()}
    return _P


def parity_config(a) -> Path:
    """从 main_v10/config_effective.json 派生对齐配置。

    phase1 每次调用都按 --config 重新加载配置；若用当前 config.json，
    任何自 main_v10 之后改过的字段（sim_bins、分块参数、judge_* …）都会悄悄
    改变下游结果。这里以 main_v10 实际生效的配置为准，并把与当前
    config.json 的差异全部列出来。"""
    global _PARITY
    if _PARITY is not None and _PARITY.exists():
        return _PARITY
    P = P1(a)
    out = a.p3 / "configs" / "parity_main.json"
    ce_p = a.main_run / "config_effective.json"
    fresh = dataclasses.asdict(P.Config())
    if ce_p.exists():
        ce = rj(ce_p)
    else:
        LOG.warning(f"{ce_p} 不存在，退回用 {a.config}（对齐性无法保证）")
        ce = rj(a.config) if a.config else {}
    keep = {k: v for k, v in ce.items() if k in fresh and not k.startswith("_")}
    dropped = sorted(k for k in ce if k not in fresh or k.startswith("_"))
    keep["root"] = str(a.runs_root)
    keep["gold_tsv"] = ""          # 防止 phase1 往软链过来的 s5 里回写
    cur = rj(a.config) if a.config and Path(a.config).exists() else {}
    diffs = {k: {"main_v10": keep.get(k), "current": cur[k]}
             for k in cur if k in keep and keep.get(k) != cur[k]
             and k not in ("run_id", "root", "gold_tsv")}
    wj(out, keep)
    wj(a.p3 / "audit" / "config_parity.json",
       {"source": str(ce_p), "source_exists": ce_p.exists(),
        "source_mtime": time.strftime("%F %T", time.localtime(ce_p.stat().st_mtime))
        if ce_p.exists() else None,
        "dropped_unknown_fields": dropped, "diff_vs_current_config": diffs})
    if diffs:
        LOG.warning(f"当前 config.json 与 main_v10 实际生效配置有 {len(diffs)} 处不同"
                    f"（以 main_v10 为准）：{sorted(diffs)[:12]}")
    _PARITY = out
    return out


def make_cfg(a, run_id: str, **over):
    P = P1(a)
    cfg = P.Config.load(str(parity_config(a)))
    cfg.run_id = run_id
    cfg.root = str(a.runs_root)
    for k, v in over.items():
        setattr(cfg, k, v)
    if a.gpus:
        cfg.gpus = [int(x) for x in a.gpus.split(",")]
    if a.no_servers:
        cfg.manage_servers = False
    return cfg.resolve()


def gpu_free_gb():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=20).stdout.decode().split()
        return [float(x) / 1024 for x in o]
    except Exception:
        return []


def gpu_procs():
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True,
                              timeout=20).stdout.strip().splitlines()
    except Exception:
        return []


def wait_gpu(gpus, need_gb, timeout=300):
    """停掉上一个 vLLM 后显存归还有延迟；autotune 量早了会把 mem_util 算低。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        f = gpu_free_gb()
        if not f or all(g < len(f) and f[g] >= need_gb for g in gpus):
            return
        time.sleep(5)
    f = gpu_free_gb()
    LOG.error(f"等 {timeout}s 后 GPU{gpus} 空闲 {[round(x,1) for x in f]} GB 仍 < 需要的 {need_gb:.0f} GB")
    for ln in gpu_procs():
        LOG.error(f"  占用中：{ln}")
    raise RuntimeError(f"GPU 显存不够（需要每卡 ≥{need_gb:.0f} GB 空闲）。看上面列出的进程；若是残留："
                       f"pkill -9 -f 'vllm serve'; pkill -9 -f EngineCore; sleep 15; nvidia-smi")


def start_ctx(a, cfg, step, role_req=None):
    """建 RunContext + autotune + ServerManager。role_req 覆盖判官显存需求（Llama 70B）。"""
    P = P1(a)
    ctx = P.RunContext(cfg)
    attach_matview(ctx, a.p3, step)
    P.ROLE_REQ["judge"] = dict(role_req) if role_req else dict(_ROLE_REQ0["judge"])
    if not a.no_servers:
        need = sum(P.ROLE_REQ["judge"].values()) + 3.0
        wait_gpu(cfg.gpus, need)
        cfg._autotune = P.autotune_hardware(cfg, ctx.log)
        ctx.wjson(ctx.base / "autotune.json", cfg._autotune)
    sm = P.ServerManager(ctx)

    def sig(s, f):
        LOG.warning(f"收到信号 {s}，停服务退出")
        with contextlib.suppress(Exception):
            sm.stop()
        sys.exit(130)
    signal.signal(signal.SIGINT, sig); signal.signal(signal.SIGTERM, sig)
    return ctx, sm


def run_phase1(a, run_id: str, stages: str, force=False):
    """调用 phase1_v13.py 跑既有 stage（走原来那条已验证的路径，不自己重写）。"""
    cmd = [a.python, str(Path(a.phase1).resolve()), "--config", str(parity_config(a)),
           "--run-id", run_id, "--stages", stages]
    if force:
        cmd.append("--force")
    if a.no_servers:
        cmd.append("--no-servers")
    lp = a.p3 / "logs" / f"phase1_{run_id}_{stages.replace(',', '-')}.log"
    LOG.info(f"  $ {' '.join(cmd)}")
    LOG.info(f"    输出 -> {lp}")
    t0 = time.time()
    with open(lp, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%F %T')} {' '.join(cmd)} =====\n"); f.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for ln in p.stdout:
            f.write(ln)
            if any(k in ln for k in ("完成", "τ*", "保证", "ERROR", "Error", "Traceback",
                                     "头部", "意图")):
                LOG.info(f"    [p1] {ln.rstrip()[:200]}")
        rc = p.wait()
    LOG.info(f"    phase1 退出码 {rc}，{(time.time()-t0)/60:.1f} min")
    if rc != 0:
        raise RuntimeError(f"phase1 {stages} 失败（rc={rc}），看 {lp}")


# ==============================================================================
#  判官调用（与 _judge1 逐字节等价，但 prompt / 思考 / 模型族显式传入）
# ==============================================================================

def judge_call(P, pool, cfg, tmpl, q, docs, thinking, max_tokens, family):
    """thinking=False 且 family=qwen3 时，请求体与 P._judge1(pool,cfg,q,docs,tmpl)
    完全相同（同一 user 消息、temperature、max_tokens、enable_thinking=False）。
    Llama 的 chat template 不认 enable_thinking，不发这个参数。"""
    kw = {}
    if family == "qwen3":
        kw = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    r = pool.chat([{"role": "user", "content": tmpl.format(q=q, docs=docs)}],
                  temperature=cfg.judge_temperature, max_tokens=max_tokens, **kw)
    base = {"tok_in": r["prompt_tokens"], "tok_out": r["completion_tokens"],
            "finish": r["finish_reason"], "truncated": r["finish_reason"] == "length"}
    j = P.xjson(r["text"])
    if not j:
        return {"ok": False, "verdict": "PARSE_FAIL", "raw": r["text"][-600:], **base}
    v = str(j.get("verdict", "")).upper().strip()
    return {"ok": True, "verdict": v, "covered": int(v == "SUFFICIENT"),
            "facts": j.get("required_facts", []), **base}


# L1_PROMPT（base, sha c2a47c11…）的逐句直译。只用于语言诊断：判断 Llama 在金标上
# 掉分是不是因为中文指令。用它跑全量时，臂名与 run 名都带 _en，报告里单独列出。
PROMPT_BASE_EN = """You need to decide whether a set of retrieval results is sufficient to answer a question.

**Question**
{q}

**Retrieval results**
{docs}

Follow these steps:
1. List the facts that are **required** to answer this question (1-3 items, specific to entities and attributes).
2. Check one by one whether the retrieval results explicitly contain each fact.
3. Give the conclusion.

Note: only consider what is in the retrieval results; do not fill in with your own knowledge. Anything not explicitly stated in the results counts as false.

Output only JSON, with no other text:
{{"required_facts":["..."],"found":[true,false],"verdict":"SUFFICIENT"|"PARTIAL"|"INSUFFICIENT","confidence":0.0}}"""


def prompt_table(a):
    t = dict(P1(a)._prompt_variants())
    t["base_en"] = PROMPT_BASE_EN
    return t


def audited_prompt(a):
    """E4/E5 取 main_v10 的 prompt：只认审计结果，且校验 sha。"""
    ap = a.p3 / "audit" / "prompt_audit.json"
    if a.prompt_override:
        pid = a.prompt_override
        LOG.warning(f"--prompt-override {pid}：跳过审计结论，后果自负")
    else:
        if not ap.exists():
            raise RuntimeError("还没跑 audit。E4/E5 必须用 main_v10 实际用过的 prompt：先跑 "
                               "`exp45.py audit`")
        au = rj(ap)
        dec = au["decision"]
        if dec["status"] != "identified" and not a.accept_static:
            raise RuntimeError(f"审计结论是 {dec['status']}（{dec.get('note','')}），"
                               f"未能确认 prompt。确认后用 --prompt-override <id> 或 "
                               f"--accept-static 继续")
        pid = dec["variant"]
    tbl = prompt_table(a)
    if pid not in tbl:
        raise RuntimeError(f"未知 prompt 变体 {pid}，可选 {sorted(tbl)}")
    text = tbl[pid]
    if not a.prompt_override and ap.exists():
        want = rj(ap)["decision"].get("sha256")
        if want and want != psha(text):
            raise RuntimeError(f"prompt {pid} 的 sha 与审计时不同 —— phase1_v13.py 被改过？")
    return pid, text


# ==============================================================================
#  数据装载
# ==============================================================================

class MainRun:
    def __init__(self, run: Path):
        self.run = run
        self.calls = {c["qid"]: c for c in rjl(run / "stages" / "s2" / "calls.jsonl")}
        self._cmap = None
        self._l1 = None

    @property
    def cmap(self):
        if self._cmap is None:
            self._cmap = {c["chunkid"]: c["text"]
                          for c in rjl(self.run / "stages" / "s1" / "chunks.jsonl")}
        return self._cmap

    @property
    def l1(self):
        if self._l1 is None:
            self._l1 = rjl(self.run / "stages" / "s6" / "l1.jsonl")
        return self._l1

    def eligible(self):
        return [r for r in self.l1 if r.get("eligible") and r.get("ok")]


def load_gold(a):
    """{pair_id: 0/1}。先读 main 的 s5/gold_pairs.jsonl，没有标签再读 gold_tsv（只读）。"""
    lab = {}
    gp = a.main_run / "stages" / "s5" / "gold_pairs.jsonl"
    if gp.exists():
        for g in rjl(gp):
            v = g.get("human_label")
            if v in (0, 1, "0", "1"):
                lab[g["pair_id"]] = int(v)
    if not lab:
        ce = a.main_run / "config_effective.json"
        tsv = (rj(ce).get("gold_tsv") if ce.exists() else "") or ""
        if tsv and Path(tsv).exists():
            import csv
            with open(tsv, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    pid = (row.get("pair_id") or "").strip()
                    v = (row.get("human_label") or "").strip()
                    if pid and v in ("0", "1"):
                        lab[pid] = int(v)
    return lab


def gold_pair_ids(a):
    gp = a.main_run / "stages" / "s5" / "gold_pairs.jsonl"
    return {g["pair_id"] for g in rjl(gp)} if gp.exists() else set()


def bin_of(cos, edges):
    for b in range(len(edges) - 1):
        if edges[b] <= cos < edges[b + 1]:
            return b
    return len(edges) - 2 if cos >= edges[-1] else 0


def strat_sample(rows, edges, n_per_bin, seed, key=lambda r: r["cos"]):
    by = defaultdict(list)
    for r in rows:
        by[bin_of(key(r), edges)].append(r)
    rng = random.Random(seed); out = []
    for b in sorted(by):
        lst = sorted(by[b], key=lambda r: r.get("pair_id", ""))
        k = n_per_bin(b) if callable(n_per_bin) else n_per_bin
        out += rng.sample(lst, min(k, len(lst)))
    return out


def is_forced_identity(r):
    return bool(r.get("same_text")) and float(r.get("jaccard") or 0) >= 0.999 \
        and float(r.get("rbo") or 0) >= 0.999


# ==============================================================================
#  统计
# ==============================================================================

def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def agree(xs, ys):
    """与 crosstab_l1.py 同一套定义：κ / PABAK / Gwet AC1，未加权。"""
    pairs = [(int(bool(x)), int(bool(y))) for x, y in zip(xs, ys)
             if x is not None and y is not None]
    n = len(pairs)
    if not n:
        return {"n": 0}
    c = Counter(pairs)
    p0 = (c[(0, 0)] + c[(1, 1)]) / n
    pa = (c[(1, 0)] + c[(1, 1)]) / n
    pb = (c[(0, 1)] + c[(1, 1)]) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    pi = (pa + pb) / 2; pe2 = 2 * pi * (1 - pi)
    return {"n": n, "agree": p0, "kappa": (p0 - pe) / (1 - pe) if pe < 1 else None,
            "pabak": 2 * p0 - 1, "ac1": (p0 - pe2) / (1 - pe2) if pe2 < 1 else None,
            "rate_a": pa, "rate_b": pb, "a1b0": c[(1, 0)], "a0b1": c[(0, 1)]}


def acc_vs(truth, pred):
    pr = [(t, p) for t, p in zip(truth, pred) if t is not None and p is not None]
    n = len(pr)
    if not n:
        return {"n": 0}
    tp = sum(1 for t, p in pr if t == 1 and p == 1)
    tn = sum(1 for t, p in pr if t == 0 and p == 0)
    fp = sum(1 for t, p in pr if t == 0 and p == 1)
    fn = sum(1 for t, p in pr if t == 1 and p == 0)
    lo, hi = wilson(tp + tn, n)
    plo, phi = wilson(tp, tp + fp) if tp + fp else (None, None)
    return {"n": n, "acc": (tp + tn) / n, "acc_ci": [lo, hi],
            "precision": tp / (tp + fp) if tp + fp else None, "precision_ci": [plo, phi],
            "recall": tp / (tp + fn) if tp + fn else None,
            "kappa": agree(truth, pred).get("kappa"),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def pct(x, d=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{100 * x:.{d}f}%"


# ==============================================================================
#  download —— 模型与数据
# ==============================================================================

def cmd_download(a):
    P = P1(a)
    cfg = make_cfg(a, "phase3_download")
    mdir = Path(cfg.models_dir)
    man_p = a.p3 / "manifest.json"
    man = rj(man_p) if man_p.exists() else {}
    man.setdefault("models", {}); man.setdefault("datasets", {})
    ep = os.environ.get("HF_ENDPOINT")
    LOG.info(f"HF_ENDPOINT={ep or '(未设置，直连 huggingface.co)'}；国内网络建议 "
             f"export HF_ENDPOINT=https://hf-mirror.com 后再跑")
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        LOG.info("hf_transfer 可用，已开启多线程下载")
    except Exception:
        pass
    from huggingface_hub import HfApi, snapshot_download

    def fetch(repos, local, need_gb):
        dest = mdir / local
        if (dest / "config.json").exists() and _weights_complete(dest):
            LOG.info(f"  {local} 已存在且权重完整：{dest}")
            return dest, man["models"].get(local, {}).get("repo"), \
                man["models"].get(local, {}).get("revision")
        free = shutil.disk_usage(mdir if mdir.exists() else mdir.parent).free / 2**30
        if free < need_gb:
            raise RuntimeError(f"{mdir} 只剩 {free:.0f} GB < 需要 {need_gb:.0f} GB")
        last = None
        for repo in repos:
            try:
                rev = None
                with contextlib.suppress(Exception):
                    rev = HfApi().model_info(repo).sha
                LOG.info(f"  下载 {repo}@{rev or 'main'} -> {dest}（断点续传；约 {need_gb:.0f} GB）")
                snapshot_download(repo_id=repo, revision=rev, local_dir=str(dest),
                                  allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                  "*.model", "*.tiktoken", "LICENSE*",
                                                  "USE_POLICY*", "*.md", "*.jinja"],
                                  max_workers=8)
                if not _weights_complete(dest):
                    raise RuntimeError("下载结束但 index 里列出的分片不全")
                return dest, repo, rev
            except Exception as e:
                last = e
                LOG.warning(f"  {repo} 失败：{type(e).__name__}: {str(e)[:200]}")
        raise RuntimeError(f"所有源都失败：{last}")

    # ---- 跨模型族判官 ----
    spec = XFAM_JUDGES[a.xfam]
    dest, repo, rev = fetch(spec["repos"], spec["local"], spec["disk_gb"])
    c = rj(dest / "config.json")
    qc = c.get("quantization_config") or {}
    LOG.info(f"  {spec['local']}: arch={c.get('architectures')} quant={qc.get('quant_method')} "
             f"bits={qc.get('bits')} group={qc.get('group_size')} zero_point={qc.get('zero_point')}")
    if str(qc.get("quant_method", "")).lower() != "awq":
        raise RuntimeError("不是 AWQ 权重；ServerManager 的判官角色写死了 --quantization awq_marlin")
    if "modules_to_not_convert" in qc and qc["modules_to_not_convert"] is None:
        LOG.warning("  quantization_config.modules_to_not_convert 为 null（模型卡提到过的告警）。"
                    "vLLM 能处理；若启动报错，把它改成 [] 再试（先备份 config.json）")
    man["models"][spec["local"]] = {
        "repo": repo, "revision": rev, "path": str(dest),
        "bytes": sum(p.stat().st_size for p in dest.glob("*.safetensors")),
        "config_sha256": fsha(dest / "config.json"), "license": spec["license"]}

    # ---- 已有模型：只校验，缺了才下 ----
    for local, repo in KNOWN_REPOS.items():
        d = mdir / local
        ok = (d / "config.json").exists() and _weights_complete(d)
        if not ok and a.fetch_missing:
            d, repo, rev = fetch([repo], local, 70.0)
            ok = True
        man["models"][local] = {**man["models"].get(local, {}), "path": str(d),
                                "present": ok,
                                "config_sha256": fsha(d / "config.json")
                                if (d / "config.json").exists() else None}
        (LOG.info if ok else LOG.error)(
            f"  {local}: {'OK' if ok else '缺失！加 --fetch-missing 自动下载 ' + repo}")
    for k in ("judge_path", "agent_path"):
        LOG.info(f"  parity {k} = {getattr(cfg, k)}  存在={Path(getattr(cfg, k), 'config.json').exists()}")

    # ---- 2WikiMultihopQA（与 phase1 同一个源、同一个 cache）----
    from datasets import load_dataset
    for repo, sub, split in P._DATASET_SRC["2wiki"]:
        try:
            kw = {"cache_dir": cfg.hf_cache, "split": split}
            d = load_dataset(repo, sub, **kw) if sub else load_dataset(repo, **kw)
            feats = list(d.features)
            LOG.info(f"  2Wiki {repo}/{split}: {len(d)} 条，字段 {feats[:10]}")
            if "evidences" not in feats:
                LOG.warning("  没有 evidences 字段：replay 子查询会退化成整题，但 E4 的 agent 不受影响")
            man["datasets"]["2wiki"] = {"repo": repo, "split": split, "n": len(d),
                                        "fingerprint": getattr(d, "_fingerprint", None),
                                        "cache_dir": cfg.hf_cache}
            break
        except Exception as e:
            LOG.error(f"  2Wiki {repo} 加载失败：{type(e).__name__}: {str(e)[:200]}")
    else:
        raise RuntimeError("2WikiMultihopQA 加载失败；检查 HF_ENDPOINT / 网络")

    # ---- vLLM 版本 ----
    vpy = Path(cfg.serve_python).parent / "python"
    with contextlib.suppress(Exception):
        v = subprocess.run([str(vpy), "-c", "import vllm;print(vllm.__version__)"],
                           capture_output=True, text=True, timeout=120).stdout.strip()
        man["vllm"] = v
        LOG.info(f"  vLLM {v}（{vpy}）")
    wj(man_p, man)
    return f"{spec['local']} @ {rev} 就绪；2Wiki {man['datasets']['2wiki']['n']} 条"


def _weights_complete(d: Path) -> bool:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        try:
            need = set(rj(idx)["weight_map"].values())
        except Exception:
            return False
        return all((d / f).exists() and (d / f).stat().st_size > 0 for f in need)
    return any(d.glob("*.safetensors"))


def write_manifest(a):
    man_p = a.p3 / "manifest.json"
    man = rj(man_p) if man_p.exists() else {}
    man["python"] = sys.version.split()[0]
    man["platform"] = platform.platform()
    man["host"] = platform.node()
    with contextlib.suppress(Exception):
        import torch
        man["torch"] = torch.__version__
    with contextlib.suppress(Exception):
        man["gpus"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True,
                                     timeout=20).stdout.strip().splitlines()
    man["scripts"] = {Path(p).name: fsha(p) for p in (__file__, a.phase1) if Path(p).exists()}
    man["main_run"] = str(a.main_run)
    man["updated"] = time.strftime("%F %T")
    wj(man_p, man)


# ==============================================================================
#  步骤 0：prompt 审计
# ==============================================================================

def cmd_audit(a):
    P = P1(a)
    tbl = prompt_table(a)
    shas = {k: psha(v) for k, v in tbl.items()}
    ev = {"prompt_sha256": {k: v[:12] for k, v in shas.items()}}

    ce_p = a.main_run / "config_effective.json"
    ce = rj(ce_p) if ce_p.exists() else {}
    ev["config_effective"] = {
        "exists": ce_p.exists(),
        "mtime": time.strftime("%F %T", time.localtime(ce_p.stat().st_mtime)) if ce_p.exists() else None,
        "has_judge_order_invariant_field": "judge_order_invariant" in ce,
        "judge_order_invariant": ce.get("judge_order_invariant"),
        "sim_bins": ce.get("sim_bins"),
        "judge_path": ce.get("judge_path"), "judge_max_tokens": ce.get("judge_max_tokens"),
        "judge_doc_chars": ce.get("judge_doc_chars"),
        "judge_temperature": ce.get("judge_temperature")}
    sb = ce.get("sim_bins") or []
    ev["code_version_hint"] = ("v13（10 档 sim_bins 含 0.98/0.999）" if (0.98 in sb and 0.999 in sb)
                               else "v12 或更早（8 档 sim_bins）" if sb else "未知")
    s22 = a.main_run / "stages" / "s22" / "selected.json"
    ev["s22_selected"] = rj(s22) if s22.exists() else None

    st_p = a.main_run / "state.json"
    if st_p.exists():
        s6 = (rj(st_p).get("stages") or {}).get("s6") or {}
        ev["s6_state"] = {k: s6.get(k) for k in ("status", "start", "duration_sec")}
        if s6.get("start"):
            ev["s6_state"]["start_h"] = time.strftime("%F %T", time.localtime(s6["start"]))

    # 日志里 s6 段落后的 [prompt] 行（v12+ 才会打）
    lp = a.main_run / "logs" / "phase1.log"
    lines = []
    if lp.exists():
        in_s6, cur = False, []
        with open(lp, encoding="utf-8", errors="ignore") as f:
            for ln in f:
                if re.search(r"\[s6\] ", ln) and "完成" not in ln and "跳过" not in ln:
                    in_s6, cur = True, [ln.strip()[:200]]
                elif re.search(r"\[s\d+\] ", ln) and "[s6]" not in ln:
                    if in_s6 and cur:
                        lines.append(cur)
                    in_s6, cur = False, []
                elif in_s6 and ("[prompt]" in ln or "顺序无关" in ln):
                    cur.append(ln.strip()[:200])
        if in_s6 and cur:
            lines.append(cur)
    ev["log_s6_sections"] = lines[-5:]
    ev["log_exists"] = lp.exists()

    # 静态推断
    if not ce:
        static, why = None, "没有 config_effective.json"
    elif ev["s22_selected"] and ev["s22_selected"].get("winner") in tbl:
        static, why = ev["s22_selected"]["winner"], "s22/selected.json 存在 → s6 用对拍胜者"
    elif "judge_order_invariant" not in ce:
        static, why = "base", "配置里没有 judge_order_invariant 字段 → v10 代码，写死 L1_PROMPT"
    elif ce.get("judge_order_invariant"):
        static, why = "order_inv", "judge_order_invariant=True 且无 s22 → L1_PROMPT_ORDER_INVARIANT"
    else:
        static, why = "base", "judge_order_invariant=False 且无 s22 → L1_PROMPT"
    ev["static_inference"] = {"variant": static, "reason": why,
                              "caveat": "config_effective.json 每次调用 phase1 都会重写，"
                                        "反映的是最后一次调用时的配置，不一定是 s6 那次"}
    LOG.info(f"静态推断：{static}（{why}）")
    for sec in lines[-3:]:
        for ln in sec:
            LOG.info(f"  log: {ln}")

    # think1 / k3 各自的 prompt（来自 fix_judge.py 的代码路径）
    fj = Path(a.phase1).resolve().parent / "fix_judge.py"
    fj_src = fj.read_text(encoding="utf-8") if fj.exists() else ""
    ev["think1_prompt"] = {
        "variant": "base" if "tmpl = prompt or P.L1_PROMPT" in fj_src else "unknown",
        "evidence": "fix_judge.judge_once 思考分支：tmpl = prompt or P.L1_PROMPT，"
                    "J2 调用时 prompt=None" if fj_src else "找不到 fix_judge.py"}
    ev["k3_prompt"] = {"variant": "order_inv" if rj(a.config).get("judge_order_invariant", True)
                       else "base",
                       "evidence": "k3 走 P._judge1(prompt=None)，fix_judge 不设 _active_prompt "
                                   "→ 取当前 config 的 judge_order_invariant（默认 True）；"
                                   "这是推断，未做指纹"} if a.config else {"variant": "unknown"}

    # 指纹
    fp = None
    if not a.no_fingerprint:
        fp = _fingerprint(a, tbl)
    ev["fingerprint"] = fp

    if fp:
        rk = sorted(fp["by_variant"].items(), key=lambda kv: -kv[1]["facts_exact"])
        (b1, s1), (b2, s2) = rk[0], rk[1]
        margin = s1["facts_exact"] - s2["facts_exact"]
        if s1["facts_exact"] >= 0.5 and margin >= 0.25:
            status, var = "identified", b1
            note = (f"{b1} 逐字复现 facts {pct(s1['facts_exact'])}，次优 {b2} {pct(s2['facts_exact'])}")
        else:
            status, var = "ambiguous", b1
            note = (f"最优 {b1} {pct(s1['facts_exact'])} 与次优 {b2} {pct(s2['facts_exact'])} "
                    f"拉不开 —— 可能判官模型 / vLLM 版本自 main_v10 后变过")
        if static and var != static:
            note += f"；⚠ 与静态推断 {static} 不一致，以指纹为准"
    else:
        status, var, note = "static_only", static, "未做指纹（--no-fingerprint），仅静态推断"
    dec = {"status": status, "variant": var, "note": note,
           "sha256": shas.get(var) if var else None}
    same = (var == ev["think1_prompt"]["variant"]) if var else None
    out = {"main_run": str(a.main_run), "decision": dec, "evidence": ev,
           "main_vs_think1_same_prompt": same,
           "paper_consequence": (
               None if same is None else
               "§6 的「两臂只差推理预算」成立（同一 prompt）。E5b 不需要。" if same else
               f"§6 的「两臂只差推理预算」不成立：main={var}，think1=base。"
               f"两臂同时差 prompt 与推理预算。E5b 的两个格子用来拆开这两个因素；"
               f"在 E5b 出结果前，正文必须改成「两种判官配置」而不是「只差推理」。")}
    wj(a.p3 / "audit" / "prompt_audit.json", out)
    LOG.info(f"结论：{status} → {var}  ({note})")
    if out["paper_consequence"]:
        LOG.warning(out["paper_consequence"])
    return f"{status}: main 用 {var}；与 think1 同 prompt={same}"


def _fingerprint(a, tbl):
    P = P1(a)
    m = MainRun(a.main_run)
    cfg = make_cfg(a, f"{a.tag}phase3_fingerprint")
    rows = [r for r in m.eligible()
            if r.get("verdict_ij") in ("SUFFICIENT", "INSUFFICIENT") and r.get("facts_ij")]
    n_per = max(2, a.fp_n // (len(cfg.sim_bins) - 1))
    samp = strat_sample(rows, cfg.sim_bins, n_per, SEED)
    LOG.info(f"[指纹] {len(samp)} 对 × {len(tbl)} 个 prompt 变体 = {len(samp)*len(tbl)} 次（ij 方向）")
    ctx, sm = start_ctx(a, cfg, "audit")
    try:
        urls = sm.ensure("judge", list(cfg.gpus))
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)
        items = [(r, v) for r in samp for v in tbl]

        def one(it):
            r, v = it
            ci, cj = m.calls[r["qid_i"]], m.calls[r["qid_j"]]
            docs = P._fmt(cj.get("retrieved_chunks") or [], m.cmap, cfg.judge_doc_chars)
            j = judge_call(P, pool, cfg, tbl[v], ci["query"], docs, False,
                           cfg.judge_max_tokens, "qwen3")
            return {"pair_id": r["pair_id"], "variant": v, "bin": r.get("bin"),
                    "verdict": j.get("verdict"), "facts": j.get("facts"),
                    "stored_verdict": r.get("verdict_ij"), "stored_facts": r.get("facts_ij")}
        res = [x for x in P.pmap(one, items, cfg.judge_concurrency, "fingerprint", ctx, pool)
               if isinstance(x, dict) and "variant" in x]
    finally:
        sm.stop()
    wjl(a.p3 / "audit" / "fingerprint.jsonl", res)

    def norm(f):
        return json.dumps(f, ensure_ascii=False, sort_keys=True) if f is not None else None
    by = {}
    for v in tbl:
        rs = [x for x in res if x["variant"] == v]
        by[v] = {"n": len(rs),
                 "verdict_agree": float(np.mean([x["verdict"] == x["stored_verdict"] for x in rs]))
                 if rs else 0.0,
                 "facts_exact": float(np.mean([norm(x["facts"]) == norm(x["stored_facts"])
                                               for x in rs])) if rs else 0.0,
                 "sha12": psha(tbl[v])[:12]}
        LOG.info(f"  {v:<18} n={by[v]['n']:4d}  verdict 一致 {pct(by[v]['verdict_agree'])}  "
                 f"facts 逐字复现 {pct(by[v]['facts_exact'])}")
    return {"n_pairs": len(samp), "by_variant": by}


# ==============================================================================
#  E5：重判引擎（E5a / E5b 共用）
# ==============================================================================

@dataclasses.dataclass
class Arm:
    name: str
    out_run: str
    model_path: str
    served: str
    family: str
    thinking: bool
    prompt_id: str
    max_tokens: int
    weights_gb: float
    min_kv_gb: float
    ctx_len: int
    max_seqs: int
    sec_per_call: float
    gate: bool = False


def link_stages(src: Path, dst: Path, stages=("s1", "s2", "s3", "s4", "s5")):
    n = 0
    for st in stages:
        sd = src / "stages" / st
        if not sd.is_dir():
            continue
        dd = dst / "stages" / st
        dd.mkdir(parents=True, exist_ok=True)
        for sp in sorted(sd.iterdir()):
            if sp.is_file() and not (dd / sp.name).exists():
                os.symlink(sp.resolve(), dd / sp.name); n += 1
    return n


def link_files(src: Path, dst: Path, rels):
    n = 0
    for rel in rels:
        sp, dp = src / "stages" / rel, dst / "stages" / rel
        if sp.exists() and not dp.exists():
            dp.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(sp.resolve(), dp); n += 1
    return n


def recompute_qc(src: Path, dst: Path, res, arm_block):
    """与 fix_judge.recompute_qc 同一规则：只继承筛选阶段的量，判定派生量不拷。"""
    KEEP = ("n_sampled_pairs", "n_eligible", "eligibility_rate",
            "eligibility_by_bin", "self_cov_rate")
    sq = src / "stages" / "s6" / "qc.json"
    old = rj(sq) if sq.exists() else {}
    qc = {k: old[k] for k in KEEP if k in old}
    good = [r for r in res if r.get("substitutable") is not None]
    st = [r for r in good if r.get("same_text")]

    def wrate(rs):
        den = sum((r.get("w") or 0.0) for r in rs)
        return float(sum((r.get("w") or 0.0) for r in rs if r["substitutable"]) / den) if den else None
    qc["judge_aggregation"] = arm_block
    qc["recomputed_from_l1"] = {
        "n_judged": len(good),
        "substitutable_rate_conditional_unweighted":
            float(np.mean([r["substitutable"] for r in good])) if good else None,
        "substitutable_rate_conditional": wrate(good),
        "asymmetry_rate": (sum(1 for r in good if r.get("cov_ij") != r.get("cov_ji"))
                           / len(good)) if good else None,
        "same_text_n": len(st),
        "same_text_substitutable_rate":
            float(np.mean([r["substitutable"] for r in st])) if st else None}
    qc["_provenance"] = {
        "inherited_from": str(src), "inherited_fields": [k for k in KEEP if k in old],
        "MIXED_ARM_NOTE": "合格性（self_cov）沿用 main_v10 的主判官；本 run 只重判覆盖。"
                          "与 think1 / k3 同一做法，所以各臂之间可比。"}
    wj(dst / "stages" / "s6" / "qc.json", qc)


def rejudge(a, arm: Arm, limit_pairs=None):
    P = P1(a)
    m = MainRun(a.main_run)
    pid_, tmpl = audited_prompt(a) if arm.prompt_id == "@main" else (arm.prompt_id, prompt_table(a)[arm.prompt_id])
    arm.prompt_id = pid_
    adir = a.p3 / "e5" / arm.name
    adir.mkdir(parents=True, exist_ok=True)
    dst = a.runs_root / arm.out_run
    l1_out = dst / "stages" / "s6" / "l1.jsonl"
    if l1_out.exists() and not a.redo:
        LOG.info(f"{l1_out} 已存在，跳过重判（--redo 重做）")
        if not a.no_calib and not (dst / "stages" / "s8" / "calib.json").exists():
            run_phase1(a, arm.out_run, "s8", force=True)
        return _arm_summary(a, arm, dst)

    elig = m.eligible()
    if limit_pairs:
        edges = rj(parity_config(a))["sim_bins"]
        elig = strat_sample(elig, edges, max(2, limit_pairs // (len(edges) - 1)), SEED)
    LOG.info(f"[{arm.name}] 模型 {arm.served}（{arm.family}） thinking={arm.thinking} "
             f"prompt={pid_}（sha {psha(tmpl)[:12]}） max_tokens={arm.max_tokens}")
    ncall = 2 * len(elig)
    LOG.info(f"  重判 {len(elig)} 对 × 2 方向 = {ncall} 次；两卡粗估 "
             f"{ncall * arm.sec_per_call / 7200:.1f} h")

    n = link_stages(a.main_run, dst)
    LOG.info(f"  s1–s5 软链 {n} 个文件：{a.main_run} -> {dst}")
    cfg = make_cfg(a, arm.out_run, judge_path=arm.model_path, judge_name=arm.served,
                   judge_ctx=arm.ctx_len, judge_max_seqs=arm.max_seqs)
    ap = Appender(adir / "partial.jsonl")
    done = {k: v for k, v in ap.load("pair_id").items() if v.get("ok")}
    LOG.info(f"  已有 {len(done)} 对判完（断点续跑）")

    block = {"arm": arm.name, "K": 1, "thinking": arm.thinking, "order": "natural",
             "rule": "single", "model": arm.served, "model_path": arm.model_path,
             "family": arm.family, "prompt_id": pid_, "prompt_sha256": psha(tmpl),
             "max_tokens": arm.max_tokens, "temperature": cfg.judge_temperature,
             "doc_chars": cfg.judge_doc_chars}

    def one_factory(pool):
        def one(r):
            ci, cj = m.calls.get(r["qid_i"]), m.calls.get(r["qid_j"])
            if not ci or not cj:
                return None
            out = dict(r)
            out["_inherited"] = {k: r.get(k) for k in
                                 ("cov_ij", "cov_ji", "substitutable", "verdict_ij",
                                  "verdict_ji", "facts_ij", "tokens", "double_judged", "raw")}
            for tag, (q, ids) in (("ij", (ci["query"], cj.get("retrieved_chunks") or [])),
                                  ("ji", (cj["query"], ci.get("retrieved_chunks") or []))):
                docs = P._fmt(ids, m.cmap, cfg.judge_doc_chars)
                j = judge_call(P, pool, cfg, tmpl, q, docs, arm.thinking, arm.max_tokens,
                               arm.family)
                out[f"cov_{tag}"] = j.get("covered") if j["ok"] else None
                out[f"verdict_{tag}"] = j["verdict"]
                out[f"facts_{tag}"] = j.get("facts")
                out[f"trunc_{tag}"] = j["truncated"]
                out[f"tok_out_{tag}"] = j["tok_out"]
                out[f"tok_in_{tag}"] = j["tok_in"]
                if not j["ok"]:
                    out[f"raw_{tag}"] = j.get("raw")
            for k in ("tokens", "double_judged", "raw", "cov_ij_shuf", "pos_consistent",
                      "substitutable_strict", "substitutable_union", "truncated"):
                out[k] = None
            out["truncated"] = bool(out.get("trunc_ij") or out.get("trunc_ji"))
            x, y = out.get("cov_ij"), out.get("cov_ji")
            out["ok"] = x is not None and y is not None
            out["substitutable"] = int(bool(x) and bool(y)) if out["ok"] else None
            out["judge_aggregation"] = block
            ap.add(out)
            return {"pair_id": r["pair_id"], "ok": out["ok"]}
        return one

    t0 = time.time()
    gate_done = (adir / "gate.json").exists() and rj(adir / "gate.json").get("passed")
    if len(done) >= len(elig) and (not arm.gate or gate_done):
        LOG.info("  全部已判完，不起服务，直接汇总")
        llm = None
    else:
        llm = _rejudge_serve(a, arm, cfg, m, elig, done, ap, adir, one_factory, gate_done)
        if isinstance(llm, str):          # --gate-only
            return llm
    wall = time.time() - t0
    return _rejudge_finish(a, arm, m, elig, ap, adir, dst, l1_out, block, llm, wall, limit_pairs)


def _rejudge_serve(a, arm, cfg, m, elig, done, ap, adir, one_factory, gate_done):
    P = P1(a)
    ctx, sm = start_ctx(a, cfg, f"e5_{arm.name}",
                        role_req={"weights": arm.weights_gb, "min_kv": arm.min_kv_gb})
    try:
        urls = sm.ensure("judge", list(cfg.gpus))
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=900)
        one = one_factory(pool)

        # ---- 闸门：金标对 + 决策带样本，先判，不过关就停 ----
        if arm.gate and not gate_done:
            g = _gate(a, arm, m, elig, done, one, ctx, pool, cfg)
            done = {k: v for k, v in ap.load("pair_id").items() if v.get("ok")}
            if a.gate_only:
                return f"闸门 {'通过' if g['passed'] else '未过'}（--gate-only，停在这里）"
            if not g["passed"] and not a.ignore_gate:
                raise RuntimeError(f"[{arm.name}] 闸门未过：{g['fail_reasons']}。"
                                   f"详见 {adir/'gate.json'}；先跑 `exp45.py e5-gate-diag` 看原因，"
                                   f"确认要继续加 --ignore-gate")

        todo = [r for r in elig if r["pair_id"] not in done]
        LOG.info(f"  本次要判 {len(todo)} 对")
        if todo:
            P.pmap(one, todo, cfg.judge_concurrency, arm.name, ctx, pool)
        # 失败的再补一轮
        got = ap.load("pair_id")
        retry = [r for r in elig if not got.get(r["pair_id"], {}).get("ok")]
        if retry:
            LOG.info(f"  {len(retry)} 对失败/缺失，重试一轮")
            P.pmap(one, retry, cfg.judge_concurrency, arm.name + "-retry", ctx, pool)
        return pool.stats()
    finally:
        sm.stop()


def _rejudge_finish(a, arm, m, elig, ap, adir, dst, l1_out, block, llm, wall, limit_pairs):
    got = ap.load("pair_id")
    elig_ids = {r["pair_id"] for r in elig}
    res = [got[k] for k in elig_ids if k in got and got[k].get("ok")]
    keep = {r["pair_id"] for r in res}
    rest, n_failed = [], 0
    for r0 in m.l1:
        if r0["pair_id"] in keep:
            continue
        r0 = dict(r0)
        if r0.get("eligible"):
            n_failed += 1
            r0["_inherited"] = {k: r0.get(k) for k in
                                ("cov_ij", "cov_ji", "substitutable", "verdict_ij", "verdict_ji")}
            for k in ("cov_ij", "cov_ji", "substitutable", "verdict_ij", "verdict_ji"):
                r0[k] = None
            r0["ok"] = False
            r0["rejudge_failed"] = True
        rest.append(r0)
    if n_failed:
        LOG.warning(f"  {n_failed} 个合格对没有本臂的有效判定（含 --limit 之外的），"
                    f"标签置空、旧值进 _inherited，不混臂")
    wjl(l1_out, res + rest)
    (dst / "stages" / "s7").mkdir(parents=True, exist_ok=True)
    wj(dst / "stages" / "s7" / "summary.json",
       {"skipped": True, "reason": f"{arm.name}：L1 标签已重判，旧 L2 锚定不适用"})
    if not (dst / "stages" / "s7" / "l2.jsonl").exists():
        (dst / "stages" / "s7" / "l2.jsonl").write_text("", encoding="utf-8")
    recompute_qc(a.main_run, dst, res, block)

    dirs = [(r.get(f"verdict_{t}"), r.get(f"trunc_{t}"), r.get(f"tok_out_{t}"))
            for r in got.values() if r["pair_id"] in elig_ids for t in ("ij", "ji")]
    info = {"arm": dataclasses.asdict(arm), "aggregation": block, "out_run": str(dst),
            "n_eligible_main": len(m.eligible()), "n_target": len(elig),
            "n_ok": len(res), "n_failed": len(elig) - len(res),
            "parse_fail_rate": float(np.mean([v == "PARSE_FAIL" for v, _, _ in dirs])) if dirs else None,
            "trunc_rate": float(np.mean([bool(t) for _, t, _ in dirs])) if dirs else None,
            "mean_tok_out": float(np.mean([o for _, _, o in dirs if o is not None])) if dirs else None,
            "p99_tok_out": float(np.percentile([o for _, _, o in dirs if o is not None], 99)) if dirs else None,
            "verdict_dist": dict(Counter(v for v, _, _ in dirs).most_common(8)),
            "substitutable_rate": float(np.mean([r["substitutable"] for r in res])) if res else None,
            "wall_h": wall / 3600, "llm": llm, "limited": bool(limit_pairs)}
    wj(adir / "rejudge.json", info)
    LOG.info(f"  有效 {len(res)}/{len(elig)}，解析失败 {pct(info['parse_fail_rate'],2)}，"
             f"截断 {pct(info['trunc_rate'],2)}，可替代率 {fmt(info['substitutable_rate'])}")

    if not a.no_calib:
        run_phase1(a, arm.out_run, "s8", force=True)
    return _arm_summary(a, arm, dst)


def _arm_summary(a, arm, dst):
    c = dst / "stages" / "s8" / "calib.json"
    if not c.exists():
        return f"{arm.name}: l1 已写出，未校准"
    tm = rj(c).get("tau_multi") or {}
    s = ", ".join(f"{k}:τ*={fmt(v.get('tau_star'),4)}"
                  f"/同文本{pct(v.get('same_text_frac'))}" for k, v in sorted(tm.items()))
    return f"{arm.name}: {s}"


def _gate(a, arm, m, elig, done, one, ctx, pool, cfg):
    P = P1(a)
    gold = load_gold(a)
    eid = {r["pair_id"]: r for r in elig}
    gs = [eid[k] for k in gold if k in eid]
    band = [r for r in elig if 0.85 <= r["cos"] < 0.999 and r["pair_id"] not in gold]
    bs = strat_sample(band, [0.85, 0.90, 0.95, 0.98, 0.999], a.gate_n // 4, SEED + 1)
    samp = {r["pair_id"]: r for r in gs + bs}
    todo = [r for k, r in samp.items() if k not in done]
    LOG.info(f"[闸门] 金标对 {len(gs)} + 决策带 {len(bs)} = {len(samp)} 对（待判 {len(todo)}）")
    if todo:
        P.pmap(one, todo, cfg.judge_concurrency, f"{arm.name}-gate", ctx, pool)
    got = Appender(a.p3 / "e5" / arm.name / "partial.jsonl").load("pair_id")
    rs = [got[k] for k in samp if k in got]
    dirs = [(r.get(f"verdict_{t}"), r.get(f"trunc_{t}")) for r in rs for t in ("ij", "ji")]
    pf = float(np.mean([v == "PARSE_FAIL" for v, _ in dirs])) if dirs else 1.0
    tr = float(np.mean([bool(t) for _, t in dirs])) if dirs else 1.0
    ga = acc_vs([gold[r["pair_id"]] for r in rs if r["pair_id"] in gold],
                [r.get("substitutable") for r in rs if r["pair_id"] in gold])
    gm = acc_vs([gold[k] for k in gold if k in eid], [eid[k].get("substitutable") for k in gold if k in eid])
    ag = agree([eid[r["pair_id"]]["substitutable"] for r in rs],
               [r.get("substitutable") for r in rs])
    fails = []
    if pf > a.gate_max_parse_fail:
        fails.append(f"解析失败 {pct(pf,2)} > {pct(a.gate_max_parse_fail,1)}")
    if tr > a.gate_max_trunc:
        fails.append(f"截断 {pct(tr,2)} > {pct(a.gate_max_trunc,1)}")
    if ga.get("n", 0) >= 30 and ga["acc"] < a.gate_min_acc:
        fails.append(f"金标准确率 {ga['acc']:.3f} < {a.gate_min_acc}")
    g = {"passed": not fails, "fail_reasons": fails, "n": len(rs),
         "parse_fail_rate": pf, "trunc_rate": tr,
         "gold_acc_this_arm": ga, "gold_acc_main_same_pairs": gm,
         "agreement_with_main_on_gate_sample": ag,
         "verdict_dist": dict(Counter(v for v, _ in dirs).most_common(8)),
         "thresholds": {"max_parse_fail": a.gate_max_parse_fail,
                        "max_trunc": a.gate_max_trunc, "min_gold_acc": a.gate_min_acc},
         "note": "金标对来自 main_v10 s5（主判官 prompt 不是靠它选的，前提是 main_v10 没跑 s22）"}
    wj(a.p3 / "e5" / arm.name / "gate.json", g)
    with contextlib.suppress(Exception):
        gate_diag(a, arm.name)
    LOG.info(f"[闸门] 解析失败 {pct(pf,2)} 截断 {pct(tr,2)} | 金标 acc 本臂 "
             f"{fmt(ga.get('acc'))} (n={ga.get('n')}) vs 主判官 {fmt(gm.get('acc'))} | "
             f"与主判官一致 {pct(ag.get('agree'))} κ={fmt(ag.get('kappa'))} → "
             f"{'通过' if not fails else '未过：' + '; '.join(fails)}")
    return g


def arm_xfam(a, check=True, lang=None):
    lang = lang or a.xfam_prompt
    spec = XFAM_JUDGES[a.xfam]
    cfg = make_cfg(a, "phase3_tmp")
    path = str(Path(cfg.models_dir) / spec["local"])
    if check and not Path(path, "config.json").exists():
        raise RuntimeError(f"{path} 不存在，先跑 `exp45.py download`")
    sfx = "_en" if lang == "en" else ""
    return Arm(name="xfam_" + a.xfam.split("-")[0] + sfx,
               out_run=f"{a.tag}e5a_xfam_{a.xfam.replace('-', '_')}{sfx}",
               model_path=path, served=spec["served"], family=spec["family"],
               thinking=False, prompt_id="base_en" if lang == "en" else "@main",
               max_tokens=cfg.judge_max_tokens, weights_gb=spec["weights_gb"],
               min_kv_gb=spec["min_kv_gb"], ctx_len=spec["ctx"], max_seqs=spec["max_seqs"],
               sec_per_call=spec["sec_per_call"], gate=True)


def arms_e5b(a):
    cfg = make_cfg(a, "phase3_tmp")
    base = dict(model_path=cfg.judge_path, served=cfg.judge_name, family="qwen3",
                weights_gb=18.0, min_kv_gb=6.0, ctx_len=cfg.judge_ctx,
                max_seqs=cfg.judge_max_seqs)
    return {
        # 与 think1 完全相同的设置（max_tokens = max(600*4, 2048) = 2400），只换 prompt
        "think_mainprompt": Arm(name="think_mainprompt", out_run=f"{a.tag}e5b_think_mainprompt",
                                thinking=True, prompt_id="@main",
                                max_tokens=max(cfg.judge_max_tokens * 4, 2048),
                                sec_per_call=0.55, **base),
        # 与 main 完全相同的设置，只换成 base prompt
        "nothink_base": Arm(name="nothink_base", out_run=f"{a.tag}e5b_nothink_base",
                            thinking=False, prompt_id="base",
                            max_tokens=cfg.judge_max_tokens, sec_per_call=0.37, **base),
    }


def arm_qwen_en(a):
    cfg = make_cfg(a, "phase3_tmp")
    return Arm(name="qwen_base_en", out_run=f"{a.tag}e5x_qwen_base_en", model_path=cfg.judge_path,
               served=cfg.judge_name, family="qwen3", thinking=False, prompt_id="base_en",
               max_tokens=cfg.judge_max_tokens, weights_gb=18.0, min_kv_gb=6.0,
               ctx_len=cfg.judge_ctx, max_seqs=cfg.judge_max_seqs, sec_per_call=0.37, gate=True)


def cmd_e5_langcheck(a):
    """语言诊断（各 ~15 min，只判闸门样本）：
       Llama + 英文直译 prompt；Qwen（主判官）+ 英文直译 prompt。
       若 Qwen 换英文后金标准确率基本不变、Llama 换英文后明显回升，
       掉分就是中文指令造成的，全量应改用英文版（两臂同语言）。"""
    pid, _ = audited_prompt(a)
    if pid != "base":
        raise RuntimeError(f"英文直译只对应 base；main 用的是 {pid}")
    a.gate_only = True
    out = []
    for arm in (arm_xfam(a, lang="en"), arm_qwen_en(a)):
        r = rejudge(a, arm, limit_pairs=None)
        out.append(f"{arm.name}: {r}")
        gp = a.p3 / "e5" / arm.name / "gate.json"
        if gp.exists():
            g = rj(gp)
            out[-1] += f" 金标 acc {fmt(g['gold_acc_this_arm'].get('acc'))}"
    return " | ".join(out)


def cmd_e5_gate_diag(a):
    names = [d.name for d in (a.p3 / "e5").iterdir() if (d / "gate.json").exists()] \
        if (a.p3 / "e5").exists() else []
    if not names:
        raise RuntimeError("还没有任何闸门结果")
    return " | ".join(gate_diag(a, n) for n in sorted(names))


def gate_diag(a, arm_name):
    """纯 CPU。回答：Llama 在金标上掉分，是能力问题还是口径问题？
    判官输出三档 SUFFICIENT/PARTIAL/INSUFFICIENT，只有 SUFFICIENT 算覆盖。
    若某模型偏爱 PARTIAL，严格映射下会被算成大量假阴性，而把 PARTIAL 也算覆盖后
    准确率回到主判官水平 —— 那就是阈值位置不同，不是不会判。"""
    m = MainRun(a.main_run)
    gold = load_gold(a)
    mainl = {r["pair_id"]: r for r in m.eligible()}
    got = Appender(a.p3 / "e5" / arm_name / "partial.jsonl").load("pair_id")
    g = rj(a.p3 / "e5" / arm_name / "gate.json")
    edges = [0.85, 0.90, 0.95, 0.98, 0.999, 1.0001]

    def cov(v, lenient):
        if v in (None, "PARSE_FAIL"):
            return None
        return int(v == "SUFFICIENT" or (lenient and v == "PARTIAL"))

    def sub(r, lenient):
        x, y = cov(r.get("verdict_ij"), lenient), cov(r.get("verdict_ji"), lenient)
        return None if x is None or y is None else int(x and y)
    ids = [k for k in got if k in mainl]
    gid = [k for k in ids if k in gold]
    out = {"arm": arm_name, "n_judged": len(ids), "n_gold": len(gid)}
    for tag, R in (("arm", got), ("main", mainl)):
        vd = Counter(R[k].get(f"verdict_{d}") for k in ids for d in ("ij", "ji"))
        tot = sum(vd.values()) or 1
        out[f"verdict_dist_{tag}"] = {k: round(v / tot, 4) for k, v in vd.most_common()}
        for lenient in (False, True):
            lab = "lenient" if lenient else "strict"
            out[f"gold_{tag}_{lab}"] = acc_vs([gold[k] for k in gid], [sub(R[k], lenient) for k in gid])
    # 与主判官的分歧方向（严格映射，单向 cov_ij）
    a1 = [(cov(mainl[k].get("verdict_ij"), False), cov(got[k].get("verdict_ij"), False)) for k in ids]
    a1 = [(x, y) for x, y in a1 if x is not None and y is not None]
    out["cov_ij_vs_main"] = {"n": len(a1), "main1_arm0": sum(1 for x, y in a1 if x == 1 and y == 0),
                             "main0_arm1": sum(1 for x, y in a1 if x == 0 and y == 1),
                             "agree": sum(1 for x, y in a1 if x == y) / max(len(a1), 1)}
    # 主判官说 SUFFICIENT、本臂不认的那些，本臂具体给了什么
    out["arm_verdict_when_main_sufficient_arm_not"] = dict(Counter(
        got[k].get(f"verdict_{d}") for k in ids for d in ("ij", "ji")
        if mainl[k].get(f"verdict_{d}") == "SUFFICIENT" and got[k].get(f"verdict_{d}") != "SUFFICIENT"))
    # 分带
    out["by_band"] = {}
    for b in range(len(edges) - 1):
        kb = [k for k in ids if edges[b] <= mainl[k]["cos"] < edges[b + 1]]
        if kb:
            out["by_band"][f"[{edges[b]},{edges[b+1]})"] = {
                "n": len(kb),
                "pos_main": float(np.mean([mainl[k]["substitutable"] for k in kb])),
                "pos_arm_strict": float(np.mean([v for v in (sub(got[k], False) for k in kb) if v is not None] or [np.nan])),
                "pos_arm_lenient": float(np.mean([v for v in (sub(got[k], True) for k in kb) if v is not None] or [np.nan]))}
    out["gate"] = {k: g.get(k) for k in ("passed", "fail_reasons")}
    wj(a.p3 / "e5" / arm_name / "gate_diag.json", out)
    ga, gm = out["gold_arm_strict"], out["gold_main_strict"]
    gal, gml = out["gold_arm_lenient"], out["gold_main_lenient"]
    LOG.info(f"[诊断 {arm_name}] 金标 n={len(gid)}")
    LOG.info(f"  严格映射（只有 SUFFICIENT=1）：本臂 acc {fmt(ga.get('acc'))} P={fmt(ga.get('precision'))} "
             f"R={fmt(ga.get('recall'))} {ga.get('confusion')} | 主判官 acc {fmt(gm.get('acc'))} "
             f"P={fmt(gm.get('precision'))} R={fmt(gm.get('recall'))} {gm.get('confusion')}")
    LOG.info(f"  宽松映射（PARTIAL 也算 1）：本臂 acc {fmt(gal.get('acc'))} P={fmt(gal.get('precision'))} "
             f"R={fmt(gal.get('recall'))} | 主判官 acc {fmt(gml.get('acc'))}")
    LOG.info(f"  判定分布 本臂 {out['verdict_dist_arm']} | 主判官 {out['verdict_dist_main']}")
    c = out["cov_ij_vs_main"]
    LOG.info(f"  cov_ij 与主判官：一致 {pct(c['agree'])}，主1本0 = {c['main1_arm0']}，主0本1 = {c['main0_arm1']}；"
             f"主判 SUFFICIENT 而本臂不认时，本臂给的是 {out['arm_verdict_when_main_sufficient_arm_not']}")
    for bn, v in out["by_band"].items():
        LOG.info(f"  {bn:<14} n={v['n']:3d} 正例率 主 {fmt(v['pos_main'])} | 本臂 严格 "
                 f"{fmt(v['pos_arm_strict'])} 宽松 {fmt(v['pos_arm_lenient'])}")
    return (f"{arm_name}: 金标 严格 {fmt(ga.get('acc'))} / 宽松 {fmt(gal.get('acc'))}"
            f"（主 {fmt(gm.get('acc'))}）")


def cmd_e5_lenient(a):
    """纯 CPU 的敏感性分析：把 PARTIAL 也算覆盖，对 main 和跨模型族臂分别重算标签并重跑 s8。
    回答「Llama 与 Qwen 的分歧是不是只是 PARTIAL 划在边界哪一侧」—— 若宽松口径下两者
    τ* 收拢，分歧主要是口径；若仍分开，分歧是实质性的。合格性（self-cov）仍沿用 main。"""
    xf = arm_xfam(a, False)
    srcs = {"main": a.main_run, xf.name: a.runs_root / xf.out_run}
    out = {}
    for name, src in srcs.items():
        if not (src / "stages" / "s6" / "l1.jsonl").exists():
            LOG.warning(f"{src} 没有 l1，跳过"); continue
        dst = a.runs_root / f"{a.tag}e5l_{name}_lenient"
        link_stages(a.main_run, dst)
        rows, res = [], []
        for r in rjl(src / "stages" / "s6" / "l1.jsonl"):
            r = dict(r)
            if r.get("eligible") and (name == "main" or r.get("judge_aggregation")):
                c = [None if v in (None, "PARSE_FAIL") else int(v in ("SUFFICIENT", "PARTIAL"))
                     for v in (r.get("verdict_ij"), r.get("verdict_ji"))]
                r["_strict"] = {k: r.get(k) for k in ("cov_ij", "cov_ji", "substitutable")}
                r["cov_ij"], r["cov_ji"] = c
                r["ok"] = None not in c
                r["substitutable"] = int(c[0] and c[1]) if r["ok"] else None
                r["judge_aggregation"] = {**(r.get("judge_aggregation") or {"arm": "main"}),
                                          "mapping": "lenient(SUFFICIENT|PARTIAL)"}
                if r["ok"]:
                    res.append(r)
            elif r.get("eligible"):
                r["ok"], r["substitutable"] = False, None
            rows.append(r)
        wjl(dst / "stages" / "s6" / "l1.jsonl", rows)
        (dst / "stages" / "s7").mkdir(parents=True, exist_ok=True)
        wj(dst / "stages" / "s7" / "summary.json", {"skipped": True, "reason": "宽松口径重算"})
        if not (dst / "stages" / "s7" / "l2.jsonl").exists():
            (dst / "stages" / "s7" / "l2.jsonl").write_text("", encoding="utf-8")
        recompute_qc(a.main_run, dst, res, {"arm": name, "mapping": "lenient"})
        LOG.info(f"  {name}: 宽松口径 {len(res)} 对，可替代率 {np.mean([r['substitutable'] for r in res]):.3f}")
        if not a.no_calib:
            run_phase1(a, dst.name, "s8", force=True)
        c = dst / "stages" / "s8" / "calib.json"
        out[name] = {"n": len(res), "sub_rate": float(np.mean([r["substitutable"] for r in res])),
                     "tau_multi": (rj(c).get("tau_multi") if c.exists() else None),
                     "tau_multi_strict": (rj(src / "stages" / "s8" / "calib.json").get("tau_multi")
                                          if (src / "stages" / "s8" / "calib.json").exists() else None)}
    wj(a.p3 / "e5" / "lenient.json", out)
    for name, v in out.items():
        for t in ("0.80", "0.85", "0.90"):
            st = ((v.get("tau_multi_strict") or {}).get(t) or {})
            le = ((v.get("tau_multi") or {}).get(t) or {})
            LOG.info(f"  {name:<14} {t}: 严格 τ*={fmt(st.get('tau_star'),4)} 同文本 {pct(st.get('same_text_frac'))}"
                     f" | 宽松 τ*={fmt(le.get('tau_star'),4)} 同文本 {pct(le.get('same_text_frac'))}")
    return "; ".join(f"{k}: 宽松可替代率 {v['sub_rate']:.3f}" for k, v in out.items())


def cmd_e5_selfcov(a):
    """把「合格性分歧」和「覆盖判定分歧」拆开（约 4–5 h GPU，Llama 两卡）。

    E5a 只重判了覆盖，合格性（两端查询各自的检索结果能否回答自己）沿用 Qwen。
    对逐字相同、检索结果也相同的一对，cov_ij 问的恰好就是 Llama 眼里 q_i 能否被自己
    的结果回答 —— 于是 Llama 每在自覆盖上和 Qwen 意见不同一次，就在曲线顶端多一个
    「相同查询却不可替代」的负例，τ* 被往上推、高档无解。这部分是混臂造成的，不是
    Llama 对「可替代性」的判断。

    做法：用 Llama（同 prompt、自然序、不思考）判这批合格对涉及的每个查询的自覆盖，
    取两个判官都认为合格的对（双合格子集），在这个共同子集上分别给 main 与 Llama
    重新标 τ*。两臂的样本集合完全相同，差别只剩覆盖判定本身。"""
    P = P1(a)
    m = MainRun(a.main_run)
    xf = arm_xfam(a)
    pid, tmpl = audited_prompt(a)
    elig = m.eligible()
    qids = sorted({q for r in elig for q in (r["qid_i"], r["qid_j"])})
    if a.limit:
        qids = qids[:a.limit]
    adir = a.p3 / "e5" / xf.name
    ap = Appender(adir / "selfcov.jsonl")
    done = {k for k, v in ap.load("qid").items() if v.get("cov") is not None}
    todo = [q for q in qids if q not in done]
    LOG.info(f"[双合格] 合格对 {len(elig)} 涉及 {len(qids)} 个查询；已判 {len(done)}，待判 {len(todo)}"
             f"（两卡粗估 {len(todo) * xf.sec_per_call / 7200:.1f} h）")
    cfg = make_cfg(a, xf.out_run, judge_path=xf.model_path, judge_name=xf.served,
                   judge_ctx=xf.ctx_len, judge_max_seqs=xf.max_seqs)
    if todo:
        ctx, sm = start_ctx(a, cfg, "e5-selfcov",
                            role_req={"weights": xf.weights_gb, "min_kv": xf.min_kv_gb})
        try:
            urls = sm.ensure("judge", list(cfg.gpus))
            pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=900)

            def one(q):
                c = m.calls[q]
                j = judge_call(P, pool, cfg, tmpl, c["query"],
                               P._fmt(c.get("retrieved_chunks") or [], m.cmap, cfg.judge_doc_chars),
                               False, xf.max_tokens, xf.family)
                ap.add({"qid": q, "cov": j.get("covered") if j["ok"] else None,
                        "verdict": j["verdict"]})
            P.pmap(one, todo, cfg.judge_concurrency, "selfcov", ctx, pool)
        finally:
            sm.stop()
    sc = {k: v.get("cov") for k, v in ap.load("qid").items()}
    n_known = sum(1 for q in qids if sc.get(q) is not None)
    rate = float(np.mean([sc[q] for q in qids if sc.get(q) is not None])) if n_known else None
    both = {r["pair_id"] for r in elig if sc.get(r["qid_i"]) == 1 and sc.get(r["qid_j"]) == 1}
    LOG.info(f"  Llama 认为能被自己结果回答的查询：{pct(rate)}（Qwen 在这批上是 100%，合格性就是它定的）")
    LOG.info(f"  双合格对 {len(both)}/{len(elig)} = {pct(len(both)/max(len(elig),1))}")

    out = {"n_queries": len(qids), "n_selfcov_known": n_known, "llama_selfcov_rate": rate,
           "n_eligible_main": len(elig), "n_both_eligible": len(both), "arms": {}}
    lab = {}
    for name, src in (("main", a.main_run), (xf.name, a.runs_root / xf.out_run)):
        dst = a.runs_root / f"{a.tag}e5d_{name}_botheligible"
        link_stages(a.main_run, dst)
        rows, res = [], []
        for r in rjl(src / "stages" / "s6" / "l1.jsonl"):
            r = dict(r)
            if r.get("eligible") == 1 and r["pair_id"] not in both:
                r["eligible"] = 0
                r["excluded_by"] = "xfam_selfcov"
            if r.get("eligible") == 1 and r.get("ok") and r.get("substitutable") is not None:
                res.append(r)
            rows.append(r)
        wjl(dst / "stages" / "s6" / "l1.jsonl", rows)
        (dst / "stages" / "s7").mkdir(parents=True, exist_ok=True)
        wj(dst / "stages" / "s7" / "summary.json", {"skipped": True, "reason": "双合格子集"})
        if not (dst / "stages" / "s7" / "l2.jsonl").exists():
            (dst / "stages" / "s7" / "l2.jsonl").write_text("", encoding="utf-8")
        recompute_qc(a.main_run, dst, res, {"arm": name, "subset": "both_eligible"})
        if not a.no_calib:
            run_phase1(a, dst.name, "s8", force=True)
        c = dst / "stages" / "s8" / "calib.json"
        lab[name] = {r["pair_id"]: r for r in res}
        fi = [r for r in res if is_forced_identity(r)]
        out["arms"][name] = {
            "n": len(res), "sub_rate": float(np.mean([r["substitutable"] for r in res])) if res else None,
            "forced_identity_pos": float(np.mean([r["substitutable"] for r in fi])) if fi else None,
            "n_forced_identity": len(fi),
            "tau_multi": rj(c).get("tau_multi") if c.exists() else None}
    if len(lab) == 2:
        ks = sorted(set(lab["main"]) & set(lab[xf.name]))
        out["agreement_substitutable"] = agree([lab["main"][k]["substitutable"] for k in ks],
                                               [lab[xf.name][k]["substitutable"] for k in ks])
        kc = [k for k in ks if lab["main"][k].get("cov_ij") is not None
              and lab[xf.name][k].get("cov_ij") is not None]
        out["agreement_cov_ij"] = agree([lab["main"][k]["cov_ij"] for k in kc],
                                        [lab[xf.name][k]["cov_ij"] for k in kc])
    wj(a.p3 / "e5" / "both_eligible.json", out)
    for name, v in out["arms"].items():
        LOG.info(f"  {name:<14} n={v['n']} 可替代率 {fmt(v['sub_rate'])} 恒等带正例率 "
                 f"{pct(v['forced_identity_pos'])}（n={v['n_forced_identity']}）")
        for t in ("0.80", "0.85", "0.90"):
            e = ((v.get("tau_multi") or {}).get(t) or {})
            LOG.info(f"      {t}: τ*={fmt(e.get('tau_star'),4)} 放行 {pct(e.get('pass_frac'))} "
                     f"同文本 {pct(e.get('same_text_frac'))}")
    return f"双合格 {len(both)} 对；Llama 自覆盖率 {pct(rate)}"


def cmd_e5a(a):
    return rejudge(a, arm_xfam(a), limit_pairs=a.limit)


def cmd_e5b(a):
    pid, _ = audited_prompt(a)
    if pid == "base" and not a.force_e5b:
        msg = "main 的 prompt 就是 base，与 think1 相同 —— 无 prompt 混杂，E5b 不需要（--force-e5b 强制跑）"
        LOG.info(msg)
        return msg
    want = [x.strip() for x in a.e5b_cells.split(",") if x.strip()]
    arms = arms_e5b(a)
    out = []
    for k in want:
        out.append(rejudge(a, arms[k], limit_pairs=a.limit))
    return " | ".join(out)


# ==============================================================================
#  E5 报告
# ==============================================================================

def arm_runs(a):
    cand = {"main": a.main_run,
            "think1": a.runs_root / a.think_run,
            "k3": a.runs_root / a.k3_run,
            arm_xfam(a, False).name: a.runs_root / arm_xfam(a, False).out_run}
    for k, arm in arms_e5b(a).items():
        cand[k] = a.runs_root / arm.out_run
    return {k: v for k, v in cand.items() if (v / "stages" / "s6" / "l1.jsonl").exists()}


def load_labels(run: Path, is_main: bool):
    d = {}
    for r in rjl(run / "stages" / "s6" / "l1.jsonl"):
        if not r.get("eligible"):
            continue
        if not is_main and not r.get("judge_aggregation"):
            continue                       # 与 crosstab_l1 一致：未重判的不算
        if r.get("substitutable") is None:
            continue
        d[r["pair_id"]] = r
    return d


def cmd_e5_report(a):
    runs = arm_runs(a)
    LOG.info(f"可用的臂：{ {k: str(v) for k, v in runs.items()} }")
    L = {k: load_labels(v, k == "main") for k, v in runs.items()}
    main = L["main"]
    edges = rj(parity_config(a))["sim_bins"]
    names = list(L)
    out = {"runs": {k: str(v) for k, v in runs.items()}, "n_labels": {k: len(v) for k, v in L.items()}}

    # 1) 两两一致性（全体合格对）
    mat = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ks = sorted(set(L[x]) & set(L[y]))
            mat[f"{x}~{y}"] = agree([L[x][k]["substitutable"] for k in ks],
                                    [L[y][k]["substitutable"] for k in ks])
    out["pairwise_all"] = mat
    # 单向 cov_ij（论文 \CrossCfgAgree / \CrossCfgKappa / \CrossCfgAC / \CrossCfgRate 的口径）
    matc = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ks = [k for k in sorted(set(L[x]) & set(L[y]))
                  if L[x][k].get("cov_ij") is not None and L[y][k].get("cov_ij") is not None]
            matc[f"{x}~{y}"] = agree([L[x][k]["cov_ij"] for k in ks], [L[y][k]["cov_ij"] for k in ks])
    out["pairwise_all_cov_ij"] = matc

    # 2) 分档：每个臂对 main 的一致性 + 各自正例率
    slices = {}
    for b in range(len(edges) - 1):
        slices[f"bin{b} [{edges[b]:.3f},{edges[b+1]:.3f})"] = \
            lambda r, b=b: bin_of(r["cos"], edges) == b and not is_forced_identity(r)
    slices["forced identity (same_text & J,RBO≥0.999)"] = is_forced_identity
    slices["all"] = lambda r: True
    per = {}
    for sname, f in slices.items():
        row = {}
        for x in names:
            ks = [k for k in L[x] if f(L[x][k])]
            row[x] = {"n": len(ks), "pos_rate": float(np.mean([L[x][k]["substitutable"] for k in ks]))
                      if ks else None}
            if x != "main":
                kk = [k for k in ks if k in main]
                row[x]["vs_main"] = agree([main[k]["substitutable"] for k in kk],
                                          [L[x][k]["substitutable"] for k in kk])
        per[sname] = row
    out["by_slice"] = per

    # 3) 金标准确率（同一批金标对，各臂都有判定的交集）
    gold = load_gold(a)
    common = [k for k in gold if all(k in L[x] for x in names)]
    out["gold"] = {"n_labeled": len(gold), "n_common": len(common),
                   "by_arm": {x: acc_vs([gold[k] for k in common],
                                        [L[x][k]["substitutable"] for k in common]) for x in names}}

    # 4) τ*（s8/calib.json 的多档）
    taus = {}
    for x, run in runs.items():
        c = run / "stages" / "s8" / "calib.json"
        if c.exists():
            taus[x] = {t: {k: v.get(k) for k in ("tau_star", "tau_point", "n_pass", "pass_frac",
                                                 "same_text_frac", "empirical_sub", "degenerate")}
                       for t, v in (rj(c).get("tau_multi") or {}).items()}
    out["tau_multi"] = taus

    # 5) 自检：main~think1 应复现论文里的 90.7% / κ 0.642 / AC1 0.875
    chk = matc.get("main~think1")
    if chk and chk.get("n"):
        out["sanity_main_think1"] = {"field": "cov_ij", "agree": chk["agree"], "kappa": chk["kappa"],
                                     "ac1": chk["ac1"], "rate_a": chk["rate_a"], "rate_b": chk["rate_b"],
                                     "n": chk["n"],
                                     "paper": {"agree": 0.907, "kappa": 0.642, "ac1": 0.875, "rate": 0.847}}
        if abs(chk["agree"] - 0.907) > 0.002 or abs((chk["kappa"] or 0) - 0.642) > 0.002:
            LOG.warning(f"⚠ main~think1（cov_ij）复现值 agree={chk['agree']:.4f} κ={chk['kappa']:.4f} "
                        f"与论文宏 0.907/0.642 不符 —— 先查清再用本报告的任何数")
        else:
            LOG.info(f"自检通过：main~think1 cov_ij agree={chk['agree']:.4f} κ={chk['kappa']:.4f} "
                     f"AC1={chk['ac1']:.4f} 覆盖率 {chk['rate_a']:.3f}/{chk['rate_b']:.3f}"
                     f"（论文 0.907/0.642/0.875/0.847）")

    # 6) 2×2 拆解（有 E5b 时）
    def pstat(x, y):
        ks = sorted(set(L[x]) & set(L[y]))
        r = agree([L[x][k]["substitutable"] for k in ks], [L[y][k]["substitutable"] for k in ks])
        r.update(a=x, b=y)
        return r
    if "main" in L and "think1" in L and ({"think_mainprompt", "nothink_base"} & set(L)):
        out["decomposition"] = {
            what: pstat(x, y) for x, y, what in (
                ("main", "think_mainprompt", "只差推理（prompt=main）"),
                ("nothink_base", "think1", "只差推理（prompt=base）"),
                ("main", "nothink_base", "只差 prompt（不思考）"),
                ("think_mainprompt", "think1", "只差 prompt（思考）"),
                ("main", "think1", "两者都差（论文 §6 现状）"))
            if x in L and y in L}

    # 7) 机械判读规则（事先写定，不是结论）
    xf = arm_xfam(a, False).name
    rules = {}
    if xf in taus and "main" in taus:
        for t in ("0.80", "0.85", "0.90"):
            e = taus[xf].get(t) or {}
            rules[f"xfam τ*@{t} 非退化"] = (e.get("tau_star") is not None
                                          and e.get("same_text_frac") is not None
                                          and e["same_text_frac"] < 0.80)
    if matc.get(f"main~{xf}", {}).get("n") and chk:
        rules["xfam 与 main 的 cov_ij κ 低于 main~think1"] = (
            (matc[f"main~{xf}"].get("kappa") or 0) < (chk.get("kappa") or 1))
    out["mechanical_rules"] = rules

    wj(a.p3 / "e5" / "compare.json", out)
    _print_e5(out, names)
    return f"{len(names)} 个臂；" + ", ".join(f"{k}={len(v)}" for k, v in L.items())


def _print_e5(o, names):
    LOG.info("")
    LOG.info("两两一致性 —— 单向 cov_ij（论文 §3/§6 的口径）")
    for k, v in o.get("pairwise_all_cov_ij", {}).items():
        if v.get("n"):
            LOG.info(f"  {k:<36} n={v['n']:6d} agree {pct(v['agree'])} κ={fmt(v['kappa'])} "
                     f"AC1={fmt(v['ac1'])}  覆盖率 {fmt(v['rate_a'])}/{fmt(v['rate_b'])}")
    LOG.info("两两一致性 —— 对级 substitutable（全体合格对，未加权）")
    for k, v in o["pairwise_all"].items():
        if v.get("n"):
            LOG.info(f"  {k:<36} n={v['n']:6d} agree {pct(v['agree'])} κ={fmt(v['kappa'])} "
                     f"PABAK={fmt(v['pabak'])} AC1={fmt(v['ac1'])}  正例率 {pct(v['rate_a'])}/{pct(v['rate_b'])}")
    LOG.info("τ*（s8 多档）")
    for x, t in o["tau_multi"].items():
        LOG.info("  " + f"{x:<18}" + "  ".join(
            f"{k}: τ*={fmt(v.get('tau_star'),4)} 放行{pct(v.get('pass_frac'))} 同文本{pct(v.get('same_text_frac'))}"
            for k, v in sorted(t.items())))
    LOG.info("金标准确率（同一批交集）")
    for x, v in o["gold"]["by_arm"].items():
        if v.get("n"):
            LOG.info(f"  {x:<18} n={v['n']} acc={fmt(v['acc'])} CI[{fmt(v['acc_ci'][0])},"
                     f"{fmt(v['acc_ci'][1])}] P={fmt(v['precision'])} R={fmt(v['recall'])}")


# ==============================================================================
#  E5c：盲测留出人工集
# ==============================================================================

ANNOT_GUIDE = (
    "每一题给出一个问题和一组检索结果。请只根据这些检索结果判断：它们是否包含"
    "回答该问题所需的全部事实？\n"
    "· 只看检索结果里有没有，不要用你自己的知识补全；没有明确写出的，一律算「不够」。\n"
    "· 结果的先后顺序与编号无关紧要。\n"
    "· 拿不准就选「不确定」，不要猜。\n"
    "标注过程中不要与其他标注者讨论，也不要查看任何模型的判定。")


def cmd_e5c_export(a):
    P = P1(a)
    m = MainRun(a.main_run)
    hdir = a.p3 / "e5" / "human"
    if (hdir / "key.json").exists() and not a.redo:
        return f"已导出过：{hdir}（--redo 重新导出会换一批样本，慎用）"
    cfg = make_cfg(a, "phase3_tmp")
    excl = gold_pair_ids(a)
    s22 = a.main_run / "stages" / "s22"
    if s22.is_dir():
        for f in s22.glob("*.jsonl"):
            with contextlib.suppress(Exception):
                excl |= {r["pair_id"] for r in rjl(f) if "pair_id" in r}
    pool = [r for r in m.eligible() if r["pair_id"] not in excl and not is_forced_identity(r)]
    rng = random.Random(SEED + 7)
    samp = []
    for name, lo, hi in HUMAN_BANDS:
        lst = sorted([r for r in pool if lo <= r["cos"] < hi], key=lambda r: r["pair_id"])
        k = min(a.human_per_band, len(lst))
        samp += [(name, r) for r in rng.sample(lst, k)]
        LOG.info(f"  {name}: 候选 {len(lst)}，抽 {k}")
    items, key = [], {}
    for band, r in samp:
        ci, cj = m.calls[r["qid_i"]], m.calls[r["qid_j"]]
        for d, (q, ids) in (("ij", (ci["query"], cj.get("retrieved_chunks") or [])),
                            ("ji", (cj["query"], ci.get("retrieved_chunks") or []))):
            iid = hashlib.sha1(f"{SEED}|{r['pair_id']}|{d}".encode()).hexdigest()[:10]
            items.append({"id": iid, "q": q,
                          "docs": [(m.cmap.get(c) or "")[:cfg.judge_doc_chars] for c in ids]})
            key[iid] = {"pair_id": r["pair_id"], "dir": d, "band": band, "cos": r["cos"]}
    rng.shuffle(items)
    hdir.mkdir(parents=True, exist_ok=True)
    wj(hdir / "key.json", {"seed": SEED + 7, "excluded_gold_or_s22": len(excl),
                           "bands": HUMAN_BANDS, "items": key})
    (hdir / "annotate.html").write_text(_annot_html(items), encoding="utf-8")
    (hdir / "GUIDE.txt").write_text(ANNOT_GUIDE + "\n", encoding="utf-8")
    LOG.info(f"  {len(samp)} 对 → {len(items)} 题 → {hdir/'annotate.html'}")
    LOG.info("  把 annotate.html 发给标注者（离线打开即可，自动保存在本机浏览器）；"
             "标完点「导出 TSV」，收回后跑 e5c-eval --labels a.tsv b.tsv")
    return f"{len(samp)} 对 / {len(items)} 题"


def _annot_html(items, store="ann45"):
    data = json.dumps(items, ensure_ascii=False).replace("</", "<\\/")
    guide = html.escape(ANNOT_GUIDE).replace("\n", "<br>")
    return """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>检索充分性标注</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--bd:#dedcd5;--acc:#2f5d8a;--ok:#2e7d32;--no:#b3261e;--un:#8a6d1f}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1a;--fg:#ecebe6;--mut:#a3a29c;--card:#252524;--bd:#3a3a37;--acc:#8fb3d9;--ok:#81c784;--no:#ef9a9a;--un:#e6c56b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
.wrap{max-width:920px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between;margin-bottom:12px}
.guide{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:10px 14px;color:var(--mut);font-size:13px;margin-bottom:12px}
.q{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:14px;margin-bottom:12px}
.q h2{font-size:17px;margin:0 0 4px}.lab{font-size:12px;color:var(--mut)}
.doc{border-top:1px solid var(--bd);padding:8px 0;font-size:14px;white-space:pre-wrap;word-break:break-word}
.doc b{color:var(--acc)}
.bar{position:sticky;bottom:0;background:var(--bg);padding:10px 0;display:flex;flex-wrap:wrap;gap:8px;border-top:1px solid var(--bd)}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--bd);background:var(--card);color:var(--fg);cursor:pointer}
button.sel{outline:3px solid var(--acc)}.b1{color:var(--ok)}.b0{color:var(--no)}.b9{color:var(--un)}
input{font:inherit;padding:6px 8px;border:1px solid var(--bd);border-radius:6px;background:var(--card);color:var(--fg)}
#note{flex:1;min-width:160px}
</style></head><body><div class="wrap">
<header><div><b>检索充分性标注</b> <span class="lab" id="prog"></span></div>
<div><input id="who" placeholder="标注者代号" size="10"> <button id="exp">导出 TSV</button></div></header>
<div class="guide">""" + guide + """<br>快捷键：1 = 够，0 = 不够，9 = 不确定，←/→ 翻题。</div>
<div id="card" class="q"></div>
<div class="bar"><button class="b1" data-v="1">1 · 够</button><button class="b0" data-v="0">0 · 不够</button>
<button class="b9" data-v="?">9 · 不确定</button><input id="note" placeholder="备注（可空）">
<button id="prev">←</button><button id="next">→</button><button id="jump">跳到未标</button></div>
</div><script>
const ITEMS=""" + data + """;
let i=0,who="",L={};
function key(){return "%STORE%:"+(who||"anon")}
function load(){try{L=JSON.parse(localStorage.getItem(key())||"{}")}catch(e){L={}}}
function save(){try{localStorage.setItem(key(),JSON.stringify(L))}catch(e){}}
function esc(s){return s.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function draw(){const it=ITEMS[i];const c=document.getElementById("card");
c.innerHTML='<div class="lab">第 '+(i+1)+' / '+ITEMS.length+' 题 · '+it.id+'</div><h2>'+esc(it.q)+'</h2>'+
it.docs.map((d,k)=>'<div class="doc"><b>['+(k+1)+']</b> '+esc(d)+'</div>').join("");
const cur=L[it.id]||{};document.querySelectorAll("[data-v]").forEach(b=>b.classList.toggle("sel",b.dataset.v===cur.v));
document.getElementById("note").value=cur.n||"";
const n=Object.values(L).filter(x=>x.v).length;document.getElementById("prog").textContent="已标 "+n+" / "+ITEMS.length;window.scrollTo(0,0)}
function setv(v){if(!who){alert("先在右上角填标注者代号");document.getElementById("who").focus();return}const it=ITEMS[i];L[it.id]={v:v,n:document.getElementById("note").value,t:Date.now()};save();if(i<ITEMS.length-1)i++;draw()}
document.querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>setv(b.dataset.v));
document.getElementById("prev").onclick=()=>{if(i>0)i--;draw()};
document.getElementById("next").onclick=()=>{if(i<ITEMS.length-1)i++;draw()};
document.getElementById("jump").onclick=()=>{const k=ITEMS.findIndex(x=>!(L[x.id]&&L[x.id].v));if(k>=0)i=k;draw()};
document.getElementById("note").onchange=e=>{const it=ITEMS[i];if(L[it.id]){L[it.id].n=e.target.value;save()}};
const W=document.getElementById("who");try{W.value=localStorage.getItem("%STORE%:who")||""}catch(e){}
who=W.value;load();W.onchange=()=>{who=W.value.trim();try{localStorage.setItem("%STORE%:who",who)}catch(e){};load();draw();W.blur()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;
if(e.key==="1")setv("1");else if(e.key==="0")setv("0");else if(e.key==="9")setv("?");
else if(e.key==="ArrowLeft"){if(i>0)i--;draw()}else if(e.key==="ArrowRight"){if(i<ITEMS.length-1)i++;draw()}});
document.getElementById("exp").onclick=()=>{if(!who){alert("先填标注者代号");return}
const rows=["item_id\\tlabel\\tnote\\tannotator\\tts"];ITEMS.forEach(it=>{const x=L[it.id];if(x&&x.v)rows.push([it.id,x.v,(x.n||"").replace(/[\\t\\n]/g," "),who,x.t].join("\\t"))});
const b=new Blob([rows.join("\\n")+"\\n"],{type:"text/tab-separated-values"});const u=URL.createObjectURL(b);
const el=document.createElement("a");el.href=u;el.download="%STORE%_labels_"+who+".tsv";el.click();URL.revokeObjectURL(u)};
draw();
</script></body></html>""".replace("%STORE%", store)


def _read_labels(paths, key):
    """{annotator: {item_id: 0/1/None}}；同一文件里可以有多位标注者（按 annotator 列分开）。"""
    import csv
    ann = defaultdict(dict)
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                iid = (row.get("item_id") or "").strip()
                if iid not in key:
                    continue
                who = (row.get("annotator") or Path(p).stem).strip()
                v = (row.get("label") or "").strip()
                ann[who][iid] = int(v) if v in ("0", "1") else None
    return ann


def cmd_e5c_eval(a):
    """设计：每题（一个方向）至少一人标；部分题两人标，用来算标注者间一致性。
    题级标签 = 所有标过它的人一致的标签（有人选「不确定」或彼此不一致 -> 待裁决）；
    对级标签 = 两个方向题级标签的 AND（两个方向可能由不同的人标）。"""
    hdir = a.p3 / "e5" / "human"
    key = rj(hdir / "key.json")["items"]
    if not a.labels:
        raise RuntimeError("用 --labels a.tsv [b.tsv ...] 指定标注结果")
    ann = _read_labels(a.labels, key)
    ws = sorted(ann)
    for w in ws:
        LOG.info(f"  标注者 {w}: {len(ann[w])} 题，其中不确定 {sum(v is None for v in ann[w].values())}")
    out = {"annotators": ws, "n_items": len(key),
           "items_labeled": len({i for w in ws for i in ann[w]}),
           "items_double_labeled": sum(1 for i in key if sum(i in ann[w] for w in ws) >= 2)}
    # 标注者间一致性：所有被 ≥2 人标过的题，两两比较
    xs, ys, pairs_n = [], [], Counter()
    for i in key:
        lab = [(w, ann[w][i]) for w in ws if i in ann[w]]
        for p in range(len(lab)):
            for q in range(p + 1, len(lab)):
                xs.append(lab[p][1]); ys.append(lab[q][1]); pairs_n[f"{lab[p][0]}~{lab[q][0]}"] += 1
    if xs:
        both = [(x, y) for x, y in zip(xs, ys)]
        out["inter_annotator_direction"] = {
            **agree(xs, ys), "n_compared": len(both),
            "n_one_unsure": sum(1 for x, y in both if (x is None) != (y is None)),
            "by_pair_of_annotators": dict(pairs_n)}
        ia = out["inter_annotator_direction"]
        LOG.info(f"  标注者间（方向级，{ia['n_compared']} 次比较，排除不确定后 n={ia.get('n')}）："
                 f"一致 {pct(ia.get('agree'))} κ={fmt(ia.get('kappa'))} AC1={fmt(ia.get('ac1'))}")
    else:
        LOG.warning("  没有被两人同时标过的题 —— 算不了标注者间一致性（跑 e5c-overlap 补一轮）")
    # 题级 -> 对级
    adj = {}
    if a.adjudicated and Path(a.adjudicated).exists():
        import csv
        with open(a.adjudicated, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if (row.get("label") or "").strip() in ("0", "1"):
                    adj[row["pair_id"].strip()] = int(row["label"])
    item_lab, unresolved = {}, []
    for i in key:
        vs = [ann[w][i] for w in ws if i in ann[w]]
        if not vs:
            continue
        if None not in vs and len(set(vs)) == 1:
            item_lab[i] = vs[0]
        else:
            unresolved.append(i)
    by_pair = defaultdict(dict)
    for i, v in item_lab.items():
        by_pair[key[i]["pair_id"]][key[i]["dir"]] = v
    cons = {pid: int(d["ij"] and d["ji"]) for pid, d in by_pair.items() if "ij" in d and "ji" in d}
    # 一个方向为 0 就足以判对为 0，即使另一方向待裁决
    for pid, d in by_pair.items():
        if pid not in cons and 0 in d.values():
            cons[pid] = 0
    cons.update(adj)
    pids = {k["pair_id"] for k in key.values()}
    pending = sorted(pids - set(cons))
    if pending:
        with open(hdir / "to_adjudicate.tsv", "w", encoding="utf-8") as f:
            f.write("pair_id\tlabel\tband\titems_unresolved\n")
            for pid in pending:
                its = [i for i in unresolved if key[i]["pair_id"] == pid]
                band = next(k["band"] for k in key.values() if k["pair_id"] == pid)
                f.write(f"{pid}\t\t{band}\t{','.join(its)}\n")
        LOG.info(f"  {len(pending)} 对因「不确定」或分歧无法定标签 → {hdir/'to_adjudicate.tsv'}"
                 f"（第三人填 label 后用 --adjudicated 传回）")
    out.update({"n_item_labels": len(item_lab), "n_items_unresolved": len(unresolved),
                "consensus_n": len(cons), "pending_n": len(pending),
                "item_pos_rate": float(np.mean(list(item_lab.values()))) if item_lab else None})
    band = {k["pair_id"]: k["band"] for k in key.values()}
    runs = arm_runs(a)
    arms = {}
    for x, run in runs.items():
        Lx = load_labels(run, x == "main")
        ks = [k for k in cons if k in Lx]
        r = {"all": acc_vs([cons[k] for k in ks], [Lx[k]["substitutable"] for k in ks])}
        # 方向级：对每个有题级标签的方向，比判官的 cov_xx
        kd = [(i, key[i]) for i in item_lab if key[i]["pair_id"] in Lx]
        r["direction"] = acc_vs([item_lab[i] for i, _ in kd],
                                [Lx[k["pair_id"]].get(f"cov_{k['dir']}") for _, k in kd])
        for bname, _, _ in HUMAN_BANDS:
            kb = [k for k in ks if band[k] == bname]
            r[bname] = acc_vs([cons[k] for k in kb], [Lx[k]["substitutable"] for k in kb])
        arms[x] = r
        LOG.info(f"  {x:<16} 对级 n={r['all'].get('n')} acc={fmt(r['all'].get('acc'))} "
                 f"P={fmt(r['all'].get('precision'))} R={fmt(r['all'].get('recall'))} | 方向级 n="
                 f"{r['direction'].get('n')} acc={fmt(r['direction'].get('acc'))} "
                 f"R={fmt(r['direction'].get('recall'))} 特异度="
                 f"{fmt((r['direction'].get('confusion') or {}).get('tn', 0) / max(((r['direction'].get('confusion') or {}).get('tn', 0) + (r['direction'].get('confusion') or {}).get('fp', 0)), 1))}")
    out["judge_vs_human"] = arms
    out["human_pos_rate_by_band"] = {b: (float(np.mean([cons[k] for k in cons if band[k] == b]))
                                         if any(band[k] == b for k in cons) else None)
                                     for b, _, _ in HUMAN_BANDS}
    out["design_note"] = ("每题单人标注（四人各标互不重叠的四分之一），另有重叠子集用于标注者间一致性。"
                          "标注时间未可靠记录。")
    wj(hdir / "eval.json", out)
    return f"对级标签 {len(cons)}，待裁决 {len(pending)}，{len(arms)} 个判官臂已对比"


def cmd_e5c_overlap(a):
    """为标注者间一致性补一轮：从每位原标注者的题里随机抽 n/4 题，交给另一位（轮换）盲标。
    生成每人一个 HTML（不含第一轮标签），存储前缀 ann45ov，不会和第一轮混在一个浏览器里。"""
    m = MainRun(a.main_run)
    hdir = a.p3 / "e5" / "human"
    key = rj(hdir / "key.json")["items"]
    if not a.labels:
        raise RuntimeError("用 --labels 指定第一轮的标注文件（要知道谁标了哪题）")
    ann = _read_labels(a.labels, key)
    ws = sorted(ann)
    if len(ws) < 2:
        raise RuntimeError("至少要两位标注者")
    cfg = make_cfg(a, "phase3_tmp")
    rng = random.Random(SEED + 11)
    per = max(1, a.overlap_n // len(ws))
    plan = {}
    for k, w in enumerate(ws):
        to = ws[(k + 1) % len(ws)]
        pool = sorted(ann[w])
        plan[to] = rng.sample(pool, min(per, len(pool)))
    odir = hdir / "overlap"
    odir.mkdir(parents=True, exist_ok=True)
    for to, iids in plan.items():
        items = []
        for iid in iids:
            k = key[iid]
            r = next(x for x in m.eligible() if x["pair_id"] == k["pair_id"])
            ci, cj = m.calls[r["qid_i"]], m.calls[r["qid_j"]]
            q, ids = ((ci["query"], cj.get("retrieved_chunks") or []) if k["dir"] == "ij"
                      else (cj["query"], ci.get("retrieved_chunks") or []))
            items.append({"id": iid, "q": q,
                          "docs": [(m.cmap.get(c) or "")[:cfg.judge_doc_chars] for c in ids]})
        rng.shuffle(items)
        (odir / f"annotate_overlap_{to}.html").write_text(_annot_html(items, store="ann45ov"),
                                                          encoding="utf-8")
        LOG.info(f"  {to}: {len(items)} 题 -> {odir/f'annotate_overlap_{to}.html'}")
    wj(odir / "plan.json", {"seed": SEED + 11, "assign": plan})
    return f"重叠子集 {sum(len(v) for v in plan.values())} 题，分给 {len(plan)} 人"



# ==============================================================================
#  E4：2Wiki 上的 agent 工作负载
# ==============================================================================

def e4_run(a):
    return a.runs_root / f"{a.tag}{a.e4_run}"


def cmd_e4_collect(a):
    P = P1(a)
    m = MainRun(a.main_run)
    edir = a.p3 / "e4"; edir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(a, f"{a.tag}e4_collect", agent_enabled=True)
    if "2wiki" not in (cfg.corpus_datasets or []) and cfg.corpus_source == "datasets":
        raise RuntimeError(f"main_v10 的语料 {cfg.corpus_datasets} 不含 2wiki —— agent 检索不到"
                           f"支撑段落，实验无意义")
    if cfg.corpus_max_docs:
        LOG.warning(f"corpus_max_docs={cfg.corpus_max_docs}：语料被截断过，2Wiki 段落可能不全")
    N = a.e4_tasks
    ctx, sm = start_ctx(a, cfg, "e4-collect")
    for rel in ("s1/chunks.jsonl", "s1/chunk_emb.npy", "s1/stats.json"):
        sp, dp = a.main_run / "stages" / rel, ctx.out(*rel.split("/"))
        if not sp.exists():
            raise RuntimeError(f"缺 {sp}")
        if not dp.exists():
            dp.parent.mkdir(parents=True, exist_ok=True); os.symlink(sp.resolve(), dp)
    tasks = P._replay_tasks(ctx, "2wiki")[:N]
    rp = {c["task_id"] for c in m.calls.values() if c["source"] == "replay:2wiki"}
    ov = sum(t["task_id"] in rp for t in tasks) / max(len(tasks), 1)
    LOG.info(f"任务 {len(tasks)}（2Wiki 验证集前 {N} 题）；与 main_v10 replay:2wiki 任务重合 {pct(ov)}")
    if ov < 0.99:
        LOG.warning("重合 < 99%：这些题的支撑段落可能不在语料里（replay_max_tasks_per_ds 限制）")
    ap = Appender(edir / "collect_partial.jsonl")
    done = {k: v for k, v in ap.load("task_id").items()
            if isinstance(v.get("res"), dict) and not (v["res"].get("error") or v["res"].get("__error__"))}
    todo = [t for t in tasks if t["task_id"] not in done]
    LOG.info(f"已完成 {len(done)}，本次 {len(todo)}")
    t0 = time.time()
    llm = None
    retr = P._mk_retriever(ctx) if todo else None
    try:
        if todo:
            # TorchRetriever 的嵌入模型是懒加载的（第一次 encode 才加载）。phase1 的 s2 里
            # replay 先在主线程跑过检索，模型早已加载；这里直接进 pmap，几十个线程同时
            # 首次调用 search，会并发加载嵌入模型 -> 冒烟里 8/8 任务失败。先在主线程预热。
            t_w = time.time()
            retr.search(["warmup: who directed the film"], cfg.topk)
            LOG.info(f"检索器已预热（{time.time()-t_w:.0f}s，GPU{cfg.retriever_gpu}）")
            agent_gpus = [g for g in cfg.gpus if g != cfg.retriever_gpu] or list(cfg.gpus)
            LOG.info(f"agent {cfg.agent_name}（{cfg.agent_quant}）独占 GPU{agent_gpus}；"
                     f"检索器 GPU{cfg.retriever_gpu}；粗估 {len(todo)/2000*4:.1f} h")
            urls = sm.ensure("agent", agent_gpus)
            pool = P.LLMPool(urls, cfg.agent_name, retries=2, timeout=900)

            def one(t):
                r = P._agent_task(pool, retr, t, cfg)
                ap.add({"task_id": t["task_id"], "res": r})
                return r
            P.pmap(one, todo, cfg.agent_concurrency, "agent-2wiki", ctx, pool)
            llm = pool.stats()
    finally:
        sm.stop()
        if retr:
            retr.free()
    got = ap.load("task_id")
    calls, nerr, errs = [], 0, Counter()
    for t in tasks:
        r = (got.get(t["task_id"]) or {}).get("res")
        if not isinstance(r, dict) or r.get("error") or r.get("__error__"):
            nerr += 1; errs[str((r or {}).get("error") or (r or {}).get("__error__"))[:120]] += 1
            continue
        for c in r.get("calls", []):
            c["source"] = "agent"; c["agent_dataset"] = "2wiki"; c["agent_model"] = cfg.agent_name
            calls.append(c)
    seen, fin = set(), []
    for c in calls:
        c["query"] = re.sub(r"\s+", " ", str(c["query"])).strip()
        if len(c["query"]) < 5:
            continue
        qid = P.sha(c["task_id"], c["turn_idx"], c["query"])
        if qid in seen:
            continue
        seen.add(qid); c["qid"] = qid
        c["cost"] = (cfg.cost_per_search_call + cfg.cost_per_context_token * c["context_tokens_added"]
                     + cfg.cost_per_rerank_doc * c["n_reranked"])
        fin.append(c)
    wjl(edir / "agent_2wiki_calls.jsonl", fin)
    st = {"model": cfg.agent_name, "quant": cfg.agent_quant, "dataset": "2wiki",
          "tasks": len(tasks), "errors": nerr, "top_errors": dict(errs.most_common(5)),
          "calls": len(fin), "calls_per_task": len(fin) / max(len(tasks) - nerr, 1),
          "multi_query_frac": float(np.mean([bool(c.get("multi_query")) for c in fin])) if fin else 0.0,
          "mean_subqueries": float(np.mean([c.get("multi_query_n", 1) for c in fin])) if fin else 0.0,
          "median_query_len": float(np.median([len(c["query"]) for c in fin])) if fin else 0.0,
          "task_overlap_with_replay_2wiki": ov, "llm": llm, "wall_h": (time.time() - t0) / 3600,
          "agent_temperature": cfg.agent_temperature, "agent_max_turns": cfg.agent_max_turns}
    wj(edir / "collect_stats.json", st)
    for msg, c in errs.most_common(5):
        LOG.error(f"  失败 ×{c}: {msg}")
    if nerr > 0.2 * len(tasks):
        raise RuntimeError(f"失败 {nerr}/{len(tasks)} > 20%，见上面的错误；重跑本步只补失败的任务")
    if nerr > 0.05 * len(tasks):
        LOG.warning(f"失败 {nerr}/{len(tasks)} > 5%，重跑本步会只补失败的任务")
    return f"{len(fin)} 调用 / {len(tasks)} 任务（{st['calls_per_task']:.2f}/任务），失败 {nerr}"


def cmd_e4_merge(a):
    m = MainRun(a.main_run)
    out = e4_run(a)
    cp = out / "stages" / "s2" / "calls.jsonl"
    if cp.exists() and not a.redo:
        return f"{cp} 已存在"
    new = rjl(a.p3 / "e4" / "agent_2wiki_calls.jsonl")
    if not new:
        raise RuntimeError("2Wiki agent 调用为 0：e4-collect 没采到东西，先修 e4-collect")
    replay = [c for c in m.calls.values() if c["source"].startswith("replay")]
    rq = {c["qid"] for c in replay}
    dup = [c for c in new if c["qid"] in rq]
    new = [c for c in new if c["qid"] not in rq]       # 与 s2 一致：replay 在前，撞 qid 的丢后者
    merged = replay + new
    LOG.info(f"main_v10 replay {len(replay)} + 2Wiki agent {len(new)}（与 replay 撞 qid 丢 {len(dup)}）"
             f" = {len(merged)}")
    (out / "stages" / "s2").mkdir(parents=True, exist_ok=True)
    wjl(cp, merged)
    qt = Counter(c["query"] for c in merged); bt = defaultdict(set)
    for c in merged:
        bt[c["query"]].add(c["task_id"])
    st = {"total_calls": len(merged), "total_tasks": len({c["task_id"] for c in merged}),
          "calls_per_task": len(merged) / max(len({c["task_id"] for c in merged}), 1),
          "total_cost": sum(c["cost"] for c in merged),
          "sanity": {
              "exact_dup_query_frac": sum(n for q, n in qt.items() if len(bt[q]) >= 2) / len(merged),
              "unresolved_placeholder_frac": sum(1 for c in merged if re.search(r"#\d+", c["query"])) / len(merged),
              "unique_query_frac": len(qt) / len(merged),
              "median_query_len": float(np.median([len(c["query"]) for c in merged])),
              "with_gold_frac": sum(1 for c in merged if c.get("gold_answer")) / len(merged)},
          "agent": rj(a.p3 / "e4" / "collect_stats.json"),
          "e4": {"agent_dataset": "2wiki", "replay_from": str(a.main_run), "dropped_qid_collisions": len(dup)}}
    for name in sorted({c["source"].split(":", 1)[1] for c in replay if ":" in c["source"]}):
        st[name] = {"calls": sum(1 for c in replay if c["source"] == f"replay:{name}"),
                    "tasks": len({c["task_id"] for c in replay if c["source"] == f"replay:{name}"})}
    wj(out / "stages" / "s2" / "stats.json", st)
    n = link_files(a.main_run, out, ["s0/preflight.json", "s1/chunks.jsonl", "s1/chunk_emb.npy",
                                     "s1/stats.json", "s6/qc.json", "s7/summary.json",
                                     "s7/l2.jsonl", "s8/calib.json", "s8/iso.pkl"])
    wj(out / "E4_PROVENANCE.json",
       {"what": "main_v10 replay + Tongyi agent on 2WikiMultihopQA",
        "linked_from_main": ["s0/preflight.json", "s1/*", "s6/qc.json", "s7/*", "s8/calib.json",
                             "s8/iso.pkl"],
        "why_s8_linked": "检测器固定为 main_v10 标定的 τ*；E4 量的是工作负载，不是重新标定",
        "s6_qc_note": "s10 里引用的判官 QC 字段（金标、κ…）来自 main_v10，不是本 run 的",
        "recompute": "s3,s4,s9,s10,s13"})
    if not (out / "state.json").exists():
        wj(out / "state.json", {"created_at": time.time(), "stages": {}})
    LOG.info(f"  软链 {n} 个文件；-> {out}")
    return f"{len(merged)} 调用 -> {out}"


def cmd_e4_downstream(a):
    run_phase1(a, e4_run(a).name, "s3,s4,s9,s10,s13", force=True)
    r = rj(e4_run(a) / "stages" / "s10" / "reuse.json")
    ag = (r.get("by_source") or {}).get("agent") or {}
    return f"2Wiki agent 头部成本覆盖 {pct(ag.get('head_cov_cost'))}，意图 {ag.get('n_intents')}"


def cmd_e4_validate(a):
    """检测器迁移：main_v10 的 τ* 在 2Wiki-agent 的对上是否仍达到保证水平。"""
    P = P1(a)
    run = e4_run(a)
    vdir = a.p3 / "e4"
    pid_, tmpl = audited_prompt(a)
    calls = {c["qid"]: c for c in rjl(run / "stages" / "s2" / "calls.jsonl")}
    cmap = {c["chunkid"]: c["text"] for c in rjl(a.main_run / "stages" / "s1" / "chunks.jsonl")}
    agq = {q for q, c in calls.items() if c.get("source") == "agent"}
    pairs = []
    with open(run / "stages" / "s4" / "pairs.jsonl", encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            p = json.loads(ln)
            if p["qid_i"] in agq and p["qid_j"] in agq:
                pairs.append(p)
    if not pairs:
        raise RuntimeError("2Wiki-agent × 2Wiki-agent 候选对为 0：先确认 e4-collect / e4-downstream")
    cfg = make_cfg(a, f"{a.tag}e4_validate")
    edges = cfg.sim_bins
    popn = Counter(bin_of(p["cos"], edges) for p in pairs)
    LOG.info(f"2Wiki-agent × 2Wiki-agent 候选对 {len(pairs)}；分层 " +
             ", ".join(f"b{b}={popn[b]}" for b in sorted(popn)))
    hi = a.val_per_bin; lo = a.val_per_bin_low
    samp = strat_sample(pairs, edges, lambda b: hi if edges[b] >= 0.80 else lo, SEED + 3)
    LOG.info(f"抽 {len(samp)} 对；判官 {cfg.judge_name}，prompt {pid_}（与 main_v10 相同）")
    ap = Appender(vdir / "validate_pairs.jsonl")
    done = {k: v for k, v in ap.load("pair_id").items() if v.get("done")}
    todo = [p for p in samp if p["pair_id"] not in done]

    ctx, sm = start_ctx(a, cfg, "e4-validate")
    try:
        urls = sm.ensure("judge", list(cfg.gpus))
        pool = P.LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)
        sc_cache, lock = {}, threading.Lock()

        def selfcov(qid):
            with lock:
                if qid in sc_cache:
                    return sc_cache[qid]
            c = calls[qid]
            j = judge_call(P, pool, cfg, tmpl, c["query"],
                           P._fmt(c["retrieved_chunks"], cmap, cfg.judge_doc_chars),
                           False, cfg.judge_max_tokens, "qwen3")
            v = j.get("covered") if j["ok"] else None
            with lock:
                sc_cache[qid] = v
            return v

        def one(p):
            ci, cj = calls[p["qid_i"]], calls[p["qid_j"]]
            out = {k: p.get(k) for k in ("pair_id", "qid_i", "qid_j", "cos", "same_text",
                                         "jaccard", "rbo")}
            out["bin"] = bin_of(p["cos"], edges)
            out["cross_task"] = ci["task_id"] != cj["task_id"]
            si, sj = selfcov(p["qid_i"]), selfcov(p["qid_j"])
            out["self_cov_i"], out["self_cov_j"] = si, sj
            out["eligible"] = int(si == 1 and sj == 1)
            if out["eligible"]:
                rij = judge_call(P, pool, cfg, tmpl, ci["query"],
                                 P._fmt(cj["retrieved_chunks"], cmap, cfg.judge_doc_chars),
                                 False, cfg.judge_max_tokens, "qwen3")
                rji = judge_call(P, pool, cfg, tmpl, cj["query"],
                                 P._fmt(ci["retrieved_chunks"], cmap, cfg.judge_doc_chars),
                                 False, cfg.judge_max_tokens, "qwen3")
                out.update({"cov_ij": rij.get("covered") if rij["ok"] else None,
                            "cov_ji": rji.get("covered") if rji["ok"] else None,
                            "verdict_ij": rij["verdict"], "verdict_ji": rji["verdict"]})
                out["ok"] = out["cov_ij"] is not None and out["cov_ji"] is not None
                out["substitutable"] = int(bool(out["cov_ij"]) and bool(out["cov_ji"])) if out["ok"] else None
            out["done"] = si is not None and sj is not None
            ap.add(out)
            return out
        if todo:
            P.pmap(one, todo, cfg.judge_concurrency, "e4-validate", ctx, pool)
    finally:
        sm.stop()

    got = ap.load("pair_id")
    rs = [got[p["pair_id"]] for p in samp if p["pair_id"] in got]
    ev = [r for r in rs if r.get("eligible") and r.get("ok")]
    mc = rj(a.main_run / "stages" / "s8" / "calib.json")
    mper = mc.get("per_bin") or {}
    # 参照：main_v10 里 MuSiQue-agent × MuSiQue-agent 的合格对
    m = MainRun(a.main_run)
    mag = [r for r in m.eligible() if m.calls.get(r["qid_i"], {}).get("source") == "agent"
           and m.calls.get(r["qid_j"], {}).get("source") == "agent"]
    per = {}
    for b in range(len(edges) - 1):
        rb = [r for r in rs if r["bin"] == b]
        eb = [r for r in ev if r["bin"] == b]
        k = sum(r["substitutable"] for r in eb)
        lo_, hi_ = wilson(k, len(eb))
        mb = [r for r in mag if bin_of(r["cos"], edges) == b]
        per[b] = {"range": [edges[b], edges[b + 1]], "N_pop": popn.get(b, 0), "n_sampled": len(rb),
                  "eligibility_rate": (sum(r["eligible"] for r in rb if r.get("done")) /
                                       max(sum(1 for r in rb if r.get("done")), 1)) if rb else None,
                  "n_eligible": len(eb), "p": k / len(eb) if eb else None, "ci": [lo_, hi_],
                  "main_v10_all_p": (mper.get(str(b)) or mper.get(b) or {}).get("p"),
                  "main_v10_musique_agent_p": float(np.mean([r["substitutable"] for r in mb])) if mb else None,
                  "main_v10_musique_agent_n": len(mb)}
        LOG.info(f"  b{b} [{edges[b]:.3f},{edges[b+1]:.3f}) N={popn.get(b,0):6d} 判 {len(eb):4d} "
                 f"p={fmt(per[b]['p'])} CI[{fmt(lo_)},{fmt(hi_)}] | main 全体 {fmt(per[b]['main_v10_all_p'])} "
                 f"main MuSiQue-agent {fmt(per[b]['main_v10_musique_agent_p'])} (n={len(mb)})")

    # 各保证档：用 main_v10 的 τ*，在 2Wiki-agent 总体上估精度（分层加权 + 分层自助）
    ops = {}
    rng = np.random.default_rng(SEED)
    for t, e in sorted((mc.get("tau_multi") or {}).items()):
        tau = e.get("tau_star")
        if tau is None:
            ops[t] = {"tau_star": None}; continue
        pass_pop = [p for p in pairs if p["cos"] >= tau]
        nb = Counter(bin_of(p["cos"], edges) for p in pass_pop)
        strata = {b: np.array([r["substitutable"] for r in ev if r["bin"] == b and r["cos"] >= tau])
                  for b in nb}
        miss = [b for b in nb if len(strata[b]) == 0]
        for b in miss:     # 跨 τ* 的层若样本都在 τ* 下，退回整层
            strata[b] = np.array([r["substitutable"] for r in ev if r["bin"] == b])
        W = sum(nb.values())
        if not W or any(len(strata[b]) == 0 for b in nb):
            empty = [b for b in nb if len(strata[b]) == 0]
            ops[t] = {"tau_star": tau, "n_pass_pop": W, "empty_bins": empty,
                      "note": "放行区间里有层没有合格且判成功的样本，无法估精度"}
            LOG.warning(f"  保证 {t}: τ*={tau:.4f} 放行 {W} 对，但 b{empty} 没有可用样本 → 无法估计；"
                        f"加大 --val-per-bin")
            continue
        est = sum(nb[b] * strata[b].mean() for b in nb) / W
        boots = []
        for _ in range(2000):
            boots.append(sum(nb[b] * rng.choice(strata[b], len(strata[b])).mean() for b in nb) / W)
        lo_b = float(np.percentile(boots, 2.5))
        st_share = sum(1 for p in pass_pop if p.get("same_text")) / W
        ops[t] = {"tau_star": tau, "n_pass_pop": W, "pass_frac_of_agent_pairs": W / max(len(pairs), 1),
                  "precision_hat": float(est), "precision_lo95": lo_b, "target": float(t),
                  "meets_target_lower": lo_b >= float(t), "meets_target_point": est >= float(t),
                  "same_text_share": st_share, "strata_fallback_bins": miss,
                  "n_judged_used": int(sum(len(v) for v in strata.values()))}
        LOG.info(f"  保证 {t}: τ*={tau:.4f} 放行 {W} 对（{pct(W/max(len(pairs),1))}），同文本 {pct(st_share)}，"
                 f"精度 {est:.3f} [下界 {lo_b:.3f}] → {'达标' if lo_b >= float(t) else '下界未达标'}")
    out = {"prompt_id": pid_, "prompt_sha256": psha(tmpl), "judge": cfg.judge_name,
           "n_candidate_pairs": len(pairs), "n_sampled": len(samp), "n_done": len(rs),
           "n_eligible_ok": len(ev),
           "eligibility_rate": sum(r["eligible"] for r in rs if r.get("done")) / max(sum(1 for r in rs if r.get("done")), 1),
           "cross_task_share_sampled": float(np.mean([r["cross_task"] for r in rs])) if rs else None,
           "per_bin": per, "operating_points": ops,
           "note": "精度=条件可替代率（两端 self-cov 都过的对），与 main_v10 的 s8 口径一致；"
                   "分层权重=2Wiki-agent 候选对总体在各层的数量"}
    wj(vdir / "validate.json", out)
    return " | ".join(f"{t}: 精度 {fmt(v.get('precision_hat'))} 下界 {fmt(v.get('precision_lo95'))}"
                      for t, v in ops.items() if v.get("precision_hat") is not None)


def cmd_e4_report(a):
    run = e4_run(a)

    def ld(r, f):
        p = r / "stages" / f
        return rj(p) if p.exists() else {}
    A, B = ld(a.main_run, "s10/reuse.json"), ld(run, "s10/reuse.json")
    keys = ("n_calls", "n_tasks", "n_intents", "head_cov_cost", "head_cov_calls",
            "cross_intent_frac", "cross_cost_frac", "head_cross_share", "mean_tasks_per_intent",
            "head_mono_text_share", "mono_text_cluster_frac", "gini_cost", "hill_alpha")
    rows = {}
    for tag, R in (("main_v10", A), ("e4", B)):
        for s, v in (R.get("by_source") or {}).items():
            rows[f"{tag}:{s}"] = {k: v.get(k) for k in keys}
    sa = (ld(a.main_run, "s2/stats.json").get("agent") or {})
    sb = rj(a.p3 / "e4" / "collect_stats.json") if (a.p3 / "e4" / "collect_stats.json").exists() else {}
    fits = {}
    for tag, r in (("main_v10", a.main_run), ("e4", run)):
        f = (ld(r, "s13/scale.json").get("fits") or {})
        for s, d in f.items():
            if d:
                fits[f"{tag}:{s}"] = {"slope": d.get("slope"), "r2": d.get("r2"),
                                      "current": d.get("current"), "pred": d.get("pred")}
    ga, gb = rows.get("main_v10:agent", {}), rows.get("e4:agent", {})
    d = (gb.get("head_cov_cost") - ga.get("head_cov_cost")) \
        if ga.get("head_cov_cost") is not None and gb.get("head_cov_cost") is not None else None
    r2a, r2b = rows.get("main_v10:replay:2wiki", {}), rows.get("e4:replay:2wiki", {})
    val = rj(a.p3 / "e4" / "validate.json") if (a.p3 / "e4" / "validate.json").exists() else {}
    out = {"by_source": rows, "agent_stats": {"musique (main_v10)": sa, "2wiki (e4)": sb},
           "scaling_fits": fits, "delta_head_cov_cost_agent_2wiki_minus_musique": d,
           "sanity_replay_2wiki_head_cov_cost": [r2a.get("head_cov_cost"), r2b.get("head_cov_cost")],
           "validation": {t: {k: v.get(k) for k in ("tau_star", "precision_hat", "precision_lo95",
                                                    "meets_target_lower", "same_text_share",
                                                    "pass_frac_of_agent_pairs")}
                          for t, v in (val.get("operating_points") or {}).items()},
           "tau_used": (ld(run, "s9/clusters.json") or {}).get("tau"),
           "mechanical_rules": {
               "|Δ 头部成本覆盖| < 5pp（对任务源不敏感）": (abs(d) < 0.05) if d is not None else None,
               "replay:2wiki 两 run 差 < 1pp（合并没破坏 replay）":
                   (abs((r2a.get("head_cov_cost") or 0) - (r2b.get("head_cov_cost") or 0)) < 0.01)
                   if r2a and r2b else None}}
    wj(a.p3 / "e4" / "compare.json", out)
    LOG.info("")
    LOG.info(f"{'':<26}{'MuSiQue agent':>16}{'2Wiki agent':>16}{'2Wiki replay':>16}")
    for k in keys:
        LOG.info(f"  {k:<24}{_c(ga.get(k), k):>16}{_c(gb.get(k), k):>16}{_c(r2b.get(k), k):>16}")
    LOG.info(f"  Δ头部成本覆盖（2Wiki−MuSiQue agent）= {pct(d) if d is not None else '—'}")
    return f"Δ head_cov_cost = {pct(d) if d is not None else '—'}"


_RAW_KEYS = ("mean_tasks_per_intent", "hill_alpha", "n_calls", "n_tasks", "n_intents")


def _c(v, k=""):
    if isinstance(v, float):
        return f"{v:.3f}" if (k in _RAW_KEYS or not 0 <= v <= 1) else pct(v)
    return str(v) if v is not None else "—"


# ==============================================================================
#  总报告
# ==============================================================================

def cmd_report(a):
    write_manifest(a)
    S, M = {}, []
    st = State(a.p3).d["steps"]
    M += ["# Phase 3 补实验（E4 / E5）总报告", "",
          f"生成于 {time.strftime('%F %T')}，主 run：`{a.main_run}`。"
          "本文件由 `exp45.py report` 从各步 JSON 汇总，数字均可追溯到 JSON。", ""]
    M += ["## 运行状态", "", "| 步骤 | 状态 | 耗时 (h) | 摘要 |", "|---|---|---|---|"]
    for k, v in st.items():
        M.append(f"| {k} | {v.get('status')} | {v.get('duration_h','')} | "
                 f"{str(v.get('summary') or v.get('error') or '').replace('|','/')[:120]} |")
    M.append("")

    au_p = a.p3 / "audit" / "prompt_audit.json"
    if au_p.exists():
        au = rj(au_p)
        S["audit.status"] = au["decision"]["status"]
        S["audit.main_prompt"] = au["decision"]["variant"]
        S["audit.main_prompt_sha12"] = (au["decision"].get("sha256") or "")[:12]
        S["audit.main_vs_think1_same_prompt"] = au.get("main_vs_think1_same_prompt")
        M += ["## 0 判官 prompt 审计", "",
              f"- 结论：**{au['decision']['status']} → `{au['decision']['variant']}`**"
              f"（sha256 `{S['audit.main_prompt_sha12']}`）。{au['decision']['note']}",
              f"- 静态推断：{au['evidence']['static_inference']}",
              f"- think1 的 prompt：`{au['evidence']['think1_prompt']['variant']}`；"
              f"k3：`{au['evidence'].get('k3_prompt',{}).get('variant')}`（推断）",
              f"- **对论文的影响**：{au.get('paper_consequence')}", ""]
        fp = au["evidence"].get("fingerprint")
        if fp:
            M += ["| prompt 变体 | sha12 | n | verdict 一致 | facts 逐字复现 |", "|---|---|---|---|---|"]
            for v, d in sorted(fp["by_variant"].items(), key=lambda kv: -kv[1]["facts_exact"]):
                M.append(f"| {v} | `{d['sha12']}` | {d['n']} | {pct(d['verdict_agree'])} | {pct(d['facts_exact'])} |")
            M.append("")

    cp = a.p3 / "e5" / "compare.json"
    if cp.exists():
        o = rj(cp)
        M += ["## E5 判官的独立验证", ""]
        for nm in o["runs"]:
            jp = a.p3 / "e5" / nm / "rejudge.json"
            if jp.exists():
                j = rj(jp)
                S[f"e5.{nm}.n_ok"] = j["n_ok"]; S[f"e5.{nm}.parse_fail_rate"] = j["parse_fail_rate"]
                S[f"e5.{nm}.trunc_rate"] = j["trunc_rate"]
                M.append(f"- `{nm}`：{j['aggregation']['model']}，prompt `{j['aggregation']['prompt_id']}`，"
                         f"thinking={j['aggregation']['thinking']}，有效 {j['n_ok']}/{j['n_target']}，"
                         f"解析失败 {pct(j['parse_fail_rate'],2)}，截断 {pct(j['trunc_rate'],2)}，"
                         f"{j['wall_h']:.1f} h")
            gp = a.p3 / "e5" / nm / "gate.json"
            if gp.exists():
                g = rj(gp)
                M.append(f"  - 闸门 {'通过' if g['passed'] else '未过'}：金标 acc "
                         f"{fmt(g['gold_acc_this_arm'].get('acc'))}（主判官同批 "
                         f"{fmt(g['gold_acc_main_same_pairs'].get('acc'))}），"
                         f"与主判官一致 {pct(g['agreement_with_main_on_gate_sample'].get('agree'))}")
        M += ["", "**两两一致性 —— 单向 cov_ij（论文 \\CrossCfg* 宏的口径）**", "",
              "| 臂对 | n | 一致率 | κ | AC1 | 覆盖率 A / B |", "|---|---|---|---|---|---|"]
        for k, v in o.get("pairwise_all_cov_ij", {}).items():
            if v.get("n"):
                S[f"e5.cov.{k}.agree"] = v["agree"]; S[f"e5.cov.{k}.kappa"] = v["kappa"]
                S[f"e5.cov.{k}.ac1"] = v["ac1"]; S[f"e5.cov.{k}.n"] = v["n"]
                S[f"e5.cov.{k}.rate_a"] = v["rate_a"]; S[f"e5.cov.{k}.rate_b"] = v["rate_b"]
                M.append(f"| {k} | {v['n']} | {pct(v['agree'])} | {fmt(v['kappa'])} | {fmt(v['ac1'])} | "
                         f"{fmt(v['rate_a'])} / {fmt(v['rate_b'])} |")
        M += ["", "**两两一致性 —— 对级 substitutable（全体合格对，未加权；定义同 crosstab_l1.py）**", "",
              "| 臂对 | n | 一致率 | κ | PABAK | AC1 | 正例率 A / B |", "|---|---|---|---|---|---|---|"]
        for k, v in o["pairwise_all"].items():
            if v.get("n"):
                S[f"e5.pair.{k}.agree"] = v["agree"]; S[f"e5.pair.{k}.kappa"] = v["kappa"]
                S[f"e5.pair.{k}.ac1"] = v["ac1"]; S[f"e5.pair.{k}.n"] = v["n"]
                M.append(f"| {k} | {v['n']} | {pct(v['agree'])} | {fmt(v['kappa'])} | {fmt(v['pabak'])} | "
                         f"{fmt(v['ac1'])} | {pct(v['rate_a'])} / {pct(v['rate_b'])} |")
        if o.get("sanity_main_think1"):
            s = o["sanity_main_think1"]
            M.append(f"\n自检 main~think1（cov_ij）：{pct(s['agree'])} / κ {fmt(s['kappa'])} / AC1 {fmt(s['ac1'])}"
                     f" / 覆盖率 {fmt(s['rate_a'])}（论文宏 90.7% / 0.642 / 0.875 / 0.847）")
        M += ["", "**τ*（各臂 s8 多档）**", "", "| 臂 | 档 | τ* | 放行 | 同文本占比 | 退化 |", "|---|---|---|---|---|---|"]
        for x, t in o["tau_multi"].items():
            for k, v in sorted(t.items()):
                S[f"e5.tau.{x}.{k}"] = v.get("tau_star"); S[f"e5.tau.{x}.{k}.same_text"] = v.get("same_text_frac")
                M.append(f"| {x} | {k} | {fmt(v.get('tau_star'),4)} | {pct(v.get('pass_frac'))} | "
                         f"{pct(v.get('same_text_frac'))} | {v.get('degenerate')} |")
        M += ["", f"**金标准确率**（{o['gold']['n_common']} 对，各臂交集）", "",
              "| 臂 | acc | 95% CI | precision | recall |", "|---|---|---|---|---|"]
        for x, v in o["gold"]["by_arm"].items():
            if v.get("n"):
                S[f"e5.gold.{x}.acc"] = v["acc"]
                M.append(f"| {x} | {fmt(v['acc'])} | [{fmt(v['acc_ci'][0])}, {fmt(v['acc_ci'][1])}] | "
                         f"{fmt(v['precision'])} | {fmt(v['recall'])} |")
        if o.get("decomposition"):
            M += ["", "**{prompt}×{推理} 拆解**", "", "| 对照 | A | B | n | 一致率 | κ | 正例率 A / B |",
                  "|---|---|---|---|---|---|---|"]
            for k, v in o["decomposition"].items():
                if v.get("n"):
                    M.append(f"| {k} | {v['a']} | {v['b']} | {v['n']} | {pct(v['agree'])} | "
                             f"{fmt(v['kappa'])} | {pct(v['rate_a'])} / {pct(v['rate_b'])} |")
        xs = [x for x in o["runs"] if x != "main"]
        M += ["", "**分档：各臂对 main 的一致率 / κ（未加权；恒等带单列）**", "",
              "| 档 | " + " | ".join(xs) + " |", "|---" * (len(xs) + 1) + "|"]
        for sname, row in o["by_slice"].items():
            cells = []
            for x in xs:
                v = (row.get(x) or {}).get("vs_main") or {}
                cells.append(f"{pct(v.get('agree'))} / {fmt(v.get('kappa'),2)} (n={v.get('n',0)})")
            M.append(f"| {sname} | " + " | ".join(cells) + " |")
        fi = o["by_slice"].get("forced identity (same_text & J,RBO≥0.999)", {})
        if fi:
            M += ["", "**恒等带（逻辑上必为 1）正例率**：" + "，".join(
                f"{x} {pct(v.get('pos_rate'))}（n={v.get('n')}）" for x, v in fi.items())]
        if o.get("mechanical_rules"):
            M += ["", "事先写定的机械判读（不是结论）：" + "；".join(f"{k} = {v}" for k, v in o["mechanical_rules"].items())]
        M.append("")

    bp = a.p3 / "e5" / "both_eligible.json"
    if bp.exists():
        bo = rj(bp)
        M += ["## E5 双合格子集（合格性也由 Llama 判，两臂样本相同）", "",
              f"Llama 认为能被自己结果回答的查询 {pct(bo.get('llama_selfcov_rate'))}；"
              f"双合格对 {bo['n_both_eligible']}/{bo['n_eligible_main']}。",
              f"一致性：cov_ij {pct((bo.get('agreement_cov_ij') or {}).get('agree'))} "
              f"κ={fmt((bo.get('agreement_cov_ij') or {}).get('kappa'))}；对级 "
              f"{pct((bo.get('agreement_substitutable') or {}).get('agree'))} "
              f"κ={fmt((bo.get('agreement_substitutable') or {}).get('kappa'))}", "",
              "| 臂 | n | 可替代率 | 恒等带正例率 | 0.80 τ* / 同文本 | 0.85 τ* / 同文本 | 0.90 τ* / 同文本 |",
              "|---|---|---|---|---|---|---|"]
        S["e5.both.n"] = bo["n_both_eligible"]; S["e5.both.llama_selfcov_rate"] = bo.get("llama_selfcov_rate")
        for nm, v in bo["arms"].items():
            cells = []
            for t in ("0.80", "0.85", "0.90"):
                e = ((v.get("tau_multi") or {}).get(t) or {})
                S[f"e5.both.{nm}.{t}.tau"] = e.get("tau_star")
                S[f"e5.both.{nm}.{t}.same_text"] = e.get("same_text_frac")
                cells.append(f"{fmt(e.get('tau_star'),4)} / {pct(e.get('same_text_frac'))}")
            M.append(f"| {nm} | {v['n']} | {fmt(v['sub_rate'])} | {pct(v['forced_identity_pos'])} | "
                     + " | ".join(cells) + " |")
        M.append("")

    lp = a.p3 / "e5" / "lenient.json"
    if lp.exists():
        lo = rj(lp)
        M += ["## E5 敏感性：PARTIAL 也算覆盖（宽松口径）", "",
              "| 臂 | 档 | 严格 τ* | 严格同文本 | 宽松 τ* | 宽松同文本 |", "|---|---|---|---|---|---|"]
        for nm, v in lo.items():
            for t in ("0.80", "0.85", "0.90"):
                st = ((v.get("tau_multi_strict") or {}).get(t) or {})
                le = ((v.get("tau_multi") or {}).get(t) or {})
                S[f"e5.lenient.{nm}.{t}.tau"] = le.get("tau_star")
                S[f"e5.lenient.{nm}.{t}.same_text"] = le.get("same_text_frac")
                M.append(f"| {nm} | {t} | {fmt(st.get('tau_star'),4)} | {pct(st.get('same_text_frac'))} | "
                         f"{fmt(le.get('tau_star'),4)} | {pct(le.get('same_text_frac'))} |")
        M.append("")

    hp = a.p3 / "e5" / "human" / "eval.json"
    if hp.exists():
        h = rj(hp)
        ia = h.get("inter_annotator_direction") or {}
        M += ["## E5c 盲测留出人工集", "",
              f"标注者 {h['annotators']}；已标 {h['items_labeled']}/{h['n_items']} 题，其中两人标过 "
              f"{h['items_double_labeled']} 题；对级标签 {h['consensus_n']} 对，待裁决 {h['pending_n']} 对；"
              f"题级正例率 {pct(h.get('item_pos_rate'))}。",
              (f"- 标注者间一致性（方向级）：{pct(ia.get('agree'))}，κ={fmt(ia.get('kappa'))}，AC1={fmt(ia.get('ac1'))}，"
               f"n={ia.get('n')}" if ia else "- 尚无重叠标注，标注者间一致性未知"), ""]
        S["e5c.iaa.kappa"] = ia.get("kappa"); S["e5c.iaa.ac1"] = ia.get("ac1")
        M += ["", "| 臂 | n | acc | 95% CI | precision | recall | κ |", "|---|---|---|---|---|---|---|"]
        for x, r in h["judge_vs_human"].items():
            v = r["all"]
            if v.get("n"):
                S[f"e5c.{x}.acc"] = v["acc"]; S[f"e5c.{x}.precision"] = v["precision"]
                M.append(f"| {x} | {v['n']} | {fmt(v['acc'])} | [{fmt(v['acc_ci'][0])}, {fmt(v['acc_ci'][1])}] | "
                         f"{fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['kappa'])} |")
        M.append("")
    elif (a.p3 / "e5" / "human" / "key.json").exists():
        M += ["## E5c 盲测留出人工集", "", "已导出标注页 `e5/human/annotate.html`，等待标注。", ""]

    ep = a.p3 / "e4" / "compare.json"
    if ep.exists():
        o = rj(ep)
        M += ["## E4 非 MuSiQue 的 agent 工作负载（Tongyi on 2WikiMultihopQA）", ""]
        sb = o["agent_stats"].get("2wiki (e4)") or {}
        if sb:
            M.append(f"- 采集：{sb.get('tasks')} 任务，{sb.get('calls')} 次检索（{fmt(sb.get('calls_per_task'),2)}/任务），"
                     f"失败 {sb.get('errors')}，多查询比例 {pct(sb.get('multi_query_frac'))}，{fmt(sb.get('wall_h'),1)} h")
        M += ["", "| 指标 | MuSiQue agent (main_v10) | 2Wiki agent (E4) | 2Wiki replay (E4) |", "|---|---|---|---|"]
        ga = o["by_source"].get("main_v10:agent", {}); gb = o["by_source"].get("e4:agent", {})
        r2 = o["by_source"].get("e4:replay:2wiki", {})
        for k in ("n_calls", "n_intents", "head_cov_cost", "head_cov_calls", "cross_intent_frac",
                  "mean_tasks_per_intent", "head_mono_text_share", "gini_cost"):
            M.append(f"| {k} | {_c(ga.get(k), k)} | {_c(gb.get(k), k)} | {_c(r2.get(k), k)} |")
            S[f"e4.agent_2wiki.{k}"] = gb.get(k); S[f"e4.agent_musique.{k}"] = ga.get(k)
        S["e4.delta_head_cov_cost"] = o["delta_head_cov_cost_agent_2wiki_minus_musique"]
        M += ["", f"Δ 头部成本覆盖（2Wiki − MuSiQue agent）= {pct(o['delta_head_cov_cost_agent_2wiki_minus_musique'])}；"
                  f"使用的 τ = {o.get('tau_used')}（main_v10 标定，固定）", ""]
        if o.get("validation"):
            M += ["**检测器迁移（main_v10 的 τ* 用在 2Wiki-agent 对上）**", "",
                  "| 档 | τ* | 放行占比 | 同文本 | 精度估计 | 95% 下界 | 下界达标 |", "|---|---|---|---|---|---|---|"]
            for t, v in o["validation"].items():
                S[f"e4.val.{t}.precision"] = v.get("precision_hat"); S[f"e4.val.{t}.lo95"] = v.get("precision_lo95")
                M.append(f"| {t} | {fmt(v.get('tau_star'),4)} | {pct(v.get('pass_frac_of_agent_pairs'))} | "
                         f"{pct(v.get('same_text_share'))} | {fmt(v.get('precision_hat'))} | "
                         f"{fmt(v.get('precision_lo95'))} | {v.get('meets_target_lower')} |")
        M += ["", "事先写定的机械判读：" + "；".join(f"{k} = {v}" for k, v in o["mechanical_rules"].items()), ""]

    man = rj(a.p3 / "manifest.json") if (a.p3 / "manifest.json").exists() else {}
    M += ["## 溯源", "", "```", json.dumps({k: man.get(k) for k in ("models", "datasets", "vllm", "torch",
                                                                   "gpus", "scripts")},
                                           ensure_ascii=False, indent=2, default=_jdef)[:4000], "```", ""]
    (a.p3 / "SUMMARY.md").write_text("\n".join(M), encoding="utf-8")
    wj(a.p3 / "summary.json", S)
    LOG.info(f"-> {a.p3/'SUMMARY.md'}  /  {a.p3/'summary.json'}（{len(S)} 个键）")
    return f"{len(S)} 个汇总键"


# ==============================================================================
#  all
# ==============================================================================

def cmd_all(a):
    skip = {x.strip() for x in (a.skip or "").split(",") if x.strip()}
    LOG.info("计划（两张 H100 串行，粗估）：download 0.3–1 h | audit 0.2 h | E5a 闸门 0.3 h + 全量 8–10 h"
             " | E4 采集 3–5 h + 下游 0.7 h + 验证 0.3 h | E5b 思考臂 ~6 h + 不思考臂 ~4 h"
             " | 合计约 23–27 h" if not a.smoke else "冒烟：约 1–1.5 h（主要是模型加载与 E4 下游）")
    plan = [("download", cmd_download), ("audit", cmd_audit), ("e5a", cmd_e5a),
            ("e4-collect", cmd_e4_collect), ("e4-merge", cmd_e4_merge),
            ("e4-downstream", cmd_e4_downstream), ("e4-validate", cmd_e4_validate),
            ("e5b", cmd_e5b), ("e5c-export", cmd_e5c_export)]
    failed = []
    for name, fn in plan:
        if name in skip:
            LOG.info(f"[{name}] --skip，跳过"); continue
        try:
            run_step(a, name, fn)
        except SystemExit:
            raise
        except BaseException as e:
            failed.append(name)
            LOG.error(f"[{name}] 失败：{type(e).__name__}: {e}")
            if name in ("download", "audit") or (name == "e5a" and "闸门" in str(e)):
                LOG.error("这一步是后续的前提，停止。修好后重跑 `all` 会从这里接着跑。")
                break
            if name.startswith("e4-") and name != "e4-report":
                skip |= {"e4-merge", "e4-downstream", "e4-validate"} - {name}
    for name, fn in (("e5-report", cmd_e5_report), ("e4-report", cmd_e4_report),
                     ("report", cmd_report)):
        if name in skip:
            continue
        setup_log(a.p3, name)
        try:
            LOG.info(f"[{name}] {fn(a)}")
        except Exception as e:
            LOG.warning(f"[{name}] 没生成：{type(e).__name__}: {e}（缺上游结果时正常）")
    setup_log(a.p3, "all")
    return f"失败 {failed}" if failed else "全部完成"


# ==============================================================================
#  CLI
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="E4 / E5 补实验",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "audit", "e5a", "e5b", "e5c-export", "e5c-eval",
                                    "e5-gate-diag", "e5-langcheck", "e5-lenient", "e5-selfcov",
                                    "e5c-overlap",
                                    "e5-report", "e4-collect", "e4-merge", "e4-downstream",
                                    "e4-validate", "e4-report", "report", "all"])
    ap.add_argument("--config", default=os.environ.get(
        "MATVIEW_CONFIG", "/mnt/data/user_workspace/zhangzizhao/matview/config.json"))
    ap.add_argument("--phase1", default="phase1_v13.py")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--main-run", default="main_v10")
    ap.add_argument("--think-run", default="main_v12_think")
    ap.add_argument("--k3-run", default="main_v11_k3")
    ap.add_argument("--e4-run", default="e4_agent2wiki")
    ap.add_argument("--python", default=sys.executable, help="跑 phase1 子进程用的解释器")
    ap.add_argument("--gpus", default=None, help="如 0,1；默认沿用 main_v10 的配置")
    ap.add_argument("--xfam", default="llama33-70b-awq", choices=sorted(XFAM_JUDGES))
    ap.add_argument("--xfam-prompt", default="main", choices=["main", "en"],
                    help="跨模型族判官用 main 的原 prompt，或其英文直译（语言诊断后决定）")
    ap.add_argument("--redo", action="store_true", help="重做已完成的步")
    ap.add_argument("--smoke", action="store_true", help="全流程小规模冒烟，输出隔离到 runs/phase3_smoke 与 smoke_* run")
    ap.add_argument("--skip", default="", help="all 时跳过的步，逗号分隔")
    ap.add_argument("--no-servers", action="store_true", help="复用外部已起的 vLLM（调试用）")
    ap.add_argument("--no-calib", action="store_true", help="重判后不跑 s8")
    ap.add_argument("--fetch-missing", action="store_true", help="download 时顺带补下缺失的已有模型")
    # 审计
    ap.add_argument("--fp-n", type=int, default=160, help="指纹实验的对数（ij 方向）")
    ap.add_argument("--no-fingerprint", action="store_true")
    ap.add_argument("--prompt-override", default=None, help="跳过审计结论，直接指定 prompt 变体 id")
    ap.add_argument("--accept-static", action="store_true", help="审计只有静态推断时也接受")
    # E5
    ap.add_argument("--limit", type=int, default=None, help="只重判分层抽的这么多对（调试）")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--ignore-gate", action="store_true")
    ap.add_argument("--gate-n", type=int, default=200)
    ap.add_argument("--gate-max-parse-fail", type=float, default=0.03)
    ap.add_argument("--gate-max-trunc", type=float, default=0.01)
    ap.add_argument("--gate-min-acc", type=float, default=0.75)
    ap.add_argument("--e5b-cells", default="think_mainprompt,nothink_base")
    ap.add_argument("--force-e5b", action="store_true")
    ap.add_argument("--human-per-band", type=int, default=40)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--adjudicated", default=None)
    ap.add_argument("--overlap-n", type=int, default=100, help="e5c-overlap：重叠子集总题数")
    # E4
    ap.add_argument("--e4-tasks", type=int, default=2000)
    ap.add_argument("--val-per-bin", type=int, default=200, help="cos≥0.80 各层抽判对数")
    ap.add_argument("--val-per-bin-low", type=int, default=30, help="cos<0.80 各层抽判对数")
    a = ap.parse_args()

    a.runs_root = Path(a.runs_root)
    if not a.runs_root.is_dir():
        sys.exit(f"找不到 {a.runs_root.resolve()}：请在 projB 目录（runs/ 所在处）下运行")
    a.main_run = a.runs_root / a.main_run
    if not (a.main_run / "stages" / "s6" / "l1.jsonl").exists():
        sys.exit(f"{a.main_run} 不完整（缺 s6/l1.jsonl）")
    a.tag = "smoke_" if a.smoke else ""
    a.p3 = a.runs_root / ("phase3_smoke" if a.smoke else "phase3")
    a.p3.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        a.limit = a.limit or 60
        a.fp_n = min(a.fp_n, 30)
        a.gate_n = min(a.gate_n, 20)
        a.e4_tasks = min(a.e4_tasks, 40)     # 8 题太少，agent×agent 候选对可能为 0
        a.val_per_bin = min(a.val_per_bin, 4)
        a.val_per_bin_low = min(a.val_per_bin_low, 2)
        a.human_per_band = min(a.human_per_band, 3)
        a.ignore_gate = True          # 冒烟样本太小，闸门只记录不拦
    if a.config and not Path(a.config).exists():
        print(f"[warn] {a.config} 不存在，只用 main_v10/config_effective.json")
        a.config = None
    setup_log(a.p3, a.cmd)
    fn = {"download": cmd_download, "audit": cmd_audit, "e5a": cmd_e5a, "e5b": cmd_e5b,
          "e5c-export": cmd_e5c_export, "e5c-eval": cmd_e5c_eval, "e5-report": cmd_e5_report,
          "e5-gate-diag": cmd_e5_gate_diag, "e5-langcheck": cmd_e5_langcheck,
          "e5-lenient": cmd_e5_lenient, "e5-selfcov": cmd_e5_selfcov,
          "e5c-overlap": cmd_e5c_overlap,
          "e4-collect": cmd_e4_collect, "e4-merge": cmd_e4_merge,
          "e4-downstream": cmd_e4_downstream, "e4-validate": cmd_e4_validate,
          "e4-report": cmd_e4_report, "report": cmd_report, "all": cmd_all}[a.cmd]
    if a.cmd in ("all", "e5c-eval", "e5-report", "e4-report", "report", "e5-gate-diag",
                 "e5-langcheck", "e5-lenient", "e5-selfcov", "e5c-overlap") or a.gate_only:
        # 这几个本身就是汇总/驱动，不进 state 的跳过逻辑
        r = fn(a)
        LOG.info(str(r))
        return 0
    run_step(a, a.cmd, fn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
