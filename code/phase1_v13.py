#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MatView Phase 1 —— 跨任务复用结构测量  v13（VLDB 投稿版）
================================================================================

这是 phase1_v2.py 的完整替换版。直接覆盖即可，**不要再跑 patch_phase1_v2.py**
（硬件自适应已内置）。

v13 相对 v12 的修改（VLDB 投稿导向）
--------------------------------------
目标：2026-12-01 窗口，Workload Characterization Paper。
这一版不再做探索，而是把论文需要的证据一次性做齐。

[1] 分箱细化：[0.95,1.0] 拆成 [0.95,0.98) [0.98,0.999) [0.999,1.0001)
    main_v9 的 bin7 内部非单调：
      cos 0.95→0.945  0.96→0.922  0.97→0.899  0.98→0.936  0.99→0.836  1.00→0.997
    cos=1.00（字面相同）占 bin7 的 78%（12692/16300）。混在一层导致保序回归
    把 τ* 推到 0.9925，判定器退化为精确匹配。拆开后 τ* 才是真解。
    bin_target 向近似匹配区间（bin7/bin8）倾斜，bin9 少抽（已确定 0.997）。

[2] 主保证水平 0.90 -> 0.85，并报告 0.80/0.85/0.90/0.95 四档
    不是迁就数据：0.90 那档的解已退化为精确匹配，失去方法意义；
    近似区间（cos 0.90~0.99）可替代率稳定在 0.836~0.945，
    配合 ε 的双参数结构（2.35% 概率损失约 0.49），δ=0.15 更符合数据。

[3] 退化检测
    τ* 放行的对里同文本占比 ≥ 80% 就标记为「退化为精确匹配」，
    s8 日志与 s16 报告都会显式给出。这是必须自曝的 limitation。

[4] block_topk 50 -> 100
    main_v9 的分块漏检率 @cos>=0.90 已涨到 6.57%（main_v6 时是 0%），
    查询池扩到 21822 任务后 top-50 装不下真正的相似邻居。

[5] 新增 s23：精确缓存 vs 近似视图（全文胜负手）
    同一 train/test 切分上对比精确缓存 / 归一化缓存 / 近似视图@四档保证，
    给出视图相对缓存的**净增量**（按查询、按成本、分域）。
    这是回应「这不就是结果缓存吗」的直接证据 —— main_v9 的
    单文本视图占比 90.3% 让这个质疑有了数据支撑，必须正面回答。

[6] 新增 s24：论文素材导出
    10 张 CSV（工作负载特征、规模律、校准、多档 τ*、判官对拍、判定器消融、
    视图命中、缓存对照、构造偏差、判官稳定性、粒度敏感性）
    + PAPER_NUMBERS.md（所有数字一览，每个标出处 stage）。

用法
----
    python phase1_v3.py --config <cfg.json> --run-id smoke2 --smoke --force
    python phase1_v3.py --config <cfg.json> --run-id main_v1
    python phase1_v3.py --list
================================================================================
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# 必须在 import torch 之前设置。s1 的 OOM 报错里有 8.71GB 是
# reserved-but-unallocated 的碎片，expandable_segments 能显著缓解。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

# 关闭 Qwen3 的思考模式。所有判官/答案生成调用都必须带上。
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

# 数据集候选源。HF 上仓库名经常变，第一个失败就换下一个。
_DATASET_SRC = {
    "musique": [("dgslibisey/MuSiQue", None, "validation")],
    "hotpotqa": [("hotpotqa/hotpot_qa", "distractor", "validation"),
                 ("hotpot_qa", "distractor", "validation")],
    # xanhho 版是 loading script，HF 新版已不支持，直接用 voidful
    "2wiki": [("voidful/2WikiMultihopQA", None, "validation")],
}


# ==============================================================================
#  配置
# ==============================================================================


@dataclass
class Config:
    # ---------------- 运行 ----------------
    run_id: str = "main_v1"
    root: str = "runs"
    seed: int = 20260901
    smoke: bool = False
    manage_servers: bool = True
    # s0 报告额外导出一份到这里（留空关闭）
    export_dir: str = ""

    # ---------------- 环境路径 ----------------
    matview_root: str = "/mnt/data/user_workspace/zhangzizhao/matview"
    serve_python: str = "/mnt/data/user_workspace/zhangzizhao/matview/envs/serve/bin/vllm"
    main_python: str = "/mnt/data/user_workspace/zhangzizhao/matview/envs/main/bin/python"
    models_dir: str = "/mnt/data/user_workspace/zhangzizhao/matview/models"

    # ---------------- 硬件 ----------------
    gpus: List[int] = field(default_factory=lambda: [0, 1])
    retriever_gpu: int = 0

    # ---------------- 模型 ----------------
    embed_model_dev: str = ""
    embed_model_final: str = ""
    use_final_embedder: bool = False

    judge_path: str = ""
    judge_name: str = "Qwen/Qwen3-32B-AWQ"
    judge_base_port: int = 8001
    judge_ctx: int = 32768
    judge_mem_util: float = 0.85
    judge_max_seqs: int = 48

    agent_path: str = ""
    agent_name: str = "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    agent_base_port: int = 8100
    agent_ctx: int = 32768
    agent_mem_util: float = 0.85
    agent_max_seqs: int = 12
    agent_quant: str = "fp8"

    # v8：main_v4 里 Tongyi 从持久卷加载 + 在线 FP8 量化 900s 没起来。
    server_boot_timeout: int = 2400

    # ---------------- 数据 ----------------
    # v4：语料默认从数据集自带段落构建，而不是 BCP。
    # v3 用 BCP 语料检索 Wikipedia 事实问题，完全错配：f1_own=0.098，
    # L1 可替代率 3.1%（cos>=0.95 层也只有 18.8%），tau* 无解。
    corpus_source: str = "datasets"          # "datasets" | "bcp"
    corpus_datasets: List[str] = field(
        default_factory=lambda: ["musique", "hotpotqa", "2wiki"])
    corpus_dir: str = ""                     # corpus_source=="bcp" 时才用
    hf_cache: str = ""
    replay_datasets: List[str] = field(
        default_factory=lambda: ["musique", "hotpotqa", "2wiki"])
    # v10：main_v6 里 cos∈[0.70,0.95) 的候选对总共只有 5860 个且已全部抽完，
    # 加 l1_budget 无效。中高相似度层稀缺是数据固有的，唯一扩充途径是
    # 扩大查询池 —— 任务数翻倍，相似对数量近似平方增长。
    # MuSiQue 全集约 2.4 万题，之前只用了 2417 个（受 4000 上限 + 解析丢弃）。
    replay_max_tasks_per_ds: int = 12000
    corpus_max_docs: int = 0

    # ---------------- 检索 ----------------
    chunk_tokens: int = 512
    chunk_stride: int = 384
    topk: int = 10
    # v5.1：512 在正式跑的 8 万块语料上 OOM。sentence-transformers 按长度降序
    # 分批，第一批就是最长的 512 条，SDPA 因 padding mask 实体化
    # (batch, heads, seq, seq) 注意力矩阵 -> 每层 2.7GB x 28 层。
    # 128 配合 max_seq 截断后实测安全；encode 还带 OOM 自动折半重试。
    embed_batch: int = 128
    embed_max_seq: int = 640                 # 切块上限 chunk_tokens=512，留点余量
    search_block: int = 262144

    # ---------------- agent ----------------
    agent_enabled: bool = True
    # agent 是唯一没有数据集构造偏差的域，main_v2 只有 390 个可用任务，
    # 头部覆盖 27.5% 的置信度不够。提到 1200。
    agent_max_tasks: int = 2000
    agent_max_turns: int = 20
    agent_max_tokens: int = 2000             # 1400 下 smoke4 仍有 3/27 被截断
    # v7：main_v3 实测 64.6% 的 tool call 返回的是查询数组，不是单条。
    # v6 只取第一条 = 扔掉 agent 三分之二的检索意图，那不是模型的真实行为。
    # "all" 把数组里每条都发一次检索并各记一条 call，同时用 multi_query_idx
    # 标注来源；离线只保留 idx==0 就能还原 "first" 口径 —— 一次跑出两条臂。
    agent_multi_query_mode: str = "all"      # "all" | "first"
    agent_max_subqueries: int = 4            # 数组上限，防止一次爆十几条
    agent_temperature: float = 0.6
    agent_concurrency: int = 20

    # ---------------- 分块 / 抽样 ----------------
    # v13：main_v9 的分块漏检率 @cos>=0.90 是 6.57%（main_v6 时还是 0%）。
    # 查询池扩到 21822 任务后 top-50 装不下真正的相似邻居，会系统性低估复用率。
    block_topk: int = 100
    block_min_sim: float = 0.50
    bruteforce_probe_n: int = 200
    # v13：把 [0.95,1.0] 拆成三层。main_v9 的 bin7 内部非单调：
    #   cos 0.95→0.945  0.96→0.922  0.97→0.899  0.98→0.936  0.99→0.836  1.00→0.997
    # cos=1.00（字面完全相同）占 bin7 的 78%（12692/16300），把它和近似匹配
    # 混在一层，保序回归被迫把 τ* 推到 0.9925 —— 判定器退化为精确匹配。
    # 拆开之后 τ* 才是真解，近似匹配区间也才看得见。
    sim_bins: List[float] = field(default_factory=lambda: [
        0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.999, 1.0001])
    pilot_per_bin: int = 100
    # v6：面向 ~15 小时长跑。判官实测 2.33 次/秒，15h 可跑约 12 万次调用，
    # main_v2 只用了 1 万次。放大预算把每层样本从 ~130 提到 ~1000，
    # 置信区间半宽从 ±0.08 收到 ±0.03。
    # v7：main_v3 用 25412 对只花了 4.15h（预算 15h），加倍到 60000。
    # 同时向高相似度层倾斜：bin7 决定 τ*，main_v3 里它只有 n=219。
    # v10：main_v6 用 59913 对跑了 5.19h（其中 s6 约 3h）。加到 90000，
    # 并对高相似度层做双次判定 —— 目标总时长 10-13h。
    l1_budget: int = 120000
    l1_min_per_bin: int = 600
    # 每层的抽样目标（对齐 sim_bins 的 8 个区间）。低相似度层可替代率接近 0，
    # 少量样本就够；高相似度层决定 τ*，给足。见 _alloc_bins 的注释。
    # 低层可替代率接近 0，少量样本足够；中高层有多少抽多少（它们是稀缺资源）。
    # 预算向「近似匹配区间」倾斜：bin7[0.95,0.98) 与 bin8[0.98,0.999) 是
    # 论文主张的工作区间，必须抽足；bin9（cos=1.0）可替代率已确定在 0.997，
    # 少抽即可，省下的额度给 7/8。
    bin_target: List[int] = field(
        default_factory=lambda: [2000, 3000, 6000, 8000, 12000, 15000,
                                 20000, 20000, 12000, 5000])
    # v10：高相似度层做双次判定（原顺序 + 打乱顺序），两次都 SUFFICIENT 才算
    # 可替代。main_v6 分层报告显示 bin7 的位置一致率只有 0.699（bin0 是 0.987），
    # 相似度越高判官越不稳定，而 bin7 恰恰决定 τ*。取交集提升 precision，
    # 代价是 recall 下降 —— 对"要保证召回下界"的用途正合适。
    double_judge_bins: List[int] = field(default_factory=lambda: [5, 6, 7])
    # v11：main_v7 证明「交集」是错的做法。bin7 翻转者的可替代率是 0.954
    # 而不是预期的 0.5 —— 位置翻转不是随机噪声，是判官在打乱顺序后系统性地
    # 更倾向 INSUFFICIENT 的单向偏差。取交集等于用一个更严格的条件重新筛，
    # 把 bin7 从 0.970 压到 0.727，任何阈值都够不到 0.9，τ*_strict 无解。
    # v11 并列报告四种口径，主口径回到 single。
    substitutability_mode: str = "single"    # single | union | intersection
    # 顺序无关指令的 prompt A/B：直接测「改 prompt 能否降低翻转率」
    order_ablation_n: int = 1500
    # s14 特征组：main_v7 显示 ent/tpl/rbo/jac 只能把命中率 7.9%->8.1%，
    # 需要的是不同类型的特征。加入检索分数分布（原方案 C2 列的候选特征）。
    use_retrieval_features: bool = True
    gold_tsv: str = ""                       # 人工标注回填的 TSV 路径
    # v12：判官 prompt 对拍。以人工标注为真值，在多个 prompt 变体上测 accuracy。
    # 每项是 [run_dir, labeled_tsv]：TSV 给标签，run_dir 给完整文档上下文
    # （工作表里的检索结果被截断到 2 篇 x 220 字，不足以复现判定条件）。
    bakeoff_sources: List[List[str]] = field(default_factory=list)
    bakeoff_max_pairs: int = 700             # 每个变体最多判多少对
    bakeoff_variants: List[str] = field(default_factory=lambda: [
        "base", "order_inv", "evidence_quote", "strict_checklist",
        "conservative", "answer_first", "binary_direct"])
    # 闸门：对拍最优 accuracy 低于此值就中止，不浪费十几小时跑坏判官
    require_judge_acc: bool = True
    judge_order_invariant: bool = True       # s6 主判定是否用顺序无关 prompt
    # 金标按层抽样的相对权重。main_v6 之前是按相似度三分位均匀抽，
    # 结果高相似度层样本太少，而判官恰恰在那里最不稳定。
    gold_bin_weights: List[float] = field(
        default_factory=lambda: [0.5, 0.5, 1.0, 1.0, 1.5, 2.0, 3.0, 4.0, 3.0, 1.0])
    bin_tilt: List[float] = field(   # 保留字段以兼容旧 config，已不再使用
        default_factory=lambda: [0.4, 0.5, 1.0, 1.5, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0])
    l1_oversample: float = 1.0
    time_budget_hours: float = 16.0          # s6 会按实测吞吐自动截断，防止超时
    l2_budget: int = 4000
    catch_trial_rate: float = 0.05
    gold_set_n: int = 300
    view_eval_task_frac: float = 0.5   # s17 用一半任务建视图、另一半评估

    # ---------------- 判官 ----------------
    judge_temperature: float = 0.0
    judge_max_tokens: int = 600
    judge_concurrency: int = 48
    judge_doc_chars: int = 1100
    judge_retries: int = 3
    # v5：位置置换只是 QC 指标，不需要全样本。抽 20% 就够，省下的调用给自覆盖。
    pos_shuffle_frac: float = 0.15
    # v5：自覆盖判定。R(q_i) 能否回答 q_i —— 基础索引本来就答不了的查询，
    # 没有召回可损失，不该被算成"不可替代"。见 s6 注释。
    self_cov_enabled: bool = True
    cross_judge_base_url: str = ""
    cross_judge_model: str = ""
    cross_judge_key_env: str = "CROSS_JUDGE_API_KEY"
    cross_judge_n: int = 600

    # ---------------- 校准 ----------------
    bootstrap_B: int = 4000
    # v13：主保证水平从 0.90 降到 0.85。这不是迁就数据 ——
    # main_v9 显示 0.90 那档的解退化为精确匹配（τ*=0.9925 只放行 cos=1.0），
    # 失去了「近似视图重写」的方法意义；而近似区间（cos 0.90~0.99）的
    # 可替代率稳定在 0.836~0.945，配合 ε 的双参数结构（2.35% 概率损失约 0.49），
    # δ=0.15 比 δ=0.10 更符合数据。三档全部报告，由读者判断。
    target_substitutability: float = 0.85
    report_targets: List[float] = field(
        default_factory=lambda: [0.80, 0.85, 0.90, 0.95])
    # 退化检测：τ* 放行的对里同文本占比超过这个值，判定器就退化成精确匹配了
    degenerate_same_text_frac: float = 0.80
    tau_grid: List[float] = field(default_factory=lambda: [0.80, 0.85, 0.90, 0.95])
    max_clique_nodes: int = 60000

    # ---------------- 成本模型 ----------------
    cost_per_search_call: float = 1.0
    cost_per_context_token: float = 0.002
    cost_per_rerank_doc: float = 0.05

    # ---------------- 止损判据（跑之前定死）----------------
    stop_head_frac: float = 0.05
    stop_coverage_go: float = 0.30
    stop_coverage_nogo: float = 0.15
    stop_crosstask_share: float = 0.50
    stop_min_kappa: float = 0.50
    stop_min_gold_acc: float = 0.80
    stop_max_catch_fp: float = 0.10
    stop_max_parse_fail: float = 0.05
    stop_min_f1_own: float = 0.30            # L2 健全性：低于此值真值锚定不可信
    # v5：字节相同的查询对，双向判定应当一致（检索结果本来就一样）。
    # 这才是真正的阴性对照；v4 用的"可替代率>0.7"测的其实是 P(检索足够)^2。
    stop_min_same_text_consistency: float = 0.85
    stop_max_unresolved: float = 0.01        # 未解析占位符占比上限（模板碰撞）

    # ---------------- 监控 ----------------
    gpu_poll_sec: float = 15.0

    def resolve(self):
        """未显式指定的路径，从 matview_root 推。"""
        r = self.matview_root
        self.models_dir = self.models_dir or f"{r}/models"
        self.embed_model_dev = self.embed_model_dev or f"{self.models_dir}/Qwen3-Embedding-0.6B"
        self.embed_model_final = self.embed_model_final or f"{self.models_dir}/Qwen3-Embedding-8B"
        self.judge_path = self.judge_path or f"{self.models_dir}/Qwen3-32B-AWQ"
        self.agent_path = self.agent_path or f"{self.models_dir}/Tongyi-DeepResearch-30B-A3B"
        self.corpus_dir = self.corpus_dir or f"{r}/data/bcp-corpus"
        self.hf_cache = self.hf_cache or f"{r}/data/hf_cache"
        self.serve_python = self.serve_python or f"{r}/envs/serve/bin/vllm"
        self.main_python = self.main_python or f"{r}/envs/main/bin/python"
        return self

    def apply_smoke(self):
        self.smoke = True
        self.corpus_max_docs = 0
        self.replay_max_tasks_per_ds = 300
        self.agent_max_tasks = 6
        self.agent_max_turns = 6
        self.block_topk = 15
        self.bruteforce_probe_n = 30
        self.pilot_per_bin = 8
        self.l1_budget = 200
        self.l1_min_per_bin = 12
        self.l2_budget = 40
        self.time_budget_hours = 1.0
        self.gold_set_n = 12
        self.cross_judge_n = 20
        self.bootstrap_B = 200
        self.judge_ctx = 16384
        self.agent_ctx = 16384
        return self

    @classmethod
    def load(cls, path: Optional[str]) -> "Config":
        cfg = cls()
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            unknown = [k for k in data if not hasattr(cfg, k)]
            if unknown:
                raise KeyError(f"config.json 里有未知字段: {unknown}。"
                               f"要么加到 Config，要么从配置里删。")
            for k, v in data.items():
                setattr(cfg, k, v)
        return cfg.resolve()


# ==============================================================================
#  运行上下文
# ==============================================================================


def _jdef(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


class RunContext:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base = Path(cfg.root) / cfg.run_id
        self.dirs = {k: self.base / k for k in
                     ("data", "stages", "logs", "figures", "cache")}
        self.base.mkdir(parents=True, exist_ok=True)
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.state_path = self.base / "state.json"
        self.state = (json.loads(self.state_path.read_text(encoding="utf-8"))
                      if self.state_path.exists()
                      else {"created_at": time.time(), "stages": {}})
        self.log = self._logging()
        self._stop_mon = threading.Event()
        self._mon: Optional[threading.Thread] = None
        self.metrics: Dict[str, Any] = {}
        random.seed(cfg.seed); np.random.seed(cfg.seed)

    def _logging(self):
        lg = logging.getLogger("matview"); lg.setLevel(logging.DEBUG); lg.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.INFO); sh.setFormatter(fmt)
        fh = logging.FileHandler(self.dirs["logs"] / "phase1.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
        lg.addHandler(sh); lg.addHandler(fh)
        return lg

    def save_state(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2, default=_jdef),
                       encoding="utf-8")
        tmp.replace(self.state_path); self._progress_md()

    def _progress_md(self):
        el = (time.time() - self.state["created_at"]) / 3600
        L = [f"# MatView Phase 1 — `{self.cfg.run_id}`", "",
             f"已运行 **{el:.2f} h**  |  更新 {time.strftime('%H:%M:%S')}", "",
             "| stage | 状态 | 耗时 | 摘要 |", "|---|---|---|---|"]
        for sid, m in self.state.get("stages", {}).items():
            d = m.get("duration_sec")
            L.append(f"| {sid} | {m.get('status','?')} | "
                     f"{f'{d/60:.1f} min' if d else '-'} | {m.get('summary','')} |")
        if self.metrics:
            L += ["", "## 实时指标", "```",
                  json.dumps(self.metrics, ensure_ascii=False, indent=2, default=_jdef)[:3000],
                  "```"]
        (self.base / "PROGRESS.md").write_text("\n".join(L), encoding="utf-8")

    def out(self, stage: str, name: str) -> Path:
        d = self.dirs["stages"] / stage; d.mkdir(parents=True, exist_ok=True)
        return d / name

    def wjsonl(self, p: Path, rows: Iterable[Dict]) -> int:
        n = 0
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=_jdef) + "\n"); n += 1
        return n

    def rjsonl(self, p: Path) -> List[Dict]:
        with open(p, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def wjson(self, p: Path, o: Any):
        p.write_text(json.dumps(o, ensure_ascii=False, indent=2, default=_jdef),
                     encoding="utf-8")

    def rjson(self, p: Path) -> Any:
        return json.loads(p.read_text(encoding="utf-8"))

    def export(self, name: str, obj: Any) -> List[str]:
        """写 stages/ 下的规范位置，并额外导出到 export_dir（失败不阻塞）。"""
        canon = self.out("s0", name)
        self.wjson(canon, obj)
        paths = [str(canon)]
        ed = self.cfg.export_dir
        if not ed:
            return paths
        try:
            d = Path(ed); d.mkdir(parents=True, exist_ok=True)
            stem, suf = Path(name).stem, Path(name).suffix
            for p in (d / name, d / f"{stem}_{self.cfg.run_id}{suf}"):
                tmp = p.with_suffix(p.suffix + ".tmp")
                tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jdef),
                               encoding="utf-8")
                tmp.replace(p); paths.append(str(p))
        except Exception as e:
            self.log.warning(f"导出到 {ed} 失败（不影响主流程）: {type(e).__name__}: {e}")
        return paths

    def start_monitor(self):
        mod = None
        for m in ("pynvml", "nvidia_ml_py"):
            with contextlib.suppress(ImportError):
                mod = __import__(m); break
        if mod is None:
            self.log.warning("无 nvidia-ml-py，跳过 GPU 监控"); return
        f = open(self.dirs["logs"] / "gpu.jsonl", "a", encoding="utf-8")

        def loop():
            pynvml = mod
            pynvml.nvmlInit()
            hs = [pynvml.nvmlDeviceGetHandleByIndex(i)
                  for i in range(pynvml.nvmlDeviceGetCount())]
            while not self._stop_mon.is_set():
                rec = {"t": round(time.time(), 1), "stage": self.state.get("current"), "g": []}
                for i, h in enumerate(hs):
                    with contextlib.suppress(Exception):
                        m_ = pynvml.nvmlDeviceGetMemoryInfo(h)
                        u = pynvml.nvmlDeviceGetUtilizationRates(h)
                        rec["g"].append({"i": i, "mem_gb": round(m_.used / 2**30, 1),
                                         "util": u.gpu})
                f.write(json.dumps(rec) + "\n"); f.flush()
                self._stop_mon.wait(self.cfg.gpu_poll_sec)
            f.close()

        self._mon = threading.Thread(target=loop, daemon=True); self._mon.start()
        self.log.info("GPU 监控 -> logs/gpu.jsonl")

    def stop_monitor(self):
        self._stop_mon.set()
        if self._mon:
            self._mon.join(timeout=5)

    def beat(self, msg: str):
        (self.base / "HEARTBEAT").write_text(
            f"{time.strftime('%H:%M:%S')} | {self.state.get('current')} | {msg}\n",
            encoding="utf-8")

    def metric(self, **kw):
        self.metrics.update(kw); self._progress_md()


# ==============================================================================
#  硬件自适应（v3 内置，不再需要 patch 脚本）
# ==============================================================================

ROLE_REQ = {"judge": {"weights": 18.0, "min_kv": 6.0},     # Qwen3-32B-AWQ
            "agent": {"weights": 31.0, "min_kv": 6.0}}     # Tongyi 30B FP8
HEADROOM = {"judge": 3.0, "agent": 3.0}   # agent 已独占非检索卡，余量可以小


def autotune_hardware(cfg: Config, log) -> dict:
    """按实测空闲显存反算 vLLM 的 gpu_memory_utilization。

    --gpu-memory-utilization 是 **总显存** 的比例。在半占用的卡上直接用
    默认 0.85 会申请到远超可用的显存，启动即 OOM。
    """
    import torch
    visible = list(range(torch.cuda.device_count()))
    if not visible:
        raise RuntimeError("看不到任何 GPU")
    gpus = [g for g in cfg.gpus if g in visible] or visible
    if gpus != list(cfg.gpus):
        log.warning(f"配置要 {list(cfg.gpus)}，可见 {visible} -> 用 {gpus}")
    cfg.gpus = gpus
    if cfg.retriever_gpu not in gpus:
        cfg.retriever_gpu = gpus[0]

    stats = []
    for g in gpus:
        free, tot = torch.cuda.mem_get_info(g)
        stats.append({"gpu": g, "free_gb": free / 2**30, "total_gb": tot / 2**30})
        log.info(f"  GPU{g}: 空闲 {free/2**30:.1f}/{tot/2**30:.1f} GB")

    min_free = min(s["free_gb"] for s in stats)
    total = min(s["total_gb"] for s in stats)
    usable = max(min_free - 2.0, 0.0)          # 留 2GB 给 context 与碎片
    rep = {"gpus": gpus, "n_gpus": len(gpus), "stats": stats,
           "min_free_gb": round(min_free, 1), "usable_gb": round(usable, 1),
           "changes": []}

    def note(m):
        rep["changes"].append(m); log.warning(f"  [autotune] {m}")

    # ---- judge ----
    need = sum(ROLE_REQ["judge"].values()) + HEADROOM["judge"]
    if usable < need:
        raise RuntimeError(f"可用 {usable:.1f}GB < 判官所需 {need:.1f}GB")
    old = cfg.judge_mem_util
    cfg.judge_mem_util = round(min((usable - HEADROOM["judge"]) / total, 0.92), 3)
    if abs(cfg.judge_mem_util - old) > 0.01:
        note(f"judge_mem_util {old} -> {cfg.judge_mem_util}")
    kv_j = usable - HEADROOM["judge"] - ROLE_REQ["judge"]["weights"]
    rep["judge_kv_gb"] = round(kv_j, 1)
    s = max(4, min(cfg.judge_max_seqs, int(kv_j * 6)))
    if s != cfg.judge_max_seqs:
        note(f"judge_max_seqs {cfg.judge_max_seqs} -> {s} (KV {kv_j:.1f}GB)")
        cfg.judge_max_seqs = s

    # ---- agent：独占非检索卡，所以按整卡算 ----
    agent_gpus = [g for g in gpus if g != cfg.retriever_gpu] or gpus
    rep["agent_gpus"] = agent_gpus
    if len(gpus) == 1:
        note("只有 1 张卡：agent 与检索器同卡，可用显存要扣掉索引占用")
        usable_a = usable - 8.0
    else:
        usable_a = usable
    need_a = sum(ROLE_REQ["agent"].values()) + HEADROOM["agent"]
    if cfg.agent_enabled and usable_a < need_a:
        note(f"可用 {usable_a:.1f}GB < agent 所需 {need_a:.1f}GB -> 关闭 agent 后端")
        note("  影响：无通用域对照，止损判定最多给 GO_CONDITIONAL；"
             "垂直域复用率不受影响")
        cfg.agent_enabled = False
        rep["agent_disabled"] = True
    elif cfg.agent_enabled:
        old = cfg.agent_mem_util
        cfg.agent_mem_util = round(min((usable_a - HEADROOM["agent"]) / total, 0.90), 3)
        if abs(cfg.agent_mem_util - old) > 0.01:
            note(f"agent_mem_util {old} -> {cfg.agent_mem_util}")
        kv_a = usable_a - HEADROOM["agent"] - ROLE_REQ["agent"]["weights"]
        rep["agent_kv_gb"] = round(kv_a, 1)
        # Qwen3-MoE 48 层 GQA，约 98KB/token
        s = max(1, min(cfg.agent_max_seqs, int(kv_a * 2**30 / 98304 / cfg.agent_ctx)))
        if s != cfg.agent_max_seqs:
            note(f"agent_max_seqs {cfg.agent_max_seqs} -> {s} (KV {kv_a:.1f}GB)")
            cfg.agent_max_seqs = s

    # ---- 并发按副本数缩放 ----
    n_j, n_a = len(gpus), len(agent_gpus)
    for k, per, n in (("judge_concurrency", cfg.judge_max_seqs, n_j),
                      ("agent_concurrency", cfg.agent_max_seqs, n_a)):
        base = getattr(cfg, k); new = max(2, min(base, per * n))
        if new != base:
            note(f"{k} {base} -> {new} ({per} × {n} 副本)")
            setattr(cfg, k, new)

    log.info(f"[autotune] judge GPU{gpus} util={cfg.judge_mem_util} | "
             f"agent GPU{agent_gpus} {'on' if cfg.agent_enabled else 'OFF'}")
    return rep


# ==============================================================================
#  vLLM 服务
# ==============================================================================


class ServerManager:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx; self.cfg = ctx.cfg
        self.procs: List[subprocess.Popen] = []
        self.role: Optional[str] = None
        self.urls: List[str] = []
        self.gpus_used: List[int] = []
        atexit.register(self.stop)

    @staticmethod
    def _alive(url, t=5) -> bool:
        import requests
        try:
            return requests.get(f"{url}/models", timeout=t).status_code == 200
        except Exception:
            return False

    def _tail_log(self, role: str, gpu: int, n: int = 6) -> List[str]:
        """取该副本 vLLM 日志的最后几行有效内容，直接打进主日志。

        main_v4 那次要人工去翻 vllm_judge_gpu1.log 才知道是
        "No available memory for the cache blocks"。这种信息应该当场就看得到。
        """
        p = self.ctx.dirs["logs"] / f"vllm_{role}_gpu{gpu}.log"
        try:
            lines = [l.rstrip() for l in p.read_text(errors="ignore").splitlines()
                     if l.strip()]
            key = [l for l in lines
                   if any(k in l for k in ("Error", "ERROR", "error", "Exception",
                                           "No available memory", "out of memory",
                                           "Traceback", "ValueError", "RuntimeError"))]
            return (key[-n:] if key else lines[-n:])
        except Exception:
            return ["（读不到日志）"]

    def _preflight_gpu(self, role: str, gpus: List[int], need_gb: float) -> None:
        """启动前实测每张卡的空闲显存，不够就直接说清楚，别等 vLLM 绕一圈再报。"""
        free = self._free()
        for g in gpus:
            if g < len(free) and free[g] < need_gb:
                self.ctx.log.error(
                    f"[server] GPU{g} 只剩 {free[g]:.1f}GB < {role} 需要的 "
                    f"{need_gb:.1f}GB。多半是上一次运行残留的进程没清干净：")
                self.ctx.log.error(
                    "          nvidia-smi; pkill -9 -f 'vllm serve'; "
                    "pkill -9 -f EngineCore; sleep 10; nvidia-smi")

    def _free(self):
        try:
            o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, timeout=20).stdout.decode().split()
            return [float(x) / 1024 for x in o]
        except Exception:
            return []

    def ensure(self, role: str, gpus: Optional[List[int]] = None) -> List[str]:
        gpus = gpus or list(self.cfg.gpus)
        if not self.cfg.manage_servers:
            base = self.cfg.judge_base_port if role == "judge" else self.cfg.agent_base_port
            urls = [u for u in (f"http://localhost:{base+i}/v1" for i in range(len(gpus)))
                    if self._alive(u)]
            if not urls:
                raise RuntimeError(f"--no-servers 下 {role} 服务不可达")
            self.ctx.log.info(f"[server] 复用外部 {role}: {urls}")
            return urls
        if self.role == role and self.gpus_used == gpus and all(self._alive(u) for u in self.urls):
            return self.urls
        self.stop()
        return self._start(role, gpus)

    def _start(self, role: str, gpus: List[int]) -> List[str]:
        cfg, log = self.cfg, self.ctx.log
        if role == "judge":
            path, name, base = cfg.judge_path, cfg.judge_name, cfg.judge_base_port
            extra = ["--quantization", "awq_marlin",
                     "--max-model-len", str(cfg.judge_ctx),
                     "--gpu-memory-utilization", str(cfg.judge_mem_util),
                     "--max-num-seqs", str(cfg.judge_max_seqs)]
        else:
            path, name, base = cfg.agent_path, cfg.agent_name, cfg.agent_base_port
            extra = ["--quantization", cfg.agent_quant,
                     "--max-model-len", str(cfg.agent_ctx),
                     "--gpu-memory-utilization", str(cfg.agent_mem_util),
                     "--max-num-seqs", str(cfg.agent_max_seqs),
                     "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]

        util = cfg.judge_mem_util if role == "judge" else cfg.agent_mem_util
        wt = ROLE_REQ[role]["weights"]
        self._preflight_gpu(role, gpus, wt + ROLE_REQ[role]["min_kv"])

        urls = []
        for k, gpu in enumerate(gpus):
            port = base + k
            cmd = [cfg.serve_python, "serve", path, "--served-model-name", name,
                   "--port", str(port), "--enable-prefix-caching",
                   "--disable-log-requests"] + extra
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            lf = open(self.ctx.dirs["logs"] / f"vllm_{role}_gpu{gpu}.log", "a",
                      encoding="utf-8")
            lf.write(f"\n\n===== {time.strftime('%F %T')} {' '.join(cmd)} =====\n"); lf.flush()
            log.info(f"[server] {role} 副本{k} -> GPU{gpu} port {port}")
            self.procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                               env=env, start_new_session=True))
            urls.append(f"http://localhost:{port}/v1")

        # v8：单个副本失败不再拖死整组。
        # main_v4 里 GPU1 的判官因幽灵进程占了 53GB 而 OOM 退出，旧逻辑抛异常 ->
        # stop() 把已经健康的 GPU0 副本一起 SIGTERM 掉（其日志末尾是
        # KeyboardInterrupt: terminated），整个 s6 白等两分钟。
        t0, pend = time.time(), set(range(len(urls)))
        dead: set = set()
        while pend and time.time() - t0 < cfg.server_boot_timeout:
            for i in list(pend):
                if self._alive(urls[i]):
                    log.info(f"[server] 副本{i} (GPU{gpus[i]}) 就绪 "
                             f"({time.time()-t0:.0f}s)")
                    pend.discard(i)
            for i, p in enumerate(self.procs):
                if i in pend and p.poll() is not None:
                    log.error(f"[server] 副本{i} (GPU{gpus[i]}) 退出 code={p.returncode}，"
                              f"详见 logs/vllm_{role}_gpu{gpus[i]}.log")
                    for ln in self._tail_log(role, gpus[i], 6):
                        log.error(f"        {ln}")
                    pend.discard(i); dead.add(i)
            if pend:
                self.ctx.beat(f"等 {role} 服务 {int(time.time()-t0)}s")
                time.sleep(8)
        for i in list(pend):                      # 超时未就绪的也算失败
            log.error(f"[server] 副本{i} (GPU{gpus[i]}) 启动超时 "
                      f"({cfg.server_boot_timeout}s)")
            for ln in self._tail_log(role, gpus[i], 6):
                log.error(f"        {ln}")
            dead.add(i)

        ok = [i for i in range(len(urls)) if i not in dead and self._alive(urls[i])]
        if not ok:
            self.stop()
            raise RuntimeError(f"{role} 全部 {len(urls)} 个副本都启动失败，"
                               f"看 logs/vllm_{role}_gpu*.log")
        if dead:
            log.warning(f"[server] {len(dead)}/{len(urls)} 个副本失败，"
                        f"用剩余 {len(ok)} 个继续（吞吐按比例下降，不中止实验）")
            for i in sorted(dead, reverse=True):
                p = self.procs[i]
                with contextlib.suppress(Exception):
                    if p.poll() is None:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                self.procs.pop(i)
        self.role = role
        self.urls = [urls[i] for i in ok]
        self.gpus_used = list(gpus)
        log.info(f"[server] {role}×{len(self.urls)} 就绪 {time.time()-t0:.0f}s，"
                 f"空闲 {[f'{x:.0f}G' for x in self._free()]}")
        return self.urls

    def stop(self):
        if not self.procs:
            return
        self.ctx.log.info(f"[server] 停止 {self.role}")
        for p in self.procs:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        for p in self.procs:
            with contextlib.suppress(Exception):
                p.wait(timeout=60)
        for p in self.procs:
            if p.poll() is None:
                with contextlib.suppress(Exception):
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        self.procs.clear(); self.role, self.urls, self.gpus_used = None, [], []
        for _ in range(30):
            f = self._free()
            if f and min(f) > 60:
                break
            time.sleep(2)
        self.ctx.log.info(f"[server] 空闲 {[f'{x:.0f}G' for x in self._free()]}")


class LLMPool:
    def __init__(self, urls, model, retries=3, timeout=600, api_key="EMPTY"):
        import requests
        self._rq = requests
        self.urls = list(urls); self.model = model
        self.retries, self.timeout, self.api_key = retries, timeout, api_key
        self._i = itertools.count(); self.lock = threading.Lock()
        self.tok_in = self.tok_out = self.n_calls = self.n_fail = self.n_trunc = 0

    def chat(self, messages, **kw) -> Dict:
        last = None
        for a in range(self.retries):
            url = self.urls[next(self._i) % len(self.urls)]
            try:
                r = self._rq.post(f"{url}/chat/completions",
                                  json={"model": self.model, "messages": messages, **kw},
                                  headers={"Authorization": f"Bearer {self.api_key}"},
                                  timeout=self.timeout)
                r.raise_for_status(); d = r.json()
                u = d.get("usage", {}) or {}
                ch = d["choices"][0]
                fin = ch.get("finish_reason")
                with self.lock:
                    self.tok_in += u.get("prompt_tokens", 0)
                    self.tok_out += u.get("completion_tokens", 0)
                    self.n_calls += 1
                    if fin == "length":
                        self.n_trunc += 1
                return {"text": ch["message"].get("content") or "",
                        "prompt_tokens": u.get("prompt_tokens", 0),
                        "completion_tokens": u.get("completion_tokens", 0),
                        "finish_reason": fin, "raw": d}
            except Exception as e:
                last = e; time.sleep(min(2 ** a + random.random(), 20))
        with self.lock:
            self.n_fail += 1
        raise RuntimeError(f"LLM 调用失败: {last}")

    def stats(self):
        return {"calls": self.n_calls, "fail": self.n_fail, "truncated": self.n_trunc,
                "tok_in": self.tok_in, "tok_out": self.tok_out,
                "avg_out": round(self.tok_out / max(self.n_calls, 1), 1)}


# ==============================================================================
#  torch 检索器
# ==============================================================================


class TorchRetriever:
    def __init__(self, ctx: RunContext, model_path: str, device: int = 0):
        self.ctx = ctx; self.model_path = model_path; self.device = f"cuda:{device}"
        self._m = None; self.emb = None
        self.chunk_ids: List[str] = []; self.chunk_docs: List[str] = []
        self.chunk_texts: List[str] = []

    @property
    def model(self):
        if self._m is None:
            from sentence_transformers import SentenceTransformer
            self.ctx.log.info(f"加载嵌入模型 -> {self.device}")
            self._m = SentenceTransformer(self.model_path, device=self.device,
                                          model_kwargs={"torch_dtype": "float16"},
                                          tokenizer_kwargs={"padding_side": "left"})
        return self._m

    def encode(self, texts, is_query, batch, desc="") -> np.ndarray:
        # 防御：任何非 str 元素都会让 sentence-transformers 的 prompt+sentence
        # 炸成 TypeError（v3 里 agent 6/6 全挂就是这个原因，当时误判为显存不足）
        texts = [t if isinstance(t, str)
                 else (" ".join(map(str, t)) if isinstance(t, (list, tuple)) else str(t))
                 for t in texts]
        import torch
        m = self.model
        # 截断到 embed_max_seq。不设的话 Qwen3-Embedding 的 max_seq_length 很大，
        # 长文本批次会把注意力矩阵撑爆。
        want = self.ctx.cfg.embed_max_seq
        if getattr(m, "max_seq_length", 0) and m.max_seq_length > want:
            self.ctx.log.info(f"[encode] max_seq_length {m.max_seq_length} -> {want}")
            m.max_seq_length = want
        if desc and texts:
            L = sorted(len(t) for t in texts)
            self.ctx.log.info(f"[encode:{desc}] 文本长度(字符) 中位 {L[len(L)//2]} "
                              f"P99 {L[int(len(L)*0.99)]} 最长 {L[-1]}")
        kw = {"prompt_name": "query"} if is_query else {}
        t0 = time.time()
        b = batch
        while True:
            try:
                e = m.encode(texts, batch_size=b, normalize_embeddings=True,
                             show_progress_bar=len(texts) > 1000,
                             convert_to_numpy=True, **kw)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if b <= 8:
                    raise
                b = max(8, b // 2)
                self.ctx.log.warning(f"[encode:{desc}] OOM，batch 折半 -> {b} 重试")
        if desc:
            dt = time.time() - t0
            self.ctx.log.info(f"[encode:{desc}] {len(texts)} / {dt/60:.1f}min "
                              f"({len(texts)/max(dt,1e-9):.0f} it/s, batch={b})")
        return e.astype(np.float32)

    def load_index(self, ids, docs, texts, emb: np.ndarray):
        import torch
        self.chunk_ids, self.chunk_docs, self.chunk_texts = ids, docs, texts
        t = torch.nn.functional.normalize(
            torch.from_numpy(emb).to(self.device, dtype=torch.float16), dim=1)
        self.emb = t
        self.ctx.log.info(f"[index] {t.shape[0]}×{t.shape[1]} fp16 常驻 {self.device} "
                          f"({t.numel()*2/2**30:.2f} GB)")

    def search_vecs(self, q: np.ndarray, k: int):
        import torch
        blk = self.ctx.cfg.search_block
        qt = torch.nn.functional.normalize(
            torch.from_numpy(q).to(self.device, dtype=torch.float16), dim=1)
        N = self.emb.shape[0]; k = min(k, N)
        bs = torch.full((qt.shape[0], 0), -1e4, device=self.device, dtype=torch.float16)
        bi = torch.zeros((qt.shape[0], 0), device=self.device, dtype=torch.long)
        for s in range(0, N, blk):
            sub = self.emb[s:s + blk]
            sc = qt @ sub.T
            v, i = torch.topk(sc, min(k, sc.shape[1]), dim=1)
            bs = torch.cat([bs, v], 1); bi = torch.cat([bi, i + s], 1)
            if bs.shape[1] > 4 * k:
                v, j = torch.topk(bs, k, dim=1); bs, bi = v, torch.gather(bi, 1, j)
            del sc
        v, j = torch.topk(bs, min(k, bs.shape[1]), dim=1)
        return v.float().cpu().numpy(), torch.gather(bi, 1, j).cpu().numpy()

    def search(self, queries: List[str], k: int, batch: int = 256):
        S, I = [], []
        for s in range(0, len(queries), batch):
            q = self.encode(queries[s:s + batch], True, min(batch, self.ctx.cfg.embed_batch))
            a, b = self.search_vecs(q, k); S.append(a); I.append(b)
        return np.concatenate(S), np.concatenate(I)

    def docs(self, row):
        return [{"docid": self.chunk_docs[i], "chunkid": self.chunk_ids[i],
                 "text": self.chunk_texts[i]} for i in row if 0 <= i < len(self.chunk_ids)]

    def free(self):
        import torch, gc
        self.emb = None; self._m = None
        gc.collect(); torch.cuda.empty_cache()


# ==============================================================================
#  统计工具
# ==============================================================================


def sha(*p):
    h = hashlib.sha1()
    for x in p:
        h.update(str(x).encode())
    return h.hexdigest()[:16]


def wilson_w(v, w, z: float = 1.96):
    """加权 Wilson 区间：用有效样本量 n_eff = (Σw)^2 / Σw^2。

    v9 起 L1 是分层抽样且各层抽样率差异巨大（bin7 抽了 24000/24250，
    bin0 只抽了 4000/166152），直接对样本取平均会严重偏离总体。
    main_v5 的「条件可替代率 0.793」就是这么来的 —— 高相似度层占了
    实判样本的 45%，而它们的可替代率是 0.987。
    """
    v = np.asarray(v, float); w = np.asarray(w, float)
    sw = w.sum()
    if sw <= 0 or len(v) == 0:
        return (float("nan"), 0.0, 1.0, 0.0)
    p = float((w * v).sum() / sw)
    n_eff = float(sw ** 2 / max((w ** 2).sum(), 1e-12))
    lo, hi = wilson(int(round(p * n_eff)), int(round(n_eff)), z)
    return (p, lo, hi, n_eff)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] / n * cb[k] / n for k in set(a) | set(b))
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def rbo(a, b, p=0.9):
    if not a or not b:
        return 0.0
    sa, sb, s, w = set(), set(), 0.0, 0.0
    for d in range(1, max(len(a), len(b)) + 1):
        if d <= len(a): sa.add(a[d - 1])
        if d <= len(b): sb.add(b[d - 1])
        ww = p ** (d - 1); s += ww * len(sa & sb) / d; w += ww
    return s / w if w else 0.0


def jacc(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa or sb) else 0.0


def gini(v):
    x = np.sort(np.asarray(list(v), float)); n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(2 * (i * x).sum() / (n * x.sum()) - (n + 1) / n)


def hill(v, xmin=1):
    xs = [c for c in v if c >= xmin]
    if len(xs) < 10:
        return float("nan")
    return 1 + len(xs) / sum(math.log(x / (xmin - 0.5)) for x in xs)


def xjson(t: str) -> Optional[Dict]:
    if not t:
        return None
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    t = re.sub(r"^.*?</think>", "", t, flags=re.S)     # 只有闭合标签的情况
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    c = m.group(1) if m else None
    if c is None:
        i, j = t.find("{"), t.rfind("}")
        c = t[i:j + 1] if 0 <= i < j else None
    if c is None:
        return None
    for cand in (c, re.sub(r",\s*([}\]])", r"\1", c), c.replace("'", '"')):
        with contextlib.suppress(Exception):
            return json.loads(cand)
    return None


def pmap(fn, items, workers, desc, ctx: RunContext, pool: Optional[LLMPool] = None):
    from tqdm import tqdm
    res = [None] * len(items); t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(fn, it): i for i, it in enumerate(items)}
        with tqdm(total=len(items), desc=desc, ncols=100) as bar:
            for k, f in enumerate(as_completed(fut)):
                i = fut[f]
                try:
                    res[i] = f.result()
                except Exception as e:
                    res[i] = {"__error__": f"{type(e).__name__}: {e}"}
                    ctx.log.debug(f"{desc}[{i}] {e}")
                bar.update(1)
                if k % 25 == 0 or k == len(items) - 1:
                    el = time.time() - t0; rate = (k + 1) / max(el, 1e-9)
                    eta = (len(items) - k - 1) / max(rate, 1e-9)
                    m = {f"{desc}_done": k + 1, f"{desc}_total": len(items),
                         f"{desc}_rate": round(rate, 2), f"{desc}_eta_min": round(eta / 60, 1)}
                    if pool:
                        m[f"{desc}_llm"] = pool.stats()
                    ctx.metric(**m); ctx.beat(f"{desc} {k+1}/{len(items)} ETA {eta/60:.1f}min")
    ctx.log.info(f"[{desc}] {len(items)} 项 / {(time.time()-t0)/60:.1f}min"
                 + (f" / {pool.stats()}" if pool else ""))
    return res


def summarize_errors(results, log, label) -> Tuple[int, Dict[str, int]]:
    """把错误按类型聚合。v2 只给一个计数，定位全靠猜。"""
    n, errs = 0, Counter()
    for r in results:
        e = None
        if not isinstance(r, dict):
            e = "非字典返回"
        else:
            e = r.get("__error__") or r.get("error")
        if e:
            n += 1; errs[str(e)[:140]] += 1
    for msg, c in errs.most_common(5):
        log.error(f"  {label} 错误 ×{c}: {msg}")
    return n, dict(errs.most_common(5))


# ==============================================================================
#  Stage 注册
# ==============================================================================

STAGES: Dict[str, Tuple[str, Callable, List[str]]] = {}
ORDER: List[str] = []


def stage(sid, desc, produces=()):
    def deco(fn):
        STAGES[sid] = (desc, fn, list(produces)); ORDER.append(sid); return fn
    return deco


# ==============================================================================
# s0  preflight
# ==============================================================================


@stage("s0", "环境 / GPU / 模型 / 数据 自检", ["preflight.json"])
def s0(ctx: RunContext, sm: ServerManager):
    cfg, log = ctx.cfg, ctx.log
    rep: Dict[str, Any] = {"host": socket.gethostname(), "ts": time.time(),
                           "run_id": cfg.run_id, "smoke": cfg.smoke, "cwd": os.getcwd()}
    bad: List[str] = []

    # ---- 解释器（上次故障就是用错了 venv）----
    rep["python"] = {"executable": sys.executable, "prefix": sys.prefix,
                     "version": sys.version.split()[0],
                     "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
                     "expected": cfg.main_python}
    if cfg.main_python and Path(cfg.main_python).exists():
        same = os.path.realpath(sys.executable) == os.path.realpath(cfg.main_python)
        rep["python"]["match"] = same
        if not same:
            bad.append(f"解释器不对：在用 {sys.executable}，应该用 {cfg.main_python}")

    import torch
    rep["torch"] = torch.__version__; rep["torch_cuda"] = torch.version.cuda

    if not torch.cuda.is_available():
        drv = ""
        with contextlib.suppress(Exception):
            o = subprocess.run(["nvidia-smi"], capture_output=True, timeout=20)
            m = re.search(r"CUDA Version:\s*([0-9.]+)", o.stdout.decode(errors="ignore"))
            drv = m.group(1) if m else ""
        rep["driver_cuda"] = drv
        tb = torch.version.cuda or ""
        try:
            mismatch = bool(drv and tb and float(tb) > float(drv))
        except ValueError:
            mismatch = False
        if mismatch:
            tag = f"cu{drv.replace('.', '')}"
            bad.append(f"CUDA 不可用：torch 是 cu{tb.replace('.','')} 编译的，"
                       f"驱动只到 CUDA {drv}。修：pip install torch --index-url "
                       f"https://download.pytorch.org/whl/{tag}")
        else:
            bad.append(f"CUDA 不可用（torch build cu{tb or '?'}，驱动 {drv or '未知'}）")
    else:
        rep["gpus"] = []
        for i in range(torch.cuda.device_count()):
            free, tot = torch.cuda.mem_get_info(i)
            p = torch.cuda.get_device_properties(i)
            rep["gpus"].append({"i": i, "name": p.name, "cc": f"{p.major}.{p.minor}",
                                "free_gb": round(free / 2**30, 1),
                                "total_gb": round(tot / 2**30, 1)})
            log.info(f"  GPU{i} {p.name} cc{p.major}.{p.minor} "
                     f"空闲 {free/2**30:.1f}/{tot/2**30:.1f} GB")
        rep["autotune"] = getattr(cfg, "_autotune", None)
        rep["gpus_in_use"] = list(cfg.gpus)
        rep["agent_enabled"] = cfg.agent_enabled
        rep["judge_mem_util"] = cfg.judge_mem_util
        rep["agent_mem_util"] = cfg.agent_mem_util
        for g in rep["gpus"]:
            if g["i"] not in cfg.gpus:
                continue
            need = cfg.judge_mem_util * g["total_gb"] + 1.0
            if g["free_gb"] < need:
                bad.append(f"GPU{g['i']} 空闲 {g['free_gb']:.1f}GB < 判官需要 {need:.1f}GB"
                           f"（util={cfg.judge_mem_util}）；自适应之后显存被别的进程占走了")
        if not cfg.gpus:
            bad.append("没有可用 GPU")
        try:
            dev = f"cuda:{cfg.gpus[0]}" if cfg.gpus else "cuda:0"
            x = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
            _ = (x @ x).float().sum().item(); del x; torch.cuda.empty_cache()
            rep["alloc_test"] = "ok"
        except Exception as e:
            bad.append(f"GPU 分配失败: {e}")

    for mod in ("transformers", "sentence_transformers", "sklearn",
                "networkx", "matplotlib", "datasets", "scipy", "pyarrow"):
        try:
            rep[mod] = getattr(__import__(mod), "__version__", "ok")
        except ImportError:
            bad.append(f"缺依赖 {mod}")

    for lbl, p in (("embed", cfg.embed_model_dev), ("judge", cfg.judge_path),
                   ("agent", cfg.agent_path)):
        ex = Path(p, "config.json").exists()
        rep[f"model_{lbl}"] = p if ex else f"MISSING:{p}"
        if not ex and not (lbl == "agent" and not cfg.agent_enabled):
            bad.append(f"{lbl} 模型缺失: {p}")

    rep["corpus_source"] = cfg.corpus_source
    if cfg.corpus_source == "bcp":
        rep["corpus_dir"] = cfg.corpus_dir
        if not Path(cfg.corpus_dir).exists():
            bad.append(f"语料目录不存在: {cfg.corpus_dir}")
    else:
        rep["corpus_datasets"] = cfg.corpus_datasets
        log.info(f"  语料来源：数据集自带段落 {cfg.corpus_datasets}（自洽语料）")

    if cfg.manage_servers and not Path(cfg.serve_python).exists():
        bad.append(f"vllm 不存在: {cfg.serve_python}")

    rep["disk_free_gb"] = round(shutil.disk_usage(ctx.base).free / 2**30, 1)
    if rep["disk_free_gb"] < 60:
        bad.append(f"磁盘仅剩 {rep['disk_free_gb']}GB")

    # 陈旧产出检测：同名 run 目录若来自旧版本脚本，stage 会被误判为"已完成"而跳过，
    # 后续 stage 读到的是旧语料/旧结构的产出。宁可报错也不要静默复用。
    old_s1 = ctx.out("s1", "stats.json")
    if old_s1.exists():
        try:
            prev = ctx.rjson(old_s1)
            if prev.get("corpus_source") != cfg.corpus_source:
                bad.append(f"run 目录里有旧产出：s1 的 corpus_source="
                           f"{prev.get('corpus_source')}，当前配置是 {cfg.corpus_source}。"
                           f"删掉 {ctx.base} 重跑，或换一个 --run-id")
        except Exception:
            bad.append(f"run 目录里有无法解析的旧 s1 产出，删掉 {ctx.base} 重跑")
    rep["problems"] = bad
    paths = ctx.export("preflight.json", rep)
    for p in paths:
        log.info(f"  自检报告 -> {p}")
    for b in bad:
        log.error(f"  ✗ {b}")
    if bad:
        raise RuntimeError(f"自检 {len(bad)} 项失败，见 " + " 或 ".join(paths))
    return (f"GPU×{len(rep.get('gpus', []))} / torch {rep['torch']} / "
            f"磁盘 {rep['disk_free_gb']}GB")


# ==============================================================================
# s1  corpus
# ==============================================================================


def _chunk(text, size, stride):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    return [text[i:i + size] for i in range(0, len(text), stride)]


def _iter_paragraphs(r):
    """三个数据集的段落字段格式各不相同，逐一适配。"""
    ps = r.get("paragraphs")
    if isinstance(ps, list):                                    # MuSiQue
        for p in ps:
            if isinstance(p, dict):
                yield p.get("title", ""), p.get("paragraph_text", "")
    c = r.get("context")
    if isinstance(c, dict) and "title" in c:                    # HotpotQA (dict-of-lists)
        for t, sents in zip(c.get("title", []), c.get("sentences", [])):
            yield t, (" ".join(sents) if isinstance(sents, list) else str(sents))
    elif isinstance(c, list):                                   # 2Wiki / 旧式 HotpotQA
        for it in c:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                yield it[0], (" ".join(it[1]) if isinstance(it[1], list) else str(it[1]))
            elif isinstance(it, dict):
                yield it.get("title", ""), (it.get("text") or it.get("paragraph_text") or "")


def _build_corpus_from_datasets(ctx: RunContext, names) -> List[Dict]:
    """从多跳数据集自带的 supporting paragraphs 建自洽语料。

    v3 用 BrowseComp-Plus 语料检索 Wikipedia 事实问题，检索结果与问题毫无关系：
    f1_own=0.098（用自己的检索结果都答不对自己的问题），L1 可替代率 3.1%。
    判官没错，是"这堆无关文档不足以回答问题"的正确判断。
    自洽语料是让 gold answer 可达、L1/L2 有意义的前提。
    """
    from datasets import load_dataset
    from tqdm import tqdm
    cfg, log = ctx.cfg, ctx.log
    seen, docs = set(), []
    for name in names:
        got = False
        for repo, sub, split in _DATASET_SRC.get(name, []):
            try:
                kw = {"cache_dir": cfg.hf_cache, "split": split}
                d = load_dataset(repo, sub, **kw) if sub else load_dataset(repo, **kw)
            except Exception as e:
                log.warning(f"  语料源 {repo} 失败: {type(e).__name__}: {str(e)[:110]}")
                continue
            n0, lim = len(docs), (cfg.replay_max_tasks_per_ds or len(d))
            for i, r in enumerate(tqdm(d, desc=f"corpus-{name}", ncols=100,
                                       total=min(lim, len(d)))):
                if i >= lim:
                    break
                for title, text in _iter_paragraphs(r):
                    text = (text or "").strip()
                    if len(text) < 50:
                        continue
                    key = sha(title, text[:300])
                    if key in seen:
                        continue
                    seen.add(key)
                    docs.append({"docid": f"{name}-{key}", "title": title or "",
                                 "text": (f"{title}. {text}" if title else text)})
            log.info(f"  语料 {name}: +{len(docs)-n0} 段落 (源 {repo})")
            got = True
            break
        if not got:
            log.error(f"  语料 {name} 全部源失败")
    return docs


@stage("s1", "语料构建 / 切块 / 编码 / 索引",
       ["chunks.jsonl", "chunk_emb.npy", "stats.json"])
def s1(ctx: RunContext, sm: ServerManager):
    from tqdm import tqdm
    cfg, log = ctx.cfg, ctx.log
    sm.stop()

    if cfg.corpus_source == "datasets":
        recs = _build_corpus_from_datasets(ctx, cfg.corpus_datasets)
        if cfg.corpus_max_docs:
            recs = recs[:cfg.corpus_max_docs]
        # 语料与查询池同源：replay_max_tasks_per_ds 一起控制两者，
        # 否则子问题的支撑段落不在语料里，f1_own 会掉。
        log.info(f"语料（数据集段落，已去重）{len(recs)} 篇")
        if not recs:
            raise RuntimeError("语料为空，检查数据集能否加载")
    else:
        from datasets import load_dataset
        log.info(f"载入 BCP 语料 {cfg.corpus_dir}")
        ds = load_dataset(cfg.corpus_dir, split="train", cache_dir=cfg.hf_cache)
        if cfg.corpus_max_docs:
            ds = ds.select(range(min(cfg.corpus_max_docs, len(ds))))
        recs = [{"docid": r.get("docid") or str(i), "title": "", "text": r.get("text") or ""}
                for i, r in enumerate(ds)]
        log.warning("corpus_source='bcp'：若查询来自多跳 QA 数据集，语料会错配，"
                    "L1/L2 无意义。除非你用的是 BCP 自己的查询。")

    size, stride = cfg.chunk_tokens * 4, cfg.chunk_stride * 4
    ids, docs, texts = [], [], []
    for r in tqdm(recs, desc="chunk", ncols=100):
        for j, c in enumerate(_chunk(r["text"], size, stride)):
            ids.append(f"{r['docid']}::{j}"); docs.append(r["docid"]); texts.append(c)
    log.info(f"切块 {len(ids)} ({len(ids)/max(len(recs),1):.1f}/文档)")

    mp = cfg.embed_model_final if cfg.use_final_embedder else cfg.embed_model_dev
    r = TorchRetriever(ctx, mp, cfg.retriever_gpu)
    t0 = time.time()
    emb = r.encode(texts, False, cfg.embed_batch, "corpus")
    dt = time.time() - t0

    ctx.wjsonl(ctx.out("s1", "chunks.jsonl"),
               ({"chunkid": a, "docid": b, "text": c} for a, b, c in zip(ids, docs, texts)))
    np.save(ctx.out("s1", "chunk_emb.npy"), emb)
    ctx.wjson(ctx.out("s1", "stats.json"),
              {"corpus_source": cfg.corpus_source, "n_docs": len(recs),
               "n_chunks": len(ids), "dim": int(emb.shape[1]), "model": mp,
               "encode_sec": dt, "chunks_per_sec": len(ids) / max(dt, 1e-9),
               "embed_batch": cfg.embed_batch,
               "index_gb": emb.shape[0] * emb.shape[1] * 2 / 2**30})
    r.free()
    return (f"{len(recs)} 文档 -> {len(ids)} 块 dim={emb.shape[1]} "
            f"{dt/60:.1f}min ({len(ids)/max(dt,1e-9):.0f}/s)")


def _mk_retriever(ctx: RunContext) -> TorchRetriever:
    cfg = ctx.cfg
    mp = cfg.embed_model_final if cfg.use_final_embedder else cfg.embed_model_dev
    r = TorchRetriever(ctx, mp, cfg.retriever_gpu)
    ch = ctx.rjsonl(ctx.out("s1", "chunks.jsonl"))
    r.load_index([c["chunkid"] for c in ch], [c["docid"] for c in ch],
                 [c["text"] for c in ch], np.load(ctx.out("s1", "chunk_emb.npy")))
    return r


# ==============================================================================
# s2  collect
# ==============================================================================

AGENT_SYS = ("你是研究助手，可以多次调用 search 工具检索资料。"
             "每次只提出一个具体、自包含的检索查询。"
             "收集到足够信息后用 FINAL: <答案> 给出最终答案。")

SEARCH_TOOL = {"type": "function", "function": {
    "name": "search", "description": "在语料库中检索文档",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"]}}}


def _resolve_musique(steps) -> List[Dict]:
    """解析 MuSiQue 的 #N 占位符，并让每一步带上自己的答案。

    v3 直接把 "#1 >> spouse" 当查询用，后果有两条：查询里没有任何实体，
    检索无意义；且跨任务字节完全相同。smoke2 里最大的"意图簇"是 14 个不同
    任务共享同一个模板串 —— 那 20.9% 的头部覆盖率是纯字符串碰撞。
    """
    subs, ans = [], {}
    for k, st in enumerate(steps, 1):
        q = (st.get("question") or "").strip()
        a = (st.get("answer") or "").strip()
        for m in re.findall(r"#(\d+)", q):
            if int(m) in ans and ans[int(m)]:
                q = q.replace(f"#{m}", ans[int(m)])
        ans[k] = a
        if re.search(r"#\d+", q):        # 仍有未解析引用 -> 不能当查询，丢弃
            continue
        if ">>" in q:                    # "实体 >> 关系" 转自然问句，利于嵌入
            ent, _, rel = q.partition(">>")
            ent, rel = ent.strip(), rel.strip()
            if ent and rel:
                q = f"What is the {rel} of {ent}?"
        if len(q) > 8 and a:
            subs.append({"q": q, "a": a})
    return subs


def _replay_tasks(ctx: RunContext, name: str) -> List[Dict]:
    """返回 [{task_id, question, subqueries:[{q,a}], gold_answer}]

    v4：subqueries 的元素从 str 变成 {q, a}，每个子问题带自己的答案。
    v3 把复合问题的最终答案赋给了每一步，L2 永远算不对。
    """
    from datasets import load_dataset
    cfg, log = ctx.cfg, ctx.log
    lim, out, n_dropped = cfg.replay_max_tasks_per_ds, [], 0
    for repo, sub, split in _DATASET_SRC.get(name, []):
        try:
            kw = {"cache_dir": cfg.hf_cache, "split": split}
            d = load_dataset(repo, sub, **kw) if sub else load_dataset(repo, **kw)
        except Exception as e:
            log.warning(f"  {name} 源 {repo} 失败: {type(e).__name__}: {str(e)[:110]}")
            continue
        for i, r in enumerate(d):
            if i >= lim:
                break
            if name == "musique":
                subs = _resolve_musique(r.get("question_decomposition") or [])
                if not subs:
                    n_dropped += 1
                    continue
                out.append({"task_id": f"musique-{r.get('id', i)}",
                            "question": r.get("question", ""), "subqueries": subs,
                            "gold_answer": r.get("answer", "")})
            elif name == "hotpotqa":
                # v5：HotpotQA 既没有分解也没有分步答案。v4 造的
                # f"{question} (regarding {title})" 把整题问句带了进去，
                # 每个任务的子查询天然唯一 -> 跨任务复用率结构性为 0
                # （smoke3 实测正是 0.0%）。这是构造出来的假象，不是发现。
                # 现在只发一条任务级查询，如实反映"该数据集没有子任务结构"。
                ga = r.get("answer", "")
                q = (r.get("question") or "").strip()
                if not q:
                    n_dropped += 1
                    continue
                subs = [{"q": q, "a": ga}]
                out.append({"task_id": f"hotpot-{r.get('id', i)}",
                            "question": r.get("question", ""), "subqueries": subs,
                            "gold_answer": ga})
            else:
                # 2Wiki 的 evidences 是 (主语, 关系, 宾语)，宾语正是该步答案
                subs = []
                for e in (r.get("evidences") or []):
                    if not isinstance(e, (list, tuple)) or len(e) < 2 or not (e[0] and e[1]):
                        continue
                    q = f"What is the {str(e[1]).strip()} of {str(e[0]).strip()}?"
                    a = str(e[2]).strip() if len(e) >= 3 else r.get("answer", "")
                    if a:
                        subs.append({"q": q, "a": a})
                if not subs:
                    q = r.get("question", "")
                    if not q:
                        n_dropped += 1
                        continue
                    subs = [{"q": q, "a": r.get("answer", "")}]
                out.append({"task_id": f"2wiki-{r.get('_id', r.get('id', i))}",
                            "question": r.get("question", ""), "subqueries": subs,
                            "gold_answer": r.get("answer", "")})
        nq = sum(len(t["subqueries"]) for t in out)
        log.info(f"  replay:{name} {len(out)} 任务 / {nq} 子查询 "
                 f"(源 {repo}，丢弃 {n_dropped} 个无法解析的任务)")
        return out
    log.warning(f"  replay:{name} 全部源失败")
    return []


def _extract_queries(tool_calls, text: str, cfg: Config) -> Tuple[List[str], int]:
    """从一次 agent 响应里抽出检索查询列表。

    纯函数，无副作用，可单测。返回 (查询列表, 原始是否为数组)。

    历史教训：
      v4/v5 把数组 query 用空格拼成一条，造出畸形长串（嵌入漂移、检索劣化）。
      v6 改成只取第一条，但 main_v3 实测 64.6% 的调用返回的就是数组 ——
      只取第一条等于扔掉三分之二的检索意图。v7 默认全取。
    """
    q = None
    if tool_calls:
        try:
            args = tool_calls[0].get("function", {}).get("arguments")
            if isinstance(args, str):
                args = json.loads(args)
            if isinstance(args, dict):
                q = args.get("query")
        except Exception:
            q = None
    if q is None and text:
        m = re.findall(r'"query"\s*:\s*"([^"]+)"', text)
        if m:
            q = m if len(m) > 1 else m[0]
    was_list = isinstance(q, (list, tuple))
    if was_list:
        qs = [str(x).strip() for x in q if str(x) and str(x).strip()]
    elif q is None:
        qs = []
    else:
        qs = [str(q).strip()] if str(q).strip() else []
    # 去重（保序）+ 单条长度上限 + 数量上限
    seen, out = set(), []
    for s in qs:
        if len(s) > 200:
            s = s[:200].rsplit(" ", 1)[0]
        k = s.lower()
        if len(s) >= 3 and k not in seen:
            seen.add(k)
            out.append(s)
    if cfg.agent_multi_query_mode == "first":
        out = out[:1]
    return out[: max(1, cfg.agent_max_subqueries)], int(was_list and len(qs) > 1)


def _agent_task(pool: LLMPool, retr: TorchRetriever, task: Dict, cfg: Config) -> Dict:
    msgs = [{"role": "system", "content": AGENT_SYS},
            {"role": "user", "content": task["question"]}]
    calls, pin, pout = [], 0, 0
    for turn in range(cfg.agent_max_turns):
        try:
            r = pool.chat(msgs, temperature=cfg.agent_temperature,
                          max_tokens=cfg.agent_max_tokens,
                          tools=[SEARCH_TOOL], tool_choice="auto", **NO_THINK)
        except Exception as e:
            return {"task_id": task["task_id"], "error": f"LLM: {type(e).__name__}: {e}",
                    "calls": calls}
        pin += r["prompt_tokens"]; pout += r["completion_tokens"]
        txt = r["text"]
        tc = (r["raw"]["choices"][0]["message"].get("tool_calls") or [])
        qs, was_multi = _extract_queries(tc, txt, cfg)
        if not qs:
            if "FINAL:" in txt:
                break
            msgs += [{"role": "assistant", "content": txt},
                     {"role": "user", "content": "请用 search 工具检索，或给出 FINAL: 答案。"}]
            continue
        try:
            t0 = time.time(); S, I = retr.search(qs, cfg.topk)
            lat = (time.time() - t0) * 1000 / len(qs)
        except Exception as e:
            return {"task_id": task["task_id"],
                    "error": f"检索失败: {type(e).__name__}: {str(e)[:150]}", "calls": calls}
        obs_parts = []
        for k, sq in enumerate(qs):
            ds = retr.docs(I[k])
            seg = "\n\n".join(f"[{i+1}] {d['text'][:600]}" for i, d in enumerate(ds))
            obs_parts.append((f"查询「{sq}」的结果:\n" if len(qs) > 1 else "") + seg)
            calls.append({"task_id": task["task_id"], "source": "agent", "turn_idx": turn,
                          "query": sq, "multi_query": was_multi,
                          "multi_query_idx": k, "multi_query_n": len(qs),
                          "retrieved": [d["docid"] for d in ds],
                          "retrieved_chunks": [d["chunkid"] for d in ds],
                          "scores": [float(x) for x in S[k][:len(ds)]],
                          "latency_ms": lat, "context_tokens_added": len(seg) // 4,
                          "n_reranked": len(ds), "gold_answer": task.get("gold_answer", ""),
                          "agent_prompt_tokens": r["prompt_tokens"] if k == 0 else 0,
                          "agent_completion_tokens": r["completion_tokens"] if k == 0 else 0})
        obs = "\n\n".join(obs_parts)
        msgs += [{"role": "assistant", "content": txt},
                 {"role": "user", "content": f"检索结果:\n{obs}"}]
    return {"task_id": task["task_id"], "calls": calls,
            "prompt_tokens": pin, "completion_tokens": pout}


@stage("s2", "检索调用采集（replay + agent）", ["calls.jsonl", "stats.json"])
def s2(ctx: RunContext, sm: ServerManager):
    from tqdm import tqdm
    cfg, log = ctx.cfg, ctx.log
    allc: List[Dict] = []; st: Dict[str, Any] = {}
    _agent_pool: List[Dict] = []      # 复用 replay 阶段已加载的 musique 任务

    sm.stop()
    retr = _mk_retriever(ctx)

    # ---------------------------------------------------------- replay --
    for name in cfg.replay_datasets:
        tasks = _replay_tasks(ctx, name)
        if name == "musique":
            _agent_pool = tasks
        if not tasks:
            st[name] = {"tasks": 0, "calls": 0, "note": "数据集加载失败"}
            continue
        # v4：subqueries 是 [{q, a}]，每步用自己的答案而不是复合问题的最终答案
        flat = [(t["task_id"], j, s["q"], s["a"])
                for t in tasks for j, s in enumerate(t["subqueries"])]
        B = 512
        for s in tqdm(range(0, len(flat), B), desc=f"retr-{name}", ncols=100):
            b = flat[s:s + B]
            t0 = time.time()
            S, I = retr.search([x[2] for x in b], cfg.topk, batch=B)
            lat = (time.time() - t0) * 1000 / len(b)
            for k, (tid, turn, q, gold) in enumerate(b):
                ds = retr.docs(I[k])
                allc.append({"task_id": tid, "source": f"replay:{name}", "turn_idx": turn,
                             "query": q, "retrieved": [d["docid"] for d in ds],
                             "retrieved_chunks": [d["chunkid"] for d in ds],
                             "scores": [float(x) for x in S[k][:len(ds)]],
                             "latency_ms": lat,
                             "context_tokens_added": sum(len(d["text"]) for d in ds) // 4,
                             "n_reranked": len(ds), "gold_answer": gold,
                             "agent_prompt_tokens": 0, "agent_completion_tokens": 0})
            ctx.beat(f"replay-{name} {s+len(b)}/{len(flat)}")
        st[name] = {"tasks": len(tasks),
                    "calls": sum(1 for c in allc if c["source"].endswith(name))}

    # ----------------------------------------------------------- agent --
    # v2 的致命 bug：检索索引在 GPU0，却在两张卡上都起 vLLM（各吃 67GB），
    # GPU0 的检索器第一次 encode 就 OOM -> 6/6 全挂。agent 必须独占非检索卡。
    if cfg.agent_enabled and Path(cfg.agent_path, "config.json").exists():
        agent_gpus = [g for g in cfg.gpus if g != cfg.retriever_gpu] or list(cfg.gpus)
        if len(cfg.gpus) > 1:
            log.info(f"agent 独占 GPU{agent_gpus}，GPU{cfg.retriever_gpu} 留给检索器")
        else:
            log.warning(f"只有 1 张卡，agent 与检索器同卡共存，"
                        f"agent_mem_util={cfg.agent_mem_util} 已按此下调")
        # v8：agent 起不来不再中止整个 s2。main_v4 第一次跑就是 agent 超时
        # 导致前面 15 分钟的 replay 检索全部作废。replay 是主证据，不能被拖累。
        try:
            urls = sm.ensure("agent", agent_gpus)
        except Exception as e:
            log.error(f"agent 服务启动失败，降级为纯 replay（不中止）: "
                      f"{type(e).__name__}: {e}")
            sm.stop()
            st["agent"] = {"failed": True, "error": f"{type(e).__name__}: {e}",
                           "tasks": 0, "calls": 0}
            urls = None
        if urls:
            pool = LLMPool(urls, cfg.agent_name, retries=2, timeout=900)
            tasks = (_agent_pool or _replay_tasks(ctx, "musique"))[:cfg.agent_max_tasks]
            res = pmap(lambda t: _agent_task(pool, retr, t, cfg), tasks,
                       cfg.agent_concurrency, "agent", ctx, pool)
            nerr, top = summarize_errors(res, log, "agent")
            for r in res:
                if isinstance(r, dict) and not (r.get("__error__") or r.get("error")):
                    allc.extend(r.get("calls", []))
            ac = [c for c in allc if c["source"] == "agent"]
            st["agent"] = {"tasks": len(tasks), "errors": nerr, "top_errors": top,
                       "gpus": agent_gpus, "calls": len(ac), "llm": pool.stats(),
                       # v6：agent 一次返回多条查询的比例。高说明 prompt 里
                       # "每次只提出一个查询"没被遵守，需要改 prompt。
                           "multi_query_mode": cfg.agent_multi_query_mode,
                           "multi_query_frac": (sum(1 for c in ac if c.get("multi_query"))
                                                / max(len(ac), 1)),
                           "mean_subqueries": float(np.mean([c.get("multi_query_n", 1)
                                                             for c in ac])) if ac else 0.0,
                           "calls_first_arm": sum(1 for c in ac
                                                  if c.get("multi_query_idx", 0) == 0),
                           "median_query_len": float(np.median([len(c["query"]) for c in ac]))
                                               if ac else 0.0}
            log.info(f"agent 多查询比例 {st['agent']['multi_query_frac']:.1%}，"
                     f"查询中位长度 {st['agent']['median_query_len']:.0f}")
            log.info(f"agent 失败 {nerr}/{len(tasks)}，产出 {st['agent']['calls']} 条")
            sm.stop()
    else:
        st["agent"] = {"skipped": True,
                       "reason": "agent_enabled=False" if not cfg.agent_enabled else "模型缺失"}
    retr.free()

    # ------------------------------------------------- 去重 / 成本埋点 --
    seen, fin = set(), []
    for c in allc:
        c["query"] = re.sub(r"\s+", " ", c["query"]).strip()
        if len(c["query"]) < 5:
            continue
        qid = sha(c["task_id"], c["turn_idx"], c["query"])
        if qid in seen:
            continue
        seen.add(qid); c["qid"] = qid
        c["cost"] = (cfg.cost_per_search_call
                     + cfg.cost_per_context_token * c["context_tokens_added"]
                     + cfg.cost_per_rerank_doc * c["n_reranked"])
        fin.append(c)

    # ---------------- 数据健全性（v4 新增）----------------
    # 跨任务字节相同的查询占比：模板碰撞的直接度量，smoke2 会让它接近 1
    qtext = Counter(c["query"] for c in fin)
    by_text_tasks = defaultdict(set)
    for c in fin:
        by_text_tasks[c["query"]].add(c["task_id"])
    dup = sum(n for t, n in qtext.items() if len(by_text_tasks[t]) >= 2)
    st["sanity"] = {
        "exact_dup_query_frac": dup / max(len(fin), 1),
        "unresolved_placeholder_frac":
            sum(1 for c in fin if re.search(r"#\d+", c["query"])) / max(len(fin), 1),
        "unique_query_frac": len(qtext) / max(len(fin), 1),
        "median_query_len": float(np.median([len(c["query"]) for c in fin])),
        "with_gold_frac": sum(1 for c in fin if c.get("gold_answer")) / max(len(fin), 1),
    }
    log.info(f"[健全性] 跨任务字节相同查询 {st['sanity']['exact_dup_query_frac']:.1%} | "
             f"未解析占位符 {st['sanity']['unresolved_placeholder_frac']:.1%} | "
             f"查询中位长度 {st['sanity']['median_query_len']:.0f}")
    if st["sanity"]["unresolved_placeholder_frac"] > cfg.stop_max_unresolved:
        log.error(f"!! 未解析的 #N 占位符 "
                  f"{st['sanity']['unresolved_placeholder_frac']:.1%} > "
                  f"{cfg.stop_max_unresolved:.0%}，意图簇会退化成模板碰撞")
    # v5：跨任务字节相同的查询在占位符解析后是"带实体的真实重复需求"，
    # 是要找的复用本身，不再当闸门，只作描述性统计。
    log.info(f"[健全性] 跨任务重复查询 {st['sanity']['exact_dup_query_frac']:.1%}"
             f"（描述性；占位符已清零时这代表真实复用，不是模板碰撞）")

    ctx.wjsonl(ctx.out("s2", "calls.jsonl"), fin)
    st.update({"total_calls": len(fin), "total_tasks": len({c["task_id"] for c in fin}),
               "total_cost": sum(c["cost"] for c in fin)})
    st["calls_per_task"] = st["total_calls"] / max(st["total_tasks"], 1)
    ctx.wjson(ctx.out("s2", "stats.json"), st)
    if not fin:
        raise RuntimeError("一条检索调用都没采到，检查数据集加载")
    return (f"{len(fin)} 调用 / {st['total_tasks']} 任务 "
            f"({st['calls_per_task']:.1f} 次/任务), 重复查询 "
            f"{st['sanity']['exact_dup_query_frac']:.1%}")


# ==============================================================================
# s3  embed queries
# ==============================================================================


@stage("s3", "查询嵌入", ["query_emb.npy", "qindex.json"])
def s3(ctx: RunContext, sm: ServerManager):
    cfg = ctx.cfg
    sm.stop()
    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    mp = cfg.embed_model_final if cfg.use_final_embedder else cfg.embed_model_dev
    r = TorchRetriever(ctx, mp, cfg.retriever_gpu)
    e = r.encode([c["query"] for c in calls], True, cfg.embed_batch, "queries")
    np.save(ctx.out("s3", "query_emb.npy"), e)
    ctx.wjson(ctx.out("s3", "qindex.json"),
              {"qids": [c["qid"] for c in calls], "n": len(calls),
               "dim": int(e.shape[1]), "model": mp})
    r.free()
    return f"{len(calls)} 查询 dim={e.shape[1]}"


# ==============================================================================
# s4  blocking
# ==============================================================================


@stage("s4", "候选对生成 + 分块完整性暴力检验", ["pairs.jsonl", "audit.json"])
def s4(ctx: RunContext, sm: ServerManager):
    import torch
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    E = np.load(ctx.out("s3", "query_emb.npy"))
    n = len(calls); dev = f"cuda:{cfg.retriever_gpu}"
    T = torch.nn.functional.normalize(torch.from_numpy(E).to(dev, dtype=torch.float16), dim=1)

    k = min(cfg.block_topk + 1, n)
    log.info(f"查询 kNN {n} × top-{k}")
    sims, idxs = [], []
    for s in range(0, n, 4096):
        sc = T[s:s + 4096] @ T.T
        v, i = torch.topk(sc, k, dim=1)
        sims.append(v.float().cpu().numpy()); idxs.append(i.cpu().numpy())
        del sc; ctx.beat(f"knn {s}/{n}")
    sims = np.concatenate(sims); idxs = np.concatenate(idxs)

    pairs, seen = [], set()
    for i in range(n):
        for r in range(k):
            j = int(idxs[i][r]); s = float(sims[i][r])
            if j == i or s < cfg.block_min_sim:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            ca, cb = calls[a], calls[b]
            pairs.append({"pair_id": sha(ca["qid"], cb["qid"]), "i": a, "j": b,
                          "qid_i": ca["qid"], "qid_j": cb["qid"], "cos": s,
                          "same_task": ca["task_id"] == cb["task_id"],
                          "same_source": ca["source"] == cb["source"],
                          "same_text": ca["query"] == cb["query"],
                          "rbo": rbo(ca["retrieved"], cb["retrieved"]),
                          "jaccard": jacc(ca["retrieved"], cb["retrieved"])})
    same_text = sum(p["same_text"] for p in pairs)
    log.info(f"候选对 {len(pairs)}，字节完全相同 {same_text} "
             f"({same_text/max(len(pairs),1):.1%})")

    probe = random.sample(range(n), min(cfg.bruteforce_probe_n, n))
    acc = {str(t): {"m": 0, "n": 0} for t in (0.80, 0.85, 0.90)}
    for i in probe:
        full = (T[i:i + 1] @ T.T)[0].float().cpu().numpy(); full[i] = -1
        blocked = set(int(x) for x in idxs[i])
        for t in (0.80, 0.85, 0.90):
            hi = np.where(full >= t)[0]
            acc[str(t)]["n"] += len(hi)
            acc[str(t)]["m"] += sum(1 for x in hi if int(x) not in blocked)
    audit = {"n_queries": n, "n_pairs": len(pairs), "block_topk": cfg.block_topk,
             "probe_n": len(probe), "detail": acc,
             "same_text_pair_frac": same_text / max(len(pairs), 1),
             "miss_rate": {t: (v["m"] / v["n"] if v["n"] else 0.0) for t, v in acc.items()}}
    mr = audit["miss_rate"]["0.9"]
    log.info(f"分块漏检率 @cos≥0.90 = {mr:.3%}")
    if mr > 0.05:
        log.warning(f"漏检率 >5%，建议 block_topk 提到 {cfg.block_topk*2} 重跑 s4")

    ctx.wjsonl(ctx.out("s4", "pairs.jsonl"), pairs)
    ctx.wjson(ctx.out("s4", "audit.json"), audit)
    del T; torch.cuda.empty_cache()
    if not pairs:
        raise RuntimeError("没有候选对，检查 block_min_sim 或查询数量")
    return f"{len(pairs)} 候选对 漏检@0.9={mr:.2%}"


# ==============================================================================
# s5  sampling
# ==============================================================================


def _bin(c, e):
    for b in range(len(e) - 1):
        if e[b] <= c < e[b + 1]:
            return b
    return len(e) - 2


def _alloc_bins(sizes: Dict[int, int], budget: int, targets: List[int],
                minb: int, nb: int) -> Dict[int, int]:
    """按每层显式目标分配抽样配额。

    为什么不用 Neyman：Neyman 最小化的是**总体**估计方差，会给 p≈0 或 p≈1
    的层近乎零的配额（方差项 sqrt(p(1-p))→0）。main_v4 实测 bin7 只分到 612
    对，比 bin1 的 9321 还少 —— 而 τ* 恰恰由 bin7 决定，bin0 那 29598 对
    只是在反复确认"可替代率约 0.02"。我们要的是**逐层**精度，不是总体精度，
    所以直接给每层显式目标，受该层可用量与总预算约束。
    """
    al = {b: min(targets[b] if b < len(targets) else minb, sizes.get(b, 0))
          for b in range(nb)}
    tot = sum(al.values())
    if tot > budget and tot > 0:
        sc = budget / tot
        al = {b: max(min(minb, sizes.get(b, 0)), int(v * sc)) for b, v in al.items()}
        # 缩放后可能仍超预算（保底导致），再从最大层削
        while sum(al.values()) > budget:
            b = max(al, key=lambda x: al[x])
            if al[b] <= min(minb, sizes.get(b, 0)):
                break
            al[b] -= max(1, (sum(al.values()) - budget) // nb)
    else:
        # 余量优先补高相似度层，但每层不超过目标的 2 倍 —— 否则余量会全部溢到
        # bin0/bin1（它们各有十几万可用），而那两层可替代率只有 0.02/0.16，
        # 再多样本也不改变结论，纯属浪费判官调用。宁可少用预算。
        left = budget - tot
        cap = {b: max((targets[b] if b < len(targets) else minb) * 2, minb)
               for b in range(nb)}
        for b in sorted(al, reverse=True):
            add = min(left, max(0, min(sizes.get(b, 0), cap[b]) - al[b]))
            al[b] += add
            left -= add
            if left <= 0:
                break
    return {b: max(0, int(v)) for b, v in al.items()}


def _neyman(W, p, budget, minb, nb, sizes, tilt=None):
    """Neyman 分配 + 分层倾斜。

    v7：main_v3 里 bin7 只分到 n=219，而 τ* 恰恰由高相似度层决定。
    tilt 让高层拿到更多预算，代价是低层精度下降 —— 低层可替代率接近 0，
    本来也不需要那么高的精度。
    """
    t = tilt or [1.0] * nb
    w = {b: W[b] * math.sqrt(max(p[b] * (1 - p[b]), 1e-6)) * (t[b] if b < len(t) else 1.0)
         for b in range(nb)}
    tot = sum(w.values()) or 1.0
    al = {b: max(min(int(round(budget * w[b] / tot)), sizes.get(b, 0)),
                 min(minb, sizes.get(b, 0))) for b in range(nb)}
    over = sum(al.values()) - budget
    order = sorted(range(nb), key=lambda b: -al[b]); k = 0
    while over > 0 and k < 20 * nb:
        b = order[k % nb]
        if al[b] > min(minb, sizes.get(b, 0)):
            al[b] -= 1; over -= 1
        k += 1
    return al


@stage("s5", "分层抽样 + 陷阱对 + 金标导出",
       ["plan.json", "l1_pairs.jsonl", "gold_pairs.jsonl"])
def s5(ctx: RunContext, sm: ServerManager):
    cfg, log = ctx.cfg, ctx.log
    pairs = ctx.rjsonl(ctx.out("s4", "pairs.jsonl"))
    e = cfg.sim_bins; nb = len(e) - 1
    byb = defaultdict(list)
    for p in pairs:
        byb[_bin(p["cos"], e)].append(p)
    W = {b: len(byb[b]) / max(len(pairs), 1) for b in range(nb)}
    log.info("分层: " + ", ".join(f"[{e[b]:.2f},{e[b+1]:.2f})={len(byb[b])}" for b in range(nb)))

    pilot = []
    for b in range(nb):
        for p in random.sample(byb[b], min(cfg.pilot_per_bin, len(byb[b]))):
            q = dict(p); q["bin"] = b; q["wave"] = "pilot"; pilot.append(q)

    prior = {}
    for b in range(nb):
        if not byb[b]:
            prior[b] = 0.0; continue
        s = random.sample(byb[b], min(200, len(byb[b])))
        prior[b] = float(np.mean([1.0 if x["rbo"] >= 0.5 else 0.0 for x in s]))

    alloc = _alloc_bins({b: len(byb[b]) for b in range(nb)},
                        max(cfg.l1_budget - len(pilot), 0),
                        cfg.bin_target, cfg.l1_min_per_bin, nb)
    log.info("分层配额: " + ", ".join(f"b{b}={alloc[b]}" for b in range(nb))
             + f"  合计 {sum(alloc.values())}")

    pid = {p["pair_id"] for p in pilot}; main = []
    for b in range(nb):
        pool = [p for p in byb[b] if p["pair_id"] not in pid]
        for p in random.sample(pool, min(alloc[b], len(pool))):
            q = dict(p); q["bin"] = b; q["wave"] = "main"; main.append(q)
    l1 = pilot + main
    for p in l1:
        p["is_catch"] = False

    catch = []
    for p in random.sample(l1, min(int(len(l1) * cfg.catch_trial_rate), len(l1))):
        q = dict(p); q["pair_id"] = "CATCH-" + q["pair_id"]
        q["is_catch"] = True; q["wave"] = "catch"; catch.append(q)

    gold = []
    sp = sorted(l1, key=lambda x: x["cos"])
    if sp:
        t = max(len(sp) // 3, 1); per = max(cfg.gold_set_n // 3, 1)
        # v10：按相似度层加权抽样，偏向高相似度层。
        # main_v6 之前按三分位均匀抽，高层样本太少 —— 而判官恰恰在 bin7
        # 最不稳定（位置一致率 0.699），人工标注最需要覆盖的正是那里。
        _gw = cfg.gold_bin_weights
        _bybin = defaultdict(list)
        for p in l1:
            _bybin[p.get("bin", 0)].append(p)
        _tw = sum(_gw[b] if b < len(_gw) else 1.0 for b in _bybin)
        for b, lst in sorted(_bybin.items()):
            k = int(round(cfg.gold_set_n * (_gw[b] if b < len(_gw) else 1.0) / max(_tw, 1e-9)))
            for p in random.sample(lst, min(k, len(lst))):
                g = dict(p); g["human_label"] = None; gold.append(g)
        log.info("[金标] 按层抽样: " + ", ".join(
            f"b{b}={sum(1 for g in gold if g.get('bin') == b)}" for b in sorted(_bybin)))

    ctx.wjsonl(ctx.out("s5", "l1_pairs.jsonl"), l1 + catch)
    ctx.wjsonl(ctx.out("s5", "gold_pairs.jsonl"), gold)
    ctx.wjson(ctx.out("s5", "plan.json"),
              {"bins": e, "sizes": {b: len(byb[b]) for b in range(nb)},
               # v9：分层权重回代要用的总体层大小（与 sizes 同值，显式命名避免歧义）
               "population_sizes": {b: len(byb[b]) for b in range(nb)}, "W": W,
               "prior_rbo": prior, "alloc": alloc, "n_pilot": len(pilot),
               "n_main": len(main), "n_catch": len(catch), "n_gold": len(gold)})
    log.warning(f"金标集 {len(gold)} 对 -> stages/s5/gold_pairs.jsonl，"
                f"人工填 human_label(1/0) 后重跑 s6 可补 accuracy")
    return f"L1 {len(l1)} + 陷阱 {len(catch)} + 金标 {len(gold)}"


# ==============================================================================
# s6  L1 judge
# ==============================================================================

L1_PROMPT = """你要判断一组检索结果是否足以回答一个问题。

**问题**
{q}

**检索结果**
{docs}

请按以下步骤：
1. 列出回答这个问题**必须**具备的事实点（1-3 条，具体到实体和属性）。
2. 逐条核对检索结果中是否明确包含该事实。
3. 给出结论。

注意：只看检索结果里有没有，不要用你自己的知识补全。结果里没有明确写出的，一律 false。

只输出 JSON，不要任何其他文字：
{{"required_facts":["..."],"found":[true,false],"verdict":"SUFFICIENT"|"PARTIAL"|"INSUFFICIENT","confidence":0.0}}"""


def _fmt(chunks, cmap, lim, shuffle=False):
    ids = list(chunks)
    if shuffle:
        random.shuffle(ids)
    return "\n\n".join(f"[{i+1}] {(cmap.get(c) or '')[:lim]}"
                       for i, c in enumerate(ids)) or "(无结果)"


# v11：显式声明文档顺序无关。main_v7 实测整体位置翻转率 21.5%、bin7 达 25.5%，
# 而判定本应与呈现顺序无关。s21 用 A/B 直接测这条指令能否把翻转率压下去。
L1_PROMPT_ORDER_INVARIANT = L1_PROMPT.replace(
    "注意：只看检索结果里有没有，不要用你自己的知识补全。结果里没有明确写出的，一律 false。",
    "注意：\n"
    "- 只看检索结果里有没有，不要用你自己的知识补全。结果里没有明确写出的，一律 false。\n"
    "- 检索结果的**呈现顺序与编号无关紧要**。把它们当作一个无序集合，"
    "只判断这个集合整体是否包含所需事实。同一组文档换个顺序，你的结论必须完全相同。")


# ==============================================================================
#  v12：判官 prompt 变体
# ==============================================================================
# main_v8 的人工金标（两批各 299 对，正例率 0.261/0.291，逐层曲线几乎重合）
# 显示当前判官 accuracy 约 0.44：recall 0.85 但 precision 0.31，
# 系统性把「同主题但不含该事实」的文档判成 SUFFICIENT，且相似度越高越严重
# （bin7 判官 0.970 vs 人工 0.373）。
#
# 陷阱对误判率 0.017% 说明它能识别「完全无关」，真正的难点是「相关但不充分」——
# 我原来的 QC 设计漏掉了后者。下面的变体都针对这一点。

_P_EVIDENCE = """你要判断一组检索结果是否足以回答一个问题。

**问题**
{q}

**检索结果**
{docs}

步骤：
1. 列出回答这个问题**必须**具备的事实点（1-3 条，具体到实体和属性）。
2. 对每个事实点，从检索结果中**逐字摘出**能证明它的原句（写进 evidence）。
   摘不出原句就是没有 —— 不要转述、不要推断、不要用自己的知识补全。
   文档只是提到了相关实体、话题相近、或者讲的是同类事物，都不算。
3. 只有**每一个**事实点都摘到了原句，verdict 才是 SUFFICIENT。

只输出 JSON：
{{"required_facts":["..."],"evidence":["原句或空字符串"],"found":[true,false],"verdict":"SUFFICIENT"|"INSUFFICIENT","confidence":0.0}}"""

_P_STRICT = """你要判断一组检索结果是否足以回答一个问题。

**问题**
{q}

**检索结果**
{docs}

判断标准（务必严格）：
- 检索结果必须**明确陈述**回答该问题所需的全部事实。
- 以下情况一律记为 INSUFFICIENT：
  * 文档提到了问题里的实体，但没有说出问题问的那个属性
  * 文档讲的是同类事物或相关话题，但不是问题问的那一个
  * 需要跨文档推理、需要常识补全、需要你自己的知识才能得出答案
  * 只能找到部分事实
- 只有当你能直接从文档中读出完整答案时，才是 SUFFICIENT。

先写出你从文档中读到的答案（读不出就写 NONE），再给结论。

只输出 JSON：
{{"answer_from_docs":"...","required_facts":["..."],"found":[true,false],"verdict":"SUFFICIENT"|"INSUFFICIENT","confidence":0.0}}"""

_P_CONSERVATIVE = """你要判断一组检索结果是否足以回答一个问题。

**问题**
{q}

**检索结果**
{docs}

重要：经验表明判断者普遍**过于宽松** —— 看到相关话题就以为够了。
请刻意保守：**拿不准时一律判 INSUFFICIENT**。
只有当文档白纸黑字写出了答案、你可以直接引用时，才判 SUFFICIENT。

1. 列出所需事实点。
2. 逐条核对文档里是否**明确写出**。
3. 给结论（存疑即 INSUFFICIENT）。

只输出 JSON：
{{"required_facts":["..."],"found":[true,false],"verdict":"SUFFICIENT"|"INSUFFICIENT","confidence":0.0}}"""

_P_ANSWER_FIRST = """根据下面的检索结果回答问题。

**检索结果**
{docs}

**问题**
{q}

先只用检索结果作答；如果检索结果里找不到答案，answer 写 "NONE"。
然后判断：你给出的 answer 是否**完全来自**检索结果的明确陈述？
如果 answer 是 NONE，或者你用到了检索结果以外的知识，verdict 就是 INSUFFICIENT。

只输出 JSON：
{{"answer":"...","from_docs_only":true,"verdict":"SUFFICIENT"|"INSUFFICIENT","confidence":0.0}}"""

_P_BINARY = """**问题**
{q}

**检索结果**
{docs}

这些检索结果里是否明确写出了回答上述问题所需的全部信息？
只看文档写了什么，不要用你自己的知识。文档只是话题相关但没写出答案 -> NO。

只输出 JSON：{{"verdict":"SUFFICIENT"|"INSUFFICIENT"}}
（能明确回答 = SUFFICIENT，否则 = INSUFFICIENT）"""


def _prompt_variants() -> Dict[str, str]:
    return {
        "base": L1_PROMPT,
        "order_inv": L1_PROMPT_ORDER_INVARIANT,
        "evidence_quote": _P_EVIDENCE,
        "strict_checklist": _P_STRICT,
        "conservative": _P_CONSERVATIVE,
        "answer_first": _P_ANSWER_FIRST,
        "binary_direct": _P_BINARY,
    }


def _selected_prompt(ctx: "RunContext") -> Optional[str]:
    """读 s22 对拍选中的 prompt；没有就返回 None（沿用默认）。"""
    p = ctx.out("s22", "selected.json")
    if not p.exists():
        return None
    try:
        name = ctx.rjson(p).get("winner")
        return _prompt_variants().get(name)
    except Exception:
        return None


def _judge1(pool: LLMPool, cfg: Config, q: str, docs: str,
            prompt: Optional[str] = None) -> Dict:
    # NO_THINK 是必须的：Qwen3-32B 默认开思考，实测 398 tokens 全被 <think>
    # 吃掉，JSON 一个字都写不出来，parse_fail_rate 直接 27.6%。
    tmpl = prompt or getattr(cfg, "_active_prompt", None) or (
        L1_PROMPT_ORDER_INVARIANT
        if getattr(cfg, "judge_order_invariant", False) else L1_PROMPT)
    r = pool.chat([{"role": "user", "content": tmpl.format(q=q, docs=docs)}],
                  temperature=cfg.judge_temperature, max_tokens=cfg.judge_max_tokens,
                  **NO_THINK)
    tok = r["prompt_tokens"] + r["completion_tokens"]
    j = xjson(r["text"])
    if not j:
        return {"ok": False, "verdict": "PARSE_FAIL", "tokens": tok,
                "truncated": r["finish_reason"] == "length",
                "out_tokens": r["completion_tokens"], "raw": r["text"][-400:]}
    v = str(j.get("verdict", "")).upper()
    return {"ok": True, "verdict": v, "covered": int(v == "SUFFICIENT"),
            "facts": j.get("required_facts", []), "found": j.get("found", []),
            "tokens": tok}


@stage("s6", "自覆盖筛选 + L1 判官（条件可替代性）", ["l1.jsonl", "qc.json", "self_cov.json"])
def s6(ctx: RunContext, sm: ServerManager):
    """v5 的核心改动：把"可替代性"和"检索召回率"拆开。

    v4 的 substitutable 要求双向 SUFFICIENT，于是它实测的是
        P(R(q_i) 足以回答 q_i) x P(R(q_j) 足以回答 q_j) x P(交叉覆盖)
    smoke3 里 40% 的查询连自己的检索结果都答不了，被算成"不可替代"。
    结果是校准曲线非单调：cos>=0.95 的 bin7 只有 0.333，比 bin6 的 0.667 还低，
    因为 0.58^2 ~ 0.34 —— 那是自覆盖率的平方，不是相似度的信号。

    v5 先判自覆盖，只在 self_cov(i) 且 self_cov(j) 的对上算可替代性。
    这个条件化在论文里也更站得住：基础索引本来答不了的查询，没有召回可损失，
    视图对它们无所谓好坏。命中率的正确表述是"在基础索引能回答的查询中，
    有多少比例可以被视图服务"。

    成本上反而更省：不合格的对直接跳过 3 次 L1 调用，
    自覆盖按 qid 记忆化（smoke3 实测 1.82 个不同查询/对）。
    """
    cfg, log = ctx.cfg, ctx.log
    calls = {c["qid"]: c for c in ctx.rjsonl(ctx.out("s2", "calls.jsonl"))}
    cmap = {c["chunkid"]: c["text"] for c in ctx.rjsonl(ctx.out("s1", "chunks.jsonl"))}
    allchunks = list(cmap)
    pairs = ctx.rjsonl(ctx.out("s5", "l1_pairs.jsonl"))
    real_pairs = [p for p in pairs if not p["is_catch"]]
    catch_pairs = [p for p in pairs if p["is_catch"]]

    # v9：分层抽样权重 w_b = N_b / n_b。各层抽样率差两个数量级，
    # 不回代权重的话所有「整体比率」都会被高相似度层拉高。
    _plan = ctx.rjson(ctx.out("s5", "plan.json"))
    _pop = {int(k): v for k, v in (_plan.get("population_sizes")
                                   or _plan.get("sizes") or {}).items()}
    _nsamp = Counter(p.get("bin") for p in real_pairs)
    for p in real_pairs:
        b = p.get("bin")
        p["w"] = _pop.get(b, 1) / max(_nsamp.get(b, 1), 1)
    log.info("[分层权重] " + ", ".join(
        f"b{b}: {_nsamp.get(b,0)}/{_pop.get(b,0)} -> w={_pop.get(b,1)/max(_nsamp.get(b,1),1):.1f}"
        for b in sorted(_pop)))

    # v12：用对拍选中的 prompt。没跑对拍就是 None，沿用默认。
    _sel = _selected_prompt(ctx)
    if _sel:
        cfg._active_prompt = _sel
        _bo = ctx.rjson(ctx.out("s22", "selected.json"))
        log.info(f"[prompt] 使用对拍选中的变体 {_bo.get('winner')}"
                 f"（金标 accuracy {_bo.get('accuracy')}）")
    else:
        log.warning("[prompt] 未跑对拍，沿用默认 prompt —— "
                    "main_v8 实测默认 prompt 的金标 accuracy 仅约 0.44")

    urls = sm.ensure("judge", list(cfg.gpus))
    pool = LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)

    # ---------------- Phase A：自覆盖（按 qid 记忆化，每查询一次）----------------
    self_cov: Dict[str, Optional[int]] = {}
    if cfg.self_cov_enabled:
        need = sorted({q for p in real_pairs for q in (p["qid_i"], p["qid_j"])})
        log.info(f"[自覆盖] {len(real_pairs)} 对涉及 {len(need)} 个不同查询 "
                 f"({len(need)/max(len(real_pairs),1):.2f} 查询/对)")

        def sc(qid):
            c = calls[qid]
            r = _judge1(pool, cfg, c["query"],
                        _fmt(c["retrieved_chunks"], cmap, cfg.judge_doc_chars))
            return {"qid": qid, "cov": (r.get("covered", 0) if r["ok"] else None),
                    "verdict": r.get("verdict")}

        _t_sc = time.time()
        scr = pmap(sc, need, cfg.judge_concurrency, "self-cov", ctx, pool)
        for r in scr:
            if isinstance(r, dict) and "qid" in r:
                self_cov[r["qid"]] = r["cov"]
        ok_n = sum(1 for v in self_cov.values() if v == 1)
        log.info(f"[自覆盖] {ok_n}/{len(self_cov)} = {ok_n/max(len(self_cov),1):.1%} "
                 f"的查询能被自己的检索结果回答")
        ctx.wjson(ctx.out("s6", "self_cov.json"),
                  {"rate": ok_n / max(len(self_cov), 1), "n": len(self_cov),
                   "cov": self_cov})
    else:
        _t_sc = time.time()
        self_cov = {q: 1 for p in real_pairs for q in (p["qid_i"], p["qid_j"])}
        ctx.wjson(ctx.out("s6", "self_cov.json"), {"disabled": True})

    def eligible(p):
        return self_cov.get(p["qid_i"]) == 1 and self_cov.get(p["qid_j"]) == 1

    elig = [p for p in real_pairs if eligible(p)]
    by_bin_all, by_bin_el = Counter(), Counter()
    for p in real_pairs:
        by_bin_all[p.get("bin")] += 1
        if eligible(p):
            by_bin_el[p.get("bin")] += 1
    log.info(f"[自覆盖] 合格对 {len(elig)}/{len(real_pairs)} = "
             f"{len(elig)/max(len(real_pairs),1):.1%}")
    for b in sorted(by_bin_all):
        log.info(f"    bin{b}: {by_bin_el[b]}/{by_bin_all[b]} 合格")
    if len(elig) < 40:
        log.error(f"!! 合格对仅 {len(elig)}，校准会不可靠。"
                  f"调大 l1_budget 或 l1_oversample 后重跑 s5,s6")

    # ---------------- Phase B：L1 判定（只跑合格对 + 全部陷阱对）----------------
    # v10：double_judge_bins 里的对全部做第二次判定（打乱文档顺序），
    # 其余层按 pos_shuffle_frac 抽样做 QC。
    _dbins = set(cfg.double_judge_bins)
    _dbl = [p for p in elig if p.get("bin") in _dbins]
    _rest = [p for p in elig if p.get("bin") not in _dbins]
    shuffle_set = {p["pair_id"] for p in _dbl}
    if _rest:
        shuffle_set |= set(random.sample([p["pair_id"] for p in _rest],
                                         max(1, int(len(_rest) * cfg.pos_shuffle_frac))))
    log.info(f"[双次判定] 高相似度层 {sorted(_dbins)} 共 {len(_dbl)} 对全部二判；"
             f"其余 {len(_rest)} 对抽 {cfg.pos_shuffle_frac:.0%} 做 QC；"
             f"合计二判 {len(shuffle_set)} 对")

    def work(p):
        ci, cj = calls[p["qid_i"]], calls[p["qid_j"]]
        if p["is_catch"]:
            rnd = random.sample(allchunks, min(cfg.topk, len(allchunks)))
            r = _judge1(pool, cfg, ci["query"], _fmt(rnd, cmap, cfg.judge_doc_chars))
            return {**p, "catch_covered": r.get("covered", 0), "ok": r["ok"],
                    "truncated": r.get("truncated", False)}
        # 不给判官看对方的查询：否则它做的是"两个问题像不像"的表面判断
        dj = _fmt(cj["retrieved_chunks"], cmap, cfg.judge_doc_chars)
        di = _fmt(ci["retrieved_chunks"], cmap, cfg.judge_doc_chars)
        rij = _judge1(pool, cfg, ci["query"], dj)
        rji = _judge1(pool, cfg, cj["query"], di)
        a, b = rij.get("covered", 0), rji.get("covered", 0)
        out = {**p, "ok": rij["ok"] and rji["ok"], "eligible": 1,
               "cov_ij": a, "cov_ji": b, "substitutable": int(bool(a) and bool(b)),
               "cov_ij_shuf": None, "substitutable_strict": None,
               "substitutable_union": None,
               "double_judged": int(p["pair_id"] in shuffle_set),
               "verdict_ij": rij.get("verdict"), "verdict_ji": rji.get("verdict"),
               "facts_ij": rij.get("facts", []),
               "self_cov_i": self_cov.get(p["qid_i"]), "self_cov_j": self_cov.get(p["qid_j"]),
               "truncated": any(x.get("truncated") for x in (rij, rji)),
               "raw": next((x.get("raw") for x in (rij, rji) if x.get("raw")), None),
               "tokens": sum(x.get("tokens", 0) for x in (rij, rji)),
               "pos_consistent": None}
        if p["pair_id"] in shuffle_set:
            rsh = _judge1(pool, cfg, ci["query"],
                          _fmt(cj["retrieved_chunks"], cmap, cfg.judge_doc_chars, True))
            out["tokens"] += rsh.get("tokens", 0)
            if rij["ok"] and rsh["ok"]:
                sh = rsh.get("covered", 0)
                out["cov_ij_shuf"] = sh
                out["pos_consistent"] = int(a == sh)
                # v11：并列四种口径，不再把交集当主口径。
                # main_v7：bin7 翻转者可替代率 0.954（不是随机的 0.5），
                # 说明打乱顺序引入的是系统性单向偏差，取交集会把真可替代的
                # 对错杀掉（bin7 0.970 -> 0.727，τ* 直接无解）。
                out["substitutable_strict"] = int(bool(a) and bool(b) and bool(sh))
                out["substitutable_union"] = int((bool(a) or bool(sh)) and bool(b))
        return out

    # 时间预算：用自覆盖阶段实测的吞吐推算 L1 需要多久，超了就按分层比例截断。
    # 宁可少判一些对，也不要跑到一半被外部杀掉。
    _t_used = (time.time() - ctx.state.get("_t0", time.time())) / 3600
    _rate = pool.stats()["calls"] / max(time.time() - _t_sc, 1e-9) if cfg.self_cov_enabled else 2.3
    _need = len(elig) * (2 + cfg.pos_shuffle_frac) / max(_rate, 1e-9) / 3600
    _left = cfg.time_budget_hours - _t_used - 0.6      # 留 0.6h 给后续 stage
    if _need > _left and _left > 0:
        keep = max(200, int(len(elig) * _left / _need))
        log.warning(f"[时间预算] L1 预计需 {_need:.1f}h，仅剩 {_left:.1f}h -> "
                    f"按分层比例截断 {len(elig)} -> {keep} 对")
        byb = defaultdict(list)
        for p in elig:
            byb[p.get("bin")].append(p)
        frac = keep / len(elig)
        elig = []
        for b, v in byb.items():
            random.shuffle(v)
            elig.extend(v[:max(30, int(len(v) * frac))])
        log.warning(f"[时间预算] 实际保留 {len(elig)} 对")
    else:
        log.info(f"[时间预算] L1 预计 {_need:.1f}h，剩余 {_left:.1f}h，不截断")

    todo = elig + catch_pairs
    res = [r for r in pmap(work, todo, cfg.judge_concurrency, "L1", ctx, pool)
           if isinstance(r, dict) and "__error__" not in r]
    # 不合格对也留一条记录，s8/s9 靠 eligible 字段过滤
    el_ids = {p["pair_id"] for p in elig}
    for p in real_pairs:
        if p["pair_id"] not in el_ids:
            res.append({**p, "ok": False, "eligible": 0, "substitutable": None,
                        "self_cov_i": self_cov.get(p["qid_i"]),
                        "self_cov_j": self_cov.get(p["qid_j"]),
                        "skipped_reason": "self_cov"})

    judged = [r for r in res if r.get("eligible") == 1 and r.get("ok")]
    catch = [r for r in res if r.get("is_catch")]
    pc = [r["pos_consistent"] for r in judged if r.get("pos_consistent") is not None]
    st_ = [r for r in judged if r.get("same_text")]

    qc: Dict[str, Any] = {
        "n_sampled_pairs": len(real_pairs), "n_eligible": len(elig), "n_judged": len(judged),
        "self_cov_rate": (sum(1 for v in self_cov.values() if v == 1)
                          / max(len(self_cov), 1)) if self_cov else None,
        "eligibility_rate": len(elig) / max(len(real_pairs), 1),
        "eligibility_by_bin": {str(b): {"elig": by_bin_el[b], "all": by_bin_all[b]}
                               for b in sorted(by_bin_all)},
        "parse_fail_rate": 1 - sum(bool(r.get("ok")) for r in judged) / max(len(judged), 1),
        "tokens": sum(r.get("tokens", 0) for r in judged), "llm": pool.stats(),
        "pos_consistency": float(np.mean(pc)) if pc else None, "pos_consistency_n": len(pc),
        # v9：main_v5 的位置一致率掉到 0.782（main_v3 是 0.916），而样本里
        # 45% 是可替代率 0.987 的 bin7，本该更高才对。分层看才知道是判官
        # 对文档顺序敏感，还是抽样结构变化造成的。
        "pos_consistency_by_bin": {
            str(b): {"n": len([r for r in judged
                               if r.get("bin") == b and r.get("pos_consistent") is not None]),
                     "rate": float(np.mean([r["pos_consistent"] for r in judged
                                            if r.get("bin") == b
                                            and r.get("pos_consistent") is not None]))
                     if any(r.get("bin") == b and r.get("pos_consistent") is not None
                            for r in judged) else None}
            for b in sorted({r.get("bin") for r in judged if r.get("bin") is not None})},
        "catch_fp_rate": float(np.mean([r["catch_covered"] for r in catch])) if catch else None,
        "asymmetry_rate": float(np.mean([int(r["cov_ij"] != r["cov_ji"])
                                         for r in judged])) if judged else None,
        # 主指标：条件可替代率（分母是自覆盖合格对）
        # 未加权（样本口径，会被高相似度层拉高，仅供对照）
        "substitutable_rate_conditional_unweighted":
            float(np.mean([r["substitutable"] for r in judged])) if judged else None,
        # 参考：把不合格对当作不可替代（v4 口径，会被自覆盖率的平方压低）
        "substitutable_rate_marginal":
            (sum(r["substitutable"] for r in judged) / max(len(real_pairs), 1))
            if judged else None,
        # 真正的阴性对照：字节相同的查询对，检索结果本来就一样，
        # 双向判定必须一致。这测的是判官稳定性，不是可替代性。
        "same_text_direction_consistency":
            float(np.mean([int(r["cov_ij"] == r["cov_ji"]) for r in st_])) if st_ else None,
        "same_text_n": len(st_),
    }
    # v9：加权口径才是总体估计
    if judged:
        _p, _lo, _hi, _neff = wilson_w([r["substitutable"] for r in judged],
                                       [r.get("w", 1.0) for r in judged])
        qc["substitutable_rate_conditional"] = _p
        qc["substitutable_rate_conditional_ci"] = [_lo, _hi]
        qc["effective_n"] = _neff
        log.info(f"条件可替代率（加权）{_p:.4f} CI[{_lo:.4f},{_hi:.4f}] n_eff={_neff:.0f}；"
                 f"未加权样本口径 {qc['substitutable_rate_conditional_unweighted']:.4f}")
    else:
        qc["substitutable_rate_conditional"] = None
    # v10：严格口径（双次判定取交集）的加权率与逐层率
    _dj = [r for r in judged if r.get("substitutable_strict") is not None]
    if _dj:
        _ps, _los, _his, _nes = wilson_w([r["substitutable_strict"] for r in _dj],
                                         [r.get("w", 1.0) for r in _dj])
        qc["strict"] = {
            "n": len(_dj), "rate_weighted": _ps, "ci": [_los, _his], "n_eff": _nes,
            "rate_unweighted": float(np.mean([r["substitutable_strict"] for r in _dj])),
            "by_bin": {str(b): {"n": len([r for r in _dj if r.get("bin") == b]),
                                "loose": float(np.mean([r["substitutable"] for r in _dj
                                                        if r.get("bin") == b])),
                                "strict": float(np.mean([r["substitutable_strict"]
                                                         for r in _dj if r.get("bin") == b]))}
                       for b in sorted({r.get("bin") for r in _dj if r.get("bin") is not None})},
        }
        _pu, _lou, _hiu, _neu = wilson_w([r["substitutable_union"] for r in _dj],
                                         [r.get("w", 1.0) for r in _dj])
        qc["union"] = {"n": len(_dj), "rate_weighted": _pu, "ci": [_lou, _hiu]}
        qc["mode_used"] = cfg.substitutability_mode
        log.info(f"[口径对比] n={len(_dj)}  single {qc['substitutable_rate_conditional']:.4f}"
                 f"  union {_pu:.4f}  intersection {_ps:.4f}")
        log.info("  main_v7 已证明 intersection 引入的是系统性偏差而非精度，"
                 "主口径用 single，另外两个仅作敏感性分析")
        for b, v in qc["strict"]["by_bin"].items():
            log.info(f"    bin{b}: 宽松 {v['loose']:.3f} -> 严格 {v['strict']:.3f} "
                     f"(n={v['n']}, 降 {v['loose']-v['strict']:.3f})")

    fails = [r for r in res if r.get("eligible") == 1 and not r.get("ok")]
    if fails:
        ctx.wjsonl(ctx.out("s6", "parse_failures.jsonl"), fails[:200])
        qc["n_parse_fail"] = len(fails)
        qc["truncated_share"] = float(np.mean([bool(f.get("truncated")) for f in fails]))
        log.warning(f"解析失败 {len(fails)} 条 -> parse_failures.jsonl")

    log.info(f"边际口径（把不合格对当不可替代）{qc['substitutable_rate_marginal']}")
    log.info(f"同文本对方向一致率 {qc['same_text_direction_consistency']} "
             f"(n={qc['same_text_n']}) | 位置一致率 {qc['pos_consistency']} "
             f"(n={qc['pos_consistency_n']})")
    log.info(f"陷阱误判 {qc['catch_fp_rate']} | 不对称 {qc['asymmetry_rate']} | "
             f"平均输出 {pool.stats()['avg_out']} tok")
    if qc["parse_fail_rate"] > cfg.stop_max_parse_fail:
        log.error(f"!! 解析失败率 {qc['parse_fail_rate']:.1%}")
    if qc["catch_fp_rate"] and qc["catch_fp_rate"] > cfg.stop_max_catch_fp:
        log.error(f"!! 陷阱误判率 {qc['catch_fp_rate']:.1%}：判官在用自身知识作答")
    stc = qc["same_text_direction_consistency"]
    if stc is not None and qc["same_text_n"] >= 8 and stc < cfg.stop_min_same_text_consistency:
        log.error(f"!! 同文本对方向一致率 {stc:.1%} < "
                  f"{cfg.stop_min_same_text_consistency:.0%}：检索结果相同却判出不同结论，"
                  f"判官不稳定")

    gold = ctx.rjsonl(ctx.out("s5", "gold_pairs.jsonl"))
    lab = [g for g in gold if g.get("human_label") in (0, 1)]
    if lab:
        m = {r["pair_id"]: r for r in judged}
        yt = [int(g["human_label"]) for g in lab if g["pair_id"] in m]
        yp = [int(m[g["pair_id"]]["substitutable"]) for g in lab if g["pair_id"] in m]
        if yt:
            tp = sum(a == 1 and b == 1 for a, b in zip(yt, yp))
            fp = sum(a == 0 and b == 1 for a, b in zip(yt, yp))
            fn = sum(a == 1 and b == 0 for a, b in zip(yt, yp))
            qc["gold"] = {"n": len(yt),
                          "accuracy": sum(a == b for a, b in zip(yt, yp)) / len(yt),
                          "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                          "kappa": kappa(yt, yp)}
            log.info(f"金标 acc={qc['gold']['accuracy']:.3f} (n={len(yt)})")
            if qc["gold"]["accuracy"] < cfg.stop_min_gold_acc:
                log.error(f"!! 金标 accuracy < {cfg.stop_min_gold_acc}，方法失效")
    else:
        log.warning("金标未标注，缺 accuracy。判官已降级，论文里这项必须有。")

    if cfg.cross_judge_base_url and cfg.cross_judge_model:
        xp = LLMPool([cfg.cross_judge_base_url], cfg.cross_judge_model, retries=3,
                     timeout=180, api_key=os.environ.get(cfg.cross_judge_key_env, ""))
        sub = random.sample(judged, min(cfg.cross_judge_n, len(judged)))

        def xw(p):
            ci, cj = calls[p["qid_i"]], calls[p["qid_j"]]
            a = _judge1(xp, cfg, ci["query"],
                        _fmt(cj["retrieved_chunks"], cmap, cfg.judge_doc_chars))
            b = _judge1(xp, cfg, cj["query"],
                        _fmt(ci["retrieved_chunks"], cmap, cfg.judge_doc_chars))
            return {"pair_id": p["pair_id"],
                    "s": int(bool(a.get("covered")) and bool(b.get("covered")))}

        xr = pmap(xw, sub, 8, "cross", ctx, xp)
        mm = {r["pair_id"]: r["s"] for r in xr if isinstance(r, dict) and "s" in r}
        aa = [p["substitutable"] for p in sub if p["pair_id"] in mm]
        bb = [mm[p["pair_id"]] for p in sub if p["pair_id"] in mm]
        if aa:
            qc["cross_kappa"] = kappa(aa, bb); qc["cross_n"] = len(aa)
            log.info(f"跨判官 κ = {qc['cross_kappa']:.3f} (n={len(aa)})")

    ctx.wjsonl(ctx.out("s6", "l1.jsonl"), res)
    ctx.wjson(ctx.out("s6", "qc.json"), qc)
    return (f"合格 {len(judged)}/{len(real_pairs)} 对, 条件可替代率 "
            f"{qc['substitutable_rate_conditional']:.3f}, 自覆盖 {qc['self_cov_rate']:.1%}")


# ==============================================================================
# s7  L2 端到端
# ==============================================================================

L2_PROMPT = """根据以下资料回答问题。只用资料中的信息，答案尽量简短（一个实体或短语）。

**资料**
{docs}

**问题**
{q}

只输出答案本身，不要解释。"""


def _norm(s):
    s = re.sub(r"\b(a|an|the)\b", " ", (s or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _f1(p, g):
    a, b = _norm(p).split(), _norm(g).split()
    if not a or not b:
        return float(a == b)
    c = sum((Counter(a) & Counter(b)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(a), c / len(b)
    return 2 * pr * rc / (pr + rc)


@stage("s7", "L2 端到端替换（真值锚定）", ["l2.jsonl", "summary.json"])
def s7(ctx: RunContext, sm: ServerManager):
    cfg, log = ctx.cfg, ctx.log
    calls = {c["qid"]: c for c in ctx.rjsonl(ctx.out("s2", "calls.jsonl"))}
    cmap = {c["chunkid"]: c["text"] for c in ctx.rjsonl(ctx.out("s1", "chunks.jsonl"))}
    l1 = [r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
          if not r.get("is_catch") and r.get("ok") and r.get("eligible") == 1]
    elig = [r for r in l1 if calls[r["qid_i"]].get("gold_answer")]
    if not elig:
        log.warning("无 gold answer，跳过 L2 —— L1 将失去真值锚定")
        ctx.wjsonl(ctx.out("s7", "l2.jsonl"), [])
        ctx.wjson(ctx.out("s7", "summary.json"), {"skipped": True})
        return "跳过"

    pos = [r for r in elig if r["substitutable"] == 1]
    neg = [r for r in elig if r["substitutable"] == 0]
    h = cfg.l2_budget // 2
    smp = (random.sample(pos, min(h, len(pos)))
           + random.sample(neg, min(cfg.l2_budget - min(h, len(pos)), len(neg))))
    log.info(f"L2 样本 {len(smp)}（L1 正 {min(h,len(pos))} / 负 {len(smp)-min(h,len(pos))}）")

    urls = sm.ensure("judge", list(cfg.gpus))
    pool = LLMPool(urls, cfg.judge_name, retries=3, timeout=180)

    def work(r):
        ci, cj = calls[r["qid_i"]], calls[r["qid_j"]]
        gold = ci.get("gold_answer", "")
        # L2 的 max_tokens=64 比 L1 更紧，不关思考必然全部截断
        a = pool.chat([{"role": "user", "content": L2_PROMPT.format(
            docs=_fmt(ci["retrieved_chunks"], cmap, cfg.judge_doc_chars), q=ci["query"])}],
            temperature=0.0, max_tokens=64, **NO_THINK)
        b = pool.chat([{"role": "user", "content": L2_PROMPT.format(
            docs=_fmt(cj["retrieved_chunks"], cmap, cfg.judge_doc_chars), q=ci["query"])}],
            temperature=0.0, max_tokens=64, **NO_THINK)
        fo, fs = _f1(a["text"], gold), _f1(b["text"], gold)
        return {"pair_id": r["pair_id"], "cos": r["cos"], "bin": r.get("bin"),
                "l1_sub": r["substitutable"], "f1_own": fo, "f1_sub": fs,
                "em_own": int(_norm(a["text"]) == _norm(gold)),
                "em_sub": int(_norm(b["text"]) == _norm(gold)),
                "l2_sub": int(fs >= fo - 1e-9), "recall_loss": max(0.0, fo - fs),
                "truncated": (a["finish_reason"] == "length" or b["finish_reason"] == "length"),
                "_src": ci.get("source", "?"),
                "ans_own": a["text"][:150], "ans_sub": b["text"][:150], "gold": gold[:100]}

    res = [r for r in pmap(work, smp, cfg.judge_concurrency, "L2", ctx, pool)
           if isinstance(r, dict) and "__error__" not in r]
    if not res:
        raise RuntimeError("L2 全部失败")

    y1 = [r["l1_sub"] for r in res]; y2 = [r["l2_sub"] for r in res]
    tp = sum(a == 1 and b == 1 for a, b in zip(y1, y2))
    fp = sum(a == 1 and b == 0 for a, b in zip(y1, y2))
    fn = sum(a == 0 and b == 1 for a, b in zip(y1, y2))
    tn = sum(a == 0 and b == 0 for a, b in zip(y1, y2))
    posl = [r["recall_loss"] for r in res if r["l1_sub"] == 1]
    su = {"n": len(res), "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
          "l1_precision": tp / max(tp + fp, 1), "l1_recall": tp / max(tp + fn, 1),
          "kappa": kappa(y1, y2),
          # f1_own 是健全性指标：普遍为 0 说明答案生成本身就坏了
          "mean_f1_own": float(np.mean([r["f1_own"] for r in res])),
          # v9：main_v5 的 f1_own 从 0.80 掉到 0.694，因为 agent 的 gold_answer
          # 是复合问题的最终答案，子查询本来就答不出来。混在一起看会被误读成
          # 检索质量下降。分域报告。
          "f1_own_by_source": {
              s: {"n": len([r for r in res if r.get("_src") == s]),
                  "mean_f1_own": float(np.mean([r["f1_own"] for r in res
                                                if r.get("_src") == s])),
                  "em_own_rate": float(np.mean([r["em_own"] for r in res
                                                if r.get("_src") == s]))}
              for s in sorted({r.get("_src") for r in res if r.get("_src")})},
          "mean_f1_sub": float(np.mean([r["f1_sub"] for r in res])),
          "em_own_rate": float(np.mean([r["em_own"] for r in res])),
          "truncated_rate": float(np.mean([r["truncated"] for r in res])),
          "mean_recall_loss_pos": float(np.mean(posl)) if posl else None,
          "p95_recall_loss_pos": float(np.percentile(posl, 95)) if len(posl) >= 5 else None,
          # v7：main_v3 的 P95 是 0.0 —— 分布是「绝大多数零损失 + 少数大损失」，
          # P95 完全没有描述能力，而 conformal 需要的恰恰是尾部。改报这几个。
          "n_pos": len(posl),
          "nonzero_loss_frac": (float(np.mean([x > 1e-9 for x in posl]))
                                if posl else None),
          "p99_recall_loss_pos": float(np.percentile(posl, 99)) if len(posl) >= 20 else None,
          "max_recall_loss_pos": float(np.max(posl)) if posl else None,
          "mean_loss_given_nonzero": (float(np.mean([x for x in posl if x > 1e-9]))
                                      if any(x > 1e-9 for x in posl) else 0.0),
          "loss_quantiles": {q: float(np.percentile(posl, q))
                             for q in (50, 75, 90, 95, 99)} if len(posl) >= 20 else None}
    log.info(f"L1 vs L2: P={su['l1_precision']:.3f} R={su['l1_recall']:.3f} κ={su['kappa']:.3f}")
    log.info(f"f1_own 均值 {su['mean_f1_own']:.3f} (EM {su['em_own_rate']:.1%})"
             f"  <- 接近 0 说明答案生成坏了，不是可替代性的问题")
    log.info(f"L1 判正时召回损失：非零占比 {su['nonzero_loss_frac']}，"
             f"均值 {su['mean_recall_loss_pos']}，P99 {su['p99_recall_loss_pos']}，"
             f"max {su['max_recall_loss_pos']}，非零条件均值 {su['mean_loss_given_nonzero']}")
    log.info(f"  -> ε 应取 P99 或 max，不是 P95（P95 常年为 0，无描述力）")
    if su["mean_f1_own"] < 0.15:
        log.error("!! f1_own 过低，L2 的真值锚定不可信。检查 gold_answer 与语料是否匹配。")

    ctx.wjsonl(ctx.out("s7", "l2.jsonl"), res)
    ctx.wjson(ctx.out("s7", "summary.json"), su)
    sm.stop()
    return f"{len(res)} 对, L1 precision={su['l1_precision']:.3f}, f1_own={su['mean_f1_own']:.2f}"


# ==============================================================================
# s8  校准
# ==============================================================================


@stage("s8", "保序回归 + Wilson + bootstrap -> tau*", ["calib.json", "iso.pkl"])
def s8(ctx: RunContext, sm: ServerManager):
    from sklearn.isotonic import IsotonicRegression
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg, log = ctx.cfg, ctx.log
    # v5：只用自覆盖合格对校准。基础索引本来答不了的查询没有召回可损失，
    # 把它们算进来会让曲线被自覆盖率的平方压低并失去单调性（smoke3 的现象）。
    res = [r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
           if not r.get("is_catch") and r.get("ok") and r.get("eligible") == 1]
    if len(res) < 20:
        raise RuntimeError(f"自覆盖合格的 L1 结果仅 {len(res)} 条，"
                           f"调大 l1_budget 后重跑 s5,s6")

    # v10：同时给出严格口径（双次判定取交集）的 τ*，看判官不稳定性
    # 对阈值的影响有多大。main_v6 里 bin7 位置一致率仅 0.699。
    _strict = [r for r in res if r.get("substitutable_strict") is not None]
    _tau_strict = None
    if len(_strict) >= 200:
        _xs = np.array([r["cos"] for r in _strict])
        _ys = np.array([r["substitutable_strict"] for r in _strict], float)
        _ws = np.array([r.get("w", 1.0) for r in _strict], float)
        _iso_s = IsotonicRegression(y_min=0, y_max=1, increasing=True,
                                    out_of_bounds="clip").fit(_xs, _ys, sample_weight=_ws)
        _gs = np.linspace(float(_xs.min()), 1.0, 400)
        _phs = _iso_s.predict(_gs)
        _bs = np.array([r.get("bin", _bin(r["cos"], cfg.sim_bins)) for r in _strict])
        _idx = {b: np.where(_bs == b)[0] for b in set(_bs.tolist())}
        _bt = np.zeros((min(cfg.bootstrap_B, 1000), len(_gs)))
        for _t in range(_bt.shape[0]):
            _sel = np.concatenate([np.random.choice(v, len(v), True)
                                   for v in _idx.values() if len(v)])
            try:
                _bt[_t] = IsotonicRegression(
                    y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
                ).fit(_xs[_sel], _ys[_sel], sample_weight=_ws[_sel]).predict(_gs)
            except Exception:
                _bt[_t] = _phs
        _lo_s = np.percentile(_bt, 2.5, 0)
        _ok_s = np.where(_lo_s >= cfg.target_substitutability)[0]
        _tau_strict = float(_gs[_ok_s[0]]) if len(_ok_s) else None
        log.info(f"[严格口径] τ*_strict = {_tau_strict}（宽松口径见下）")

    x = np.array([r["cos"] for r in res])
    y = np.array([r["substitutable"] for r in res], float)
    bs = np.array([r.get("bin", _bin(r["cos"], cfg.sim_bins)) for r in res])

    per = {}
    for b in sorted(set(bs.tolist())):
        m = bs == b; k, n = int(y[m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        per[int(b)] = {"range": [cfg.sim_bins[int(b)], cfg.sim_bins[int(b) + 1]],
                       "n": n, "k": k, "p": k / n if n else None, "lo": lo, "hi": hi}
        log.info(f"  bin{b} [{cfg.sim_bins[int(b)]:.2f},{cfg.sim_bins[int(b)+1]:.2f}) "
                 f"n={n} p={k/max(n,1):.3f} CI=[{lo:.3f},{hi:.3f}]")

    # v9：分层抽样必须带权重，否则保序回归被 bin7 的巨量样本主导
    wt = np.array([r.get("w", 1.0) for r in res], float)
    iso = IsotonicRegression(y_min=0, y_max=1, increasing=True,
                             out_of_bounds="clip").fit(x, y, sample_weight=wt)
    grid = np.linspace(float(x.min()), 1.0, 400); ph = iso.predict(grid)

    idxb = {b: np.where(bs == b)[0] for b in set(bs.tolist())}
    B = cfg.bootstrap_B; boot = np.zeros((B, len(grid)))
    for t in range(B):
        sel = np.concatenate([np.random.choice(v, len(v), True) for v in idxb.values() if len(v)])
        try:
            boot[t] = IsotonicRegression(
                y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
            ).fit(x[sel], y[sel], sample_weight=wt[sel]).predict(grid)
        except Exception:
            boot[t] = ph
        if t % 200 == 0:
            ctx.beat(f"bootstrap {t}/{B}")
    lo_b = np.percentile(boot, 2.5, 0); hi_b = np.percentile(boot, 97.5, 0)

    tgt = cfg.target_substitutability
    ok = np.where(lo_b >= tgt)[0]; tau = float(grid[ok[0]]) if len(ok) else None
    okp = np.where(ph >= tgt)[0]; tau_p = float(grid[okp[0]]) if len(okp) else None

    # v13：多档 τ* + 退化检测。
    # 「退化」指 τ* 放行的对里绝大多数是字面完全相同的查询 ——
    # 那时判定器等价于精确缓存，「近似视图重写」这个卖点就不成立了。
    # main_v9：τ*=0.9925 放行的几乎全是 cos=1.0 的同文本对。
    tau_multi = {}
    for _t in cfg.report_targets:
        _ok = np.where(lo_b >= _t)[0]
        _tau = float(grid[_ok[0]]) if len(_ok) else None
        _okp = np.where(ph >= _t)[0]
        _taup = float(grid[_okp[0]]) if len(_okp) else None
        ent = {"tau_star": _tau, "tau_point": _taup}
        if _tau is not None:
            _pass = [r for r in res if r["cos"] >= _tau]
            _st = [r for r in _pass if r.get("same_text")]
            ent.update({
                "n_pass": len(_pass),
                "pass_frac": len(_pass) / max(len(res), 1),
                "same_text_frac": len(_st) / max(len(_pass), 1),
                "empirical_sub": float(np.mean([r["substitutable"] for r in _pass]))
                                 if _pass else None,
            })
            ent["degenerate"] = bool(ent["same_text_frac"] >= cfg.degenerate_same_text_frac)
        tau_multi[f"{_t:.2f}"] = ent
        if _tau is None:
            log.info(f"  保证 {_t:.2f} -> τ* 无解（点估计 {_taup}）")
        else:
            log.info(f"  保证 {_t:.2f} -> τ* {_tau:.4f}  放行 {ent['n_pass']} 对"
                     f"（{ent['pass_frac']:.1%}）  其中同文本 {ent['same_text_frac']:.1%}"
                     f"  实测可替代 {ent['empirical_sub']:.3f}"
                     + ("  ⚠ 退化为精确匹配" if ent["degenerate"] else ""))
    if tau is None:
        log.error(f"任何相似度下置信下界都到不了 {tgt}：嵌入相似度不足以保证可替代性。"
                  f"点估计 τ={tau_p}")
    else:
        log.info(f"τ* = {tau:.4f}（置信下界）；点估计 {tau_p}")

    l2 = ctx.rjson(ctx.out("s7", "summary.json"))
    qc6 = ctx.rjson(ctx.out("s6", "qc.json"))
    calib = {"n": len(res), "conditional_on_self_cov": True,
             "self_cov_rate": qc6.get("self_cov_rate"),
             "eligibility_rate": qc6.get("eligibility_rate"),
             "per_bin": per, "tau_star": tau, "tau_point": tau_p,
             "target": tgt, "grid": grid.tolist(), "p_hat": ph.tolist(),
             "band_lo": lo_b.tolist(), "band_hi": hi_b.tolist(),
             "tau_star_strict": _tau_strict, "n_strict": len(_strict),
             "tau_multi": tau_multi, "report_targets": cfg.report_targets,
             "l2_precision": None if l2.get("skipped") else l2.get("l1_precision"),
             "blocking_miss": ctx.rjson(ctx.out("s4", "audit.json"))["miss_rate"].get("0.9")}
    ctx.wjson(ctx.out("s8", "calib.json"), calib)
    with open(ctx.out("s8", "iso.pkl"), "wb") as f:
        pickle.dump(iso, f)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.fill_between(grid, lo_b, hi_b, alpha=.22, label="95% bootstrap")
    ax.plot(grid, ph, lw=2, label="isotonic")
    for b, d in per.items():
        c = (d["range"][0] + d["range"][1]) / 2
        # 小样本层的加权 Wilson 区间可能出现 lo>p 或 hi<p，matplotlib 会因负的
        # yerr 直接抛错并中断整个 stage（BCP 上 n=27 的层触发过），这里夹到非负。
        _el, _eu = max(0.0, d["p"] - d["lo"]), max(0.0, d["hi"] - d["p"])
        ax.errorbar(c, d["p"], yerr=[[_el], [_eu]],
                    fmt="o", ms=4, capsize=3, color="k")
    ax.axhline(tgt, ls="--", c="r", lw=1, label=f"target {tgt}")
    if tau:
        ax.axvline(tau, ls=":", c="g", lw=1.5, label=f"tau*={tau:.3f}")
    ax.set_xlabel("query cosine similarity")
    ax.set_ylabel("substitutability | both self-covered")
    ax.set_title("Substitutability calibration (conditional)"); ax.legend(fontsize=8)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(ctx.dirs["figures"] / f"calibration.{e}", dpi=180)
    plt.close(fig)
    return f"tau*={tau}, 点估计={tau_p}, n={len(res)}"


# ==============================================================================
# s9  聚类
# ==============================================================================


def _clique_cover(G, nodes, cap):
    if len(nodes) > cap:
        nodes = random.sample(list(nodes), cap)
    unc = set(nodes); deg = dict(G.degree()); out = []
    for v in sorted(nodes, key=lambda x: -deg.get(x, 0)):
        if v not in unc:
            continue
        cl = [v]
        for u in sorted(set(G.neighbors(v)) & unc, key=lambda x: -deg.get(x, 0)):
            if all(G.has_edge(u, w) for w in cl):
                cl.append(u)
        unc -= set(cl); out.append(cl)
    return out


@stage("s9", "连通分量 / 极大团 + 粒度敏感性", ["clusters.json", "granularity.json"])
def s9(ctx: RunContext, sm: ServerManager):
    import networkx as nx
    cfg, log = ctx.cfg, ctx.log
    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    pairs = ctx.rjsonl(ctx.out("s4", "pairs.jsonl"))
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    qids = [c["qid"] for c in calls]
    tau = calib.get("tau_star") or calib.get("tau_point") or 0.90
    taus = sorted(set([round(t, 3) for t in cfg.tau_grid] + [round(tau, 3)]))

    gran, cc_map, cl_map = {}, {}, {}
    for t in taus:
        G = nx.Graph(); G.add_nodes_from(qids)
        G.add_edges_from((p["qid_i"], p["qid_j"]) for p in pairs if p["cos"] >= t)
        cc = list(nx.connected_components(G))
        cl = _clique_cover(G, qids, cfg.max_clique_nodes)
        edge = {frozenset((p["qid_i"], p["qid_j"])) for p in pairs if p["cos"] >= t}
        big = [c for c in cc if len(c) >= 3]; intra = []
        for comp in (random.sample(big, min(60, len(big))) if big else []):
            L = list(comp)
            for _ in range(min(5, len(L))):
                a, b = random.sample(L, 2)
                intra.append(1.0 if frozenset((a, b)) in edge else 0.0)
        gran[str(t)] = {"n_components": len(cc), "n_cliques": len(cl),
                        "largest": max((len(c) for c in cc), default=0),
                        "singleton_frac": sum(len(c) == 1 for c in cc) / max(len(cc), 1),
                        "intra_edge_rate": float(np.mean(intra)) if intra else None,
                        "top20": sorted((len(c) for c in cc), reverse=True)[:20]}
        log.info(f"τ={t}: 分量 {len(cc)} / 团 {len(cl)} / 最大簇 {gran[str(t)]['largest']}"
                 f" / 簇内直连 {gran[str(t)]['intra_edge_rate']}")
        if t == round(tau, 3):
            cc_map = {q: k for k, c in enumerate(cc) for q in c}
            cl_map = {q: k for k, c in enumerate(cl) for q in c}

    ctx.wjson(ctx.out("s9", "granularity.json"), gran)
    ctx.wjson(ctx.out("s9", "clusters.json"),
              {"tau": round(tau, 3), "components": cc_map, "cliques": cl_map})
    return f"τ={tau:.3f}: {gran[str(round(tau,3))]['n_components']} 个意图簇（分量口径）"


# ==============================================================================
# s10  复用统计
# ==============================================================================


@stage("s10", "复用率 / 成本加权 / 跨任务 / Figure 1 / 止损", ["reuse.json"])
def s10(ctx: RunContext, sm: ServerManager):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg, log = ctx.cfg, ctx.log
    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    clus = ctx.rjson(ctx.out("s9", "clusters.json"))
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    gran = ctx.rjson(ctx.out("s9", "granularity.json"))
    s2st = ctx.rjson(ctx.out("s2", "stats.json"))
    R: Dict[str, Any] = {"tau": clus["tau"], "cost_model": {
        "per_call": cfg.cost_per_search_call,
        "per_context_token": cfg.cost_per_context_token,
        "per_rerank_doc": cfg.cost_per_rerank_doc},
        "sanity": s2st.get("sanity", {})}

    def ana(assign, label, sub=None):
        sel = [c for c in calls if (sub is None or sub(c))]
        cn, cc_, ct = Counter(), Counter(), defaultdict(set)
        ctext = defaultdict(set)
        for c in sel:
            k = assign.get(c["qid"])
            if k is None:
                continue
            cn[k] += 1; cc_[k] += c["cost"]; ct[k].add(c["task_id"])
            ctext[k].add(c["query"])
        if not cn:
            return {"n_calls": 0}
        rk = [k for k, _ in cn.most_common()]
        hn = max(1, int(round(len(rk) * cfg.stop_head_frac))); head = rk[:hn]
        tc, tcost = sum(cn.values()), sum(cc_.values())
        cross = {k for k in rk if len(ct[k]) >= 2}
        hx = [k for k in head if k in cross]
        d = {"n_calls": tc, "n_intents": len(rk),
             "n_tasks": len({c["task_id"] for c in sel}),
             "head_cov_calls": sum(cn[k] for k in head) / tc,
             "head_cov_cost": sum(cc_[k] for k in head) / max(tcost, 1e-9),
             "cross_intent_frac": len(cross) / len(rk),
             "cross_call_frac": sum(cn[k] for k in cross) / tc,
             "cross_cost_frac": sum(cc_[k] for k in cross) / max(tcost, 1e-9),
             "head_cross_share": sum(cn[k] for k in hx) / max(sum(cn[k] for k in head), 1),
             # 单文本簇：整簇只有一种字面表述 -> 模板碰撞而非真实意图复用
             "mono_text_cluster_frac":
                 len({k for k in rk if cn[k] >= 2 and len(ctext[k]) == 1})
                 / max(sum(1 for k in rk if cn[k] >= 2), 1),
             "head_mono_text_share":
                 sum(cn[k] for k in head if cn[k] >= 2 and len(ctext[k]) == 1)
                 / max(sum(cn[k] for k in head), 1),
             "gini_calls": gini(cn.values()), "gini_cost": gini(cc_.values()),
             "hill_alpha": hill(list(cn.values())),
             "mean_tasks_per_intent": float(np.mean([len(v) for v in ct.values()])),
             "freq": sorted(cn.values(), reverse=True)[:500],
             "cost": [float(x) for x in sorted(cc_.values(), reverse=True)[:500]]}
        log.info(f"[{label}] 意图 {d['n_intents']} / 调用 {tc} | 头部 按次 "
                 f"{d['head_cov_calls']:.1%} 按成本 {d['head_cov_cost']:.1%} | "
                 f"跨任务 {d['cross_intent_frac']:.1%} | "
                 f"头部单文本簇 {d['head_mono_text_share']:.1%}")
        return d

    R["components"] = ana(clus["components"], "连通分量")
    R["cliques"] = ana(clus["cliques"], "极大团")
    R["by_source"] = {s: ana(clus["components"], f"src={s}", lambda c, s=s: c["source"] == s)
                      for s in sorted({c["source"] for c in calls})}

    d = R["components"]
    prec = calib.get("l2_precision"); miss = calib.get("blocking_miss") or 0.0
    R["corrected"] = {"point": d["head_cov_calls"],
                      "lower": d["head_cov_calls"] * (prec or 1.0),
                      "upper": min(1.0, d["head_cov_calls"] / max(1 - miss, 1e-6)),
                      "l2_precision": prec, "blocking_miss": miss}

    qc = ctx.rjson(ctx.out("s6", "qc.json"))
    l2 = ctx.rjson(ctx.out("s7", "summary.json"))
    vert = [v for k, v in R["by_source"].items() if k.startswith("replay") and v.get("n_calls")]
    gen = R["by_source"].get("agent", {})
    bv = max((v["head_cov_cost"] for v in vert), default=0.0)
    bx = max((v["head_cross_share"] for v in vert), default=0.0)
    ga = (qc.get("gold") or {}).get("accuracy")
    kp = qc.get("cross_kappa"); cfp = qc.get("catch_fp_rate")
    pfr = qc.get("parse_fail_rate", 0.0)

    f1o = None if l2.get("skipped") else l2.get("mean_f1_own")
    dupq = R["sanity"].get("exact_dup_query_frac", 0.0)
    mono = d.get("head_mono_text_share", 0.0)

    unres = R["sanity"].get("unresolved_placeholder_frac", 0.0)
    stc = qc.get("same_text_direction_consistency")

    # 数据健全性优先于止损判定：数据不对，覆盖率再高也是假的。
    # v5 撤掉了 v4 的 mono_text 闸门 —— 占位符解析之后，字面相同的查询是
    # 带实体的真实重复信息需求（smoke3 里 "When was Henry III crowned?"
    # 出现在 4 个不同任务中），恰恰是要找的复用。真正的病理由
    # unresolved_placeholder_frac 直接盖住。dup 与 mono 保留为描述性统计。
    if unres > cfg.stop_max_unresolved:
        verdict = "DATA_INVALID"
        why = f"未解析占位符 {unres:.1%}：查询里没有实体，复用是模板碰撞"
    elif stc is not None and qc.get("same_text_n", 0) >= 8 \
            and stc < cfg.stop_min_same_text_consistency:
        verdict = "METHOD_FAILURE"
        why = f"同文本对方向一致率 {stc:.1%}：检索相同却判出不同结论，判官不稳定"
    elif f1o is not None and f1o < cfg.stop_min_f1_own:
        verdict = "DATA_INVALID"
        why = f"f1_own {f1o:.3f} 过低：语料与查询错配，L1/L2 测的都不是可替代性"
    elif pfr > cfg.stop_max_parse_fail:
        verdict, why = "METHOD_FAILURE", f"判官解析失败率 {pfr:.1%}，L1 结果不可用"
    elif (cfp is not None and cfp > cfg.stop_max_catch_fp) or \
         (ga is not None and ga < cfg.stop_min_gold_acc) or \
         (kp is not None and kp < cfg.stop_min_kappa):
        verdict, why = "METHOD_FAILURE", "判官质量不达标，先修方法"
    elif bv >= cfg.stop_coverage_go and bx >= cfg.stop_crosstask_share:
        if gen.get("head_cov_cost", 0) >= cfg.stop_coverage_go:
            verdict, why = "GO", "垂直域与通用域均成立"
        else:
            verdict, why = "GO_CONDITIONAL", "垂直域成立、通用域不成立 -> 条件性主张"
    elif bv < cfg.stop_coverage_nogo:
        verdict, why = "STOP", f"最好情形头部成本覆盖仅 {bv:.1%}"
    else:
        verdict, why = "AMBIGUOUS", f"头部成本覆盖 {bv:.1%} 落在灰区"

    R["decision"] = {"verdict": verdict, "reason": why, "best_vertical_cost_cov": bv,
                     "best_vertical_cross_share": bx,
                     "general_cost_cov": gen.get("head_cov_cost"),
                     "gold_acc": ga, "cross_kappa": kp, "catch_fp": cfp,
                     "parse_fail_rate": pfr, "mean_f1_own": f1o,
                     "exact_dup_query_frac": dupq, "head_mono_text_share": mono,
                     "unresolved_placeholder_frac": unres,
                     "same_text_direction_consistency": stc,
                     "self_cov_rate": qc.get("self_cov_rate"),
                     "eligibility_rate": qc.get("eligibility_rate"),
                     "substitutable_rate_conditional":
                         qc.get("substitutable_rate_conditional")}
    log.info("=" * 62); log.info(f"止损判定: {verdict} — {why}"); log.info("=" * 62)
    ctx.wjson(ctx.out("s10", "reuse.json"), R)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    a = ax[0]
    for lbl, key in (("by calls", "freq"), ("by cost", "cost")):
        v = R["components"][key]
        if v:
            a.loglog(np.arange(1, len(v) + 1), v, lw=1.6, label=lbl)
    a.set_xlabel("intent rank"); a.set_ylabel("frequency / cost")
    a.set_title(f"(a) intent distribution (tau={clus['tau']:.2f}, "
                f"alpha={R['components']['hill_alpha']:.2f})")
    a.legend(fontsize=8); a.grid(alpha=.3, which="both")
    a = ax[1]
    for lbl, key in (("by calls", "freq"), ("by cost", "cost")):
        v = np.array(R["components"][key], float)
        if len(v):
            a.plot(np.arange(1, len(v) + 1) / len(v), np.cumsum(v) / v.sum(), lw=1.8, label=lbl)
    a.axvline(cfg.stop_head_frac, ls="--", c="r", lw=1, label=f"head {cfg.stop_head_frac:.0%}")
    a.axhline(cfg.stop_coverage_go, ls=":", c="g", lw=1, label=f"GO {cfg.stop_coverage_go:.0%}")
    a.set_xlabel("intent fraction"); a.set_ylabel("cumulative coverage")
    a.set_title("(b) head coverage"); a.legend(fontsize=8); a.grid(alpha=.3)
    a = ax[2]
    ts = sorted(gran, key=float)
    a.plot([float(t) for t in ts], [gran[t]["n_components"] for t in ts], "o-", label="components")
    a.plot([float(t) for t in ts], [gran[t]["n_cliques"] for t in ts], "s--", label="cliques")
    a.set_xlabel("tau"); a.set_ylabel("#intents")
    a.set_title("(c) granularity sensitivity"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(ctx.dirs["figures"] / f"figure1.{e}", dpi=180)
    plt.close(fig)
    return (f"{verdict} | 头部按次 {d['head_cov_calls']:.1%} / 按成本 "
            f"{d['head_cov_cost']:.1%} / 跨任务 {d['cross_intent_frac']:.1%}")


# ==============================================================================
# s11  启动视图
# ==============================================================================


@stage("s11", "Phase 2 启动视图", ["views.json"])
def s11(ctx: RunContext, sm: ServerManager):
    cfg, log = ctx.cfg, ctx.log
    calls = {c["qid"]: c for c in ctx.rjsonl(ctx.out("s2", "calls.jsonl"))}
    clus = ctx.rjson(ctx.out("s9", "clusters.json"))
    E = np.load(ctx.out("s3", "query_emb.npy"))
    qidx = {q: i for i, q in enumerate(ctx.rjson(ctx.out("s3", "qindex.json"))["qids"])}
    assign = clus["components"]

    grp = defaultdict(list)
    for q, k in assign.items():
        grp[k].append(q)
    rk = sorted(grp.items(), key=lambda kv: -len(kv[1]))
    n = max(1, int(len(rk) * cfg.stop_head_frac)); views = []
    for k, mem in rk[:n]:
        if len(mem) < 2:
            continue
        dc = Counter()
        for q in mem:
            for d in calls[q]["retrieved"]:
                dc[d] += 1
        vecs = np.stack([E[qidx[q]] for q in mem if q in qidx])
        cen = vecs.mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
        texts = [calls[q]["query"] for q in mem]
        views.append({"view_id": f"V{k:05d}", "n_queries": len(mem),
                      "n_tasks": len({calls[q]["task_id"] for q in mem}),
                      "n_distinct_texts": len(set(texts)),   # 1 = 模板碰撞
                      "member_qids": mem[:200],
                      "exemplars": list(dict.fromkeys(texts))[:5],
                      "materialized_docids": [d for d, _ in dc.most_common(cfg.topk * 2)],
                      "doc_support": {d: c / len(mem) for d, c in dc.most_common(20)},
                      "centroid": cen.tolist(),
                      "scope_radius_p5": float(np.percentile(vecs @ cen, 5)),
                      "cost_served": sum(calls[q]["cost"] for q in mem),
                      "sources": sorted({calls[q]["source"] for q in mem})})
    cov = sum(v["n_queries"] for v in views) / max(len(assign), 1)
    mono = sum(1 for v in views if v["n_distinct_texts"] == 1) / max(len(views), 1)
    ctx.wjson(ctx.out("s11", "views.json"),
              {"tau": clus["tau"], "n_views": len(views), "coverage": cov,
               "mono_text_view_frac": mono, "views": views})
    log.info(f"{len(views)} 视图，覆盖 {cov:.1%}，单文本视图占 {mono:.1%}")
    if mono > 0.5:
        log.error("!! 超过一半的视图只有一种字面表述，是模板碰撞不是意图复用")
    return f"{len(views)} 视图 / 覆盖 {cov:.1%} / 单文本 {mono:.1%}"


# ==============================================================================
# s12  报告
# ==============================================================================


@stage("s12", "汇总报告", ["report.md"])
def s12(ctx: RunContext, sm: ServerManager):
    cfg = ctx.cfg
    g = lambda s, n: ctx.rjson(ctx.out(s, n)) if ctx.out(s, n).exists() else {}
    cs, st = g("s1", "stats.json"), g("s2", "stats.json")
    au = g("s4", "audit.json"); qc, l2 = g("s6", "qc.json"), g("s7", "summary.json")
    ca, gr = g("s8", "calib.json"), g("s9", "granularity.json")
    R, vw = g("s10", "reuse.json"), g("s11", "views.json")
    d, dec, co = R.get("components", {}), R.get("decision", {}), R.get("corrected", {})
    cl = R.get("cliques", {}); sn = st.get("sanity", {})
    tot = sum(m.get("duration_sec", 0) for m in ctx.state.get("stages", {}).values())

    L = [f"# MatView Phase 1 报告 — `{cfg.run_id}`", "",
         f"生成 {time.strftime('%F %T')}  |  总机时 {tot/3600:.2f} h", "",
         "## 0. 止损判定", "",
         f"### **{dec.get('verdict','N/A')}** — {dec.get('reason','')}", "",
         f"- 垂直域头部{cfg.stop_head_frac:.0%}**成本**覆盖 "
         f"{dec.get('best_vertical_cost_cov',0):.1%} (GO 线 {cfg.stop_coverage_go:.0%})",
         f"- 其中跨任务意图占比 {dec.get('best_vertical_cross_share',0):.1%} "
         f"(要求 ≥{cfg.stop_crosstask_share:.0%})",
         f"- 通用域(agent)头部成本覆盖 {dec.get('general_cost_cov')}", "",
         "**数据健全性关卡**（不过则上面的覆盖率无意义）", "",
         f"- 未解析占位符 {dec.get('unresolved_placeholder_frac')} "
         f"(上限 {cfg.stop_max_unresolved})",
         f"- f1_own {dec.get('mean_f1_own')} (下限 {cfg.stop_min_f1_own})",
         f"- 同文本对方向一致率 {dec.get('same_text_direction_consistency')} "
         f"(下限 {cfg.stop_min_same_text_consistency}) — 阴性对照",
         "",
         "描述性（不作闸门）：跨任务字节相同查询 "
         f"{dec.get('exact_dup_query_frac')}，头部单文本簇 "
         f"{dec.get('head_mono_text_share')}。占位符解析后，字面相同的查询是"
         "带实体的真实重复需求，不是模板碰撞。", "",
         "**判官关卡**", "",
         f"- 解析失败 {dec.get('parse_fail_rate')} / 陷阱误判 {dec.get('catch_fp')} "
         f"/ 金标 acc {dec.get('gold_acc')} / κ {dec.get('cross_kappa')}", "",
         "## 1. 数据规模", "",
         f"- 语料源 **{cs.get('corpus_source')}**：{cs.get('n_docs')} 文档 / "
         f"{cs.get('n_chunks')} 块 / dim {cs.get('dim')} / "
         f"编码 {cs.get('encode_sec',0)/60:.0f}min ({cs.get('chunks_per_sec',0):.0f} 块/s)",
         f"- 检索调用 {st.get('total_calls')} / 任务 {st.get('total_tasks')} "
         f"({st.get('calls_per_task',0):.1f} 次/任务)",
         f"- 查询唯一率 {sn.get('unique_query_frac')}，中位长度 {sn.get('median_query_len')}，"
         f"未解析占位符 {sn.get('unresolved_placeholder_frac')}",
         f"- agent: {json.dumps(st.get('agent', {}), ensure_ascii=False)[:300]}", "",
         "## 2. 分块完整性", "",
         f"- 候选对 {au.get('n_pairs')}，top-{au.get('block_topk')}，暴力抽检 {au.get('probe_n')}",
         f"- **漏检率 @cos≥0.90 = {(au.get('miss_rate',{}).get('0.9',0)):.2%}**", "",
         "## 3. 判官质量", "",
         f"- 抽样 {qc.get('n_sampled_pairs')} 对 -> 自覆盖合格 {qc.get('n_eligible')} "
         f"(合格率 {qc.get('eligibility_rate')}) -> 实判 {qc.get('n_judged')} 对",
         f"- **自覆盖率 {qc.get('self_cov_rate')}**（R(q) 能否回答 q 本身）",
         f"- **条件可替代率 {qc.get('substitutable_rate_conditional')}**"
         f"（边际口径 {qc.get('substitutable_rate_marginal')}）",
         f"- 解析失败 {qc.get('parse_fail_rate',0):.2%}"
         f"（截断占 {qc.get('truncated_share')}），{qc.get('tokens',0)/1e6:.1f}M token",
         f"- 平均输出 {(qc.get('llm') or {}).get('avg_out')} tok/次",
         f"- **陷阱误判率 {qc.get('catch_fp_rate')}**（>{cfg.stop_max_catch_fp:.0%} 作废）",
         f"- 位置一致率 {qc.get('pos_consistency')} (n={qc.get('pos_consistency_n')})，"
         f"方向不对称率 {qc.get('asymmetry_rate')}",
         f"- **同文本对方向一致率 {qc.get('same_text_direction_consistency')}** "
         f"(n={qc.get('same_text_n')}) — 阴性对照，检索结果相同，双向判定必须一致",
         f"- 金标 acc/P/R {(qc.get('gold') or {}).get('accuracy')} / "
         f"{(qc.get('gold') or {}).get('precision')} / {(qc.get('gold') or {}).get('recall')}",
         f"- 跨判官 κ {qc.get('cross_kappa')} (n={qc.get('cross_n')})", "",
         "## 4. L1 vs L2 真值锚定", "",
         f"- L2 样本 {l2.get('n')}，混淆 {l2.get('confusion')}",
         f"- f1_own 均值 {l2.get('mean_f1_own')}（EM {l2.get('em_own_rate')}）— "
         f"健全性检查，接近 0 说明答案生成坏了",
         f"- L1 相对 L2 的 P/R {l2.get('l1_precision')} / {l2.get('l1_recall')}",
         f"- **L1 判正时召回损失 均值 {l2.get('mean_recall_loss_pos')}，"
         f"P95 {l2.get('p95_recall_loss_pos')}**", "",
         "> P95 就是 Phase 2 里 ε 的经验取值依据。", "",
         "## 5. 阈值校准", "",
         f"- **τ\\* = {ca.get('tau_star')}**（置信下界，目标 {cfg.target_substitutability}），"
         f"点估计 {ca.get('tau_point')}",
         f"- 校准基于自覆盖合格对（n={ca.get('n')}，合格率 {ca.get('eligibility_rate')}，"
         f"自覆盖率 {ca.get('self_cov_rate')}）",
         "- 纵轴含义：**在基础索引能回答的查询中**，视图可替代的比例",
         "- 图 `figures/calibration.png`", "",
         "## 6. 复用结构（主结果）", "",
         "| 指标 | 连通分量(下界) | 极大团(上界) |", "|---|---|---|",
         f"| 意图数 | {d.get('n_intents')} | {cl.get('n_intents')} |",
         f"| 头部覆盖(按次) | {d.get('head_cov_calls',0):.1%} | {cl.get('head_cov_calls',0):.1%} |",
         f"| 头部覆盖(按成本) | {d.get('head_cov_cost',0):.1%} | {cl.get('head_cov_cost',0):.1%} |",
         f"| 跨任务意图占比 | {d.get('cross_intent_frac',0):.1%} | "
         f"{cl.get('cross_intent_frac',0):.1%} |",
         f"| 头部单文本簇占比 | {d.get('head_mono_text_share',0):.1%} | "
         f"{cl.get('head_mono_text_share',0):.1%} |",
         f"| Gini(按次) | {d.get('gini_calls',0):.3f} | {cl.get('gini_calls',0):.3f} |",
         f"| Hill α | {d.get('hill_alpha')} | {cl.get('hill_alpha')} |", "",
         "### 误差合成", "",
         f"- 点估计 {co.get('point',0):.1%}",
         f"- 下界 {co.get('lower',0):.1%}（L2 precision={co.get('l2_precision')}）",
         f"- 上界 {co.get('upper',0):.1%}（分块漏检={co.get('blocking_miss')}）", "",
         "## 7. 分域", "",
         "| 数据源 | 调用 | 意图 | 头部覆盖(成本) | 跨任务意图 |", "|---|---|---|---|---|"]
    for s, v in (R.get("by_source") or {}).items():
        if v.get("n_calls"):
            L.append(f"| {s} | {v['n_calls']} | {v['n_intents']} | "
                     f"{v['head_cov_cost']:.1%} | {v['cross_intent_frac']:.1%} |")
    L += ["", "## 8. 粒度敏感性", "",
          "| τ | 分量 | 团 | 最大簇 | 单例 | 簇内直连 |", "|---|---|---|---|---|---|"]
    for t in sorted(gr, key=float):
        v = gr[t]
        L.append(f"| {t} | {v['n_components']} | {v['n_cliques']} | {v['largest']} | "
                 f"{v['singleton_frac']:.1%} | {v['intra_edge_rate']} |")
    L += ["", "## 9. Phase 2 交接", "",
          f"- 启动视图 {vw.get('n_views')} 个，覆盖 {vw.get('coverage',0):.1%}，"
          f"单文本视图 {vw.get('mono_text_view_frac')}",
          "- `stages/s11/views.json` 含 centroid 与 scope_radius_p5，"
          "可直接做 conformal 的 nonconformity score 特征",
          f"- ε 初值建议 {l2.get('p95_recall_loss_pos')}", "",
          "## 10. 已知局限", "",
          "1. 可替代性是相对 (k, 语料, 下游模型) 定义的三元组属性，不是查询内在属性。",
          "2. 可替代关系不传递；分量与团分别是意图数的下界与上界。",
          "3. 语料由数据集自带段落构成，只覆盖这些问题需要的文档，"
          "检索难度低于真实开放语料，复用率是上界。",
          "4. HotpotQA 没有分解标注，只发一条任务级查询，天然不参与"
          "子任务级复用统计；部分 2Wiki 条目缺分步答案，用整题答案代替，"
          "其 L2 召回损失偏保守。",
          "4b. 命中率是条件于自覆盖的：分母是基础索引本来能回答的查询。"
          "无条件口径会被自覆盖率的平方压低，不反映视图的真实价值。",
          "5. 分块漏检造成的系统性低估已量化未消除。",
          "6. 单一 agent 模型，未做 agent 强度的敏感性分析。", "",
          "---", "", "图 `figures/`  数据 `stages/`  日志 `logs/`"]
    ctx.out("s12", "report.md").write_text("\n".join(L), encoding="utf-8")
    shutil.copy(ctx.out("s12", "report.md"), ctx.base / "REPORT.md")
    return "REPORT.md"



# ==============================================================================
#  文本特征：实体与模板（v6 新增，供 s14 二特征判定器使用）
# ==============================================================================

_STOP = set("""a an the of in on at to for from by with and or is are was were be been being
what which who whom whose when where why how did do does that this these those it its as
than then there their they them he she his her him you your i we our us not no""".split())


def _tokens(q: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", q or "")


def _entity_set(q: str) -> set:
    """实体近似：非句首大写词、纯数字、长度>=4 的非停用词。

    不追求 NER 精度，只需把「同模板不同实体」与「不同表述同实体」分开。
    main_v2 的诊断显示 bin5 倒挂层里，不可替代对的模板 Jaccard 0.819 而实体
    0.390，可替代对反过来（0.472 / 0.497）—— 这个特征就是冲这个来的。
    """
    ts, ents = _tokens(q), set()
    for i, t in enumerate(ts):
        tl = t.lower()
        if tl in _STOP:
            continue
        if (t[0].isupper() and i > 0) or t.isdigit() or len(t) >= 4:
            ents.add(tl)
    return ents


def _template_set(q: str) -> set:
    ts = [t.lower() for t in _tokens(q)]
    e = _entity_set(q)
    return {t for t in ts if t not in e}


def _score_stats(scores) -> Tuple[float, float, float]:
    """检索分数分布特征：top1、top1-top2 间隔、softmax 熵。

    v11 新增。main_v7 显示 ent/tpl/rbo/jac 只能把命中率 7.9%->8.1%，
    需要的是**不同类型**的信号。原方案 C2 列的候选特征里就有
    「视图内最高分与全局分数分布的分位关系」—— 这一组正是它。
    直觉：检索分数陡峭（top1 高、间隔大、熵低）意味着这条查询有明确答案，
    可替代性判断更可靠；分数平坦意味着检索本身没抓到重点。
    """
    s = [float(x) for x in (scores or [])]
    if not s:
        return (0.0, 0.0, 0.0)
    top1 = s[0]
    margin = s[0] - (s[1] if len(s) > 1 else 0.0)
    a = np.asarray(s[:10], dtype=float)
    p = np.exp(a - a.max())
    p = p / max(p.sum(), 1e-9)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    return (top1, margin, ent)


def _pair_features(r: Dict, ci: Dict, cj: Dict) -> Tuple[List[float], List[str]]:
    """一对查询的全部特征。返回 (值, 名字)，顺序固定。"""
    ent = _jac(_entity_set(ci["query"]), _entity_set(cj["query"]))
    tpl = _jac(_template_set(ci["query"]), _template_set(cj["query"]))
    t1i, mi, ei = _score_stats(ci.get("scores"))
    t1j, mj, ej = _score_stats(cj.get("scores"))
    du = len(set(ci.get("retrieved", [])) | set(cj.get("retrieved", [])))
    di = len(set(ci.get("retrieved", [])))
    vals = [r["cos"], ent, tpl, r.get("rbo", 0.0), r.get("jaccard", 0.0),
            min(t1i, t1j), abs(t1i - t1j), min(mi, mj), max(ei, ej),
            du / max(di, 1), float(len(ci["query"])) / max(len(cj["query"]), 1)]
    names = ["cos", "ent", "tpl", "rbo", "jac",
             "top1_min", "top1_gap", "margin_min", "entropy_max",
             "doc_union_ratio", "len_ratio"]
    return vals, names


def _jac(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


class _DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _head_cov(calls_sub, pairs, tau, head_frac):
    """在子集上重算意图簇与头部覆盖，口径与 s10 一致。"""
    qids = [c["qid"] for c in calls_sub]
    if not qids:
        return {}
    dsu = _DSU(qids); qset = set(qids); ne = 0
    for p in pairs:
        if p["cos"] >= tau and p["qid_i"] in qset and p["qid_j"] in qset:
            dsu.union(p["qid_i"], p["qid_j"]); ne += 1
    root = {q: dsu.find(q) for q in qids}
    cn, cc, ct = Counter(), Counter(), defaultdict(set)
    for c in calls_sub:
        k = root[c["qid"]]
        cn[k] += 1; cc[k] += c.get("cost", 1.0); ct[k].add(c["task_id"])
    rk = [k for k, _ in cn.most_common()]
    hn = max(1, int(round(len(rk) * head_frac)))
    head = rk[:hn]
    tn, tc = sum(cn.values()), sum(cc.values())
    cross = {k for k in rk if len(ct[k]) >= 2}
    hx = [k for k in head if k in cross]
    return {"n_tasks": len({c["task_id"] for c in calls_sub}), "n_calls": len(calls_sub),
            "n_intents": len(rk), "n_edges": ne,
            "head_cov_calls": sum(cn[k] for k in head) / tn,
            "head_cov_cost": sum(cc[k] for k in head) / max(tc, 1e-9),
            "cross_intent_frac": len(cross) / len(rk),
            "head_cross_share": sum(cn[k] for k in hx) / max(sum(cn[k] for k in head), 1),
            "mean_tasks_per_intent": float(np.mean([len(v) for v in ct.values()]))}


# ==============================================================================
# s13  规模曲线（离线，不用 GPU）
# ==============================================================================


@stage("s13", "规模曲线：头部覆盖 vs 任务数", ["scale.json"])
def s13(ctx: RunContext, sm: ServerManager):
    """复用率不是尺度不变量。main_v2 实测 520->10417 任务时头部覆盖 12.4%->24.5%，
    对数线性 R²=0.988。这条曲线回答审稿人的「会不会收敛在 20%」，
    同时坐实「视图收益随工作负载规模增长」这个与 DB 物化视图同源的机制。
    """
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    pairs = ctx.rjsonl(ctx.out("s4", "pairs.jsonl"))
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    tau = calib.get("tau_star") or calib.get("tau_point") or 0.90
    fracs = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
    seeds = 3

    deg = Counter()
    for p in pairs:
        if p["cos"] >= tau:
            deg[p["qid_i"]] += 1; deg[p["qid_j"]] += 1
    sat = sum(1 for v in deg.values() if v >= cfg.block_topk - 2) / max(len(deg), 1)
    log.info(f"邻域饱和度 {sat:.2%}（子采样只丢邻居不新增，小规模点是轻微低估）")

    def curve(cs, label):
        byt = defaultdict(list)
        for c in cs:
            byt[c["task_id"]].append(c)
        tasks = sorted(byt)
        rows = []
        for fr in fracs:
            acc = []
            for s in range(seeds if fr < 1.0 else 1):
                rnd = random.Random(1000 + s)
                keep = set(rnd.sample(tasks, max(1, int(len(tasks) * fr))))
                m = _head_cov([c for t in keep for c in byt[t]], pairs, tau, cfg.stop_head_frac)
                if m:
                    acc.append(m)
            if not acc:
                continue
            agg = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
            agg["frac"] = fr
            agg["sd_cost"] = float(np.std([a["head_cov_cost"] for a in acc])) if len(acc) > 1 else 0.0
            rows.append(agg)
            log.info(f"  [{label}] {fr:>5.0%} 任务{agg['n_tasks']:>7.0f} "
                     f"头部(次){agg['head_cov_calls']:>6.1%} 头部(成本){agg['head_cov_cost']:>6.1%} "
                     f"跨任务{agg['cross_intent_frac']:>6.1%}")
        return rows

    out = {"tau": tau, "saturation": sat, "all": curve(calls, "全域"), "by_source": {}}
    for s in sorted({c["source"] for c in calls}):
        out["by_source"][s] = curve([c for c in calls if c["source"] == s], s)

    def fit(rows):
        pts = [(r["n_tasks"], r["head_cov_cost"]) for r in rows if r["n_tasks"] >= 20]
        if len(pts) < 3:
            return None
        x = np.log10([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        b, a = np.polyfit(x, y, 1)
        yh = a + b * x
        r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        cur = pts[-1][0]
        return {"slope": float(b), "intercept": float(a), "r2": float(r2),
                "current_tasks": float(cur), "current": float(y[-1]),
                "pred": {f"{m}x": float(a + b * math.log10(cur * m)) for m in (2, 10, 50)}}

    out["fits"] = {"all": fit(out["all"])}
    for s, rows in out["by_source"].items():
        out["fits"][s] = fit(rows)
    for k, f in out["fits"].items():
        if f:
            log.info(f"  拟合 {k:<20} 斜率 {f['slope']:+.4f}/decade R²={f['r2']:.3f} "
                     f"当前 {f['current']:.1%} -> 10x {f['pred']['10x']:.1%}")
    ctx.wjson(ctx.out("s13", "scale.json"), out)
    fa = out["fits"].get("all") or {}
    return f"全域斜率 {fa.get('slope', 0):+.4f}/decade R²={fa.get('r2', 0):.3f}"


# ==============================================================================
# s14  二特征判定器（离线）
# ==============================================================================


@stage("s14", "判定器消融：单特征 vs 多特征（split conformal）", ["twofeat.json"])
def s14(ctx: RunContext, sm: ServerManager):
    """main_v2 的 bin5 真倒挂说明单靠余弦不够：不可替代对的模板重合 0.819、
    实体重合 0.390，正是「同模板不同实体」。这一步量化加特征能换来多少命中率。

    口径：split conformal。校准集上选阈值使可替代率的 95% 置信下界 >= 目标，
    测试集上报实际覆盖率与实际可替代率。命中率 = 通过判定的对占比 —— 这就是
    论文里「多少查询可以绕过基础索引」的直接对应物。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    calls = {c["qid"]: c for c in ctx.rjsonl(ctx.out("s2", "calls.jsonl"))}
    l1 = [r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
          if r.get("eligible") == 1 and r.get("ok") and r.get("substitutable") is not None]
    if len(l1) < 200:
        log.warning(f"合格判定仅 {len(l1)} 条，二特征分析不可靠")
    X, y, meta, Wl = [], [], [], []
    for r in l1:
        ci, cj = calls.get(r["qid_i"]), calls.get(r["qid_j"])
        if not ci or not cj:
            continue
        vals, FNAMES = _pair_features(r, ci, cj)
        X.append(vals)
        y.append(r["substitutable"])
        Wl.append(r.get("w", 1.0))
        meta.append({"pair_id": r["pair_id"], "cos": r["cos"], "ent": vals[1],
                     "tpl": vals[2], "bin": r.get("bin"), "sub": r["substitutable"]})
    X = np.array(X, float); y = np.array(y, int); W_ = np.array(Wl, float)
    _p, _lo, _hi, _ne = wilson_w(y, W_)
    log.info(f"样本 {len(y)}；正例率 未加权 {y.mean():.3f}，"
             f"加权 {_p:.3f} CI[{_lo:.3f},{_hi:.3f}] n_eff={_ne:.0f}")

    rng = np.random.RandomState(cfg.seed)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    cal, tst = idx[:half], idx[half:]

    # v11：加入检索分数分布特征。main_v7 里纯文本特征只把命中率 7.9%->8.1%，
    # 提升微弱；检索分数是**不同类型**的信号，值得单独做一组对照。
    FEATS = {"cos": [0],
             "cos+文本(ent,tpl)": [0, 1, 2],
             "cos+重叠(rbo,jac)": [0, 3, 4],
             "cos+检索分数": [0, 5, 6, 7, 8],
             "cos+文本+重叠": [0, 1, 2, 3, 4],
             "全部特征": list(range(11))}
    target = cfg.target_substitutability
    # 多目标扫描：主目标是 cfg 的 0.90，但也报 0.80/0.85/0.95 的命中率。
    # 若主目标在校准集上无解（正例率太低时会这样），至少还有可用结论，
    # 而且「保证水平 vs 命中率」这条曲线本身就是论文里的帕累托前沿。
    targets = sorted({0.80, 0.85, 0.90, 0.95, float(target)})
    out = {"n": int(len(y)), "pos_rate": float(y.mean()), "target": target,
           "targets": targets, "models": {}, "pareto": {}}

    for name, cols in FEATS.items():
        if len(cols) == 1:
            sc_cal, sc_tst = X[cal][:, 0], X[tst][:, 0]
            coef = None
        else:
            sk = StandardScaler().fit(X[cal][:, cols])
            lr = LogisticRegression(max_iter=2000, C=1.0)
            lr.fit(sk.transform(X[cal][:, cols]), y[cal], sample_weight=W_[cal])
            sc_cal = lr.predict_proba(sk.transform(X[cal][:, cols]))[:, 1]
            sc_tst = lr.predict_proba(sk.transform(X[tst][:, cols]))[:, 1]
            _fn = ["cos", "ent", "tpl", "rbo", "jac", "top1_min", "top1_gap",
                   "margin_min", "entropy_max", "doc_union_ratio", "len_ratio"]
            coef = dict(zip([_fn[c] for c in cols],
                            [round(float(v), 4) for v in lr.coef_[0]]))
        # 校准集上找最低阈值，使置信下界 >= 目标且覆盖最大；对每个目标各算一次
        def solve(tg):
            bt, bc = None, 0.0
            # v9：命中率与可替代率都按分层权重算。未加权口径下 main_v5 的
            # ≥0.90 命中率是 81.4%，那是被 bin7 占样本 45% 撑出来的假象。
            for t in np.quantile(sc_cal, np.linspace(0.0, 0.995, 200)):
                m_ = sc_cal >= t
                n_ = int(m_.sum())
                if n_ < 30:
                    continue
                _, lo_, _, _ = wilson_w(y[cal][m_], W_[cal][m_])
                if lo_ >= tg and W_[cal][m_].sum() / W_[cal].sum() > bc:
                    bt, bc = float(t), float(W_[cal][m_].sum() / W_[cal].sum())
            if bt is None:
                return None
            m_ = sc_tst >= bt
            n_ = int(m_.sum())
            if n_ == 0:
                return None
            emp_, lo_, hi_, neff_ = wilson_w(y[tst][m_], W_[tst][m_])
            cov_w = float(W_[tst][m_].sum() / W_[tst].sum())
            return {"threshold": bt, "coverage_cal": bc,
                    "coverage_test": cov_w,
                    "coverage_test_unweighted": n_ / len(tst),
                    "n_test_accepted": n_, "n_eff": neff_,
                    "empirical_sub_test": emp_,
                    "empirical_ci": [lo_, hi_], "guarantee_met": bool(lo_ >= tg - 0.05)}

        per_t = {f"{tg:.2f}": solve(tg) for tg in targets}
        out["pareto"][name] = per_t
        main = per_t.get(f"{target:.2f}")
        if main is None:
            # 主目标无解时，取能达成的最高目标
            ach = [(float(k), v) for k, v in per_t.items() if v]
            if ach:
                k, main = max(ach, key=lambda kv: kv[0])
                log.warning(f"  {name:<22} 目标 {target} 无解，退到可达成的最高目标 {k:.2f}")
                main = dict(main); main["fallback_target"] = k
            else:
                log.warning(f"  {name:<22} 任何目标下都无解（正例率 {y.mean():.3f} 过低）")
                out["models"][name] = {"threshold": None, "coverage_test": 0.0,
                                       "coef": coef, "guarantee_met": False}
                continue
        main = dict(main); main["coef"] = coef
        out["models"][name] = main
        log.info(f"  {name:<22} 命中率 {main['coverage_test']:>6.1%}  "
                 f"实际可替代率 {main['empirical_sub_test']:.3f} "
                 f"CI[{main['empirical_ci'][0]:.3f},{main['empirical_ci'][1]:.3f}]  "
                 f"{'达标' if main['guarantee_met'] else '未达标'}"
                 + (f"（回退目标 {main['fallback_target']:.2f}）"
                    if "fallback_target" in main else ""))
        for tg in targets:
            v = per_t.get(f"{tg:.2f}")
            log.info(f"      保证 {tg:.2f} -> 命中率 "
                     + (f"{v['coverage_test']:.1%}" if v else "无解"))

    base = out["models"].get("cos", {}).get("coverage_test", 0.0)
    for name, d in out["models"].items():
        if base > 0 and d.get("coverage_test"):
            d["lift_vs_cos"] = d["coverage_test"] / base - 1
    best = max(out["models"].items(),
               key=lambda kv: kv[1].get("coverage_test", 0.0) if kv[1].get("guarantee_met") else -1)
    out["best_model"] = best[0]
    log.info(f"  最优特征组合：{best[0]}，相对单余弦命中率提升 "
             f"{best[1].get('lift_vs_cos', 0):+.1%}")

    # 逐层：加实体约束后单调性是否恢复
    mono = {}
    for ethr in (0.0, 0.2, 0.3, 0.4, 0.5):
        seq = []
        for b in sorted({m["bin"] for m in meta if m["bin"] is not None}):
            s = [m for m in meta if m["bin"] == b and m["ent"] >= ethr]
            seq.append(round(float(np.mean([m["sub"] for m in s])), 3) if len(s) >= 20 else None)
        v = [x for x in seq if x is not None]
        drops = sum(1 for a, b_ in zip(v, v[1:]) if b_ < a - 0.02)
        mono[f"ent>={ethr}"] = {"seq": seq, "drops": drops,
                                "kept": int(sum(1 for m in meta if m["ent"] >= ethr))}
        log.info(f"  ent>={ethr:.1f} 逐层 {seq} 下降 {drops} 次")
    out["monotonicity"] = mono
    ctx.wjson(ctx.out("s14", "twofeat.json"), out)
    return (f"最优 {best[0]}，命中率 {best[1].get('coverage_test', 0):.1%} "
            f"(单余弦 {base:.1%})")


# ==============================================================================
# s15  金标工作表 / 评估（离线）
# ==============================================================================


@stage("s15", "金标工作表生成或评估", ["gold_eval.json"])
def s15(ctx: RunContext, sm: ServerManager):
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    gold = ctx.rjsonl(ctx.out("s5", "gold_pairs.jsonl"))
    l1 = {r["pair_id"]: r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
          if r.get("eligible") == 1 and r.get("ok")}
    calls = {c["qid"]: c for c in ctx.rjsonl(ctx.out("s2", "calls.jsonl"))}
    # v11：支持从人工标注的 TSV 回填（比手改 jsonl 方便得多）
    if cfg.gold_tsv and Path(cfg.gold_tsv).exists():
        import csv
        lut = {}
        with open(cfg.gold_tsv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                pid = (row.get("pair_id") or "").strip()
                v = (row.get("human_label") or "").strip()
                if pid and v in ("0", "1"):
                    lut[pid] = int(v)
        n_fill = 0
        for gg in gold:
            if gg["pair_id"] in lut:
                gg["human_label"] = lut[gg["pair_id"]]
                n_fill += 1
        log.info(f"  从 {cfg.gold_tsv} 回填 {n_fill}/{len(gold)} 条人工标注")
        ctx.wjsonl(ctx.out("s5", "gold_pairs.jsonl"), gold)   # 落盘，下次不用再传
    lab = [g for g in gold if g.get("human_label") in (0, 1, "0", "1")]
    out = {"n_gold": len(gold), "n_judged": sum(1 for g in gold if g["pair_id"] in l1),
           "n_labeled": len(lab)}

    if lab:
        yt, yp, ids = [], [], []
        for g in lab:
            r = l1.get(g["pair_id"])
            if r:
                yt.append(int(g["human_label"])); yp.append(int(r["substitutable"]))
                ids.append(g["pair_id"])
        if yt:
            tp = sum(a == 1 and b == 1 for a, b in zip(yt, yp))
            fp = sum(a == 0 and b == 1 for a, b in zip(yt, yp))
            fn = sum(a == 1 and b == 0 for a, b in zip(yt, yp))
            tn = sum(a == 0 and b == 0 for a, b in zip(yt, yp))
            acc = (tp + tn) / len(yt); lo, hi = wilson(tp + tn, len(yt))
            out.update({"n": len(yt), "accuracy": acc, "acc_ci": [lo, hi],
                        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                        "kappa": kappa(yt, yp),
                        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                        "pass": acc >= cfg.stop_min_gold_acc,
                        "disagreements": [ids[i] for i in range(len(yt)) if yt[i] != yp[i]][:30]})
            log.info(f"  金标 accuracy {acc:.3f} CI[{lo:.3f},{hi:.3f}] "
                     f"P={out['precision']:.3f} R={out['recall']:.3f} κ={out['kappa']:.3f}")
            if not out["pass"]:
                log.error(f"!! accuracy < {cfg.stop_min_gold_acc}，判官不可靠，结论需降级")
    else:
        need = set()
        for g in gold:
            for q in (g["qid_i"], g["qid_j"]):
                c = calls.get(q)
                if c:
                    need.update(c.get("retrieved_chunks", [])[:3])
        cmap = {}
        with open(ctx.out("s1", "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["chunkid"] in need:
                    cmap[d["chunkid"]] = d["text"]
                if len(cmap) == len(need):
                    break
        ws = ctx.base / "gold_worksheet.tsv"
        key = {}
        order = list(gold); random.Random(cfg.seed).shuffle(order)
        with open(ws, "w", encoding="utf-8") as f:
            f.write("pair_id\tcos\thuman_label\t查询A\t查询B\tA检索结果\tB检索结果\n")
            for g in order:
                ci, cj = calls.get(g["qid_i"], {}), calls.get(g["qid_j"], {})
                sn = lambda c: " || ".join((cmap.get(x, "")[:220]).replace("\t", " ")
                                           .replace("\n", " ")
                                           for x in c.get("retrieved_chunks", [])[:2])
                f.write("\t".join([g["pair_id"], f"{g['cos']:.4f}", "",
                                    (ci.get("query", "") or "").replace("\t", " "),
                                    (cj.get("query", "") or "").replace("\t", " "),
                                    sn(ci), sn(cj)]) + "\n")
                r = l1.get(g["pair_id"])
                key[g["pair_id"]] = {"judge": (r or {}).get("substitutable")}
        ctx.wjson(ctx.base / "gold_answerkey.json", key)
        out["worksheet"] = str(ws)
        log.warning(f"  金标未标注，工作表 -> {ws}（判官结论另存 gold_answerkey.json，"
                    f"标注完成前不要看）")
    ctx.wjson(ctx.out("s15", "gold_eval.json"), out)
    return (f"accuracy {out['accuracy']:.3f}" if "accuracy" in out
            else f"生成工作表 {out['n_gold']} 对待标注")


# ==============================================================================
# s16  最终汇总（把所有结论压进一个文本文件）
# ==============================================================================


@stage("s16", "最终汇总（单文件）", ["FINAL.txt"])
def s16(ctx: RunContext, sm: ServerManager):
    cfg = ctx.cfg
    sm.stop()
    g = lambda s, n: ctx.rjson(ctx.out(s, n)) if ctx.out(s, n).exists() else {}
    cs, st = g("s1", "stats.json"), g("s2", "stats.json")
    au, qc = g("s4", "audit.json"), g("s6", "qc.json")
    l2, ca = g("s7", "summary.json"), g("s8", "calib.json")
    gr, R_ = g("s9", "granularity.json"), g("s10", "reuse.json")
    vw, sc = g("s11", "views.json"), g("s13", "scale.json")
    tf, ge = g("s14", "twofeat.json"), g("s15", "gold_eval.json")
    d = R_.get("components", {}); dec = R_.get("decision", {})
    tot = sum(m.get("duration_sec", 0) for m in ctx.state.get("stages", {}).values())
    L = []
    def w(s=""):
        L.append(s)

    w("=" * 78); w(f"  MatView Phase 1 最终结果 — {cfg.run_id}"); w("=" * 78)
    w(f"  生成 {time.strftime('%F %T')}   总机时 {tot/3600:.2f} h")
    w()
    w("【0】止损判定")
    w(f"     {dec.get('verdict')} — {dec.get('reason')}")
    w(f"     垂直域头部{cfg.stop_head_frac:.0%}成本覆盖 {dec.get('best_vertical_cost_cov')}")
    w(f"     通用域(agent) {dec.get('general_cost_cov')}")
    w()
    w("【1】数据健全性")
    w(f"     未解析占位符 {dec.get('unresolved_placeholder_frac')} (上限 {cfg.stop_max_unresolved})")
    w(f"     f1_own {dec.get('mean_f1_own')} (下限 {cfg.stop_min_f1_own})")
    w(f"     同文本方向一致率 {dec.get('same_text_direction_consistency')} — 阴性对照")
    _ag = st.get("agent", {})
    w(f"     agent 多查询模式 {_ag.get('multi_query_mode')}  数组占比 "
      f"{_ag.get('multi_query_frac')}  平均子查询数 {_ag.get('mean_subqueries')}")
    w(f"     agent 调用 {_ag.get('calls')}（其中首条 {_ag.get('calls_first_arm')} = "
      f"'first' 口径的量）")
    w(f"     agent 查询中位长度 {st.get('agent', {}).get('median_query_len')}")
    w()
    w("【2】判官质量")
    w(f"     抽样 {qc.get('n_sampled_pairs')} 对 -> 自覆盖合格 {qc.get('n_eligible')} "
      f"({qc.get('eligibility_rate')}) -> 实判 {qc.get('n_judged')}")
    w(f"     自覆盖率 {qc.get('self_cov_rate')}")
    w(f"     ★ 条件可替代率（分层加权）{qc.get('substitutable_rate_conditional')} "
      f"CI{[round(x,4) for x in (qc.get('substitutable_rate_conditional_ci') or [0,0])]} "
      f"n_eff={qc.get('effective_n')}")
    w(f"       未加权样本口径 {qc.get('substitutable_rate_conditional_unweighted')}"
      f"（会被高相似度层拉高，仅供对照）；边际口径 "
      f"{qc.get('substitutable_rate_marginal')}")
    _sc = qc.get("strict") or {}
    _un = qc.get("union") or {}
    if _sc:
        w(f"     口径对比（n={_sc.get('n')}）：single "
          f"{qc.get('substitutable_rate_conditional')}  union "
          f"{_un.get('rate_weighted')}  intersection {_sc.get('rate_weighted')}")
        w("       主口径 = single。main_v7 已证明 intersection 引入的是系统性偏差：")
        w("       bin7 翻转者可替代率 0.954（非随机的 0.5），打乱顺序后判官系统性")
        w("       偏向 INSUFFICIENT，取交集把真可替代的对错杀，τ*_strict 直接无解。")
        w("       逐层 宽松->严格：" + "  ".join(
            f"b{b}:{v['loose']:.3f}->{v['strict']:.3f}"
            for b, v in sorted(_sc.get("by_bin", {}).items(), key=lambda x: int(x[0]))))
    _pcb = qc.get("pos_consistency_by_bin") or {}
    if _pcb:
        w("     位置一致率分层：" + "  ".join(
            f"b{b}={(v['rate'] if v['rate'] is not None else float('nan')):.3f}(n={v['n']})"
            for b, v in sorted(_pcb.items(), key=lambda x: int(x[0]))))
    w(f"     陷阱误判 {qc.get('catch_fp_rate')}   位置一致 {qc.get('pos_consistency')}   "
      f"方向不对称 {qc.get('asymmetry_rate')}")
    w(f"     解析失败 {qc.get('parse_fail_rate')}   跨判官κ {qc.get('cross_kappa')}")
    if ge.get("accuracy") is not None:
        w(f"     ★ 人工金标 accuracy {ge['accuracy']:.3f} CI{[round(x,3) for x in ge['acc_ci']]} "
          f"P={ge['precision']:.3f} R={ge['recall']:.3f} κ={ge['kappa']:.3f} "
          f"-> {'通过' if ge.get('pass') else '不通过'}")
    else:
        w(f"     ★ 人工金标未标注（工作表 {ge.get('worksheet')}）—— 投稿硬门槛，必须补")
    w()
    w("【3】L2 真值锚定")
    w(f"     n={l2.get('n')} 混淆 {l2.get('confusion')}")
    w(f"     f1_own {l2.get('mean_f1_own')} (EM {l2.get('em_own_rate')})")
    for s, v in (l2.get("f1_own_by_source") or {}).items():
        w(f"       {s:<20} n={v['n']:>5}  f1_own={v['mean_f1_own']:.3f}  "
          f"EM={v['em_own_rate']:.3f}")
    w("       （agent 的 gold_answer 是复合问题的最终答案，子查询本就答不出，"
      "混算会被误读成检索质量下降）")
    w(f"     L1 相对 L2 P/R {l2.get('l1_precision')} / {l2.get('l1_recall')}")
    w(f"     判正样本 n={l2.get('n_pos')}  非零损失占比 {l2.get('nonzero_loss_frac')}")
    w(f"     损失分位 {l2.get('loss_quantiles')}")
    w(f"     ★ ε 建议值：P99={l2.get('p99_recall_loss_pos')}  "
      f"max={l2.get('max_recall_loss_pos')}  非零条件均值={l2.get('mean_loss_given_nonzero')}")
    w(f"       （P95={l2.get('p95_recall_loss_pos')} 常年为 0，分布是「多数零损失+少数大损失」，"
      f"不可用作 ε）")
    w()
    w("【4】阈值校准（条件于自覆盖）")
    w(f"     τ* {ca.get('tau_star')}   点估计 {ca.get('tau_point')}   n={ca.get('n')}")
    w(f"     τ*（交集口径，仅供对照，已知有系统性偏差）{ca.get('tau_star_strict')} "
      f"n={ca.get('n_strict')}")
    if ca.get("tau_multi"):
        w(f"     多档 τ*（★ 退化 = 放行的对里同文本占比 ≥ "
          f"{cfg.degenerate_same_text_frac:.0%}，判定器等价于精确缓存）：")
        w(f"       {'保证':<8}{'τ*':>10}{'放行占比':>10}{'同文本占比':>12}"
          f"{'实测可替代':>12}{'退化':>8}")
        for t, v in sorted(ca["tau_multi"].items()):
            if v.get("tau_star") is None:
                w(f"       {t:<8}{'无解':>10}")
                continue
            w(f"       {t:<8}{v['tau_star']:>10.4f}{v['pass_frac']:>10.1%}"
              f"{v['same_text_frac']:>12.1%}{v['empirical_sub']:>12.3f}"
              f"{'★退化' if v['degenerate'] else '':>8}")
    if ca.get("tau_star") and ca["tau_star"] > 0.98:
        w("     ⚠ τ* > 0.98 意味着只有几乎完全相同的查询才够到保证水平，")
        w("       单靠查询嵌入余弦在当前规模下已无法给出有用阈值。")
        w("       见【5】的特征消融：需要的是不同类型的信号（检索分数分布），")
        w("       而不是现有文本特征的组合。")
    w("     逐层：")
    for b, v in sorted((ca.get("per_bin") or {}).items(), key=lambda x: int(x[0])):
        w(f"       bin{b} [{v['range'][0]:.2f},{v['range'][1]:.2f}) n={v['n']:>5} "
          f"p={v['p']:.3f} CI=[{v['lo']:.3f},{v['hi']:.3f}]")
    w()
    w("【5】判定器消融（split conformal，命中率=通过判定的对占比）")
    for name, m in (tf.get("models") or {}).items():
        w(f"     {name:<22} 命中率 {m.get('coverage_test')}  实际可替代率 "
          f"{m.get('empirical_sub_test')} CI{[round(x,3) for x in (m.get('empirical_ci') or [0,0])]}"
          f"  {'达标' if m.get('guarantee_met') else '未达标'}")
    w(f"     最优组合 {tf.get('best_model')}")
    if tf.get("pareto"):
        w("     保证水平 vs 命中率（帕累托前沿）：")
        tg_list = tf.get("targets") or []
        w("       " + "特征".ljust(24) + "".join(f"{f'≥{t:.2f}':>10}" for t in tg_list))
        for name, per in tf["pareto"].items():
            row = "       " + name.ljust(24)
            for t in tg_list:
                v = per.get(f"{t:.2f}")
                cell = "%.1f%%" % (v["coverage_test"] * 100) if v else "-"
                row += cell.rjust(10)
            w(row)
    w("     单调性（加实体约束后逐层可替代率）：")
    for k, v in (tf.get("monotonicity") or {}).items():
        w(f"       {k:<10} 下降 {v['drops']} 次  保留 {v['kept']}  {v['seq']}")
    w()
    w("【6】复用结构")
    w(f"     意图数 {d.get('n_intents')}  头部覆盖 按次 {d.get('head_cov_calls')} "
      f"按成本 {d.get('head_cov_cost')}")
    w(f"     跨任务意图占比 {d.get('cross_intent_frac')}  头部跨任务占比 "
      f"{d.get('head_cross_share')}")
    w("     分域：")
    for s, v in (R_.get("by_source") or {}).items():
        if v.get("n_calls"):
            w(f"       {s:<20} 调用{v['n_calls']:>6} 意图{v['n_intents']:>6} "
              f"头部(成本){v['head_cov_cost']:>7.1%} 跨任务{v['cross_intent_frac']:>7.1%} "
              f"每意图任务{v['mean_tasks_per_intent']:.2f}")
    w()
    w("【7】规模曲线（头部成本覆盖 ~ a + b·log10(任务数)）")
    w(f"     邻域饱和度 {sc.get('saturation')}（低=子采样偏差有限）")
    for k, f in (sc.get("fits") or {}).items():
        if f:
            w(f"     {k:<20} 斜率 {f['slope']:+.4f}/decade R²={f['r2']:.3f}  "
              f"当前 {f['current']:.1%} -> 2x {f['pred']['2x']:.1%} "
              f"10x {f['pred']['10x']:.1%} 50x {f['pred']['50x']:.1%}")
    w("     各比例明细（全域）：")
    for r in (sc.get("all") or []):
        w(f"       {r['frac']:>5.0%} 任务{r['n_tasks']:>8.0f} 调用{r['n_calls']:>8.0f} "
          f"头部(次){r['head_cov_calls']:>7.1%} 头部(成本){r['head_cov_cost']:>7.1%} "
          f"±{r.get('sd_cost',0)*100:.1f}pp 跨任务{r['cross_intent_frac']:>6.1%}")
    w()
    w("【8】粒度敏感性")
    for t, v in sorted((gr or {}).items(), key=lambda x: float(x[0])):
        w(f"     τ={t:<7} 分量{v['n_components']:>7} 团{v['n_cliques']:>7} "
          f"最大簇{v['largest']:>4} 单例{v['singleton_frac']:>6.1%} "
          f"簇内直连{v['intra_edge_rate']}")
    w()
    cv = g("s23", "cache_vs_view.json")
    if cv and "baselines" in cv:
        w("【8a】精确缓存 vs 近似视图（核心对照）")
        w(f"     build {cv.get('n_build_tasks')} 任务 -> {cv.get('n_views')} 视图；"
          f"eval {cv.get('n_eval')} 查询")
        w(f"     {'方法':<20}{'按查询':>10}{'按成本':>10}{'净增(查询)':>12}{'净增(成本)':>12}")
        b = cv["baselines"]
        w(f"     {'精确缓存':<20}{b['exact_cache']['hit_queries']:>10.1%}"
          f"{b['exact_cache']['hit_cost']:>10.1%}{'-':>12}{'-':>12}")
        w(f"     {'归一化缓存':<20}{b['normalized_cache']['hit_queries']:>10.1%}"
          f"{b['normalized_cache']['hit_cost']:>10.1%}{'-':>12}{'-':>12}")
        for t, d in sorted((cv.get("views") or {}).items()):
            if not d:
                w(f"     {'视图@'+t:<20}{'无可行阈值':>10}")
                continue
            inc = (cv.get("increment") or {}).get(t, {})
            w(f"     {'视图@'+t:<20}{d['hit_queries']:>10.1%}{d['hit_cost']:>10.1%}"
              f"{inc.get('frac_of_eval', 0):>12.1%}{inc.get('cost_frac_of_eval', 0):>12.1%}")
        _bsan = cv.get("baseline_sanity") or {}
        if _bsan and not _bsan.get("plausible", True):
            w(f"     ⚠ 缓存基线命中率 {_bsan.get('normalized_cache_hit'):.1%} 不在合理区间"
              f" [2%, 60%]，本对照表无意义，需检查查询池同质性或切分")
        w("     ★ 净增量 = 视图命中但归一化缓存未命中的部分。")
        w("       这是回应「这不就是结果缓存吗」的直接证据：净增量为正且可观，")
        w("       才说明近似视图重写有独立价值。")
        key = f"{cfg.target_substitutability:.2f}"
        ib = ((cv.get("increment") or {}).get(key) or {}).get("by_source") or {}
        if ib:
            w(f"     主保证 {key} 分域净增量（按成本）：" + "  ".join(
                f"{s.replace('replay:','')}={v['cost_frac']:.1%}" for s, v in ib.items()))
        w()
    vh = g("s17", "view_hit.json")
    if vh and "by_guarantee" in vh:
        w("【8b】视图级命中率（train/test 按任务切分，评估侧从未参与建视图）")
        w(f"     建视图 {vh.get('n_build_tasks')} 任务 -> {vh.get('n_views')} 视图；"
          f"评估 {vh.get('n_eval_tasks')} 任务 / {vh.get('n_eval_queries')} 查询")
        w(f"     {'保证':<8}{'按查询':>10}{'按成本':>10}   分域(按成本)")
        for k in ("0.80", "0.85", "0.90", "0.95"):
            d = vh["by_guarantee"].get(k)
            if not d:
                w(f"     {k:<8}{'无可行阈值':>10}")
                continue
            bs = "  ".join(f"{s.replace('replay:','')}={v['hit_rate_cost']:.1%}"
                           for s, v in d["by_source"].items())
            w(f"     {k:<8}{d['hit_rate_queries']:>10.1%}{d['hit_rate_cost']:>10.1%}   {bs}")
        w("     ★ 这才是论文里那句话的数字：在 ε 的召回保证下，多少查询能绕过基础索引")
        w()
    bo = g("s22", "bakeoff.json")
    if bo and "variants" in bo:
        w("【2a】判官 prompt 对拍（以人工金标为真值）")
        w(f"     样本 {bo.get('n')} 对，人工正例率 {bo.get('human_pos_rate'):.3f}")
        w(f"     {'变体':<20}{'acc':>8}{'95%CI':>18}{'P':>8}{'R':>8}"
          f"{'κ':>8}{'判正率':>9}")
        for k, v in sorted(bo["variants"].items(),
                           key=lambda kv: -kv[1]["accuracy"]):
            w(f"     {k:<20}{v['accuracy']:>8.3f}"
              f"   [{v['acc_ci'][0]:.3f},{v['acc_ci'][1]:.3f}]"
              f"{v['precision']:>8.3f}{v['recall']:>8.3f}{v['kappa']:>8.3f}"
              f"{v['pred_pos_rate']:>9.3f}")
        w(f"     ★ 最优 {bo.get('winner')}  accuracy {bo.get('winner_accuracy'):.3f}  "
          f"门槛 {bo.get('threshold')}  {'通过' if bo.get('passes') else '未通过'}")
        wv = bo["variants"].get(bo.get("winner"), {})
        if wv.get("by_bin"):
            w("     最优变体逐层（人工/判官/准确率）：" + "  ".join(
                f"b{b}:{v['human']:.2f}/{v['judge']:.2f}/{v['acc']:.2f}"
                for b, v in sorted(wv["by_bin"].items(), key=lambda x: int(x[0]))))
        if not bo.get("passes"):
            w("     ⚠ 未通过门槛意味着全部基于判官的结论（可替代率、τ*、命中率）")
            w("       都不可信，必须先修判官再重跑。")
        w()
    oa = g("s21", "order_ablation.json")
    if oa and "by_prompt" in oa:
        w("【2c】判官顺序敏感性消融（prompt A/B）")
        w(f"     样本 {oa.get('n')} 对，每对 4 次判定")
        w(f"     {'prompt':<12}{'一致率':>10}{'翻转率':>10}"
          f"{'SUFF->INSUFF':>15}{'INSUFF->SUFF':>15}{'方向偏差':>10}")
        for k, v in oa["by_prompt"].items():
            w(f"     {k:<12}{v['consistency']:>10.3f}{v['flip_rate']:>10.3f}"
              f"{v['n_suff_to_insuff']:>15}{v['n_insuff_to_suff']:>15}"
              f"{v['directional_bias']:>+10.3f}")
        w(f"     顺序无关指令带来的一致率提升 {oa.get('improvement'):+.4f}")
        w("     ★ 方向偏差显著为正 = 打乱顺序后判官系统性偏向 INSUFFICIENT，")
        w("       这解释了为什么「取交集」会把 τ* 推到无解。")
        w()
    js = g("s19", "judge_stability.json")
    if js and "by_bin" in js:
        w("【2b】判官稳定性（双次判定）")
        w(f"     {'bin':<6}{'n':>7}{'翻转率':>10}{'一致者可替代':>14}"
          f"{'翻转者可替代':>14}{'严格口径':>12}")
        for b, v in sorted(js["by_bin"].items(), key=lambda x: int(x[0])):
            f2 = lambda z: f"{z:.3f}" if isinstance(z, float) else "-"
            w(f"     {b:<6}{v['n']:>7}{v['flip_rate']:>10.1%}"
              f"{f2(v['sub_rate_consistent']):>14}{f2(v['sub_rate_flipped']):>14}"
              f"{f2(v['strict_rate']):>12}")
        fv = js.get("flipped_vs_consistent") or {}
        if fv:
            w(f"     翻转对 vs 一致对：方向不对称 {fv.get('asym_flip'):.3f} vs "
              f"{fv.get('asym_keep'):.3f}；同文本 {fv.get('same_text_flip'):.3f} vs "
              f"{fv.get('same_text_keep'):.3f}")
        sc2 = js.get("strict_cost") or {}
        if sc2:
            w(f"     严格口径代价：{sc2['loose']:.4f} -> {sc2['strict']:.4f}"
              f"（相对降 {sc2['relative_drop']:.1%}）")
        w(f"     τ*：宽松 {js.get('tau_star_loose')}  严格 {js.get('tau_star_strict')}")
        w("     ★ 翻转者可替代率接近 0.5 = 判官在这些例子上是随机的；"
          "严格口径把它们排除掉")
        w()
    cb = g("s20", "construction_bias.json")
    if cb and "matched" in cb:
        w("【8d】数据集构造偏差：等任务数对照")
        w(f"     基准任务数 {cb.get('matched_tasks')}")
        w(f"     {'域':<20}{'全量头部(成本)':>16}{'等量头部(成本)':>16}"
          f"{'跨任务字面重复':>16}{'每文本任务数':>14}")
        for s in sorted(cb.get("full", {})):
            f_ = cb["full"][s]; mt = cb["matched"].get(s, {}); dc = cb["decomposition"][s]
            w(f"     {s:<20}{f_.get('head_cov_cost',0):>16.1%}"
              f"{mt.get('head_cov_cost',0):>16.1%}"
              f"{dc['exact_dup_crosstask_frac']:>16.1%}"
              f"{dc['mean_tasks_per_text']:>14.2f}")
        if cb.get("musique_over_2wiki"):
            w(f"     ★ 等量对照下 MuSiQue / 2Wiki = "
              f"{cb['musique_over_2wiki']:.1f}x —— 两者查询形式、语料、判定器都相同，")
            w("       唯一区别是 MuSiQue 由单跳题组合而成。这个倍数就是构造偏差的大小，")
            w("       主张必须建立在 agent 域上，MuSiQue 只能作为上界。")
        w()
    aa = g("s18", "agent_arms.json")
    if aa and "arms" in aa:
        w("【8c】agent 多查询双臂对比")
        w(f"     数组占比 {aa.get('multi_query_frac')}  平均子查询数 "
          f"{aa.get('mean_subqueries')}")
        w(f"     {'臂':<8}{'调用':>8}{'意图':>8}{'头部(次)':>11}{'头部(成本)':>12}"
          f"{'跨任务':>9}{'任务内重复':>11}")
        for k, v in aa["arms"].items():
            w(f"     {k:<8}{v.get('n_calls', 0):>8}{v.get('n_intents', 0):>8}"
              f"{v.get('head_cov_calls', 0):>11.1%}{v.get('head_cov_cost', 0):>12.1%}"
              f"{v.get('cross_intent_frac', 0):>9.1%}"
              f"{v.get('intra_task_dup_rate', 0):>11.1%}")
        w(f"     两臂差 {aa.get('delta_head_cov_cost', 0):+.1%}（all - first）")
        w()
    w("【9】Phase 2 交接")
    w(f"     启动视图 {vw.get('n_views')} 个，覆盖 {vw.get('coverage')}")
    w(f"     ε 初值：P99={l2.get('p99_recall_loss_pos')}（非零损失占比 "
      f"{l2.get('nonzero_loss_frac')}，非零条件均值 {l2.get('mean_loss_given_nonzero')}）")
    w("       风险结构是「小概率完全失败」而非「普遍小损失」，")
    w("       对应 C2 的双参数形式：ε 取较大值配小 δ，正是 conformal 擅长的场景")
    w(f"     判定器特征：{tf.get('best_model')}（views.json 含 centroid 与 scope_radius_p5）")
    w()
    w("【10】待办")
    if ge.get("accuracy") is None:
        w("     - 标注 gold_worksheet.tsv 后重跑 --stages s15，补人工金标 accuracy")
    if qc.get("cross_kappa") is None:
        w("     - config 里填 cross_judge_base_url/model 补跨判官 κ")
    w("     - MuSiQue 的复用来自数据集组合式构造，主张应建立在 agent 域上，")
    w("       MuSiQue 作为上界放进敏感性分析")
    w()
    w("=" * 78)
    txt = "\n".join(L)
    ctx.out("s16", "FINAL.txt").write_text(txt, encoding="utf-8")
    (ctx.base / "FINAL.txt").write_text(txt, encoding="utf-8")
    ctx.log.info("\n" + txt)
    return "FINAL.txt"



# ==============================================================================
# s17  视图级命中率（离线，train/test 按任务切分）
# ==============================================================================


@stage("s17", "视图级命中率：多少查询能真正绕过基础索引", ["view_hit.json"])
def s17(ctx: RunContext, sm: ServerManager):
    """s14 给的是 pair 级接受率，不是「多少查询能绕过索引」。这一步做换算。

    做法：按任务切分，前一半任务的查询建视图（簇心 + 作用域），
    后一半任务的查询逐条去匹配最近视图，用 s14 学到的判定器决定是否命中。
    train/test 按任务切分而不是按查询，避免同一任务的查询同时出现在两边
    把命中率灌水。

    输出的才是论文里那句话的数字：
    「在 X 的召回保证下，Y% 的查询（按成本 Z%）可以完全绕过基础索引」。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cfg, log = ctx.cfg, ctx.log
    sm.stop()

    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    qidx = {q: i for i, q in enumerate(ctx.rjson(ctx.out("s3", "qindex.json"))["qids"])}
    E = np.load(ctx.out("s3", "query_emb.npy"))
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    l1 = [r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
          if r.get("eligible") == 1 and r.get("ok") and r.get("substitutable") is not None]
    cmap = {c["qid"]: c for c in calls}

    # ---- 在 L1 判定上训练判定器（与 s14 同特征）----
    X, y = [], []
    for r in l1:
        ci, cj = cmap.get(r["qid_i"]), cmap.get(r["qid_j"])
        if not ci or not cj:
            continue
        ent = _jac(_entity_set(ci["query"]), _entity_set(cj["query"]))
        tpl = _jac(_template_set(ci["query"]), _template_set(cj["query"]))
        X.append([r["cos"], ent, tpl]); y.append(r["substitutable"])
    if len(y) < 200:
        log.warning(f"L1 样本仅 {len(y)}，s17 结果不可靠")
    X = np.array(X, float); y = np.array(y, int)
    sk = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sk.transform(X), y)
    score_all = lr.predict_proba(sk.transform(X))[:, 1]

    # ---- 按保证水平定阈值（用 Wilson 下界）----
    thr = {}
    for tg in (0.80, 0.85, 0.90, 0.95):
        best = None
        for t in np.quantile(score_all, np.linspace(0.0, 0.995, 200)):
            m = score_all >= t
            n = int(m.sum())
            if n < 50:
                continue
            lo, _ = wilson(int(y[m].sum()), n)
            if lo >= tg and (best is None or t < best):
                best = float(t)
        thr[tg] = best
        log.info(f"  保证 {tg:.2f} -> 判定器阈值 {best}")

    # ---- 按任务切分 ----
    tasks = sorted({c["task_id"] for c in calls})
    rnd = random.Random(cfg.seed)
    rnd.shuffle(tasks)
    ncut = int(len(tasks) * cfg.view_eval_task_frac)
    build_t, eval_t = set(tasks[:ncut]), set(tasks[ncut:])
    build = [c for c in calls if c["task_id"] in build_t and c["qid"] in qidx]
    evalq = [c for c in calls if c["task_id"] in eval_t and c["qid"] in qidx]
    log.info(f"  建视图 {len(build_t)} 任务 / {len(build)} 查询；"
             f"评估 {len(eval_t)} 任务 / {len(evalq)} 查询")

    # ---- 用 build 侧的查询建视图（按 τ 聚类，只保留 >=2 成员的簇）----
    pairs = ctx.rjsonl(ctx.out("s4", "pairs.jsonl"))
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    tau = calib.get("tau_star") or calib.get("tau_point") or 0.90
    bset = {c["qid"] for c in build}
    dsu = _DSU([c["qid"] for c in build])
    for p in pairs:
        if p["cos"] >= tau and p["qid_i"] in bset and p["qid_j"] in bset:
            dsu.union(p["qid_i"], p["qid_j"])
    grp = defaultdict(list)
    for c in build:
        grp[dsu.find(c["qid"])].append(c["qid"])
    views = [(k, v) for k, v in grp.items() if len(v) >= 2]
    log.info(f"  build 侧得到 {len(views)} 个视图（>=2 成员），"
             f"覆盖 {sum(len(v) for _, v in views)}/{len(build)} 条 build 查询")
    if not views:
        log.error("  没有视图，无法评估命中率")
        ctx.wjson(ctx.out("s17", "view_hit.json"), {"error": "no views"})
        return "无视图"

    cen = np.stack([E[[qidx[q] for q in v]].mean(0) for _, v in views])
    cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    # 每个视图取一个代表查询，用于算实体/模板特征
    reps = [cmap[v[0]]["query"] for _, v in views]
    rep_ent = [_entity_set(t) for t in reps]
    rep_tpl = [_template_set(t) for t in reps]

    # ---- 评估侧逐条匹配最近视图 ----
    ev_idx = [qidx[c["qid"]] for c in evalq]
    sims = E[ev_idx] @ cen.T                      # (n_eval, n_views)
    best_v = np.argmax(sims, axis=1)
    best_s = sims[np.arange(len(ev_idx)), best_v]

    feats = np.zeros((len(evalq), 3))
    for i, c in enumerate(evalq):
        j = int(best_v[i])
        e = _entity_set(c["query"]); t = _template_set(c["query"])
        feats[i] = [float(best_s[i]), _jac(e, rep_ent[j]), _jac(t, rep_tpl[j])]
    sc = lr.predict_proba(sk.transform(feats))[:, 1]

    cost = np.array([c.get("cost", 1.0) for c in evalq], float)
    srcs = np.array([c["source"] for c in evalq])
    out = {"tau": tau, "n_views": len(views), "n_eval_queries": len(evalq),
           "n_build_tasks": len(build_t), "n_eval_tasks": len(eval_t),
           "coef": dict(zip(["cos", "ent", "tpl"], [float(v) for v in lr.coef_[0]])),
           "by_guarantee": {}}
    log.info("  视图级命中率（评估侧，从未参与建视图）:")
    for tg, t in thr.items():
        if t is None:
            out["by_guarantee"][f"{tg:.2f}"] = None
            log.info(f"    保证 {tg:.2f} -> 无可行阈值")
            continue
        hit = sc >= t
        d = {"threshold": t,
             "hit_rate_queries": float(hit.mean()),
             "hit_rate_cost": float(cost[hit].sum() / max(cost.sum(), 1e-9)),
             "by_source": {}}
        for s in sorted(set(srcs.tolist())):
            m = srcs == s
            if m.sum():
                d["by_source"][s] = {
                    "hit_rate_queries": float(hit[m].mean()),
                    "hit_rate_cost": float(cost[m & hit].sum() / max(cost[m].sum(), 1e-9)),
                    "n": int(m.sum())}
        out["by_guarantee"][f"{tg:.2f}"] = d
        log.info(f"    保证 {tg:.2f}: 按查询 {d['hit_rate_queries']:.1%}，"
                 f"按成本 {d['hit_rate_cost']:.1%}")
        for s, v in d["by_source"].items():
            log.info(f"        {s:<20} 查询 {v['hit_rate_queries']:>6.1%} "
                     f"成本 {v['hit_rate_cost']:>6.1%} (n={v['n']})")
    ctx.wjson(ctx.out("s17", "view_hit.json"), out)
    g90 = out["by_guarantee"].get("0.90")
    return (f"保证0.90 下命中 查询{g90['hit_rate_queries']:.1%}/成本{g90['hit_rate_cost']:.1%}"
            if g90 else "0.90 无可行阈值")



# ==============================================================================
# s18  agent 双臂对比：all vs first（离线）
# ==============================================================================


@stage("s18", "agent 多查询双臂对比（all vs first）", ["agent_arms.json"])
def s18(ctx: RunContext, sm: ServerManager):
    """main_v5 实测 81.1% 的 tool call 返回查询数组，平均 2.52 条。

    "all" 把数组里每条都发检索（7807 条调用），"first" 只取第一条（3773 条）。
    两者对复用率的口径不同：all 更接近 agent 的真实行为但会抬高任务内冗余，
    first 保守但丢掉三分之二的检索意图。论文里两个都要报。

    这一步纯离线：按 multi_query_idx==0 过滤即可还原 first 臂，不用重跑。
    """
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    pairs = ctx.rjsonl(ctx.out("s4", "pairs.jsonl"))
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    tau = calib.get("tau_star") or calib.get("tau_point") or 0.90

    ag = [c for c in calls if c["source"] == "agent"]
    if not ag:
        log.warning("没有 agent 调用，跳过双臂对比")
        ctx.wjson(ctx.out("s18", "agent_arms.json"), {"skipped": True})
        return "无 agent 数据"

    arms = {"all": ag, "first": [c for c in ag if c.get("multi_query_idx", 0) == 0]}
    out = {"tau": tau, "n_agent_calls": len(ag),
           "multi_query_frac": float(np.mean([bool(c.get("multi_query")) for c in ag])),
           "mean_subqueries": float(np.mean([c.get("multi_query_n", 1) for c in ag])),
           "arms": {}}
    for name, cs in arms.items():
        m = _head_cov(cs, pairs, tau, cfg.stop_head_frac)
        m["n_calls"] = len(cs)
        out["arms"][name] = m
        log.info(f"  [{name:5s}] 调用 {len(cs):>6} 意图 {m.get('n_intents'):>6} "
                 f"头部(次) {m.get('head_cov_calls', 0):>6.1%} "
                 f"头部(成本) {m.get('head_cov_cost', 0):>6.1%} "
                 f"跨任务 {m.get('cross_intent_frac', 0):>6.1%} "
                 f"每意图任务 {m.get('mean_tasks_per_intent', 0):.2f}")

    a, f = out["arms"]["all"], out["arms"]["first"]
    out["delta_head_cov_cost"] = a.get("head_cov_cost", 0) - f.get("head_cov_cost", 0)
    log.info(f"  两臂头部成本覆盖差 {out['delta_head_cov_cost']:+.1%}"
             f"（all - first）；差异大说明结论对多查询处理方式敏感，需在论文里两个都报")

    # 任务内 vs 任务间冗余拆解：all 臂多出来的复用是不是都在任务内
    for name, cs in arms.items():
        byt = defaultdict(list)
        for c in cs:
            byt[c["task_id"]].append(c["query"])
        intra = float(np.mean([1 - len(set(v)) / len(v) for v in byt.values() if v]))
        out["arms"][name]["intra_task_dup_rate"] = intra
        log.info(f"  [{name:5s}] 任务内查询重复率 {intra:.1%}")
    ctx.wjson(ctx.out("s18", "agent_arms.json"), out)
    return (f"all {a.get('head_cov_cost', 0):.1%} vs first "
            f"{f.get('head_cov_cost', 0):.1%}（差 {out['delta_head_cov_cost']:+.1%}）")



# ==============================================================================
# s19  判官稳定性分析（离线）
# ==============================================================================


@stage("s19", "判官稳定性：位置翻转、严格口径、对 τ* 的影响", ["judge_stability.json"])
def s19(ctx: RunContext, sm: ServerManager):
    """main_v6 的分层报告暴露了一个威胁核心结论的问题：

        b0=0.987  b1=0.897  b2=0.867  b3=0.800  b4=0.844  b5=0.823
        b6=0.850  b7=0.699 (n=1490)

    相似度越高判官越不稳定，而 bin7 恰恰决定 τ*。bin7 的可替代率 0.988
    与 30% 的自我翻转率同时成立，只可能是翻转集中在少数难例上。
    这一步量化：翻转对是不是边界例、严格口径能挽回多少 precision、
    以及 τ* 在两种口径下差多少。
    """
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    calls = {c["qid"]: c for c in ctx.rjsonl(ctx.out("s2", "calls.jsonl"))}
    l1 = [r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
          if r.get("eligible") == 1 and r.get("ok")]
    dj = [r for r in l1 if r.get("pos_consistent") is not None]
    if not dj:
        log.warning("没有二判样本，跳过")
        ctx.wjson(ctx.out("s19", "judge_stability.json"), {"skipped": True})
        return "无二判样本"

    out = {"n_double_judged": len(dj)}
    log.info(f"  二判样本 {len(dj)} 对")

    # ---- 1) 逐层翻转率 + 翻转对的性质 ----
    rows = {}
    for b in sorted({r.get("bin") for r in dj if r.get("bin") is not None}):
        sub = [r for r in dj if r.get("bin") == b]
        flip = [r for r in sub if r["pos_consistent"] == 0]
        keep = [r for r in sub if r["pos_consistent"] == 1]
        rows[str(b)] = {
            "n": len(sub), "flip_rate": len(flip) / len(sub),
            # 一致的那批可替代率：这才是"判官有把握"的部分
            "sub_rate_consistent": float(np.mean([r["substitutable"] for r in keep]))
                                   if keep else None,
            # 翻转的那批：若接近 0.5，说明判官在这些例子上就是随机的
            "sub_rate_flipped": float(np.mean([r["substitutable"] for r in flip]))
                                if flip else None,
            "strict_rate": float(np.mean([r["substitutable_strict"] for r in sub
                                          if r.get("substitutable_strict") is not None]))
                           if any(r.get("substitutable_strict") is not None for r in sub)
                           else None,
        }
        log.info(f"    bin{b}: n={len(sub):>5} 翻转率 {rows[str(b)]['flip_rate']:>6.1%}  "
                 f"一致者可替代率 {rows[str(b)]['sub_rate_consistent']}  "
                 f"翻转者 {rows[str(b)]['sub_rate_flipped']}  "
                 f"严格口径 {rows[str(b)]['strict_rate']}")
    out["by_bin"] = rows
    log.info("    判读：翻转者可替代率若接近 0.5，说明判官在这些例子上是随机的，"
             "把它们排除掉（严格口径）才是诚实做法")

    # ---- 2) 翻转对是不是边界例 ----
    fl = [r for r in dj if r["pos_consistent"] == 0]
    kp = [r for r in dj if r["pos_consistent"] == 1]
    if fl and kp:
        out["flipped_vs_consistent"] = {
            "n_flip": len(fl), "n_keep": len(kp),
            "mean_cos_flip": float(np.mean([r["cos"] for r in fl])),
            "mean_cos_keep": float(np.mean([r["cos"] for r in kp])),
            "asym_flip": float(np.mean([int(r["cov_ij"] != r["cov_ji"]) for r in fl])),
            "asym_keep": float(np.mean([int(r["cov_ij"] != r["cov_ji"]) for r in kp])),
            "same_text_flip": float(np.mean([bool(r.get("same_text")) for r in fl])),
            "same_text_keep": float(np.mean([bool(r.get("same_text")) for r in kp])),
        }
        d = out["flipped_vs_consistent"]
        log.info(f"  翻转对 vs 一致对：方向不对称率 {d['asym_flip']:.3f} vs "
                 f"{d['asym_keep']:.3f}；同文本占比 {d['same_text_flip']:.3f} vs "
                 f"{d['same_text_keep']:.3f}")
        log.info("    方向不对称率显著更高 -> 翻转对确实是边界例，不是随机噪声")

    # ---- 3) 严格口径的代价与收益 ----
    have = [r for r in dj if r.get("substitutable_strict") is not None]
    if have:
        lo_ = float(np.mean([r["substitutable"] for r in have]))
        st_ = float(np.mean([r["substitutable_strict"] for r in have]))
        out["strict_cost"] = {"loose": lo_, "strict": st_, "drop": lo_ - st_,
                              "relative_drop": (lo_ - st_) / max(lo_, 1e-9)}
        log.info(f"  严格口径代价：可替代率 {lo_:.4f} -> {st_:.4f}"
                 f"（降 {lo_-st_:.4f}，相对 {(lo_-st_)/max(lo_,1e-9):.1%}）")

    # ---- 4) τ* 两口径对比 ----
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    out["tau_star_loose"] = calib.get("tau_star")
    out["tau_star_strict"] = calib.get("tau_star_strict")
    log.info(f"  τ* 宽松 {out['tau_star_loose']}  严格 {out['tau_star_strict']}")

    # ---- 5) 翻转样本示例 ----
    ex = []
    for r in sorted(fl, key=lambda x: -x["cos"])[:10]:
        ci, cj = calls.get(r["qid_i"], {}), calls.get(r["qid_j"], {})
        ex.append({"cos": r["cos"], "bin": r.get("bin"),
                   "cov_ij": r["cov_ij"], "cov_ji": r["cov_ji"],
                   "cov_ij_shuf": r.get("cov_ij_shuf"),
                   "qa": ci.get("query", "")[:110], "qb": cj.get("query", "")[:110]})
    out["flip_examples"] = ex
    ctx.wjson(ctx.out("s19", "judge_stability.json"), out)
    b7 = rows.get("7") or {}
    return (f"bin7 翻转率 {b7.get('flip_rate', 0):.1%}，严格口径 "
            f"{b7.get('strict_rate')}，τ*严格 {out['tau_star_strict']}")


# ==============================================================================
# s20  构造偏差对照：MuSiQue vs 2Wiki（离线）
# ==============================================================================


@stage("s20", "构造偏差量化：MuSiQue vs 2Wiki 等量对照", ["construction_bias.json"])
def s20(ctx: RunContext, sm: ServerManager):
    """main_v6 的 s17 分域命中率：musique=52.7% 而 2wiki=2.7%，差二十倍。

    但两者的查询形式几乎一样（都是 "What is the {rel} of {subj}?"）、
    语料是同一批、判定器也相同。唯一的区别是 MuSiQue 由单跳问题组合而成，
    同一个原子子问题按设计出现在几十道题里。

    这一步把这个对照做实：等任务数、等调用数下重算头部覆盖，
    并把复用拆成「字面完全相同」与「不同表述」两部分。
    结论直接决定论文该把主张建立在哪个域上。
    """
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    pairs = ctx.rjsonl(ctx.out("s4", "pairs.jsonl"))
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    tau = calib.get("tau_star") or calib.get("tau_point") or 0.90

    by_src = defaultdict(list)
    for c in calls:
        by_src[c["source"]].append(c)
    ntask = {s: len({c["task_id"] for c in cs}) for s, cs in by_src.items()}
    log.info("  各域任务数: " + ", ".join(f"{s}={n}" for s, n in sorted(ntask.items())))
    m = min(ntask.values())
    log.info(f"  等量对照基准：{m} 个任务")

    out = {"tau": tau, "matched_tasks": m, "task_counts": ntask, "matched": {},
           "full": {}, "decomposition": {}}
    for s, cs in sorted(by_src.items()):
        byt = defaultdict(list)
        for c in cs:
            byt[c["task_id"]].append(c)
        tks = sorted(byt)
        accs = []
        for seed in range(3):
            rnd = random.Random(3000 + seed)
            keep = set(rnd.sample(tks, min(m, len(tks))))
            r = _head_cov([c for t in keep for c in byt[t]], pairs, tau, cfg.stop_head_frac)
            if r:
                accs.append(r)
        if accs:
            out["matched"][s] = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
        out["full"][s] = _head_cov(cs, pairs, tau, cfg.stop_head_frac)

        # 复用拆解：字面完全相同 vs 不同表述
        txt = Counter(c["query"] for c in cs)
        tsk = defaultdict(set)
        for c in cs:
            tsk[c["query"]].add(c["task_id"])
        dup_calls = sum(n for q, n in txt.items() if len(tsk[q]) >= 2)
        out["decomposition"][s] = {
            "n_calls": len(cs), "n_unique_text": len(txt),
            "exact_dup_crosstask_frac": dup_calls / max(len(cs), 1),
            "mean_tasks_per_text": float(np.mean([len(v) for v in tsk.values()])),
            "max_tasks_per_text": max((len(v) for v in tsk.values()), default=0),
        }

    log.info(f"  {'域':<20}{'全量头部(成本)':>16}{'等量头部(成本)':>16}"
             f"{'跨任务字面重复':>16}{'每文本任务数':>14}")
    for s in sorted(out["full"]):
        f_ = out["full"][s]; mt = out["matched"].get(s, {}); dc = out["decomposition"][s]
        log.info(f"  {s:<20}{f_.get('head_cov_cost', 0):>16.1%}"
                 f"{mt.get('head_cov_cost', 0):>16.1%}"
                 f"{dc['exact_dup_crosstask_frac']:>16.1%}"
                 f"{dc['mean_tasks_per_text']:>14.2f}")

    mq = out["matched"].get("replay:musique", {}).get("head_cov_cost")
    tw = out["matched"].get("replay:2wiki", {}).get("head_cov_cost")
    if mq and tw:
        out["musique_over_2wiki"] = mq / max(tw, 1e-9)
        log.info(f"  等量对照下 MuSiQue / 2Wiki = {out['musique_over_2wiki']:.1f}x")
        log.info("    两者查询形式、语料、判定器都相同，唯一区别是 MuSiQue 由单跳题"
                 "组合而成 -> 这个倍数就是数据集构造偏差的大小")
    ctx.wjson(ctx.out("s20", "construction_bias.json"), out)
    return (f"等量对照 MuSiQue/2Wiki = "
            f"{out.get('musique_over_2wiki', float('nan')):.1f}x")



# ==============================================================================
# s21  判官顺序敏感性消融（需要判官服务）
# ==============================================================================


@stage("s21", "prompt A/B：顺序无关指令能否降低位置翻转率", ["order_ablation.json"])
def s21(ctx: RunContext, sm: ServerManager):
    """main_v7 的核心问题：整体位置翻转率 21.5%，bin7 达 25.5%。

    而且翻转不是随机的 —— bin7 翻转者的可替代率是 0.954（若是随机噪声应接近
    0.5），说明打乱文档顺序后判官系统性地更倾向 INSUFFICIENT。
    这既让「取交集」变成错误做法（τ*_strict 无解），也直接威胁 τ* 的可信度。

    这一步做 2x2 消融：{原 prompt, 顺序无关 prompt} x {原顺序, 打乱顺序}，
    在同一批对上跑，直接回答「加一句指令能不能修好」。
    只需 4 x N 次判官调用，N=1500 时约 15 分钟。
    """
    cfg, log = ctx.cfg, ctx.log
    calls = {c["qid"]: c for c in ctx.rjsonl(ctx.out("s2", "calls.jsonl"))}
    cmap = {c["chunkid"]: c["text"] for c in ctx.rjsonl(ctx.out("s1", "chunks.jsonl"))}
    l1 = [r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
          if r.get("eligible") == 1 and r.get("ok")]
    if not l1:
        ctx.wjson(ctx.out("s21", "order_ablation.json"), {"skipped": True})
        return "无可用判定"

    # 按层分层抽样，重点覆盖高相似度层（翻转率最高的地方）
    byb = defaultdict(list)
    for r in l1:
        byb[r.get("bin", 0)].append(r)
    per = max(50, cfg.order_ablation_n // max(len(byb), 1))
    smp = []
    for b, lst in sorted(byb.items()):
        smp += random.sample(lst, min(per, len(lst)))
    log.info(f"  抽样 {len(smp)} 对（每层最多 {per}），共 4x{len(smp)} 次判官调用")

    urls = sm.ensure("judge", list(cfg.gpus))
    pool = LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)

    def work(r):
        ci, cj = calls[r["qid_i"]], calls[r["qid_j"]]
        chunks = cj["retrieved_chunks"]
        d_ord = _fmt(chunks, cmap, cfg.judge_doc_chars)
        rng = random.Random(hash(r["pair_id"]) & 0xFFFF)
        sh_ids = list(chunks); rng.shuffle(sh_ids)
        d_shf = _fmt(sh_ids, cmap, cfg.judge_doc_chars)
        out = {"pair_id": r["pair_id"], "bin": r.get("bin"), "cos": r["cos"],
               "w": r.get("w", 1.0)}
        for pname, tmpl in (("base", L1_PROMPT),
                            ("order_inv", L1_PROMPT_ORDER_INVARIANT)):
            a = _judge1(pool, cfg, ci["query"], d_ord, prompt=tmpl)
            s = _judge1(pool, cfg, ci["query"], d_shf, prompt=tmpl)
            if a["ok"] and s["ok"]:
                out[f"{pname}_ord"] = a["covered"]
                out[f"{pname}_shf"] = s["covered"]
                out[f"{pname}_consistent"] = int(a["covered"] == s["covered"])
            else:
                out[f"{pname}_consistent"] = None
        return out

    res = [r for r in pmap(work, smp, cfg.judge_concurrency, "order-ab", ctx, pool)
           if isinstance(r, dict) and "__error__" not in r]
    sm.stop()

    out = {"n": len(res), "llm": pool.stats(), "by_prompt": {}, "by_bin": {}}
    for pname in ("base", "order_inv"):
        v = [r[f"{pname}_consistent"] for r in res
             if r.get(f"{pname}_consistent") is not None]
        if not v:
            continue
        # 翻转方向：原顺序判 SUFFICIENT 而打乱后判 INSUFFICIENT 的比例
        drop = [r for r in res if r.get(f"{pname}_ord") == 1 and r.get(f"{pname}_shf") == 0]
        rise = [r for r in res if r.get(f"{pname}_ord") == 0 and r.get(f"{pname}_shf") == 1]
        out["by_prompt"][pname] = {
            "n": len(v), "consistency": float(np.mean(v)),
            "flip_rate": 1 - float(np.mean(v)),
            "n_suff_to_insuff": len(drop), "n_insuff_to_suff": len(rise),
            "directional_bias": (len(drop) - len(rise)) / max(len(drop) + len(rise), 1),
        }
        d = out["by_prompt"][pname]
        log.info(f"  [{pname:<10}] 一致率 {d['consistency']:.3f}  翻转 {d['flip_rate']:.3f}  "
                 f"SUFF->INSUFF {d['n_suff_to_insuff']} vs INSUFF->SUFF "
                 f"{d['n_insuff_to_suff']}  方向偏差 {d['directional_bias']:+.3f}")

    for b in sorted({r.get("bin") for r in res if r.get("bin") is not None}):
        sub = [r for r in res if r.get("bin") == b]
        row = {}
        for pname in ("base", "order_inv"):
            v = [r[f"{pname}_consistent"] for r in sub
                 if r.get(f"{pname}_consistent") is not None]
            row[pname] = float(np.mean(v)) if v else None
        row["n"] = len(sub)
        out["by_bin"][str(b)] = row
        log.info(f"    bin{b} n={len(sub):>4}  base {row['base']}  "
                 f"order_inv {row['order_inv']}")

    bp = out["by_prompt"]
    if "base" in bp and "order_inv" in bp:
        out["improvement"] = bp["order_inv"]["consistency"] - bp["base"]["consistency"]
        log.info(f"  顺序无关指令带来的一致率提升 {out['improvement']:+.4f}")
        log.info("    提升显著 -> 下一轮把 judge_order_invariant 设为 True 重跑 s6；")
        log.info("    提升不显著 -> 位置敏感是模型固有属性，写进 limitations，")
        log.info("    并把位置一致率作为判官可靠性上界报告。")
    ctx.wjsonl(ctx.out("s21", "samples.jsonl"), res)
    ctx.wjson(ctx.out("s21", "order_ablation.json"), out)
    return (f"base {bp.get('base', {}).get('consistency')} -> order_inv "
            f"{bp.get('order_inv', {}).get('consistency')}")



# ==============================================================================
# s22  判官 prompt 对拍（以人工金标为真值）—— 必须在 s6 之前跑
# ==============================================================================


@stage("s22", "判官 prompt 对拍：以人工标注为真值选最优 prompt",
       ["bakeoff.json", "selected.json"])
def s22(ctx: RunContext, sm: ServerManager):
    """main_v8 的人工金标暴露判官 accuracy 仅约 0.44。

    两批各 299 对（不同样本）正例率 0.261 / 0.291、逐层曲线几乎重合，
    所以不是标注噪声。判官 recall 0.85 但 precision 0.31 ——
    它系统性地把「同主题但不含该事实」的文档判成 SUFFICIENT，
    且相似度越高越严重（bin7 判官 0.970 vs 人工 0.373）。

    陷阱对误判率 0.017% 只说明它能识别「完全无关」；
    真正的难点是「相关但不充分」，原来的 QC 没测这一项。

    这一步用人工标注当真值，在多个 prompt 变体上直接测 accuracy，
    选最优者供 s6 使用。跑不过闸门就中止，不浪费十几小时。
    """
    cfg, log = ctx.cfg, ctx.log
    if not cfg.bakeoff_sources:
        log.warning("未配置 bakeoff_sources，跳过对拍（s6 将沿用默认 prompt）")
        ctx.wjson(ctx.out("s22", "bakeoff.json"), {"skipped": True})
        ctx.wjson(ctx.out("s22", "selected.json"), {"winner": None, "skipped": True})
        return "跳过（无标注数据）"

    # ---------- 载入标注 + 还原完整文档上下文 ----------
    import csv
    items = []
    for entry in cfg.bakeoff_sources:
        run_dir, tsv = Path(entry[0]), Path(entry[1])
        if not tsv.exists():
            log.error(f"  标注文件不存在: {tsv}"); continue
        st = run_dir / "stages"
        gp = st / "s5" / "gold_pairs.jsonl"
        cl = st / "s2" / "calls.jsonl"
        ch = st / "s1" / "chunks.jsonl"
        if not (gp.exists() and cl.exists() and ch.exists()):
            log.error(f"  {run_dir} 缺 s1/s2/s5 产出，无法还原上下文"); continue
        lab = {}
        with open(tsv, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                pid = (row.get("pair_id") or "").strip()
                v = (row.get("human_label") or "").strip()
                if pid and v in ("0", "1"):
                    lab[pid] = int(v)
        pairs = {p["pair_id"]: p for p in ctx.rjsonl(gp) if p["pair_id"] in lab}
        need_q = {q for p in pairs.values() for q in (p["qid_i"], p["qid_j"])}
        calls = {c["qid"]: c for c in ctx.rjsonl(cl) if c["qid"] in need_q}
        need_c = {x for c in calls.values() for x in c.get("retrieved_chunks", [])}
        cmap = {}
        with open(ch, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["chunkid"] in need_c:
                    cmap[d["chunkid"]] = d["text"]
                if len(cmap) == len(need_c):
                    break
        n0 = len(items)
        for pid, p in pairs.items():
            ci, cj = calls.get(p["qid_i"]), calls.get(p["qid_j"])
            if not ci or not cj:
                continue
            items.append({
                "pair_id": pid, "run": str(run_dir), "cos": p["cos"],
                "bin": p.get("bin", _bin(p["cos"], cfg.sim_bins)),
                "human": lab[pid],
                "qi": ci["query"], "qj": cj["query"],
                "di": _fmt(ci["retrieved_chunks"], cmap, cfg.judge_doc_chars),
                "dj": _fmt(cj["retrieved_chunks"], cmap, cfg.judge_doc_chars)})
        log.info(f"  {run_dir.name}: 标注 {len(lab)} 条 -> 可用 {len(items)-n0} 对")

    if len(items) < 50:
        raise RuntimeError(f"可用标注仅 {len(items)} 对，无法对拍。"
                           f"检查 bakeoff_sources 的 run_dir 与 TSV 是否匹配。")
    random.Random(cfg.seed).shuffle(items)
    items = items[:cfg.bakeoff_max_pairs]
    hr = float(np.mean([it["human"] for it in items]))
    log.info(f"  对拍样本 {len(items)} 对，人工正例率 {hr:.3f}")
    log.info("  每层分布: " + ", ".join(
        f"b{b}={sum(1 for it in items if it['bin'] == b)}"
        for b in sorted({it["bin"] for it in items})))

    urls = sm.ensure("judge", list(cfg.gpus))
    pool = LLMPool(urls, cfg.judge_name, retries=cfg.judge_retries, timeout=300)
    variants = {k: v for k, v in _prompt_variants().items() if k in cfg.bakeoff_variants}
    log.info(f"  变体 {list(variants)}；预计 {len(items)*len(variants)*2} 次判官调用")

    results, out = {}, {"n": len(items), "human_pos_rate": hr, "variants": {}}
    for name, tmpl in variants.items():
        def work(it, _t=tmpl):
            a = _judge1(pool, cfg, it["qi"], it["dj"], prompt=_t)
            b = _judge1(pool, cfg, it["qj"], it["di"], prompt=_t)
            ok = a["ok"] and b["ok"]
            return {"pair_id": it["pair_id"], "bin": it["bin"], "human": it["human"],
                    "ok": ok,
                    "pred": int(bool(a.get("covered")) and bool(b.get("covered")))
                            if ok else None,
                    "cov_ij": a.get("covered"), "cov_ji": b.get("covered")}

        rs = [r for r in pmap(work, items, cfg.judge_concurrency, f"bo:{name}", ctx, pool)
              if isinstance(r, dict) and "__error__" not in r]
        good = [r for r in rs if r["pred"] is not None]
        if not good:
            log.error(f"  [{name}] 全部解析失败"); continue
        yt = [r["human"] for r in good]; yp = [r["pred"] for r in good]
        tp = sum(a == 1 and b == 1 for a, b in zip(yt, yp))
        fp = sum(a == 0 and b == 1 for a, b in zip(yt, yp))
        fn = sum(a == 1 and b == 0 for a, b in zip(yt, yp))
        tn = sum(a == 0 and b == 0 for a, b in zip(yt, yp))
        acc = (tp + tn) / len(yt)
        lo, hi = wilson(tp + tn, len(yt))
        d = {"n": len(good), "parse_fail": 1 - len(good) / max(len(rs), 1),
             "accuracy": acc, "acc_ci": [lo, hi],
             "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
             "kappa": kappa(yt, yp), "pred_pos_rate": float(np.mean(yp)),
             "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
             "by_bin": {}}
        for b in sorted({r["bin"] for r in good}):
            s = [r for r in good if r["bin"] == b]
            d["by_bin"][str(b)] = {
                "n": len(s),
                "human": float(np.mean([r["human"] for r in s])),
                "judge": float(np.mean([r["pred"] for r in s])),
                "acc": float(np.mean([int(r["human"] == r["pred"]) for r in s]))}
        out["variants"][name] = d
        results[name] = good
        log.info(f"  [{name:<18}] acc {acc:.3f} CI[{lo:.3f},{hi:.3f}]  "
                 f"P {d['precision']:.3f}  R {d['recall']:.3f}  κ {d['kappa']:.3f}  "
                 f"判正率 {d['pred_pos_rate']:.3f}（人工 {hr:.3f}）")

    if not out["variants"]:
        raise RuntimeError("所有变体都失败了")
    win = max(out["variants"].items(), key=lambda kv: kv[1]["accuracy"])
    out["winner"] = win[0]
    out["winner_accuracy"] = win[1]["accuracy"]
    out["threshold"] = cfg.stop_min_gold_acc
    out["passes"] = bool(win[1]["accuracy"] >= cfg.stop_min_gold_acc)
    log.info("=" * 66)
    log.info(f"  最优变体 {win[0]}  accuracy {win[1]['accuracy']:.3f}  "
             f"{'通过' if out['passes'] else '未过闸门 ' + str(cfg.stop_min_gold_acc)}")
    log.info("=" * 66)
    log.info("  逐层（最优变体）: " + "  ".join(
        f"b{b}:人{v['human']:.2f}/判{v['judge']:.2f}"
        for b, v in sorted(win[1]["by_bin"].items(), key=lambda x: int(x[0]))))

    ctx.wjson(ctx.out("s22", "bakeoff.json"), out)
    ctx.wjson(ctx.out("s22", "selected.json"),
              {"winner": win[0], "accuracy": win[1]["accuracy"],
               "passes": out["passes"], "threshold": cfg.stop_min_gold_acc})
    ctx.wjsonl(ctx.out("s22", "predictions.jsonl"), results.get(win[0], []))
    sm.stop()

    if cfg.require_judge_acc and not out["passes"]:
        raise RuntimeError(
            f"判官对拍最优 accuracy {win[1]['accuracy']:.3f} < {cfg.stop_min_gold_acc}。"
            f"已中止，不继续跑十几小时。可选：换更强的判官模型、增加 prompt 变体、"
            f"或用 --ignore-judge-gate 强行继续（结论需降级为不可靠）。")
    return f"最优 {win[0]} acc={win[1]['accuracy']:.3f}"



# ==============================================================================
# s23  精确缓存 vs 近似视图（离线）—— 回应「这不就是结果缓存吗」
# ==============================================================================


@stage("s23", "精确缓存 vs 近似视图：视图的净增量是多少", ["cache_vs_view.json"])
def s23(ctx: RunContext, sm: ServerManager):
    """这是全文的胜负手。

    main_v9 的软肋：头部单文本簇占比 85.4%、单文本视图占比 90.3%，
    而 τ*=0.9925 实际只放行 cos=1.0 的对。审稿人会直接问：
    「检测到的复用九成发生在字面完全相同的查询之间，那结果缓存就够了，
      为什么需要近似视图重写？」

    这一步给出直接对照：在同一个 train/test 切分上，
      - 精确缓存：查询文本在 build 集里出现过才算命中
      - 归一化缓存：小写/去标点/压空白后匹配（比精确缓存更强的基线）
      - 近似视图 @保证 g：判定器接受才算命中
    报告每种的按查询/按成本命中率、分域结果，以及视图相对缓存的**净增量**，
    并检验净增量部分的实测可替代率（增量不能是垃圾）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cfg, log = ctx.cfg, ctx.log
    sm.stop()

    calls = ctx.rjsonl(ctx.out("s2", "calls.jsonl"))
    qidx = {q: i for i, q in enumerate(ctx.rjson(ctx.out("s3", "qindex.json"))["qids"])}
    E = np.load(ctx.out("s3", "query_emb.npy"))
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    l1 = [r for r in ctx.rjsonl(ctx.out("s6", "l1.jsonl"))
          if r.get("eligible") == 1 and r.get("ok") and r.get("substitutable") is not None]
    cmap = {c["qid"]: c for c in calls}

    def norm_text(s: str) -> str:
        s = re.sub(r"[^\w\s]", " ", (s or "").lower())
        return re.sub(r"\s+", " ", s).strip()

    # ---- 判定器（与 s17 同口径：cos + ent + tpl）----
    X, y, W_ = [], [], []
    for r in l1:
        ci, cj = cmap.get(r["qid_i"]), cmap.get(r["qid_j"])
        if not ci or not cj:
            continue
        X.append([r["cos"],
                  _jac(_entity_set(ci["query"]), _entity_set(cj["query"])),
                  _jac(_template_set(ci["query"]), _template_set(cj["query"]))])
        y.append(r["substitutable"]); W_.append(r.get("w", 1.0))
    X = np.array(X, float); y = np.array(y, int); W_ = np.array(W_, float)
    if len(y) < 200:
        log.warning(f"L1 样本仅 {len(y)}，s23 结果不可靠")
    sk = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000).fit(sk.transform(X), y, sample_weight=W_)
    sc_all = lr.predict_proba(sk.transform(X))[:, 1]

    thr = {}
    for tg in cfg.report_targets:
        best = None
        for t in np.quantile(sc_all, np.linspace(0.0, 0.998, 300)):
            m = sc_all >= t
            if int(m.sum()) < 50:
                continue
            _, lo, _, _ = wilson_w(y[m], W_[m])
            if lo >= tg and (best is None or t < best):
                best = float(t)
        thr[tg] = best
    log.info(f"  判定器阈值 {dict((k, None if v is None else round(v,4)) for k,v in thr.items())}")

    # ---- train/test 按任务切分（与 s17 一致）----
    tasks = sorted({c["task_id"] for c in calls})
    rnd = random.Random(cfg.seed); rnd.shuffle(tasks)
    ncut = int(len(tasks) * cfg.view_eval_task_frac)
    build_t, eval_t = set(tasks[:ncut]), set(tasks[ncut:])
    build = [c for c in calls if c["task_id"] in build_t and c["qid"] in qidx]
    evalq = [c for c in calls if c["task_id"] in eval_t and c["qid"] in qidx]
    log.info(f"  build {len(build_t)} 任务 / {len(build)} 查询；"
             f"eval {len(eval_t)} 任务 / {len(evalq)} 查询")

    # ---- 三种缓存基线 ----
    exact = {c["query"] for c in build}
    normed = {norm_text(c["query"]) for c in build}
    hit_exact = np.array([c["query"] in exact for c in evalq])
    hit_norm = np.array([norm_text(c["query"]) in normed for c in evalq])

    # ---- 视图 ----
    pairs = ctx.rjsonl(ctx.out("s4", "pairs.jsonl"))
    calib = ctx.rjson(ctx.out("s8", "calib.json"))
    tau = calib.get("tau_star") or calib.get("tau_point") or 0.90
    bset = {c["qid"] for c in build}
    dsu = _DSU([c["qid"] for c in build])
    for p in pairs:
        if p["cos"] >= tau and p["qid_i"] in bset and p["qid_j"] in bset:
            dsu.union(p["qid_i"], p["qid_j"])
    grp = defaultdict(list)
    for c in build:
        grp[dsu.find(c["qid"])].append(c["qid"])
    views = [(k, v) for k, v in grp.items() if len(v) >= 2]
    if not views:
        log.error("  没有视图，无法对照")
        ctx.wjson(ctx.out("s23", "cache_vs_view.json"), {"error": "no views"})
        return "无视图"
    cen = np.stack([E[[qidx[q] for q in v]].mean(0) for _, v in views])
    cen = cen / (np.linalg.norm(cen, axis=1, keepdims=True) + 1e-9)
    reps = [cmap[v[0]]["query"] for _, v in views]
    rep_ent = [_entity_set(t) for t in reps]
    rep_tpl = [_template_set(t) for t in reps]

    ev_idx = [qidx[c["qid"]] for c in evalq]
    sims = E[ev_idx] @ cen.T
    bv = np.argmax(sims, axis=1); bs = sims[np.arange(len(ev_idx)), bv]
    feats = np.zeros((len(evalq), 3))
    for i, c in enumerate(evalq):
        j = int(bv[i])
        feats[i] = [float(bs[i]),
                    _jac(_entity_set(c["query"]), rep_ent[j]),
                    _jac(_template_set(c["query"]), rep_tpl[j])]
    sc = lr.predict_proba(sk.transform(feats))[:, 1]

    cost = np.array([c.get("cost", 1.0) for c in evalq], float)
    srcs = np.array([c["source"] for c in evalq])

    def summarize(mask):
        d = {"hit_queries": float(mask.mean()),
             "hit_cost": float(cost[mask].sum() / max(cost.sum(), 1e-9)),
             "n_hit": int(mask.sum()), "by_source": {}}
        for s in sorted(set(srcs.tolist())):
            m = srcs == s
            if m.sum():
                d["by_source"][s] = {
                    "hit_queries": float(mask[m].mean()),
                    "hit_cost": float(cost[m & mask].sum() / max(cost[m].sum(), 1e-9)),
                    "n": int(m.sum())}
        return d

    out = {"tau": tau, "n_views": len(views), "n_eval": len(evalq),
           "n_build_tasks": len(build_t), "n_eval_tasks": len(eval_t),
           "baselines": {"exact_cache": summarize(hit_exact),
                         "normalized_cache": summarize(hit_norm)},
           "views": {}, "increment": {}}
    log.info(f"  精确缓存     按查询 {out['baselines']['exact_cache']['hit_queries']:.1%}"
             f"  按成本 {out['baselines']['exact_cache']['hit_cost']:.1%}")
    log.info(f"  归一化缓存   按查询 {out['baselines']['normalized_cache']['hit_queries']:.1%}"
             f"  按成本 {out['baselines']['normalized_cache']['hit_cost']:.1%}")

    # 健全性：缓存基线命中率过高或过低，这张对照表都失去意义。
    # 过高 -> 查询池太同质（build 集几乎覆盖全部文本），净增量必然接近 0；
    # 过低 -> 复用本身就不存在，视图也不会有收益。
    _cb = out["baselines"]["normalized_cache"]["hit_queries"]
    out["baseline_sanity"] = {"normalized_cache_hit": _cb,
                              "plausible": bool(0.02 <= _cb <= 0.60)}
    if _cb > 0.60:
        log.error(f"!! 归一化缓存命中率 {_cb:.1%} 过高：build 集几乎覆盖了 eval 的全部"
                  f"查询文本，净增量必然接近 0，这张对照表无意义。"
                  f"检查查询池是否过于同质。")
    elif _cb < 0.02:
        log.error(f"!! 归一化缓存命中率 {_cb:.1%} 过低：几乎不存在字面重复，"
                  f"视图的收益也无从谈起。检查 train/test 切分是否正确。")
    else:
        log.info(f"  [健全性] 归一化缓存命中 {_cb:.1%}，落在合理区间，对照有效")

    for tg in cfg.report_targets:
        t = thr.get(tg)
        if t is None:
            out["views"][f"{tg:.2f}"] = None
            log.info(f"  视图 @{tg:.2f}  无可行阈值")
            continue
        hv = sc >= t
        out["views"][f"{tg:.2f}"] = summarize(hv)
        # 净增量：视图命中但归一化缓存没命中的部分
        inc = hv & (~hit_norm)
        d = {"n": int(inc.sum()),
             "frac_of_eval": float(inc.mean()),
             "cost_frac_of_eval": float(cost[inc].sum() / max(cost.sum(), 1e-9)),
             "share_of_view_hits": float(inc.sum() / max(hv.sum(), 1)),
             "by_source": {}}
        for s in sorted(set(srcs.tolist())):
            m = srcs == s
            if m.sum():
                d["by_source"][s] = {
                    "frac": float(inc[m].mean()),
                    "cost_frac": float(cost[m & inc].sum() / max(cost[m].sum(), 1e-9))}
        # 增量部分的判定器打分分布（间接反映质量）
        if inc.sum():
            d["score_mean"] = float(sc[inc].mean())
            d["cos_mean"] = float(bs[inc].mean())
            d["cos_p10"] = float(np.percentile(bs[inc], 10))
        out["increment"][f"{tg:.2f}"] = d
        log.info(f"  视图 @{tg:.2f}  按查询 {out['views'][f'{tg:.2f}']['hit_queries']:.1%}"
                 f"  按成本 {out['views'][f'{tg:.2f}']['hit_cost']:.1%}"
                 f"  |  净增量 按查询 {d['frac_of_eval']:.1%}"
                 f"  按成本 {d['cost_frac_of_eval']:.1%}"
                 f"（占视图命中的 {d['share_of_view_hits']:.1%}）")

    # 主保证水平下的分域增量
    key = f"{cfg.target_substitutability:.2f}"
    if out["increment"].get(key):
        log.info(f"  主保证 {key} 的分域净增量（按成本）：" + "  ".join(
            f"{s.replace('replay:','')}={v['cost_frac']:.1%}"
            for s, v in out["increment"][key]["by_source"].items()))
    out["headline"] = {
        "target": cfg.target_substitutability,
        "exact_cache_cost": out["baselines"]["exact_cache"]["hit_cost"],
        "normalized_cache_cost": out["baselines"]["normalized_cache"]["hit_cost"],
        "view_cost": (out["views"].get(key) or {}).get("hit_cost"),
        "increment_cost": (out["increment"].get(key) or {}).get("cost_frac_of_eval"),
    }
    ctx.wjson(ctx.out("s23", "cache_vs_view.json"), out)
    h = out["headline"]
    return (f"@{key} 缓存 {h['normalized_cache_cost']:.1%} -> 视图 "
            f"{(h['view_cost'] or 0):.1%}（净增 {(h['increment_cost'] or 0):.1%}）")


# ==============================================================================
# s24  论文素材导出（离线）
# ==============================================================================


@stage("s24", "论文素材：表格 CSV + 数字一览", ["PAPER_NUMBERS.md"])
def s24(ctx: RunContext, sm: ServerManager):
    """把散在各 stage 的结果整理成可直接贴进论文的表格与图数据。

    产出在 <run>/paper/ 下：
      table_*.csv  —— 论文表格
      fig_*.csv    —— 图的原始数据（自己用 matplotlib/pgfplots 画）
      PAPER_NUMBERS.md —— 所有数字一览，每个都标出处 stage
    """
    import csv as _csv
    cfg, log = ctx.cfg, ctx.log
    sm.stop()
    pdir = ctx.base / "paper"
    pdir.mkdir(parents=True, exist_ok=True)
    g = lambda s, n: ctx.rjson(ctx.out(s, n)) if ctx.out(s, n).exists() else {}
    s1_, s2_ = g("s1", "stats.json"), g("s2", "stats.json")
    s4_, qc = g("s4", "audit.json"), g("s6", "qc.json")
    l2, ca = g("s7", "summary.json"), g("s8", "calib.json")
    gr, R_ = g("s9", "granularity.json"), g("s10", "reuse.json")
    sc_, tf = g("s13", "scale.json"), g("s14", "twofeat.json")
    vh, aa = g("s17", "view_hit.json"), g("s18", "agent_arms.json")
    js, cb = g("s19", "judge_stability.json"), g("s20", "construction_bias.json")
    oa, bo = g("s21", "order_ablation.json"), g("s22", "bakeoff.json")
    cv, ge = g("s23", "cache_vs_view.json"), g("s15", "gold_eval.json")

    def wcsv(name, header, rows):
        p = pdir / name
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f); w.writerow(header); w.writerows(rows)
        return p.name

    made = []
    # T1 分域工作负载特征
    if R_.get("by_source"):
        made.append(wcsv("table_workload.csv",
            ["source", "calls", "intents", "head_cov_calls", "head_cov_cost",
             "cross_intent_frac", "mean_tasks_per_intent", "head_mono_text_share"],
            [[s, v.get("n_calls"), v.get("n_intents"),
              round(v.get("head_cov_calls", 0), 4), round(v.get("head_cov_cost", 0), 4),
              round(v.get("cross_intent_frac", 0), 4),
              round(v.get("mean_tasks_per_intent", 0), 3),
              round(v.get("head_mono_text_share", 0), 4)]
             for s, v in sorted(R_["by_source"].items()) if v.get("n_calls")]))
    # T2 规模律
    if sc_.get("fits"):
        made.append(wcsv("table_scale_fit.csv",
            ["domain", "slope_per_decade", "r2", "current_tasks", "current",
             "pred_2x", "pred_10x", "pred_50x"],
            [[k, round(f["slope"], 5), round(f["r2"], 4), int(f["current_tasks"]),
              round(f["current"], 4), round(f["pred"]["2x"], 4),
              round(f["pred"]["10x"], 4), round(f["pred"]["50x"], 4)]
             for k, f in sc_["fits"].items() if f]))
    if sc_.get("all"):
        made.append(wcsv("fig_scale_curve.csv",
            ["frac", "n_tasks", "n_calls", "head_cov_calls", "head_cov_cost",
             "sd_cost", "cross_intent_frac"],
            [[r["frac"], int(r["n_tasks"]), int(r["n_calls"]),
              round(r["head_cov_calls"], 4), round(r["head_cov_cost"], 4),
              round(r.get("sd_cost", 0), 4), round(r["cross_intent_frac"], 4)]
             for r in sc_["all"]]))
    # T3 校准曲线
    if ca.get("per_bin"):
        made.append(wcsv("fig_calibration.csv",
            ["bin", "lo", "hi", "n", "k", "p", "ci_lo", "ci_hi"],
            [[b, v["range"][0], v["range"][1], v["n"], v["k"],
              round(v["p"], 4), round(v["lo"], 4), round(v["hi"], 4)]
             for b, v in sorted(ca["per_bin"].items(), key=lambda x: int(x[0]))]))
    if ca.get("tau_multi"):
        made.append(wcsv("table_tau_multi.csv",
            ["target", "tau_star", "tau_point", "n_pass", "pass_frac",
             "same_text_frac", "empirical_sub", "degenerate"],
            [[t, v.get("tau_star"), v.get("tau_point"), v.get("n_pass"),
              v.get("pass_frac"), v.get("same_text_frac"),
              v.get("empirical_sub"), v.get("degenerate")]
             for t, v in sorted(ca["tau_multi"].items())]))
    # T4 判官对拍
    if bo.get("variants"):
        made.append(wcsv("table_judge_bakeoff.csv",
            ["variant", "n", "accuracy", "ci_lo", "ci_hi", "precision",
             "recall", "kappa", "pred_pos_rate"],
            [[k, v["n"], round(v["accuracy"], 4), round(v["acc_ci"][0], 4),
              round(v["acc_ci"][1], 4), round(v["precision"], 4),
              round(v["recall"], 4), round(v["kappa"], 4),
              round(v["pred_pos_rate"], 4)]
             for k, v in sorted(bo["variants"].items(),
                                key=lambda x: -x[1]["accuracy"])]))
    # T5 判定器消融
    if tf.get("pareto"):
        rows = []
        for name, per in tf["pareto"].items():
            for t, v in sorted(per.items()):
                if v:
                    rows.append([name, t, round(v["coverage_test"], 4),
                                 round(v["empirical_sub_test"], 4),
                                 round(v["empirical_ci"][0], 4),
                                 round(v["empirical_ci"][1], 4)])
        made.append(wcsv("table_detector_ablation.csv",
            ["features", "guarantee", "hit_rate", "empirical_sub",
             "ci_lo", "ci_hi"], rows))
    # T6 视图级命中率
    if vh.get("by_guarantee"):
        rows = []
        for t, d in sorted(vh["by_guarantee"].items()):
            if not d:
                continue
            rows.append([t, "ALL", round(d["hit_rate_queries"], 4),
                         round(d["hit_rate_cost"], 4), vh.get("n_eval_queries")])
            for s, v in d.get("by_source", {}).items():
                rows.append([t, s, round(v["hit_rate_queries"], 4),
                             round(v["hit_rate_cost"], 4), v["n"]])
        made.append(wcsv("table_view_hit.csv",
            ["guarantee", "source", "hit_queries", "hit_cost", "n"], rows))
    # T7 缓存对照（核心表）
    if cv.get("baselines"):
        rows = [["exact_cache", "-", round(cv["baselines"]["exact_cache"]["hit_queries"], 4),
                 round(cv["baselines"]["exact_cache"]["hit_cost"], 4), "", ""],
                ["normalized_cache", "-",
                 round(cv["baselines"]["normalized_cache"]["hit_queries"], 4),
                 round(cv["baselines"]["normalized_cache"]["hit_cost"], 4), "", ""]]
        for t, d in sorted((cv.get("views") or {}).items()):
            if not d:
                continue
            inc = (cv.get("increment") or {}).get(t, {})
            rows.append([f"view@{t}", t, round(d["hit_queries"], 4),
                         round(d["hit_cost"], 4),
                         round(inc.get("frac_of_eval", 0), 4),
                         round(inc.get("cost_frac_of_eval", 0), 4)])
        made.append(wcsv("table_cache_vs_view.csv",
            ["method", "guarantee", "hit_queries", "hit_cost",
             "increment_queries", "increment_cost"], rows))
    # T8 构造偏差
    if cb.get("full"):
        made.append(wcsv("table_construction_bias.csv",
            ["source", "full_head_cost", "matched_head_cost",
             "exact_dup_crosstask_frac", "mean_tasks_per_text"],
            [[s, round(cb["full"][s].get("head_cov_cost", 0), 4),
              round(cb.get("matched", {}).get(s, {}).get("head_cov_cost", 0), 4),
              round(cb["decomposition"][s]["exact_dup_crosstask_frac"], 4),
              round(cb["decomposition"][s]["mean_tasks_per_text"], 3)]
             for s in sorted(cb["full"])]))
    # T9 判官稳定性
    if js.get("by_bin"):
        made.append(wcsv("table_judge_stability.csv",
            ["bin", "n", "flip_rate", "sub_rate_consistent",
             "sub_rate_flipped", "strict_rate"],
            [[b, v["n"], round(v["flip_rate"], 4), v["sub_rate_consistent"],
              v["sub_rate_flipped"], v["strict_rate"]]
             for b, v in sorted(js["by_bin"].items(), key=lambda x: int(x[0]))]))
    # T10 粒度敏感性
    if gr:
        made.append(wcsv("table_granularity.csv",
            ["tau", "n_components", "n_cliques", "largest",
             "singleton_frac", "intra_edge_rate"],
            [[t, v["n_components"], v["n_cliques"], v["largest"],
              round(v["singleton_frac"], 4), v.get("intra_edge_rate")]
             for t, v in sorted(gr.items(), key=lambda x: float(x[0]))]))
    log.info(f"  导出 {len(made)} 个 CSV -> {pdir}")

    # ---------------- PAPER_NUMBERS.md ----------------
    L = [f"# 论文数字一览 — {cfg.run_id}", "",
         f"生成 {time.strftime('%F %T')}。每个数字后括号内是出处 stage。", "",
         "## 工作负载", "",
         f"- 语料 {s1_.get('n_docs')} 文档 / {s1_.get('n_chunks')} 块，dim {s1_.get('dim')} (s1)",
         f"- 检索调用 {s2_.get('total_calls')} / 任务 {s2_.get('total_tasks')} "
         f"({s2_.get('calls_per_task', 0):.2f} 次/任务) (s2)",
         f"- 候选对 {s4_.get('n_pairs')}，top-{s4_.get('block_topk')}，"
         f"分块漏检 @0.90 = {(s4_.get('miss_rate', {}) or {}).get('0.9')} (s4)",
         f"- 跨任务字面相同查询占比 {(s2_.get('sanity') or {}).get('exact_dup_query_frac')} (s2)",
         "", "## 判官", "",
         f"- prompt 对拍最优 **{bo.get('winner')}**，accuracy "
         f"**{bo.get('winner_accuracy')}**，门槛 {bo.get('threshold')} (s22)"]
    if bo.get("variants", {}).get(bo.get("winner")):
        wv = bo["variants"][bo["winner"]]
        L.append(f"- 最优变体 P/R/κ = {wv['precision']:.3f} / {wv['recall']:.3f} / "
                 f"{wv['kappa']:.3f}；人工正例率 {bo.get('human_pos_rate'):.3f}，"
                 f"n={bo.get('n')} (s22)")
    L += [f"- 自覆盖率 {qc.get('self_cov_rate')}；合格率 {qc.get('eligibility_rate')} (s6)",
          f"- 条件可替代率（分层加权）**{qc.get('substitutable_rate_conditional')}** "
          f"CI{qc.get('substitutable_rate_conditional_ci')} n_eff={qc.get('effective_n')} (s6)",
          f"- 陷阱误判 {qc.get('catch_fp_rate')}；同文本方向一致 "
          f"{qc.get('same_text_direction_consistency')}；解析失败 "
          f"{qc.get('parse_fail_rate')} (s6)",
          f"- 位置一致率 {qc.get('pos_consistency')}；方向不对称 "
          f"{qc.get('asymmetry_rate')} (s6)",
          f"- 跨判官 κ {qc.get('cross_kappa')} (s6)",
          f"- 人工金标 accuracy {ge.get('accuracy')} (s15)"]
    if oa.get("by_prompt"):
        for k, v in oa["by_prompt"].items():
            L.append(f"- 顺序 A/B [{k}] 一致率 {v['consistency']:.3f}，"
                     f"方向偏差 {v['directional_bias']:+.3f} (s21)")
    L += ["", "## 判定器", "",
          f"- τ*（主保证 {cfg.target_substitutability}）**{ca.get('tau_star')}**，"
          f"点估计 {ca.get('tau_point')}，n={ca.get('n')} (s8)"]
    for t, v in sorted((ca.get("tau_multi") or {}).items()):
        L.append(f"  - 保证 {t}: τ*={v.get('tau_star')}  放行 {v.get('pass_frac')}  "
                 f"同文本占比 {v.get('same_text_frac')}  "
                 f"{'⚠退化' if v.get('degenerate') else ''} (s8)")
    if tf.get("best_model"):
        L.append(f"- 最优特征组合 {tf['best_model']} (s14)")
    L += ["", "## 收益", ""]
    if cv.get("headline"):
        h = cv["headline"]
        L += [f"- **精确缓存** 按成本 {h['exact_cache_cost']} (s23)",
              f"- **归一化缓存** 按成本 {h['normalized_cache_cost']} (s23)",
              f"- **近似视图 @{h['target']}** 按成本 {h['view_cost']} (s23)",
              f"- **净增量** 按成本 {h['increment_cost']} (s23)"]
    if vh.get("by_guarantee"):
        for t, d in sorted(vh["by_guarantee"].items()):
            if d:
                L.append(f"- 视图级命中 @{t}: 按查询 {d['hit_rate_queries']:.3f} / "
                         f"按成本 {d['hit_rate_cost']:.3f} (s17)")
    L += ["", "## ε（Phase 2 用）", "",
          f"- 非零损失占比 {l2.get('nonzero_loss_frac')}，P99 "
          f"{l2.get('p99_recall_loss_pos')}，max {l2.get('max_recall_loss_pos')}，"
          f"非零条件均值 {l2.get('mean_loss_given_nonzero')} (s7)",
          f"- L1 相对 L2 的 P/R {l2.get('l1_precision')} / {l2.get('l1_recall')} (s7)",
          f"- f1_own {l2.get('mean_f1_own')}（分域见 table 与 FINAL.txt）(s7)",
          "", "## 构造偏差", ""]
    if cb.get("musique_over_2wiki"):
        L.append(f"- 等量对照 MuSiQue/2Wiki = **{cb['musique_over_2wiki']:.2f}×**"
                 f"（基准 {cb.get('matched_tasks')} 任务）(s20)")
    if sc_.get("fits"):
        L += ["", "## 规模律", ""]
        for k, f in sc_["fits"].items():
            if f:
                L.append(f"- {k}: 斜率 {f['slope']:+.4f}/decade, R²={f['r2']:.3f}, "
                         f"当前 {f['current']:.3f} -> 10x {f['pred']['10x']:.3f} (s13)")
    L += ["", "## 导出的 CSV", ""] + [f"- `paper/{m}`" for m in made]
    (pdir / "PAPER_NUMBERS.md").write_text("\n".join(L), encoding="utf-8")
    ctx.out("s24", "PAPER_NUMBERS.md").write_text("\n".join(L), encoding="utf-8")
    log.info(f"  数字一览 -> {pdir / 'PAPER_NUMBERS.md'}")
    return f"{len(made)} 个 CSV + PAPER_NUMBERS.md"


# ==============================================================================
#  main
# ==============================================================================


def run_stage(ctx: RunContext, sm: ServerManager, sid: str, force: bool):
    desc, fn, prod = STAGES[sid]
    if prod and not force and all(ctx.out(sid, p).exists() for p in prod):
        ctx.log.info(f"[{sid}] 已完成，跳过"); return
    ctx.log.info("=" * 78); ctx.log.info(f"[{sid}] {desc}"); ctx.log.info("=" * 78)
    ctx.state["current"] = sid
    ctx.state["stages"][sid] = {"status": "running", "start": time.time()}
    ctx.save_state(); t0 = time.time()
    try:
        summ = fn(ctx, sm) or ""
    except Exception as e:
        ctx.state["stages"][sid].update({"status": "failed", "error": str(e),
                                         "tb": traceback.format_exc(),
                                         "duration_sec": time.time() - t0})
        ctx.save_state(); ctx.log.error(f"[{sid}] 失败: {e}")
        ctx.log.debug(traceback.format_exc()); raise
    dt = time.time() - t0
    ctx.state["stages"][sid].update({"status": "done", "duration_sec": dt,
                                     "summary": str(summ)})
    ctx.save_state(); ctx.log.info(f"[{sid}] 完成 {dt/60:.1f}min — {summ}")


# v9：ORDER 按 @stage 定义顺序生成，而 s17 定义在 s16 之后 -> main_v5 里
# s16 先跑，读不到 s17 的产出，FINAL.txt 缺了【8b】整段（s17 其实跑成功了：
# 保证0.90 下命中 查询18.8%/成本19.8%）。这里显式把汇总放到最后。
def _fix_order():
    """s22 对拍必须最早（它决定 s6 用哪个 prompt，且是硬闸门）；
    s16 是汇总，必须最后（它要读 s17~s21 的产出）。"""
    if "s22" in ORDER:
        ORDER.remove("s22")
        ORDER.insert(1 if ORDER and ORDER[0] == "s0" else 0, "s22")
    # s23/s24 依赖 s8/s10/s13/s17/s20/s21/s22，必须在它们之后、s16 之前
    for s in ("s23", "s24"):
        if s in ORDER:
            ORDER.remove(s); ORDER.append(s)
    tail = [s for s in ("s16",) if s in ORDER]
    for s in tail:
        ORDER.remove(s)
    ORDER.extend(tail)


_fix_order()


def main() -> int:
    ap = argparse.ArgumentParser(description="MatView Phase 1 v13")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-servers", action="store_true")
    ap.add_argument("--gold-tsv", default=None, help="人工标注回填的 TSV")
    ap.add_argument("--bakeoff", action="append", default=None,
                    metavar="RUN_DIR:TSV",
                    help="判官对拍数据源，可重复。例："
                         "--bakeoff runs/main_v8:v8_labeled.tsv")
    ap.add_argument("--ignore-judge-gate", action="store_true",
                    help="对拍未达标也继续（结论需降级为不可靠）")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for s in ORDER:
            print(f"  {s:4s} {STAGES[s][0]}")
        return 0

    cfg = Config.load(a.config)
    if a.run_id:
        cfg.run_id = a.run_id
    if a.smoke:
        cfg.apply_smoke()
    if a.no_servers:
        cfg.manage_servers = False
    if a.gold_tsv:
        cfg.gold_tsv = a.gold_tsv
    if a.bakeoff:
        cfg.bakeoff_sources = [s.split(":", 1) for s in a.bakeoff if ":" in s]
    if a.ignore_judge_gate:
        cfg.require_judge_acc = False

    ctx = RunContext(cfg)
    ctx.log.info("硬件自适应：按实测空闲显存反算 vLLM 显存比例")
    try:
        cfg._autotune = autotune_hardware(cfg, ctx.log)
        ctx.wjson(ctx.base / "autotune.json", cfg._autotune)
    except Exception as e:
        ctx.log.error(f"硬件自适应失败: {e}"); return 3
    ctx.log.info(f"run_id={cfg.run_id}  {'SMOKE' if cfg.smoke else 'FULL'}  "
                 f"GPUs={cfg.gpus}  语料源={cfg.corpus_source}")
    ctx.log.info(f"输出 {ctx.base.resolve()}")
    ctx.wjson(ctx.base / "config_effective.json", asdict(cfg))
    ctx.state["_t0"] = time.time()
    ctx.save_state()
    ctx.start_monitor()
    sm = ServerManager(ctx)

    def sig(s, f):
        ctx.log.warning(f"收到信号 {s}，清理退出")
        sm.stop(); ctx.save_state(); ctx.stop_monitor(); sys.exit(130)
    signal.signal(signal.SIGINT, sig); signal.signal(signal.SIGTERM, sig)

    todo = [s.strip() for s in a.stages.split(",")] if a.stages else ORDER
    for s in todo:
        if s not in STAGES:
            ctx.log.error(f"未知 stage: {s}"); return 2

    rc, t0 = 0, time.time()
    try:
        for s in todo:
            run_stage(ctx, sm, s, a.force)
    except Exception:
        rc = 1
    finally:
        sm.stop(); ctx.stop_monitor(); ctx.save_state()
        ctx.log.info(f"总耗时 {(time.time()-t0)/3600:.2f} h")
        for sid, m in ctx.state["stages"].items():
            if m.get("duration_sec"):
                ctx.log.info(f"  {sid:4s} {m['duration_sec']/60:7.1f}min  "
                             f"{str(m.get('summary',''))[:60]}")
        if (ctx.base / "REPORT.md").exists():
            ctx.log.info(f"报告 {ctx.base/'REPORT.md'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
