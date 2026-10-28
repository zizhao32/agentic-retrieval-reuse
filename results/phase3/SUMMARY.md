# Phase 3 补实验（E4 / E5）总报告

生成于 2026-10-24 10:44:56，主 run：`runs/main_v10`。本文件由 `exp45.py report` 从各步 JSON 汇总，数字均可追溯到 JSON。

## 运行状态

| 步骤 | 状态 | 耗时 (h) | 摘要 |
|---|---|---|---|
| download | done | 0.172 | Llama-3.3-70B-Instruct-AWQ-INT4 @ a3c28ea93c041acfaadb85401ad6493749a5aa04 就绪；2Wiki 12576 条 |
| audit | done | 0.258 | identified: main 用 base；与 think1 同 prompt=True |
| e5a | done | 9.811 | xfam_llama33: 0.80:τ*=0.9875/同文本76.9%, 0.85:τ*=—/同文本—, 0.90:τ*=—/同文本—, 0.95:τ*=—/同文本— |
| e4-collect | done | 1.983 | 10901 调用 / 2000 任务（5.47/任务），失败 7 |
| e4-merge | done | 0.001 | 54451 调用 -> runs/e4_agent2wiki |
| e4-downstream | done | 0.038 | 2Wiki agent 头部成本覆盖 17.4%，意图 6845 |
| e4-validate | done | 0.196 | 0.80: 精度 0.958 下界 0.935 / 0.85: 精度 0.987 下界 0.971 / 0.90: 精度 1.000 下界 1.000 / 0.95: 精度 1.000 下界 1.000 |
| e5b | done | 0.0 | main 的 prompt 就是 base，与 think1 相同 —— 无 prompt 混杂，E5b 不需要（--force-e5b 强制跑） |
| e5c-export | done | 0.001 | 200 对 / 400 题 |

## 0 判官 prompt 审计

- 结论：**identified → `base`**（sha256 `c2a47c11a8ba`）。base 逐字复现 facts 98.1%，次优 order_inv 40.6%
- 静态推断：{'variant': 'base', 'reason': 's22/selected.json 存在 → s6 用对拍胜者', 'caveat': 'config_effective.json 每次调用 phase1 都会重写，反映的是最后一次调用时的配置，不一定是 s6 那次'}
- think1 的 prompt：`base`；k3：`order_inv`（推断）
- **对论文的影响**：§6 的「两臂只差推理预算」成立（同一 prompt）。E5b 不需要。

| prompt 变体 | sha12 | n | verdict 一致 | facts 逐字复现 |
|---|---|---|---|---|
| base | `c2a47c11a8ba` | 160 | 100.0% | 98.1% |
| order_inv | `781d9f4dd5c9` | 160 | 98.8% | 40.6% |
| evidence_quote | `a1cd7ee0119e` | 160 | 95.0% | 30.6% |
| strict_checklist | `ff5b44d4e7f7` | 160 | 92.5% | 5.0% |
| conservative | `896d7e4e8cd5` | 160 | 91.9% | 4.4% |
| answer_first | `f1bc68eade56` | 160 | 98.1% | 0.0% |
| binary_direct | `787795df6068` | 160 | 92.5% | 0.0% |

## E5 判官的独立验证

- `xfam_llama33`：Llama-3.3-70B-Instruct-AWQ，prompt `base`，thinking=False，有效 37719/37733，解析失败 0.02%，截断 0.00%，9.8 h
  - 闸门 未过：金标 acc 0.733（主判官同批 0.868），与主判官一致 77.6%

**两两一致性 —— 单向 cov_ij（论文 \CrossCfg* 宏的口径）**

| 臂对 | n | 一致率 | κ | AC1 | 覆盖率 A / B |
|---|---|---|---|---|---|
| main~think1 | 37600 | 90.7% | 0.642 | 0.875 | 0.847 / 0.847 |
| main~k3 | 37733 | 84.1% | 0.500 | 0.768 | 0.847 / 0.762 |
| main~xfam_llama33 | 37719 | 83.9% | 0.497 | 0.764 | 0.847 / 0.758 |
| think1~k3 | 37600 | 85.5% | 0.545 | 0.788 | 0.847 / 0.761 |
| think1~xfam_llama33 | 37586 | 87.1% | 0.596 | 0.810 | 0.847 / 0.758 |
| k3~xfam_llama33 | 37719 | 84.3% | 0.570 | 0.753 | 0.762 / 0.758 |

**两两一致性 —— 对级 substitutable（全体合格对，未加权；定义同 crosstab_l1.py）**

| 臂对 | n | 一致率 | κ | PABAK | AC1 | 正例率 A / B |
|---|---|---|---|---|---|---|
| main~think1 | 37600 | 84.0% | 0.536 | 0.680 | 0.756 | 76.6% / 79.2% |
| main~k3 | 37733 | 80.3% | 0.518 | 0.607 | 0.670 | 76.6% / 67.3% |
| main~xfam_llama33 | 37719 | 78.0% | 0.450 | 0.561 | 0.636 | 76.6% / 68.9% |
| think1~k3 | 37600 | 80.3% | 0.505 | 0.605 | 0.675 | 79.2% / 67.3% |
| think1~xfam_llama33 | 37586 | 83.9% | 0.587 | 0.679 | 0.739 | 79.2% / 68.9% |
| k3~xfam_llama33 | 37719 | 80.8% | 0.558 | 0.615 | 0.660 | 67.3% / 68.9% |

自检 main~think1（cov_ij）：90.7% / κ 0.642 / AC1 0.875 / 覆盖率 0.847（论文宏 90.7% / 0.642 / 0.875 / 0.847）

**τ*（各臂 s8 多档）**

| 臂 | 档 | τ* | 放行 | 同文本占比 | 退化 |
|---|---|---|---|---|---|
| main | 0.80 | 0.8810 | 69.5% | 20.7% | False |
| main | 0.85 | 0.9398 | 38.9% | 37.0% | False |
| main | 0.90 | 0.9937 | 17.3% | 83.2% | True |
| main | 0.95 | 0.9937 | 17.3% | 83.2% | True |
| think1 | 0.80 | 0.8797 | 69.8% | 20.6% | False |
| think1 | 0.85 | 0.8835 | 69.0% | 20.8% | False |
| think1 | 0.90 | 1.0000 | 12.0% | 99.9% | True |
| think1 | 0.95 | — | — | — | None |
| xfam_llama33 | 0.80 | 0.9875 | 18.7% | 76.9% | False |
| xfam_llama33 | 0.85 | — | — | — | None |
| xfam_llama33 | 0.90 | — | — | — | None |
| xfam_llama33 | 0.95 | — | — | — | None |

**金标准确率**（90 对，各臂交集）

| 臂 | acc | 95% CI | precision | recall |
|---|---|---|---|---|
| main | 0.878 | [0.794, 0.930] | 0.879 | 0.951 |
| think1 | 0.878 | [0.794, 0.930] | 0.879 | 0.951 |
| k3 | 0.689 | [0.587, 0.775] | 0.811 | 0.705 |
| xfam_llama33 | 0.733 | [0.634, 0.814] | 0.849 | 0.738 |

**分档：各臂对 main 的一致率 / κ（未加权；恒等带单列）**

| 档 | think1 | k3 | xfam_llama33 |
|---|---|---|---|
| bin0 [0.500,0.600) | 99.5% / 0.73 (n=1304) | 99.7% / 0.78 (n=1304) | 99.8% / 0.82 (n=1303) |
| bin1 [0.600,0.700) | 96.8% / 0.81 (n=1480) | 97.3% / 0.83 (n=1483) | 96.4% / 0.76 (n=1483) |
| bin2 [0.700,0.750) | 88.9% / 0.76 (n=1839) | 89.7% / 0.77 (n=1848) | 86.7% / 0.70 (n=1848) |
| bin3 [0.750,0.800) | 82.4% / 0.62 (n=1724) | 81.7% / 0.62 (n=1731) | 78.0% / 0.55 (n=1731) |
| bin4 [0.800,0.850) | 76.5% / 0.30 (n=2597) | 74.8% / 0.36 (n=2605) | 71.7% / 0.30 (n=2602) |
| bin5 [0.850,0.900) | 73.8% / 0.20 (n=4198) | 74.6% / 0.35 (n=4203) | 67.8% / 0.20 (n=4198) |
| bin6 [0.900,0.950) | 82.1% / 0.22 (n=12115) | 85.4% / 0.54 (n=12148) | 75.5% / 0.21 (n=12144) |
| bin7 [0.950,0.980) | 84.7% / 0.09 (n=4724) | 76.5% / 0.07 (n=4762) | 77.1% / 0.14 (n=4761) |
| bin8 [0.980,0.999) | 82.7% / 0.05 (n=2185) | 52.6% / -0.11 (n=2189) | 77.6% / -0.05 (n=2189) |
| bin9 [0.999,1.000) | 95.1% / -0.02 (n=465) | 80.0% / -0.01 (n=469) | 80.4% / -0.04 (n=469) |
| forced identity (same_text & J,RBO≥0.999) | 90.9% / 0.01 (n=4969) | 77.4% / 0.01 (n=4991) | 82.5% / -0.00 (n=4991) |
| all | 84.0% / 0.54 (n=37600) | 80.3% / 0.52 (n=37733) | 78.0% / 0.45 (n=37719) |

**恒等带（逻辑上必为 1）正例率**：main 99.8%（n=4991），think1 91.0%（n=4969），k3 77.4%（n=4991），xfam_llama33 82.7%（n=4991）

事先写定的机械判读（不是结论）：xfam τ*@0.80 非退化 = True；xfam τ*@0.85 非退化 = False；xfam τ*@0.90 非退化 = False；xfam 与 main 的 cov_ij κ 低于 main~think1 = True

## E5 双合格子集（合格性也由 Llama 判，两臂样本相同）

Llama 认为能被自己结果回答的查询 89.2%；双合格对 29229/37733。
一致性：cov_ij 94.4% κ=0.761；对级 91.0% κ=0.697

| 臂 | n | 可替代率 | 恒等带正例率 | 0.80 τ* / 同文本 | 0.85 τ* / 同文本 | 0.90 τ* / 同文本 |
|---|---|---|---|---|---|---|
| main | 29229 | 0.805 | 99.8% | 0.8772 / 21.7% | 0.8885 / 22.5% | 0.8972 / 23.1% |
| xfam_llama33 | 29227 | 0.835 | 100.0% | 0.8308 / 19.6% | 0.8321 / 19.7% | 0.8810 / 21.9% |

## E5 敏感性：PARTIAL 也算覆盖（宽松口径）

| 臂 | 档 | 严格 τ* | 严格同文本 | 宽松 τ* | 宽松同文本 |
|---|---|---|---|---|---|
| main | 0.80 | 0.8810 | 20.7% | 0.8070 | 17.5% |
| main | 0.85 | 0.9398 | 37.0% | 0.8271 | 18.1% |
| main | 0.90 | 0.9937 | 83.2% | 0.8709 | 20.2% |
| xfam_llama33 | 0.80 | 0.9875 | 76.9% | 0.8622 | 19.7% |
| xfam_llama33 | 0.85 | — | — | 0.8810 | 20.7% |
| xfam_llama33 | 0.90 | — | — | 0.9461 | 41.0% |

## E5c 盲测留出人工集

已导出标注页 `e5/human/annotate.html`，等待标注。

## E4 非 MuSiQue 的 agent 工作负载（Tongyi on 2WikiMultihopQA）

- 采集：2000 任务，10901 次检索（5.47/任务），失败 7，多查询比例 84.0%，2.0 h

| 指标 | MuSiQue agent (main_v10) | 2Wiki agent (E4) | 2Wiki replay (E4) |
|---|---|---|---|
| n_calls | 14206 | 10901 | 29741 |
| n_intents | 7592 | 6845 | 24764 |
| head_cov_cost | 28.6% | 17.4% | 14.3% |
| head_cov_calls | 28.9% | 17.6% | 14.0% |
| cross_intent_frac | 10.7% | 2.8% | 12.4% |
| mean_tasks_per_intent | 1.230 | 1.034 | 1.190 |
| head_mono_text_share | 3.9% | 29.1% | 82.1% |
| gini_cost | 41.5% | 31.5% | 23.1% |

Δ 头部成本覆盖（2Wiki − MuSiQue agent）= -11.2%；使用的 τ = 0.94（main_v10 标定，固定）

**检测器迁移（main_v10 的 τ* 用在 2Wiki-agent 对上）**

| 档 | τ* | 放行占比 | 同文本 | 精度估计 | 95% 下界 | 下界达标 |
|---|---|---|---|---|---|---|
| 0.80 | 0.8810 | 15.0% | 35.8% | 0.958 | 0.935 | True |
| 0.85 | 0.9398 | 10.8% | 50.0% | 0.987 | 0.971 | True |
| 0.90 | 0.9937 | 5.9% | 90.5% | 1.000 | 1.000 | True |
| 0.95 | 0.9937 | 5.9% | 90.5% | 1.000 | 1.000 | True |

事先写定的机械判读：|Δ 头部成本覆盖| < 5pp（对任务源不敏感） = False；replay:2wiki 两 run 差 < 1pp（合并没破坏 replay） = True

## 溯源

```
{
  "models": {
    "Llama-3.3-70B-Instruct-AWQ-INT4": {
      "repo": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
      "revision": "a3c28ea93c041acfaadb85401ad6493749a5aa04",
      "path": "/mnt/data/user_workspace/zhangzizhao/matview/models/Llama-3.3-70B-Instruct-AWQ-INT4",
      "bytes": 39767996256,
      "config_sha256": "1da1d369c9996294fe4312600eaa1f632aa33cdc8068c079a5b6930aba29d6f4",
      "license": "Llama 3.3 Community License（发表需注明 Built with Llama）"
    },
    "Qwen3-32B-AWQ": {
      "path": "/mnt/data/user_workspace/zhangzizhao/matview/models/Qwen3-32B-AWQ",
      "present": true,
      "config_sha256": "0e0c09d430b12e84773ef83ca83dba7c9174caf5a94bb940cd895b82df6c6528"
    },
    "Tongyi-DeepResearch-30B-A3B": {
      "path": "/mnt/data/user_workspace/zhangzizhao/matview/models/Tongyi-DeepResearch-30B-A3B",
      "present": true,
      "config_sha256": "aaf562dec00d45914019f2d9f184a759b112c576d32ebdeb858198a06b8c2f74"
    }
  },
  "datasets": {
    "2wiki": {
      "repo": "voidful/2WikiMultihopQA",
      "split": "validation",
      "n": 12576,
      "fingerprint": "ada731f87faf9caa",
      "cache_dir": "/mnt/data/user_workspace/zhangzizhao/matview/data/hf_cache"
    }
  },
  "vllm": null,
  "torch": "2.6.0+cu124",
  "gpus": [
    "NVIDIA H100 80GB HBM3, 81559 MiB, 550.54.15",
    "NVIDIA H100 80GB HBM3, 81559 MiB, 550.54.15"
  ],
  "scripts": {
    "exp45.py": "eb2ae80cd27804d48731b89cdbf3680dc914ded6d4f3c5cb60c72a56202f342c",
    "phase1_v13.py": "77ad83a4807196a724a613e6766b31398737c943ec898f16e3900b23329aa404"
  }
}
```
