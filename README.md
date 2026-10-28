# Cross-Task Reuse in Agentic Retrieval Workloads 


> Language note: code comments, log messages and the judge/agent prompts are in
> Chinese. The prompts are reproduced with English translations in
> [`docs/PROMPTS.md`](docs/PROMPTS.md).

## Repository layout

```
code/            every script that produced a number in the paper (flat: the scripts
                 import each other by file name and expect runs/ in the working dir)
environment/     setup script, requirements, model list with revisions
docs/PROMPTS.md  exact judge and agent prompts (+ SHA-256 of the judge prompt)
results/         small outputs: per-run configs, calibration curves (s8/calib.json),
                 reuse statistics (s10), scaling fits (s13), E4/E5 summaries
human_labels/    development labels (used for prompt selection) and held-out labels
tools/           export_artifacts.py (builds results/ and the Zenodo bundle)
```

## Runs

Every number in the paper traces to one of these runs. Each run directory has the
layout `runs/<run>/stages/<stage>/...`; `results/runs/<run>/config_effective.json` is the
exact configuration that run used.

| Run | What it is |
|---|---|
| `main_v10` | Primary run. 21,822 tasks; replay sub-questions from MuSiQue, HotpotQA, 2WikiMultihopQA and Tongyi-DeepResearch traces on MuSiQue tasks; 57,756 retrieval calls; judge Qwen3-32B-AWQ, thinking off, base prompt |
| `main_v12_think` | Same eligible pairs re-judged with reasoning enabled (2,400-token budget) |
| `main_v11_k3` | Same pairs, order-invariant prompt, majority of 3 shuffled passage orders |
| `agent_b` | Second agent model (Qwen3-32B-AWQ) on the same MuSiQue tasks |
| `distractor` | Corpus expanded with distractor passages |
| `e4_agent2wiki` | Tongyi agent on 2,000 2WikiMultihopQA tasks + `main_v10` replays (E4) |
| `e5a_xfam_llama33_70b_awq` | Same pairs re-judged by Llama-3.3-70B-Instruct (cross-family judge, E5) |
| `e5d_*_botheligible` | Primary and cross-family judges restricted to pairs both judges deem eligible |
| `e5l_*_lenient` | Primary and cross-family labels with `PARTIAL` counted as covered |

Runs derived from `main_v10` (`main_v12_think`, `main_v11_k3`, `e5*`) reuse its corpus,
traces, candidate pairs and sample (stages s1–s5) and differ only in the judge labels
(s6) and what is computed from them.

## Where each part of the paper comes from

| Paper content | Script (in `code/`) | Outputs |
|---|---|---|
| Corpus, retrieval-call collection (replay + agent) | `phase1_v13.py` stages s1–s2 | `s1/`, `s2/calls.jsonl` |
| Query embedding, candidate blocking, stratified sampling | `phase1_v13.py` s3–s5 | `s4/pairs.jsonl`, `s5/` |
| Self-coverage screening and pairwise judge | `phase1_v13.py` s6 | `s6/l1.jsonl`, `s6/qc.json` |
| Calibration (isotonic + bootstrap), multi-level τ* | `phase1_v13.py` s8 | `s8/calib.json` |
| Intent clustering, reuse, head coverage, by-source breakdown | `phase1_v13.py` s9–s10 | `s10/reuse.json` |
| Scaling with workload size | `phase1_v13.py` s13 | `s13/scale.json` |
| View hit rate, cache vs. view | `phase1_v13.py` s17, s23 | `s17/`, `s23/` |
| End-to-end view serving; uncalibrated semantic-cache baseline | `phase2_experiments.py` (A, C) | `results/phase2/` |
| Cache ladder, density control, second-pass facts, k sensitivity | `review_fixes.py` (R1–R4), `fill_gaps.py` | `results/phase2/R/` |
| Second agent model | `agent_b.py` | `agent_b` run |
| Distractor corpus | `distractor.py` | `distractor` run |
| Judge configurations (reasoning budget, shuffled orders) | `fix_judge.py` (J1 gate, J2 re-judge) | `main_v12_think`, `main_v11_k3` |
| Cross-configuration agreement, forced-identity pairs | `crosstab_l1.py`, `identity_probe.py`, `compare_l1.py`, `judge_quality.py`, `dump_cases.py` | stdout / `results/phase3/e5/compare.json` |
| Prompt audit, cross-family judge, eligibility and verdict-mapping sensitivity, held-out human evaluation | `exp45.py` (`audit`, `e5a`, `e5-gate-diag`, `e5-langcheck`, `e5-selfcov`, `e5-lenient`, `e5c-*`, `e5-report`) | `results/phase3/` |
| 2WikiMultihopQA agent workload | `exp45.py` (`e4-collect`, `e4-merge`, `e4-downstream`, `e4-validate`, `e4-report`) | `e4_agent2wiki`, `results/phase3/e4/` |
| Figures (Fig. 1 of the paper: `make_fig_pipeline.py`, input `results/phase3/e5/compare.json`) | `make_fig_pipeline.py`; earlier drafts: `make_figs.py`, `make_fig4.py`, `make_fig_twoarm.py` | — |
| Check that every number in the paper exists in a run artifact | `audit_macros.py` | — |

`results/phase3/summary.json` is a flat key → value file of every E4/E5 number, generated
by `exp45.py report`.

## Reproducing

**Hardware.** The paper's runs used 2× NVIDIA H100 80GB. One GPU hosts the retriever;
LLM judges and agents are served by vLLM, one replica per GPU. Approximate GPU time:
primary run 10–15 h; each full re-judge of ~37,700 pairs 6–10 h; the 2Wiki agent
workload 2 h.

**Environment.** `environment/setup_matview_v2.sh` creates two Python environments
(`envs/main` for analysis, `envs/serve` for vLLM), downloads models and datasets, and
writes a `config.json`. Requirements are listed in `environment/requirements-*.txt` and
models in `environment/MODELS.md`.

**Working directory.** Scripts expect to be run from a directory that contains both the
scripts and `runs/`:

```bash
mkdir work && cp code/* work/ && cd work
R=/path/to/matview          # created by setup_matview_v2.sh
PY=$R/envs/main/bin/python
```

**Primary run and judge configurations.**

```bash
$PY phase1_v13.py --config $R/config.json --run-id main_v10            # all stages
$PY fix_judge.py  --config $R/config.json --exp J1                       # gate
$PY fix_judge.py  --config $R/config.json --exp J2 --arm think1 --main-run runs/main_v10 --out-run main_v12_think
$PY fix_judge.py  --config $R/config.json --exp J2 --arm k3     --main-run runs/main_v10 --out-run main_v11_k3
$PY phase1_v13.py --config $R/config.json --run-id main_v12_think --stages s8
```

**Supplementary experiments.** `agent_b.py`, `distractor.py`, `phase2_experiments.py`,
`review_fixes.py` and `fill_gaps.py` each document their usage in the module docstring.

**E4 / E5.**

```bash
bash run_exp45.sh download        # cross-family judge weights, dataset check
bash run_exp45.sh all             # audit -> E5a -> E4 -> (E5b if needed) -> E5c export -> reports
$PY exp45.py --config $R/config.json e5-selfcov   # both-eligible subset
$PY exp45.py --config $R/config.json e5-lenient   # PARTIAL counted as covered
$PY exp45.py --config $R/config.json e5c-eval --labels ../human_labels/heldout/labels.tsv
$PY exp45.py --config $R/config.json report       # results/phase3/SUMMARY.md + summary.json
```

**Recomputing reported statistics without GPUs.** Download the Zenodo bundle, unpack it
into `runs/` (the paths inside the archive mirror `runs/<run>/stages/...`), copy the
JSON files from `results/runs/` alongside, and run e.g.

```bash
$PY crosstab_l1.py main_v10 main_v12_think      # cross-configuration agreement
$PY identity_probe.py main_v10 main_v11_k3 main_v12_think
$PY exp45.py --config runs/main_v10/config_effective.json e5-report
```

## Determinism and known caveats

* Judges run at temperature 0, but vLLM batching is not bit-deterministic: re-running a
  judge reproduces stored verdicts for the large majority of pairs, not all of them
  (see the prompt audit: 100% verdict and 98.1% exact `required_facts` reproduction on
  160 pairs).
* Agents run at temperature 0.6; re-collecting agent traces yields a different but
  statistically similar workload. The released traces are the ones analysed.
* `config_effective.json` is rewritten each time `phase1_v13.py` is invoked on a run, so
  it reflects the last invocation; the judge prompt actually used by `main_v10` was
  verified separately (`docs/PROMPTS.md`).
* Each held-out human item carries a single label; see `human_labels/README.md`.
* BrowseComp-Plus is distributed encrypted to avoid benchmark contamination; the code
  that reads it is included, but no plaintext derived from it is redistributed.

## License

Code: MIT (see `LICENSE`). Retrieval traces, judge verdicts and human labels: CC BY 4.0.
Passages and questions derived from MuSiQue, HotpotQA and 2WikiMultihopQA remain under
their original licenses.

The cross-family judge verdicts (`e5a_*`) were produced with Llama 3.3, licensed under
the Llama 3.3 Community License Agreement. Built with Llama.
