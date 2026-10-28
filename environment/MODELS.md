# Models

| Role in the paper | Hugging Face repository | Local name | Serving |
|---|---|---|---|
| Embedder for blocking and similarity (all runs) | `Qwen/Qwen3-Embedding-0.6B` | `Qwen3-Embedding-0.6B` | sentence-transformers, fp16 |
| Primary judge (`main_v10`, `main_v12_think`, `main_v11_k3`, E4 validation) | `Qwen/Qwen3-32B-AWQ` | `Qwen3-32B-AWQ` | vLLM, `--quantization awq_marlin`, thinking disabled except `main_v12_think` |
| Agent A (MuSiQue traces, E4 2Wiki traces) | `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` | `Tongyi-DeepResearch-30B-A3B` | vLLM, fp8, temperature 0.6, max 20 turns |
| Agent B (agent sensitivity, `agent_b`) | `Qwen/Qwen3-32B-AWQ` | `Qwen3-32B-AWQ` | vLLM, `awq_marlin` |
| Cross-family judge (E5) | `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4` @ `a3c28ea93c041acfaadb85401ad6493749a5aa04` | `Llama-3.3-70B-Instruct-AWQ-INT4` | vLLM, `awq_marlin`, max model len 16384 |

Judge decoding: temperature 0, `max_tokens` 600 (2,400 for the reasoning-enabled
configuration), passages in retrieval-rank order, 1,100 characters per passage.

`tools/export_artifacts.py --models-dir ...` records the exact revision of every local
model from its Hugging Face download metadata into `results/EXPORT_MANIFEST.json`.

`setup_matview_v2.sh` is the setup script used on the experiment machine. It also
downloads models that the paper does not use (e.g. a larger embedder and an alternative
judge); only the models in the table above back reported numbers.

Llama 3.3 is used under the Llama 3.3 Community License ("Built with Llama").
