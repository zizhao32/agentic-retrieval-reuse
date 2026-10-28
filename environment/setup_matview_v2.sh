#!/usr/bin/env bash
# =============================================================================
#  MatView Phase 1 — 一键环境搭建  v2
# =============================================================================
#
#  相对 v1 的变更（决策已定，不需要你再选）：
#
#    判官  Qwen3-32B-AWQ  ->  Qwen/Qwen3.8-27B-FP8   [主]
#                            Qwen/Qwen3-32B-AWQ      [对照，用于判官对拍]
#          理由：Qwen3.8-27B 64 层里仅 16 层全注意力，48 层 Gated DeltaNet
#          线性注意力，KV cache 极小 —— 判官任务是 2-3K token 文档 prefill +
#          200 token 输出，并发受 KV 限制，这个架构正好打在点上。
#          但"更强的判官"在本协议里不必然更好：模型越强越容易用自身知识
#          补全文档里没有的事实。所以两个都下，用金标集+陷阱对对拍决定。
#
#    Agent Tongyi-DeepResearch-30B-A3B  [主，领域专家 + 文献可比]
#          Qwen3.8-27B-FP8              [A/B，与判官共用同一份权重，零额外下载]
#          理由：复用率对 agent 强度敏感，审稿人必问。两条曲线是防御。
#
#    环境  vLLM 装最新版（Qwen3.8 是 Qwen3_5ForConditionalGeneration 架构，
#          需要 transformers>=5.8.0 和相当新的 vLLM），并新增架构兼容性实测。
#
#  用法:
#      chmod +x setup_matview_v2.sh
#      ./setup_matview_v2.sh --dry-run
#      ./setup_matview_v2.sh
#      ./setup_matview_v2.sh --only models --force
#
#  步骤: preflight / tools / envserve / envmain / models / data / launchers / verify
# =============================================================================

set -Eeuo pipefail

# ----------------------------------------------------------------- 默认配置 --
MATVIEW_ROOT="${MATVIEW_ROOT:-$HOME/matview}"
PY_VERSION="${PY_VERSION:-3.12}"
DOWNLOAD_MODELS=1
DOWNLOAD_DATA=1
DOWNLOAD_CHALLENGER=1     # 判官对照 Qwen3-32B-AWQ
DOWNLOAD_AGENT=1
DRY_RUN=0; FORCE=0; ONLY=""; ASSUME_YES=0
HF_PARALLEL="${HF_PARALLEL:-8}"

MIN_DISK_GB=1000
MIN_RAM_GB=64
MIN_DRIVER=535

# ---- 模型清单 ----
EMB_DEV="Qwen/Qwen3-Embedding-0.6B"            # ~1.2 GB
EMB_FINAL="Qwen/Qwen3-Embedding-8B"            # ~16  GB
JUDGE_MAIN="Qwen/Qwen3.8-27B-FP8"              # ~30  GB  判官主选 + agent A/B
JUDGE_CHALLENGER="Qwen/Qwen3-32B-AWQ"          # ~20  GB  判官对照
AGENT_MAIN="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"  # ~61 GB

CORPUS_REPO="Tevatron/browsecomp-plus-corpus"
DRBENCH_REPO="ServiceNow/drbench"
BCP_GIT="https://github.com/texttron/BrowseComp-Plus.git"
JDK_URL="https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse"

# --------------------------------------------------------------- 参数解析 --
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)             MATVIEW_ROOT="$2"; shift 2 ;;
    --python)           PY_VERSION="$2"; shift 2 ;;
    --skip-models)      DOWNLOAD_MODELS=0; shift ;;
    --skip-data)        DOWNLOAD_DATA=0; shift ;;
    --no-challenger)    DOWNLOAD_CHALLENGER=0; shift ;;
    --no-agent)         DOWNLOAD_AGENT=0; shift ;;
    --only)             ONLY="$2"; shift 2 ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --force)            FORCE=1; shift ;;
    -y|--yes)           ASSUME_YES=1; shift ;;
    -h|--help)          sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
done

# ------------------------------------------------------------------- 路径 --
ENV_SERVE="$MATVIEW_ROOT/envs/serve"
ENV_MAIN="$MATVIEW_ROOT/envs/main"
MODELS_DIR="$MATVIEW_ROOT/models"
DATA_DIR="$MATVIEW_ROOT/data"
TOOLS_DIR="$MATVIEW_ROOT/tools"
STATE_DIR="$MATVIEW_ROOT/.state"
LOG_DIR="$MATVIEW_ROOT/logs"
BIN_DIR="$MATVIEW_ROOT/bin"
JDK_DIR="$TOOLS_DIR/jdk21"
mkdir -p "$MATVIEW_ROOT"/{models,data,tools,logs,bin,envs} "$STATE_DIR"

SETUP_LOG="$LOG_DIR/setup_$(date +%Y%m%d_%H%M%S).log"; touch "$SETUP_LOG"

C_R='\033[0;31m'; C_G='\033[0;32m'; C_Y='\033[0;33m'; C_B='\033[0;36m'; C_D='\033[2m'; C_N='\033[0m'
log()  { echo -e "${C_D}$(date +%H:%M:%S)${C_N} $*" | tee -a "$SETUP_LOG"; }
info() { echo -e "${C_B}[INFO]${C_N} $*" | tee -a "$SETUP_LOG"; }
ok()   { echo -e "${C_G}[ OK ]${C_N} $*" | tee -a "$SETUP_LOG"; }
warn() { echo -e "${C_Y}[WARN]${C_N} $*" | tee -a "$SETUP_LOG"; }
err()  { echo -e "${C_R}[FAIL]${C_N} $*" | tee -a "$SETUP_LOG"; }
hdr()  { echo "" | tee -a "$SETUP_LOG"
         echo -e "${C_B}$(printf '=%.0s' {1..76})${C_N}" | tee -a "$SETUP_LOG"
         echo -e "${C_B}  $*${C_N}" | tee -a "$SETUP_LOG"
         echo -e "${C_B}$(printf '=%.0s' {1..76})${C_N}" | tee -a "$SETUP_LOG"; }

step_done()  { [[ $FORCE -eq 0 && -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { [[ $DRY_RUN -eq 0 ]] && touch "$STATE_DIR/$1.done"; }
should_run() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

TRAP_STEP="init"
on_error(){ err "步骤 [$TRAP_STEP] 失败（第 $1 行）。日志: $SETUP_LOG"
            err "修复后重跑即可，已完成步骤会自动跳过。"; exit 1; }
trap 'on_error $LINENO' ERR
STEP_T0=0
begin_step(){ TRAP_STEP="$1"; STEP_T0=$(date +%s); }
end_step(){ local d=$(( $(date +%s)-STEP_T0 )); ok "[$1] 完成 $((d/60))m$((d%60))s"; mark_done "$1"; }

# =============================================================================
# 1  preflight
# =============================================================================
step_preflight(){
  should_run preflight || return 0
  hdr "1/8  preflight — 系统自检"; begin_step preflight
  local p=0

  info "OS: $( . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -a )"
  [[ "$(uname -m)" == "x86_64" ]] || { err "仅支持 x86_64"; p=$((p+1)); }

  local glibc; glibc=$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0)
  info "glibc: $glibc"
  awk -v g="$glibc" 'BEGIN{exit !(g+0<2.28)}' && warn "glibc<2.28，faiss-gpu wheel 可能装不上（会自动降级 CPU）"

  if ! command -v nvidia-smi >/dev/null 2>&1; then err "找不到 nvidia-smi"; p=$((p+1));
  else
    local drv; drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1)
    info "驱动: $drv"
    [[ ${drv%%.*} -lt $MIN_DRIVER ]] && { err "驱动 < $MIN_DRIVER"; p=$((p+1)); }
    echo | tee -a "$SETUP_LOG"
    nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv 2>&1 | tee -a "$SETUP_LOG"
    echo | tee -a "$SETUP_LOG"
    local ngpu vram cc
    ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)
    vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|sort -n|head -1)
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -1)
    info "GPU 数 $ngpu / 最小显存 ${vram}MiB / compute_cap $cc"
    [[ $ngpu -lt 1 ]] && { err "无 GPU"; p=$((p+1)); }
    [[ ${vram:-0} -lt 70000 ]] && warn "显存<70GB：Tongyi bf16 跑不动，务必用 FP8；判官 FP8 仍可"
    if awk -v c="$cc" 'BEGIN{exit !(c+0 < 8.9)}'; then
      err "compute_cap $cc < 8.9，官方 block-wise FP8 权重需要 Ada/Hopper 及以后"; p=$((p+1))
    fi
    [[ "$cc" == "9.0" ]] && warn "Hopper(SM90)：faiss-gpu wheel 支持范围文档自相矛盾，verify 步会实测"
  fi

  local fg; fg=$(( $(df -Pk "$MATVIEW_ROOT"|awk 'NR==2{print $4}')/1024/1024 ))
  info "磁盘可用: ${fg}GB"
  if [[ $fg -lt $MIN_DISK_GB ]]; then
    warn "低于建议 ${MIN_DISK_GB}GB（模型~128GB / 语料+索引~40GB / Wikipedia~25GB / 中间产物~50GB）"
    [[ $fg -lt 280 ]] && { err "低于 280GB，装不下"; p=$((p+1)); }
  fi

  local ram; ram=$(awk '/MemTotal/{printf "%d",$2/1024/1024}' /proc/meminfo)
  info "内存 ${ram}GB / CPU $(nproc) 核"
  [[ $ram -lt $MIN_RAM_GB ]] && warn "内存偏小，编码大语料时调小 embed_batch"

  for h in huggingface.co pypi.org github.com api.adoptium.net; do
    curl -fsS --max-time 12 -o /dev/null "https://$h" 2>/dev/null && ok "可达 $h" \
      || { err "不可达 $h"; p=$((p+1)); }
  done
  [[ -n "${HF_ENDPOINT:-}" ]] && info "HF_ENDPOINT=$HF_ENDPOINT"

  if [[ -n "${HF_TOKEN:-}" || -f "$HOME/.cache/huggingface/token" ]]; then ok "HF token 已配置"
  else warn "无 HF token；BrowseComp-Plus 查询与 GAIA 是 gated，需 export HF_TOKEN=hf_xxx"; fi

  [[ $p -gt 0 ]] && { err "自检 $p 项阻塞问题"; exit 1; }
  end_step preflight
}

# =============================================================================
# 2  tools — uv + JDK21
# =============================================================================
step_tools(){
  should_run tools || return 0
  hdr "2/8  tools — uv 与 JDK 21"
  step_done tools && { ok "已完成，跳过"; return 0; }
  begin_step tools

  local UV=""
  if command -v uv >/dev/null 2>&1; then UV=$(command -v uv)
  elif [[ -x "$HOME/.local/bin/uv" ]]; then UV="$HOME/.local/bin/uv"
  else
    info "安装 uv"
    [[ $DRY_RUN -eq 0 ]] && { curl -LsSf https://astral.sh/uv/install.sh | sh >>"$SETUP_LOG" 2>&1 || true; }
    [[ -x "$HOME/.local/bin/uv" ]] && UV="$HOME/.local/bin/uv"
  fi
  [[ -n "$UV" ]] && ok "uv: $($UV --version)" || warn "uv 不可用，降级 venv+pip（慢）"
  echo "$UV" > "$STATE_DIR/uv_path"

  if [[ -x "$JDK_DIR/bin/java" ]]; then
    ok "JDK 已存在: $("$JDK_DIR/bin/java" -version 2>&1|head -1)"
  elif [[ $DRY_RUN -eq 0 ]]; then
    info "下载 Temurin JDK 21（~190MB，免 root）"
    mkdir -p "$JDK_DIR"
    curl -fL --retry 3 --progress-bar "$JDK_URL" -o "$TOOLS_DIR/jdk21.tar.gz"
    tar -xzf "$TOOLS_DIR/jdk21.tar.gz" -C "$JDK_DIR" --strip-components=1
    rm -f "$TOOLS_DIR/jdk21.tar.gz"
    ok "JDK: $("$JDK_DIR/bin/java" -version 2>&1|head -1)"
  fi
  end_step tools
}

# =============================================================================
# 3  envserve — vLLM（必须够新才认识 Qwen3_5ForConditionalGeneration）
# =============================================================================
step_envserve(){
  should_run envserve || return 0
  hdr "3/8  envserve — vLLM 推理环境"
  step_done envserve && { ok "已完成，跳过"; return 0; }
  begin_step envserve
  local UV; UV=$(cat "$STATE_DIR/uv_path" 2>/dev/null||echo "")

  info "创建 venv: $ENV_SERVE (python $PY_VERSION)"
  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -n "$UV" ]]; then "$UV" venv --python "$PY_VERSION" "$ENV_SERVE" >>"$SETUP_LOG" 2>&1
    else python${PY_VERSION} -m venv "$ENV_SERVE" >>"$SETUP_LOG" 2>&1 || python3 -m venv "$ENV_SERVE" >>"$SETUP_LOG" 2>&1; fi
  fi

  info "安装最新 vLLM + transformers>=5.8（Qwen3.8 架构要求；约 3-6GB，10-20 分钟）"
  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -n "$UV" ]]; then
      VIRTUAL_ENV="$ENV_SERVE" "$UV" pip install --upgrade \
        vllm "transformers>=5.8.0" "huggingface_hub[cli]" hf_transfer \
        2>&1 | tail -20 | tee -a "$SETUP_LOG"
    else
      "$ENV_SERVE/bin/pip" install -q --upgrade pip
      "$ENV_SERVE/bin/pip" install --upgrade vllm "transformers>=5.8.0" \
        "huggingface_hub[cli]" hf_transfer 2>&1 | tail -20 | tee -a "$SETUP_LOG"
    fi
  fi

  if [[ $DRY_RUN -eq 0 ]]; then
    local v t
    v=$("$ENV_SERVE/bin/python" -c "import vllm;print(vllm.__version__)" 2>/dev/null||echo FAIL)
    t=$("$ENV_SERVE/bin/python" -c "import transformers;print(transformers.__version__)" 2>/dev/null||echo FAIL)
    [[ "$v" == FAIL ]] && { err "vLLM 导入失败"; exit 1; }
    ok "vLLM $v / transformers $t"
    if awk -v x="${t%%.*}" 'BEGIN{exit !(x+0 < 5)}'; then
      err "transformers $t < 5.8.0，无法加载 Qwen3.8（Qwen3_5ForConditionalGeneration）"
      err "  手动: $ENV_SERVE/bin/pip install -U 'transformers>=5.8.0'"
      exit 1
    fi
    # 探测 vLLM 支持的开关，写进 serve_flags.env 供 launcher 用
    local H; H=$("$ENV_SERVE/bin/vllm" serve --help 2>/dev/null || echo "")
    : > "$STATE_DIR/serve_flags.env"
    grep -q -- "--language-model-only" <<<"$H" && echo 'LM_ONLY=--language-model-only' >> "$STATE_DIR/serve_flags.env" \
      && ok "支持 --language-model-only（可丢掉视觉塔省显存）"
    grep -q -- "--reasoning-parser"    <<<"$H" && echo 'REASON_PARSER=--reasoning-parser qwen3' >> "$STATE_DIR/serve_flags.env"
    grep -q -- "--enable-prefix-caching" <<<"$H" && echo 'PREFIX_CACHE=--enable-prefix-caching' >> "$STATE_DIR/serve_flags.env"
    grep -q -- "--enable-auto-tool-choice" <<<"$H" && echo 'TOOLS=--enable-auto-tool-choice' >> "$STATE_DIR/serve_flags.env"
  fi
  end_step envserve
}

# =============================================================================
# 4  envmain — 流水线
# =============================================================================
step_envmain(){
  should_run envmain || return 0
  hdr "4/8  envmain — 流水线环境（编码/索引/聚类/统计）"
  step_done envmain && { ok "已完成，跳过"; return 0; }
  begin_step envmain
  local UV; UV=$(cat "$STATE_DIR/uv_path" 2>/dev/null||echo "")
  local PIP=()

  info "创建 venv: $ENV_MAIN"
  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -n "$UV" ]]; then
      "$UV" venv --python "$PY_VERSION" "$ENV_MAIN" >>"$SETUP_LOG" 2>&1
      PIP=(env "VIRTUAL_ENV=$ENV_MAIN" "$UV" pip install)
    else
      python${PY_VERSION} -m venv "$ENV_MAIN" >>"$SETUP_LOG" 2>&1 || python3 -m venv "$ENV_MAIN" >>"$SETUP_LOG" 2>&1
      "$ENV_MAIN/bin/pip" install -q --upgrade pip
      PIP=("$ENV_MAIN/bin/pip" install)
    fi
  else PIP=(echo "[dry-run] pip install"); fi

  info "[1/5] torch (cu121，与 faiss-gpu-cu12 的 CUDA 主版本对齐)"
  "${PIP[@]}" torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1|tail -5|tee -a "$SETUP_LOG"

  info "[2/5] transformers / sentence-transformers"
  "${PIP[@]}" "transformers>=4.51" "sentence-transformers>=3.4" "accelerate>=1.0" \
              "tokenizers>=0.21" safetensors 2>&1|tail -5|tee -a "$SETUP_LOG"

  info "[3/5] faiss-gpu-cu12（GPU 可用性由 verify 实测）"
  if [[ $DRY_RUN -eq 0 ]]; then
    if "${PIP[@]}" faiss-gpu-cu12 >>"$SETUP_LOG" 2>&1; then
      ok "faiss-gpu-cu12 已装"; echo gpu-untested > "$STATE_DIR/faiss_mode"
    else
      warn "faiss-gpu-cu12 失败，降级 faiss-cpu"
      "${PIP[@]}" faiss-cpu 2>&1|tail -3|tee -a "$SETUP_LOG"; echo cpu > "$STATE_DIR/faiss_mode"
    fi
  fi

  info "[4/5] datasets / pyarrow / 聚类 / 统计 / 绘图"
  "${PIP[@]}" "datasets>=3.0" "pyarrow>=15" "pandas>=2.0" "numpy>=1.26,<2.3" \
      "scipy>=1.11" "scikit-learn>=1.4" "hdbscan>=0.8.38" "umap-learn>=0.5.6" "numba>=0.59" \
      "networkx>=3.2" "matplotlib>=3.8" seaborn tqdm requests pynvml psutil \
      "huggingface_hub[cli]" hf_transfer "openai>=1.40" orjson rich \
      2>&1|tail -5|tee -a "$SETUP_LOG"

  info "[5/5] pyserini (BM25)"
  if [[ $DRY_RUN -eq 0 ]]; then
    if JAVA_HOME="$JDK_DIR" "${PIP[@]}" pyserini >>"$SETUP_LOG" 2>&1; then
      ok "pyserini 已装"; echo 1 > "$STATE_DIR/pyserini_ok"
    else
      warn "pyserini 失败。BM25 基线缺失，Phase 1 主流程不受影响，Phase 2 前要补。"
      warn "  手动: export JAVA_HOME=$JDK_DIR && $ENV_MAIN/bin/pip install pyserini"
      echo 0 > "$STATE_DIR/pyserini_ok"
    fi
  fi
  end_step envmain
}

# =============================================================================
# 5  models
# =============================================================================
hf_dl(){
  local repo="$1" sub="$2" desc="$3" type="${4:-model}"
  local dest="$MODELS_DIR/$sub"
  [[ "$type" == dataset ]] && dest="$DATA_DIR/$sub"
  if [[ -f "$dest/.complete" && $FORCE -eq 0 ]]; then ok "已存在，跳过: $repo"; return 0; fi
  info "下载 $desc — $repo"
  [[ $DRY_RUN -eq 1 ]] && { echo "  [dry-run] hf download $repo -> $dest"; return 0; }
  mkdir -p "$dest"
  local HF="$ENV_MAIN/bin/hf"; [[ -x "$HF" ]] || HF="$ENV_MAIN/bin/huggingface-cli"
  local t0; t0=$(date +%s)
  local extra=(); [[ "$type" == dataset ]] && extra=(--repo-type dataset)
  if HF_HUB_ENABLE_HF_TRANSFER=1 "$HF" download "$repo" "${extra[@]}" \
       --local-dir "$dest" --max-workers "$HF_PARALLEL" 2>&1|tail -3|tee -a "$SETUP_LOG"; then
    touch "$dest/.complete"
    ok "$repo 完成 ($(du -sh "$dest" 2>/dev/null|cut -f1), $(( ($(date +%s)-t0)/60 ))min)"
  else
    err "$repo 下载失败。gated 仓库需先网页申请并 export HF_TOKEN"; return 1
  fi
}

step_models(){
  should_run models || return 0
  [[ $DOWNLOAD_MODELS -eq 0 ]] && { info "--skip-models"; return 0; }
  hdr "5/8  models — 模型权重"
  step_done models && { ok "已完成，跳过"; return 0; }
  begin_step models

  cat <<EOF | tee -a "$SETUP_LOG"

  即将下载（合计 ~128GB）：
    嵌入-开发   $EMB_DEV                       ~1.2GB
    嵌入-正式   $EMB_FINAL                     ~16GB
    判官-主     $JUDGE_MAIN                    ~30GB   (同时用作 agent A/B)
$( [[ $DOWNLOAD_CHALLENGER -eq 1 ]] && echo "    判官-对照   $JUDGE_CHALLENGER   ~20GB   (判官对拍用)" )
$( [[ $DOWNLOAD_AGENT -eq 1 ]] && echo "    Agent-主    $AGENT_MAIN   ~61GB" )
EOF
  if [[ $ASSUME_YES -eq 0 && $DRY_RUN -eq 0 ]]; then
    read -r -p "  继续? [Y/n] " a; [[ "${a:-Y}" =~ ^[Nn] ]] && { warn "已取消"; exit 0; }
  fi

  hf_dl "$EMB_DEV"    "Qwen3-Embedding-0.6B"  "嵌入(开发迭代)"
  hf_dl "$EMB_FINAL"  "Qwen3-Embedding-8B"    "嵌入(正式数字)"
  hf_dl "$JUDGE_MAIN" "Qwen3.8-27B-FP8"       "判官主选 / agent A/B"
  [[ $DOWNLOAD_CHALLENGER -eq 1 ]] && hf_dl "$JUDGE_CHALLENGER" "Qwen3-32B-AWQ" "判官对照"
  [[ $DOWNLOAD_AGENT -eq 1 ]]      && hf_dl "$AGENT_MAIN" "Tongyi-DeepResearch-30B-A3B" "Agent 主选"

  info "模型目录: $(du -sh "$MODELS_DIR" 2>/dev/null|cut -f1)"
  end_step models
}

# =============================================================================
# 6  data
# =============================================================================
step_data(){
  should_run data || return 0
  [[ $DOWNLOAD_DATA -eq 0 ]] && { info "--skip-data"; return 0; }
  hdr "6/8  data — 语料与数据集"
  step_done data && { ok "已完成，跳过"; return 0; }
  begin_step data
  local PY="$ENV_MAIN/bin/python"

  hf_dl "$CORPUS_REPO" "bcp-corpus" "BrowseComp-Plus 语料 (100K 文档)" dataset
  hf_dl "$DRBENCH_REPO" "drbench"   "DRBench 企业域"                    dataset || true

  info "BrowseComp-Plus 代码仓"
  if [[ $DRY_RUN -eq 0 ]]; then
    if [[ -d "$DATA_DIR/BrowseComp-Plus/.git" ]]; then
      (cd "$DATA_DIR/BrowseComp-Plus" && git pull --ff-only >>"$SETUP_LOG" 2>&1) || true
    else
      git clone --depth 1 "$BCP_GIT" "$DATA_DIR/BrowseComp-Plus" >>"$SETUP_LOG" 2>&1
    fi
    ok "代码仓就位"
  fi

  info "多跳数据集 MuSiQue / HotpotQA / 2WikiMultihopQA"
  if [[ $DRY_RUN -eq 0 ]]; then
    HF_DATASETS_CACHE="$DATA_DIR/hf_cache" "$PY" - <<'PYEOF' 2>&1 | tee -a "$SETUP_LOG"
import os
from datasets import load_dataset
cache = os.environ["HF_DATASETS_CACHE"]
for repo, cfg, split, name in [
    ("dgslibisey/MuSiQue",      None,        "validation", "MuSiQue"),
    ("hotpotqa/hotpot_qa",      "distractor","validation", "HotpotQA"),
    ("xanhho/2WikiMultihopQA",  None,        "validation", "2WikiMultihopQA"),
]:
    try:
        kw = {"cache_dir": cache, "split": split}
        if repo.startswith("hotpotqa"): kw["trust_remote_code"] = True
        ds = load_dataset(repo, cfg, **kw) if cfg else load_dataset(repo, **kw)
        print(f"  OK  {name:18s} {len(ds):>7d} 条")
    except Exception as e:
        print(f"  !!  {name:18s} 失败: {type(e).__name__}: {str(e)[:140]}")
        print("      三者缺一不致命，但 MuSiQue 必须有（唯一带 question_decomposition，"
              "是子任务级查询的主要来源）")
PYEOF
  fi

  warn "GAIA 是 gated，需网页手动申请（可能等几天，现在就去点）："
  warn "  https://huggingface.co/datasets/gaia-benchmark/GAIA"
  info "数据目录: $(du -sh "$DATA_DIR" 2>/dev/null|cut -f1)"
  end_step data
}

# =============================================================================
# 7  launchers
# =============================================================================
step_launchers(){
  should_run launchers || return 0
  hdr "7/8  launchers — 启动脚本 / env.sh / 判官对拍 / 思考开关探测"
  begin_step launchers

  local JUDGE_LOCAL="$MODELS_DIR/Qwen3.8-27B-FP8"
  local CHAL_LOCAL="$MODELS_DIR/Qwen3-32B-AWQ"
  local AGENT_LOCAL="$MODELS_DIR/Tongyi-DeepResearch-30B-A3B"
  [[ -f "$STATE_DIR/serve_flags.env" ]] || : > "$STATE_DIR/serve_flags.env"

  # -------------------------------------------------------------- env.sh --
  cat > "$MATVIEW_ROOT/env.sh" <<EOF
export MATVIEW_ROOT="$MATVIEW_ROOT"
export MATVIEW_MODELS="$MODELS_DIR"
export MATVIEW_DATA="$DATA_DIR"
export JAVA_HOME="$JDK_DIR"
export PATH="\$JAVA_HOME/bin:$BIN_DIR:\$PATH"
export HF_HOME="$DATA_DIR/hf_home"
export HF_DATASETS_CACHE="$DATA_DIR/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

# 与 phase1.py 默认值一致
export MATVIEW_JUDGE_URL="http://localhost:8001/v1"
export MATVIEW_JUDGE_MODEL="$JUDGE_MAIN"
export MATVIEW_AGENT_URL="http://localhost:8000/v1"
export MATVIEW_AGENT_MODEL="$AGENT_MAIN"

echo "MatView 环境已加载"
echo "  流水线 python: $ENV_MAIN/bin/python"
echo "  vLLM   python: $ENV_SERVE/bin/python"
EOF
  ok "env.sh"

  # -------------------------------------------------------- serve_judge --
  cat > "$BIN_DIR/serve_judge.sh" <<EOF
#!/usr/bin/env bash
# 判官主服务 — Qwen3.8-27B-FP8（默认卡 1，端口 8001）
#
# 为什么并发能开这么大：64 层里只有 16 层全注意力，48 层是 Gated DeltaNet
# 线性注意力，KV 是常数状态。判官负载是 2-3K token prefill + 200 token 输出，
# 正好吃这个架构的红利。
#
# FP8 是官方 block-wise 权重，vLLM 从 config 自动识别，不要加 --quantization。
set -e
source "$STATE_DIR/serve_flags.env" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=\${JUDGE_GPU:-1}
exec "$ENV_SERVE/bin/vllm" serve "$JUDGE_LOCAL" \\
    --served-model-name "$JUDGE_MAIN" \\
    --port \${JUDGE_PORT:-8001} \\
    --max-model-len \${JUDGE_CTX:-40960} \\
    --gpu-memory-utilization \${JUDGE_MEM:-0.90} \\
    --max-num-seqs \${JUDGE_SEQS:-128} \\
    --trust-remote-code \\
    \${PREFIX_CACHE:-} \${REASON_PARSER:-} \${LM_ONLY:-} \\
    --disable-log-requests \\
    "\$@"
EOF

  # ---------------------------------------------------- serve_judge_alt --
  cat > "$BIN_DIR/serve_judge_alt.sh" <<EOF
#!/usr/bin/env bash
# 判官对照 — Qwen3-32B-AWQ（默认卡 1，端口 8011），仅用于判官对拍
set -e
source "$STATE_DIR/serve_flags.env" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=\${JUDGE_GPU:-1}
exec "$ENV_SERVE/bin/vllm" serve "$CHAL_LOCAL" \\
    --served-model-name "$JUDGE_CHALLENGER" \\
    --port \${JUDGE_ALT_PORT:-8011} \\
    --max-model-len \${JUDGE_CTX:-32768} \\
    --gpu-memory-utilization \${JUDGE_MEM:-0.90} \\
    --max-num-seqs \${JUDGE_SEQS:-64} \\
    --quantization awq_marlin \\
    \${PREFIX_CACHE:-} --disable-log-requests "\$@"
EOF

  # -------------------------------------------------------- serve_agent --
  cat > "$BIN_DIR/serve_agent.sh" <<EOF
#!/usr/bin/env bash
# Agent 主服务 — Tongyi-DeepResearch-30B-A3B（默认卡 0，端口 8000，FP8）
#
# 显存账：30.5B bf16 = 61GB，80GB 卡只剩 19GB KV，32K 上下文下并发只有 5-6。
# FP8 把权重压到 ~31GB，并发能到 15+。仍不够就 AGENT_GPU=0,1 AGENT_TP=2
# （代价是判官要挪走或排队）。
set -e
source "$STATE_DIR/serve_flags.env" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=\${AGENT_GPU:-0}
exec "$ENV_SERVE/bin/vllm" serve "$AGENT_LOCAL" \\
    --served-model-name "$AGENT_MAIN" \\
    --port \${AGENT_PORT:-8000} \\
    --max-model-len \${AGENT_CTX:-65536} \\
    --quantization \${AGENT_QUANT:-fp8} \\
    --tensor-parallel-size \${AGENT_TP:-1} \\
    --gpu-memory-utilization \${AGENT_MEM:-0.92} \\
    --max-num-seqs \${AGENT_SEQS:-16} \\
    \${PREFIX_CACHE:-} \${TOOLS:-} --tool-call-parser hermes \\
    --disable-log-requests "\$@"
EOF

  # ---------------------------------------------------- serve_agent_alt --
  cat > "$BIN_DIR/serve_agent_alt.sh" <<EOF
#!/usr/bin/env bash
# Agent A/B — Qwen3.8-27B-FP8（默认卡 0，端口 8010）
#
# 与判官共用同一份权重，零额外下载。用途：测"复用率对 agent 强度的敏感性"。
# 审稿人一定会问"更强的 agent 会不会让你的视图失效"，两条曲线就是答案。
# 注意：跑 agent 时思考模式的开关要和判官保持一致口径并写进论文。
set -e
source "$STATE_DIR/serve_flags.env" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=\${AGENT_GPU:-0}
exec "$ENV_SERVE/bin/vllm" serve "$JUDGE_LOCAL" \\
    --served-model-name "$JUDGE_MAIN" \\
    --port \${AGENT_ALT_PORT:-8010} \\
    --max-model-len \${AGENT_CTX:-65536} \\
    --gpu-memory-utilization \${AGENT_MEM:-0.90} \\
    --max-num-seqs \${AGENT_SEQS:-24} \\
    --trust-remote-code \\
    \${PREFIX_CACHE:-} \${REASON_PARSER:-} \${TOOLS:-} --tool-call-parser hermes \\
    --disable-log-requests "\$@"
EOF

  # ---------------------------------------------------------- serve_all --
  cat > "$BIN_DIR/serve_all.sh" <<EOF
#!/usr/bin/env bash
# 后台起判官(8001)+agent(8000)并等待就绪
set -e
mkdir -p "$LOG_DIR"
nohup "$BIN_DIR/serve_judge.sh" > "$LOG_DIR/vllm_judge.log" 2>&1 & echo \$! > "$STATE_DIR/judge.pid"
nohup "$BIN_DIR/serve_agent.sh" > "$LOG_DIR/vllm_agent.log" 2>&1 & echo \$! > "$STATE_DIR/agent.pid"
echo "已后台启动，等待就绪（首次加载 3-10 分钟）..."
for port in 8001 8000; do
  for i in \$(seq 1 120); do
    curl -fsS "http://localhost:\$port/v1/models" >/dev/null 2>&1 && { echo "  port \$port 就绪"; break; }
    [[ \$i -eq 120 ]] && echo "  port \$port 超时，看 $LOG_DIR/"
    sleep 10
  done
done
EOF

  cat > "$BIN_DIR/stop_all.sh" <<EOF
#!/usr/bin/env bash
for f in "$STATE_DIR"/*.pid; do
  [[ -f "\$f" ]] || continue
  pid=\$(cat "\$f"); kill "\$pid" 2>/dev/null && echo "已停止 \$(basename "\$f" .pid) (\$pid)"; rm -f "\$f"
done
EOF

  # ------------------------------------------------- probe_thinking.py --
  # Qwen3.8 默认 reasoning effort 是 xhigh。12000 次判官调用开着思考会把
  # 这一步从 1 小时变成 1 天，而且协议的 prompt 本身就是显式 CoT，冗余。
  # 这个脚本探测你这个 vLLM 版本认哪种关闭方式。
  cat > "$BIN_DIR/probe_thinking.py" <<'EOF'
#!/usr/bin/env python3
"""探测本机 vLLM 上关闭 Qwen3.8 思考模式的正确姿势。

用法: python probe_thinking.py [--url http://localhost:8001/v1] [--model NAME]
把 WINNER 那一行的写法填进 phase1.py 的 _judge_one。
"""
import argparse, json, time, requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8001/v1")
ap.add_argument("--model", default=None)
a = ap.parse_args()

model = a.model or requests.get(f"{a.url}/models", timeout=15).json()["data"][0]["id"]
print(f"model = {model}\n")

MSG = [{"role": "user", "content":
        '只输出 JSON，不要任何其他文字: {"verdict":"SUFFICIENT","confidence":0.9}'}]

VARIANTS = {
    "baseline(默认)":          {},
    "chat_template_kwargs":    {"chat_template_kwargs": {"enable_thinking": False}},
    "reasoning_effort=minimal":{"reasoning_effort": "minimal"},
    "reasoning_effort=low":    {"reasoning_effort": "low"},
    "extra_body 嵌套":          {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
}

results = []
for name, extra in VARIANTS.items():
    body = {"model": model, "messages": MSG, "temperature": 0.0,
            "max_tokens": 800, **extra}
    t0 = time.time()
    try:
        r = requests.post(f"{a.url}/chat/completions", json=body, timeout=180)
        if r.status_code != 200:
            print(f"  {name:28s} HTTP {r.status_code}: {r.text[:110]}")
            continue
        d = r.json()
        m = d["choices"][0]["message"]
        out = (m.get("content") or "")
        think = m.get("reasoning_content") or m.get("reasoning") or ""
        ct = d.get("usage", {}).get("completion_tokens", 0)
        dt = time.time() - t0
        results.append((ct, name, dt, len(think)))
        print(f"  {name:28s} completion={ct:>4d} tok  思考={len(think):>5d} 字  "
              f"{dt:5.2f}s  输出前60字: {out.strip()[:60]!r}")
    except Exception as e:
        print(f"  {name:28s} 异常: {type(e).__name__}: {str(e)[:90]}")

if results:
    results.sort()
    best_ct, best_name, best_dt, _ = results[0]
    base = next((c for c, n, _, _ in results if n.startswith("baseline")), None)
    print(f"\n  WINNER: {best_name}  ({best_ct} completion tokens, {best_dt:.2f}s)")
    if base and base > 0:
        print(f"  相对默认省下 {100*(1-best_ct/base):.0f}% 的输出 token")
    if best_name.startswith("baseline"):
        print("  注意: 默认就已经不思考，或该版本未接通思考开关。跑判官前"
              "务必用 gold_pairs 抽 20 对实测一次单对耗时，再推算 s6 总时长。")
EOF

  # ----------------------------------------------------- judge_bakeoff --
  cat > "$BIN_DIR/judge_bakeoff.sh" <<EOF
#!/usr/bin/env bash
# 判官对拍：Qwen3.8-27B-FP8 vs Qwen3-32B-AWQ
#
# 为什么必须做：本协议的 L1 prompt 要求判官"只看检索结果，不要用自己的知识
# 补全"。模型越强越知道答案，就越容易违反这条 —— 更强的通用能力在这里
# 不必然是更好的判官。判据是陷阱对误判率和金标 precision，不是 benchmark。
#
# 前提：stages/s5/gold_pairs.jsonl 的 human_label 已人工填好。
set -e
cd "$MATVIEW_ROOT"
RUN=\${1:-bakeoff}
echo "=== A: Qwen3.8-27B-FP8 (port 8001) ==="
"$ENV_MAIN/bin/python" phase1.py --run-id "\${RUN}_qwen38" --stages s6 --force
echo
echo "=== B: Qwen3-32B-AWQ (port 8011) ==="
JUDGE_ALT_PORT=8011 nohup "$BIN_DIR/serve_judge_alt.sh" > "$LOG_DIR/vllm_judge_alt.log" 2>&1 &
echo \$! > "$STATE_DIR/judge_alt.pid"
for i in \$(seq 1 90); do curl -fsS localhost:8011/v1/models >/dev/null 2>&1 && break; sleep 10; done
cat > /tmp/mv_alt.json <<JSON
{"judge_base_url":"http://localhost:8011/v1","judge_model":"$JUDGE_CHALLENGER"}
JSON
"$ENV_MAIN/bin/python" phase1.py --run-id "\${RUN}_qwen332" --config /tmp/mv_alt.json --stages s6 --force
echo
echo "=== 对比（按此表选判官，然后冻结）==="
for r in "\${RUN}_qwen38" "\${RUN}_qwen332"; do
  echo "--- \$r"
  "$ENV_MAIN/bin/python" -c "
import json,sys
q=json.load(open('runs/\$r/stages/s6/judge_qc.json'))
g=q.get('gold') or {}
print(f\"  陷阱误判率 {q.get('catch_false_positive_rate')}   <- 越低越好，这是核心判据\")
print(f\"  金标 acc/P/R {g.get('accuracy')}/{g.get('precision')}/{g.get('recall')}\")
print(f\"  位置一致率 {q.get('pos_consistency')}   方向不对称率 {q.get('asymmetry_rate')}\")
print(f\"  解析失败率 {q.get('parse_fail_rate')}   耗时 {q.get('wall_sec',0)/60:.1f}min\")
"
done
echo
echo "选定后写进 config.json 并冻结。中途换判官会让 s6/s8 全部作废。"
EOF

  chmod +x "$BIN_DIR"/*.sh "$BIN_DIR"/*.py 2>/dev/null || true
  ok "生成 $BIN_DIR/{serve_judge,serve_judge_alt,serve_agent,serve_agent_alt,serve_all,stop_all}.sh"
  ok "生成 $BIN_DIR/{probe_thinking.py,judge_bakeoff.sh}"
  end_step launchers
}

# =============================================================================
# 8  verify
# =============================================================================
step_verify(){
  should_run verify || return 0
  hdr "8/8  verify — 端到端验证"
  begin_step verify
  [[ $DRY_RUN -eq 1 ]] && { info "[dry-run] 跳过"; end_step verify; return 0; }

  export JAVA_HOME="$JDK_DIR"; export PATH="$JAVA_HOME/bin:$PATH"

  # ---- 8a. Qwen3.8 架构兼容性：serve 环境里只读 config，不加载权重 ----
  info "Qwen3.8 架构兼容性（serve 环境，不加载权重）"
  MATVIEW_MODELS="$MODELS_DIR" "$ENV_SERVE/bin/python" - <<'PYEOF' 2>&1 | tee -a "$SETUP_LOG"
import os, sys, json
p = os.path.join(os.environ["MATVIEW_MODELS"], "Qwen3.8-27B-FP8")
if not os.path.isdir(p):
    print("  [WARN] 判官模型未下载（--skip-models?），跳过"); sys.exit(0)
try:
    cfg = json.load(open(os.path.join(p, "config.json")))
    arch = cfg.get("architectures", ["?"])[0]
    print(f"  架构        {arch}")
    print(f"  量化        {cfg.get('quantization_config', {}).get('quant_method')}")
    print(f"  最大位置    {cfg.get('max_position_embeddings')}")
    print(f"  视觉塔      {'有' if 'vision_config' in cfg else '无'}")
    from transformers import AutoConfig
    AutoConfig.from_pretrained(p, trust_remote_code=True)
    print("  [ OK ] transformers 能解析该 config")
except Exception as e:
    print(f"  [FAIL] transformers 解析失败: {type(e).__name__}: {str(e)[:160]}")
    print("         多半是 transformers 版本不够，需 >=5.8.0")
    sys.exit(1)
try:
    from vllm.model_executor.models.registry import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    print(f"  [{'  OK' if arch in supported else 'FAIL'}] vLLM {'认识' if arch in supported else '不认识'} {arch}")
    if arch not in supported:
        print("         升级 vLLM: envs/serve/bin/pip install -U vllm  （或用 nightly）")
        sys.exit(1)
except Exception as e:
    print(f"  [WARN] 无法查询 vLLM 架构注册表（{type(e).__name__}），起服务时再看")
PYEOF

  # ---- 8b. 流水线环境 ----
  info "流水线环境验证"
  MATVIEW_MODELS="$MODELS_DIR" MATVIEW_STATE="$STATE_DIR" \
  HF_DATASETS_CACHE="$DATA_DIR/hf_cache" "$ENV_MAIN/bin/python" - <<'PYEOF' 2>&1 | tee -a "$SETUP_LOG"
import os, sys, time, glob, json
import numpy as np
FAIL = []
def check(name, fn, critical=True):
    t0 = time.time()
    try:
        msg = fn(); print(f"  [ OK ] {name:30s} {msg or ''}  ({time.time()-t0:.1f}s)"); return True
    except Exception as e:
        print(f"  [{'FAIL' if critical else 'WARN'}] {name:30s} {type(e).__name__}: {str(e)[:110]}")
        if critical: FAIL.append(name)
        return False

print("\n-- 依赖 --")
def _t():
    import torch; assert torch.cuda.is_available(), "CUDA 不可用"
    n = torch.cuda.device_count()
    return f"torch {torch.__version__}, {n} GPU: " + ", ".join(torch.cuda.get_device_name(i) for i in range(n))
check("torch + CUDA", _t)
for m, c in [("transformers",1),("sentence_transformers",1),("datasets",1),("pyarrow",1),
             ("sklearn",1),("scipy",1),("networkx",1),("matplotlib",1),
             ("hdbscan",0),("umap",0),("pynvml",0),("pyserini",0)]:
    check(f"import {m}", (lambda k=m: getattr(__import__(k),"__version__","ok")), bool(c))

print("\n-- faiss GPU 实测（唯一靠谱的判定方式）--")
def _fg():
    import faiss
    d,n = 128, 20000
    xb = np.random.rand(n,d).astype("float32"); faiss.normalize_L2(xb)
    idx = faiss.IndexFlatIP(d)
    res = faiss.StandardGpuResources(); idx = faiss.index_cpu_to_gpu(res, 0, idx)
    idx.add(xb); D,I = idx.search(xb[:64], 10); assert I.shape==(64,10)
    open(os.path.join(os.environ["MATVIEW_STATE"],"faiss_mode"),"w").write("gpu")
    return f"faiss {faiss.__version__} [GPU] 通过"
if not check("faiss GPU", _fg, critical=False):
    def _fc():
        import faiss
        d,n = 128, 20000
        xb = np.random.rand(n,d).astype("float32"); faiss.normalize_L2(xb)
        idx = faiss.IndexFlatIP(d); idx.add(xb); idx.search(xb[:64],10)
        open(os.path.join(os.environ["MATVIEW_STATE"],"faiss_mode"),"w").write("cpu")
        return f"faiss {faiss.__version__} [CPU] 通过 -> 把 phase1.py 的 faiss_use_gpu 设 false"
    check("faiss CPU 回退", _fc, critical=True)

print("\n-- 嵌入模型冒烟 --")
def _e():
    from sentence_transformers import SentenceTransformer
    p = os.path.join(os.environ["MATVIEW_MODELS"], "Qwen3-Embedding-0.6B")
    if not os.path.isdir(p): raise FileNotFoundError("未下载")
    m = SentenceTransformer(p, model_kwargs={"torch_dtype":"float16"},
                            tokenizer_kwargs={"padding_side":"left"})
    e = m.encode(["谁创立了苹果公司","Apple 公司由谁创办"],
                 normalize_embeddings=True, show_progress_bar=False)
    s = float(e[0]@e[1]); assert 0.3 < s < 1.01, f"同义句相似度异常 {s}"
    return f"dim={e.shape[1]}, 同义句 cos={s:.3f}"
check("Qwen3-Embedding-0.6B", _e, critical=False)

print("\n-- 权重分片完整性（不加载）--")
def _w():
    root = os.environ["MATVIEW_MODELS"]; out=[]
    for d in sorted(os.listdir(root)):
        p = os.path.join(root,d)
        if not os.path.isdir(p): continue
        st = glob.glob(os.path.join(p,"*.safetensors"))
        idx = os.path.join(p,"model.safetensors.index.json"); missing=[]
        if os.path.exists(idx):
            need = set(json.load(open(idx))["weight_map"].values())
            missing = sorted(need - {os.path.basename(x) for x in st})
        sz = sum(os.path.getsize(x) for x in st)/2**30
        flag = "OK" if (os.path.exists(os.path.join(p,"config.json")) and st and not missing) else "INCOMPLETE"
        out.append(f"      {d:32s} {len(st):>3d} shard {sz:6.1f}GB  {flag}" +
                   (f"  缺 {len(missing)} 片" if missing else ""))
    return "\n"+"\n".join(out) if out else "（无模型）"
check("权重校验", _w, critical=False)

print("\n-- 数据 --")
def _d():
    from datasets import load_dataset
    ds = load_dataset("dgslibisey/MuSiQue", split="validation[:20]",
                      cache_dir=os.environ.get("HF_DATASETS_CACHE"))
    assert "question_decomposition" in ds[0], "缺 question_decomposition"
    return f"MuSiQue 可读，样例 {len(ds[0]['question_decomposition'])} 个子问题"
check("MuSiQue", _d, critical=False)

def _j():
    import subprocess
    o = subprocess.run(["java","-version"], capture_output=True, timeout=30)
    return (o.stderr or o.stdout).decode(errors="ignore").splitlines()[0]
check("java -version", _j, critical=False)

print()
if FAIL:
    print(f"  >>> {len(FAIL)} 项关键失败: {', '.join(FAIL)}"); sys.exit(1)
print("  >>> 关键检查全部通过")
PYEOF
  end_step verify
}

# =============================================================================
summary(){
  hdr "搭建完成"
  local fm; fm=$(cat "$STATE_DIR/faiss_mode" 2>/dev/null || echo unknown)
  cat <<EOF | tee -a "$SETUP_LOG"

  根目录  $MATVIEW_ROOT
  模型    $(du -sh "$MODELS_DIR" 2>/dev/null|cut -f1)
  数据    $(du -sh "$DATA_DIR" 2>/dev/null|cut -f1)
  剩余    $(df -Ph "$MATVIEW_ROOT"|awk 'NR==2{print $4}')
  日志    $SETUP_LOG

  角色分配
    判官-主    $JUDGE_MAIN         port 8001  (卡 1)
    判官-对照  $JUDGE_CHALLENGER   port 8011  (卡 1，对拍时才起)
    Agent-主   $AGENT_MAIN         port 8000  (卡 0，FP8)
    Agent-A/B  $JUDGE_MAIN         port 8010  (卡 0，与判官共用权重)

  --------------------------------------------------------------------------
  下一步（按顺序）
  --------------------------------------------------------------------------
  1) source $MATVIEW_ROOT/env.sh

  2) $BIN_DIR/serve_all.sh          # 起判官+agent，等 3-10 分钟

  3) $ENV_MAIN/bin/python $BIN_DIR/probe_thinking.py
     ！这一步不能跳。Qwen3.8 的 reasoning effort 默认是 xhigh，12000 次
       判官调用开着思考会把 s6 从 1 小时拖成 1 天。而协议的 prompt 本身
       就要求模型先列 required_facts 再逐条核对，已经是显式 CoT，隐藏思考
       是纯冗余。把 WINNER 那一行的写法填进 phase1.py 的 _judge_one。

  4) cd $MATVIEW_ROOT
     $ENV_MAIN/bin/python phase1.py --stages s0
     $ENV_MAIN/bin/python phase1.py --smoke        # 约 20 分钟全流程

  5) 人工标注 runs/<id>/stages/s5/gold_pairs.jsonl 的 human_label (1/0)
     然后 $BIN_DIR/judge_bakeoff.sh
     按陷阱误判率和金标 precision 选判官，选完冻结，别再改。

  6) $ENV_MAIN/bin/python phase1.py --run-id main_v1

  --------------------------------------------------------------------------
  仍需手动
  --------------------------------------------------------------------------
  [1] GAIA gated 申请（今天就去点）
      https://huggingface.co/datasets/gaia-benchmark/GAIA

  [2] BrowseComp-Plus 查询解密（需 HF 登录）
      cd $DATA_DIR/BrowseComp-Plus && $ENV_MAIN/bin/python \\
        scripts_build_index/decrypt_dataset.py \\
        --output data/browsecomp_plus_decrypted.jsonl \\
        --generate-tsv topics-qrels/queries.tsv

  [3] phase1.py 的 config.json
      judge_model       = "$JUDGE_MAIN"
      agent_model       = "$AGENT_MAIN"
      embed_model_dev   = "$MODELS_DIR/Qwen3-Embedding-0.6B"
      embed_model_final = "$MODELS_DIR/Qwen3-Embedding-8B"
      faiss_use_gpu     = $( [[ "$fm" == cpu ]] && echo "false  <- faiss 已降级 CPU" || echo "true" )

EOF
  [[ "$(cat "$STATE_DIR/pyserini_ok" 2>/dev/null)" == "0" ]] && \
    warn "pyserini 未装上，BM25 基线缺失。Phase 1 不受影响，Phase 2 前补。"
  return 0
}

hdr "MatView Phase 1 环境搭建 v2"
log "根目录 $MATVIEW_ROOT   Python $PY_VERSION"
log "判官 $JUDGE_MAIN (对照 $JUDGE_CHALLENGER)   Agent $AGENT_MAIN"
[[ $DRY_RUN -eq 1 ]] && warn "DRY-RUN，不会实际改动"
[[ -n "$ONLY" ]] && info "只执行: $ONLY"

step_preflight; step_tools; step_envserve; step_envmain
step_models; step_data; step_launchers; step_verify; summary
