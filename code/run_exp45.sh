#!/usr/bin/env bash
# ==============================================================================
# run_exp45.sh —— 后台跑 E4 / E5 补实验（exp45.py 的 nohup 包装）
#
#   bash run_exp45.sh download        # ① 下载 Llama-3.3-70B AWQ（~40 GB）+ 校验已有模型与 2Wiki 数据
#   bash run_exp45.sh smoke           # ② 冒烟：全流程小规模，约 1–1.5 h，输出在 runs/phase3_smoke
#   bash run_exp45.sh all             # ③ 正式：约 23–27 h（两张 H100，串行）
#   bash run_exp45.sh audit           #    任意单步：audit / e5a / e5b / e4-collect / ... / report
#   bash run_exp45.sh e5a --gate-only #    额外参数原样传给 exp45.py
#
# 在 projB 目录（runs/ 所在处）下运行；exp45.py、phase1_v13.py、fix_judge.py 放在同一目录。
# 中途断了直接重跑同一条命令：已完成的步跳过，长任务从断点接着跑。
# ==============================================================================
set -euo pipefail

R=${MATVIEW_ROOT:-/mnt/data/user_workspace/zhangzizhao/matview}
PY=${PY:-$R/envs/main/bin/python}
CFG=${CFG:-$R/config.json}

[[ -d runs/main_v10 ]] || { echo "找不到 runs/main_v10：请在 projB 目录下运行"; exit 1; }
for f in exp45.py phase1_v13.py fix_judge.py; do
  [[ -f $f ]] || { echo "缺 $f（与 exp45.py 放同一目录）"; exit 1; }
done
[[ -x $PY ]] || { echo "找不到解释器 $PY（用 PY=... 指定）"; exit 1; }

# HF 连通性：没设 HF_ENDPOINT 且直连不通，就用镜像
if [[ -z "${HF_ENDPOINT:-}" ]]; then
  if ! curl -sI --max-time 8 https://huggingface.co >/dev/null 2>&1; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "[info] huggingface.co 直连不通，改用 HF_ENDPOINT=$HF_ENDPOINT"
  fi
fi

# 残留的 vLLM 进程会让 autotune 量错显存
if pgrep -f "vllm serve" >/dev/null 2>&1; then
  echo "[warn] 有 vllm serve 进程在跑："
  pgrep -af "vllm serve" | head -5
  echo "       如果是上一次残留的：pkill -9 -f 'vllm serve'; pkill -9 -f EngineCore; sleep 10"
  read -r -t 20 -p "       仍然继续？[y/N] " yn || yn=n
  [[ "$yn" == "y" ]] || exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

mode=${1:-all}; shift || true
case "$mode" in
  smoke) args=(all --smoke "$@"); P3=runs/phase3_smoke ;;
  all)   args=(all "$@");          P3=runs/phase3 ;;
  *)     args=("$mode" "$@");      P3=runs/phase3 ;;
esac
mkdir -p "$P3/logs"
LOG="$P3/logs/nohup_${mode}_$(date +%m%d_%H%M%S).log"

# setsid：放进独立会话。只用 nohup 时，网页终端断开/关闭会对整个进程树发 SIGTERM
# （2026-10-23 的 E5a 就是这样在 56 分钟时被停掉的），nohup 只挡 SIGHUP。
setsid nohup "$PY" exp45.py --config "$CFG" "${args[@]}" > "$LOG" 2>&1 < /dev/null &
echo $! > "$P3/exp45.pid"
echo "已在后台启动  PID $(cat "$P3/exp45.pid")"
echo "  实时日志   tail -f $LOG"
echo "  每步状态   cat $P3/state.json"
echo "  分步日志   ls $P3/logs/"
echo "  停止       kill \$(cat $P3/exp45.pid)   （会先停 vLLM 再退出）"
