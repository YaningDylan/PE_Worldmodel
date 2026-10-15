#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server in the background INSIDE a Slurm job, wait until it is
# ready, and print the base URL (the only line written to stdout; everything else goes to stderr).
#
#   BASE_URL=$($PE_ROOT/serve/launch_vllm.sh --model Qwen/Qwen3.5-9B) || exit 1
#   ... use $BASE_URL (e.g. http://127.0.0.1:8000/v1) ...
#   $PE_ROOT/serve/stop_vllm.sh                # stops the server started from this job
#
# Every option can also be given as an env var (flag wins). Defaults in brackets.
#   --model ID|PATH          PE_VLLM_MODEL        [Qwen/Qwen3.5-9B]  HF repo id (resolved to the local
#                                                 snapshot in $HF_HOME, served offline) or a local dir
#   --served-name NAME       PE_VLLM_SERVED_NAME  [repo id, or basename of the path]
#   --port N                 PE_VLLM_PORT         [8000]  if busy, the next free port is used
#   --host H                 PE_VLLM_HOST         [127.0.0.1]
#   --max-model-len N        PE_VLLM_MAX_MODEL_LEN [131072]
#   --gpu-mem-util F         PE_VLLM_GPU_MEM_UTIL [0.90]
#   --tp N                   PE_VLLM_TP           [1]
#   --attn-backend B         PE_VLLM_ATTN_BACKEND [$VLLM_ATTENTION_BACKEND or TRITON_ATTN]; "auto" = let
#                                                 vLLM choose. (vLLM 0.22.1 ignores the
#                                                 VLLM_ATTENTION_BACKEND env var; only the flag works.)
#   --ncpus N                PE_VLLM_NCPUS        [4]  cores (taken from this job's allowed set) vLLM is
#                                                 pinned to; the rest are exported as PE_FREE_CPUS
#   --mtp N                  PE_VLLM_MTP          [0]  >0 enables MTP speculative decoding with N tokens
#   --max-num-seqs N         PE_VLLM_MAX_NUM_SEQS [vLLM default]
#   --enforce-eager          PE_VLLM_ENFORCE_EAGER=1  [off]
#   --no-language-model-only PE_VLLM_LM_ONLY=0    [on: skip the vision encoder, text only]
#   --runtime R              PE_VLLM_RUNTIME      [apptainer if $PE_VLLM_SIF exists, else conda]
#                                                 apptainer: official vllm/vllm-openai image (SIF) with --nv;
#                                                 conda: $VLLM_ENV (ocrvl). PE_VLLM_SIF defaults to
#                                                 $PE_SCRATCH/containers/vllm-openai_v0.22.1.sif
#   --run-dir DIR            PE_VLLM_RUN_DIR      [$PE_ROOT/logs/vllm/<name>_j<jobid>_<time>]
#   --timeout SEC            PE_VLLM_READY_TIMEOUT [1800]
#   -- ARGS...               extra args passed verbatim to `vllm serve` (also PE_VLLM_EXTRA_ARGS)
#
# Run dir contents: vllm.log, vllm.pid (process-group leader), cmd.txt, server.env (sourceable:
# PE_VLLM_BASE_URL, PE_VLLM_MODEL_NAME, PE_VLLM_PID, PE_VLLM_RUN_DIR, PE_VLLM_CPUS, PE_FREE_CPUS),
# ready_seconds. A pointer to the latest run dir is kept in $PE_SCRATCH/vllm_current_<jobid>.
set -euo pipefail

source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
VLLM_ENV=${VLLM_ENV:-/home/dvdai_mit/miniconda3/envs/ocrvl}

MODEL=${PE_VLLM_MODEL:-Qwen/Qwen3.5-9B}
SERVED_NAME=${PE_VLLM_SERVED_NAME:-}
PORT=${PE_VLLM_PORT:-8000}
HOST=${PE_VLLM_HOST:-127.0.0.1}
MAX_LEN=${PE_VLLM_MAX_MODEL_LEN:-131072}
GPU_UTIL=${PE_VLLM_GPU_MEM_UTIL:-0.90}
TP=${PE_VLLM_TP:-1}
ATTN=${PE_VLLM_ATTN_BACKEND:-${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}}
NCPUS=${PE_VLLM_NCPUS:-4}
MTP=${PE_VLLM_MTP:-0}
MAX_NUM_SEQS=${PE_VLLM_MAX_NUM_SEQS:-}
ENFORCE_EAGER=${PE_VLLM_ENFORCE_EAGER:-0}
LM_ONLY=${PE_VLLM_LM_ONLY:-1}
RUN_DIR=${PE_VLLM_RUN_DIR:-}
TIMEOUT=${PE_VLLM_READY_TIMEOUT:-1800}
SIF=${PE_VLLM_SIF:-$PE_SCRATCH/containers/vllm-openai_v0.22.1.sif}
RUNTIME=${PE_VLLM_RUNTIME:-}
read -r -a EXTRA <<< "${PE_VLLM_EXTRA_ARGS:-}"

die() { echo "[launch_vllm] ERROR: $*" >&2; exit 1; }
log() { echo "[launch_vllm] $*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2;;
    --served-name|--served-model-name) SERVED_NAME=$2; shift 2;;
    --port) PORT=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --max-model-len) MAX_LEN=$2; shift 2;;
    --gpu-mem-util|--gpu-memory-utilization) GPU_UTIL=$2; shift 2;;
    --tp|--tensor-parallel-size) TP=$2; shift 2;;
    --attn-backend|--attention-backend) ATTN=$2; shift 2;;
    --ncpus) NCPUS=$2; shift 2;;
    --mtp) MTP=$2; shift 2;;
    --max-num-seqs) MAX_NUM_SEQS=$2; shift 2;;
    --enforce-eager) ENFORCE_EAGER=1; shift;;
    --no-language-model-only) LM_ONLY=0; shift;;
    --runtime) RUNTIME=$2; shift 2;;
    --run-dir) RUN_DIR=$2; shift 2;;
    --timeout) TIMEOUT=$2; shift 2;;
    --) shift; EXTRA+=("$@"); break;;
    -h|--help) sed -n '2,32p' "$0" >&2; exit 0;;
    *) die "unknown option $1 (use -- to pass raw vllm args)";;
  esac
done

[ -n "$RUNTIME" ] || { [ -s "$SIF" ] && RUNTIME=apptainer || RUNTIME=conda; }
case "$RUNTIME" in
  apptainer) [ -s "$SIF" ] || die "SIF not found: $SIF (sbatch serve/scripts/pull_vllm_image.sbatch)";;
  conda) [ -x "$VLLM_ENV/bin/vllm" ] || die "vllm not found in $VLLM_ENV";;
  *) die "unknown runtime $RUNTIME (apptainer|conda)";;
esac
command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || die "no GPU visible (run this inside a GPU Slurm job)"

# ---- resolve model: HF repo id -> local snapshot (serve offline); local dir -> as is ----
if [ -d "$MODEL" ]; then
  MODEL_PATH=$(readlink -f "$MODEL")
  [ -n "$SERVED_NAME" ] || SERVED_NAME=$(basename "$MODEL_PATH")
else
  REPO_DIR="$HF_HOME/hub/models--${MODEL//\//--}"
  [ -f "$REPO_DIR/refs/main" ] || die "repo $MODEL not in HF cache ($REPO_DIR); run serve/scripts/download_models.sh $MODEL on the login node"
  MODEL_PATH="$REPO_DIR/snapshots/$(cat "$REPO_DIR/refs/main")"
  [ -f "$MODEL_PATH/config.json" ] && compgen -G "$MODEL_PATH/*.safetensors" >/dev/null \
    || die "incomplete snapshot $MODEL_PATH (verify with serve/scripts/verify_download.py $MODEL)"
  [ -n "$SERVED_NAME" ] || SERVED_NAME=$MODEL
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ---- CPU pinning: first NCPUS cores of this job's allowed set ----
read -r VLLM_CPUS FREE_CPUS < <("$PE_PY" - "$NCPUS" <<'EOF'
import os, sys
cpus = sorted(os.sched_getaffinity(0)); n = int(sys.argv[1])
if n <= 0 or n >= len(cpus):
    print(",".join(map(str, cpus)), "-")
else:
    print(",".join(map(str, cpus[:n])), ",".join(map(str, cpus[n:])))
EOF
)
[ "$FREE_CPUS" = "-" ] && FREE_CPUS=""
NPIN=$(awk -F, '{print NF}' <<< "$VLLM_CPUS")

# ---- port: first free one at or above $PORT (other jobs may share the node) ----
PORT=$("$PE_PY" - "$PORT" "$HOST" <<'EOF'
import socket, sys
p, host = int(sys.argv[1]), sys.argv[2]
for q in range(p, p + 200):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, q)); s.close(); print(q); break
    except OSError:
        s.close()
else:
    sys.exit("no free port")
EOF
)

TAG=$(basename "$SERVED_NAME" | tr -c 'A-Za-z0-9._-' '_')
JOB=${SLURM_JOB_ID:-nojob}
[ -n "$RUN_DIR" ] || RUN_DIR="$PE_ROOT/logs/vllm/${TAG}_j${JOB}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
RUN_DIR=$(readlink -f "$RUN_DIR")

ARGS=(serve "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --reasoning-parser qwen3
  --disable-uvicorn-access-log)
[ "$LM_ONLY" = 1 ] && ARGS+=(--language-model-only)
[ "$ATTN" != auto ] && ARGS+=(--attention-backend "$ATTN")
[ "$ENFORCE_EAGER" = 1 ] && ARGS+=(--enforce-eager)
# FlashInfer autotune (default on for SM >= 9.0) is pointless unless FlashInfer serves attention, and its
# JIT paths need nvcc, which the nodes lack. PE_VLLM_FI_AUTOTUNE=1 keeps vLLM's default.
[ "$ATTN" != FLASHINFER ] && [ "${PE_VLLM_FI_AUTOTUNE:-0}" != 1 ] && ARGS+=(--no-enable-flashinfer-autotune)
[ -n "$MAX_NUM_SEQS" ] && ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
[ "$MTP" -gt 0 ] 2>/dev/null && ARGS+=(--speculative-config "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$MTP}")
ARGS+=("${EXTRA[@]}")

# vLLM 0.22.1 does not read VLLM_ATTENTION_BACKEND (it warns "Unknown vLLM environment variable");
# the backend is passed via --attention-backend above instead.
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export OMP_NUM_THREADS=$NPIN MKL_NUM_THREADS=$NPIN TORCHINDUCTOR_COMPILE_THREADS=$NPIN
export TOKENIZERS_PARALLELISM=false
export PATH="$VLLM_ENV/bin:$PATH"

if [ "$RUNTIME" = apptainer ]; then
  # Same pattern as the group's working B200 stacks: the server lives in an Apptainer image; every cache
  # is redirected to a writable dir on /work because the image's package tree is read-only.
  AC="$PE_SCRATCH/apptainer_cache"
  mkdir -p "$PE_SCRATCH/apptainer_home" "$AC"/{xdg,triton,vllm,inductor,flashinfer}
  unset APPTAINER_BIND
  VLLM_CMD=(apptainer exec --nv --cleanenv --home "$PE_SCRATCH/apptainer_home" -B /work
    --env "HF_HOME=$HF_HOME" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
    --env "XDG_CACHE_HOME=$AC/xdg" --env "TRITON_CACHE_DIR=$AC/triton" --env "VLLM_CACHE_ROOT=$AC/vllm"
    --env "TORCHINDUCTOR_CACHE_DIR=$AC/inductor" --env "FLASHINFER_WORKSPACE_BASE=$AC/flashinfer"
    --env "VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    --env "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-0}" --env "VLLM_MOE_USE_DEEP_GEMM=${VLLM_MOE_USE_DEEP_GEMM:-0}"
    --env VLLM_NO_USAGE_STATS=1 --env TOKENIZERS_PARALLELISM=false
    --env "OMP_NUM_THREADS=$NPIN" --env "MKL_NUM_THREADS=$NPIN" --env "TORCHINDUCTOR_COMPILE_THREADS=$NPIN")
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && VLLM_CMD+=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
  VLLM_CMD+=("$SIF" vllm)
else
  VLLM_CMD=("$VLLM_ENV/bin/vllm")
fi

{
  echo "# $(date -Is) host=$(hostname) job=$JOB gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  echo "# runtime=$RUNTIME cpus(vllm)=$VLLM_CPUS free=$FREE_CPUS"
  printf 'taskset -c %q' "$VLLM_CPUS"; printf ' %q' "${VLLM_CMD[@]}" "${ARGS[@]}"; echo
} > "$RUN_DIR/cmd.txt"

T0=$(date +%s.%N)
# setsid: own process group so stop_vllm.sh can kill API server + engine core together.
rm -f "$RUN_DIR/vllm.pid"
setsid bash -c 'echo $$ > "$0/vllm.pid"; exec "$@"' "$RUN_DIR" \
  taskset -c "$VLLM_CPUS" "${VLLM_CMD[@]}" "${ARGS[@]}" > "$RUN_DIR/vllm.log" 2>&1 < /dev/null &
for _ in $(seq 50); do [ -s "$RUN_DIR/vllm.pid" ] && break; sleep 0.1; done
PID=$(cat "$RUN_DIR/vllm.pid" 2>/dev/null) || die "vLLM process did not start"
BASE_URL="http://$HOST:$PORT/v1"
cat > "$RUN_DIR/server.env" <<EOF
export PE_VLLM_BASE_URL=$BASE_URL
export PE_VLLM_MODEL_NAME=$SERVED_NAME
export PE_VLLM_PID=$PID
export PE_VLLM_RUN_DIR=$RUN_DIR
export PE_VLLM_RUNTIME=$RUNTIME
export PE_VLLM_CPUS=$VLLM_CPUS
export PE_FREE_CPUS=$FREE_CPUS
EOF
mkdir -p "$PE_SCRATCH"; echo "$RUN_DIR" > "$PE_SCRATCH/vllm_current_${JOB}"
log "model=$SERVED_NAME path=$MODEL_PATH"
log "runtime=$RUNTIME pid=$PID port=$PORT attn=$ATTN cpus=$VLLM_CPUS run_dir=$RUN_DIR"

# ---- wait for readiness ----
while :; do
  if ! kill -0 "$PID" 2>/dev/null; then
    log "vLLM exited before becoming ready. Root-cause lines, then the last 60 log lines ($RUN_DIR/vllm.log):"
    grep -aE "EngineCore.*(RuntimeError|Error:|error:)|unrecognized arguments|usage:" "$RUN_DIR/vllm.log" \
      | sort -u | head -8 | cut -c1-400 >&2
    tail -n 60 "$RUN_DIR/vllm.log" >&2
    exit 1
  fi
  if curl -sf -m 5 "$BASE_URL/models" 2>/dev/null | grep -q "\"$SERVED_NAME\""; then
    break
  fi
  ELAPSED=$(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%d", b-a}')
  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    log "not ready after ${TIMEOUT}s; stopping it. Last 60 log lines:"
    tail -n 60 "$RUN_DIR/vllm.log" >&2
    kill -TERM -- "-$PID" 2>/dev/null || true
    exit 1
  fi
  sleep 3
done
READY=$(awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
echo "$READY" > "$RUN_DIR/ready_seconds"
log "ready after ${READY}s at $BASE_URL (stop: $PE_ROOT/serve/stop_vllm.sh $RUN_DIR)"
echo "$BASE_URL"
