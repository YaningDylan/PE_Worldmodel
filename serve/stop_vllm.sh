#!/usr/bin/env bash
# Stop a vLLM server started by serve/launch_vllm.sh (kills its whole process group).
#   stop_vllm.sh [RUN_DIR]     default: the latest server launched in this Slurm job
#                              ($PE_SCRATCH/vllm_current_$SLURM_JOB_ID)
# Can also be sourced to get a `pe_stop_vllm [RUN_DIR]` function.
pe_stop_vllm() {
  local scratch=/work/mit/ppliang_mit/yaning/workspace/pe_scratch
  local run_dir=${1:-$(cat "$scratch/vllm_current_${SLURM_JOB_ID:-nojob}" 2>/dev/null)}
  [ -n "$run_dir" ] && [ -f "$run_dir/vllm.pid" ] || { echo "[stop_vllm] no run dir / pid file (${run_dir:-unset})" >&2; return 1; }
  local pid; pid=$(cat "$run_dir/vllm.pid")
  if ! kill -0 "$pid" 2>/dev/null; then echo "[stop_vllm] pid $pid already gone" >&2; return 0; fi
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"
  local i
  for i in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop_vllm] pid $pid still alive after 30s, SIGKILL" >&2
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid"
  fi
  # engine-core children live in the same process group; make sure none survive
  pkill -KILL -g "$pid" 2>/dev/null || true
  echo "[stop_vllm] stopped pid $pid ($run_dir)" >&2
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then pe_stop_vllm "$@"; fi
