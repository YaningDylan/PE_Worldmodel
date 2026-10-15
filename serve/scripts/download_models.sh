#!/usr/bin/env bash
# Download the serving checkpoints into $HF_HOME (login node; network-bound).
# Usage: bash serve/scripts/download_models.sh [repo_id ...]   (default: Qwen3.5-9B then Qwen3.6-27B)
set -uo pipefail
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0   # no duplicate chunk cache on /work
HF=/work/mit/ppliang_mit/yaning/workspace/envs/pe/bin/hf
repos=("$@"); [ ${#repos[@]} -eq 0 ] && repos=(Qwen/Qwen3.5-9B Qwen/Qwen3.6-27B)
for r in "${repos[@]}"; do
  echo "[$(date -Is)] START $r"
  t0=$(date +%s)
  # NB: hf 1.16.1 sometimes prints a spurious click Exit(0) traceback and returns non-zero after a
  # successful download, so completeness is checked separately against the Hub file sizes.
  "$HF" download "$r" --exclude "*.gguf" --exclude "original/*" --max-workers 8
  echo "[$(date -Is)] hf exit=$? for $r after $(( $(date +%s) - t0 ))s"
  "$PE_PY" "$PE_ROOT/serve/scripts/verify_download.py" "$r"
done
echo "[$(date -Is)] ALL DONE"
