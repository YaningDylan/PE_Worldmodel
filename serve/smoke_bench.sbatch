#!/usr/bin/env bash
# Single-B200 smoke test + throughput benchmark for one model.
#   sbatch --export=ALL,PE_SMOKE_MODEL=Qwen/Qwen3.5-9B  serve/smoke_bench.sbatch
#   sbatch --export=ALL,PE_SMOKE_MODEL=Qwen/Qwen3.6-27B serve/smoke_bench.sbatch
# Optional: PE_SMOKE_EXTRAS="flashinfer mtp" (extra configs to benchmark after the main one),
#           PE_SMOKE_BACKENDS="TRITON_ATTN TRITON_ATTN:eager FLASHINFER" (fallback order for the main run).
#SBATCH --job-name=pe_vllm_smoke
#SBATCH --partition=b200-devel,b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:30:00
#SBATCH --account=ppliang_mit
#SBATCH --output=/work/mit/ppliang_mit/yaning/workspace/potential_estimation/logs/%x-%j.out
set -uo pipefail
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
MODEL=${PE_SMOKE_MODEL:-Qwen/Qwen3.5-9B}
EXTRAS=${PE_SMOKE_EXTRAS:-flashinfer mtp}
BACKENDS=${PE_SMOKE_BACKENDS:-TRITON_ATTN TRITON_ATTN:eager FLASHINFER}
TAG=$(basename "$MODEL")
OUT=$PE_ROOT/serve/smoke_results/${TAG}_j${SLURM_JOB_ID}
mkdir -p "$OUT"
L=$PE_ROOT/serve/launch_vllm.sh
S=$PE_ROOT/serve/stop_vllm.sh
BENCH="$PE_PY $PE_ROOT/serve/bench_throughput.py"
CHECK="$PE_PY $PE_ROOT/serve/client_check.py"
ts() { date +%H:%M:%S; }
gpumem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | head -1; }
summ() {  # key lines from a vLLM log
  grep -aE "Using .*backend|attention backend|Model loading took|KV cache|Maximum concurrency|prefix caching|torch.compile|CUDA graph|Graph capturing|WARNING|Error|error" "$1" \
    | grep -v "Unknown vLLM environment" | sed -e 's/^/    /' | cut -c1-300 | head -60
}

echo "== $(ts) host=$(hostname) job=$SLURM_JOB_ID model=$MODEL"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo "affinity: $(taskset -pc $$)"; echo "nproc=$(nproc)"
"$VLLM_ENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability())"

# ---------- main run (cold compile cache for this model), with backend fallback ----------
# Runtime ladder first (Apptainer image, then the conda env), attention backends inside each.
BASE_URL=""; MAIN_BACKEND=""; MAIN_DIR=""
for rt in ${PE_SMOKE_RUNTIMES:-apptainer conda}; do
  for b in $BACKENDS; do
    be=${b%%:*}; extra=(); [ "$b" != "$be" ] && extra=(--enforce-eager)
    echo "== $(ts) launching main (runtime=$rt backend=$b)"
    if BASE_URL=$($L --model "$MODEL" --runtime "$rt" --attn-backend "$be" "${extra[@]}" --run-dir "$OUT/main_${rt}_$b" --timeout 2400); then
      MAIN_BACKEND=$b; MAIN_DIR="$OUT/main_${rt}_$b"; break 2
    fi
    echo "!! runtime $rt backend $b failed"; summ "$OUT/main_${rt}_$b/vllm.log"
  done
done
[ -n "$BASE_URL" ] || { echo "!! all runtimes/backends failed"; exit 1; }
source "$MAIN_DIR/server.env"   # also exports PE_VLLM_RUNTIME, so the warm restart and extras reuse it
echo "== $(ts) READY runtime=$PE_VLLM_RUNTIME backend=$MAIN_BACKEND ready_s=$(cat "$PE_VLLM_RUN_DIR/ready_seconds") gpu_mem=$(gpumem)"
echo "-- log summary"; summ "$PE_VLLM_RUN_DIR/vllm.log"

echo "== $(ts) client_check thinking ON"
$CHECK --max-tokens 16384 --json
echo "== $(ts) client_check thinking OFF"
$CHECK --max-tokens 2048 --no-thinking --json

echo "== $(ts) bench main (2k prompt, 4096 gen, thinking on)"
$BENCH --concurrency 1 4 16 --prompt-tokens 2048 --gen-tokens 4096 --label "main_${PE_VLLM_RUNTIME}_$MAIN_BACKEND" --out "$OUT/bench.jsonl"
echo "== $(ts) long-context check (100k prompt, 256 gen)"
$BENCH --concurrency 1 --prompt-tokens 100000 --gen-tokens 256 --label "long100k_${PE_VLLM_RUNTIME}_$MAIN_BACKEND" --out "$OUT/bench.jsonl"
echo "gpu_mem after bench: $(gpumem)"
$S "$PE_VLLM_RUN_DIR"; sleep 5

# ---------- warm restart (compile caches populated, weights likely in page cache) ----------
echo "== $(ts) warm relaunch ($MAIN_BACKEND)"
be=${MAIN_BACKEND%%:*}; extra=(); [ "$MAIN_BACKEND" != "$be" ] && extra=(--enforce-eager)
if BASE_URL=$($L --model "$MODEL" --attn-backend "$be" "${extra[@]}" --run-dir "$OUT/warm_$MAIN_BACKEND"); then
  echo "== $(ts) WARM READY ready_s=$(cat "$OUT/warm_$MAIN_BACKEND/ready_seconds")"
  $S "$OUT/warm_$MAIN_BACKEND"; sleep 5
fi

# ---------- extras ----------
for x in $EXTRAS; do
  case $x in
    flashinfer) args=(--attn-backend FLASHINFER);;
    mtp) args=(--attn-backend "$be" --mtp 2);;
    *) echo "unknown extra $x"; continue;;
  esac
  echo "== $(ts) extra $x: launching"
  if BASE_URL=$($L --model "$MODEL" "${args[@]}" --run-dir "$OUT/extra_$x" --timeout 2400); then
    source "$OUT/extra_$x/server.env"
    echo "== $(ts) extra $x READY ready_s=$(cat "$OUT/extra_$x/ready_seconds") gpu_mem=$(gpumem)"
    summ "$OUT/extra_$x/vllm.log" | head -25
    $CHECK --max-tokens 16384 --json
    $BENCH --concurrency 1 4 16 --prompt-tokens 2048 --gen-tokens 4096 --label "$x" --out "$OUT/bench.jsonl"
    $S "$OUT/extra_$x"; sleep 5
  else
    echo "!! extra $x failed to start"; summ "$OUT/extra_$x/vllm.log"
  fi
done
echo "== $(ts) DONE"
