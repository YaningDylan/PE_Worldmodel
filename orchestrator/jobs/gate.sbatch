#!/usr/bin/env bash
# Gate 0b + Gate 1 in one B200 job (avoids several queue waits):
#   1. raw Docker / execution-model checks on the GPU node, failure classification
#   2. vLLM (Qwen3.5-9B) on 4 cores; thinking on/off check
#   3. one shared judge on the remaining cores: official Docker image if usable, else native copy
#   4. judge validation: per-case reference / gpt5 / baseline (+ pre-#120 gpt5), judge vs local_eval
#   5. score noise with K concurrent submitters while vLLM is under load -> sigma, tau
#   6. Gate 1: if 2-4 pass, C3 smoke run (5 iterations) through the shared judge with that tau
# Everything lands in logs/gate_<jobid>/.
#SBATCH --job-name=pe_gate
#SBATCH --partition=b200-devel,b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=/work/mit/ppliang_mit/yaning/workspace/potential_estimation/logs/%x-%j.out
set -uo pipefail
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
source "$PE_ROOT/orchestrator/jobs/lib.sh"
cd "$PE_ROOT"
OUT="$PE_ROOT/logs/gate_${SLURM_JOB_ID}"; mkdir -p "$OUT"
MODEL=${PE_MODEL:-Qwen/Qwen3.5-9B}
K=${PE_NOISE_K:-5}
step() { echo "=== [$(date +%T)] $*"; }

step "1. Docker / execution-model checks"
pe_docker_checks "$OUT/docker_checks.txt"; cat "$OUT/docker_checks.txt"

step "2. vLLM $MODEL"
if ! pe_start_vllm "$MODEL"; then echo "GATE FAIL: vLLM did not start"; exit 1; fi
echo "vLLM variant=$PE_VLLM_VARIANT $PE_VLLM_BASE_URL run_dir=$PE_VLLM_RUN_DIR vllm_cpus=$PE_VLLM_CPUS free=$PE_FREE_CPUS ready=$(cat "$PE_VLLM_RUN_DIR/ready_seconds")s"
VLLM_OK=1
# max_tokens as in the run config (32768): with thinking on, Qwen3.5-9B spends 5-8k tokens reasoning on
# this toy prompt, so a small cap ends mid-thought with empty content (job 1323967 failed that way at 2048).
for mode in "" "--no-thinking"; do
  taskset -c "$PE_VLLM_CPUS" "$PE_PY" serve/client_check.py --max-tokens 32768 $mode --json \
    > "$OUT/client_check${mode:+_nothink}.txt" 2>&1 || VLLM_OK=0
  tail -3 "$OUT/client_check${mode:+_nothink}.txt"
done

step "3. shared judge"
if ! pe_start_judge; then echo "GATE FAIL: judge did not start"; exit 1; fi

step "4. judge validation (per case)"
taskset -c "$PE_FREE_CPUS" "$PE_PY" orchestrator/jobs/judge_validate.py --url "$PE_JUDGE_URL" \
  --judge-type "$PE_JUDGE_TYPE" --out "$OUT/validate"
JUDGE_OK=$([ $? -eq 0 ] && echo 1 || echo 0)

step "5. noise with K=$K concurrent submitters while vLLM is busy"
taskset -c "$PE_VLLM_CPUS" "$PE_PY" serve/bench_throughput.py --concurrency 8 --waves 400 --prompt-tokens 2048 \
  --gen-tokens 2048 --label noise_background --out "$OUT/background_load.jsonl" > "$OUT/background_load.log" 2>&1 &
LOAD_PID=$!
taskset -c "$PE_FREE_CPUS" "$PE_PY" orchestrator/jobs/noise_concurrent.py --url "$PE_JUDGE_URL" \
  --judge-type "$PE_JUDGE_TYPE" --concurrency "$K" --reps 4 --out "$OUT/noise.json"
kill "$LOAD_PID" 2>/dev/null; wait "$LOAD_PID" 2>/dev/null
TAU=$("$PE_PY" -c "import json;print(round(json.load(open('$OUT/noise.json'))['tau'],4))" 2>/dev/null || echo 1.0)
echo "tau=$TAU"

echo "{\"docker_ok\": $PE_DOCKER_OK, \"docker_fail\": \"${PE_DOCKER_FAIL:-}\", \"vllm_ok\": $VLLM_OK, \"vllm_variant\": \"$PE_VLLM_VARIANT\", \"judge_ok\": $JUDGE_OK, \"judge\": \"$PE_JUDGE_ID\", \"tau\": $TAU}" > "$OUT/gate.json"
cat "$OUT/gate.json"
if [ "$VLLM_OK" != 1 ] || [ "$JUDGE_OK" != 1 ]; then
  echo "GATE FAIL: not starting the Gate 1 smoke run (vllm_ok=$VLLM_OK judge_ok=$JUDGE_OK)"; exit 1
fi

step "6. Gate 1 smoke: C3, 5 iterations, judge=$PE_JUDGE_TYPE, tau=$TAU"
RUN_ID=${PE_SMOKE_RUN_ID:-gate1_smoke_C3_s0}
mkdir -p "$PE_ROOT/runs/$RUN_ID"
"$PE_PY" -m orchestrator.run --condition C3 --seed 0 --run-id "$RUN_ID" \
  --set run.budget_T=5 --set "loop.tau=$TAU" \
  --set "llm.base_url=$PE_VLLM_BASE_URL" --set "llm.model=$PE_VLLM_MODEL_NAME" \
  --set eval.backend=judge --set "eval.judge_url=$PE_JUDGE_URL" --set "eval.judge_type=$PE_JUDGE_TYPE" \
  2>&1 | tee -a "$PE_ROOT/runs/$RUN_ID/stdout.log"
step "done (exit ${PIPESTATUS[0]})"
