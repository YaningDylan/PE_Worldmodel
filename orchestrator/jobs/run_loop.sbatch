#!/usr/bin/env bash
# Main-experiment job: one vLLM server + one shared judge + several (condition, seed) runs.
#
#   PE_RUNS="C0:0 C3:0 C0:1 C3:1"   specs hosted by this job (one seed of every condition per job: balanced)
#   PE_TAG=main27b                  run ids are {PE_TAG}_{cond}_s{seed}; the same id resumes from its checkpoint
#   PE_MODELS="Qwen/Qwen3.8-27B Qwen/Qwen3.6-27B"   the first that starts is used; a resumed run only
#                                   accepts the model it started with (orchestrator.run refuses otherwise)
#   PE_CONFIG=orchestrator/configs/model_qwen27b.yaml   model overlay (sampling per model card)
#   PE_SETS="run.budget_T=30"       extra --set overrides for every run
#   PE_BENCH=1                      also run the throughput benchmark (1/4/16 streams) before the runs
#
#   sbatch --export=ALL,PE_RUNS="C0:0 C3:0",PE_TAG=main27b orchestrator/jobs/run_loop.sbatch
# Chain 4-h chunks with --dependency=afterany:<previous job>; a chunk whose runs are all finished exits at once.
#SBATCH --job-name=pe_main
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
OUT="$PE_ROOT/logs/main_${SLURM_JOB_ID}"; mkdir -p "$OUT"
read -r -a SPECS <<< "${PE_RUNS:?set PE_RUNS}"
read -r -a SETS <<< "${PE_SETS:-run.budget_T=30}"
TAG=${PE_TAG:?set PE_TAG}
CONFIG=${PE_CONFIG:-orchestrator/configs/model_qwen27b.yaml}
step() { echo "=== [$(date +%T)] $*"; }
step "job=$SLURM_JOB_ID host=$(hostname) runs=${SPECS[*]} tag=$TAG"

# --- which runs still need work, and which model they are bound to -----------------------------------
read -r TODO PINNED < <("$PE_PY" - "$TAG" "${SPECS[@]}" <<'PY'
import json, sys, yaml
from pathlib import Path
tag, specs = sys.argv[1], sys.argv[2:]
todo, models = [], set()
for s in specs:
    cond, seed = s.split(":")
    rd = Path("runs") / f"{tag}_{cond}_s{seed}"
    st = json.loads((rd / "state.json").read_text()) if (rd / "state.json").exists() else None
    cfg = yaml.safe_load((rd / "config.yaml").read_text()) if (rd / "config.yaml").exists() else None
    T = int(cfg["run"]["budget_T"]) if cfg else None
    if st is None or T is None or st["step"] < T:
        todo.append(s)
    if cfg:
        models.add(cfg["llm"]["model"])
print(",".join(todo) or "-", ",".join(sorted(models)) or "-")
PY
)
if [ "$TODO" = "-" ]; then step "all runs finished; nothing to do"; exit 0; fi
IFS=, read -r -a TODO_SPECS <<< "$TODO"
MODELS=${PE_MODELS:-Qwen/Qwen3.8-27B Qwen/Qwen3.6-27B}
if [ "$PINNED" != "-" ]; then
  case "$PINNED" in *,*) echo "runs of this job disagree on the model: $PINNED"; exit 1;; esac
  MODELS=$PINNED
fi
step "todo=${TODO_SPECS[*]} models=$MODELS"

# --- vLLM (model ladder), then the shared judge on the remaining cores --------------------------------
OK=0
for m in $MODELS; do
  step "vLLM $m"
  if pe_start_vllm "$m"; then OK=1; break; fi
  echo "[job] model $m failed to start"
done
[ "$OK" = 1 ] || { echo "JOB FAIL: no model started"; exit 1; }
echo "vLLM model=$PE_VLLM_MODEL_NAME variant=$PE_VLLM_VARIANT ready=$(cat "$PE_VLLM_RUN_DIR/ready_seconds")s cpus=$PE_VLLM_CPUS"
pe_start_judge || { echo "JOB FAIL: judge did not start"; exit 1; }

step "judge sanity (warm-up, baseline, reference)"
taskset -c "$PE_VLLM_CPUS" "$PE_PY" - "$PE_JUDGE_URL" "$PE_JUDGE_TYPE" > "$OUT/judge_sanity.json" <<'PY'
import json, sys
sys.path.insert(0, ".")
from orchestrator.evaluator import JudgeEvaluator
ev = JudgeEvaluator(sys.argv[1], sys.argv[2])
p0 = "third_party/Frontier-CS/algorithmic/problems/0/examples/reference.cpp"
ev.evaluate(open("solutions/baseline_naive.cpp").read(), tag="warmup")
b = ev.evaluate(open("solutions/baseline_naive.cpp").read(), tag="baseline")
r = ev.evaluate(open(p0).read(), tag="reference")
ok = b["status"] == "ok" and abs(b["score"] - 45.99985793142856) < 1e-6 and r["status"] == "ok" and abs(r["score"] - 89.1) < 0.3
print(json.dumps({"ok": ok, "baseline": b.get("score"), "reference": r.get("score")}))
PY
cat "$OUT/judge_sanity.json"
grep -q '"ok": true' "$OUT/judge_sanity.json" || { echo "JOB FAIL: judge sanity check failed"; exit 1; }

if [ "${PE_BENCH:-0}" = 1 ]; then
  step "throughput benchmark ($PE_VLLM_MODEL_NAME; 2k prompt, 4096 gen, thinking on)"
  taskset -c "$PE_VLLM_CPUS" "$PE_PY" serve/bench_throughput.py --concurrency 1 4 16 --prompt-tokens 2048 \
    --gen-tokens 4096 --label "main_${PE_VLLM_RUNTIME}_TRITON_ATTN" --out "$OUT/bench.jsonl" 2>&1 | tail -4
fi

# --- runs (orchestrators are light HTTP clients: they share vLLM's cores, the judge keeps its own) ------
PIDS=()
for spec in "${TODO_SPECS[@]}"; do
  cond=${spec%%:*}; seed=${spec##*:}
  run_id="${TAG}_${cond}_s${seed}"
  args=(--condition "$cond" --seed "$seed" --run-id "$run_id" --config "$CONFIG"
        --set "llm.base_url=$PE_VLLM_BASE_URL" --set "llm.model=$PE_VLLM_MODEL_NAME"
        --set eval.backend=judge --set "eval.judge_url=$PE_JUDGE_URL" --set "eval.judge_type=$PE_JUDGE_TYPE")
  for s in "${SETS[@]}"; do args+=(--set "$s"); done
  mkdir -p "$PE_ROOT/runs/$run_id"
  echo "[job] start $run_id"
  taskset -c "$PE_VLLM_CPUS" "$PE_PY" -m orchestrator.run "${args[@]}" >> "$PE_ROOT/runs/$run_id/stdout.log" 2>&1 &
  PIDS+=($!)
done
status=0
for p in "${PIDS[@]}"; do wait "$p" || status=1; done
step "done status=$status"
exit $status
