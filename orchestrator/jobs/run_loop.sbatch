#!/usr/bin/env bash
# Experiment job: one vLLM server + one shared judge + several (condition, seed) runs of the search loop.
#
#   PE_RUNS="vanilla:0 pe_labeled:0"  specs hosted by this job; a spec may carry its own overrides after a second
#                                   colon: "pe_labeled:0:run.max_iters=5,loop.tau=2"
#   PE_TAG=oe27b                    run ids are {PE_TAG}_{cond}_s{seed}; the same id resumes from its checkpoint
#   PE_MODELS="Qwen/Qwen3.8-27B Qwen/Qwen3.6-27B"   the first that starts is used; a resumed run only
#                                   accepts the model it started with (orchestrator.run refuses otherwise)
#   PE_CONFIG=orchestrator/configs/model_qwen27b.yaml   model overlay (sampling per model card)
#   PE_SETS="run.full_evals=100"    extra --set overrides for every run
#   PE_BENCH=1                      also run the throughput benchmark (1/4/16 streams) before the runs
#
#   sbatch --export=ALL,PE_RUNS="vanilla:0 pe_labeled:0",PE_TAG=oe27b orchestrator/jobs/run_loop.sbatch
# Several queued jobs may host the same runs (e.g. a 4-h chain and a preemptable job): a run is written only by
# the job recorded in runs/{run_id}/OWNER while that job is alive; the others skip it (or exit if all are taken).
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
read -r -a SETS <<< "${PE_SETS:-run.full_evals=100}"
TAG=${PE_TAG:?set PE_TAG}
CONFIG=${PE_CONFIG:-orchestrator/configs/model_qwen27b.yaml}
step() { echo "=== [$(date +%T)] $*"; }
step "job=$SLURM_JOB_ID host=$(hostname) runs=${SPECS[*]} tag=$TAG"

# --- which runs still need work, and which model they are bound to -----------------------------------
read -r TODO PINNED < <("$PE_PY" - "$TAG" "${SPECS[@]}" <<'PY'
import json, sys, yaml
from pathlib import Path
tag, specs = sys.argv[1], sys.argv[2:]
import os, subprocess
todo, models = [], set()
me = os.environ.get("SLURM_JOB_ID")
for s in specs:
    cond, seed = s.split(":")[:2]
    rd = Path("runs") / f"{tag}_{cond}_s{seed}"
    st = json.loads((rd / "state.json").read_text()) if (rd / "state.json").exists() else None
    cfg = yaml.safe_load((rd / "config.yaml").read_text()) if (rd / "config.yaml").exists() else None
    owner = (rd / "OWNER").read_text().split()[0] if (rd / "OWNER").exists() else None
    busy = owner and owner != me and subprocess.run(["squeue", "-j", owner, "-h", "-o", "%T"], capture_output=True,
                                                    text=True).stdout.strip() in ("RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED")
    if (st is None or not st.get("done")) and not busy:
        todo.append(s)
    if cfg:
        models.add(cfg["llm"]["model"])
print(",".join(todo) or "-", ",".join(sorted(models)) or "-")
PY
)
if [ "$TODO" = "-" ]; then step "all runs finished or owned by another running job; nothing to do"; exit 0; fi
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
# orchestrators + their stage-1 checks (compile + 5 small cases) get their own cores, apart from vLLM and the judge
IFS=, read -r -a FREE <<< "$PE_FREE_CPUS"
NORCH=${PE_ORCH_NCPUS:-6}
ORCH_CPUS=$(IFS=,; echo "${FREE[*]:0:$NORCH}")
PE_FREE_CPUS=$(IFS=,; echo "${FREE[*]:$NORCH}"); export PE_FREE_CPUS
echo "cpus: vllm=$PE_VLLM_CPUS orchestrators=$ORCH_CPUS judge=$PE_FREE_CPUS"
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

# --- runs: no iteration starts within PE_DEADLINE_MARGIN_S of the time limit; an infrastructure failure
# (exit 4: iteration not committed) is retried in place after a pause ------------------------------------
LEFT=$(squeue -j "$SLURM_JOB_ID" -h -o %L)   # [d-]hh:mm:ss | mm:ss
LEFT_S=$("$PE_PY" -c "import sys; s=sys.argv[1]; d,_,s=s.rpartition('-'); p=[int(x) for x in s.split(':')]; p=[0]*(3-len(p))+p; print((int(d or 0)*24+p[0])*3600+p[1]*60+p[2])" "$LEFT")
export PE_DEADLINE_TS=$(( $(date +%s) + LEFT_S - ${PE_DEADLINE_MARGIN_S:-1500} ))
echo "time left ${LEFT} -> no new iteration after $(date -d @"$PE_DEADLINE_TS" +%T)"
run_one() {  # run_id args...
  local id=$1; shift
  local rc=4 tries=0
  while [ "$rc" = 4 ] && [ "$tries" -lt "${PE_INFRA_RETRIES:-5}" ] && [ "$(date +%s)" -lt "$PE_DEADLINE_TS" ]; do
    [ "$tries" -gt 0 ] && { echo "[job] $id infra failure, retry $tries in 60 s"; sleep 60; }
    taskset -c "$ORCH_CPUS" "$PE_PY" -m orchestrator.run "$@" >> "$PE_ROOT/runs/$id/stdout.log" 2>&1
    rc=$?; tries=$((tries + 1))
  done
  echo "[job] $id exit=$rc"
  return "$rc"
}
PIDS=()
for spec in "${TODO_SPECS[@]}"; do
  IFS=: read -r cond seed extra <<< "$spec"
  run_id="${TAG}_${cond}_s${seed}"
  args=(--condition "$cond" --seed "$seed" --run-id "$run_id" --config "$CONFIG"
        --set "llm.base_url=$PE_VLLM_BASE_URL" --set "llm.model=$PE_VLLM_MODEL_NAME"
        --set eval.backend=judge --set "eval.judge_url=$PE_JUDGE_URL" --set "eval.judge_type=$PE_JUDGE_TYPE")
  for s in "${SETS[@]}"; do args+=(--set "$s"); done
  if [ -n "${extra:-}" ]; then IFS=, read -r -a EX <<< "$extra"; for s in "${EX[@]}"; do args+=(--set "$s"); done; fi
  mkdir -p "$PE_ROOT/runs/$run_id"
  echo "[job] start $run_id"
  run_one "$run_id" "${args[@]}" &
  PIDS+=($!)
done
status=0
for p in "${PIDS[@]}"; do wait "$p"; rc=$?; [ "$rc" = 0 ] || [ "$rc" = 5 ] || [ "$rc" = 6 ] || status=1; done
step "done status=$status"
exit $status
