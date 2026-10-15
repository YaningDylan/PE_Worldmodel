#!/bin/bash
# Calibration + noise on ONE cpu node (16 cores): official native judge vs local_eval.py.
#   sbatch --partition=cpu --cpus-per-task=16 --mem=32G --time=01:30:00 --account=ppliang_mit \
#          --job-name=pe_calib --output=$PE_ROOT/logs/pe_calib_%j.log eval/jobs/calibrate.sh
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
set -u
OUT=$PE_ROOT/eval/results/calib_${SLURM_JOB_ID}
mkdir -p $OUT
REPS_OFFICIAL=${REPS_OFFICIAL:-3}
REPS_LOCAL=${REPS_LOCAL:-5}
PORT=$((20000 + SLURM_JOB_ID % 10000))
SOLS=("ref:$PROBLEM_DIR/examples/reference.cpp" "gpt5:$PROBLEM_DIR/examples/gpt5.cpp" "baseline:$PE_ROOT/solutions/baseline_naive.cpp")
echo "host=$(hostname) cpus=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))') port=$PORT date=$(date -Is)" | tee $OUT/node.txt
lscpu | grep -E "Model name|^CPU\(s\)|MHz" >> $OUT/node.txt; g++ --version | head -1 >> $OUT/node.txt

# ---------- official judge (native go-judge + node server.js, pinned commit)
export JUDGE_DEBUG_LOG=${JUDGE_DEBUG_LOG:-1}   # log raw non-Accepted go-judge results (diagnostics only)
trap "$PE_ROOT/eval/start_judge_native.sh stop $PORT" EXIT
$PE_ROOT/eval/start_judge_native.sh start $PORT || exit 1
cp $PE_LOCAL_TMP/judge_$PORT/logs/gojudge.log $OUT/gojudge_start.log
for rep in $(seq 1 $REPS_OFFICIAL); do
  for s in "${SOLS[@]}"; do
    name=${s%%:*}; f=${s#*:}
    $PE_PY $PE_ROOT/eval/official_judge_client.py --url http://127.0.0.1:$PORT $f --json $OUT/official_${name}_r${rep}.json | tee -a $OUT/official_summary.jsonl
  done
done
# official Python API path (frontier SingleEvaluator) once per solution, for the record
for s in "${SOLS[@]}"; do
  name=${s%%:*}; f=${s#*:}
  $PE_PY - $f $PORT $name <<'PY' | tee -a $OUT/official_api.jsonl
import sys, time, json
from frontier_cs import SingleEvaluator
f, port, name = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
ev = SingleEvaluator(register_cleanup=False, judge_url=f"http://127.0.0.1:{port}")
r = ev.evaluate("algorithmic", problem_id=0, code=open(f).read())
md = r.metadata or {}
print(json.dumps({"name": name, "status": str(r.status), "score": r.score, "score_unbounded": r.score_unbounded,
                  "n_cases": len(md.get("cases") or []), "wall_s": round(time.time() - t0, 2)}))
PY
done
# frontier CLI once (reference), as documented in the plan
/usr/bin/env time -p true 2>/dev/null; t0=$(date +%s.%N)
$PE_ROOT/../envs/pe/bin/frontier eval algorithmic 0 $PROBLEM_DIR/examples/reference.cpp --judge-url http://127.0.0.1:$PORT --json > $OUT/frontier_cli_ref.json 2>$OUT/frontier_cli_ref.err
echo "frontier_cli_ref_wall=$(python3 -c "print(round($(date +%s.%N)-$t0,2))")" | tee -a $OUT/node.txt
cp $PE_LOCAL_TMP/judge_$PORT/logs/server.log $OUT/server.log
$PE_ROOT/eval/start_judge_native.sh stop $PORT
trap - EXIT

# ---------- local evaluator (same node, judge stopped)
for w in 16 8; do
  for rep in $(seq 1 $REPS_LOCAL); do
    for s in "${SOLS[@]}"; do
      name=${s%%:*}; f=${s#*:}
      $PE_PY $PE_ROOT/eval/local_eval.py $f --workers $w --json $OUT/local_w${w}_${name}_r${rep}.json | tee -a $OUT/local_summary.jsonl
    done
  done
done
echo "done $(date -Is)"
