#!/bin/bash
# Noise measurement (plan §3.3) on one cpu node with 16 allocated cores.
#   sbatch --partition=cpu --cpus-per-task=16 --mem=32G --time=01:00:00 --account=ppliang_mit \
#          --job-name=pe_noise --output=$PE_ROOT/logs/pe_noise_%j.log eval/jobs/noise.sh
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
OUT=$PE_ROOT/eval/results/noise_${SLURM_JOB_ID}; mkdir -p $OUT
REPS=${REPS:-5}
echo "host=$(hostname) cpus=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))') date=$(date -Is)" | tee $OUT/node.txt
SOLS=("ref:$PROBLEM_DIR/examples/reference.cpp" "gpt5:$PROBLEM_DIR/examples/gpt5.cpp" "baseline:$PE_ROOT/solutions/baseline_naive.cpp")
run() { # tag workers file name rep
  $PE_PY $PE_ROOT/eval/local_eval.py $3 --workers $2 --json $OUT/$1_$4_r$5.json | cut -c1-300 | sed "s/^/[$1 $4 r$5] /" | tee -a $OUT/summary.txt
}
# 1) clean condition, 16 workers on 16 cores; and 8 workers
for rep in $(seq 1 $REPS); do for s in "${SOLS[@]}"; do run w16 16 ${s#*:} ${s%%:*} $rep; done; done
for rep in $(seq 1 $REPS); do run w8 8 $PROBLEM_DIR/examples/reference.cpp ref $rep; done
# 2) oversubscribed: 32 workers on 16 cores
for rep in $(seq 1 $REPS); do run w32 32 $PROBLEM_DIR/examples/reference.cpp ref $rep; done
# 3) 16 workers + 8 busy-loop processes in the same allocation (stand-in for a CPU-hungry vLLM)
for i in $(seq 1 8); do (timeout 600 bash -c 'while :; do :; done') & done
for rep in $(seq 1 $REPS); do run w16hog8 16 $PROBLEM_DIR/examples/reference.cpp ref $rep; done
for rep in $(seq 1 $REPS); do run w8hog8 8 $PROBLEM_DIR/examples/reference.cpp ref $rep; done
pkill -P $$ -f "while :; do :; done" 2>/dev/null; kill $(jobs -p) 2>/dev/null
echo "done $(date -Is)"
