#!/bin/bash
# Official judge (native go-judge) with the official image's Ubuntu 22.04 userland: 3 reps x 3 solutions.
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
OUT=$PE_ROOT/eval/results/calib_rootfs_${SLURM_JOB_ID}; mkdir -p $OUT
PORT=$((20000 + SLURM_JOB_ID % 10000))
export JUDGE_ROOTFS=${JUDGE_ROOTFS:-$PE_SCRATCH/rootfs/frontiercs_judge_ubuntu2204}
echo "host=$(hostname) cpus=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))') port=$PORT rootfs=$JUDGE_ROOTFS date=$(date -Is)" | tee $OUT/node.txt
trap "$PE_ROOT/eval/start_judge_native.sh stop $PORT" EXIT
$PE_ROOT/eval/start_judge_native.sh start $PORT || exit 1
cp $PE_LOCAL_TMP/judge_$PORT/logs/gojudge.log $OUT/gojudge_start.log
for rep in 1 2 3; do
  for s in "ref:$PROBLEM_DIR/examples/reference.cpp" "gpt5:$PROBLEM_DIR/examples/gpt5.cpp" "baseline:$PE_ROOT/solutions/baseline_naive.cpp"; do
    name=${s%%:*}; f=${s#*:}
    $PE_PY $PE_ROOT/eval/official_judge_client.py --url http://127.0.0.1:$PORT $f --json $OUT/official_${name}_r${rep}.json | tee -a $OUT/official_summary.jsonl
  done
done
