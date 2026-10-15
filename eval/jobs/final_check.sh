#!/bin/bash
# Final smoke check of the delivered scripts on a cpu node.
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
cd $PE_ROOT
$PE_ROOT/eval/jobs/test_native_judge.sh 2>&1 | grep -v "^{\"level\"" | cut -c1-250
PORT=$((20000 + SLURM_JOB_ID % 10000))
$PE_ROOT/eval/start_judge_native.sh start $PORT && $PE_ROOT/eval/start_judge_native.sh status $PORT
$PE_PY eval/official_judge_client.py --url http://127.0.0.1:$PORT solutions/baseline_naive.cpp
$PE_ROOT/eval/start_judge_native.sh stop $PORT; $PE_ROOT/eval/start_judge_native.sh status $PORT
pgrep -u $USER -fa "go-judge|server.js" | grep -v pgrep || echo "no judge processes left"
$PE_PY -m pytest -q eval/tests -p no:cacheprovider 2>&1 | tail -3
$PE_PY eval/local_eval.py solutions/baseline_naive.cpp | cut -c1-300
ls $PE_LOCAL_TMP
