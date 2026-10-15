#!/bin/bash
# Start the native official judge in this job, check /problems, evaluate the two official examples.
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
PORT=${PORT:-$((20000 + RANDOM % 10000))}
export GJ_CGROUP_MODE=${GJ_CGROUP_MODE:-rlimit}
trap "$PE_ROOT/eval/start_judge_native.sh stop $PORT" EXIT
echo "host=$(hostname) cpus=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))') mode=$GJ_CGROUP_MODE port=$PORT"
$PE_ROOT/eval/start_judge_native.sh start $PORT || exit 1
STATE=$PE_LOCAL_TMP/judge_$PORT
head -12 $STATE/logs/gojudge.log | cut -c1-400
curl -sS http://127.0.0.1:$PORT/problems; echo
for s in reference gpt5; do
  t0=$(date +%s.%N)
  $PE_ROOT/../envs/pe/bin/frontier eval algorithmic 0 $PROBLEM_DIR/examples/$s.cpp --judge-url http://127.0.0.1:$PORT --json > $PE_ROOT/logs/native_${s}_${SLURM_JOB_ID}.json 2> $PE_ROOT/logs/native_${s}_${SLURM_JOB_ID}.err
  t1=$(date +%s.%N)
  echo "$s wall=$(python3 -c "print(round($t1-$t0,2))")"
  $PE_PY - $PE_ROOT/logs/native_${s}_${SLURM_JOB_ID}.json <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
r=d[0] if isinstance(d,list) else d
md=r.get('metadata') or {}
cases=md.get('cases') or []
print({k:r.get(k) for k in ('status','score','score_unbounded','message','duration_seconds')}, 'n_cases', len(cases))
st={}
for c in cases: st[c.get('status')]=st.get(c.get('status'),0)+1
print('case statuses', st, 'max time ms', max([c.get('time',0) for c in cases] or [0])/1e6, 'max mem MB', max([c.get('memory',0) for c in cases] or [0])/2**20)
PY
done
tail -5 $STATE/logs/server.log
