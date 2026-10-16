#!/usr/bin/env bash
# Short CPU-partition validation of the gate-job scripts before the GPU job is released:
#   docker/execution-model checks on a cpu node, shared native judge via lib.sh, judge_validate.py,
#   noise_concurrent.py (small K/reps), and clean judge shutdown.
# Results: logs/validate_cpu_<jobid>/summary.json
#SBATCH --job-name=pe_validate_cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --account=ppliang_mit
#SBATCH --output=/work/mit/ppliang_mit/yaning/workspace/potential_estimation/logs/%x-%j.out
set -uo pipefail
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
source "$PE_ROOT/orchestrator/jobs/lib.sh"
cd "$PE_ROOT"
OUT="$PE_ROOT/logs/validate_cpu_${SLURM_JOB_ID}"; mkdir -p "$OUT"
step() { echo "=== [$(date +%T)] $*"; }
rc() { [ "$1" -eq 0 ] && echo true || echo false; }

step "1. docker / execution-model checks (cpu node)"
pe_docker_checks "$OUT/docker_checks.txt"; tail -1 "$OUT/docker_checks.txt"

step "2. shared native judge on all job CPUs"
pe_start_judge || { echo "judge failed to start"; exit 1; }

step "3. judge_validate.py"
"$PE_PY" orchestrator/jobs/judge_validate.py --url "$PE_JUDGE_URL" --judge-type "$PE_JUDGE_TYPE" --out "$OUT/validate"
V=$(rc $?)

step "4. noise_concurrent.py (K=3, reps=2: script check, not the real measurement)"
"$PE_PY" orchestrator/jobs/noise_concurrent.py --url "$PE_JUDGE_URL" --judge-type "$PE_JUDGE_TYPE" \
  --concurrency 3 --reps 2 --out "$OUT/noise_small.json"
N=$(rc $?)
NOISE_CHECK=$("$PE_PY" - "$OUT/noise_small.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))["solutions"]
tl = d["reference_tl1500"]
ok = (all(s == "ok" for v in d.values() for s in v["isolated"]["statuses"] + v["concurrent"]["statuses"])
      and tl["concurrent"]["max_case_cpu_s"] is not None and tl["concurrent"]["max_case_cpu_s"] < 1.75)
print("true" if ok else "false", "tl1500_max_case_cpu_s=%s" % tl["concurrent"]["max_case_cpu_s"],
      {k: v["concurrent"]["fail_reasons"] for k, v in d.items() if v["concurrent"]["fail_reasons"]})
PY
)
echo "noise check: $NOISE_CHECK"

step "5. judge shutdown"
pe_stop_all; trap - EXIT; sleep 2
LEFT=$(pgrep -u "$USER" -f "$PE_JUDGE_STATE_DIR" | wc -l)
S=$([ "$LEFT" -eq 0 ] && echo true || echo false)

cat > "$OUT/summary.json" <<EOF
{"job": "$SLURM_JOB_ID", "node": "$(hostname)", "docker_ok": $PE_DOCKER_OK, "docker_fail": "${PE_DOCKER_FAIL:-}",
 "judge": "$PE_JUDGE_ID", "judge_validate": $V, "noise_script": $N, "noise_check": ${NOISE_CHECK%% *},
 "clean_shutdown": $S}
EOF
cat "$OUT/summary.json"
