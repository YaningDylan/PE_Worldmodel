#!/usr/bin/env bash
# Start / stop the official Frontier-CS judge from an Apptainer image that replicates the official
# Docker image (Ubuntu 22.04, g++ 11.4, node 20, go-judge v1.11.1, /app = pinned server.js+judge/*).
# Image: $PE_SCRATCH/sif/frontiercs_judge.sif, built by eval/container_attempts/build_apptainer_judge.sh
# from eval/container_attempts/frontiercs_judge.def (only deviation from the Dockerfile: npm not upgraded).
#
#   eval/start_judge_apptainer.sh start [PORT] | status [PORT] | stop [PORT]
#
# Same command line as the image's entrypoint.sh (go-judge -parallelism N & node /app/server.js) plus
# an explicit go-judge -http-addr/-dir (unique per port; nodes are shared) and writable binds for
# /app/data and /app/submissions (the SIF is read-only). go-judge runs rootless inside the container
# and falls back to rlimit/rusage mode (no systemd bus inside --containall).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"
SIF="${JUDGE_SIF:-$PE_SCRATCH/sif/frontiercs_judge.sif}"
cmd="${1:-start}"
PORT="${2:-${PORT:-8081}}"
GJ_PORT="${GJ_PORT:-$((PORT + 1000))}"
STATE="${JUDGE_STATE_DIR:-$PE_LOCAL_TMP/ajudge_$PORT}"
PIDF="$STATE/apptainer.pid"

stop_judge() {
  if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    pid="$(cat "$PIDF")"
    pkill -TERM -P "$pid" 2>/dev/null || true
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
    kill -KILL "$pid" 2>/dev/null || true
    echo "[ajudge] stopped (pid $pid)"
  fi
  rm -f "$PIDF"; rm -rf "$STATE/gjfiles" 2>/dev/null || true
}

case "$cmd" in
  stop) stop_judge; exit 0 ;;
  status) curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "[ajudge] running on $PORT"; exit 0; } || { echo "[ajudge] not running"; exit 1; } ;;
  start) ;;
  *) echo "usage: $0 {start|stop|status} [PORT]" >&2; exit 2 ;;
esac

N="$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')"
GJ_PARALLELISM="${GJ_PARALLELISM:-$N}"
JUDGE_WORKERS="${JUDGE_WORKERS:-$GJ_PARALLELISM}"
mkdir -p "$STATE/gjfiles" "$STATE/data" "$STATE/submissions" "$STATE/logs"

apptainer exec --containall --cleanenv \
  --bind "$FRONTIER_DIR/algorithmic/problems:/app/problems:ro" \
  --bind "$STATE/data:/app/data" --bind "$STATE/submissions:/app/submissions" \
  --bind "$STATE/gjfiles:/gjfiles" \
  --env "PORT=$PORT,GJ_ADDR=http://127.0.0.1:$GJ_PORT,JUDGE_WORKERS=$JUDGE_WORKERS,GJ_PARALLELISM=$GJ_PARALLELISM,SAVE_OUTPUTS=false" \
  "$SIF" bash -c "ulimit -s unlimited 2>/dev/null; cd /app && \
     /usr/local/bin/go-judge -parallelism $GJ_PARALLELISM -http-addr 127.0.0.1:$GJ_PORT -dir /gjfiles > /app/data/gojudge.log 2>&1 & \
     sleep 0.5; cd /app && exec node /app/server.js" \
  > "$STATE/logs/server.log" 2>&1 &
echo $! > "$PIDF"

deadline=$(( $(date +%s) + ${JUDGE_START_TIMEOUT:-180} ))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null || (( $(date +%s) > deadline )); then
    echo "[ajudge] failed to start" >&2; tail -20 "$STATE/logs/server.log" "$STATE/data/gojudge.log" >&2 || true
    stop_judge; exit 1
  fi
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$PORT/problems" | grep -q '"id":"0"' || { echo "[ajudge] problem 0 missing" >&2; stop_judge; exit 1; }
mode="$(grep -m1 -o 'falling back to rlimit / rusage mode\|creating containers cgroup' "$STATE/data/gojudge.log" || true)"
echo "[ajudge] ready: http://127.0.0.1:$PORT (image $SIF, parallelism=$GJ_PARALLELISM, go-judge: ${mode:-?})"
