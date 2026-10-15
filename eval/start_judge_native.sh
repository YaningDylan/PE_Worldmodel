#!/usr/bin/env bash
# Start / stop the OFFICIAL Frontier-CS algorithmic judge natively (no Docker) inside a Slurm job.
#
#   go-judge v1.11.1 (release binary, sha256-verified)  +  node 20.19.4 running the pinned
#   server.js / judge/src (byte-identical copies of Frontier-CS e6ef46eb, see judge_runtime/app).
#
# Usage (run on a compute node, inside the job that will use the judge):
#   eval/start_judge_native.sh start  [PORT]   # default PORT=8081; waits for /health + /problems
#   eval/start_judge_native.sh status [PORT]
#   eval/start_judge_native.sh stop   [PORT]
#
# Environment knobs (all optional):
#   GJ_PORT          go-judge HTTP port (default PORT+1000, bound to 127.0.0.1 only)
#   GJ_PARALLELISM   go-judge concurrent runs   (default: #CPUs in this job's affinity, like entrypoint.sh's nproc)
#   JUDGE_WORKERS    concurrent submissions     (default: same as GJ_PARALLELISM, like entrypoint.sh)
#   GJ_CGROUP_MODE   "rlimit" (default): hide the systemd user bus from go-judge so it never moves itself
#                    out of the Slurm job cgroup; it then runs in its documented rlimit/rusage fallback.
#                    "systemd": let go-judge create a transient systemd user scope (cgroup v2 limits);
#                    NOTE this moves go-judge OUT of the Slurm job's cgroup/cpuset - avoid on shared nodes.
#   JUDGE_ROOTFS     optional: unpacked official judge image (e.g. $PE_SCRATCH/rootfs/frontiercs_judge_ubuntu2204);
#                    the sandbox then compiles/runs with the image's Ubuntu 22.04 g++ 11.4 + glibc instead
#                    of the host RHEL 9 g++ 11.5 (see eval/README.md)
#   JUDGE_STATE_DIR  node-local state dir (default $PE_LOCAL_TMP/judge_$PORT)
#   SUBMISSIONS_DIR  where server.js stores submissions (default $JUDGE_STATE_DIR/submissions)
#   JUDGE_START_TIMEOUT  seconds to wait for health (default 120)
#   JUDGE_DEBUG_LOG  if non-empty, preload app/debug_run_logger.mjs: logs raw non-Accepted go-judge results to
#                    logs/server.log (debug only; judge code/behaviour unchanged)
#   GJ_EXTRA_ARGS    extra go-judge flags (default "-release"; set to e.g. "-pre-fork 1" for verbose logs)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/../env.sh"

RT="$HERE/judge_runtime"
APP="$RT/app"
NODE="$RT/node-v20.19.4-linux-x64/bin/node"
GOJUDGE="$RT/go-judge/go-judge"

cmd="${1:-start}"
PORT="${2:-${PORT:-8081}}"
GJ_PORT="${GJ_PORT:-$((PORT + 1000))}"
STATE="${JUDGE_STATE_DIR:-$PE_LOCAL_TMP/judge_$PORT}"

ncpu() { python3 -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || nproc; }

is_alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

stop_judge() {
  local rc=0
  for name in node gojudge; do
    local pf="$STATE/$name.pid"
    if is_alive "$pf"; then
      local pid; pid="$(cat "$pf")"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
      if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" 2>/dev/null || true; fi
      echo "[judge] stopped $name (pid $pid)"
    fi
    rm -f "$pf"
  done
  # go-judge file store lives in the state dir; drop it (keep logs + submissions).
  rm -rf "$STATE/gjfiles" "$STATE/tmp" 2>/dev/null || true
  return $rc
}

case "$cmd" in
  stop)
    stop_judge; exit 0 ;;
  status)
    if is_alive "$STATE/node.pid" && is_alive "$STATE/gojudge.pid" && curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[judge] running: http://127.0.0.1:$PORT (go-judge 127.0.0.1:$GJ_PORT, state $STATE)"; exit 0
    fi
    echo "[judge] not running on port $PORT"; exit 1 ;;
  start) ;;
  *) echo "usage: $0 {start|stop|status} [PORT]" >&2; exit 2 ;;
esac

if is_alive "$STATE/node.pid" || is_alive "$STATE/gojudge.pid"; then
  echo "[judge] already running for port $PORT (state $STATE); stop it first" >&2; exit 1
fi
lo_ephemeral="$(cut -f1 /proc/sys/net/ipv4/ip_local_port_range 2>/dev/null || echo 32768)"
for p in "$PORT" "$GJ_PORT"; do
  if (( p >= lo_ephemeral )); then
    echo "[judge] WARNING: port $p is inside the ephemeral range (>= $lo_ephemeral); it may be taken by an outgoing connection" >&2
  fi
done
for p in "$PORT" "$GJ_PORT"; do
  if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
    echo "[judge] port $p on 127.0.0.1 is already in use (another user's judge on this node?) - pick another PORT" >&2; exit 1
  fi
done

N="$(ncpu)"
GJ_PARALLELISM="${GJ_PARALLELISM:-$N}"
JUDGE_WORKERS="${JUDGE_WORKERS:-$GJ_PARALLELISM}"
GJ_CGROUP_MODE="${GJ_CGROUP_MODE:-rlimit}"

mkdir -p "$STATE/gjfiles" "$STATE/tmp" "$STATE/logs"
SUBMISSIONS_DIR="${SUBMISSIONS_DIR:-$STATE/submissions}"
mkdir -p "$SUBMISSIONS_DIR"

# mount.yaml: host userland + pinned testlib include dir at /testlib (default), or, with
# JUDGE_ROOTFS=<dir>, the unpacked official-image userland (Ubuntu 22.04, testlib at /lib/testlib).
if [[ -n "${JUDGE_ROOTFS:-}" ]]; then
  [[ -x "$JUDGE_ROOTFS/usr/bin/g++" && -f "$JUDGE_ROOTFS/lib/testlib/testlib.h" ]] || {
    echo "[judge] JUDGE_ROOTFS=$JUDGE_ROOTFS lacks usr/bin/g++ or lib/testlib/testlib.h" >&2; exit 1; }
  sed "s|__ROOTFS__|$JUDGE_ROOTFS|g" "$RT/go-judge/mount.rootfs.yaml.in" > "$STATE/mount.yaml"
  TESTLIB_INSIDE=/lib/testlib
else
  sed "s|__TESTLIB_PATH__|$APP/include|g" "$RT/go-judge/mount.yaml.in" > "$STATE/mount.yaml"
  TESTLIB_INSIDE=/testlib
fi

# docker-compose.yml sets an unlimited stack ulimit for the container; mirror it (best effort).
ulimit -s unlimited 2>/dev/null || true

GJ_ENV=(env)
if [[ "$GJ_CGROUP_MODE" != "systemd" ]]; then
  # Unsetting the bus variables is NOT enough: godbus then probes /run/user/$UID/bus, which exists on
  # the compute nodes, and go-judge moves itself into user@UID.service (outside the Slurm job).
  GJ_ENV=(env -u XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS=unix:path=/nonexistent/pe_no_session_bus)
fi

(
  cd "$STATE"
  exec "${GJ_ENV[@]}" "$GOJUDGE" \
    -mount-conf "$STATE/mount.yaml" \
    -http-addr "127.0.0.1:$GJ_PORT" \
    -dir "$STATE/gjfiles" \
    -parallelism "$GJ_PARALLELISM" \
    ${GJ_EXTRA_ARGS:--release}
) > "$STATE/logs/gojudge.log" 2>&1 &
echo $! > "$STATE/gojudge.pid"

for _ in $(seq 1 100); do
  curl -fsS "http://127.0.0.1:$GJ_PORT/version" >/dev/null 2>&1 && break
  if ! is_alive "$STATE/gojudge.pid"; then
    echo "[judge] go-judge died during startup:" >&2; tail -20 "$STATE/logs/gojudge.log" >&2; stop_judge; exit 1
  fi
  sleep 0.2
done

(
  cd "$APP"
  exec env PORT="$PORT" GJ_ADDR="http://127.0.0.1:$GJ_PORT" JUDGE_WORKERS="$JUDGE_WORKERS" \
    GJ_PARALLELISM="$GJ_PARALLELISM" SAVE_OUTPUTS=false TESTLIB_INSIDE="$TESTLIB_INSIDE" \
    SUBMISSIONS_DIR="$SUBMISSIONS_DIR" TMPDIR="$STATE/tmp" \
    "$NODE" ${JUDGE_DEBUG_LOG:+--import "$APP/debug_run_logger.mjs"} "$APP/server.js"
) > "$STATE/logs/server.log" 2>&1 &
echo $! > "$STATE/node.pid"

deadline=$(( $(date +%s) + ${JUDGE_START_TIMEOUT:-120} ))
until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! is_alive "$STATE/node.pid" || ! is_alive "$STATE/gojudge.pid"; then
    echo "[judge] judge exited during startup" >&2
    tail -20 "$STATE/logs/server.log" "$STATE/logs/gojudge.log" >&2 || true
    stop_judge; exit 1
  fi
  if (( $(date +%s) > deadline )); then
    echo "[judge] timed out waiting for health" >&2; stop_judge; exit 1
  fi
  sleep 0.5
done
if ! curl -fsS "http://127.0.0.1:$PORT/problems" | grep -q '"id":"0"'; then
  echo "[judge] /problems does not list problem 0" >&2; stop_judge; exit 1
fi
mode_line="$(grep -m1 -o 'falling back to rlimit / rusage mode\|creating containers cgroup' "$STATE/logs/gojudge.log" || true)"
echo "[judge] ready: http://127.0.0.1:$PORT  (go-judge 127.0.0.1:$GJ_PORT, parallelism=$GJ_PARALLELISM, workers=$JUDGE_WORKERS, cgroup_mode=$GJ_CGROUP_MODE: ${mode_line:-?}, userland=${JUDGE_ROOTFS:-host}, state=$STATE)"
