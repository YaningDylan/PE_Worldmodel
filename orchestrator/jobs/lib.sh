# Shared job helpers (source inside a Slurm job after env.sh).
#   pe_docker_checks FILE      raw Docker/container facts of this node -> FILE; sets PE_DOCKER_OK=1|0 and
#                              PE_DOCKER_FAIL (no-binary | daemon-unreachable | socket-permission-denied |
#                              privileged-refused | only-enroot-apptainer-podman)
#   pe_start_vllm MODEL        vLLM pinned to PE_VLLM_NCPUS cores; exports PE_VLLM_* and PE_FREE_CPUS
#   pe_start_judge             one judge for the whole job on PE_FREE_CPUS: official Docker image if
#                              PE_DOCKER_OK=1, else the native copy; exports PE_JUDGE_URL/TYPE/ID
#   pe_stop_all                stop judge + vLLM (installed as EXIT trap by pe_start_*)

pe_docker_checks() {
  local f=$1
  {
    echo "### $(date -Is) job=${SLURM_JOB_ID:-} node=$(hostname)"
    echo "\$ hostname; id; groups"; hostname; id; groups
    echo "\$ which docker docker-compose podman enroot apptainer"; which docker docker-compose podman enroot apptainer 2>&1
    echo "\$ docker info | head -20"; docker info 2>&1 | head -20
    echo "\$ docker compose version"; docker compose version 2>&1
    echo "\$ ls -l /var/run/docker.sock /run/user/\$UID/docker.sock"; ls -l /var/run/docker.sock /run/user/$UID/docker.sock 2>&1
    echo "DOCKER_HOST=${DOCKER_HOST:-}"
    echo "\$ module avail | grep -i -E 'docker|container'"; (module avail 2>&1 || true) | grep -i -E "docker|container" || echo "(no match)"
    echo "\$ sinfo -o '%P %f %G'"; sinfo -o "%P %f %G" 2>&1
    echo "\$ docker run --rm --privileged hello-world"; docker run --rm --privileged hello-world 2>&1 | head -5
    echo "--- execution model"
    echo "\$ cat /proc/1/cgroup"; cat /proc/1/cgroup
    echo "\$ cat /proc/self/cgroup"; cat /proc/self/cgroup
    echo "\$ ls -la /.dockerenv /run/.containerenv"; ls -la /.dockerenv /run/.containerenv 2>&1
    echo "\$ systemd-detect-virt -c"; systemd-detect-virt -c 2>&1
    echo "\$ systemd-detect-virt"; systemd-detect-virt 2>&1
    echo "PID1=$(ps -p 1 -o comm=)"
    echo "\$ cat /etc/os-release | head -4; uname -r"; head -4 /etc/os-release; uname -r
    echo "\$ lscpu (model)"; lscpu | grep -E "Model name|^CPU\(s\)|Socket|Thread|Core"
    echo "\$ ls /usr/bin | grep -i -E 'docker|containerd|nerdctl|rootless|enroot|podman|apptainer|singularity|crun|runc'"
    ls /usr/bin | grep -i -E "docker|containerd|nerdctl|rootless|enroot|podman|apptainer|singularity|crun|runc" || true
    echo "\$ grep \$USER /etc/subuid /etc/subgid"; grep "$USER" /etc/subuid /etc/subgid 2>&1 || echo "(none)"
  } > "$f" 2>&1
  PE_DOCKER_OK=0
  if ! command -v docker >/dev/null 2>&1; then
    if command -v podman >/dev/null || command -v apptainer >/dev/null || command -v enroot >/dev/null; then
      PE_DOCKER_FAIL=only-enroot-apptainer-podman
    else
      PE_DOCKER_FAIL=no-binary
    fi
  elif docker info 2>&1 | grep -qi "permission denied"; then PE_DOCKER_FAIL=socket-permission-denied
  elif ! docker info >/dev/null 2>&1; then PE_DOCKER_FAIL=daemon-unreachable
  elif ! docker run --rm --privileged hello-world >/dev/null 2>&1; then PE_DOCKER_FAIL=privileged-refused
  else PE_DOCKER_OK=1; PE_DOCKER_FAIL=; fi
  echo "### classification: PE_DOCKER_OK=$PE_DOCKER_OK PE_DOCKER_FAIL=${PE_DOCKER_FAIL:-none}" >> "$f"
  export PE_DOCKER_OK PE_DOCKER_FAIL
}

pe_stop_all() {
  if [ "${PE_JUDGE_TYPE:-}" = native ]; then
    JUDGE_STATE_DIR="$PE_JUDGE_STATE_DIR" "$PE_ROOT/eval/start_judge_native.sh" stop "$PE_JUDGE_PORT" || true
  fi
  if [ "${PE_JUDGE_TYPE:-}" = official-docker ]; then
    (cd "$FRONTIER_DIR/algorithmic" && docker compose -p "pe${SLURM_JOB_ID:-x}" down) || true
  fi
  if [ -n "${PE_VLLM_RUN_DIR:-}" ]; then "$PE_ROOT/serve/stop_vllm.sh" "$PE_VLLM_RUN_DIR" || true; fi
}

pe_start_vllm() {
  # Fallback ladder so one bad setting does not cost another queue wait: the Apptainer image first (how
  # the group's working B200 stacks run), then the conda env; each with compiled graphs, then eager.
  # PE_VLLM_VARIANT records what worked.
  local model=$1 v ok=0 rt var
  local ladder=(apptainer/default conda/default apptainer/eager conda/eager)
  [ -s "${PE_VLLM_SIF:-$PE_SCRATCH/containers/vllm-openai_v0.22.1.sif}" ] || ladder=(conda/default conda/eager)
  for v in ${PE_VLLM_LADDER:-${ladder[@]}}; do
    rt=${v%%/*}; var=${v##*/}
    local extra=(--runtime "$rt")
    [ "$var" = eager ] && extra+=(--enforce-eager)
    echo "[vllm] launching variant=$v"
    # max-num-seqs 64 (we never exceed ~20 concurrent requests): with vLLM's default (1024) the hybrid
    # model's Mamba cache on a 96 GB RTX PRO 6000 is too small for CUDA-graph capture (job 1345086).
    if "$PE_ROOT/serve/launch_vllm.sh" --model "$model" --ncpus "${PE_VLLM_NCPUS:-4}" \
         --max-num-seqs "${PE_VLLM_MAX_NUM_SEQS:-64}" "${extra[@]}" -- --enable-prefix-caching > /dev/null; then
      ok=1; break
    fi
    echo "[vllm] variant=$v failed"
  done
  [ "$ok" = 1 ] || return 1
  export PE_VLLM_VARIANT=$v
  # shellcheck disable=SC1090
  source "$(cat "$PE_SCRATCH/vllm_current_${SLURM_JOB_ID:-nojob}")/server.env"
  export PE_VLLM_BASE_URL PE_VLLM_MODEL_NAME PE_VLLM_RUN_DIR PE_FREE_CPUS
  trap pe_stop_all EXIT
}

pe_start_judge() {
  local cpus=${PE_FREE_CPUS:-$("$PE_PY" -c 'import os;print(",".join(map(str,sorted(os.sched_getaffinity(0)))))')}
  local n; n=$(awk -F, '{print NF}' <<< "$cpus")
  PE_JUDGE_PORT=${PE_JUDGE_PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 1000))}
  if [ "${PE_DOCKER_OK:-0}" = 1 ]; then
    (cd "$FRONTIER_DIR/algorithmic" && \
      JUDGE_WORKERS=$n GJ_PARALLELISM=$n docker compose -p "pe${SLURM_JOB_ID:-x}" up -d --build) || return 1
    PE_JUDGE_PORT=8081; PE_JUDGE_TYPE=official-docker
  else
    PE_JUDGE_STATE_DIR="$PE_LOCAL_TMP/judge_${SLURM_JOB_ID:-x}"
    export PE_JUDGE_STATE_DIR  # stop must see the same state dir, or it silently stops nothing
    JUDGE_STATE_DIR="$PE_JUDGE_STATE_DIR" GJ_PARALLELISM=$n JUDGE_WORKERS=$n \
      taskset -c "$cpus" "$PE_ROOT/eval/start_judge_native.sh" start "$PE_JUDGE_PORT" || return 1
    PE_JUDGE_TYPE=native
  fi
  PE_JUDGE_URL="http://127.0.0.1:$PE_JUDGE_PORT"
  curl -fsS "$PE_JUDGE_URL/problems" || return 1; echo
  PE_JUDGE_ID="$PE_JUDGE_TYPE@$(hostname):$PE_JUDGE_PORT/job${SLURM_JOB_ID:-x}/workers$n"
  export PE_JUDGE_URL PE_JUDGE_TYPE PE_JUDGE_ID PE_JUDGE_PORT
  trap pe_stop_all EXIT
  echo "[judge] $PE_JUDGE_ID cpus=$cpus"
}
