#!/usr/bin/env bash
# Minimal single-B200 job: 1 GPU + 16 cores, print GPU/CPU facts and exit.
#SBATCH --job-name=pe_gpu_smoke
#SBATCH --partition=b200-devel,b200-batch
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --account=ppliang_mit
#SBATCH --output=/work/mit/ppliang_mit/yaning/workspace/potential_estimation/logs/%x-%j.out
set -euo pipefail
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
echo "host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION cpus=$SLURM_CPUS_PER_TASK nproc=$(nproc)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
which docker podman apptainer 2>&1 || true
podman info 2>&1 | grep -iE "rootless|cgroupVersion" || true
df -h /tmp | tail -1
timeout 10 curl -sI https://huggingface.co | head -1 || echo "no HF access"
timeout 10 curl -sI https://github.com | head -1 || echo "no GitHub access"
timeout 10 curl -sI https://pypi.org/simple/ | head -1 || echo "no PyPI access"
g++ --version | head -1
ulimit -a | grep -iE "stack|virtual|processes"
cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || true; cat /proc/sys/user/max_user_namespaces
echo DONE
