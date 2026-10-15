#!/usr/bin/env bash
# Pull the official vLLM image into a SIF (cpu partition, as the cluster docs recommend) and run every
# check that does not need a GPU: versions, DeepGEMM/FlashInfer state, nvcc, exact CLI flag spellings
# (also for the conda fallback env), and that our Qwen checkpoints' architecture is registered.
#   sbatch serve/scripts/pull_vllm_image.sbatch            # PE_VLLM_IMAGE_TAG=v0.22.1 by default
#SBATCH --job-name=pe_pull_vllm
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --account=ppliang_mit
#SBATCH --output=/work/mit/ppliang_mit/yaning/workspace/potential_estimation/logs/%x-%j.out
set -uo pipefail
source /work/mit/ppliang_mit/yaning/workspace/potential_estimation/env.sh
TAG=${PE_VLLM_IMAGE_TAG:-v0.22.1}
SIF_DIR=$PE_SCRATCH/containers; mkdir -p "$SIF_DIR"
SIF=$SIF_DIR/vllm-openai_${TAG}.sif
# layer cache + build temp on node-local disk: nothing but the final SIF lands on /work (96% full)
export APPTAINER_CACHEDIR=/tmp/pe_${USER}_apptainer_${SLURM_JOB_ID}/cache
export APPTAINER_TMPDIR=/tmp/pe_${USER}_apptainer_${SLURM_JOB_ID}/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
trap 'rm -rf "/tmp/pe_${USER}_apptainer_${SLURM_JOB_ID}"' EXIT
unset APPTAINER_BIND
echo "== $(date +%T) node=$(hostname) tag=$TAG sif=$SIF  /tmp: $(df -h /tmp | tail -1 | awk '{print $4" free"}')"

if [ ! -s "$SIF" ]; then
  apptainer pull "$SIF.partial" "docker://vllm/vllm-openai:$TAG" || { echo "PULL FAILED"; exit 1; }
  mv "$SIF.partial" "$SIF"
fi
ls -la "$SIF"; sha256sum "$SIF" | cut -c1-64

PROBE=$(cat <<'PY'
import importlib, json, shutil, sys
out = {}
import vllm, torch, transformers
out["vllm"], out["torch"], out["torch_cuda"], out["transformers"] = vllm.__version__, torch.__version__, torch.version.cuda, transformers.__version__
for m in ("deep_gemm", "flashinfer", "triton"):
    try:
        mod = importlib.import_module(m); out[m] = getattr(mod, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        out[m] = f"MISSING ({type(e).__name__})"
from vllm.utils import deep_gemm as dg
out["has_deep_gemm"] = dg.has_deep_gemm()
try:
    dg._lazy_init()
    out["deep_gemm_mk_alignment_impl"] = getattr(dg, "_get_mk_alignment_for_contiguous_layout_impl", "n/a") is not None
except Exception as e:  # noqa: BLE001
    out["deep_gemm_lazy_init_error"] = repr(e)[:200]
import vllm.envs as envs
out["env_VLLM_USE_DEEP_GEMM"] = envs.VLLM_USE_DEEP_GEMM
out["nvcc"] = shutil.which("nvcc")
from vllm.model_executor.models.registry import ModelRegistry
out["qwen3_5_registered"] = "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs()
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser
p = make_arg_parser(FlexibleArgumentParser())
opts = {s for a in p._actions for s in a.option_strings}
need = ["--language-model-only", "--attention-backend", "--reasoning-parser", "--enable-prefix-caching",
        "--no-enable-flashinfer-autotune", "--enforce-eager", "--max-model-len", "--gpu-memory-utilization",
        "--served-model-name", "--disable-uvicorn-access-log", "--speculative-config", "--max-num-seqs"]
out["flags_missing"] = [f for f in need if f not in opts]
out["flashinfer_flags"] = sorted(s for s in opts if "flashinfer" in s)
ra = next(a for a in p._actions if "--reasoning-parser" in a.option_strings)
out["reasoning_parser_has_qwen3"] = ("qwen3" in (ra.choices or [])) if ra.choices else "no choices list"
print("PROBE " + json.dumps(out))
PY
)

echo "== $(date +%T) probe inside the SIF (no GPU on this node)"
mkdir -p "$PE_SCRATCH/apptainer_home"
apptainer exec --cleanenv --home "$PE_SCRATCH/apptainer_home" -B /work \
  --env VLLM_USE_DEEP_GEMM=0 --env VLLM_NO_USAGE_STATS=1 --env HF_HUB_OFFLINE=1 \
  "$SIF" bash -c 'which vllm python3; cat /etc/os-release | head -2; python3 -c "$0"' "$PROBE" 2>&1 \
  | grep -v -i -E "warn|raw_cnt|ccache|paddle" | tail -8

echo "== $(date +%T) same probe in the conda fallback env"
"$VLLM_ENV/bin/python" -c "$PROBE" 2>&1 | grep -E "^PROBE" | tail -1
echo "== $(date +%T) DONE"
