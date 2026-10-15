# Source me: shared paths/caches for the potential-estimation project.
export PE_ROOT=/work/mit/ppliang_mit/yaning/workspace/potential_estimation
export PE_SCRATCH=/work/mit/ppliang_mit/yaning/workspace/pe_scratch
export PE_PY=/work/mit/ppliang_mit/yaning/workspace/envs/pe/bin/python
export VLLM_ENV=/home/dvdai_mit/miniconda3/envs/ocrvl          # vLLM 0.22.1 / torch 2.11 cu130 (read-only reuse)
export FRONTIER_DIR=$PE_ROOT/third_party/Frontier-CS
export PROBLEM_DIR=$FRONTIER_DIR/algorithmic/problems/0
# /home quota is nearly full and ~/.config is read-only on compute nodes: keep every cache on /work.
export HF_HOME=$PE_SCRATCH/hf
export UV_CACHE_DIR=$PE_SCRATCH/uv_cache
export XDG_CACHE_HOME=$PE_SCRATCH/xdg_cache
export XDG_CONFIG_HOME=$PE_SCRATCH/xdg_config
export TRITON_CACHE_DIR=$PE_SCRATCH/triton_cache
export VLLM_CACHE_ROOT=$PE_SCRATCH/vllm_cache
export TORCHINDUCTOR_CACHE_DIR=$PE_SCRATCH/inductor_cache
# Blackwell nodes have no nvcc: avoid FlashInfer JIT paths.
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
# B200 (SM 10.0): vLLM 0.22.1's kernel warm-up calls into deep_gemm, and the copy in the ocrvl env is too old
# ("DeepGEMM backend is not available or outdated" -> engine dies after model load; jobs 1211675/1211676/
# 1212451). Our BF16 checkpoints never use FP8 DeepGEMM kernels, so switch it off.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
# Node-local scratch for compiling/running candidate binaries (NFS is slow and shared).
export PE_LOCAL_TMP=${SLURM_TMPDIR:-/tmp}/pe_${USER}
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$PE_LOCAL_TMP" 2>/dev/null || true
