"""GPU-free facts about a vLLM installation (run inside the SIF or the conda env)."""
import dataclasses, importlib, json, shutil
out = {}
import vllm, torch, transformers
out.update(vllm=vllm.__version__, torch=torch.__version__, torch_cuda=torch.version.cuda,
           transformers=transformers.__version__)
for m in ("deep_gemm", "flashinfer", "triton"):
    try:
        mod = importlib.import_module(m); out[m] = getattr(mod, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        out[m] = f"MISSING ({type(e).__name__})"
from vllm.utils import deep_gemm as dg
import vllm.envs as envs
out["has_deep_gemm"] = dg.has_deep_gemm()
out["env_VLLM_USE_DEEP_GEMM"] = envs.VLLM_USE_DEEP_GEMM
try:
    import deep_gemm as _d
    out["deep_gemm_has_mk_alignment"] = hasattr(_d, "get_mk_alignment_for_contiguous_layout")
except Exception as e:  # noqa: BLE001
    out["deep_gemm_has_mk_alignment"] = f"n/a ({type(e).__name__})"
out["nvcc"] = shutil.which("nvcc")
from vllm.model_executor.models.registry import ModelRegistry
out["qwen3_5_registered"] = "Qwen3_5ForConditionalGeneration" in ModelRegistry.get_supported_archs()
from vllm.engine.arg_utils import EngineArgs
ef = {f.name for f in dataclasses.fields(EngineArgs)}
need = ["language_model_only", "enable_prefix_caching", "enable_flashinfer_autotune", "enforce_eager",
        "max_model_len", "gpu_memory_utilization", "served_model_name", "speculative_config", "max_num_seqs"]
out["engine_fields_missing"] = [n for n in need if n not in ef]
out["attention_fields"] = sorted(n for n in ef if "attention" in n or "attn" in n)
print("PROBE " + json.dumps(out))
