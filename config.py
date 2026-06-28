"""
Central config. Everything is overridable by environment variable so you can
sweep models/quant without editing code:

    MODEL_ID="meta-llama/Llama-3.3-70B-Instruct" QUANT=fp8 python run_vllm.py
"""
import os

# ---------------------------------------------------------------------------
# Model to profile.
#
# Validate the harness on a SMALL model first (downloads in seconds, runs on any
# GPU), then swap to the real targets once the plumbing works.
#
#   Smoke test   : "meta-llama/Llama-3.2-1B-Instruct"
#                  "Qwen/Qwen2.5-1.5B-Instruct"        (not gated, easy)
#   Real targets : "meta-llama/Llama-3.3-70B-Instruct" -> ~40 GB resident (fp8/4bit)
#                  "openai/gpt-oss-120b"               -> ~63 GB resident (the driver)
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-70B-Instruct")

# Quantization.
#   vanilla (HF)  : None | "4bit" | "8bit"        (bitsandbytes)
#   vLLM          : None | "fp8" | "awq" | "gptq"
# 70B in bf16 is ~140 GB and will NOT fit one 96 GB card -> use fp8 (vLLM) or 4bit (HF).
QUANT = os.environ.get("QUANT", "fp8") or None

# Input prompt sizes to sweep, in approximate *input* tokens.
PROMPT_SIZES = [128, 1024, 4096, 8192]

# Tokens to generate per request.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))

# vLLM: fraction of the card vLLM may use (model weights + KV cache).
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", "0.90"))

# vLLM: context length to size the KV cache for.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "16384"))

# vLLM: batch sizes for the throughput test (continuous batching is the point).
VLLM_BATCH_SIZES = [1, 8, 16]

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
