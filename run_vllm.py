"""
vLLM offline profiler.

Two things it shows that the vanilla run can't:
  1. Single-prompt latency per size (apples-to-apples vs vanilla).
  2. Batched throughput -- continuous batching means aggregate tok/s should
     RISE with batch size, where vanilla stays flat. This is the real reason
     one card can serve many users.

    python run_vllm.py
    MODEL_ID="meta-llama/Llama-3.3-70B-Instruct" QUANT=fp8 python run_vllm.py
"""
import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import config
from gpustats import Timer, vram_total_mb, vram_used_mb
from prompts import build_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--quant", default=config.QUANT)
    ap.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    ap.add_argument("--max-model-len", type=int, default=config.MAX_MODEL_LEN)
    ap.add_argument("--gpu-mem-util", type=float, default=config.GPU_MEM_UTIL)
    args = ap.parse_args()

    print(f"[vllm] loading {args.model} (quant={args.quant}) ...")
    llm_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    if args.quant:
        llm_kwargs["quantization"] = args.quant
    llm = LLM(**llm_kwargs)

    # IMPORTANT: vLLM pre-reserves KV cache up to gpu_memory_utilization, so
    # nvidia-smi "used" sits near the cap by design -- that is NOT an OOM.
    # For the true breakdown read vLLM's own startup log lines:
    #   "model weights take X GiB", "GPU KV cache size: N tokens",
    #   "Maximum concurrency for M tokens per request: K.xx x"
    # That K is roughly how many concurrent requests the card can serve.
    print(
        f"[vllm] device used after load: "
        f"{vram_used_mb():.0f}/{vram_total_mb():.0f} MB "
        f"(KV pre-reserved -- read the vLLM log above for the real split)"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = build_prompts(tokenizer, config.PROMPT_SIZES)
    sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    runs = []

    # 1) Single-prompt latency per size.
    for size, prompt in prompts.items():
        with Timer() as t:
            out = llm.generate([prompt], sp, use_tqdm=False)
        n_out = len(out[0].outputs[0].token_ids)
        runs.append(
            dict(mode="single", target_in_tok=size, batch=1,
                 n_out=n_out, total_s=round(t.dt, 3),
                 tok_s=round(n_out / t.dt, 1),
                 output=out[0].outputs[0].text)
        )
        print(f"[vllm] single ~{size} tok: {round(n_out/t.dt,1)} tok/s "
              f"({round(t.dt,2)}s)")

    # 2) Batched throughput at a mid prompt size.
    mid = config.PROMPT_SIZES[len(config.PROMPT_SIZES) // 2]
    batch_prompt = prompts[mid]
    for bs in config.VLLM_BATCH_SIZES:
        with Timer() as t:
            out = llm.generate([batch_prompt] * bs, sp, use_tqdm=False)
        total_out = sum(len(o.outputs[0].token_ids) for o in out)
        runs.append(
            dict(mode="batch", target_in_tok=mid, batch=bs,
                 n_out=total_out, total_s=round(t.dt, 3),
                 tok_s=round(total_out / t.dt, 1),
                 outputs=[o.outputs[0].text for o in out])
        )
        print(f"[vllm] batch={bs} @~{mid} tok: "
              f"{round(total_out/t.dt,1)} tok/s aggregate")

    out = dict(
        backend="vllm",
        model=args.model,
        quant=args.quant,
        device_total_mb=round(vram_total_mb()),
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        runs=runs,
    )
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "vllm.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[vllm] wrote {path}")


if __name__ == "__main__":
    main()
