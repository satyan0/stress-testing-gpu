"""
Vanilla HuggingFace transformers profiler -- the baseline.

Sequential, one request at a time. This is what vLLM has to beat. Measures, per
prompt size: resident VRAM, time-to-first-token (TTFT), decode throughput, and
peak VRAM during generation.

    python run_vanilla.py
    MODEL_ID="meta-llama/Llama-3.3-70B-Instruct" QUANT=4bit python run_vanilla.py
"""
import argparse
import json
import os
import threading
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

import config
from gpustats import PeakVRAM, Timer, vram_total_mb, vram_used_mb
from prompts import build_prompts


def load_model(model_id, quant):
    kwargs = dict(device_map="cuda", torch_dtype=torch.bfloat16)
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        kwargs.pop("torch_dtype")
        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def run_once(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    n_in = inputs["input_ids"].shape[1]

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        streamer=streamer,
    )

    ttft, n_out = None, 0
    chunks = []
    with PeakVRAM() as peak, Timer() as total:
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        t0 = time.perf_counter()
        thread.start()
        for i, chunk in enumerate(streamer):
            if i == 0:
                ttft = time.perf_counter() - t0
            chunks.append(chunk)
            n_out += 1
        thread.join()

    decode_tps = (
        n_out / (total.dt - ttft) if (ttft and total.dt > ttft) else float("nan")
    )
    return dict(
        n_in=n_in,
        n_out=n_out,
        ttft_s=round(ttft, 3) if ttft else None,
        total_s=round(total.dt, 3),
        decode_tok_s=round(decode_tps, 1),
        peak_vram_mb=round(peak.peak),
        output="".join(chunks),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.MODEL_ID)
    ap.add_argument("--quant", default=config.QUANT)
    ap.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    args = ap.parse_args()

    print(f"[vanilla] loading {args.model} (quant={args.quant}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_vram = vram_used_mb()
    model = load_model(args.model, args.quant)
    resident = vram_used_mb()
    print(
        f"[vanilla] model weights resident: {resident - base_vram:.0f} MB "
        f"({resident:.0f}/{vram_total_mb():.0f} MB device used)"
    )

    prompts = build_prompts(tokenizer, config.PROMPT_SIZES)
    runs = []
    for size, prompt in prompts.items():
        print(f"[vanilla] prompt ~{size} tok ...", end=" ", flush=True)
        r = run_once(model, tokenizer, prompt, args.max_new_tokens)
        r["target_in_tok"] = size
        runs.append(r)
        print(
            f"TTFT {r['ttft_s']}s  decode {r['decode_tok_s']} tok/s  "
            f"peak {r['peak_vram_mb']} MB"
        )

    out = dict(
        backend="vanilla-hf",
        model=args.model,
        quant=args.quant,
        device_total_mb=round(vram_total_mb()),
        model_resident_mb=round(resident - base_vram),
        runs=runs,
    )
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR, "vanilla.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[vanilla] wrote {path}")


if __name__ == "__main__":
    main()
