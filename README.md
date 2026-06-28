# gpu-profiler

A tiny harness to answer one question on the GCP G4 (RTX PRO 6000, 96 GB):
**will one card hold the strong model and still leave room to serve users?**

It pulls a model, prompts it at several context sizes, and profiles VRAM +
latency + throughput two ways: **vanilla HuggingFace** (sequential baseline) and
**vLLM** (continuous batching — the production path).

## Layout

```
config.py        model / quant / prompt sizes / vLLM knobs (all env-overridable)
prompts.py       builds prompts at a target token length using the tokenizer
gpustats.py      NVML VRAM sampling + timing (works across both backends)
run_vanilla.py   HF transformers profiler  -> results/vanilla.json
run_vllm.py      vLLM profiler             -> results/vllm.json
compare.py       prints both side by side
```

## Setup (on the G4 instance)

```bash
pip install -r requirements.txt
huggingface-cli login          # only needed for gated models (Llama)
```

Point the HF cache at a disk with room — 70B/120B weights are large:

```bash
export HF_HOME=/mnt/scratch/hf      # use a big disk, not the 40 GB boot disk
```

## Run

Start small to prove the harness works on any GPU:

```bash
python run_vanilla.py
python run_vllm.py
python compare.py
```

Then swap to the real target and re-run:

```bash
# 70B — leaves real KV headroom on 96 GB (recommended)
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct" QUANT=fp8 python run_vllm.py

# vanilla can't fit 70B in bf16 (~140 GB); use 4-bit to test the footprint
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct" QUANT=4bit python run_vanilla.py

# the 120B ceiling — confirm whether vision can stay co-resident or runs off-peak
MODEL_ID="openai/gpt-oss-120b" python run_vllm.py
```

## Reading the results — the go/no-go

**1. Resident footprint (does the model even fit, with room to spare?)**
- vanilla prints `model weights resident: X MB` straight from NVML.
- vLLM *pre-reserves* KV cache up to `GPU_MEM_UTIL`, so `nvidia-smi` looks ~90%
  full **by design — that is not an OOM**. Read vLLM's own startup lines:
  `model weights take X GiB` and `GPU KV cache size: N tokens`.

**2. Concurrency (can it serve 20–25 users?)**
- In the vLLM log, find: `Maximum concurrency for <ctx> tokens per request: K.xx x`.
  That `K` ≈ how many simultaneous requests fit in KV cache. With think-time,
  20–25 users → ~3–8 truly concurrent, so you want `K` comfortably above that.
- The batched run confirms it empirically: aggregate `tok/s` should **rise** with
  batch size. Vanilla stays flat — that contrast is the whole argument for vLLM.

**3. The trap to watch for**
- `gpt-oss-120b` (~63 GB) + vision co-resident (~20 GB) + reranker ≈ 88 GB →
  only ~8 GB for KV → concurrency `K` collapses. If you see that, it confirms the
  proposal's two outs: run OCR/vision **off-peak**, or use **Llama-3.3-70B** (40 GB)
  which leaves ~30 GB of KV headroom.

## Notes
- `do_sample=False` / `temperature=0` everywhere for repeatable numbers.
- Prompt sizes are approximate (tokenizer-sliced); good enough for sizing.
- Delete the instance when done — a *stopped* GPU VM still bills for its disk.
