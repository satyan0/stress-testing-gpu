# BASELINE — Single 96 GB GPU, Llama-3.1-70B (fp8 / vLLM)

**Date:** 2026-06-28 (UTC) · **Host:** GCP G4 Spot VM · **Card:** 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB)
**Baseline config (production path):** `meta-llama/Llama-3.1-70B-Instruct`, **fp8**, **vLLM 0.23.0**, `max_model_len=16384`, `max_new_tokens=256`.
Vanilla HF **4-bit** is a **separate cross-check**, not the headline.

---

## 1. Executive summary

**Verdict: the single-96 GB-GPU / 70B decision is validated for the LLM, with one correction to the resident-stack budget.** A single RTX PRO 6000 Blackwell holds Llama-3.1-70B at fp8 (67.7 GiB of weights) and runs it through vLLM with a 16k context. Continuous batching is real and strong: aggregate throughput scales **19 → 144 → 270 tok/s** from batch 1 → 8 → 16, which is the mechanism that lets one card serve many interactive users. The KV cache holds **47,648 tokens** (LLM-only) or **39,872 tokens** with a realistic auxiliary-stack reservation — comfortably covering the **~3–8 truly-concurrent** target and approaching the 20–25 registered-user pool once think-time is accounted for.

**The one correction:** the proposal's literal "reserve ~24 GB for vision-OCR + embedder + reranker" does **not** fit alongside fp8 70B at 16k. fp8 weights alone are 67.7 GiB; reserving 24 GB (`gpu_memory_utilization=0.75`) leaves **0.3 GiB** for KV and vLLM **refuses to start**. The realistic co-resident budget is **~10–12 GB**, which is enough for a quantized embedder + reranker + a small OCR/vision head, but not a heavy vision model. Plan the auxiliary stack to ~12 GB, or move it to a second card / smaller LLM quant if 24 GB is truly required. This is documented in §5 with the exact error.

---

## 2. Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell **Server Edition** |
| GPU memory (total) | 97,887 MiB (≈ 95.6 GiB / “96 GB”) |
| Driver | 580.159.03 |
| CUDA (driver / runtime) | 13.0 / torch built for cu130 (`torch.version.cuda = 13.0`) |
| Python | 3.12.3 |
| vLLM | 0.23.0 |
| transformers | 5.12.1 |
| torch | 2.11.0+cu130 |
| Model | meta-llama/Llama-3.1-70B-Instruct |
| Quant (baseline) | fp8 (vLLM) · cross-check: 4-bit nf4 (bitsandbytes, HF) |
| Sampling | greedy (`temperature=0`), `max_new_tokens=256` |
| max_model_len | 16384 |

Full capture: `results/baseline_env.txt`.

---

## 3. Capacity — VRAM footprint

| Config | Weights resident | KV cache (vLLM-managed) | Device used after load | % of 96 GB |
|---|---|---|---|---|
| **fp8, util=0.90** (LLM-only / off-peak) | **67.7 GiB** | 14.54 GiB → 47,648 tok | 89,609 MiB | 91.5 % |
| **fp8, util=0.875** (reserved-stack, ~12 GB held back) | 67.7 GiB | 12.17 GiB → 39,872 tok | 86,993 MiB | 88.9 % |
| _4-bit (HF) cross-check_ | _41.1 GiB (42,060 MiB)_ | _n/a (HF, no paged KV)_ | _47,683 MiB peak @8k_ | _48.7 % peak_ |

> **Reading the numbers correctly:** vLLM **pre-reserves** KV cache up to `gpu_memory_utilization`, so `nvidia-smi` sits near the cap **by design — that is not an OOM**. The *true* footprint is the **67.7 GiB** of weights (vLLM log: `Model loading took 67.7 GiB`); everything above that is KV headroom vLLM chose to claim. The 4-bit cross-check is a real (non-pre-reserved) measurement: 41.1 GiB resident, 47.7 GiB peak at 8k context.

---

## 4. Latency & throughput (vLLM fp8, single stream)

| Input size (tok) | Output tok | End-to-end s | tok/s (end-to-end) |
|---|---|---|---|
| 128 | 232 | 12.09 | 19.2 |
| 1024 | 116 | 6.25 | 18.6 |
| 4096 | 256 | 14.20 | 18.0 |
| 8192 | 256 | 14.75 | 17.4 |

> The vLLM harness times the full `generate()` call (prefill + decode), so the tok/s above is **end-to-end**, not a pure decode rate; it does not instrument TTFT separately. For a TTFT/decode split, see the HF cross-check below — its TTFT row shows prefill cost scaling cleanly with context (0.5 s → 3.8 s from 128 → 8192 tokens).

**HF 4-bit cross-check (TTFT + decode split):**

| Input size | n_in | TTFT (s) | Decode tok/s | Peak VRAM |
|---|---|---|---|---|
| 128 | 262 | 0.515 | 23.4 | 43,323 MB |
| 1024 | 1155 | 0.608 | 23.8 | 43,771 MB |
| 4096 | 4224 | 1.847 | 22.3 | 45,859 MB |
| 8192 | 8328 | 3.763 | 21.2 | 47,683 MB |

Single-stream rates (fp8 ~17–19 tok/s end-to-end; 4-bit ~21–24 tok/s decode) are **modest** — single-user latency is not where this card wins.

---

## 5. Concurrency — the headline

**Continuous-batching throughput sweep (fp8, ~4096-token requests):**

| Batch | Aggregate tok/s |
|---|---|
| 1 | 19.0 |
| 8 | 144.1 |
| 16 | 269.7 |

Aggregate throughput **rises ~14×** from batch 1 → 16 — the single-card-serves-many-users mechanism, confirmed.

**vLLM "Maximum concurrency" (each request holding the *full* 16,384-token context — worst case):**

| Config | KV cache | Max concurrency @ 16k/req |
|---|---|---|
| util=0.90 (LLM-only) | 47,648 tok | **2.91×** |
| util=0.875 (reserved-stack, ~12 GB held back) | 39,872 tok | **2.43×** |
| util=0.75 (24 GB reserved) | 0.3 GiB | **engine failed to start — see below** |

> **What "2.91×" means vs the ~3–8 concurrent target.** vLLM's `K` is the number of requests that fit **if every one simultaneously pins a maxed 16,384-token context** — the pathological worst case. Real tender-desk turns are far shorter. Concurrency is really `KV_tokens ÷ tokens_per_request`:
>
> | Avg tokens / request | Concurrent @ util=0.90 (47,648 tok) | Concurrent @ util=0.875 (39,872 tok) |
> |---|---|---|
> | ~2,000 (typical Q&A turn) | ~24 | ~20 |
> | ~4,500 (heavy 4k-context turn) | ~10 | ~9 |
> | 16,384 (full window, worst case) | 2.9 (`K`) | 2.4 (`K`) |
>
> So the **~3–8 truly-concurrent** target is met with margin at any realistic request size, and the **20–25 registered users** (with think-time, ~3–8 active) is reachable when turns average ≤ ~2k tokens. The card only falls below the target if many users *simultaneously* submit maxed 16k contexts — which production would shed to a triage model anyway.

### The reserved-stack failure (util=0.75, 24 GB reserved) — exact error

```
INFO  gpu_worker.py:480] Available KV cache memory: 0.3 GiB
ValueError: To serve at least one request with the model's max seq len (16384),
  5.0 GiB KV cache is needed, which is larger than the available KV cache memory (0.3 GiB).
  Based on the available memory, the estimated maximum model length is 976.
RuntimeError: Engine core initialization failed.
```

fp8 weights (67.7 GiB) + vLLM overhead consume essentially all of the `0.75 × 95.6 = 71.7 GiB` budget. **Conclusion:** the auxiliary stack must fit in **~10–12 GB** co-resident with fp8 70B at 16k (the util=0.875 run above is exactly that scenario and works, giving 2.43× / ~20 concurrent at 2k-token turns). Reserving a full 24 GB requires either a smaller LLM quant (4-bit vLLM weights ≈ 38 GiB) or a second GPU. Log preserved at `results/vllm_fp8_util075.log`.

---

## 6. Correctness evidence (tender-executive tasks)

The model cites specific clauses/figures and flags missing information rather than hallucinating.

**(a) EMD question — fp8, ~1024-token context** (exact figures + clause quoted):
> The EMD amount is INR 18,42,000, which is 1% of the tender value. The acceptable modes of submission are via Bank Guarantee (BG) or online payment on the GeM/CPPP portal. The EMD should be valid for 180 days.
> Reference: … "Earnest Money Deposit (EMD): INR 18,42,000 (1% of tender value), via BG or online payment on the GeM/CPPP portal; valid 180 days."

**(b) Eligibility — fp8, ~128-token context** (correctly flags missing info instead of inventing criteria):
> **Technical Eligibility**: The NIT does not provide specific technical eligibility criteria such as experience, turnover, or personnel requirements. **We need more information** to assess our technical eligibility.

**(c) Onerous commercial clauses — fp8, ~8192-token context** (cites clause numbers + gives risk pricing):
> Liquidated Damages (LD): Clause C2 … 0.5% of contract value per week of delay … ceiling of 10%. … Risk pricing: 2-3% of the contract value.
> Retention: Clause C4 … 10% of each running account bill … released against the defect liability bank guarantee.

**Cross-check — HF 4-bit, same EMD question** (matches fp8, cites the same clause):
> The EMD amount is INR 18,42,000, which is 1% of the tender value. … via Bank Guarantee (BG) or online payment on the GeM/CPPP portal. … valid for 180 days.

fp8 and 4-bit agree on the substantive answers, supporting fp8 as production-equivalent for these tasks.

---

## 7. Caveats (read before quoting numbers)

- **fp8 ≈ production-equivalent, not full bf16 quality.** fp8 is the production path; it is not the full-precision ceiling. Treat answer quality as “production-grade,” not “best-possible.”
- **Spot VM, Server-Edition card.** Single-stream tok/s (~17–24) is **indicative, not final** — clocks/thermals on a Spot instance vary. Re-confirm on the committed instance type before publishing latency SLAs.
- **Latency is not the strength; throughput is.** ~20 tok/s single-stream is modest. The case rests on continuous-batching aggregate throughput (270 tok/s @ batch 16) and KV-cache concurrency, not per-request speed.
- **Quick/short turns would route to a smaller triage model** in production, reserving the 70B for reasoning-heavy bid analysis — this keeps the 70B’s KV budget for the requests that need it and lifts effective concurrency.
- **Reserved-stack budget is ~12 GB, not 24 GB** (see §5). Size the vision-OCR + embedder + reranker to ~12 GB co-resident, or plan a second card.
- **“Maximum concurrency K” is a worst-case floor** (full 16k context per request), not the expected interactive concurrency (§5 table).

---

## 8. Conclusion

**Validated:** one 96 GB RTX PRO 6000 Blackwell runs Llama-3.1-70B (fp8, vLLM, 16k) and serves the ~3–8 concurrent / 20–25-user interactive load via continuous batching — provided the co-resident vision/embedder/reranker stack is budgeted to ~12 GB rather than 24 GB.

---

### Artifact index
- `baseline_env.txt` — environment capture
- `vllm_fp8_util090.{log,json}` — LLM-only baseline (util 0.90)
- `vllm_fp8_util0875.{log,json}` — reserved-stack run (~12 GB held back, util 0.875)
- `vllm_fp8_util075.log` — **failed** 24 GB-reserved run (negative finding, exact error)
- `vanilla_4bit.{log,json}` — HF 4-bit cross-check
- `BASELINE.md` — this report
