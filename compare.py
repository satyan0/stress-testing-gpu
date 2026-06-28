"""Pretty-print vanilla vs vLLM results side by side. Run after both profilers."""
import json
import os

import config


def load(name):
    path = os.path.join(config.RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    v = load("vanilla.json")
    x = load("vllm.json")

    if v:
        print(f"\n=== vanilla-hf  |  {v['model']}  quant={v['quant']} ===")
        print(f"model resident: {v['model_resident_mb']} MB "
              f"of {v['device_total_mb']} MB")
        print(f"{'in_tok':>8} {'TTFT s':>8} {'decode t/s':>11} {'peak MB':>9}")
        for r in v["runs"]:
            print(f"{r['target_in_tok']:>8} {str(r['ttft_s']):>8} "
                  f"{r['decode_tok_s']:>11} {r['peak_vram_mb']:>9}")

    if x:
        print(f"\n=== vllm  |  {x['model']}  quant={x['quant']} ===")
        single = [r for r in x["runs"] if r["mode"] == "single"]
        batch = [r for r in x["runs"] if r["mode"] == "batch"]
        print(f"{'in_tok':>8} {'tok/s':>8}   (single-stream)")
        for r in single:
            print(f"{r['target_in_tok']:>8} {r['tok_s']:>8}")
        print(f"{'batch':>8} {'agg tok/s':>10}   (continuous batching)")
        for r in batch:
            print(f"{r['batch']:>8} {r['tok_s']:>10}")

    print("\nWhat to look for:")
    print("  * model resident must leave clear headroom on a 96 GB card.")
    print("  * vLLM aggregate tok/s should RISE with batch size; vanilla won't.")
    print("  * read vLLM's 'Maximum concurrency' log line for users-per-card.")


if __name__ == "__main__":
    main()
