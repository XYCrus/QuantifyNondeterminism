"""
RQ2: Benchmark latency overhead of deterministic (batch-invariant) execution.

Compares three execution modes on GPT-2 Large:
  - Eager: standard PyTorch attention
  - SDPA: Flash Attention / Scaled Dot-Product Attention
  - Batch Invariant: custom Triton kernels with fixed accumulation order
"""
import torch
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from BatchInvariant import set_batch_invariant_mode

# --- CONFIGURATION ---
MODEL_ID = "gpt2-large"
DEVICE = "cuda"
OUTPUT_FILE = os.path.join("outputs", "rq2_results.json")
os.makedirs("outputs", exist_ok=True)
ITERATIONS_PER_RUN = 20  # forward passes per timing measurement

# Same 10 prompts as RQ1 for consistency
PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the world.",
    "To be or not to be, that is the question.",
    "The grand unifying theory of physics remains elusive.",
    "Python is a popular programming language for data science.",
    "In the beginning God created the heaven and the earth.",
    "A journey of a thousand miles begins with a single step.",
    "Quantum mechanics describes nature at the smallest scales.",
    "The stock market experienced a significant volatility event.",
    "Climate change poses a significant threat to global stability."
]


def benchmark_pass(model, inputs, enable_invariant, iterations):
    """
    Time forward pass latency with optional batch-invariant mode.
    
    Uses CUDA events for precise GPU timing.
    Returns average latency in milliseconds.
    """
    ctx = set_batch_invariant_mode(enable_invariant)
    
    with ctx:
        # Warmup passes to stabilize GPU clocks and cache
        for _ in range(3):
            with torch.no_grad():
                model(**inputs)
        torch.cuda.synchronize()
        
        # Timed passes using CUDA events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            with torch.no_grad():
                model(**inputs)
        end_event.record()
        torch.cuda.synchronize()
        
    return start_event.elapsed_time(end_event) / iterations


def run_experiment():
    """Benchmark all three execution modes across 10 prompts."""
    print(f"Loading Models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Eager attention baseline (standard PyTorch)
    model_eager = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    
    # SDPA / Flash Attention reference (fastest non-deterministic)
    model_fast = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map=DEVICE, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    results = []
    
    print("\n--- Starting RQ2 Multi-Run Benchmark ---")
    
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[Run {i+1}/{len(PROMPTS)}] Benchmarking prompt...")
        
        # Batch of 8 to simulate realistic inference server load
        inputs = tokenizer([prompt] * 8, return_tensors="pt", padding=True).to(DEVICE)
        
        # Benchmark each mode
        t_eager = benchmark_pass(model_eager, inputs, False, ITERATIONS_PER_RUN)
        t_inv = benchmark_pass(model_eager, inputs, True, ITERATIONS_PER_RUN)  # invariant uses eager model
        t_fast = benchmark_pass(model_fast, inputs, False, ITERATIONS_PER_RUN)
        
        # Compute overhead percentages
        ovh_vs_eager = ((t_inv - t_eager) / t_eager) * 100
        ovh_vs_fast = ((t_inv - t_fast) / t_fast) * 100
        
        print(f"  Eager: {t_eager:.2f}ms | Invariant: {t_inv:.2f}ms | Fast: {t_fast:.2f}ms")
        print(f"  Overhead vs Eager: {ovh_vs_eager:.2f}%")
        
        results.append({
            "run_id": i + 1,
            "latency_eager_ms": t_eager,
            "latency_invariant_ms": t_inv,
            "latency_fast_ms": t_fast,
            "overhead_vs_eager_pct": ovh_vs_eager,
            "overhead_vs_fast_pct": ovh_vs_fast
        })

    # Aggregate overhead statistics
    avg_ovh_eager = np.mean([r["overhead_vs_eager_pct"] for r in results])
    avg_ovh_fast = np.mean([r["overhead_vs_fast_pct"] for r in results])
    
    summary = {
        "model": MODEL_ID,
        "batch_size": 8,
        "iterations_per_run": ITERATIONS_PER_RUN,
        "avg_overhead_vs_eager": float(avg_ovh_eager),
        "avg_overhead_vs_fast": float(avg_ovh_fast),
        "detailed_runs": results
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nExperiment Complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiment()
