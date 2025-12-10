"""
RQ1: Quantify batch-induced inference non-determinism in Transformer LLMs.

Compares solo vs batched inference logits under standard and batch-invariant modes.
Demonstrates that floating-point non-associativity in attention causes divergence.
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
OUTPUT_FILE = os.path.join("outputs", "rq1_results.json")
OS_MAKER = os.makedirs("outputs", exist_ok=True)

# 10 diverse prompts covering various domains
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

# Long distractor to force padding misalignment on target prompt
DISTRACTOR = (
    "This is a significantly longer distractor sentence intended to ensure that "
    "the target prompt is heavily padded within the batch. By checking this, "
    "we introduce the misalignment in memory layout that causes non-determinism."
) * 5


def setup_model():
    """Load GPT-2 Large with eager attention (no Flash Attention)."""
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        low_cpu_mem_usage=True,
        attn_implementation="eager"  # standard PyTorch attention, not Flash
    )
    return model, tokenizer


def get_logits(model, tokenizer, prompts, target_idx, enable_invariant):
    """
    Run inference and extract logits at last real token position.
    
    Args:
        prompts: list of input strings
        target_idx: which prompt in the batch to extract logits from
        enable_invariant: whether to use batch-invariant Triton kernels
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    
    # Find last real token index (before padding)
    target_len = len(tokenizer.encode(prompts[target_idx]))
    last_real_token_index = target_len - 1
    
    ctx = set_batch_invariant_mode(enable_invariant)
    with ctx:
        with torch.no_grad():
            outputs = model(**inputs)
            
    # Extract logit vector at last real token for the target prompt
    return outputs.logits[target_idx, last_real_token_index, :].float()


def run_experiment():
    """Main experiment loop: compare solo vs batched inference under both modes."""
    model, tokenizer = setup_model()
    results = []
    
    print("\n--- Starting RQ1 Multi-Run Experiment ---")
    
    for i, prompt in enumerate(PROMPTS):
        print(f"\n[Run {i+1}/{len(PROMPTS)}] Processing prompt...")
        
        # Standard mode: measure divergence between solo and batched
        logits_solo_std = get_logits(model, tokenizer, [prompt], 0, False)
        logits_batch_std = get_logits(model, tokenizer, [prompt, DISTRACTOR], 0, False)
        diff_std = (logits_solo_std - logits_batch_std).abs().max().item()
        
        # Batch-invariant mode: should achieve zero divergence
        logits_solo_inv = get_logits(model, tokenizer, [prompt], 0, True)
        logits_batch_inv = get_logits(model, tokenizer, [prompt, DISTRACTOR], 0, True)
        diff_inv = (logits_solo_inv - logits_batch_inv).abs().max().item()
        
        print(f"  Standard Diff:  {diff_std:.10f}")
        print(f"  Invariant Diff: {diff_inv:.10f}")
        
        results.append({
            "run_id": i + 1,
            "prompt_excerpt": prompt[:30] + "...",
            "diff_standard": diff_std,
            "diff_invariant": diff_inv,
            "success": diff_inv < 1e-4  # bitwise identity threshold
        })

    # Aggregate statistics
    avg_std = np.mean([r["diff_standard"] for r in results])
    avg_inv = np.mean([r["diff_invariant"] for r in results])
    
    summary = {
        "model": MODEL_ID,
        "precision": "bfloat16",
        "num_runs": len(PROMPTS),
        "avg_diff_standard": float(avg_std),
        "avg_diff_invariant": float(avg_inv),
        "detailed_runs": results
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nExperiment Complete. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiment()
