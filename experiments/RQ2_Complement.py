"""
RQ2 Complement: Benchmark deterministic vs fast convolution in CNNs.

Provides context for RQ2 by measuring the overhead of torch.use_deterministic_algorithms()
on a single Conv2d layer. CNN determinism has lower overhead than Transformer attention.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_conv(channels: int, device: torch.device, precision: str, base_state: dict) -> nn.Module:
    """Create a Conv2d layer with given weights."""
    conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
    conv.load_state_dict(base_state)
    conv.to(device)
    if precision == "fp16":
        conv.half()
    conv.eval()
    return conv


def measure_once(
    mode: str,
    batch_size: int,
    channels: int,
    h: int,
    w: int,
    iters: int,
    warmup: int,
    device: torch.device,
    precision: str,
    base_state: dict,
) -> dict:
    """
    Run one measurement pass for a given mode.
    
    Modes:
      - "fast": cuDNN benchmark enabled, non-deterministic
      - "deterministic": torch.use_deterministic_algorithms(True)
    """
    # Configure PyTorch determinism settings
    torch.backends.cudnn.benchmark = mode == "fast"
    torch.backends.cudnn.deterministic = mode == "deterministic"
    torch.use_deterministic_algorithms(mode == "deterministic")

    conv = build_conv(channels, device, precision, base_state)
    x = torch.randn(batch_size, channels, h, w, device=device)
    if precision == "fp16":
        x = x.half()

    # Warmup and timed forward passes
    with torch.no_grad():
        for _ in range(warmup):
            conv(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            conv(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latency = (time.time() - start) / iters

    # Verify determinism by comparing two consecutive outputs
    with torch.no_grad():
        y1 = conv(x)
        y2 = conv(x)
        diff = (y1 - y2).abs().max().item()

    del conv
    torch.cuda.empty_cache()
    throughput = (batch_size / latency) if latency > 0 else 0.0
    return {"latency_ms": latency * 1000.0, "max_abs_diff": diff, "throughput": throughput}


def measure(
    mode: str,
    batch_size: int,
    channels: int,
    h: int,
    w: int,
    iters: int,
    warmup: int,
    device: torch.device,
    precision: str,
    base_state: dict,
    repeats: int,
) -> dict:
    """Run multiple measurement passes and aggregate statistics."""
    lat = []
    diffs = []
    thpt = []
    for _ in range(repeats):
        metrics = measure_once(
            mode=mode,
            batch_size=batch_size,
            channels=channels,
            h=h,
            w=w,
            iters=iters,
            warmup=warmup,
            device=device,
            precision=precision,
            base_state=base_state,
        )
        lat.append(metrics["latency_ms"])
        diffs.append(metrics["max_abs_diff"])
        thpt.append(metrics["throughput"])
    return {
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_std": float(np.std(lat)),
        "latency_ms_p50": float(np.percentile(lat, 50)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
        "throughput_mean": float(np.mean(thpt)),
        "max_abs_diff_max": float(np.max(diffs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark deterministic vs fast conv")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default=os.path.join(os.getcwd(), "outputs", "rq2_complement_results.json"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create base conv weights for consistent comparison
    base_conv = nn.Conv2d(args.channels, args.channels, kernel_size=3, padding=1, bias=False)
    base_state = base_conv.state_dict()
    del base_conv

    # Benchmark both modes across all batch sizes
    modes = ["fast", "deterministic"]
    results = {}
    for bs in args.batch_sizes:
        results[str(bs)] = {}
        for mode in modes:
            metrics = measure(
                mode=mode,
                batch_size=bs,
                channels=args.channels,
                h=args.image_size,
                w=args.image_size,
                iters=args.iters,
                warmup=args.warmup,
                device=device,
                precision=args.precision,
                base_state=base_state,
                repeats=args.repeats,
            )
            results[str(bs)][mode] = metrics

    payload = {
        "device": str(device),
        "precision": args.precision,
        "batch_sizes": args.batch_sizes,
        "channels": args.channels,
        "image_size": args.image_size,
        "iters": args.iters,
        "warmup": args.warmup,
        "seed": args.seed,
        "repeats": args.repeats,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
