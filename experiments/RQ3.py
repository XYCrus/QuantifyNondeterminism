"""
RQ3: Evaluate stochastic rounding as a mitigation for low-precision training instability.

Trains ResNet-18 on CIFAR-10 under three rounding modes:
  - fp32: full precision baseline
  - nearest: round-to-nearest (biased, can lose gradient signal)
  - stochastic: probabilistic rounding (unbiased, preserves expected gradient)

Stochastic rounding provides statistical stability without strict determinism.
"""
import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class StochasticRoundFn(torch.autograd.Function):
    """
    Stochastic rounding: round up/down probabilistically based on fractional part.
    E[SR(x)] = x, making gradients unbiased in expectation.
    """
    @staticmethod
    def forward(ctx, x, scale):
        scaled = x * scale
        base = torch.floor(scaled)
        prob = scaled - base  # fractional part = probability of rounding up
        noise = torch.rand_like(prob)
        rounded = base + (noise < prob).float()
        return rounded / scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: pass gradients unchanged
        return grad_output, None


class NearestRoundFn(torch.autograd.Function):
    """
    Standard round-to-nearest: deterministic but biased.
    Small gradients can be systematically lost.
    """
    @staticmethod
    def forward(ctx, x, scale):
        return torch.round(x * scale) / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def round_tensor(x: torch.Tensor, scale: float, mode: str) -> torch.Tensor:
    """Apply rounding based on mode."""
    if mode == "stochastic":
        return StochasticRoundFn.apply(x, scale)
    if mode == "nearest":
        return NearestRoundFn.apply(x, scale)
    return x  # fp32: no rounding


class QuantConv2d(nn.Conv2d):
    """
    Quantized Conv2d that rounds activations, weights, and biases.
    Simulates fixed-point arithmetic with configurable fractional bits.
    """
    def __init__(self, *args, frac_bits: int, mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.frac_bits = frac_bits
        self.mode = mode

    def forward(self, x):
        scale = float(1 << self.frac_bits)  # 2^frac_bits
        q_weight = round_tensor(self.weight, scale, self.mode)
        q_bias = round_tensor(self.bias, scale, self.mode) if self.bias is not None else None
        q_x = round_tensor(x, scale, self.mode)
        return F.conv2d(
            q_x,
            q_weight,
            q_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def swap_convs(module: nn.Module, mode: str, frac_bits: int) -> None:
    """Recursively replace all Conv2d layers with QuantConv2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = QuantConv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                frac_bits=frac_bits,
                mode=mode,
            )
            new_conv.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_conv.bias.data.copy_(child.bias.data)
            setattr(module, name, new_conv)
        else:
            swap_convs(child, mode, frac_bits)


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(mode: str, frac_bits: int, device: torch.device, base_state: dict) -> nn.Module:
    """Build ResNet-18 with optional quantized convolutions."""
    model = models.resnet18(weights=None, num_classes=10)
    if mode in {"stochastic", "nearest"}:
        swap_convs(model, mode=mode, frac_bits=frac_bits)
    model.load_state_dict(base_state, strict=False)
    model.to(device)
    return model


def get_loaders(data_root: str, batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train/test data loaders with standard augmentation."""
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    train_set = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    )
    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model, loader, device, criterion, optimizer) -> tuple[float, float]:
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    total = 0
    correct = 0
    running_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        correct += (out.argmax(dim=1) == labels).sum().item()
    return running_loss / len(loader), correct / total


def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    """Evaluate on validation set, return average loss and accuracy."""
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item()
            total += labels.size(0)
            correct += (out.argmax(dim=1) == labels).sum().item()
    return running_loss / len(loader), correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet18 with low-precision rounding")
    parser.add_argument("--data-root", default=os.path.join(os.getcwd(), "data"))
    parser.add_argument("--modes", nargs="+", choices=["fp32", "nearest", "stochastic"], default=["fp32", "nearest", "stochastic"])
    parser.add_argument("--frac-bits", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default=os.path.join(os.getcwd(), "outputs", "rq3_results.json"))
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = get_loaders(args.data_root, args.batch_size, args.workers)

    # Create base weights for consistent initialization across modes
    base_model = models.resnet18(weights=None, num_classes=10)
    base_state = base_model.state_dict()
    del base_model

    runs = []
    total_start = time.time()

    for idx, mode in enumerate(args.modes):
        # Use different seed per mode to avoid identical training trajectories
        set_seed(args.seed + idx)
        model = build_model(mode, args.frac_bits, device, base_state)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

        history = []
        best_val = 0.0
        best_state = None
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc = evaluate(model, test_loader, device, criterion)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })
            if val_acc > best_val:
                best_val = val_acc
                if args.save_checkpoint:
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.cuda.empty_cache()

        elapsed = time.time() - start

        # Save best checkpoint if requested
        if args.save_checkpoint and best_state is not None:
            ckpt_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(best_state, os.path.join(ckpt_dir, f"best_{mode}.pt"))

        runs.append({
            "mode": mode,
            "history": history,
            "best_val_acc": best_val,
            "final_val_acc": history[-1]["val_acc"],
            "elapsed_sec": elapsed,
        })
        del model
        torch.cuda.empty_cache()

    payload = {
        "device": str(device),
        "modes": args.modes,
        "frac_bits": args.frac_bits,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "workers": args.workers,
        "seed": args.seed,
        "total_elapsed_sec": time.time() - total_start,
        "runs": runs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
