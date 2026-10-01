"""Measure validation NMSE, parameter count, approximate MACs and batch-1 latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.cost2100 import COST2100Dataset
from src.metrics import sample_nmse_linear
from src.models.lite_dpas import LiteDPASCsiNet, LiteDPASCsiNetV2, count_parameters


def approximate_macs(model: torch.nn.Module, sample: torch.Tensor) -> int:
    """Count Conv2d/Linear MACs; FFT and elementwise operations are excluded."""
    total = 0
    handles = []

    def count(module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, torch.nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total += output.numel() * (module.in_channels // module.groups) * kernel
        elif isinstance(module, torch.nn.Linear):
            total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return total


def measure_latency_ms(
    model: torch.nn.Module,
    sample: torch.Tensor,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, float]:
    def synchronize() -> None:
        if sample.is_cuda:
            torch.cuda.synchronize(sample.device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize()
        times = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            model(sample)
            synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "warmup_runs": warmup,
        "timed_runs": repeats,
    }


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    encoded_dim = int(checkpoint.get("encoded_dim", 512))
    model_name = checkpoint.get("model_name", "litedpas")
    model_class = LiteDPASCsiNetV2 if model_name == "litedpas_v2" else LiteDPASCsiNet
    model = model_class(encoded_dim=encoded_dim)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    data_dir = args.data_dir or args.project_root / "Data"
    val = COST2100Dataset(data_dir / "DATA_Hvalin.mat")
    count = min(args.samples, len(val))
    loader = DataLoader(Subset(val, range(count)), batch_size=64, shuffle=False, num_workers=0)
    ratios = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            ratios.append(sample_nmse_linear(batch, model(batch)).cpu())
    val_nmse_db = float(10.0 * np.log10(max(torch.cat(ratios).mean().item(), 1e-12)))

    sample = torch.zeros(1, 2, 32, 32, device=device)
    report = {
        "model": type(model).__name__,
        "encoded_dim_real_values": encoded_dim,
        "compression_ratio_convention": f"1/{2048 // encoded_dim}",
        "parameters": count_parameters(model),
        "approx_conv_linear_macs_batch1": approximate_macs(model, sample),
        "mac_count_note": "Conv2d/Linear only; excludes FFT, normalization, activations and interpolation.",
        "batch1_latency": measure_latency_ms(model, sample),
        "latency_device": str(device),
        "validation_samples": count,
        "validation_nmse_db_subset": val_nmse_db,
        "test_split_used": False,
    }
    out_dir = args.project_root / "runs" / args.checkpoint.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "profile_validation.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved profile to: {output_path}")


if __name__ == "__main__":
    main()
