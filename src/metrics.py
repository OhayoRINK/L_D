"""Shared metrics for CSI reconstruction comparisons."""

import torch


def sample_nmse_linear(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return one complex-equivalent NMSE ratio per sample.

    Inputs are normalized real/imag channels shaped [B, 2, 32, 32]. The
    normalized COST2100 offset is removed before measuring complex power.
    """
    if reference.shape != estimate.shape:
        raise ValueError(
            f"Reference and estimate shapes differ: {reference.shape} vs {estimate.shape}"
        )
    if reference.ndim != 4 or reference.shape[1:] != (2, 32, 32):
        raise ValueError(f"Expected [B, 2, 32, 32], got {reference.shape}")

    reference_centered = reference - 0.5
    estimate_centered = estimate - 0.5
    error_power = (estimate_centered - reference_centered).square().sum(dim=(1, 2, 3))
    reference_power = reference_centered.square().sum(dim=(1, 2, 3)).clamp_min(epsilon)
    return error_power / reference_power


def nmse_db(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """Return 10*log10(mean(sample-wise complex-equivalent NMSE))."""
    ratios = sample_nmse_linear(reference, estimate)
    return 10.0 * torch.log10(ratios.mean().clamp_min(1e-12))
