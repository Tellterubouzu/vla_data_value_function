from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - utility functions still testable without torch.
    torch = None  # type: ignore[assignment]


def bin_centers(num_bins: int = 201):
    if torch is None:
        return np.linspace(-1.0, 0.0, num_bins, dtype=np.float32)
    return torch.linspace(-1.0, 0.0, num_bins, dtype=torch.float32)


def logits_to_continuous(logits, num_bins: int = 201):
    if torch is None:
        raise ModuleNotFoundError("torch is required for logits_to_continuous")

    probs = torch.softmax(logits, dim=-1)
    centers = bin_centers(num_bins=num_bins).to(logits.device, dtype=logits.dtype)
    return torch.sum(probs * centers.unsqueeze(0), dim=-1)


def mean_absolute_error(pred, target):
    if torch is None:
        pred_np = np.asarray(pred)
        target_np = np.asarray(target)
        return float(np.mean(np.abs(pred_np - target_np)))
    return torch.mean(torch.abs(pred - target))


def compute_batch_metrics(logits, v_norm, num_bins: int = 201) -> Dict[str, float]:
    if torch is None:
        raise ModuleNotFoundError("torch is required for compute_batch_metrics")

    v_hat = logits_to_continuous(logits=logits, num_bins=num_bins)
    mae = mean_absolute_error(v_hat, v_norm)
    return {
        "mae": float(mae.detach().cpu().item()),
        "v_hat_mean": float(v_hat.detach().mean().cpu().item()),
    }
