"""Metrics calculation utilities."""

from typing import Dict
import torch


def calculate_attack_success_rate(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate attack success rate."""
    correct = (predictions == targets).sum().item()
    total = len(targets)
    return correct / total if total > 0 else 0.0
