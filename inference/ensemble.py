"""Model ensemble for inference."""

import torch
import torch.nn as nn
from typing import List, Dict
from pathlib import Path


class ModelEnsemble:
    """Weighted average ensemble of multiple models."""

    def __init__(self, models: List[nn.Module], weights: List[float] = None):
        self.models = models
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
        assert len(self.weights) == len(self.models)
        assert abs(sum(self.weights) - 1.0) < 1e-6

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run ensemble prediction."""
        predictions = []
        for model, weight in zip(self.models, self.weights):
            model.eval()
            pred = model(x)
            predictions.append(pred * weight)

        return torch.stack(predictions).sum(dim=0)


def load_ensemble(
    checkpoint_paths: List[str],
    device: torch.device,
    weights: List[float] = None,
) -> ModelEnsemble:
    """Load ensemble from multiple checkpoints."""
    from evaluate import load_model

    models = []
    for path in checkpoint_paths:
        model, _ = load_model(path, device)
        models.append(model)

    return ModelEnsemble(models, weights)
