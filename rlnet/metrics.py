from __future__ import annotations

import torch


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).to(torch.int64)
    labels_int = labels.to(torch.int64)

    tp = int(((predictions == 1) & (labels_int == 1)).sum().item())
    tn = int(((predictions == 0) & (labels_int == 0)).sum().item())
    fp = int(((predictions == 1) & (labels_int == 0)).sum().item())
    fn = int(((predictions == 0) & (labels_int == 1)).sum().item())

    total = max(tp + tn + fp + fn, 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
