"""Post-prediction transition counts; target labels never enter a readout."""
from __future__ import annotations

import numpy as np


def transition_counts(base: np.ndarray, proposal: np.ndarray, target: np.ndarray,
                      classes: int, chunk_size: int = 1_000_000) -> np.ndarray:
    """Count [old prediction, proposed prediction, ground truth], including ties."""
    if classes < 1 or chunk_size < 1:
        raise ValueError("Positive class count and chunk size are required.")
    base, proposal, target = map(np.asarray, (base, proposal, target))
    if base.shape != proposal.shape or base.shape != target.shape:
        raise ValueError("Prediction and mask shapes must agree.")
    if any(not np.issubdtype(values.dtype, np.integer)
           for values in (base, proposal, target)):
        raise ValueError("Class indices must be integers.")
    base, proposal, target = (values.reshape(-1) for values in (base, proposal, target))
    counts = np.zeros(classes ** 3, dtype=np.int64)
    for start in range(0, target.size, chunk_size):
        end = start + chunk_size
        truth = target[start:end]
        valid = (truth >= 0) & (truth < classes)
        old, new, truth = (values[start:end][valid].astype(np.int64)
                           for values in (base, proposal, target))
        if np.any((old < 0) | (old >= classes) | (new < 0) | (new >= classes)):
            raise ValueError("Invalid prediction on a scored pixel.")
        encoded = (old * classes + new) * classes + truth
        counts += np.bincount(encoded, minlength=classes ** 3)
    return counts.reshape(classes, classes, classes)


def transition_summary(counts: np.ndarray) -> dict:
    counts = np.asarray(counts)
    if (counts.ndim != 3 or len(set(counts.shape)) != 1
            or not counts.shape[0] or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)):
        raise ValueError("Expected a nonnegative cubic integer transition tensor.")
    old, new, truth = np.indices(counts.shape)
    changed = old != new
    beneficial = changed & (new == truth)
    harmful = changed & (old == truth)
    wrong_to_wrong = changed & (old != truth) & (new != truth)
    return {"valid": int(counts.sum()), "changed": int(counts[changed].sum()),
            "beneficial": int(counts[beneficial].sum()),
            "harmful": int(counts[harmful].sum()),
            "wrong_to_wrong": int(counts[wrong_to_wrong].sum()),
            "base_confusion": counts.sum(axis=1).T.tolist(),
            "proposal_confusion": counts.sum(axis=0).T.tolist()}
