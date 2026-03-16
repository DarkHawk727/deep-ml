import numpy as np


def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    N, C = preds.shape
    one_hot_target = np.zeros_like(preds)
    one_hot_target[np.arange(N), target] = 1
    per_sample_loss = -np.sum(one_hot_target * np.log(preds + 1e-15), axis=1)
    grad = preds - one_hot_target
    if reduction == "mean":
        loss = np.mean(per_sample_loss)
        grad /= N
    elif reduction == "sum":
        loss = np.sum(per_sample_loss)
    elif reduction == "none":
        loss = per_sample_loss
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    return loss, grad
