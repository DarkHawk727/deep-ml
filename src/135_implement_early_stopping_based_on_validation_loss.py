def early_stopping(val_losses: list[float], patience: int, min_delta: float) -> tuple[int, int]:
    best_epoch = 0
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            return epoch, best_epoch

    return len(val_losses) - 1, best_epoch


print(early_stopping([0.9, 0.8, 0.75, 0.77, 0.76, 0.77, 0.78], 2, 0.01)) # (4, 2)