def model_fit_quality(training_accuracy: float, test_accuracy: float) -> int:
    if training_accuracy - test_accuracy > 0.2:  # Overfitting
        return 1
    elif training_accuracy < 0.7 and test_accuracy < 0.7:  # Underfitting
        return -1
    else:
        return 0


print(model_fit_quality(0.95, 0.65))  # 1
