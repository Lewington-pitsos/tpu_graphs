import numpy as np

def slowdown(y_true, y_pred, k):
    y_fastest = np.min(y_true)

    top_k_indices = y_pred[:k]

    top_k_predicted_times = np.min(y_true[top_k_indices])

    ratio = top_k_predicted_times / y_fastest

    return ratio - 1

def speed_score(y_true, y_pred, k):
    return 1 - slowdown(y_true, y_pred, k)

# print(speed_score(np.array([1.24, 0.231, 4.2, 2.01]), np.array([2, 0, 3, 1]), 1))
