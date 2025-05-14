import numpy as np


def acc(pred, true, weight=None):
    if pred.ndim == 3:
        pred = pred.astype(np.float32).reshape(pred.shape[0], -1)
        true = true.astype(np.float32).reshape(true.shape[0], -1)

    pred_anomaly = pred - pred.mean(axis=0)
    true_anomaly = true - true.mean(axis=0)
    if weight is not None:
        numerator = np.sum(weight * pred_anomaly * true_anomaly, axis=0)
        denominator = np.sqrt(np.sum(weight * pred_anomaly**2, axis=0)) * np.sqrt(
            np.sum(weight * true_anomaly**2, axis=0)
        )
    else:
        numerator = np.sum(pred_anomaly * true_anomaly, axis=0)
        denominator = np.sqrt(np.sum(pred_anomaly**2, axis=0)) * np.sqrt(
            np.sum(true_anomaly**2, axis=0)
        )

    acc = numerator / denominator
    return np.mean(acc)
