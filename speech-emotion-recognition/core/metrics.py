import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def get_metrics_df(pred_class, gt_class, model_name=None):
    metric_dict = calculate_metrics(pred_class=pred_class, gt_class=gt_class)
    metrics_df = pd.DataFrame([metric_dict]).T.round(4)

    if model_name is not None:
        metrics_df.columns = [model_name]

    return metrics_df


def weighted_accuracy(y_true, y_pred, n_classes=4):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    class_accuracies = []
    for i in range(n_classes):
        gt_class_mask = y_true == i
        pred_class_mask = y_pred == i
        class_accuracies.append((gt_class_mask * pred_class_mask).sum() / gt_class_mask.sum())

    return np.mean(class_accuracies)


def calculate_metrics(gt_class, pred_class, **kwargs):
    n_classes = 5
    metrics_dict = {
        "accuracy": accuracy_score(y_true=gt_class, y_pred=pred_class),
        "WA": weighted_accuracy(y_true=gt_class, y_pred=pred_class, n_classes=n_classes),
        "f1_macro": f1_score(y_true=gt_class, y_pred=pred_class, average="macro"),
        "f1_by_class": f1_score(y_true=gt_class, y_pred=pred_class, average=None),
        "recall_by_class": recall_score(y_true=gt_class, y_pred=pred_class, average=None),
        "precision_by_class": precision_score(y_true=gt_class, y_pred=pred_class, average=None),
    }

    return metrics_dict
