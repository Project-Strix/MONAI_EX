from typing import Callable

import os
import json
import numpy as np
from functools import partial
import matplotlib as mpl
from numpy.lib.function_base import average

mpl.use("Agg")
import matplotlib.pyplot as plt

from monai.utils import exact_version, optional_import

Metric, _ = optional_import("ignite.metrics", "0.4.7", exact_version, "Metric")
EpochMetric, _ = optional_import(
    "ignite.metrics", "0.4.7", exact_version, "EpochMetric"
)

from sklearn.metrics import (
    f1_score,
    roc_curve,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


def cutoff_youdens(fpr, tpr, thresholds):
    scores = tpr - fpr
    orders = sorted(zip(scores, thresholds, range(len(scores))))
    return orders[-1][1], orders[-1][-1]


def compute_CI(y_true, y_pred, statistic: Callable, CI_index=0.95, seed=1):
    n_bootstraps = 1000
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = statistic(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int((1.0 - CI_index) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1.0 - (1.0 - CI_index) / 2 * len(sorted_scores))]
    CI = [confidence_lower, confidence_upper]

    return CI


def save_roc_curve_fn(
    y_preds, y_targets, save_dir, average_type="macro", CI=True, suffix=""
):
    print("y_targets, y_preds shapes:", y_targets.shape, y_preds.shape)
    y_true, y_pred = y_targets.numpy(), y_preds.numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    fig = plt.figure(figsize=(8, 5), dpi=200)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 16
    plt.plot(fpr, tpr, label=f"Test AUC={auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right", fontsize="x-large")
    plt.savefig(os.path.join(save_dir, f"roc{suffix}.png"))

    np.savetxt(
        os.path.join(save_dir, f"roc_scores{suffix}.csv"),
        np.stack([thresholds, fpr, tpr]).transpose(),
        delimiter=",",
        fmt="%f",
        header="Threshold,FPR,TPR",
        comments="",
    )

    best_th, best_idx = cutoff_youdens(fpr, tpr, thresholds)
    y_pred_binary = (y_pred > best_th).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average=average_type
    )
    acc = accuracy_score(y_true, y_pred_binary)
    bal_acc = balanced_accuracy_score(y_true, y_pred_binary)

    auc_ci = acc_ci = bal_acc_ci = ppv_ci = tpr_ci = f1_ci = 0
    if CI:
        ci = 0.95
        auc_ci = compute_CI(y_true, y_pred_binary, roc_auc_score, CI_index=ci)
        acc_ci = compute_CI(y_true, y_pred_binary, accuracy_score, CI_index=ci)
        bal_acc_ci = compute_CI(y_true, y_pred_binary, balanced_accuracy_score, CI_index=ci)
        ppv_ci = compute_CI(y_true, y_pred_binary, partial(precision_score, average=average_type), CI_index=ci)
        tpr_ci = compute_CI(y_true, y_pred_binary, partial(recall_score, average=average_type), CI_index=ci)
        f1_ci = compute_CI(y_true, y_pred_binary, partial(f1_score, average=average_type), CI_index=ci)

    print("Best precision, recall, f1:", precision, recall, f1)
    with open(os.path.join(save_dir, f"classification_results{suffix}.json"), "w") as f:
        json.dump(
            {
                "AUC": float(auc),
                "AUC_CI": auc_ci,
                "Best threshold": float(best_th),
                "Precision(PPV)": float(precision),
                "PPV_CI": ppv_ci,
                "Recall(TPR)": float(recall),
                "Recall_CI": tpr_ci,
                "Accuracy": float(acc),
                "ACC_CI": acc_ci,
                "Balanced accuracy": float(bal_acc),
                "Bal_ACC_CI": bal_acc_ci,
                "f1": float(f1),
                "F1_CI": f1_ci,
            },
            f,
            indent=2,
        )

    return 0


def save_roc_curve_multiclass_fn(y_preds, y_targets, save_dir):
    assert y_preds.shape == y_targets.shape, (
        "y_preds, y_targets shape should be same,"
        f"but got {y_preds.shape} != {y_targets.shape}"
    )
    for idx in range(y_preds.shape[-1]):
        save_roc_curve_fn(
            y_preds[:, idx],
            y_targets[:, idx],
            save_dir,
            average_type="macro",
            CI=True,
            suffix=f"-class{idx}",
        )
    return 0


class DrawRocCurve(EpochMetric):
    """Draw ROC curve for binary classification task.

    Args:
        EpochMetric ([type]): [description]
    """

    def __init__(
        self,
        save_dir,
        output_transform=lambda x: x,
        check_compute_fn: bool = False,
        is_multilabel: bool = False,
    ):
        if is_multilabel:
            super(DrawRocCurve, self).__init__(
                partial(save_roc_curve_multiclass_fn, save_dir=save_dir),
                output_transform=output_transform,
                check_compute_fn=check_compute_fn,
            )
        else:
            super(DrawRocCurve, self).__init__(
                partial(save_roc_curve_fn, save_dir=save_dir, average_type="binary"),
                output_transform=output_transform,
                check_compute_fn=check_compute_fn,
            )
