import os
import json
import numpy as np
from functools import partial
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from monai.utils import exact_version, optional_import
Metric, _ = optional_import("ignite.metrics", "0.4.4", exact_version, "Metric")
EpochMetric, _ = optional_import("ignite.metrics", "0.4.4", exact_version, "EpochMetric")

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support
)


def cutoff_youdens(fpr, tpr, thresholds):
    scores = tpr-fpr
    orders = sorted(zip(scores, thresholds, range(len(scores))))
    return orders[-1][1], orders[-1][-1]


def save_roc_curve_fn(y_preds, y_targets, save_dir, average_type='binary', suffix=''):
    print("y_targets, y_preds shapes:", y_targets.shape, y_preds.shape)
    fpr, tpr, thresholds = roc_curve(y_targets.numpy(), y_preds.numpy())
    auc = roc_auc_score(y_targets.numpy(), y_preds.numpy())

    fig = plt.figure(figsize=(8, 5), dpi=200)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 16
    plt.plot(fpr, tpr, label=f'Test AUC={auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.savefig(os.path.join(save_dir, f'roc{suffix}.png'))

    np.savetxt(
        os.path.join(save_dir, f'roc_scores{suffix}.csv'),
        np.stack([thresholds, fpr, tpr]).transpose(),
        delimiter=',',
        fmt='%f',
        header='Threshold,FPR,TPR',
        comments=''
    )

    best_th, best_idx = cutoff_youdens(fpr, tpr, thresholds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_targets.numpy(), y_preds.numpy() > best_th, average=average_type
    )
    acc = accuracy_score(y_targets.numpy(), y_preds.numpy() > best_th)
    bal_acc = balanced_accuracy_score(y_targets.numpy(), y_preds.numpy() > best_th)
    print('Best precision, recall, f1:', precision, recall, f1)
    with open(os.path.join(save_dir, f'classification_results{suffix}.json'), 'w') as f:
        json.dump({
            'AUC': float(auc),
            'Best threshold': float(best_th),
            'Precision(PPV)': float(precision),
            'Recall(TPR)': float(recall),
            'Accuracy': float(acc),
            'Balanced accuracy': float(bal_acc),
            'f1': float(f1),
            }, f, indent=2
        )

    return 0


def save_roc_curve_multiclass_fn(y_preds, y_targets, save_dir):
    assert y_preds.shape == y_targets.shape,\
        'y_preds, y_targets shape should be same,'\
        f'but got {y_preds.shape} != {y_targets.shape}'
    for idx in range(y_preds.shape[-1]):
        save_roc_curve_fn(
            y_preds[:, idx],
            y_targets[:, idx],
            save_dir,
            average_type='macro',
            suffix=f'-class{idx}'
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
        is_multilabel: bool = False
    ):
        if is_multilabel:
            super(DrawRocCurve, self).__init__(
                partial(save_roc_curve_multiclass_fn, save_dir=save_dir),
                output_transform=output_transform,
                check_compute_fn=check_compute_fn
            )
        else:
            super(DrawRocCurve, self).__init__(
                partial(save_roc_curve_fn, save_dir=save_dir, average_type='binary'),
                output_transform=output_transform,
                check_compute_fn=check_compute_fn
            )
