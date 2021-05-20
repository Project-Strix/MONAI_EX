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
    precision_recall_fscore_support
)


def cutoff_youdens(fpr, tpr, thresholds):
    scores = tpr-fpr
    orders = sorted(zip(scores,thresholds, range(len(scores))))
    return orders[-1][1], orders[-1][-1]


def save_roc_curve_fn(y_preds, y_targets, save_dir, is_multilabel=False):
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
    plt.savefig(os.path.join(save_dir, 'roc.png'))

    np.savetxt(
        os.path.join(save_dir, 'roc_scores.csv'),
        np.stack([thresholds, fpr, tpr]).transpose(),
        delimiter=',',
        fmt='%f',
        header='Threshold,FPR,TPR',
        comments=''
    )

    average_type = 'binary' if not is_multilabel else None
    best_th, best_idx = cutoff_youdens(fpr, tpr, thresholds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_targets.numpy(), y_preds.numpy()>best_th, average=average_type)
    acc = accuracy_score(y_targets.numpy(), y_preds.numpy()>best_th)
    print('Best precision, recall, f1:', precision, recall, f1)
    with open(os.path.join(save_dir, 'classification_results.json'), 'w') as f:
        json.dump( {
            'AUC': float(auc),
            'Best threshold': float(best_th),
            'Precision': float(precision),
            'Recall': float(recall),
            'Accuracy': float(acc),
            'False positive rate': float(fpr[best_idx]),
            'True positive rate': float(tpr[best_idx]),
            'f1': float(f1),
            }, f, indent=2 )

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
        super(DrawRocCurve, self).__init__(
            partial(save_roc_curve_fn, save_dir=save_dir, is_multilabel=is_multilabel),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn
        )
