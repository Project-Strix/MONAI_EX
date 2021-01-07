import os
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from ignite.metrics import EpochMetric, Metric
from ignite.contrib.metrics.roc_auc import roc_auc_curve_compute_fn, roc_auc_compute_fn

def save_roc_curve_fn(y_preds, y_targets, save_dir):
    fpr, tpr, thresholds = roc_auc_curve_compute_fn(y_preds, y_targets)
    auc = roc_auc_compute_fn(y_preds, y_targets)

    fig = plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(fpr, tpr, label=f'Test AUC={auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.savefig(os.path.join(save_dir, 'roc.png'))

    np.savetxt(
        os.path.join(save_dir, 'roc_scores.csv'),
        np.stack([fpr, tpr]).transpose(),
        delimiter=',',
        fmt='%f',
        header='FPR,TPR',
        comments=''
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
        check_compute_fn: bool = False
    ):
        super(DrawRocCurve, self).__init__(
            partial(save_roc_curve_fn, save_dir=save_dir),
            output_transform=output_transform, 
            check_compute_fn=check_compute_fn
        )

