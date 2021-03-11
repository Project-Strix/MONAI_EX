import os
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from functools import partial

from ignite.metrics import EpochMetric, Metric
from ignite.contrib.metrics.roc_auc import roc_auc_curve_compute_fn, roc_auc_compute_fn
from sklearn.metrics import precision_recall_curve, average_precision_score


def save_pr_curve_fn(y_preds, y_targets, save_dir):
    precisions, recalls, thresholds = precision_recall_curve(y_targets.numpy(), y_preds.numpy())
    ap = average_precision_score(y_targets.numpy(), y_preds.numpy())

    fig = plt.figure(figsize=(8, 5), dpi=200)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 16
    plt.plot(recalls, precisions, label=f'Test AP={ap:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right', fontsize='x-large')
    plt.savefig(os.path.join(save_dir, 'prc.png'))

    np.savetxt(
        os.path.join(save_dir, 'prc_scores.csv'),
        np.stack([thresholds, recalls, precisions]).transpose(),
        delimiter=',',
        fmt='%f',
        header='Threshold,Recall,Precision',
        comments=''
    )

class DrawPRCurve(EpochMetric):
    """Draw Precision-Recall curve for binary classification task.

    Args:
        EpochMetric ([type]): [description]
    """
    def __init__(
        self,
        save_dir,
        output_transform=lambda x: x, 
        check_compute_fn: bool = False
    ):
        super(DrawPRCurve, self).__init__(
            partial(save_pr_curve_fn, save_dir=save_dir),
            output_transform=output_transform,
            check_compute_fn=check_compute_fn
        )
