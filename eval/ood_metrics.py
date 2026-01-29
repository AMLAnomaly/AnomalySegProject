import numpy as np
from sklearn.metrics import roc_curve

def fpr_at_95_tpr(scores, labels):

    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return fpr[-1]
    return fpr[idx[0]]

def calc_metrics(*args, **kwargs):
    pass

def plot_roc(*args, **kwargs):
    pass

def plot_pr(*args, **kwargs):
    pass

def plot_barcode(*args, **kwargs):
    pass
