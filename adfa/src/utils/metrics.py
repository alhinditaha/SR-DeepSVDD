import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

def compute_confusion_stats(y_true, y_pred):
    y = np.asarray(y_true, int); p = np.asarray(y_pred, int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0,1]).ravel()
    eps = 1e-12
    recall = tp / (tp + fn + eps)        # DR/TPR
    precision = tp / (tp + fp + eps)     # PPV
    specificity = tn / (tn + fp + eps)   # TNR
    far = fp / (fp + tn + eps)
    acc = accuracy_score(y, p)
    bacc = 0.5 * (recall + specificity)
    f1 = 2*precision*recall / (precision + recall + eps)
    return dict(FAR=float(far), ACC=float(acc), BACC=float(bacc), F1=float(f1),
                DR=float(recall), PREC=float(precision), SPEC=float(specificity))

def per_type_stats(y_true, g, y_pred):
    """
    Precision/Recall/F1/Specificity/Balanced-Acc for each anomaly type (1 and 2).
    Slice = normals + that specific anomaly type.
    """
    y = np.asarray(y_true, int); gg = np.asarray(g, int); p = np.asarray(y_pred, int)
    eps=1e-12
    out={}
    for t, tag in [(1,'T1'), (2,'T2')]:
        mask = (gg==0) | (gg==t)
        ym = y[mask]; pm = p[mask]
        tn, fp, fn, tp = confusion_matrix(ym, pm, labels=[0,1]).ravel()
        rec  = tp / (tp + fn + eps)
        prec = tp / (tp + fp + eps)
        spec = tn / (tn + fp + eps)
        f1   = 2*prec*rec / (prec + rec + eps)
        bacc = 0.5*(rec + spec)
        out.update({f'DR_{tag}':float(rec), f'PREC_{tag}':float(prec),
                    f'F1_{tag}':float(f1), f'SPEC_{tag}':float(spec), f'BACC_{tag}':float(bacc)})
    return out

def auc_from_scores(y_true, scores):
    y = np.asarray(y_true, int)
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float('nan')
