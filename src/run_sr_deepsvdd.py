# src/run_sr_deepsvdd.py
# Path-robust imports: allow running from src/ or project root
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from pathlib import Path
import itertools
import numpy as np
import json

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix, average_precision_score
)

import torch

# Dataset
from src.utils.dataset_banana import (
    BananaParams, make_banana_data, plot_scatter
)
from src.base.base_dataset import DictDataset, BaseADDataset

# Models
from src.models.sr_deep_svdd import SRDeepSVDD
from src.models.deep_svdd import DeepSVDD
from src.models.svdd_linear import LinearSVDD
from src.baselines.ocsvm import OCSVMWrapper
from src.baselines.svdd_rbf import KernelSVDD

# Use the project's score_plots (works from project root or from src/)
try:
    from src.utils import score_plots
except ImportError:
    from utils import score_plots

# Minimal local plotting for the *dataset-level* train scatter
import matplotlib
matplotlib.use("Agg")  # safe for headless runs
import matplotlib.pyplot as plt

# Optional stats libs
try:
    from scipy.stats import wilcoxon as _wilcoxon, ttest_rel as _ttest_rel
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False
    _wilcoxon = None
    _ttest_rel = None


# ---- CSV schema (fixed order; model + metrics first) ----
CSV_HEADER = [
    # identity + metrics first
    'model','seed','boundary','thr','test_AUC','test_AUPRC',
    'ACC','BACC','F1','DR','DR1','DR2','PREC','SPEC','FAR',
    # per-type PRF/FAR (add these!)
    'PREC1','REC1','F11','FAR1',
    'PREC2','REC2','F12','FAR2',

    # confusion matrix (overall)
    'TN','FP','FN','TP',
    # confusion matrix (normal vs Type-1)
    'TN1','FP1','FN1','TP1',
    # confusion matrix (normal vs Type-2)
    'TN2','FP2','FN2','TP2',

    # hyperparams (leave blank when N/A)
    'nu','rep_dim','gamma','lr','epochs','batch','wd',
    'sev1','sev2','margin1','margin2',

    # dataset/meta
    'rotate','n_norm','n_anom1','n_anom2','b','s1','s2',
    'far_cap','target_type',

    # validation info
    'val_crit',
]



# -------------------- utilities --------------------
def ensure_dirs():
    Path('plots').mkdir(exist_ok=True, parents=True)
    Path('checkpoints').mkdir(exist_ok=True, parents=True)
    Path('logs').mkdir(exist_ok=True, parents=True)


def parse_list(raw, cast=float, allow_symbols=None):
    """
    Parse comma-separated list string into a list of casted values.
    If allow_symbols is provided, tokens equal to any symbol are kept as str.
    """
    if raw is None:
        return None
    toks = [t.strip() for t in str(raw).split(',') if t.strip() != '']
    out = []
    for t in toks:
        if allow_symbols and t in allow_symbols:
            out.append(t)
        else:
            try:
                out.append(cast(t))
            except Exception:
                out.append(t)
    return out


def parse_pair(raw, cast=float, name="pair"):
    if raw is None:
        return None
    try:
        a, b = [cast(x) for x in str(raw).split(',')]
        return (a, b)
    except Exception as e:
        raise ValueError(f'--{name} must be "a,b"') from e


def parse_triple(raw, cast=float, name="triple"):
    if raw is None:
        return None
    try:
        a, b, c = [cast(x) for x in str(raw).split(',')]
        return (a, b, c)
    except Exception as e:
        raise ValueError(f'--{name} must be "a,b,c"') from e


def stratified_split_indices(y, g, ratios=(0.6, 0.2, 0.2), seed=123):
    """
    Stratify by combined label: normal (0) vs anomalies types (1,2).
    We use key = y*10 + g to preserve normal (0) and two anomaly types (11,12).
    Returns train_idx, val_idx, test_idx
    """
    rng = np.random.default_rng(seed)
    keys = y * 10 + g
    uniq = np.unique(keys)
    train_idx, val_idx, test_idx = [], [], []
    for k in uniq:
        idx = np.where(keys == k)[0]
        rng.shuffle(idx)
        n = idx.size
        n_train = int(round(ratios[0] * n))
        n_val   = int(round(ratios[1] * n))
        n_test  = n - n_train - n_val
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train+n_val])
        test_idx.extend(idx[n_train+n_val:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def make_splits_from_params(params: BananaParams, split_seed=123, ratios=(0.6, 0.2, 0.2)):
    """
    Generate a single banana dataset from params, then split by 'ratios'.
    Returns three DictDataset objects (train, val, test) and also (Xtr,ytr,gtr),(Xva,yva,gva),(Xte,yte,gte)
    """
    X, y, g = make_banana_data(params)
    tr, va, te = stratified_split_indices(y, g, ratios=ratios, seed=split_seed)
    Xtr, ytr, gtr = X[tr], y[tr], g[tr]
    Xva, yva, gva = X[va], y[va], g[va]
    Xte, yte, gte = X[te], y[te], g[te]
    train = DictDataset(Xtr, ytr, gtr)
    val   = DictDataset(Xva, yva, gva)
    test  = DictDataset(Xte, yte, gte)
    return (train, val, test), (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte)


def select_threshold_by_f1(y_true, scores):
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=int)
    uniq = np.unique(s)
    candidates = np.concatenate([[-np.inf], uniq, [np.inf]])
    best_f1, best_t = -1.0, 0.0
    for t in candidates:
        yhat = (s > t).astype(int)
        f1 = f1_score(y, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def select_threshold_f1_with_far_cap(y_true, scores, far_cap=None):
    if far_cap is None:
        return select_threshold_by_f1(y_true, scores)
    s = np.asarray(scores, float)
    y = np.asarray(y_true, int)
    uniq = np.unique(s)
    candidates = np.concatenate([[-np.inf], uniq, [np.inf]])
    best_f1, best_t = -1.0, 0.0
    for t in candidates:
        yhat = (s > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        far = fp / (fp + tn + 1e-12)
        if far <= far_cap:
            f1 = f1_score(y, yhat, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    if best_f1 < 0:
        return select_threshold_by_f1(y, s)
    return best_t, best_f1


def select_threshold_sr_targetDR_with_far_cap(y_true, g_val, scores, target_type=1, far_cap=None):
    s = np.asarray(scores, float)
    y = np.asarray(y_true, int)
    g = np.asarray(g_val, int)
    uniq = np.unique(s)
    candidates = np.concatenate([[-np.inf], uniq, [np.inf]])
    best_dr, best_t = -1.0, 0.0
    for t in candidates:
        yhat = (s > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        far = fp / (fp + tn + 1e-12)
        if (far_cap is None) or (far <= far_cap):
            mask_t = (y == 1) & (g == target_type)
            if mask_t.any():
                tp_t = np.sum((yhat == 1) & mask_t)
                fn_t = np.sum((yhat == 0) & mask_t)
                dr_t = tp_t / (tp_t + fn_t + 1e-12)
            else:
                dr_t = 0.0
            if dr_t > best_dr:
                best_dr, best_t = dr_t, t
    if best_dr < 0:
        return select_threshold_by_f1(y, s)
    return best_t, best_dr


def compute_confusion_stats(y_true, y_pred):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    eps = 1e-12
    recall = tp / (tp + fn + eps)           # DR = TPR
    precision = tp / (tp + fp + eps)
    specificity = tn / (tn + fp + eps)      # TNR
    far = fp / (fp + tn + eps)

    acc = accuracy_score(y, p)
    bacc = 0.5 * (recall + specificity)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return dict(
        FAR=float(far),
        ACC=float(acc),
        BACC=float(bacc),
        F1=float(f1),
        DR=float(recall),
        PREC=float(precision),
        SPEC=float(specificity),
    )


def compute_confusions_with_types(y_true, y_pred, g, type_ids=(1, 2)):
    """
    Adds:
      TN, FP, FN, TP  (overall, y in {0,1})
      TNt, FPt, FNt, TPt for each t in type_ids,
        computed on the subset: normals (y=0) OR (anomaly & g==t).
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    g = np.asarray(g, dtype=int)

    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    out = dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

    for t in type_ids:
        mask = (y == 0) | ((y == 1) & (g == t))
        if np.any(mask):
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y[mask], p[mask], labels=[0, 1]).ravel()
        else:
            tn_t, fp_t, fn_t, tp_t = 0, 0, 0, 0
        out.update({
            f'TN{t}': int(tn_t), f'FP{t}': int(fp_t),
            f'FN{t}': int(fn_t), f'TP{t}': int(tp_t),
        })
    return out


def typewise_detection_rates(y_true, y_pred, g, type_ids=(1, 2)):
    """
    DRt = recall on normals U anomalies of type t, i.e. TPt / (TPt + FNt).
    """
    y = np.asarray(y_true, int)
    p = np.asarray(y_pred, int)
    g = np.asarray(g, int)
    out = {}
    for t in type_ids:
        mask = (y == 0) | ((y == 1) & (g == t))
        if np.any(mask):
            tn_t, fp_t, fn_t, tp_t = confusion_matrix(y[mask], p[mask], labels=[0, 1]).ravel()
            out[f'DR{t}'] = float(tp_t / (tp_t + fn_t + 1e-12))
        else:
            out[f'DR{t}'] = 0.0
    return out

import numpy as np
from sklearn.metrics import confusion_matrix

def _typewise_precision_recall_f1_far(y_true, y_pred, g, type_ids=(1, 2)):
    """
    For each type t, restrict to {normals} ∪ {anomalies of type t},
    then compute Precision_t, Recall_t, F1_t, FAR_t.
    Returns a flat dict: {'PREC1', 'REC1', 'F11', 'FAR1', 'PREC2', ...}.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    g = np.asarray(g, dtype=int)
    out, eps = {}, 1e-12

    for t in type_ids:
        mask = (y == 0) | ((y == 1) & (g == t))
        if not np.any(mask):
            tn = fp = fn = tp = 0
        else:
            tn, fp, fn, tp = confusion_matrix(y[mask], p[mask], labels=[0, 1]).ravel()
        rec = tp / (tp + fn + eps)
        prec = tp / (tp + fp + eps)
        f1 = (2 * prec * rec) / (prec + rec + eps) if (prec + rec) > 0 else 0.0
        far = fp / (fp + tn + eps)  # normals predicted as anomalies
        out.update({f'PREC{t}': float(prec), f'REC{t}': float(rec),
                    f'F1{t}': float(f1), f'FAR{t}': float(far)})
    return out


def torch_scores_svdd_like(model, X, device='cpu'):
    """
    Compute scores for SR-DeepSVDD / DeepSVDD / Linear SVDD models:
      s(x) = d2 - R^2 (soft-boundary)  or  s(x)=d2 (one-class)
    """
    device = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    net = model.net.to(device) if hasattr(model, 'net') else None

    with torch.no_grad():
        if net is not None:
            xb = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
            z = net(xb)

            c_attr = getattr(model.trainer, 'c', None)
            if isinstance(c_attr, torch.Tensor):
                c = c_attr.detach().to(device)
            else:
                c = torch.as_tensor(c_attr, device=device, dtype=torch.float32)

            d2 = torch.sum((z - c) ** 2, dim=1)

            if getattr(model, 'objective', None) == 'soft-boundary':
                R_attr = getattr(model.trainer, 'R', 0.0)
                if isinstance(R_attr, torch.Tensor):
                    R2 = (R_attr.detach() ** 2)
                else:
                    R2 = torch.as_tensor(R_attr, device=device, dtype=torch.float32) ** 2
                s = (d2 - R2).detach().cpu().numpy()
            else:
                s = d2.detach().cpu().numpy()
        else:
            # Linear SVDD (center a and radius in input space)
            a = model.trainer.a.detach().cpu().numpy()
            R_attr = model.trainer.R
            if isinstance(R_attr, torch.Tensor):
                R2 = float((R_attr.detach() ** 2).cpu().numpy())
            else:
                R2 = float(R_attr ** 2)
            X = np.asarray(X, dtype=np.float32)
            d2 = np.sum((X - a) ** 2, axis=1)
            s = d2 - R2
    return s


def log_result_row(csv_path, row_dict, header=CSV_HEADER):
    import csv, os

    # Decide if we need to (re)write the header
    needs_header = True
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as fr:
                reader = csv.reader(fr)
                first = next(reader, None)
                if first is not None:
                    # Compare exactly to requested header
                    needs_header = (list(first) != list(header))
        except Exception:
            needs_header = True

    mode = 'w' if needs_header else 'a'
    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        if needs_header:
            writer.writeheader()
        # Fill missing keys as empty strings for consistent CSV
        row_filled = {k: ('' if row_dict.get(k) is None else row_dict.get(k, '')) for k in header}
        writer.writerow(row_filled)



class _XYTrainWrapper:
    """Minimal adapter for KernelSVDD.fit_on_dataset: exposes .train_set = (X, y)."""
    def __init__(self, X, y):
        self.train_set = (np.asarray(X), np.asarray(y))


def _ensure_dir(d):
    if d and len(d):
        Path(d).mkdir(parents=True, exist_ok=True)


def save_train_scatter(X, y, g, savepath, title="Train scatter"):
    """One scatter for the training split (normal vs Type-1 vs Type-2)."""
    _ensure_dir(os.path.dirname(savepath))
    X = np.asarray(X); y = np.asarray(y); g = np.asarray(g)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=600)
    ax.scatter(X[y==0,0], X[y==0,1], s=28, alpha=0.7, label='Train Normal', edgecolors='none')
    ax.scatter(X[(y==1)&(g==1),0], X[(y==1)&(g==1),1], s=34, alpha=0.95, label='Train Anomaly Type 1', marker='^')
    ax.scatter(X[(y==1)&(g==2),0], X[(y==1)&(g==2),1], s=34, alpha=0.95, label='Train Anomaly Type 2', marker='v')
    ax.legend(frameon=False)
    ax.set_title(title); ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.axis('equal'); ax.grid(True, ls=':', alpha=0.35)
    fig.tight_layout(); fig.savefig(savepath); plt.close(fig)


# -------------------- single-run (with new controls) --------------------
def single_run_sr(params_ds, split_ratios, device='cuda',
                  nu=0.05, rep_dim=16,
                  sev1=3.0, sev2=1.0, margin1=0.5, margin2=0.0,
                  lr=1e-3, n_epochs=50, batch_size=128, weight_decay=1e-6,
                  boundary='val', far_cap=None, target_type=1,
                  plots_dir='plots', plot_scores=1, grid_res=300):
    ensure_dirs()
    (train, val, test), (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = make_splits_from_params(
        params_ds, split_seed=params_ds.seed, ratios=split_ratios
    )

    model = SRDeepSVDD(objective='soft-boundary', nu=nu,
                       severity_weights={1: sev1, 2: sev2},
                       margin_per_group={1: margin1, 2: margin2})
    model.set_network('mlp', input_dim=2, hidden_dims=(64, 32), rep_dim=rep_dim)

    ds_train_val = BaseADDataset(train, val)
    model.train(ds_train_val, optimizer_name='adamw', lr=lr, n_epochs=n_epochs,
                lr_milestones=(int(0.7*n_epochs),), batch_size=batch_size,
                weight_decay=weight_decay, device=device)

    # validation threshold
    s_val = torch_scores_svdd_like(model, Xva, device=device)
    if boundary == 'val':
        thr, crit = select_threshold_sr_targetDR_with_far_cap(
            y_true=yva, g_val=gva, scores=s_val, target_type=target_type, far_cap=far_cap
        )
    else:
        thr, crit = 0.0, None

    # test scoring + metrics
    s_test = torch_scores_svdd_like(model, Xte, device=device)
    auc = float(roc_auc_score(yte, s_test))
    auprc = float(average_precision_score(yte, s_test))
    yhat_test = (s_test > thr).astype(int)

    stats  = compute_confusion_stats(yte, yhat_test)              # includes overall FAR, ACC, F1, DR, PREC, SPEC if using your helper
    counts = compute_confusions_with_types(yte, yhat_test, gte)   # TNt, FPt, FNt, TPt per type
    type_dr = typewise_detection_rates(yte, yhat_test, gte)       # DR1, DR2

    # NEW: per-type Precision/Recall/F1/FAR (normals ∪ anomalies of that type)
    type_prf_far = _typewise_precision_recall_f1_far(yte, yhat_test, gte, type_ids=(1, 2))

    if plot_scores:
        score_plots.plot_score_and_quantile(
            model, 'sr',
            (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte),
            thr,
            device=device,
            grid_res=grid_res,
            save_prefix=os.path.join(plots_dir, 'sr'),
            boundary=boundary
        )

    row = {
        'model': 'SR-DeepSVDD', 'boundary': boundary, 'thr': thr,
        'test_AUC': auc, 'test_AUPRC': auprc,
        **stats, **type_dr, **type_prf_far, **counts,
        'nu': nu, 'rep_dim': rep_dim, 'gamma': '',
        'lr': lr, 'epochs': n_epochs, 'batch': batch_size, 'wd': weight_decay,
        'sev1': sev1, 'sev2': sev2, 'margin1': margin1, 'margin2': margin2,
        'rotate': params_ds.rotate_deg, 'n_norm': params_ds.n_norm,
        'n_anom1': params_ds.n_anom1, 'n_anom2': params_ds.n_anom2,
        'b': params_ds.b, 's1': params_ds.s1, 's2': params_ds.s2,
        'far_cap': far_cap, 'target_type': target_type,
        'seed': params_ds.seed,
        'val_crit': crit
    }
    log_result_row('logs/grid_results.csv', row)
    print("[SR-DeepSVDD]", row)
    return row



def grid_sr(params_ds, split_ratios, device, nu_list, rep_list, sev1_list, sev2_list,
            margin1_list, margin2_list, lr_list, epoch_list, batch_list, wd_list,
            boundary, far_cap, target_type, plots_dir, plot_scores, grid_res):
    results = []
    for nu, rep, sev1, sev2, m1, m2, lr, ne, bs, wd in itertools.product(
        nu_list, rep_list, sev1_list, sev2_list, margin1_list, margin2_list,
        lr_list, epoch_list, batch_list, wd_list
    ):
        try:
            row = single_run_sr(params_ds, split_ratios, device=device, nu=nu, rep_dim=rep,
                                sev1=sev1, sev2=sev2, margin1=m1, margin2=m2,
                                lr=lr, n_epochs=int(ne), batch_size=int(bs), weight_decay=wd,
                                boundary=boundary, far_cap=far_cap, target_type=target_type,
                                plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            results.append(row)
        except Exception as e:
            print(f"[SR-DeepSVDD] failed for params: nu={nu}, rep={rep}, sev1={sev1}, sev2={sev2}, m1={m1}, m2={m2}, lr={lr}, ep={ne}, bs={bs}, wd={wd}. Error: {e}")
    return results


def single_run_deepsvdd(params_ds, split_ratios, device='cuda', nu=0.05, rep_dim=16,
                        lr=1e-3, n_epochs=50, batch_size=128, weight_decay=1e-6,
                        boundary='val', far_cap=None,
                        plots_dir='plots', plot_scores=1, grid_res=300):
    ensure_dirs()
    (train, val, test), (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = make_splits_from_params(
        params_ds, split_seed=params_ds.seed, ratios=split_ratios
    )

    model = DeepSVDD(objective='soft-boundary', nu=nu, hinge_power=2)
    model.set_network('mlp', input_dim=2, hidden_dims=(64, 32), rep_dim=rep_dim)

    ds_train_val = BaseADDataset(train, val)
    model.train(ds_train_val, optimizer_name='adamw', lr=lr, n_epochs=n_epochs,
                lr_milestones=(int(0.7*n_epochs),), batch_size=batch_size,
                weight_decay=weight_decay, device=device)

    s_val = torch_scores_svdd_like(model, Xva, device=device)
    if boundary == 'val':
        thr, _ = select_threshold_f1_with_far_cap(yva, s_val, far_cap=far_cap)
    else:
        thr = 0.0

    s_test = torch_scores_svdd_like(model, Xte, device=device)
    auc = float(roc_auc_score(yte, s_test))
    auprc = float(average_precision_score(yte, s_test))
    yhat_test = (s_test > thr).astype(int)

    stats  = compute_confusion_stats(yte, yhat_test)
    counts = compute_confusions_with_types(yte, yhat_test, gte)
    type_dr = typewise_detection_rates(yte, yhat_test, gte)
    # NEW:
    type_prf_far = _typewise_precision_recall_f1_far(yte, yhat_test, gte, type_ids=(1, 2))

    if plot_scores:
        score_plots.plot_score_and_quantile(
            model, 'deepsvdd',
            (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte),
            thr,
            device=device,
            grid_res=grid_res,
            save_prefix=os.path.join(plots_dir, 'deepsvdd'),
            boundary=boundary
        )

    row = {
        'model': 'DeepSVDD', 'boundary': boundary, 'thr': thr,
        'test_AUC': auc, 'test_AUPRC': auprc,
        **stats, **type_dr, **type_prf_far, **counts,
        'nu': nu, 'rep_dim': rep_dim, 'gamma': '',
        'lr': lr, 'epochs': n_epochs, 'batch': batch_size, 'wd': weight_decay,
        'sev1': '', 'sev2': '', 'margin1': '', 'margin2': '',
        'rotate': params_ds.rotate_deg, 'n_norm': params_ds.n_norm,
        'n_anom1': params_ds.n_anom1, 'n_anom2': params_ds.n_anom2,
        'b': params_ds.b, 's1': params_ds.s1, 's2': params_ds.s2,
        'far_cap': far_cap, 'target_type': '',
        'seed': params_ds.seed, 'val_crit': ''
    }
    log_result_row('logs/grid_results.csv', row)
    print("[DeepSVDD]", row)
    return row


def grid_deepsvdd(params_ds, split_ratios, device, nu_list, rep_list, lr_list, epoch_list, batch_list, wd_list,
                  boundary, far_cap, plots_dir, plot_scores, grid_res):
    results = []
    for nu, rep, lr, ne, bs, wd in itertools.product(
        nu_list, rep_list, lr_list, epoch_list, batch_list, wd_list
    ):
        try:
            row = single_run_deepsvdd(params_ds, split_ratios, device=device, nu=nu, rep_dim=rep,
                                      lr=lr, n_epochs=int(ne), batch_size=int(bs), weight_decay=wd,
                                      boundary=boundary, far_cap=far_cap,
                                      plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            results.append(row)
        except Exception as e:
            print(f"[DeepSVDD] failed for params: nu={nu}, rep={rep}, lr={lr}, ep={ne}, bs={bs}, wd={wd}. Error: {e}")
    return results


def single_run_svdd_linear(params_ds, split_ratios, nu=0.05, lr=1e-2, n_epochs=300, batch_size=256,
                           boundary='val', far_cap=None,
                           plots_dir='plots', plot_scores=1, grid_res=300):
    ensure_dirs()
    (train, val, test), (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = make_splits_from_params(
        params_ds, split_seed=params_ds.seed, ratios=split_ratios
    )

    model = LinearSVDD(objective='soft-boundary', nu=nu, hinge_power=2)
    model.train(BaseADDataset(train, val), lr=lr, n_epochs=n_epochs, batch_size=batch_size, device='cpu')

    a = model.trainer.a.detach().cpu().numpy()
    R2 = float((model.trainer.R ** 2).detach().cpu().numpy())
    d2_val = np.sum((np.asarray(Xva) - a) ** 2, axis=1)
    s_val = d2_val - R2
    if boundary == 'val':
        thr, _ = select_threshold_f1_with_far_cap(yva, s_val, far_cap=far_cap)
    else:
        thr = 0.0

    d2_test = np.sum((np.asarray(Xte) - a) ** 2, axis=1)
    s_test = d2_test - R2
    auc = float(roc_auc_score(yte, s_test))
    auprc = float(average_precision_score(yte, s_test))
    yhat_test = (s_test > thr).astype(int)
    stats = compute_confusion_stats(yte, yhat_test)
    counts = compute_confusions_with_types(yte, yhat_test, gte)
    type_dr = typewise_detection_rates(yte, yhat_test, gte)

    if plot_scores:
        score_plots.plot_score_and_quantile(
            model, 'svdd_linear',
            (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte),
            thr,
            device='cpu',
            grid_res=grid_res,
            save_prefix=os.path.join(plots_dir, 'lsvdd'),
            boundary=boundary
        )

    row = {
        'model': 'SVDD-Linear', 'boundary': boundary, 'thr': thr,
        'test_AUC': auc, 'test_AUPRC': auprc,
        **stats, **type_dr, **counts,
        'nu': nu, 'rep_dim': '', 'gamma': '',
        'lr': lr, 'epochs': n_epochs, 'batch': batch_size, 'wd': '',
        'sev1': '', 'sev2': '', 'margin1': '', 'margin2': '',
        'rotate': params_ds.rotate_deg, 'n_norm': params_ds.n_norm,
        'n_anom1': params_ds.n_anom1, 'n_anom2': params_ds.n_anom2,
        'b': params_ds.b, 's1': params_ds.s1, 's2': params_ds.s2,
        'far_cap': far_cap, 'target_type': '',
        'seed': params_ds.seed,
        'val_crit': ''
    }
    log_result_row('logs/grid_results.csv', row)
    print("[SVDD-Linear]", row)
    return row


def grid_svdd_linear(params_ds, split_ratios, nu_list, lr_list, epoch_list, batch_list,
                     boundary, far_cap, plots_dir, plot_scores, grid_res):
    results = []
    for nu, lr, ne, bs in itertools.product(nu_list, lr_list, epoch_list, batch_list):
        try:
            row = single_run_svdd_linear(params_ds, split_ratios, nu=nu, lr=lr, n_epochs=int(ne), batch_size=int(bs),
                                         boundary=boundary, far_cap=far_cap,
                                         plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            results.append(row)
        except Exception as e:
            print(f"[SVDD-Linear] failed for params: nu={nu}, lr={lr}, ep={ne}, bs={bs}. Error: {e}")
    return results


def single_run_ocsvm(params_ds, split_ratios, nu=0.05, gamma='scale',
                     boundary='val', far_cap=None,
                     plots_dir='plots', plot_scores=1, grid_res=300):
    ensure_dirs()
    (train, val, test), (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = make_splits_from_params(
        params_ds, split_seed=params_ds.seed, ratios=split_ratios
    )

    oc = OCSVMWrapper(nu=nu, kernel='rbf', gamma=gamma)
    oc.train(BaseADDataset(train, val))   # fits on normals from train

    # OC-SVM: higher s = more anomalous (we flip sign of decision_function)
    s_val = -oc.model.decision_function(np.asarray(Xva)).ravel()
    if boundary == 'val':
        thr, _ = select_threshold_f1_with_far_cap(yva, s_val, far_cap=far_cap)
    else:
        thr = 0.0

    s_test = -oc.model.decision_function(np.asarray(Xte)).ravel()
    auc = float(roc_auc_score(yte, s_test))
    auprc = float(average_precision_score(yte, s_test))
    yhat_test = (s_test > thr).astype(int)

    stats  = compute_confusion_stats(yte, yhat_test)             # overall: ACC, F1, DR, PREC, SPEC, FAR, ...
    counts = compute_confusions_with_types(yte, yhat_test, gte)  # TNt, FPt, FNt, TPt per type
    type_dr = typewise_detection_rates(yte, yhat_test, gte)      # DR1, DR2

    # NEW: per-type Precision/Recall/F1/FAR (normals ∪ anomalies of that type)
    type_prf_far = _typewise_precision_recall_f1_far(yte, yhat_test, gte, type_ids=(1, 2))

    if plot_scores:
        score_plots.plot_score_and_quantile(
            oc, 'ocsvm',
            (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte),
            thr,
            device='cpu',
            grid_res=grid_res,
            save_prefix=os.path.join(plots_dir, 'ocsvm'),
            boundary=boundary
        )

    row = {
        'model': 'OC-SVM', 'boundary': boundary, 'thr': thr,
        'test_AUC': auc, 'test_AUPRC': auprc,
        **stats, **type_dr, **type_prf_far, **counts,
        'nu': nu, 'rep_dim': '', 'gamma': gamma,
        'lr': '', 'epochs': '', 'batch': '', 'wd': '',
        'sev1': '', 'sev2': '', 'margin1': '', 'margin2': '',
        'rotate': params_ds.rotate_deg, 'n_norm': params_ds.n_norm,
        'n_anom1': params_ds.n_anom1, 'n_anom2': params_ds.n_anom2,
        'b': params_ds.b, 's1': params_ds.s1, 's2': params_ds.s2,
        'far_cap': far_cap, 'target_type': '',
        'seed': params_ds.seed,
        'val_crit': ''
    }
    log_result_row('logs/grid_results.csv', row)
    print("[OC-SVM]", row)
    return row


def grid_ocsvm(params_ds, split_ratios, nu_list, gamma_list, boundary, far_cap, plots_dir, plot_scores, grid_res):
    results = []
    for nu, gamma in itertools.product(nu_list, gamma_list):
        try:
            row = single_run_ocsvm(params_ds, split_ratios, nu=nu, gamma=gamma,
                                   boundary=boundary, far_cap=far_cap,
                                   plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            results.append(row)
        except Exception as e:
            print(f"[OC-SVM] failed for params: nu={nu}, gamma={gamma}. Error: {e}")
    return results


def single_run_ksvdd(params_ds, split_ratios, nu=0.05, gamma=0.2,
                     boundary='val', far_cap=None,
                     plots_dir='plots', plot_scores=1, grid_res=300):
    ensure_dirs()
    (train, val, test), (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = make_splits_from_params(
        params_ds, split_seed=params_ds.seed, ratios=split_ratios
    )

    svdd = KernelSVDD(nu=nu, gamma=gamma)  # C = 1/(nu*N) internally

    # Train on normals from the train split (mirrors OC-SVM wrapper)
    mask_norm = (np.asarray(ytr) == 0)
    X_train_norm = np.asarray(Xtr)[mask_norm]
    y_train_norm = np.asarray(ytr)[mask_norm]
    ds_xy = _XYTrainWrapper(X_train_norm, y_train_norm)
    svdd.fit_on_dataset(ds_xy)

    # Scores: higher should mean "more anomalous" for consistency
    s_val = svdd.decision_function(np.asarray(Xva))
    if boundary == 'val':
        thr, _ = select_threshold_f1_with_far_cap(yva, s_val, far_cap=far_cap)
    else:
        thr = 0.0

    s_test = svdd.decision_function(np.asarray(Xte))
    auc = float(roc_auc_score(yte, s_test))
    auprc = float(average_precision_score(yte, s_test))
    yhat_test = (s_test > thr).astype(int)

    stats  = compute_confusion_stats(yte, yhat_test)             # overall metrics incl. FAR
    counts = compute_confusions_with_types(yte, yhat_test, gte)  # TNt/FPt/FNt/TPt per type
    type_dr = typewise_detection_rates(yte, yhat_test, gte)      # DR1, DR2

    # NEW: per-type Precision/Recall/F1/FAR (normals ∪ anomalies of that type)
    type_prf_far = _typewise_precision_recall_f1_far(yte, yhat_test, gte, type_ids=(1, 2))

    if plot_scores:
        score_plots.plot_score_and_quantile(
            svdd, 'svdd_rbf',
            (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte),
            thr,
            device='cpu',
            grid_res=grid_res,
            save_prefix=os.path.join(plots_dir, 'ksvdd'),
            boundary=boundary
        )

    row = {
        'model': 'SVDD-RBF', 'boundary': boundary, 'thr': thr,
        'test_AUC': auc, 'test_AUPRC': auprc,
        **stats, **type_dr, **type_prf_far, **counts,
        'nu': nu, 'rep_dim': '', 'gamma': gamma,
        'lr': '', 'epochs': '', 'batch': '', 'wd': '',
        'sev1': '', 'sev2': '', 'margin1': '', 'margin2': '',
        'rotate': params_ds.rotate_deg, 'n_norm': params_ds.n_norm,
        'n_anom1': params_ds.n_anom1, 'n_anom2': params_ds.n_anom2,
        'b': params_ds.b, 's1': params_ds.s1, 's2': params_ds.s2,
        'far_cap': far_cap, 'target_type': '',
        'seed': params_ds.seed,
        'val_crit': ''
    }
    log_result_row('logs/grid_results.csv', row)
    print("[SVDD-RBF]", row)
    return row



def grid_ksvdd(params_ds, split_ratios, nu_list, gamma_list, boundary, far_cap, plots_dir, plot_scores, grid_res):
    results = []
    for nu, gamma in itertools.product(nu_list, gamma_list):
        try:
            row = single_run_ksvdd(params_ds, split_ratios, nu=nu, gamma=gamma,
                                   boundary=boundary, far_cap=far_cap,
                                   plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            results.append(row)
        except Exception as e:
            print(f"[SVDD-RBF] failed for params: nu={nu}, gamma={gamma}. Error: {e}")
    return results


# -------------------- stats helpers (multi-seed) --------------------
def _paired_test(x, y, method='wilcoxon'):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = min(len(x), len(y))
    if n == 0:
        return np.nan
    x = x[:n]; y = y[:n]
    if method == 'wilcoxon' and _SCIPY_OK:
        try:
            stat, pval = _wilcoxon(x, y, zero_method='wilcox', alternative='two-sided', method='auto')
            return float(pval)
        except Exception:
            pass
    if method == 'ttest' and _SCIPY_OK:
        try:
            stat, pval = _ttest_rel(x, y)
            return float(pval)
        except Exception:
            pass
    # Fallback: normal approx for paired t-test
    d = x - y
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return 1.0
    t = m / (sd / np.sqrt(n))
    # p ~ 2*(1 - Phi(|t|))
    p = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(t) / np.sqrt(2.0))))
    return float(p)


def summarize_and_pvalues(all_rows, seeds, select_metric='test_AUC', select_dir='max',
                          pval_metric='test_AUC', pval_dir='max',
                          ref_model='SR-DeepSVDD', test='wilcoxon'):
    """
    Select best row per (model, seed) by select_metric (dir max/min),
    then compute mean/std across seeds and paired-test p-values vs ref_model.
    """
    if not all_rows:
        return [], []

    # Build per (model -> seed -> best value) and also keep test metric vector.
    best_by_model_seed = {}
    metrics_pool = {}  # (model -> {seed: {metric: value, ...}})

    # Decide comparator
    def better(a, b):
        return (a > b) if select_dir == 'max' else (a < b)

    for row in all_rows:
        model = row.get('model', 'UNK')
        seed = int(row.get('seed', -1))
        if seed < 0:
            continue
        metrics_pool.setdefault(model, {}).setdefault(seed, {})
        # this run's value for selection metric
        val = float(row.get(select_metric, np.nan))
        # keep "best" config per model, per seed
        prev = best_by_model_seed.get((model, seed), None)
        if prev is None or (np.isfinite(val) and better(val, prev[0])):
            best_by_model_seed[(model, seed)] = (val, row)
        # cache metrics (not strictly needed)
        metrics_pool[model][seed].update(row)

    # Reduce to best rows
    best_rows = {}
    for (model, seed), (_, row) in best_by_model_seed.items():
        best_rows.setdefault(model, {})[seed] = row

    # Prepare summaries
    seeds_sorted = sorted(set(seeds))
    models = sorted(best_rows.keys())
    # Build per-model vectors for pval_metric
    vecs = {}
    for m in models:
        v = []
        for s in seeds_sorted:
            r = best_rows[m].get(s)
            if r is None:
                continue
            v.append(float(r.get(pval_metric, np.nan)))
        vecs[m] = np.array(v, float)

    # Summaries
    summary_rows = []
    for m in models:
        arr = vecs[m]
        arr = arr[np.isfinite(arr)]
        summary_rows.append(dict(
            model=m, metric=pval_metric, select_metric=select_metric,
            n=int(arr.size), mean=float(np.mean(arr)) if arr.size else np.nan,
            std=float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
        ))

    # P-values vs reference
    pval_rows = []
    if ref_model in vecs:
        x = vecs[ref_model]
        for m in models:
            if m == ref_model:
                continue
            y = vecs[m]
            n = int(min(x.size, y.size))
            if n == 0:
                pval = np.nan
            else:
                pval = _paired_test(x[:n], y[:n], method=test)
            pval_rows.append(dict(
                ref_model=ref_model, model=m, metric=pval_metric,
                test=test, n=n,
                mean_ref=float(np.mean(x[:n])) if n else np.nan,
                mean_model=float(np.mean(y[:n])) if n else np.nan,
                pvalue=float(pval)
            ))

    # Write to disk
    import csv
    with open('logs/seed_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model','metric','select_metric','n','mean','std'])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    with open(f'logs/pvalues_{pval_metric}.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ref_model','model','metric','test','n','mean_ref','mean_model','pvalue'])
        writer.writeheader()
        for r in pval_rows:
            writer.writerow(r)

    # Also print a compact table
    print("\n=== Seed Summary ({} by {}, dir {}), {} seeds ===".format(
        pval_metric, select_metric, select_dir, len(seeds_sorted)))
    for r in summary_rows:
        print(f"  {r['model']:>12s} : n={r['n']:>2d}  mean={r['mean']:.4f}  std={r['std']:.4f}")
    if pval_rows:
        print("\n=== Paired p-values vs {} ({}), metric={} ===".format(ref_model, test, pval_metric))
        for r in pval_rows:
            print(f"  {r['model']:>12s} : n={r['n']:>2d}  mean_ref={r['mean_ref']:.4f}  mean_model={r['mean_model']:.4f}  p={r['pvalue']:.4g}")

    return summary_rows, pval_rows


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="SR-DeepSVDD & baselines on banana dataset with grid search, multi-seed runs, and statistical validation.")

    # mode
    p.add_argument('--mode', type=str, default='grid',
                   choices=['sr', 'deepsvdd', 'svdd_linear', 'ocsvm', 'svdd_rbf', 'all', 'grid'],
                   help='Which method to run (grid = run grid for selected models).')
    p.add_argument('--models', type=str, default='sr,deepsvdd,svdd_linear,ocsvm,svdd_rbf',
                   help='When --mode grid, comma-separated list of models to include.')

    # dataset params (BananaParams)
    p.add_argument('--n-norm', type=int, default=700)
    p.add_argument('--n-anom1', type=int, default=150)
    p.add_argument('--n-anom2', type=int, default=150)
    p.add_argument('--b', type=float, default=0.2)
    p.add_argument('--s1', type=float, default=2.0)
    p.add_argument('--s2', type=float, default=1.5)
    p.add_argument('--rotate', type=float, default=90.0)
    p.add_argument('--seed', type=int, default=123)

    # NEW: anomaly covariances and centers
    p.add_argument('--cov-a1', type=str, default='0.04,0.81', help='Anomaly Type-1 covariance controls (as "var,offdiag_scale").')
    p.add_argument('--cov-a2', type=str, default='0.04,0.81', help='Anomaly Type-2 covariance controls (as "var,offdiag_scale").')
    p.add_argument('--mu-a1',  type=str, default=None, help='Anomaly Type-1 center as "x,y".')
    p.add_argument('--mu-a2',  type=str, default=None, help='Anomaly Type-2 center as "x,y".')

    # NEW: split controls (sizes win if both given)
    p.add_argument('--split-frac', type=str, default='0.6,0.2,0.2', help='"train,val,test" fractions (sum≈1)')
    p.add_argument('--split-size', type=str, default=None, help='"train,val,test" absolute sizes (converted to fractions)')

    # Cross-cutting validation knobs
    p.add_argument('--far-target', type=float, default=None, help='FAR cap used in validation (None disables).')
    p.add_argument('--target-type', type=int, default=1, help='Targeted anomaly type (group id) for SR validation.')

    # Per-model boundary selection
    p.add_argument('--sr-boundary',     default='val', choices=['val', 'zero'])
    p.add_argument('--dsvdd-boundary',  default='val', choices=['val', 'zero'])
    p.add_argument('--lsvdd-boundary',  default='val', choices=['val', 'zero'])
    p.add_argument('--ocsvm-boundary',  default='val', choices=['val', 'zero'])
    p.add_argument('--ksvdd-boundary',  default='val', choices=['val', 'zero'])

    # SR-DeepSVDD grid params (vectors)
    p.add_argument('--sr-nu', type=str, default='0.05')
    p.add_argument('--sr-rep', type=str, default='16')
    p.add_argument('--sr-sev1', type=str, default='3.0')
    p.add_argument('--sr-sev2', type=str, default='1.0')
    p.add_argument('--sr-margin1', type=str, default='0.5')
    p.add_argument('--sr-margin2', type=str, default='0.0')
    p.add_argument('--sr-lr', type=str, default='0.001')
    p.add_argument('--sr-epochs', type=str, default='50')
    p.add_argument('--sr-batch', type=str, default='128')
    p.add_argument('--sr-weight-decay', type=str, default='1e-6')
    p.add_argument('--device', type=str, default='cuda')

    # DeepSVDD grid params
    p.add_argument('--dsvdd-nu', type=str, default='0.05')
    p.add_argument('--dsvdd-rep', type=str, default='16')
    p.add_argument('--dsvdd-lr', type=str, default='0.001')
    p.add_argument('--dsvdd-epochs', type=str, default='50')
    p.add_argument('--dsvdd-batch', type=str, default='128')
    p.add_argument('--dsvdd-weight-decay', type=str, default='1e-6')

    # Linear SVDD grid params
    p.add_argument('--lsvdd-nu', type=str, default='0.05')
    p.add_argument('--lsvdd-lr', type=str, default='0.01')
    p.add_argument('--lsvdd-epochs', type=str, default='300')
    p.add_argument('--lsvdd-batch', type=str, default='256')

    # OC-SVM grid params
    p.add_argument('--ocsvm-nu', type=str, default='0.05')
    p.add_argument('--ocsvm-gamma', type=str, default='scale')  # allow 'scale' or numbers

    # Kernel SVDD grid params
    p.add_argument('--ksvdd-nu', type=str, default='0.05')
    p.add_argument('--ksvdd-gamma', type=str, default='0.2')

    # Plotting controls
    p.add_argument('--plots-dir',   type=str, default='plots', help='Where to save plots.')
    p.add_argument('--grid-res',    type=int, default=300,     help='Grid resolution for score_plots surface.')
    p.add_argument('--plot-train',  type=int, default=1,       help='1=save ONE training scatter per dataset.')
    p.add_argument('--plot-scores', type=int, default=1,       help='1=call score_plots for each model.')

    # ---- Multi-seed + statistics ----
    p.add_argument('--seeds', type=str, default=None, help='Comma-separated list of seeds to run. Overrides --seed.')
    p.add_argument('--n-seeds', type=int, default=None, help='If set, run seeds [seed, seed+1, ..., seed+n_seeds-1].')
    p.add_argument('--select-metric', type=str, default='test_AUC',
                   choices=['test_AUC','test_AUPRC','ACC','BACC','F1','DR','DR1','DR2','PREC','SPEC','FAR','val_crit'],
                   help='If multiple configs per model per seed, pick the best by this metric.')
    p.add_argument('--select-direction', type=str, default='max', choices=['max','min'],
                   help='Direction for selecting best config by --select-metric.')
    p.add_argument('--pval-metric', type=str, default='test_AUC',
                   choices=['test_AUC','test_AUPRC','ACC','BACC','F1','DR','DR1','DR2','PREC','SPEC','FAR','val_crit'],
                   help='Metric used for p-value computation.')
    p.add_argument('--pval-direction', type=str, default='max', choices=['max','min'],
                   help='(Info only) desired direction for the pval metric.')
    p.add_argument('--pval-ref', type=str, default='SR-DeepSVDD',
                   help='Reference model name to compare others against.')
    p.add_argument('--pval-test', type=str, default='wilcoxon', choices=['wilcoxon','ttest'],
                   help='Paired statistical test to use across seeds.')

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    # Resolve seeds to run
# Resolve seeds to run
    if args.seeds:  # explicit list wins
        seeds = [int(s) for s in parse_list(args.seeds, int)]
    elif ',' in str(args.seed):  # allow comma list in --seed
        seeds = [int(s) for s in parse_list(args.seed, int)]
    elif args.n_seeds and int(args.n_seeds) > 0:
        start = int(args.seed)
        seeds = [start + i for i in range(int(args.n_seeds))]
    else:
        seeds = [int(args.seed)]


    # Split: sizes win if given; otherwise fractions.
    split_frac_triple = parse_triple(args.split_frac, float, 'split-frac') if args.split_frac else (0.6, 0.2, 0.2)

    # Plotting flags
    plots_dir   = args.plots_dir
    grid_res    = int(getattr(args, "grid_res", 300))
    plot_train  = int(getattr(args, "plot_train", 1))
    plot_scores = int(getattr(args, "plot_scores", 1))

    # Parse vectors
    sr_nu = parse_list(args.sr_nu, float)
    sr_rep = parse_list(args.sr_rep, int)
    sr_sev1 = parse_list(args.sr_sev1, float)
    sr_sev2 = parse_list(args.sr_sev2, float)
    sr_margin1 = parse_list(args.sr_margin1, float)
    sr_margin2 = parse_list(args.sr_margin2, float)
    sr_lr = parse_list(args.sr_lr, float)
    sr_epochs = parse_list(args.sr_epochs, int)
    sr_batch = parse_list(args.sr_batch, int)
    sr_wd = parse_list(args.sr_weight_decay, float)

    dsvdd_nu = parse_list(args.dsvdd_nu, float)
    dsvdd_rep = parse_list(args.dsvdd_rep, int)
    dsvdd_lr = parse_list(args.dsvdd_lr, float)
    dsvdd_epochs = parse_list(args.dsvdd_epochs, int)
    dsvdd_batch = parse_list(args.dsvdd_batch, int)
    dsvdd_wd = parse_list(args.dsvdd_weight_decay, float)

    lsvdd_nu = parse_list(args.lsvdd_nu, float)
    lsvdd_lr = parse_list(args.lsvdd_lr, float)
    lsvdd_epochs = parse_list(args.lsvdd_epochs, int)
    lsvdd_batch = parse_list(args.lsvdd_batch, int)

    ocsvm_nu = parse_list(args.ocsvm_nu, float)
    ocsvm_gamma = parse_list(args.ocsvm_gamma, cast=str, allow_symbols={'scale'})

    ksvdd_nu = parse_list(args.ksvdd_nu, float)
    ksvdd_gamma = parse_list(args.ksvdd_gamma, float)

    # Short-hands for per-model boundaries & FAR cap
    sr_boundary     = args.sr_boundary
    dsvdd_boundary  = args.dsvdd_boundary
    lsvdd_boundary  = args.lsvdd_boundary
    ocsvm_boundary  = args.ocsvm_boundary
    ksvdd_boundary  = args.ksvdd_boundary
    far_cap         = args.far_target
    tgt_type        = args.target_type

    all_rows = []

    for seed in seeds:
        # Build dataset params object per seed
        params_ds = BananaParams(
            n_norm=args.n_norm, n_anom1=args.n_anom1, n_anom2=args.n_anom2,
            seed=seed, b=args.b, s1=args.s1, s2=args.s2, rotate_deg=args.rotate
        )
        # Inject optional centers/covariances if provided
        cov_a1 = parse_pair(args.cov_a1, float, name='cov-a1')
        cov_a2 = parse_pair(args.cov_a2, float, name='cov-a2')
        setattr(params_ds, 'cov_a1', cov_a1)
        setattr(params_ds, 'cov_a2', cov_a2)
        if args.mu_a1 is not None:
            setattr(params_ds, 'mu_a1', parse_pair(args.mu_a1, float, name='mu-a1'))
        if args.mu_a2 is not None:
            setattr(params_ds, 'mu_a2', parse_pair(args.mu_a2, float, name='mu-a2'))

        if args.split_size:
            total = float(args.n_norm + args.n_anom1 + args.n_anom2)
            ts, vs, ss = parse_triple(args.split_size, int, 'split-size')
            split_ratios = (ts / total, vs / total, ss / total)
        else:
            split_ratios = split_frac_triple

        # ONE dataset-level train scatter (shared by all models for this seed)
        if plot_train:
            (_, _, _), (Xtr, ytr, gtr), (Xva, yva, gva), (Xte, yte, gte) = make_splits_from_params(
                params_ds, split_seed=params_ds.seed, ratios=split_ratios
            )
            save_train_scatter(
                Xtr, ytr, gtr,
                os.path.join(plots_dir, f'train_scatter_seed{seed}.png'),
                title=f"Training dataset"
            )

        # Dispatch per mode
        if args.mode == 'sr':
            all_rows += grid_sr(params_ds, split_ratios, args.device,
                                sr_nu, sr_rep, sr_sev1, sr_sev2, sr_margin1, sr_margin2,
                                sr_lr, sr_epochs, sr_batch, sr_wd,
                                boundary=sr_boundary, far_cap=far_cap, target_type=tgt_type,
                                plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)

        elif args.mode == 'deepsvdd':
            all_rows += grid_deepsvdd(params_ds, split_ratios, args.device,
                                      dsvdd_nu, dsvdd_rep, dsvdd_lr, dsvdd_epochs, dsvdd_batch, dsvdd_wd,
                                      boundary=dsvdd_boundary, far_cap=far_cap,
                                      plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)

        elif args.mode == 'svdd_linear':
            all_rows += grid_svdd_linear(params_ds, split_ratios, lsvdd_nu, lsvdd_lr, lsvdd_epochs, lsvdd_batch,
                                         boundary=lsvdd_boundary, far_cap=far_cap,
                                         plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)

        elif args.mode == 'ocsvm':
            all_rows += grid_ocsvm(params_ds, split_ratios, ocsvm_nu, ocsvm_gamma,
                                   boundary=ocsvm_boundary, far_cap=far_cap,
                                   plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)

        elif args.mode == 'svdd_rbf':
            all_rows += grid_ksvdd(params_ds, split_ratios, ksvdd_nu, ksvdd_gamma,
                                   boundary=ksvdd_boundary, far_cap=far_cap,
                                   plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)

        elif args.mode == 'all':
            all_rows.append(single_run_sr(params_ds, split_ratios, device=args.device,
                                          nu=sr_nu[0], rep_dim=sr_rep[0], sev1=sr_sev1[0], sev2=sr_sev2[0],
                                          margin1=sr_margin1[0], margin2=sr_margin2[0],
                                          lr=sr_lr[0], n_epochs=sr_epochs[0], batch_size=sr_batch[0], weight_decay=sr_wd[0],
                                          boundary=sr_boundary, far_cap=far_cap, target_type=tgt_type,
                                          plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res))

            all_rows.append(single_run_deepsvdd(params_ds, split_ratios, device=args.device,
                                                nu=dsvdd_nu[0], rep_dim=dsvdd_rep[0], lr=dsvdd_lr[0],
                                                n_epochs=dsvdd_epochs[0], batch_size=dsvdd_batch[0], weight_decay=dsvdd_wd[0],
                                                boundary=dsvdd_boundary, far_cap=far_cap,
                                                plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res))

            all_rows.append(single_run_svdd_linear(params_ds, split_ratios,
                                                   nu=lsvdd_nu[0], lr=lsvdd_lr[0], n_epochs=lsvdd_epochs[0], batch_size=lsvdd_batch[0],
                                                   boundary=lsvdd_boundary, far_cap=far_cap,
                                                   plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res))

            all_rows.append(single_run_ocsvm(params_ds, split_ratios,
                                             nu=ocsvm_nu[0], gamma=ocsvm_gamma[0],
                                             boundary=ocsvm_boundary, far_cap=far_cap,
                                             plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res))

            all_rows.append(single_run_ksvdd(params_ds, split_ratios,
                                             nu=ksvdd_nu[0], gamma=ksvdd_gamma[0],
                                             boundary=ksvdd_boundary, far_cap=far_cap,
                                             plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res))

        else:  # grid over selected models
            models = [m.strip() for m in args.models.split(',') if m.strip()]
            if 'sr' in models:
                all_rows += grid_sr(params_ds, split_ratios, args.device,
                                    sr_nu, sr_rep, sr_sev1, sr_sev2, sr_margin1, sr_margin2,
                                    sr_lr, sr_epochs, sr_batch, sr_wd,
                                    boundary=sr_boundary, far_cap=far_cap, target_type=tgt_type,
                                    plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            if 'deepsvdd' in models:
                all_rows += grid_deepsvdd(params_ds, split_ratios, args.device,
                                          dsvdd_nu, dsvdd_rep, dsvdd_lr, dsvdd_epochs, dsvdd_batch, dsvdd_wd,
                                          boundary=dsvdd_boundary, far_cap=far_cap,
                                          plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            if 'svdd_linear' in models:
                all_rows += grid_svdd_linear(params_ds, split_ratios, lsvdd_nu, lsvdd_lr, lsvdd_epochs, lsvdd_batch,
                                             boundary=lsvdd_boundary, far_cap=far_cap,
                                             plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            if 'ocsvm' in models:
                all_rows += grid_ocsvm(params_ds, split_ratios, ocsvm_nu, ocsvm_gamma,
                                       boundary=ocsvm_boundary, far_cap=far_cap,
                                       plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)
            if 'svdd_rbf' in models:
                all_rows += grid_ksvdd(params_ds, split_ratios, ksvdd_nu, ksvdd_gamma,
                                       boundary=ksvdd_boundary, far_cap=far_cap,
                                       plots_dir=plots_dir, plot_scores=plot_scores, grid_res=grid_res)

    # ---- Summaries & p-values across seeds ----
    summarize_and_pvalues(
        all_rows,
        seeds,
        select_metric=args.select_metric,
        select_dir=args.select_direction,
        pval_metric=args.pval_metric,
        pval_dir=args.pval_direction,
        ref_model=args.pval_ref,
        test=args.pval_test
    )


if __name__ == '__main__':
    main()
