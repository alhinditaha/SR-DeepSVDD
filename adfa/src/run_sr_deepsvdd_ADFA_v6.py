#!/usr/bin/env python3
"""
run_sr_deepsvdd_ADFA_v5.py
ADFA-LD experiment driver (tabular + TF-IDF -> bias-free MLP -> Deep/SR-DeepSVDD)

Key additions:
- On-load targeting controls: --target-families / --target-frac / --target-override
  (lets you mark targeted anomalies at train/val/test load time, even if not baked in during --prepare)
- Robust family substring parsing (CSV or list-like strings)
- Device resolution with safe CUDA fallback
- FAR-cap evaluation + split summaries
- Kernel SVDD baseline import supports svdd_rbf.py or kernel_svdd.py
- NEW: Score orientation auto-fix (flip if needed) so 'higher = more anomalous' for every model.
"""

# --- path bootstrap so 'src.*' works even when cwd is .../src ---
import torch
from pathlib import Path
import sys, os
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
# ---------------------------------------------------------------

import argparse, numpy as np, ast
from typing import Tuple, Dict, Any, List, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score

# === Repo imports ===
from src.datasets.adfa_dataset import build_and_save_features
from src.base.base_dataset import DictDataset, BaseADDataset
from src.models.sr_deep_svdd import SRDeepSVDD
from src.networks.main import build_network  # used by SRDeepSVDD.set_network internally
from src.models.deep_svdd import DeepSVDD 

# ---- Optional baselines (import if exist) ----
_HAS_LSVDD = False
_HAS_OCSVM = False
_HAS_KSVDD = False
try:
    from src.models.linear_svdd import LinearSVDD
    _HAS_LSVDD = True
except Exception:
    pass
try:
    from src.baselines.ocsvm import OCSVM
    _HAS_OCSVM = True
except Exception:
    pass
# Try both names for the kernel baseline
try:
    from src.baselines.svdd_rbf import KernelSVDD
    _HAS_KSVDD = True
except Exception:
    try:
        from src.baselines.kernel_svdd import KernelSVDD  # fallback name
        _HAS_KSVDD = True
    except Exception:
        pass


# =========================
# FAR-cap evaluation utils
# =========================
def seed_everything(seed: int = 42):
    import os, random, numpy as np, torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"  # or ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _pick_threshold_at_far(val_scores_norm: np.ndarray, far_cap: float) -> float:
    """Pick tau so that FAR on validation normals is ≤ far_cap (higher score = more anomalous)."""
    if val_scores_norm.size == 0:
        return float("inf")  # degenerate
    q = np.quantile(val_scores_norm, 1.0 - float(far_cap))
    return float(q)

def _confusion(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp, tn, fp, fn = _confusion(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)  # = Detection Rate (DR)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc  = accuracy_score(y_true, y_pred)
    return dict(TP=tp, TN=tn, FP=fp, FN=fn,
                Precision=prec, Recall=rec, F1=f1, Specificity=spec, Accuracy=acc)

def _infer_scores_via_center(model, X: np.ndarray, device: str):
    """
    Generic DeepSVDD-style score: squared distance to center c in rep space.
    Requires model.net (torch nn.Module) and model.c (torch tensor or array-like).
    """
    if not hasattr(model, "net") or not hasattr(model, "c"):
        raise AttributeError("Model lacks 'net' or 'c' for center-based scoring.")
    net = model.net
    net.eval()
    dev = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
    net.to(dev)

    with torch.no_grad():
        Xb = torch.from_numpy(np.asarray(X)).float().to(dev)
        Z = net(Xb)  # (n, rep_dim)

        c = model.c
        if isinstance(c, (list, tuple, np.ndarray)):
            c = torch.as_tensor(c, dtype=torch.float32, device=dev)
        else:
            c = c.detach().to(dev).float()

        d2 = torch.sum((Z - c) ** 2, dim=1)
        scores = d2.detach().cpu().numpy()
    return scores

def _get_scores(model, X: np.ndarray, device: str):
    """
    Try common APIs; fall back to center-based scorer.
    NOTE: Returned scores may be 'normality' or 'anomaly' depending on model.
          We orient them later with _orient_scores_for_anomaly(...).
    """
    # 1) model-specific APIs
    for attr in ("decision_function", "score_samples", "predict_scores"):
        if hasattr(model, attr):
            fn = getattr(model, attr)
            try:
                return np.asarray(fn(X, device=device))
            except TypeError:
                return np.asarray(fn(X))

    # 2) some repos expose 'test_on_array' returning scores
    for attr in ("test_on_array", "eval_on_array"):
        if hasattr(model, attr):
            fn = getattr(model, attr)
            code = getattr(fn, "__code__", None)
            if code and "device" in code.co_varnames:
                out = fn(X, device=device)
            else:
                out = fn(X)
            if isinstance(out, (tuple, list)) and len(out) > 0:
                return np.asarray(out[0])
            return np.asarray(out)

    # 3) center-based fallback
    return _infer_scores_via_center(model, X, device)

# =========================
# Score orientation helper
# =========================
def _orient_scores_for_anomaly(scores, y_binary):
    """
    Ensure 'higher = more anomalous'.
    If the score looks like 'higher = more normal' (AUROC < 0.5), flip its sign.
    y_binary must be 0=normal, 1=anomaly on the SAME samples used for 'scores'.
    Returns (scores_oriented, flipped_flag).
    """
    s = np.asarray(scores, float)
    y = np.asarray(y_binary, int)
    flipped = False
    try:
        if len(np.unique(y)) == 2 and roc_auc_score(y, s) < 0.5:
            s = -s
            flipped = True
    except Exception:
        pass
    return s, flipped

# =================
# Metrics printers
# =================
def _print_overall_aucs(scores: np.ndarray, y_true: np.ndarray, tag: str):
    # orient first using anomaly labels
    s, flipped = _orient_scores_for_anomaly(scores, y_true)
    try:
        if len(np.unique(y_true)) < 2:
            raise ValueError("Only one class in y_true; AUROC undefined.")
        auroc = roc_auc_score(y_true, s)
        auprc = average_precision_score(y_true, s)
        flip_note = " [flipped]" if flipped else ""
        print(f"[{tag}] Test AUROC={auroc:.4f} AUPRC={auprc:.4f}{flip_note}")
    except Exception as e:
        print(f"[{tag}] Could not compute overall AUROC/AUPRC:", e)

def _print_subset_aucs(scores: np.ndarray, y: np.ndarray, g: np.ndarray, tag: str):
    """
    Print AUROC/AUPRC for TARGETED and NON-TARGETED subsets, mirroring the
    thresholded definitions used in evaluate_at_far():
      - TARGETED positives: y==1 & g==1; everything else is negative
      - NON-TARGETED positives: y==1 & g==0; everything else is negative
    Also prints (n_pos, n_neg) used for each curve.
    """
    # orient once using overall anomaly labels
    scores, flipped = _orient_scores_for_anomaly(scores, y)
    flip_note = " [flipped]" if flipped else ""

    def _one(pos_mask: np.ndarray, name: str):
        y_bin = np.zeros_like(y)
        y_bin[pos_mask] = 1
        n_pos = int(np.sum(y_bin == 1))
        n_neg = int(np.sum(y_bin == 0))
        if n_pos == 0 or n_neg == 0:
            print(f"[{tag}] {name} AUROC/AUPRC: undefined (n_pos={n_pos}, n_neg={n_neg})")
            return
        try:
            auroc = roc_auc_score(y_bin, scores)
            auprc = average_precision_score(y_bin, scores)
            print(f"[{tag}] {name} AUROC={auroc:.4f} AUPRC={auprc:.4f} (n_pos={n_pos}, n_neg={n_neg}){flip_note}")
        except Exception as e:
            print(f"[{tag}] {name} AUROC/AUPRC error: {e} (n_pos={n_pos}, n_neg={n_neg})")

    _one((y == 1) & (g == 1), "TARGETED")
    _one((y == 1) & (g == 0), "NON-TARGETED")


# =========================
# FAR-cap evaluation (with orientation)
# =========================
def evaluate_at_far(model, ds, device, far_cap=0.05, boundary=None):
    """
    Evaluate with either:
      - boundary='val'  : choose tau on validation normals so FAR <= far_cap (uses oriented scores)
      - boundary='zero' : use native boundary tau = 0.0 (no score flipping)

    Reports:
      OVERALL  (all anomalies vs all)
      TARGETED (positives = targeted anomalies; negatives = normals only)
      NON-TARGETED (positives = non-targeted anomalies; negatives = normals only)
    """
    # ----- resolve boundary mode -----
    mode = (boundary or getattr(model, "boundary", "val")).lower()
    if mode not in ("val", "zero"):
        mode = "val"

    # ----- validation -----
    Xv = ds.val_set.X
    yv = ds.val_set.y.numpy()  # 0 normal, 1 anomaly
    n_valn = int(np.sum(yv == 0))

    # raw scores (do NOT flip yet)
    val_scores_all_raw = _get_scores(model, Xv, device=device)

    if mode == "val":
        # Orient so 'higher = more anomalous' and pick tau at FAR cap on validation normals
        val_scores_all, flipped = _orient_scores_for_anomaly(val_scores_all_raw, yv)
        val_scores_norm = np.asarray(val_scores_all[:n_valn], dtype=float)
        tau = _pick_threshold_at_far(val_scores_norm, far_cap)
    else:
        # Native boundary (e.g., OC-SVM: 0, Kernel SVDD: d^2 - R^2 = 0); do NOT flip
        flipped = False
        tau = 0.0

    # ----- test -----
    Xt = ds.test_set.X
    test_scores_raw = _get_scores(model, Xt, device=device)

    if mode == "val" and flipped:
        test_scores = -np.asarray(test_scores_raw, float)
    else:
        test_scores = np.asarray(test_scores_raw, float)

    y = ds.test_set.y.numpy()  # 0 normal, 1 anomaly
    g = ds.test_set.g.numpy()  # 1 targeted, 0 non-targeted (normals have 0)
    y_pred = (test_scores > tau).astype(int)

    # OVERALL: positives = all anomalies (1-vs-all)
    overall = _metrics(y, y_pred)

    # TARGETED: positives = targeted anomalies; negatives = normals only
    mask_t    = (y == 0) | ((y == 1) & (g == 1))
    y_true_t  = ((y == 1) & (g == 1))[mask_t].astype(int)
    y_pred_t  = y_pred[mask_t]
    targeted  = _metrics(y_true_t, y_pred_t)

    # NON-TARGETED: positives = non-targeted anomalies; negatives = normals only
    mask_nt   = (y == 0) | ((y == 1) & (g == 0))
    y_true_nt = ((y == 1) & (g == 0))[mask_nt].astype(int)
    y_pred_nt = y_pred[mask_nt]
    non_targeted = _metrics(y_true_nt, y_pred_nt)

    return dict(
        threshold=float(tau),
        FAR_cap=float(far_cap),
        boundary_mode=mode,
        overall=overall,
        targeted=targeted,
        non_targeted=non_targeted,
    )


# =================
# Targeting helpers
# =================
import json, ast

def _parse_target_families(arg):
    """
    Accepts:
      CSV:      Adduser,Hydra_SSH
      JSON:     ["Adduser","Hydra_SSH"]
      Python:   ['Adduser','Hydra_SSH']
      And also handles accidental surrounding quotes.
    Returns all-lowercase substrings.
    """
    if arg is None:
        return []
    if isinstance(arg, (list, tuple)):
        return [str(x).strip().lower() for x in arg]

    s = str(arg).strip()
    # strip accidental outer quotes
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # try JSON list
    if s.startswith("[") and s.endswith("]"):
        try:
            return [str(x).strip().lower() for x in json.loads(s)]
        except Exception:
            pass
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return [str(x).strip().lower() for x in obj]
        except Exception:
            pass

    # fallback: CSV
    return [t.strip().lower() for t in s.split(",") if t.strip()]


def _extract_family_name(meta: dict) -> str:
    for k in ("anomaly_label","family","attack_family","attack","name","label_name","category"):
        v = meta.get(k, None)
        if isinstance(v, str) and v:
            return v
    return ""


def _derive_target_flags_from_metas(metas, labels, target_families=None, target_frac=None, seed=42):
    """
    Build g (0/1) from metas for anomalies only.
    - If target_families is provided: g=1 iff family contains any substring (case-insensitive).
    - Else if target_frac is provided: random Bernoulli per anomaly.
    - Else: fall back to metas['targeted_flag'] if present, otherwise zeros.
    """
    labels = np.asarray(labels, int)
    n = len(metas)
    g = np.zeros(n, dtype=int)

    # Prefer explicit families rule
    if target_families:
        tf = [s.lower() for s in target_families]
        for i, m in enumerate(metas):
            if labels[i] == 1:
                fam = _extract_family_name(m).lower()
                g[i] = int(any(sub in fam for sub in tf))
        return g

    # Then randomized split if requested
    if target_frac is not None:
        rng = np.random.default_rng(seed)
        for i in range(n):
            if labels[i] == 1:
                g[i] = int(rng.random() < float(target_frac))
        return g

    # Finally, respect any existing targeted_flag
    for i, m in enumerate(metas):
        if labels[i] == 1:
            g[i] = int(m.get("targeted_flag", 0))
    return g


# =================
# Data utilities
# =================
def _stack_val(X_valn, metas_valn, X_vala, metas_vala,
               target_families, target_frac, seed, override_flags) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_val = np.vstack([X_valn, X_vala])
    y_val = np.concatenate([np.zeros(X_valn.shape[0], dtype=int),
                            np.ones(X_vala.shape[0], dtype=int)])

    # normals -> g=0; anomalies determine via rules
    g_norm = np.zeros(X_valn.shape[0], dtype=int)

    # compute anomaly g from metas_vala
    if override_flags or not any(("targeted_flag" in m) for m in metas_vala):
        g_anom = _derive_target_flags_from_metas(
            metas_vala, labels=np.ones(len(metas_vala), dtype=int),
            target_families=target_families, target_frac=target_frac, seed=seed
        )
    else:
        g_anom = np.array([int(m.get("targeted_flag", 0)) for m in metas_vala], dtype=int)

    g_val = np.concatenate([g_norm, g_anom])
    return X_val, y_val, g_val

def make_dataset_from_features(data_dir: str,
                               target_families: List[str],
                               target_frac: Optional[float],
                               override_flags: bool,
                               seed: int) -> BaseADDataset:
    tr = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
    X_train, M_train = tr["X"], tr["metas"]
    y_train = np.array([int(m.get("label", 0)) for m in M_train], dtype=int)

    vn = np.load(os.path.join(data_dir, "val_norm.npz"), allow_pickle=True)
    X_valn, M_valn = vn["X"], vn["metas"]
    va = np.load(os.path.join(data_dir, "val_anom.npz"), allow_pickle=True)
    X_vala, M_vala = va["X"], va["metas"]

    # Train g
    if override_flags or not any(("targeted_flag" in m) for m in M_train):
        g_train = _derive_target_flags_from_metas(
            M_train, labels=y_train, target_families=target_families,
            target_frac=target_frac, seed=seed
        )
    else:
        g_train = np.array([int(m.get("targeted_flag", 0)) for m in M_train], dtype=int)

    # Val stacked
    X_val, y_val, g_val = _stack_val(
        X_valn, M_valn, X_vala, M_vala,
        target_families=target_families,
        target_frac=target_frac,
        seed=seed,
        override_flags=override_flags
    )

    # Test
    t = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
    X_test, M_test = t["X"], t["metas"]
    y_test = np.array([int(m.get("label", 0)) for m in M_test], dtype=int)

    if override_flags or not any(("targeted_flag" in m) for m in M_test):
        g_test = _derive_target_flags_from_metas(
            M_test, labels=y_test, target_families=target_families,
            target_frac=target_frac, seed=seed
        )
    else:
        g_test = np.array([int(m.get("targeted_flag", 0)) if int(m.get("label", 0)) == 1 else 0
                           for m in M_test], dtype=int)

    train_set = DictDataset(X_train, y_train, g_train)  # SR uses anomalies; baselines subset normals
    val_set   = DictDataset(X_val,   y_val,   g_val)
    test_set  = DictDataset(X_test,  y_test,  g_test)
    return BaseADDataset(train_set, val_set, test_set)

def _subset_normals(ds: BaseADDataset) -> BaseADDataset:
    """Return a view where train_set contains only normals; val/test unchanged."""
    idx = (ds.train_set.y.numpy() == 0)
    X = ds.train_set.X[idx]
    y = ds.train_set.y.numpy()[idx]
    g = ds.train_set.g.numpy()[idx]
    return BaseADDataset(DictDataset(X, y, g), ds.val_set, ds.test_set)

def _print_split_summary(ds):
    def _one(split, name):
        y = split.y.numpy()
        g = split.g.numpy()
        n_norm = int(np.sum(y == 0))
        n_anom = int(np.sum(y == 1))
        n_targ = int(np.sum((y == 1) & (g == 1)))
        n_nont = int(np.sum((y == 1) & (g == 0)))
        print(f"[ADFA] {name}: normals={n_norm} anomalies={n_anom} (targeted={n_targ}, non-targeted={n_nont})")
    _one(ds.train_set, "Train")
    _one(ds.val_set,   "Val")
    _one(ds.test_set,  "Test")
    # Warn if no targeted anomalies in val/test
    yv, gv = ds.val_set.y.numpy(), ds.val_set.g.numpy()
    yt, gt = ds.test_set.y.numpy(), ds.test_set.g.numpy()
    if (np.sum((yv==1)&(gv==1))==0) or (np.sum((yt==1)&(gt==1))==0):
        print("[WARN] No targeted anomalies found in val/test — TARGETED metrics will be zeros.")


# ==========================
# Single-model run wrappers
# ==========================
def run_sr_deepsvdd(ds: BaseADDataset, args):
    # g=1 targeted, g=0 non-targeted  (use same keys everywhere)
    severity = {1: float(args.sr_sev1), 0: float(args.sr_sev2)}
    margins  = {1: float(args.sr_margin1), 0: float(args.sr_margin2)}

    model = SRDeepSVDD(
        objective='soft-boundary',        # <<< ensure soft-boundary
        nu=args.sr_nu,
        severity_weights=severity,
        margin_per_group=margins
    )
    model.boundary = args.sr_boundary 


    hidden = tuple(int(x) for x in str(args.sr_hidden).split(',')) if args.sr_hidden else (256, 128)
    model.set_network(args.sr_network, input_dim=ds.train_set.X.shape[1],
                      hidden_dims=hidden, rep_dim=int(args.sr_rep))

    model.train(ds, optimizer_name='adamw', lr=args.sr_lr, n_epochs=args.sr_epochs,
                batch_size=args.sr_batch, weight_decay=args.sr_weight_decay, device=args.device)
    model.test(ds, device=args.device)

    if hasattr(model, "results") and "test_scores" in model.results and ds.test_set is not None:
        y_test = ds.test_set.y.numpy()
        g_test = ds.test_set.g.numpy()
        scores = np.asarray(model.results["test_scores"])
        _print_overall_aucs(scores, y_test, "SR-DeepSVDD")
        _print_subset_aucs(scores, y_test, g_test, "SR-DeepSVDD")

    rpt = evaluate_at_far(model, ds, device=args.device, far_cap=args.far_target,boundary=args.sr_boundary)
    print(f"[SR-DeepSVDD] tau={rpt['threshold']:.6f}  FAR_cap={rpt['FAR_cap']}")
    print(f"[SR-DeepSVDD] OVERALL     -> {rpt['overall']}")
    print(f"[SR-DeepSVDD] TARGETED    -> {rpt['targeted']}")
    print(f"[SR-DeepSVDD] NON-TARGETED-> {rpt['non_targeted']}")
    return getattr(model, "results", {})

def run_deepsvdd(ds: BaseADDataset, args):
    ds_norm = _subset_normals(ds)

    # ✅ use DeepSVDD, not SRDeepSVDD
    model = DeepSVDD(objective='soft-boundary', nu=args.dsvdd_nu)
    model.boundary = args.dsvdd_boundary  

    hidden = tuple(int(x) for x in str(args.dsvdd_hidden).split(',')) if args.dsvdd_hidden else (256, 128)
    model.set_network(args.dsvdd_network,
                      input_dim=ds_norm.train_set.X.shape[1],
                      hidden_dims=hidden,
                      rep_dim=int(args.dsvdd_rep))

    model.train(ds_norm, optimizer_name='adamw', lr=args.dsvdd_lr, n_epochs=args.dsvdd_epochs,
                batch_size=args.dsvdd_batch, weight_decay=args.dsvdd_weight_decay, device=args.device)
    model.test(ds_norm, device=args.device)

    if hasattr(model, "results") and "test_scores" in model.results and ds_norm.test_set is not None:
        y_test = ds_norm.test_set.y.numpy()
        g_test = ds_norm.test_set.g.numpy()
        scores = np.asarray(model.results["test_scores"])
        _print_overall_aucs(scores, y_test, "DeepSVDD")
        _print_subset_aucs(scores, y_test, g_test, "DeepSVDD")

    rpt = evaluate_at_far(model, ds_norm, device=args.device, far_cap=args.far_target,boundary=args.dsvdd_boundary)
    print(f"[DeepSVDD]   tau={rpt['threshold']:.6f}  FAR_cap={rpt['FAR_cap']}")
    print(f"[DeepSVDD]   OVERALL     -> {rpt['overall']}")
    print(f"[DeepSVDD]   TARGETED    -> {rpt['targeted']}")
    print(f"[DeepSVDD]   NON-TARGETED-> {rpt['non_targeted']}")
    return getattr(model, "results", {})

def run_lsvdd(ds: BaseADDataset, args) -> Dict[str, Any]:
    if not _HAS_LSVDD:
        print("[LinearSVDD] Not available in repo. Skipping.")
        return {}
    ds_norm = _subset_normals(ds)
    model = LinearSVDD(nu=args.lsvdd_nu)
    model.train(ds_norm, lr=args.lsvdd_lr, n_epochs=args.lsvdd_epochs, batch_size=args.lsvdd_batch, device=args.device)
    model.test(ds_norm, device=args.device)

    rpt = evaluate_at_far(model, ds_norm, device=args.device, far_cap=args.far_target)
    print(f"[LinearSVDD] tau={rpt['threshold']:.6f}  FAR_cap={rpt['FAR_cap']}")
    print(f"[LinearSVDD] OVERALL     -> {rpt['overall']}")
    print(f"[LinearSVDD] TARGETED    -> {rpt['targeted']}")
    print(f"[LinearSVDD] NON-TARGETED-> {rpt['non_targeted']}")
    return getattr(model, "results", {})

def run_ocsvm(ds: BaseADDataset, args) -> Dict[str, Any]:
    if not _HAS_OCSVM:
        print("[OC-SVM] Not available in repo. Skipping.")
        return {}

    class _XYTrainView:
        def __init__(self, X, y):
            self.train_set = (np.asarray(X), np.asarray(y))

    ds_norm = _subset_normals(ds)
    model = OCSVM(nu=args.ocsvm_nu, gamma=args.ocsvm_gamma)
    model.boundary = args.ocsvm_boundary 

    # Prefer the dataset API if present, but pass exactly (X,y)
    if hasattr(model, "fit_on_dataset"):
        Xn = ds_norm.train_set.X
        yn = ds_norm.train_set.y.numpy()
        model.fit_on_dataset(_XYTrainView(Xn, yn))
    elif hasattr(model, "fit"):
        Xn = ds_norm.train_set.X
        if torch.is_tensor(Xn):
            Xn = Xn.detach().cpu().numpy()
        model.fit(np.asarray(Xn))
    else:
        raise AttributeError("OCSVM baseline lacks fit/train API.")

    rpt = evaluate_at_far(model, ds_norm, device=args.device, far_cap=args.far_target,
                          boundary=args.ocsvm_boundary)  # ← pass boundary
    print(f"[OC-SVM]     tau={rpt['threshold']:.6f}  FAR_cap={rpt['FAR_cap']}  boundary={rpt['boundary_mode']}")
    print(f"[OC-SVM]     OVERALL     -> {rpt['overall']}")
    print(f"[OC-SVM]     TARGETED    -> {rpt['targeted']}")
    print(f"[OC-SVM]     NON-TARGETED-> {rpt['non_targeted']}")
    return {}

def run_ksvdd(ds: BaseADDataset, args) -> Dict[str, Any]:
    if not _HAS_KSVDD:
        print("[Kernel SVDD] Not available in repo. Skipping.")
        return {}

    class _XYTrainView:
        def __init__(self, X, y):
            self.train_set = (np.asarray(X), np.asarray(y))

    ds_norm = _subset_normals(ds)
    model = KernelSVDD(nu=args.ksvdd_nu, gamma=args.ksvdd_gamma)
    model.boundary = args.ksvdd_boundary 

    if hasattr(model, "fit_on_dataset"):
        Xn = ds_norm.train_set.X
        yn = ds_norm.train_set.y.numpy()
        model.fit_on_dataset(_XYTrainView(Xn, yn))
    elif hasattr(model, "fit"):
        Xn = ds_norm.train_set.X
        if torch.is_tensor(Xn):
            Xn = Xn.detach().cpu().numpy()
        model.fit(np.asarray(Xn))
    else:
        raise AttributeError("KernelSVDD baseline lacks fit API.")

    rpt = evaluate_at_far(model, ds_norm, device=args.device, far_cap=args.far_target,
                          boundary=args.ksvdd_boundary)  
    print(f"[KernelSVDD] tau={rpt['threshold']:.6f}  FAR_cap={rpt['FAR_cap']}  boundary={rpt['boundary_mode']}")
    print(f"[KernelSVDD] OVERALL     -> {rpt['overall']}")
    print(f"[KernelSVDD] TARGETED    -> {rpt['targeted']}")
    print(f"[KernelSVDD] NON-TARGETED-> {rpt['non_targeted']}")
    return {}

# =========
#   Main
# =========
def _resolve_device(dev):
    """Return 'cuda' only if actually available; otherwise fall back to 'cpu'."""
    d = str(dev).lower() if isinstance(dev, str) else 'cpu'
    if d == 'auto':
        return 'cuda' if (torch.cuda.is_available() and getattr(torch.version, 'cuda', None)) else 'cpu'
    if d.startswith('cuda') and not torch.cuda.is_available():
        print("[WARN] CUDA requested but torch is CPU-only or no GPU present; falling back to 'cpu'.")
        return 'cpu'
    return d

def main():
    

    p = argparse.ArgumentParser()
    # Prep / paths
    p.add_argument("--prepare", action="store_true", help="Build features from raw ADFA traces")
    p.add_argument("--adfa-root", default="data/ADFA-LD", help="Root folder with ADFA trace subfolders")
    p.add_argument("--data-dir",  default="data/adfa_features", help="Where to store/find prepared features")
    p.add_argument("--ngram", default="1,3", help="n-gram range as 'min,max' (e.g., 1,3)")
    p.add_argument("--max-features", type=int, default=5000, help="TF-IDF vocabulary cap")
    p.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction of normals when preparing")
    p.add_argument("--seed", type=int, default=42, help="Seed used in feature prep split")
    p.add_argument("--anom-train-frac", type=float, default=0.2, help="Fraction of anomalies per family sent to TRAIN")
    
    p.add_argument("--tfidf-mode", default="all", choices=["all","normals"],
                  help="TF-IDF fitting mode: 'all' = train normals + train anomalies; 'normals' = train normals only.")
    # Targeting controls (new)
    p.add_argument("--target-families", default=None,
                   help="Comma-separated substrings or list string to label anomalies as targeted by family/name "
                        "(e.g., 'Hydra,Adduser' or \"['Hydra_SSH','Hydra_FTP']\").")
    p.add_argument("--target-frac", type=float, default=None,
                   help="If set, randomly mark this fraction of anomalies as targeted (per split).")
    p.add_argument("--target-override", action="store_true",
                   help="Override any saved targeted_flag in metas with rules above.")

    # Modes / orchestration
    p.add_argument("--mode", default="sr", choices=["sr","deepsvdd","svdd_linear","ocsvm","svdd_rbf","all","grid"])
    p.add_argument("--models", default="sr,deepsvdd,svdd_linear,ocsvm,svdd_rbf", help="Comma-list for --mode grid/all")

    # Device / runtime
    p.add_argument("--device", default="cpu")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--grid-res", type=int, default=300)
    p.add_argument("--plot-train", type=int, default=0)
    p.add_argument("--plot-scores", type=int, default=0)

    # Validation / SR policy
    p.add_argument("--far-target", type=float, default=0.05, help="FAR cap used for threshold selection")
    p.add_argument("--target-type", type=int, default=1, help="Targeted anomaly group index (kept for parity)")

    # Multi-seed hooks (parity with earlier runners)
    p.add_argument("--seeds", default=None, help="Comma-separated seeds (handled by external runner)")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--select-metric", default="DR1")
    p.add_argument("--select-direction", default="max")
    p.add_argument("--pval-metric", default="DR1")
    p.add_argument("--pval-direction", default="max")
    p.add_argument("--pval-ref", default="SR-DeepSVDD")
    p.add_argument("--pval-test", default="wilcoxon")

    # --- SR-DeepSVDD knobs & boundary ---
    p.add_argument("--sr-boundary", default="val", choices=["val","zero"])
    p.add_argument("--sr-nu", type=float, default=0.05)
    p.add_argument("--sr-rep", type=int, default=64)
    p.add_argument("--sr-sev1", type=float, default=5.0)    # reserved
    p.add_argument("--sr-sev2", type=float, default=2.0)    # reserved
    p.add_argument("--sr-margin1", type=float, default=0.5) # reserved
    p.add_argument("--sr-margin2", type=float, default=0.5) # reserved
    p.add_argument("--sr-lr", type=float, default=1e-3)
    p.add_argument("--sr-epochs", type=int, default=50)
    p.add_argument("--sr-batch", type=int, default=128)
    p.add_argument("--sr-weight-decay", type=float, default=1e-6)
    p.add_argument("--sr-network", default="mlp_nobias")
    p.add_argument("--sr-hidden", default="256,128")

    # --- DeepSVDD knobs & boundary ---
    p.add_argument("--dsvdd-boundary", default="val", choices=["val","zero"])
    p.add_argument("--dsvdd-nu", type=float, default=0.05)
    p.add_argument("--dsvdd-rep", type=int, default=64)
    p.add_argument("--dsvdd-lr", type=float, default=1e-3)
    p.add_argument("--dsvdd-epochs", type=int, default=50)
    p.add_argument("--dsvdd-batch", type=int, default=128)
    p.add_argument("--dsvdd-weight-decay", type=float, default=1e-6)
    p.add_argument("--dsvdd-network", default="mlp_nobias")
    p.add_argument("--dsvdd-hidden", default="256,128")

    # --- Linear SVDD boundary & knobs ---
    p.add_argument("--lsvdd-boundary", default="val", choices=["val","zero"])
    p.add_argument("--lsvdd-nu", type=float, default=0.05)
    p.add_argument("--lsvdd-lr", type=float, default=1e-2)
    p.add_argument("--lsvdd-epochs", type=int, default=300)
    p.add_argument("--lsvdd-batch", type=int, default=256)

    # --- OCSVM boundary & knobs ---
    p.add_argument("--ocsvm-boundary", default="val", choices=["val","zero"])
    p.add_argument("--ocsvm-nu", type=float, default=0.05)
    p.add_argument("--ocsvm-gamma", default="scale")

    # --- Kernel SVDD boundary & knobs ---
    p.add_argument("--ksvdd-boundary", default="val", choices=["val","zero"])
    p.add_argument("--ksvdd-nu", type=float, default=0.05)
    p.add_argument("--ksvdd-gamma", type=float, default=1.0)
    
    

    args = p.parse_args()
    seed_everything(args.seed)
    args.device = _resolve_device(args.device)

    # Normalize paths to repo root (folder containing 'src')
    ROOT = Path(__file__).resolve().parents[1]
    def _norm(pth):
        pth = Path(pth)
        return pth if pth.is_absolute() else (ROOT / pth)
    args.adfa_root = str(_norm(args.adfa_root))
    args.data_dir  = str(_norm(args.data_dir))

    target_families = _parse_target_families(args.target_families)
    print(f"[ADFA] Parsed target families: {target_families or '[] (none)'}")

    # 1) Prepare features if requested
    if args.prepare:
        n_min, n_max = [int(x) for x in args.ngram.split(",")]
        build_and_save_features(root_dir=args.adfa_root,
                            out_dir=args.data_dir,
                            ngram_range=(n_min, n_max),
                            max_features=args.max_features,
                            val_frac=args.val_frac,
                            seed=args.seed,
                            anom_train_frac=args.anom_train_frac,
                            target_families=target_families,
                            target_frac=args.target_frac,
                            tfidf_mode=args.tfidf_mode)
        print("[ADFA] Preparation done.")
        return

    # 2) Load feature splits (with targeting rules applied if requested)
    ds = make_dataset_from_features(args.data_dir,
                                    target_families=target_families,
                                    target_frac=args.target_frac,
                                    override_flags=args.target_override,
                                    seed=args.seed)
    _print_split_summary(ds)
    # show a few anomaly family names if no targeted found
    yv, gv = ds.val_set.y.numpy(), ds.val_set.g.numpy()
    yt, gt = ds.test_set.y.numpy(), ds.test_set.g.numpy()
    if (gv[yv==1].sum()==0) or (gt[yt==1].sum()==0):
        import numpy as np, collections, os
        def top_fams(npz_path):
            metas = np.load(npz_path, allow_pickle=True)["metas"]
            fams  = [m.get("anomaly_label","") for m in metas if int(m.get("label",0))==1]
            return collections.Counter(fams).most_common(10)
        print("[DBG] Val anomaly families:", top_fams(os.path.join(args.data_dir, "val_anom.npz")))
        print("[DBG] Test anomaly families:", top_fams(os.path.join(args.data_dir, "test.npz")))

    # 3) Execute requested mode(s)
    def _do(m: str):
        m = m.strip().lower()
        if m == "sr":
            return run_sr_deepsvdd(ds, args)
        if m == "deepsvdd":
            return run_deepsvdd(ds, args)
        if m == "svdd_linear":
            return run_lsvdd(ds, args)
        if m == "ocsvm":
            return run_ocsvm(ds, args)
        if m == "svdd_rbf":
            return run_ksvdd(ds, args)
        print(f"[WARN] Unknown mode '{m}'")
        return {}

    if args.mode == "all":
        for m in [s for s in args.models.split(",") if s]:
            _do(m)
    elif args.mode == "grid":
        for m in [s for s in args.models.split(",") if s]:
            _do(m)
    else:
        _do(args.mode)


if __name__ == "__main__":
    main()
