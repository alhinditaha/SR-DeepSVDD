#!/usr/bin/env python3
"""
SR-DeepSVDD — Code Runner v4
- Adds anomaly centers control: --mu-a1, --mu-a2 (strings "x,y")
- Moves boundary selection to per-model flags:
    --sr-boundary, --dsvdd-boundary, --lsvdd-boundary, --ocsvm-boundary, --ksvdd-boundary
- Keeps sr_sev2, sr_margin2, and split controls (fractions/sizes).
- Vectorized grid (internal/explode), one-shot “all”, and per-model runs.
- NEW: passes plotting flags, device, and multi-seed/statistics flags to the main script.
"""

import os, shlex, subprocess
from pathlib import Path
from itertools import product
from datetime import datetime
from typing import Iterable, List, Union

# =========================
# GLOBAL EXECUTION CONTROLS
# =========================
PY: str = "python"        # interpreter; "python3" or a full path if needed
EXECUTE: bool = True      # True: run commands; False: only print them
PARALLEL: bool = False    # parallelize only for GRID_MODE='explode'
N_WORKERS: int = 4
SAVE_CMDS_TO: Union[str, None] = "commands_run_log.txt"
GRID_MODE: str = "internal"   # 'internal' (comma-lists to --mode grid) or 'explode' (cartesian here)

# =================
# FILES & PATHS
# =================
SRC = Path(__file__).resolve().parent
MAIN = SRC / "run_sr_deepsvdd.py"

# =================
# DATASET SETTINGS
# =================
# All dataset parameters can be scalars or lists (vectors).
# The script will pick the FIRST value for single runs and pass comma-joined values for internal grid.
# For external grid, the cartesian product will be expanded across any lists here as well.

# Geometric shape (banana) + class sizes
rotate      = [0]          # rotation in degrees of the base banana prior; larger rotates the manifold
n_norm      = [700]        # number of normal samples
n_anom1     = [200]        # number of anomaly Type-1 (targeted/severe)
n_anom2     = [200]        # number of anomaly Type-2 (non-targeted)
b           = [0.2]        # curvature/bend of the banana; ↑b => more curvature (harder separation)
s1          = [2.0]        # scale factor for normal cluster 1 spread; ↑s widens distribution
s2          = [1.5]        # scale factor for normal cluster 2 spread
seed        = [123]        # RNG seed(s); you can list multiple to average robustness across seeds

# Anomaly covariance + centers (strings the main script expects)
cov_a1      = ["0.5,0.9"]  # covariance for anomaly type 1 (variance/off-diagonal scale)
cov_a2      = ["0.5,0.9"]  # covariance for anomaly type 2
mu_a1       = ["0.0,4"]     # << NEW: anomaly Type-1 center; "x,y". ↑|μ| moves cluster away (usually easier to detect)
mu_a2       = ["0.0,-5"]    # << NEW: anomaly Type-2 center; "x,y"

# =====================
# DATA SPLIT CONTROLS
# =====================
# Choose either fractions OR sizes (if both provided, sizes win).
# Fractions must sum to 1.0; sizes must sum to n_total for each class internally.
train_frac  = [0.6]       # e.g., [0.6, 0.7] for a sweep
val_frac    = [0.2]
test_frac   = [0.2]

train_size  = [None]      # e.g., [100] to force exact counts; keep None to ignore
val_size    = [None]
test_size   = [None]

# ======================================
# VALIDATION / CALIBRATION (CROSS-CUTTING)
# ======================================
far_target  = [0.05]      # FAR cap (None to disable). SR uses it as a hard cap for targeted-DR selection; baselines use as cap for F1 thresholding
target_type = [1]         # which anomaly group SR targets in validation (1=Type-1)

# =========================================
# PER-MODEL BOUNDARY SELECTION (MOVED HERE)
# =========================================
# 'val' uses validation-picked threshold; 'zero' uses raw s(x)=0 boundary.
# Keep model-specific control so you can compare, e.g., SR with 'val' vs OC-SVM with 'zero'.
sr_boundary     = ["val"]
dsvdd_boundary  = ["val"]
lsvdd_boundary  = ["val"]
ocsvm_boundary  = ["zero"]
ksvdd_boundary  = ["zero"]

# =================
# SR-DeepSVDD PARAMS
# =================
# NOTE: All can be lists for grid. The repo expects these flags:
#   --sr-nu, --sr-rep, --sr-sev1, --sr-sev2, --sr-margin1, --sr-margin2, --sr-lr, --sr-epochs, --sr-batch, --sr-weight-decay
sr_nu            = [0.05]     # ν for soft-boundary; ↑ν tighter boundary
sr_rep           = [16]       # embedding dim; ↑rep may help complex manifolds but can overfit
sr_sev1          = [5]      # severity weight for targeted anomalies; ↑sev1 pushes targeted further out
sr_sev2          = [2]      # severity weight for non-targeted anomalies; ↑sev2 pushes them further out
sr_margin1       = [0.5]      # extra margin beyond R² for targeted group; ↑margin1 widens safety gap (may ↑FAR if normals near boundary)
sr_margin2       = [0.5]      # extra margin beyond R² for non-targeted group; ↑ widens gap for group 2 (may ↑FAR)
sr_lr            = [1e-3]     # learning rate for SR-DeepSVDD
sr_epochs        = [50]       # training epochs
sr_batch         = [128]      # batch size
sr_weight_decay  = [1e-6]     # AdamW weight decay

# =================
# DeepSVDD PARAMS
# =================
# Flags:
#   --dsvdd-nu, --dsvdd-rep, --dsvdd-lr, --dsvdd-epochs, --dsvdd-batch, --dsvdd-weight-decay
dsvdd_nu            = [0.05] # ν for soft-boundary; ↑ν tighter boundary
dsvdd_rep           = [16]
dsvdd_lr            = [1e-3]
dsvdd_epochs        = [50]
dsvdd_batch         = [128]
dsvdd_weight_decay  = [1e-6]

# =================
# Linear SVDD PARAMS
# =================
# Flags:
#   --lsvdd-nu, --lsvdd-lr, --lsvdd-epochs, --lsvdd-batch
lsvdd_nu     = [0.05]
lsvdd_lr     = [1e-2]
lsvdd_epochs = [300]
lsvdd_batch  = [256]

# =================
# OC-SVM PARAMS
# =================
# Flags:
#   --ocsvm-nu, --ocsvm-gamma
ocsvm_nu    = [0.05]   # ν for OC-SVM; controls expected outlier fraction
ocsvm_gamma = ["scale", 2] # 'scale' or numeric; bandwidth of RBF; ↑gamma => tighter boundary

# =================
# Kernel SVDD (RBF) PARAMS
# =================
# Flags:
#   --ksvdd-nu, --ksvdd-gamma
ksvdd_nu    = [0.05]
ksvdd_gamma = [2]

# =================
# DEVICE & PLOTTING (passed through)
# =================
DEVICE       = "cuda"    # --device (e.g., "cuda" or "cpu")
PLOTS_DIR    = "plots"   # --plots-dir
GRID_RES     = 300       # --grid-res
PLOT_TRAIN   = 1         # --plot-train (1/0)
PLOT_SCORES  = 1         # --plot-scores (1/0)

# ==========================
# MULTI-SEED & STATISTICS
# ==========================
# You can either:
#   (A) leave SEEDS_CSV=None and N_SEEDS=None to behave like before (use --seed)
#   (B) set SEEDS_CSV to a comma string like "123,124,125" to use --seeds
#   (C) set N_SEEDS to an int (e.g., 10) to use --n-seeds starting at --seed
SEEDS_CSV        = None#"123,1,2,3,4,5,6,7,8,9"            # e.g., "123,124,125"
N_SEEDS: Union[int, None] = None   # e.g., 10

# Selection for best-config-per-seed (used when you pass vectors / grids)
SELECT_METRIC       = "DR1"   # <- use DR1 for Type-1 (or "DR2" for Type-2)
SELECT_DIRECTION    = "max"

# P-value report settings across seeds
PVAL_METRIC         = "DR1"   # <- same as above
PVAL_DIRECTION      = "max"

PVAL_REF            = "SR-DeepSVDD" # --pval-ref
PVAL_TEST           = "wilcoxon"    # --pval-test ('wilcoxon' or 'ttest')

# =================
# WHICH RUNS TO DO
# =================
RUN_ONE_SHOT_ALL  = True   # runs --mode all once (uses FIRST values of each param)
RUN_INDIVIDUAL    = False   # runs each model individually once (FIRST values)
RUN_GRID          = False  # runs grid search (internal or external based on GRID_MODE)

GRID_MODELS = ["sr", "deepsvdd", "svdd_linear", "ocsvm", "svdd_rbf"]


# ================
# Helper Functions
# ================
def to_csv(xs: Iterable) -> str:
    """Comma-join values for flags that accept comma-lists."""
    return ",".join(str(x) for x in xs)

def first(xs: Iterable):
    """Pick the first value for single/one-shot runs."""
    return list(xs)[0]

def run(cmd: str):
    print(cmd)
    if SAVE_CMDS_TO:
        with open(SAVE_CMDS_TO, "a", encoding="utf-8") as f:
            f.write(cmd + "\n")
    if EXECUTE:
        try:
            # capture output so argparse errors are visible
            res = subprocess.run(shlex.split(cmd), check=True, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.stdout: print(res.stdout)
            if res.stderr: print(res.stderr)
        except subprocess.CalledProcessError as e:
            print("RUN FAILED:")
            if e.stdout: print(e.stdout)
            if e.stderr: print(e.stderr)


def build_base_cmd(far_target_val, target_type_val,
                   rotate_val, n_norm_val, n_anom1_val, n_anom2_val,
                   b_val, s1_val, s2_val, seed_val, cov_a1_val, cov_a2_val, mu_a1_val, mu_a2_val,
                   split_frac_csv: Union[str, None],
                   split_size_csv: Union[str, None]):
    cmd = (f'{PY} "{MAIN}"'
           f' --rotate {rotate_val}'
           f' --n-norm {n_norm_val}'
           f' --n-anom1 {n_anom1_val}'
           f' --n-anom2 {n_anom2_val}'
           f' --b {b_val}'
           f' --s1 {s1_val}'
           f' --s2 {s2_val}'
           f' --seed {seed_val}'
           f' --cov-a1 {cov_a1_val}'
           f' --cov-a2 {cov_a2_val}')
    # New anomaly centers (optional for backward compatibility)
    if mu_a1_val is not None and str(mu_a1_val).lower() != "none":
        cmd += f' --mu-a1 {mu_a1_val}'
    if mu_a2_val is not None and str(mu_a2_val).lower() != "none":
        cmd += f' --mu-a2 {mu_a2_val}'

    if far_target_val is not None and str(far_target_val).lower() != "none":
        cmd += f' --far-target {far_target_val} --target-type {target_type_val}'

    # Prefer explicit sizes if provided; otherwise pass fractions
    if split_size_csv and split_size_csv.lower() != "none":
        cmd += f' --split-size {split_size_csv}'
    elif split_frac_csv and split_frac_csv.lower() != "none":
        cmd += f' --split-frac {split_frac_csv}'

    # NEW: always pass device & plotting controls
    cmd += (f' --device {DEVICE}'
            f' --plots-dir {PLOTS_DIR} --grid-res {GRID_RES}'
            f' --plot-train {PLOT_TRAIN} --plot-scores {PLOT_SCORES}')

    # NEW: optional multi-seed & statistics flags
    if SEEDS_CSV:
        cmd += f' --seeds {SEEDS_CSV}'
    if N_SEEDS is not None:
        cmd += f' --n-seeds {int(N_SEEDS)}'
    if SELECT_METRIC:
        cmd += f' --select-metric {SELECT_METRIC} --select-direction {SELECT_DIRECTION}'
    if PVAL_METRIC:
        cmd += (f' --pval-metric {PVAL_METRIC} --pval-direction {PVAL_DIRECTION}'
                f' --pval-ref {PVAL_REF} --pval-test {PVAL_TEST}')
    return cmd

def add_sr_flags(cmd, boundary, nu, rep, sev1, sev2, margin1, margin2, lr, epochs, batch, wd):
    cmd = (cmd
           + f' --sr-boundary {boundary}'
           + f' --sr-nu {nu}'
           + f' --sr-rep {rep}'
           + f' --sr-sev1 {sev1}'
           + f' --sr-margin1 {margin1}'
           + f' --sr-lr {lr}'
           + f' --sr-epochs {epochs}'
           + f' --sr-batch {batch}'
           + f' --sr-weight-decay {wd}')
    # Optional to maintain backward compatibility with older argparse
    if sev2 is not None:
        cmd += f' --sr-sev2 {sev2}'
    if margin2 is not None:
        cmd += f' --sr-margin2 {margin2}'
    return cmd

def add_dsvdd_flags(cmd, boundary, nu, rep, lr, epochs, batch, wd):
    return (cmd
            + f' --dsvdd-boundary {boundary}'
            + f' --dsvdd-nu {nu}'
            + f' --dsvdd-rep {rep}'
            + f' --dsvdd-lr {lr}'
            + f' --dsvdd-epochs {epochs}'
            + f' --dsvdd-batch {batch}'
            + f' --dsvdd-weight-decay {wd}')

def add_lsvdd_flags(cmd, boundary, nu, lr, epochs, batch):
    return (cmd
            + f' --lsvdd-boundary {boundary}'
            + f' --lsvdd-nu {nu}'
            + f' --lsvdd-lr {lr}'
            + f' --lsvdd-epochs {epochs}'
            + f' --lsvdd-batch {batch}')

def add_ocsvm_flags(cmd, boundary, nu, gamma):
    return (cmd
            + f' --ocsvm-boundary {boundary}'
            + f' --ocsvm-nu {nu}'
            + f' --ocsvm-gamma {gamma}')

def add_ksvdd_flags(cmd, boundary, nu, gamma):
    return (cmd
            + f' --ksvdd-boundary {boundary}'
            + f' --ksvdd-nu {nu}'
            + f' --ksvdd-gamma {gamma}')


# ==================
# Main Orchestration
# ==================
def main():
    # log header
    if SAVE_CMDS_TO:
        with open(SAVE_CMDS_TO, "a", encoding="utf-8") as f:
            f.write(f"\n# ===== {datetime.now().isoformat()} =====\n")

    # Build split strings (single triple). For sweeping splits, use GRID_MODE='explode'.
    split_frac_csv = f"{first(train_frac)},{first(val_frac)},{first(test_frac)}" if all(
        v is not None for v in (first(train_frac), first(val_frac), first(test_frac))) else None

    _ts, _vs, _ss = first(train_size), first(val_size), first(test_size)
    split_size_csv = (f"{_ts},{_vs},{_ss}" if (_ts is not None and _vs is not None and _ss is not None) else None)

    # --------
    # ONE-SHOT
    # --------
    if RUN_ONE_SHOT_ALL:
        base = build_base_cmd(
            first(far_target), first(target_type),
            first(rotate), first(n_norm), first(n_anom1), first(n_anom2),
            first(b), first(s1), first(s2), first(seed),
            first(cov_a1), first(cov_a2), first(mu_a1), first(mu_a2),
            split_frac_csv, split_size_csv
        )
        # We pass per-model boundaries even for --mode all (your script should read them).
        cmd = base + " --mode all"
        cmd = add_sr_flags(cmd, first(sr_boundary),
                           first(sr_nu), first(sr_rep), first(sr_sev1), first(sr_sev2),
                           first(sr_margin1), first(sr_margin2),
                           first(sr_lr), first(sr_epochs), first(sr_batch), first(sr_weight_decay))
        cmd = add_dsvdd_flags(cmd, first(dsvdd_boundary),
                              first(dsvdd_nu), first(dsvdd_rep),
                              first(dsvdd_lr), first(dsvdd_epochs), first(dsvdd_batch), first(dsvdd_weight_decay))
        cmd = add_lsvdd_flags(cmd, first(lsvdd_boundary),
                              first(lsvdd_nu), first(lsvdd_lr), first(lsvdd_epochs), first(lsvdd_batch))
        cmd = add_ocsvm_flags(cmd, first(ocsvm_boundary),
                              first(ocsvm_nu), first(ocsvm_gamma))
        cmd = add_ksvdd_flags(cmd, first(ksvdd_boundary),
                              first(ksvdd_nu), first(ksvdd_gamma))
        run(cmd)

    # ------------
    # INDIVIDUALS
    # ------------
    if RUN_INDIVIDUAL:
        base = build_base_cmd(
            first(far_target), first(target_type),
            first(rotate), first(n_norm), first(n_anom1), first(n_anom2),
            first(b), first(s1), first(s2), first(seed),
            first(cov_a1), first(cov_a2), first(mu_a1), first(mu_a2),
            split_frac_csv, split_size_csv
        )
        # SR
        cmd = add_sr_flags(base + " --mode sr", first(sr_boundary),
                           first(sr_nu), first(sr_rep), first(sr_sev1), first(sr_sev2),
                           first(sr_margin1), first(sr_margin2),
                           first(sr_lr), first(sr_epochs), first(sr_batch), first(sr_weight_decay))
        run(cmd)
        # DeepSVDD
        cmd = add_dsvdd_flags(base + " --mode deepsvdd", first(dsvdd_boundary),
                              first(dsvdd_nu), first(dsvdd_rep),
                              first(dsvdd_lr), first(dsvdd_epochs),
                              first(dsvdd_batch), first(dsvdd_weight_decay))
        run(cmd)
        # Linear SVDD
        cmd = add_lsvdd_flags(base + " --mode svdd_linear", first(lsvdd_boundary),
                              first(lsvdd_nu), first(lsvdd_lr),
                              first(lsvdd_epochs), first(lsvdd_batch))
        run(cmd)
        # OC-SVM
        cmd = add_ocsvm_flags(base + " --mode ocsvm", first(ocsvm_boundary),
                              first(ocsvm_nu), first(ocsvm_gamma))
        run(cmd)
        # Kernel SVDD
        cmd = add_ksvdd_flags(base + " --mode svdd_rbf", first(ksvdd_boundary),
                              first(ksvdd_nu), first(ksvdd_gamma))
        run(cmd)

    # ----
    # GRID
    # ----
    if RUN_GRID:
        if GRID_MODE == "internal":
            # Pass comma-joined lists to the repo's internal grid mode
            base = build_base_cmd(
                to_csv(far_target), to_csv(target_type),
                to_csv(rotate), to_csv(n_norm), to_csv(n_anom1), to_csv(n_anom2),
                to_csv(b), to_csv(s1), to_csv(s2), to_csv(seed),
                to_csv(cov_a1), to_csv(cov_a2), to_csv(mu_a1), to_csv(mu_a2),
                # For sweeping splits, prefer GRID_MODE='explode' for clarity
                f"{first(train_frac)},{first(val_frac)},{first(test_frac)}" if split_frac_csv else None,
                f"{first(train_size)},{first(val_size)},{first(test_size)}" if split_size_csv else None
            )
            # If you want internal multi-seed directly, specify SEEDS_CSV above (base already adds flags).
            cmd = base + f' --mode grid --models {to_csv(GRID_MODELS)}'
            cmd = add_sr_flags(cmd, to_csv(sr_boundary),
                               to_csv(sr_nu), to_csv(sr_rep), to_csv(sr_sev1), to_csv(sr_sev2),
                               to_csv(sr_margin1), to_csv(sr_margin2),
                               to_csv(sr_lr), to_csv(sr_epochs), to_csv(sr_batch), to_csv(sr_weight_decay))
            cmd = add_dsvdd_flags(cmd, to_csv(dsvdd_boundary),
                                  to_csv(dsvdd_nu), to_csv(dsvdd_rep),
                                  to_csv(dsvdd_lr), to_csv(dsvdd_epochs), to_csv(dsvdd_batch), to_csv(dsvdd_weight_decay))
            cmd = add_lsvdd_flags(cmd, to_csv(lsvdd_boundary),
                                  to_csv(lsvdd_nu), to_csv(lsvdd_lr), to_csv(lsvdd_epochs), to_csv(lsvdd_batch))
            cmd = add_ocsvm_flags(cmd, to_csv(ocsvm_boundary),
                                  to_csv(ocsvm_nu), to_csv(ocsvm_gamma))
            cmd = add_ksvdd_flags(cmd, to_csv(ksvdd_boundary),
                                  to_csv(ksvdd_nu), to_csv(ksvdd_gamma))
            run(cmd)

        elif GRID_MODE == "explode":
            # Expand the grid externally and run individual commands per combo.
            ds_space = list(product(
                far_target, target_type, rotate, n_norm, n_anom1, n_anom2,
                b, s1, s2, seed, cov_a1, cov_a2, mu_a1, mu_a2
            ))

            pending_cmds: List[str] = []

            for (farc, tgt, rot, nn, na1, na2, bb, ss1, ss2, sd, ca1, ca2, mua1, mua2) in ds_space:
                # recompute split strings for each loop if you sweep them too
                _frac_csv = f"{first(train_frac)},{first(val_frac)},{first(test_frac)}" if split_frac_csv else None
                _ts, _vs, _ss = first(train_size), first(val_size), first(test_size)
                _size_csv = (f"{_ts},{_vs},{_ss}" if split_size_csv else None)

                base = build_base_cmd(farc, tgt, rot, nn, na1, na2, bb, ss1, ss2, sd, ca1, ca2, mua1, mua2,
                                      _frac_csv, _size_csv)

                if "sr" in GRID_MODELS:
                    for (p_bd, p_nu, p_rep, p_sev1, p_sev2, p_margin1, p_margin2, p_lr, p_ep, p_bs, p_wd) in product(
                        sr_boundary, sr_nu, sr_rep, sr_sev1, sr_sev2, sr_margin1, sr_margin2,
                        sr_lr, sr_epochs, sr_batch, sr_weight_decay
                    ):
                        pending_cmds.append(
                            add_sr_flags(base + " --mode sr", p_bd, p_nu, p_rep, p_sev1, p_sev2,
                                         p_margin1, p_margin2, p_lr, p_ep, p_bs, p_wd)
                        )

                if "deepsvdd" in GRID_MODELS:
                    for (p_bd, p_nu, p_rep, p_lr, p_ep, p_bs, p_wd) in product(
                        dsvdd_boundary, dsvdd_nu, dsvdd_rep, dsvdd_lr, dsvdd_epochs, dsvdd_batch, dsvdd_weight_decay
                    ):
                        pending_cmds.append(
                            add_dsvdd_flags(base + " --mode deepsvdd", p_bd, p_nu, p_rep, p_lr, p_ep, p_bs, p_wd)
                        )

                if "svdd_linear" in GRID_MODELS:
                    for (p_bd, p_nu, p_lr, p_ep, p_bs) in product(
                        lsvdd_boundary, lsvdd_nu, lsvdd_lr, lsvdd_epochs, lsvdd_batch
                    ):
                        pending_cmds.append(
                            add_lsvdd_flags(base + " --mode svdd_linear", p_bd, p_nu, p_lr, p_ep, p_bs)
                        )

                if "ocsvm" in GRID_MODELS:
                    for (p_bd, p_nu, p_gamma) in product(ocsvm_boundary, ocsvm_nu, ocsvm_gamma):
                        pending_cmds.append(
                            add_ocsvm_flags(base + " --mode ocsvm", p_bd, p_nu, p_gamma)
                        )

                if "svdd_rbf" in GRID_MODELS:
                    for (p_bd, p_nu, p_gamma) in product(ksvdd_boundary, ksvdd_nu, ksvdd_gamma):
                        pending_cmds.append(
                            add_ksvdd_flags(base + " --mode svdd_rbf", p_bd, p_nu, p_gamma)
                        )

            if PARALLEL and EXECUTE and pending_cmds:
                try:
                    import concurrent.futures as _fut
                    with _fut.ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
                        list(ex.map(run, pending_cmds))
                except Exception as e:
                    print("Parallel exec failed; running sequentially. Reason:", e)
                    for c in pending_cmds:
                        run(c)
            else:
                for c in pending_cmds:
                    run(c)

        else:
            print(f"[WARN] Unknown GRID_MODE='{GRID_MODE}'. Skipping grid.")


if __name__ == "__main__":
    main()
