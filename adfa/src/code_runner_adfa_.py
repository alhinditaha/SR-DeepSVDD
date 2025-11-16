#!/usr/bin/env python3
"""
SR-DeepSVDD — ADFA Code Runner
"""
import os, shlex, subprocess, json
from pathlib import Path
from itertools import product
from datetime import datetime
from typing import Iterable, List, Union

# =========================
# GLOBAL EXECUTION CONTROLS
# =========================
PY: str = "python"        # interpreter
EXECUTE: bool = True      # True: run; False: print only
PARALLEL: bool = False    # only for GRID_MODE='explode'
N_WORKERS: int = 0
SAVE_CMDS_TO: Union[str, None] = "commands_run_adfa_log.txt"
GRID_MODE: str = "internal"   # 'internal' (comma-lists) or 'explode' (cartesian here)

# =================
# FILES & PATHS
# =================
SRC = Path(__file__).resolve().parent
MAIN = SRC / "run_sr_deepsvdd_ADFA_v6.py"

# =================
# ADFA PREP SETTINGS
# =================
# =================
# ADFA PREP SETTINGS
# =================
adfa_root    = ["src/data/ADFA-LD"]
data_dir     = ["src/data/adfa_features"]
ngram        = ["1,4"]      # "min,max"
max_features = [20000]
val_frac     = [0.20]
seed         = [1]
tfidf_mode   = ["normals"]      # NEW: 'all' or 'normals' for TF-IDF fitting


# =========================
# Targeting controls (load-time)
# =========================
# e.g., ["Hydra_SSH","Hydra_FTP"] or ["Adduser"] — substring, case-insensitive
TARGET_FAMILIES = ["Adduser"]
TARGET_FRAC = None          # e.g., 0.5 to randomly mark 50% of anomalies as targeted
TARGET_OVERRIDE = True      # IMPORTANT: force recompute g from families/frac even if metas have targeted_flag

RUN_PREPARE = True

# ======================================
# VALIDATION / CALIBRATION (CROSS-CUTTING)
# ======================================
far_target  = [0.05]      # FAR cap (None to disable)
target_type = [1]         # which anomaly group SR targets in validation (parity w/ earlier runner)

# =========================================
# PER-MODEL BOUNDARY SELECTION
# =========================================
sr_boundary     = ["val"]
dsvdd_boundary  = ["val"]
lsvdd_boundary  = ["zero"]
ocsvm_boundary  = ["zero"]
ksvdd_boundary  = ["zero"]

# =================
# SR-DeepSVDD PARAMS
# =================
sr_nu            = [0.05]
sr_rep           = [64]
sr_sev1          = [1] #1.5
sr_sev2          = [1] #0.1
sr_margin1       = [0.5] #0.5 worked well
sr_margin2       = [0.5] #0.5 worked well
sr_lr            = [1e-3]
sr_epochs        = [50]
sr_batch         = [128]
sr_weight_decay  = [1e-6]
sr_network       = ["mlp_nobias"]
sr_hidden        = ["512,256"]

# =================
# DeepSVDD PARAMS
# =================
dsvdd_nu            = [0.05]
dsvdd_rep           = [64]
dsvdd_lr            = [1e-4]
dsvdd_epochs        = [10]
dsvdd_batch         = [128]
dsvdd_weight_decay  = [1e-6]
dsvdd_network       = ["mlp_nobias"]
dsvdd_hidden        = ["512,256"]

# =================
# Linear SVDD PARAMS
# =================
lsvdd_nu     = [0.05]
lsvdd_lr     = [1e-2]
lsvdd_epochs = [300]
lsvdd_batch  = [256]

# =================
# OC-SVM PARAMS
# =================
ocsvm_nu    = [0.05]
ocsvm_gamma = ["scale"]  # or numeric

# =================
# Kernel SVDD (RBF) PARAMS
# =================
ksvdd_nu    = [0.05]
ksvdd_gamma = [1]

# =================
# DEVICE & PLOTTING (passed through)
# =================
DEVICE       = "cuda"    # or "cpu"
PLOTS_DIR    = "plots"
GRID_RES     = 300
PLOT_TRAIN   = 0
PLOT_SCORES  = 0

# ==========================
# MULTI-SEED & STATISTICS
# ==========================
SEEDS_CSV        = None            # e.g., "123,124,125"
N_SEEDS: Union[int, None] = None   # e.g., 10

SELECT_METRIC       = "DR1"
SELECT_DIRECTION    = "max"
PVAL_METRIC         = "DR1"
PVAL_DIRECTION      = "max"
PVAL_REF            = "SR-DeepSVDD"
PVAL_TEST           = "wilcoxon"

# =================
# WHICH RUNS TO DO
# =================
RUN_ONE_SHOT_ALL  = True
RUN_INDIVIDUAL    = False
RUN_GRID          = False
GRID_MODELS = ["sr", "deepsvdd", "svdd_linear", "ocsvm", "svdd_rbf"]

# ================
# Helper Functions
# ================
def to_csv(xs: Iterable) -> str:
    return ",".join(str(x) for x in xs)

def first(xs: Iterable):
    return list(xs)[0]

def run(cmd: str):
    print(cmd)
    if SAVE_CMDS_TO:
        with open(SAVE_CMDS_TO, "a", encoding="utf-8") as f:
            f.write(cmd + "\n")
    if EXECUTE:
        try:
            res = subprocess.run(shlex.split(cmd), check=True, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.stdout: print(res.stdout)
            if res.stderr: print(res.stderr)
        except subprocess.CalledProcessError as e:
            print("RUN FAILED:")
            if e.stdout: print(e.stdout)
            if e.stderr: print(e.stderr)

def _as_json_list(xs):
    return json.dumps(list(xs))  # e.g. ["Adduser","Hydra"]

# ==================
# Command builders
# ==================
def build_prep_cmd():
    cmd = (f'{PY} "{MAIN}" --prepare'
           f' --adfa-root {first(adfa_root)}'
           f' --data-dir {first(data_dir)}'
           f' --ngram {first(ngram)}'
           f' --max-features {first(max_features)}'
           f' --val-frac {first(val_frac)}'
           f' --seed {first(seed)}'
           f' --tfidf-mode {first(tfidf_mode)}')
    # pass CSV to avoid quoting issues
    if TARGET_FAMILIES:
        cmd += f' --target-families {",".join(TARGET_FAMILIES)}'
    return cmd


def build_base_cmd(far_target_val, target_type_val):
    cmd = (f'{PY} "{MAIN}"'
           f' --data-dir {first(data_dir)}'
           f' --device {DEVICE}'
           f' --plots-dir {PLOTS_DIR} --grid-res {GRID_RES}'
           f' --plot-train {PLOT_TRAIN} --plot-scores {PLOT_SCORES}'
           f' --seed {first(seed)}')
    if far_target_val is not None and str(far_target_val).lower() != "none":
        cmd += f' --far-target {far_target_val} --target-type {target_type_val}'

    # pass CSV here as well
    if TARGET_FAMILIES:
        cmd += f' --target-families {",".join(TARGET_FAMILIES)}'

    # only add --target-override if you *really* want to ignore saved flags
    if TARGET_OVERRIDE:
        cmd += ' --target-override'

    if TARGET_FRAC is not None:
        cmd += f' --target-frac {TARGET_FRAC}'

    # ... rest unchanged
    return cmd

def add_sr_flags(cmd, boundary, nu, rep, sev1, sev2, margin1, margin2, lr, epochs, batch, wd, net, hidden):
    return (cmd + f' --sr-boundary {boundary} --sr-nu {nu} --sr-rep {rep}'
            + f' --sr-sev1 {sev1} --sr-sev2 {sev2} --sr-margin1 {margin1} --sr-margin2 {margin2}'
            + f' --sr-lr {lr} --sr-epochs {epochs} --sr-batch {batch} --sr-weight-decay {wd}'
            + f' --sr-network {net} --sr-hidden {hidden}')

def add_dsvdd_flags(cmd, boundary, nu, rep, lr, epochs, batch, wd, net, hidden):
    return (cmd + f' --dsvdd-boundary {boundary} --dsvdd-nu {nu} --dsvdd-rep {rep}'
            + f' --dsvdd-lr {lr} --dsvdd-epochs {epochs} --dsvdd-batch {batch} --dsvdd-weight-decay {wd}'
            + f' --dsvdd-network {net} --dsvdd-hidden {hidden}')

def add_lsvdd_flags(cmd, boundary, nu, lr, epochs, batch):
    return (cmd + f' --lsvdd-boundary {boundary} --lsvdd-nu {nu} --lsvdd-lr {lr}'
            + f' --lsvdd-epochs {epochs} --lsvdd-batch {batch}')

def add_ocsvm_flags(cmd, boundary, nu, gamma):
    return (cmd + f' --ocsvm-boundary {boundary} --ocsvm-nu {nu} --ocsvm-gamma {gamma}')

def add_ksvdd_flags(cmd, boundary, nu, gamma):
    return (cmd + f' --ksvdd-boundary {boundary} --ksvdd-nu {nu} --ksvdd-gamma {gamma}')

# ==================
# Orchestration
# ==================
def main():
    # log header
    if SAVE_CMDS_TO:
        with open(SAVE_CMDS_TO, "a", encoding="utf-8") as f:
            f.write(f"\n# ===== {datetime.now().isoformat()} =====\n")

    # 1) Prepare
    if RUN_PREPARE:
        run(build_prep_cmd())

    # 2) One-shot "all"
    if RUN_ONE_SHOT_ALL:
        base = build_base_cmd(first(far_target), first(target_type))
        cmd = base + " --mode all --models " + ",".join(GRID_MODELS)
        cmd = add_sr_flags(cmd, first(sr_boundary), first(sr_nu), first(sr_rep),
                           first(sr_sev1), first(sr_sev2), first(sr_margin1), first(sr_margin2),
                           first(sr_lr), first(sr_epochs), first(sr_batch), first(sr_weight_decay),
                           first(sr_network), first(sr_hidden))
        cmd = add_dsvdd_flags(cmd, first(dsvdd_boundary), first(dsvdd_nu), first(dsvdd_rep),
                              first(dsvdd_lr), first(dsvdd_epochs), first(dsvdd_batch), first(dsvdd_weight_decay),
                              first(dsvdd_network), first(dsvdd_hidden))
        cmd = add_lsvdd_flags(cmd, first(lsvdd_boundary), first(lsvdd_nu), first(lsvdd_lr),
                              first(lsvdd_epochs), first(lsvdd_batch))
        cmd = add_ocsvm_flags(cmd, first(ocsvm_boundary), first(ocsvm_nu), first(ocsvm_gamma))
        cmd = add_ksvdd_flags(cmd, first(ksvdd_boundary), first(ksvdd_nu), first(ksvdd_gamma))
        run(cmd)

    # 3) Individuals
    if RUN_INDIVIDUAL:
        base = build_base_cmd(first(far_target), first(target_type))
        run(add_sr_flags(base + " --mode sr", first(sr_boundary), first(sr_nu), first(sr_rep),
                         first(sr_sev1), first(sr_sev2), first(sr_margin1), first(sr_margin2),
                         first(sr_lr), first(sr_epochs), first(sr_batch), first(sr_weight_decay),
                         first(sr_network), first(sr_hidden)))
        run(add_dsvdd_flags(base + " --mode deepsvdd", first(dsvdd_boundary), first(dsvdd_nu), first(dsvdd_rep),
                            first(dsvdd_lr), first(dsvdd_epochs), first(dsvdd_batch), first(dsvdd_weight_decay),
                            first(dsvdd_network), first(dsvdd_hidden)))
        run(add_lsvdd_flags(base + " --mode svdd_linear", first(lsvdd_boundary), first(lsvdd_nu),
                            first(lsvdd_lr), first(lsvdd_epochs), first(lsvdd_batch)))
        run(add_ocsvm_flags(base + " --mode ocsvm", first(ocsvm_boundary), first(ocsvm_nu), first(ocsvm_gamma)))
        run(add_ksvdd_flags(base + " --mode svdd_rbf", first(ksvdd_boundary), first(ksvdd_nu), first(ksvdd_gamma)))

    # 4) Grid
    if RUN_GRID:
        if GRID_MODE == "internal":
            base = build_base_cmd(to_csv(far_target), to_csv(target_type))
            cmd = base + f' --mode grid --models {",".join(GRID_MODELS)}'
            cmd = add_sr_flags(cmd, to_csv(sr_boundary), to_csv(sr_nu), to_csv(sr_rep),
                               to_csv(sr_sev1), to_csv(sr_sev2), to_csv(sr_margin1), to_csv(sr_margin2),
                               to_csv(sr_lr), to_csv(sr_epochs), to_csv(sr_batch), to_csv(sr_weight_decay),
                               to_csv(sr_network), to_csv(sr_hidden))
            cmd = add_dsvdd_flags(cmd, to_csv(dsvdd_boundary), to_csv(dsvdd_nu), to_csv(dsvdd_rep),
                                  to_csv(dsvdd_lr), to_csv(dsvdd_epochs), to_csv(dsvdd_batch), to_csv(dsvdd_weight_decay),
                                  to_csv(dsvdd_network), to_csv(dsvdd_hidden))
            cmd = add_lsvdd_flags(cmd, to_csv(lsvdd_boundary), to_csv(lsvdd_nu), to_csv(lsvdd_lr),
                                  to_csv(lsvdd_epochs), to_csv(lsvdd_batch))
            cmd = add_ocsvm_flags(cmd, to_csv(ocsvm_boundary), to_csv(ocsvm_nu), to_csv(ocsvm_gamma))
            cmd = add_ksvdd_flags(cmd, to_csv(ksvdd_boundary), to_csv(ksvdd_nu), to_csv(ksvdd_gamma))
            run(cmd)

        elif GRID_MODE == "explode":
            pending_cmds: List[str] = []
            base_fixed = build_base_cmd(first(far_target), first(target_type))
            for (p_sr_bd, p_sr_nu, p_sr_rep, p_sr_s1, p_sr_s2, p_sr_m1, p_sr_m2, p_sr_lr, p_sr_ep, p_sr_bs, p_sr_wd, p_sr_net, p_sr_hid) in product(
                sr_boundary, sr_nu, sr_rep, sr_sev1, sr_sev2, sr_margin1, sr_margin2,
                sr_lr, sr_epochs, sr_batch, sr_weight_decay, sr_network, sr_hidden
            ):
                pending_cmds.append(
                    add_sr_flags(base_fixed + " --mode sr", p_sr_bd, p_sr_nu, p_sr_rep,
                                 p_sr_s1, p_sr_s2, p_sr_m1, p_sr_m2, p_sr_lr, p_sr_ep, p_sr_bs, p_sr_wd,
                                 p_sr_net, p_sr_hid)
                )
            for (p_bd, p_nu, p_rep, p_lr, p_ep, p_bs, p_wd, p_net, p_hid) in product(
                dsvdd_boundary, dsvdd_nu, dsvdd_rep, dsvdd_lr, dsvdd_epochs, dsvdd_batch, dsvdd_weight_decay,
                dsvdd_network, dsvdd_hidden
            ):
                pending_cmds.append(
                    add_dsvdd_flags(base_fixed + " --mode deepsvdd", p_bd, p_nu, p_rep, p_lr, p_ep, p_bs, p_wd, p_net, p_hid)
                )
            for (p_bd, p_nu, p_lr, p_ep, p_bs) in product(lsvdd_boundary, lsvdd_nu, lsvdd_lr, lsvdd_epochs, lsvdd_batch):
                pending_cmds.append(add_lsvdd_flags(base_fixed + " --mode svdd_linear", p_bd, p_nu, p_lr, p_ep, p_bs))
            for (p_bd, p_nu, p_gamma) in product(ocsvm_boundary, ocsvm_nu, ocsvm_gamma):
                pending_cmds.append(add_ocsvm_flags(base_fixed + " --mode ocsvm", p_bd, p_nu, p_gamma))
            for (p_bd, p_nu, p_gamma) in product(ksvdd_boundary, ksvdd_nu, ksvdd_gamma):
                pending_cmds.append(add_ksvdd_flags(base_fixed + " --mode svdd_rbf", p_bd, p_nu, p_gamma))

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

