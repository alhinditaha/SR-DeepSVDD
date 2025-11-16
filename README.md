# SR-DeepSVDD – Severity‑Regularized Deep Support Vector Data Description

A research-grade implementation of **Severity-Regularized Deep SVDD (SR-DeepSVDD)** with runners for
simulated data experiments and a host-based intrusion detection **case study on ADFA-LD**.  
The repo also includes classic baselines (OC-SVM, kernel SVDD, DeepSVDD) and a consistent evaluation
pipeline (per-type DR/PREC/F1, FAR caps, and validation policies).

> **Why SR‑DeepSVDD?**> Not all anomalies are equal. SR‑DeepSVDD lets you *weight anomaly types by severity*, improving recall on
> high‑impact events while keeping overall false alarms under control.

## ✨ Features
- Severity-aware DeepSVDD loss (targeted vs. non-targeted anomalies, margins, and weights).
- Reproducible runners for **simulated datasets** (e.g., banana-shaped data).
- **ADFA-LD intrusion detection case study** with a dedicated runner under `adfa/`.
- Baselines: OC-SVM, (RBF) SVDD, Linear/Deep SVDD with unified reporting.
- Validation policies with optional **FAR ≤ τ** constraints.
- Clean logs & CSV exports (seed included), plus plots.
- Packaging, linting, CI, and pre-commit hooks ready for GitHub.

## 🧩 Repository structure (key folders)
```
.
├─ src/                 # Library code (models, trainers, data utils, SIM runners)
├─ adfa/                # ADFA-LD case study (runner + README_ADFA-LD.md)
├─ tests/               # Minimal smoke tests
├─ notebooks/           # (optional) exploration notebooks
├─ .github/workflows/   # CI pipeline
└─ README.md
```
- Simulated experiments (banana, etc.) are driven from src/code_runner.py.
- The ADFA-LD intrusion detection experiments live under adfa/ with their own README.

## 🔧 Installation
**Python 3.10+** recommended. Create a fresh venv, then:

```bash
pip install -U pip
pip install -e .
# or: pip install -r requirements.txt
```

If you prefer PEP 517/pyproject builds:
```bash
pip install build
python -m build
pip install dist/*.whl
```

## 🚀 Quickstart

### 1. Simulated data experiments (banana, etc.)

The main scripted entry point for simulated experiments is `src/code_runner.py`.

From the repository root, you can either use the provided Make target:

```bash
make run-sim
```

or call the runner directly:

```bash
python src/code_runner.py
```

By default, `code_runner.py` runs a set of SR-DeepSVDD and baseline experiments on the simulated datasets
(e.g., banana), using the common evaluation pipeline.
You can open `src/code_runner.py` to:

* Select which datasets to run,
* Enable/disable specific methods (SR-DeepSVDD, DeepSVDD, OC-SVM, SVDD-RBF),
* Adjust seeds, plotting, or logging options.

If you prefer the module-style entry point, you can still run the original simple runner:

```bash
python -m src.run_sr_deepsvdd --dataset banana --seed 42
```

### 2. ADFA-LD intrusion detection case study

The ADFA-LD case study (Linux host-based intrusion detection) is organized under the `adfa/` folder.

* **Dataset:** The ADFA-LD data are *not* included in this repository.
  You must obtain the host-based dataset directly from the original sources (Harvard Dataverse / Creech & Hu).
* **Runner:** The main entry point is `adfa/code_runner_adfa_.py`.
* **Instructions:** See the detailed guide in:

```text
adfa/README_ADFA-LD.md
```

for:

* How to place the ADFA-LD raw data under `data/ADFA-LD/`,
* How to prepare TF–IDF + tabular features,
* How to run SR-DeepSVDD and baselines on ADFA-LD,
* Example commands and reproducibility notes.

A typical command from the repository root is:

```bash
python adfa/code_runner_adfa_.py
```

---


## 📜 License
MIT – see [LICENSE](LICENSE).

## 🙌 Acknowledgments
This repository draws inspiration from Deep SVDD literature and implements
a severity‑aware variant for practical use‑cases.
