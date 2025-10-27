# SR-DeepSVDD – Severity‑Regularized Deep Support Vector Data Description

A research‑grade implementation of **Severity‑Regularized Deep SVDD (SR‑DeepSVDD)** with runners for
simulated data and **ADFA‑LD** intrusion detection experiments. The repo also includes classic baselines
(OC‑SVM, kernel SVDD, DeepSVDD) and a consistent evaluation pipeline (per‑type DR/PREC/F1, FAR caps, and
validation policies).

> **Why SR‑DeepSVDD?**> Not all anomalies are equal. SR‑DeepSVDD lets you *weight anomaly types by severity*, improving recall on
> high‑impact events while keeping overall false alarms under control.

## ✨ Features
- Severity‑aware DeepSVDD loss (targeted vs. non‑targeted anomalies, margins, and weights).
- Reproducible runners for **simulated banana** data and **ADFA‑LD**.
- Baselines: OC‑SVM, (RBF) SVDD, Linear/Deep SVDD with unified reporting.
- Validation policies with optional **FAR ≤ τ** constraints.
- Clean logs & CSV exports (seed included), plus plots.
- Packaging, linting, CI, and pre‑commit hooks ready for GitHub.

## 🧩 Repository structure (key folders)
```
.
├─ src/                 # Library code (models, trainers, data utils, runners)
├─ scripts/             # Convenience scripts for common experiments
├─ tests/               # Minimal smoke tests
├─ notebooks/           # (optional) exploration notebooks
├─ .github/workflows/   # CI pipeline
└─ README.md
```

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
Simulated banana data:
```bash
make run-sim
# or
python -m src.run_sr_deepsvdd --dataset banana --seed 42
```

ADFA‑LD (assumes features prepared or auto-prepared by the runner):
```bash
make run-adfa
# or roughly equivalent:
python -m src.run_sr_deepsvdd_ADFA_v4 --prepare   --adfa-root src/data/ADFA-LD --data-dir stc/data/adfa_features   --ngram 1,3 --max-features 5000 --val-frac 0.2 --seed 42
```

> The ADFA runner supports targeting controls: `--target-families`, `--target-frac`, `--target-override`, and FAR caps via `--far-target`.

## 🧪 Tests & Lint
```bash
make lint        # ruff + black
make test        # pytest smoke tests
make typecheck   # mypy (if configured)
```

## 📊 Results logging
All runs export metrics with **seed** to `results/*.csv` so you can aggregate and run
statistical tests (Wilcoxon/t‑test) later.

## 📚 Citation
If you use this repository, please cite:
- Ruff et al., **Deep One‑Class Classification (Deep SVDD)**, ICML 2018.
- Kang, Jeong, Kong, **A differentiated one-class classification method**, ESWA 2012.

You can also use the provided `CITATION.cff` in this repo.

## 📜 License
MIT – see [LICENSE](LICENSE).

## 🙌 Acknowledgments
This repository draws inspiration from Deep SVDD and differentiated SVDD literature and implements
a severity‑aware variant for practical cybersecurity use‑cases.
