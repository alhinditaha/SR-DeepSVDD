# ADFA-LD Experiments for SR-DeepSVDD

This folder contains the scripts used to run the **ADFA-LD** intrusion detection experiments for **SR-DeepSVDD** and the baseline methods.

The main entry point is:

* `code_runner_adfa_.py` — orchestrates **data preparation** and **model runs** on ADFA-LD.

---

## 1. Dataset

These experiments use the **ADFA-LD** host-based intrusion detection dataset (Linux system-call traces).
The dataset is **not** redistributed in this repository. You must obtain it directly from the original sources:

* UNSW Canberra, *ADFA IDS (Intrusion detection systems) datasets comprising labeled host, network and windows stealthy attacks settings*, Harvard Dataverse, 2024, V1.
  DOI: [https://doi.org/10.7910/DVN/IFTZPF](https://doi.org/10.7910/DVN/IFTZPF)

* Creech, G.; Hu, J., *A Semantic Approach to Host-Based Intrusion Detection Systems Using Contiguous and Discontiguous System Call Patterns*,
  IEEE Transactions on Computers, 2014, 63(4), 807–819.
  DOI: [https://doi.org/10.1109/TC.2013.13](https://doi.org/10.1109/TC.2013.13)

After obtaining the **host-based ADFA-LD** data:

1. Create the following directory relative to the repository root (if it does not already exist):

   ```text
   data/ADFA-LD/
   ```

2. Place the raw ADFA-LD host-based files (normal and attack sessions) inside `data/ADFA-LD/`, preserving the original folder structure as provided by the dataset authors.

The scripts in this folder assume this layout and will read the data from `data/ADFA-LD/`.

---

## 2. Environment Setup

From the **repository root**, set up the Python environment:

1. Create and activate a Python environment (Python 3.10 recommended).
2. Install the project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

This installs PyTorch, NumPy, scikit-learn, and other required libraries used by SR-DeepSVDD and the baselines.

---

## 3. Running the ADFA-LD Experiments

The script `code_runner_adfa_.py` automates the full ADFA-LD pipeline:

* **Feature preparation**
  (tokenization, TF–IDF, and sequence-level statistics; saving features to disk)
* **Model evaluation**
  (running SR-DeepSVDD and baselines on the prepared features)
* **Logging**
  (printing metrics and recording executed commands for reproducibility)

You can launch the pipeline in either of the following ways:

### Option A — From the repository root

```bash
python adfa/code_runner_adfa_.py
```

### Option B — From inside the `adfa/` folder

```bash
cd adfa
python code_runner_adfa_.py
```

By default, the runner:

1. **Prepares ADFA-LD features** (if the corresponding flag in the script is enabled):

   * Reads raw sessions from `data/ADFA-LD/`
   * Builds TF–IDF features plus sequence-level statistics
   * Splits sessions into training, validation, and test sets
   * Saves processed feature matrices under:

     ```text
     data/adfa_features/
     ```

2. **Runs the experiments**:

   * Loads features from `data/adfa_features/`
   * Trains and evaluates SR-DeepSVDD and the configured baselines
   * Applies the validation-based operating point calibration (including the FAR constraint)
   * Prints overall and per-type performance metrics to the console

All commands executed by the runner are also appended to:

```text
adfa/commands_run_adfa_log.txt
```

You can inspect this log to see the exact commands used or to re-run any configuration manually.

---

## 4. Modifying or Re-running Specific Models

The behavior of `code_runner_adfa_.py` is controlled by configuration flags defined near the top of the file, for example:

* Whether to (re)run feature preparation
* Whether to run all models in one pass or invoke them individually
* Which models are enabled
* Device and plotting options

To adjust the behavior:

1. Open `adfa/code_runner_adfa_.py` in a text editor.
2. Modify the relevant configuration variables (e.g., enabling/disabling preparation, selecting models).
3. Run the script again:

   ```bash
   python adfa/code_runner_adfa_.py
   ```

For more advanced usage, you may call the underlying ADFA driver script directly with your own command-line options; the commands recorded in `commands_run_adfa_log.txt` serve as concrete examples of how `code_runner_adfa_.py` invokes it.

---

## 5. Reproducibility

The ADFA-LD experiments used in the paper correspond to a specific commit of this repository (see the main `README.md` and the paper Appendix for the exact commit hash).

The runner sets a fixed random seed via a `--seed` argument when invoking the ADFA driver, so:

* All methods share the **same train/validation/test partitions**, and
* They see **identical mini-batch orders** for a given experiment,

which ensures that performance differences arise from the learning objectives and hyperparameters, not from different data splits.
