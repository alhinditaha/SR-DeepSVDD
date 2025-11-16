import os
import math
import json
import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from src.base.base_dataset import DictDataset, BaseADDataset  # kept for parity; not used here

# ---------------------------
# I/O + basic utils
# ---------------------------
def read_trace_file(path):
    """Read a syscall trace file into a list of tokens (one per line or whitespace-separated)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read().strip()
    return txt.split() if txt else []

def shannon_entropy_from_counts(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    p = np.array([c / total for c in counts if c > 0], dtype=np.float64)
    return float(-np.sum(p * np.log(p + 1e-12)))

def gini_coefficient(freqs):
    arr = np.array(sorted(freqs))
    if arr.size == 0:
        return 0.0
    n = arr.size
    cum = np.cumsum(arr, dtype=float)
    if cum[-1] == 0:
        return 0.0
    return float((2.0 * np.sum((np.arange(1, n + 1) * arr))) / (n * cum[-1]) - (n + 1.0) / n)

def burst_cv(tokens, win=50):
    T = len(tokens)
    if T == 0:
        return 0.0
    counts = np.array([len(tokens[i:i + win]) for i in range(0, T, win)], dtype=float)
    mu = counts.mean()
    if mu == 0:
        return 0.0
    return float(counts.std() / (mu + 1e-12))

# ---------------------------
# Targeting helpers
# ---------------------------
def _parse_family_spec(spec):
    """
    Accept CSV string (e.g., "Hydra_FTP,Adduser") OR a JSON/Python list string (e.g., '["Hydra","Adduser"]'),
    OR a list/tuple that’s already parsed. Returns a list[str] of lowercase substrings.
    """
    if not spec:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(x).strip().lower() for x in spec if str(x).strip()]
    s = str(spec).strip()
    if s.startswith("["):
        # JSON / Python-list literal
        try:
            obj = json.loads(s)
        except Exception:
            import ast
            obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            return [str(x).strip().lower() for x in obj if str(x).strip()]
    # Fallback: CSV
    return [t.strip().lower() for t in s.split(",") if t.strip()]

def _is_target_family(fam, subs):
    """Substring, case-insensitive match."""
    f = (fam or "").lower()
    return any(sub in f for sub in subs)

# ---------------------------
# Tabular feature block
# ---------------------------
def tabular_features(tokens, win=50):
    T = len(tokens)
    if T == 0:
        return {
            "len_tokens": 0, "log_len_tokens": 0, "uniq_syscalls": 0,
            "unigram_entropy": 0.0, "bigram_entropy": 0.0,
            "top1_prop": 0.0, "top3_prop": 0.0,
            "gini_unigram": 0.0, "burst_cv_win50": 0.0,
            "ngram_diversity_ratio": 0.0
        }

    c_uni = Counter(tokens)
    uniq = len(c_uni)
    H1 = shannon_entropy_from_counts(list(c_uni.values()))

    # bigrams
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    c_bi = Counter(bigrams)
    H2 = shannon_entropy_from_counts(list(c_bi.values())) if c_bi else 0.0

    top1 = max(c_uni.values()) / T
    top3 = sum(sorted(c_uni.values(), reverse=True)[:3]) / T
    gini = gini_coefficient(list(c_uni.values()))
    burst = burst_cv(tokens, win=win)
    div_ratio = (len(c_bi) / (T - 1)) if T > 1 else 0.0

    return {
        "len_tokens": T,
        "log_len_tokens": float(math.log1p(T)),
        "uniq_syscalls": uniq,
        "unigram_entropy": float(H1),
        "bigram_entropy": float(H2),
        "top1_prop": float(top1),
        "top3_prop": float(top3),
        "gini_unigram": float(gini),
        "burst_cv_win50": float(burst),
        "ngram_diversity_ratio": float(div_ratio),
    }

# ---------------------------
# Feature builder
# ---------------------------
class ADFAFeatureBuilder:
    def __init__(self, ngram_range=(1, 3), max_features=5000):
        # fixed tabular headers (do not change)
        self.tab_cols = [
            "len_tokens", "log_len_tokens", "uniq_syscalls",
            "unigram_entropy", "bigram_entropy",
            "top1_prop", "top3_prop",
            "gini_unigram", "burst_cv_win50",
            "ngram_diversity_ratio",
        ]
        self.ngram_range = ngram_range
        self.max_features = max_features

        # Accept numeric/single-char tokens; do not force lowercase; sublinear_tf for robustness
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
            min_df=1,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
        )

    def fit_vectorizer(self, token_sequences):
        docs = [" ".join(toks) for toks in token_sequences]
        self.vectorizer.fit(docs)

    def transform_tfidf(self, tokens):
        doc = " ".join(tokens)
        return self.vectorizer.transform([doc])  # sparse row

    def fit(self, sessions):
        token_seqs = [s["tokens"] for s in sessions]
        self.fit_vectorizer(token_seqs)

    def build_feature_matrix(self, sessions, scaler=None):
        """Return (X, metas) where X = [tabular || tfidf]."""
        from scipy.sparse import vstack

        tfidf_rows, tab_rows, metas = [], [], []
        for s in sessions:
            tokens = s["tokens"]
            tab = tabular_features(tokens)
            tab_rows.append([tab[c] for c in self.tab_cols])
            tfidf_rows.append(self.transform_tfidf(tokens))
            metas.append({
                "session_id": s.get("session_id"),
                "label": int(s.get("label", 0)),
                "anomaly_label": s.get("anomaly_label"),
                "targeted_flag": int(s.get("targeted_flag", 0)),
            })

        X_tfidf = vstack(tfidf_rows)  # (n, d_tfidf) sparse
        X_tab = np.array(tab_rows, dtype=np.float32)

        # scale tab block if scaler provided
        if scaler is not None:
            means = np.array([scaler["means"][c] for c in self.tab_cols], dtype=np.float32)
            stds = np.array([scaler["stds"][c] for c in self.tab_cols], dtype=np.float32)
            stds[stds == 0] = 1.0
            X_tab = (X_tab - means) / stds

        # convert tfidf to dense for MLP convenience
        X_tfidf_dense = X_tfidf.toarray().astype(np.float32)
        X = np.hstack([X_tab, X_tfidf_dense])
        return X, metas

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "ngram_range": self.ngram_range,
                    "max_features": self.max_features,
                    "tab_cols": self.tab_cols,
                    "vectorizer": self.vectorizer,  # save the fitted vectorizer object
                },
                f,
            )

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.ngram_range = tuple(data["ngram_range"])
        self.max_features = data["max_features"]
        self.tab_cols = data["tab_cols"]
        self.vectorizer = data["vectorizer"]

# ---------------------------
# ADFA file loader
# ---------------------------
def load_adfa_from_dir(root_dir):
    """
    Canonical ADFA-LD layout:
      - Attack_Data_Master/<FAMILY>/*.txt  -> anomalies (label=1, anomaly_label=<FAMILY>)
      - Training_Data_Master/*.txt         -> normals (label=0)
      - Validation_Set/*.txt               -> normals (label=0)
    """
    sessions = []

    # anomalies
    attack_root = os.path.join(root_dir, "Attack_Data_Master")
    if os.path.isdir(attack_root):
        for fam in sorted(os.listdir(attack_root)):
            fam_dir = os.path.join(attack_root, fam)
            if not os.path.isdir(fam_dir):
                continue
            for fn in sorted(os.listdir(fam_dir)):
                if fn.lower().endswith(".txt"):
                    toks = read_trace_file(os.path.join(fam_dir, fn))
                    sessions.append({
                        "session_id": f"Attack_Data_Master/{fam}/{fn}",
                        "tokens": toks,
                        "label": 1,
                        "anomaly_label": fam,
                    })

    # normals from training + validation
    for bucket in ["Training_Data_Master", "Validation_Set"]:
        nb_dir = os.path.join(root_dir, bucket)
        if not os.path.isdir(nb_dir):
            continue
        for fn in sorted(os.listdir(nb_dir)):
            if fn.lower().endswith(".txt"):
                toks = read_trace_file(os.path.join(nb_dir, fn))
                sessions.append({
                    "session_id": f"{bucket}/{fn}",
                    "tokens": toks,
                    "label": 0,
                    "anomaly_label": None,
                })

    return sessions

# ---------------------------
# Build + save feature splits
# ---------------------------
def build_and_save_features(
    root_dir,
    out_dir,
    ngram_range=(1, 3),
    max_features=5000,
    val_frac=0.2,
    seed=42,
    anom_train_frac=0.2,
    target_families=None,   # NEW: substrings for targeting (CSV/list)
    target_frac=None,       # NEW: if set, randomly mark this fraction of anomalies as targeted
    tfidf_mode="all"        # NEW: 'all' (normals+anoms) or 'normals' (normals only) for TF-IDF fitting
):

    os.makedirs(out_dir, exist_ok=True)
    sessions = load_adfa_from_dir(root_dir)
    normals = [s for s in sessions if s["label"] == 0]
    anoms = [s for s in sessions if s["label"] == 1]

    print(f"[ADFA] Loaded sessions: normals={len(normals)} anomalies={len(anoms)}")

    rng = np.random.default_rng(seed)
    rng.shuffle(normals)

    # normals: ~60/20/20 if val_frac=0.2
    n = len(normals)
    n_train = int((1.0 - 2 * val_frac) * n)
    n_val = int(val_frac * n)
    train_norm = normals[:n_train]
    val_norm = normals[n_train:n_train + n_val]
    test_norm = normals[n_train + n_val:]

    print(f"[ADFA] Normals split -> train={len(train_norm)} val={len(val_norm)} test={len(test_norm)}")

    if len(train_norm) == 0:
        # Fallback to ensure we can fit TF-IDF
        take = min(max(1, int(0.6 * n)), n)
        train_norm = normals[:take]
        remain = normals[take:]
        half = len(remain) // 2
        val_norm = remain[:half]
        test_norm = remain[half:]
        print("[ADFA][WARN] No train normals with requested val/test frac; applied fallback split.")
        print(f"[ADFA] New normals split -> train={len(train_norm)} val={len(val_norm)} test={len(test_norm)}")

    # anomalies per family: train/val/test
    groups = {}
    for a in anoms:
        groups.setdefault(a["anomaly_label"], []).append(a)

    train_anoms, val_anoms, test_anoms = [], [], []
    for fam, lst in groups.items():
        rng.shuffle(lst)
        nt = int(max(0, round(anom_train_frac * len(lst))))
        nv = max(1, int(0.2 * len(lst))) if len(lst) >= 5 else max(0, int(0.2 * len(lst)))
        nt = min(nt, max(0, len(lst) - nv - 1)) if len(lst) > nv else 0
        train_anoms += lst[:nt]
        val_anoms += lst[nt:nt + nv]
        test_anoms += lst[nt + nv:]

    print(f"[ADFA] Anomalies split -> train={len(train_anoms)} val={len(val_anoms)} test={len(test_anoms)}")

    # === targeted assignment at PREPARE TIME ===
    subs = _parse_family_spec(target_families)

    def set_target_flags(lst):
        for s in lst:
            fam = s.get("anomaly_label")
            if subs:
                s["targeted_flag"] = 1 if _is_target_family(fam, subs) else 0
            elif target_frac is not None:
                s["targeted_flag"] = int(rng.random() < float(target_frac))
            else:
                s["targeted_flag"] = 0

    set_target_flags(train_anoms)
    set_target_flags(val_anoms)
    set_target_flags(test_anoms)

    # quick summary
    def _summ(norm, anom, tag):
        targ = sum(int(x.get("targeted_flag", 0)) for x in anom)
        print(f"[ADFA] {tag}: normals={len(norm)} anomalies={len(anom)} (targeted={targ}, non-targeted={len(anom)-targ})")

    _summ(train_norm, train_anoms, "Train")
    _summ(val_norm,   val_anoms,   "Val")
    _summ(test_norm,  test_anoms,  "Test")
    if (sum(int(x.get("targeted_flag", 0)) for x in val_anoms) == 0 or
        sum(int(x.get("targeted_flag", 0)) for x in test_anoms) == 0):
        print("[WARN] No targeted anomalies found in val/test with the current target spec.")

    # ==== features ====
    # ==== features ====
    fb = ADFAFeatureBuilder(ngram_range=ngram_range, max_features=max_features)
    # Control how TF-IDF is fitted:
    #   - "all":     use both train normals and train anomalies
    #   - "normals": use train normals only
    tfidf_mode = (tfidf_mode or "all").lower()
    if tfidf_mode == "normals":
        print("[ADFA] TF-IDF fitting mode: normals-only")
        fb.fit(train_norm)
    else:
        if tfidf_mode != "all":
            print(f"[ADFA] [WARN] Unknown tfidf_mode={tfidf_mode!r}; defaulting to 'all'.")
        print("[ADFA] TF-IDF fitting mode: all train (normals + anomalies)")
        fb.fit(train_norm + train_anoms)

    # scaler from train normals (tabular block)
    X_tmp, _ = fb.build_feature_matrix(train_norm)
    k = len(fb.tab_cols)
    means = X_tmp[:, :k].mean(axis=0).tolist()
    stds = X_tmp[:, :k].std(axis=0).tolist()
    scaler = {"means": dict(zip(fb.tab_cols, means)), "stds": dict(zip(fb.tab_cols, stds))}

    # build matrices
    X_trn, M_trn = fb.build_feature_matrix(train_norm + train_anoms, scaler=scaler)
    for i in range(len(train_norm)):
        M_trn[i]["label"] = 0
        M_trn[i]["targeted_flag"] = 0
    for i in range(len(train_norm), len(M_trn)):
        M_trn[i]["label"] = 1  # anomaly; targeted_flag already set per family

    X_valn, M_valn = fb.build_feature_matrix(val_norm, scaler=scaler)
    for m in M_valn:
        m["label"] = 0
        m["targeted_flag"] = 0
    X_vala, M_vala = fb.build_feature_matrix(val_anoms, scaler=scaler)
    for m in M_vala:
        m["label"] = 1

    X_tn, M_tn = fb.build_feature_matrix(test_norm, scaler=scaler)
    for m in M_tn:
        m["label"] = 0
        m["targeted_flag"] = 0
    X_ta, M_ta = fb.build_feature_matrix(test_anoms, scaler=scaler)
    for m in M_ta:
        m["label"] = 1

    X_test = np.vstack([X_tn, X_ta])
    M_test = M_tn + M_ta

    # save
    np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_trn, metas=M_trn)
    np.savez_compressed(os.path.join(out_dir, "val_norm.npz"), X=X_valn, metas=M_valn)
    np.savez_compressed(os.path.join(out_dir, "val_anom.npz"), X=X_vala, metas=M_vala)
    np.savez_compressed(os.path.join(out_dir, "test.npz"),  X=X_test, metas=M_test)

    fb.save(os.path.join(out_dir, "feature_builder.pkl"))
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump(scaler, f)

    print(f"Saved features to {out_dir}")
