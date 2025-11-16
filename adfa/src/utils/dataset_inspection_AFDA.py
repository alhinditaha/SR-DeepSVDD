import os, json, pickle, numpy as np, pandas as pd

DATA_DIR = r"data/adfa_features"  # adjust if needed

def load_split_df(data_dir, split):
    if split == "train":
        z = np.load(os.path.join(data_dir, "train.npz"), allow_pickle=True)
        metas = pd.DataFrame(z["metas"].tolist())
        X = z["X"]
        metas["split"] = "train"
        return metas, X

    if split == "val":
        vn = np.load(os.path.join(data_dir, "val_norm.npz"), allow_pickle=True)
        va = np.load(os.path.join(data_dir, "val_anom.npz"), allow_pickle=True)
        metas = pd.DataFrame(list(vn["metas"]) + list(va["metas"]))
        X = np.vstack([vn["X"], va["X"]])
        metas["split"] = ["val_norm"] * len(vn["metas"]) + ["val_anom"] * len(va["metas"])
        return metas, X

    if split == "test":
        z = np.load(os.path.join(data_dir, "test.npz"), allow_pickle=True)
        metas = pd.DataFrame(z["metas"].tolist())
        X = z["X"]
        metas["split"] = "test"
        return metas, X

    raise ValueError("split must be one of: train, val, test")

# Load dataframes (metas) for each split
df_train, X_train = load_split_df(DATA_DIR, "train")
df_val,   X_val   = load_split_df(DATA_DIR, "val")
df_test,  X_test  = load_split_df(DATA_DIR, "test")

# --- Quick inspections ---

# Counts by class (0=normal, 1=anomaly) and targeted_flag
print("TRAIN label/targeted:\n", pd.crosstab(df_train["label"], df_train["targeted_flag"]))
print("VAL   label/targeted:\n", pd.crosstab(df_val["label"],   df_val["targeted_flag"]))
print("TEST  label/targeted:\n", pd.crosstab(df_test["label"],  df_test["targeted_flag"]))

# Anomaly families distribution and targeted vs non-targeted
def fam_counts(df):
    fam = df.query("label==1")["anomaly_label"]
    return fam.value_counts().head(20)

print("\nTop anomaly families (VAL):\n", fam_counts(df_val))
print("\nTop anomaly families (TEST):\n", fam_counts(df_test))

print("\nTEST targeted families (top 20):\n",
      df_test.query("label==1 and targeted_flag==1")["anomaly_label"].value_counts().head(20))

# Save the three metas tables if you want to browse in Excel
# df_train.to_csv("train_metas.csv", index=False)
# df_val.to_csv("val_metas.csv", index=False)
# df_test.to_csv("test_metas.csv", index=False)

# --- (Optional) Inspect feature columns with names ---
# The first block of columns are tabular; the rest are TF-IDF terms.
with open(os.path.join(DATA_DIR, "feature_builder.pkl"), "rb") as f:
    fb = pickle.load(f)   # dict with 'tab_cols' and fitted 'vectorizer'

tab_cols = list(fb["tab_cols"])
tfidf_terms = list(fb["vectorizer"].get_feature_names_out())
feat_cols = tab_cols + [f"tfidf::{t}" for t in tfidf_terms]

# Example: show 5 targeted test anomalies with their meta + first 10 feature columns
ix = df_test.query("label==1 and targeted_flag==1").index[:5]
feat_view = pd.DataFrame(X_test[ix][:, :len(tab_cols)+10], columns=feat_cols[:len(tab_cols)+10])
print("\nSample targeted rows (TEST) with first 10 feature cols:\n",
      pd.concat([df_test.loc[ix].reset_index(drop=True), feat_view.reset_index(drop=True)], axis=1))


import os, numpy as np, pandas as pd

DATA_DIR = r"data/adfa_features"

def load_npz(fn):
    z = np.load(os.path.join(DATA_DIR, fn), allow_pickle=True)
    return pd.DataFrame(z["metas"].tolist()), z["X"]

df_tr, X_tr = load_npz("train.npz")
df_vn, X_vn = load_npz("val_norm.npz")
df_va, X_va = load_npz("val_anom.npz")
df_te, X_te = load_npz("test.npz")

# Build one VAL df and reset its index
df_val = pd.concat(
    [df_vn.assign(split="val_norm"), df_va.assign(split="val_anom")],
    ignore_index=True
)

print("TRAIN shape:", X_tr.shape)
print(pd.crosstab(df_tr["label"], df_tr["targeted_flag"]))

print("\nVAL shape:", (X_vn.shape[0] + X_va.shape[0], X_val := None))
print(pd.crosstab(df_val["label"], df_val["targeted_flag"]))

print("\nTEST shape:", X_te.shape)
print(pd.crosstab(df_te["label"], df_te["targeted_flag"]))

# Quick sanity: targeted counts among anomalies
print("\nVAL targeted anomalies:", int(df_val.query("label == 1")["targeted_flag"].sum()))
print("TEST targeted anomalies:", int(df_te.query("label == 1")["targeted_flag"].sum()))

# See which families are targeted
print("\nTargeted families in VAL (top 10):")
print(df_val.query("label == 1 and targeted_flag == 1")["anomaly_label"].value_counts().head(10))

print("\nTargeted families in TEST (top 10):")
print(df_te.query("label == 1 and targeted_flag == 1")["anomaly_label"].value_counts().head(10))
