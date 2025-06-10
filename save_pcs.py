from pathlib import Path
import re, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT        = Path(".")
TRAIN_DIR   = ROOT / "train"
TEST_DIR    = ROOT / "test"
OUT_DIR     = ROOT / "pca_outputs";  OUT_DIR.mkdir(exist_ok=True)

MIN_COVERAGE = 0.60       

def extract_weight(name: str, frame=None) -> float:
    m = re.search(r"\d{5}", Path(name).stem)
    if m:
        return float(m.group()) / 10.0
    if frame is not None:
        return float(str(frame).rjust(5, "0")[-5:]) / 10.0
    raise ValueError(f"cannot find weight in {name!r}")


def load_csv_features(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, usecols=["frame", "bone", "distance_mm", "distance_mm_mask"],
                     low_memory=False)
    df["distance_mm"]      = pd.to_numeric(df.distance_mm,      errors="coerce")
    df["distance_mm_mask"] = pd.to_numeric(df.distance_mm_mask, errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["frame", "bone"], keep="first")

    dist_w = df.pivot(index="frame", columns="bone", values="distance_mm").add_prefix("dist_")
    mask_w = df.pivot(index="frame", columns="bone", values="distance_mm_mask").add_prefix("mask_")

    flat = [c.removeprefix("dist_") for c in dist_w.columns]
    mask_cols = [f"mask_{d}" for d in flat if f"mask_{d}" in mask_w.columns]
    if not mask_cols:
        return pd.DataFrame()

    coverage = (mask_w[mask_cols] > 0).sum(axis=1) / len(flat)
    keep_idx = coverage[coverage >= MIN_COVERAGE].index
    if keep_idx.empty:
        return pd.DataFrame()

    dist_w = (dist_w.loc[keep_idx].fillna(0.0) *
              mask_w.loc[keep_idx].rename(columns=lambda c: c.replace("mask_", "dist_")))

    model_cols = [f"dist_{d}" for d in flat if f"dist_{d}" in dist_w.columns]
    if not model_cols:
        return pd.DataFrame()

    wide = dist_w[model_cols].reset_index()
    wide["weight_mg"] = extract_weight(csv.name)
    return wide.astype(np.float32, errors="ignore")


def build_dataset(folder: Path) -> pd.DataFrame:
    parts = [load_csv_features(f) for f in sorted(folder.glob("*.csv"))]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)

print(" loading datasets …")
train_df = build_dataset(TRAIN_DIR)
test_df  = build_dataset(TEST_DIR)
assert not train_df.empty and not test_df.empty, "found no usable data"

feat_cols = [c for c in train_df.columns if c.startswith("dist_")]
X_train   = train_df[feat_cols].values
X_test    = test_df[feat_cols].values
print(f"✓ shapes: train {X_train.shape}, test {X_test.shape}")


scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

pca = PCA(n_components=None, svd_solver="full", random_state=0)
train_scores = pca.fit_transform(X_train)
test_scores  = pca.transform(X_test)

cum_var = np.cumsum(pca.explained_variance_ratio_)

def pcs_for(thresh: float) -> int:

    return int(np.searchsorted(cum_var, thresh) + 1)

print("\nCumulative variance:")
for thr in (0.85, 0.90, 0.95):
    print(f"  {thr*100:>4.0f}% ⇒ {pcs_for(thr)} PCs")


k90 = pcs_for(0.90)
print(f"\nWill additionally save the first {k90} PCs (≥90% variance).")


def save_scores(arr: np.ndarray, df: pd.DataFrame, fname: str, n_pc: int | None = None):
  
    n = arr.shape[1] if n_pc is None else n_pc
    pd.DataFrame(arr[:, :n],
                 columns=[f"pca_{i}" for i in range(n)]) \
      .assign(weight_mg=df.weight_mg) \
      .to_csv(OUT_DIR / fname, index=False)


save_scores(train_scores, train_df, "train_pca_scores.csv")
save_scores(test_scores,  test_df,  "test_pca_scores.csv")


save_scores(train_scores, train_df, f"train_pca_scores_90var.csv", n_pc=k90)
save_scores(test_scores,  test_df,  f"test_pca_scores_90var.csv",  n_pc=k90)

ev_df = pd.DataFrame({"pc": [f"PC{i+1}" for i in range(len(cum_var))],
                      "var_ratio": pca.explained_variance_ratio_,
                      "cum_var": cum_var})
ev_df.to_csv(OUT_DIR / "explained_variance.csv", index=False)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cum_var)+1), cum_var*100, marker="o")
plt.xlabel("Principal component"); plt.ylabel("Cumulative variance (%)")
plt.title("Scree plot"); 
plt.tight_layout()
plt.savefig(OUT_DIR / "scree_plot.png", dpi=150); plt.close()


print("\n✓ outputs saved in", OUT_DIR)
