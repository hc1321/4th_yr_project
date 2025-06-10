
import random, warnings
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble      import RandomForestRegressor
from sklearn.metrics       import (r2_score, mean_absolute_error,
                                   mean_absolute_percentage_error)
from sklearn.model_selection import KFold, ParameterGrid
from scipy.stats import spearmanr

random.seed(0);  np.random.seed(0)

ROOT      = Path(".")
TRAIN_DIR = ROOT / "distances 0.9 threshold" / "train"
TEST_DIR  = ROOT / "distances 0.9 threshold" / "test"
OUT_DIR   = ROOT / "plots_multi_sets_RF"; OUT_DIR.mkdir(exist_ok=True)

MIN_COVERAGE = 0.60
# Grid for Random-Forest – tweak freely ☺
PARAM_GRID   = {"n_estimators": [200],
                "max_depth":    [40]}
KFOLD        = KFold(n_splits=5, shuffle=True, random_state=0)


DISTANCE_GROUPS: dict[str, list[str] | list[list[str]] | str | None] = {
    
  
    "all": "all",   
    # "pca": "pca" 
}
""" "skeleton": [
        "b_t-b_a_1","b_a_1-b_a_2","b_a_2-b_a_3","b_a_3-b_a_4","b_a_4-b_a_5",
        "b_t-l_1_co_r","l_1_co_r-l_1_tr_r","l_1_tr_r-l_1_fe_r","l_1_fe_r-l_1_ti_r",
        "l_1_ti_r-l_1_ta_r","l_1_ta_r-l_1_pt_r",
        "l_1_co_r-l_2_co_r","l_2_co_r-l_2_tr_r","l_2_tr_r-l_2_fe_r","l_2_fe_r-l_2_ti_r",
        "l_2_ti_r-l_2_ta_r","l_2_ta_r-l_2_pt_r",
        "l_2_co_r-l_3_co_r","l_3_co_r-l_3_tr_r","l_3_tr_r-l_3_fe_r","l_3_fe_r-l_3_ti_r",
        "l_3_ti_r-l_3_ta_r","l_3_ta_r-l_3_pt_r",
        "b_t-l_1_co_l","l_1_co_l-l_1_tr_l","l_1_tr_l-l_1_fe_l","l_1_fe_l-l_1_ti_l",
        "l_1_ti_l-l_1_ta_l","l_1_ta_l-l_1_pt_l",
        "l_1_co_l-l_2_co_l","l_2_co_l-l_2_tr_l","l_2_tr_l-l_2_fe_l","l_2_fe_l-l_2_ti_l",
        "l_2_ti_l-l_2_ta_l","l_2_ta_l-l_2_pt_l",
        "l_2_co_l-l_3_co_l","l_3_co_l-l_3_tr_l","l_3_tr_l-l_3_fe_l","l_3_fe_l-l_3_ti_l",
        "l_3_ti_l-l_3_ta_l","l_3_ta_l-l_3_pt_l",
        "b_t-b_h","b_h-an_1_r","an_1_r-an_2_r","an_2_r-an_3_r",
        "b_h-an_1_l","an_1_l-an_2_l","an_2_l-an_3_l"
    ],"""

def extract_weight(name: str, frame: int | None = None) -> float:
    stem = Path(name).stem
    m = re.search(r'\d{5}', stem)
    if m:
        return float(m.group()) / 10.0
    if frame is not None:
        digits = ''.join(filter(str.isdigit, str(frame))).zfill(5)
        return float(digits[-5:]) / 10.0
    raise ValueError(f"Cannot recover weight from {name!r}")

def load_csv_features(path: Path, distances) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False,
                     usecols=["frame", "bone", "distance_mm", "distance_mm_mask"])
    df["distance_mm"]      = pd.to_numeric(df["distance_mm"],      errors="coerce")
    df["distance_mm_mask"] = pd.to_numeric(df["distance_mm_mask"], errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["frame", "bone"], keep="first")

    dist_w = df.pivot(index="frame", columns="bone",
                      values="distance_mm").add_prefix("dist_")
    mask_w = df.pivot(index="frame", columns="bone",
                      values="distance_mm_mask").add_prefix("mask_")

    circle_mode = isinstance(distances, list) and distances and isinstance(distances[0], list)

    if distances is None or distances == "all":
        groups = None
        flat   = [c.removeprefix("dist_") for c in dist_w.columns]
    elif isinstance(distances, str):
        if distances == "pca":
            groups = None
            flat   = [c.removeprefix("dist_") for c in dist_w.columns]
        else:
            raise ValueError(f"Unknown distance spec '{distances}'")
    elif circle_mode:
        groups = distances
        flat   = [d for grp in distances for d in grp]
    else:
        groups = None
        flat   = distances

    mask_cols = [f"mask_{d}" for d in flat if f"mask_{d}" in mask_w.columns]
    if not mask_cols:
        return pd.DataFrame()

    coverage = (mask_w[mask_cols] > 0).sum(axis=1) / len(flat)
    keep_idx = coverage[coverage >= MIN_COVERAGE].index
    if keep_idx.empty:
        return pd.DataFrame()

    dist_w = dist_w.loc[keep_idx].fillna(0.0)
    dist_w = dist_w * mask_w.loc[keep_idx].rename(
        columns=lambda c: c.replace("mask_", "dist_"))

    model_cols: list[str] = []
    if circle_mode:
        for gi, grp in enumerate(groups):
            valid = [f"dist_{d}" for d in grp if f"dist_{d}" in dist_w.columns]
            if valid:
                cname = f"group_sum_{gi}"
                dist_w[cname] = dist_w[valid].sum(axis=1)
                model_cols.append(cname)
    else:
        model_cols = [f"dist_{d}" for d in flat if f"dist_{d}" in dist_w.columns]

    if not model_cols:
        return pd.DataFrame()

    wide = dist_w[model_cols].reset_index()
    wide["weight_mg"] = extract_weight(path.name)
    return wide.astype(np.float32, errors="ignore")

def build_dataset(folder: Path, distances) -> pd.DataFrame:
    parts     = [load_csv_features(f, distances) for f in sorted(folder.glob("*.csv"))]
    non_empty = [p for p in parts if not p.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


mean_data, mape_data = {}, {}


for set_name, dist_spec in DISTANCE_GROUPS.items():
    print(f"\n=== {set_name.upper()} ===")

   
    train_df = build_dataset(TRAIN_DIR, dist_spec)
    test_df  = build_dataset(TEST_DIR , dist_spec)
    if train_df.empty or test_df.empty:
        print(" No usable frames – skipping.");  continue

    feat_cols = [c for c in train_df.columns
                     if c.startswith("dist_") or c.startswith("group_sum")]

    X_train = train_df[feat_cols].values
    y_train = np.log10(train_df.weight_mg.values)        
    X_test  = test_df[feat_cols].values
    y_test  = np.log10(test_df.weight_mg.values)

    best_params, best_mape = None, float("inf")
    for p in ParameterGrid(PARAM_GRID):
        fold_mapes = []
        for tr_idx, va_idx in KFOLD.split(X_train):
            rf = RandomForestRegressor(
                     n_estimators=p["n_estimators"],
                     max_depth=p["max_depth"],
                     n_jobs=-1, random_state=0)
            rf.fit(X_train[tr_idx], y_train[tr_idx])
            pred = rf.predict(X_train[va_idx])

            mape = mean_absolute_percentage_error(10**y_train[va_idx], 10**pred)
            fold_mapes.append(mape)

        mean_mape = float(np.mean(fold_mapes))
        print(f"  CV {p} → MAPE {mean_mape:.4f}")
        if mean_mape < best_mape:
            best_mape, best_params = mean_mape, p
    print(f" → Best {best_params}  (CV MAPE {best_mape:.4f})")

   
    rf = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth   =best_params["max_depth"],
            n_jobs=-1, random_state=0, oob_score=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rf.fit(X_train, y_train)

    y_pred_log = rf.predict(X_test)
    y_pred, y_true = 10**y_pred_log, 10**y_test

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    rho  = spearmanr(y_true, y_pred).correlation
    print(f"   R² {r2:.3f} | MAE {mae:.3f} mg | MAPE {mape:.2f}% | Spearman {rho:.3f}")

    PLOT_DIR = OUT_DIR / set_name;  PLOT_DIR.mkdir(exist_ok=True, parents=True)

  
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--')
    plt.xscale("log");  plt.yscale("log")
    plt.xlabel("True (mg)");  plt.ylabel("Pred (mg)")
    plt.title(f"Parity – {set_name} (RF)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "parity.png", dpi=150);  plt.close()

  
    df_mean = (pd.DataFrame({"true": y_true, "pred": y_pred})
               .groupby("true", as_index=False)["pred"].mean())
    mean_data[set_name] = df_mean

    df_mean.to_csv(PLOT_DIR / f"{set_name}_mean_parity.csv", index=False)

    plt.figure(figsize=(6, 6))
    plt.plot([df_mean.true.min(), df_mean.true.max()],
             [df_mean.true.min(), df_mean.true.max()], 'k--')
    plt.scatter(df_mean.true, df_mean.pred, s=40)
    plt.xscale("log");  plt.yscale("log")
    plt.xlabel("True");  plt.ylabel("Mean Pred")
    plt.title(f"Mean-Pred – {set_name} (RF)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mean_pred.png", dpi=150);  plt.close()


    df_mape = (pd.DataFrame({"true": y_true, "pred": y_pred})
               .groupby("true")
               .apply(lambda d: mean_absolute_percentage_error(d.true, d.pred) * 100)
               .reset_index(name="mape"))
    mape_data[set_name] = df_mape


    df_mape.to_csv(PLOT_DIR / f"{set_name}_mape.csv", index=False)

  
    bins    = np.logspace(np.log10(y_true.min()), np.log10(y_true.max()), 21)
    bin_ids = np.clip(np.digitize(y_true, bins, right=True) - 1, 0, 19)
    dfrm    = pd.DataFrame({"true": y_true, "pred": y_pred, "bin": bin_ids})
    bin_mape = (dfrm.groupby("bin", group_keys=False)
                     .apply(lambda d: mean_absolute_percentage_error(d.true, d.pred)*100)
                     .reindex(range(len(bins)-1)))
    centres = (bins[:-1] * bins[1:]) ** 0.5
    plt.figure(figsize=(7, 4))
    plt.plot(centres, bin_mape.values, marker="o")
    plt.xscale("log")
    plt.xlabel("True (mg)");  plt.ylabel("MAPE %")
    plt.title(f"MAPE per log-bin – {set_name} (RF)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mape_bin.png", dpi=150);  plt.close()


plt.figure(figsize=(6, 6))
for label, df in mean_data.items():
    plt.scatter(df.true, df.pred, label=label, alpha=0.7)
lims = [min(min(d.true.min() for d in mean_data.values()),
            min(d.pred.min() for d in mean_data.values())),
        max(max(d.true.max() for d in mean_data.values()),
            max(d.pred.max() for d in mean_data.values()))]
plt.plot(lims, lims, 'k--')
plt.xscale("log");  plt.yscale("log")
plt.xlabel("True (mg)");  plt.ylabel("Mean Pred (mg)")
plt.title("Mean parity – all sets (RF)");  plt.legend()
plt.tight_layout();  plt.savefig(OUT_DIR / "combined_mean_parity.png", dpi=150);  plt.close()

plt.figure(figsize=(7, 4))
for label, df in mape_data.items():
    plt.plot(df.true, df.mape, marker='o', label=label)
plt.xscale("log")
plt.xlabel("True weight (mg)");  plt.ylabel("MAPE %")
plt.title("MAPE versus weight – all sets (RF)")
plt.legend();  plt.tight_layout()
plt.savefig(OUT_DIR / "combined_mape_weight.png", dpi=150);  plt.close()


combined_mean = pd.concat(
    [df.assign(set_name=label) for label, df in mean_data.items()],
    ignore_index=True
)
combined_mean.to_csv(OUT_DIR / "combined_mean_parity_all_sets.csv", index=False)


combined_mape = pd.concat(
    [df.assign(set_name=label) for label, df in mape_data.items()],
    ignore_index=True
)
combined_mape.to_csv(OUT_DIR / "combined_mape_all_sets.csv", index=False)

print("\n✓ All variants finished – combined plots saved in", OUT_DIR)
print("✓ CSVs saved:")
print("   • Per-set mean parity  →  <each PLOT_DIR>/<set_name>_mean_parity.csv")
print("   • Per-set MAPE         →  <each PLOT_DIR>/<set_name>_mape.csv")
print("   • Combined mean parity →  combined_mean_parity_all_sets.csv")
print("   • Combined MAPE        →  combined_mape_all_sets.csv")
