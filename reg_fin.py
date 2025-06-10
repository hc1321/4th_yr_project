import random
from pathlib import Path
import pickle                   

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import KFold, ParameterGrid
from torch.utils.data import DataLoader, TensorDataset
import re
from pathlib import Path

random.seed(0);  np.random.seed(0);  torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT      = Path(".")
TRAIN_DIR = ROOT / "train"
TEST_DIR  = ROOT  / "test"
OUT_DIR   = ROOT / "plots_multi_sets_REG_REDO_SUMMED"; OUT_DIR.mkdir(exist_ok=True)

MIN_COVERAGE = 0.60
PARAM_GRID   = {"lr": [1e-3, 5e-4], "weight_decay": [1e-4, 1e-5]}
KFOLD        = KFold(n_splits=5, shuffle=True, random_state=0)


MAX_EPOCHS = 75     
PATIENCE   = 5        
DELTA      = 1e-4     
DISTANCE_GROUPS: dict[str, list[str] | list[list[str]] | str | None] = {
    "circle": [
        ["l_3_fe_l-l_2_fe_l","l_2_fe_l-l_1_fe_l","l_1_fe_l-an_1_l","an_1_l-an_1_r",
         "an_1_r-l_1_fe_r","l_1_fe_r-l_2_fe_r","l_2_fe_r-l_3_fe_r","l_3_fe_r-b_a_5",
         "b_a_5-l_3_fe_l"],
        ["l_1_co_r-l_2_co_r","l_2_co_r-l_3_co_r","l_3_co_r-b_a_1","b_a_1-l_3_co_l",
         "l_3_co_l-l_2_co_l","l_2_co_l-l_1_co_l","l_1_co_l-b_h","b_h-l_1_co_r"],
        ["b_a_5-b_a_4","b_a_4-b_a_3","b_a_3-b_a_2","b_a_2-b_a_1","b_a_1-b_h"],
        ["b_a_5-an_1_l"], ["b_a_5-an_1_r"], ["b_a_5-b_h"],
        ["l_1_co_r-l_1_tr_r","l_1_tr_r-l_1_fe_r"],
        ["l_2_co_r-l_2_tr_r","l_2_tr_r-l_2_fe_r"],
        ["l_3_co_r-l_3_tr_r","l_3_tr_r-l_3_fe_r"],
        ["l_1_co_l-l_1_tr_l","l_1_tr_l-l_1_fe_l"],
        ["l_2_co_l-l_2_tr_l","l_2_tr_l-l_2_fe_l"],
        ["l_3_co_l-l_3_tr_l","l_3_tr_l-l_3_fe_l"]
    ],

    # FULL skeleton list (unchanged)
 #   "skeleton": [
  #      "b_t-b_a_1","b_a_1-b_a_2","b_a_2-b_a_3","b_a_3-b_a_4","b_a_4-b_a_5",
   #     "b_t-l_1_co_r","l_1_co_r-l_1_tr_r","l_1_tr_r-l_1_fe_r","l_1_fe_r-l_1_ti_r",
    #    "l_1_ti_r-l_1_ta_r","l_1_ta_r-l_1_pt_r",
#        "l_1_co_r-l_2_co_r","l_2_co_r-l_2_tr_r","l_2_tr_r-l_2_fe_r","l_2_fe_r-l_2_ti_r",
 #       "l_2_ti_r-l_2_ta_r","l_2_ta_r-l_2_pt_r",
  #      "l_2_co_r-l_3_co_r","l_3_co_r-l_3_tr_r","l_3_tr_r-l_3_fe_r","l_3_fe_r-l_3_ti_r",
   #     "l_3_ti_r-l_3_ta_r","l_3_ta_r-l_3_pt_r",
    #    "b_t-l_1_co_l","l_1_co_l-l_1_tr_l","l_1_tr_l-l_1_fe_l","l_1_fe_l-l_1_ti_l",
#        "l_1_ti_l-l_1_ta_l","l_1_ta_l-l_1_pt_l",
 #       "l_1_co_l-l_2_co_l","l_2_co_l-l_2_tr_l","l_2_tr_l-l_2_fe_l","l_2_fe_l-l_2_ti_l",
  #      "l_2_ti_l-l_2_ta_l","l_2_ta_l-l_2_pt_l",
   #     "l_2_co_l-l_3_co_l","l_3_co_l-l_3_tr_l","l_3_tr_l-l_3_fe_l","l_3_fe_l-l_3_ti_l",
    #    "l_3_ti_l-l_3_ta_l","l_3_ta_l-l_3_pt_l",
     #   "b_t-b_h","b_h-an_1_r","an_1_r-an_2_r","an_2_r-an_3_r",
      #  "b_h-an_1_l","an_1_l-an_2_l","an_2_l-an_3_l"
 #   ],
  #  "body": [
   #     "an_1_r-an_1_l", "b_a_1-b_a_2", "b_a_2-b_a_3", "b_a_3-b_a_4", "b_a_4-b_a_5",
    #    "b_h-an_1_r", "b_h-an_1_l",
     #   "b_t-l_1_co_r", "b_t-l_1_co_l",
      #  "b_t-l_2_co_r", "b_t-l_2_co_l",
       # "b_t-l_3_co_r", "b_t-l_3_co_l",
        # Coxa–coxa spans (same side)
#        "l_1_co_r-l_2_co_r", "l_1_co_l-l_2_co_l",
 #       "l_2_co_r-l_3_co_r", "l_2_co_l-l_3_co_l",
        # Coxa–coxa spans (across body)
  #      "l_1_co_r-l_1_co_l", "l_2_co_r-l_2_co_l", "l_3_co_r-l_3_co_l"
  #  ],

 #   "all": "all",   # use every available distance as-is
 #   "pca": "pca"    # reduce all distances to 40 PCs first
}


from pathlib import Path


def extract_weight(name: str, frame: int | str | None = None) -> float:
   
    stem = Path(name).stem
    if stem.isdigit() and len(stem) == 5:     
        digits = stem
    else:                                        
        if frame is None:
            raise ValueError(
                "Filename is not a 5-digit weight and no frame value was supplied."
            )
        digits_only = re.sub(r"\D", "", str(frame))
        if not digits_only:
            raise ValueError(f"No digits found in frame {frame!r}")
        digits = digits_only[-5:].zfill(5)        

    return int(digits) / 10.0                  





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
    dist_w = dist_w * mask_w.loc[keep_idx].rename(columns=lambda c: c.replace("mask_", "dist_"))

    model_cols = []
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
    wide["weight_mg"] = wide["frame"].apply(
        lambda fr: extract_weight(path.name, fr)
    )
    return wide.astype(np.float32, errors="ignore")

def build_dataset(folder: Path, distances) -> pd.DataFrame:
    parts = [load_csv_features(f, distances) for f in sorted(folder.glob("*.csv"))]
    non_empty = [p for p in parts if not p.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

class MLPReg(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32),  nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

FRAME_RESULTS, MEAN_RESULTS, MAPE_RESULTS, BIN_RESULTS = [], [], [], []
mean_data: dict[str, pd.DataFrame] = {}
mape_data: dict[str, pd.DataFrame] = {}


for name, dist_spec in DISTANCE_GROUPS.items():
    print(f"\n=== {name.upper()} ===")

    train_df = build_dataset(TRAIN_DIR, dist_spec)
    test_df  = build_dataset(TEST_DIR , dist_spec)
    if train_df.empty or test_df.empty:
        print(" No usable frames – skipping.")
        continue


    pca = None
    if dist_spec == "pca":
        from sklearn.decomposition import PCA
        raw_feats = [c for c in train_df.columns if c.startswith("dist_")]
        pca = PCA(n_components=40, random_state=0)
        train_p = pca.fit_transform(train_df[raw_feats])
        test_p  = pca.transform(test_df[raw_feats])
        feat_cols = [f"pca_{i}" for i in range(train_p.shape[1])]
        train_df[feat_cols] = train_p;  test_df[feat_cols] = test_p
    else:
        feat_cols = [c for c in train_df.columns
                     if c.startswith("dist_") or c.startswith("group_sum")]


    X_train = torch.tensor(train_df[feat_cols].values, dtype=torch.float32, device=DEVICE)
    y_train = torch.tensor(np.log10(train_df.weight_mg.values),
                           dtype=torch.float32, device=DEVICE).unsqueeze(1)
    X_test  = torch.tensor(test_df[feat_cols].values , dtype=torch.float32, device=DEVICE)
    y_test  = torch.tensor(np.log10(test_df.weight_mg.values),
                           dtype=torch.float32, device=DEVICE).unsqueeze(1)


    best_params, best_mape = None, float("inf")
    for p in ParameterGrid(PARAM_GRID):
        fold_mapes = []
        for fold_i, (tr_idx, va_idx) in enumerate(KFOLD.split(X_train), 1):
            mdl, crit = MLPReg(len(feat_cols)).to(DEVICE), nn.MSELoss()
            opt = torch.optim.Adam(mdl.parameters(), **p)
            loader = DataLoader(TensorDataset(X_train[tr_idx], y_train[tr_idx]),
                                batch_size=128, shuffle=True)

            best_val, counter = float("inf"), 0
            epoch_stop = MAX_EPOCHS
            for ep in range(MAX_EPOCHS):
                # one epoch
                for xb, yb in loader:
                    loss = crit(mdl(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()

                with torch.no_grad():
                    pred = mdl(X_train[va_idx]).cpu().squeeze()
                    true = y_train[va_idx].cpu().squeeze()
                    val_mape = mean_absolute_percentage_error(10**true, 10**pred)

                if val_mape < best_val - DELTA:
                    best_val, counter = val_mape, 0
                else:
                    counter += 1
                    if counter >= PATIENCE:
                        epoch_stop = ep + 1
                        break

            print(f"    {name} – CV fold {fold_i}: "f"loss {loss.item():.4f} | stopped at epoch {epoch_stop} | best MAPE {best_val:.4f}")
            fold_mapes.append(best_val)

        mean_mape = float(np.mean(fold_mapes))
        print(f"  CV {p} → MAPE {mean_mape:.4f}")
        if mean_mape < best_mape:
            best_mape, best_params = mean_mape, p
    print(f" → Best {best_params}  (CV MAPE {best_mape:.4f})")


    model = MLPReg(len(feat_cols)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), **best_params)
    crit  = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=128, shuffle=True)

    best_train, counter = float("inf"), 0
    for ep in range(MAX_EPOCHS):
        total = 0.0
        for xb, yb in loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)

        epoch_mse = total / len(loader.dataset)
        if epoch_mse < best_train - DELTA:
            best_train, counter = epoch_mse, 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  ⏹ {name}: early-stopped final fit at epoch {ep+1}")
                break

        if ep % 10 == 0 or ep == MAX_EPOCHS-1:
            print(f"  Ep {ep:2d}: train MSE {epoch_mse:.4f}")


    model.eval()
    with torch.no_grad():
        y_pred_log = model(X_test).cpu().squeeze().numpy()

    y_pred, y_true = 10**y_pred_log, 10**y_test.cpu().squeeze().numpy()
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    rho  = spearmanr(y_true, y_pred).correlation
    print(f"   R² {r2:.3f} | MAE {mae:.3f} mg | MAPE {mape:.2f}% | Spearman {rho:.3f}")

    torch.save({
        "state_dict": model.state_dict(),
        "feat_cols": feat_cols,
        "dist_spec": dist_spec,
        "pca": pca
    }, OUT_DIR / f"model_{name}.pt")
    print(f" model saved → model_{name}.pt")

    PLOT_DIR = OUT_DIR / name;  PLOT_DIR.mkdir(exist_ok=True, parents=True)

    bins = np.logspace(np.log10(1.0), np.log10(50.0), 26)   
    plt.figure(figsize=(7, 4))

    ax1 = plt.gca()
    ax1.hist(train_df["weight_mg"], bins=bins, alpha=0.6,
            label="Train", edgecolor="none")
    ax1.set_xscale("log")
    ax1.set_xlabel("Weight (mg, log scale)")
    ax1.set_ylabel("Train – number of frames")

    ax2 = ax1.twinx()
    ax2.hist(test_df["weight_mg"], bins=bins, alpha=0.6,
            label="Test", edgecolor="none", color="tab:orange")
    ax2.set_ylabel("Test – number of frames")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center",
            ncol=2, frameon=False)

    plt.title(f"Weight distribution – {name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "weight_hist_logbins.png", dpi=150)
    plt.close()


    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--')
    plt.xscale("log");  plt.yscale("log")
    plt.xlabel("True (mg)");  plt.ylabel("Pred (mg)")
    plt.title(f"Parity – {name}");  plt.tight_layout()
    plt.savefig(PLOT_DIR / "parity.png", dpi=150);  plt.close()

    df_mean = (pd.DataFrame({"true": y_true, "pred": y_pred})
               .groupby("true", as_index=False)["pred"].mean())
    mean_data[name] = df_mean
    plt.figure(figsize=(6, 6))
    plt.plot([df_mean.true.min(), df_mean.true.max()],
             [df_mean.true.min(), df_mean.true.max()], 'k--')
    plt.scatter(df_mean.true, df_mean.pred, s=40)
    plt.xscale("log");  plt.yscale("log")
    plt.xlabel("True");  plt.ylabel("Mean Pred")
    plt.title(f"Mean-Pred – {name}");  plt.tight_layout()
    plt.savefig(PLOT_DIR / "mean_pred.png", dpi=150);  plt.close()


    df_mape = (pd.DataFrame({"true": y_true, "pred": y_pred})
               .groupby("true")
               .apply(lambda d: mean_absolute_percentage_error(d.true, d.pred) * 100)
               .reset_index(name="mape"))
    mape_data[name] = df_mape

    bins      = np.logspace(np.log10(y_true.min()), np.log10(y_true.max()), 21)
    bin_ids   = np.clip(np.digitize(y_true, bins, right=True) - 1, 0, 19)
    dfrm      = pd.DataFrame({"true": y_true, "pred": y_pred, "bin": bin_ids})

    bin_mape  = (
        dfrm.groupby("bin", group_keys=False)     
            .apply(lambda d: mean_absolute_percentage_error(d.true, d.pred) * 100)
    )

    bin_mape  = bin_mape.reindex(range(len(bins) - 1))
    centers   = (bins[:-1] * bins[1:]) ** 0.5            

    plt.figure(figsize=(7, 4))
    plt.plot(centers, bin_mape.values, marker="o")
    plt.xscale("log")
    plt.xlabel("True (mg)");  plt.ylabel("MAPE %")
    plt.title(f"MAPE per log-bin – {name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "mape_bin.png", dpi=150)
    plt.close()


    tmp = test_df.loc[:, ["frame", "weight_mg"]].copy()
    tmp["pred_weight_mg"] = y_pred
    tmp["set"] = name
    FRAME_RESULTS.append(tmp)

    MEAN_RESULTS.append(df_mean.assign(set=name))
    MAPE_RESULTS.append(df_mape.assign(set=name))
    BIN_RESULTS.append(pd.DataFrame({
        "weight_center": centers,
        "mape": bin_mape.values,
        "set": name
    }))


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
plt.title("Mean parity – all sets")
plt.legend();  plt.tight_layout()
plt.savefig(OUT_DIR / "combined_mean_parity.png", dpi=150);  plt.close()

plt.figure(figsize=(7, 4))
for label, df in mape_data.items():
    plt.plot(df.true, df.mape, marker='o', label=label)
plt.xscale("log")
plt.xlabel("True weight (mg)");  plt.ylabel("MAPE %")
plt.title("MAPE versus weight – all sets")
plt.legend();  plt.tight_layout()
plt.savefig(OUT_DIR / "combined_mape_weight.png", dpi=150);  plt.close()



pd.concat(FRAME_RESULTS, ignore_index=True).to_csv(
    OUT_DIR / "all_frame_predictions.csv", index=False)

pd.concat(MEAN_RESULTS, ignore_index=True).rename(
    columns={"true": "true_weight_mg",
             "pred": "mean_pred_weight_mg"}
).to_csv(OUT_DIR / "mean_predictions_per_weight.csv", index=False)

pd.concat(MAPE_RESULTS, ignore_index=True).rename(
    columns={"true": "true_weight_mg"}
).to_csv(OUT_DIR / "mape_per_weight.csv", index=False)

pd.concat(BIN_RESULTS, ignore_index=True).to_csv(
    OUT_DIR / "mape_per_bin.csv", index=False)

print("✓ All variants finished – plots, CSVs and models saved in", OUT_DIR)