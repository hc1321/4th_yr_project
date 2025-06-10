
import os, random, re, json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold, ParameterGrid
from scipy.stats import mode, spearmanr


random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT        = Path(".")
TRAIN_DIR   = ROOT  / "train"
TEST_DIR    = ROOT / "test"
OUT_DIR     = ROOT / "plots_class_csv";  OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES   = 20    
SIGMA         = 4.0         
MIN_COVERAGE  = 0.6       


MAX_EPOCHS = 70
PATIENCE   = 5
DELTA      = 1e-4
PARAM_GRID = {"lr": [1e-3, 5e-4], "weight_decay": [1e-4, 1e-5]}
KFOLD      = KFold(n_splits=5, shuffle=True, random_state=0)


DISTANCE_GROUPS: Dict[str, Any] = {
    "skeleton": [
        "b_t-b_a_1","b_a_1-b_a_2","b_a_2-b_a_3","b_a_3-b_a_4","b_a_4-b_a_5",
        "b_t-l_1_co_r","l_1_co_r-l_1_tr_r","l_1_tr_r-l_1_fe_r","l_1_fe_r-l_1_ti_r",
        "l_1_ti_r-l_1_ta_r","l_1_ta_r-l_1_pt_r",
        "l_1_co_r-l_2_co_r","l_2_co_r-l_2_tr_r","l_2_tr_r-l_2_fe_r","l_2_fe_r-l_2_ti_r",
        "l_2_ti_r-l_2_ta_r","l_2_ta_r-l_2_pt_r",
        "l_2_co_r-l_3_co_r","l_3_co_r-l_3_tr_r","l_3_tr_r-l_3_fe_r","l_3_fe_r-l_3_ti_r",
        "l_3_ti_r-l_3_ta_r","l_3_ta_r-l_3_pt_r",
        "b_t-l_1_co_l","l_1_co_l-l_1_tr_l","l_1_tr_r-l_1_fe_l","l_1_fe_l-l_1_ti_l",
        "l_1_ti_r-l_1_ta_l","l_1_ta_r-l_1_pt_l",
        "l_1_co_l-l_2_co_l","l_2_co_l-l_2_tr_l","l_2_tr_r-l_2_fe_l","l_2_fe_l-l_2_ti_l",
        "l_2_ti_r-l_2_ta_l","l_2_ta_r-l_2_pt_l",
        "l_2_co_l-l_3_co_l","l_3_co_l-l_3_tr_l","l_3_tr_r-l_3_fe_l","l_3_fe_l-l_3_ti_l",
        "l_3_ti_r-l_3_ta_l","l_3_ta_r-l_3_pt_l",
        "b_t-b_h","b_h-an_1_r","an_1_r-an_2_r","an_2_r-an_3_r",
        "b_h-an_1_l","an_1_l-an_2_l","an_2_l-an_3_l"
    ],
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
        ["l_1_co_l-l_1_tr_l","l_1_tr_r-l_1_fe_l"],
        ["l_2_co_l-l_2_tr_l","l_2_tr_r-l_2_fe_l"],
        ["l_3_co_l-l_3_tr_l","l_3_tr_r-l_3_fe_l"]
    ],
    "body": [
        "an_1_r-an_1_l", "b_a_1-b_a_2", "b_a_2-b_a_3", "b_a_3-b_a_2", "b_a_4-b_a_5",
        "b_h-an_1_r", "b_h-an_1_l", "b_t-l_1_co_r", "b_t-l_1_co_l",
        "b_t-l_2_co_r", "b_t-l_2_co_l", "b_t-l_3_co_r", "b_t-l_3_co_l",
        "l_1_co_r-l_2_co_r", "l_1_co_l-l_2_co_l", "l_2_co_r-l_3_co_r", "l_2_co_l-l_3_co_l",
        "l_1_co_r-l_1_co_l", "l_2_co_r-l_2_co_l", "l_3_co_r-l_3_co_l"
    ],
    #"pca": "pca",  note pca was implemented in a separate script
    "all": "all",
}



def weight_to_class(w: float) -> int:
    bins = np.logspace(np.log10(1), np.log10(51), NUM_CLASSES + 1)
    return np.digitize(w, bins[:-1]) - 1

def extract_weight(name: str, frame: int | None = None) -> float:

    stem = Path(name).stem             
    first5 = stem[:5]                

    if len(first5) == 5 and first5.isdigit():
        return float(first5) / 10.0       

    if frame is not None:
        last5 = str(frame).zfill(5)[-5:] 
        if last5.isdigit():
            return float(last5) / 10.0

    raise ValueError(f"Cannot recover weight from {name!r} (frame={frame})")

def _flatten(spec):
    if spec is None or spec == "all": return None
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return [d for grp in spec for d in grp]
    return spec

def load_csv_wide_distances(path: Path, distances) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["distance_mm"] = pd.to_numeric(df["distance_mm"], errors="coerce")
    df["distance_mm_mask"] = pd.to_numeric(df["distance_mm_mask"], errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["frame", "bone"], keep="first")

    d = df.pivot(index="frame", columns="bone", values="distance_mm").add_prefix("dist_")
    m = df.pivot(index="frame", columns="bone", values="distance_mm_mask").add_prefix("mask_")

    if distances is not None and distances != "all":
        want = _flatten(distances)
        d_keep = [f"dist_{w}" for w in want if f"dist_{w}" in d.columns]
        m_keep = [f"mask_{w}" for w in want if f"mask_{w}" in m.columns]
        if not d_keep:
            return pd.DataFrame()
        d, m = d[d_keep], m[m_keep]

    keep = ((m > 0).mean(1) >= MIN_COVERAGE)
    if keep.sum() == 0:
        return pd.DataFrame()

    d, m = d.loc[keep], m.loc[keep]
    d = d.fillna(0) * m.rename(columns=lambda c: c.replace("mask_", "dist_"))

    wide = pd.concat([d, m], axis=1).astype(np.float32).reset_index()
    wide["weight_mg"] = wide["frame"].apply(lambda frame: extract_weight(path.name, frame))

    return wide

def build_dataset(folder: Path, distances) -> pd.DataFrame:
    parts = []
    for csv in sorted(folder.glob("*.csv")):
        wide = load_csv_wide_distances(csv, distances)
        if not wide.empty:
            wide["csv_file"] = csv.name
            parts.append(wide)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def crals_vec(c: int, k: int = NUM_CLASSES, s: float = SIGMA) -> np.ndarray:
    idx = np.arange(k)
    v = np.exp(-0.5 * ((idx - c) / s) ** 2)
    return v / v.sum()


class MLP(nn.Module):
    def __init__(self, din: int, k: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, k)
        )
    def forward(self, x):
        return self.net(x)


metrics_rows: List[Dict[str, Any]] = []
modal_data: Dict[str, pd.DataFrame] = {}


for set_name, dist_spec in DISTANCE_GROUPS.items():
    print(f"\n=== {set_name.upper()} ===")
    PLOT_DIR = OUT_DIR / set_name;  PLOT_DIR.mkdir(parents=True, exist_ok=True)


    train_df = build_dataset(TRAIN_DIR, dist_spec)
    test_df  = build_dataset(TEST_DIR , dist_spec)
    if train_df.empty or test_df.empty:
        print(" no usable frames")
        continue

    for df in (train_df, test_df):
        df.query("1 <= weight_mg <= 50", inplace=True); df.reset_index(drop=True, inplace=True)


    all_cols = sorted(set(train_df.columns) | set(test_df.columns))
    train_df = train_df.reindex(columns=all_cols, fill_value=0)
    test_df  = test_df .reindex(columns=all_cols, fill_value=0)
    feat_cols = [c for c in all_cols if c.startswith(("dist_", "mask_"))]


    Xtr = train_df[feat_cols].to_numpy(np.float32)
    Xte = test_df [feat_cols].to_numpy(np.float32)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd


    ytr_idx = np.array([weight_to_class(w) for w in train_df.weight_mg])
    yte_idx = np.array([weight_to_class(w) for w in test_df .weight_mg])

    ytr_soft = np.vstack([crals_vec(i) for i in ytr_idx]).astype(np.float32)
    yte_soft = np.vstack([crals_vec(i) for i in yte_idx]).astype(np.float32)


    XtrT = torch.tensor(Xtr)
    ytrT = torch.tensor(ytr_soft)
    XteT = torch.tensor(Xte)


    best_params, best_val = None, -1.0    

    for p in ParameterGrid(PARAM_GRID):
        fold_scores: List[float] = []

        for fold_i, (tr_idx, va_idx) in enumerate(KFOLD.split(XtrT), 1):
            mdl = MLP(len(feat_cols)).to(DEVICE)
            opt = torch.optim.Adam(mdl.parameters(),
                                lr=p["lr"], weight_decay=p["weight_decay"])
            crit = nn.KLDivLoss(reduction="batchmean")

            loader = DataLoader(TensorDataset(XtrT[tr_idx], ytrT[tr_idx]),
                                batch_size=128, shuffle=True)

            best_fold_mcc, counter = -1.0, 0
            stop_ep = MAX_EPOCHS

            for ep in range(MAX_EPOCHS):

                mdl.train()
                for xb, yb in loader:
                    logp = F.log_softmax(mdl(xb.to(DEVICE)), dim=1)
                    loss = crit(logp, yb.to(DEVICE))
                    opt.zero_grad(); loss.backward(); opt.step()

 
                mdl.eval()
                with torch.no_grad():
                    preds = torch.argmax(mdl(XtrT[va_idx].to(DEVICE)), 1).cpu().numpy()
                    val_mcc = matthews_corrcoef(ytr_idx[va_idx], preds)

                if val_mcc > best_fold_mcc + DELTA:
                    best_fold_mcc, counter = val_mcc, 0
                else:
                    counter += 1
                    if counter >= PATIENCE:
                        stop_ep = ep + 1
                        break

            print(f"    CV {p} – fold {fold_i}: stopped at epoch {stop_ep}, "
                f"best MCC {best_fold_mcc:.4f}")
            fold_scores.append(best_fold_mcc)

        mean_mcc = float(np.mean(fold_scores))
        print(f"  CV {p} → mean MCC {mean_mcc:.4f}")
        if mean_mcc > best_val:
            best_val, best_params = mean_mcc, p

    print(f" → Best params {best_params}  (mean MCC {best_val:.4f})")


    model = MLP(len(feat_cols)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=best_params["lr"],
                             weight_decay=best_params["weight_decay"])
    crit  = nn.KLDivLoss(reduction="batchmean")

    loader = DataLoader(TensorDataset(XtrT, ytrT), batch_size=128, shuffle=True)
    best_train, counter = float("inf"), 0
    for ep in range(MAX_EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in loader:
            logp = F.log_softmax(model(xb.to(DEVICE)), dim=1)
            loss = crit(logp, yb.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)

        epoch_loss = tot / len(loader.dataset)
        if epoch_loss < best_train - DELTA:
            best_train, counter = epoch_loss, 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  ⏹ early-stopped final fit at epoch {ep+1}")
                break

        if ep % 10 == 0 or ep == MAX_EPOCHS - 1:
            print(f"  Ep {ep:3d}: train KL {epoch_loss:.4f}")

    print("  ✓ training done")


    model.eval(); logits = model(XteT.to(DEVICE)).cpu()
    pred_classes = torch.argmax(logits, dim=1).numpy()
    true_classes = yte_idx

    bins = np.logspace(np.log10(1), np.log10(51), NUM_CLASSES + 1)
    cent = (bins[:-1] + bins[1:]) / 2
    true_w = cent[true_classes];  pred_w = cent[pred_classes]

    acc  = accuracy_score(true_classes, pred_classes)
    mcc  = matthews_corrcoef(true_classes, pred_classes)
    srcc,_ = spearmanr(true_w, pred_w)
    mape = np.mean(np.abs((pred_w - true_w) / true_w)) * 100
    print(f"   ACC {acc:.3f} | SRCC {srcc:.3f} | MAPE {mape:.2f}% | MCC {mcc:.3f}")


    metrics_rows.append({
        "distance_set": set_name,
        "ACC": acc,
        "SRCC": srcc,
        "MAPE": mape,
        "MCC": mcc,
        "best_lr": best_params["lr"],
        "best_weight_decay": best_params["weight_decay"],
        "best_cv_KL": best_val,
    })


    pred_df = pd.DataFrame({
        "csv_file": test_df["csv_file"],
        "frame":    test_df["frame"],
        "true_w":   true_w,
        "pred_w":   pred_w,
        "true_class": true_classes,
        "pred_class": pred_classes,
    })
    pred_df.to_csv(PLOT_DIR / "per_frame_predictions.csv", index=False)


    cm = confusion_matrix(true_classes, pred_classes, labels=list(range(NUM_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(NUM_CLASSES)],
                 columns=[f"pred_{i}" for i in range(NUM_CLASSES)]) \
        .to_csv(PLOT_DIR / "confusion_raw.csv")
    pd.DataFrame(cm_norm, index=[f"true_{i}" for i in range(NUM_CLASSES)],
                 columns=[f"pred_{i}" for i in range(NUM_CLASSES)]) \
        .to_csv(PLOT_DIR / "confusion_norm.csv")


    plt.figure(figsize=(6,5)); plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"CM – {set_name}"); plt.colorbar(); plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion.png", dpi=150); plt.close()

    plt.figure(figsize=(6,6)); plt.plot([1,60],[1,60],"k--")
    plt.scatter(true_w, pred_w, s=10, alpha=0.4, edgecolors="none")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("True (mg)"); plt.ylabel("Pred (mg)")
    plt.title(f"Per-frame parity – {set_name}")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "frame_parity.png", dpi=150); plt.close()

    mean_df  = (pred_df.groupby("true_w", as_index=False)["pred_w"].mean()
                       .rename(columns={"pred_w":"mean_pred_w"}))
    modal_df = (pred_df.groupby("true_w")["pred_w"]
                       .agg(lambda x: mode(x, keepdims=True).mode[0])
                       .reset_index())
    mean_df.to_csv(PLOT_DIR / "mean_parity_values.csv", index=False)
    modal_df.to_csv(PLOT_DIR / "modal_parity_values.csv", index=False)

    modal_data[set_name] = modal_df

    plt.figure(figsize=(6,6)); plt.plot([1,60],[1,60],"k--")
    plt.scatter(modal_df.true_w, modal_df.pred_w, s=60, alpha=0.8, edgecolors="none")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("True (mg)"); plt.ylabel("Modal pred (mg)")
    plt.title(f"Modal parity – {set_name}")
    plt.tight_layout(); plt.savefig(PLOT_DIR / "modal_parity.png", dpi=150); plt.close()

    print(f"   → saved CSV & plots in {PLOT_DIR}")


if modal_data:
    plt.figure(figsize=(6,6))
    for label, md in modal_data.items():
        plt.scatter(md.true_w, md.pred_w, s=50, alpha=0.7, label=label, edgecolors="none")
    lims = [1, 60]
    plt.plot(lims, lims, "k--")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("True weight (mg)"); plt.ylabel("Modal predicted (mg)")
    plt.title("Modal parity – all distance sets")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "combined_modal_parity.png", dpi=150); plt.close()


metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
print("\n✓ finished – results (CSV & PNG) in", OUT_DIR)
