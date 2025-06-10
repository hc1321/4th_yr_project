
import random, os, json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np, pandas as pd
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from scipy.stats import spearmanr, mode


random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ROOT        = Path(".")
TRAIN_DIR   = ROOT / "distances 0.9 threshold" / "train"
TEST_DIR    = ROOT / "distances 0.9 threshold" / "test"
OUT_DIR     = ROOT / "plots_class_plain_NEW"; OUT_DIR.mkdir(exist_ok=True)

MIN_COVERAGE = 0.60
NUM_CLASSES  = 20


MAX_EPOCHS = 70
PATIENCE   = 5
DELTA      = 1e-4
PARAM_GRID = {"lr": [1e-3, 5e-4], "weight_decay": [1e-4, 1e-5]}
KFOLD      = KFold(n_splits=5, shuffle=True, random_state=0)


DISTANCE_GROUPS: Dict[str, List[str] | List[List[str]] | str | None] = {
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

    "skeleton": [
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
    ],

    "body": [
        "an_1_r-an_1_l","b_a_1-l_3_co_l","l_1_co_r-l_1_co_l","l_2_co_r-l_2_tr",
        "b_a_2-b_a_3","b_a_3-b_a_4","b_h-an_1_r","l_2_co_l-l_2_tr",
        "b_a_1-l_3_co_r","l_1_co_r-l_1_tr","l_3_co_l-l_3_tr","l_2_co_l-l_3_co_l",
        "l_2_co_r-l_2_co_l","l_2_co_r-l_3_co_r","b_t-l_1_co_r","l_3_co_r-l_3_co_l",
        "b_a_4-b_a_5","b_h-an_1_l","l_1_co_l-l_1_tr","l_1_co_r-l_2_co_r",
        "b_t-l_1_co_l","b_a_1-l_3_tr","l_3_co_r-l_3_tr","b_a_1-l_2_co","b_a_1-b_a_2"
    ],

    "all": "all",
}

def _flatten(spec):
    if spec in (None, "all"): return None
    if isinstance(spec[0], list):
        return [d for g in spec for d in g]
    return spec


def weight_to_class(w: float) -> int:
    bins = np.logspace(np.log10(1), np.log10(51), NUM_CLASSES+1)
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

def load_csv_features(path: Path, distances):
    df = pd.read_csv(path, low_memory=False)
    df["distance_mm"]      = pd.to_numeric(df["distance_mm"], errors="coerce")
    df["distance_mm_mask"] = pd.to_numeric(df["distance_mm_mask"], errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["frame", "bone"], keep="first")

    dist = df.pivot(index="frame", columns="bone", values="distance_mm").add_prefix("dist_")
    mask = df.pivot(index="frame", columns="bone", values="distance_mm_mask").add_prefix("mask_")

    if distances not in (None, "all"):
        want = _flatten(distances)
        dist = dist[[f"dist_{d}" for d in want if f"dist_{d}" in dist.columns]]
        mask = mask[[f"mask_{d}" for d in want if f"mask_{d}" in mask.columns]]

    coverage = (mask > 0).mean(axis=1)
    keep = coverage >= MIN_COVERAGE
    if keep.sum() == 0:
        return pd.DataFrame()

    dist = dist.loc[keep].fillna(0) * \
           mask.loc[keep].rename(columns=lambda c: c.replace("mask_", "dist_"))
    dist["coverage"] = coverage.loc[keep].values
    return dist.reset_index()

def build_dataset(folder: Path, distances):
    parts = []
    for csv in sorted(folder.glob("*.csv")):
        wide = load_csv_features(csv, distances)
        if not wide.empty:
            wide["weight_mg"] = wide["frame"].apply(lambda f, fn=csv: extract_weight(fn.name, f))
            parts.append(wide.assign(csv_file=csv.name))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df.query("1 <= weight_mg <= 50").reset_index(drop=True)




class MLP(nn.Module):
    def __init__(self, d_in: int, k: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 32),   nn.ReLU(),
            nn.Linear(32, k)
        )
    def forward(self, x):
        return self.net(x)


metrics_rows: List[Dict[str, Any]] = []
modal_parity: Dict[str, pd.DataFrame] = {}


for name, dist_spec in DISTANCE_GROUPS.items():
    print(f"\n=== {name.upper()} ===")
    PLOT_DIR = OUT_DIR / name; PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("  building train")
    train_df = build_dataset(TRAIN_DIR, dist_spec)
    print("BUILDING TEST")
    test_df  = build_dataset(TEST_DIR , dist_spec)
    if train_df.empty or test_df.empty:
        print("skipped (no usable frames)")
        continue

    feats = [c for c in train_df.columns if c.startswith("dist_")]
    mu, sd = train_df[feats].mean(0), train_df[feats].std(0)+1e-6
    train_df[feats] = (train_df[feats]-mu)/sd
    test_df [feats] = (test_df [feats]-mu)/sd

    Xtr = torch.tensor(train_df[feats].values, dtype=torch.float32)
    ytr = torch.tensor([weight_to_class(w) for w in train_df.weight_mg], dtype=torch.long)
    Xte = torch.tensor(test_df[feats].values,  dtype=torch.float32)
    yte = torch.tensor([weight_to_class(w) for w in test_df.weight_mg ], dtype=torch.long)


    best_p: Dict[str, float] = {}
    best_score: float = -1.0

    for p in ParameterGrid(PARAM_GRID):
        print("  testing hyper-params:", p)
        fold_scores: List[float] = []

        for fold_i, (tr_idx, va_idx) in enumerate(KFOLD.split(Xtr), 1):
            mdl = MLP(len(feats)).to(DEVICE)
            opt = torch.optim.Adam(mdl.parameters(), **p)

            cls_cnt = np.bincount(ytr[tr_idx].numpy(), minlength=NUM_CLASSES)
            weights = (1/(cls_cnt+1e-6))*NUM_CLASSES/np.sum(1/(cls_cnt+1e-6))
            crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))

            train_data = DataLoader(TensorDataset(Xtr[tr_idx], ytr[tr_idx]),
                                    batch_size=128, shuffle=True)

            best_fold_loss: float = float("inf")   
            best_fold_mcc : float = -1.0         
            counter: int = 0
            stop_ep: int = MAX_EPOCHS

            for ep in range(MAX_EPOCHS):
               
                mdl.train()
                for xb, yb in train_data:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    loss = crit(mdl(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()

                
                mdl.eval()
                with torch.no_grad():
                    logits   = mdl(Xtr[va_idx].to(DEVICE))
                    val_loss = crit(logits, ytr[va_idx].to(DEVICE)).item()
                    preds    = torch.argmax(logits, 1).cpu().numpy()
                    val_mcc  = matthews_corrcoef(ytr[va_idx].cpu().numpy(), preds)

              
                if val_loss < best_fold_loss - DELTA:
                    best_fold_loss, counter = val_loss, 0
                else:
                    counter += 1
                    if counter >= PATIENCE:
                        stop_ep = ep + 1 
                        break

          
                if val_mcc > best_fold_mcc:
                    best_fold_mcc = val_mcc

            print(f"    CV {p} – fold {fold_i}: stopped at epoch {stop_ep}, "
                  f"best loss {best_fold_loss:.4f}, best MCC {best_fold_mcc:.4f}")
            fold_scores.append(best_fold_mcc) 

        mean_mcc = float(np.mean(fold_scores))
        print(f"  CV {p} → mean MCC {mean_mcc:.4f}")
        if mean_mcc > best_score:
            best_score, best_p = mean_mcc, p

    print("  best hyper-params:", best_p, "CV MCC:", best_score)


    model = MLP(len(feats)).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), **best_p)

    cls_cnt = np.bincount(ytr.numpy(), minlength=NUM_CLASSES)
    weights = (1/(cls_cnt+1e-6))*NUM_CLASSES/np.sum(1/(cls_cnt+1e-6))
    crit    = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    loader  = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)

    best_train_loss: float = float("inf")
    counter: int = 0

    for ep in range(MAX_EPOCHS):
        model.train(); tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = crit(model(xb), yb)
            optim.zero_grad(); loss.backward(); optim.step()
            tot += loss.item() * xb.size(0)

        epoch_loss = tot / len(loader.dataset)
        if epoch_loss < best_train_loss - DELTA:
            best_train_loss, counter = epoch_loss, 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  ⏹ early-stopped final fit at epoch {ep+1}")
                break

        if ep % 10 == 0:
            print(f"  ep{ep:2d}: train loss {epoch_loss:.4f}")
    print("  ✓ training finished")

  
    bins = np.logspace(np.log10(1), np.log10(51), NUM_CLASSES+1)
    cent = (bins[:-1] + bins[1:]) / 2

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(Xte.to(DEVICE)), 1).cpu().numpy()

    true   = yte.numpy()
    true_w = cent[true];  pred_w = cent[preds]

    acc  = accuracy_score(true, preds)
    mcc  = matthews_corrcoef(true, preds)
    srcc,_ = spearmanr(true_w, pred_w)
    mape = np.mean(np.abs((pred_w - true_w) / true_w)) * 100
    print(f"   acc {acc:.3f} srcc {srcc:.3f} mape {mape:.2f}% mcc {mcc:.3f}")


    metrics_rows.append({
        "distance_set": name,
        "ACC": acc,
        "SRCC": srcc,
        "MAPE": mape,
        "MCC": mcc,
        "best_lr": best_p["lr"],
        "best_weight_decay": best_p["weight_decay"],
        "CV_ACC": best_score,
    })

    per_frame_df = pd.DataFrame({
        "csv_file": test_df["csv_file"],
        "frame": test_df["frame"],
        "true_w": true_w,
        "pred_w": pred_w,
        "true_class": true,
        "pred_class": preds,
    })
    per_frame_df.to_csv(PLOT_DIR / "per_frame_predictions.csv", index=False)


    cm = confusion_matrix(true, preds, labels=list(range(NUM_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(NUM_CLASSES)],
                 columns=[f"pred_{i}" for i in range(NUM_CLASSES)]) \
        .to_csv(PLOT_DIR / "confusion_raw.csv")
    pd.DataFrame(cm_norm, index=[f"true_{i}" for i in range(NUM_CLASSES)],
                 columns=[f"pred_{i}" for i in range(NUM_CLASSES)]) \
        .to_csv(PLOT_DIR / "confusion_norm.csv")

    plt.figure(figsize=(6,5)); plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"CM – {name}"); plt.colorbar(); plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion.png", dpi=150); plt.close()


    plt.figure(figsize=(6,6))
    plt.plot([1,60], [1,60], "k--")
    plt.scatter(true_w, pred_w, s=10, alpha=.4, edgecolors="none")
    plt.xscale("log"); plt.yscale("log")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "frame_parity.png", dpi=150)
    plt.close()

modal_df = (per_frame_df.groupby("true_w")["pred_w"]
              .agg(lambda x: mode(x, keepdims=True).mode[0])
              .reset_index())
modal_df.to_csv(PLOT_DIR / "modal_parity_values.csv", index=False)

plt.figure(figsize=(6,6))
plt.plot([1,60],[1,60],"k--")
plt.scatter(modal_df.true_w, modal_df.pred_w, s=60, alpha=.8, edgecolors="none")
plt.xscale("log"); plt.yscale("log")
plt.tight_layout()
plt.savefig(PLOT_DIR / "modal_parity.png", dpi=150)
plt.close()

modal_parity[name] = modal_df
if modal_parity:
    plt.figure(figsize=(6,6))
    for lbl, md in modal_parity.items():
        plt.scatter(md.true_w, md.pred_w, label=lbl,
                    s=50, alpha=.7, edgecolors="none")
    plt.plot([1,60],[1,60],"k--")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("True (mg)"); plt.ylabel("Modal pred (mg)")
    plt.title("Modal parity – all sets (hard targets)")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "combined_modal_parity.png", dpi=150)
    plt.close()

pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "summary_metrics.csv", index=False)
print("\n✓ all distance sets completed – outputs in", OUT_DIR)
