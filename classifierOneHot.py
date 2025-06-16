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

ROOT       = Path(".")
DF_OUT_DIR = ROOT / "dataframes" 
OUT_DIR    = ROOT / "plots_class"
OUT_DIR.mkdir(exist_ok=True)

MIN_COVERAGE = 0.60
NUM_CLASSES  = 20
MAX_EPOCHS   = 70
PATIENCE     = 5
DELTA        = 1e-4
PARAM_GRID   = {"lr": [1e-3, 5e-4], "weight_decay": [1e-4, 1e-5]}
KFOLD        = KFold(n_splits=5, shuffle=True, random_state=0)

centers = np.logspace(np.log10(1.0), np.log10(50.0), NUM_CLASSES)
edges = np.zeros(NUM_CLASSES + 1)
edges[1:-1] = (centers[:-1] + centers[1:]) / 2
edges[0]      = centers[0]**2 / edges[1]
edges[-1]     = centers[-1]**2 / edges[-2]

def weight_to_class(w: float) -> int:
    return np.digitize(w, edges[1:-1])

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

for train_csv in sorted(DF_OUT_DIR.glob("train_*.csv")):
    set_name = train_csv.stem.replace("train_", "")
    test_csv = DF_OUT_DIR / f"test_{set_name}.csv"
    print(f"\n=== {set_name.upper()} ===")

    if not test_csv.exists():
        print("  missing test – skipping."); continue

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)


    feature_prefixes = ("dist_", "group_sum_", "pca_")
    feats = sorted(
        set(c for c in train_df.columns if c.startswith(feature_prefixes)) |
        set(c for c in test_df.columns  if c.startswith(feature_prefixes))
    )
    if not feats:
        print("  no feature columns found – skipping."); continue

    train_df = train_df.reindex(columns=["frame","weight_mg"] + feats, fill_value=0)
    test_df  = test_df .reindex(columns=["frame","weight_mg"] + feats, fill_value=0)


    mu = train_df[feats].mean(0)
    sd = train_df[feats].std(0) + 1e-6
    train_df[feats] = (train_df[feats] - mu) / sd
    test_df [feats] = (test_df [feats] - mu) / sd


    Xtr = torch.tensor(train_df[feats].values, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(test_df [feats].values, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor([weight_to_class(w) for w in train_df.weight_mg],
                       dtype=torch.long, device=DEVICE)
    yte = torch.tensor([weight_to_class(w) for w in test_df .weight_mg],
                       dtype=torch.long, device=DEVICE)

    best_p, best_score = {}, -1.0
    for p in ParameterGrid(PARAM_GRID):
        print(f"  testing params {p}...")
        fold_mcc = []
        fold = 1
        for tr_idx, va_idx in KFOLD.split(Xtr.cpu()):
            mdl = MLP(len(feats)).to(DEVICE)
            opt = torch.optim.Adam(mdl.parameters(), **p)

            crit = nn.CrossEntropyLoss()

            loader = DataLoader(TensorDataset(Xtr[tr_idx], ytr[tr_idx]),
                                batch_size=128, shuffle=True)
            best_loss, best_mcc, wait = float("inf"), -1.0, 0
            for ep in range(MAX_EPOCHS):
                mdl.train()
                for xb, yb in loader:
                    loss = crit(mdl(xb), yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                mdl.eval()
                with torch.no_grad():
                    logits   = mdl(Xtr[va_idx])
                    val_loss = crit(logits, ytr[va_idx]).item()
                    preds    = torch.argmax(logits,1).cpu().numpy()
                    val_mcc  = matthews_corrcoef(ytr[va_idx].cpu().numpy(), preds)
                if val_loss < best_loss - DELTA:
                    best_loss, wait = val_loss, 0
                else:
                    wait += 1
                    if wait >= PATIENCE: break
                best_mcc = max(best_mcc, val_mcc)
            stop_epoch = ep + 1
            print(f"    fold {fold}: , loss {best_loss}, stopped at epoch {stop_epoch:02d} | CV-MCC {best_mcc:.4f}")
            fold_mcc.append(best_mcc)
            fold += 1

        mean_mcc = float(np.mean(fold_mcc))
        if mean_mcc > best_score:
            best_score, best_p = mean_mcc, p
    print(f"best params {best_p} (CV MCC {best_score:.4f})")

    model = MLP(len(feats)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), **best_p)
    crit  = nn.CrossEntropyLoss()  
    loader= DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)

    best_loss, wait = float("inf"), 0
    for ep in range(MAX_EPOCHS):
        model.train(); total=0.0
        for xb, yb in loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*xb.size(0)
        epoch_loss = total/len(loader.dataset)
        if epoch_loss < best_loss - DELTA:
            best_loss, wait = epoch_loss, 0
        else:
            wait +=1
            if wait >= PATIENCE: break
    print(" ✓ training done")

    model.eval()
    with torch.no_grad():
        logits = model(Xte)
    preds = torch.argmax(logits,1).cpu().numpy()
    true  = yte.cpu().numpy()

    true_w = centers[true]
    pred_w = centers[preds]

    acc   = accuracy_score(true, preds)
    mcc   = matthews_corrcoef(true, preds)
    srcc, _ = spearmanr(true_w, pred_w)
    mape  = np.mean(np.abs((pred_w - true_w)/true_w))*100
    print(f"   ACC {acc:.3f} | SRCC {srcc:.3f} | MAPE {mape:.2f}% | MCC {mcc:.3f}")

    metrics_rows.append({
        "distance_set": set_name, "ACC": acc, "SRCC": srcc,
        "MAPE": mape, "MCC": mcc,
        "best_lr": best_p["lr"], "best_weight_decay": best_p["weight_decay"],
        "CV_MCC": best_score
    })

    outd = OUT_DIR / set_name; outd.mkdir(exist_ok=True)
    pd.DataFrame({
        "frame":      test_df["frame"],
        "true_w":     true_w,
        "pred_w":     pred_w,
        "true_class": true,
        "pred_class": preds
    }).to_csv(outd/"per_frame_predictions.csv", index=False)

    cm     = confusion_matrix(true, preds, labels=list(range(NUM_CLASSES)))
    cm_norm= cm/cm.sum(axis=1,keepdims=True)
    pd.DataFrame(cm_norm).to_csv(outd/"confusion_norm.csv", index=False)
    plt.figure(figsize=(6,5))
    plt.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.title(f"CM – {set_name}")
    plt.colorbar(); plt.tight_layout()
    plt.savefig(outd/"confusion.png", dpi=150); plt.close()

pd.DataFrame(metrics_rows).to_csv(OUT_DIR/"summary_metrics.csv", index=False)
print("\n✓ finished – results in", OUT_DIR)
