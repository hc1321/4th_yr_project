import os, random, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import KFold, ParameterGrid
from scipy.stats import mode, spearmanr

random.seed(0); np.random.seed(0); torch.manual_seed(0)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT       = Path('.')
DF_OUT_DIR = ROOT / 'data_frames'
OUT_DIR    = ROOT / 'plots_class_csv'
OUT_DIR.mkdir(parents=True, exist_ok=True)


NUM_CLASSES  = 20
SIGMA        = 4.0
MIN_COVERAGE = 0.6
MAX_EPOCHS   = 70
PATIENCE     = 5
DELTA        = 1e-4
PARAM_GRID   = {'lr': [1e-3, 5e-4], 'weight_decay': [1e-4, 1e-5]}
KFOLD        = KFold(n_splits=5, shuffle=True, random_state=0)

centers = np.logspace(np.log10(1.0), np.log10(50.0), NUM_CLASSES)
edges = np.zeros(NUM_CLASSES + 1)
edges[1:-1] = (centers[:-1] + centers[1:]) / 2
edges[0] = centers[0]**2 / edges[1]
edges[-1] = centers[-1]**2 / edges[-2]

def weight_to_class(w: float) -> int:
    return np.digitize(w, edges[1:-1], right=False)

def crals_vec(c: int, k: int = NUM_CLASSES, s: float = SIGMA) -> np.ndarray:
    idx = np.arange(k)
    v = np.exp(-0.5 * ((idx - c) / s) ** 2)
    return v / v.sum()


class MLP(nn.Module):
    def __init__(self, din: int, k: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, k)
        )
    def forward(self, x):
        return self.net(x)

metrics_rows: List[Dict[str, Any]] = []
modal_data: Dict[str, pd.DataFrame] = {}

for train_csv in sorted(DF_OUT_DIR.glob('train_*.csv')):
    set_name = train_csv.stem.replace('train_', '')
    test_csv = DF_OUT_DIR / f'test_{set_name}.csv'
    print(f"\n=== {set_name.upper()} ===")

    if not train_csv.exists() or not test_csv.exists():
        print("  missing train or test CSV – skipping.")
        continue

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    feature_prefixes = ('dist_', 'group_sum_', 'pca_')
    feat_cols = sorted(
        set(c for c in train_df.columns if c.startswith(feature_prefixes)) |
        set(c for c in test_df.columns  if c.startswith(feature_prefixes))
    )

    if not feat_cols:
        print("  no feature columns found – skipping.")
        continue

    train_df = train_df.reindex(columns=['frame', 'weight_mg'] + feat_cols, fill_value=0)
    test_df  = test_df.reindex(columns=['frame', 'weight_mg'] + feat_cols, fill_value=0)

    mu = train_df[feat_cols].mean(0)
    sd = train_df[feat_cols].std(0) + 1e-6    
    train_df[feat_cols] = (train_df[feat_cols] - mu) / sd
    test_df [feat_cols] = (test_df [feat_cols] - mu) / sd

    Xtr = train_df[feat_cols].to_numpy(np.float32)
    Xte = test_df [feat_cols].to_numpy(np.float32)

    ytr_idx = np.array([weight_to_class(w) for w in train_df.weight_mg])
    yte_idx = np.array([weight_to_class(w) for w in test_df.weight_mg])
    ytr_soft = np.vstack([crals_vec(i) for i in ytr_idx]).astype(np.float32)

    XtrT = torch.tensor(Xtr, device=DEVICE)
    ytrT = torch.tensor(ytr_soft, device=DEVICE)
    XteT = torch.tensor(Xte, device=DEVICE)


    best_params, best_val = None, -1.0
    for p in ParameterGrid(PARAM_GRID):
        print(f"  testing params {p}...")
        fold_scores = []
        fold = 1
        for tri, vai in KFOLD.split(XtrT.cpu()):
            mdl = MLP(len(feat_cols)).to(DEVICE)
            opt = torch.optim.Adam(
                mdl.parameters(),
                lr=p['lr'],
                weight_decay=p['weight_decay']
            )
            crit = nn.KLDivLoss(reduction='batchmean')
            loader = DataLoader(
                TensorDataset(XtrT[tri], ytrT[tri]),
                batch_size=128,
                shuffle=True
            )

            best_val_loss, cnt = float('inf'), 0
            best_fold_mcc = -1.0

            for ep in range(MAX_EPOCHS):
                mdl.train()
                for xb, yb in loader:
                    logp = F.log_softmax(mdl(xb), dim=1)
                    loss = crit(logp, yb)
                    opt.zero_grad(); loss.backward(); opt.step()

                mdl.eval()
                with torch.no_grad():
                    logp_val = F.log_softmax(mdl(XtrT[vai]), dim=1)
                    val_loss = crit(logp_val, ytrT[vai]).item()
                    preds    = torch.argmax(logp_val, dim=1).cpu().numpy()
                    mcc       = matthews_corrcoef(ytr_idx[vai], preds)

                if val_loss < best_val_loss - DELTA:
                    best_val_loss, cnt = val_loss, 0
                else:
                    cnt += 1
                    if cnt >= PATIENCE:
                        break

                best_fold_mcc = max(best_fold_mcc, mcc)
            stop_epoch = ep + 1
            print(f"    fold {fold}: , loss {best_val_loss}, stopped at epoch {stop_epoch:02d} | CV-MCC {best_fold_mcc:.4f}")
            fold_scores.append(best_fold_mcc)
            fold += 1

        mean_mcc = float(np.mean(fold_scores))
        if mean_mcc > best_val:
            best_val, best_params = mean_mcc, p

    print(f" → Best params {best_params} (mean MCC {best_val:.4f})")

    model = MLP(len(feat_cols)).to(DEVICE)
    opt   = torch.optim.Adam(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    crit  = nn.KLDivLoss(reduction='batchmean')
    loader= DataLoader(TensorDataset(XtrT, ytrT), batch_size=128, shuffle=True)

    best_loss, cnt = float('inf'), 0
    for ep in range(MAX_EPOCHS):
        model.train(); total = 0.0
        for xb, yb in loader:
            logp = F.log_softmax(model(xb), dim=1)
            loss = crit(logp, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * xb.size(0)

        epoch_loss = total / len(loader.dataset)
        if epoch_loss < best_loss - DELTA:
            best_loss, cnt = epoch_loss, 0
        else:
            cnt += 1
            if cnt >= PATIENCE:
                break

    print("training done")

    model.eval()
    with torch.no_grad():
        logits = model(XteT).cpu()
    pred_classes = torch.argmax(logits, dim=1).numpy()
    true_classes = yte_idx

    true_w = centers[true_classes]
    pred_w = centers[pred_classes]

    acc   = accuracy_score(true_classes, pred_classes)
    mcc   = matthews_corrcoef(true_classes, pred_classes)
    srcc, _ = spearmanr(true_w, pred_w)
    mape  = np.mean(np.abs((pred_w - true_w) / true_w)) * 100
    print(f"   ACC {acc:.3f} | SRCC {srcc:.3f} | MAPE {mape:.2f}% | MCC {mcc:.3f}")

    metrics_rows.append({
        'distance_set': set_name, 'ACC': acc, 'SRCC': srcc,
        'MAPE': mape, 'MCC': mcc,
        'best_lr': best_params['lr'], 'best_weight_decay': best_params['weight_decay'],
        'best_cv_KL': best_val
    })

    PLOT_DIR = OUT_DIR / set_name
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        'frame':      test_df['frame'],
        'true_w':     true_w,
        'pred_w':     pred_w,
        'true_class': true_classes,
        'pred_class': pred_classes
    })
    pred_df.to_csv(PLOT_DIR / 'per_frame_predictions.csv', index=False)

    cm = confusion_matrix(true_classes, pred_classes, labels=list(range(NUM_CLASSES)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    pd.DataFrame(cm_norm).to_csv(PLOT_DIR / 'confusion_norm.csv', index=False)
    plt.figure(figsize=(6,5)); plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.title(f"CM – {set_name}"); plt.colorbar(); plt.tight_layout()
    plt.savefig(PLOT_DIR / 'confusion.png', dpi=150); plt.close()

summary_df = pd.DataFrame(metrics_rows)
summary_df.to_csv(OUT_DIR / 'summary_metrics.csv', index=False)
print("\n✓ finished – results in", OUT_DIR)
