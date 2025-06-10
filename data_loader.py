from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch


random.seed(0);  np.random.seed(0);  torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


ROOT      = Path(".")
TRAIN_DIR = ROOT /  "train"
TEST_DIR  = ROOT /"test"


DF_OUT_DIR = ROOT / "plots_multi_sets" / "dataframes"
DF_OUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_COVERAGE = 0.60



DISTANCE_GROUPS: dict[str, list[str] | list[list[str]] | str | None] = {
    "summed": [
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
        "an_1_r-an_1_l", "b_a_1-b_a_2", "b_a_2-b_a_3", "b_a_3-b_a_4", "b_a_4-b_a_5",
        "b_h-an_1_r", "b_h-an_1_l",
        "b_t-l_1_co_r", "b_t-l_1_co_l",
        "b_t-l_2_co_r", "b_t-l_2_co_l",
        "b_t-l_3_co_r", "b_t-l_3_co_l",

        "l_1_co_r-l_2_co_r", "l_1_co_l-l_2_co_l",
        "l_2_co_r-l_3_co_r", "l_2_co_l-l_3_co_l",

        "l_1_co_r-l_1_co_l", "l_2_co_r-l_2_co_l", "l_3_co_r-l_3_co_l"
    ],

    "all": "all", 
 #   "pca": "pca" 
}

from pathlib import Path

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
    wide["weight_mg"] = extract_weight(path.name)
    return wide.astype(np.float32, errors="ignore")

def build_dataset(folder: Path, distances) -> pd.DataFrame:
    parts = [load_csv_features(f, distances) for f in sorted(folder.glob("*.csv"))]
    non_empty = [p for p in parts if not p.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

for name, dist_spec in DISTANCE_GROUPS.items():
    print(f"Processing distance set: {name}")

    train_df = build_dataset(TRAIN_DIR, dist_spec)
    test_df  = build_dataset(TEST_DIR , dist_spec)

    if train_df.empty or test_df.empty:
        print("  No usable frames – skipped.")
        continue

    out_train = DF_OUT_DIR / f"train_{name}.csv"
    out_test  = DF_OUT_DIR / f"test_{name}.csv"

    train_df.to_csv(out_train, index=False)
    test_df.to_csv(out_test , index=False)

    print(f"   ✓ Saved {len(train_df):,} train rows  → {out_train.name}")
    print(f"   ✓ Saved {len(test_df):,} test  rows  → {out_test.name}")

print("\nAll sets done – dataframes are in:", DF_OUT_DIR)