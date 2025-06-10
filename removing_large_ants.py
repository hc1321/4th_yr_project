
from pathlib import Path
import sys
import pandas as pd


LEARN_CSV        = Path("data cleaning\\00010DLC_resnet101_ANT-POSE-MIXEDMar27shuffle1_1200000_filtered.csv")
REF_FRAME_LIST   = Path("data cleaning\\distances_00010_test.txt")
THRESHOLDS_PATH  = Path("data cleaning\\00010_threshold_table.csv")
THRESH_PERCENT   = 0.35     


TARGET_CSVS = [
    Path("data cleaning\\00010DLC_resnet101_ANT-POSE-MIXEDMar27shuffle1_1200000_filtered_TRAIN.csv"),
   

]

FRAME_COL = "frame"
BONE_COL  = "bone"
DIST_COL  = "distance_mm"
MASK_COL  = "distance_mm_mask"   

def load_reference_frames(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"Reference‑frame list not found: {path}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def learn_thresholds(df: pd.DataFrame, ref_frames: list[str], pct: float) -> pd.DataFrame:
    """Return DataFrame indexed by bone with columns low & high."""
    ref = df[df[FRAME_COL].isin(ref_frames)].copy()
    ref = ref[ref[DIST_COL] != 0]  
    if ref.empty:
        sys.exit(" No non‑zero distances found in reference frames")
    means = ref.groupby(BONE_COL)[DIST_COL].mean()
    return pd.DataFrame({
        "low" : means * (1 - pct),
        "high": means * (1 + pct),
    })


def apply_thresholds(df: pd.DataFrame, thresh: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return a *new* DataFrame with thresholds applied + count of zeroed values."""
    df2 = df.merge(thresh, left_on=BONE_COL, right_index=True, how="left")
    out_of_band = (df2[DIST_COL] < df2["low"]) | (df2[DIST_COL] > df2["high"])
    df2.loc[out_of_band, DIST_COL] = 0
    if MASK_COL in df2.columns:
        df2.loc[out_of_band, MASK_COL] = 0
    df2.drop(columns=["low", "high"], inplace=True)
    return df2, int(out_of_band.sum())


def output_path(src: Path) -> Path:
    """Return <stem>_thresholded.csv next to *src*."""
    return src.with_name(src.stem + "_thresholded.csv")


def main() -> None:
    if not LEARN_CSV.exists():
        sys.exit(f"Learning CSV not found: {LEARN_CSV}")


    if THRESHOLDS_PATH.exists():
        thresh = pd.read_csv(THRESHOLDS_PATH).set_index("bone")
        print(f"→ Loaded thresholds from {THRESHOLDS_PATH}")
    else:
        ref_frames = load_reference_frames(REF_FRAME_LIST)
        if not ref_frames:
            sys.exit("Reference‑frame list is empty")
        df_learn_raw = pd.read_csv(LEARN_CSV, low_memory=False)
        need_cols = {FRAME_COL, BONE_COL, DIST_COL}
        if not need_cols.issubset(df_learn_raw.columns):
            sys.exit("Learning CSV is missing required columns: frame, bone, distance_mm")
        thresh = learn_thresholds(df_learn_raw, ref_frames, THRESH_PERCENT)
        THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        thresh.reset_index().to_csv(THRESHOLDS_PATH, index=False)
        print(f"→ Saved thresholds to {THRESHOLDS_PATH}")

        df_learn, changed = apply_thresholds(df_learn_raw, thresh)
        out_path = output_path(LEARN_CSV)
        df_learn.to_csv(out_path, index=False)
        print(f"   Filtered learning CSV   : {out_path}  (zeroed {changed} values)")


    for path in TARGET_CSVS:
        if not path.exists():
            print(f"Skipping – file not found: {path}")
            continue
        df_raw = pd.read_csv(path, low_memory=False)
        df_filtered, changed = apply_thresholds(df_raw, thresh.copy())
        out_path = output_path(path)
        df_filtered.to_csv(out_path, index=False)
        print(f"   Filtered target CSV    : {out_path}  (zeroed {changed} values)")

    print("All done!")


if __name__ == "__main__":
    main()
