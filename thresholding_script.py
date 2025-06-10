from pathlib import Path
import sys
import pandas as pd


REF_CSV          = Path("distances 0.9 threshold\\un thresholded csvs\\00019DLC_resnet101_ANT-POSE-MIXEDMar27shuffle1_1200000_filtered_TEST.csv")
REF_FRAME_LIST   = Path("data cleaning\\distances_00019_test.txt")
OUT_CSV          = Path("data cleaning\\cleaned\\00019_threshold_table_TRAIN.csv")
THRESH_PERCENT   = 0.35


FRAME_COL = "frame"
BONE_COL  = "bone"
DIST_COL  = "distance_mm"

def load_reference_frames(path: Path) -> list[str]:
    if not path.exists():
        sys.exit(f"Reference-frame list not found: {path}")
    frames = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not frames:
        sys.exit("Reference-frame list is empty")
    return frames

def make_thresholds(df: pd.DataFrame, frames: list[str], pct: float) -> pd.DataFrame:

    subset = df[df[FRAME_COL].isin(frames)].copy()
    subset = subset[subset[DIST_COL] != 0]                
    if subset.empty:
        sys.exit("No non-zero distances in the specified frames")

    means = subset.groupby(BONE_COL, sort=False)[DIST_COL].mean()
    return pd.DataFrame({
        "low" : means * (1 - pct),
        "high": means * (1 + pct),
    })

def main() -> None:

    if not REF_CSV.exists():
        sys.exit(f"Reference CSV not found: {REF_CSV}")
    df = pd.read_csv(REF_CSV, low_memory=False)


    need_cols = {FRAME_COL, BONE_COL, DIST_COL}
    if not need_cols.issubset(df.columns):
        missing = ", ".join(sorted(need_cols - set(df.columns)))
        sys.exit(f"Reference CSV is missing required columns: {missing}")


    frames = load_reference_frames(REF_FRAME_LIST)


    thresholds = make_thresholds(df, frames, THRESH_PERCENT)


    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    thresholds.reset_index().rename(columns={"index": "bone"}).to_csv(OUT_CSV, index=False)
    print(f"Threshold table written to: {OUT_CSV}  ({len(thresholds)} bones)")

if __name__ == "__main__":
    main()
