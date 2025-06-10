
from pathlib import Path
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "data cleaning/00010DLC_resnet101_ANT-POSE-MIXEDMar27shuffle1_1200000_filtered_TRAIN_thresholded2.csv"
)

DISTANCES = [
    "b_t-b_a_1", "b_a_1-b_a_2", "b_a_2-b_a_3", "b_a_3-b_a_4", "b_a_4-b_a_5",
    "b_t-l_1_co_r", "l_1_co_r-l_1_tr_r", "l_1_tr_r-l_1_fe_r", "l_1_fe_r-l_1_ti_r",
    "l_1_ti_r-l_1_ta_r", "l_1_ta_r-l_1_pt_r", "l_1_co_r-l_2_co_r", "l_2_co_r-l_2_tr_r",
    "l_2_tr_r-l_2_fe_r", "l_2_fe_r-l_2_ti_r", "l_2_ti_r-l_2_ta_r", "l_2_ta_r-l_2_pt_r",
    "l_2_co_r-l_3_co_r", "l_3_co_r-l_3_tr_r", "l_3_tr_r-l_3_fe_r", "l_3_fe_r-l_3_ti_r",
    "l_3_ti_r-l_3_ta_r", "l_3_ta_r-l_3_pt_r", "b_t-l_1_co_l", "l_1_co_l-l_1_tr_l",
    "l_1_tr_l-l_1_fe_l", "l_1_fe_l-l_1_ti_l", "l_1_ti_l-l_1_ta_l", "l_1_ta_l-l_1_pt_l",
    "l_1_co_l-l_2_co_l", "l_2_co_l-l_2_tr_l", "l_2_tr_l-l_2_fe_l", "l_2_fe_l-l_2_ti_l",
    "l_2_ti_l-l_2_ta_l", "l_2_ta_l-l_2_pt_l", "l_2_co_l-l_3_co_l", "l_3_co_l-l_3_tr_l",
    "l_3_tr_l-l_3_fe_l", "l_3_fe_l-l_3_ti_l", "l_3_ti_l-l_3_ta_l", "l_3_ta_l-l_3_pt_l",
    "b_t-b_h", "b_h-an_1_r", "an_1_r-an_2_r", "an_2_r-an_3_r", "b_h-an_1_l",
    "an_1_l-an_2_l", "an_2_l-an_3_l"
]

MIN_COVERAGE = 0.65 



def count_present_per_frame(csv_path: Path) -> pd.Series:

    df = pd.read_csv(csv_path)


    df = df[df["bone"].isin(DISTANCES)].copy()


    df["distance_mm_mask"] = pd.to_numeric(
        df["distance_mm_mask"], errors="coerce"
    ).fillna(0)

    counts = (
        df.assign(is_present=df["distance_mm_mask"] > 0)
          .groupby("frame")["is_present"]
          .sum()
          .astype(int)
    )

    return counts.sort_index()


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    counts = count_present_per_frame(CSV_PATH)

    total_expected = len(DISTANCES)
    threshold = int(np.ceil(total_expected * MIN_COVERAGE))

    passing = counts[counts >= threshold]

    print(f"Checked file          : {CSV_PATH}")
    print(f"Total frames examined : {len(counts):,}")
    print(
        f"Frames with ≥ {MIN_COVERAGE*100:.0f}% "
        f"({threshold}/{total_expected}) of listed distances present: {len(passing):,}"
    )
    print() 

    if passing.empty:
        print("(no frames meet the threshold)")
    else:
        print("frame,n_present")
        for frame, n in passing.items():

            print(f"{frame},{n}")


if __name__ == "__main__":
    main()
