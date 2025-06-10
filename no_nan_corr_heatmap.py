import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

KEYPOINTS = [
    "b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5",
    "l_1_co_r", "l_1_tr_r", "l_1_fe_r", "l_1_ti_r", "l_1_ta_r", "l_1_pt_r",
    "l_2_co_r", "l_2_tr_r", "l_2_fe_r", "l_2_ti_r", "l_2_ta_r", "l_2_pt_r",
    "l_3_co_r", "l_3_tr_r", "l_3_fe_r", "l_3_ti_r", "l_3_ta_r", "l_3_pt_r",
    "l_1_co_l", "l_1_tr_l", "l_1_fe_l", "l_1_ti_l", "l_1_ta_l", "l_1_pt_l",
    "l_2_co_l", "l_2_tr_l", "l_2_fe_l", "l_2_ti_l", "l_2_ta_r", "l_2_pt_l",
    "l_3_co_l", "l_3_tr_l", "l_3_fe_l", "l_3_ti_l", "l_3_ta_l", "l_3_pt_l",
    "b_h", "an_1_r", "an_2_r", "an_3_r", "an_1_l", "an_2_l", "an_3_l"
]

def load_and_concatenate_distance_csvs(distance_folder):
 
    print(f"Loading CSV files from folder: {distance_folder}")
    all_dfs = []
    files = glob.glob(os.path.join(distance_folder, "*.csv"))
    print(f"Found {len(files)} CSV files.")
    for csv_file in files:
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
            print(f"Loaded {csv_file} with shape {df.shape}")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    if all_dfs:
        concatenated = pd.concat(all_dfs, ignore_index=True)
        print(f"Concatenated DataFrame shape: {concatenated.shape}")
        return concatenated
    else:
        print("No CSVs loaded!")
        return pd.DataFrame()

def aggregate_by_video_and_bone(df):
   
    print("Aggregating data by video_id and bone")
    df_agg = df.groupby(["video_id", "bone"], as_index=False)["distance_mm"].mean()
    print(f"Aggregated DataFrame shape: {df_agg.shape}")
    return df_agg

def merge_with_weight(df_agg, weight_csv):
    
    print(f"Loading weight data from: {weight_csv}")
    weights_df = pd.read_csv(weight_csv)
    df_merged = pd.merge(df_agg, weights_df, on="video_id", how="left")
    print(f" Merged DataFrame shape: {df_merged.shape}")
    return df_merged

def compute_bone_weight_correlation_matrix(df_merged):
    
    n = len(KEYPOINTS)
    corr_matrix = np.full((n, n), np.nan)
    print("Computing bone-weight correlations for each keypoint pair...")

    for i in range(n):
        for j in range(i+1, n):
            bone_name = f"{KEYPOINTS[i]}-{KEYPOINTS[j]}"
            sub = df_merged[df_merged["bone"] == bone_name]
            if len(sub) >= 2:
                r = sub["distance_mm"].corr(sub["weight"])
            else:
                r = np.nan
            corr_matrix[i, j] = r
            corr_matrix[j, i] = r  
        if i % 5 == 0:
            print(f"Processed keypoint {i+1}/{n}")
    return pd.DataFrame(corr_matrix, index=KEYPOINTS, columns=KEYPOINTS)

def plot_heatmap(matrix, title, output_file):
   
    print(f"Creating heatmap: {title}")
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(matrix.index, fontsize=8)
    
    """ 
   
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", va="center", ha="center", fontsize=6,
                        color="white" if abs(val) > 0.5 else "black")"""
    plt.title(title, pad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"[PLOT] Heatmap saved to: {output_file}")

def main():
   
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    weight_csv = r"data_paths.csv"

    heatmap_output_base = r"bone_weight_heatmaps_pearson3"
    os.makedirs(heatmap_output_base, exist_ok=True)
    
    for thresh in thresholds:
        print(f"\n=== Processing threshold = {thresh} ===")
        distance_folder = f"keypoints_mm2_all_lengths_threshold3_{thresh}_modified"

        print(f" Loading distance data from: {distance_folder}")
        df_all = load_and_concatenate_distance_csvs(distance_folder)
        if df_all.empty:
            print(f" No distance data loaded for threshold {thresh}. Skipping.")
            continue
        

        zero_count = (df_all["distance_mm"] == 0).sum()
        print(f"Excluding {zero_count} rows where distance_mm == 0.")
        df_all = df_all[df_all["distance_mm"] != 0]
        if df_all.empty:
            print(f" After excluding zeros, no data remains for threshold {thresh}. Skipping.")
            continue
        
        print("Aggregating data by video_id and bone...")
        df_agg = aggregate_by_video_and_bone(df_all)
        
        print("Merging aggregated data with weight data...")
        df_merged = merge_with_weight(df_agg, weight_csv)

        print("Calculating most informative bones based on correlation with weight...")
        bone_corrs = (
            df_merged.groupby("bone")[["distance_mm", "weight"]]
            .apply(lambda df: df["distance_mm"].corr(df["weight"]))
            .dropna()
        )
   
        top_10 = bone_corrs.abs().sort_values(ascending=False).head(15)
        top_10_signed = bone_corrs.loc[top_10.index] 

        plt.figure(figsize=(10, 5))
        top_10_signed.plot(kind='bar', color='skyblue')
        plt.title(f"10 most informative{thresh})")
        plt.ylabel("Pearson Correlation")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()

        info_bar_path = os.path.join(heatmap_output_base, f"top10_bones_corr_{thresh}.png")
        plt.savefig(info_bar_path, dpi=150)
        plt.close()
        print(f"Top 10 informative bones saved to: {info_bar_path}")


        print(" Computing 49x49 bone-weight correlation matrix...")
        corr_df = compute_bone_weight_correlation_matrix(df_merged)
        print(f"Correlation matrix shape: {corr_df.shape}")
        
        title = f"Bone-Weight Correlation Matrix (Threshold={thresh})"
        output_file = os.path.join(heatmap_output_base, f"bone_weight_corr_matrix_{thresh}.png")
        plot_heatmap(corr_df, title, output_file)
    
    print("All thresholds processed. Bone-weight correlation heatmaps generated.")

if __name__ == "__main__":
    main()
