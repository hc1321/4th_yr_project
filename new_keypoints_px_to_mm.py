import math
import os
import pandas as pd

def distance_bet_keypoints(kp_data, idx1, idx2, conv_factor):
    x1, y1 = float(kp_data[idx1 * 3]), float(kp_data[idx1 * 3 + 1])
    x2, y2 = float(kp_data[idx2 * 3]), float(kp_data[idx2 * 3 + 1])
    dist_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist_px / conv_factor 

def process_video(input_csv, video_id, conv_factor):
    bodyparts = [
        "b_t", "b_a_1", "b_a_2", "b_a_3", "b_a_4", "b_a_5",
        "l_1_co_r", "l_1_tr_r", "l_1_fe_r", "l_1_ti_r", "l_1_ta_r", "l_1_pt_r",
        "l_2_co_r", "l_2_tr_r", "l_2_fe_r", "l_2_ti_r", "l_2_ta_r", "l_2_pt_r",
        "l_3_co_r", "l_3_tr_r", "l_3_fe_r", "l_3_ti_r", "l_3_ta_r", "l_3_pt_r",
        "l_1_co_l", "l_1_tr_l", "l_1_fe_l", "l_1_ti_l", "l_1_ta_l", "l_1_pt_l",
        "l_2_co_l", "l_2_tr_l", "l_2_fe_l", "l_2_ti_l", "l_2_ta_l", "l_2_pt_l",
        "l_3_co_l", "l_3_tr_l", "l_3_fe_l", "l_3_ti_l", "l_3_ta_l", "l_3_pt_l",
        "b_h", "an_1_r", "an_2_r", "an_3_r", "an_1_l", "an_2_l", "an_3_l"
    ]
    
    skeleton = [
        ["b_t", "b_a_1"], ["b_a_1", "b_a_2"], ["b_a_2", "b_a_3"], ["b_a_3", "b_a_4"], ["b_a_4", "b_a_5"],
        ["b_t", "l_1_co_r"], ["l_1_co_r", "l_1_tr_r"], ["l_1_tr_r", "l_1_fe_r"], ["l_1_fe_r", "l_1_ti_r"],
        ["l_1_ti_r", "l_1_ta_r"], ["l_1_ta_r", "l_1_pt_r"], ["l_1_co_r", "l_2_co_r"], ["l_2_co_r", "l_2_tr_r"],
        ["l_2_tr_r", "l_2_fe_r"], ["l_2_fe_r", "l_2_ti_r"], ["l_2_ti_r", "l_2_ta_r"], ["l_2_ta_r", "l_2_pt_r"],
        ["l_2_co_r", "l_3_co_r"], ["l_3_co_r", "l_3_tr_r"], ["l_3_tr_r", "l_3_fe_r"], ["l_3_fe_r", "l_3_ti_r"],
        ["l_3_ti_r", "l_3_ta_r"], ["l_3_ta_r", "l_3_pt_r"], ["b_t", "l_1_co_l"], ["l_1_co_l", "l_1_tr_l"],
        ["l_1_tr_l", "l_1_fe_l"], ["l_1_fe_l", "l_1_ti_l"], ["l_1_ti_l", "l_1_ta_l"], ["l_1_ta_l", "l_1_pt_l"],
        ["l_1_co_l", "l_2_co_l"], ["l_2_co_l", "l_2_tr_l"], ["l_2_tr_l", "l_2_fe_l"], ["l_2_fe_l", "l_2_ti_l"],
        ["l_2_ti_l", "l_2_ta_l"], ["l_2_ta_l", "l_2_pt_l"], ["l_2_co_l", "l_3_co_l"], ["l_3_co_l", "l_3_tr_l"],
        ["l_3_tr_l", "l_3_fe_l"], ["l_3_fe_l", "l_3_ti_l"], ["l_3_ti_l", "l_3_ta_l"], ["l_3_ta_l", "l_3_pt_l"],
        ["b_t", "b_h"], ["b_h", "an_1_r"], ["an_1_r", "an_2_r"], ["an_2_r", "an_3_r"], ["b_h", "an_1_l"],
        ["an_1_l", "an_2_l"], ["an_2_l", "an_3_l"]
    ]
    

    df = pd.read_csv(input_csv, header=[0,1,2], index_col=0)

    scorer = df.columns.levels[0][0]
    
    results = []
    
    for frame, row in df.iterrows():
        kp_data = []
        for bp in bodyparts:
            try:
                x = float(row[(scorer, bp, 'x')])
                y = float(row[(scorer, bp, 'y')])
                likelihood = float(row[(scorer, bp, 'likelihood')])
            except Exception as e:
                print(f"Warning: Missing data for bodypart '{bp}' in frame {frame} of video {video_id}. Error: {e}")
                x, y, likelihood = float('nan'), float('nan'), float('nan')
            kp_data.extend([x, y, likelihood])
        
        for bone in skeleton:
            bp1, bp2 = bone
            if bp1 not in bodyparts or bp2 not in bodyparts:
                print(f"Warning: {bp1} or {bp2} not found in bodyparts list for video {video_id}.")
                continue
            idx1 = bodyparts.index(bp1)
            idx2 = bodyparts.index(bp2)
            distance_mm = distance_bet_keypoints(kp_data, idx1, idx2, conv_factor)
            results.append({
                "video_id": video_id,
                "frame": frame,
                "bone": f"{bp1}-{bp2}",
                "distance_mm": distance_mm
            })
    
    return pd.DataFrame(results)

def main():
    input_folder = r"Keypoints_px2"      
    output_dataframes = r"keypoints_mm3"      
    video_mapping_path = r"data_paths.csv"    
    conversion_factors_path = r"conv_H_id_map.csv" 

    os.makedirs(output_dataframes, exist_ok=True)

    video_mapping = pd.read_csv(video_mapping_path)
    conversion_factors = pd.read_csv(conversion_factors_path).set_index("conversion_factor_id")

    for _, row in video_mapping.iterrows():
        video_id = str(row["video_id"])
        conv_factor_id = row["conversion_factor_id"]

        if conv_factor_id in conversion_factors.index:
            conv_factor = conversion_factors.loc[conv_factor_id, "conversion_factor"]
        else:
            print(f"Warning: Conversion factor ID {conv_factor_id} not found. Skipping video {video_id}.")
            continue
 
        keypoints_file = os.path.join(input_folder, f"{video_id}.csv")
        if not os.path.exists(keypoints_file):
            print(f"Warning: Keypoints file {keypoints_file} not found. Skipping video {video_id}.")
            continue

        df_distances = process_video(keypoints_file, video_id, conv_factor)

        output_file = os.path.join(output_dataframes, f"{video_id}.csv")
        df_distances.to_csv(output_file, index=False)
        print(f"Saved bone distances for video {video_id} to {output_file}")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
