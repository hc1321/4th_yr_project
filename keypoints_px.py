### 
## THIS HAS TO BE RUN IN DLC CONDA ENVIRONMENT


import os
import glob
import time
import pandas as pd
import deeplabcut

def process_videos_from_mapping(mapping_csv, config_path, video_dir, output_csv_folder, wait_time=20):
    
    os.makedirs(output_csv_folder, exist_ok=True)
    

    mapping = pd.read_csv(mapping_csv)
    
    for video_id in mapping["video_id"]:
        video_id = str(video_id).strip()  
        video_path = os.path.join(video_dir, video_id + ".mp4")
        
        if not os.path.exists(video_path):
            print(f"Video file not found for video_id {video_id} at {video_path}. Skipping.")
            continue
        
        print(f"Processing video {video_id} at {video_path}")

        deeplabcut.analyze_videos(config_path, [video_path], save_as_csv=True)

        #csv saving time
        time.sleep(wait_time)
        

        csv_pattern = os.path.join(video_dir, video_id + "*csv")
        csv_files = glob.glob(csv_pattern)
        
        if csv_files:
            original_csv = csv_files[0]
            desired_csv = os.path.join(output_csv_folder, video_id + ".csv")
            try:
                os.rename(original_csv, desired_csv)
                print(f"CSV for video {video_id} moved to: {desired_csv}")
            except Exception as e:
                print(f"Error moving CSV for video {video_id}: {e}")
        else:
            print(f"No CSV file found for video {video_id}.")

if __name__ == "__main__":
    
    mapping_csv = r"data_paths.csv"           
    config_path = r"dlc_new\\config.yaml"
    video_dir = r"Cropped Videos"
    output_csv_folder = r"Keypoints_px2"   
    
    process_videos_from_mapping(mapping_csv, config_path, video_dir, output_csv_folder, wait_time=20)
