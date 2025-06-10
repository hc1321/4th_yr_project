import cv2
import pandas as pd
import os
import glob

output_directory = "Cropped_Frames3`    "
os.makedirs(output_directory, exist_ok=True)  

dates = ["Original Videos\\22.07", "Original Videos\\23.07", "Original Videos\\24.07"]

for date in dates:
    date_path = date

    for frag in range(1, 30):
        frag_path = os.path.join(date_path, str(frag))

        if not os.path.exists(frag_path):
            print("Skipping:", frag_path)
            continue

        for view in ["top", "side"]:
            view_path = os.path.join(frag_path, view)
            if not os.path.exists(view_path):
                print("Skipping:", view_path)
                continue

            csv_file = glob.glob(os.path.join(view_path, "*.csv"))
            if not csv_file:
                print("No CSV found for:", view_path)
                continue

            csv_path = csv_file[0]

            video_file = glob.glob(os.path.join(view_path, "*.mov"))
            if not video_file:
                print("No video found for:", view_path)
                continue

            video_path = video_file[0]

            output_folder = os.path.join(output_directory, f"{date}_{frag}_{view}")
            os.makedirs(output_folder, exist_ok=True)

            try:
                df = pd.read_csv(csv_path, encoding="utf-8")
            except Exception as e:
                print(f"Error loading CSV {csv_path} with default delimiter: {e}")
                df = None

            #delimiter error (;)
            if df is not None and 'frame' not in df.columns and 'Frame' not in df.columns:
                try:
                    print("Retrying with ';' delimiter")
                    df = pd.read_csv(csv_path, delimiter=';', encoding="utf-8")
                except Exception as e:
                    print(f"Error loading CSV {csv_path} with ';' delimiter: {e}")
                    continue  
            

            if 'frame' in df.columns:
                df.rename(columns={'frame': 'frame_number'}, inplace=True)
            elif 'Frame' in df.columns:
                df.rename(columns={'Frame': 'frame_number'}, inplace=True)
            else:
                print("Error: 'frame' column not found in CSV. Skipping.")
                continue

            first_valid_frame = df['frame_number'].min()

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error opening video stream or file:", video_path)
                continue

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            frame_index = 1 
            bounding_box = 250  

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  
                
 
                if frame_index < first_valid_frame:
                    frame_index += 1
                    continue

                if frame_index in df['frame_number'].values:
                    point = df[df['frame_number'] == frame_index].iloc[0]
                    y = frame_height - int(point['y'])  
                    x, y = int(point['x']), y

                    x1 = x - bounding_box
                    y1 = y - bounding_box
                    x2 = x + bounding_box
                    y2 = y + bounding_box


                    if x1 < 0:
                        x2 -= x1  
                        x1 = 0
                    if y1 < 0:
                        y2 -= y1
                        y1 = 0
                    if x2 > frame_width:
                        x1 -= (x2 - frame_width)  
                        x2 = frame_width
                    if y2 > frame_height:
                        y1 -= (y2 - frame_height)  
                        y2 = frame_height
                    
                  #  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cropped_frame = frame[y1:y2, x1:x2]

                    if cropped_frame.shape[0] >= 500 and cropped_frame.shape[1] >= 500:
                        cropped_image_path = os.path.join(output_folder, f"frame_{frame_index}.jpg")
                        cv2.imwrite(cropped_image_path, cropped_frame)

                frame_index += 1

            cap.release()
            print("Saved cropped frames in", output_folder)

print("Batch processing completed.")
