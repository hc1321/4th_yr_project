import cv2 
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def create_folder_overlay(video_files, method='max', output_size=None):
    if not video_files:
        print("[WARNING] No videos found in folder.")
        return None
    
    first_valid_frame = None
    
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            first_valid_frame = frame
            break
    
    if first_valid_frame is None:
        print("[ERROR] No valid frames found in videos.")
        return None
    
    if output_size is not None:
        first_valid_frame = cv2.resize(first_valid_frame, output_size)
    else:
        output_size = (first_valid_frame.shape[1], first_valid_frame.shape[0])
    
    height, width, channels = first_valid_frame.shape
    
    if method == 'max':
        accum = np.zeros((height, width, channels), dtype=np.uint8)
    else:  # 'avg'
        accum = np.zeros((height, width, channels), dtype=np.float32)
        count = 0
    
    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        while ret:
            if output_size is not None:
                frame = cv2.resize(frame, output_size)
            if method == 'max':
                accum = np.maximum(accum, frame)
            else:  # 'avg'
                accum += frame.astype(np.float32)
                count += 1
            ret, frame = cap.read()
        cap.release()
    
    if method == 'avg' and count > 0:
        accum /= count
        accum = accum.astype(np.uint8)
    
    return accum

def process_all_folders(base_path, method='max'):
    
    overlay_dir = os.path.join(base_path, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)
    
    for folder_name in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        video_files = glob.glob(os.path.join(folder_path, "*.mov"))
        if not video_files:
            print(f"[WARNING] No videos found in {folder_name}, skipping.")
            continue
        
        print(f"[INFO] Processing folder '{folder_name}' with {len(video_files)} videos.")
        overlay = create_folder_overlay(video_files, method=method)
        
        if overlay is not None:
            output_path = os.path.join(overlay_dir, f"{folder_name}_overlay.jpeg")
            cv2.imwrite(output_path, overlay)
            print(f"[INFO] Saved {output_path}")
            
            plt.figure(figsize=(10, 6))
            plt.title(f"Folder {folder_name}: Composite ({method} projection)")
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

def main():
    base_path = "Videos_for_overlaying"  # Change if needed
    if not os.path.exists(base_path):
        print(f"[ERROR] Base directory '{base_path}' not found.")
        return
    process_all_folders(base_path, method='max')

if __name__ == "__main__":
    main()