
from pathlib import Path
import re
import cv2           


ROOT_DIR   = Path(r"Cropped_Videos - USED IN DLC")  
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}            
SKIP_PAT   = re.compile(r"(?:[_\-]|^)side\d*$", re.IGNORECASE) 

def count_frames(video_path: Path) -> int:

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  Could not open {video_path}")
        return 0

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frames > 0:
        cap.release()
        return frames

    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    return frames

grand_total = 0

for vid in ROOT_DIR.rglob("*"):
    if vid.is_file() and vid.suffix.lower() in VIDEO_EXTS:
        if SKIP_PAT.search(vid.stem.lower()):
            continue

        n_frames = count_frames(vid)
        grand_total += n_frames
        print(f"{vid.relative_to(ROOT_DIR)}  →  {n_frames:,} frames")

print("\n" + "-" * 60)
print(f"TOTAL FRAMES (excluding *side* videos):  {grand_total:,}")
