import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def select_two_points(image_path): 
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    pts = plt.ginput(2, timeout=0)  
    plt.close()
    pts = np.array(pts, dtype=np.float32)
    return pts, image

def compute_distance(pt1, pt2):
    return np.linalg.norm(pt2 - pt1)

def draw_line(image, pt1, pt2, color=(0, 0, 255), thickness=2):
    cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, thickness)

def main(image_known, json_known, image_unknown):

    with open(json_known, "r") as f:
        data = json.load(f)
    
    if "conversion_factor" not in data:
        print(f" No conversion factor found in {json_known}")
        return
    
    known_conversion = data["conversion_factor"]

    print(f"Select two points on the known image ({image_known})")
    pts_known, img_known = select_two_points(image_known)


    pixel_distance_known = compute_distance(pts_known[0], pts_known[1])

    real_world_distance = pixel_distance_known * known_conversion

    print(f"[INFO] Select the same two points on the unknown image ({image_unknown})")
    pts_unknown, img_unknown = select_two_points(image_unknown)

    pixel_distance_unknown = compute_distance(pts_unknown[0], pts_unknown[1])

    new_conversion = real_world_distance / pixel_distance_unknown

    print(f"\n### Conversion Calculation ###")
    print(f"Known Image: {image_known}")
    print(f" - Pixel Distance: {pixel_distance_known:.2f} px")
    print(f" - Known Conversion: {known_conversion:.6f} mm/px")
    print(f" - Real-World Distance: {real_world_distance:.2f} mm")

    print(f"\nUnknown Image: {image_unknown}")
    print(f" - Pixel Distance: {pixel_distance_unknown:.2f} px")
    print(f" - Computed Conversion: {new_conversion:.6f} mm/px")

    new_json_path = os.path.splitext(image_unknown)[0] + "_lines.json"
    new_data = {"image": os.path.basename(image_unknown), "conversion_factor": new_conversion}

    with open(new_json_path, "w") as f:
        json.dump(new_data, f, indent=4)
    
    print(f"\nConversion factor saved to {new_json_path}")

 
    draw_line(img_known, pts_known[0], pts_known[1], color=(255, 0, 0))  
    draw_line(img_unknown, pts_unknown[0], pts_unknown[1], color=(0, 255, 0))  


    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_known, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_unknown, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()
if __name__ == "__main__":
    
    image_known = "Videos_for_overlaying\\overlays\\23_overlay.jpeg"  # Image with known conversion
    json_known = "Videos_for_overlaying\\overlays\\23_overlay_lines.json"  # JSON with known conversion
    image_unknown = "Videos_for_overlaying\\overlays\\todo\\24_overlay.jpeg"  # Image to compute conversion

    main(image_known, json_known, image_unknown)
