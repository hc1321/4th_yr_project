import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def select_three_points(image_path):

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,8))
    plt.imshow(image_rgb)
    plt.title("Click 3 points: (P1, P2 for reference line; P3 for parallel line)")
    plt.axis("off")
    pts = plt.ginput(3, timeout=0)
    plt.close()
    pts = np.array(pts, dtype=np.float32)
    return pts, image

def line_from_points(p1, p2):

    a = p2[1] - p1[1]
    b = p1[0] - p2[0]
    c = p2[0]*p1[1] - p1[0]*p2[1]
    return np.array([a, b, c], dtype=np.float32)

def extend_line_xy(x1, y1, x2, y2, img_w, img_h):
   
    if x1 == x2:
        return np.array([x1, 0], dtype=np.float32), np.array([x1, img_h], dtype=np.float32)
    
    m = (y2 - y1) / (x2 - x1)  
    c = y1 - m * x1 

    points = []
   
    y_left = c
    if 0 <= y_left <= img_h:
        points.append((0, y_left))
   
    y_right = m * img_w + c
    if 0 <= y_right <= img_h:
        points.append((img_w, y_right))
    
    if m != 0:
        x_top = -c / m
        if 0 <= x_top <= img_w:
            points.append((x_top, 0))

    if m != 0:
        x_bottom = (img_h - c) / m
        if 0 <= x_bottom <= img_w:
            points.append((x_bottom, img_h))
    if len(points) < 2:
        raise ValueError("Could not compute intersections for line extension.")
   
    max_dist = 0
    pt1_out, pt2_out = points[0], points[1]
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = np.hypot(points[i][0]-points[j][0], points[i][1]-points[j][1])
            if d > max_dist:
                max_dist = d
                pt1_out, pt2_out = points[i], points[j]
    return np.array(pt1_out, dtype=np.float32), np.array(pt2_out, dtype=np.float32)

def compute_perpendicular_distance(line, point):
    a, b, c = line
    x, y = point
    return np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

def draw_line(img, pt1, pt2, color, thickness=2):
    cv2.line(img, (int(round(pt1[0])), int(round(pt1[1]))),
                  (int(round(pt2[0])), int(round(pt2[1]))), color, thickness)

def main(image_path):

    pts, image = select_three_points(image_path)
    p1, p2, p3 = pts

    ref_line = line_from_points(p1, p2)
    
    pixel_distance = compute_perpendicular_distance(ref_line, p3)
    print(f"Measured perpendicular distance: {pixel_distance:.2f} px")
    known_distance_mm = float(input("Enter known perpendicular distance (in mm): "))
    mm_per_pixel = known_distance_mm / pixel_distance
    print(f"Conversion factor: {mm_per_pixel:.4f} mm/px")
    

    a, b, c = ref_line
    factor = (a*p3[0] + b*p3[1] + c) / (a**2 + b**2)
    foot = np.array([p3[0] - a*factor, p3[1] - b*factor])
    
    img_h, img_w, _ = image.shape
    ext_pt1, ext_pt2 = extend_line_xy(p1[0], p1[1], p2[0], p2[1], img_w, img_h)

    c2 = - (a*p3[0] + b*p3[1])
    par_pt1, par_pt2 = extend_line_xy(p3[0], p3[1],
                                      p3[0] + (p2[0]-p1[0]),
                                      p3[1] + (p2[1]-p1[1]),
                                      img_w, img_h)
    

    vis = image.copy()
    draw_line(vis, ext_pt1, ext_pt2, (255, 0, 0), thickness=2)  
    draw_line(vis, par_pt1, par_pt2, (0, 255, 0), thickness=2)    
    draw_line(vis, p3, foot, (0, 0, 255), thickness=2)             
    for pt in pts:
        cv2.circle(vis, (int(round(pt[0])), int(round(pt[1]))), 5, (0, 255, 255), -1)
    cv2.putText(vis, f"{pixel_distance*mm_per_pixel:.2f} mm", (int(round(foot[0])), int(round(foot[1])-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
    plt.figure(figsize=(10,8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Extended Lines from Clicked Points")
    plt.axis("off")
    plt.show()
    
    output_path = os.path.splitext(image_path)[0] + "_measurement.jpg"
    cv2.imwrite(output_path, vis)
    print(f"Measurement visualization saved as '{output_path}'")
    
    lines_data = {
        "image": os.path.basename(image_path),
        "conversion_factor": mm_per_pixel
    }
    json_path = os.path.splitext(image_path)[0] + "_lines.json"
    with open(json_path, "w") as f:
        json.dump(lines_data, f, indent=4)
    print(f"Line data saved to '{json_path}'")

if __name__ == "__main__":
    root = "Videos_for_overlaying/overlays"
    
    if not os.path.exists(root):
        print(f"[ERROR] Folder '{root}' not found.")
    else:
        for overlay in sorted(os.listdir(root)):
            overlay_path = os.path.join(root, overlay)
            if overlay.lower().endswith(('.jpg', '.jpeg', '.png')):  
                print(f"[INFO] Processing {overlay_path}")
                main(overlay_path)
