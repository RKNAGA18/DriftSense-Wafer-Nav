import numpy as np
import cv2
import os
import csv
import argparse

def apply_sem_noise(image, is_search=False):
    noise_level = 65.0 if is_search else 15.0
    noisy = np.random.poisson(image / 255.0 * noise_level) / noise_level * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def generate_wafer_base(size=12000):
    base = np.full((size, size), 100, dtype=np.uint8)
    
    # Physically Accurate 60px Pitch in Search Image
    pitch_x, pitch_y = 600, 600 
    for i in range(0, size, pitch_x):
        w = 200 + np.random.randint(-20, 30)
        base[:, i:i+w] = 180
    for i in range(0, size, pitch_y):
        w = 200 + np.random.randint(-20, 30)
        base[i:i+w, :] = 60
                
    for _ in range(500):
        lx, ly = np.random.randint(0, size), np.random.randint(0, size)
        lw, lh = np.random.randint(100, 400), np.random.randint(100, 400)
        cv2.rectangle(base, (lx, ly), (lx+lw, ly+lh), int(np.random.randint(50, 200)), -1)
    return base

def create_pair(index, out_dir):
    base = generate_wafer_base(12000)
    scale = 10.0  
    rot_deg = np.random.uniform(-2.0, 2.0)
    
    search_fov = int(1000 * scale)
    search_start_x, search_start_y = 1000, 1000 
    
    search_crop = base[search_start_y:search_start_y+search_fov, search_start_x:search_start_x+search_fov]
    search_img = cv2.resize(search_crop, (1000, 1000), interpolation=cv2.INTER_AREA)
    search_img = apply_sem_noise(search_img, True)
    
    drift_x_search_px = np.random.uniform(-30.0, 30.0)
    drift_y_search_px = np.random.uniform(-30.0, 30.0)
    
    gt_search_x = 500.0 + drift_x_search_px
    gt_search_y = 500.0 + drift_y_search_px
    
    ref_start_x = search_start_x + int(gt_search_x * scale) - 500
    ref_start_y = search_start_y + int(gt_search_y * scale) - 500
    
    ref_img = base[ref_start_y:ref_start_y+1000, ref_start_x:ref_start_x+1000]
    M = cv2.getRotationMatrix2D((500, 500), rot_deg, 1.0)
    ref_img = cv2.warpAffine(ref_img, M, (1000, 1000), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ref_img = apply_sem_noise(ref_img, False)
    
    pair_dir = os.path.join(out_dir, f"{index:04d}")
    os.makedirs(pair_dir, exist_ok=True)
    cv2.imwrite(os.path.join(pair_dir, "search.png"), search_img)
    cv2.imwrite(os.path.join(pair_dir, "reference.png"), ref_img)
    
    return gt_search_x, gt_search_y, scale, rot_deg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=100)
    args = parser.parse_args()
    
    out_dir = "data"
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "search_path", "ref_path", "gt_x", "gt_y", "scale", "rotation"])
        for i in range(args.num_pairs):
            x, y, sc, rot = create_pair(i, os.path.join(out_dir, "pairs"))
            s_path = os.path.join(out_dir, "pairs", f"{i:04d}", "search.png")
            r_path = os.path.join(out_dir, "pairs", f"{i:04d}", "reference.png")
            writer.writerow([f"{i:04d}", s_path, r_path, x, y, sc, rot])
