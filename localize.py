import csv
import time
import numpy as np
import cv2

MANIFEST_PATH = "data/manifest.csv"
PRED_PATH = "results/predictions.csv"
TEMPLATE_SIZE = 100          
TIE_BREAK_RATIO = 0.95       
CENTER = np.array([500.0, 500.0])

COARSE_ANGLES = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
FINE_HALF_RANGE = 0.4
FINE_STEP = 0.2

def rotate_template(template, angle_deg):
    if angle_deg == 0.0: return template
    h, w = template.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(template, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def parabolic_subpixel(corr, row, col):
    h, w = corr.shape
    r, c = min(max(row, 1), h - 2), min(max(col, 1), w - 2)

    cx, cy, cz = corr[r, c - 1], corr[r, c], corr[r, c + 1]
    denom_x = (cx - 2.0 * cy + cz)
    dx = 0.5 * (cx - cz) / denom_x if abs(denom_x) > 1e-9 else 0.0

    cx2, cz2 = corr[r - 1, c], corr[r + 1, c]
    denom_y = (cx2 - 2.0 * cy + cz2)
    dy = 0.5 * (cx2 - cz2) / denom_y if abs(denom_y) > 1e-9 else 0.0

    return col + float(np.clip(dx, -1.0, 1.0)), row + float(np.clip(dy, -1.0, 1.0))

def localize_pair(ref_path, search_path):
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    template0 = cv2.resize(ref, (TEMPLATE_SIZE, TEMPLATE_SIZE), interpolation=cv2.INTER_AREA)
    
    # Center Crop for Speed
    OFFSET = 300
    search_crop = search[OFFSET:700, OFFSET:700]

    best_angle, best_max, best_corr = 0.0, -2.0, None
    for ang in COARSE_ANGLES:
        tmpl = rotate_template(template0, ang)
        corr = cv2.matchTemplate(search_crop, tmpl, cv2.TM_CCOEFF_NORMED)
        m = float(corr.max())
        if m > best_max:
            best_max, best_angle, best_corr = m, ang, corr

    fine_angles = np.arange(best_angle - FINE_HALF_RANGE, best_angle + FINE_HALF_RANGE + 1e-9, FINE_STEP)
    for ang in fine_angles:
        tmpl = rotate_template(template0, float(ang))
        corr = cv2.matchTemplate(search_crop, tmpl, cv2.TM_CCOEFF_NORMED)
        m = float(corr.max())
        if m > best_max:
            best_max, best_angle, best_corr = m, float(ang), corr

    # THE TRUE AMAT RULE: NMS + Phase-Noise Buffer -> Pure Distance Sort
    max_val = float(best_corr.max())
    dilated = cv2.dilate(best_corr, np.ones((5, 5), np.float32))
    mask = (best_corr >= (dilated - 1e-6)) & (best_corr >= TIE_BREAK_RATIO * max_val)
    ys, xs = np.where(mask)

    best_dist, best_center, best_score = None, None, None
    half = TEMPLATE_SIZE / 2.0
    
    for row, col in zip(ys, xs):
        sub_col, sub_row = parabolic_subpixel(best_corr, row, col)
        
        cx = sub_col + half + OFFSET
        cy = sub_row + half + OFFSET
        dist = float(np.hypot(cx - CENTER[0], cy - CENTER[1]))
        
        # Pure Distance Tie-Breaker (No arbitrary weights)
        if best_dist is None or dist < best_dist:
            best_dist, best_center, best_score = dist, np.array([cx, cy]), best_corr[row, col]

    if best_center is None:
        return 500.0, 500.0, best_angle, 0.0
        
    return float(best_center[0]), float(best_center[1]), best_angle, float(best_score)

def main():
    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows, times, errors = [], [], []
    
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        pred_x, pred_y, angle, score = localize_pair(row["ref_path"], row["search_path"])
        dt = time.perf_counter() - t0
        times.append(dt)

        out_rows.append({
            **row, "pred_x": repr(pred_x), "pred_y": repr(pred_y), "inference_time_s": repr(dt),
        })
        
        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        err = ((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2) ** 0.5
        errors.append(err)
        
        print(f"[{i+1:2d}/{len(rows)}] pred=({pred_x:8.4f},{pred_y:8.4f}) "
              f"gt=({gt_x:8.4f},{gt_y:8.4f}) err={err:6.3f}px "
              f"angle={angle:+.2f}deg score={score:.4f} t={dt*1000:.1f}ms")

    with open(PRED_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
        
    errors = np.array(errors)
    print(f"\nMean Error: {np.mean(errors):.3f} px")
    print(f"Pass @ 1px: {np.mean(errors <= 1.0)*100:.1f}%")
    print(f"Mean inference time: {np.mean(times)*1000:.2f} ms")

if __name__ == "__main__":
    main()
