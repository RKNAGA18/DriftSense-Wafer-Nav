import csv
import time
import numpy as np
from localize import localize_pair

MANIFEST_PATH = "data/manifest.csv"

def main():
    print("🚀 Running Verified Evaluation Pipeline...")
    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    errors, times = [], []
    
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        pred_x, pred_y, angle, score = localize_pair(row["ref_path"], row["search_path"])
        dt = time.perf_counter() - t0
        times.append(dt)

        gt_x, gt_y = float(row["gt_x"]), float(row["gt_y"])
        err = float(np.hypot(pred_x - gt_x, pred_y - gt_y))
        errors.append(err)
        
        print(f"[{i+1:3d}/{len(rows)}] Error: {err:6.4f} px | Time: {dt*1000:6.1f} ms")

    errors = np.array(errors)
    times = np.array(times)
    
    print("\n" + "="*40)
    print(f"📊 FINAL EVALUATION REPORT ({len(rows)} Pairs)")
    print("="*40)
    print(f"Mean Error      : {np.mean(errors):.3f} px")
    print(f"Median Error    : {np.median(errors):.3f} px")
    print(f"Worst-case Error: {np.max(errors):.3f} px")
    print(f"Pass @ 1px      : {np.mean(errors <= 1.0)*100:.1f}%")
    print(f"Mean Time       : {np.mean(times)*1000:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()
