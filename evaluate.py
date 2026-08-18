import pandas as pd
import numpy as np
import subprocess
import time
import re
import argparse
import sys

def evaluate(manifest_path):
    df = pd.read_csv(manifest_path)
    errors = []
    runtimes = []
    
    print("Evaluating Pipeline...")
    for idx, row in df.iterrows():
        start = time.time()
        res = subprocess.run(["python", "localize.py", row['ref_path'], row['search_path']], capture_output=True, text=True)
        t_ms = (time.time() - start) * 1000
        
        # Guardrail: Print exact crash logs if the script fails
        if res.returncode != 0 or not res.stdout.strip():
            print(f"\n[CRASH LOG] Script Failed on Pair {idx+1}")
            print(f"STDERR: {res.stderr}")
            sys.exit(1)
            
        coords = re.findall(r"[-+]?\d*\.\d+|\d+", res.stdout)
        px, py = float(coords[0]), float(coords[1])
        
        err = np.sqrt((px - row['gt_x'])**2 + (py - row['gt_y'])**2)
        errors.append(err)
        runtimes.append(t_ms)
        print(f"[{idx+1}/{len(df)}] Error: {err:.4f} px | Time: {t_ms:.1f} ms")
        
    errors = np.array(errors)
    print("\n=== FINAL METRICS ===")
    print(f"Pass @ 5px: {np.mean(errors <= 5.0)*100:.1f}%")
    print(f"Pass @ 1px: {np.mean(errors <= 1.0)*100:.1f}%")
    print(f"Sub-pixel (<1px): {np.mean(errors < 1.0)*100:.1f}%")
    print(f"Mean Error: {np.mean(errors):.4f} px")
    print(f"Worst-Case Error: {np.max(errors):.4f} px")
    print(f"Mean Runtime: {np.mean(runtimes):.1f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    evaluate(args.manifest)
