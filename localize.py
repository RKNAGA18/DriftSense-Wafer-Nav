#!/usr/bin/env python3
"""Localize reference patterns in search images.

Usage:
    # Single pair
    python localize.py --ref data/pairs/0001/reference.png --search data/pairs/0001/search.png
    
    # Batch mode from manifest
    python localize.py --manifest data/manifest.csv --output_dir results/

Output:
    Predicted (x, y) center coordinates in search-image pixels.
    Origin (0,0) is top-left; x increases right, y increases downward.
"""
import argparse
import os
import sys
import time
import yaml
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.template_matcher import TemplateMatcher

def main():
    parser = argparse.ArgumentParser(description='Localize reference patterns in search images')
    
    # Single pair mode
    parser.add_argument('--ref', type=str, help='Path to reference image')
    parser.add_argument('--search', type=str, help='Path to search image')
    
    # Batch mode
    parser.add_argument('--manifest', type=str, help='Path to manifest CSV for batch processing')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for predictions')
    
    # Config
    parser.add_argument('--config', type=str, default='configs/dram_config.yaml',
                        help='Path to configuration YAML')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    matcher = TemplateMatcher(config)
    
    if args.ref and args.search:
        # Single pair mode
        ref = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
        
        if ref is None or search is None:
            print(f"Error: Could not load images")
            sys.exit(1)
        
        start = time.perf_counter()
        result = matcher.localize(ref, search)
        elapsed = time.perf_counter() - start
        
        print(f"\nLocalization Result:")
        print(f"  Predicted center: ({result['x']:.4f}, {result['y']:.4f})")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Best scale: {result['best_scale']:.2f}")
        print(f"  Best angle: {result['best_angle']:.2f}°")
        print(f"  Correlation peak: {result['correlation_peak']:.4f}")
        print(f"  Candidates found: {result['num_candidates']}")
        print(f"  Runtime: {elapsed*1000:.1f} ms")
        
    elif args.manifest:
        # Batch mode
        df = pd.read_csv(args.manifest)
        os.makedirs(args.output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Drift-Sense Batch Localization")
        print(f"{'='*60}")
        print(f"Pairs to process: {len(df)}")
        print(f"Output: {args.output_dir}")
        print(f"{'='*60}\n")
        
        predictions = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Localizing"):
            ref = cv2.imread(row['ref_path'], cv2.IMREAD_GRAYSCALE)
            search = cv2.imread(row['search_path'], cv2.IMREAD_GRAYSCALE)
            
            if ref is None or search is None:
                print(f"Warning: Could not load pair {row['pair_id']}")
                predictions.append({
                    'pair_id': row['pair_id'],
                    'gt_x': row['gt_x'],
                    'gt_y': row['gt_y'],
                    'pred_x': np.nan,
                    'pred_y': np.nan,
                    'confidence': 0.0,
                    'runtime_ms': 0.0,
                    'error_px': np.nan
                })
                continue
            
            start = time.perf_counter()
            result = matcher.localize(ref, search)
            elapsed = time.perf_counter() - start
            
            error = np.sqrt((result['x'] - row['gt_x'])**2 + 
                          (result['y'] - row['gt_y'])**2)
            
            predictions.append({
                'pair_id': row['pair_id'],
                'ref_path': row['ref_path'],
                'search_path': row['search_path'],
                'gt_x': row['gt_x'],
                'gt_y': row['gt_y'],
                'pred_x': round(result['x'], 4),
                'pred_y': round(result['y'], 4),
                'confidence': round(result['confidence'], 4),
                'best_scale': result['best_scale'],
                'best_angle': result['best_angle'],
                'correlation_peak': round(result['correlation_peak'], 4),
                'num_candidates': result['num_candidates'],
                'runtime_ms': round(elapsed * 1000, 1),
                'error_px': round(error, 4),
                'difficulty': row.get('difficulty', 'unknown'),
                'scale': row.get('scale', 10.0),
                'rotation_deg': row.get('rotation_deg', 0.0)
            })
        
        # Save predictions CSV (contains BOTH ground truth AND predictions)
        pred_df = pd.DataFrame(predictions)
        pred_path = os.path.join(args.output_dir, 'predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        
        # Print summary
        errors = pred_df['error_px'].dropna()
        print(f"\n{'='*60}")
        print(f"Batch Localization Complete")
        print(f"{'='*60}")
        print(f"Mean error: {errors.mean():.4f} px")
        print(f"Median error: {errors.median():.4f} px")
        print(f"Worst error: {errors.max():.4f} px")
        print(f"Mean runtime: {pred_df['runtime_ms'].mean():.1f} ms")
        for thresh in [5, 4, 2, 1, 0.5]:
            rate = (errors <= thresh).mean() * 100
            print(f"Pass rate @ {thresh}px: {rate:.1f}%")
        print(f"\nPredictions saved to: {pred_path}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
