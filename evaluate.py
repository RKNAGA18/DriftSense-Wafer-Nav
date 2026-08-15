#!/usr/bin/env python3
"""Evaluate localization results with detailed failure analysis.

Usage:
    python evaluate.py --predictions results/predictions.csv --output_dir results/
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List
import platform

def classify_failure(row: pd.Series, threshold: float = 5.0) -> str:
    """Classify the type of failure for a single prediction."""
    if pd.isna(row['error_px']) or row['error_px'] <= threshold:
        return 'SUCCESS'
    
    if row.get('confidence', 1.0) < 0.05:
        return 'PERIODIC_AMBIGUITY'
    
    scale = row.get('scale', 10.0)
    if abs(scale - 10.0) > 0.5:
        return 'SCALE_MISMATCH'
        
    rot = row.get('rotation_deg', 0.0)
    if abs(rot) > 5.0:
        return 'ROTATION_SENSITIVITY'
        
    diff = str(row.get('difficulty', '')).lower()
    if diff in ['hard', 'extreme']:
        return 'NOISE_DEGRADATION'
        
    pred_x, pred_y = row.get('pred_x', 500), row.get('pred_y', 500)
    if pred_x < 70 or pred_x > 930 or pred_y < 70 or pred_y > 930:
        return 'EDGE_POSITION'
        
    return 'UNKNOWN'

def generate_ppt_summary(metrics: Dict, df: pd.DataFrame, threshold: float) -> str:
    """Generate a PPT-ready text block for Slide 9."""
    hardware = f"{platform.system()} {platform.machine()}"
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    
    total_pairs = len(df)
    failures = df[df['failure_type'] != 'SUCCESS']
    fail_counts = failures['failure_type'].value_counts()
    
    summary = f"""=== SLIDE 9: RESULTS SUMMARY ===
Dataset: {total_pairs} synthetic DRAM 6F² pairs
Python: {python_ver} | Hardware: {hardware}

Localization Accuracy:
  Pass @ 5px:  {metrics.get('pass_5', 0):.1f}%
  Pass @ 4px:  {metrics.get('pass_4', 0):.1f}%
  Pass @ 2px:  {metrics.get('pass_2', 0):.1f}%
  Pass @ 1px:  {metrics.get('pass_1', 0):.1f}%
  Pass @ 0.5px: {metrics.get('pass_0.5', 0):.1f}%

Error Statistics:
  Mean:   {metrics.get('mean_error', 0):.4f} px
  Median: {metrics.get('median_error', 0):.4f} px
  Worst:  {metrics.get('worst_error', 0):.4f} px

Runtime:
  Mean: {metrics.get('mean_runtime', 0):.1f} ms/pair
  Timing: time.perf_counter() (wall clock)

Failure Analysis:
  Total failures (>{threshold}px): {len(failures)} ({len(failures)/total_pairs*100:.1f}%)"""

    categories = ['PERIODIC_AMBIGUITY', 'NOISE_DEGRADATION', 'SCALE_MISMATCH', 'ROTATION_SENSITIVITY', 'EDGE_POSITION', 'UNKNOWN']
    for cat in categories:
        count = fail_counts.get(cat, 0)
        if count > 0:
            name = cat.replace('_', ' ').capitalize()
            summary += f"\n  - {name}: {count}"
            
    return summary

def main():
    parser = argparse.ArgumentParser(description='Evaluate localization results')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Path to predictions CSV')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for plots and reports')
    parser.add_argument('--failure_threshold', type=float, default=5.0,
                        help='Error threshold (px) above which a case is considered failed')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    fail_dir = os.path.join(args.output_dir, 'failure_cases')
    os.makedirs(fail_dir, exist_ok=True)
    
    df = pd.read_csv(args.predictions)
    
    errors = df['error_px'].dropna()
    metrics = {
        'mean_error': errors.mean(),
        'median_error': errors.median(),
        'worst_error': errors.max(),
        'mean_runtime': df['runtime_ms'].mean()
    }
    
    for thresh in [5, 4, 2, 1, 0.5]:
        metrics[f'pass_{thresh}'] = (errors <= thresh).mean() * 100
        
    df['failure_type'] = df.apply(lambda row: classify_failure(row, args.failure_threshold), axis=1)
    
    # Error histogram
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Localization Errors')
    plt.xlabel('Error (px)')
    plt.ylabel('Count')
    plt.axvline(args.failure_threshold, color='red', linestyle='dashed', linewidth=1, label=f'Threshold ({args.failure_threshold}px)')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'error_histogram.png'), bbox_inches='tight')
    plt.close()
    
    # Error vs Noise
    if 'difficulty' in df.columns:
        plt.figure(figsize=(8, 6))
        # Simple boxplot replacement without seaborn
        diff_levels = ['easy', 'medium', 'hard', 'extreme']
        data_to_plot = [df[df['difficulty'] == d]['error_px'].dropna() for d in diff_levels]
        plt.boxplot(data_to_plot, labels=diff_levels)
        plt.title('Error vs Difficulty Level')
        plt.ylabel('Error (px)')
        plt.savefig(os.path.join(args.output_dir, 'error_vs_noise.png'), bbox_inches='tight')
        plt.close()
        
    # Error vs Scale
    if 'scale' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['scale'], df['error_px'], alpha=0.6)
        plt.title('Error vs Scale')
        plt.xlabel('Scale Factor')
        plt.ylabel('Error (px)')
        plt.savefig(os.path.join(args.output_dir, 'error_vs_scale.png'), bbox_inches='tight')
        plt.close()
        
    # Error vs Rotation
    if 'rotation_deg' in df.columns:
        plt.figure(figsize=(8, 6))
        plt.scatter(df['rotation_deg'], df['error_px'], alpha=0.6)
        plt.title('Error vs Rotation Angle')
        plt.xlabel('Rotation (degrees)')
        plt.ylabel('Error (px)')
        plt.savefig(os.path.join(args.output_dir, 'error_vs_rotation.png'), bbox_inches='tight')
        plt.close()
        
    # Pass rates
    plt.figure(figsize=(8, 6))
    thresholds = [0.5, 1, 2, 4, 5]
    rates = [metrics[f'pass_{t}'] for t in thresholds]
    bars = plt.bar([str(t) for t in thresholds], rates, color='skyblue', edgecolor='black')
    plt.title('Pass Rates at Different Error Thresholds')
    plt.xlabel('Threshold (px)')
    plt.ylabel('Pass Rate (%)')
    plt.ylim(0, 100)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
    plt.savefig(os.path.join(args.output_dir, 'threshold_pass_rates.png'), bbox_inches='tight')
    plt.close()
    
    # Failure Pie Chart
    failures = df[df['failure_type'] != 'SUCCESS']
    if len(failures) > 0:
        plt.figure(figsize=(8, 8))
        fail_counts = failures['failure_type'].value_counts()
        plt.pie(fail_counts, labels=fail_counts.index, autopct='%1.1f%%', startangle=140)
        plt.title('Failure Analysis Categories')
        plt.savefig(os.path.join(args.output_dir, 'failure_analysis.png'), bbox_inches='tight')
        plt.close()
        
    # Worst Cases Visualization
    worst_cases = df.nlargest(5, 'error_px')
    for i, (_, row) in enumerate(worst_cases.iterrows()):
        if pd.isna(row.get('search_path')) or not os.path.exists(row['search_path']):
            continue
        search_img = cv2.imread(row['search_path'])
        if search_img is not None:
            gt_pt = (int(row['gt_x']), int(row['gt_y']))
            pred_pt = (int(row['pred_x']), int(row['pred_y']))
            
            cv2.circle(search_img, gt_pt, 15, (0, 255, 0), 2)
            
            length = 15
            cv2.line(search_img, (pred_pt[0]-length, pred_pt[1]-length), (pred_pt[0]+length, pred_pt[1]+length), (0, 0, 255), 2)
            cv2.line(search_img, (pred_pt[0]-length, pred_pt[1]+length), (pred_pt[0]+length, pred_pt[1]-length), (0, 0, 255), 2)
            
            cv2.putText(search_img, f"Error: {row['error_px']:.1f}px ({row['failure_type']})", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out_path = os.path.join(fail_dir, f"worst_{i+1}_pair{int(row['pair_id'])}.png")
            cv2.imwrite(out_path, search_img)

    summary = generate_ppt_summary(metrics, df, args.failure_threshold)
    with open(os.path.join(args.output_dir, 'ppt_summary.txt'), 'w') as f:
        f.write(summary)
        
    print(summary)
    print(f"\nEvaluation complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
