#!/usr/bin/env python3
"""Generate synthetic DRAM SEM image pairs for Drift-Sense evaluation.

Usage:
    python generate_dataset.py --config configs/dram_config.yaml --num_pairs 50 --seed 42
    python generate_dataset.py --config configs/dram_config.yaml --num_pairs 10 --output_dir data/test_pairs

Outputs:
    data/pairs/{pair_id:04d}/reference.png  - 1000x1000 grayscale reference (100x)
    data/pairs/{pair_id:04d}/search.png     - 1000x1000 grayscale search (10x)
    data/manifest.csv                       - Full manifest with metadata
"""
import argparse
import os
import sys
import yaml
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.pair_builder import PairBuilder

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic DRAM SEM image pairs')
    parser.add_argument('--config', type=str, default='configs/dram_config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--num_pairs', type=int, default=None,
                        help='Number of pairs to generate (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with CLI args
    if args.num_pairs is not None:
        config['dataset']['num_pairs'] = args.num_pairs
    if args.output_dir is not None:
        config['dataset']['output_dir'] = args.output_dir
    if args.seed is not None:
        config['dataset']['seed'] = args.seed
    
    num_pairs = config['dataset']['num_pairs']
    output_dir = config['dataset']['output_dir']
    base_seed = config['dataset']['seed']
    manifest_file = config['dataset'].get('manifest_file', 'data/manifest.csv')
    
    # Get difficulty distribution
    diff_dist = config['dataset'].get('difficulty_distribution', 
                                       {'easy': 0.2, 'medium': 0.4, 'hard': 0.3, 'extreme': 0.1})
    
    print(f"\n{'='*60}")
    print(f"Drift-Sense Synthetic Dataset Generator")
    print(f"{'='*60}")
    print(f"Architecture: DRAM 6F²")
    print(f"Number of pairs: {num_pairs}")
    print(f"Output directory: {output_dir}")
    print(f"Base seed: {base_seed}")
    print(f"Difficulty distribution: {diff_dist}")
    print(f"{'='*60}\n")
    
    # Initialize builder
    builder = PairBuilder(config)
    
    # Assign difficulty levels based on distribution
    rng = np.random.default_rng(base_seed)
    difficulties = []
    for level, fraction in diff_dist.items():
        count = int(round(fraction * num_pairs))
        difficulties.extend([level] * count)
    # Pad or trim to exact num_pairs
    while len(difficulties) < num_pairs:
        difficulties.append('medium')
    difficulties = difficulties[:num_pairs]
    rng.shuffle(difficulties)
    
    # Generate pairs
    manifest_rows = []
    total_time = 0
    
    for i in tqdm(range(num_pairs), desc="Generating pairs"):
        pair_id = i + 1
        pair_seed = base_seed + pair_id
        difficulty = difficulties[i]
        
        start_time = time.perf_counter()
        result = builder.generate_pair(pair_id, pair_seed, difficulty)
        gen_time = time.perf_counter() - start_time
        total_time += gen_time
        
        # Save images
        pair_dir = os.path.join(output_dir, f"{pair_id:04d}")
        os.makedirs(pair_dir, exist_ok=True)
        
        ref_path = os.path.join(pair_dir, 'reference.png')
        search_path = os.path.join(pair_dir, 'search.png')
        
        import cv2
        cv2.imwrite(ref_path, result['reference'])
        cv2.imwrite(search_path, result['search'])
        
        # Save metadata JSON
        import json
        meta_path = os.path.join(pair_dir, 'metadata.json')
        # Convert numpy types for JSON serialization
        meta = {}
        for k, v in result['metadata'].items():
            if isinstance(v, (np.integer,)):
                meta[k] = int(v)
            elif isinstance(v, (np.floating,)):
                meta[k] = float(v)
            elif isinstance(v, np.ndarray):
                meta[k] = v.tolist()
            else:
                meta[k] = v
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Manifest row
        manifest_rows.append({
            'pair_id': pair_id,
            'ref_path': ref_path,
            'search_path': search_path,
            'gt_x': result['gt_x'],
            'gt_y': result['gt_y'],
            'scale': result['metadata']['scale'],
            'rotation_deg': result['metadata']['rotation_deg'],
            'difficulty': difficulty,
            'seed': pair_seed,
            'generation_time_s': round(gen_time, 3)
        })
    
    # Save manifest CSV
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
    df = pd.DataFrame(manifest_rows)
    df.to_csv(manifest_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Generation Complete!")
    print(f"{'='*60}")
    print(f"Total pairs: {num_pairs}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per pair: {total_time/num_pairs:.2f}s")
    print(f"Manifest saved to: {manifest_file}")
    print(f"Images saved to: {output_dir}/")
    print(f"\nDifficulty breakdown:")
    for level in ['easy', 'medium', 'hard', 'extreme']:
        count = difficulties.count(level)
        print(f"  {level}: {count} pairs")

if __name__ == '__main__':
    main()
