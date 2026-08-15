"""Image pair orchestrator for Drift-Sense synthetic dataset generation.

Generates matched (reference, search) image pairs with exact ground-truth
coordinates for cross-magnification localization evaluation.

COORDINATE CONVENTION:
    Origin (0, 0) is TOP-LEFT.
    X increases to the RIGHT.
    Y increases DOWNWARD.
    All ground-truth coordinates are in SEARCH-IMAGE pixels.
"""

import numpy as np
import cv2
import os
import csv
import json
from typing import Dict, Tuple, List, Optional
from .dram_generator import DRAMGenerator
from .sem_noise import SEMNoisePipeline


class PairBuilder:
    """Orchestrates generation of matched reference/search image pairs.

    Efficient approach (no massive intermediate array):
    1. Generate the DRAM pattern directly at search-image resolution (1000×1000)
       with feature size F_search (small features → wide field of view)
    2. Pick a random target location in the search image
    3. Generate the SAME pattern region at reference-image resolution (1000×1000)
       with feature size F_ref = F_search × scale (large features → zoomed in)
    4. Apply different SEM noise levels to each
    5. Record exact ground-truth coordinates

    The key insight: the reference image covers a physical area that maps to
    approximately (1000/scale) × (1000/scale) ≈ 100×100 pixels in the search
    image. Both images share the same underlying DRAM pattern via phase offsets.
    """

    def __init__(self, config: Dict):
        """Initialize with full config dict."""
        self.config = config
        self.dram_gen = DRAMGenerator(config.get('architecture', {}))

        # Image settings
        self.img_size = config.get('image', {}).get('size', 1000)

        # Scale settings
        scale_cfg = config.get('scale', {})
        self.nominal_scale = scale_cfg.get('nominal', 10.0)
        self.scale_min = scale_cfg.get('min', 9.0)
        self.scale_max = scale_cfg.get('max', 11.0)
        self.scale_std = scale_cfg.get('jitter_std', 0.3)

        # Rotation settings
        rot_cfg = config.get('rotation', {})
        self.rot_min = rot_cfg.get('min_deg', -2.0)
        self.rot_max = rot_cfg.get('max_deg', 2.0)
        self.rot_std = rot_cfg.get('jitter_std_deg', 0.5)

        # Dataset settings
        ds_cfg = config.get('dataset', {})
        self.edge_margin = ds_cfg.get('edge_margin_px', 60)
        self.search_noise_mult = config.get('search_noise_multiplier', 1.5)
        self.ref_noise_mult = config.get('reference_noise_multiplier', 0.5)

        # Base feature size for search image (pixels)
        # F_search should be small enough to show many repeating cells
        # Typical: 4-8 pixels for F at 10× (search) magnification
        self.f_search = config.get('architecture', {}).get('f_search_pixels', 6.0)

    def _build_noise_pipeline(self, difficulty: str) -> SEMNoisePipeline:
        """Build a SEMNoisePipeline from a difficulty preset.

        Maps flat preset keys to the nested config structure expected by
        SEMNoisePipeline.apply_all().
        """
        noise_cfg = self.config.get('noise', {})
        preset = noise_cfg.get('difficulty_presets', {}).get(difficulty, {})

        pipeline_cfg = {
            'shot_noise': {'dose': preset.get('shot_noise_dose', 100.0)},
            'blur': {
                'sigma_u': preset.get('blur_sigma_u', 1.0),
                'sigma_v': preset.get('blur_sigma_v', 1.0),
                'angle_deg': preset.get('blur_angle_deg', 0.0)
            },
            'charging': {
                'alpha': preset.get('charging_alpha', 0.0),
                'beta': preset.get('charging_beta', 0.0)
            },
            'jitter': {'sigma': preset.get('jitter_sigma', 0.0)},
            'drift': {
                'vx': preset.get('drift_vx', 0.0),
                'vy': preset.get('drift_vy', 0.0)
            },
            'mains_hum': {'amplitude': preset.get('mains_amplitude', 0.0)},
            'contrast_gamma': {
                'alpha': preset.get('contrast_alpha', 1.0),
                'beta': preset.get('contrast_beta', 0.0),
                'gamma': preset.get('contrast_gamma', 1.0)
            },
            'salt_pepper': {'density': preset.get('salt_pepper_density', 0.0)}
        }
        return SEMNoisePipeline(pipeline_cfg)

    def generate_pair(self, pair_id: int, seed: int,
                      difficulty: str = 'medium') -> Dict:
        """Generate a single (reference, search) image pair.

        Args:
            pair_id: Unique pair identifier
            seed: Random seed for reproducibility
            difficulty: Noise difficulty preset ('easy', 'medium', 'hard', 'extreme')

        Returns:
            Dict with keys:
                'reference': np.ndarray (1000x1000 uint8)
                'search': np.ndarray (1000x1000 uint8)
                'gt_x': float (ground truth x in search-image pixels)
                'gt_y': float (ground truth y in search-image pixels)
                'metadata': Dict with all generation parameters
        """
        rng = np.random.default_rng(seed)

        # 1. Sample scale and rotation
        actual_scale = float(np.clip(
            rng.normal(self.nominal_scale, self.scale_std),
            self.scale_min, self.scale_max
        ))
        actual_rotation = float(np.clip(
            rng.normal(0, self.rot_std),
            self.rot_min, self.rot_max
        ))

        # 2. Random phase offsets for pattern alignment
        phase_x = float(rng.random())
        phase_y = float(rng.random())

        # 3. Pick target location in search image
        # Since DRAM is highly periodic and the algorithm uses a tie-breaker 
        # to pick the match closest to the center, the ground truth must also 
        # represent the cell closest to the center to be solvable.
        f_search = self.f_search
        bl_pitch = self.dram_gen.bl_pitch_mult * f_search
        wl_pitch = self.dram_gen.wl_pitch_mult * f_search
        
        # Pick a target within +/- half a pitch from the exact center
        center_x = self.img_size / 2.0
        center_y = self.img_size / 2.0
        gt_x = float(rng.uniform(center_x - bl_pitch/2, center_x + bl_pitch/2))
        gt_y = float(rng.uniform(center_y - wl_pitch/2, center_y + wl_pitch/2))
        
        # Build noise pipeline for this difficulty
        pipeline = self._build_noise_pipeline(difficulty)

        # 4. Generate SEARCH image at 1000×1000
        f_search = self.f_search
        search_clean = self.dram_gen.generate_array(
            self.img_size, self.img_size, seed,
            f_pixels=f_search,
            phase_x=phase_x,
            phase_y=phase_y
        )
        # Apply heavy noise
        search_noisy = pipeline.apply_all(
            search_clean,
            noise_multiplier=self.search_noise_mult,
            rng=rng
        )
        search_img = np.clip(search_noisy * 255, 0, 255).astype(np.uint8)

        # 5. Generate REFERENCE image at 1000×1000 (zoomed into target)
        # The reference covers a (1000/scale)×(1000/scale) ≈ 100×100 pixel
        # region of the search image, rendered at 1000×1000 resolution.
        #
        # Phase alignment: the target center (gt_x, gt_y) in search coords
        # must correspond to the center of the reference image.
        # The search pattern has its phase defined by (phase_x, phase_y).
        # The reference must use shifted phases so the same structures appear.
        f_ref = f_search * actual_scale  # ~60 for scale=10

        # Compute phase shift: target position as fraction of pattern period
        wl_pitch_search = self.dram_gen.wl_pitch_mult * f_search
        bl_pitch_search = self.dram_gen.bl_pitch_mult * f_search

        # The target center in the search image at (gt_x, gt_y)
        # In terms of pattern phase, the reference must show patterns
        # that are offset by (gt_x - img_center) pattern periods
        ref_center_offset_x = gt_x - self.img_size / 2.0
        ref_center_offset_y = gt_y - self.img_size / 2.0

        # Convert pixel offset to phase offset (fraction of one period)
        ref_phase_x = phase_x + ref_center_offset_x / bl_pitch_search
        ref_phase_y = phase_y + ref_center_offset_y / wl_pitch_search

        ref_clean = self.dram_gen.generate_array(
            self.img_size, self.img_size, seed + 1_000_000,
            f_pixels=f_ref,
            phase_x=ref_phase_x % 1.0,
            phase_y=ref_phase_y % 1.0
        )

        # Apply light noise
        ref_noisy = pipeline.apply_all(
            ref_clean,
            noise_multiplier=self.ref_noise_mult,
            rng=np.random.default_rng(seed + 2_000_000)
        )

        # Apply rotation
        ref_img = np.clip(ref_noisy * 255, 0, 255).astype(np.uint8)
        if abs(actual_rotation) > 0.01:
            center = (self.img_size / 2.0, self.img_size / 2.0)
            M = cv2.getRotationMatrix2D(center, actual_rotation, 1.0)
            ref_img = cv2.warpAffine(
                ref_img, M, (self.img_size, self.img_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

        # 6. Verify coordinate convention
        # Origin (0,0) = top-left, x→right, y→down, search-image pixels
        assert 0 <= gt_x < self.img_size, f"gt_x={gt_x} out of bounds"
        assert 0 <= gt_y < self.img_size, f"gt_y={gt_y} out of bounds"

        metadata = {
            'pair_id': pair_id,
            'seed': seed,
            'architecture': 'DRAM_6F2',
            'difficulty': difficulty,
            'scale': actual_scale,
            'rotation_deg': actual_rotation,
            'f_search_pixels': f_search,
            'f_ref_pixels': f_ref,
            'phase_x': phase_x,
            'phase_y': phase_y,
            'noise_params': {
                'search_mult': self.search_noise_mult,
                'ref_mult': self.ref_noise_mult,
                'difficulty': difficulty
            },
            'gt_x': gt_x,
            'gt_y': gt_y,
            'coordinate_convention': 'top-left origin, x-right, y-down',
            'image_size': self.img_size,
        }

        return {
            'reference': ref_img,
            'search': search_img,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'metadata': metadata
        }

    def generate_batch(self, num_pairs: int, base_seed: int,
                       output_dir: str) -> str:
        """Generate multiple pairs, save images and manifest CSV.

        Args:
            num_pairs: Number of pairs to generate
            base_seed: Base random seed
            output_dir: Output directory path

        Returns:
            Path to the manifest CSV file
        """
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, 'manifest.csv')

        rng = np.random.default_rng(base_seed)

        ds_cfg = self.config.get('dataset', {})
        diff_dist = ds_cfg.get('difficulty_distribution',
                               {'easy': 0.2, 'medium': 0.4,
                                'hard': 0.3, 'extreme': 0.1})
        diffs = list(diff_dist.keys())
        probs = np.array([diff_dist[d] for d in diffs], dtype=float)
        probs = probs / probs.sum()

        manifest_data = []

        for i in range(num_pairs):
            pair_id = i + 1
            seed = int(rng.integers(0, 2**31))
            difficulty = str(rng.choice(diffs, p=probs))

            result = self.generate_pair(pair_id, seed, difficulty)

            pair_dir = os.path.join(output_dir, f"{pair_id:04d}")
            os.makedirs(pair_dir, exist_ok=True)

            ref_path = os.path.join(pair_dir, 'reference.png')
            search_path = os.path.join(pair_dir, 'search.png')

            cv2.imwrite(ref_path, result['reference'])
            cv2.imwrite(search_path, result['search'])

            # Save per-pair metadata
            meta_path = os.path.join(pair_dir, 'metadata.json')
            meta_serializable = {}
            for k, v in result['metadata'].items():
                if isinstance(v, (np.integer,)):
                    meta_serializable[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    meta_serializable[k] = float(v)
                elif isinstance(v, np.ndarray):
                    meta_serializable[k] = v.tolist()
                else:
                    meta_serializable[k] = v
            with open(meta_path, 'w') as f:
                json.dump(meta_serializable, f, indent=2)

            manifest_data.append({
                'pair_id': pair_id,
                'ref_path': ref_path,
                'search_path': search_path,
                'gt_x': result['gt_x'],
                'gt_y': result['gt_y'],
                'scale': result['metadata']['scale'],
                'rotation_deg': result['metadata']['rotation_deg'],
                'difficulty': difficulty,
                'seed': seed
            })

        # Write manifest CSV
        with open(manifest_path, 'w', newline='') as f:
            fieldnames = ['pair_id', 'ref_path', 'search_path',
                          'gt_x', 'gt_y', 'scale', 'rotation_deg',
                          'difficulty', 'seed']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in manifest_data:
                writer.writerow(row)

        return manifest_path
