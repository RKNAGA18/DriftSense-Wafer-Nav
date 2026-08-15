"""DRAM 6F² geometry engine for synthetic SEM image generation.

Generates realistic top-down DRAM array layouts at any resolution by
parameterizing feature sizes in pixel units. All structures are based on
published 6F² DRAM architecture specifications.

References:
    - IEEE IRDS 2022/2023 "More Moore" chapters (pitch/scaling rules)
    - TechInsights DRAM cell architecture analysis (layout geometry)
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List


class DRAMGenerator:
    """Generates realistic top-down DRAM 6F² array layouts.

    The generator creates DRAM patterns directly at the requested resolution
    by scaling feature size F in pixel units. This avoids generating massive
    intermediate arrays and enables fast, memory-efficient rendering.

    Architecture layers (composited in order):
        1. Active Area (AA): Diagonal pill-shaped Si islands in STI oxide
        2. Wordlines (WL): Horizontal parallel metal lines (pitch = 2F)
        3. Bitlines (BL): Vertical parallel metal lines (pitch = 3F)
        4. Capacitor Array: Hexagonal honeycomb holes with electrode rims
    """

    def __init__(self, config: Dict):
        """Initialize with architecture config dict.

        Args:
            config: Architecture configuration with keys like
                    feature_size_nm, wl_pitch_multiplier, etc.
        """
        self.config = config
        # Base feature size F will be set dynamically per generate call
        # based on desired pixel scale. Store nm value for reference.
        self.feature_size_nm = config.get('feature_size_nm', 18.0)

        # Layout multipliers
        self.wl_pitch_mult = config.get('wl_pitch_multiplier', 2)
        self.bl_pitch_mult = config.get('bl_pitch_multiplier', 3)
        self.aa_tilt_deg = config.get('aa_tilt_deg', 26.57)
        self.aa_length_mult = config.get('aa_length_multiplier', 3.5)
        self.aa_width_mult = config.get('aa_width_multiplier', 1.2)
        self.cap_pitch_mult = config.get('cap_pitch_multiplier', 2.5)
        self.cap_diameter_mult = config.get('cap_diameter_multiplier', 1.2)

        # Edge bloom
        self.edge_bloom_amplitude = config.get('edge_bloom_amplitude', 0.12)

        # LER settings
        self.ler_enabled = config.get('ler_enabled', True)
        self.ler_sigma_mult = config.get('ler_sigma_multiplier', 0.04)

    def generate_array(self, width: int, height: int, seed: int,
                       f_pixels: float = 12.0,
                       phase_x: float = 0.0,
                       phase_y: float = 0.0) -> np.ndarray:
        """Generate a full DRAM array at the specified resolution.

        Args:
            width: Output width in pixels
            height: Output height in pixels
            seed: Random seed for reproducibility
            f_pixels: Feature size F in pixels (controls density)
            phase_x: Horizontal phase offset [0, 1) for pattern alignment
            phase_y: Vertical phase offset [0, 1) for pattern alignment

        Returns:
            Grayscale image as float64 array [0, 1]
        """
        rng = np.random.Generator(np.random.PCG64(seed))
        F = f_pixels

        # Generate individual architecture layers
        aa_layer = self._generate_aa_layer(width, height, F, phase_x, phase_y, rng)
        wl_layer = self._generate_wl_layer(width, height, F, phase_x, phase_y, rng)
        bl_layer = self._generate_bl_layer(width, height, F, phase_x, phase_y, rng)
        cap_layer = self._generate_cap_layer(width, height, F, phase_x, phase_y, rng)

        # Composite layers into final layout
        img = self._composite_layers([aa_layer, wl_layer, bl_layer, cap_layer])

        # Apply simulated SEM edge brightness blooming
        img = self._apply_edge_bloom(img)

        # Optionally apply LER (skip for very small F to save time)
        if self.ler_enabled and F >= 4.0:
            img = self._apply_ler_fast(img, F, rng)

        return np.clip(img, 0.0, 1.0)

    def _generate_aa_layer(self, width: int, height: int, F: float,
                           px: float, py: float,
                           rng: np.random.Generator) -> np.ndarray:
        """Generate the Active Area (substrate/STI) layer.

        Pill-shaped Si islands tilted at ~26.5° in 6F² diagonal grid,
        embedded in SiO2 shallow trench isolation.
        """
        layer = np.zeros((height, width, 2), dtype=np.float64)

        # Material intensities (normalized to [0, 1])
        sti_val = rng.uniform(40, 70) / 255.0
        aa_val = rng.uniform(120, 150) / 255.0

        # Base layer: STI oxide (fully opaque)
        layer[..., 0] = sti_val
        layer[..., 1] = 1.0

        mask = np.zeros((height, width), dtype=np.uint8)

        # 6F² layout: 3F horizontal pitch, 2F vertical pitch
        x_pitch = self.bl_pitch_mult * F
        y_pitch = self.wl_pitch_mult * F

        # Phase offsets
        x_off = px * x_pitch
        y_off = py * y_pitch

        # AA island dimensions
        length = self.aa_length_mult * F
        width_aa = self.aa_width_mult * F

        half_len = max(1, int(length / 2))
        half_wid = max(1, int(width_aa / 2))

        # Draw all AA islands
        for y_idx in range(-1, int(height / y_pitch) + 2):
            for x_idx in range(-1, int(width / (1.5 * F)) + 2):
                # Diagonal grid: stagger odd rows
                if (x_idx + y_idx) % 2 == 0:
                    cx = int(x_idx * 1.5 * F + x_off) % (width + int(x_pitch))
                    cy = int(y_idx * y_pitch + y_off)

                    if -half_len <= cx <= width + half_len and -half_len <= cy <= height + half_len:
                        center = (int(cx), int(cy))
                        axes = (half_len, half_wid)
                        cv2.ellipse(mask, center, axes,
                                    self.aa_tilt_deg, 0, 360, 255, -1)

        layer[mask > 127, 0] = aa_val
        return layer

    def _generate_wl_layer(self, width: int, height: int, F: float,
                           px: float, py: float,
                           rng: np.random.Generator) -> np.ndarray:
        """Generate Wordline layer (horizontal parallel metal lines)."""
        layer = np.zeros((height, width, 2), dtype=np.float64)

        pitch = self.wl_pitch_mult * F
        wl_width = max(1, int(0.8 * F))
        y_off = py * pitch

        val = rng.uniform(200, 230) / 255.0

        # Vectorized: create Y coordinate array and compute mask
        y_coords = np.arange(height)
        # Distance from nearest wordline center
        y_phase = (y_coords + y_off) % pitch
        mask = (y_phase < wl_width) | (y_phase > pitch - wl_width)

        # Apply to full image
        mask_2d = np.tile(mask[:, np.newaxis], (1, width))
        layer[mask_2d, 0] = val
        layer[mask_2d, 1] = 0.6  # Semi-transparent

        return layer

    def _generate_bl_layer(self, width: int, height: int, F: float,
                           px: float, py: float,
                           rng: np.random.Generator) -> np.ndarray:
        """Generate Bitline layer (vertical parallel metal lines)."""
        layer = np.zeros((height, width, 2), dtype=np.float64)

        pitch = self.bl_pitch_mult * F
        bl_width = max(1, int(0.6 * F))
        x_off = px * pitch

        val = rng.uniform(180, 210) / 255.0

        # Vectorized
        x_coords = np.arange(width)
        x_phase = (x_coords + x_off) % pitch
        mask = (x_phase < bl_width) | (x_phase > pitch - bl_width)

        mask_2d = np.tile(mask[np.newaxis, :], (height, 1))
        layer[mask_2d, 0] = val
        layer[mask_2d, 1] = 0.7

        return layer

    def _generate_cap_layer(self, width: int, height: int, F: float,
                            px: float, py: float,
                            rng: np.random.Generator) -> np.ndarray:
        """Generate Capacitor array layer (hexagonal honeycomb holes)."""
        layer = np.zeros((height, width, 2), dtype=np.float64)

        pitch = self.cap_pitch_mult * F
        row_spacing = pitch * np.sqrt(3) / 2

        x_off = px * pitch
        y_off = py * row_spacing

        hole_radius = max(1, int((self.cap_diameter_mult * F) / 2))
        rim_radius = max(hole_radius + 1, int(hole_radius + 0.3 * F))

        rim_val = rng.uniform(160, 190) / 255.0
        hole_val = rng.uniform(20, 40) / 255.0

        mask_rim = np.zeros((height, width), dtype=np.uint8)
        mask_hole = np.zeros((height, width), dtype=np.uint8)

        for row in range(-1, int(height / max(1, row_spacing)) + 2):
            stagger = (row % 2) * (pitch / 2)
            cy = int(row * row_spacing + y_off)
            if cy < -rim_radius or cy > height + rim_radius:
                continue
            for col in range(-1, int(width / max(1, pitch)) + 2):
                cx = int(col * pitch + stagger + x_off)
                if cx < -rim_radius or cx > width + rim_radius:
                    continue
                cv2.circle(mask_rim, (cx, cy), rim_radius, 255, -1)
                cv2.circle(mask_hole, (cx, cy), hole_radius, 255, -1)

        # Bright electrode rim
        rim_only = (mask_rim > 127) & (mask_hole <= 127)
        layer[rim_only, 0] = rim_val
        layer[rim_only, 1] = 0.9

        # Dark hole punches through
        layer[mask_hole > 127, 0] = hole_val
        layer[mask_hole > 127, 1] = 1.0

        return layer

    def _apply_edge_bloom(self, image: np.ndarray) -> np.ndarray:
        """Apply edge brightness blooming (SE yield effect at edges).

        I_edge = I_0 + A * |grad(I)|
        Simulates increased secondary electron yield at steep feature
        boundaries (Reimer 1998, sec(θ) model).
        """
        # Use Scharr for better gradient estimation
        grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        max_grad = grad_mag.max()
        if max_grad > 1e-6:
            grad_mag = grad_mag / max_grad

        return image + self.edge_bloom_amplitude * grad_mag

    def _apply_ler_fast(self, image: np.ndarray, F: float,
                        rng: np.random.Generator) -> np.ndarray:
        """Apply fast Line Edge Roughness via small displacement noise.

        Uses low-resolution noise upscaled to image size for efficiency.
        Displaces pixels by small random amounts to simulate stochastic
        lithographic edge roughness (Bunday et al. PSD model).
        """
        H, W = image.shape
        sigma = self.ler_sigma_mult * F  # Displacement in pixels

        if sigma < 0.2:
            return image  # Too small to matter

        # Generate low-res noise for efficiency (1/8 resolution)
        scale = max(1, min(8, int(F)))
        small_h = max(4, H // scale)
        small_w = max(4, W // scale)

        dx_small = rng.standard_normal((small_h, small_w)).astype(np.float32) * sigma
        dy_small = rng.standard_normal((small_h, small_w)).astype(np.float32) * sigma

        # Smooth the noise (correlated roughness)
        ksize = max(3, int(2 * (sigma / scale) + 1))
        if ksize % 2 == 0:
            ksize += 1
        dx_small = cv2.GaussianBlur(dx_small, (ksize, ksize), 0)
        dy_small = cv2.GaussianBlur(dy_small, (ksize, ksize), 0)

        # Upscale to full resolution
        dx = cv2.resize(dx_small, (W, H), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(dy_small, (W, H), interpolation=cv2.INTER_LINEAR)

        # Build remap coordinates
        y, x = np.mgrid[0:H, 0:W]
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        return cv2.remap(image.astype(np.float32), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT).astype(np.float64)

    def _composite_layers(self, layers: List[np.ndarray]) -> np.ndarray:
        """Composite all layers into a final image using alpha blending.

        Each layer has shape (H, W, 2): channel 0 = intensity, channel 1 = alpha.
        Layers are blended back-to-front (first layer = background).
        """
        h, w = layers[0].shape[:2]
        result = np.zeros((h, w), dtype=np.float64)

        for layer in layers:
            intensity = layer[..., 0]
            alpha = layer[..., 1]
            result = intensity * alpha + result * (1.0 - alpha)

        return result
