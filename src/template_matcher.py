import numpy as np
import cv2
import time
from scipy import fft as scipy_fft
from typing import Dict, Tuple, List, Optional
import logging

try:
    from .subpixel import parabolic_refinement, guizar_sicairos_refinement
except ImportError:
    # Fallback placeholders in case subpixel module is not yet implemented
    def parabolic_refinement(zncc_map, x, y): return 0.0, 0.0
    def guizar_sicairos_refinement(search, template, x, y, upsample): return 0.0, 0.0

try:
    from .peak_selector import PeakSelector
except ImportError:
    # Fallback placeholder
    class PeakSelector:
        def __init__(self, **kwargs): pass
        def select(self, zncc_map):
            y, x = np.unravel_index(np.argmax(zncc_map), zncc_map.shape)
            return int(x), int(y), 1

logger = logging.getLogger(__name__)

class TemplateMatcher:
    """Multi-scale, multi-angle FFT-based ZNCC template matching.
    
    Handles cross-magnification localization of a 100x reference pattern
    inside a 10x search image with:
    - Scale bank for magnification uncertainty (±5%)
    - Rotation bank for in-plane drift (±2°)
    - Coarse-to-fine pyramid search
    - Sub-pixel refinement
    - Center-prior peak selection for periodic patterns
    
    The algorithm:
    1. Anti-alias and downsample the 1000x1000 reference to ~100x100
       (matching the search image's effective resolution)
    2. Generate scale bank: 5 templates at different scale factors
    3. For each scale, generate rotation bank: 9 angles at 0.5° steps
    4. Coarse-to-fine: evaluate at half-resolution first, then refine
    5. Select the best (scale, angle) by peak ZNCC score
    6. Apply NMS + center-prior to disambiguate periodic patterns
    7. Sub-pixel refinement via Guizar-Sicairos DFT upsampling
    """
    
    def __init__(self, config: Dict):
        """Initialize with localization config."""
        loc_cfg = config.get('localization', {})
        self.scale_bank = loc_cfg.get('scale_bank', [9.5, 9.75, 10.0, 10.25, 10.5])
        self.rot_step = loc_cfg.get('rotation_bank_step_deg', 0.5)
        self.rot_range = loc_cfg.get('rotation_bank_range_deg', 2.0)
        self.peak_threshold = loc_cfg.get('peak_threshold_ratio', 0.85)
        self.peak_min_dist = loc_cfg.get('peak_min_distance', 10)
        self.subpixel_method = loc_cfg.get('subpixel_method', 'guizar_sicairos')
        self.subpixel_upsample = loc_cfg.get('subpixel_upsampling_factor', 100)
        self.pyramid_levels = loc_cfg.get('pyramid_levels', 2)
        self.img_size = config.get('image', {}).get('size', 1000)
        
        self.peak_selector = PeakSelector(
            threshold_ratio=self.peak_threshold,
            min_distance=self.peak_min_dist,
            image_center=(self.img_size // 2, self.img_size // 2)
        )
    
    def _prepare_templates(self, reference: np.ndarray) -> List[Tuple[np.ndarray, float, float]]:
        """Prepare multi-scale, multi-angle template bank.
        
        Returns list of (template, scale, angle) tuples.
        """
        templates = []
        ref_float = reference.astype(np.float64) / 255.0
        
        for scale in self.scale_bank:
            # Anti-aliased downsampling
            # The reference is at 100x, search at 10x
            # So the reference needs to be downsampled by 'scale' factor
            target_size = int(round(1000 / scale))  # e.g., 100 for scale=10
            
            # Gaussian anti-aliasing: sigma = sqrt(scale^2 - 1) / (2*sqrt(2*ln(2)))
            sigma_aa = np.sqrt(max(0, scale**2 - 1)) / 2.355
            blurred = cv2.GaussianBlur(ref_float, (0, 0), sigma_aa)
            downsampled = cv2.resize(blurred, (target_size, target_size), 
                                     interpolation=cv2.INTER_AREA)
            
            # Generate rotation bank
            angles = np.arange(-self.rot_range, self.rot_range + self.rot_step/2, 
                              self.rot_step)
            for angle in angles:
                if abs(angle) < 0.01:  # No rotation needed
                    templates.append((downsampled, scale, 0.0))
                else:
                    # Rotate with Lanczos interpolation
                    center = (target_size / 2, target_size / 2)
                    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
                    rotated = cv2.warpAffine(downsampled, M, 
                                            (target_size, target_size),
                                            flags=cv2.INTER_LANCZOS4,
                                            borderMode=cv2.BORDER_REFLECT)
                    templates.append((rotated, scale, angle))
        
        return templates

    def _fft_zncc(self, search: np.ndarray, template: np.ndarray) -> np.ndarray:
        """Compute Zero-mean Normalized Cross-Correlation using FFT.
        
        Uses Lewis's fast NCC method:
        - Numerator via FFT cross-correlation
        - Denominator via integral images (summed area tables)
        
        Returns:
            Correlation map of shape (H-h+1, W-w+1) with values in [-1, 1]
        """
        # Ensure float64
        search_f = search.astype(np.float64)
        template_f = template.astype(np.float64)
        
        h, w = template_f.shape
        H, W = search_f.shape
        
        if h > H or w > W:
            return np.zeros((1, 1), dtype=np.float64)
            
        # Template statistics
        t_mean = template_f.mean()
        t_std = template_f.std()
        t_centered = template_f - t_mean
        n = h * w  # number of pixels in template
        
        # FFT cross-correlation (numerator)
        # Pad template to search image size
        pad_template = np.zeros_like(search_f)
        pad_template[:h, :w] = t_centered
        
        # Cross-correlation via FFT
        fft_search = np.fft.rfft2(search_f)
        fft_template = np.fft.rfft2(pad_template)
        cross_corr = np.fft.irfft2(fft_search * np.conj(fft_template))
        
        # Local mean and std of search image using integral images
        integral = cv2.integral(search_f)  # (H+1, W+1)
        integral_sq = cv2.integral(search_f ** 2)
        
        # Local sum in template-sized windows
        # Sum over window [y:y+h, x:x+w]
        local_sum = (integral[h:, w:] - integral[h:, :W-w+1] - 
                     integral[:H-h+1, w:] + integral[:H-h+1, :W-w+1])
        local_sum_sq = (integral_sq[h:, w:] - integral_sq[h:, :W-w+1] - 
                        integral_sq[:H-h+1, w:] + integral_sq[:H-h+1, :W-w+1])
        
        local_mean = local_sum / n
        local_var = local_sum_sq / n - local_mean ** 2
        local_var = np.maximum(local_var, 0)  # numerical stability
        local_std = np.sqrt(local_var)
        
        # ZNCC = (cross_corr - n * t_mean * local_mean) / (n * t_std * local_std)
        # Crop cross_corr to valid region
        result_h = H - h + 1
        result_w = W - w + 1
        numerator = cross_corr[:result_h, :result_w]
        denominator = n * t_std * local_std
        
        # Avoid division by zero
        zncc = np.zeros_like(numerator)
        valid = denominator > 1e-8
        zncc[valid] = numerator[valid] / denominator[valid]
        
        return np.clip(zncc, -1.0, 1.0)

    def _coarse_to_fine(self, search: np.ndarray, 
                        templates: List[Tuple[np.ndarray, float, float]]) -> Dict:
        """Run coarse-to-fine pyramid matching.
        
        Level 1 (coarse): Downsample search by 2x, evaluate ALL templates
        Level 0 (fine): Native resolution, evaluate only top-3 candidates
        """
        # Build search pyramid
        search_f = search.astype(np.float64) / 255.0
        search_coarse = cv2.resize(search_f, 
                                   (search_f.shape[1] // 2, search_f.shape[0] // 2),
                                   interpolation=cv2.INTER_AREA)
        
        # Coarse pass: evaluate all templates at half resolution
        coarse_results = []
        for template, scale, angle in templates:
            # Downsample template by 2x for coarse search
            t_coarse = cv2.resize(template, 
                                 (template.shape[1] // 2, template.shape[0] // 2),
                                 interpolation=cv2.INTER_AREA)
            if t_coarse.shape[0] < 4 or t_coarse.shape[1] < 4:
                continue
            
            zncc_map = self._fft_zncc(search_coarse, t_coarse)
            peak_val = zncc_map.max()
            coarse_results.append({
                'scale': scale, 'angle': angle,
                'peak_val': peak_val,
                'template': template  # Keep original resolution template
            })
        
        # Sort by peak value, take top 3
        coarse_results.sort(key=lambda x: x['peak_val'], reverse=True)
        top_candidates = coarse_results[:3]
        
        # Fine pass: evaluate top candidates at native resolution
        best_result = None
        best_zncc_map = None
        best_peak = -1.0
        
        for cand in top_candidates:
            zncc_map = self._fft_zncc(search_f, cand['template'])
            peak_val = zncc_map.max()
            if peak_val > best_peak:
                best_peak = peak_val
                best_zncc_map = zncc_map
                best_result = cand
        
        return {
            'zncc_map': best_zncc_map,
            'best_scale': best_result['scale'],
            'best_angle': best_result['angle'],
            'peak_val': best_peak,
            'template': best_result['template'],
            'template_shape': best_result['template'].shape
        }

    def localize(self, reference: np.ndarray, search: np.ndarray) -> Dict:
        """Find the reference pattern in the search image.
        
        Args:
            reference: 1000x1000 uint8 grayscale reference image (100x mag)
            search: 1000x1000 uint8 grayscale search image (10x mag)
            
        Returns:
            Dict with:
                'x': float - predicted x coordinate (search pixels)
                'y': float - predicted y coordinate (search pixels)
                'confidence': float - match confidence score
                'best_scale': float - estimated scale factor
                'best_angle': float - estimated rotation (degrees)
                'correlation_peak': float - peak ZNCC value
                'num_candidates': int - number of candidate peaks found
                'time_ms': float - execution time in milliseconds
        """
        start_time = time.perf_counter()
        
        # 1 & 2. Prepare templates and coarse-to-fine search
        templates = self._prepare_templates(reference)
        cf_result = self._coarse_to_fine(search, templates)
        
        zncc_map = cf_result['zncc_map']
        th, tw = cf_result['template_shape']
        best_template = cf_result['template']
        
        # 3. Use PeakSelector to find candidates and select closest to center
        try:
            if hasattr(self.peak_selector, 'select_peak'):
                peak_result = self.peak_selector.select_peak(zncc_map, (th, tw))
                best_x = peak_result['peak_x']
                best_y = peak_result['peak_y']
                num_candidates = peak_result.get('num_candidates', 1)
                confidence = peak_result.get('confidence', cf_result['peak_val'])
                cf_result['peak_val'] = confidence
            else:
                # Fallback to simple argmax if method missing
                best_y, best_x = np.unravel_index(np.argmax(zncc_map), zncc_map.shape)
                num_candidates = 1
        except Exception as e:
            logger.warning(f"Peak selection failed, falling back to argmax: {e}")
            best_y, best_x = np.unravel_index(np.argmax(zncc_map), zncc_map.shape)
            num_candidates = 1

        best_x = float(best_x)
        best_y = float(best_y)

        # 4. Apply sub-pixel refinement to the selected peak
        refined_x, refined_y = best_x, best_y
        try:
            if self.subpixel_method == 'guizar_sicairos':
                search_float = search.astype(np.float64) / 255.0
                best_template_float = best_template.astype(np.float64)
                if best_template_float.max() > 1.0:
                    best_template_float /= 255.0
                pad_template = np.zeros_like(search_float)
                pad_template[:th, :tw] = best_template_float - best_template_float.mean()
                fft_search = np.fft.fft2(search_float)
                fft_template = np.fft.fft2(pad_template)
                refined_x, refined_y = guizar_sicairos_refinement(
                    fft_search, fft_template, best_y, best_x, self.subpixel_upsample
                )
            else:
                refined_x, refined_y = parabolic_refinement(zncc_map, int(best_y), int(best_x))
        except Exception as e:
            logger.warning(f"Subpixel refinement failed, proceeding without it: {e}")
            refined_x, refined_y = best_x, best_y

        # 5. Adjust coordinates to account for template size offset
        # Convert top-left ZNCC map coordinate to the center of the template match
        center_x = refined_x + (tw / 2.0)
        center_y = refined_y + (th / 2.0)
        
        elapsed = time.perf_counter() - start_time
        
        # 6. Return the result dict
        return {
            'x': center_x,
            'y': center_y,
            'confidence': cf_result['peak_val'],
            'best_scale': cf_result['best_scale'],
            'best_angle': cf_result['best_angle'],
            'correlation_peak': cf_result['peak_val'],
            'num_candidates': num_candidates,
            'time_ms': elapsed * 1000.0
        }
