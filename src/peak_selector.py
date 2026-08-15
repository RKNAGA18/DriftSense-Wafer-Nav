import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

class PeakSelector:
    """Peak detection with center-prior selection for periodic semiconductor patterns.
    
    In highly periodic structures (DRAM arrays, FinFET grids), the correlation
    map produces a lattice of nearly identical peaks. This module:
    1. Detects all significant peaks via Non-Maximum Suppression (NMS)
    2. Selects the peak closest to the image center (the specified tie-breaker rule)
    3. Reports confidence based on peak discrimination
    """
    
    def __init__(self, threshold_ratio: float = 0.85,
                 min_distance: int = 10,
                 image_center: Tuple[int, int] = (500, 500)):
        """
        Args:
            threshold_ratio: Peaks must be >= this ratio of max peak (0.85 = 85%)
            min_distance: Minimum pixel distance between detected peaks
            image_center: Center of the search image for tie-breaking
        """
        self.threshold_ratio = threshold_ratio
        self.min_distance = max(1, min_distance)
        self.image_center = image_center
    
    def select_peak(self, zncc_map: np.ndarray, 
                    template_shape: Tuple[int, int]) -> Dict:
        """Detect all peaks and select the one closest to image center.
        
        Args:
            zncc_map: ZNCC correlation map
            template_shape: (height, width) of the template used
            
        Returns:
            Dict with:
                'peak_x': int - integer peak x (in zncc_map coordinates)
                'peak_y': int - integer peak y
                'center_x': float - center x in search image coordinates
                'center_y': float - center y in search image coordinates  
                'peak_value': float - ZNCC value at selected peak
                'num_candidates': int - total peaks found
                'confidence': float - discrimination score
                'all_peaks': List[Tuple[int,int,float]] - all detected peaks
        """
        if zncc_map is None or zncc_map.size == 0:
            raise ValueError("Invalid or empty ZNCC map provided.")
            
        # 1. Peak Detection via NMS
        max_val = np.max(zncc_map)
        abs_threshold = 0.3
        threshold = max(self.threshold_ratio * max_val, abs_threshold)
        
        # Kernel size ensures minimum separation between peaks
        kernel_size = self.min_distance * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Morphological dilation to find local maxima
        zncc_float = zncc_map.astype(np.float32)
        dilated = cv2.dilate(zncc_float, kernel)
        
        # A peak is where original equals dilated AND exceeds our threshold
        peak_mask = (zncc_float == dilated) & (zncc_float >= threshold)
        ys, xs = np.where(peak_mask)
        
        all_peaks = []
        for y, x in zip(ys, xs):
            val = float(zncc_map[y, x])
            all_peaks.append((int(x), int(y), val))
            
        if not all_peaks:
            # Fallback to the absolute maximum if no peaks met the threshold
            y, x = np.unravel_index(np.argmax(zncc_map), zncc_map.shape)
            val = float(zncc_map[y, x])
            all_peaks.append((int(x), int(y), val))
            
        # 2. Center-Distance Selection
        template_h, template_w = template_shape
        target_cx, target_cy = self.image_center
        
        best_peak = None
        min_dist = float('inf')
        
        for x, y, val in all_peaks:
            # Convert peak from zncc_map coords to search image center coords
            # Assuming origin (0,0) is top-left
            center_x = x + template_w / 2.0
            center_y = y + template_h / 2.0
            
            dist = np.sqrt((center_x - target_cx)**2 + (center_y - target_cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_peak = (x, y, val, center_x, center_y)
                
        selected_x, selected_y, selected_val, selected_cx, selected_cy = best_peak
        
        # 3. Confidence Score
        # Sort all candidates by peak value descending
        sorted_peaks = sorted(all_peaks, key=lambda p: p[2], reverse=True)
        
        if len(sorted_peaks) > 1:
            highest_val = sorted_peaks[0][2]
            second_highest_val = sorted_peaks[1][2]
            # Use discrimination between top 2 global peaks as confidence of the pattern
            confidence = highest_val - second_highest_val
        else:
            confidence = float(selected_val)
            
        return {
            'peak_x': selected_x,
            'peak_y': selected_y,
            'center_x': float(selected_cx),
            'center_y': float(selected_cy),
            'peak_value': selected_val,
            'num_candidates': len(all_peaks),
            'confidence': confidence,
            'all_peaks': all_peaks
        }
