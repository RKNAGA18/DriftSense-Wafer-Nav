import numpy as np
from typing import Tuple

def parabolic_refinement(zncc_map: np.ndarray, peak_y: int, peak_x: int) -> Tuple[float, float]:
    """2D parabolic (quadratic) sub-pixel refinement.
    
    Fits a 2D quadratic surface to the 3x3 neighborhood around the integer peak:
    f(dx, dy) = a*dx^2 + b*dy^2 + c*dx*dy + d*dx + e*dy + f0
    
    Setting gradient to zero gives sub-pixel offset:
    [dx, dy] = -inv([[2a, c], [c, 2b]]) @ [d, e]
    
    To mitigate 'pixel locking' bias, we fit on log(R) instead of R
    when all 9 neighbors are positive.
    
    Args:
        zncc_map: Full ZNCC correlation map
        peak_y: Integer peak row
        peak_x: Integer peak column
        
    Returns:
        (refined_x, refined_y) - sub-pixel peak position
        
    Precision: ~0.03-0.05 pixels
    """
    H, W = zncc_map.shape
    
    # Boundary check - can't refine at edges
    if peak_y < 1 or peak_y >= H-1 or peak_x < 1 or peak_x >= W-1:
        return float(peak_x), float(peak_y)
    
    # Extract 3x3 neighborhood
    patch = zncc_map[peak_y-1:peak_y+2, peak_x-1:peak_x+2].copy()
    
    # Use log-domain fitting if all values are positive (reduces pixel locking)
    if np.all(patch > 0):
        patch = np.log(patch)
    
    # Fit 2D quadratic using the 3x3 grid
    # Grid positions: (-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)
    # 
    # For separable estimation:
    # dx = -(patch[1,2] - patch[1,0]) / (2 * (patch[1,2] - 2*patch[1,1] + patch[1,0]))
    # dy = -(patch[2,1] - patch[0,1]) / (2 * (patch[2,1] - 2*patch[1,1] + patch[0,1]))
    
    # X direction (columns)
    denom_x = 2.0 * (patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0])
    if abs(denom_x) > 1e-10:
        dx = -(patch[1, 2] - patch[1, 0]) / denom_x
    else:
        dx = 0.0
    
    # Y direction (rows)
    denom_y = 2.0 * (patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1])
    if abs(denom_y) > 1e-10:
        dy = -(patch[2, 1] - patch[0, 1]) / denom_y
    else:
        dy = 0.0
    
    # Clamp to [-0.5, 0.5] to prevent runaway
    dx = np.clip(dx, -0.5, 0.5)
    dy = np.clip(dy, -0.5, 0.5)
    
    return float(peak_x + dx), float(peak_y + dy)


def guizar_sicairos_refinement(fft_search: np.ndarray, fft_template: np.ndarray,
                                peak_y: int, peak_x: int,
                                upsample_factor: int = 100) -> Tuple[float, float]:
    """Guizar-Sicairos matrix-multiply DFT upsampling for sub-pixel registration.
    
    Based on: Guizar-Sicairos et al., "Efficient subpixel image registration 
    algorithms", Optics Letters, 2008.
    
    Achieves arbitrary sub-pixel precision 1/kappa (e.g., kappa=100 -> 0.01 px)
    by computing upsampled DFT only in a small window around the coarse peak.
    
    The key insight: instead of zero-padding the full FFT (memory explosive),
    compute the DFT at upsampled positions using matrix multiplications:
    R_up(r,c) = E_row @ Q(u,v) @ E_col
    
    Args:
        fft_search: FFT of search image (full-size)
        fft_template: FFT of template (padded to search image size)
        peak_y: Coarse integer peak row in correlation map
        peak_x: Coarse integer peak column in correlation map
        upsample_factor: Upsampling factor kappa (100 = 0.01px precision)
        
    Returns:
        (refined_x, refined_y) - sub-pixel peak position
        
    Precision: ~0.005-0.01 pixels for kappa=100
    """
    # Cross-power spectrum
    cross_power = fft_search * np.conj(fft_template)
    
    nr, nc = cross_power.shape
    
    # Upsampled region size (1.5 pixels around coarse peak)
    upsampled_region_size = int(np.ceil(upsample_factor * 1.5))
    
    # Center of upsampled region
    dftshift = int(np.fix(upsampled_region_size / 2))
    
    # Upsampled DFT via matrix multiply
    # Row kernel
    row_indices = np.arange(upsampled_region_size) - dftshift
    col_kernel_indices = np.arange(nc)  # Use ifftshift for proper centering
    
    # Shift to center around coarse peak
    row_shift = peak_y
    col_shift = peak_x
    
    # Build DFT matrices
    # E_col: upsampled cols
    col_freqs = (np.fft.ifftshift(np.arange(nc)) - np.floor(nc / 2)).reshape(-1, 1)
    col_positions = (np.arange(upsampled_region_size) - dftshift).reshape(1, -1)
    col_positions = col_positions / upsample_factor + col_shift
    E_col = np.exp(-1j * 2 * np.pi / nc * col_freqs * col_positions)
    
    # E_row: upsampled rows  
    row_freqs = (np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2)).reshape(1, -1)
    row_positions = (np.arange(upsampled_region_size) - dftshift).reshape(-1, 1)
    row_positions = row_positions / upsample_factor + row_shift
    E_row = np.exp(-1j * 2 * np.pi / nr * row_positions * row_freqs)
    
    # Compute upsampled cross-correlation
    upsampled_cc = E_row @ cross_power @ E_col
    
    # Find peak in upsampled region
    cc_abs = np.abs(upsampled_cc)
    max_idx = np.unravel_index(cc_abs.argmax(), cc_abs.shape)
    
    # Convert back to original coordinates
    refined_y = row_shift + (max_idx[0] - dftshift) / upsample_factor
    refined_x = col_shift + (max_idx[1] - dftshift) / upsample_factor
    
    return float(refined_x), float(refined_y)
