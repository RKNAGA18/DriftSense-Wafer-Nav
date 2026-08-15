import numpy as np
import cv2
from scipy import ndimage
from typing import Dict, Tuple, Optional

class SEMNoisePipeline:
    """Physics-based SEM image degradation engine.
    
    Applies realistic Scanning Electron Microscope artifacts to 
    synthetic semiconductor layout images. Each degradation is
    independently controllable and based on published SEM physics.
    
    References:
        - Goldstein et al. (2018) SEM and X-ray Microanalysis, Springer
        - Reimer (1998) SEM: Physics of Image Formation, Springer
        - Bunday et al. (2014-2020) CD-SEM Metrology, IEEE/SPIE
    """
    
    def __init__(self, config: Dict):
        """Initialize with noise configuration parameters."""
        self.config = config
    
    def apply_all(self, image: np.ndarray, noise_multiplier: float = 1.0,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Apply the full SEM degradation pipeline.
        
        Order matters! Apply in physically correct sequence:
        1. Defocus / astigmatism blur (optical path)
        2. Edge brightness blooming (SE yield physics) - handled via blur/contrast or skipped if not configured
        3. Charging streaks (specimen interaction)
        4. Scan line jitter (scan electronics)
        5. Drift (thermal/mechanical stage)
        6. AC mains hum (EMI)
        7. Contrast / gamma adjustment (detector)
        8. Shot noise (Poisson statistics) 
        9. Salt & pepper noise (dead pixels)
        
        Args:
            image: Input grayscale image as float64 [0, 1]
            noise_multiplier: Scale factor for noise intensity
            rng: Random number generator for reproducibility
            
        Returns:
            Degraded image as float64 [0, 1]
        """
        if rng is None:
            rng = np.random.default_rng()
            
        img = image.copy()
        
        # 1. Defocus / astigmatism blur
        if 'blur' in self.config:
            cfg = self.config['blur']
            sigma_u = cfg.get('sigma_u', 1.0) * noise_multiplier
            sigma_v = cfg.get('sigma_v', 1.0) * noise_multiplier
            angle_deg = cfg.get('angle_deg', 0.0)
            if sigma_u > 0 and sigma_v > 0:
                img = self.apply_blur(img, sigma_u, sigma_v, angle_deg)
                
        # 3. Charging streaks
        if 'charging' in self.config:
            cfg = self.config['charging']
            # charging alpha doesn't necessarily scale linearly with multiplier, but we'll scale beta
            alpha = cfg.get('alpha', 0.95)
            beta = cfg.get('beta', 0.05) * noise_multiplier
            if beta > 0:
                img = self.apply_charging(img, alpha, beta, rng)
                
        # 4. Scan line jitter
        if 'jitter' in self.config:
            cfg = self.config['jitter']
            sigma = cfg.get('sigma', 1.0) * noise_multiplier
            if sigma > 0:
                img = self.apply_jitter(img, sigma, rng)
                
        # 5. Drift
        if 'drift' in self.config:
            cfg = self.config['drift']
            vx = cfg.get('vx', 0.0) * noise_multiplier
            vy = cfg.get('vy', 0.0) * noise_multiplier
            if vx != 0 or vy != 0:
                img = self.apply_drift(img, vx, vy)
                
        # 6. AC mains hum
        if 'mains_hum' in self.config:
            cfg = self.config['mains_hum']
            amplitude = cfg.get('amplitude', 1.0) * noise_multiplier
            if amplitude > 0:
                img = self.apply_mains_hum(img, amplitude, rng)
                
        # 7. Contrast / gamma adjustment
        if 'contrast_gamma' in self.config:
            cfg = self.config['contrast_gamma']
            alpha = cfg.get('alpha', 1.0)
            beta = cfg.get('beta', 0.0)
            gamma = cfg.get('gamma', 1.0)
            img = self.apply_contrast_gamma(img, alpha, beta, gamma)
            
        # 8. Shot noise
        if 'shot_noise' in self.config:
            cfg = self.config['shot_noise']
            # Higher multiplier -> lower dose (more noise)
            base_dose = cfg.get('dose', 50.0)
            dose = max(1.0, base_dose / max(1e-6, noise_multiplier))
            img = self.apply_shot_noise(img, dose, rng)
            
        # 9. Salt & pepper noise
        if 'salt_pepper' in self.config:
            cfg = self.config['salt_pepper']
            density = cfg.get('density', 0.001) * noise_multiplier
            if density > 0:
                img = self.apply_salt_pepper(img, density, rng)
                
        return np.clip(img, 0.0, 1.0)

    def apply_shot_noise(self, image: np.ndarray, dose: float,
                         rng: np.random.Generator) -> np.ndarray:
        """Apply Poisson shot noise.
        
        Models the discrete nature of electron detection:
        I_noisy = Poisson(I * S) / S
        
        Higher dose S = cleaner image (more electrons per pixel).
        Typical S range: [5 (very noisy) to 500 (clean)].
        
        Args:
            image: Input image [0, 1]
            dose: Dose parameter S (higher = less noise)
            rng: Random generator
        """
        noisy = rng.poisson(np.clip(image, 0, 1) * dose) / dose
        return np.clip(noisy, 0.0, 1.0)
    
    def apply_blur(self, image: np.ndarray, sigma_u: float, sigma_v: float,
                   angle_deg: float) -> np.ndarray:
        """Apply anisotropic Gaussian PSF (defocus + astigmatism).
        
        When sigma_u == sigma_v: isotropic defocus blur.
        When sigma_u != sigma_v: astigmatic blur at given angle.
        
        Uses rotated 2D Gaussian kernel:
        Sigma = R(theta) @ [[sigma_u^2, 0], [0, sigma_v^2]] @ R(theta)^T
        """
        # Kernel size
        max_sigma = max(sigma_u, sigma_v)
        ksize = int(2 * np.ceil(3 * max_sigma) + 1)
        if ksize < 3:
            ksize = 3
            
        # Create meshgrid
        x = np.linspace(-ksize//2, ksize//2, ksize)
        y = np.linspace(-ksize//2, ksize//2, ksize)
        X, Y = np.meshgrid(x, y)
        
        # Rotate coordinates
        theta = np.deg2rad(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
        
        X_rot = c * X - s * Y
        Y_rot = s * X + c * Y
        
        # Calculate kernel
        var_u = sigma_u**2
        var_v = sigma_v**2
        
        if var_u < 1e-6: var_u = 1e-6
        if var_v < 1e-6: var_v = 1e-6
        
        kernel = np.exp(-0.5 * (X_rot**2 / var_u + Y_rot**2 / var_v))
        kernel = kernel / np.sum(kernel)
        
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    def apply_charging(self, image: np.ndarray, alpha: float, beta: float,
                       rng: np.random.Generator) -> np.ndarray:
        """Apply charging streak artifacts.
        
        Simulates charge buildup on insulating surfaces during raster scan.
        Horizontal IIR filter along each scan line:
        C(x, y) = alpha * C(x-1, y) + beta * I(x, y)
        I_out(x, y) = I(x, y) + C(x, y)
        
        alpha: decay constant [0.9 - 0.99] (higher = longer streaks)
        beta: charge accumulation rate
        """
        out = np.zeros_like(image)
        h, w = image.shape
        
        for y in range(h):
            charge = 0.0
            row_in = image[y, :]
            row_out = np.zeros_like(row_in)
            for x in range(w):
                charge = alpha * charge + beta * row_in[x]
                row_out[x] = row_in[x] + charge
            out[y, :] = row_out
            
        return np.clip(out, 0.0, 1.0)
    
    def apply_jitter(self, image: np.ndarray, sigma: float,
                     rng: np.random.Generator) -> np.ndarray:
        """Apply scan line jitter.
        
        Per-row horizontal displacement: dx(y) ~ N(0, sigma^2)
        Simulates timing jitter in the scan electronics.
        """
        h, w = image.shape
        shifts = rng.normal(0, sigma, size=h)
        
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        X_shifted = X - shifts[:, np.newaxis]
        
        # We use map_coordinates or cv2.remap. cv2.remap is often faster.
        map_x = X_shifted.astype(np.float32)
        map_y = Y.astype(np.float32)
        
        return cv2.remap(image.astype(np.float32), map_x, map_y, 
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    
    def apply_drift(self, image: np.ndarray, vx: float, vy: float) -> np.ndarray:
        """Apply slow linear stage drift.
        
        Accumulating displacement over the image scan:
        dx(x,y) = vx * (y * W + x) / f_pixel
        dy(x,y) = vy * (y * W + x) / f_pixel
        """
        h, w = image.shape
        
        # Simplified: let f_pixel be simply W * H, so total drift is vx, vy across the whole frame
        f_pixel = h * w
        
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Pixel index
        idx = Y * w + X
        
        dx = vx * idx / f_pixel
        dy = vy * idx / f_pixel
        
        map_x = (X - dx).astype(np.float32)
        map_y = (Y - dy).astype(np.float32)
        
        return cv2.remap(image.astype(np.float32), map_x, map_y, 
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    
    def apply_mains_hum(self, image: np.ndarray, amplitude: float,
                        rng: np.random.Generator) -> np.ndarray:
        """Apply AC mains (50/60Hz) electromagnetic interference.
        
        Sinusoidal horizontal displacement along slow scan axis:
        dx(y) = A * sin(2*pi*f_mains/f_line * y + phi)
        
        Creates wavy vertical edges characteristic of EMI.
        """
        h, w = image.shape
        
        # 3 to 8 cycles across the image height
        cycles = rng.uniform(3, 8)
        phi = rng.uniform(0, 2 * np.pi)
        
        y_coords = np.arange(h)
        dx = amplitude * np.sin(2 * np.pi * cycles * y_coords / h + phi)
        
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        X_shifted = X - dx[:, np.newaxis]
        
        map_x = X_shifted.astype(np.float32)
        map_y = Y.astype(np.float32)
        
        return cv2.remap(image.astype(np.float32), map_x, map_y, 
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    
    def apply_contrast_gamma(self, image: np.ndarray, alpha: float,
                             beta: float, gamma: float) -> np.ndarray:
        """Apply contrast, brightness, and gamma adjustment.
        
        I_out = clip(alpha * I^gamma + beta/255, 0, 1)
        Simulates PMT non-linearity and detector gain.
        """
        # Add small epsilon for gamma correction to avoid zero to power
        safe_img = np.clip(image, 1e-6, 1.0)
        out = alpha * (safe_img ** gamma) + (beta / 255.0)
        return np.clip(out, 0.0, 1.0)
    
    def apply_salt_pepper(self, image: np.ndarray, density: float,
                          rng: np.random.Generator) -> np.ndarray:
        """Apply salt-and-pepper noise (dead/hot pixels).
        
        Randomly sets pixels to 0 (dead) or 1 (hot).
        density: fraction of affected pixels [0, 0.01]
        """
        h, w = image.shape
        out = image.copy()
        
        # Generate random values for each pixel
        rand_matrix = rng.random((h, w))
        
        # Salt (hot pixels) -> 1
        out[rand_matrix < (density / 2)] = 1.0
        
        # Pepper (dead pixels) -> 0
        out[(rand_matrix >= (density / 2)) & (rand_matrix < density)] = 0.0
        
        return out
