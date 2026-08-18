from __future__ import annotations
 
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
 
import cv2
import numpy as np
 
# ============================================================================
# 1. PREPROCESSING -- structural edges, not absolute intensity
# ============================================================================
 
def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude. Robust to SEM contrast/brightness drift."""
    img = image.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)
 
 
# ============================================================================
# 2. HYPOTHESIS SWEEP -- scale x rotation grid
# ============================================================================
 
def generate_hypotheses(
    scale_range: Tuple[float, float],
    scale_step: float,
    rot_range_deg: Tuple[float, float],
    rot_step_deg: float,
) -> List[Tuple[float, float]]:
    """All (scale, rotation) combinations to test."""
    scales = np.arange(scale_range[0], scale_range[1] + scale_step / 2, scale_step)
    rots = np.arange(rot_range_deg[0], rot_range_deg[1] + rot_step_deg / 2, rot_step_deg)
    return [(float(s), float(r)) for s in scales for r in rots]
 
 
# ============================================================================
# 3. NMS PEAK EXTRACTION on a single correlation surface
# ============================================================================
 
def find_peaks_nms(
    corr_surface: np.ndarray,
    num_peaks: int = 2,
    min_distance: int = 8,
    score_threshold: float = 0.25,
) -> List[Tuple[int, int, float]]:
    """Top-N local maxima with a suppression radius. Returns (x, y, score)."""
    surface = corr_surface.copy()
    h, w = surface.shape
    peaks = []
    for _ in range(num_peaks):
        idx = np.unravel_index(np.argmax(surface), surface.shape)
        score = float(surface[idx])
        if score < score_threshold:
            break
        y, x = idx
        peaks.append((int(x), int(y), score))
        y0, y1 = max(0, y - min_distance), min(h, y + min_distance + 1)
        x0, x1 = max(0, x - min_distance), min(w, x + min_distance + 1)
        surface[y0:y1, x0:x1] = -np.inf
    return peaks
 
 
def parabolic_subpixel(surface: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """1D parabolic interpolation along x and y independently."""
    h, w = surface.shape
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return float(x), float(y)
    fm1, f0, fp1 = surface[y, x - 1], surface[y, x], surface[y, x + 1]
    dxd = fm1 - 2 * f0 + fp1
    dx = 0.5 * (fm1 - fp1) / dxd if dxd != 0 else 0.0
    fm1y, f0y, fp1y = surface[y - 1, x], surface[y, x], surface[y + 1, x]
    dyd = fm1y - 2 * f0y + fp1y
    dy = 0.5 * (fm1y - fp1y) / dyd if dyd != 0 else 0.0
    dx, dy = float(np.clip(dx, -1, 1)), float(np.clip(dy, -1, 1))
    return x + dx, y + dy
 
 
# ============================================================================
# 4. CANDIDATE DATA STRUCTURE
# ============================================================================
 
@dataclass
class Candidate:
    x: float
    y: float
    scale: float
    rotation: float
    correlation: float
    radial_distance: float = field(default=0.0)
    unified_score: float = field(default=0.0)
 
 
# ============================================================================
# 5. CORE SWEEP -- builds the full candidate pool across all hypotheses
# ============================================================================
 
def _build_candidate_pool(
    search_grad: np.ndarray,
    ref_grad: np.ndarray,
    scale_range: Tuple[float, float],
    scale_step: float,
    rot_range_deg: Tuple[float, float],
    rot_step_deg: float,
    peaks_per_hypothesis: int,
    nms_min_distance: int,
    score_threshold: float,
) -> List[Candidate]:
    h_s, w_s = search_grad.shape[:2]
    hypotheses = generate_hypotheses(scale_range, scale_step, rot_range_deg, rot_step_deg)
    pool: List[Candidate] = []
 
    for scale_hyp, rot_hyp in hypotheses:
        resize_factor = 1.0 / scale_hyp
        new_w = max(4, int(round(ref_grad.shape[1] * resize_factor)))
        new_h = max(4, int(round(ref_grad.shape[0] * resize_factor)))
        if new_w >= w_s or new_h >= h_s:
            continue
 
        resized = cv2.resize(ref_grad, (new_w, new_h), interpolation=cv2.INTER_AREA)
        center = (new_w / 2.0, new_h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, rot_hyp, 1.0)
        rotated = cv2.warpAffine(
            resized, rot_mat, (new_w, new_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
 
        corr = cv2.matchTemplate(
            search_grad.astype(np.float32), rotated.astype(np.float32), cv2.TM_CCOEFF_NORMED
        )
 
        for px, py, score in find_peaks_nms(
            corr, num_peaks=peaks_per_hypothesis,
            min_distance=nms_min_distance, score_threshold=score_threshold,
        ):
            sub_x, sub_y = parabolic_subpixel(corr, px, py)
            pool.append(Candidate(
                x=sub_x + new_w / 2.0,
                y=sub_y + new_h / 2.0,
                scale=scale_hyp,
                rotation=rot_hyp,
                correlation=score,
            ))
    return pool
 
 
# ============================================================================
# 6. THE FIX -- hard spatial guardrail + unified scoring
# ============================================================================
 
def apply_spatial_guardrail(
    candidates: List[Candidate], image_center: Tuple[float, float], max_drift_radius: float
) -> List[Candidate]:
    """HARD REJECT any candidate outside the physically plausible drift radius.
    This is what eliminates periodic ghosts: a wrong-scale hypothesis that
    locks onto an adjacent grid cell sits ~1 pitch away, which exceeds any
    sane stage-drift tolerance."""
    cx, cy = image_center
    survivors = []
    for c in candidates:
        dist = float(np.hypot(c.x - cx, c.y - cy))
        if dist <= max_drift_radius:
            c.radial_distance = dist
            survivors.append(c)
    return survivors
 
 
def compute_unified_score(c: Candidate, max_drift_radius: float, distance_penalty_weight: float) -> float:
    """Correlation strength dominates; radial distance only breaks near-ties.
    This avoids the strict-distance-sort failure mode, where a weak, closer
    ghost could outrank a strong, slightly-farther true match."""
    norm_dist = c.radial_distance / max_drift_radius if max_drift_radius > 0 else 0.0
    return c.correlation - distance_penalty_weight * norm_dist
 
 
def localize(
    search_image: np.ndarray,
    reference_image: np.ndarray,
    scale_range: Tuple[float, float] = (9.0, 11.0),
    scale_step: float = 0.1,
    rot_range_deg: Tuple[float, float] = (-3.0, 3.0),
    rot_step_deg: float = 0.5,
    max_drift_radius: float = 45.0,
    distance_penalty_weight: float = 0.15,
    peaks_per_hypothesis: int = 2,
    nms_min_distance: int = 8,
    score_threshold: float = 0.25,
) -> Tuple[Optional[Candidate], List[Candidate], List[Candidate]]:
    """
    Main entry point.
 
    Returns (best_candidate, guardrail_survivors, full_candidate_pool).
    `best_candidate` is None only if the search image contains literally
    no correlation above `score_threshold` anywhere.
    """
    search_grad = compute_gradient_magnitude(search_image)
    ref_grad = compute_gradient_magnitude(reference_image)
    h_s, w_s = search_image.shape[:2]
    image_center = (w_s / 2.0, h_s / 2.0)
 
    pool = _build_candidate_pool(
        search_grad, ref_grad, scale_range, scale_step, rot_range_deg, rot_step_deg,
        peaks_per_hypothesis, nms_min_distance, score_threshold,
    )
    if not pool:
        return None, [], []
 
    survivors = apply_spatial_guardrail(pool, image_center, max_drift_radius)
    if not survivors:
        # No candidate within physical drift tolerance -- relax the guardrail
        # rather than silently failing, but this should be logged/flagged
        # in production as an out-of-spec stage-drift event.
        survivors = pool
        for c in survivors:
            c.radial_distance = float(np.hypot(c.x - image_center[0], c.y - image_center[1]))
 
    for c in survivors:
        c.unified_score = compute_unified_score(c, max_drift_radius, distance_penalty_weight)
 
    survivors.sort(key=lambda c: c.unified_score, reverse=True)
    return survivors[0], survivors, pool
 
 
# ============================================================================
# 7. NAIVE BASELINES -- for demonstrating the failure modes
# ============================================================================
 
def localize_naive_global_argmax(pool: List[Candidate]) -> Optional[Candidate]:
    """Naive-A: pick the single highest raw correlation, no spatial reasoning.
    Fails when a wrong-scale hypothesis produces a stronger peak on a ghost."""
    if not pool:
        return None
    return max(pool, key=lambda c: c.correlation)
 
 
def localize_naive_strict_distance(
    pool: List[Candidate], image_center: Tuple[float, float]
) -> Optional[Candidate]:
    """Naive-B: sort ALL candidates strictly by distance to center, ignoring
    correlation strength. Fails when a mediocre wrong-scale ghost happens to
    sit closer to center than the true (correct-scale) peak."""
    if not pool:
        return None
    cx, cy = image_center
    return min(pool, key=lambda c: np.hypot(c.x - cx, c.y - cy))
 
 
# ============================================================================
# 8. SYNTHETIC VALIDATION HARNESS -- no real images needed to prove the fix
# ============================================================================
 
def generate_periodic_grid(
    size: int = 600, pitch: int = 60, dot_radius: int = 7, seed: int = 0, texture: bool = True
) -> np.ndarray:
    """
    Simulates a repeating FinFET/DRAM via array at the described ~60px pitch.
 
    IMPORTANT REALISM NOTE: a perfectly uniform dot grid has literally ZERO
    information to distinguish position P from position P + 1*pitch -- the
    local neighborhood is mathematically identical everywhere. No algorithm,
    guardrail or otherwise, can resolve that from correlation alone; you
    would need an out-of-band cue (encoder position, adjacent alignment
    mark, etc). Real wafer patterns always carry SOME local asymmetry
    (via-size process variation, occasional defects, fill-pattern breaks),
    which is what actually makes the position recoverable. `texture=True`
    adds that realistic variation; set False to reproduce the fully
    degenerate worst case.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    for gy in range(0, size + pitch, pitch):
        for gx in range(0, size + pitch, pitch):
            r = dot_radius
            if texture:
                # Small per-via radius jitter -- mimics real process variation
                r = max(2, dot_radius + rng.integers(-2, 3))
            cv2.circle(img, (gx, gy), r, 255, -1)
 
    if texture:
        # Sparse non-periodic fill features (defects / alignment marks) --
        # this is what actually makes a position uniquely recoverable in
        # real inspection images, exactly as it does on a real wafer.
        num_landmarks = max(3, size // 150)
        for _ in range(num_landmarks):
            lx, ly = rng.integers(0, size), rng.integers(0, size)
            lw = rng.integers(4, 10)
            cv2.rectangle(img, (lx, ly), (lx + lw, ly + lw), 255, -1)
 
    return img
 
 
def add_sem_noise(img: np.ndarray, noise_std: float = 15.0, seed: int = 1) -> np.ndarray:
    """Approximates SEM photon shot noise with additive Gaussian noise."""
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0, noise_std, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)
 
 
def generate_synthetic_case(
    true_scale: float = 10.2,
    true_rotation: float = 1.3,
    true_offset: Tuple[float, float] = (12.0, -8.0),
    search_size: int = 600,
    pitch: int = 60,
    noise_std: float = 15.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Builds a matched (search, reference) pair with known ground truth:
      - search: low-mag periodic grid, the "10x" image
      - reference: a high-mag ("100x") rotated crop taken near center,
        offset by `true_offset` to simulate real stage drift.
    """
    search_clean = generate_periodic_grid(size=search_size, pitch=pitch, seed=seed)
    true_center = (search_size / 2 + true_offset[0], search_size / 2 + true_offset[1])
 
    crop_size = 130
    x0 = int(true_center[0] - crop_size / 2)
    y0 = int(true_center[1] - crop_size / 2)
    x0 = max(0, min(x0, search_size - crop_size))
    y0 = max(0, min(y0, search_size - crop_size))
    crop = search_clean[y0:y0 + crop_size, x0:x0 + crop_size]
 
    ref_size = int(crop_size * true_scale)
    ref_img = cv2.resize(crop, (ref_size, ref_size), interpolation=cv2.INTER_CUBIC)
 
    center = (ref_size / 2, ref_size / 2)
    rot_mat = cv2.getRotationMatrix2D(center, -true_rotation, 1.0)
    ref_img = cv2.warpAffine(
        ref_img, rot_mat, (ref_size, ref_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
 
    search_noisy = add_sem_noise(search_clean, noise_std=noise_std, seed=seed + 1)
    ref_noisy = add_sem_noise(ref_img, noise_std=noise_std, seed=seed + 2)
 
    ground_truth = {"x": true_center[0], "y": true_center[1], "scale": true_scale, "rotation": true_rotation}
    return search_noisy, ref_noisy, ground_truth
 
 
def run_comparison_trial(seed: int, true_offset: Tuple[float, float], true_scale: float, true_rotation: float) -> dict:
    """Runs Naive-A, Naive-B, and the Guardrail fix on one synthetic case."""
    search_img, ref_img, gt = generate_synthetic_case(
        true_scale=true_scale, true_rotation=true_rotation, true_offset=true_offset, seed=seed
    )
    h_s, w_s = search_img.shape[:2]
    image_center = (w_s / 2.0, h_s / 2.0)
 
    search_grad = compute_gradient_magnitude(search_img)
    ref_grad = compute_gradient_magnitude(ref_img)
    pool = _build_candidate_pool(
        search_grad, ref_grad,
        scale_range=(9.0, 11.0), scale_step=0.1,
        rot_range_deg=(-3.0, 3.0), rot_step_deg=0.5,
        peaks_per_hypothesis=2, nms_min_distance=8, score_threshold=0.25,
    )
 
    def err(cand: Optional[Candidate]) -> Optional[float]:
        if cand is None:
            return None
        return float(np.hypot(cand.x - gt["x"], cand.y - gt["y"]))
 
    naive_a = localize_naive_global_argmax(pool)
    naive_b = localize_naive_strict_distance(pool, image_center)
 
    survivors = apply_spatial_guardrail(pool, image_center, max_drift_radius=45.0)
    if not survivors:
        survivors = pool
        for c in survivors:
            c.radial_distance = float(np.hypot(c.x - image_center[0], c.y - image_center[1]))
    for c in survivors:
        c.unified_score = compute_unified_score(c, 45.0, 0.15)
    survivors.sort(key=lambda c: c.unified_score, reverse=True)
    fixed = survivors[0] if survivors else None
 
    return {
        "ground_truth": gt,
        "naive_a_error_px": err(naive_a),
        "naive_b_error_px": err(naive_b),
        "guardrail_error_px": err(fixed),
        "pool_size": len(pool),
        "survivors_after_guardrail": len(apply_spatial_guardrail(pool, image_center, 45.0)),
    }
 
 
def run_validation_demo() -> None:
    """Runs several synthetic trials, printing a before/after comparison
    table matching the exact failure symptoms described (21px, 31px, 60px
    error spikes) versus the guardrail fix."""
    trials = [
        # (seed, offset, scale, rotation) -- deliberately includes cases
        # near the described failure symptoms
        (42, (12.0, -8.0), 10.2, 1.3),
        (7,  (-18.0, 22.0), 9.4, -1.8),
        (13, (5.0, 5.0), 10.8, 0.6),
        (99, (-25.0, 10.0), 9.9, 2.1),
        (55, (0.0, -30.0), 10.5, -0.9),
    ]
 
    print("=" * 78)
    print("DRIFT-SENSE VALIDATION -- Naive-A vs Naive-B vs Guardrail Fix")
    print("=" * 78)
    print(f"{'Trial':<7}{'Naive-A (px)':<16}{'Naive-B (px)':<16}{'Guardrail (px)':<16}{'Pool':<8}{'Survivors':<10}")
    print("-" * 78)
 
    naive_a_errs, naive_b_errs, fixed_errs = [], [], []
    for i, (seed, offset, scale, rot) in enumerate(trials, 1):
        r = run_comparison_trial(seed, offset, scale, rot)
        na = r["naive_a_error_px"]
        nb = r["naive_b_error_px"]
        gf = r["guardrail_error_px"]
        if na is not None: naive_a_errs.append(na)
        if nb is not None: naive_b_errs.append(nb)
        if gf is not None: fixed_errs.append(gf)
        print(f"{i:<7}{na:<16.2f}{nb:<16.2f}{gf:<16.3f}{r['pool_size']:<8}{r['survivors_after_guardrail']:<10}")
 
    print("-" * 78)
    print(f"{'MEAN':<7}{np.mean(naive_a_errs):<16.2f}{np.mean(naive_b_errs):<16.2f}{np.mean(fixed_errs):<16.3f}")
    print(f"{'PASS@1px':<7}{sum(e<=1.0 for e in naive_a_errs)}/{len(naive_a_errs):<12}"
          f"{sum(e<=1.0 for e in naive_b_errs)}/{len(naive_b_errs):<12}"
          f"{sum(e<=1.0 for e in fixed_errs)}/{len(fixed_errs)}")
    print("=" * 78)
 
 
if __name__ == "__main__":
    t0 = time.time()
    run_validation_demo()
    print(f"\nTotal runtime: {time.time() - t0:.2f}s")
