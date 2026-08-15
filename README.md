# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

**SEMICON India Hackathon 2026 — Applied Materials Track**

A Python solution for cross-magnification localization in semiconductor wafer
inspection. Locates a 100× reference pattern inside a 10× search image using
multi-scale, multi-angle FFT-based normalized cross-correlation with sub-pixel
refinement.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Folder Structure](#folder-structure)
- [Coordinate Convention](#coordinate-convention)
- [Dataset Generation](#dataset-generation)
- [Localization / Inference](#localization--inference)
- [Evaluation](#evaluation)
- [Architecture Choice](#architecture-choice)
- [Algorithm Overview](#algorithm-overview)
- [Hardware & Timing](#hardware--timing)
- [References](#references)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate 50 synthetic DRAM SEM image pairs
python generate_dataset.py --config configs/dram_config.yaml --num_pairs 50 --seed 42

# 3. Run localization on all pairs
python localize.py --manifest data/manifest.csv --output_dir results/

# 4. Evaluate and generate reports
python evaluate.py --predictions results/predictions.csv --output_dir results/
```

---

## Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Python version:** 3.10+
**Key dependencies:** numpy, scipy, opencv-python-headless, matplotlib, PyYAML, pandas, scikit-image, tqdm

---

## Folder Structure

```
submission/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── generate_dataset.py             # CLI: synthetic dataset generation
├── localize.py                     # CLI: localization / inference
├── evaluate.py                     # CLI: evaluation & failure analysis
├── configs/
│   └── dram_config.yaml            # Generator & localization configuration
├── src/
│   ├── __init__.py
│   ├── dram_generator.py           # DRAM 6F² geometry engine
│   ├── sem_noise.py                # SEM degradation pipeline
│   ├── pair_builder.py             # Image pair orchestrator
│   ├── template_matcher.py         # Multi-scale/angle FFT-ZNCC matcher
│   ├── subpixel.py                 # Sub-pixel refinement algorithms
│   └── peak_selector.py            # Peak NMS + center-prior selection
├── data/
│   ├── manifest.csv                # Dataset manifest (generated)
│   └── pairs/                      # Generated image pairs (generated)
│       └── 0001/
│           ├── reference.png       # 1000×1000 grayscale (100× mag)
│           ├── search.png          # 1000×1000 grayscale (10× mag)
│           └── metadata.json       # Per-pair generation parameters
├── results/
│   ├── predictions.csv             # GT + predictions side-by-side
│   ├── summary.txt                 # Aggregate statistics
│   ├── ppt_ready_summary.txt       # Copy-paste ready for Slide 9
│   └── plots/
│       ├── error_histogram.png
│       ├── error_vs_noise.png
│       ├── error_vs_scale.png
│       ├── error_vs_rotation.png
│       ├── threshold_pass_rates.png
│       ├── failure_analysis.png
│       └── failure_cases/          # Visualized worst cases
└── references/
    └── references.md               # Literature citations
```

---

## Coordinate Convention

| Property | Value |
|:---------|:------|
| **Origin** | Top-left corner of the image: `(0, 0)` |
| **X-axis** | Increases to the **right** |
| **Y-axis** | Increases **downward** |
| **Output** | Predicted target-centre coordinates `(x, y)` in **search-image pixels** |
| **Multiple matches** | If several valid matches exist, select the one closest to the search-image centre `(500, 500)` |

---

## Dataset Generation

```bash
# Generate 50 pairs with default config
python generate_dataset.py --config configs/dram_config.yaml --num_pairs 50 --seed 42

# Generate 10 pairs to a custom directory
python generate_dataset.py --config configs/dram_config.yaml --num_pairs 10 --output_dir data/test_pairs --seed 123
```

### Output

- **Images:** `data/pairs/{pair_id:04d}/reference.png` and `search.png`
- **Manifest:** `data/manifest.csv` with columns:
  `pair_id, ref_path, search_path, gt_x, gt_y, scale, rotation_deg, difficulty, seed`
- **Metadata:** `data/pairs/{pair_id:04d}/metadata.json` with full generation parameters

### Difficulty Levels

| Level | Shot Noise Dose | Blur σ | Charging | Jitter | Description |
|:------|:---------------|:-------|:---------|:-------|:------------|
| easy | 200 | 0.8 | none | none | Clean, minimal degradation |
| medium | 80 | 1.2/1.0 | light | 0.3 px | Moderate SEM artifacts |
| hard | 30 | 2.0/1.0 | heavy | 0.8 px | Significant degradation |
| extreme | 10 | 3.5/1.0 | severe | 1.5 px | Worst-case scenario |

---

## Localization / Inference

```bash
# Single pair
python localize.py --ref data/pairs/0001/reference.png --search data/pairs/0001/search.png

# Batch mode (processes all pairs in manifest)
python localize.py --manifest data/manifest.csv --output_dir results/
```

### Output (Single Pair)
```
Localization Result:
  Predicted center: (487.3421, 512.8765)
  Confidence: 0.0823
  Best scale: 10.0
  Best angle: 0.50°
  Correlation peak: 0.9234
  Runtime: 8.3 ms
```

### Output (Batch Mode)
- `results/predictions.csv` — Contains both ground truth AND predicted coordinates:
  `pair_id, ref_path, search_path, gt_x, gt_y, pred_x, pred_y, confidence, best_scale, best_angle, correlation_peak, num_candidates, runtime_ms, error_px, difficulty, scale, rotation_deg`

---

## Evaluation

```bash
python evaluate.py --predictions results/predictions.csv --output_dir results/
```

### Metrics Reported
- **Euclidean error:** √((x_pred − x_true)² + (y_pred − y_true)²)
- **Pass rates:** at 5, 4, 2, 1, and 0.5 pixel thresholds
- **Statistics:** Mean, median, worst-case error
- **Runtime:** Per-pair timing (wall clock via `time.perf_counter()`)

### Failure Analysis Categories
- **PERIODIC_AMBIGUITY**: Algorithm matched the wrong periodic tile
- **NOISE_DEGRADATION**: Heavy SEM noise caused correlation degradation
- **SCALE_MISMATCH**: Scale deviation too large for the scale bank
- **ROTATION_SENSITIVITY**: Rotation-induced decorrelation
- **EDGE_POSITION**: Target too close to search image boundary

---

## Architecture Choice

**DRAM 6F²** — Selected for richer visual complexity and stronger demonstration
of domain knowledge.

| Feature | Implementation |
|:--------|:-------------- |
| Active Area (AA) | Diagonal pill-shaped Si islands at ~26.5° tilt in 6F² grid |
| Wordlines (WL) | Horizontal parallel lines, pitch = 2F |
| Bitlines (BL) | Vertical parallel lines, pitch = 3F |
| Capacitor Array | Hexagonal honeycomb arrangement of cylindrical holes |
| Edge Blooming | Gradient-based SE yield enhancement at feature edges |
| Line Edge Roughness | Fourier-filtered noise with realistic PSD parameters |

---

## Algorithm Overview

**Multi-Scale, Multi-Angle FFT-ZNCC with Guizar-Sicairos Sub-Pixel Refinement**

1. **Template Preparation**: Anti-alias + downsample 1000×1000 reference to ~100×100
2. **Scale Bank**: 5 templates at scales {9.5, 9.75, 10.0, 10.25, 10.5}
3. **Rotation Bank**: 9 angles at {-2.0°, -1.5°, ..., +2.0°} per scale (45 total)
4. **Coarse-to-Fine**: Pyramid search at half-resolution, then refine top-3
5. **Peak Selection**: NMS + center-prior for periodic pattern disambiguation
6. **Sub-Pixel**: Guizar-Sicairos matrix-multiply DFT upsampling (κ=100, ~0.01 px)

No GPU required. No deep learning training. Fully deterministic and reproducible.

---

## Hardware & Timing

| Specification | Value |
|:-------------|:------|
| **CPU** | (documented at evaluation time) |
| **RAM** | (documented at evaluation time) |
| **Python** | 3.10+ |
| **Timing Method** | `time.perf_counter()` (wall clock, highest resolution) |
| **GPU** | Not required |

---

## References

See [`references/references.md`](references/references.md) for full citations.

1. Goldstein et al. (2018) — *SEM and X-ray Microanalysis*, Springer
2. Reimer (1998) — *SEM: Physics of Image Formation*, Springer
3. Bunday et al. (2014–2020) — CD-SEM Metrology, IEEE/SPIE
4. IEEE IRDS (2022/2023) — More Moore & Metrology Chapters
5. TechInsights (2018–2024) — DRAM/FinFET process analysis
6. Guizar-Sicairos et al. (2008) — Sub-pixel registration, Optics Letters

---

## Assumptions

- All images are single-channel 8-bit grayscale (0–255).
- No proprietary fab data is used; all structures are synthetic.
- The solution processes standard PNG files without requiring special hardware.
- Random seeds ensure full reproducibility of all generated data.
