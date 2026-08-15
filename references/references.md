# References

Public sources supporting the semiconductor structures, SEM image formation,
noise models, and transformations used in this solution.

---

## Semiconductor Device Architectures

1. **IEEE / SEMI International Roadmap for Devices and Systems (IRDS), 2022/2023 Editions.**
   *IRDS Reports: "More Moore" and "Metrology" Chapters.*
   IEEE IRDS Standards Committee.
   - Official roadmap specifying contacted poly pitch (CPP), fin pitch (FP),
     gate length, and DRAM 6F² physical scaling rules.
   - Source for DRAM cell area formulas, pitch values across technology nodes
     (20nm through 1a/1b/1c), and metrology precision requirements.
   - URL: https://irds.ieee.org/

2. **TechInsights Memory & Logic Process Analysis Reports, 2018–2024.**
   *"Samsung, SK Hynix, and Micron 1x/1y/1z/1a/1b DRAM Cell Architecture Analysis"*
   & *"TSMC / Intel 14nm, 10nm, 7nm, 5nm FinFET Structural Micrographs."*
   TechInsights Inc.
   - Detailed cross-sectional and top-down SEM/TEM analysis of commercial
     6F² honeycomb capacitor layouts, buried wordline geometries, and
     FinFET multi-fin standard cell layouts.
   - Source for active area tilt angles (~26.5°), hexagonal capacitor arrangements,
     and multi-layer compositing in top-down views.

---

## SEM Imaging Physics & Noise Models

3. **Goldstein, J. I., Newbury, D. E., Michael, J. R., Ritchie, N. W.,
   Scott, J. H. J., & Joy, D. C. (2018).**
   *Scanning Electron Microscopy and X-ray Microanalysis* (4th ed.).
   Springer, New York. ISBN: 978-1-4939-6674-5.
   - Definitive reference for electron-matter interaction physics,
     secondary electron escape depth models, specimen charging mechanics,
     and detector collection efficiency.
   - Source for Poisson shot noise model, material Z-contrast
     (backscatter yield vs atomic number), and charging streak formation.

4. **Reimer, L. (1998).**
   *Scanning Electron Microscopy: Physics of Image Formation and
   Microanalysis* (2nd ed.). Springer Series in Optical Sciences, Vol. 45.
   ISBN: 978-3-540-63909-8.
   - Mathematical derivations of the sec(θ) secondary electron yield model
     for edge brightness blooming, electron beam astigmatism/defocus
     aberration formulations, and Monte Carlo trajectory simulations.
   - Source for the anisotropic 2D Gaussian PSF model used in our
     defocus/astigmatism simulation.

5. **Bunday, B., et al. (2014–2020).**
   *"CD-SEM Metrology for Advanced Technology Nodes"* &
   *"Impact of SEM Image Noise on Line-Edge Roughness and Critical
   Dimension Measurement."*
   IEEE Transactions on Semiconductor Manufacturing / SPIE Advanced
   Lithography, Vol. 9050, 905009.
   - Comprehensive metrology analysis of SEM shot noise impact on
     measurement precision, power spectral density (PSD) models for
     line edge roughness (LER) and line width roughness (LWR).
   - Source for our LER simulation parameters (σ_LER ≈ 1.5 nm,
     correlation length ξ ≈ 20 nm, roughness exponent α ≈ 0.7).

---

## Sub-Pixel Registration Algorithm

6. **Guizar-Sicairos, M., Thurman, S. T., & Fienup, J. R. (2008).**
   *"Efficient subpixel image registration algorithms."*
   Optics Letters, 33(2), 156–158. DOI: 10.1364/OL.33.000156.
   - Efficient matrix-multiply DFT upsampling algorithm for sub-pixel
     image registration without full zero-padded FFT computation.
   - Achieves arbitrary sub-pixel precision (1/κ) with computational
     cost independent of image size.
   - Source for our Guizar-Sicairos sub-pixel refinement implementation.
