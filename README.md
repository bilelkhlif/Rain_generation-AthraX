# SandStorm-Video Generator

Physics-based sand and dust video degradation for autonomous driving datasets.
The pipeline renders temporally coherent degraded video pairs — clean RGB,
degraded RGB, and physical ground-truth maps (optical depth, transmission,
extinction coefficient) — from any input video and a monocular depth estimate.

---

## Table of Contents

1. [Features](#features)
2. [Physical Model](#physical-model)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Parameter Reference](#parameter-reference)
6. [Output Structure](#output-structure)
7. [Citation](#citation)
8. [License](#license)

---

## Features

- Kolmogorov turbulence density field with k^{−11/6} amplitude spectrum and log-normal transform
- Beer–Lambert spectral extinction with per-pixel optical depth computed by ray-marching (N=64 steps)
- Mie forward-scattering PSF applied via Gaussian pyramid scale-space interpolation — O(M·H·W)
- Per-pixel multiple-scattering glow using the Henyey–Greenstein phase function
- Temporally coherent sequences via curl-noise advection and controllable `rho_refresh_rate`
- Per-frame MiDaS monocular depth estimation (Intel/dpt-hybrid-midas, ~500 MB, auto-downloaded)
- Streamlit GUI with interactive parameter sliders and real-time progress
- CLI for batch dataset generation
- Raw float32 `.npy` ground-truth maps alongside 16-bit PNG previews

---

## Physical Model

### 1 — Kolmogorov turbulence density field

```
ρ(x) = exp( ρ_G(x) − σ²/2 )      ρ_G ~ N(0, σ²)
```

White noise is shaped by k^{−11/6} in Fourier space (Obukhov–Corrsin inertial
range for a passive scalar) and passed through a log-normal transform to
guarantee a positive-definite concentration field.

### 2 — Extinction and optical depth (ray-marching)

```
β(x)  = β₀ · ρ(x) / ⟨ρ⟩              (Doc B, Eq. 2)
τ(x)  = Σᵢ β(sᵢ) Δs    (N = 64 steps)
t(x)  = exp(−τ(x))                    Beer–Lambert
```

### 3 — Mie PSF (depth-varying Gaussian blur)

```
P(θ) = (1 − g²) / [ 4π (1 + g² − 2g cosθ)^{3/2} ]   (HG phase function)
σ_PSF(x) = σ₀ · τ(x)^{0.6}                            (Doc A, Alg. 1)
```

### 4 — Multiple-scattering glow and final composition

```
I_MS = γ · (J_blur * G_{σ₀√τ}) · (1 − t)             (Doc B, Eq. 11)
I    = J_blur · t  +  A · (1 − t)  +  I_MS            (Doc A, §5.4)
```

---

## Installation

```bash
git clone https://github.com/bilelkhlif/SandStorm_generation-AthraX.git
cd SandStorm_generation-AthraX
pip install -r requirements.txt
```

### Depth model (MiDaS)

On the **first run** the pipeline downloads Intel/dpt-hybrid-midas (~500 MB)
from HuggingFace Hub automatically and caches it in `~/.cache/huggingface/hub`.
No manual download is required.

If you prefer to work offline, place the model files in a folder named
`midas_model/` in the project root:

```
midas_model/
  model.safetensors      (or pytorch_model.bin)
  config.json
  preprocessor_config.json
```

Direct download links:

```
https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/model.safetensors
https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/config.json
https://huggingface.co/Intel/dpt-hybrid-midas/resolve/main/preprocessor_config.json
```

If you downloaded `pytorch_model.bin` instead of `model.safetensors`, convert
it once with:

```bash
python -c "
import torch
from safetensors.torch import save_file
state = torch.load('midas_model/pytorch_model.bin', map_location='cpu', weights_only=True)
save_file(state, 'midas_model/model.safetensors')
"
```

---

## Usage

### GUI (recommended)

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a video, adjust parameters, click **Run pipeline**.

### CLI

```bash
python process_test_video.py --input path/to/video.mp4 --output_dir output/
```

All parameters are sampled from Table 4 ranges with `--seed 42` by default.

---

## Parameter Reference

| Symbol | Name | Range | Default | Physical meaning |
|--------|------|-------|---------|-----------------|
| β₀ | Mean extinction | 0.002 – 0.020 m⁻¹ | 0.008 | Light attenuation per metre; controls storm opacity |
| g | HG asymmetry | 0.70 – 0.90 | 0.80 | Forward-scattering strength of dust particles |
| C²ᵨ | Turbulence constant | 10⁻⁴ – 10⁻² | 10⁻³ | Spatial patchiness of particle concentration |
| γ | MS glow strength | 0.20 – 0.60 | 0.40 | Intensity of multiple-scattering halo |
| σ₀ | PSF base spread | 0.5 – 2.0 px | 1.0 | Blur radius at τ = 1 |
| **A** | Atmospheric light | RGB colour | #F0D8A0 | Warm dust-tone colour filling occluded regions |
| r | Refresh rate | 0.0 – 1.0 | 0.10 | Temporal correlation: 0 = steady, 1 = per-frame random |
| N | Ray steps | 16 – 128 | 64 | Integration accuracy along each camera ray |

---

## Output Structure

```
output_sandstorm/
├── clean_rgb/              frame_NNNN.png   — 8-bit RGB input
├── degraded_rgb/           frame_NNNN.png   — 8-bit RGB degraded output
├── depth_maps/             frame_NNNN.png   — 16-bit preview
│                           frame_NNNN.png.npy — float32 depth [m]
├── transmission_maps/      frame_NNNN.png + .npy — t ∈ [0, 1]
├── beta_maps/              frame_NNNN.png + .npy — β [m⁻¹]
├── tau_maps/               frame_NNNN.png + .npy — τ (dimensionless)
├── sandstorm_video.mp4     — re-encoded degraded video
└── metadata.json           — all parameters + per-frame normalisation factors
```

### Recovering physical units from PNG previews

Each 16-bit PNG is normalised by its per-frame maximum before saving.
The raw physical value is always in the `.npy` file, or can be recovered:

```python
import numpy as np, cv2, json

meta    = json.load(open("output_sandstorm/metadata.json"))
max_val = meta["frame_map_maxvals"][0]["beta_per_m"]   # m⁻¹
png     = cv2.imread("output_sandstorm/beta_maps/frame_0000.png", cv2.IMREAD_UNCHANGED)
beta    = png / 65535.0 * max_val   # physical β in m⁻¹
```

---

## Citation

```bibtex
@article{khlif2026sandstorm,
  title   = {SandStorm-Video: A Benchmark for Sand and Dust Video Degradation},
  author  = {Khlif, Bilel et al.},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
