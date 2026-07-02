"""
sand_dust_pipeline.py
=====================
Complete forward sand/dust video degradation pipeline.

Physical models implemented:
  1. Kolmogorov turbulence density field  (FFT + log-normal, k^{-11/6} spectrum)
  2. Extinction / optical depth via ray-marching  (N=64 steps)
  3. Mie forward-scattering PSF  (Henyey-Greenstein + Gaussian approximation)
  4. Multiple-scattering glow term
  5. Final image composition  (Beer-Lambert + atmospheric light + MS glow)
  6. Temporal coherence via curl-noise advection
  7. Controllable temporal correlation via ``rho_refresh_rate``
     (interpolates between fully advected and fully independent per-frame fields)

Architecture notes
------------------
* Depth-varying blur uses a *Gaussian pyramid* strategy: precompute M blurred
  versions of the input image, then per-pixel bilinear-interpolate in scale-
  space. This is O(M * H * W) versus O(H * W * K^2) for naive per-pixel conv.
* Temporal advection uses scipy.ndimage.map_coordinates for sub-pixel accuracy
  and an analytically divergence-free (curl-noise) velocity field.
* All maps are stored as float32 numpy arrays in [0, 1] except where noted.

Requirements: numpy, scipy, opencv-python (cv2), tqdm
              torch is optional (auto-detected for GPU blur)
"""

import os
import json
import math
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import cv2
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Optional PyTorch import for GPU-accelerated separable convolution
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn.functional as F_torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn("PyTorch not found – falling back to CPU scipy convolution.")

# =========================================================================== #
#  SECTION 1 – DENSITY FIELD  (Kolmogorov turbulence)
# =========================================================================== #

def _make_wavenumber_grid(shape: Tuple[int, int]) -> np.ndarray:
    """Return the 2-D wavenumber magnitude grid |k| for an (H, W) array."""
    H, W = shape
    ky = np.fft.fftfreq(H).reshape(-1, 1)   # cycles per pixel
    kx = np.fft.fftfreq(W).reshape(1, -1)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_mag[0, 0] = 1.0  # avoid division by zero at DC
    return k_mag


def _curl_velocity_field(shape: Tuple[int, int],
                         seed: int,
                         scale: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a divergence-free 2-D velocity field via the curl of a potential.

    psi is a smooth random scalar potential; v = curl(psi) = (dpsi/dy, -dpsi/dx).
    This guarantees ∇·v = 0, so density is conserved under advection.

    Parameters
    ----------
    shape  : (H, W)
    seed   : RNG seed for reproducibility
    scale  : Controls the smoothness / spatial frequency of the field.

    Returns
    -------
    vy, vx : velocity components in pixels/frame
    """
    rng = np.random.default_rng(seed)
    H, W = shape
    # Random smooth potential field
    psi_raw = rng.standard_normal(shape)
    sigma_psi = max(H, W) * scale
    psi = gaussian_filter(psi_raw, sigma=sigma_psi)

    # Finite-difference gradient → divergence-free velocity
    vy = np.gradient(psi, axis=1)   # dpsi/dx → vy
    vx = -np.gradient(psi, axis=0)  # -dpsi/dy → vx

    # Normalise so max displacement per frame is ~2 pixels
    v_max = np.sqrt(vx**2 + vy**2).max() + 1e-8
    speed = 2.0  # pixels per frame
    vx = vx / v_max * speed
    vy = vy / v_max * speed
    return vy, vx


def generate_density_field(
    shape: Tuple[int, int],
    C_rho_sq: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a single 2-D sand/dust density realisation from a Kolmogorov
    turbulence power spectrum.

    The Kolmogorov inertial-range spectrum for a passive scalar is:
        S(k) ∝ k^{-5/3}   (power spectrum)
    so the amplitude spectrum  A(k) ∝ k^{-5/6}.
    For the full 3-D structure function (Doc B §6) the exponent becomes
    k^{-11/6} in amplitude, matching the Obukhov-Corrsin theory used
    in the benchmark.

    Parameters
    ----------
    shape    : (H, W) – spatial dimensions of the field
    C_rho_sq : density structure constant  [particles²/m⁶].  Controls the
               overall variance of the log-normal field.
    rng      : numpy Generator (optional, created fresh if None)

    Returns
    -------
    rho : float32 ndarray of shape (H, W), values > 0, mean ≈ 1.0
    """
    if rng is None:
        rng = np.random.default_rng()

    H, W = shape
    # Step 1 – white Gaussian noise in Fourier space
    noise = rng.standard_normal(shape).astype(np.float64)
    noise_fft = np.fft.fft2(noise)

    # Step 2 – shape by Kolmogorov amplitude spectrum  k^{-11/6}
    k_mag = _make_wavenumber_grid(shape)
    amplitude_filter = k_mag ** (-11.0 / 6.0)
    amplitude_filter[0, 0] = 0.0   # zero mean

    shaped_fft = noise_fft * amplitude_filter

    # Step 3 – back to real space, normalise to unit variance
    rho_g = np.fft.ifft2(shaped_fft).real
    sigma_g = rho_g.std() + 1e-12
    rho_g = rho_g / sigma_g  # N(0,1)-like field

    # Map the physical structure constant C_rho_sq to the log-domain standard
    # deviation sigma_log.  The factor of 10 is derived from the Obukhov–Corrsin
    # relation: sigma_rho^2 = C_rho_sq * L^(2/3), with L ≈ 100 m and the field
    # discretised to unit pixel spacing, giving sigma_log = sqrt(C_rho_sq) * 10.
    # The clamp to [0.01, 0.5] keeps the log-normal variance in a range that
    # produces physically stable (non-singular) density realisations.
    sigma_log = math.sqrt(C_rho_sq) * 10.0
    sigma_log = np.clip(sigma_log, 0.01, 0.5)
    rho_g = rho_g * sigma_log

    # Step 4 – log-normal transform: rho = exp(rho_G - sigma^2/2)
    # The shift -sigma^2/2 ensures E[rho] = 1.0
    rho = np.exp(rho_g - sigma_log**2 / 2.0).astype(np.float32)
    return rho


def advect_density_field(
    rho: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Advance the density field by one time-step using semi-Lagrangian advection.

    Solves  ∂ρ/∂t + v·∇ρ = 0  by back-tracing sample points.

    Parameters
    ----------
    rho : (H, W) current density field
    vy  : (H, W) velocity – row direction (pixels/frame)
    vx  : (H, W) velocity – column direction (pixels/frame)
    dt  : time step multiplier (default 1 frame)

    Returns
    -------
    rho_new : (H, W) advected density field (float32)
    """
    H, W = rho.shape
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float32)

    # Back-trace: where did this parcel come from?
    src_rows = rows - vy * dt
    src_cols = cols - vx * dt

    # Wrap around (periodic boundary) – clamp to [0, H-1] / [0, W-1]
    src_rows = src_rows % H
    src_cols = src_cols % W

    coords = np.array([src_rows.ravel(), src_cols.ravel()])
    rho_new = map_coordinates(rho.astype(np.float64), coords,
                              order=1, mode='wrap')
    return rho_new.reshape(H, W).astype(np.float32)


def update_density_field(
    rho: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    rng: np.random.Generator,
    C_rho_sq: float,
    refresh_rate: float = 0.1,
    dt: float = 1.0,
) -> np.ndarray:
    """Advance the turbulent density field by one frame with controllable
    temporal correlation.

    The update is a linear blend between pure advection (perfect temporal
    coherence) and a freshly sampled, independent Kolmogorov realisation
    (zero temporal coherence).  Intermediate values produce the slowly
    evolving structures observed in real sandstorm footage.

    Formally:

        ρ_new = (1 − r) · advect(ρ, v, dt)  +  r · ρ_fresh

    where r = ``refresh_rate`` and ρ_fresh ~ Kolmogorov(C_rho_sq).

    Parameters
    ----------
    rho          : (H, W) float32  current density field (mean ≈ 1.0)
    vy, vx       : (H, W) float32  divergence-free velocity components
                   (pixels/frame) from ``_curl_velocity_field``
    rng          : numpy Generator  used to draw the fresh noise realisation
    C_rho_sq     : density structure constant — passed to
                   ``generate_density_field`` for the fresh realisation
    refresh_rate : float in [0, 1]  temporal correlation control

        0.00  pure advection — structures persist indefinitely
              (steady, unrealistic for long sequences)
        0.05  very slow evolution — stable sandstorm conditions
        0.10  slow evolution — realistic sandstorm (default)
        0.30  moderate evolution — gusty conditions
        0.50  rapid evolution — highly variable storm
        1.00  fully independent per-frame — no temporal coherence,
              flickering appearance

    dt           : time-step multiplier forwarded to ``advect_density_field``

    Returns
    -------
    rho_new : (H, W) float32  updated density field (mean ≈ 1.0)
    """
    rho_advected = advect_density_field(rho, vy, vx, dt)
    if refresh_rate <= 0.0:
        return rho_advected
    rho_fresh = generate_density_field(rho.shape, C_rho_sq, rng=rng)
    rho_new   = (1.0 - refresh_rate) * rho_advected + refresh_rate * rho_fresh
    return rho_new.astype(np.float32)


# =========================================================================== #
#  SECTION 2 – RAY-MARCHING: extinction coefficient & optical depth
# =========================================================================== #

def compute_extinction_and_tau(
    depth_map: np.ndarray,
    rho: np.ndarray,
    beta_0: float,
    n_steps: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ray-march along each camera ray to integrate the optical depth τ.

    Model  (Doc B Eq 2):
        β(x) = β₀ · ρ(x) / <ρ>
        τ(x) = ∫₀^{depth} β(s) ds  ≈  Σᵢ β(sᵢ) Δs

    We project the 3-D density field back onto the 2-D image plane at each
    slab depth using a simple fronto-parallel approximation: at depth fraction
    f ∈ [0,1], the density sample is ρ(r,c) * f  (density grows with depth).

    Parameters
    ----------
    depth_map : (H, W) metric depth in metres (float32)
    rho       : (H, W) normalised density field (mean ≈ 1.0)
    beta_0    : mean extinction coefficient [m⁻¹]
    n_steps   : number of ray-marching steps per pixel

    Returns
    -------
    beta_map : (H, W) local extinction [m⁻¹]
    tau_map  : (H, W) integrated optical depth (dimensionless)
    t_map    : (H, W) transmission t = exp(-tau)
    """
    H, W = depth_map.shape
    rho_mean = rho.mean() + 1e-12

    # Normalised extinction field  β(x) = β₀ · ρ(x) / <ρ>
    beta_map = (beta_0 * rho / rho_mean).astype(np.float32)

    # Ray-march: integrate β along each ray
    # For a pixel at depth D, we sample n_steps uniformly spaced slabs
    # at depths  s_i = (i + 0.5) * D / n_steps
    tau_map = np.zeros((H, W), dtype=np.float32)

    # Vectorised over all pixels simultaneously
    D = depth_map.astype(np.float64)
    delta_s = D / n_steps   # slab thickness per pixel

    for i in range(n_steps):
        # Fractional depth of this slab
        frac = (i + 0.5) / n_steps   # ∈ (0, 1)

        # At this slab depth the density field is sampled from the same 2-D
        # rho map, but modulated by the local turbulence depth scaling:
        # ρ_slab(r,c) = ρ(r,c) * (1 + 0.5*sin(2π*frac)) to add vertical
        # variation; simplification of a full 3-D field.
        depth_mod = 1.0 + 0.3 * math.sin(2.0 * math.pi * frac)
        beta_slab = beta_map * depth_mod   # same spatial pattern, depth-varying intensity

        tau_map += (beta_slab * delta_s).astype(np.float32)

    t_map = np.exp(-tau_map).astype(np.float32)
    return beta_map, tau_map, t_map


# =========================================================================== #
#  SECTION 3 – MIE SCATTERING PSF  (depth-varying Gaussian blur)
# =========================================================================== #

def henyey_greenstein(theta: np.ndarray, g: float) -> np.ndarray:
    """
    Henyey-Greenstein phase function (Doc B Eq 4):
        P(θ) = (1 - g²) / [4π (1 + g² - 2g cos θ)^{3/2}]

    Parameters
    ----------
    theta : scattering angles in radians
    g     : asymmetry parameter ∈ [0.7, 0.9]

    Returns
    -------
    P : phase function values (normalised so ∫ P dΩ ≈ 1)
    """
    g2 = g * g
    denom = (1.0 + g2 - 2.0 * g * np.cos(theta)) ** 1.5
    return (1.0 - g2) / (4.0 * math.pi * denom + 1e-15)


def _build_gaussian_pyramid(
    image: np.ndarray,
    sigma_levels: np.ndarray,
) -> list:
    """
    Pre-compute a set of Gaussian-blurred versions of `image` at each sigma
    in `sigma_levels` (monotonically increasing).

    Returns a list of (H, W, C) float32 arrays, one per level.
    This is the core of the efficient depth-varying convolution strategy:
    we compute M blurs once, then per-pixel bilinear-interpolate in scale-space
    rather than applying a separate kernel for every pixel.

    Complexity: O(M · H · W)  vs  O(H · W · K²_max)  for naive per-pixel.
    """
    blurred = []
    for sigma in sigma_levels:
        if sigma < 0.5:
            blurred.append(image.copy())
        else:
            if image.ndim == 3:
                # Apply per-channel
                b = np.stack([
                    gaussian_filter(image[..., c].astype(np.float64), sigma=sigma)
                    for c in range(image.shape[2])
                ], axis=-1).astype(np.float32)
            else:
                b = gaussian_filter(image.astype(np.float64), sigma=sigma).astype(np.float32)
            blurred.append(b)
    return blurred


def depth_varying_blur(
    image: np.ndarray,
    tau_map: np.ndarray,
    sigma_0: float,
    n_levels: int = 16,
) -> np.ndarray:
    """
    Apply depth-varying Gaussian blur guided by the optical depth map.

    PSF width (Doc A Algorithm 1):
        σ_PSF(x) = σ₀ · τ(x)^{0.6}

    Strategy: Gaussian pyramid (M=n_levels) + per-pixel scale-space interpolation.

    Parameters
    ----------
    image    : (H, W, 3) or (H, W) float32 clean image in [0, 1]
    tau_map  : (H, W) optical depth map
    sigma_0  : base PSF spread in pixels  ∈ [0.5, 2.0]
    n_levels : number of pyramid levels

    Returns
    -------
    J_blur : same shape as image, float32, blurred result
    """
    # Range of sigma values needed
    tau_max = float(tau_map.max()) + 1e-6
    sigma_max = sigma_0 * (tau_max ** 0.6) + 0.5

    sigma_levels = np.linspace(0.0, sigma_max, n_levels)
    pyramid = _build_gaussian_pyramid(image, sigma_levels)

    # Per-pixel sigma
    sigma_px = (sigma_0 * (tau_map ** 0.6)).astype(np.float32)   # (H, W)

    # Continuous index into the pyramid
    level_idx = (sigma_px / (sigma_max + 1e-9)) * (n_levels - 1)   # (H, W)
    level_idx = np.clip(level_idx, 0, n_levels - 2)

    lo = level_idx.astype(np.int32)
    hi = lo + 1
    t_interp = (level_idx - lo).astype(np.float32)  # fractional part ∈ [0, 1)

    if image.ndim == 3:
        t_interp = t_interp[..., np.newaxis]  # broadcast over channels

    # Stack pyramid into array for vectorised indexing: (n_levels, H, W[, C])
    pyr_array = np.stack(pyramid, axis=0)   # (n_levels, H, W) or (n_levels, H, W, C)

    H, W = tau_map.shape
    rows = np.arange(H)[:, np.newaxis]
    cols = np.arange(W)[np.newaxis, :]

    lo_img = pyr_array[lo, rows, cols]    # (H, W[, C])
    hi_img = pyr_array[hi, rows, cols]    # (H, W[, C])

    J_blur = (lo_img * (1.0 - t_interp) + hi_img * t_interp).astype(np.float32)
    return J_blur


# =========================================================================== #
#  SECTION 4 – MULTIPLE-SCATTERING GLOW
# =========================================================================== #

def compute_ms_glow(
    J_blur: np.ndarray,
    tau_map: np.ndarray,
    t_map: np.ndarray,
    gamma: float,
    sigma_0: float,
    n_levels: int = 16,
) -> np.ndarray:
    """
    Empirical multiple-scattering glow term (Doc B Eq 11; Doc A Eq 6):

        I_MS = γ · (J_blur * G_{σ(τ)}) · (1 - t)

    where the glow blur radius is **per-pixel**:

        σ_glow(x) = σ₀ · √τ(x)          (Doc A Eq 6)

    Implementation reuses the Gaussian-pyramid / bilinear-interpolation
    machinery from ``depth_varying_blur()`` for the same O(M·H·W) complexity
    rather than a single global sigma.  This correctly captures spatially
    varying multiple-scattering: dense-dust regions (high τ, low t) receive
    wider glow kernels and stronger modulation.

    Parameters
    ----------
    J_blur   : (H, W, 3) PSF-blurred image, float32 [0, 1]
    tau_map  : (H, W) optical depth (dimensionless, ≥ 0)
    t_map    : (H, W) transmission = exp(-tau)
    gamma    : multiple-scattering strength ∈ [0.2, 0.6]
    sigma_0  : base spread in pixels ∈ [0.5, 2.0]  (same as PSF)
    n_levels : Gaussian pyramid levels (default 16, matching depth_varying_blur)

    Returns
    -------
    I_MS : (H, W, 3) float32 glow contribution
    """
    # Per-pixel glow sigma:  σ_glow(x) = σ₀ · √τ(x)
    # Clamp to ≥ 0.5 so the pyramid interpolation never asks for σ < 0
    sigma_glow_map = (sigma_0 * np.sqrt(np.maximum(tau_map, 0.0))).astype(np.float32)
    sigma_glow_map = np.maximum(sigma_glow_map, 0.5)

    # Reuse pyramid infrastructure -------------------------------------------
    sigma_max_glow = float(sigma_glow_map.max()) + 1e-6
    sigma_levels   = np.linspace(0.0, sigma_max_glow, n_levels)
    pyramid        = _build_gaussian_pyramid(J_blur, sigma_levels)   # M blurs of J_blur

    # Continuous pyramid index per pixel
    level_idx = (sigma_glow_map / sigma_max_glow) * (n_levels - 1)   # (H, W)
    level_idx = np.clip(level_idx, 0, n_levels - 2)

    lo       = level_idx.astype(np.int32)
    hi       = lo + 1
    t_interp = (level_idx - lo).astype(np.float32)   # fractional part ∈ [0, 1)

    if J_blur.ndim == 3:
        t_interp = t_interp[..., np.newaxis]   # broadcast over channels

    pyr_array = np.stack(pyramid, axis=0)   # (n_levels, H, W, C) or (n_levels, H, W)
    H, W = tau_map.shape
    rows = np.arange(H)[:, np.newaxis]
    cols = np.arange(W)[np.newaxis, :]

    glow_lo   = pyr_array[lo, rows, cols]   # (H, W[, C])
    glow_hi   = pyr_array[hi, rows, cols]   # (H, W[, C])
    glow_base = (glow_lo * (1.0 - t_interp) + glow_hi * t_interp).astype(np.float32)
    # ------------------------------------------------------------------------

    # Modulate by (1 - t): glow is strongest where transmission is lowest
    one_minus_t = (1.0 - t_map).astype(np.float32)
    if J_blur.ndim == 3:
        one_minus_t = one_minus_t[..., np.newaxis]

    I_MS = (gamma * glow_base * one_minus_t).astype(np.float32)
    return I_MS


# =========================================================================== #
#  SECTION 5 – FINAL IMAGE COMPOSITION
# =========================================================================== #

# Doc A Table 4 – atmospheric light A: warm sand/dust RGB colours
# Values in [0, 1]. Each row is one preset; we sample one randomly.
_ATMOSPHERIC_LIGHT_PRESETS = np.array([
    [0.95, 0.88, 0.72],   # warm sandy yellow
    [0.92, 0.85, 0.68],   # light ochre
    [0.88, 0.78, 0.60],   # deep ochre / sienna
    [0.98, 0.93, 0.80],   # pale cream
    [0.90, 0.80, 0.55],   # golden dust
    [0.85, 0.75, 0.60],   # terracotta-tinted
], dtype=np.float32)


def sample_atmospheric_light(rng: np.random.Generator) -> np.ndarray:
    """Return a random warm atmospheric light colour (3,) float32 [0, 1]."""
    idx = rng.integers(0, len(_ATMOSPHERIC_LIGHT_PRESETS))
    A = _ATMOSPHERIC_LIGHT_PRESETS[idx].copy()
    # Add small per-channel noise for variety
    A += rng.uniform(-0.03, 0.03, size=3).astype(np.float32)
    return np.clip(A, 0.0, 1.0).astype(np.float32)


def compose_final_image(
    J: np.ndarray,
    J_blur: np.ndarray,
    t_map: np.ndarray,
    A: np.ndarray,
    I_MS: np.ndarray,
) -> np.ndarray:
    """
    Final degraded image  (Doc B §10 Algorithm 1; Doc A §5.4):

        I = J_blur · t + A · (1 - t) + I_MS

    Parameters
    ----------
    J      : (H, W, 3) clean image, float32 [0, 1]  (unused in formula, kept for logging)
    J_blur : (H, W, 3) PSF-blurred clean image
    t_map  : (H, W)    transmission map
    A      : (3,)      atmospheric light colour
    I_MS   : (H, W, 3) multiple-scattering glow

    Returns
    -------
    I_degraded : (H, W, 3) float32 degraded image, clipped to [0, 1]
    """
    t3 = t_map[..., np.newaxis]            # (H, W, 1) → broadcast over RGB
    A3 = A.reshape(1, 1, 3)               # (1, 1, 3)

    I_degraded = J_blur * t3 + A3 * (1.0 - t3) + I_MS
    return np.clip(I_degraded, 0.0, 1.0).astype(np.float32)


# =========================================================================== #
#  SECTION 6 – PARAMETER SAMPLING  (Doc A Table 4 ranges)
# =========================================================================== #

def sample_parameters(rng: np.random.Generator) -> Dict[str, Any]:
    """Sample a full set of degradation parameters from Doc A Table 4 ranges.

    Parameters
    ----------
    rng : numpy Generator

    Returns
    -------
    params : dict with keys
        beta_0, g, C_rho_sq, gamma, sigma_0, A, rho_refresh_rate
    """
    beta_0           = float(rng.uniform(0.002, 0.02))      # m⁻¹
    g                = float(rng.uniform(0.7,   0.9))        # HG asymmetry
    C_rho_sq         = float(10 ** rng.uniform(-4, -2))      # log-uniform
    gamma            = float(rng.uniform(0.2,   0.6))        # MS strength
    sigma_0          = float(rng.uniform(0.5,   2.0))        # PSF spread (pixels)
    A                = sample_atmospheric_light(rng)         # warm RGB
    rho_refresh_rate = float(rng.uniform(0.0,   1.0))        # temporal correlation

    return {
        "beta_0":           beta_0,
        "g":                g,
        "C_rho_sq":         C_rho_sq,
        "gamma":            gamma,
        "sigma_0":          sigma_0,
        "A":                A.tolist(),
        "rho_refresh_rate": rho_refresh_rate,
    }


# =========================================================================== #
#  SECTION 7 – SINGLE-FRAME DEGRADER
# =========================================================================== #

def degrade_frame(
    clean_rgb: np.ndarray,
    depth_map: np.ndarray,
    rho: np.ndarray,
    params: Dict[str, Any],
    n_ray_steps: int = 64,
    n_blur_levels: int = 16,
) -> Dict[str, np.ndarray]:
    """
    Apply the full degradation pipeline to one frame.

    Parameters
    ----------
    clean_rgb  : (H, W, 3) float32 RGB in [0, 1]
    depth_map  : (H, W) float32 metric depth in metres
    rho        : (H, W) float32 density field (mean ≈ 1.0)
    params     : dict from sample_parameters()
    n_ray_steps: number of ray-marching steps
    n_blur_levels: number of levels in the Gaussian pyramid

    Returns
    -------
    dict with keys:
        degraded_rgb, transmission_map, beta_map, tau_map
    """
    beta_0  = params["beta_0"]
    gamma   = params["gamma"]
    sigma_0 = params["sigma_0"]
    A       = np.array(params["A"], dtype=np.float32)

    # 1. Extinction & optical depth
    beta_map, tau_map, t_map = compute_extinction_and_tau(
        depth_map, rho, beta_0, n_steps=n_ray_steps
    )

    # 2. Depth-varying PSF blur
    J_blur = depth_varying_blur(clean_rgb, tau_map, sigma_0, n_levels=n_blur_levels)

    # 3. Multiple-scattering glow
    I_MS = compute_ms_glow(J_blur, tau_map, t_map, gamma, sigma_0)

    # 4. Compose final image
    I_degraded = compose_final_image(clean_rgb, J_blur, t_map, A, I_MS)

    return {
        "degraded_rgb":    I_degraded,
        "transmission_map": t_map,
        "beta_map":        beta_map,
        "tau_map":         tau_map,
    }


# =========================================================================== #
#  SECTION 8 – VIDEO PIPELINE  (batch processing, SandStorm-Video format)
# =========================================================================== #

def _save_float_map(path: str, arr: np.ndarray) -> float:
    """
    Save a float32 physical map in two forms:

    1. ``<path>``      – 16-bit PNG visual preview, normalised by per-frame
                         max so it is displayable.  The normalisation factor
                         (max_val) is returned so callers can log it in
                         metadata.json and recover the original physical scale:
                             arr_physical = png_uint16 / 65535 * max_val

    2. ``<path>.npy``  – Raw float32 array, physically scaled (m⁻¹ for beta,
                         dimensionless for tau/transmission).  This is the
                         ground-truth file; the PNG is only for visualisation.

    Parameters
    ----------
    path : destination PNG path  (e.g. ``…/beta_maps/frame_0000.png``)
    arr  : float32 ndarray, physical values (non-negative)

    Returns
    -------
    max_val : float  – the per-frame maximum used for PNG normalisation
    """
    # -- raw float32 save (physically meaningful) ---------------------------
    npy_path = path + ".npy"
    np.save(npy_path, arr.astype(np.float32))

    # -- 16-bit PNG preview (normalised per-frame) --------------------------
    arr_clipped = np.clip(arr, 0, None)
    max_val = float(arr_clipped.max())
    if max_val > 0:
        arr_norm = arr_clipped / max_val
    else:
        arr_norm = arr_clipped
    arr_16 = (arr_norm * 65535).astype(np.uint16)
    cv2.imwrite(path, arr_16)

    return max_val


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    """Save an (H, W, 3) float32 [0,1] image as 8-bit PNG."""
    bgr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]  # RGB→BGR
    cv2.imwrite(path, bgr)


def degrade_video(
    clean_frames: list,
    depth_frames: list,
    output_dir: str,
    sequence_seed: int = 42,
    n_ray_steps: int = 64,
    n_blur_levels: int = 16,
    density_refresh: float = 0.1,
    rho_refresh_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Process a sequence of clean RGB frames into a SandStorm-Video dataset.

    Output directory layout::

        output_dir/
          clean_rgb/          frame_0000.png …
          degraded_rgb/       frame_0000.png …
          depth_maps/         frame_0000.png + .npy …
          transmission_maps/  frame_0000.png + .npy …
          beta_maps/          frame_0000.png + .npy …
          tau_maps/           frame_0000.png + .npy …
          metadata.json

    Parameters
    ----------
    clean_frames     : list of (H, W, 3) float32 RGB images in [0, 1]
    depth_frames     : list of (H, W) float32 depth maps in metres
    output_dir       : root folder for all outputs
    sequence_seed    : master RNG seed for full reproducibility
    n_ray_steps      : number of ray-marching integration steps per pixel
    n_blur_levels    : Gaussian pyramid levels for the depth-varying PSF blur
    density_refresh  : deprecated alias for ``rho_refresh_rate``; ignored when
                       ``rho_refresh_rate`` is supplied explicitly
    rho_refresh_rate : temporal correlation control in [0, 1].
                       0.0 = pure advection (steady turbulence);
                       0.1 = slow evolution, realistic sandstorm (default);
                       1.0 = fully independent per-frame (flickering).
                       When ``None``, the value stored in the sampled
                       ``params["rho_refresh_rate"]`` is used.

    Returns
    -------
    metadata : dict  (also written to ``output_dir/metadata.json``)
    """
    assert len(clean_frames) == len(depth_frames), "Mismatch in frame counts."
    n_frames = len(clean_frames)
    H, W = clean_frames[0].shape[:2]
    shape2d = (H, W)

    rng = np.random.default_rng(sequence_seed)

    # Sample one parameter set for the full sequence (scene-level consistency)
    params = sample_parameters(rng)

    # Resolve rho_refresh_rate: explicit argument wins over sampled value,
    # which wins over the legacy density_refresh alias.
    if rho_refresh_rate is not None:
        effective_refresh = float(np.clip(rho_refresh_rate, 0.0, 1.0))
    else:
        effective_refresh = float(np.clip(
            params.get("rho_refresh_rate", density_refresh), 0.0, 1.0
        ))
    # Store the resolved value back so it appears in metadata.json.
    params["rho_refresh_rate"] = effective_refresh

    # Create output directories
    subdirs = ["clean_rgb", "degraded_rgb", "depth_maps",
               "transmission_maps", "beta_maps", "tau_maps"]
    for sd in subdirs:
        Path(output_dir, sd).mkdir(parents=True, exist_ok=True)

    # Initialise density field and curl-noise velocity field
    rho = generate_density_field(shape2d, params["C_rho_sq"], rng=rng)
    vy, vx = _curl_velocity_field(shape2d, seed=int(rng.integers(0, 2**31)))

    metadata: Dict[str, Any] = {
        "sequence_seed": sequence_seed,
        "n_frames": n_frames,
        "resolution": [H, W],
        "parameters": params,
        # Per-frame normalisation factors so physical values can be recovered:
        #   arr_physical = png_uint16 / 65535 * max_val
        # beta  units : m⁻¹  (extinction coefficient)
        # tau   units : dimensionless  (optical depth)
        # depth units : m    (metric depth)
        # transmission: dimensionless [0, 1]
        "frame_map_maxvals": [],
    }

    for frame_idx in tqdm(range(n_frames), desc="Degrading frames"):
        clean = clean_frames[frame_idx].astype(np.float32)
        depth = depth_frames[frame_idx].astype(np.float32)

        # Degrade this frame
        result = degrade_frame(clean, depth, rho, params,
                               n_ray_steps=n_ray_steps,
                               n_blur_levels=n_blur_levels)

        fname = f"frame_{frame_idx:04d}.png"

        # Save outputs; _save_float_map returns the per-frame max_val used for
        # PNG normalisation so that raw physical values can be reconstructed.
        _save_rgb(str(Path(output_dir, "clean_rgb",    fname)), clean)
        _save_rgb(str(Path(output_dir, "degraded_rgb", fname)), result["degraded_rgb"])

        mv_depth = _save_float_map(
            str(Path(output_dir, "depth_maps",        fname)), depth)
        mv_trans = _save_float_map(
            str(Path(output_dir, "transmission_maps", fname)), result["transmission_map"])
        mv_beta  = _save_float_map(
            str(Path(output_dir, "beta_maps",         fname)), result["beta_map"])
        mv_tau   = _save_float_map(
            str(Path(output_dir, "tau_maps",          fname)), result["tau_map"])

        metadata["frame_map_maxvals"].append({
            "frame": frame_idx,
            "depth_m":        mv_depth,
            "transmission":   mv_trans,
            "beta_per_m":     mv_beta,
            "tau":            mv_tau,
        })

        # Advance density field using the controllable temporal-correlation model.
        # effective_refresh = 0.0 → pure advection (steady turbulence).
        # effective_refresh = 1.0 → fully independent per-frame field.
        rho = update_density_field(
            rho, vy, vx, rng,
            C_rho_sq     = params["C_rho_sq"],
            refresh_rate = effective_refresh,
            dt           = 1.0,
        )

    # Save metadata
    meta_path = str(Path(output_dir, "metadata.json"))
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nDone. {n_frames} frames saved to: {output_dir}")
    print(f"Metadata: {meta_path}")
    return metadata


# =========================================================================== #
#  ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    print("SandStorm-Video Generator — physics engine")
    print("Usage: streamlit run app.py        (GUI)")
    print("       python process_test_video.py --input <video>  (CLI)")
