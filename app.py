"""
app.py
======
Streamlit interface for the SandStorm-Video Generator.

All long-running operations run in the main thread with live progress
feedback.  Backend logic lives entirely in sand_dust_pipeline.py and
process_test_video.py — this module is UI only.

Usage
-----
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------------------
# Physics engine
# ---------------------------------------------------------------------------
try:
    from sand_dust_pipeline import degrade_video
except ImportError as _exc:
    st.error(
        f"Cannot import sand_dust_pipeline: {_exc}  "
        "Ensure sand_dust_pipeline.py is in the same directory."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Depth-estimation helpers
# ---------------------------------------------------------------------------
try:
    from process_test_video import (
        _load_midas,
        _estimate_depth_midas,
        _fake_depth_for_video,
        DEPTH_MIN_M,
        DEPTH_MAX_M,
    )
    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False
    DEPTH_MIN_M, DEPTH_MAX_M = 2.0, 200.0

# =========================================================================== #
#  PAGE CONFIG & GLOBAL CSS
# =========================================================================== #

st.set_page_config(
    page_title="SandStorm-Video Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Typography ─────────────────────────────────────────────────────────── */
h1 { font-size: 1.55rem !important; font-weight: 700 !important;
     letter-spacing: -0.01em; color: #0f1117; }
h2 { font-size: 1.1rem  !important; font-weight: 600 !important;
     color: #0f1117; margin-top: 1.4rem !important; }
h3 { font-size: 0.9rem  !important; font-weight: 600 !important;
     text-transform: uppercase; letter-spacing: 0.06em;
     color: #555; margin-top: 1rem !important; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #f7f8fa; }
section[data-testid="stSidebar"] h2 {
    font-size: 0.75rem !important; text-transform: uppercase;
    letter-spacing: 0.08em; color: #888; border-bottom: 1px solid #e0e0e0;
    padding-bottom: 4px; margin-bottom: 0.5rem !important; }
section[data-testid="stSidebar"] .stSlider > label,
section[data-testid="stSidebar"] .stSelectbox > label,
section[data-testid="stSidebar"] .stNumberInput > label,
section[data-testid="stSidebar"] .stColorPicker > label {
    font-size: 0.82rem !important; color: #333; font-weight: 500; }

/* ── Metric cards ────────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: #f7f8fa;
    border: 1px solid #e8e8e8;
    border-radius: 6px;
    padding: 14px 18px !important;
}
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important; color: #666 !important;
    text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.45rem !important; font-weight: 700 !important; color: #0f1117; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
div[data-testid="stDownloadButton"] button {
    width: 100%; font-size: 0.82rem; font-weight: 500;
    border-radius: 4px; padding: 6px 14px; }

/* ── Upload zone ─────────────────────────────────────────────────────────── */
section[data-testid="stFileUploadDropzone"] {
    border: 1.5px dashed #c8ccd4 !important;
    border-radius: 6px; background: #fafbfc; }

/* ── Info / warning boxes ───────────────────────────────────────────────── */
div[data-testid="stAlert"] { border-radius: 4px; font-size: 0.85rem; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border: none; border-top: 1px solid #e8e8e8; margin: 1.2rem 0; }

/* ── Code block (log) ───────────────────────────────────────────────────── */
pre { font-size: 0.78rem !important; background: #f3f4f6 !important; }
</style>
""", unsafe_allow_html=True)

# =========================================================================== #
#  SESSION STATE DEFAULTS
# =========================================================================== #

_DEFAULTS: Dict[str, Any] = {
    "beta_0":            0.008,
    "g":                 0.80,
    "C_rho_sq_log":      -3.0,
    "gamma":             0.40,
    "sigma_0":           1.0,
    "A_hex":             "#F0D8A0",
    "sequence_seed":     42,
    "n_ray_steps":       64,
    "rho_refresh_rate":  0.10,
    "depth_source":      "Auto (MiDaS)",
    "run_done":          False,
    "output_dir":        None,
    "log_lines":         [],
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# =========================================================================== #
#  UTILITY HELPERS
# =========================================================================== #

def _hex_to_rgb01(hex_str: str) -> np.ndarray:
    """Convert a CSS hex colour string to a float32 [0, 1] RGB array."""
    h = hex_str.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)], dtype=np.float32)


def _log(msg: str) -> None:
    """Append a timestamped entry to the session-state processing log."""
    st.session_state["log_lines"].append(msg)


def _build_params() -> Dict[str, Any]:
    """Assemble a pipeline-compatible parameter dict from session state."""
    return {
        "beta_0":           st.session_state["beta_0"],
        "g":                st.session_state["g"],
        "C_rho_sq":         10.0 ** st.session_state["C_rho_sq_log"],
        "gamma":            st.session_state["gamma"],
        "sigma_0":          st.session_state["sigma_0"],
        "A":                _hex_to_rgb01(st.session_state["A_hex"]).tolist(),
        "rho_refresh_rate": st.session_state["rho_refresh_rate"],
    }


def _npy_paths(output_dir: Path) -> List[Path]:
    """Return all .npy physical-map files under output_dir, sorted."""
    return sorted(output_dir.rglob("*.npy"))


def _load_rgb(path: Path) -> Optional[np.ndarray]:
    """Load a BGR PNG from disk and return it as uint8 RGB, or None."""
    bgr = cv2.imread(str(path))
    return bgr[..., ::-1] if bgr is not None else None


# =========================================================================== #
#  DEPTH ESTIMATION
# =========================================================================== #

def _planar_depth(H: int, W: int) -> np.ndarray:
    """Return a synthetic planar depth gradient (far at top, near at bottom)."""
    depth = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        depth[r, :] = DEPTH_MAX_M - (r / H) * (DEPTH_MAX_M - DEPTH_MIN_M)
    return depth


def _estimate_depth_batch(
    frames: List[np.ndarray],
    depth_source: str,
    stage_bar,
    status,
) -> List[np.ndarray]:
    """Estimate or generate depth maps for all frames.

    Parameters
    ----------
    frames:
        List of uint8 RGB frames.
    depth_source:
        ``"Auto (MiDaS)"`` or ``"Synthetic (testing only)"``.
    stage_bar:
        ``st.progress`` object updated per frame.
    status:
        ``st.empty`` placeholder for status text.

    Returns
    -------
    List[np.ndarray]
        Per-frame float32 depth maps in pseudo-metric metres.
    """
    n = len(frames)
    H, W = frames[0].shape[:2]

    if depth_source == "Synthetic (testing only)":
        fake = _fake_depth_for_video(H, W) if _CLI_AVAILABLE else _planar_depth(H, W)
        stage_bar.progress(1.0)
        status.text("Depth: synthetic planar gradient applied to all frames.")
        _log(f"[depth] Synthetic planar depth ({H}x{W}) applied to all {n} frames.")
        return [fake] * n

    status.text("Loading depth model from local folder...")
    _load_midas()
    depth_frames = []
    for i, frame in enumerate(frames):
        depth_frames.append(_estimate_depth_midas(frame))
        stage_bar.progress((i + 1) / n)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            status.text(f"Depth estimation: {i+1}/{n} frames")
    _log(f"[depth] MiDaS depth estimated for {n} frames.")
    return depth_frames


# =========================================================================== #
#  VIDEO I/O
# =========================================================================== #

def _extract_frames(video_bytes: bytes) -> Tuple[List[np.ndarray], float, int, int]:
    """Decode a video from raw bytes.  Writes to a temp file for OpenCV.

    Returns
    -------
    Tuple[list, float, int, int]
        ``(frames_rgb, fps, width, height)``
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap    = cv2.VideoCapture(tmp_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames: List[np.ndarray] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frames.append(bgr[..., ::-1].copy())

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)
    return frames, fps, width, height


def _encode_video(
    frames_rgb: List[np.ndarray],
    fps: float,
    width: int,
    height: int,
) -> bytes:
    """Encode RGB frames to an in-memory MP4 byte string."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    writer = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    for frame in frames_rgb:
        bgr = frame[..., ::-1]
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height))
        writer.write(bgr)
    writer.release()

    data = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink(missing_ok=True)
    return data


def _zip_npy_maps(output_dir: Path) -> bytes:
    """Compress all .npy physical maps under output_dir into a ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in _npy_paths(output_dir):
            zf.write(p, p.relative_to(output_dir))
    return buf.getvalue()

# =========================================================================== #
#  PIPELINE RUNNER
# =========================================================================== #

def run_pipeline(
    frames_uint8: List[np.ndarray],
    fps: float,
    width: int,
    height: int,
    params: Dict[str, Any],
    depth_source: str,
    seed: int,
    n_ray_steps: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute the full degradation pipeline with live Streamlit progress.

    Stages
    ------
    1. Depth estimation  (MiDaS or synthetic)
    2. float32 conversion
    3. Physics-based degradation  (sand_dust_pipeline.degrade_video)
    4. Video encoding  (OpenCV mp4v)

    Returns
    -------
    dict
        Annotated metadata dict written to ``output_dir/metadata.json``.
    """
    n_frames = len(frames_uint8)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall   = st.progress(0, text="Initialising...")
    stage_bar = st.progress(0)
    status    = st.empty()

    # Stage 1 — depth
    overall.progress(5, text="Stage 1 / 4  —  Depth Estimation")
    depth_frames = _estimate_depth_batch(frames_uint8, depth_source, stage_bar, status)
    overall.progress(25, text="Stage 2 / 4  —  Pre-processing")

    # Stage 2 — float32 conversion
    status.text("Converting frames to float32 RGB...")
    clean_frames = [f.astype(np.float32) / 255.0 for f in frames_uint8]
    stage_bar.progress(100)
    _log("[preprocess] Converted all frames to float32 [0, 1].")
    overall.progress(30, text="Stage 3 / 4  —  Degradation")

    # Stage 3 — physics degradation
    status.text("Applying sand/dust degradation model...")
    metadata = degrade_video(
        clean_frames     = clean_frames,
        depth_frames     = depth_frames,
        output_dir       = str(output_dir),
        sequence_seed    = seed,
        n_ray_steps      = n_ray_steps,
        n_blur_levels    = 16,
        rho_refresh_rate = params.get("rho_refresh_rate", 0.1),
    )
    _log(f"[degrade] {n_frames} frames degraded. Output: {output_dir}")
    overall.progress(85, text="Stage 4 / 4  —  Video Encoding")

    # Stage 4 — re-encode to MP4
    status.text("Encoding output video...")
    degraded_pngs = sorted((output_dir / "degraded_rgb").glob("frame_*.png"))
    degraded_rgb  = []
    for i, p in enumerate(degraded_pngs):
        img = _load_rgb(p)
        if img is not None:
            degraded_rgb.append(img)
        stage_bar.progress(int((i + 1) / max(len(degraded_pngs), 1) * 100))

    video_bytes = _encode_video(degraded_rgb, fps, width, height)
    video_out   = output_dir / "sandstorm_video.mp4"
    video_out.write_bytes(video_bytes)
    _log(f"[encode] Output video: {video_out}")

    # Annotate metadata with depth provenance
    depth_label = (
        f"Intel/dpt-hybrid-midas — relative inverse depth rescaled to "
        f"[{DEPTH_MIN_M}, {DEPTH_MAX_M}] m  (not calibrated metric ground truth)"
        if depth_source == "Auto (MiDaS)"
        else "Synthetic planar depth gradient — for pipeline testing only"
    )
    metadata["depth_source"]     = depth_label
    metadata["input_resolution"] = [height, width]
    metadata["source_fps"]       = fps

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    overall.progress(100, text="Complete.")
    status.text("Pipeline finished.")
    _log("[pipeline] Done.")
    return metadata


# =========================================================================== #
#  SIDEBAR
# =========================================================================== #

def _render_sidebar() -> None:
    """Render all parameter controls in the sidebar.

    Each widget is bound to session state via its ``key`` argument so values
    survive reruns.  Labels use the standard physical notation from Table 4
    of Khlif et al. (2026).
    """
    sb = st.sidebar

    sb.markdown("## Parameters")
    sb.caption("Ranges follow Khlif et al. (2026), Table 4.")

    # ── Extinction ──────────────────────────────────────────────────────────
    sb.markdown("### Extinction & Scattering")

    sb.slider(
        "β₀  —  mean extinction coefficient  (m⁻¹)",
        min_value=0.002, max_value=0.020, step=0.001,
        key="beta_0",
        help=(
            "Specifies how much light is attenuated per metre of propagation "
            "through the particle cloud.  "
            "0.002 corresponds to light haze; 0.020 to a severe sandstorm."
        ),
    )

    sb.slider(
        "g  —  Henyey–Greenstein asymmetry",
        min_value=0.70, max_value=0.90, step=0.01,
        key="g",
        help=(
            "Controls the angular distribution of single-scattered radiance.  "
            "g = 1 is perfectly forward-scattering; g = 0 is isotropic.  "
            "Sand and dust particles are strongly forward-scattering."
        ),
    )

    # ── Turbulence ──────────────────────────────────────────────────────────
    sb.markdown("### Turbulence Structure")

    sb.slider(
        "log₁₀(C²ᵨ)  —  density structure constant",
        min_value=-4.0, max_value=-2.0, step=0.1,
        key="C_rho_sq_log",
        help=(
            "The Obukhov–Corrsin density structure constant C²ᵨ on a log₁₀ scale.  "
            "−4 produces a nearly uniform concentration field; "
            "−2 produces strong spatial patchiness."
        ),
    )
    sb.caption(f"C²ᵨ = {10.0 ** st.session_state['C_rho_sq_log']:.2e}  (particles/m³)²")

    # ── Multiple scattering ─────────────────────────────────────────────────
    sb.markdown("### Multiple Scattering")

    sb.slider(
        "γ  —  multiple-scattering glow strength",
        min_value=0.20, max_value=0.60, step=0.05,
        key="gamma",
        help=(
            "Scales the glow term I_MS (Eq. 11).  Higher values produce "
            "brighter halos and more diffuse back-fill in optically dense regions."
        ),
    )

    sb.slider(
        "σ₀  —  PSF base spread  (pixels)",
        min_value=0.5, max_value=2.0, step=0.1,
        key="sigma_0",
        help=(
            "Base width of the Mie-scattering point-spread function at τ = 1.  "
            "The per-pixel blur radius scales as σ_PSF = σ₀ · τ^0.6."
        ),
    )

    # ── Atmospheric light ───────────────────────────────────────────────────
    sb.markdown("### Atmospheric Light")

    sb.color_picker(
        "A  —  atmospheric light colour",
        key="A_hex",
        help=(
            "Colour of the diffuse light scattered into the scene by the dust.  "
            "Warm ochres and sandy yellows are appropriate for Saharan conditions."
        ),
    )
    rgb = _hex_to_rgb01(st.session_state["A_hex"])
    sb.caption(f"Linear RGB  ({rgb[0]:.2f},  {rgb[1]:.2f},  {rgb[2]:.2f})")

    # ── Simulation control ──────────────────────────────────────────────────
    sb.markdown("### Simulation Control")

    sb.number_input(
        "Random seed",
        min_value=0, max_value=9999, step=1,
        key="sequence_seed",
        help=(
            "Seed for the turbulence field RNG.  "
            "Identical seed + parameters always yields identical output."
        ),
    )

    sb.slider(
        "Ray-marching steps  (N)",
        min_value=16, max_value=128, step=8,
        key="n_ray_steps",
        help=(
            "Number of integration slabs along each camera ray (Eq. 2, Doc B).  "
            "Higher values improve accuracy; 64 is a practical default."
        ),
    )

    # ── Temporal correlation ────────────────────────────────────────────────
    sb.markdown("### Temporal Correlation")

    sb.slider(
        "Turbulence persistence  (refresh rate)",
        min_value=0.0, max_value=1.0, step=0.01,
        key="rho_refresh_rate",
        help=(
            "Controls how much the turbulent density field changes between "
            "consecutive frames.  "
            "0.0: the field only advects — structures persist indefinitely.  "
            "0.1: slow evolution, characteristic of a sustained sandstorm.  "
            "0.5: moderate gustiness — large structures reform within ~2 frames.  "
            "1.0: fully independent per frame — incoherent flickering."
        ),
    )

    _r = st.session_state["rho_refresh_rate"]
    if _r <= 0.05:
        _regime, _regime_color = "Steady (unrealistic for long sequences)", "#888"
    elif _r <= 0.20:
        _regime, _regime_color = "Slow evolution — realistic sandstorm", "#2a7a2a"
    elif _r <= 0.50:
        _regime, _regime_color = "Gusty conditions", "#b06000"
    else:
        _regime, _regime_color = "Rapid / flickering", "#b00000"

    sb.markdown(
        f"<span style='font-size:0.78rem;color:{_regime_color}'>"
        f"{int(_r * 100)}%  —  {_regime}</span>",
        unsafe_allow_html=True,
    )

    # ── Depth source ────────────────────────────────────────────────────────
    sb.markdown("### Depth Source")

    sb.selectbox(
        "Depth estimation method",
        options=["Auto (MiDaS)", "Synthetic (testing only)"],
        key="depth_source",
        help=(
            "Auto (MiDaS): loads Intel/dpt-hybrid-midas — from the local "
            "midas_model/ folder if present, otherwise downloads it automatically "
            "from HuggingFace Hub (~500 MB, one-time).  "
            "Synthetic: planar gradient — use only to verify the pipeline runs, "
            "not for research-grade results."
        ),
    )

    if st.session_state["depth_source"] == "Synthetic (testing only)":
        sb.warning(
            "Synthetic depth does not reflect actual scene geometry.  "
            "Results are not physically accurate."
        )

# =========================================================================== #
#  OUTPUT SECTION
# =========================================================================== #

def _render_output(output_dir: Path, metadata: Dict[str, Any]) -> None:
    """Render the results panel after a successful pipeline run.

    Parameters
    ----------
    output_dir:
        Root directory produced by run_pipeline.
    metadata:
        Annotated metadata dict from the pipeline runner.
    """
    st.markdown("---")
    st.markdown("## Results")

    # ── Frame comparison ────────────────────────────────────────────────────
    clean_pngs    = sorted((output_dir / "clean_rgb").glob("frame_*.png"))
    degraded_pngs = sorted((output_dir / "degraded_rgb").glob("frame_*.png"))
    n = len(clean_pngs)

    if n > 0:
        st.markdown("### Frame Comparison")

        frame_idx = st.slider(
            "Frame index",
            min_value=0,
            max_value=max(n - 1, 0),
            value=min(n // 2, n - 1),
            help="Step through the processed frames.",
        )

        col_a, col_b = st.columns(2, gap="medium")
        clean_img = _load_rgb(clean_pngs[frame_idx])
        deg_img   = _load_rgb(degraded_pngs[frame_idx])

        with col_a:
            st.caption(f"Clean input  —  frame {frame_idx:04d}")
            if clean_img is not None:
                st.image(clean_img, use_container_width=True)

        with col_b:
            st.caption(f"Degraded output  —  frame {frame_idx:04d}")
            if deg_img is not None:
                st.image(deg_img, use_container_width=True)

    # ── Physical statistics ─────────────────────────────────────────────────
    st.markdown("### Physical Statistics  (frame 0)")

    tau_npy   = output_dir / "tau_maps"          / "frame_0000.png.npy"
    trans_npy = output_dir / "transmission_maps" / "frame_0000.png.npy"
    beta_npy  = output_dir / "beta_maps"         / "frame_0000.png.npy"
    depth_npy = output_dir / "depth_maps"        / "frame_0000.png.npy"

    if tau_npy.exists() and trans_npy.exists():
        tau   = np.load(str(tau_npy))
        trans = np.load(str(trans_npy))
        beta  = np.load(str(beta_npy))  if beta_npy.exists()  else None
        depth = np.load(str(depth_npy)) if depth_npy.exists() else None

        cols = st.columns(5)
        cols[0].metric("Mean τ",       f"{tau.mean():.3f}",
                       help="Mean optical depth across the frame.")
        cols[1].metric("Max τ",        f"{tau.max():.3f}",
                       help="Peak optical depth (densest region).")
        cols[2].metric("Mean t̄",       f"{trans.mean():.3f}",
                       help="Mean transmission.  0 = fully opaque, 1 = clear.")
        cols[3].metric("Min t",        f"{trans.min():.4f}",
                       help="Minimum transmission (most occluded pixel).")
        if beta is not None:
            cols[4].metric("Mean β (m⁻¹)", f"{beta.mean():.5f}",
                           help="Mean extinction coefficient.")
        if depth is not None:
            fmv = metadata.get("frame_map_maxvals", [{}])
            if fmv:
                max_d = fmv[0].get("depth_m", 1.0)
                depth_m = np.load(str(depth_npy))
                st.caption(
                    f"Depth range: {depth_m.min():.1f} – {depth_m.max():.1f} m  "
                    f"(MiDaS relative, rescaled to [{DEPTH_MIN_M}, {DEPTH_MAX_M}] m)"
                )

    # ── Depth source note ───────────────────────────────────────────────────
    st.info(metadata.get("depth_source", "Depth source not recorded."))

    # ── Parameters used ─────────────────────────────────────────────────────
    with st.expander("Parameters used for this run", expanded=False):
        p = metadata.get("parameters", {})

        # Regime label for rho_refresh_rate
        _r = p.get("rho_refresh_rate", 0.1)
        if _r <= 0.05:
            _rlabel = "steady (unrealistic)"
        elif _r <= 0.20:
            _rlabel = "slow evolution — realistic sandstorm"
        elif _r <= 0.50:
            _rlabel = "gusty"
        else:
            _rlabel = "rapid / flickering"

        cols = st.columns(3)
        display_params = {k: v for k, v in p.items() if k != "rho_refresh_rate"}
        items = list(display_params.items())
        for i, (k, v) in enumerate(items):
            val_str = f"{v:.5f}" if isinstance(v, float) else str(v)
            cols[i % 3].code(f"{k} = {val_str}", language="")

        st.caption(
            f"rho_refresh_rate = {_r:.2f}  ({int(_r*100)}%)  —  {_rlabel}"
        )

    # ── Downloads ───────────────────────────────────────────────────────────
    st.markdown("### Downloads")

    col1, col2, col3 = st.columns(3, gap="small")

    video_path = output_dir / "sandstorm_video.mp4"
    with col1:
        if video_path.exists():
            st.download_button(
                "Download sandstorm_video.mp4",
                data=video_path.read_bytes(),
                file_name="sandstorm_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

    meta_path = output_dir / "metadata.json"
    with col2:
        if meta_path.exists():
            st.download_button(
                "Download metadata.json",
                data=meta_path.read_text(encoding="utf-8"),
                file_name="metadata.json",
                mime="application/json",
                use_container_width=True,
            )

    npy_files = _npy_paths(output_dir)
    with col3:
        if npy_files:
            st.download_button(
                f"Download physical_maps.zip  ({len(npy_files)} files)",
                data=_zip_npy_maps(output_dir),
                file_name="physical_maps.zip",
                mime="application/zip",
                use_container_width=True,
            )

    # ── Processing log ──────────────────────────────────────────────────────
    with st.expander("Processing log", expanded=False):
        st.code("\n".join(st.session_state["log_lines"]), language="")


# =========================================================================== #
#  MAIN
# =========================================================================== #

def main() -> None:
    """Application entry point.

    Renders the sidebar controls, the input panel, the run button, and — once
    a run has completed — the results panel.  All mutable state is persisted in
    st.session_state so widget interactions do not reset the page.
    """
    _render_sidebar()

    # ── Page header ─────────────────────────────────────────────────────────
    st.markdown("# SandStorm-Video Generator")
    st.markdown(
        "Physics-based sand and dust degradation for autonomous driving datasets.  "
        "Implements Kolmogorov turbulence, Mie PSF, multiple-scattering glow, "
        "and Beer–Lambert extinction  (Khlif et al., 2026)."
    )
    st.markdown("---")

    # ── Input ────────────────────────────────────────────────────────────────
    st.markdown("## Input Video")

    uploaded = st.file_uploader(
        "Select an MP4, AVI, or MOV file",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
        help="The video is processed entirely on your local machine.",
    )

    if uploaded is None:
        st.caption(
            "Upload a clean RGB video to begin.  "
            "Adjust degradation parameters in the sidebar before running."
        )
        st.markdown("---")
        st.caption(
            "SandStorm-Video Generator  ·  "
            "Based on Khlif et al. (2026)  ·  "
            "Depth estimation via Intel/dpt-hybrid-midas"
        )
        return

    # ── File info row ────────────────────────────────────────────────────────
    info_col, btn_col = st.columns([4, 1], gap="large")

    with info_col:
        st.markdown(
            f"**{uploaded.name}**  &nbsp;·&nbsp;  {uploaded.size / 1e6:.1f} MB",
            unsafe_allow_html=True,
        )

    with btn_col:
        run = st.button(
            "Run pipeline",
            type="primary",
            use_container_width=True,
        )

    # ── Pipeline execution ───────────────────────────────────────────────────
    if run:
        st.session_state["run_done"]   = False
        st.session_state["output_dir"] = None
        st.session_state["log_lines"]  = []

        with st.spinner("Decoding video..."):
            frames, fps, W, H = _extract_frames(uploaded.read())

        if not frames:
            st.error("No frames could be decoded from the uploaded file.")
            return

        st.markdown(
            f"`{len(frames)} frames`  &nbsp;·&nbsp;  "
            f"`{W} × {H} px`  &nbsp;·&nbsp;  "
            f"`{fps:.1f} fps`",
            unsafe_allow_html=True,
        )

        output_dir = Path(__file__).parent / "output_sandstorm"
        params     = _build_params()

        metadata = run_pipeline(
            frames_uint8 = frames,
            fps          = fps,
            width        = W,
            height       = H,
            params       = params,
            depth_source = st.session_state["depth_source"],
            seed         = int(st.session_state["sequence_seed"]),
            n_ray_steps  = int(st.session_state["n_ray_steps"]),
            output_dir   = output_dir,
        )

        st.session_state["run_done"]    = True
        st.session_state["output_dir"]  = str(output_dir)
        st.session_state["metadata"]    = metadata

    # ── Results ──────────────────────────────────────────────────────────────
    if st.session_state.get("run_done") and st.session_state.get("output_dir"):
        _render_output(
            output_dir = Path(st.session_state["output_dir"]),
            metadata   = st.session_state.get("metadata", {}),
        )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "SandStorm-Video Generator  ·  "
        "Khlif et al. (2026) — *A Benchmark for Sand and Dust Video Degradation*  ·  "
        "Built with [Streamlit](https://streamlit.io)"
    )


if __name__ == "__main__":
    main()
