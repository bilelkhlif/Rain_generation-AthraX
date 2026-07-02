"""
process_test_video.py
=====================
Apply the physics-based sand/dust degradation pipeline from sand_dust_pipeline.py
to a real video file.

Usage
-----
    # Auto-locate a file named test.{mp4,avi,mov,mkv} in the working directory:
    python process_test_video.py

    # Explicit input / output:
    python process_test_video.py --input /path/to/clip.mp4 --output_dir my_output

Pipeline
--------
1. Extract frames as float32 RGB [0, 1]                   (OpenCV)
2. Estimate per-frame depth with Intel/dpt-hybrid-midas   (transformers + torch)
   → relative inverse depth is linearly rescaled to 2–200 m  (**approximation**)
3. Degrade via sand_dust_pipeline.degrade_video()
4. Re-encode degraded frames → <stem>_sandstorm.mp4       (OpenCV VideoWriter)

Depth note
----------
MiDaS produces *relative inverse depth* (unitless, monotone with distance).
The linear rescaling to [2, 200] m used here is a reasonable approximation for
driving scenes — sky pixels map to ~200 m, near-ground pixels to ~2 m —
but is NOT calibrated metric ground truth.  This is recorded explicitly in
metadata.json under "depth_source".

Requirements
------------
    pip install torch torchvision transformers accelerate
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── import the pipeline ────────────────────────────────────────────────────
try:
    from sand_dust_pipeline import (
        degrade_video,
        sample_parameters,
    )
except ImportError as exc:
    sys.exit(
        f"[ERROR] Cannot import sand_dust_pipeline: {exc}\n"
        "Make sure sand_dust_pipeline.py is in the same directory or on PYTHONPATH."
    )

# =========================================================================== #
#  DEPTH ESTIMATION  (Intel/dpt-hybrid-midas via HuggingFace transformers)
# =========================================================================== #

# MiDaS relative-disparity output is linearly rescaled to this pseudo-metric
# range.  Chosen to match typical autonomous-driving scene extents.
# APPROXIMATION — not calibrated metric ground truth.
DEPTH_MIN_M = 2.0    # closest objects (e.g. vehicle hood)
DEPTH_MAX_M = 200.0  # farthest visible scene elements

_midas_model     = None
_midas_processor = None
_midas_device    = None


def _load_midas() -> None:
    """Load Intel/dpt-hybrid-midas for per-frame depth estimation.

    Resolution strategy (tried in order):

    1. **Local folder** — ``<script_dir>/midas_model/`` with either
       ``model.safetensors`` or ``pytorch_model.bin`` present alongside
       ``config.json`` and ``preprocessor_config.json``.
       Suitable for air-gapped environments or faster repeated runs.

    2. **HuggingFace Hub** — if the local folder is absent or incomplete,
       the model is downloaded automatically from ``Intel/dpt-hybrid-midas``
       and cached in the standard HuggingFace cache directory
       (``~/.cache/huggingface/hub``).  This requires an internet connection
       on the first run and downloads approximately 500 MB.

    The function exits with a descriptive message if neither ``torch`` nor
    ``transformers`` are installed.
    """
    global _midas_model, _midas_processor, _midas_device

    import os
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    try:
        import torch
        from transformers import DPTForDepthEstimation, DPTImageProcessor
    except ImportError:
        sys.exit(
            "[ERROR] 'transformers' or 'torch' not found.\n"
            "Install with:  pip install torch torchvision transformers accelerate"
        )

    # ── resolve model source ───────────────────────────────────────────────
    script_dir  = Path(__file__).parent
    local_model = script_dir / "midas_model"

    has_safetensors = (local_model / "model.safetensors").exists()
    has_bin         = (local_model / "pytorch_model.bin").exists()
    required_base   = ["config.json", "preprocessor_config.json"]

    local_ok = (
        local_model.is_dir()
        and (has_safetensors or has_bin)
        and all((local_model / f).exists() for f in required_base)
    )

    if local_ok:
        model_source     = str(local_model)
        use_local        = True
        use_safetensors  = has_safetensors
        weight_file      = "model.safetensors" if has_safetensors else "pytorch_model.bin"
        print(f"[depth] Loading MiDaS from local folder ({weight_file}).")
    else:
        model_source     = "Intel/dpt-hybrid-midas"
        use_local        = False
        use_safetensors  = True   # Hub always provides safetensors
        print(
            "[depth] Local midas_model/ folder not found.  "
            "Downloading Intel/dpt-hybrid-midas from HuggingFace Hub (~500 MB).\n"
            "[depth] This is a one-time download; subsequent runs use the cache."
        )

    # ── load ───────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[depth] Loading model on {device}.")
    try:
        kwargs = {"local_files_only": True} if use_local else {}
        _midas_processor = DPTImageProcessor.from_pretrained(model_source, **kwargs)
        _midas_model     = DPTForDepthEstimation.from_pretrained(
            model_source,
            use_safetensors=use_safetensors,
            **kwargs,
        )
        _midas_model.to(device).eval()
        _midas_device = device
        print("[depth] Model loaded successfully.")
    except Exception as exc:
        sys.exit(
            f"[ERROR] Failed to load MiDaS from '{model_source}':\n  {exc}\n"
            "Check your internet connection or place the model files in midas_model/."
        )


def _estimate_depth_midas(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Run MiDaS DPT on one uint8 RGB frame and return pseudo-metric depth.

    MiDaS outputs relative inverse depth (disparity-like: higher = closer).
    Steps:
      1. Normalise disparity to [0, 1].
      2. Invert  →  [0, 1] depth proxy  (0 = near, 1 = far).
      3. Linearly rescale to [DEPTH_MIN_M, DEPTH_MAX_M].

    NOTE: result is an approximation, not calibrated metric depth.

    Parameters
    ----------
    rgb_uint8 : (H, W, 3) uint8 RGB frame

    Returns
    -------
    depth_m : (H, W) float32  pseudo-metric depth in metres
    """
    import torch
    from PIL import Image as PilImage

    pil_img = PilImage.fromarray(rgb_uint8)
    inputs  = _midas_processor(images=pil_img, return_tensors="pt")
    inputs  = {k: v.to(_midas_device) for k, v in inputs.items()}

    with torch.no_grad():
        pred = _midas_model(**inputs).predicted_depth.squeeze().cpu().numpy()

    # Resize output to original frame resolution
    H, W = rgb_uint8.shape[:2]
    pred = cv2.resize(pred.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)

    d_min, d_max = pred.min(), pred.max()
    if d_max - d_min < 1e-6:
        # Degenerate / uniform frame → mid-range depth
        return np.full((H, W), 0.5 * (DEPTH_MIN_M + DEPTH_MAX_M), dtype=np.float32)

    # Invert disparity → depth proxy in [0, 1]
    depth_proxy = 1.0 - (pred - d_min) / (d_max - d_min)   # 0=near, 1=far

    # Rescale to pseudo-metric metres
    depth_m = (DEPTH_MIN_M + depth_proxy * (DEPTH_MAX_M - DEPTH_MIN_M)).astype(np.float32)
    return depth_m


# =========================================================================== #
#  VIDEO I/O HELPERS
# =========================================================================== #

_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")


def _find_test_video(search_dir: Path) -> Path:
    """
    Locate a file whose stem is 'test' (case-insensitive) in search_dir.
    Returns the first match or raises FileNotFoundError.
    """
    for ext in _VIDEO_EXTENSIONS:
        candidate = search_dir / f"test{ext}"
        if candidate.exists():
            return candidate
    # Case-insensitive fallback (Windows is case-insensitive, but be explicit)
    for p in search_dir.iterdir():
        if p.suffix.lower() in {e.lower() for e in _VIDEO_EXTENSIONS}:
            if p.stem.lower() == "test":
                return p
    raise FileNotFoundError(
        f"No file named 'test.<ext>' found in {search_dir}.\n"
        f"Supported extensions: {_VIDEO_EXTENSIONS}\n"
        "Use --input <path> to specify a different file."
    )


def _extract_frames(video_path: Path) -> tuple:
    """
    Extract all frames from a video file.

    Returns
    -------
    frames_rgb : list of (H, W, 3) uint8 numpy arrays (RGB order)
    fps        : float  – source frame rate
    width      : int
    height     : int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        sys.exit("[ERROR] Video reports 0 frames. Check the file is a valid video.")

    frames_rgb = []
    with tqdm(total=total if total > 0 else None,
              desc="Extracting frames", unit="frame") as pbar:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            frames_rgb.append(bgr[..., ::-1].copy())   # BGR→RGB, contiguous
            pbar.update(1)

    cap.release()

    if len(frames_rgb) == 0:
        sys.exit("[ERROR] No frames could be read from the video.")

    return frames_rgb, fps, width, height


def _encode_output_video(
    degraded_dir: Path,
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    n_frames: int,
) -> None:
    """
    Re-encode degraded_rgb PNG frames back to an MP4 video.

    Tries H.264 (avc1) first, falls back to mp4v if the codec is unavailable.
    """
    # Try H.264 first, then mp4v
    for fourcc_str in ("avc1", "mp4v", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )
        if writer.isOpened():
            break
        writer.release()
    else:
        print("[WARNING] Could not open any VideoWriter codec. Skipping video encode.")
        return

    frame_files = sorted(degraded_dir.glob("frame_*.png"))
    if not frame_files:
        print("[WARNING] No degraded PNG frames found for video encoding.")
        writer.release()
        return

    with tqdm(total=len(frame_files), desc="Encoding output video", unit="frame") as pbar:
        for fpath in frame_files:
            img_bgr = cv2.imread(str(fpath))
            if img_bgr is None:
                continue
            # Resize to exactly match source dimensions (MiDaS might change size)
            if img_bgr.shape[1] != width or img_bgr.shape[0] != height:
                img_bgr = cv2.resize(img_bgr, (width, height))
            writer.write(img_bgr)
            pbar.update(1)

    writer.release()
    print(f"[encode] Output video saved: {output_path}")


# =========================================================================== #
#  MAIN PIPELINE
# =========================================================================== #

def process_video(input_path: Path, output_dir: Path) -> None:
    """
    Full pipeline: extract → depth → degrade → encode.
    """
    print("\n" + "=" * 65)
    print("  SandStorm Video Degradation Pipeline")
    print("=" * 65)

    # ── 1. Extract frames ──────────────────────────────────────────────────
    print(f"\n[input] {input_path}")
    frames_uint8, fps, W, H = _extract_frames(input_path)
    n_frames = len(frames_uint8)

    print(f"[input] {W}×{H} px  |  {fps:.2f} fps  |  {n_frames} frames")
    if n_frames == 0:
        sys.exit("[ERROR] 0 frames extracted.")

    # ── 2. Depth estimation ────────────────────────────────────────────────
    _load_midas()   # exits cleanly if torch / transformers missing

    depth_source_label = (
        "Intel/dpt-hybrid-midas – relative inverse depth "
        f"linearly rescaled to [{DEPTH_MIN_M}, {DEPTH_MAX_M}] m  "
        "(APPROXIMATION – not calibrated metric ground truth)"
    )
    print(f"\n[depth] Estimating depth for {n_frames} frames …")
    depth_frames_float32 = []
    for frame_u8 in tqdm(frames_uint8, desc="Depth estimation", unit="frame"):
        depth_frames_float32.append(_estimate_depth_midas(frame_u8))

    # ── 3. Convert frames to float32 [0, 1] ───────────────────────────────
    print("\n[preprocess] Converting frames to float32 RGB …")
    clean_frames_float32 = [
        f.astype(np.float32) / 255.0 for f in frames_uint8
    ]

    # ── 4. Sample degradation parameters ──────────────────────────────────
    rng    = np.random.default_rng(42)
    params = sample_parameters(rng)
    print("\n[params] Degradation parameters (seed=42):")
    for k, v in params.items():
        print(f"  {k:12s}: {v}")

    # ── 5. Run degradation pipeline ────────────────────────────────────────
    print(f"\n[degrade] Writing output to {output_dir} …")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = degrade_video(
        clean_frames      = clean_frames_float32,
        depth_frames      = depth_frames_float32,
        output_dir        = str(output_dir),
        sequence_seed     = 42,
        n_ray_steps       = 64,
        n_blur_levels     = 16,
    )

    # Annotate metadata with depth provenance
    metadata["depth_source"] = depth_source_label
    metadata["input_video"]  = str(input_path)
    metadata["source_fps"]   = fps
    metadata["source_resolution"] = [H, W]

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"[meta] metadata.json updated with depth_source annotation.")

    # ── 6. Re-encode degraded frames to video ─────────────────────────────
    output_video_path = output_dir / f"{input_path.stem}_sandstorm.mp4"
    degraded_rgb_dir  = output_dir / "degraded_rgb"

    _encode_output_video(
        degraded_dir  = degraded_rgb_dir,
        output_path   = output_video_path,
        fps           = fps,
        width         = W,
        height        = H,
        n_frames      = n_frames,
    )

    # ── 7. Print summary ───────────────────────────────────────────────────
    # Physical stats from the first frame's saved metadata
    fmv0 = metadata.get("frame_map_maxvals", [{}])[0]
    # Load first frame's tau and transmission .npy for accurate stats
    tau_npy  = output_dir / "tau_maps"          / "frame_0000.png.npy"
    trans_npy= output_dir / "transmission_maps" / "frame_0000.png.npy"
    try:
        mean_tau   = float(np.load(str(tau_npy)).mean())
        mean_trans = float(np.load(str(trans_npy)).mean())
    except Exception:
        mean_tau   = fmv0.get("tau",           float("nan"))
        mean_trans = fmv0.get("transmission",  float("nan"))

    print("\n" + "=" * 65)
    print("  PIPELINE SUMMARY")
    print("=" * 65)
    print(f"  Input           : {input_path}")
    print(f"  Resolution      : {W} × {H} px")
    print(f"  FPS             : {fps:.2f}")
    print(f"  Frames processed: {n_frames}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Output video    : {output_video_path}")
    print(f"  Metadata        : {meta_path}")
    print(f"\n  First-frame physical statistics:")
    print(f"    Mean transmission t̄  : {mean_trans:.4f}  "
          f"(1.0=clear, 0.0=opaque)")
    print(f"    Mean optical depth τ̄ : {mean_tau:.4f}  (dimensionless)")
    print(
        f"\n  ⚠  DEPTH NOTE: {depth_source_label[:80]} …"
        if len(depth_source_label) > 80 else
        f"\n  ⚠  DEPTH NOTE: {depth_source_label}"
    )
    print("=" * 65 + "\n")


# =========================================================================== #
#  CLI ENTRY POINT
# =========================================================================== #

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the CLI entry point.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes ``input`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Apply physics-based sand/dust degradation to a video.\n"
            "Depth is estimated with Intel/dpt-hybrid-midas (MiDaS) and is an\n"
            "APPROXIMATION, not metric ground truth."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help=(
            "Path to the input video. "
            "If omitted, the script looks for test.{mp4,avi,mov,mkv} "
            "in the current working directory."
        ),
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="output_sandstorm",
        help="Root output directory  (default: output_sandstorm/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("SandStorm-Video Generator — CLI")
    print("Usage: python process_test_video.py --input <video.mp4> [--output_dir <dir>]")
    print("       streamlit run app.py  (GUI)")
    _args = _parse_args()
    _input = Path(_args.input) if _args.input else None
    if _input is None:
        try:
            _input = _find_test_video(Path.cwd())
        except FileNotFoundError as e:
            sys.exit(f"[ERROR] {e}")
    elif not _input.exists():
        sys.exit(f"[ERROR] Input file not found: {_input}")
    process_video(_input, Path(_args.output_dir))
