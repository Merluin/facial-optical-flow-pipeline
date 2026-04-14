"""
Facial Expression Optical Flow Pipeline
========================================
Batch processor for multiple video datasets

Steps (per video):
  1. Load video frames
  2. Detect face using OpenCV Haar cascade
  3a. Smooth bounding boxes (temporal stability)
  3b. Standardise face position (align, crop) + affine registration to frame 0
  4. Create circular face mask (optional)
  5. Detect the expression apex frame (max motion from neutral)
  6. Segment a fixed window around the apex
  7. Apply dense Farneback optical flow on the segment
  8. Save flow vectors to NPZ
  9. Export a GIF with colour-wheel + arrow visualisation

Usage (from project root, with face_of conda env active):
    python scripts/face_of_pipeline.py

Output structure:
    output/
    ├── optical_flow_npz/
    │   ├── ADFES_video1_name.npz
    │   ├── ADFES_video2_name.npz
    │   └── JeFEE_video1_name.npz
    └── flow_visualization_gif/
        ├── ADFES_video1_name.gif
        ├── ADFES_video2_name.gif
        └── JeFEE_video1_name.gif
"""
#conda activate face_of 

import math
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# Configuration — edit these to change behaviour
# ---------------------------------------------------------------------------
VIDEO_ROOT  = "video"        # scan this folder for all videos
OUTPUT_ROOT = "output"       # organize outputs by file type

CROP_SIZE       = 256        # output face square in pixels
PAD_FACTOR      = 0.35       # extra padding fraction around face bounding box
APEX_WINDOW_BEFORE = 40      # frames BEFORE apex to include in segment
APEX_WINDOW_AFTER  = 5       # frames AFTER apex to include in segment

# Face mask for optical flow (isolate motion to face region only)
USE_FACE_MASK   = True       # enable/disable face masking for OF
FACE_MASK_SIZE  = 0.85       # fraction of frame size (0.0-1.0); 0.85 = 85% of 256px

ARROW_STEP  = 16             # grid spacing for arrow visualisation (px)
ARROW_SCALE = 5.0            # multiplier to make arrows visible
GIF_SPEED   = 0.5            # fraction of real speed for GIF (0.5 = half speed)


# Farneback parameters
OF_PYR_SCALE  = 0.5
OF_LEVELS     = 3
OF_WINSIZE    = 15
OF_ITERATIONS = 3
OF_POLY_N     = 5
OF_POLY_SIGMA = 1.2

# Video file extensions to search for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_all_videos(root_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Recursively find all video files.
    Returns list of (video_path, dataset_name, video_stem).
    dataset_name is the immediate parent folder (e.g., 'ADFES', 'JeFEE').
    """
    videos = []
    for video_path in root_dir.rglob("*"):
        if video_path.suffix.lower() in VIDEO_EXTENSIONS and video_path.is_file():
            dataset_name = video_path.parent.name
            video_stem = video_path.stem
            videos.append((video_path, dataset_name, video_stem))
    return sorted(videos)


def load_video(path: Path) -> tuple[list[np.ndarray], float]:
    """Return list of BGR frames and the video FPS."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}")
        return [], 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    print(f"[INFO] Loaded {len(frames)} frames at {fps:.2f} fps")
    return frames, fps


def detect_face_bbox(frame_bgr: np.ndarray, cascade) -> tuple[int, int, int, int] | None:
    """
    Detect face using Haar cascade.
    Returns (x, y, w, h) or None if no face found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(50, 50))
    if len(faces) == 0:
        return None
    # return largest face
    return tuple(faces[np.argmax(faces[:, 2] * faces[:, 3])])


def estimate_eye_points(x: int, y: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate left and right eye centers from face bounding box.
    Simple geometry: eyes are ~1/3 down the face, at 1/3 and 2/3 horizontal positions.
    """
    eye_y = y + int(0.35 * h)
    left_eye  = np.array([x + int(0.25 * w), eye_y], dtype=np.float32)
    right_eye = np.array([x + int(0.75 * w), eye_y], dtype=np.float32)
    return left_eye, right_eye


def get_face_keypoints(x: int, y: int, w: int, h: int) -> np.ndarray:
    """
    Estimate 3 key facial points from bounding box: left eye, right eye, nose.
    Returns (3, 2) array of (x, y) coordinates.
    """
    left_eye  = np.array([x + int(0.25 * w), y + int(0.35 * h)], dtype=np.float32)
    right_eye = np.array([x + int(0.75 * w), y + int(0.35 * h)], dtype=np.float32)
    nose      = np.array([x + int(0.50 * w), y + int(0.55 * h)], dtype=np.float32)
    return np.array([left_eye, right_eye, nose], dtype=np.float32)


def align_and_crop(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                   crop_size: int, pad_factor: float) -> np.ndarray:
    """
    1. Rotate frame so eyes are horizontal.
    2. Crop a padded square around the face.
    3. Resize to crop_size × crop_size.
    """
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox

    # estimate eye positions
    le, re = estimate_eye_points(x, y, bw, bh)

    # --- rotation ---
    dx, dy = re - le
    angle = math.degrees(math.atan2(dy, dx))
    cx, cy = w / 2, h / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame_bgr, M_rot, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

    # rotate bbox center
    bbox_center = np.array([x + bw/2, y + bh/2, 1.0], dtype=np.float32)
    bbox_center_rot = M_rot @ bbox_center

    # --- padded square bounding box ---
    pad = pad_factor * max(bw, bh)
    x_min = bbox_center_rot[0] - bw/2 - pad
    y_min = bbox_center_rot[1] - bh/2 - pad
    x_max = x_min + bw + 2*pad
    y_max = y_min + bh + 2*pad

    # force square
    side = max(x_max - x_min, y_max - y_min)
    cx_box = (x_min + x_max) / 2
    cy_box = (y_min + y_max) / 2
    x1 = max(0, int(cx_box - side / 2))
    y1 = max(0, int(cy_box - side / 2))
    x2 = min(w, int(x1 + side))
    y2 = min(h, int(y1 + side))

    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame_bgr, (crop_size, crop_size))
    return cv2.resize(crop, (crop_size, crop_size))


def smooth_bboxes(bboxes: list[tuple], kernel_size: int = 5) -> list[tuple]:
    """Apply median filter to smooth bounding boxes across frames."""
    bboxes_array = np.array(bboxes, dtype=np.float32)
    smoothed = np.zeros_like(bboxes_array)
    for i in range(4):  # smooth x, y, w, h
        smoothed[:, i] = ndimage.median_filter(bboxes_array[:, i], size=kernel_size)
    return [tuple(int(v) for v in b) for b in smoothed]


def register_to_reference(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                          ref_keypoints: np.ndarray, ref_bbox: tuple[int, int, int, int],
                          crop_size: int, pad_factor: float) -> np.ndarray:
    """
    Register frame to reference by computing affine transform from current keypoints
    to reference keypoints. This removes rigid head motion.
    """
    x, y, bw, bh = bbox
    curr_keypoints = get_face_keypoints(x, y, bw, bh)

    # Compute affine transformation to align current to reference
    M = cv2.getAffineTransform(curr_keypoints, ref_keypoints)

    # Apply affine warp
    h, w = frame_bgr.shape[:2]
    warped = cv2.warpAffine(frame_bgr, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    # Crop around the reference frame's fixed position
    ref_x, ref_y, ref_w, ref_h = ref_bbox

    # --- crop around reference position ---
    pad = pad_factor * max(ref_w, ref_h)
    x_min = int(ref_x + ref_w/2 - ref_w/2 - pad)
    y_min = int(ref_y + ref_h/2 - ref_h/2 - pad)
    x_max = int(x_min + ref_w + 2*pad)
    y_max = int(y_min + ref_h + 2*pad)

    # force square
    side = max(x_max - x_min, y_max - y_min)
    cx_box = (x_min + x_max) / 2
    cy_box = (y_min + y_max) / 2
    x1 = max(0, int(cx_box - side / 2))
    y1 = max(0, int(cy_box - side / 2))
    x2 = min(w, int(x1 + side))
    y2 = min(h, int(y1 + side))

    crop = warped[y1:y2, x1:x2]
    if crop.size == 0:
        return cv2.resize(frame_bgr, (crop_size, crop_size))
    return cv2.resize(crop, (crop_size, crop_size))


def create_face_mask(h: int, w: int, mask_size: float) -> np.ndarray:
    """
    Create an elliptical (oval) face mask that frames the face.
    h, w: frame dimensions
    mask_size: fraction of frame to mask (0.0-1.0)
    Returns binary mask (0=outside, 255=inside).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    # Ellipse: taller than wide to match face proportions
    axes_x = int((w / 2) * mask_size * 0.9)      # narrower (90% of horizontal)
    axes_y = int((h / 2) * mask_size)             # taller (100% of vertical, to frame face)
    cv2.ellipse(mask, center, (axes_x, axes_y), 0, 0, 360, 255, -1)  # filled ellipse
    return mask


def apply_mask_to_gray(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to grayscale frame (sets masked-out regions to 0)."""
    return cv2.bitwise_and(gray, gray, mask=mask)


def motion_from_neutral(frames_standardised: list[np.ndarray], mask: np.ndarray | None = None) -> np.ndarray:
    """
    For each frame, compute total optical flow magnitude from the first (neutral) frame.
    This is used to detect the apex (peak displacement from neutral).
    If mask is provided, only compute flow within the mask region.
    Returns 1-D array of length n_frames.
    """
    motion = [0.0]  # first frame has 0 motion
    neutral_gray = cv2.cvtColor(frames_standardised[0], cv2.COLOR_BGR2GRAY)
    if mask is not None:
        neutral_gray = apply_mask_to_gray(neutral_gray, mask)

    for i in range(1, len(frames_standardised)):
        curr_gray = cv2.cvtColor(frames_standardised[i], cv2.COLOR_BGR2GRAY)
        if mask is not None:
            curr_gray = apply_mask_to_gray(curr_gray, mask)

        flow = cv2.calcOpticalFlowFarneback(
            neutral_gray, curr_gray, None,
            OF_PYR_SCALE, OF_LEVELS, OF_WINSIZE,
            OF_ITERATIONS, OF_POLY_N, OF_POLY_SIGMA, 0,
        )
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).sum()
        motion.append(mag)

    return np.array(motion, dtype=np.float32)


def flow_to_color(flow: np.ndarray, max_mag: float) -> np.ndarray:
    """
    Convert (H, W, 2) flow to a BGR colour image using HSV colour wheel.
    Direction → hue, magnitude → value, saturation=1.
    """
    dx, dy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)  # -pi .. pi

    hue = ((angle + math.pi) / (2 * math.pi) * 179).astype(np.uint8)
    if max_mag > 0:
        val = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    else:
        val = np.zeros_like(hue)
    sat = np.full_like(hue, 255)

    hsv = np.stack([hue, sat, val], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_arrows(canvas: np.ndarray, flow: np.ndarray,
                step: int, scale: float) -> np.ndarray:
    """
    Draw a regular grid of flow arrows on canvas.
    Skips arrows with very small magnitude.
    """
    out = canvas.copy()
    h, w = canvas.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx = flow[y, x, 0] * scale
            dy = flow[y, x, 1] * scale
            mag = math.sqrt(dx**2 + dy**2)
            if mag < 0.3:  # skip very small vectors
                continue
            x2 = int(x + dx)
            y2 = int(y + dy)
            cv2.arrowedLine(out, (x, y), (x2, y2),
                            color=(255, 255, 255), thickness=1,
                            tipLength=0.3)
    return out


def process_video(video_path: Path, dataset_name: str, video_stem: str,
                  output_root: Path, cascade) -> bool:
    """
    Process a single video. Returns True if successful.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}/{video_stem}")
    print(f"{'='*70}")

    # Output folders organized by file type
    npz_dir = output_root / "optical_flow_npz"
    gif_dir = output_root / "flow_visualization_gif"
    npz_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    # File names with dataset prefix
    file_prefix = f"{dataset_name}_{video_stem}"
    npz_path = npz_dir / f"{file_prefix}.npz"
    gif_path = gif_dir / f"{file_prefix}.gif"

    # ── 1. Load video ────────────────────────────────────────────────────────
    frames_raw, fps = load_video(video_path)
    if not frames_raw:
        print(f"[ERROR] Failed to load video: {video_path}")
        return False
    n_frames = len(frames_raw)

    # ── 2. Face detection (all frames) ───────────────────────────────────────
    print("[INFO] Detecting face bounding boxes …")
    bboxes = []
    for i, f in enumerate(frames_raw):
        bbox = detect_face_bbox(f, cascade)
        bboxes.append(bbox)
        if (i + 1) % 20 == 0:
            print(f"  … {i+1}/{n_frames}")

    detected = sum(1 for bbox in bboxes if bbox is not None)
    print(f"[INFO] Face detected in {detected}/{n_frames} frames")
    if detected == 0:
        print(f"[ERROR] No face detected in any frame, skipping.")
        return False

    # fill None entries with nearest valid bbox
    prev_bbox = next((bbox for bbox in bboxes if bbox is not None), None)
    filled_bbox = []
    for bbox in bboxes:
        if bbox is not None:
            prev_bbox = bbox
        filled_bbox.append(prev_bbox)

    # ── 3a. Smooth bounding boxes (reduce jitter) ────────────────────────────
    print("[INFO] Smoothing bounding boxes (temporal stability) …")
    filled_bbox = smooth_bboxes(filled_bbox, kernel_size=5)

    # ── 3b. Face standardisation with affine registration ──────────────────────
    print("[INFO] Standardising face position + affine registration …")
    # First frame as reference
    ref_bbox = filled_bbox[0]
    ref_keypoints = get_face_keypoints(*ref_bbox)

    # Align first frame without registration (it's the reference)
    std_frames = [align_and_crop(frames_raw[0], filled_bbox[0], CROP_SIZE, PAD_FACTOR)]

    # Register all other frames to the reference
    for i in range(1, n_frames):
        frame_registered = register_to_reference(
            frames_raw[i], filled_bbox[i], ref_keypoints, ref_bbox, CROP_SIZE, PAD_FACTOR
        )
        std_frames.append(frame_registered)

    # ── 4. Create face mask (optional) ───────────────────────────────────────
    face_mask = None
    if USE_FACE_MASK:
        print("[INFO] Creating face mask for OF computation …")
        face_mask = create_face_mask(CROP_SIZE, CROP_SIZE, FACE_MASK_SIZE)

    # ── 5. Apex detection (max motion) ───────────────────────────────────────
    print("[INFO] Computing motion for apex detection …")
    motion = motion_from_neutral(std_frames, face_mask)
    apex_global = int(np.argmax(motion))
    print(f"[INFO] Apex frame: {apex_global} (motion={motion[apex_global]:.1f})")

    # ── 6. Segmentation around apex (asymmetric window) ──────────────────────
    seg_start = max(0, apex_global - APEX_WINDOW_BEFORE)
    seg_end   = min(n_frames, apex_global + APEX_WINDOW_AFTER + 1)
    apex_local = apex_global - seg_start
    seg_frames = std_frames[seg_start:seg_end]
    T = len(seg_frames)
    print(f"[INFO] Segment: frames {seg_start}–{seg_end-1} ({T} frames), "
          f"apex at local index {apex_local} "
          f"({APEX_WINDOW_BEFORE} before + {APEX_WINDOW_AFTER} after)")

    # ── 7. Dense Optical Flow (Farneback) ────────────────────────────────────
    print("[INFO] Computing dense optical flow …")
    flows = []
    for i in range(T - 1):
        prev_gray = cv2.cvtColor(seg_frames[i],     cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(seg_frames[i + 1], cv2.COLOR_BGR2GRAY)

        # Apply face mask if enabled
        if face_mask is not None:
            prev_gray = apply_mask_to_gray(prev_gray, face_mask)
            curr_gray = apply_mask_to_gray(curr_gray, face_mask)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            OF_PYR_SCALE, OF_LEVELS, OF_WINSIZE,
            OF_ITERATIONS, OF_POLY_N, OF_POLY_SIGMA, 0,
        )
        flows.append(flow)
    print(f"[INFO] Computed {len(flows)} flow maps, each {flows[0].shape}")

    # ── 8. Save NPZ ──────────────────────────────────────────────────────────
    np.savez_compressed(
        str(npz_path),
        flow              = np.stack(flows),                    # (T-1, H, W, 2)
        frames            = np.stack(seg_frames),              # (T, H, W, 3)
        apex_local        = np.int32(apex_local),
        apex_global       = np.int32(apex_global),
        seg_start         = np.int32(seg_start),
        seg_end           = np.int32(seg_end),
        fps               = np.float32(fps),
        motion_all        = motion,                            # (n_frames,)
        face_mask_used    = USE_FACE_MASK,
        face_mask_size    = FACE_MASK_SIZE,
    )
    print(f"[INFO] Saved → {npz_path}")

    # ── 9. GIF visualisation ─────────────────────────────────────────────────
    print("[INFO] Rendering GIF …")
    flow_stack = np.stack(flows)
    max_mag = float(np.sqrt(flow_stack[..., 0]**2 +
                            flow_stack[..., 1]**2).max())
    max_mag = max(max_mag, 1e-6)

    gif_frames = []
    for i, flow in enumerate(flows):
        base = seg_frames[i].copy()

        # colour-wheel overlay
        color_map = flow_to_color(flow, max_mag)
        blended = cv2.addWeighted(base, 0.5, color_map, 0.5, 0)

        # arrow overlay
        blended = draw_arrows(blended, flow, ARROW_STEP, ARROW_SCALE)

        # apply mask: black out background (if enabled)
        if face_mask is not None:
            center = (CROP_SIZE // 2, CROP_SIZE // 2)
            axes_x = int((CROP_SIZE / 2) * FACE_MASK_SIZE * 0.9)
            axes_y = int((CROP_SIZE / 2) * FACE_MASK_SIZE)

            # Create inverse mask to black out background
            mask_inv = cv2.bitwise_not(face_mask)
            # Apply black overlay to background
            black = np.zeros_like(blended)
            blended = cv2.copyTo(blended, face_mask) + cv2.copyTo(black, mask_inv)

            # Draw oval boundary (orange)
            cv2.ellipse(blended, center, (axes_x, axes_y), 0, 0, 360, (100, 200, 255), 2)
            cv2.putText(blended, "MASK", (center[0] - 20, center[1] - axes_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

        # mark apex frame
        local_frame_idx = i
        if local_frame_idx == apex_local or local_frame_idx == apex_local - 1:
            cv2.putText(blended, "APEX", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # frame counter
        cv2.putText(blended, f"{seg_start + i}", (4, CROP_SIZE - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        gif_frames.append(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

    gif_fps  = max(1.0, fps * GIF_SPEED)
    imageio.mimsave(str(gif_path), gif_frames, fps=gif_fps, loop=0)
    print(f"[INFO] Saved → {gif_path}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = Path(__file__).resolve().parent.parent
    video_root = root / VIDEO_ROOT
    output_root = root / OUTPUT_ROOT

    # Load Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        sys.exit("[ERROR] Could not load Haar cascade.")

    # Find all videos
    videos = find_all_videos(video_root)
    if not videos:
        sys.exit(f"[ERROR] No video files found in {video_root}")

    print(f"\n{'='*70}")
    print(f"Found {len(videos)} video(s)")
    print(f"{'='*70}")
    for video_path, dataset_name, video_stem in videos:
        print(f"  • {dataset_name}/{video_stem}")

    # Process each video
    succeeded = 0
    failed = 0
    for video_path, dataset_name, video_stem in videos:
        try:
            success = process_video(video_path, dataset_name, video_stem,
                                    output_root, cascade)
            if success:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Exception processing {dataset_name}/{video_stem}:")
            print(f"  {e}")
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos    : {len(videos)}")
    print(f"Successful      : {succeeded}")
    print(f"Failed          : {failed}")
    print(f"\nOutput structure:")
    print(f"  NPZ files       : {output_root / 'optical_flow_npz'}")
    print(f"  GIF files       : {output_root / 'flow_visualization_gif'}")
    print(f"  Naming          : <dataset>_<video_stem>.<ext>")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
