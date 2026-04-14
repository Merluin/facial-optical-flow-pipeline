"""
Facial Expression Landmark Tracking Pipeline
=============================================
Complete pipeline for analyzing facial expressions using landmark tracking.

Instead of dense pixel-level optical flow, this tracks 106 facial landmarks
from insightface, computing their motion and displacement during expression.

Steps (per video):
  1. Load video frames
  2. Detect face using OpenCV Haar cascade
  3. Smooth bounding boxes (temporal stability)
  4. Standardise face position (align, crop) + affine registration to frame 0
  5. Detect 106 landmarks using insightface on standardized frames
  6. Compute landmark motion magnitude (apex detection)
  7. Segment a fixed window around the apex
  8. Save landmark trajectories to NPZ
  9. Export GIF with landmark motion visualization

Output structure:
    output/
    ├── landmark_npz/
    │   ├── ADFES_video_name.npz
    │   └── JeFEE_video_name.npz
    └── landmark_gif/
        ├── ADFES_video_name.gif
        └── JeFEE_video_name.gif

Usage (from project root, with face_of conda env active):
    python scripts/face_of_pipeline.py
"""

import math
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy import ndimage

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] insightface not installed. Install with:")
    print("  pip install insightface onnxruntime")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_ROOT = "video"
OUTPUT_ROOT = "output"

CROP_SIZE = 256        # output face square in pixels
PAD_FACTOR = 0.3       # extra padding fraction around face bounding box
APEX_WINDOW_BEFORE = 40  # frames BEFORE apex to include in segment
APEX_WINDOW_AFTER = 5    # frames AFTER apex to include in segment

# Visualization
LANDMARK_RADIUS = 3
TRAIL_LENGTH = 10      # show trail for last N frames
GIF_SPEED = 0.5        # fraction of real speed for GIF (0.5 = half speed)

# Landmarks: insightface provides 106 points
N_LANDMARKS = 106

# Video file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_all_videos(root_dir: Path) -> list[tuple[Path, str, str]]:
    """Recursively find all video files."""
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
    """Detect face using Haar cascade. Returns (x, y, w, h) or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                     minSize=(50, 50))
    if len(faces) == 0:
        return None
    return tuple(faces[np.argmax(faces[:, 2] * faces[:, 3])])


def estimate_eye_points(x: int, y: int, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate left and right eye centers from face bounding box."""
    eye_y = y + int(0.35 * h)
    left_eye  = np.array([x + int(0.25 * w), eye_y], dtype=np.float32)
    right_eye = np.array([x + int(0.75 * w), eye_y], dtype=np.float32)
    return left_eye, right_eye


def get_face_keypoints(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Estimate 3 key facial points from bounding box: left eye, right eye, nose."""
    left_eye  = np.array([x + int(0.25 * w), y + int(0.35 * h)], dtype=np.float32)
    right_eye = np.array([x + int(0.75 * w), y + int(0.35 * h)], dtype=np.float32)
    nose      = np.array([x + int(0.50 * w), y + int(0.55 * h)], dtype=np.float32)
    return np.array([left_eye, right_eye, nose], dtype=np.float32)


def align_and_crop(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                   crop_size: int, pad_factor: float) -> np.ndarray:
    """Rotate frame so eyes are horizontal, then crop to square."""
    h, w = frame_bgr.shape[:2]
    x, y, bw, bh = bbox

    le, re = estimate_eye_points(x, y, bw, bh)
    dx, dy = re - le
    angle = math.degrees(math.atan2(dy, dx))
    cx, cy = w / 2, h / 2
    M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame_bgr, M_rot, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

    bbox_center = np.array([x + bw/2, y + bh/2, 1.0], dtype=np.float32)
    bbox_center_rot = M_rot @ bbox_center

    pad = pad_factor * max(bw, bh)
    x_min = bbox_center_rot[0] - bw/2 - pad
    y_min = bbox_center_rot[1] - bh/2 - pad
    x_max = x_min + bw + 2*pad
    y_max = y_min + bh + 2*pad

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
    for i in range(4):
        smoothed[:, i] = ndimage.median_filter(bboxes_array[:, i], size=kernel_size)
    return [tuple(int(v) for v in b) for b in smoothed]


def register_to_reference(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int],
                          ref_keypoints: np.ndarray, ref_bbox: tuple[int, int, int, int],
                          crop_size: int, pad_factor: float) -> np.ndarray:
    """Register frame to reference by computing affine transform."""
    x, y, bw, bh = bbox
    curr_keypoints = get_face_keypoints(x, y, bw, bh)
    M = cv2.getAffineTransform(curr_keypoints, ref_keypoints)

    h, w = frame_bgr.shape[:2]
    warped = cv2.warpAffine(frame_bgr, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    ref_x, ref_y, ref_w, ref_h = ref_bbox
    pad = pad_factor * max(ref_w, ref_h)
    x_min = int(ref_x + ref_w/2 - ref_w/2 - pad)
    y_min = int(ref_y + ref_h/2 - ref_h/2 - pad)
    x_max = int(x_min + ref_w + 2*pad)
    y_max = int(y_min + ref_h + 2*pad)

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


def detect_landmarks_on_frame(frame_bgr: np.ndarray, face_analyzer) -> np.ndarray | None:
    """
    Detect 106 landmarks using insightface.
    Returns (106, 2) array of landmark coordinates or None if no face.
    """
    faces = face_analyzer.get(frame_bgr)
    if not faces:
        return None

    # Use first (largest) face
    kps = faces[0].kps  # Shape: (106, 2)
    return np.array(kps, dtype=np.float32)


def compute_landmark_motion(landmarks_sequence: list[np.ndarray | None]) -> np.ndarray:
    """
    Compute total landmark motion magnitude for each frame.
    Returns array of shape (n_frames,) with total displacement from frame 0.
    """
    motion = np.zeros(len(landmarks_sequence), dtype=np.float32)

    # Find first valid landmarks as reference
    ref_landmarks = None
    for lm in landmarks_sequence:
        if lm is not None:
            ref_landmarks = lm
            break

    if ref_landmarks is None:
        return motion

    for i in range(len(landmarks_sequence)):
        if landmarks_sequence[i] is None:
            motion[i] = motion[i - 1] if i > 0 else 0
            continue

        # Sum displacement of all landmarks
        displacements = landmarks_sequence[i] - ref_landmarks
        total_motion = np.sum(np.linalg.norm(displacements, axis=1))
        motion[i] = total_motion

    return motion


def get_landmark_color(idx: int) -> tuple:
    """Generate distinct colors for landmarks (BGR)."""
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
    ]
    return colors[idx % len(colors)]


def process_video(video_path: Path, dataset_name: str, video_stem: str,
                  output_root: Path, cascade, face_analyzer) -> bool:
    """Process a single video: detect, standardize, track landmarks."""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}/{video_stem}")
    print(f"{'='*70}")

    # Output folders
    npz_dir = output_root / "landmark_npz"
    gif_dir = output_root / "landmark_gif"
    npz_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{dataset_name}_{video_stem}"
    npz_path = npz_dir / f"{file_prefix}.npz"
    gif_path = gif_dir / f"{file_prefix}.gif"

    # ── 1. Load video ────────────────────────────────────────────────────────
    frames_raw, fps = load_video(video_path)
    if not frames_raw:
        print(f"[ERROR] Failed to load video")
        return False
    n_frames = len(frames_raw)

    # ── 2. Face detection ────────────────────────────────────────────────────
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
        print(f"[ERROR] No face detected, skipping")
        return False

    # Fill None entries with nearest valid bbox
    prev_bbox = next((bbox for bbox in bboxes if bbox is not None), None)
    filled_bbox = []
    for bbox in bboxes:
        if bbox is not None:
            prev_bbox = bbox
        filled_bbox.append(prev_bbox)

    # ── 3a. Smooth bounding boxes ────────────────────────────────────────────
    print("[INFO] Smoothing bounding boxes …")
    filled_bbox = smooth_bboxes(filled_bbox, kernel_size=5)

    # ── 3b. Face standardisation with affine registration ──────────────────────
    print("[INFO] Standardising face position + affine registration …")
    ref_bbox = filled_bbox[0]
    ref_keypoints = get_face_keypoints(*ref_bbox)

    std_frames = [align_and_crop(frames_raw[0], filled_bbox[0], CROP_SIZE, PAD_FACTOR)]

    for i in range(1, n_frames):
        frame_registered = register_to_reference(
            frames_raw[i], filled_bbox[i], ref_keypoints, ref_bbox, CROP_SIZE, PAD_FACTOR
        )
        std_frames.append(frame_registered)

    # ── 4. Detect landmarks on standardized frames ───────────────────────────
    print("[INFO] Detecting 106 landmarks on standardized frames …")
    landmarks_sequence = []
    for i, frame in enumerate(std_frames):
        lm = detect_landmarks_on_frame(frame, face_analyzer)
        landmarks_sequence.append(lm)
        if (i + 1) % 20 == 0:
            print(f"  … {i+1}/{n_frames}")

    detected_lm = sum(1 for lm in landmarks_sequence if lm is not None)
    print(f"[INFO] Landmarks detected in {detected_lm}/{n_frames} frames")

    # ── 5. Compute landmark motion ───────────────────────────────────────────
    print("[INFO] Computing landmark motion …")
    motion = compute_landmark_motion(landmarks_sequence)
    apex_global = int(np.argmax(motion))
    print(f"[INFO] Apex frame: {apex_global} (motion={motion[apex_global]:.1f})")

    # ── 6. Segmentation around apex ──────────────────────────────────────────
    seg_start = max(0, apex_global - APEX_WINDOW_BEFORE)
    seg_end   = min(n_frames, apex_global + APEX_WINDOW_AFTER + 1)
    apex_local = apex_global - seg_start
    seg_frames = std_frames[seg_start:seg_end]
    seg_landmarks = landmarks_sequence[seg_start:seg_end]
    T = len(seg_frames)
    print(f"[INFO] Segment: frames {seg_start}–{seg_end-1} ({T} frames), apex at index {apex_local}")

    # ── 7. Save NPZ ──────────────────────────────────────────────────────────
    print("[INFO] Saving landmark trajectories …")
    landmarks_array = {}
    for i in range(N_LANDMARKS):
        lm_track = []
        for lm_dict in seg_landmarks:
            if lm_dict is not None and i < len(lm_dict):
                lm_track.append(lm_dict[i])
            else:
                lm_track.append(np.array([0, 0], dtype=np.float32))
        landmarks_array[f"landmark_{i}"] = np.stack(lm_track)  # (T, 2)

    np.savez_compressed(
        str(npz_path),
        frames            = np.stack(seg_frames),              # (T, H, W, 3)
        apex_local        = np.int32(apex_local),
        apex_global       = np.int32(apex_global),
        seg_start         = np.int32(seg_start),
        seg_end           = np.int32(seg_end),
        fps               = np.float32(fps),
        motion_all        = motion,
        **landmarks_array
    )
    print(f"[INFO] Saved → {npz_path}")

    # ── 8. GIF visualisation ─────────────────────────────────────────────────
    print("[INFO] Rendering GIF with landmark tracks …")

    gif_frames = []
    for i, frame in enumerate(seg_frames):
        canvas = frame.copy()

        # Draw landmarks and trails
        if seg_landmarks[i] is not None:
            landmarks = seg_landmarks[i]
            for lm_idx, lm in enumerate(landmarks):
                color = get_landmark_color(lm_idx)

                # Draw trail (previous positions)
                trail_start = max(0, i - TRAIL_LENGTH)
                for j in range(trail_start, i):
                    if seg_landmarks[j] is not None and lm_idx < len(seg_landmarks[j]):
                        prev_lm = seg_landmarks[j][lm_idx].astype(int)
                        alpha = (j - trail_start + 1) / (i - trail_start + 1)
                        fade_color = tuple(int(c * alpha) for c in color)
                        cv2.circle(canvas, tuple(prev_lm), LANDMARK_RADIUS // 2, fade_color, 1)

                # Draw current landmark
                pos = lm.astype(int)
                cv2.circle(canvas, tuple(pos), LANDMARK_RADIUS, color, -1)
                cv2.circle(canvas, tuple(pos), LANDMARK_RADIUS, (255, 255, 255), 1)

        # Mark apex
        if i == apex_local or i == apex_local - 1:
            cv2.putText(canvas, "APEX", (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # Frame counter
        cv2.putText(canvas, f"{seg_start + i}", (4, CROP_SIZE - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        gif_frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    gif_fps = max(1.0, fps * GIF_SPEED)
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

    # Initialize insightface
    print("[INFO] Initializing insightface (106-point landmark detector) …")
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

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
                                    output_root, cascade, face_analyzer)
            if success:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Exception processing {dataset_name}/{video_stem}:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos    : {len(videos)}")
    print(f"Successful      : {succeeded}")
    print(f"Failed          : {failed}")
    print(f"\nOutput structure:")
    print(f"  NPZ files       : {output_root / 'landmark_npz'}")
    print(f"  GIF files       : {output_root / 'landmark_gif'}")
    print(f"  Naming          : <dataset>_<video_stem>.<ext>")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
