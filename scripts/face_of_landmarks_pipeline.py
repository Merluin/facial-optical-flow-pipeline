"""
Facial Expression Landmark Tracking Pipeline
=============================================
Tracks specific anatomical landmarks instead of pixel-level optical flow.

Landmarks tracked:
  - Left eye center
  - Right eye center
  - Left mouth corner (modiolus angoli oris)
  - Right mouth corner (modiolus angoli oris)

Steps (per video):
  1. Load video frames
  2. Detect face using OpenCV Haar cascade
  3a. Smooth bounding boxes (temporal stability)
  3b. Standardise face position (align, crop) + affine registration to frame 0
  4. Detect facial landmarks using MediaPipe Face Mesh
  5. Extract motion of target landmarks
  6. Detect the expression apex frame (max landmark motion)
  7. Segment a fixed window around the apex
  8. Save landmark trajectories to NPZ
  9. Export visualization with landmark tracks

Usage (from project root, with face_of conda env active):
    python scripts/face_of_landmarks_pipeline.py

Output structure:
    output/
    ├── landmark_tracks_npz/
    │   ├── ADFES_video1_name.npz
    │   └── JeFEE_video1_name.npz
    └── landmark_visualization_gif/
        ├── ADFES_video1_name.gif
        └── JeFEE_video1_name.gif
"""

import math
import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy import ndimage
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

# ---------------------------------------------------------------------------
# Configuration — edit these to change behaviour
# ---------------------------------------------------------------------------
VIDEO_ROOT  = "video"        # scan this folder for all videos
OUTPUT_ROOT = "output"       # organize outputs by file type

CROP_SIZE       = 256        # output face square in pixels
PAD_FACTOR      = 0.3        # extra padding fraction around face bounding box
APEX_WINDOW_BEFORE = 40      # frames BEFORE apex to include in segment
APEX_WINDOW_AFTER  = 5       # frames AFTER apex to include in segment

# Visualization
LANDMARK_RADIUS = 4          # circle radius for landmarks (px)
TRAIL_LENGTH    = 10         # show trail for last N frames
GIF_SPEED       = 0.5        # fraction of real speed for GIF (0.5 = half speed)

# MediaPipe Face Mesh landmarks to track
# Reference: https://mediapipe.dev/images/face_mesh_io_updated.png
LANDMARKS_TO_TRACK = {
    "left_eye_center": 468,           # left eye iris center
    "right_eye_center": 473,          # right eye iris center
    "left_mouth_corner": 61,          # left mouth corner (modiolus angoli oris)
    "right_mouth_corner": 291,        # right mouth corner (modiolus angoli oris)
}

# Video file extensions to search for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_all_videos(root_dir: Path) -> list[tuple[Path, str, str]]:
    """
    Recursively find all video files.
    Returns list of (video_path, dataset_name, video_stem).
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


def detect_landmarks(frame_bgr: np.ndarray, face_mesh) -> dict[str, np.ndarray] | None:
    """
    Detect facial landmarks using MediaPipe Face Mesh.
    Returns dict of {landmark_name: (x, y)} or None if no face detected.
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = {}
    face_landmarks = results.multi_face_landmarks[0]

    for name, idx in LANDMARKS_TO_TRACK.items():
        lm = face_landmarks.landmark[idx]
        # Convert normalized coordinates to pixel coordinates
        x = int(lm.x * w)
        y = int(lm.y * h)
        landmarks[name] = np.array([x, y], dtype=np.float32)

    return landmarks


def compute_landmark_motion(landmarks_sequence: list[dict]) -> np.ndarray:
    """
    Compute total motion magnitude for each frame.
    Returns array of shape (n_frames,) with total displacement from frame 0.
    """
    motion = np.zeros(len(landmarks_sequence), dtype=np.float32)

    if not landmarks_sequence[0]:
        return motion

    ref_landmarks = landmarks_sequence[0]

    for i in range(len(landmarks_sequence)):
        if landmarks_sequence[i] is None:
            motion[i] = motion[i - 1] if i > 0 else 0
            continue

        total_disp = 0
        for name in LANDMARKS_TO_TRACK:
            ref_pos = ref_landmarks[name]
            curr_pos = landmarks_sequence[i][name]
            disp = np.linalg.norm(curr_pos - ref_pos)
            total_disp += disp

        motion[i] = total_disp

    return motion


def process_video(video_path: Path, dataset_name: str, video_stem: str,
                  output_root: Path, cascade, face_mesh) -> bool:
    """Process a single video for landmark tracking."""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}/{video_stem}")
    print(f"{'='*70}")

    npz_dir = output_root / "landmark_tracks_npz"
    gif_dir = output_root / "landmark_visualization_gif"
    npz_dir.mkdir(parents=True, exist_ok=True)
    gif_dir.mkdir(parents=True, exist_ok=True)

    file_prefix = f"{dataset_name}_{video_stem}"
    npz_path = npz_dir / f"{file_prefix}.npz"
    gif_path = gif_dir / f"{file_prefix}.gif"

    # ── 1. Load video ────────────────────────────────────────────────────────
    frames_raw, fps = load_video(video_path)
    if not frames_raw:
        print(f"[ERROR] Failed to load video: {video_path}")
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
        print(f"[ERROR] No face detected in any frame, skipping.")
        return False

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

    # ── 4. Detect landmarks in standardized frames ───────────────────────────
    print("[INFO] Detecting facial landmarks …")
    landmarks_sequence = []
    for i, frame in enumerate(std_frames):
        lm = detect_landmarks(frame, face_mesh)
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
    print(f"[INFO] Segment: frames {seg_start}–{seg_end-1} ({T} frames), "
          f"apex at local index {apex_local}")

    # ── 7. Save NPZ ──────────────────────────────────────────────────────────
    # Convert landmarks to arrays for saving
    landmarks_array = {}
    for name in LANDMARKS_TO_TRACK:
        lm_track = []
        for lm_dict in seg_landmarks:
            if lm_dict is not None:
                lm_track.append(lm_dict[name])
            else:
                lm_track.append(np.array([0, 0], dtype=np.float32))
        landmarks_array[name] = np.stack(lm_track)  # (T, 2)

    np.savez_compressed(
        str(npz_path),
        frames            = np.stack(seg_frames),              # (T, H, W, 3)
        left_eye_center   = landmarks_array["left_eye_center"],
        right_eye_center  = landmarks_array["right_eye_center"],
        left_mouth_corner = landmarks_array["left_mouth_corner"],
        right_mouth_corner = landmarks_array["right_mouth_corner"],
        apex_local        = np.int32(apex_local),
        apex_global       = np.int32(apex_global),
        seg_start         = np.int32(seg_start),
        seg_end           = np.int32(seg_end),
        fps               = np.float32(fps),
        motion_all        = motion,
    )
    print(f"[INFO] Saved → {npz_path}")

    # ── 8. GIF visualisation ─────────────────────────────────────────────────
    print("[INFO] Rendering GIF with landmark tracks …")

    # Color map for landmarks
    colors = {
        "left_eye_center": (255, 0, 0),       # Blue
        "right_eye_center": (255, 0, 0),      # Blue
        "left_mouth_corner": (0, 255, 0),     # Green
        "right_mouth_corner": (0, 255, 0),    # Green
    }

    gif_frames = []
    for i, frame in enumerate(seg_frames):
        canvas = frame.copy()

        # Draw landmarks and trails
        for name in LANDMARKS_TO_TRACK:
            color = colors[name]

            # Draw trail (previous positions)
            trail_start = max(0, i - TRAIL_LENGTH)
            for j in range(trail_start, i):
                if seg_landmarks[j] is not None:
                    prev_pos = seg_landmarks[j][name].astype(int)
                    # Fade trail
                    alpha = (j - trail_start + 1) / (i - trail_start + 1)
                    fade_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(canvas, tuple(prev_pos), LANDMARK_RADIUS // 2, fade_color, 1)

            # Draw current landmark
            if seg_landmarks[i] is not None:
                pos = seg_landmarks[i][name].astype(int)
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

    # Initialize MediaPipe Face Mesh
    face_mesh = solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

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
                                    output_root, cascade, face_mesh)
            if success:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Exception processing {dataset_name}/{video_stem}:")
            print(f"  {e}")
            failed += 1

    face_mesh.close()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total videos    : {len(videos)}")
    print(f"Successful      : {succeeded}")
    print(f"Failed          : {failed}")
    print(f"\nOutput structure:")
    print(f"  NPZ files       : {output_root / 'landmark_tracks_npz'}")
    print(f"  GIF files       : {output_root / 'landmark_visualization_gif'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
