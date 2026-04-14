"""
Raw Facial Landmark Tracking (Frame-by-Frame)
==============================================
Tracks facial landmarks on raw frames with no preprocessing.

- No face crop
- No video stabilization
- No normalization
- Just landmarks frame-by-frame

Landmarks: 106 points (insightface buffalo_l landmark_2d_106)

GIF output shows:
  - All 106 landmarks as colored dots
  - Optical flow arrows: per-landmark displacement vector between
    consecutive frames (frame t-1 → frame t), scaled for visibility

Usage:
    python scripts/face_landmarks_raw.py

Output:
    output/landmarks_raw_npz/
        ├── ADFES_video_name.npz
        └── JeFEE_video_name.npz
    output/landmarks_raw_gif/
        ├── ADFES_video_name.gif
        └── JeFEE_video_name.gif
"""

import sys
from pathlib import Path

import cv2
import imageio
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("[ERROR] insightface not installed. Install with:")
    print("  pip install insightface onnxruntime")
    sys.exit(1)

# Configuration
VIDEO_ROOT = "video"
OUTPUT_ROOT = "output"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Visualization
LANDMARK_RADIUS = 3
BBOX_COLOR = (0, 255, 0)  # Green
GIF_SPEED = 0.5  # Fraction of real speed
ARROW_SCALE = 3.0          # Multiply raw displacement to make arrows visible
ARROW_MIN_LENGTH = 1.0     # Skip arrows shorter than this (pixels) to reduce clutter
ARROW_COLOR = (0, 200, 255)  # Yellow-ish (BGR) for all OF arrows

# Landmark colors: rotate through distinct colors for visibility
def get_landmark_color(idx):
    """Generate distinct colors for landmarks."""
    colors = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
    ]
    return colors[idx % len(colors)]

# insightface generates 106 landmarks per face
# We'll track all of them for comprehensive facial analysis
N_LANDMARKS = 106
LANDMARKS = [f"landmark_{i}" for i in range(N_LANDMARKS)]


def find_all_videos(root_dir: Path) -> list[tuple[Path, str, str]]:
    """Find all video files recursively."""
    videos = []
    for video_path in root_dir.rglob("*"):
        if video_path.suffix.lower() in VIDEO_EXTENSIONS and video_path.is_file():
            dataset_name = video_path.parent.name
            video_stem = video_path.stem
            videos.append((video_path, dataset_name, video_stem))
    return sorted(videos)


def load_video(path: Path) -> tuple[list[np.ndarray], float]:
    """Load all frames from video."""
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


def detect_and_estimate_landmarks(frame_bgr: np.ndarray, face_analyzer) -> tuple[tuple[int, int, int, int] | None, dict[str, np.ndarray]]:
    """
    Detect face and estimate 106 landmarks using insightface.
    Returns (bbox, landmarks_dict).
    """
    landmarks = {}
    bbox = None

    faces = face_analyzer.get(frame_bgr)

    if faces:
        # Use first (largest) face
        face = faces[0]

        # Get bbox in format (x, y, w, h)
        bbox_orig = face.bbox
        x1, y1, x2, y2 = [int(v) for v in bbox_orig]
        w = x2 - x1
        h = y2 - y1
        bbox = (x1, y1, w, h)

        # Get all 106 landmarks via landmark_2d_106 (not face.kps which is 5-point)
        lm106 = getattr(face, "landmark_2d_106", None)
        if lm106 is not None and len(lm106) == N_LANDMARKS:
            for i, kp in enumerate(lm106):
                landmarks[f"landmark_{i}"] = np.array(kp, dtype=np.float32)
        else:
            # Fallback: insightface model doesn't expose landmark_2d_106
            for name in LANDMARKS:
                landmarks[name] = None
    else:
        # No face detected
        for name in LANDMARKS:
            landmarks[name] = None

    return bbox, landmarks


def create_visualization_gif(frames: list[np.ndarray], bboxes: list, landmarks_sequence: list,
                             gif_path: Path, fps: float) -> None:
    """Create GIF visualization with landmarks."""
    print("[INFO] Creating visualization GIF …")

    gif_frames = []
    n_frames = len(frames)

    for i, frame in enumerate(frames):
        canvas = frame.copy()

        # Draw bbox
        bbox = bboxes[i]
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), BBOX_COLOR, 2)

        lm_dict = landmarks_sequence[i]
        prev_lm_dict = landmarks_sequence[i - 1] if i > 0 else None

        # Draw OF arrows (landmark displacement frame t-1 → t)
        if lm_dict is not None and prev_lm_dict is not None:
            for name in LANDMARKS:
                curr = lm_dict.get(name)
                prev = prev_lm_dict.get(name)
                if curr is None or prev is None:
                    continue
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                length = np.sqrt(dx * dx + dy * dy)
                if length < ARROW_MIN_LENGTH:
                    continue
                pt1 = tuple(prev.astype(int))
                pt2 = (int(prev[0] + dx * ARROW_SCALE),
                        int(prev[1] + dy * ARROW_SCALE))
                cv2.arrowedLine(canvas, pt1, pt2, ARROW_COLOR, 1,
                                tipLength=0.3)

        # Draw landmarks (on top of arrows)
        if lm_dict is not None:
            for name, lm in lm_dict.items():
                if lm is not None:
                    pos = tuple(lm.astype(int))
                    idx = int(name.split('_')[1])
                    color = get_landmark_color(idx)
                    cv2.circle(canvas, pos, LANDMARK_RADIUS, color, -1)
                    cv2.circle(canvas, pos, LANDMARK_RADIUS, (255, 255, 255), 1)

        # Frame counter
        cv2.putText(canvas, f"Frame {i+1}/{n_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show face detection status
        status = "Face detected" if bbox is not None else "No face"
        cv2.putText(canvas, status, (10, canvas.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if bbox is not None else (0, 0, 255), 1)

        gif_frames.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

        if (i + 1) % 50 == 0 or (i + 1) == n_frames:
            print(f"  … rendered {i+1}/{n_frames} frames")

    # Save GIF
    gif_fps = max(1.0, fps * GIF_SPEED)
    imageio.mimsave(str(gif_path), gif_frames, fps=gif_fps, loop=0)
    print(f"[INFO] Saved → {gif_path}")


def process_video(video_path: Path, dataset_name: str, video_stem: str,
                  output_root: Path, face_analyzer) -> bool:
    """Process video: detect faces and landmarks frame-by-frame with insightface."""
    print(f"\n{'='*70}")
    print(f"Processing: {dataset_name}/{video_stem}")
    print(f"{'='*70}")

    # Load video
    frames, fps = load_video(video_path)
    if not frames:
        print(f"[ERROR] Failed to load video")
        return False

    n_frames = len(frames)

    # Detect faces and landmarks frame-by-frame
    print("[INFO] Detecting faces and landmarks …")
    bboxes = []
    landmarks_sequence = []
    for i, frame in enumerate(frames):
        bbox, landmarks = detect_and_estimate_landmarks(frame, face_analyzer)
        bboxes.append(bbox)
        landmarks_sequence.append(landmarks)
        if (i + 1) % 50 == 0 or (i + 1) == n_frames:
            detected = sum(1 for b in bboxes if b is not None)
            print(f"  … {i+1}/{n_frames} frames ({detected} with face)")

    detected = sum(1 for b in bboxes if b is not None)
    print(f"[INFO] Face detected in {detected}/{n_frames} frames")

    # Create GIF visualization
    gif_dir = output_root / "landmarks_raw_gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"{dataset_name}_{video_stem}.gif"
    create_visualization_gif(frames, bboxes, landmarks_sequence, gif_path, fps)

    # Save NPZ
    npz_dir = output_root / "landmarks_raw_npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{dataset_name}_{video_stem}.npz"

    print("[INFO] Saving landmarks …")

    # Convert to arrays
    landmarks_array = {}
    for name in LANDMARKS:
        lm_track = []
        for lm_dict in landmarks_sequence:
            lm_track.append(lm_dict[name])
        landmarks_array[name] = np.array(lm_track)  # (N, 2) or (N,) if None

    # Save
    np.savez_compressed(
        str(npz_path),
        fps=np.float32(fps),
        n_frames=np.int32(n_frames),
        bboxes=np.array([(b if b is not None else (0, 0, 0, 0)) for b in bboxes]),
        bboxes_detected=[b is not None for b in bboxes],
        **landmarks_array
    )
    print(f"[INFO] Saved → {npz_path}")

    return True


def main():
    root = Path(__file__).resolve().parent.parent
    video_root = root / VIDEO_ROOT
    output_root = root / OUTPUT_ROOT

    # Initialize insightface
    print("[INFO] Initializing insightface face analyzer …")
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    # Find videos
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
                                  output_root, face_analyzer)
            if success:
                succeeded += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[ERROR] Exception: {e}")
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
    print(f"Output:")
    print(f"  NPZ data      : {output_root / 'landmarks_raw_npz'}")
    print(f"  GIF viz       : {output_root / 'landmarks_raw_gif'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
