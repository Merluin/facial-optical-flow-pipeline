"""
Raw Facial Landmark Tracking (Frame-by-Frame)
==============================================
Tracks 106 facial landmarks on raw frames with head movement stabilization.

Features:
  - Detects all 106 face landmarks using insightface
  - Stabilizes head movement using face boundary landmarks (affine registration)
  - Computes expression-only optical flow (excludes head rotation/translation)
  - Creates GIF with per-frame motion arrows
  - Identifies expression apex (max expression intensity)
  - Saves apex frame visualization with weighted motion arrows

Landmarks: 106 points (insightface buffalo_l landmark_2d_106)

GIF visualization:
  - All 106 landmarks as colored dots
  - Yellow arrows: optical flow between consecutive frames (t-1 → t)
    (arrows show expression motion after head stabilization)

Apex image:
  - Blue dots: landmark positions at frame 0 (baseline)
  - Colored dots: landmark positions at apex frame
  - Arrows: cumulative motion from frame 0 → apex
  - Arrow thickness & brightness: proportional to motion magnitude

Usage:
    python scripts/face_landmarks_raw.py

Output:
    output/landmarks_raw_npz/        # Raw 106 landmarks per frame
        ├── ADFES_video_name.npz
        └── JeFEE_video_name.npz
    output/landmarks_raw_gif/        # Animation (stabilized landmarks)
        ├── ADFES_video_name.gif
        └── JeFEE_video_name.gif
    output/landmarks_raw_apex/       # Apex frame (expression intensity)
        ├── ADFES_video_name_apex.png
        └── JeFEE_video_name_apex.png
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
ARROW_SCALE = 15.0         # Exaggerate displacement 15x for visibility (typical ~0.3-0.5 px/frame motion)
ARROW_MIN_LENGTH = 0.05    # Only skip if real displacement < 0.05 px (almost no motion)
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


def stabilize_landmarks_affine(landmarks_dict: dict, ref_landmarks_dict: dict) -> dict:
    """
    Register landmarks to reference frame using affine transform.
    Uses face boundary landmarks (contour) for robust alignment.

    insightface 106-point layout:
      - 0-32: Face contour (jaw + cheeks)
      - Uses corners/edges of face boundary to compute transform

    Returns registered landmarks dict (same structure, transformed coordinates).
    """
    # Face boundary landmark indices (approx jaw & contour corners)
    BOUNDARY_INDICES = [0, 8, 16, 33, 50, 58, 68]  # key contour points

    # Extract boundary points from reference and current frame
    ref_pts = []
    curr_pts = []
    for idx in BOUNDARY_INDICES:
        ref_lm = ref_landmarks_dict.get(f"landmark_{idx}")
        curr_lm = landmarks_dict.get(f"landmark_{idx}")
        if ref_lm is not None and curr_lm is not None:
            ref_pts.append(ref_lm)
            curr_pts.append(curr_lm)

    if len(ref_pts) < 3:
        # Not enough points for affine — return original
        return landmarks_dict

    ref_pts = np.array(ref_pts, dtype=np.float32)
    curr_pts = np.array(curr_pts, dtype=np.float32)

    # Compute affine transform: curr_pts → ref_pts (register to reference)
    M = cv2.getAffineTransform(curr_pts[:3], ref_pts[:3])

    # Apply transform to all landmarks
    registered = {}
    for name, lm in landmarks_dict.items():
        if lm is None:
            registered[name] = None
        else:
            # Transform point: [x, y, 1] @ M^T → [x', y']
            pt_homog = np.array([lm[0], lm[1], 1.0], dtype=np.float32)
            pt_warped = M @ pt_homog
            registered[name] = pt_warped.astype(np.float32)

    return registered


def compute_landmark_motion(landmarks_sequence: list, stabilized_sequence: list = None) -> np.ndarray:
    """
    Compute cumulative motion magnitude per frame (from frame 0 to current).
    If stabilized_sequence provided, use that (head-stabilized landmarks).
    Otherwise use raw landmarks (includes head movement).

    Returns array of shape (n_frames,) with total displacement from frame 0.
    """
    # Use stabilized landmarks if provided, otherwise raw landmarks
    lm_seq = stabilized_sequence if stabilized_sequence is not None else landmarks_sequence

    motion = np.zeros(len(lm_seq), dtype=np.float32)

    # Find first frame with valid landmarks as reference
    ref_landmarks = None
    for lm_dict in lm_seq:
        if lm_dict is not None:
            has_valid = any(lm is not None for lm in lm_dict.values())
            if has_valid:
                ref_landmarks = lm_dict
                break

    if ref_landmarks is None:
        return motion

    for i, lm_dict in enumerate(lm_seq):
        if lm_dict is None:
            motion[i] = motion[i - 1] if i > 0 else 0
            continue

        total_displacement = 0.0
        for name in LANDMARKS:
            curr = lm_dict.get(name)
            ref = ref_landmarks.get(name)
            if curr is None or ref is None:
                continue
            displacement = np.linalg.norm(curr - ref)
            total_displacement += displacement

        motion[i] = total_displacement

    return motion


def create_apex_visualization(frame: np.ndarray, landmarks_0: dict, landmarks_apex: dict,
                              apex_idx: int, output_path: Path) -> None:
    """
    Create visualization of apex frame with arrows showing cumulative motion from frame 0.
    Arrow thickness and color intensity scaled by motion magnitude.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Compute per-landmark motion magnitudes (for arrow scaling)
    max_motion = 0.0
    motions = {}
    for name in LANDMARKS:
        curr = landmarks_apex.get(name)
        prev = landmarks_0.get(name)
        if curr is None or prev is None:
            motions[name] = 0.0
            continue
        displacement = np.linalg.norm(curr - prev)
        motions[name] = displacement
        max_motion = max(max_motion, displacement)

    # Draw arrows (baseline → apex)
    if max_motion > 0:
        norm = Normalize(vmin=0, vmax=max_motion)
    else:
        norm = None

    for name in LANDMARKS:
        curr = landmarks_apex.get(name)
        prev = landmarks_0.get(name)
        if curr is None or prev is None:
            continue

        displacement = motions[name]
        if displacement < 0.1:
            continue

        # Arrow thickness and color based on motion magnitude
        if norm is not None:
            intensity = norm(displacement)  # 0.0 to 1.0
        else:
            intensity = 0.5

        thickness = max(1, int(1 + intensity * 3))
        color_val = int(50 + intensity * 200)
        arrow_color = (color_val, color_val, 255)  # Red-ish, brighter = more motion

        # Arrow originates at APEX position, points forward along motion direction
        # motion_vec = apex - frame0 (direction and magnitude of motion)
        pt1 = tuple(curr.astype(int))  # Origin at apex landmark
        motion_vec = curr - prev  # frame0 -> apex displacement (direction + magnitude)
        pt2_pos = curr + motion_vec  # Points forward (continuing motion direction)
        pt2 = tuple(pt2_pos.astype(int))

        # Clamp to image bounds
        pt1 = (max(0, min(w - 1, pt1[0])), max(0, min(h - 1, pt1[1])))
        pt2 = (max(0, min(w - 1, pt2[0])), max(0, min(h - 1, pt2[1])))

        cv2.arrowedLine(canvas, pt1, pt2, arrow_color, thickness, tipLength=0.2)

    # Draw only apex landmarks (colored dots, no frame-0 landmarks)
    for name in LANDMARKS:
        lm_apex = landmarks_apex.get(name)
        if lm_apex is not None:
            idx = int(name.split('_')[1])
            color = get_landmark_color(idx)
            pos = tuple(lm_apex.astype(int))
            pos = (max(0, min(w - 1, pos[0])), max(0, min(h - 1, pos[1])))
            cv2.circle(canvas, pos, LANDMARK_RADIUS, color, -1)
            cv2.circle(canvas, pos, LANDMARK_RADIUS, (255, 255, 255), 1)

    # Add text annotations
    cv2.putText(canvas, f"APEX (Frame {apex_idx})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(canvas, "Expression Motion Trajectories (Frame 0 → Apex)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, "Colored dots=Apex  Arrows originate at apex, point forward (motion direction)", (10, canvas.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"[INFO] Saved apex visualization → {output_path}")


class Normalize:
    """Simple min-max normalization for arrow scaling."""
    def __init__(self, vmin=0, vmax=1):
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, value):
        if self.vmax == self.vmin:
            return 0.5
        return (value - self.vmin) / (self.vmax - self.vmin)


# Printed once to help diagnose which landmark attributes the model exposes
_face_debug_printed = False


def detect_and_estimate_landmarks(frame_bgr: np.ndarray, face_analyzer) -> tuple[tuple[int, int, int, int] | None, dict[str, np.ndarray]]:
    """
    Detect face and estimate 106 landmarks using insightface.
    Returns (bbox, landmarks_dict).

    insightface Face is a dict subclass, so use dict.get() to safely access
    optional keys — getattr with a default does NOT work because __getattr__
    raises KeyError (not AttributeError) on missing keys.
    """
    global _face_debug_printed
    landmarks = {}
    bbox = None

    faces = face_analyzer.get(frame_bgr)

    if faces:
        # Use first (largest) face
        face = faces[0]

        # One-time debug: show every key the model populated
        if not _face_debug_printed:
            print(f"[DEBUG] Face object keys: {list(face.keys())}")
            _face_debug_printed = True

        # bbox (x1, y1, w, h)
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        bbox = (x1, y1, x2 - x1, y2 - y1)

        # 106-point landmarks — buffalo_l runs 2d106det which sets this key
        # Use dict .get() because face is a dict subclass; getattr would KeyError
        lm106 = face.get("landmark_2d_106")

        if lm106 is not None and len(lm106) == N_LANDMARKS:
            for i, kp in enumerate(lm106):
                landmarks[f"landmark_{i}"] = np.array(kp, dtype=np.float32)
        else:
            print(f"[WARN] landmark_2d_106 not available "
                  f"(got {None if lm106 is None else len(lm106)} points). "
                  f"Check [DEBUG] line above for available keys.")
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

        # Draw OF arrows (landmark displacement from frame t-1 → frame t)
        if i > 0:
            prev_lm_dict = landmarks_sequence[i - 1]
            for name in LANDMARKS:
                curr = lm_dict.get(name)
                prev = prev_lm_dict.get(name)
                if curr is None or prev is None:
                    continue

                # Compute displacement and magnitude
                dx = float(curr[0] - prev[0])
                dy = float(curr[1] - prev[1])
                length = np.sqrt(dx * dx + dy * dy)

                # Skip near-zero motion
                if length < ARROW_MIN_LENGTH:
                    continue

                # Draw arrow: from prev position, scaled displacement
                pt1 = tuple(prev.astype(int))
                scaled_pt2 = prev + np.array([dx * ARROW_SCALE, dy * ARROW_SCALE])
                pt2 = tuple(scaled_pt2.astype(int))
                cv2.arrowedLine(canvas, pt1, pt2, ARROW_COLOR, 1, tipLength=0.3)

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

    # Head stabilization: register all landmarks to frame 0 using face boundary
    print("[INFO] Stabilizing head movement using face boundary landmarks …")
    stabilized_landmarks_sequence = [None] * n_frames
    if landmarks_sequence[0] is not None:
        ref_frame_landmarks = landmarks_sequence[0]
        for i, lm_dict in enumerate(landmarks_sequence):
            if lm_dict is None:
                stabilized_landmarks_sequence[i] = None
            elif i == 0:
                # Frame 0 is already the reference
                stabilized_landmarks_sequence[i] = lm_dict
            else:
                # Register frame i to frame 0
                stabilized_landmarks_sequence[i] = stabilize_landmarks_affine(lm_dict, ref_frame_landmarks)
        print(f"  … stabilized {n_frames} frames")
    else:
        stabilized_landmarks_sequence = landmarks_sequence
        print(f"  [WARN] No landmarks in frame 0, skipping stabilization")

    # Create GIF visualization (using stabilized landmarks)
    gif_dir = output_root / "landmarks_raw_gif"
    gif_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"{dataset_name}_{video_stem}.gif"
    create_visualization_gif(frames, bboxes, stabilized_landmarks_sequence, gif_path, fps)

    # Compute motion and find apex frame (on stabilized landmarks)
    print("[INFO] Computing expression motion to find apex …")
    motion = compute_landmark_motion(stabilized_landmarks_sequence)
    apex_idx = int(np.argmax(motion))
    print(f"[INFO] Apex frame: {apex_idx} (expression motion={motion[apex_idx]:.1f})")

    # Create apex visualization with arrows (baseline → apex, stabilized)
    if apex_idx > 0 and stabilized_landmarks_sequence[0] is not None and stabilized_landmarks_sequence[apex_idx] is not None:
        apex_dir = output_root / "landmarks_raw_apex"
        apex_dir.mkdir(parents=True, exist_ok=True)
        apex_path = apex_dir / f"{dataset_name}_{video_stem}_apex.png"
        create_apex_visualization(frames[apex_idx], stabilized_landmarks_sequence[0],
                                 stabilized_landmarks_sequence[apex_idx],
                                 apex_idx, apex_path)
    else:
        print(f"[WARN] Could not create apex visualization (invalid stabilized landmarks at frame 0 or {apex_idx})")

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
    print(f"  Apex image    : {output_root / 'landmarks_raw_apex'}")
    print(f"                  (cumulative motion from frame 0 → apex)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
