"""
Raw Facial Landmark Tracking Pipeline
======================================
Complete analysis of facial expression using 106-point landmark tracking with
head movement stabilization and optical flow visualization.

FEATURES
────────
✓ 106-point facial landmark detection (insightface buffalo_l model)
✓ Head stabilization via affine registration (uses face contour landmarks)
✓ Expression-only optical flow (eliminates head rotation & translation)
✓ Per-frame motion visualization (GIF with optical flow arrows)
✓ Apex detection (identifies peak expression frame)
✓ Motion trajectory visualization (apex frame with weighted arrows)
✓ Expression arc analysis (intensity curve over time)
✓ Professional summary report (PDF with all visualizations)

WORKFLOW
────────
1. Load video frames
2. Detect 106 landmarks on each frame using insightface
3. Stabilize landmarks using face boundary points (affine registration to frame 0)
4. Compute frame-to-frame optical flow (expression motion)
5. Identify apex frame (maximum cumulative displacement)
6. Generate outputs:
   - GIF: stabilized video with optical flow arrows
   - Apex image: peak expression with motion vectors
   - Arc plot: expression intensity curve
   - Summary PDF: professional report with all analyses

OUTPUTS
───────

NPZ Archives (Raw Data):
  output/landmarks_raw_npz/
    └── <dataset>_<video>.npz        # 106 landmarks per frame (raw coordinates)

GIF Animation (Frame-by-Frame):
  output/landmarks_raw_gif/
    └── <dataset>_<video>.gif        # Stabilized video with optical flow arrows
                                      # Yellow arrows show expression motion
                                      # Between consecutive frames (t-1 → t)

Expression Analysis (Images & Report):
  output/landmarks_raw_apex/
    ├── <dataset>_<video>_apex.png       # Apex frame with motion vectors
    │                                     # • Colored dots: landmark positions
    │                                     # • Arrows: motion direction & magnitude
    │                                     # • Origin: apex position
    │                                     # • Direction: forward along motion
    ├── <dataset>_<video>_arc.png        # Expression arc plot
    │                                     # • X-axis: frame number
    │                                     # • Y-axis: motion magnitude
    │                                     # • Red line: marks apex frame
    └── <dataset>_<video>_summary.pdf    # Professional summary report
                                         # 2×2 grid layout with metadata:
                                         # ① Stabilized video frame (apex)
                                         # ② Apex image with motion vectors
                                         # ③ Expression arc plot
                                         # + Title, frame info, footer

VISUALIZATIONS EXPLAINED
────────────────────────

GIF (Stabilized Video):
  - 106 landmarks (colored dots) on each frame
  - Yellow optical flow arrows between consecutive frames
  - Arrows show expression motion (head movement already removed)
  - Green bbox shows detected face region
  - Green text shows frame counter and detection status

Apex Image (Peak Expression):
  - Background: the actual apex frame from the video
  - Colored dots: all 106 landmarks at peak expression
  - Arrows: originate at each landmark, point forward
  - Arrow length: proportional to displacement from frame 0
  - Arrow thickness & brightness: brighter/thicker = more motion
  - Visualizes which facial regions moved most (e.g., mouth, eyebrows)

Expression Arc:
  - Shows motion magnitude throughout the video
  - Smooth curve from onset to offset of expression
  - Peak marked with red vertical dashed line (apex)
  - Used to identify expression phases:
    * Onset: gradual increase
    * Apex: peak plateau
    * Offset: return to baseline

Summary PDF:
  - Professional 1-page report with all key visualizations
  - Title bar: video name, apex frame number, motion intensity
  - Top section: GIF frame + apex image side-by-side
  - Bottom section: expression arc plot
  - Footer: pipeline info and generation timestamp

PARAMETERS
──────────
  ARROW_SCALE = 15.0              # Exaggerate optical flow arrows by 15×
  ARROW_MIN_LENGTH = 0.05         # Skip arrows < 0.05 px (noise filtering)
  LANDMARK_RADIUS = 3             # Radius of landmark dot markers
  GIF_SPEED = 0.5                 # GIF playback speed (0.5 = half speed)

USAGE
─────
    python scripts/face_landmarks_raw.py

REQUIREMENTS
────────────
  - insightface with onnxruntime (106-point landmark detection)
  - opencv-python (cv2)
  - numpy
  - imageio (GIF creation)
  - matplotlib (plotting & PDF generation)
  - Pillow/PIL (image processing)
"""

import sys
from pathlib import Path

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt

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
N_LANDMARKS = 106
LANDMARKS = [f"landmark_{i}" for i in range(N_LANDMARKS)]

# KEY LANDMARKS FOR EXPRESSION ANALYSIS (4 points minimum)
# Anguli oris (mouth corners) + middle forehead (between eyebrows)
KEY_LANDMARK_INDICES = [
    84,    # Left mouth corner (anguli oris sinister)
    90,    # Right mouth corner (anguli oris dexter)
    38,    # Left eyebrow inner point (middle face, over left eyebrow)
    48,    # Right eyebrow inner point (middle face, over right eyebrow)
]
KEY_LANDMARKS = [f"landmark_{i}" for i in KEY_LANDMARK_INDICES]
N_KEY_LANDMARKS = len(KEY_LANDMARK_INDICES)


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


def create_apex_visualization_key(frame: np.ndarray, landmarks_0: dict, landmarks_apex: dict,
                                  apex_idx: int, output_path: Path) -> None:
    """
    Create apex visualization using only KEY LANDMARKS (subset for expression).
    Exaggerated arrow magnitude for clarity.
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Exaggeration factor for key landmarks visualization
    ARROW_SCALE_KEY = 40.0  # Highly exaggerate for visibility

    # Compute per-landmark motion magnitudes (for key landmarks only)
    max_motion = 0.0
    motions = {}
    for idx in KEY_LANDMARK_INDICES:
        name = f"landmark_{idx}"
        curr = landmarks_apex.get(name)
        prev = landmarks_0.get(name)
        if curr is None or prev is None:
            motions[name] = 0.0
            continue
        displacement = np.linalg.norm(curr - prev)
        motions[name] = displacement
        max_motion = max(max_motion, displacement)

    # Draw arrows (key landmarks only)
    if max_motion > 0:
        norm = Normalize(vmin=0, vmax=max_motion)
    else:
        norm = None

    for idx in KEY_LANDMARK_INDICES:
        name = f"landmark_{idx}"
        curr = landmarks_apex.get(name)
        prev = landmarks_0.get(name)
        if curr is None or prev is None:
            continue

        displacement = motions[name]
        if displacement < 0.05:  # Lower threshold for key landmarks
            continue

        # Arrow thickness and color based on motion magnitude (exaggerated)
        if norm is not None:
            intensity = norm(displacement)
        else:
            intensity = 0.5

        thickness = max(2, int(2 + intensity * 5))  # Thicker arrows
        color_val = int(100 + intensity * 155)  # Brighter colors
        arrow_color = (color_val, color_val, 255)

        # Arrow originates at apex, points forward (EXAGGERATED)
        pt1 = tuple(curr.astype(int))
        motion_vec = curr - prev
        pt2_pos = curr + motion_vec * ARROW_SCALE_KEY  # 40× exaggeration
        pt2 = tuple(pt2_pos.astype(int))

        # Clamp to bounds
        pt1 = (max(0, min(w - 1, pt1[0])), max(0, min(h - 1, pt1[1])))
        pt2 = (max(0, min(w - 1, pt2[0])), max(0, min(h - 1, pt2[1])))

        cv2.arrowedLine(canvas, pt1, pt2, arrow_color, thickness, tipLength=0.2)

    # Draw key landmarks (colored dots)
    for idx in KEY_LANDMARK_INDICES:
        name = f"landmark_{idx}"
        lm_apex = landmarks_apex.get(name)
        if lm_apex is not None:
            color = get_landmark_color(idx)
            pos = tuple(lm_apex.astype(int))
            pos = (max(0, min(w - 1, pos[0])), max(0, min(h - 1, pos[1])))
            cv2.circle(canvas, pos, LANDMARK_RADIUS + 1, color, -1)
            cv2.circle(canvas, pos, LANDMARK_RADIUS + 1, (255, 255, 255), 1)

    # Add text annotations
    cv2.putText(canvas, f"APEX (Frame {apex_idx}) - KEY LANDMARKS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(canvas, "Expression Motion Trajectories (Frame 0 → Apex)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, f"Key Landmarks: {N_KEY_LANDMARKS} points (2× anguli oris + 2× inner eyebrow)",
                (10, canvas.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"[INFO] Saved key landmarks apex visualization → {output_path}")


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


def create_expression_arc_plot(motion: np.ndarray, apex_idx: int, output_path: Path) -> None:
    """
    Create a plot showing expression intensity (motion magnitude) over time.
    Marks the apex (peak expression) with a vertical red dashed line.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)

    frames = np.arange(len(motion))
    ax.plot(frames, motion, linewidth=2, color='#1f77b4')
    ax.axvline(x=apex_idx, color='red', linestyle='--', linewidth=2, label='Apex')

    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Motion Magnitude', fontsize=12)
    ax.set_title('Expression Arc', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved expression arc plot → {output_path}")


def create_summary_pdf(gif_path: Path, apex_img_path: Path, arc_img_path: Path,
                       output_pdf_path: Path, apex_idx: int, dataset_name: str = "",
                       video_stem: str = "", n_frames: int = 0, apex_motion: float = 0.0) -> None:
    """
    Create a professional summary PDF with 2x2 grid layout and metadata.

    Layout:
      [Title with video info]
      [GIF frame (apex)] [Apex image with motion vectors]
      [Expression arc plot - spans both columns]
    """
    from PIL import Image
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle

    # Extract key frames from GIF (onset, mid, apex)
    gif_frames = imageio.mimread(str(gif_path))
    n_gif_frames = len(gif_frames)

    # Select frames to show progression: start, middle, apex
    frame_onset = gif_frames[0] if n_gif_frames > 0 else None
    frame_mid = gif_frames[n_gif_frames // 2] if n_gif_frames > 1 else frame_onset
    frame_apex = gif_frames[min(apex_idx, n_gif_frames - 1)] if n_gif_frames > 0 else None

    # Create a horizontal strip showing progression
    if frame_onset is not None and frame_mid is not None and frame_apex is not None:
        onset_pil = Image.fromarray(frame_onset)
        mid_pil = Image.fromarray(frame_mid)
        apex_pil = Image.fromarray(frame_apex)

        # Resize frames to same height for stitching
        h_target = 250
        aspect_onset = onset_pil.width / onset_pil.height
        aspect_mid = mid_pil.width / mid_pil.height
        aspect_apex = apex_pil.width / apex_pil.height

        onset_pil = onset_pil.resize((int(h_target * aspect_onset), h_target), Image.Resampling.LANCZOS)
        mid_pil = mid_pil.resize((int(h_target * aspect_mid), h_target), Image.Resampling.LANCZOS)
        apex_pil = apex_pil.resize((int(h_target * aspect_apex), h_target), Image.Resampling.LANCZOS)

        # Stitch frames horizontally
        total_width = onset_pil.width + mid_pil.width + apex_pil.width + 10
        gif_progression = Image.new('RGB', (total_width, h_target + 40), color='white')
        gif_progression.paste(onset_pil, (0, 0))
        gif_progression.paste(mid_pil, (onset_pil.width + 5, 0))
        gif_progression.paste(apex_pil, (onset_pil.width + mid_pil.width + 10, 0))
    else:
        gif_progression = Image.fromarray(frame_apex if frame_apex is not None else gif_frames[-1])

    # Load apex image
    apex_img = Image.open(str(apex_img_path))

    # Load arc plot
    arc_img = Image.open(str(arc_img_path))

    # Create PDF figure with GridSpec (title + 2x2 content + footer)
    fig = plt.figure(figsize=(16, 14), dpi=100, facecolor='white')
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.8, 3, 3, 0.8],
                          hspace=0.35, wspace=0.25, top=0.95, bottom=0.05)

    # ─── Title Section ───
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.set_xlim(0, 10)
    ax_title.set_ylim(0, 2)

    # Add colored background bar
    title_bar = Rectangle((0, 0), 10, 2, facecolor='#2c3e50', edgecolor='none', zorder=0)
    ax_title.add_patch(title_bar)

    # Title text
    title_text = f"Facial Expression Analysis: {dataset_name}/{video_stem}"
    ax_title.text(5, 1.4, title_text, fontsize=18, fontweight='bold', color='white',
                 ha='center', va='center', family='sans-serif')

    # Subtitle with frame info
    subtitle = f"Apex Frame: #{apex_idx} / {n_frames}  |  Expression Intensity: {apex_motion:.1f}"
    ax_title.text(5, 0.5, subtitle, fontsize=12, color='#ecf0f1', ha='center', va='center',
                 family='monospace', style='italic')

    # ─── GIF Progression (onset → apex) ───
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(gif_progression)
    ax1.set_title('① GIF Progression: Onset → Mid → Apex', fontsize=13, fontweight='bold',
                 loc='left', pad=10, color='#2c3e50')
    ax1.axis('off')

    # ─── Apex Image with Motion Vectors ───
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(apex_img)
    ax2.set_title('② Expression Motion Vectors (Frame 0 → Apex)', fontsize=13, fontweight='bold',
                 loc='left', pad=10, color='#2c3e50')
    ax2.axis('off')

    # ─── Arc Plot (spans both columns) ───
    ax3 = fig.add_subplot(gs[2, :])
    ax3.imshow(arc_img)
    ax3.set_title('③ Expression Arc: Motion Magnitude Over Time', fontsize=13, fontweight='bold',
                 loc='left', pad=10, color='#2c3e50')
    ax3.axis('off')

    # ─── Footer Section ───
    ax_footer = fig.add_subplot(gs[3, :])
    ax_footer.axis('off')
    ax_footer.set_xlim(0, 10)
    ax_footer.set_ylim(0, 2)

    # Footer bar
    footer_bar = Rectangle((0, 0), 10, 2, facecolor='#ecf0f1', edgecolor='#bdc3c7', linewidth=2, zorder=0)
    ax_footer.add_patch(footer_bar)

    # Footer text
    footer_text = ("Raw Facial Landmark Tracking Pipeline  |  106-point Landmark Detection with Head Stabilization  |  "
                  "Optical Flow based on Expression")
    ax_footer.text(5, 1.3, footer_text, fontsize=10, color='#2c3e50', ha='center', va='center',
                  family='sans-serif', wrap=True)

    date_text = f"Generated: {np.datetime64('today')}"
    ax_footer.text(5, 0.4, date_text, fontsize=9, color='#7f8c8d', ha='center', va='center',
                  family='monospace', style='italic')

    # Save PDF
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_pdf_path), format='pdf', bbox_inches='tight', dpi=100, facecolor='white')
    plt.close()
    print(f"[INFO] Saved summary PDF → {output_pdf_path}")


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

    # Create expression arc plot (motion magnitude over time)
    arc_dir = output_root / "landmarks_raw_apex"
    arc_dir.mkdir(parents=True, exist_ok=True)
    arc_path = arc_dir / f"{dataset_name}_{video_stem}_arc.png"
    create_expression_arc_plot(motion, apex_idx, arc_path)

    # Create apex visualization with arrows (baseline → apex, stabilized)
    apex_path = None
    apex_key_path = None
    if apex_idx > 0 and stabilized_landmarks_sequence[0] is not None and stabilized_landmarks_sequence[apex_idx] is not None:
        apex_dir = output_root / "landmarks_raw_apex"
        apex_dir.mkdir(parents=True, exist_ok=True)

        # Full 106 landmarks version
        apex_path = apex_dir / f"{dataset_name}_{video_stem}_apex.png"
        create_apex_visualization(frames[apex_idx], stabilized_landmarks_sequence[0],
                                 stabilized_landmarks_sequence[apex_idx],
                                 apex_idx, apex_path)

        # Key landmarks only version
        apex_key_path = apex_dir / f"{dataset_name}_{video_stem}_apex_key.png"
        create_apex_visualization_key(frames[apex_idx], stabilized_landmarks_sequence[0],
                                     stabilized_landmarks_sequence[apex_idx],
                                     apex_idx, apex_key_path)
    else:
        print(f"[WARN] Could not create apex visualization (invalid stabilized landmarks at frame 0 or {apex_idx})")

    # Create summary PDF (2x2 grid: GIF, Apex, Arc)
    if apex_path is not None and arc_path.exists():
        pdf_dir = output_root / "landmarks_raw_apex"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{dataset_name}_{video_stem}_summary.pdf"
        create_summary_pdf(gif_path, apex_path, arc_path, pdf_path, apex_idx,
                          dataset_name=dataset_name, video_stem=video_stem,
                          n_frames=n_frames, apex_motion=motion[apex_idx])
    else:
        print(f"[WARN] Could not create summary PDF (missing apex image or arc plot)")

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
