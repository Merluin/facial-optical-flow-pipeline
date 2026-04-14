# Facial Expression Landmark Tracking Pipeline

A comprehensive Python pipeline for analyzing facial expressions in video using 106-point landmark tracking. Designed for datasets like ADFES and JeFEE.

## Overview

This pipeline processes facial expression videos to track 106 facial landmarks during the peak of emotional expression. It standardizes face position, detects the expression apex, segments video around the peak, and tracks landmark motion throughout the segment.

**Key Features:**
- ✅ Batch processing of multiple video datasets
- ✅ Automatic face detection and standardization (rotation + square crop)
- ✅ Affine registration to remove rigid head motion
- ✅ 106-point landmark detection using insightface
- ✅ Apex detection using landmark motion magnitude
- ✅ Asymmetric segmentation (configurable before/after apex)
- ✅ Multi-format output (NPZ landmark trajectories + animated GIF visualization)

## Quick Start

### 1. Clone & Setup

```bash
# Navigate to project directory
cd path/to/facial-optical-flow-pipeline

# Create conda environment
conda env create -f environment.yml
conda activate face_of
```

### 2. Organize Videos

Place videos in dataset folders:
```
video/
├── ADFES/
│   ├── F01-Joy-Face Forward.mp4
│   ├── F03-Joy-Face Forward.mp4
│   └── ...
└── JeFEE/
    ├── JF1_Happiness.mp4
    ├── JF2_Happiness.mp4
    └── ...
```

The script automatically finds all videos recursively.

### 3. Run Pipeline

**Full Preprocessing + 106-Point Landmark Tracking (recommended):**
```bash
python scripts/face_of_pipeline.py
```

This includes:
- Face detection with Haar cascades
- Bounding box smoothing
- Face alignment and cropping
- Affine registration to remove head motion
- 106-point landmark detection from insightface
- Apex detection via landmark motion
- Asymmetric temporal segmentation

**Alternative: Raw Frame Tracking (no preprocessing):**
```bash
python scripts/face_landmarks_raw.py
```

Raw tracking on unprocessed video frames (no face alignment, no motion compensation).

### 4. Check Output

Full pipeline output:
```
output/
├── landmark_npz/
│   ├── ADFES_F01-Joy-Face Forward.npz
│   ├── ADFES_F03-Joy-Face Forward.npz
│   └── ...
└── landmark_gif/
    ├── ADFES_F01-Joy-Face Forward.gif
    ├── ADFES_F03-Joy-Face Forward.gif
    └── ...
```

Raw tracking output:
```
output/
├── landmarks_raw_npz/
│   └── ...
└── landmarks_raw_gif/
    └── ...
```

## Configuration

Edit `scripts/face_of_pipeline.py` configuration section:

```python
# Face standardization
CROP_SIZE       = 256        # output face square (pixels)
PAD_FACTOR      = 0.3        # padding around face (fraction of bbox)

# Apex segmentation
APEX_WINDOW_BEFORE = 40      # frames before apex
APEX_WINDOW_AFTER  = 5       # frames after apex

# Landmark visualization
LANDMARK_RADIUS = 3          # radius of landmark circles (pixels)
TRAIL_LENGTH    = 10         # show trail for last N frames

# GIF export
GIF_SPEED       = 0.5        # fraction of real speed (0.5 = half speed)
```

## Loading Results in Python

```python
import numpy as np

# Load landmark data
data = np.load("output/landmark_npz/ADFES_F01-Joy-Face Forward.npz")

# Access core data
frames = data["frames"]              # shape: (T, 256, 256, 3) — standardized face frames (BGR)
apex_local = int(data["apex_local"])    # frame index within segment
apex_global = int(data["apex_global"])  # frame index in full video
seg_start = int(data["seg_start"])      # segment start in full video
seg_end = int(data["seg_end"])          # segment end in full video
fps = float(data["fps"])
motion_all = data["motion_all"]     # landmark motion magnitude per frame

# Access landmark trajectories (106 landmarks, all tracked through segment)
landmark_0 = data["landmark_0"]     # shape: (T, 2) — [x, y] per frame
landmark_1 = data["landmark_1"]
# ... landmark_2 through landmark_105

# Example: compute displacement from first frame for a landmark
disp = landmark_0 - landmark_0[0]  # displacement relative to frame 0
speed = np.linalg.norm(disp, axis=1)  # speed magnitude per frame

print(f"Total frames in segment: {len(frames)}")
print(f"Apex at local frame {apex_local} (global {apex_global})")
print(f"Peak landmark motion: {motion_all.max():.2f} pixels")
print(f"FPS: {fps:.2f}")
```

## Output File Format

### NPZ (Landmark Tracking Data)

Binary compressed NumPy archive containing:

| Variable | Shape | Type | Description |
|----------|-------|------|-------------|
| `frames` | (T, 256, 256, 3) | uint8 | Standardized face frames (BGR) |
| `apex_local` | () | int32 | Apex frame index within segment |
| `apex_global` | () | int32 | Apex frame index in full video |
| `seg_start` | () | int32 | Segment start frame (full video) |
| `seg_end` | () | int32 | Segment end frame (full video) |
| `fps` | () | float32 | Video FPS |
| `motion_all` | (N,) | float32 | Landmark motion magnitude per frame (full video) |
| `landmark_0` through `landmark_105` | (T, 2) | float32 | [x, y] coordinates per frame for each of 106 landmarks |

Where T = number of frames in segment.

### GIF (Visualization)

Animated GIF showing per-frame landmark positions with:
- **Colored circles**: 106 landmarks (color cycles through 6 distinct hues)
- **Fading trails**: previous positions with alpha fade (shows motion)
- **White outline**: bright border around each landmark
- **Frame counter**: segment frame index (bottom left)
- **"APEX" label**: marks the apex frame

## Methodology

See `METHODOLOGY.html` for detailed step-by-step explanation of:
1. Video loading & preprocessing
2. Face detection with Haar cascades
3. Face standardization (alignment + cropping)
4. Bbox smoothing & affine registration
5. 106-point landmark detection (insightface)
6. Apex detection (landmark motion magnitude)
7. Asymmetric temporal segmentation
8. Landmark trajectory tracking
9. Visualization & output

## System Requirements

- **Python**: 3.10+
- **OS**: macOS, Linux, Windows
- **RAM**: 2-4 GB (depends on video resolution & duration)
- **Disk**: ~1 GB per 100 videos

## Dependencies

| Package | Purpose |
|---------|---------|
| opencv-python | Face detection, image processing, visualization |
| numpy | Array operations |
| scipy | Temporal smoothing (median filter) |
| imageio[ffmpeg] | GIF export |
| pillow | Image I/O |
| insightface | 106-point facial landmark detection |
| onnxruntime | Inference backend for insightface |

## Troubleshooting

**Issue**: "No face detected in any frame"
- **Solution**: Check video resolution (min 256×256), lighting, and face visibility

**Issue**: Landmarks are scattered or not tracking the face
- **Solution**: Landmarks may be detected on background; check that `PAD_FACTOR` and `CROP_SIZE` are appropriate for your video resolution

**Issue**: Apex frame seems incorrect
- **Solution**: Check motion curve: `plt.plot(data["motion_all"])`; apex is at `argmax(motion_all)`

**Issue**: GIF file is very large
- **Solution**: Reduce `GIF_SPEED` (try 0.25), `TRAIL_LENGTH`, or `APEX_WINDOW_BEFORE`/`APEX_WINDOW_AFTER`; use compression post-processing

## Paper References

**Landmark Detection:**
- Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). "RetinaFace: Single-stage Dense Face Localisation in the Wild"

**Face Detection:**
- Viola, P., & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features"

**Facial Expression Analysis:**
- Ekman, P., & Friesen, W. V. (1978). "Manual for the Facial Action Coding System"

## License

This project is provided for research purposes.

## Raw Frame Landmark Tracking

**Alternative approach:** Track 106 landmarks on raw video frames without preprocessing:

```bash
python scripts/face_landmarks_raw.py
```

This skips face standardization and produces landmarks in the original video coordinate system (no rotation alignment, no face cropping). Useful for:
- Tracking landmarks on unaligned faces
- Analyzing head motion separately
- Validating preprocessing effects

### Output
Saves raw landmark detection as NPZ files with:
- `landmark_0` through `landmark_105` — (T, 2) arrays of [x, y] positions in original video coordinates
- `bboxes` — (T, 4) face bounding boxes [x, y, w, h]
- `bboxes_detected` — (T,) boolean array indicating frames where face was detected
- `fps` — video frame rate

### Example Usage
```python
import numpy as np

data = np.load("output/landmarks_raw_npz/ADFES_F01-Joy-Face Forward.npz")

# Access raw landmarks (no preprocessing)
landmark_0 = data["landmark_0"]      # shape: (T, 2) — [x, y] in video coords
landmark_105 = data["landmark_105"]

# Face detections
bboxes = data["bboxes"]              # shape: (T, 4) — [x, y, w, h]
detected = data["bboxes_detected"]   # shape: (T,) — True if face found

print(f"Frames with detected face: {np.sum(detected)}/{len(detected)}")
```

## Citation

If you use this pipeline, please cite:
```
Quettier, T. (2026). Facial Expression Landmark Tracking Pipeline. 
ORIS Vector Project.
```

## Contact

For questions or issues, contact: research@tcjq.eu

---

**Last Updated**: April 2026  
**Version**: 1.0
