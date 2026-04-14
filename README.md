# Facial Expression Optical Flow Pipeline

A comprehensive Python pipeline for analyzing facial expressions in video using optical flow. Designed for datasets like ADFES and JeFEE.

## Overview

This pipeline processes facial expression videos to extract optical flow vectors during the peak of emotional expression. It standardizes face position, detects the expression apex, segments video around the peak, and applies dense optical flow with optional face masking.

**Key Features:**
- ✅ Batch processing of multiple video datasets
- ✅ Automatic face detection and standardization (rotation + square crop)
- ✅ Affine registration to remove rigid head motion
- ✅ Apex detection using optical flow magnitude
- ✅ Asymmetric segmentation (configurable before/after apex)
- ✅ Dense Farneback optical flow with optional circular face mask
- ✅ Multi-format output (NPZ data + animated GIF visualization)

## Quick Start

### 1. Clone & Setup

```bash
# Navigate to project directory
cd path/to/04.COMPUTATIONAL

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

```bash
python scripts/face_of_pipeline.py
```

### 4. Check Output

```
output/
├── optical_flow_npz/
│   ├── ADFES_F01-Joy-Face Forward.npz
│   ├── ADFES_F03-Joy-Face Forward.npz
│   └── ...
└── flow_visualization_gif/
    ├── ADFES_F01-Joy-Face Forward.gif
    ├── ADFES_F03-Joy-Face Forward.gif
    └── ...
```

## Configuration

Edit `scripts/face_of_pipeline.py` top section:

```python
# Face normalization
CROP_SIZE       = 256        # output face square (pixels)
PAD_FACTOR      = 0.35       # padding around face (0.0-1.0)

# Apex segmentation
APEX_WINDOW_BEFORE = 40      # frames before apex
APEX_WINDOW_AFTER  = 5       # frames after apex

# Face masking for OF
USE_FACE_MASK   = True       # enable/disable
FACE_MASK_SIZE  = 0.85       # mask radius (0.0-1.0)

# Visualization
ARROW_STEP  = 16             # arrow grid spacing (pixels)
ARROW_SCALE = 5.0            # arrow size multiplier
GIF_SPEED   = 0.5            # GIF playback speed
```

## Loading Results in Python

```python
import numpy as np

# Load optical flow data
data = np.load("output/optical_flow_npz/ADFES_F01-Joy-Face Forward.npz")

# Access arrays
flow = data["flow"]              # shape: (T-1, 256, 256, 2) — [dx, dy]
frames = data["frames"]          # shape: (T, 256, 256, 3) — BGR
apex_local = int(data["apex_local"])    # frame index within segment
apex_global = int(data["apex_global"])  # frame index in full video
fps = float(data["fps"])
motion_all = data["motion_all"]  # motion magnitude per frame

# Extract flow components
u = flow[..., 0]  # horizontal component
v = flow[..., 1]  # vertical component
magnitude = np.sqrt(u**2 + v**2)
angle = np.arctan2(v, u)

print(f"Total frames: {len(frames)}")
print(f"Apex at frame {apex_local} (global {apex_global})")
print(f"Max flow magnitude: {magnitude.max():.2f} pixels/frame")
```

## Output File Format

### NPZ (Optical Flow Data)

Binary compressed NumPy archive containing:

| Variable | Shape | Type | Description |
|----------|-------|------|-------------|
| `flow` | (T-1, H, W, 2) | float32 | Dense optical flow, dx/dy components |
| `frames` | (T, H, W, 3) | uint8 | Standardized face frames (BGR) |
| `apex_local` | () | int32 | Apex frame index within segment |
| `apex_global` | () | int32 | Apex frame index in full video |
| `seg_start` | () | int32 | Segment start frame (full video) |
| `seg_end` | () | int32 | Segment end frame (full video) |
| `fps` | () | float32 | Video FPS |
| `motion_all` | (N,) | float32 | Motion magnitude per frame (full video) |
| `face_mask_used` | () | bool | Whether mask was applied |
| `face_mask_size` | () | float32 | Mask radius fraction |

### GIF (Visualization)

Animated GIF showing per-frame optical flow with:
- **Colour wheel**: direction (hue) + magnitude (brightness)
- **White arrows**: sparse flow vectors
- **Orange circle**: face mask boundary
- **Frame counter**: segment frame index
- **"APEX" label**: marks apex frames
- **"MASK" label**: indicates mask region

## Methodology

See `METHODOLOGY.html` for detailed step-by-step explanation of:
1. Video loading & preprocessing
2. Face detection with Haar cascades
3. Face standardization (alignment + cropping)
4. Bbox smoothing & affine registration
5. Apex detection (motion-based)
6. Asymmetric segmentation
7. Dense optical flow (Farneback algorithm)
8. Visualization & output

## System Requirements

- **Python**: 3.10+
- **OS**: macOS, Linux, Windows
- **RAM**: 2-4 GB (depends on video resolution & duration)
- **Disk**: ~1 GB per 100 videos

## Dependencies

| Package | Purpose |
|---------|---------|
| opencv-python | Face detection, optical flow, image processing |
| numpy | Array operations |
| scipy | Temporal smoothing (median filter) |
| imageio[ffmpeg] | GIF export |
| pillow | Image I/O |

## Troubleshooting

**Issue**: "No face detected in any frame"
- **Solution**: Check video resolution (min 256×256), lighting, face visibility

**Issue**: Arrows on shoulders/hair instead of face
- **Solution**: Reduce `PAD_FACTOR` (try 0.05-0.15)

**Issue**: Apex frame seems incorrect
- **Solution**: Check motion curve: `plt.plot(data["motion_all"])`; apex is `argmax`

**Issue**: GIF file is very large
- **Solution**: Reduce `GIF_SPEED` or segment size; use compression post-processing

## Paper References

**Optical Flow:**
- Farneback, G. (2003). "Two-Frame Motion Estimation Based on Polynomial Expansion"

**Face Detection:**
- Viola, P., & Jones, M. (2001). "Rapid Object Detection using a Boosted Cascade of Simple Features"

**Facial Expression Analysis:**
- Ekman, P., & Friesen, W. V. (1978). "Manual for the Facial Action Coding System"

## License

This project is provided for research purposes.

## Citation

If you use this pipeline, please cite:
```
Quettier, T. (2026). Facial Expression Optical Flow Pipeline. 
ORIS Vector Project.
```

## Contact

For questions or issues, contact: thomas.quettier@example.com

---

**Last Updated**: April 2026  
**Version**: 1.0
