# ePaper Target

A Raspberry Pi-based laser target simulator that uses an LCD display and USB camera with IR detection to recreate the paper shooting experience.

## Overview

The ePaper Target replaces paper targets with a reusable electronic system. A USB camera with an IR-pass filter watches the target face while four corner IR LEDs establish a coordinate system. When a laser shot is detected, the hit is localized on the display in real time, scoring shots and rendering target graphics without ever replacing paper.

## Hardware

| Component | Description | Notes |
|-----------|-------------|-------|
| Raspberry Pi | Host compute platform | RPi 4B or 5 recommended |
| LCD Display | Target face display | Sized to match standard target dimensions |
| USB Camera | IR laser hit detection | Fitted with IR-pass filter to reject visible light |
| IR LEDs (x4) | Corner fiducials | Establish camera-to-display coordinate mapping |
| IR-pass filter | Camera lens filter | Isolates IR laser dot from LCD background |

## Software Architecture

```
epaper_target/
├── config.py        # Tunable parameters (camera, detection, calibration, display)
├── capture.py       # Background camera thread — latest-frame semantics
├── calibration.py   # Corner LED detection, subpixel refinement, homography
├── detection.py     # Temporal frame differencing hit detector
├── display.py       # Pygame fullscreen display and hit rendering
├── geometry.py      # Scoring rings and shot group statistics
└── util/fps.py      # Sliding-window FPS counter
```

- **Capture** — daemon thread keeps the latest grayscale frame available; manual V4L2 exposure control
- **Calibration** — thresholds for 4 corner IR LED blobs, refines centroids with `cornerSubPix`, orders TL/TR/BL/BR, computes perspective homography; re-detects every frame to handle drift
- **Hit Detection** — `absdiff` between consecutive frames cancels constant IR sources; blob filtering by area and circularity localizes the laser dot; hit mapped to display coordinates via homography
- **Display** — pygame fullscreen surface renders target graphics and cumulative hit markers with scores
- **Scoring** — hits scored against concentric `ScoringRing` definitions; shot group stats (spread, extreme spread) computed per session

## Setup

### Requirements

- Raspberry Pi OS Lite 64-bit (RPi 4B or 5)
- Python 3.9+

### Dependencies

```bash
pip install opencv-python numpy pygame pyyaml
```

> On Raspberry Pi, `python3-opencv` from apt is more reliable than the pip package:
> ```bash
> sudo apt install python3-opencv
> pip install numpy pygame pyyaml
> ```

### Run

```bash
python main.py
# or with a custom config:
python main.py config.yaml
```

## Hardware Bench Tests

Before running the full system, validate each subsystem in order:

```bash
python tests/bench_capture_rate.py    # Camera FPS + exposure control
python tests/bench_ir_bleed.py        # LCD IR bleed vs laser pulse SNR (critical)
python tests/bench_corner_detect.py   # Corner LED detection + jitter
python tests/bench_homography.py      # Homography accuracy + stability
python tests/bench_temporal_diff.py   # End-to-end hit detection
```

Each script is standalone — no install required. Run `bench_ir_bleed.py` first; it determines whether the IR-pass filter and display combination has sufficient SNR for reliable hit detection.
