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

- **Hit Detection** — OpenCV pipeline captures frames, thresholds for the IR laser dot, and localizes hit coordinates
- **Coordinate Calibration** — 4-point homography from corner IR LEDs maps camera pixel space to display coordinates
- **Display Rendering** — Target graphics and hit markers rendered on the LCD in real time
- **Scoring** — Hit coordinates scored against target ring geometry

## Setup

> Setup instructions will be added as the project develops.

### Dependencies

```bash
pip install opencv-python numpy pillow
```

### Run

```bash
python main.py
```
