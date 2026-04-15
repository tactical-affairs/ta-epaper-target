"""
Bench Test 3: Corner IR LED Detection + Ordering
=================================================
Verifies the four corner IR LEDs are detectable, correctly ordered as
TL/TR/BL/BR, and that subpixel centroid estimates are stable.

Requires: camera with IR-pass filter + 4 corner IR LEDs powered.

Controls:
  q   quit

Pass criteria:
  - >= 95% of frames detect all 4 blobs
  - Centroid position jitter < 0.3px stddev over 100-frame window
  - Ordering algorithm succeeds consistently
"""

import sys
import time
import statistics
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from epaper_target.config import CameraConfig, CalibrationConfig
from epaper_target.calibration import CalibrationManager

DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
EXPOSURE = -6
DISPLAY_SIZE = (1024, 600)

CORNER_LABELS = ["TL", "TR", "BL", "BR"]
CORNER_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (0, 0, 255)]

cal_cfg = CalibrationConfig(led_threshold=180, min_stability_frames=1)
cal = CalibrationManager(config=cal_cfg, display_size=DISPLAY_SIZE)

cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

if not cap.isOpened():
    print(f"ERROR: cannot open camera {DEVICE_INDEX}")
    sys.exit(1)

# Rolling statistics
total_frames = 0
detect_frames = 0
corner_history: list[deque] = [deque(maxlen=100) for _ in range(4)]

print("=== Bench Test 3: Corner LED Detection ===")
print(f"LED threshold: {cal_cfg.led_threshold}")
print("Ensure all 4 corner LEDs are powered and in the camera's field of view.")
print("Press q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    total_frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detection = cal.detect_corners(gray)

    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if detection.stable:
        detect_frames += 1
        for i, (cx, cy) in enumerate(detection.corners):
            corner_history[i].append((cx, cy))
            color = CORNER_COLORS[i]
            cv2.circle(display, (int(cx), int(cy)), 8, color, -1)
            cv2.putText(display, CORNER_LABELS[i], (int(cx) + 10, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        blob_count = len(detection.centroids)
        cv2.putText(display, f"Only {blob_count}/4 blobs found", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    detect_rate = detect_frames / total_frames * 100 if total_frames else 0
    cv2.putText(display, f"Detect rate: {detect_rate:.1f}%", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detect_rate >= 95 else (0, 165, 255), 2)

    # Compute per-corner jitter every 100 frames
    if total_frames % 100 == 0 and all(len(h) >= 10 for h in corner_history):
        print(f"\nFrame {total_frames}  detect_rate={detect_rate:.1f}%")
        all_pass = True
        for i, hist in enumerate(corner_history):
            xs = [p[0] for p in hist]
            ys = [p[1] for p in hist]
            jx = statistics.stdev(xs) if len(xs) > 1 else 0
            jy = statistics.stdev(ys) if len(ys) > 1 else 0
            jitter = (jx ** 2 + jy ** 2) ** 0.5
            ok = jitter < 0.3
            if not ok:
                all_pass = False
            print(f"  {CORNER_LABELS[i]}  jitter={jitter:.3f}px  {'OK' if ok else 'HIGH'}")
        print(f"  Detect rate: {detect_rate:.1f}%  {'OK' if detect_rate >= 95 else 'LOW'}")
        if detect_rate >= 95 and all_pass:
            print("  -> PASS")

    cv2.imshow("Bench: Corner Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n=== Final Report ===")
detect_rate = detect_frames / total_frames * 100 if total_frames else 0
print(f"  Frames: {total_frames}   Detected: {detect_frames}   Rate: {detect_rate:.1f}%")
if all(len(h) >= 2 for h in corner_history):
    for i, hist in enumerate(corner_history):
        xs = [p[0] for p in hist]
        ys = [p[1] for p in hist]
        jitter = (statistics.stdev(xs) ** 2 + statistics.stdev(ys) ** 2) ** 0.5
        print(f"  {CORNER_LABELS[i]}  jitter={jitter:.3f}px")
pass_rate = detect_rate >= 95
print(f"  RESULT: {'PASS' if pass_rate else 'FAIL'}")
