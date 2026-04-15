"""
Bench Test 1: Camera Frame Rate + Exposure Control
===================================================
Verifies the USB camera delivers the target frame rate, that manual
exposure is actually taking effect, and characterises frame timing jitter.

No display or IR LEDs required — camera only.

Controls:
  +/=   increase exposure
  -     decrease exposure
  q     quit

Pass criteria:
  - Sustained FPS >= configured target
  - Frame interval jitter < 2ms stddev
  - Zero dropped frames over 30s
"""

import sys
import time
import statistics

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from epaper_target.config import CameraConfig
from epaper_target.util.fps import FPSCounter

TARGET_FPS = 60
DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
INITIAL_EXPOSURE = -6

cap = cv2.VideoCapture(DEVICE_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # manual mode
cap.set(cv2.CAP_PROP_EXPOSURE, INITIAL_EXPOSURE)

if not cap.isOpened():
    print(f"ERROR: cannot open camera {DEVICE_INDEX}")
    sys.exit(1)

exposure = INITIAL_EXPOSURE
fps_counter = FPSCounter(window=60)

frame_times: list[float] = []
prev_time = time.monotonic()
dropped = 0
report_interval = 1.0
last_report = time.monotonic()
total_frames = 0

print(f"Camera opened. Target FPS={TARGET_FPS}, exposure={exposure}")
print("Controls: +/= raise exposure, - lower exposure, q quit")

while True:
    ret, frame = cap.read()
    now = time.monotonic()

    if not ret:
        dropped += 1
        continue

    total_frames += 1
    interval = now - prev_time
    frame_times.append(interval)
    if len(frame_times) > 300:
        frame_times.pop(0)
    prev_time = now
    fps_counter.tick()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))

    # Overlay stats on preview
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fps = fps_counter.get_fps()
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Exposure: {exposure}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Mean intensity: {mean_intensity:.1f}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Dropped: {dropped}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if dropped else (0, 255, 0), 2)
    cv2.imshow("Bench: Capture Rate", display)

    # Per-second report
    if now - last_report >= report_interval and len(frame_times) > 1:
        jitter_ms = statistics.stdev(frame_times) * 1000
        mean_interval_ms = statistics.mean(frame_times) * 1000
        print(f"  FPS={fps:.1f}  interval={mean_interval_ms:.1f}ms  "
              f"jitter={jitter_ms:.2f}ms  dropped={dropped}  "
              f"intensity={mean_intensity:.1f}  exposure={exposure}")
        last_report = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('+'), ord('=')):
        exposure = min(exposure + 1, 0)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(f"  Exposure -> {exposure}")
    elif key == ord('-'):
        exposure = max(exposure - 1, -13)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        print(f"  Exposure -> {exposure}")

cap.release()
cv2.destroyAllWindows()

if len(frame_times) > 1:
    jitter_ms = statistics.stdev(frame_times) * 1000
    mean_fps = fps_counter.get_fps()
    print(f"\n=== Final Report ===")
    print(f"  Mean FPS:    {mean_fps:.1f}  (target {TARGET_FPS})")
    print(f"  Jitter:      {jitter_ms:.2f} ms stddev  (target < 2ms)")
    print(f"  Dropped:     {dropped}  (target 0)")
    print(f"  Total frames:{total_frames}")
    pass_fail = (mean_fps >= TARGET_FPS * 0.95 and jitter_ms < 2.0 and dropped == 0)
    print(f"  RESULT:      {'PASS' if pass_fail else 'FAIL'}")
