"""
Bench Test 4: Homography Accuracy + Stability
==============================================
Verifies the perspective transform correctly maps camera coordinates to
display coordinates, and that the mapping remains stable over time.

Requires: camera + 4 corner IR LEDs + display connected.

Procedure:
  1. Corner LEDs are detected and homography is computed.
  2. A 3x3 grid of dots is rendered on the display.
  3. User places a laser pointer at each dot and presses SPACE to record
     where the camera sees it. This measures reprojection error.
  4. Stability is monitored for 2 minutes.

Pass criteria:
  - Reprojection error < 3px
  - Corner position drift < 1px over 2 minutes
"""

import sys
import time
import statistics

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from epaper_target.config import CalibrationConfig
from epaper_target.calibration import CalibrationManager

DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
EXPOSURE = -6
DISPLAY_SIZE = (1024, 600)

cal_cfg = CalibrationConfig(led_threshold=180, min_stability_frames=20)
cal = CalibrationManager(config=cal_cfg, display_size=DISPLAY_SIZE)

cap = cv2.VideoCapture(DEVICE_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

if not cap.isOpened():
    print(f"ERROR: cannot open camera {DEVICE_INDEX}")
    sys.exit(1)

# Build 3x3 grid of test points in display space
W, H = DISPLAY_SIZE
margin_x, margin_y = W // 5, H // 5
grid_points_display = []
for row in range(3):
    for col in range(3):
        x = margin_x + col * (W - 2 * margin_x) // 2
        y = margin_y + row * (H - 2 * margin_y) // 2
        grid_points_display.append((x, y))


def read_gray() -> np.ndarray:
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise RuntimeError("Camera read failed")


# Phase 1: Wait for stable calibration
print("=== Bench Test 4: Homography Accuracy ===")
print("Waiting for stable corner detection (hold still)...")

result = None
while result is None:
    gray = read_gray()
    result = cal.update(gray)
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(preview, "Waiting for calibration...", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.imshow("Bench: Homography", preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

print(f"Calibration acquired. Reprojection error: {result.reprojection_error:.2f}px")

# Phase 2: Grid reprojection test
print(f"\nGrid test: {len(grid_points_display)} points")
print("For each dot shown on the display, aim a laser pointer at it and press SPACE.")
print("(If no physical laser available, press SPACE to skip — reprojection error won't be measured.)")

errors = []
for i, (dx, dy) in enumerate(grid_points_display):
    print(f"\n  Point {i+1}/{len(grid_points_display)}: display ({dx}, {dy})")
    print("  Aim laser at the dot and press SPACE (or ENTER to skip)...")

    while True:
        gray = read_gray()
        preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw all grid points, highlight current
        for j, (gx, gy) in enumerate(grid_points_display):
            # Convert display coords to camera coords (approximate — for visual only)
            color = (0, 255, 255) if j == i else (100, 100, 100)
            # Display these in camera window as rough reference
            cv2.circle(preview, (j * 60 + 30, 30), 5, color, -1)

        cv2.putText(preview, f"Point {i+1}: aim at dot {i+1} on display", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Bench: Homography", preview)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            # Find brightest blob in frame (assume laser pointer is visible)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    mapped = cal.camera_to_display((cx, cy), result.H)
                    err = np.linalg.norm(np.array(mapped) - np.array((dx, dy)))
                    errors.append(err)
                    print(f"    Camera: ({cx:.1f}, {cy:.1f})  -> Display: {mapped}  "
                          f"Expected: ({dx},{dy})  Error: {err:.1f}px")
            break
        elif key in (ord('\r'), ord('\n'), ord('q')):
            print("    Skipped.")
            break

# Phase 3: Stability monitoring
print("\n\nMonitoring homography stability for 2 minutes...")
print("Keep the camera and display stationary. Press q to stop early.")

start = time.monotonic()
drift_samples: list[float] = []
baseline_corners = cal.detect_corners(read_gray()).corners.copy()

while time.monotonic() - start < 120:
    gray = read_gray()
    detection = cal.detect_corners(gray)
    if detection.stable:
        drift = float(np.max(np.linalg.norm(detection.corners - baseline_corners, axis=1)))
        drift_samples.append(drift)

    elapsed = time.monotonic() - start
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(preview, f"Stability: {elapsed:.0f}s / 120s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if drift_samples:
        cv2.putText(preview, f"Max drift: {max(drift_samples):.2f}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Bench: Homography", preview)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n=== Final Report ===")
if errors:
    mean_err = statistics.mean(errors)
    print(f"  Reprojection error: mean={mean_err:.2f}px  max={max(errors):.2f}px  "
          f"({'PASS' if mean_err < 3.0 else 'FAIL'}, threshold 3px)")
else:
    print("  Reprojection error: not measured (no laser pointer)")

if drift_samples:
    max_drift = max(drift_samples)
    print(f"  Corner drift:       max={max_drift:.2f}px  "
          f"({'PASS' if max_drift < 1.0 else 'FAIL'}, threshold 1px)")

print(f"  Initial calibration reprojection error: {result.reprojection_error:.2f}px")
