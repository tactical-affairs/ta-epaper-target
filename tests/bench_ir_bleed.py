"""
Bench Test 2: LCD IR Bleed vs Laser Pulse SNR
==============================================
Quantifies how much IR the LCD backlight emits through the camera's IR-pass
filter, then measures the SNR of a real laser pulse in both raw and temporal
diff frames.

THIS IS THE MAKE-OR-BREAK TEST. If laser SNR < 50 in the diff frame, the
basic approach needs a different filter or display technology.

No IR LEDs required — camera + powered LCD only.

Steps (prompted interactively):
  1. Baseline: LCD off or covered
  2. LCD on, solid white (maximum backlight emission)
  3. LCD on, solid black
  4. LCD on, dark target image
  Then: fire laser at camera field of view when prompted.

Pass criteria:
  - Laser SNR > 50 in diff frame
  - If raw SNR < 5, output recommendation for narrow bandpass filter
"""

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
EXPOSURE = -6
MEASURE_FRAMES = 60   # frames to average for background measurement
LASER_WAIT_FRAMES = 90  # max frames to wait for a laser hit

cap = cv2.VideoCapture(DEVICE_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)

if not cap.isOpened():
    print(f"ERROR: cannot open camera {DEVICE_INDEX}")
    sys.exit(1)


def read_gray() -> np.ndarray:
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise RuntimeError("Camera read failed")


def measure_background(label: str) -> dict:
    input(f"\n[SETUP] {label}\nPress ENTER when ready to measure...")
    frames = []
    for _ in range(MEASURE_FRAMES):
        frames.append(read_gray().astype(np.float32))
    stack = np.stack(frames)
    mean_frame = np.mean(stack, axis=0)
    return {
        "label": label,
        "mean": float(np.mean(mean_frame)),
        "max": float(np.max(mean_frame)),
        "stddev": float(np.std(mean_frame)),
        "frame": mean_frame,
    }


def measure_laser_snr(background: dict) -> dict:
    bg_mean = background["mean"]
    bg_frame = background["frame"]

    print(f"\n[LASER TEST] Background mean = {bg_mean:.2f}")
    print("Fire the laser at the center of the camera's field of view.")
    print("Waiting for a bright transient...")

    prev = read_gray().astype(np.float32)
    best_raw_snr = 0.0
    best_diff_snr = 0.0
    best_raw_peak = 0.0
    best_diff_peak = 0.0

    for i in range(LASER_WAIT_FRAMES):
        frame = read_gray().astype(np.float32)

        raw_peak = float(np.max(frame))
        raw_snr = raw_peak / (bg_mean + 1e-6)

        diff = cv2.absdiff(frame, prev)
        diff_peak = float(np.max(diff))
        diff_snr = diff_peak / (float(np.mean(diff)) + 1e-6)

        if raw_snr > best_raw_snr:
            best_raw_snr = raw_snr
            best_raw_peak = raw_peak
        if diff_snr > best_diff_snr:
            best_diff_snr = diff_snr
            best_diff_peak = diff_peak

        prev = frame

        cv2.imshow("IR view (fire laser now)", frame.astype(np.uint8))
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return {
        "raw_snr": best_raw_snr,
        "raw_peak": best_raw_peak,
        "diff_snr": best_diff_snr,
        "diff_peak": best_diff_peak,
    }


print("=== Bench Test 2: IR Bleed / Laser SNR ===")
print(f"Exposure: {EXPOSURE}, Resolution: {WIDTH}x{HEIGHT}")

conditions = [
    "LCD off (or lens covered) — baseline",
    "LCD on, solid WHITE — maximum backlight",
    "LCD on, solid BLACK — minimum backlight",
    "LCD on, dark target image (open in any viewer)",
]

measurements = []
for cond in conditions:
    m = measure_background(cond)
    measurements.append(m)
    print(f"  {m['label'][:40]:<40}  mean={m['mean']:6.2f}  max={m['max']:6.2f}  std={m['stddev']:5.2f}")

# Use LCD-on target image as the operating background
operating_bg = measurements[3]

laser = measure_laser_snr(operating_bg)

print(f"\n=== SNR Results ===")
print(f"  Background (target image): mean={operating_bg['mean']:.2f}")
print(f"  Raw frame:   peak={laser['raw_peak']:.1f}   SNR={laser['raw_snr']:.1f}")
print(f"  Diff frame:  peak={laser['diff_peak']:.1f}   SNR={laser['diff_snr']:.1f}")

print(f"\n=== Verdict ===")
if laser['diff_snr'] >= 50:
    print(f"  PASS  Diff SNR={laser['diff_snr']:.1f} >= 50. Approach is viable.")
elif laser['diff_snr'] >= 10:
    print(f"  MARGINAL  Diff SNR={laser['diff_snr']:.1f}. May work with tuning.")
    print("  Consider a narrow bandpass filter (e.g., 850nm +/- 20nm).")
else:
    print(f"  FAIL  Diff SNR={laser['diff_snr']:.1f} < 10.")
    print("  Recommendation: switch to narrow bandpass filter or OLED display.")

if laser['raw_snr'] < 5:
    print("  Note: raw frame SNR < 5 — LCD bleed is severe at this exposure.")
    print("  A narrowband filter will help significantly.")

cap.release()
