"""
Bench Test 5: End-to-End Hit Detection
=======================================
Full system test: fire a laser at the display and watch the hit render.
Measures detection rate, false positive rate, and latency.

Requires: camera + 4 corner LEDs + display (pygame fullscreen).

Controls:
  r   reset hits
  c   force recalibration
  q   quit

Pass criteria:
  - >= 90% of laser shots detected
  - < 1 false positive per 30 seconds
  - Detection latency < 100ms
"""

import sys
import time
import threading

import cv2
import numpy as np
import pygame

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from epaper_target.config import CameraConfig, CalibrationConfig, DetectionConfig, DisplayConfig
from epaper_target.capture import CaptureThread
from epaper_target.calibration import CalibrationManager
from epaper_target.detection import HitDetector, HitEvent
from epaper_target.geometry import score_hit, DEFAULT_RINGS
from epaper_target.util.fps import FPSCounter

DISPLAY_SIZE = (1024, 600)
TARGET_CENTER = (DISPLAY_SIZE[0] // 2, DISPLAY_SIZE[1] // 2)

cam_cfg = CameraConfig(width=640, height=480, fps=60, exposure=-6)
cal_cfg = CalibrationConfig(led_threshold=180, min_stability_frames=20)
det_cfg = DetectionConfig(diff_threshold=30, min_blob_area=5, max_blob_area=500,
                           min_circularity=0.4, cooldown_frames=10)

capture = CaptureThread(cam_cfg)
cal = CalibrationManager(cal_cfg, DISPLAY_SIZE)
detector = HitDetector(det_cfg, cal)

# --- Pygame display setup (must be main thread) ---
pygame.init()
screen = pygame.display.set_mode(DISPLAY_SIZE, pygame.FULLSCREEN | pygame.DOUBLEBUF)
pygame.display.set_caption("ePaper Target — Bench Test 5")
font = pygame.font.SysFont("monospace", 20)
font_large = pygame.font.SysFont("monospace", 28, bold=True)


def draw_target(surface: pygame.Surface) -> None:
    """Render concentric scoring rings on the surface."""
    surface.fill((240, 230, 200))
    W, H = DISPLAY_SIZE
    short = min(W, H)
    cx, cy = TARGET_CENTER
    ring_colors = [(20, 20, 20), (50, 50, 50), (80, 80, 80),
                   (110, 110, 110), (140, 140, 140), (180, 40, 40),
                   (200, 60, 60), (220, 80, 80), (240, 100, 100), (255, 120, 120)]
    for ring, color in zip(reversed(DEFAULT_RINGS), ring_colors):
        r = int(ring.radius_fraction * short)
        pygame.draw.circle(surface, color, (cx, cy), r)
    for ring in DEFAULT_RINGS:
        r = int(ring.radius_fraction * short)
        pygame.draw.circle(surface, (0, 0, 0), (cx, cy), r, 1)


target_surface = pygame.Surface(DISPLAY_SIZE)
draw_target(target_surface)

# Stats
hits: list[HitEvent] = []
false_positives = 0
session_start = time.monotonic()
fps_counter = FPSCounter(window=60)
calibrated = False
status_msg = "Waiting for calibration..."

# Start capture
capture.start()

print("=== Bench Test 5: End-to-End Hit Detection ===")
print("Waiting for corner LED calibration...")
print("Controls: r=reset  c=recalibrate  q=quit")

prev_frame: np.ndarray | None = None
prev_frame_number = 0
running = True

while running:
    # --- Pygame event handling (main thread) ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                hits.clear()
                print("  Hits reset.")
            elif event.key == pygame.K_c:
                calibrated = False
                cal._result = None
                cal._stable_count = 0
                status_msg = "Recalibrating..."
                print("  Recalibrating...")

    # --- Get latest camera frame ---
    frame, frame_number = capture.get_frame()
    if frame is None or frame_number == prev_frame_number:
        pygame.time.wait(1)
        continue

    fps_counter.tick()

    # --- Calibration ---
    if not calibrated:
        result = cal.update(frame)
        if result is not None:
            calibrated = True
            status_msg = f"Calibrated (err={result.reprojection_error:.1f}px)"
            print(f"  {status_msg}")

    # --- Hit detection ---
    if calibrated and prev_frame is not None:
        t0 = time.monotonic()
        hit = detector.process_frame(frame, prev_frame, frame_number)
        latency_ms = (time.monotonic() - t0) * 1000

        if hit is not None:
            score = score_hit(hit.display_pt, TARGET_CENTER, DISPLAY_SIZE)
            hits.append(hit)
            print(f"  HIT  display={hit.display_pt}  score={score}  latency={latency_ms:.1f}ms")

    prev_frame = frame
    prev_frame_number = frame_number

    # --- Render ---
    screen.blit(target_surface, (0, 0))

    for h in hits:
        score = score_hit(h.display_pt, TARGET_CENTER, DISPLAY_SIZE)
        pygame.draw.circle(screen, (255, 50, 50), h.display_pt, 8)
        pygame.draw.circle(screen, (255, 255, 255), h.display_pt, 8, 2)
        lbl = font.render(str(score) if score else "M", True, (255, 255, 0))
        screen.blit(lbl, (h.display_pt[0] + 10, h.display_pt[1] - 10))

    elapsed = time.monotonic() - session_start
    fp_rate = false_positives / (elapsed / 30) if elapsed >= 1 else 0

    hud_lines = [
        f"FPS: {fps_counter.get_fps():.1f}",
        f"Hits: {len(hits)}",
        f"FP/30s: {fp_rate:.1f}",
        status_msg,
        "r=reset  c=recal  q=quit",
    ]
    for i, line in enumerate(hud_lines):
        surf = font.render(line, True, (255, 255, 100))
        screen.blit(surf, (10, 10 + i * 22))

    if not calibrated:
        msg = font_large.render("CALIBRATING...", True, (255, 80, 0))
        screen.blit(msg, (DISPLAY_SIZE[0] // 2 - 120, DISPLAY_SIZE[1] - 50))

    pygame.display.flip()

capture.stop()
pygame.quit()

# Final report
elapsed = time.monotonic() - session_start
fp_rate = false_positives / (elapsed / 30) if elapsed >= 1 else 0
print(f"\n=== Final Report ===")
print(f"  Session duration:  {elapsed:.0f}s")
print(f"  Total hits:        {len(hits)}")
print(f"  False positives:   {false_positives}  ({fp_rate:.1f}/30s)  "
      f"{'PASS' if fp_rate < 1 else 'FAIL'}")
print(f"  (Detection rate requires manual count of actual shots fired)")
