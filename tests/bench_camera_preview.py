"""
Camera Preview — shows live camera feed fullscreen on the display.

Useful for verifying camera position, focus, and IR filter effect.

Controls:
  +/=   increase exposure
  -     decrease exposure
  g     toggle grayscale / color
  q     quit
"""
import sys
import subprocess

import cv2
import numpy as np
import pygame

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from epaper_target.config import CameraConfig

DEVICE_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 120
EXPOSURE = 50   # 100μs units

# --- Camera setup ---
subprocess.run(
    ["v4l2-ctl", "-d", f"/dev/video{DEVICE_INDEX}",
     "--set-ctrl=auto_exposure=1",
     f"--set-ctrl=exposure_time_absolute={EXPOSURE}",
     "--set-ctrl=exposure_dynamic_framerate=0"],
    check=False, capture_output=True,
)

cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

if not cap.isOpened():
    print(f"ERROR: cannot open camera {DEVICE_INDEX}")
    sys.exit(1)

# --- Pygame display setup ---
pygame.init()
info = pygame.display.Info()
W, H = info.current_w, info.current_h
screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
font = pygame.font.SysFont("monospace", 20, bold=True)

exposure = EXPOSURE
grayscale = False

from epaper_target.util.fps import FPSCounter
fps_counter = FPSCounter(window=60)

clock = pygame.time.Clock()

while True:
    # --- Events ---
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            cap.release(); pygame.quit(); sys.exit()
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_q or e.key == pygame.K_ESCAPE:
                cap.release(); pygame.quit(); sys.exit()
            elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                exposure = min(exposure + 10, 5000)
                subprocess.run(["v4l2-ctl", "-d", f"/dev/video{DEVICE_INDEX}",
                                 f"--set-ctrl=exposure_time_absolute={exposure}"],
                                check=False, capture_output=True)
            elif e.key == pygame.K_MINUS:
                exposure = max(exposure - 10, 1)
                subprocess.run(["v4l2-ctl", "-d", f"/dev/video{DEVICE_INDEX}",
                                 f"--set-ctrl=exposure_time_absolute={exposure}"],
                                check=False, capture_output=True)
            elif e.key == pygame.K_g:
                grayscale = not grayscale

    # --- Capture ---
    ret, frame = cap.read()
    if not ret:
        continue

    fps_counter.tick()

    if grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # OpenCV BGR → pygame RGB surface
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Scale to fit screen
    scale = min(W / WIDTH, H / HEIGHT)
    nw, nh = int(WIDTH * scale), int(HEIGHT * scale)
    frame_resized = cv2.resize(frame_rgb, (nw, nh))
    surface = pygame.surfarray.make_surface(np.transpose(frame_resized, (1, 0, 2)))

    screen.fill((0, 0, 0))
    screen.blit(surface, ((W - nw) // 2, (H - nh) // 2))

    # HUD
    fps = fps_counter.get_fps()
    hud = f"FPS:{fps:.0f}  Exp:{exposure}  {'GRAY' if grayscale else 'COLOR'}  +/-=exposure  g=gray  q=quit"
    s = font.render(hud, True, (0, 255, 0))
    screen.blit(s, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)
