"""
Default target display — shows the USPSA target fullscreen on boot.
Launched automatically via labwc autostart.
"""
import sys
import os
import numpy as np
import cv2
import pygame

ASSET = os.path.join(os.path.dirname(__file__), "assets", "uspsa_target.webp")

pygame.init()
info = pygame.display.Info()
W, H = info.current_w, info.current_h
screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
screen.fill((0, 0, 0))

# Load image, flood-fill background from all four corners then replace with black
raw = cv2.imread(ASSET)
ih, iw = raw.shape[:2]
mask = np.zeros((ih + 2, iw + 2), dtype=np.uint8)
for corner in [(0, 0), (iw - 1, 0), (0, ih - 1), (iw - 1, ih - 1)]:
    cv2.floodFill(raw, mask, corner, (0, 0, 0), loDiff=(30,)*3, upDiff=(30,)*3)

# Scale to fit screen
ih, iw = raw.shape[:2]
scale = min(W / iw, H / ih)
nw, nh = int(iw * scale), int(ih * scale)
raw = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_AREA)

# BGR → RGB for pygame
rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
img = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
screen.blit(img, ((W - nw) // 2, (H - nh) // 2))
pygame.display.flip()

clock = pygame.time.Clock()
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
    clock.tick(30)
