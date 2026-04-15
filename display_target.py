"""
Default target display — shows the USPSA target fullscreen on boot.
Launched automatically via labwc autostart.
"""
import sys
import os
import pygame

ASSET = os.path.join(os.path.dirname(__file__), "assets", "uspsa_target.webp")

pygame.init()
info = pygame.display.Info()
W, H = info.current_w, info.current_h
screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
screen.fill((0, 0, 0))

img = pygame.image.load(ASSET).convert_alpha()
iw, ih = img.get_size()
scale = min(W / iw, H / ih)
nw, nh = int(iw * scale), int(ih * scale)
img = pygame.transform.smoothscale(img, (nw, nh))
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
