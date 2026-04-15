from __future__ import annotations

import numpy as np
import pygame


class DisplayManager:
    """
    Fullscreen pygame display for target rendering and hit visualization.

    Must be instantiated and updated on the main thread (pygame requirement).
    """

    def __init__(self, display_size: tuple[int, int], fullscreen: bool = True):
        self._size = display_size  # (width, height)
        pygame.init()
        flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF if fullscreen else 0
        self._screen = pygame.display.set_mode(display_size, flags)
        pygame.display.set_caption("ePaper Target")
        self._target_surface: pygame.Surface | None = None
        self._hits: list[tuple[tuple[int, int], int]] = []  # [(display_pt, score)]
        self._font = pygame.font.SysFont("monospace", 20)

    def set_target(self, image: np.ndarray) -> None:
        """
        Set the target background from an OpenCV BGR image (numpy array).
        Resized to fill the display.
        """
        rgb = image[:, :, ::-1]  # BGR → RGB
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self._target_surface = pygame.transform.scale(surface, self._size)
        self._hits.clear()

    def add_hit(self, pt: tuple[int, int], score: int) -> None:
        """Record a hit at display coordinates with its score."""
        self._hits.append((pt, score))

    def update(self) -> None:
        """Render current state to screen. Call once per main loop iteration."""
        if self._target_surface:
            self._screen.blit(self._target_surface, (0, 0))
        else:
            self._screen.fill((30, 30, 30))

        for pt, score in self._hits:
            pygame.draw.circle(self._screen, (255, 50, 50), pt, 8)
            pygame.draw.circle(self._screen, (255, 255, 255), pt, 8, 2)
            label = self._font.render(str(score), True, (255, 255, 0))
            self._screen.blit(label, (pt[0] + 10, pt[1] - 10))

        pygame.display.flip()

    def show_solid_color(self, color: tuple[int, int, int]) -> None:
        """Fill the display with a solid RGB color and flip immediately. Used for IR bleed testing."""
        self._screen.fill(color)
        pygame.display.flip()

    def clear_hits(self) -> None:
        """Remove all recorded hits."""
        self._hits.clear()

    def pump_events(self) -> bool:
        """
        Process pygame event queue. Returns False if a quit event is received.
        Must be called regularly on the main thread.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def quit(self) -> None:
        """Shut down pygame."""
        pygame.quit()
