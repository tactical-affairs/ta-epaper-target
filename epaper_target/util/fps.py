import time
from collections import deque


class FPSCounter:
    """Sliding-window FPS counter."""

    def __init__(self, window: int = 30):
        self._window = window
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Record a frame timestamp."""
        self._times.append(time.monotonic())

    def get_fps(self) -> float:
        """Return current FPS averaged over the last window frames."""
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._times) - 1) / elapsed
