from __future__ import annotations

import threading
import time
import cv2
import numpy as np

from .config import CameraConfig


class CaptureThread:
    """
    Background camera capture thread with latest-frame semantics.

    Only the most recent frame is kept — callers always get the newest
    available frame without buffering latency. Runs as a daemon thread.
    """

    def __init__(self, config: CameraConfig):
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._frame_number: int = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Open the camera and start the capture thread."""
        self._cap = cv2.VideoCapture(self._config.device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
        # Disable auto-exposure and set manual value
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode on V4L2
        self._cap.set(cv2.CAP_PROP_EXPOSURE, self._config.exposure)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self._config.device_index}")
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> tuple[np.ndarray | None, int]:
        """
        Return the latest captured frame and its frame number.

        Returns (None, 0) if no frame has been captured yet.
        Frame is grayscale uint8.
        """
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self._frame_number

    def set_exposure(self, value: int) -> None:
        """Adjust manual exposure value at runtime."""
        self._config.exposure = value
        if self._cap:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, value)

    def _run(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            with self._lock:
                self._frame = gray
                self._frame_number += 1
