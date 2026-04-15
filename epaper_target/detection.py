from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectionConfig
from .calibration import CalibrationManager


@dataclass
class HitEvent:
    """A confirmed laser hit."""
    camera_pt: tuple[float, float]
    display_pt: tuple[int, int]
    frame_number: int
    timestamp: float


class HitDetector:
    """
    Detects laser hits using temporal frame differencing.

    Subtracts consecutive frames so that constant IR sources (corner LEDs,
    ambient IR) cancel out, leaving only transient events like a laser pulse.
    """

    def __init__(self, config: DetectionConfig, calibration: CalibrationManager):
        self._cfg = config
        self._cal = calibration
        self._cooldown = 0
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process_frame(self, frame: np.ndarray, prev_frame: np.ndarray,
                      frame_number: int) -> HitEvent | None:
        """
        Compare frame against prev_frame for a laser hit.

        Pipeline:
          absdiff → threshold → morphological open (denoise) →
          findContours → area + circularity filter → centroid →
          cooldown check → camera_to_display

        Returns a HitEvent if a hit is detected, None otherwise.
        Requires a valid calibration to map to display coordinates.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        result = self._cal.current_result
        if result is None:
            return None

        diff = cv2.absdiff(frame, prev_frame)
        _, thresh = cv2.threshold(diff, self._cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if not (self._cfg.min_blob_area <= area <= self._cfg.max_blob_area):
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self._cfg.min_circularity:
                continue

            if area > best_area:
                best_area = area
                best = c

        if best is None:
            return None

        M = cv2.moments(best)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        display_pt = self._cal.camera_to_display((cx, cy), result.H)

        self._cooldown = self._cfg.cooldown_frames
        return HitEvent(
            camera_pt=(cx, cy),
            display_pt=display_pt,
            frame_number=frame_number,
            timestamp=time.monotonic(),
        )
