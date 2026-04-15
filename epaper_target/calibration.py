from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CalibrationConfig


@dataclass
class CornerDetection:
    """Result of a single-frame corner LED detection attempt."""
    corners: np.ndarray        # shape (4, 2), float32, ordered TL/TR/BL/BR
    centroids: list[tuple]     # raw detected blob centroids before ordering
    stable: bool               # True if detection was clean and unambiguous


@dataclass
class CalibrationResult:
    """A computed homography mapping camera space → display space."""
    H: np.ndarray              # 3x3 perspective transform matrix
    reprojection_error: float  # mean reprojection error in display pixels
    timestamp: float           # time.monotonic() when computed


class CalibrationManager:
    """
    Detects the four corner IR LEDs and computes the camera-to-display homography.

    Call update() each frame. It returns a CalibrationResult once corners have
    been stable for min_stability_frames consecutive frames.
    """

    def __init__(self, config: CalibrationConfig, display_size: tuple[int, int]):
        self._cfg = config
        self._display_size = display_size  # (width, height)
        self._stable_count = 0
        self._prev_corners: np.ndarray | None = None
        self._result: CalibrationResult | None = None

    def detect_corners(self, frame: np.ndarray) -> CornerDetection:
        """
        Detect four corner IR LED blobs in a grayscale frame.

        Thresholds at cfg.led_threshold, finds contours, filters by area,
        refines centroids with cornerSubPix, then orders TL/TR/BL/BR.
        Returns CornerDetection with stable=False if fewer than 4 blobs found.
        """
        _, thresh = cv2.threshold(frame, self._cfg.led_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for c in contours:
            area = cv2.contourArea(c)
            if self._cfg.min_blob_area <= area <= self._cfg.max_blob_area:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    centroids.append((cx, cy))

        if len(centroids) != 4:
            return CornerDetection(
                corners=np.zeros((4, 2), dtype=np.float32),
                centroids=centroids,
                stable=False,
            )

        # Subpixel refinement
        pts = np.array(centroids, dtype=np.float32).reshape(-1, 1, 2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        pts = cv2.cornerSubPix(frame, pts, (self._cfg.subpixel_window, self._cfg.subpixel_window),
                               (-1, -1), criteria)
        refined = pts.reshape(-1, 2)

        ordered = self.order_corners([(p[0], p[1]) for p in refined])
        return CornerDetection(corners=ordered, centroids=centroids, stable=True)

    def order_corners(self, blobs: list[tuple]) -> np.ndarray:
        """
        Order four blob centroids as [TL, TR, BL, BR].

        Splits into top/bottom halves by median Y, then left/right within each half.
        """
        pts = sorted(blobs, key=lambda p: p[1])   # sort by Y
        top = sorted(pts[:2], key=lambda p: p[0])  # left/right among top two
        bot = sorted(pts[2:], key=lambda p: p[0])  # left/right among bottom two
        return np.array([top[0], top[1], bot[0], bot[1]], dtype=np.float32)

    def compute_homography(self, corners: np.ndarray,
                           display_size: tuple[int, int]) -> CalibrationResult:
        """
        Compute perspective homography from camera corner points to display corners.

        camera corners: TL/TR/BL/BR in pixel coordinates
        display corners: mapped to the full display_size rectangle
        """
        w, h = display_size
        dst = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32)
        H, _ = cv2.findHomography(corners, dst, method=0)

        # Compute reprojection error
        src_h = np.array([[[p[0], p[1]]] for p in corners], dtype=np.float32)
        projected = cv2.perspectiveTransform(src_h, H).reshape(-1, 2)
        error = float(np.mean(np.linalg.norm(projected - dst, axis=1)))

        return CalibrationResult(H=H, reprojection_error=error, timestamp=time.monotonic())

    def camera_to_display(self, pt: tuple[float, float], H: np.ndarray) -> tuple[int, int]:
        """Map a single camera-space point to display-space using the homography."""
        src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, H)
        x, y = dst[0][0]
        return int(round(x)), int(round(y))

    def update(self, frame: np.ndarray) -> CalibrationResult | None:
        """
        Process a frame. Returns a CalibrationResult once corners are stable,
        or None while still accumulating stability frames.
        """
        detection = self.detect_corners(frame)
        if not detection.stable:
            self._stable_count = 0
            return None

        if self._prev_corners is not None:
            drift = np.max(np.linalg.norm(detection.corners - self._prev_corners, axis=1))
            if drift > self._cfg.max_corner_drift_px:
                self._stable_count = 0

        self._prev_corners = detection.corners
        self._stable_count += 1

        if self._stable_count >= self._cfg.min_stability_frames:
            self._result = self.compute_homography(detection.corners, self._display_size)
            self._stable_count = 0  # reset so re-calibration can trigger again on drift
            return self._result

        return None

    @property
    def current_result(self) -> CalibrationResult | None:
        """The most recently computed calibration, or None if not yet calibrated."""
        return self._result
