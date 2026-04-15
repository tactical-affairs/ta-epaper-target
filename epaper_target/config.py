from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 60
    exposure: int = -6          # V4L2 manual exposure value (log scale)


@dataclass
class DetectionConfig:
    diff_threshold: int = 30    # absdiff threshold for frame differencing
    min_blob_area: int = 5      # minimum contour area (px^2) for a hit candidate
    max_blob_area: int = 500    # maximum contour area (px^2) for a hit candidate
    min_circularity: float = 0.5
    cooldown_frames: int = 10   # frames to ignore after a detected hit


@dataclass
class CalibrationConfig:
    led_threshold: int = 200    # brightness threshold for corner LED blobs
    min_blob_area: int = 5
    max_blob_area: int = 200
    min_stability_frames: int = 15   # consecutive stable frames before accepting calibration
    subpixel_window: int = 5         # half-window size for cornerSubPix refinement
    max_corner_drift_px: float = 2.0 # drift tolerance before recalibration


@dataclass
class DisplayConfig:
    width: int = 1024
    height: int = 600
    fullscreen: bool = True
    hit_marker_radius: int = 8
    hit_marker_color: tuple = (255, 50, 50)   # BGR


@dataclass
class LEDConfig:
    gpio_pins: list[int] = field(default_factory=lambda: [17, 18, 27, 22])  # TL/TR/BL/BR


@dataclass
class SystemConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    led: LEDConfig = field(default_factory=LEDConfig)

    @classmethod
    def load(cls, path: str) -> SystemConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        if "camera" in data:
            cfg.camera = CameraConfig(**data["camera"])
        if "detection" in data:
            cfg.detection = DetectionConfig(**data["detection"])
        if "calibration" in data:
            cfg.calibration = CalibrationConfig(**data["calibration"])
        if "display" in data:
            cfg.display = DisplayConfig(**data["display"])
        if "led" in data:
            cfg.led = LEDConfig(**data["led"])
        return cfg

    def save(self, path: str) -> None:
        """Save config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)
