"""
ePaper Target — main entry point (placeholder).

Run the full system once all subsystems have been validated with bench tests.
"""
import sys
from epaper_target.config import SystemConfig


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = SystemConfig.load(config_path) if config_path else SystemConfig()
    print("ePaper Target — system not yet fully implemented.")
    print(f"Config: camera={cfg.camera.device_index}, "
          f"{cfg.camera.width}x{cfg.camera.height}@{cfg.camera.fps}fps")


if __name__ == "__main__":
    main()
