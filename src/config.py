from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

RECOGNITION_VIDEO_PATH = (DATA_DIR / "videos/sample-day-1.mp4").resolve()
RECOGNITION_MODEL_PATH = (PROJECT_DIR / "best_rknn_model/best-rk3588.rknn").resolve()
RECOGNITION_OUTPUT_PATH = (DATA_DIR / "recognized/sample_recognized_tracked.mp4").resolve()

RECOGNITION_SAVE_VIDEO = True
RECOGNITION_MAX_FRAMES = None
RECOGNITION_CONFIDENCE = 0.25
RECOGNITION_IOU = 0.5
RECOGNITION_INPUT_SIZE = 736
RECOGNITION_LOG_INTERVAL = 1.0
