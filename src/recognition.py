from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import config
import numpy as np
from rknnlite.api import RKNNLite

VIDEO_PATH = config.RECOGNITION_VIDEO_PATH
MODEL_PATH = config.RECOGNITION_MODEL_PATH
OUTPUT_PATH = config.RECOGNITION_OUTPUT_PATH
SAVE_VIDEO = config.RECOGNITION_SAVE_VIDEO
MAX_FRAMES = config.RECOGNITION_MAX_FRAMES
CONFIDENCE = config.RECOGNITION_CONFIDENCE
IOU = config.RECOGNITION_IOU
INPUT_SIZE = getattr(config, "RECOGNITION_INPUT_SIZE", 736)
LOG_INTERVAL = getattr(config, "RECOGNITION_LOG_INTERVAL", 1.0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    score: float
    cls: int


@dataclass
class RecognitionStats:
    frame_count: int = 0
    preprocess_time: float = 0.0
    inference_time: float = 0.0
    postprocess_time: float = 0.0
    annotation_time: float = 0.0
    detections_total: int = 0
    rolling_fps_samples: List[float] = field(default_factory=list)

    @property
    def detection_time(self) -> float:
        return self.preprocess_time + self.inference_time + self.postprocess_time

    @property
    def total_time(self) -> float:
        return self.detection_time + self.annotation_time

    @property
    def fps(self) -> float:
        return 0.0 if self.detection_time == 0 else self.frame_count / self.detection_time

    @property
    def avg_time_per_frame(self) -> float:
        return 0.0 if self.frame_count == 0 else self.detection_time / self.frame_count

    @property
    def avg_rolling_fps(self) -> float:
        if not self.rolling_fps_samples:
            return 0.0
        return sum(self.rolling_fps_samples) / len(self.rolling_fps_samples)


class TemporalTracker:
    def __init__(self, iou_thresh: float = 0.4, min_frames: int = 2, max_missing: int = 5):
        self.tracks: List[dict] = []
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.min_frames = min_frames
        self.max_missing = max_missing

    def _iou(self, a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)

    def update(self, detections: Sequence[Detection]) -> List[dict]:
        updated_tracks: List[dict] = []
        used = set()

        for track in self.tracks:
            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in used or track["cls"] != det.cls:
                    continue
                iou = self._iou(track["bbox"], det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= self.iou_thresh and best_idx >= 0:
                det = detections[best_idx]
                used.add(best_idx)
                track["bbox"] = det.bbox
                track["score"] = 0.7 * track["score"] + 0.3 * det.score
                track["frames"] += 1
                track["missing"] = 0
                updated_tracks.append(track)
            else:
                track["missing"] += 1
                if track["missing"] <= self.max_missing:
                    updated_tracks.append(track)

        for i, det in enumerate(detections):
            if i in used:
                continue
            updated_tracks.append(
                {
                    "id": self.next_id,
                    "bbox": det.bbox,
                    "score": float(det.score),
                    "cls": int(det.cls),
                    "frames": 1,
                    "missing": 0,
                }
            )
            self.next_id += 1

        self.tracks = updated_tracks
        return [t for t in self.tracks if t["frames"] >= self.min_frames and t["missing"] == 0]


class RollingFPS:
    def __init__(self, window_seconds: float = 1.0):
        self.window_seconds = window_seconds
        self.timestamps: deque[float] = deque()

    def update(self, timestamp: float) -> float:
        self.timestamps.append(timestamp)
        cutoff = timestamp - self.window_seconds
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        if len(self.timestamps) <= 1:
            return float(len(self.timestamps))
        elapsed = max(1e-6, self.timestamps[-1] - self.timestamps[0])
        return len(self.timestamps) / elapsed


class RKNNDetector:
    def __init__(self, model_path: Path, input_size: int, conf: float, iou: float):
        self.model_path = model_path
        self.input_size = int(input_size)
        self.conf = conf
        self.iou = iou
        self.rknn = RKNNLite()
        self.names = load_names(model_path)

    def __enter__(self) -> "RKNNDetector":
        ret = self.rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"Failed to load RKNN model: {self.model_path}")

        core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", None)
        if core_mask is None:
            ret = self.rknn.init_runtime()
        else:
            ret = self.rknn.init_runtime(core_mask=core_mask)
        if ret != 0:
            raise RuntimeError("Failed to init RKNN runtime")
        return self

    def __exit__(self, *_exc) -> None:
        self.rknn.release()

    def predict(self, frame: np.ndarray) -> tuple[List[Detection], dict]:
        t0 = time.perf_counter()
        model_input, ratio, pad = letterbox(frame, self.input_size)
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
        model_input = np.expand_dims(np.ascontiguousarray(model_input), axis=0)
        t1 = time.perf_counter()

        outputs = self.rknn.inference(inputs=[model_input])
        t2 = time.perf_counter()

        detections = postprocess_yolo_output(
            outputs[0],
            original_shape=frame.shape[:2],
            ratio=ratio,
            pad=pad,
            conf_thres=self.conf,
            iou_thres=self.iou,
        )
        t3 = time.perf_counter()
        timings = {
            "preprocess": t1 - t0,
            "inference": t2 - t1,
            "postprocess": t3 - t2,
        }
        return detections, timings


class VideoSink:
    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        self.proc = None
        self.writer = None
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which("ffmpeg"):
            self.proc = open_ffmpeg_writer(output_path, width, height, fps)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            if self.proc.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")
            self.proc.stdin.write(frame.tobytes())
        elif self.writer is not None:
            self.writer.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
            self.proc.wait()
        if self.writer is not None:
            self.writer.release()


def load_names(model_path: Path) -> List[str]:
    metadata_path = model_path.with_name("metadata.yaml")
    if not metadata_path.exists():
        return []

    text = metadata_path.read_text(encoding="utf-8")
    names: dict[int, str] = {}
    in_names = False
    for line in text.splitlines():
        if line.strip() == "names:":
            in_names = True
            continue
        if in_names and line and not line.startswith(" "):
            break
        if not in_names:
            continue
        match = re.match(r"\s+(\d+):\s*(.+)\s*$", line)
        if not match:
            continue
        idx = int(match.group(1))
        raw_value = match.group(2)
        try:
            value = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            value = raw_value.strip().strip("'\"")
        names[idx] = str(value)
    return [names[i] for i in range(max(names) + 1)] if names else []


def ensure_paths(video_path: Path, model_path: Path) -> None:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if model_path.suffix.lower() != ".rknn":
        raise ValueError(f"Rock5B mode expects an .rknn model, got: {model_path}")


def letterbox(image: np.ndarray, input_size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(input_size / height, input_size / width)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2
    padded[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return padded, ratio, (float(pad_x), float(pad_y))


def postprocess_yolo_output(
    output: np.ndarray,
    original_shape: tuple[int, int],
    ratio: float,
    pad: tuple[float, float],
    conf_thres: float,
    iou_thres: float,
) -> List[Detection]:
    pred = np.squeeze(output)
    if pred.ndim != 2:
        raise ValueError(f"Unexpected RKNN output shape: {output.shape}")
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    if pred.shape[1] < 5:
        return []

    boxes_xywh = pred[:, :4].astype(np.float32, copy=False)
    class_scores = pred[:, 4:].astype(np.float32, copy=False)
    finite_mask = np.isfinite(boxes_xywh).all(axis=1) & np.isfinite(class_scores).all(axis=1)
    if not np.any(finite_mask):
        return []

    boxes_xywh = boxes_xywh[finite_mask]
    class_scores = class_scores[finite_mask]
    classes = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), classes]
    keep = scores >= conf_thres
    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    classes = classes[keep]

    x, y, w, h = boxes_xywh.T
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0

    pad_x, pad_y = pad
    orig_h, orig_w = original_shape
    x1 = np.clip((x1 - pad_x) / ratio, 0, orig_w - 1)
    y1 = np.clip((y1 - pad_y) / ratio, 0, orig_h - 1)
    x2 = np.clip((x2 - pad_x) / ratio, 0, orig_w - 1)
    y2 = np.clip((y2 - pad_y) / ratio, 0, orig_h - 1)

    valid = (x2 > x1) & (y2 > y1)
    if not np.any(valid):
        return []

    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    scores = scores[valid]
    classes = classes[valid]
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_thres, iou_thres)
    if len(indices) == 0:
        return []

    indices = np.array(indices).reshape(-1)
    return [
        Detection(
            bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
            score=float(scores[i]),
            cls=int(classes[i]),
        )
        for i in indices
    ]


def annotate_frame(
    frame: np.ndarray,
    detections: Sequence[Detection],
    names: Sequence[str],
    tracks: Sequence[dict] | None = None,
    fps: float | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    items: Iterable = tracks if tracks else detections

    for item in items:
        if isinstance(item, Detection):
            bbox, cls, score, ident = item.bbox, item.cls, item.score, None
        else:
            bbox, cls, score, ident = item["bbox"], item["cls"], item["score"], item["id"]
        x1, y1, x2, y2 = map(int, bbox)
        label = names[cls] if 0 <= cls < len(names) else str(cls)
        text = f"{label} {score:.2f}" if ident is None else f"{label}#{ident} {score:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(annotated, text, (x1, max(14, y1 - 6)), FONT, 0.5, (0, 0, 0), 2)
        cv2.putText(annotated, text, (x1, max(14, y1 - 6)), FONT, 0.5, (255, 255, 255), 1)

    if fps is not None:
        text = f"avg recognition/sec: {fps:.2f}"
        cv2.putText(annotated, text, (12, 28), FONT, 0.75, (0, 0, 0), 3)
        cv2.putText(annotated, text, (12, 28), FONT, 0.75, (255, 255, 255), 1)
    return annotated


def open_ffmpeg_writer(output_path: Path, width: int, height: int, fps: float):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_video(
    video_path: Path,
    model_path: Path,
    output_path: Path,
    save_video: bool,
    max_frames: int | None,
    conf: float,
    iou: float,
    input_size: int,
    log_interval: float,
) -> RecognitionStats:
    ensure_paths(video_path, model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_sink = VideoSink(output_path, width, height, fps) if save_video else None
    tracker = TemporalTracker()
    rolling = RollingFPS(window_seconds=1.0)
    stats = RecognitionStats()
    next_log = time.perf_counter() + max(0.1, log_interval)

    try:
        with RKNNDetector(model_path, input_size, conf, iou) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and stats.frame_count >= max_frames:
                    break

                detections, timings = detector.predict(frame)
                now = time.perf_counter()
                current_fps = rolling.update(now)
                tracks = tracker.update(detections)

                stats.frame_count += 1
                stats.detections_total += len(detections)
                stats.preprocess_time += timings["preprocess"]
                stats.inference_time += timings["inference"]
                stats.postprocess_time += timings["postprocess"]
                stats.rolling_fps_samples.append(current_fps)

                if video_sink is not None:
                    t0 = time.perf_counter()
                    annotated = annotate_frame(frame, detections, detector.names, tracks, current_fps)
                    video_sink.write(annotated)
                    stats.annotation_time += time.perf_counter() - t0

                if log_interval > 0 and now >= next_log:
                    print(
                        f"frames={stats.frame_count} "
                        f"recent_recognition_per_sec={current_fps:.2f} "
                        f"overall_recognition_per_sec={stats.fps:.2f} "
                        f"detections={stats.detections_total}"
                    )
                    next_log = now + log_interval
    finally:
        cap.release()
        if video_sink is not None:
            video_sink.close()

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rock5B/RK3588 RKNN traffic sign recognition.")
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--no-save-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--conf", type=float, default=CONFIDENCE)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE)
    parser.add_argument("--log-interval", type=float, default=LOG_INTERVAL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = process_video(
        video_path=args.video.resolve(),
        model_path=args.model.resolve(),
        output_path=args.output.resolve(),
        save_video=SAVE_VIDEO and not args.no_save_video,
        max_frames=args.max_frames,
        conf=args.conf,
        iou=args.iou,
        input_size=args.input_size,
        log_interval=args.log_interval,
    )
    print("=== Recognition Summary ===")
    print(f"Frames processed: {stats.frame_count}")
    print(f"Preprocess time: {stats.preprocess_time:.3f}s")
    print(f"RKNN inference time: {stats.inference_time:.3f}s")
    print(f"Postprocess/NMS time: {stats.postprocess_time:.3f}s")
    print(f"Detection pipeline time: {stats.detection_time:.3f}s")
    print(f"Average per frame: {stats.avg_time_per_frame:.4f}s")
    print(f"Average recognition per second: {stats.fps:.2f}")
    print(f"Average rolling 1s recognition per second: {stats.avg_rolling_fps:.2f}")
    print(f"Detections counted: {stats.detections_total}")


if __name__ == "__main__":
    main()
