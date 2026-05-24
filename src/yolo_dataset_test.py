from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

from recognition import Detection, RKNNDetector, annotate_frame

IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


@dataclass
class GroundTruth:
    bbox: tuple[float, float, float, float]
    cls: int


@dataclass
class ImageTestResult:
    predictions: dict[Path, list[Detection]]
    speed_ms: dict[str, float]
    overall: dict[str, float | None]
    per_class: list[dict[str, float | int | str | None]]
    image_count: int


def run_dataset_test(
    dataset_root: Path,
    split: str,
    model_path: Path,
    output_dir: Path,
    max_images: int,
    names: list[str],
    input_size: int,
    conf: float,
    iou: float,
    input_layout: str = "nhwc",
) -> ImageTestResult:
    images = image_paths(dataset_root, split)
    predictions: dict[Path, list[Detection]] = {}
    timings = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    preview_images = set(images[:max_images])

    print(f"RKNN model: {model_path}")
    print(f"Images: {len(images)} ({split})")
    print(f"Output directory: {output_dir}")

    with RKNNDetector(model_path, input_size, conf, iou, input_layout=input_layout) as detector:
        names = names or detector.names
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, image_path in enumerate(images, start=1):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {image_path}")

            detections, image_timings = detector.predict(frame)
            predictions[image_path] = detections
            for key, value in image_timings.items():
                timings[key] += value

            if image_path in preview_images:
                annotated = annotate_frame(frame, detections, names)
                cv2.imwrite(str(output_dir / image_path.name), annotated)

            if index == 1 or index % 25 == 0 or index == len(images):
                print(f"RKNN tested {index}/{len(images)} images")

    ground_truths = {
        image_path: read_yolo_label(label_path_for_image(image_path), image_path)
        for image_path in images
    }
    overall, per_class = evaluate_predictions(predictions, ground_truths, names)
    image_count = len(images)
    speed_ms = {
        key: (value * 1000.0 / image_count if image_count else 0.0)
        for key, value in timings.items()
    }
    return ImageTestResult(
        predictions=predictions,
        speed_ms=speed_ms,
        overall=overall,
        per_class=per_class,
        image_count=image_count,
    )


def image_paths(dataset_root: Path, split: str) -> list[Path]:
    split_dir = "valid" if split == "val" else split
    image_dir = dataset_root / split_dir / "images"
    paths: list[Path] = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(image_dir.glob(pattern))
    return sorted(paths)


def names_from_data_yaml(data_yaml: Path | None) -> list[str]:
    if data_yaml is None or not data_yaml.exists():
        return []
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", [])
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    return [str(name) for name in names]


def label_path_for_image(image_path: Path) -> Path:
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def read_yolo_label(label_path: Path, image_path: Path) -> list[GroundTruth]:
    if not label_path.exists():
        return []

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to read image for labels: {image_path}")
    height, width = frame.shape[:2]
    rows: list[GroundTruth] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        if len(values) == 4:
            x_center, y_center, box_width, box_height = values
            x1 = (x_center - box_width / 2.0) * width
            y1 = (y_center - box_height / 2.0) * height
            x2 = (x_center + box_width / 2.0) * width
            y2 = (y_center + box_height / 2.0) * height
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x1 = min(xs) * width
            y1 = min(ys) * height
            x2 = max(xs) * width
            y2 = max(ys) * height
        else:
            continue
        rows.append(GroundTruth(bbox=(x1, y1, x2, y2), cls=cls))
    return rows


def evaluate_predictions(
    predictions: dict[Path, list[Detection]],
    ground_truths: dict[Path, list[GroundTruth]],
    names: list[str],
) -> tuple[dict[str, float | None], list[dict[str, float | int | str | None]]]:
    class_ids = sorted(
        {
            item.cls
            for items in list(predictions.values()) + list(ground_truths.values())
            for item in items
        }
    )
    if not class_ids:
        return empty_metrics(), []

    per_class = []
    ap50_values = []
    map_values = []
    precision_values = []
    recall_values = []
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    for class_id in class_ids:
        ap_by_threshold = [
            average_precision_for_class(predictions, ground_truths, class_id, threshold)
            for threshold in thresholds
        ]
        precision, recall = precision_recall_for_class(predictions, ground_truths, class_id, 0.5)
        ap50 = ap_by_threshold[0]
        map50_95 = float(np.mean(ap_by_threshold)) if ap_by_threshold else 0.0
        ap50_values.append(ap50)
        map_values.append(map50_95)
        precision_values.append(precision)
        recall_values.append(recall)
        per_class.append(
            {
                "class_id": class_id,
                "name": names[class_id] if 0 <= class_id < len(names) else str(class_id),
                "precision": precision,
                "recall": recall,
                "map50": ap50,
                "map50_95": map50_95,
            }
        )

    overall = {
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "map50": float(np.mean(ap50_values)),
        "map50_95": float(np.mean(map_values)),
        "fitness": float(np.mean(map_values)),
    }
    return overall, per_class


def empty_metrics() -> dict[str, float | None]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map50_95": 0.0,
        "fitness": 0.0,
    }


def average_precision_for_class(
    predictions: dict[Path, list[Detection]],
    ground_truths: dict[Path, list[GroundTruth]],
    class_id: int,
    iou_threshold: float,
) -> float:
    y_true, _y_score, gt_count = match_class_predictions(predictions, ground_truths, class_id, iou_threshold)
    if gt_count == 0 or not y_true:
        return 0.0

    true_positive = np.cumsum(np.array(y_true, dtype=np.float32))
    false_positive = np.cumsum(1.0 - np.array(y_true, dtype=np.float32))
    recall = true_positive / max(gt_count, 1)
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-9)
    return interpolated_ap(recall, precision)


def precision_recall_for_class(
    predictions: dict[Path, list[Detection]],
    ground_truths: dict[Path, list[GroundTruth]],
    class_id: int,
    iou_threshold: float,
) -> tuple[float, float]:
    y_true, _y_score, gt_count = match_class_predictions(predictions, ground_truths, class_id, iou_threshold)
    if not y_true:
        return 0.0, 0.0
    tp = float(sum(y_true))
    fp = float(len(y_true) - sum(y_true))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / gt_count if gt_count else 0.0
    return precision, recall


def match_class_predictions(
    predictions: dict[Path, list[Detection]],
    ground_truths: dict[Path, list[GroundTruth]],
    class_id: int,
    iou_threshold: float,
) -> tuple[list[int], list[float], int]:
    detections = [
        (image_path, detection)
        for image_path, image_detections in predictions.items()
        for detection in image_detections
        if detection.cls == class_id
    ]
    detections.sort(key=lambda item: item[1].score, reverse=True)
    class_ground_truths = {
        image_path: [gt for gt in image_ground_truths if gt.cls == class_id]
        for image_path, image_ground_truths in ground_truths.items()
    }
    gt_count = sum(len(items) for items in class_ground_truths.values())
    matched: dict[Path, set[int]] = {image_path: set() for image_path in class_ground_truths}
    y_true: list[int] = []
    y_score: list[float] = []

    for image_path, detection in detections:
        candidates = class_ground_truths.get(image_path, [])
        best_iou = 0.0
        best_index = -1
        for index, gt in enumerate(candidates):
            if index in matched[image_path]:
                continue
            value = box_iou(detection.bbox, gt.bbox)
            if value > best_iou:
                best_iou = value
                best_index = index
        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_path].add(best_index)
            y_true.append(1)
        else:
            y_true.append(0)
        y_score.append(float(detection.score))
    return y_true, y_score, gt_count


def interpolated_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1]))


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    return inter / max(area_a + area_b - inter, 1e-9)
