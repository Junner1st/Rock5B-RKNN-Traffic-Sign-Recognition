from __future__ import annotations

import argparse
import ctypes.util
from pathlib import Path
from typing import Any

import yaml

from export_rknn import ensure_rknn_model, export_plan


def main() -> None:
    args = parse_args()
    if args.command == "test":
        run_test(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bridge Rock5B RKNN testing for yolo-ver-comp.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--weights", type=Path, required=True)
    test_parser.add_argument("--target", default="rk3588")
    test_parser.add_argument("--input-size", type=int, required=True)
    test_parser.add_argument("--input-layout", choices=("nchw", "nhwc"), default="nhwc")
    test_parser.add_argument("--conf", type=float, required=True)
    test_parser.add_argument("--iou", type=float, required=True)
    test_parser.add_argument("--split", default="test")
    test_parser.add_argument("--dataset-root", type=Path)
    test_parser.add_argument("--data-yaml", type=Path)
    test_parser.add_argument("--video", type=Path)
    test_parser.add_argument("--output-dir", type=Path, required=True)
    test_parser.add_argument("--output-video", type=Path)
    test_parser.add_argument("--report-yaml", type=Path, required=True)
    test_parser.add_argument("--report-md", type=Path, required=True)
    test_parser.add_argument("--max-images", type=int, default=12)
    test_parser.add_argument("--no-save-video", action="store_true")
    test_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_test(args: argparse.Namespace) -> None:
    plan = export_plan(args.weights.resolve(), args.target, args.input_size)
    rknn_model = ensure_rknn_model(args.weights.resolve(), args.target, args.input_size, dry_run=args.dry_run)

    if args.dry_run:
        print(
            "Rock5B RKNN bridge: "
            f"source={plan.source_weights}, model={rknn_model}, target={args.target}, "
            f"input_size={args.input_size}, output_dir={args.output_dir}"
        )
        return

    if args.video:
        ensure_librknnrt()
        report = run_video_mode(args, rknn_model)
    else:
        ensure_librknnrt()
        report = run_dataset_mode(args, rknn_model)
    write_reports(report, args.report_yaml, args.report_md)


def ensure_librknnrt() -> None:
    if ctypes.util.find_library("rknnrt"):
        return
    common_paths = [
        Path("/usr/lib/librknnrt.so"),
        Path("/usr/lib/aarch64-linux-gnu/librknnrt.so"),
        Path("/usr/local/lib/librknnrt.so"),
    ]
    if any(path.exists() for path in common_paths):
        return
    raise SystemExit(
        "librknnrt.so was not found. Install the RKNN runtime shared library first:\n"
        "  git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git /tmp/rknn-toolkit2\n"
        "  sudo cp /tmp/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/\n"
        "  sudo ldconfig\n"
    )


def run_video_mode(args: argparse.Namespace, rknn_model: Path) -> dict[str, Any]:
    from recognition import process_video

    output_video = args.output_video or args.output_dir / f"{args.video.stem}_rknn.mp4"
    stats = process_video(
        video_path=args.video.resolve(),
        model_path=rknn_model,
        output_path=output_video.resolve(),
        save_video=not args.no_save_video,
        max_frames=None,
        conf=args.conf,
        iou=args.iou,
        input_size=args.input_size,
        input_layout=args.input_layout,
        log_interval=1.0,
    )
    speed = {
        "preprocess": seconds_to_ms_per_frame(stats.preprocess_time, stats.frame_count),
        "inference": seconds_to_ms_per_frame(stats.inference_time, stats.frame_count),
        "postprocess": seconds_to_ms_per_frame(stats.postprocess_time, stats.frame_count),
        "total": seconds_to_ms_per_frame(stats.detection_time, stats.frame_count),
    }
    return {
        "split": args.split,
        "backend": "rknn",
        "mode": "video",
        "input_layout": args.input_layout,
        "video": str(args.video.resolve()),
        "output_video": str(output_video.resolve()) if not args.no_save_video else None,
        "image_count": None,
        "frame_count": stats.frame_count,
        "detections_total": stats.detections_total,
        "overall": {},
        "per_class": [],
        "speed": speed,
        "ms_per_img": speed,
    }


def run_dataset_mode(args: argparse.Namespace, rknn_model: Path) -> dict[str, Any]:
    from yolo_dataset_test import names_from_data_yaml, run_dataset_test

    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required when --video is not provided.")
    names = names_from_data_yaml(args.data_yaml)
    result = run_dataset_test(
        dataset_root=args.dataset_root.resolve(),
        split=args.split,
        model_path=rknn_model,
        output_dir=args.output_dir.resolve(),
        max_images=args.max_images,
        names=names,
        input_size=args.input_size,
        conf=args.conf,
        iou=args.iou,
        input_layout=args.input_layout,
    )
    speed = dict(result.speed_ms)
    if "total" not in speed:
        speed["total"] = sum(speed.get(key, 0.0) for key in ("preprocess", "inference", "postprocess"))
    return {
        "split": args.split,
        "backend": "rknn",
        "mode": "dataset",
        "input_layout": args.input_layout,
        "image_count": result.image_count,
        "overall": result.overall,
        "per_class": result.per_class,
        "speed": speed,
        "ms_per_img": speed,
    }


def write_reports(report: dict[str, Any], yaml_path: Path, md_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(report, allow_unicode=True, sort_keys=False), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")


def markdown_report(report: dict[str, Any]) -> str:
    overall = report.get("overall", {})
    lines = [
        f"# {str(report.get('split', 'test')).title()} RKNN Report",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.append(f"| mode | {report.get('mode', '')} |")
    lines.append(f"| input_layout | {report.get('input_layout', '')} |")
    lines.append(f"| images | {report.get('image_count') or ''} |")
    lines.append(f"| frames | {report.get('frame_count') or ''} |")
    for key in ("precision", "recall", "map50", "map50_95", "fitness"):
        lines.append(f"| {key} | {format_value(overall.get(key))} |")

    speed = report.get("ms_per_img", {})
    if speed:
        lines.extend(["", "## Speed", "", "| Metric | ms/img |", "| --- | ---: |"])
        for key in ("preprocess", "inference", "postprocess", "total"):
            if key in speed:
                lines.append(f"| {key} | {format_value(speed.get(key))} |")

    rows = report.get("per_class", [])
    if rows:
        lines.extend(["", "## Per Class", "", "| Class ID | Name | mAP50-95 |", "| ---: | --- | ---: |"])
        for row in rows:
            lines.append(f"| {row['class_id']} | {row['name']} | {format_value(row.get('map50_95'))} |")

    lines.append("")
    return "\n".join(lines)


def seconds_to_ms_per_frame(seconds: float, count: int) -> float:
    return 0.0 if count == 0 else seconds * 1000.0 / count


def format_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
