from __future__ import annotations

import argparse
import ctypes.util
import html
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
    test_parser.add_argument("--temperature-interval", type=float, default=1.0)
    test_parser.add_argument("--no-temperature-log", action="store_true")
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
        temperature_interval=None if args.no_temperature_log else args.temperature_interval,
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
        "temperature": stats.temperature,
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
        temperature_interval=None if args.no_temperature_log else args.temperature_interval,
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
        "temperature": result.temperature,
    }


def write_reports(report: dict[str, Any], yaml_path: Path, md_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    temperature_plot = write_npu_temperature_plot(report, md_path)
    yaml_path.write_text(yaml.safe_dump(report, allow_unicode=True, sort_keys=False), encoding="utf-8")
    md_path.write_text(
        markdown_report(report, temperature_plot.name if temperature_plot else None),
        encoding="utf-8",
    )


def markdown_report(report: dict[str, Any], temperature_plot_name: str | None = None) -> str:
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

    temperature = report.get("temperature", {})
    sensors = temperature.get("sensors", {})
    if sensors:
        lines.extend(
            [
                "",
                "## Temperature",
                "",
                f"- Samples: {temperature.get('sample_count', 0)}",
                f"- Interval: {format_value(temperature.get('interval_s'))} s",
            ]
        )
        if temperature_plot_name:
            lines.extend(["", f"![NPU temperature curve]({temperature_plot_name})"])
        lines.extend(
            [
                "",
                "| Sensor | Start C | End C | Min C | Max C | Avg C |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, summary in sensors.items():
            lines.append(
                f"| {name} | {format_value(summary.get('start_c'))} | "
                f"{format_value(summary.get('end_c'))} | {format_value(summary.get('min_c'))} | "
                f"{format_value(summary.get('max_c'))} | {format_value(summary.get('avg_c'))} |"
            )

    rows = report.get("per_class", [])
    if rows:
        lines.extend(["", "## Per Class", "", "| Class ID | Name | mAP50-95 |", "| ---: | --- | ---: |"])
        for row in rows:
            lines.append(f"| {row['class_id']} | {row['name']} | {format_value(row.get('map50_95'))} |")

    lines.append("")
    return "\n".join(lines)


def seconds_to_ms_per_frame(seconds: float, count: int) -> float:
    return 0.0 if count == 0 else seconds * 1000.0 / count


def write_npu_temperature_plot(report: dict[str, Any], md_path: Path) -> Path | None:
    temperature = report.get("temperature", {})
    samples = temperature.get("samples", [])
    if not isinstance(samples, list):
        return None

    sensor_name = npu_sensor_name(samples)
    if sensor_name is None:
        remove_stale_temperature_plot(md_path)
        return None

    points = temperature_points(samples, sensor_name)
    if len(points) < 2:
        remove_stale_temperature_plot(md_path)
        return None

    plot_path = md_path.with_name(f"{md_path.stem}_npu_temperature.svg")
    plot_path.write_text(render_temperature_svg(points, sensor_name), encoding="utf-8")
    return plot_path


def remove_stale_temperature_plot(md_path: Path) -> None:
    try:
        md_path.with_name(f"{md_path.stem}_npu_temperature.svg").unlink()
    except FileNotFoundError:
        pass


def npu_sensor_name(samples: list[dict[str, Any]]) -> str | None:
    for sample in samples:
        temperatures = sample.get("temperatures_c", {})
        if not isinstance(temperatures, dict):
            continue
        for name in temperatures:
            if "npu" in str(name).lower():
                return str(name)
    return None


def temperature_points(samples: list[dict[str, Any]], sensor_name: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for sample in samples:
        temperatures = sample.get("temperatures_c", {})
        if not isinstance(temperatures, dict) or sensor_name not in temperatures:
            continue
        try:
            elapsed = float(sample.get("elapsed_s", 0.0))
            value = float(temperatures[sensor_name])
        except (TypeError, ValueError):
            continue
        points.append((elapsed, value))
    return points


def render_temperature_svg(points: list[tuple[float, float]], sensor_name: str) -> str:
    width, height = 720, 260
    left, right, top, bottom = 58, 22, 30, 45
    plot_width = width - left - right
    plot_height = height - top - bottom
    min_time, max_time = points[0][0], points[-1][0]
    temps = [point[1] for point in points]
    min_temp, max_temp = min(temps), max(temps)
    if max_temp == min_temp:
        min_temp -= 0.5
        max_temp += 0.5
    else:
        padding = max(0.25, (max_temp - min_temp) * 0.12)
        min_temp -= padding
        max_temp += padding

    def sx(value: float) -> float:
        if max_time == min_time:
            return left + plot_width / 2.0
        return left + (value - min_time) * plot_width / (max_time - min_time)

    def sy(value: float) -> float:
        return top + (max_temp - value) * plot_height / (max_temp - min_temp)

    polyline = " ".join(f"{sx(elapsed):.2f},{sy(temp):.2f}" for elapsed, temp in points)
    circles = "\n".join(
        f'    <circle cx="{sx(elapsed):.2f}" cy="{sy(temp):.2f}" r="3.5">'
        f"<title>{elapsed:.2f}s: {temp:.3f} C</title></circle>"
        for elapsed, temp in points
    )
    title = html.escape(f"{sensor_name} temperature")
    sample_count = len(points)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <title>{title}</title>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="20" font-family="Arial, sans-serif" font-size="14" font-weight="700" fill="#1f2933">{title} ({sample_count} samples)</text>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#94a3b8" stroke-width="1"/>
  <line x1="{left}" y1="{sy(max(temps)):.2f}" x2="{left + plot_width}" y2="{sy(max(temps)):.2f}" stroke="#e2e8f0" stroke-width="1"/>
  <line x1="{left}" y1="{sy(min(temps)):.2f}" x2="{left + plot_width}" y2="{sy(min(temps)):.2f}" stroke="#e2e8f0" stroke-width="1"/>
  <text x="{left - 8}" y="{sy(max(temps)) + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#475569">{max(temps):.1f} C</text>
  <text x="{left - 8}" y="{sy(min(temps)) + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#475569">{min(temps):.1f} C</text>
  <text x="{left}" y="{height - 16}" text-anchor="start" font-family="Arial, sans-serif" font-size="11" fill="#475569">{min_time:.1f}s</text>
  <text x="{left + plot_width}" y="{height - 16}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#475569">{max_time:.1f}s</text>
  <polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
  <g fill="#ffffff" stroke="#2563eb" stroke-width="2">
{circles}
  </g>
</svg>
"""


def format_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
