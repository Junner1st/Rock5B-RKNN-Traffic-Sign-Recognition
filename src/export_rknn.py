from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportPlan:
    source_weights: Path
    target: str
    input_size: int
    output_dir: Path
    output_model: Path


def export_plan(source_weights: Path, target: str, input_size: int) -> ExportPlan:
    if source_weights.suffix.lower() == ".rknn":
        return ExportPlan(
            source_weights=source_weights,
            target=target,
            input_size=input_size,
            output_dir=source_weights.parent,
            output_model=source_weights,
        )

    output_dir = source_weights.with_name(f"{source_weights.stem}_rknn_model")
    output_model = output_dir / f"{source_weights.stem}-{target}.rknn"
    return ExportPlan(
        source_weights=source_weights,
        target=target,
        input_size=input_size,
        output_dir=output_dir,
        output_model=output_model,
    )


def ensure_rknn_model(source_weights: Path, target: str, input_size: int, dry_run: bool = False) -> Path:
    plan = export_plan(source_weights, target, input_size)
    if plan.output_model.exists():
        return plan.output_model.resolve()

    if source_weights.suffix.lower() == ".rknn":
        raise FileNotFoundError(f"RKNN model not found: {source_weights}")
    if source_weights.suffix.lower() != ".pt":
        raise ValueError(f"RKNN export expects a .pt source model, got: {source_weights}")
    if not source_weights.exists():
        raise FileNotFoundError(f"Source weights not found: {source_weights}")
    if dry_run:
        return plan.output_model.resolve()

    exported_dir = export_with_ultralytics(source_weights, target, input_size)
    exported_model = find_exported_model(exported_dir, source_weights.stem, target)
    if not exported_model.exists():
        raise FileNotFoundError(f"RKNN export finished but model was not found: {exported_model}")
    return exported_model.resolve()


def export_with_ultralytics(source_weights: Path, target: str, input_size: int) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(source_weights))
    exported = model.export(format="rknn", name=target, imgsz=input_size, batch=1)
    exported_path = Path(exported[0] if isinstance(exported, tuple) else exported)
    return exported_path.resolve()


def find_exported_model(exported_dir: Path, source_stem: str, target: str) -> Path:
    expected = exported_dir / f"{source_stem}-{target}.rknn"
    if expected.exists():
        return expected
    matches = sorted(exported_dir.glob("*.rknn")) if exported_dir.exists() else []
    if len(matches) == 1:
        return matches[0]
    return expected

