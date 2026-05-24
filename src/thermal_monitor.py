from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


THERMAL_ROOT = Path("/sys/class/thermal")


@dataclass(frozen=True)
class ThermalZone:
    name: str
    temp_path: Path


class TemperatureMonitor:
    def __init__(self, interval_seconds: float | None = 1.0, thermal_root: Path = THERMAL_ROOT):
        self.enabled = interval_seconds is not None
        self.interval_seconds = max(0.0, float(interval_seconds or 0.0))
        self.thermal_root = thermal_root
        self.zones = discover_thermal_zones(thermal_root) if self.enabled else []
        self.samples: list[dict[str, Any]] = []
        self.started_at: float | None = None
        self.last_sample_at: float | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.started_at = time.perf_counter()
        self.sample(force=True)

    def sample_due(self) -> None:
        if not self.enabled:
            return
        if self.last_sample_at is None:
            self.sample(force=True)
            return
        if time.perf_counter() - self.last_sample_at >= self.interval_seconds:
            self.sample(force=True)

    def sample(self, force: bool = False) -> None:
        if not self.enabled or not self.zones:
            return
        now = time.perf_counter()
        if not force and self.last_sample_at is not None and now - self.last_sample_at < self.interval_seconds:
            return
        if self.started_at is None:
            self.started_at = now

        temperatures = {
            zone.name: value
            for zone in self.zones
            if (value := read_temperature_c(zone.temp_path)) is not None
        }
        if temperatures:
            self.samples.append(
                {
                    "elapsed_s": round(now - self.started_at, 6),
                    "temperatures_c": temperatures,
                }
            )
            self.last_sample_at = now

    def finish(self) -> dict[str, Any]:
        if self.enabled:
            self.sample(force=True)
        return self.report()

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": bool(self.zones),
            "interval_s": self.interval_seconds if self.enabled else None,
            "sample_count": len(self.samples),
            "sensors": summarize_samples(self.samples),
            "samples": self.samples,
        }


def discover_thermal_zones(thermal_root: Path = THERMAL_ROOT) -> list[ThermalZone]:
    zones: list[ThermalZone] = []
    used_names: dict[str, int] = {}
    for zone_dir in sorted(thermal_root.glob("thermal_zone*")):
        temp_path = zone_dir / "temp"
        if not temp_path.exists():
            continue
        raw_name = read_text(zone_dir / "type") or zone_dir.name
        base_name = raw_name.strip() or zone_dir.name
        suffix_count = used_names.get(base_name, 0)
        used_names[base_name] = suffix_count + 1
        name = base_name if suffix_count == 0 else f"{base_name}_{suffix_count + 1}"
        zones.append(ThermalZone(name=name, temp_path=temp_path))
    return zones


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def read_temperature_c(path: Path) -> float | None:
    raw = read_text(path)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if abs(value) > 1000:
        value /= 1000.0
    return round(value, 3)


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    values_by_sensor: dict[str, list[float]] = {}
    for sample in samples:
        for name, value in sample.get("temperatures_c", {}).items():
            values_by_sensor.setdefault(name, []).append(float(value))

    summary: dict[str, dict[str, float]] = {}
    for name, values in values_by_sensor.items():
        summary[name] = {
            "start_c": round(values[0], 3),
            "end_c": round(values[-1], 3),
            "min_c": round(min(values), 3),
            "max_c": round(max(values), 3),
            "avg_c": round(sum(values) / len(values), 3),
        }
    return summary
