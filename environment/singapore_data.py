"""Validated loading of reader-supplied Singapore greenhouse observations."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = (
    "timestamp",
    "outdoor_temperature_c",
    "outdoor_rh_percent",
    "solar_irradiance_w_m2",
)
OPTIONAL_COLUMNS = (
    "indoor_temperature_c",
    "indoor_rh_percent",
    "indoor_co2_ppm",
    "temperature_setpoint_c",
    "supplemental_light_w_m2",
    "co2_injection",
    "humidity_setpoint_percent",
)


@dataclass(frozen=True)
class SingaporeGreenhouseData:
    timestamps: tuple
    outdoor_temperature_c: np.ndarray
    outdoor_rh_percent: np.ndarray
    solar_irradiance_w_m2: np.ndarray
    optional: dict[str, np.ndarray] = field(default_factory=dict)
    source: str = ""


def _numeric(rows, name):
    try:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {name} must contain numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"column {name} contains non-finite values")
    return values


def load_singapore_csv(path, horizon=24):
    """Load the first consecutive hourly window from a timestamped CSV file."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if len(rows) < horizon:
        raise ValueError(f"at least {horizon} hourly records are required")
    rows = rows[:horizon]

    timestamps = []
    for row in rows:
        try:
            timestamp = datetime.fromisoformat(row["timestamp"])
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamps must use ISO 8601 format") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamps must include a timezone offset")
        timestamps.append(timestamp)
    if any(right - left != timedelta(hours=1) for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("records must be strictly chronological and hourly")

    temperature = _numeric(rows, "outdoor_temperature_c")
    humidity = _numeric(rows, "outdoor_rh_percent")
    solar = _numeric(rows, "solar_irradiance_w_m2")
    if np.any((humidity < 0.0) | (humidity > 100.0)):
        raise ValueError("outdoor_rh_percent must be within [0, 100]")
    if np.any(solar < 0.0):
        raise ValueError("solar_irradiance_w_m2 must be nonnegative")

    optional = {}
    for name in OPTIONAL_COLUMNS:
        if name in columns and any(row.get(name, "").strip() for row in rows):
            optional[name] = _numeric(rows, name)
    return SingaporeGreenhouseData(
        timestamps=tuple(timestamps),
        outdoor_temperature_c=temperature,
        outdoor_rh_percent=humidity,
        solar_irradiance_w_m2=solar,
        optional=optional,
        source=str(path.resolve()),
    )


def _assign_profiles(model, data):
    if model.config.T_steps != len(data.timestamps):
        raise ValueError("data horizon does not match the greenhouse configuration")
    model._T_out = data.outdoor_temperature_c.copy()
    model._RH_out = data.outdoor_rh_percent.copy()
    model._solar = data.solar_irradiance_w_m2.copy()


def apply_singapore_data(env, data):
    """Inject validated outdoor profiles into a base or perturbation environment."""
    if hasattr(env, "base_env"):
        _assign_profiles(env.base_env, data)
        env.eval_env = env._build_eval_env()
    else:
        _assign_profiles(env, data)
    return env
