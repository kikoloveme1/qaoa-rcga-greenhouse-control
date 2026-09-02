from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from environment.greenhouse_model import GreenhouseEnv
from environment.singapore_data import apply_singapore_data, load_singapore_csv


HEADER = (
    "timestamp,outdoor_temperature_c,outdoor_rh_percent,"
    "solar_irradiance_w_m2\n"
)


def _write_weather(path, spacing_hours=1):
    start = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=8)))
    rows = [HEADER]
    for hour in range(24):
        timestamp = start + timedelta(hours=hour * spacing_hours)
        rows.append(f"{timestamp.isoformat()},{27 + hour / 10:.1f},82,{hour * 10}\n")
    path.write_text("".join(rows), encoding="utf-8")


def test_loader_reads_hourly_timezone_aware_weather_and_injects_it(tmp_path):
    path = tmp_path / "singapore.csv"
    _write_weather(path)
    data = load_singapore_csv(path, horizon=24)
    env = GreenhouseEnv(seed=1)
    apply_singapore_data(env, data)

    assert len(data.timestamps) == 24
    assert np.isclose(env.T_out[0], 27.0)
    assert np.isclose(env.T_out[-1], 29.3)
    assert np.isclose(env.solar_profile[-1], 230.0)


def test_loader_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp,outdoor_temperature_c\n"
        "2026-01-01T00:00:00+08:00,27\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outdoor_rh_percent"):
        load_singapore_csv(path)


def test_loader_rejects_non_hourly_records(tmp_path):
    path = tmp_path / "bad-spacing.csv"
    _write_weather(path, spacing_hours=2)
    with pytest.raises(ValueError, match="hourly"):
        load_singapore_csv(path, horizon=24)
