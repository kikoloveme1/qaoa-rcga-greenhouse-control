# Singapore greenhouse input data

The repository does not distribute greenhouse observations. Obtain a lawful Singapore dataset from its provider and convert one consecutive hourly window to CSV.

Required columns:

| Column | Unit | Meaning |
|---|---:|---|
| `timestamp` | ISO 8601 | Time with an explicit Singapore offset, such as `+08:00` |
| `outdoor_temperature_c` | °C | Outdoor dry-bulb temperature |
| `outdoor_rh_percent` | % | Outdoor relative humidity in `[0, 100]` |
| `solar_irradiance_w_m2` | W/m² | Nonnegative global solar irradiance |

Optional measured columns are `indoor_temperature_c`, `indoor_rh_percent`, `indoor_co2_ppm`, `temperature_setpoint_c`, `supplemental_light_w_m2`, `co2_injection`, and `humidity_setpoint_percent`.

Optional columns are loaded for reader-side analysis, but are not automatically used to fit the model, validate predictions, or replace optimized controls. This repository provides no automatic data download or assurance about a reader's chosen dataset.

Records must be strictly chronological, exactly one hour apart, numeric, and free of gaps. The program reads the first 24 records unless a different model horizon is selected. Keep downloaded and converted observations outside version control.
