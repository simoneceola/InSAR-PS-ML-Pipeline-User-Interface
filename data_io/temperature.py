"""Temperature series: measured `tavg` CSV if available, else a climatological fallback."""

import numpy as np
import pandas as pd

from app_config import TEMP_CSV_PATH


def build_reference_temperature(dates_dt: pd.DatetimeIndex) -> np.ndarray:
    """Climatological air temperature (°C) fallback for the Netherlands."""
    doy = dates_dt.dayofyear.to_numpy(dtype=float)
    return 10.5 - 7.5 * np.cos(2.0 * np.pi * (doy - 23.0) / 365.25)


def load_temperature_series(dates_dt: pd.DatetimeIndex) -> tuple[np.ndarray, bool]:
    """Load daily average temperatures (°C) from filtered_date_tavg.csv next to app.py."""
    if TEMP_CSV_PATH.exists():
        df_temp = pd.read_csv(TEMP_CSV_PATH)
        if "tavg" in df_temp.columns:
            date_cols = [c for c in df_temp.columns if c.lower() in {"date", "time", "dt", "datetime"}]
            if date_cols:
                d_col = date_cols[0]
                aligned = (
                    df_temp[[d_col, "tavg"]]
                    .assign(_dt=lambda d: pd.to_datetime(d[d_col]).dt.normalize())
                    .dropna(subset=["_dt"])
                    .drop_duplicates(subset=["_dt"])
                    .set_index("_dt")["tavg"]
                    .reindex(dates_dt.normalize())
                )
                if aligned.notna().sum() >= max(1, int(0.8 * len(dates_dt))):
                    filled = aligned.interpolate(method="linear", limit_direction="both").bfill().ffill()
                    return filled.to_numpy(dtype=float), True

            if len(df_temp) == len(dates_dt):
                s = pd.Series(df_temp["tavg"].values, dtype=float)
                filled = s.interpolate(method="linear", limit_direction="both").bfill().ffill()
                return filled.to_numpy(dtype=float), True

    return build_reference_temperature(dates_dt), False
