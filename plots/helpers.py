"""Small helpers shared by several figure builders."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def pack_multilines(dates_dt: pd.DatetimeIndex, series_2d: np.ndarray) -> tuple[list[str | None], np.ndarray]:
    """Pack an (N, T) matrix into None-separated 1D arrays for fast multi-line Plotly rendering."""
    n_pts = series_2d.shape[0]
    dates_str = dates_dt.strftime("%Y-%m-%d").tolist() + [None]
    x_all = dates_str * n_pts
    y_all = np.column_stack([series_2d, np.full(n_pts, np.nan)]).ravel()
    return x_all, y_all


def add_season_shading(
    fig: go.Figure,
    dates_dt: pd.DatetimeIndex,
    cols: list[int] | None = None,
    row: int = 1,
) -> None:
    """Shade summer (Jun–Aug, orange) and winter (Dec–Feb, blue).

    cols=None → plain figure (no subplots); otherwise shade the given subplot columns.
    """
    d_min, d_max = dates_dt.min(), dates_dt.max()

    def _rect(x0, x1, color: str) -> None:
        if cols is None:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer="below")
        else:
            for c in cols:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0, layer="below", row=row, col=c)

    for yr in range(d_min.year - 1, d_max.year + 2):
        s0, s1 = pd.Timestamp(yr, 6, 1), pd.Timestamp(yr, 8, 31)
        if s1 >= d_min and s0 <= d_max:
            _rect(max(s0, d_min), min(s1, d_max), "rgba(255, 165, 0, 0.08)")
        w0 = pd.Timestamp(yr, 12, 1)
        w1 = pd.Timestamp(yr + 1, 2, 28) + pd.offsets.MonthEnd(0)
        if w1 >= d_min and w0 <= d_max:
            _rect(max(w0, d_min), min(w1, d_max), "rgba(66, 153, 225, 0.08)")
