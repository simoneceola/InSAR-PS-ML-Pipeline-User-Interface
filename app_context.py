"""Everything a page section needs, bundled in one object passed to `section.render(ctx)`."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app_config import AnalysisConfig
from ui.point_selector import render_point_selector
from ui.sidebar import Settings


@dataclass
class AppContext:
    building_datasets: dict[int, dict]
    intra_df: pd.DataFrame
    filtered_df: pd.DataFrame
    dates_dt: pd.DatetimeIndex
    temp_c: np.ndarray
    temp_is_measured: bool
    period: int
    cfg: AnalysisConfig
    settings: Settings
    selected_id: int
    res: dict[str, Any]  # full single-building analysis result
    _selected_pt: int | None = field(default=None, init=False, repr=False)

    @property
    def building_data(self) -> dict:
        return self.building_datasets[self.selected_id]

    def selected_point(self) -> int:
        """Render the scatterer dropdown once (at the first section that needs it) and reuse it."""
        if self._selected_pt is None:
            self._selected_pt = render_point_selector(self.res)
        return self._selected_pt
