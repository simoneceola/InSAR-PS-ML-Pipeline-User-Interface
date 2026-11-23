"""
Section registry.

Maps the section keys used in page_layout.SECTIONS to their render functions.
To add a new section: create sections/my_section.py with `render(ctx)`,
register it below and add its key to page_layout.SECTIONS.
"""

from dataclasses import dataclass
from typing import Callable

from app_context import AppContext
from page_layout import SECTIONS
from . import (
    all_timeseries,
    clustered_timeseries,
    diagnosis,
    height_strata,
    kpis,
    scatterer_3d,
    single_point,
)


@dataclass(frozen=True)
class Section:
    render: Callable[[AppContext], None]
    divider: bool = True  # draw a horizontal rule before the section


REGISTRY: dict[str, Section] = {
    "diagnosis": Section(diagnosis.render),
    "kpis": Section(kpis.render, divider=False),
    "scatterer_3d": Section(scatterer_3d.render),
    "all_timeseries": Section(all_timeseries.render),
    "clustered_timeseries": Section(clustered_timeseries.render),
    "height_strata": Section(height_strata.render),
    "single_point": Section(single_point.render),
}


def get_active_sections() -> list[tuple[str, Section]]:
    """Enabled sections from page_layout.SECTIONS, in the order listed there."""
    active = []
    for key, enabled in SECTIONS:
        if key not in REGISTRY:
            raise KeyError(f"page_layout.SECTIONS has unknown section '{key}'. Valid keys: {sorted(REGISTRY)}")
        if enabled:
            active.append((key, REGISTRY[key]))
    return active
