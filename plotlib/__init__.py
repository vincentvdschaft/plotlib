from imagelib import Extent

from .animation import *
from .boxconnection import *
from .constants import IEEE_COLUMN_WIDTH, IEEE_DOUBLE_COLUMN_WIDTH
from .dimensions import (
    DimensionsGrid,
    DimensionsSingle,
    FloatShape,
    IntShape,
    Margins,
    Spacing,
)
from .plotlib import (
    MPLFigure,
    remove_axes,
    remove_internal_labels,
    remove_internal_last_ticks_grid,
    remove_internal_ticks,
    remove_internal_ticks_labels,
    remove_internal_titles,
    remove_ticks_labels,
)
from .quicksfigs import quickfig_grid, quickfig_single
from .styles import STYLE_DARK, STYLE_LIGHT, STYLE_PAPER, use_style

__all__ = [
    "MPLFigure",
    "Margins",
    "DimensionsSingle",
    "quickfig_single",
    "DimensionsGrid",
    "quickfig_grid",
    "FloatShape",
    "IntShape",
    "Spacing",
    "IEEE_COLUMN_WIDTH",
    "IEEE_DOUBLE_COLUMN_WIDTH",
    "STYLE_LIGHT",
    "STYLE_DARK",
    "STYLE_PAPER",
    "use_style",
    "Extent",
    "remove_axes",
    "remove_internal_labels",
    "remove_internal_last_ticks_grid",
    "remove_internal_ticks",
    "remove_internal_ticks_labels",
    "remove_internal_titles",
    "remove_ticks_labels",
]
