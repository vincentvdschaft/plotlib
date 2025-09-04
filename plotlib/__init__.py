from imagelib import Extent

from .animation import *
from .boxconnection import *
from .constants import IEEE_COLUMN_WIDTH, IEEE_DOUBLE_COLUMN_WIDTH
from .dimensions import (
    DimensionsGrid,
    DimensionsSingle,
    DimensionsSingleBesidesGrid,
    FloatShape,
    IntShape,
    Margins,
    Spacing,
)
from .plotlib import (
    MPLFigure,
    interpret_width_height_aspect,
    mm_formatter_ax,
    mmplot,
    remove_axes,
    remove_internal_labels,
    remove_internal_last_ticks_grid,
    remove_internal_ticks,
    remove_internal_ticks_labels,
    remove_internal_titles,
    remove_ticks_labels,
)
from .quicksfigs import quickfig_grid, quickfig_single, quickfig_single_besides_grid
from .styles import (
    ALLOWED_STYLES,
    STYLE_DARK,
    STYLE_LIGHT,
    STYLE_NAMES,
    STYLE_PAPER,
    STYLE_POSTER,
    use_style,
)

__all__ = [
    "MPLFigure",
    "Margins",
    "DimensionsSingle",
    "DimensionsGrid",
    "DimensionsSingleBesidesGrid",
    "quickfig_single",
    "quickfig_grid",
    "quickfig_single_besides_grid",
    "FloatShape",
    "IntShape",
    "Spacing",
    "IEEE_COLUMN_WIDTH",
    "IEEE_DOUBLE_COLUMN_WIDTH",
    "STYLE_LIGHT",
    "STYLE_DARK",
    "STYLE_PAPER",
    "STYLE_POSTER",
    "ALLOWED_STYLES",
    "STYLE_NAMES",
    "use_style",
    "Extent",
    "remove_axes",
    "remove_internal_labels",
    "remove_internal_last_ticks_grid",
    "remove_internal_ticks",
    "remove_internal_ticks_labels",
    "remove_internal_titles",
    "remove_ticks_labels",
    "interpret_width_height_aspect",
    "mm_formatter_ax",
    "mmplot",
]
