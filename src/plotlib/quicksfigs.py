import warnings
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from plotlib import (
    DimensionsGrid,
    DimensionsSingle,
    DimensionsSingleBesidesGrid,
)
from plotlib.plotlib import MPLFigure

AxesArray: TypeAlias = npt.NDArray[np.object_]


def _warn_deprecated(function_name: str) -> None:
    warnings.warn(
        f"{function_name} is deprecated, use dimensions.initialize_figure() instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def quickfig_single(dimensions: DimensionsSingle) -> tuple[MPLFigure, Axes]:
    """Create a quick figure with single dimensions.

    Deprecated: use `DimensionsSingle.initialize_figure()` instead.

    Args:
        dimensions: The single dimensions to use.

    Returns:
        The created figure and axis.
    """
    _warn_deprecated("quickfig_single")
    return dimensions.initialize_figure()


def quickfig_grid(dimensions: DimensionsGrid) -> tuple[MPLFigure, AxesArray]:
    """Create a quick figure with grid dimensions.

    Deprecated: use `DimensionsGrid.initialize_figure()` instead.

    Args:
        dimensions: The grid dimensions to use.

    Returns:
        The created figure and axes.
    """
    _warn_deprecated("quickfig_grid")
    return dimensions.initialize_figure()


def quickfig_single_besides_grid(
    dimensions: DimensionsSingleBesidesGrid, grid_on_right: bool = True
) -> tuple[MPLFigure, Axes, AxesArray]:
    """Create a quick figure with single beside grid dimensions.

    Deprecated: use `DimensionsSingleBesidesGrid.initialize_figure()` instead.

    Args:
        dimensions: The single beside grid dimensions to use.
        grid_on_right: Place the grid right of the single axis if True.

    Returns:
        The created figure, the single axis and the grid axes.
    """
    _warn_deprecated("quickfig_single_besides_grid")
    return dimensions.initialize_figure(grid_on_right)
