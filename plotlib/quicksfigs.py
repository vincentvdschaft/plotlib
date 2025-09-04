from plotlib.dimensions import DimensionsGrid, DimensionsSingle
from plotlib.plotlib import MPLFigure


def quickfig_single(dimensions: DimensionsSingle) -> MPLFigure:
    """Create a quick figure with single dimensions.

    Args:
        dimensions (DimensionsSingle): The single dimensions to use.

    Returns:
        MPLFigure: The created figure.
    """
    assert isinstance(dimensions, DimensionsSingle)

    fig = MPLFigure(figsize=dimensions.figsize)
    ax = fig.add_ax(
        x=dimensions.margins.left,
        y=dimensions.margins.top,
        width=dimensions.axis_size.width,
        height=dimensions.axis_size.height,
    )
    return fig, ax


def quickfig_grid(dimensions: DimensionsGrid) -> MPLFigure:
    """Create a quick figure with grid dimensions.

    Args:
        dimensions_grid (DimensionsGrid): The grid dimensions to use.

    Returns:
        MPLFigure: The created figure.
    """
    assert isinstance(dimensions, DimensionsGrid)

    fig = MPLFigure(figsize=dimensions.figsize)

    axes = fig.add_axes_grid(
        n_rows=dimensions.grid_shape.n_rows,
        n_cols=dimensions.grid_shape.n_cols,
        x=dimensions.margins.left,
        y=dimensions.margins.top,
        width=dimensions.axis_size.width,
        height=dimensions.axis_size.height,
        spacing=dimensions.grid_spacing,
    )
    return fig, axes
