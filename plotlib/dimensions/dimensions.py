from .margins import Margins
from .shape import FloatShape, IntShape
from .spacing import Spacing


class DimensionsSingle:
    def __init__(self, margins: Margins, figsize: FloatShape):
        assert isinstance(margins, Margins)

        self._margins = margins
        self._figsize = FloatShape(figsize[0], figsize[1])

    @property
    def margins(self):
        return self._margins.copy()

    @property
    def figsize(self):
        return self._figsize

    @property
    def axis_size(self):
        axis_width = self._figsize[0] - self._margins.width
        axis_height = self._figsize[1] - self._margins.height
        return FloatShape(axis_width, axis_height)

    @classmethod
    def from_no_height(cls, margins, fig_width, aspect):
        axis_width = fig_width - margins.width
        axis_height = axis_width * aspect
        figsize = (fig_width, axis_height + margins.height)
        return cls(margins, figsize)

    @classmethod
    def from_no_width(cls, margins, fig_height, aspect):
        axis_height = fig_height - margins.height
        axis_width = axis_height / aspect
        figsize = (axis_width + margins.width, fig_height)
        return cls(margins, figsize)


class DimensionsGrid:
    def __init__(
        self,
        margins: Margins,
        grid_shape: IntShape,
        figsize: FloatShape,
        grid_spacing: Spacing,
    ):
        assert isinstance(margins, Margins)
        assert isinstance(grid_spacing, Spacing)
        self._margins = margins
        self._grid_shape = grid_shape
        self._figsize = FloatShape(figsize[0], figsize[1])
        self._grid_spacing = Spacing(grid_spacing[0], grid_spacing[1])

    @property
    def margins(self):
        return self._margins.copy()

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def figsize(self):
        return self._figsize

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @property
    def axis_size(self):
        axis_width = (
            self._figsize[0]
            - self._margins.width
            - self._grid_spacing.horizontal * (self._grid_shape.n_cols - 1)
        ) / self._grid_shape.n_cols

        axis_height = (
            self._figsize[1]
            - self._margins.height
            - self._grid_spacing.vertical * (self._grid_shape.n_rows - 1)
        ) / self._grid_shape.n_rows

        return FloatShape(axis_width, axis_height)

    @classmethod
    def from_no_height(
        cls,
        margins: Margins,
        fig_width: float,
        aspect: float,
        grid_spacing: Spacing,
        grid_shape: IntShape,
    ):
        assert isinstance(margins, Margins)
        grid_shape = IntShape(grid_shape[0], grid_shape[1])
        grid_spacing = Spacing(grid_spacing[0], grid_spacing[1])
        aspect = float(aspect)
        fig_width = float(fig_width)

        axis_width = (
            fig_width
            - margins.width
            - grid_spacing.horizontal * (grid_shape.n_cols - 1)
        )

        axis_height = axis_width * aspect

        fig_height = (
            axis_height * grid_shape.n_rows
            + margins.height
            + grid_spacing.vertical * (grid_shape.n_rows - 1)
        )
        figsize = FloatShape(fig_width, fig_height)

        return cls(
            margins=margins,
            grid_shape=grid_shape,
            figsize=figsize,
            grid_spacing=grid_spacing,
        )
