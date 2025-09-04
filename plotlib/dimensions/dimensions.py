from .margins import Margins
from .shape import FloatShape, IntShape
from .spacing import Spacing


class DimensionsSingle:
    def __init__(self, margins: Margins, fig_size: IntShape):
        assert isinstance(margins, Margins)
        assert isinstance(fig_size, IntShape)

        self._margins = margins
        self._fig_size = fig_size

    @property
    def margins(self):
        return self._margins.copy()

    @property
    def fig_size(self):
        return self._fig_size

    @classmethod
    def from_no_height(cls, margins, fig_width, aspect):
        axis_width = fig_width - margins.width
        axis_height = axis_width * aspect
        fig_size = (fig_width, axis_height + margins.height)
        return cls(margins, fig_size)

    @classmethod
    def from_no_width(cls, margins, fig_height, aspect):
        axis_height = fig_height - margins.height
        axis_width = axis_height / aspect
        fig_size = (axis_width + margins.width, fig_height)
        return cls(margins, fig_size)


class DimensionsGrid:
    def __init__(self, margins: Margins, grid_shape, fig_size, spacing: Spacing):
        assert isinstance(margins, Margins)
        assert isinstance(spacing, Spacing)
        self._margins = margins
        self._grid_shape = grid_shape
        self._fig_size = (float(fig_size[0]), float(fig_size[1]))
        self._spacing = spacing

    @property
    def margins(self):
        return self._margins.copy()

    @property
    def grid_shape(self):
        return self._grid_shape

    @property
    def fig_size(self):
        return self._fig_size

    @classmethod
    def from_no_height(
        cls,
        margins: Margins,
        fig_width: float,
        aspect: float,
        spacing: Spacing,
        grid_shape,
    ):
        axis_width = fig_width - margins.width
        axis_height = axis_width * aspect
        fig_size = (fig_width, axis_height + margins.height)
        return cls(margins, grid_shape, fig_size, spacing)
