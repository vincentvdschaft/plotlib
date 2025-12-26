import numpy as np

from ..plotlib import MPLFigure
from .margins import Margins
from .shape import FloatShape, IntShape
from .spacing import Spacing


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
        axis_aspect: float,
        grid_spacing: Spacing,
        grid_shape: IntShape,
    ):
        assert isinstance(margins, Margins)
        grid_shape = IntShape(grid_shape[0], grid_shape[1])
        grid_spacing = Spacing(grid_spacing[0], grid_spacing[1])
        axis_aspect = float(axis_aspect)
        fig_width = float(fig_width)

        axis_width = (
            fig_width
            - margins.width
            - grid_spacing.horizontal * (grid_shape.n_cols - 1)
        ) / grid_shape.n_cols

        axis_height = axis_width * axis_aspect

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

    @classmethod
    def from_solve(
        cls,
        grid_shape: IntShape,
        fig_width=None,
        fig_height=None,
        margins_left=None,
        margins_right=None,
        margins_top=None,
        margins_bottom=None,
        grid_horizontal_spacing=None,
        grid_vertical_spacing=None,
        axis_aspect=None,
        spacings_equal=True,
    ):
        grid_shape = IntShape(grid_shape[0], grid_shape[1])

        # 0  fig_width
        # 1  fig_height
        # 2  margins_left
        # 3  margins_right
        # 4  margins_top
        # 5  margins_bottom
        # 6  grid_horizontal_spacing
        # 7  grid_vertical_spacing
        # 8  axis_aspect

        row_fig_width = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
        row_fig_height = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])
        row_margins_left = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
        row_margins_right = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])
        row_margins_top = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
        row_margins_bottom = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
        row_grid_horizontal_spacing = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])
        row_grid_vertical_spacing = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])
        row_target = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

        system_matrix_rows = []
        if fig_width is not None:
            new_row = row_fig_width + row_target * fig_width
            system_matrix_rows.append(new_row)

        if fig_height is not None:
            new_row = row_fig_height + row_target * fig_height
            system_matrix_rows.append(new_row)

        if margins_left is not None:
            new_row = row_margins_left + row_target * margins_left
            system_matrix_rows.append(new_row)

        if margins_right is not None:
            new_row = row_margins_right + row_target * margins_right
            system_matrix_rows.append(new_row)

        if margins_top is not None:
            new_row = row_margins_top + row_target * margins_top
            system_matrix_rows.append(new_row)

        if margins_bottom is not None:
            new_row = row_margins_bottom + row_target * margins_bottom
            system_matrix_rows.append(new_row)

        if grid_horizontal_spacing is not None:
            new_row = row_grid_horizontal_spacing + row_target * grid_horizontal_spacing
            system_matrix_rows.append(new_row)

        if grid_vertical_spacing is not None:
            new_row = row_grid_vertical_spacing + row_target * grid_vertical_spacing
            system_matrix_rows.append(new_row)

        if axis_aspect is not None:
            print((grid_shape.n_cols - 1) * row_grid_horizontal_spacing)
            axis_width = (
                row_fig_width
                - row_margins_left
                - row_margins_right
                - (grid_shape.n_cols - 1) * row_grid_horizontal_spacing
            ) / grid_shape.n_cols
            axis_height = (
                row_fig_height
                - row_margins_top
                - row_margins_bottom
                - (grid_shape.n_rows - 1) * row_grid_vertical_spacing
            ) / grid_shape.n_rows
            new_row = -axis_width * axis_aspect + axis_height
            system_matrix_rows.append(new_row)

        if spacings_equal:
            new_row = row_grid_horizontal_spacing - row_grid_vertical_spacing
            system_matrix_rows.append(new_row)

        system_matrix_rows = [np.array(row) for row in system_matrix_rows]
        system_matrix = np.vstack(system_matrix_rows)
        target_vector = system_matrix[:, -1]
        coeff_matrix = system_matrix[:, :-1]

        rank = np.linalg.matrix_rank(coeff_matrix)
        if rank < coeff_matrix.shape[1]:
            print("Underdetermined system: Consider providing more constraints.")
        elif rank > coeff_matrix.shape[1]:
            print(
                "Overdetermined system: No exact solution found for the given "
                "constraints. Providing least squares solution."
            )

        solution = np.linalg.lstsq(coeff_matrix, target_vector, rcond=None)[0]
        if np.any(solution < 0):
            print("Negative value found in solution for the given constraints.")
        return cls(
            margins=Margins(
                left=solution[2],
                right=solution[3],
                top=solution[4],
                bottom=solution[5],
            ),
            figsize=FloatShape(width=solution[0], height=solution[1]),
            grid_shape=grid_shape,
            grid_spacing=Spacing(horizontal=solution[6], vertical=solution[7]),
        )

    def initialize_figure(self):
        fig = MPLFigure(figsize=self.figsize)
        axes = fig.add_axes_grid(
            n_rows=self.grid_shape.n_rows,
            n_cols=self.grid_shape.n_cols,
            x=self.margins.left,
            y=self.margins.top,
            width=(
                self.figsize[0]
                - self.margins.left
                - self.margins.right
                - self.grid_spacing.horizontal * (self.grid_shape.n_cols - 1)
            )
            / self.grid_shape.n_cols,
            height=(
                self.figsize[1]
                - self.margins.top
                - self.margins.bottom
                - self.grid_spacing.vertical * (self.grid_shape.n_rows - 1)
            )
            / self.grid_shape.n_rows,
            spacing=self.grid_spacing,
        )
        return fig, axes

    def __repr__(self):
        return (
            f"DimensionsGrid(margins={self.margins}, figsize={self.figsize}, "
            f"grid_shape={self.grid_shape}, grid_spacing={self.grid_spacing})"
        )
