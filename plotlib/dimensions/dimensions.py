import numpy as np

from ..plotlib import MPLFigure
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
    def from_no_height(cls, margins, fig_width, axis_aspect):
        axis_width = fig_width - margins.width
        axis_height = axis_width * axis_aspect
        figsize = (fig_width, axis_height + margins.height)
        return cls(margins, figsize)

    @classmethod
    def from_no_width(cls, margins, fig_height, axis_aspect):
        axis_height = fig_height - margins.height
        axis_width = axis_height / axis_aspect
        figsize = (axis_width + margins.width, fig_height)
        return cls(margins, figsize)

    @classmethod
    def from_solve(
        cls,
        fig_width=None,
        fig_height=None,
        margins_left=None,
        margins_right=None,
        margins_top=None,
        margins_bottom=None,
        axis_aspect=None,
    ):
        system_matrix_rows = []
        if fig_width is not None:
            system_matrix_rows.append([1, 0, 0, 0, 0, 0, fig_width])
        if fig_height is not None:
            system_matrix_rows.append([0, 1, 0, 0, 0, 0, fig_height])
        if margins_left is not None:
            system_matrix_rows.append([0, 0, 1, 0, 0, 0, margins_left])
        if margins_right is not None:
            system_matrix_rows.append([0, 0, 0, 1, 0, 0, margins_right])
        if margins_top is not None:
            system_matrix_rows.append([0, 0, 0, 0, 1, 0, margins_top])
        if margins_bottom is not None:
            system_matrix_rows.append([0, 0, 0, 0, 0, 1, margins_bottom])
        if axis_aspect is not None:
            system_matrix_rows.append([-axis_aspect, 1, 0, 0, 0, 0, 0])

        system_matrix_rows = [np.array(row) for row in system_matrix_rows]
        system_matrix = np.vstack(system_matrix_rows)
        target_vector = system_matrix[:, -1]
        coeff_matrix = system_matrix[:, :-1]

        # Check if there is a solution
        if np.linalg.matrix_rank(coeff_matrix) < np.linalg.matrix_rank(
            np.column_stack((coeff_matrix, target_vector))
        ):
            print(
                "No solution found for the given constraints. Providing least squares solution."
            )
            solution = np.linalg.lstsq(coeff_matrix, target_vector, rcond=None)[0]
        else:
            solution = np.linalg.solve(coeff_matrix, target_vector)
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
        )

    def initialize_figure(self):
        fig = MPLFigure(figsize=self.figsize)
        ax = fig.add_ax(
            x=self.margins.left,
            y=self.margins.top,
            width=self.figsize[0] - self.margins.left - self.margins.right,
            height=self.figsize[1] - self.margins.top - self.margins.bottom,
        )
        return fig, ax


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
        # Check if there is a solution
        if np.linalg.matrix_rank(coeff_matrix) < np.linalg.matrix_rank(
            np.column_stack((coeff_matrix, target_vector))
        ):
            print(
                "No solution found for the given constraints. Providing least squares solution."
            )
        else:
            solution = np.linalg.lstsq(coeff_matrix, target_vector, rcond=None)[0]
            # solution = np.linalg.solve(coeff_matrix, target_vector)
            if np.any(solution < 0):
                print("Negative value found in solution for the given constraints.")
        print(solution)
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


class DimensionsSingleBesidesGrid:
    def __init__(
        self,
        margins: Margins,
        grid_shape: IntShape,
        figsize: FloatShape,
        grid_spacing: Spacing,
        middle_spacing: float,
        single_axis_shape: FloatShape,
    ):
        assert isinstance(margins, Margins)
        assert isinstance(grid_spacing, Spacing)
        self._margins = margins
        self._grid_shape = IntShape(grid_shape[0], grid_shape[1])
        self._figsize = FloatShape(figsize[0], figsize[1])
        self._grid_spacing = Spacing(grid_spacing[0], grid_spacing[1])
        self._single_axis_shape = FloatShape(single_axis_shape[0], single_axis_shape[1])
        self._middle_spacing = float(middle_spacing)

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
    def single_axis_shape(self):
        return self._single_axis_shape

    @property
    def middle_spacing(self):
        return self._middle_spacing

    @property
    def grid_spacing(self):
        return self._grid_spacing

    @property
    def grid_axis_shape(self):
        grid_width = (
            self._figsize.width
            - self._margins.width
            - self._single_axis_shape.width
            - self._middle_spacing
            - self._grid_spacing.horizontal * (self._grid_shape.n_cols - 1)
        ) / self._grid_shape.n_cols

        grid_height = (
            self._figsize.height
            - self._margins.height
            - self._grid_spacing.vertical * (self._grid_shape.n_rows - 1)
        ) / self._grid_shape.n_rows

        return FloatShape(grid_width, grid_height)

    @property
    def grid_total_size(self):
        grid_width = (
            self._figsize.width
            - self._margins.width
            - self._single_axis_shape.width
            - self._middle_spacing
        )
        grid_height = self._figsize.height - self._margins.height
        return FloatShape(width=grid_width, height=grid_height)

    @classmethod
    def from_no_height(
        cls,
        margins: Margins,
        fig_width: float,
        single_axis_aspect: float,
        single_axis_width: float,
        grid_axis_aspect: float,
        grid_shape: IntShape,
        grid_horizontal_spacing: float,
        middle_spacing: float,
    ):
        assert isinstance(margins, Margins)
        grid_shape = IntShape(grid_shape[0], grid_shape[1])
        grid_horizontal_spacing = float(grid_horizontal_spacing)
        single_axis_width = float(single_axis_width)
        single_axis_aspect = float(single_axis_aspect)
        fig_width = float(fig_width)
        grid_axis_aspect = float(grid_axis_aspect)
        middle_spacing = float(middle_spacing)

        single_axis_shape = FloatShape(
            single_axis_width, single_axis_width * single_axis_aspect
        )

        grid_total_width = (
            fig_width - margins.width - single_axis_width - middle_spacing
        )
        grid_total_height = single_axis_shape.height

        fig_height = single_axis_shape.height + margins.height

        # Compute the maximum width for each grid axis based on the figure width
        grid_axis_width_max = (
            grid_total_width - (grid_horizontal_spacing * (grid_shape.n_cols - 1))
        ) / grid_shape.n_cols

        grid_axis_height_max = grid_total_height / grid_shape.n_rows

        grid_axis_width = min(
            grid_axis_width_max,
            grid_axis_height_max / grid_axis_aspect,
        )

        grid_axis_shape = FloatShape(
            width=grid_axis_width,
            height=grid_axis_width * grid_axis_aspect,
        )

        grid_vertical_spacing = (
            grid_total_height - (grid_axis_shape.height * grid_shape.n_rows)
        ) / (grid_shape.n_rows - 1)

        grid_spacing = Spacing(
            horizontal=grid_horizontal_spacing, vertical=grid_vertical_spacing
        )
        figsize = FloatShape(fig_width, fig_height)

        return cls(
            margins=margins,
            grid_shape=grid_shape,
            figsize=figsize,
            single_axis_shape=single_axis_shape,
            grid_spacing=grid_spacing,
            middle_spacing=middle_spacing,
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
        single_axis_width=None,
        single_axis_height=None,
        grid_axis_width=None,
        grid_axis_height=None,
        grid_horizontal_spacing=None,
        grid_vertical_spacing=None,
        middle_spacing=None,
        single_axis_aspect=None,
        grid_axis_aspect=None,
        spacings_equal=True,
        margin_left_right_equal=False,
    ):
        # 0  fig_width
        # 1  fig_height
        # 2  margins_left
        # 3  margins_right
        # 4  margins_top
        # 5  margins_bottom
        # 6  single_axis_width
        # 7  single_axis_height
        # 8  grid_axis_width
        # 9  grid_axis_height
        # 10 grid_horizontal_spacing
        # 11 grid_vertical_spacing
        # 12 middle_spacing
        grid_shape = IntShape(grid_shape[0], grid_shape[1])

        system_matrix_rows = []
        if fig_width is not None:
            system_matrix_rows.append(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, fig_width]
            )
        if fig_height is not None:
            system_matrix_rows.append(
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, fig_height]
            )
        if margins_left is not None:
            system_matrix_rows.append(
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, margins_left]
            )
        if margins_right is not None:
            system_matrix_rows.append(
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, margins_right]
            )
        if margins_top is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, margins_top]
            )
        if margins_bottom is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, margins_bottom]
            )
        if single_axis_width is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, single_axis_width]
            )
        if single_axis_height is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, single_axis_height]
            )

        if grid_axis_width is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, grid_axis_width]
            )
        if grid_axis_height is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, grid_axis_height]
            )
        if grid_horizontal_spacing is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, grid_horizontal_spacing]
            )
        if grid_vertical_spacing is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, grid_vertical_spacing]
            )
        if middle_spacing is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, middle_spacing]
            )

        # ======================================================================
        # Ensure correct aspect ratios
        # ======================================================================
        if single_axis_aspect is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, -single_axis_aspect, 1, 0, 0, 0, 0, 0, 0]
            )
        if grid_axis_aspect is not None:
            system_matrix_rows.append(
                [0, 0, 0, 0, 0, 0, 0, 0, -grid_axis_aspect, 1, 0, 0, 0, 0]
            )

        # ======================================================================
        # Further constraints
        # ======================================================================
        if spacings_equal is not None:
            system_matrix_rows.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0])

        if margin_left_right_equal:
            system_matrix_rows.append([0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Ensure the height is equal to the single axis height + margins
        system_matrix_rows.append([0, -1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

        # Ensure the total grid height is equal to the single axis height
        system_matrix_rows.append(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -1,
                0,
                grid_shape.n_rows,
                0,
                grid_shape.n_rows - 1,
                0,
                0,
            ]
        )
        system_matrix_rows.append(
            [
                -1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                grid_shape.n_cols,
                0,
                grid_shape.n_cols - 1,
                0,
                1,
                0,
            ]
        )

        system_matrix_rows.append([0, -1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])

        system_matrix_rows = [np.array(row) for row in system_matrix_rows]
        system_matrix = np.vstack(system_matrix_rows)
        target_vector = system_matrix[:, -1]
        coeff_matrix = system_matrix[:, :-1]
        print(system_matrix)
        # Check if there is a solution
        if np.linalg.matrix_rank(coeff_matrix) < np.linalg.matrix_rank(
            np.column_stack((coeff_matrix, target_vector))
        ):
            print(
                "No solution found for the given constraints. Providing least squares solution."
            )
            solution = np.linalg.lstsq(coeff_matrix, target_vector, rcond=None)[0]
        else:
            print(coeff_matrix.shape)
            # solution = np.linalg.solve(coeff_matrix, target_vector)
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
            grid_spacing=Spacing(horizontal=solution[10], vertical=solution[11]),
            middle_spacing=solution[12],
            single_axis_shape=FloatShape(width=solution[6], height=solution[7]),
        )
