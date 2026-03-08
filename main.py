import matplotlib.pyplot as plt

import plotlib

dims = plotlib.DimensionsSingle.from_solve(
    fig_width=5,
    margins_left=0.5,
    margins_right=0.5,
    margins_top=0.5,
    axis_aspect=(0, 40e-3, 0, 20e-3),
    margins_bottom=0.5,
)

fig, ax = plotlib.quickfig_single(dims)

plt.show()
