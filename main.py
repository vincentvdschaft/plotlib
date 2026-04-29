import matplotlib.pyplot as plt

from plotlib import DimensionsSingle, quickfig_single
from plotlib.ruler import add_ruler

fig, ax = quickfig_single(
    DimensionsSingle.from_solve(
        fig_width=12,
        axis_aspect=1.0,
        margins_left=0.5,
        margins_right=0.5,
        margins_top=0.5,
        margins_bottom=0.5,
    )
)
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)

add_ruler(
    ax=ax,
    start=(5.5, 5.5),
    end=(11.5, 0.5),
    formatter=lambda d: f"{d:.1f} mm",
    color="red",
    linewidth=4.0,
    label_side="above",
    label_offset=0.4,
    fontsize=12,
)
plt.show()
