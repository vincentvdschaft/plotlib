from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow, Circle, ConnectionPatch, Rectangle
from matplotlib.transforms import Bbox

from plotlib import *


def _get_tempdir():
    tempdir = Path(__file__).parent / "temp"
    tempdir.mkdir(exist_ok=True)
    return tempdir


def _create_example_lineplot(title="Example plot"):
    fig = MPLFigure()
    ax = fig.add_ax(0, 0, 4, 2)

    t = np.linspace(0, 2 * np.pi, 200)
    for i in range(1, 5):
        ax.plot(t, np.sin(t * i) + np.cos(t * i * 2.5), label=f"Line {i}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(title)
    ax.legend(loc="upper right")

    return fig


def test_example_lineplots():

    for style in ALLOWED_STYLES:
        use_style(style)
        fig = _create_example_lineplot(f"Style: {STYLE_NAMES[style]}")
        fig.savefig(
            _get_tempdir() / f"style_{STYLE_NAMES[style]}.png",
            bbox_inches="tight",
            dpi=600,
        )


if __name__ == "__main__":
    test_example_lineplots()
