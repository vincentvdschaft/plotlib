import numpy as np


def linear_clipped(t, factor=0.9):
    t = (t - 0.5) / factor + 0.5

    return np.clip(t, 0, 1)


def smooth(t: float) -> float:
    """Smooth transition function for animations, copied from manim.

    Zero first and second derivatives at t=0 and t=1.
    Equivalent to bezier([0, 0, 0, 1, 1, 1])
    """
    s = 1 - t
    return (t**3) * (10 * s * s + 5 * s * t + t * t)


def map_range(t, start, end, init_start=0, init_end=1):
    return start + (end - start) * (t - init_start) / (init_end - init_start)


def smooth_range(t, start, end):
    return map_range(smooth(t), start, end)
