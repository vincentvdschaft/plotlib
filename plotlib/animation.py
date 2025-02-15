def smooth(t: float) -> float:
    """Smooth transition function for animations, copied from manim.

    Zero first and second derivatives at t=0 and t=1.
    Equivalent to bezier([0, 0, 0, 1, 1, 1])
    """
    s = 1 - t
    return (t**3) * (10 * s * s + 5 * s * t + t * t)
