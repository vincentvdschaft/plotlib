import matplotlib.pyplot as plt
import numpy as np
from imagelib import Image
from scipy.interpolate import PchipInterpolator


def apply_dynamic_range_curve(setpoints: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Apply a dynamic range curve to values in [0, 1].

    Parameters
    ----------
    curve : (n,) np.ndarray
        Control points sampled uniformly over x in [0, 1]. For example,
        curve[0] is the output at x=0, curve[-1] at x=1.
        Values are expected to be in [0, 1].
    values : np.ndarray
        Array of any shape with values in [0, 1] to which the curve is applied.

    Returns
    -------
    np.ndarray
        Array of same shape as `values`, transformed by the spline and clipped to [0, 1].
    """
    if setpoints.ndim != 1:
        raise ValueError("setpoints must be a 1D array")

    setpoints = np.zeros((setpoints.size + 2,))
    setpoints[-1] = 1.0

    # x-positions for the uniformly spaced control points in [0, 1]
    x = np.linspace(0.0, 1.0, num=setpoints.size)

    print(x)

    # shape-preserving cubic spline (monotone where data are monotone)
    spline = PchipInterpolator(x, setpoints, extrapolate=True)

    # evaluate spline at the input values
    transformed = spline(values)

    # ensure the result stays in [0, 1]
    transformed = np.clip(transformed, 0.0, 1.0)

    # preserve input dtype if it is floating, otherwise return float32
    if np.issubdtype(values.dtype, np.floating):
        return transformed.astype(values.dtype, copy=False)
    return transformed.astype(np.float32, copy=False)


def view_curve(setpoints):
    vals_in = np.linspace(0, 1, 100)
    vals_out = apply_dynamic_range_curve(setpoints, vals_in)
    fig, ax = plt.subplots()
    ax.plot(vals_in, vals_out)
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_title("Dynamic Range Curve")
    plt.show()


image = np.arange(64 * 64).reshape((64, 64))
image = image / np.max(image)
setpoints = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

view_curve(setpoints)
image_out = apply_dynamic_range_curve(setpoints, image)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[1].imshow(image_out)
plt.show()
