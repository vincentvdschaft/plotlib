import pytest
from pytest import fixture
import numpy as np
from imagelib import Image


@fixture
def fixture_x_vals():
    """Return a test x-values."""
    return np.linspace(0, 2 * np.pi, 200)


@fixture
def fixture_y_vals(fixture_x_vals):
    """Return a test y-values."""
    curves = []
    for i in range(1, 5):
        curve = np.sin(fixture_x_vals * i) + np.cos(fixture_x_vals * i * 2.5)
        curves.append(curve)

    return curves


@fixture
def fixture_image():
    """Return a test image."""
    return Image.test_image()
