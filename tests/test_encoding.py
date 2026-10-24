from io import BytesIO
import warnings

import numpy as np
from PIL import Image
import pytest

from face_search.encoding import load_image


@pytest.mark.parametrize("transparency", [bytes([0, 128, 255]), 0, None])
def test_palette_png_keeps_rgb_values_without_transparency_warning(transparency):
    image = Image.new("P", (3, 1))
    colors = [(240, 20, 40), (30, 210, 60), (70, 80, 220)]
    image.putpalette([channel for color in colors for channel in color] + [0] * (768 - 9))
    image.putdata([0, 1, 2])
    if transparency is not None:
        image.info["transparency"] = transparency
    stream = BytesIO()
    image.save(stream, format="PNG")

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        actual = load_image(stream.getvalue())

    # Retain the existing RGB preprocessing, including colors beneath transparency.
    # A new matte/background would change embeddings for already indexed images.
    np.testing.assert_array_equal(actual, np.array([colors], dtype=np.uint8))
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_truecolor_loading_retains_existing_rgb_preprocessing(mode):
    color = (25, 50, 75) if mode == "RGB" else (25, 50, 75, 128)
    image = Image.new(mode, (2, 2), color)
    stream = BytesIO()
    image.save(stream, format="PNG")
    np.testing.assert_array_equal(load_image(stream.getvalue()), np.full((2, 2, 3), color[:3], dtype=np.uint8))
