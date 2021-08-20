"""Tests for the neural style transfer methods"""

from PIL import Image

from guertena import style_transfer

def test_style_transfer():
    data_folder = "./tests/data"
    content = Image.open(f"{data_folder}/dancing.jpg")
    style = Image.open(f"{data_folder}/picasso.jpg")
    result = style_transfer(content, style, num_steps=1)
    assert isinstance(result, Image.Image)
    assert result.size == content.size
