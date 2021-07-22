"""Tests for the neural style transfer methods"""

from PIL import Image

from neuralstyle.driver import run_style_transfer

def test_run_style_transfer():
    data_folder = "./tests/data"
    content = Image.open(f"{data_folder}/dancing.jpg")
    style = Image.open(f"{data_folder}/picasso.jpg")
    result = run_style_transfer(content, style, num_steps=500)
    result.save("output.png")
