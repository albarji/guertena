"""Main entrypoing for running style transfer as a python library

References:
    - Pytorch Neural Style tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

from copy import deepcopy
import logging
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from neuralstyle.styletransfernet import StyleTransferNet, get_device


def style_transfer(content_img, style_img, num_steps=500, style_weight=1000000, content_weight=1,
    tv_weight=1, output_resolution=None):
    """Run the style transfer"""
    # Format input images
    logging.info(f"Loaded content image size: {np.array(content_img).shape[:2]}")
    logging.info(f"Loaded style image size: {np.array(style_img).shape[:2]}")
    if output_resolution:
        logging.info(f"Preparing for output resolution {output_resolution}")
    content_img, style_img = _format_input_images(content_img, style_img, output_resolution)
    logging.info(f"Preprocessed content image size: {tuple(content_img.shape[2:])}")
    logging.info(f"Preprocessed style image size: {tuple(content_img.shape[2:])}")

    # Initialize style network
    style_network = StyleTransferNet(content_img, style_img, content_weight, style_weight, tv_weight).to(get_device())
    # Initialize synthetic image with content image
    input_img = deepcopy(content_img).to(get_device())
    # Alternatively, initialize synthetic image with random image
    #torch.manual_seed(12345)
    #input_img = torch.rand(content_img.shape).to(get_device())

    optimizer = optim.LBFGS([input_img.requires_grad_()], tolerance_grad=0, line_search_fn="strong_wolfe")
    #optimizer = optim.Adam([input_img.requires_grad_()])

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            # correct the values of updated input image
            # input_img.data.clamp_(0, 1)  # FIXME: not sure if this is helping. Produces training divergence

            optimizer.zero_grad()
            style_network(input_img)
            style_score = 0
            content_score = 0

            for sl in style_network.style_losses:
                style_score += sl.loss
            for cl in style_network.content_losses:
                content_score += cl.loss
            tv_score = style_network.tv_loss.loss

            loss = style_score + content_score + tv_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                logging.info(f'Step: {run[0]}, Style Loss: {style_score.item():6f}, Content Loss: {content_score.item():6f}, TV Loss: {tv_score.item():6f}, Total Loss: {loss.item():6f}')

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return _format_output_image(input_img)


def _format_input_images(content, style, output_resolution=None):
    """Scale and format input PIL images for good performance of the Style Transfer method.
    
    This means:
        - Transform PIL images to pytorch tensors
        - Content image is rescaled to desired output resolution
        - Style image is rescaled to the same dimensions as the content image
    """
    # Content tranformations
    if output_resolution is not None:
        try:
            output_resolution = [int(x) for x in output_resolution.split("x")]
            assert 1 <= len(output_resolution) <= 2
            if len(output_resolution) == 1:
                output_columns = output_resolution[0]
                content_columns, content_rows, _ = np.array(content).shape
                output_rows = int(np.floor(output_columns / content_columns * content_rows))
                output_resolution = [output_columns, output_rows]
        except:
            raise ValueError("Output resolution must be in format 'COLUMNS' or 'COLUMNSxROWS' (e.g. 640 or 640x480")
        content_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(output_resolution)])
    else:
        content_transforms = transforms.ToTensor()
    # Style transformations
    style_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(output_resolution if output_resolution else content.size)
    ])
    # Apply and return
    return (
        content_transforms(content).unsqueeze(0).to(get_device()),
        style_transforms(style).unsqueeze(0).to(get_device())
    )

def _format_output_image(output):
    """Recovers the output of the style transfer as a PIL image"""
    output_transforms = transforms.ToPILImage()
    return output_transforms(output.detach().squeeze(0))
