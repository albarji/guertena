"""Main entrypoing for running style transfer as a python library

References:
    - Pytorch Neural Style tutorial: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

from copy import deepcopy
import logging
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms

from neuralstyle.styletransfernet import StyleTransferNet, get_device


def style_transfer(content_img, style_img, num_steps=500, style_weight=1000000, content_weight=1,
    tv_weight=1, output_resolution=None, init_img=None):
    """Run the style transfer"""
    # Format input images
    logging.info(f"Received content image size: {np.array(content_img).shape[:2]}")
    logging.info(f"Received style image size: {np.array(style_img).shape[:2]}")
    if init_img is not None:
        logging.info(f"Received initialization image size: {np.array(init_img).shape[:2]}")
    if output_resolution:
        logging.info(f"Preparing for output resolution {output_resolution}")
    content_img, style_img, init_img = _format_input_images(
        content_img,
        style_img,
        init_img,
        output_resolution
    )
    logging.info(f"Preprocessed content image size: {tuple(content_img.shape[2:])}")
    logging.info(f"Preprocessed style image size: {tuple(content_img.shape[2:])}")
    if init_img is not None:
        logging.info(f"Preprocessed initialization image size: {tuple(init_img.shape[2:])}")

    # Initialize style network
    style_network = StyleTransferNet(content_img, style_img, content_weight, style_weight, tv_weight).to(get_device())

    # Initialize synthetic image
    input_img = deepcopy(init_img if init_img is not None else content_img).to(get_device())

    # Prepare optimizer
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


def style_transfer_multiresolution(content_img, style_img, num_steps=1000, style_weight=1000000, content_weight=1,
    tv_weight=0, output_resolution=None, num_rounds=5, upscales_per_round=7, starting_resolution=256):
    """Runs a multiresolution version of style transfer, much lower but of better quality"""
    logging.info("Starting multiresolution strategy")
    # Check arguments
    output_shape = _get_target_shape(content_img, output_resolution)
    if starting_resolution > output_shape[0]:
        logging.warning(f"Output resolution {output_shape[0]} is smaller than starting resolution {starting_resolution}, adjusting staring to half of output")
        starting_resolution = int(np.floor(output_shape[0] / 2))

    # Multiresolution rounds
    seed = None
    for round in range(num_rounds):
        logging.info(f"Multiresolution strategy round {round}")
        resolutions = np.linspace(starting_resolution, output_shape[0], upscales_per_round, dtype=int)
        iters = num_steps
        for stepnumber, res in enumerate(resolutions):
            logging.info(f"Multiresolution round {round} step {stepnumber}: upscaling to resolution {res}")
            seed = style_transfer(
                content_img, 
                style_img, 
                iters, 
                style_weight, 
                content_weight, 
                tv_weight,
                output_resolution=res,
                init_img=seed
            )
            iters = max(iters / 2, 100)

    return seed


def _get_target_shape(content_img, output_resolution=None):
    """Returns the shape that the resultant image will take for given content and desired output resolution"""
    if output_resolution is not None:
        try:
            if isinstance(output_resolution, np.integer):
                output_shape = [output_resolution]
            else:
                output_shape = [int(x) for x in output_resolution.split("x")]
            assert 1 <= len(output_shape) <= 2
            if len(output_shape) == 1:
                output_columns = output_shape[0]
                content_columns, content_rows, _ = np.array(content_img).shape
                output_rows = int(np.floor(output_columns / content_columns * content_rows))
                output_shape = [output_columns, output_rows]
        except:
            raise ValueError("Output resolution must be in format 'COLUMNS' or 'COLUMNSxROWS' (e.g. 640 or 640x480")
    else:
        output_shape = np.array(content_img).size
    return output_shape


def _format_input_images(content, style, init=None, output_resolution=None):
    """Scale and format input PIL images for correct performance of the Style Transfer method.
    
    This means:
        - Transform PIL images to pytorch tensors
        - Content image is rescaled to desired output resolution
        - Style image is rescaled to the same dimensions as the content image
        - Initialization image (if provided) is rescaled to the same dimensions as the content image
    """
    # Content tranformations
    output_shape = _get_target_shape(content, output_resolution)
    if output_shape != content.size:
        content_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(output_shape)])
    else:
        content_transforms = transforms.ToTensor()
    # Style transformations
    style_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(output_shape)])
    # Initialization image transformations (if provided)
    if init is not None:
        init_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(output_shape)])
    # Apply and return
    return (
        content_transforms(content).unsqueeze(0).to(get_device()),
        style_transforms(style).unsqueeze(0).to(get_device()),
        init_transforms(init).unsqueeze(0).to(get_device()) if init is not None else None
    )

def _format_output_image(output):
    """Recovers the output of the style transfer as a PIL image"""
    output_transforms = transforms.ToPILImage()
    return output_transforms(output.detach().squeeze(0))
