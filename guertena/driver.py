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
from tqdm import tqdm

from .styletransfernet import StyleTransferLoss, UnetGenerator, get_device

VGG19_RECEPTIVE_FIELD_SIZE = 212  # https://distill.pub/2019/computing-receptive-fields/


def _style_transfer_pass(content_img, style_img, num_steps=1000, style_weight=1e6, content_weight=1,
    tv_weight=1, valid_pixels_weight=1, output_resolution=None, init_img=None, content_layers=None,
    style_layers=None, generator="raw", downscaling_losses=True, verbosity=0):
    """Runs a style transfer pass using Gatys method
    
    Arguments:
        content_img: original content image on which to apply style transfer. Image must be in PIL format.
        style_img: image with the style to transfer. Image must be in PIL format.
        num_steps: number of optimizer iterations to run
        style_weight: weight of the style loss in the optimization
        content_weight: weight of the content loss in the optimization
        tv_weight: weight of the Total Variation loss in the optimization
        valid_pixels_weight: weight of the Valid Pixels loss in the optimization
        output_resolution: desired resolution of the resultant image.
            Must be in format 'COLUMNS' or 'COLUMNSxROWS' (e.g. 640 or 640x480)
        init_img: image to initialize the optimization procedure. Must be a PIL image. If None, use content image.
        content_layers: iterable of VGG19 layers names into which to impose content losses. If None, a default choice is used.
        style_layers: iterable of VGG19 layers names into which to impose content losses. If None, a default choice is used.
        generator: generator network, among "raw" or "u-net"
        downscaling_losses: whether to use auxiliary downscaling losses
        verbosity: level of verbosity during style transfer, from 0 to 2.
    
    Returns a PIL image with the result of the style transfer.
    """
    # Format input images
    logging.debug(f"Starting style transfer pass...")
    logging.debug(f"Received content image size: {np.array(content_img).shape[:2]}")
    logging.debug(f"Received style image size: {np.array(style_img).shape[:2]}")
    if init_img is not None:
        logging.debug(f"Received initialization image size: {np.array(init_img).shape[:2]}")
    if output_resolution:
        logging.debug(f"Preparing for output resolution {output_resolution}")
    content_img, style_img, init_img = _format_input_images(
        content_img,
        style_img,
        init_img,
        output_resolution
    )
    logging.debug(f"Preprocessed content image size: {tuple(content_img.shape[2:])}")
    logging.debug(f"Preprocessed style image size: {tuple(content_img.shape[2:])}")
    if init_img is not None:
        logging.debug(f"Preprocessed initialization image size: {tuple(init_img.shape[2:])}")
    # Compute number of possible downscales
    possible_downscales = int(np.ceil(min(
        np.log2(content_img.shape[2]/VGG19_RECEPTIVE_FIELD_SIZE), 
        np.log2(content_img.shape[3]/VGG19_RECEPTIVE_FIELD_SIZE)
    )))
    logging.debug(f"Number of possible image downscales: {possible_downscales}")

    # Initialize generator network
    generator_network = UnetGenerator(3, 3, 7).to(get_device()) if generator == "u-net" else None  # TODO: adapt downscales to image size
    if generator_network is not None:
        logging.debug(f"Generator network created by the following architecture:\n{generator_network.model}")

    # Initialize style loss network
    loss_network = StyleTransferLoss(
        content_img,
        style_img,
        content_weight,
        style_weight,
        tv_weight,
        valid_pixels_weight,
        content_layers,
        style_layers,
        normalize_input=True
    ).to(get_device())
    logging.debug(f"Loss network created by the following architecture:\n{loss_network.model}")

    # Initialize auxiliary losses at reduced resolutions
    if downscaling_losses:
        reduced_content_img = content_img
        reduced_style_img = style_img
        weight_scale = 1.0
        auxiliary_losses = []
        for _ in range(possible_downscales):
            reduced_content_img = torch.nn.functional.avg_pool2d(reduced_content_img, 2)
            reduced_style_img = torch.nn.functional.avg_pool2d(reduced_style_img, 2)
            weight_scale /= 2.0
            auxiliary_losses.append(
                StyleTransferLoss(
                    reduced_content_img,
                    reduced_style_img,
                    content_weight * weight_scale,
                    style_weight * weight_scale,
                    tv_weight=0,
                    valid_pixels_weight=0,
                    content_layers=content_layers,
                    style_layers=style_layers,
                    normalize_input=True
                ).to(get_device())
            )

    # Initialize pixels of synthetic image if no generator is to be used
    if generator_network is None:
        raw_generated_image = deepcopy(init_img if init_img is not None else content_img).to(get_device())
    # Else prepare generator inputs
    else:
        generator_inputs = (init_img if init_img else content_img).to(get_device())

    # Prepare optimizer
    params = [raw_generated_image.requires_grad_()] if generator_network is None else generator_network.parameters()
    optimizer = optim.LBFGS(params, tolerance_grad=0, line_search_fn="strong_wolfe", history_size=1)

    logging.debug('Optimizing...')
    pbar = tqdm(total=num_steps, disable=verbosity==0)
    run = [0]
    generated_image = [None]  # TODO: this is extremely ugly. Look for another way to do this
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            if generator_network:
                generated_image[0] = generator_network(generator_inputs)
            else:
                generated_image[0] = raw_generated_image
            
            # Apply main loss
            loss_network(generated_image[0])
            style_score = 0
            for sl in loss_network.style_losses:
                style_score += sl.loss  # Note in Gatys the different styles losses are averaged, not summed, but this follows jcjohnson implementation
            content_score = 0
            for cl in loss_network.content_losses:
                content_score += cl.loss
            tv_score = loss_network.tv_loss.loss
            valid_pixels_score = loss_network.valid_pixels_loss.loss

            # Apply loss network at different image scales, until minimal size accepted by VGG19 (32x32 pixels)
            if downscaling_losses:
                current_image = generated_image[0]
                for i in range(possible_downscales):
                    current_image = torch.nn.functional.avg_pool2d(current_image, 2)
                    auxiliary_losses[i](current_image)
                    for sl in auxiliary_losses[i].style_losses:
                        style_score += sl.loss  # Note in Gatys the different styles losses are averaged, not summed, but this follows jcjohnson implementation
                    for cl in auxiliary_losses[i].content_losses:
                        content_score += cl.loss

            loss = style_score + content_score + tv_score + valid_pixels_score
            loss.backward()

            run[0] += 1
            pbar.update(1)
            if run[0] % 50 == 0:
                logging.debug(f'Step: {run[0]}, Style Loss: {style_score.item():6f}, Content Loss: {content_score.item():6f}, TV Loss: {tv_score.item():6f}, Valid Pixels Loss: {valid_pixels_score.item():6f}, Total Loss: {loss.item():6f}')

            return loss

        optimizer.step(closure)
    pbar.close()

    # a last correction...
    generated_image[0].data.clamp_(0, 1)

    return _format_output_image(generated_image[0])


def style_transfer(content_img, style_img, num_steps=1000, style_weight=1000000, content_weight=1,
    tv_weight=1, valid_pixels_weight=1, output_resolution=None, content_layers=None, style_layers=None, upscaling_rounds=1,
    upscales_per_round=7, starting_resolution=256, generator="raw", verbosity=0):
    """Transfers the style from one image to another.
    
    Arguments:
        content_img: original content image on which to apply style transfer. Image must be in PIL format.
        style_img: image with the style to transfer. Image must be in PIL format.
        num_steps: number of optimizer iterations to run
        style_weight: weight of the style loss in the optimization
        content_weight: weight of the content loss in the optimization
        tv_weight: weight of the Total Variation loss in the optimization
        valid_pixels_weight: weight of the Valid Pixels loss in the optimization
        output_resolution: desired resolution of the resultant image.
            Must be in format 'COLUMNS' or 'COLUMNSxROWS' (e.g. 640 or 640x480)
        init_img: image to initialize the optimization procedure. Must be a PIL image. If None, use content image.
        content_layers: iterable of VGG19 layers names into which to impose content losses. If None, a default choice is used.
        style_layers: iterable of VGG19 layers names into which to impose content losses. If None, a default choice is used.
        upscaling_rounds: number of upscaling rounds to perform. If 0 or None, the transfer is directly performed at
            target resolution, which is faster but yields poorer quality.
        upscales_per_round: number of upscaling steps to perform in each upscaling round.
        starting_resolution: resolution at which to start upscaling steps.
        generator: generator network, among "raw" or "u-net"
        verbosity: level of verbosity during style transfer, from 0 to 2
    
    Returns a PIL image with the result of the style transfer.
    """
    # Set verbosity
    assert 0 <= verbosity <= 2, ValueError("Verbosity must be an int between 0 and 2")
    logging.getLogger().setLevel([logging.ERROR, logging.INFO, logging.DEBUG][verbosity])

    # If 0 or None rounds, just performe a simple pass
    if upscaling_rounds is None or upscaling_rounds == 0:
        return _style_transfer_pass(
            content_img, 
            style_img, 
            num_steps, 
            style_weight, 
            content_weight, 
            tv_weight,
            valid_pixels_weight,
            output_resolution=output_resolution,
            content_layers=content_layers,
            style_layers=style_layers,
            generator=generator,
            verbosity=verbosity
        )

    # Check arguments
    output_shape = _get_target_shape(content_img, output_resolution)
    if starting_resolution > output_shape[0]:
        logging.warning(f"Output resolution {output_shape[0]} is smaller than starting resolution {starting_resolution}, adjusting staring to half of output")
        starting_resolution = int(np.floor(output_shape[0] / 2))

    # Multiresolution rounds
    seed = None
    for round in range(upscaling_rounds):
        logging.info(f"Upscaling strategy round {round}")
        iterations = num_steps
        resolutions = np.linspace(starting_resolution, output_shape[0], upscales_per_round, dtype=int)
        for stepnumber, res in enumerate(resolutions):
            logging.info(f"Upscaling round {round} step {stepnumber}: upscaling to resolution {res}")
            seed = _style_transfer_pass(
                content_img, 
                style_img, 
                iterations, 
                style_weight, 
                content_weight, 
                tv_weight,
                output_resolution=res,
                init_img=seed,
                content_layers=content_layers,
                style_layers=style_layers,
                generator=generator,
                verbosity=verbosity
            )
            # iterations = max(iterations / 2, 100)

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
        output_shape = np.array(content_img).shape[0:2]
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
