"""Main script to run style transfer"""

import argparse
import logging
from PIL import Image, ImageOps

from guertena import style_transfer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neural style transfer')
    parser.add_argument('content', type=str, help='Path to content image')
    parser.add_argument('style', type=str, help='Path to style image')
    parser.add_argument('output', type=str, help='Path to output image')
    parser.add_argument('--output_resolution', type=str, default=None,
                        help='Resolution of output image, in format ROWSxCOLUMNS')
    parser.add_argument('--content_weight', type=float, default=1,
                        help='Weight of the content loss')
    parser.add_argument('--style_weight', type=float, default=1e6,
                        help='Weight of the style loss')
    parser.add_argument('--tv_weight', type=float, default=1,
                        help='Weight of the total variation loss')
    parser.add_argument('--content_layers', type=str, default=None,
                        help='Comma-separated string with the names of VGG-19 layers where to inject content losses')
    parser.add_argument('--style_layers', type=str, default=None,
                        help='Comma-separated string with the names of VGG-19 layers where to inject style losses')
    parser.add_argument('--num_steps', type=int, default=1000,
                        help='Number of gradient descent iterations')
    parser.add_argument('--upscaling_rounds', type=int, default=1,
                        help='Number of resolution upscaling rounds.')
    parser.add_argument('--verbosity', type=int, default=0,
                        help='Verbosity level, from 0 to 2')
    args = parser.parse_args()

    if args.content_layers is not None:
        args.content_layers = args.content_layers.split(",")
    if args.style_layers is not None:
        args.style_layers = args.style_layers.split(",")

    result = style_transfer(
        content_img=ImageOps.exif_transpose(Image.open(args.content).convert('RGB')),
        style_img=ImageOps.exif_transpose(Image.open(args.style).convert('RGB')),
        num_steps=args.num_steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        output_resolution=args.output_resolution,
        content_layers=args.content_layers,
        style_layers=args.style_layers,
        upscaling_rounds=args.upscaling_rounds,
        verbosity=args.verbosity
    )

    result.save(args.output)
