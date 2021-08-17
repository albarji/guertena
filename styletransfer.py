"""Main script to run style transfer"""

import argparse
import logging
from neuralstyle.driver import style_transfer_multiresolution
from PIL import Image, ImageOps

from neuralstyle import style_transfer

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
    parser.add_argument('--num_steps', type=int, default=500,
                        help='Number of gradient descent iterations')
    parser.add_argument('--content_weight', type=float, default=1,
                        help='Weight of the content loss')
    parser.add_argument('--style_weight', type=float, default=1000000,
                        help='Weight of the style loss')
    parser.add_argument('--tv_weight', type=float, default=1,
                        help='Weight of the total variation loss')
    parser.add_argument('--output_resolution', type=str, default=None,
                        help='Resolution of output image, in format ROWSxCOLUMNS')
    parser.add_argument('--strategy', type=str, default="single-pass", choices=['single-pass', 'multiresolution'],
                        help='Strategy to generate the image')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Number of rounds for the multiresolution strategy.')
    args = parser.parse_args()

    if args.strategy == "single-pass":
        result = style_transfer(
            content_img=ImageOps.exif_transpose(Image.open(args.content)),
            style_img=ImageOps.exif_transpose(Image.open(args.style)),
            num_steps=args.num_steps,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            tv_weight=args.tv_weight,
            output_resolution=args.output_resolution
        )
    else:
        result = style_transfer_multiresolution(
            content_img=ImageOps.exif_transpose(Image.open(args.content)),
            style_img=ImageOps.exif_transpose(Image.open(args.style)),
            num_steps=args.num_steps,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            tv_weight=args.tv_weight,
            output_resolution=args.output_resolution,
            num_rounds=args.num_rounds
        )

    result.save(args.output)
