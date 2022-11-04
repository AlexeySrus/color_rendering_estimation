from argparse import ArgumentParser, Namespace
import cv2
import os
from tqdm import tqdm


def parse_args() -> Namespace:
    parser = ArgumentParser('Calibrate images in folder')
    parser.add_argument(
        '-i', '--input', type=str,
        help='Path to input folder with images'
    )
    parser.add_argument(
        '-o', '--output', type=str,
        help='Path to output folder with calibrated'
    )
    parser.add_argument(
        '-a', '--algorithm', type=str, choices=['nlmd'], required=False,
        default='nlmd'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    for fname in tqdm(os.listdir(args.input)):
        image_path = os.path.join(args.input, fname)
        output_path = os.path.join(args.output, fname)

        img = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )

        denoise_img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

        cv2.imwrite(
            output_path,
            cv2.cvtColor(denoise_img, cv2.COLOR_RGB2BGR)
        )
