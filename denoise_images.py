from argparse import ArgumentParser, Namespace
import cv2
import os
from tqdm import tqdm

from denoising.danet_inference import DANetInference

DEFAULT_DANET_PATH = './data/models/DANet.pt'
DEFAULT_DANETPP_PATH = './data/models/DANetPlus.pt'
DEFAULT_UFORMER_PATH = './data/models/Uformer.pt'


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
        '-a', '--algorithm', type=str, choices=['nlmd', 'danet-pp', 'danet', 'danet-t'],
        required=False,
        default='nlmd',
        help='Denoising approach: ' \
             '\'nlmd\' is Non-local means denoising, ' \
             '\'danet\' is Unet-likely model trained with DANet strategy, ' \
             '\'danet-pp\' is Unet-likely model trained with DANet++ strategy, ' \
             '\'danet-t\' is Uformer model  trained with DANet strategy'
    ),
    parser.add_argument(
        '--device', type=str, default='cpu', required=False,
        choices=['cuda', 'cpu', 'mps'],
        help='Inference device (for Apple M1/M2 chip use \'mps\')'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    denoising_function = None

    if args.algorithm == 'nlmd':
        denoising_function = lambda x: cv2.fastNlMeansDenoisingColored(
            x, None, 10, 10, 7, 21
        )
    elif args.algorithm == 'danet':
        denoising_function = DANetInference(
            'danet', DEFAULT_DANET_PATH, args.device
        )
    elif args.algorithm == 'danet-pp':
        denoising_function = DANetInference(
            'danet++', DEFAULT_DANETPP_PATH, args.device
        )
    elif args.algorithm == 'danet-t':
        denoising_function = DANetInference(
            't', DEFAULT_UFORMER_PATH, args.device
        )
    else:
        raise RuntimeError(
            'Unsupported denoise approach: {}'.format(args.algorithm)
        )

    for fname in tqdm(os.listdir(args.input)):
        image_path = os.path.join(args.input, fname)
        output_path = os.path.join(args.output, fname)

        img = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )

        denoise_img = denoising_function(img)

        cv2.imwrite(
            output_path,
            cv2.cvtColor(denoise_img, cv2.COLOR_RGB2BGR)
        )
