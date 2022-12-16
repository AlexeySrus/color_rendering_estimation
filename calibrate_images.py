from argparse import ArgumentParser, Namespace
import cv2
import os
from tqdm import tqdm

from utils.table_tools import read_colorchecker_table, read_gt_table
from utils.calibration import get_rgb_lut, calibrate_image


def parse_args() -> Namespace:
    parser = ArgumentParser('Calibrate images in folder')
    parser.add_argument(
        '-i', '--input', type=str,
        help='Path to input folder with images'
    )
    parser.add_argument(
        '-c', '--calibration', type=str,
        help='Path to folder with detected colors table'
    )
    parser.add_argument(
        '-o', '--output', type=str,
        help='Path to output folder with calibrated'
    )
    parser.add_argument(
        '--gt-table', type=str, required=False, default='./gt_colors.csv',
        help='Path to table with ground truth colors'
    )
    parser.add_argument(
        '--x-rite', action='store_true',
        help='Use this option if you use color checker from x-rite'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    calibration_tables_basenames = [
        fname.split('-')[0]
        for fname in os.listdir(args.calibration)
        if fname.endswith('.csv')
    ]

    os.makedirs(args.output, exist_ok=True)

    rgbs_gt, _ = read_gt_table(args.gt_table, args.x_rite)

    for fname in tqdm(calibration_tables_basenames):
        image_path = os.path.join(args.input, fname + '.JPG')
        output_path = os.path.join(args.output, fname + '.JPG')
        table_path = os.path.join(args.calibration, fname + '-MCC-values.csv')

        img = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )

        rgbs_colors, _ = read_colorchecker_table(table_path)

        rgb_lut = get_rgb_lut(rgbs_colors, rgbs_gt)

        calib_img = calibrate_image(img, rgb_lut)

        cv2.imwrite(
            output_path,
            cv2.cvtColor(calib_img, cv2.COLOR_RGB2BGR)
        )
