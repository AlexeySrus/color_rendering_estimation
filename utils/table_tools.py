from typing import List, Tuple
import numpy as np
import pandas as pd

from utils.color_transformations import rgb2Lab, Lab2rgb


def read_colorchecker_table(table_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Color Checker table
    Args:
        table_path: csv table path with ";" separator

    Returns:
        Lists of 24 sRGB detected colors and CIE Lab colors
    """
    rgb_values = []
    lab_values = []
    table = pd.read_csv(table_path, sep=';')
    for patch_no in range(1, 24 + 1):
        patch_rgb = table[(table['patch'] == patch_no) & (table['space'] == 'RGB')]
        rgb = patch_rgb['average']
        lab = rgb2Lab(rgb)

        rgb_values.append(rgb)
        lab_values.append(lab)

    rgb_values = np.array(rgb_values, dtype=np.float32)
    lab_values = np.array(lab_values, dtype=np.float32)
    return rgb_values, lab_values


def read_gt_table(table_path: str, x_rite: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read table with ground truth pathes colors
    Args:
       table_path: csv table path with "," separator
       x_rite: Set True if used color table from x-rite company

    Returns:
       Lists of 24 sRGB detected colors and CIE Lab colors
    """
    rgb_values = []
    lab_values = []
    table = pd.read_csv(table_path, sep=',')
    for parch_no in range(1, 24 + 1):
        line = table[table['patch'] == parch_no].loc[:, ['L', 'a', 'b']]
        lab = line.to_numpy()[0]
        rgb = Lab2rgb(lab)

        rgb_values.append(rgb)
        lab_values.append(lab)

    rgb_values = np.array(rgb_values, dtype=np.float32)
    lab_values = np.array(lab_values, dtype=np.float32)

    if x_rite:
        rgb_values = rgb_values.reshape((4, 6, 3))
        rgb_values[:3] = rgb_values[:3, ::-1]
        rgb_values[1] = rgb_values[1, ::-1]
        rgb_values = rgb_values.reshape((24, 3))

    return rgb_values, lab_values


if __name__ == '__main__':
    from utils.metrics import calculate_delta_e_between_tables, rmse, tables_psnr

    rgbs1, labs1 = read_colorchecker_table('../data/images_dataset/sony_calibrated_out/DSC01212-MCC-values.csv')
    rgbs2, labs2 = read_gt_table('../gt_colors.csv')
    print('Avg Delta E: {:.5f}'.format(calculate_delta_e_between_tables(labs1, labs2)))
    print('RMSE: {:.5f}'.format(rmse(rgbs1, rgbs2)))
    print('PSNR: {:.5f}'.format(tables_psnr(rgbs1, rgbs2)))
