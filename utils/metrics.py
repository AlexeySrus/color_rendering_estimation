import cv2
import numpy as np
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie1976


def calculate_delta_e(lab_color1: np.ndarray, lab_color2: np.ndarray) -> float:
    """
    Calculate Delta E CIE 1976 distance between two colors in CIE Lab space
    Args:
        lab_color1: Lab color
        lab_color2: Lab color

    Returns:
        Delta E distance value
    """
    _color1 = LabColor(*lab_color1)
    _color2 = LabColor(*lab_color2)
    return delta_e_cie1976(_color1, _color2)


def calculate_delta_e_between_tables(lab_table_1: np.ndarray, lab_table_2: np.ndarray) -> float:
    """
    Calculate average Delta E CIE 1976 distance between correspondent colors in two tables with Lab colors
    Args:
        lab_table_1: List of Lab colors
        lab_table_2: List of Lab colors

    Returns:
        Average Delta E distance value
    """
    assert lab_table_1.shape == lab_table_2.shape, 'Shape of tables are not equal'
    avg_delta_e = 0
    for _patch_no in range(lab_table_1.shape[0]):
        avg_delta_e += calculate_delta_e(lab_table_1[_patch_no], lab_table_2[_patch_no])

    return avg_delta_e / lab_table_1.shape[0]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2) / a.shape[0])


def tables_psnr(rgb_table1: np.ndarray, rgb_table2: np.ndarray) -> float:
    return cv2.PSNR(
        np.expand_dims(rgb_table1, axis=0),
        np.expand_dims(rgb_table2, axis=0)
    )
