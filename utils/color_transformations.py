from typing import Tuple, List

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy as np


def Lab2rgb(lab: List[float]) -> np.ndarray:
    lab_color = LabColor(*lab)
    rgb_color = convert_color(lab_color, sRGBColor)
    rgb_color_arr = np.array(
       rgb_color.get_value_tuple(),
        dtype=np.float32
    ) * 255
    return rgb_color_arr


def rgb2Lab(rgb: List[float]) -> np.ndarray:
    rgb_color = sRGBColor(*[col / 255 for col in rgb], is_upscaled=False)
    lab_color = convert_color(rgb_color, LabColor, target_illuminant='d50')
    lab_color_arr = np.array(
        lab_color.get_value_tuple(),
        dtype=np.float32
    )
    return lab_color_arr


if __name__ == '__main__':
    print(Lab2rgb(rgb2Lab([100, 220, 50])))
