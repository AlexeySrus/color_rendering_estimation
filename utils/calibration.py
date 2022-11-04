import numpy as np
import cv2


# From this page: https://stackoverflow.com/questions/18897730/how-i-make-color-calibration-in-opencv-using-a-colorchecker
def polynomialFit(x, y):
    # calculate polynomial
    z = np.polyfit(x, y, 3) # 3 degree polynomial fit
    f = np.poly1d(z) # create a function

    # calculate new x's and y's
    x_new = np.arange(256)
    y_new = np.clip(f(x_new),0,255)
    return y_new


def get_rgb_lut(src_table_rgb: np.ndarray, gt_table_rgb: np.ndarray) -> np.ndarray:
    lineR = polynomialFit(src_table_rgb[:, 0], gt_table_rgb[:, 0])
    lineG = polynomialFit(src_table_rgb[:, 1], gt_table_rgb[:, 1])
    lineB = polynomialFit(src_table_rgb[:, 2], gt_table_rgb[:, 2])

    lutR = np.uint8(lineR)
    lutG = np.uint8(lineG)
    lutB = np.uint8(lineB)

    return np.stack((lutR, lutG, lutB), axis=0)


def calibrate_image(image: np.ndarray, lut_tables: np.ndarray) -> np.ndarray:
    """
    Calibrate RGB image by RGB LUT
    Args:
        image: RGB uint8 image array
        lut_tables: RGB LUT table

    Returns:
        Calibrated image
    """
    res = image.copy()
    res[:, :, 0] = lut_tables[0][image[:, :, 0]]
    res[:, :, 1] = lut_tables[1][image[:, :, 1]]
    res[:, :, 2] = lut_tables[2][image[:, :, 2]]
    return res


if __name__ == '__main__':
    from utils.table_tools import read_colorchecker_table, read_gt_table

    img = cv2.cvtColor(
        cv2.imread('../data/images_dataset/sony/DSC01210.JPG'),
        cv2.COLOR_BGR2RGB
    )

    rgbs_colors, _ = read_colorchecker_table('../data/images_dataset/sony_out/DSC01210-MCC-values.csv')
    rgbs_gt, _ = read_gt_table('../gt_colors.csv')

    rgb_lut = get_rgb_lut(rgbs_colors, rgbs_gt)

    calib_img = calibrate_image(img, rgb_lut)

    cv2.imwrite(
        '../data/calibrated_10.png',
        cv2.cvtColor(calib_img, cv2.COLOR_RGB2BGR)
    )
