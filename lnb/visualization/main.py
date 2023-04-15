"""Visualize the data."""

import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import tifffile as tif


def mask(img_mask):
    """Transform an S2 mask (values between 0 and 9) to float32 binary.

    It uses the simple filter:
    0, 1, 7, 8, 9 -> 0 (incorrect data)
    other -> 1 (correct data)
    """
    interm = np.where(img_mask < 2, 0.0, 1.0)
    return np.where(img_mask > 6, 0.0, interm)


def visualize_mask(n_img, data_path, csv_name):
    """Visualize the mask with colors."""

    def val_to_rgb(array):
        rgb = np.zeros((array.shape[0], array.shape[1], 3))
        for i in range(12):
            rgb = np.where(array[..., None] == i, colors[i], rgb)
        return rgb

    csv_path = osp.join(data_path, csv_name)
    series = pd.read_csv(csv_path)
    # randomly shuffle the dataset
    series = series.sample(frac=1).reset_index(drop=True)
    s2m_path = osp.join(data_path, "s2-mask")

    imgs = np.empty((0, 256, 3 * 256, 3))
    colors = {
        0: [0, 0, 0],
        1: [20, 20, 20],
        2: [70, 70, 70],
        3: [140, 140, 140],
        4: [1, 122, 5],
        5: [88, 41, 0],
        6: [0, 0, 255],
        7: [255, 255, 0],
        8: [200, 200, 200],
        9: [255, 255, 255],
        10: [140, 180, 180],
        11: [255, 0, 255],
    }

    for i_img in range(n_img):
        img = np.hstack(
            [
                val_to_rgb(tif.imread(osp.join(s2m_path, series["0"][i_img]))),  # t-2
                val_to_rgb(tif.imread(osp.join(s2m_path, series["1"][i_img]))),  # t-1
                val_to_rgb(tif.imread(osp.join(s2m_path, series["2"][i_img]))),  # t
            ]
        )
        imgs = np.concatenate((imgs, img[None, ...]), axis=0)

    tif.imshow(imgs, title="mask")
    plt.show()


def visualize(n_img, data_path, csv_name, kind="all"):
    """Plot data with tffile under the form:
    if kind == 'all':
    ---------------------------------------------
    |           |          |          |          |
    |normalized |          |          |          |
    |    LAI    | LAI mask |    VV    |    VH    |
    |           |          |          |          |
    |           |          |          |          |
    ---------------------------------------------
    if kind == 'lai':
    --------------
    |             |
    |unnormalized |
    |     LAI     |
    |             |
    |             |
    --------------

    Vertical index is time step:
        t-2
        t-1
        t
    With sliders to control the image index below.
    """

    csv_path = osp.join(data_path, csv_name)
    series = pd.read_csv(csv_path)
    # randomly shuffle the dataset
    series = series.sample(frac=1).reset_index(drop=True)

    s1_path = osp.join(data_path, "s1")  # contains VV and VH
    s2_path = osp.join(data_path, "s2")  # contains LAI
    s2m_path = osp.join(data_path, "s2-mask")

    imgs = np.empty((0, 3 * 256, 4 * 256, 1))

    for i_img in range(n_img):
        img = np.vstack(
            [
                np.hstack(
                    [  # t-2
                        tif.imread(osp.join(s2_path, series["0"][i_img])),  # LAI
                        mask(
                            tif.imread(osp.join(s2m_path, series["0"][i_img]))
                        ),  # LAI mask
                        tif.imread(osp.join(s1_path, series["0"][i_img]))[..., 0],  # VV
                        tif.imread(osp.join(s1_path, series["0"][i_img]))[..., 1],  # VH
                    ]
                ),
                np.hstack(
                    [  # t-1
                        tif.imread(osp.join(s2_path, series["1"][i_img])),  # LAI
                        mask(
                            tif.imread(osp.join(s2m_path, series["1"][i_img]))
                        ),  # LAI mask
                        tif.imread(osp.join(s1_path, series["1"][i_img]))[..., 0],  # VV
                        tif.imread(osp.join(s1_path, series["1"][i_img]))[..., 1],  # VH
                    ]
                ),
                np.hstack(
                    [  # t
                        tif.imread(osp.join(s2_path, series["2"][i_img])),  # LAI
                        mask(
                            tif.imread(osp.join(s2m_path, series["2"][i_img]))
                        ),  # LAI mask
                        tif.imread(osp.join(s1_path, series["2"][i_img]))[..., 0],  # VV
                        tif.imread(osp.join(s1_path, series["2"][i_img]))[..., 1],  # VH
                    ]
                ),
            ]
        )

        if kind == "all":
            # Normalize LAI between 0 and 1 inside an image
            img[..., :256] = img[..., :256] / 12.0

        imgs = np.concatenate((imgs, img[None, ..., None]), axis=0)
        # imgs shape: (i_img, 3, 256, 4*256, 1)

    # Normalize VV and VH between 0 and 1 (range is -30 to 0)
    # New unit: "-30dB"
    imgs[..., 512:, 0] /= -30
    if kind == "all":
        tif.imshow(imgs, title="LAI; LAI mask; VV; VH", vmin=0, vmax=1)
    else:  # kind == 'lai'
        tif.imshow(imgs[..., :256, :], title="LAI")
    plt.show()


if __name__ == "__main__":
    DATA_PATH = "../data"
    CSV_NAME = "train_regular.csv"
    N_IMG = 30
    # visualize(N_IMG, DATA_PATH, CSV_NAME, kind='all')
    visualize_mask(N_IMG, DATA_PATH, CSV_NAME)
