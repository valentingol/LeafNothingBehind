"""Visualize the data."""

import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
from tifffile import tifffile as tif
import pandas as pd


def mask(img_mask):
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    0, 1, 7, 8, 9 -> 0 (incorrect data)
    other -> 1 (correct data)
    """
    interm = np.where(img_mask < 2, 0.0, 1.0)
    return np.where(img_mask > 6, 0.0, interm)


def visualize(n_img, data_path, kind='all'):
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

    csv_path = osp.join(data_path, 'image_series.csv')
    series = pd.read_csv(csv_path)
    # randomly shuffle the dataset
    series = series.sample(frac=1).reset_index(drop=True)

    s1_path = osp.join(data_path, 's1')  # contains VV and VH
    s2_path = osp.join(data_path, 's2')  # contains LAI
    s2m_path = osp.join(data_path, 's2-mask')

    imgs = np.empty((0, 3*256, 4*256, 1))

    for i_img in range(n_img):

        img = np.vstack([
            np.hstack([  # t-2
                tif.imread(osp.join(s2_path, series['0'][i_img])),  # LAI
                mask(tif.imread(osp.join(s2m_path, series['0'][i_img]))),  # LAI mask
                tif.imread(osp.join(s1_path, series['0'][i_img]))[..., 0],  # VV
                tif.imread(osp.join(s1_path, series['0'][i_img]))[..., 1],  # VH
            ]),
            np.hstack([  # t-1
                tif.imread(osp.join(s2_path, series['1'][i_img])),  # LAI
                mask(tif.imread(osp.join(s2m_path, series['1'][i_img]))),  # LAI mask
                tif.imread(osp.join(s1_path, series['1'][i_img]))[..., 0],  # VV
                tif.imread(osp.join(s1_path, series['1'][i_img]))[..., 1],  # VH
            ]),
            np.hstack([  # t
                tif.imread(osp.join(s2_path, series['2'][i_img])),  # LAI
                mask(tif.imread(osp.join(s2m_path, series['2'][i_img]))),  # LAI mask
                tif.imread(osp.join(s1_path, series['2'][i_img]))[..., 0],  # VV
                tif.imread(osp.join(s1_path, series['2'][i_img]))[..., 1],  # VH
            ]),
        ])

        if kind == 'all':
            # Normalize LAI between 0 and 1 inside an image
            img[..., :256] = img[..., :256] / 12.0

        imgs = np.concatenate((imgs, img[None, ..., None]), axis=0)
        # imgs shape: (i_img, 3, 256, 4*256, 1)

    # Normalize VV and VH between 0 and 1 (range is -30 to 0)
    # New unit: "-30dB"
    imgs[..., 512:, 0] /= -30
    if kind == 'all':
        tif.imshow(imgs, title='LAI; LAI mask; VV; VH', vmin=0, vmax=1)
    else:  # kind == 'lai'
        tif.imshow(imgs[..., :256, :], title='LAI')
    plt.show()


if __name__ == '__main__':
    DATA_PATH = '../../data'
    N_IMG = 30
    visualize(N_IMG, DATA_PATH, kind='all')
