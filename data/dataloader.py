"""Dataset classes."""

import os.path as osp
from typing import Optional

from einops import rearrange
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tifffile import tifffile as tif
from torchvision import transforms


class TrainDataset(Dataset):
    """Training dataloader for LNB dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder.
    csv_data : str
        Name of the csv file containing the training data.
    csv_grid : str or None, optional
        Name of the csv file containing the 2*2 grid data.
        Required only if grid_augmentation=True. By default None.
    grid_augmentation : bool, optional
        Whether to use grid augmentation or not. If activates, the dataset will
        sample an augmented data (zoom, rotation, translation) on the fly with a
        probability of 1/2 (other data unchanged). It also double
        the length of the dataset. By default False.

    Data
    ----
    data : torch.Tensor
        Tensor of shape (3, 4, 256, 256) containing the 3 time steps
        and the 4 channels (LAI, LAI mask, VV, VH).
    """

    def __init__(self, dataset_path: str, csv_data: str,
                 csv_grid: Optional[str] = None,
                 grid_augmentation: bool = False) -> None:
        # Paths
        self.dataset_path = dataset_path
        self.s1_path = osp.join(self.dataset_path, 's1')
        self.s2_path = osp.join(self.dataset_path, 's2')
        self.s2m_path = osp.join(self.dataset_path, 's2-mask')
        # Data augmentation
        self.grid_augmentation = grid_augmentation
        # Data frames
        self.series_df = pd.read_csv(osp.join(self.dataset_path,
                                              csv_data))
        if csv_grid:
            self.grid_df = pd.read_csv(osp.join(self.dataset_path,
                                                csv_grid))
        else:
            self.grid_df = None
            if self.grid_augmentation:
                raise ValueError('Rotation augmentation is not possible '
                                 'without 2*2 grid data.')

    def __len__(self) -> int:
        """Length of the dataset."""
        if self.grid_augmentation:
            return 2 * len(self.series_df)
        return len(self.series_df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item at index idx."""
        # TODO: add time feature
        if (not self.grid_augmentation) or np.random.rand() < 1/2:
            # If no grid augmentation or 50% of the time -> normal data
            # without augmentation
            if self.grid_augmentation:
                idx = idx // 2  # Two times more data with grid augmentation
            samples_list = []
            for tstep in ['0', '1', '2']:
                samples_list.append(np.concatenate([
                    tif.imread(osp.join(self.s2_path,
                                        self.series_df[tstep][idx]))[..., None],
                    mask(tif.imread(osp.join(self.s2m_path,
                                             self.series_df[tstep][idx])))[..., None],
                    tif.imread(osp.join(self.s1_path,
                                        self.series_df[tstep][idx])),
                ], axis=-1))
            data_np = np.stack(samples_list, axis=0)  # shape (3, 256, 256, 4)
            data_np = normalize_data_np(data_np)
            data = torch.from_numpy(data_np).float().permute(0, 3, 1, 2)
            # shape (3, 4, 256, 256)
        else:
            idx = idx // 2  # Two times more data with grid augmentation
            # Change idx to get a data from grids dataset
            idx_grid = int(idx / len(self.series_df) * len(self.grid_df))

            # Get a 2*2 grid of data (shape (3, 4, 512, 512))
            grid = self._get_2by2_grid(idx_grid)

            # 90° rotation(s)
            for _ in range(np.random.randint(4)):
                grid = torch.rot90(grid, 1, (2, 3))
            # Random flip (horizontal or vertical)
            if np.random.rand() < 1/2:
                grid = grid.flip(2)
            if np.random.rand() < 1/2:
                grid = grid.flip(3)

            # 0-90° rotation and zoom augmentation
            zoom = 1 + (np.random.rand() - 0.5) * 0.6   # zoom > 1 => close to earth
            theta = np.random.rand() * 90
            interpolation = transforms.InterpolationMode.BILINEAR
            size = int(256 * (2-zoom))  # size of the crop
            # Rotate without cropping
            grid_r = transforms.functional.rotate(grid, theta,
                                                  interpolation=interpolation,
                                                  expand=True,
                                                  fill=0.0)
            # Find a random crop that avoids black borders
            ext = len(grid_r[0, 0])
            x_min = min(256, int(size * np.sin(np.deg2rad(theta))))
            x_max = ext - size - x_min
            x = int(np.random.rand() * (x_max - x_min) + x_min)
            y = int(np.random.rand() * (x_max - x_min) + x_min)
            data = grid_r[:, :, y:y+size, x:x+size]
            # Resize to 256x256
            try:
                data = transforms.functional.resize(data, (256, 256),
                                                    interpolation=interpolation)
            except RuntimeError:  # When the crop is unbounded -> take the center
                data = grid_r[:, :, 128:384, 128:384]
        return data

    def _get_2by2_grid(self, idx: int):
        """Get a 2*2 grid of available data."""
        grid_np_list = []
        for key in self.grid_df.keys():
            # Keys are uleft0, uright0, bleft0, bright0, uleft1, ... etc (3 time steps)
            sample_list = []
            for path in [self.s2_path, self.s2m_path, self.s1_path]:
                data = tif.imread(osp.join(path, self.grid_df[key][idx]))
                if path == self.s2m_path:
                    data = mask(data)
                if data.ndim == 2:
                    data = data[..., None]
                sample_list.append(data)
            sample = np.concatenate(sample_list, axis=-1)
            grid_np_list.append(sample)
        grid_np = np.stack(grid_np_list, axis=0)  # shape (4, 256, 256, 4)
        grid_np = rearrange(grid_np,
                            '(time loc) h w c -> time loc h w c',
                            time=3, loc=4)
        grid_np = rearrange(grid_np,
                            'time (loc1 loc2) h w c -> time (loc1 w) (loc2 h) c',
                            loc1=2, loc2=2)
        grid_np = normalize_data_np(grid_np)
        grid = torch.from_numpy(grid_np).float()
        grid = rearrange(grid, 'time H W c -> time c H W')
        return grid  # shape (3, 4, 512, 512)


def normalize_data_np(data):
    """Normalize numpy data under the format [LAI, LAI mask, VV, VH]."""
    data[..., 0] = np.clip(data[..., 0], 0, 10.0) / 5.0 # in [0, 2]
    data[..., 2:] = data[..., 2:]/30.0 + 1.0  # in [0, 1]
    return data


def mask(img_mask):
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    0, 1, 7, 8, 9 -> 0 (incorrect data)
    other -> 1 (correct data)
    """
    interm = np.where(img_mask < 2, 0.0, 1.0)
    return np.where(img_mask > 6, 0.0, interm)


if __name__ == '__main__':
    # Test the dataset and explore some data
    dataset = TrainDataset(dataset_path='../../data', csv_data='image_series.csv',
                           csv_grid='square.csv', grid_augmentation=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                             num_workers=6)
    for data in dataloader:
        print(data.shape)
        print(data[0, 0, :, :5, :5])
        break
