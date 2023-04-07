"""Dataset classes."""

import os.path as osp
from typing import Callable, Optional, Tuple

from einops import rearrange
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from tifffile import tifffile as tif
from torchvision import transforms


def normalize_fn(data):
    """Normalize numpy data under the format [LAI, LAI mask, VV, VH]."""
    # LAI
    data[..., 0] = np.clip(data[..., 0], 0, 10.0) / 5.0  # in [0, 2]
    # VV and VH
    data[..., -2:] = data[..., -2:] / 30.0 + 1.0  # in [0, 1]
    return data


def mask_fn(img_mask):
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    0, 1, 7, 8, 9 -> 0 (incorrect data)
    other -> 1 (correct data)
    """
    interm = np.where(img_mask < 2, 0.0, 1.0)
    return np.where(img_mask > 6, 0.0, interm)


class TrainDataset(Dataset):
    """Training dataloader for LNB dataset.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset folder.
    csv_name : str
        Name of the csv file containing the training data.
    csv_grid_name : str or None, optional
        Name of the csv file containing the 2*2 grid data.
        Required only if grid_augmentation=True. By default None.
    grid_augmentation : bool, optional
        Whether to use grid augmentation or not. If activates, the dataset will
        sample an augmented data (zoom, rotation, translation) on the fly with a
        probability of 1/2 (other data unchanged). It also double
        the length of the dataset. By default False.
    mask_fn : Callable, optional
        Function to apply on the raw mask data (under numpy format).
        By default, a simple binary mask is used.
    normalize_fn : Callable, optional
        Function to apply on the raw data (under numpy format) to normalize it.
        A simple normalization is used by default.

    Data
    ----
    data : torch.Tensor
        Tensor of shape (3, 3+c, 256, 256) containing the 3 time steps, t-2, t-1, t
        and the 3+c channels: (LAI, (c LAI mask channels), VV, VH).
    time_info : torch.Tensor
        Tensor two floats between 0 and 1 continuous and periodic
        containing time information.
    """

    def __init__(self, dataset_path: str, csv_name: str,
                 csv_grid_name: Optional[str] = None,
                 grid_augmentation: bool = False,
                 mask_fn: Callable = mask_fn,
                 normalize_fn: Callable = normalize_fn) -> None:
        # Paths
        self.dataset_path = dataset_path
        self.s1_path = osp.join(self.dataset_path, 's1')
        self.s2_path = osp.join(self.dataset_path, 's2')
        self.s2m_path = osp.join(self.dataset_path, 's2-mask')
        # Data augmentation
        self.grid_augmentation = grid_augmentation
        # Functions
        self.mask_fn = mask_fn
        self.normalize_fn = normalize_fn
        # Data frames
        self.series_df = pd.read_csv(osp.join(dataset_path, csv_name))
        if csv_grid_name and grid_augmentation:
            self.grid_df = pd.read_csv(osp.join(self.dataset_path,
                                                csv_grid_name))
        else:
            self.grid_df = None
            if self.grid_augmentation:
                raise ValueError('Rotation augmentation is not possible '
                                 'without the 2*2 grid data.')

    def __len__(self) -> int:
        """Length of the dataset."""
        if self.grid_augmentation:
            return 2 * len(self.series_df)
        return len(self.series_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item at index idx."""
        if (not self.grid_augmentation) or np.random.rand() < 1 / 2:
            # If no grid augmentation or 50% of the time -> normal data
            # without augmentation
            if self.grid_augmentation:
                idx = idx // 2  # Two times more data with grid augmentation
            samples_list = []
            time_info = self._name_to_time_info(self.series_df['0'][idx])
            for tstep in ['0', '1', '2']:
                mask_lai = self.mask_fn(tif.imread(
                    osp.join(self.s2m_path, self.series_df[tstep][idx])
                ))
                if mask_lai.ndim == 2:
                    mask_lai = mask_lai[..., None]
                samples_list.append(np.concatenate([
                    tif.imread(osp.join(self.s2_path,
                                        self.series_df[tstep][idx]))[..., None],
                    mask_lai,
                    tif.imread(osp.join(self.s1_path,
                                        self.series_df[tstep][idx])),
                ], axis=-1))
            data_np = np.stack(samples_list, axis=0)  # shape (3, 256, 256, 4)
            data_np = self.normalize_fn(data_np)
            data = torch.from_numpy(data_np).float().permute(0, 3, 1, 2)
            # shape (3, 3+c, 256, 256)
        else:
            idx = idx // 2  # Two times more data with grid augmentation
            # Change idx to get a data from grids dataset
            idx_grid = int(idx / len(self.series_df) * len(self.grid_df))

            time_info = self._name_to_time_info(self.grid_df['uleft0'][idx_grid])

            # Get a 2*2 grid of data (shape (3, 4, 512, 512))
            grid = self._get_2by2_grid(idx_grid)

            # 90° rotation(s)
            for _ in range(np.random.randint(4)):
                grid = torch.rot90(grid, 1, (2, 3))
            # Random flip (horizontal or vertical)
            if np.random.rand() < 1 / 2:
                grid = grid.flip(2)
            if np.random.rand() < 1 / 2:
                grid = grid.flip(3)

            # 0-90° rotation and zoom augmentation
            zoom = 1 + (np.random.rand() - 0.5) * 0.6   # zoom > 1 => close to earth
            theta = np.random.rand() * 90
            interpolation = transforms.InterpolationMode.BILINEAR
            size = int(256 * (2 - zoom))  # size of the crop
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
            data = grid_r[:, :, y:y + size, x:x + size]
            # Resize to 256x256
            try:
                data = transforms.functional.resize(data, (256, 256),
                                                    interpolation=interpolation)
            except RuntimeError:  # When the crop is outbounded -> take the center
                data = grid_r[:, :, 128:384, 128:384]
        return data, time_info

    def _get_2by2_grid(self, idx: int):
        """Get a 2*2 grid of available data."""
        grid_np_list = []
        for key in self.grid_df.keys():
            # Keys are uleft0, uright0, bleft0, bright0, uleft1, ... etc (3 time steps)
            sample_list = []
            for path in [self.s2_path, self.s2m_path, self.s1_path]:
                data = tif.imread(osp.join(path, self.grid_df[key][idx]))
                if path == self.s2m_path:
                    data = self.mask_fn(data)
                if data.ndim == 2:
                    data = data[..., None]
                sample_list.append(data)
            sample = np.concatenate(sample_list, axis=-1)
            grid_np_list.append(sample)
        grid_np = np.stack(grid_np_list, axis=0)  # shape (4, 256, 256, 4)
        # Split time and 2*2 locations on two axes
        grid_np = rearrange(grid_np,
                            '(time loc) h w c -> time loc h w c',
                            time=3, loc=4)
        # Rearrange locations to get a 2*2 grid
        grid_np = rearrange(grid_np,
                            'time (loc1 loc2) h w c -> time (loc1 w) (loc2 h) c',
                            loc1=2, loc2=2)
        grid_np = self.normalize_fn(grid_np)
        grid = torch.from_numpy(grid_np).float()
        grid = rearrange(grid, 'time H W c -> time c H W')
        return grid  # shape (3, 4, 512, 512)

    def _name_to_time_info(self, filename: str):
        """Parse the name of the file to get the period of time in the year
        and transform it in two shifted periodic, linear and continuous values
        between -1 and 1. The firt value describe the summer-winter axis and
        the second the spring-autumn axis.
        """
        def periodlin_map(linear_date: float):
            """Map a date in [0, 1] to a value in [-1, 1] with a
            periodic linear pattern."""
            sign = - (int((linear_date // 0.5) % 2) * 2 - 1)
            return 4 * sign * (linear_date - 0.5 * linear_date // 0.5) + 2 * (1 - sign) - 1

        split = filename.split('-')
        month, day = float(split[1]), float(split[2].split('_')[0])
        linear_date = (day / 30.0 + month - 1) / 12.0  # in [0, 1]
        return torch.tensor([periodlin_map(linear_date),
                             periodlin_map(linear_date - 0.25)])


if __name__ == '__main__':
    # Test the dataset and explore some data
    dataset = TrainDataset(dataset_path='../data', csv_name='train_regular.csv',
                           csv_grid_name='train_regular_grids.csv',
                           grid_augmentation=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                             num_workers=6)
    for data, time_info in dataloader:
        print('time info:')
        print(time_info)
        print('data:')
        print('- shape:', data.shape)
        print('- ex:', data[0, 0, :, :5, :5])
        break
