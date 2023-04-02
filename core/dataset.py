import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tifffile import tifffile as tif

import numpy as np
import pandas as pd
import os
import time

def mask(img_mask):
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    0, 1, 7, 8, 9 -> 0 (incorrect data)
    other -> 1 (correct data)
    """
    interm = np.where(img_mask < 2, 0.0, 1.0)
    return np.where(img_mask > 6, 0.0, interm)


def read_img(data_path: str, name: str):
    s1_path = os.path.join(data_path, "s1")  # contains VV and VH
    s2_path = os.path.join(data_path, "s2")  # contains LAI
    s2m_path = os.path.join(data_path, "s2-mask")

    result = {
        "lai": tif.imread(os.path.join(s2_path, name)),
        "lai_mask": mask(tif.imread(os.path.join(s2m_path, name))),
        "vv": tif.imread(os.path.join(s1_path, name))[..., 0],
        "vh": tif.imread(os.path.join(s1_path, name))[..., 1],
    }

    # Normalize LAI
    lai_min, lai_max = result["lai"].min(), result["lai"].max()
    if lai_max > lai_min:
        result["lai"] = (result["lai"] - lai_min) / (lai_max - lai_min)

        # Normalize VV and VH
        result["vv"] /= -30
        result["vh"] /= -30
    else:
        return None
    return result


class LNBDataset(Dataset):
    def __init__(self, data_path: str, stackify: bool = False):
        self.data_path = data_path
        self.series = pd.read_csv(os.path.join(data_path, "image_series.csv"))

        self.s1_path = os.path.join(data_path, "s1")  # contains VV and VH
        self.s2_path = os.path.join(data_path, "s2")  # contains LAI
        self.s2m_path = os.path.join(data_path, "s2-mask")

        self.stackify = stackify

    def __len__(self):
        return self.series.shape[0]

    def __getitem__(self, idx):
        """Get a sample and its prediction.

        Args:
            idx (int): Index of the sample

        Returns:
            list: Sample & prediction
        """
        # start = time.time()

        t_2 = read_img(self.data_path, self.series["0"][idx])
        t_1 = read_img(self.data_path, self.series["1"][idx])
        t = read_img(self.data_path, self.series["2"][idx])

        # print("Read imgs", time.time() - start)

        if t_2 is None or t_1 is None or t is None:
            return None

        sample = {"t_2": t_2, "t_1": t_1, "t": {"vv": t["vv"], "vh": t["vh"]}}
        prediction = {"lai": t["lai"], "lai_mask": t["lai_mask"]}

        if self.stackify:
            # Stackify the sample dictionary into a (256, 256, 10) array
            sample = np.array(
                [
                    *sample["t_2"].values(),
                    *sample["t_1"].values(),
                    *sample["t"].values(),
                ]
            )#.transpose(1, 2, 0)

            # Stackify the prediction dictionary into a (256, 256, 1) array
            prediction = np.array(prediction["lai"])[..., np.newaxis]

        # if self.transform:
        #     sample = self.transform(sample)

        return torch.tensor(sample), torch.tensor(prediction)

    def plot(self, idx):
        sample, prediction = self.__getitem__(idx)

        img_sample = np.vstack(
            [
                np.hstack(sample["t_2"].values()),
                np.hstack(sample["t_1"].values()),
            ]
        )

        img_prediction = np.vstack(
            [
                np.hstack(prediction.values()),
            ]
        )

        print(np.hstack(sample["t_2"].values()).shape)
        print(np.hstack(sample["t_1"].values()).shape)
        print(img_sample.shape)

        tif.imshow(img_sample)
        tif.imshow(img_prediction)

        plt.show()


if __name__ == "__main__":
    dataset = LNBDataset(
        "E:\Antoine\Compétitions\Leaf Nothing Behind\\assignment-2023\\assignment-2023",
        stackify=True,
    )

    # dataset.plot(12)
    sample, prediction = dataset[0]

    # tif.imshow(sample["t_2"]["lai"])
    # tif.imshow(sample["t_1"]["lai"])
    # tif.imshow(prediction["lai"])

    # plt.show()
