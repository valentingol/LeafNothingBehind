import os
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from tifffile import tifffile as tif

def mask(img_mask):
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    0, 1, 7, 8, 9 -> 0 (incorrect data)
    other -> 1 (correct data)
    """
    interm = np.where(img_mask < 2, 0.0, 1.0)
    return np.where(img_mask > 6, 0.0, interm)

def load_sample(data_path: str, name: str):
    s1_path = os.path.join(data_path, "s1")  # contains VV and VH
    s2_path = os.path.join(data_path, "s2")  # contains LAI
    s2m_path = os.path.join(data_path, "s2-mask")

    result = {
        "lai": tif.imread(os.path.join(s2_path, name)),
        "lai_mask": mask(tif.imread(os.path.join(s2m_path, name))),
        "vv": tif.imread(os.path.join(s1_path, name))[..., 0],
        "vh": tif.imread(os.path.join(s1_path, name))[..., 1],
    }

    lai_min, lai_max = result["lai"].min(), result["lai"].max()
    if lai_max > lai_min:
        result["lai"] = (result["lai"] - lai_min) / (lai_max - lai_min)

        # Normalize VV and VH
        result["vv"] /= -30
        result["vh"] /= -30

        return result

    return None


def extract(data_path: str):
    series = pd.read_csv(os.path.join(data_path, "image_series.csv"))

    for idx in trange(5000):
        sample_t_2 = load_sample(data_path, series.iloc[idx, 0])
        sample_t_1 = load_sample(data_path, series.iloc[idx, 1])
        sample_t = load_sample(data_path, series.iloc[idx, 2])

        if sample_t_2 is None or sample_t_1 is None or sample_t is None:
            continue

        sample = np.array(
            [
                *sample_t_2.values(),
                *sample_t_1.values(),
                *sample_t.values(),
            ]
        )
        sample = sample[np.newaxis, ...]

if __name__ == "__main__":
    extract("E:\Antoine\Compétitions\Leaf Nothing Behind\\assignment-2023\\assignment-2023")