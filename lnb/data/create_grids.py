"""Safe grids to numpy arrays."""
import os
import os.path as osp

import numpy as np
import pandas as pd
import tifffile as tif
from einops import rearrange


def create_grids(data_path: str, csv_name: str) -> None:
    """Create grids and save them as numpy arrays."""
    # Save csv file containing 2*2 grids
    csv_grid_path = save_csv_grids(data_path, csv_name)
    # Save grids as numpy arrays
    save_grids(data_path=data_path, csv_grid_path=csv_grid_path)


def save_csv_grids(data_path: str, csv_name: str) -> str:
    """Save csv grid file."""
    csv_path = osp.join(data_path, csv_name)
    data_df = pd.read_csv(csv_path)

    grids = {}
    for line in data_df.itertuples():
        name = line[1]
        split = name.split("-")
        head, row, col = "-".join(split[:5]), int(split[7]), int(split[8].split(".")[0])
        if head not in grids:
            grids[head] = np.zeros((33, 33), dtype=np.uint8)
        grids[head][row, col] = 1

    list_square = []

    for head, grid in grids.items():
        for i in range(0, 32):
            for j in range(0, 32):
                if (
                    grid[i, j] + grid[i + 1, j] + grid[i, j + 1] + grid[i + 1, j + 1]
                    == 4
                ):
                    list_square.append(
                        [
                            f"{head}-0-0-{i}-{j}.tiff",
                            f"{head}-0-0-{i}-{j+1}.tiff",
                            f"{head}-0-0-{i+1}-{j}.tiff",
                            f"{head}-0-0-{i+1}-{j+1}.tiff",
                            f"{head}-1-0-{i}-{j}.tiff",
                            f"{head}-1-0-{i}-{j+1}.tiff",
                            f"{head}-1-0-{i+1}-{j}.tiff",
                            f"{head}-1-0-{i+1}-{j+1}.tiff",
                            f"{head}-2-0-{i}-{j}.tiff",
                            f"{head}-2-0-{i}-{j+1}.tiff",
                            f"{head}-2-0-{i+1}-{j}.tiff",
                            f"{head}-2-0-{i+1}-{j+1}.tiff",
                        ],
                    )
    # save to csv
    grids_df = pd.DataFrame(
        list_square,
        columns=[
            "uleft0",
            "uright0",
            "bleft0",
            "bright0",
            "uleft1",
            "uright1",
            "bleft1",
            "bright1",
            "uleft2",
            "uright2",
            "bleft2",
            "bright2",
        ],
    )
    csv_grid_path = osp.join(data_path, "train_regular_grids.csv")
    grids_df.to_csv(csv_grid_path, index=False)
    return csv_grid_path


def save_grids(data_path: str, csv_grid_path: str) -> None:
    """Save 2*2 grids as numpy arrays."""
    grid_df = pd.read_csv(osp.join(data_path, csv_grid_path))
    csv_base_name, _ = osp.splitext(csv_grid_path)
    s2_path = osp.join(data_path, "s2")
    s2m_path = osp.join(data_path, "s2-mask")
    s1_path = osp.join(data_path, "s1")
    os.makedirs(osp.join(data_path, csv_base_name), exist_ok=True)

    total_grids = len(grid_df["uleft0"])
    for i in range(len(grid_df["uleft0"])):
        grid_np_list = []
        for key in grid_df:
            # Keys are uleft0, uright0, bleft0, bright0, uleft1, ... etc (3 time steps)
            sample_list = []
            for path in [s2_path, s2m_path, s1_path]:
                data = tif.imread(osp.join(path, grid_df[key][i]))
                if data.ndim == 2:
                    data = data[..., None]
                data = data.astype(np.float32)
                sample_list.append(data)
            sample = np.concatenate(sample_list, axis=-1)
            grid_np_list.append(sample)
        grid_np = np.stack(grid_np_list, axis=0)  # shape (12, 256, 256, c)
        # channels: [LAI, (LAI mask), VV, VH]
        # Split time and 2*2 locations on two axes
        grid_np = rearrange(
            grid_np, "(time loc) h w c -> time loc h w c", time=3, loc=4,
        )
        # Rearrange locations to get a 2*2 grid
        grid_np = rearrange(
            grid_np,
            "time (loc1 loc2) h w c -> time (loc1 w) (loc2 h) c",
            loc1=2,
            loc2=2,
        )  # shape (3, 512, 512, c)
        # Pre-processing should occur here -> save grid_np
        out_path = osp.join(data_path, f"{csv_base_name}", f"grid_{i}.npy")
        np.save(out_path, grid_np)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{total_grids} grids saved.  ", end="\r")
    print(f"{total_grids}/{total_grids} grids saved.  ", end="\r")
    print()


if __name__ == "__main__":
    DATA_PATH = "../data"
    CSV_PATH = "train_regular.csv"
    #  tiff_to_np(DATA_PATH)
    create_grids(DATA_PATH, CSV_PATH)
