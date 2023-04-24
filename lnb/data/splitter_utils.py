"""Utilities for splitting data."""
import copy
import os.path as osp
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import tifffile as tif


def grid_show(grid: dict, title: str, clim_max: Optional[float] = None) -> None:
    """Show a grid of images."""
    if clim_max is None:
        maxi = 0
        for key, _ in grid.items():
            if np.max(grid[key]) > maxi:
                maxi = grid[key].max()
        clim_max = maxi
    fig, axes = plt.subplots(6, 4, figsize=(20, 30))
    plt.title(title)

    for i, key in enumerate(grid.keys()):
        ax_ = axes[i // 4, i % 4]
        ax_.set_title(f"{key} || {title}")
        ax_.axis("off")
        # Add color bar
        cmap = mpl.colormaps.get_cmap("jet")
        norm = plt.Normalize(vmin=0, vmax=clim_max)
        img = ax_.imshow(grid[key], cmap="jet", norm=norm)
        img.set_cmap(cmap)
        img.set_clim(0, clim_max)
        cbar = fig.colorbar(img, ax=ax_, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Valeur")

    plt.show(block=False)


def grid_exploration(series: pd.DataFrame) -> Tuple[dict, dict]:
    """Grid exploration."""
    grids = {}
    grids_size = {}
    for line in series.itertuples():
        name = line[1]
        split = name.split("-")
        head, row, col = "-".join(split[:5]), int(split[7]), int(split[8].split(".")[0])
        if head not in grids:
            grids[head] = np.zeros((33, 33), dtype=np.uint8)
            grids_size[head] = 0
        grids[head][row, col] = 1
        grids_size[head] += 1
    return grids, grids_size


def valtest_prop_in_grids(
    valtest_percent: int,
    grids_size: dict,
) -> Tuple[dict, int]:
    """Compute validation/test proportion for each grid."""
    grids_valtest_size = {}
    total_valtest_size = 0
    for grids_name, size in grids_size.items():
        grids_valtest_size[grids_name] = int(size * valtest_percent / 100)
        total_valtest_size += grids_valtest_size[grids_name]
    return grids_valtest_size, total_valtest_size


def get_data_corner_limits(grid: np.ndarray) -> Tuple[List[int], List[int]]:
    """Get data corner limits."""
    up_right_limits = [0, 32]
    down_left_limits = [32, 0]
    for row in range(33):
        for col in range(33):
            if grid[row, col] == 1:
                if row > up_right_limits[0]:
                    up_right_limits[0] = row
                if row < down_left_limits[0]:
                    down_left_limits[0] = row
                if col < up_right_limits[1]:
                    up_right_limits[1] = col
                if col > down_left_limits[1]:
                    down_left_limits[1] = col

    return down_left_limits, up_right_limits


def get_missing_data_indexes(grid: np.ndarray) -> Tuple[list, int]:
    """Get missing data indexes."""
    missing_data_indexes = []

    down_left_limits, up_right_limits = get_data_corner_limits(grid)

    missing_size = 0

    for row in range(down_left_limits[0], up_right_limits[0] + 1):
        for col in range(up_right_limits[1], down_left_limits[1] + 1):
            if grid[row, col] == 0:
                missing_data_indexes.append([row, col])
                missing_size += 1

    return missing_data_indexes, missing_size


def cross_check_for_whole(
    grid: np.ndarray,
    valid_datas_index: np.ndarray,
    data_index_0: int,
    data_index_1: int,
) -> bool:
    """Cross-check for whole data.

    Check for empty space with valid data between two data
    already selected for the validation/test dataset.
    """
    if (
        data_index_0 < 31
        and (
            valid_datas_index[data_index_0 + 2, data_index_1] == 1
            and grid[data_index_0 + 1, data_index_1] == 1
        )
        and valid_datas_index[data_index_0 + 1][data_index_1] == 0
    ):
        valid_datas_index[data_index_0 + 1][data_index_1] = 1
        return True
    if (
        data_index_0 > 1
        and (
            valid_datas_index[data_index_0 - 2, data_index_1] == 1
            and grid[data_index_0 - 1, data_index_1] == 1
        )
        and valid_datas_index[data_index_0 - 1][data_index_1] == 0
    ):
        valid_datas_index[data_index_0 - 1][data_index_1] = 1
        return True
    if (
        data_index_1 < 31
        and (
            valid_datas_index[data_index_0, data_index_1 + 2] == 1
            and grid[data_index_0, data_index_1 + 1] == 1
        )
        and valid_datas_index[data_index_0][data_index_1 + 1] == 0
    ):
        valid_datas_index[data_index_0][data_index_1 + 1] = 1
        return True
    if (
        data_index_1 > 1
        and (
            valid_datas_index[data_index_0, data_index_1 - 2] == 1
            and grid[data_index_0, data_index_1 - 1] == 1
        )
        and valid_datas_index[data_index_0][data_index_1 - 1] == 0
    ):
        valid_datas_index[data_index_0][data_index_1 - 1] = 1
        return True
    return False


def get_border_data(
    grid: np.ndarray,
    n_valtest_per_grid: int,
    valid_datas_index: np.ndarray,
    expect_grid_valtest_size: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Get border data."""
    down_left_limits, up_right_limits = get_data_corner_limits(grid)

    for row in range(down_left_limits[0], up_right_limits[0] + 1):
        for col in range(up_right_limits[1], down_left_limits[1] + 1):
            if (
                grid[row, col] == 1
                and valid_datas_index[row, col] == 0
                and n_valtest_per_grid < expect_grid_valtest_size
            ):
                valid_datas_index[row, col] = 1
                n_valtest_per_grid += 1

    return n_valtest_per_grid, valid_datas_index


# pylint: disable=too-many-branches
def valid_data_around_missing_data(  # noqa: C901
    grid: np.ndarray,
    missing_data_indexes: list,
    expect_grid_valtest_size: int,
    valid_datas_index: np.ndarray,
    n_valtest_per_grid: int,
) -> Tuple[np.ndarray, int]:
    """Get val/test data around missing data.

    Get the validation/test data from near to the more missing data possible
    and then get the data around (of the missing data)
    and then get the data from the border if it's not enough.
    """
    for _ in range(16):
        for data_index in missing_data_indexes:
            if (n_valtest_per_grid < expect_grid_valtest_size
                    and data_index[0] < 32
                    and grid[data_index[0] + 1, data_index[1]] == 1):
                # Perform a vertical and horizontal cross check
                # to get the valid data around the missing data
                if valid_datas_index[data_index[0] + 1][data_index[1]] == 0:
                    valid_datas_index[data_index[0] + 1][data_index[1]] = 1
                    n_valtest_per_grid += 1
                if (cross_check_for_whole(grid, valid_datas_index, data_index[0] + 1,
                                          data_index[1])
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue
            if (n_valtest_per_grid < expect_grid_valtest_size
                    and data_index[0] > 0
                    and grid[data_index[0] - 1, data_index[1]] == 1):
                if valid_datas_index[data_index[0] - 1][data_index[1]] == 0:
                    valid_datas_index[data_index[0] - 1][data_index[1]] = 1
                    n_valtest_per_grid += 1
                if (cross_check_for_whole(grid, valid_datas_index, data_index[0] - 1,
                                          data_index[1])
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue
            if ((n_valtest_per_grid < expect_grid_valtest_size)
                    and data_index[1] < 32
                    and grid[data_index[0], data_index[1] + 1] == 1):
                if valid_datas_index[data_index[0]][data_index[1] + 1] == 0:
                    valid_datas_index[data_index[0]][data_index[1] + 1] = 1
                    n_valtest_per_grid += 1
                if (cross_check_for_whole(grid, valid_datas_index, data_index[0],
                                          data_index[1] + 1)
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue
            if ((n_valtest_per_grid < expect_grid_valtest_size)
                    and data_index[1] > 0
                    and grid[data_index[0], data_index[1] - 1] == 1):
                if valid_datas_index[data_index[0]][data_index[1] - 1] == 0:
                    valid_datas_index[data_index[0]][data_index[1] - 1] = 1
                    n_valtest_per_grid += 1
                if (cross_check_for_whole(grid, valid_datas_index, data_index[0],
                                          data_index[1] - 1)
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue

                # Perform a diagonal cross to get the valid data
                # around the missing data
            if (n_valtest_per_grid < expect_grid_valtest_size
                    and data_index[0] < 32
                    and data_index[1] > 0
                    and grid[data_index[0] + 1, data_index[1] - 1] == 1):
                if valid_datas_index[data_index[0] + 1][data_index[1] - 1] == 0:
                    valid_datas_index[data_index[0] + 1][data_index[1] - 1] = 1
                    n_valtest_per_grid += 1

                if (cross_check_for_whole(grid, valid_datas_index, data_index[0] + 1,
                                          data_index[1] - 1)
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue
            if (n_valtest_per_grid < expect_grid_valtest_size
                    and data_index[0] > 0
                    and data_index[1] > 0
                    and grid[data_index[0] - 1, data_index[1] - 1] == 1):
                if valid_datas_index[data_index[0] - 1][data_index[1] - 1] == 0:
                    valid_datas_index[data_index[0] - 1][data_index[1] - 1] = 1
                    n_valtest_per_grid += 1

                if (cross_check_for_whole(grid, valid_datas_index, data_index[0] - 1,
                                          data_index[1] - 1)
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue
            if (n_valtest_per_grid < expect_grid_valtest_size
                    and data_index[1] < 32
                    and data_index[0] > 0
                    and grid[data_index[0] - 1, data_index[1] + 1] == 1):
                if valid_datas_index[data_index[0] - 1][data_index[1] + 1] == 0:
                    valid_datas_index[data_index[0] - 1][data_index[1] + 1] = 1
                    n_valtest_per_grid += 1

                if (cross_check_for_whole(grid, valid_datas_index, data_index[0] - 1,
                                          data_index[1] + 1)
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue
            if (n_valtest_per_grid < expect_grid_valtest_size
                    and data_index[0] < 32
                    and data_index[1] < 32
                    and grid[data_index[0] + 1, data_index[1] + 1] == 1):
                if valid_datas_index[data_index[0] + 1][data_index[1] + 1] == 0:
                    valid_datas_index[data_index[0] + 1][data_index[1] + 1] = 1
                    n_valtest_per_grid += 1

                if (cross_check_for_whole(grid, valid_datas_index, data_index[0] + 1,
                                          data_index[1] + 1)
                        and n_valtest_per_grid < expect_grid_valtest_size):
                    n_valtest_per_grid += 1
                    continue

    return valid_datas_index, n_valtest_per_grid


def get_around_missing(
    grids: dict,
    valtest_percent: int,
    grids_size: dict,
) -> Tuple[dict, dict]:
    """Get sample data around missing data."""
    expect_grids_valtest_size, _ = valtest_prop_in_grids(
        valtest_percent,
        grids_size,
    )

    valtest_data_per_grid = {}
    valtest_data_num_per_grid = {}

    for grid_name, grid in grids.items():
        valtest_data_per_grid[grid_name] = np.zeros(
            (33, 33),
            dtype=np.uint,
        )
        valtest_data_num_per_grid[grid_name] = 0
        missing_datas_indexes, _ = get_missing_data_indexes(grid)

        (
            valtest_data_per_grid[grid_name],
            valtest_data_num_per_grid[grid_name],
        ) = valid_data_around_missing_data(
            grid,
            missing_datas_indexes,
            expect_grids_valtest_size[grid_name],
            valtest_data_per_grid[grid_name],
            valtest_data_num_per_grid[grid_name],
        )

        # Get data from the border
        if (
            valtest_data_num_per_grid[grid_name]
            < expect_grids_valtest_size[grid_name]
        ):
            (
                valtest_data_num_per_grid[grid_name],
                valtest_data_per_grid[grid_name],
            ) = get_border_data(
                grid,
                valtest_data_num_per_grid[grid_name],
                valtest_data_per_grid[grid_name],
                expect_grids_valtest_size[grid_name],
            )

    return valtest_data_per_grid, valtest_data_num_per_grid


def average_differences(
    line: List[str],
    data_path: str,
) -> Tuple[float, float]:
    """Get average absolute differences between two images."""
    s1_path = osp.join(data_path, "s2-mask")  # contains VV and VH

    t_vv = tif.imread(osp.join(s1_path, line[3]))[..., 0]  # VV
    t_vh = tif.imread(osp.join(s1_path, line[3]))[..., 1]  # VH

    t1_vv = tif.imread(osp.join(s1_path, line[2]))[..., 0]  # VV
    t1_vh = tif.imread(osp.join(s1_path, line[2]))[..., 1]  # VH

    vv_differences = np.mean(np.absolute(t_vv - t1_vv))
    vh_differences = np.mean(np.absolute(t_vh - t1_vh))

    # print(vv_differences, vh_differences)
    return vv_differences, vh_differences  # type: ignore


def mask(img_mask: np.ndarray) -> np.ndarray:
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    8, 9 -> 1 (cloudy data)
    other -> 0 (not cloudy data)
    """
    interm = np.where(img_mask < 8, 0.0, 1.0)
    return np.where(img_mask > 9, 0.0, interm)


def cloudy_pixels_prop(
    line: List[str],
    from_cloudy_percentage: int,
    data_path: str,
) -> Tuple[List[int], List[int]]:
    """Return cloudy pixels proportions."""
    s2m_path = osp.join(data_path, "s2-mask")  # contains mask

    t1_mask = tif.imread(osp.join(s2m_path, line[2]))
    t2_mask = tif.imread(osp.join(s2m_path, line[1]))

    proportions_array = [from_cloudy_percentage * 256 * 256, 0.95 * 256 * 256]

    res = [-1, -1]

    for i, n in enumerate(proportions_array):
        t1_mask = mask(tif.imread(osp.join(s2m_path, line[2])))
        t2_mask = mask(tif.imread(osp.join(s2m_path, line[1])))

        nb_pixel_cloudy_t1 = np.count_nonzero(t1_mask == 1)
        nb_pixel_cloudy_t2 = np.count_nonzero(t2_mask == 1)

        if i < len(proportions_array) - 1:
            if proportions_array[i + 1] > nb_pixel_cloudy_t2 >= n:
                res[1] = i
            if proportions_array[i + 1] > nb_pixel_cloudy_t1 >= n:
                res[0] = i
        else:
            if nb_pixel_cloudy_t2 >= n:
                res[1] = i
            if nb_pixel_cloudy_t1 >= n:
                res[0] = i

    return res, proportions_array  # type: ignore


def get_vh_vv_differences(differences: List[dict]) -> dict:
    """Get the VH and VV differences."""
    s1_difference_dataset = {}
    for idx, difference in enumerate(differences):
        conca_array: Optional[np.ndarray] = None
        no_in_dataset = 0
        in_dataset = 0

        # Segment correspond to the limit value beyond witch the difference
        # is considered HIGH, first fort VV and second for VH
        segment = 1.9 if idx == 0 else 1.6

        for key, _ in difference.items():
            if key not in s1_difference_dataset:
                s1_difference_dataset[key] = np.zeros((33, 33))
            for i in range(difference[key].shape[0]):
                for j in range(difference[key].shape[1]):
                    if difference[key][i, j] != 0:
                        if difference[key][i, j] < segment:
                            no_in_dataset += 1
                        else:
                            s1_difference_dataset[key][i, j] = 1
                            in_dataset += 1
            # print(key)
            if conca_array is None:
                conca_array_np = copy.deepcopy(difference[key])
            else:
                conca_array_np = np.concatenate((conca_array, difference[key]))

        s1_type = "VV" if idx == 0 else "VH"
        segment_end = np.round(conca_array_np.max(), decimals=2)
        proportion_in = np.round(
            100 / (no_in_dataset + in_dataset) * in_dataset,
            decimals=2,
        )
        proportion_no_in = np.round(
            100 / (no_in_dataset + in_dataset) * no_in_dataset,
            decimals=2,
        )
        print(
            f"{s1_type} difference entre 0 et {segment} = "
            f"{no_in_dataset} => {proportion_no_in} %",
        )
        print(
            f"{s1_type} difference entre {segment} et "
            f"{segment_end} = {in_dataset} => {proportion_in} %",
        )

    return s1_difference_dataset


def get_s1_differences_dataset(dataset_to_check: pd.DataFrame, data_path: str) -> dict:
    """Get Sentinel 1 differences."""
    csv_path = osp.join(data_path, "image_series.csv")
    series = pd.read_csv(csv_path)

    vv_differences = {}
    vh_differences = {}

    for line in series.itertuples():
        name = line[1]
        split = name.split("-")
        head, row, col = "-".join(split[:5]), int(split[7]), int(split[8].split(".")[0])
        if head in dataset_to_check and dataset_to_check[head][row, col] == 1:
            if head not in vv_differences or head not in vv_differences:
                vv_differences[head] = np.zeros((33, 33))
                vh_differences[head] = np.zeros((33, 33))
            (
                vv_differences[head][row, col],
                vh_differences[head][row, col],
            ) = average_differences(line, data_path)

    s1_difference_dataset = get_vh_vv_differences([vv_differences, vh_differences])

    return s1_difference_dataset
