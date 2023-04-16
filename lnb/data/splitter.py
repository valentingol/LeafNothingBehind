"""Train/validation/test data splitter functions."""
import copy
import csv
import os
import os.path as osp

import shutil
import argparse


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tifffile import tifffile as tif


def grid_show(grid, tittle, clim_max=None):
    """Show a grid of images."""
    if clim_max is None:
        maxi = 0
        for key, _ in grid.items():
            if np.max(grid[key]) > maxi:
                maxi = grid[key].max()
        clim_max = maxi
    fig, axes = plt.subplots(6, 4, figsize=(20, 30))
    plt.title(tittle)

    for i, key in enumerate(grid.keys()):
        ax_ = axes[i // 4, i % 4]
        ax_.set_title(f"{key} || {tittle}")
        ax_.axis("off")
        # Add color bar
        cmap = matplotlib.colormaps.get_cmap("jet")
        norm = plt.Normalize(vmin=0, vmax=clim_max)
        img = ax_.imshow(grid[key], cmap="jet", norm=norm)
        img.set_cmap(cmap)
        img.set_clim(0, clim_max)
        cbar = fig.colorbar(img, ax=ax_, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Valeur")

    plt.show(block=False)


def goss_show(array, tittle):
    """Show distribution of a numpy array."""
    data = array.flatten()
    data = np.delete(data, np.where(data == 0))
    # Compute data histogram
    hist, bins = np.histogram(data, bins=150)
    # Compute histogram percentage
    hist_percent = hist / np.sum(hist)
    # Draw distribution curve
    plt.plot(bins[:-1], hist_percent)

    plt.xlabel("Valeurs")
    plt.ylabel("Pourcentage")
    plt.title(f"Courbe de distribution {tittle}")
    plt.show(block=False)


def grid_exploration(series):
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


def val_test_proportion_for_each_grid(validation_test_percent, grids_size):
    """Compute validation/test proportion for each grid."""
    grids_valtest_size = {}
    total_val_test_data_size = 0
    for grids_name, size in grids_size.items():
        grids_valtest_size[grids_name] = int(size * validation_test_percent / 100)
        total_val_test_data_size += grids_valtest_size[grids_name]
    return grids_valtest_size, total_val_test_data_size


def get_data_corner_limits(grid):
    """Get data corner limits."""
    up_right_corner_limits = [0, 32]
    down_left_corner_limits = [32, 0]
    for row in range(33):
        for col in range(33):
            if grid[row, col] == 1:
                if row > up_right_corner_limits[0]:
                    up_right_corner_limits[0] = row
                if row < down_left_corner_limits[0]:
                    down_left_corner_limits[0] = row
                if col < up_right_corner_limits[1]:
                    up_right_corner_limits[1] = col
                if col > down_left_corner_limits[1]:
                    down_left_corner_limits[1] = col

    return down_left_corner_limits, up_right_corner_limits


def get_missing_data_indexes(grid):
    """Get missing data indexes."""
    missing_data_indexes = []

    down_left_corner_limits, up_right_corner_limits = get_data_corner_limits(grid)

    missing_size = 0

    for row in range(down_left_corner_limits[0], up_right_corner_limits[0] + 1):
        for col in range(up_right_corner_limits[1], down_left_corner_limits[1] + 1):
            if grid[row, col] == 0:
                missing_data_indexes.append([row, col])
                missing_size += 1

    return missing_data_indexes, missing_size


def cross_check_for_whole(grid, valid_datas_index, data_index_0, data_index_1):
    """Check for empty space with valid data between two data
    already selected for the validation/test dataset.
    """
    if data_index_0 < 31 and (
        valid_datas_index[data_index_0 + 2, data_index_1] == 1
        and grid[data_index_0 + 1, data_index_1] == 1
    ) and valid_datas_index[data_index_0 + 1][data_index_1] == 0:
        valid_datas_index[data_index_0 + 1][data_index_1] = 1
        return True
    if data_index_0 > 1 and (
        valid_datas_index[data_index_0 - 2, data_index_1] == 1
        and grid[data_index_0 - 1, data_index_1] == 1
    ) and valid_datas_index[data_index_0 - 1][data_index_1] == 0:
        valid_datas_index[data_index_0 - 1][data_index_1] = 1
        return True
    if data_index_1 < 31 and (
        valid_datas_index[data_index_0, data_index_1 + 2] == 1
        and grid[data_index_0, data_index_1 + 1] == 1
    ) and valid_datas_index[data_index_0][data_index_1 + 1] == 0:
        valid_datas_index[data_index_0][data_index_1 + 1] = 1
        return True
    if data_index_1 > 1 and (
        valid_datas_index[data_index_0, data_index_1 - 2] == 1
        and grid[data_index_0, data_index_1 - 1] == 1
    ) and valid_datas_index[data_index_0][data_index_1 - 1] == 0:
        valid_datas_index[data_index_0][data_index_1 - 1] = 1
        return True
    return False


def get_border_data(
    grid,
    number_of_data_for_valtest_for_a_grid,
    valid_datas_index,
    expected_grid_validation_test_size,
):
    """Get border data."""
    down_left_corner_limits, up_right_corner_limits = get_data_corner_limits(grid)

    for row in range(down_left_corner_limits[0], up_right_corner_limits[0] + 1):
        for col in range(up_right_corner_limits[1], down_left_corner_limits[1] + 1):
            if (
                grid[row, col] == 1
                and valid_datas_index[row, col] == 0
                and number_of_data_for_valtest_for_a_grid
                < expected_grid_validation_test_size
            ):
                valid_datas_index[row, col] = 1
                number_of_data_for_valtest_for_a_grid += 1

    return number_of_data_for_valtest_for_a_grid, valid_datas_index


def valid_datas_arround_missing_datas(
    grid,
    missing_data_indexes,
    expected_grid_validation_test_size,
    valid_datas_index,
    number_of_data_for_valtest_for_a_grid,
):
    """
    Get the validation/test data from near to the more missing data possible
    and then get the data around (of the missing data)
    and then get the data from the border if it's not enough.
    """
    for _ in range(16):
        for data_index in missing_data_indexes:

            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                # Perform a vertical and horizontal cross check
                # to get the valid data around the missing data
                if data_index[0] < 32:
                    if grid[data_index[0] + 1, data_index[1]] == 1:
                        if valid_datas_index[data_index[0] + 1][data_index[1]] == 0:
                            valid_datas_index[data_index[0] + 1][data_index[1]] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] + 1, data_index[1]) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                if data_index[0] > 0:
                    if grid[data_index[0] - 1, data_index[1]] == 1:
                        if valid_datas_index[data_index[0] - 1][data_index[1]] == 0:
                            valid_datas_index[data_index[0] - 1][data_index[1]] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index, data_index[0]
                                                 - 1, data_index[1]) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                if data_index[1] < 32:
                    if grid[data_index[0], data_index[1] + 1] == 1:
                        if valid_datas_index[data_index[0]][data_index[1] + 1] == 0:
                            valid_datas_index[data_index[0]][data_index[1] + 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0], data_index[1] + 1) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                if data_index[1] > 0:
                    if grid[data_index[0], data_index[1] - 1] == 1:
                        if valid_datas_index[data_index[0]][data_index[1] - 1] == 0:
                            valid_datas_index[data_index[0]][data_index[1] - 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0], data_index[1] - 1) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:
                            number_of_data_for_valtest_for_a_grid += 1
                            continue

                # Perform a diagonal cross to get the valid data
                # around the missing data
            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                if data_index[0] < 32 and data_index[1] > 0:
                    if grid[data_index[0] + 1, data_index[1] - 1] == 1:
                        if valid_datas_index[data_index[0] + 1][data_index[1] - 1] == 0:
                            valid_datas_index[data_index[0] + 1][data_index[1] - 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1

                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] + 1,
                                                 data_index[1] - 1) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:

                            number_of_data_for_valtest_for_a_grid += 1
                            continue
            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                if data_index[0] > 0 and data_index[1] > 0:
                    if grid[data_index[0] - 1, data_index[1] - 1] == 1:
                        if valid_datas_index[data_index[0] - 1][data_index[1] - 1] == 0:
                            valid_datas_index[data_index[0] - 1][data_index[1] - 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1

                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] - 1,
                                                 data_index[1] - 1) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:

                            number_of_data_for_valtest_for_a_grid += 1
                            continue
            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                if data_index[1] < 32 and data_index[0] > 0:
                    if grid[data_index[0] - 1, data_index[1] + 1] == 1:
                        if valid_datas_index[data_index[0] - 1][data_index[1] + 1] == 0:
                            valid_datas_index[data_index[0] - 1][data_index[1] + 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1

                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] - 1,
                                                 data_index[1] + 1) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:

                            number_of_data_for_valtest_for_a_grid += 1
                            continue
            if (number_of_data_for_valtest_for_a_grid <
                    expected_grid_validation_test_size):
                if data_index[0] < 32 and data_index[1] < 32:
                    if grid[data_index[0] + 1, data_index[1] + 1] == 1:
                        if valid_datas_index[data_index[0] + 1][data_index[1] + 1] == 0:
                            valid_datas_index[data_index[0] + 1][data_index[1] + 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1

                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] + 1,
                                                 data_index[1] + 1) and number_of_data_for_valtest_for_a_grid < expected_grid_validation_test_size:

                            number_of_data_for_valtest_for_a_grid += 1
                            continue

    return valid_datas_index, number_of_data_for_valtest_for_a_grid


def get_datas_arround_missing_datas(
        grids, validation_test_percent, grids_size):
    """Get sample data around missing data."""

    expected_grids_validation_test_size, _ = val_test_proportion_for_each_grid(
        validation_test_percent, grids_size)

    validation_test_data_for_each_grid = {}
    valtest_data_num_per_grid = {}

    for grid_name, grid in grids.items():

        validation_test_data_for_each_grid[grid_name] = np.zeros((33, 33),
                                                                 dtype=np.uint)
        valtest_data_num_per_grid[grid_name] = 0
        missing_datas_indexes, missing_size = get_missing_data_indexes(grid)

        (validation_test_data_for_each_grid[grid_name],
            valtest_data_num_per_grid[grid_name]) = valid_datas_arround_missing_datas(
            grid, missing_datas_indexes,
            expected_grids_validation_test_size[grid_name],
            validation_test_data_for_each_grid[grid_name],
            valtest_data_num_per_grid[grid_name])

        # Get data from the border
        if (valtest_data_num_per_grid[grid_name]
                < expected_grids_validation_test_size[grid_name]):
            (valtest_data_num_per_grid[grid_name],
                validation_test_data_for_each_grid[grid_name]) = get_border_data(
                grid, valtest_data_num_per_grid[grid_name],
                validation_test_data_for_each_grid[grid_name],
                expected_grids_validation_test_size[grid_name])

    return validation_test_data_for_each_grid, valtest_data_num_per_grid


def average_absolute_differences(line, data_path):
    s1_path = osp.join(data_path, "s2-mask")  # contains VV and VH

    t_vv = tif.imread(osp.join(s1_path, line[3]))[..., 0]  # VV
    t_vh = tif.imread(osp.join(s1_path, line[3]))[..., 1]  # VH

    t1_vv = tif.imread(osp.join(s1_path, line[2]))[..., 0]  # VV
    t1_vh = tif.imread(osp.join(s1_path, line[2]))[..., 1]  # VH

    vv_differences = np.mean(np.absolute(t_vv - t1_vv))
    vh_differences = np.mean(np.absolute(t_vh - t1_vh))

    # print(vv_differences, vh_differences)

    return vv_differences, vh_differences


def mask(img_mask):
    """Transform an S2 mask (values between 1 and 9) to float32 binary.

    It uses the simple filter:
    8, 9 -> 1 (cloudy data)
    other -> 0 (not cloudy data)
    """
    interm = np.where(img_mask < 8, 0.0, 1.0)
    return np.where(img_mask > 9, 0.0, interm)


def cloudy_pixels_proportions(line, from_cloudy_percentage, data_path):
    s2m_path = osp.join(data_path, 's2-mask')  # contains mask

    t1_mask = tif.imread(osp.join(s2m_path, line[2]))
    t2_mask = tif.imread(osp.join(s2m_path, line[1]))

    proportions_array = [
        from_cloudy_percentage * 256 * 256,
        0.95 * 256 * 256]

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

    return res, proportions_array


def get_vh_vv_differences(differences):
    s1_difference_dataset = {}
    for o, difference in enumerate(differences):
        conca_array = None
        no_in_dataset = 0
        in_dataset = 0

        """
        Segment correspond to the limit value beyond witch the difference
        is considered HIGH, first fort VV and second for VH
        """
        segment = 1.9 if o == 0 else 1.6

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
                conca_array = copy.deepcopy(difference[key])
            else:
                conca_array = np.concatenate((conca_array, difference[key]))

        s1_type = "VV" if o == 0 else "VH"

        segment_end = np.round(conca_array.max(), decimals=2)

        proportion_in = np.round(
            100 / (no_in_dataset + in_dataset) * in_dataset, decimals=2,
        )

        proportion_no_in = np.round(
            100 / (no_in_dataset + in_dataset) * no_in_dataset, decimals=2,
        )

        print(
            f"{s1_type} difference entre 0 et {segment} = "
            f"{no_in_dataset} => {proportion_no_in} %",
        )
        print(
            f"{s1_type} difference entre {segment} et "
            f"{segment_end} = {in_dataset} => {proportion_in} %",
        )

        # goss_show(conca_array, f"{s1_type}_differences entre t et t-1")

    # Show the difference from the grid view
    # grid_show(differences[0], "VV differences t et t-1")
    # grid_show(differences[1], "VH differences t et t-1")

    return s1_difference_dataset


def get_s1_differences_dataset(dataset_to_check, data_path):
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
            ) = average_absolute_differences(line, data_path)

    s1_difference_dataset = get_vh_vv_differences([vv_differences, vh_differences])

    return s1_difference_dataset


def select(selecter, head, row, col, train_data_set, val_dataset, test_dataset, N_prop):
    temp = selecter
    rand = np.random.randint(0, 100)
    if rand < 100 - N_prop:
        selecter = 1

    if (selecter == 1 or selecter == 0):
        if head not in train_data_set:
            train_data_set[head] = np.zeros((33, 33))
        train_data_set[head][row][col] = 1
        return temp
    if selecter == 2:
        if head not in val_dataset:
            val_dataset[head] = np.zeros((33, 33))
        val_dataset[head][row][col] = 1
        return 3
    if selecter == 3:
        if head not in test_dataset:
            test_dataset[head] = np.zeros((33, 33))
        test_dataset[head][row][col] = 1
        return 2


def split_cloudy_set(cloudy_pixels, N_prop):
    train_dataset = {}
    val_dataset = {}
    test_dataset = {}

    selecter = [2, 2, 2]

    for head, _ in cloudy_pixels.items():
        for row in range(33):
            for col in range(33):
                cloudiest = cloudy_pixels[head][row][col]

                # Ensure that the distribution of double clouded imager (t-1 and t-2) is
                # egal between the 3 datasets
                if cloudiest[0] != -1 and cloudiest[1] != -1:
                    selecter[0] = select(
                        selecter[0],
                        head,
                        row,
                        col,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        N_prop
                    )
                # Ensure that the distribution of highly clouded images (t-1 or t-2) is
                # egal between the 3 datasets
                elif cloudiest[0] == 1 or cloudiest[1] == 1:
                    selecter[1] = select(
                        selecter[1],
                        head,
                        row,
                        col,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        N_prop
                    )
                # Ensure that the distribution of the images that are not very cloudy (t-1
                # or t-2) is egal between the 3 datasets
                elif cloudiest[0] == 0 or cloudiest[1] == 0:
                    selecter[2] = select(
                        selecter[2],
                        head,
                        row,
                        col,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        N_prop
                    )

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_cloudy(cloudy_pixels, pro_array, N_prop):
    proportions_array = copy.deepcopy(pro_array)
    for i, element in enumerate(proportions_array):
        proportions_array[i] = element / (256 * 256)

    clouded_repartition = np.zeros(len(proportions_array))
    repartition = np.zeros(len(proportions_array))
    double = 0

    for head, _ in cloudy_pixels.items():
        for row in range(33):
            for col in range(33):
                cloudiest = cloudy_pixels[head][row][col]
                if cloudiest[0] != -1:
                    repartition[cloudiest[0]] += 1
                if cloudiest[1] != -1:
                    repartition[cloudiest[1]] += 1
                if cloudiest[0] != -1 and cloudiest[1] != -1:
                    double += 1
                    clouded_repartition[cloudiest[0]] += 1
                    clouded_repartition[cloudiest[1]] += 1

    return split_cloudy_set(cloudy_pixels, N_prop)


def get_cloudy_dataset(dataset_to_check, from_cloudy_percentage,
                       series, data_path, N_prop):

    cloudy_pixels = {}

    for line in series.itertuples():
        name = line[1]

        split = name.split('-')
        head, row, col = '-'.join(split[:5]), int(split[7]), int(split[8].split('.')[0])
        if head in dataset_to_check:
            if dataset_to_check[head][row, col] == 1:
                if head not in cloudy_pixels:
                    cloudy_pixels[head] = np.full((33, 33, 2), -1)
                cloudy_pixels[head][row, col], proportions_array = cloudy_pixels_proportions(
                    line, from_cloudy_percentage, data_path)

    train_mask_cloudy, val_mask_cloudy, test_mask_cloudy = get_train_val_test_cloudy(
        cloudy_pixels, proportions_array, N_prop
    )

    return train_mask_cloudy, val_mask_cloudy, test_mask_cloudy


def remove_dataset_from_another(dataset_to_clean, dataset_to_remove):
    for key, _ in dataset_to_clean.items():
        for i in range(33):
            for j in range(33):
                if key in dataset_to_remove and dataset_to_remove[key][i, j] == 1:
                    dataset_to_clean[key][i, j] = 0

    return dataset_to_clean


def get_dataset_size(grids):
    size = 0
    for key, _ in grids.items():
        for i in range(33):
            for j in range(33):
                if grids[key][i, j] == 1:
                    size += 1

    return size


def check_no_duplication(train_data_set,
                         val_regular_dataset,
                         test_regular_dataset,
                         val_mask_cloudy,
                         test_mask_cloudy
                         ):
    for key, _ in train_data_set.items():
        for i in range(32):
            for j in range(32):
                if train_data_set[key][i, j] == 1:
                    if key in val_regular_dataset:
                        if val_regular_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and "
                                f"val_regular_dataset dataset at {key} index {i}{j}",
                            )
                    if key in test_regular_dataset:
                        if test_regular_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and "
                                f"test_regular_dataset dataset at {key} index {i}{j}",
                            )
                    if key in val_mask_cloudy and val_mask_cloudy[key][i, j] == 1:
                        raise ValueError(
                            f"You have commun data between your train and "
                            f"val_mask_cloudy dataset at {key} index {i}{j}",
                        )
                    if key in test_mask_cloudy and test_mask_cloudy[key][i, j] == 1:
                        raise ValueError(
                            f"You have commun data between your train and "
                            f"test_mask_cloudy dataset at {key} index {i}{j}",
                        )


def check_no_duplication_val_test(dataset_1, dataset_2):
    for key, _ in dataset_1.items():
        for i in range(33):
            for j in range(33):
                if key in dataset_2:
                    if dataset_1[key][i, j] == 1 and dataset_2[key][i, j] == 1:
                        raise ValueError(
                            "You have commun data between your "
                            "validation and testing dataset")


def final_checkers(train_data_set,
                   val_regular_dataset,
                   test_regular_dataset,
                   val_mask_cloudy,
                   test_mask_cloudy,
                   data_path):

    check_no_duplication(
        train_data_set,
        val_regular_dataset,
        test_regular_dataset,
        val_mask_cloudy,
        test_mask_cloudy,
    )

    check_no_duplication_val_test(val_regular_dataset, test_regular_dataset)
    check_no_duplication_val_test(val_mask_cloudy, test_mask_cloudy)


def get_one_of_two_datas(dataset):
    cuted_dataset = {}
    one_of_two = True
    for key, _ in dataset.items():
        for i in range(33):
            for j in range(33):
                if dataset[key][i, j] == 1:
                    if one_of_two:
                        if key in cuted_dataset:
                            cuted_dataset[key][i, j] = 1
                            one_of_two = False
                        else:
                            cuted_dataset[key] = np.zeros((33, 33))
                            cuted_dataset[key][i, j] = 1
                            one_of_two = False
                    else:
                        one_of_two = True

    original_cuted_dataset = remove_dataset_from_another(dataset, cuted_dataset)
    return cuted_dataset, original_cuted_dataset


def split_val_test(regular_dataset):
    val_regular_dataset = {}
    test_regular_dataset = {}

    val_regular_dataset, test_regular_dataset = get_one_of_two_datas(regular_dataset)

    return (val_regular_dataset, test_regular_dataset)


def clean_empty_key(array):
    data_seen = False
    for element in array:
        keys_to_erase = []
        for key, _ in element.items():
            for i in range(33):
                for j in range(33):
                    if element[key][i, j] == 1:
                        data_seen = True
            if data_seen is False:
                keys_to_erase.append(key)
            data_seen = False

        for k in keys_to_erase:
            element.pop(k)


def create_csv(data_path, datasets):
    dir = data_path

    for dataset_name, _ in datasets.items():
        with open(os.path.join(dir, dataset_name), "w", newline="") as csvfile:
            # Création de l'objet writer CSV
            writer = csv.writer(csvfile)

            writer.writerow(["0", "1", "2"])

            for key, _ in datasets[dataset_name].items():
                for i in range(33):
                    for j in range(33):
                        if datasets[dataset_name][key][i, j] == 1:
                            writer.writerow(
                                [
                                    f"{key}-0-0-{i}-{j}.tiff",
                                    f"{key}-1-0-{i}-{j}.tiff",
                                    f"{key}-2-0-{i}-{j}.tiff",
                                ],
                            )


def size_check(data_path, file):
    dir = data_path

    csv_path = osp.join(dir, file)

    series = pd.read_csv(csv_path)

    grids, _ = grid_exploration(series)

    return get_dataset_size(grids)


def get_all_grids(data_path, array):
    dir = data_path

    grids_array = {}
    for el in array:
        csv_path = osp.join(dir, el)

        series = pd.read_csv(csv_path)

        grids, _ = grid_exploration(series)

        grids_array[el] = grids

    return grids_array


def test_csv(data_path):

    # print(
    #     "Check size for train dataset taken from CSV",
    #     size_check(data_path, "train_regular.csv"),
    # )
    # print(
    #     "Check size for train cloudy dataset taken from CSV",
    #     size_check(data_path, "train_cloudy.csv"),
    # )

    # print("Check size for val regular dataset taken from CSV = ", size_check(data_path,
    #                                                                          "validation_regular.csv"))
    # print("Check size for test regular dataset taken from CSV = ", size_check(data_path,
    #                                                                           "test_regular.csv"))

    # print("Check size for val_mask_cloudy dataset taken from CSV = ", size_check(data_path,
    #                                                                              "validation_mask_cloudy.csv"))
    # print("Check size for test_mask_cloudy dataset taken from CSV = ", size_check(data_path,
    #                                                                               "test_mask_cloudy.csv"))

    grids_array = get_all_grids(data_path, ["train_regular.csv",
                                            "train_cloudy.csv",
                                            "validation_regular.csv",
                                            "test_regular.csv",
                                            "validation_mask_cloudy.csv",
                                            "test_mask_cloudy.csv"])

    final_checkers(grids_array["train_regular.csv"],
                   grids_array["validation_regular.csv"],
                   grids_array["test_regular.csv"],
                   grids_array["validation_mask_cloudy.csv"],
                   grids_array["test_mask_cloudy.csv"],
                   data_path)
    final_checkers(grids_array["train_cloudy.csv"],
                   grids_array["validation_regular.csv"],
                   grids_array["test_regular.csv"],
                   grids_array["validation_mask_cloudy.csv"],
                   grids_array["test_mask_cloudy.csv"],
                   data_path)


def join_dataset(dataset_1, dataset_2):
    # All in join in the dataset_2
    for key, _ in dataset_1.items():
        for i in range(33):
            for j in range(33):
                if dataset_1[key][i][j] == 1:
                    if key not in dataset_2:
                        dataset_2[key] = np.zeros((33, 33))
                    dataset_2[key][i][j] = 1
    return dataset_2


def get_train_cloudy_dataset(train_data_set, from_cloudy_percentage, DATA_PATH):
    dataset1, dataset2, dataset3 = get_cloudy_dataset(
        train_data_set, from_cloudy_percentage, DATA_PATH)

    final_dataset = join_dataset(dataset1, dataset2)
    final_dataset = join_dataset(dataset3, final_dataset)

    return final_dataset


if __name__ == "__main__":
    DATA_PATH = "../data"
    VALIDATION_TEST_PERCENT = 20  # Size of the VALIDATION and TEST dataset in Percent
    GRID_SHOW = False
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_cloudy_percentage', type=int, required=True,
                        default=2.5)
    parser.add_argument('--prop_cloudy_val_test', type=int, required=True,
                        default=20)
    args = parser.parse_args()

    from_cloudy_percentage = args.from_cloudy_percentage / 100

    prop_cloudy_val_test = args.prop_cloudy_val_test

    csv_path = osp.join(DATA_PATH, "image_series.csv")
    series = pd.read_csv(csv_path)

    grids, grids_size = grid_exploration(series)

    train_cloudy_dataset, val_cloudy_dataset, test_cloudy_dataset = get_cloudy_dataset(
        grids, from_cloudy_percentage, series, DATA_PATH, prop_cloudy_val_test)

    grids = remove_dataset_from_another(grids, train_cloudy_dataset)
    grids = remove_dataset_from_another(grids, val_cloudy_dataset)
    grids = remove_dataset_from_another(grids, test_cloudy_dataset)

    val_test_regular_dataset, number_of_data_for_valtest_for_each_grids = get_datas_arround_missing_datas(
        grids, VALIDATION_TEST_PERCENT, grids_size)

    train_dataset = remove_dataset_from_another(grids, val_test_regular_dataset)

    val_regular_dataset, test_regular_dataset = split_val_test(val_test_regular_dataset)

    train_dataset = remove_dataset_from_another(grids, val_test_regular_dataset)

    print("========")
    print("Train dataset size =", get_dataset_size(train_dataset))
    print("Train cloudy dataset size =", get_dataset_size(train_cloudy_dataset))
    print("Val regular dataset size = ", get_dataset_size(val_regular_dataset))
    print("Test regular dataset size = ", get_dataset_size(test_regular_dataset))
    print("val_mask_cloudy_dataset size", get_dataset_size(val_cloudy_dataset))
    print("test_mask_cloudy_dataset size", get_dataset_size(test_cloudy_dataset))
    print("========")

    clean_empty_key([
                    val_regular_dataset,
                    test_regular_dataset])

    final_checkers(
        train_dataset,
        val_regular_dataset,
        test_regular_dataset,
        val_cloudy_dataset,
        test_cloudy_dataset,
        DATA_PATH,
    )

    final_checkers(
        train_cloudy_dataset,
        val_regular_dataset,
        test_regular_dataset,
        val_cloudy_dataset,
        test_cloudy_dataset,
        DATA_PATH)

    create_csv(DATA_PATH,
               {"train_regular.csv": train_dataset,
                "train_cloudy.csv": train_cloudy_dataset,
                "validation_regular.csv": val_regular_dataset,
                "test_regular.csv": test_regular_dataset,
                "validation_mask_cloudy.csv": val_cloudy_dataset,
                "test_mask_cloudy.csv": test_cloudy_dataset})

    test_csv(DATA_PATH)

    if GRID_SHOW:
        grid_show(train_dataset, "Train")
        grid_show(train_cloudy_dataset, "Train clo")
        grid_show(val_regular_dataset, "Val reg")
        grid_show(test_regular_dataset, "Test reg")
        grid_show(val_cloudy_dataset, "Val clo")
        grid_show(test_cloudy_dataset, "Test clo")

    print(f"FROM CLOUDY PERCENTAGE : {from_cloudy_percentage * 100}% ")
    print(f"PROPORTION CLOUDY VAL TEST : {prop_cloudy_val_test}% ")

    if GRID_SHOW:
        plt.ion()
        plt.show()
        plt.pause(0.001)
        input("\nPress [enter] to continue and close all the windows.")
