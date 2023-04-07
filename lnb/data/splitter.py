"""Train/validation/test data splitter functions."""
import copy
import csv
import os
import os.path as osp
import shutil

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

    plt.xlabel('Valeurs')
    plt.ylabel('Pourcentage')
    plt.title(f'Courbe de distribution {tittle}')
    plt.show(block=False)


def grid_exploration(series):
    """Grid exploration."""
    grids = {}
    grids_size = {}
    for line in series.itertuples():
        name = line[1]
        split = name.split('-')
        head, row, col = '-'.join(split[:5]), int(split[7]), int(split[8].split('.')[0])
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
        if grids_name == "KALININGRAD_GUSEV_2018-04-07_2018-04-19":
            grids_valtest_size[grids_name] = size
        else:
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
    if data_index_0 < 31:
        if (valid_datas_index[data_index_0 + 2, data_index_1] == 1
                and grid[data_index_0 + 1, data_index_1] == 1):
            if valid_datas_index[data_index_0 + 1][data_index_1] == 0:
                valid_datas_index[data_index_0 + 1][data_index_1] = 1
                return True
    if data_index_0 > 1:
        if (valid_datas_index[data_index_0 - 2, data_index_1] == 1
                and grid[data_index_0 - 1, data_index_1] == 1):
            if valid_datas_index[data_index_0 - 1][data_index_1] == 0:
                valid_datas_index[data_index_0 - 1][data_index_1] = 1
                return True
    if data_index_1 < 31:
        if (valid_datas_index[data_index_0, data_index_1 + 2] == 1
                and grid[data_index_0, data_index_1 + 1] == 1):
            if valid_datas_index[data_index_0][data_index_1 + 1] == 0:
                valid_datas_index[data_index_0][data_index_1 + 1] = 1
                return True
    if data_index_1 > 1:
        if (valid_datas_index[data_index_0, data_index_1 - 2] == 1
                and grid[data_index_0, data_index_1 - 1] == 1):
            if valid_datas_index[data_index_0][data_index_1 - 1] == 0:
                valid_datas_index[data_index_0][data_index_1 - 1] = 1
                return True


def get_border_data(grid, number_of_data_for_valtest_for_a_grid,
                    valid_datas_index, expected_grid_validation_test_size):
    """Get border data."""

    down_left_corner_limits, up_right_corner_limits = get_data_corner_limits(grid)

    for row in range(down_left_corner_limits[0], up_right_corner_limits[0] + 1):
        for col in range(up_right_corner_limits[1], down_left_corner_limits[1] + 1):
            if (grid[row, col] == 1 and valid_datas_index[row, col] == 0
                    and number_of_data_for_valtest_for_a_grid
                    < expected_grid_validation_test_size):
                valid_datas_index[row, col] = 1
                number_of_data_for_valtest_for_a_grid += 1

    return number_of_data_for_valtest_for_a_grid, valid_datas_index


def valid_datas_arround_missing_datas(grid, missing_data_indexes,
                                      expected_grid_validation_test_size,
                                      valid_datas_index,
                                      number_of_data_for_valtest_for_a_grid):
    """
    Get the validation/test data from near to the more missing data possible
    and then get the data around (of the missing data)
    and then get the data from the border if it's not enough
    """

    for _ in range(16):
        for data_index in missing_data_indexes:
            if (number_of_data_for_valtest_for_a_grid
                    < expected_grid_validation_test_size):
                # Perform a vertical and horizontal cross check
                # to get the valid data around the missing data
                if data_index[0] < 32:
                    if grid[data_index[0] + 1, data_index[1]] == 1:
                        if valid_datas_index[data_index[0] + 1][data_index[1]] == 0:
                            valid_datas_index[data_index[0] + 1][data_index[1]] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] + 1, data_index[1]):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
                if data_index[0] > 0:
                    if grid[data_index[0] - 1, data_index[1]] == 1:
                        if valid_datas_index[data_index[0] - 1][data_index[1]] == 0:
                            valid_datas_index[data_index[0] - 1][data_index[1]] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index, data_index[0]
                                                 - 1, data_index[1]):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
                if data_index[1] < 32:
                    if grid[data_index[0], data_index[1] + 1] == 1:
                        if valid_datas_index[data_index[0]][data_index[1] + 1] == 0:
                            valid_datas_index[data_index[0]][data_index[1] + 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0], data_index[1] + 1):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
                if data_index[1] > 0:
                    if grid[data_index[0], data_index[1] - 1] == 1:
                        if valid_datas_index[data_index[0]][data_index[1] - 1] == 0:
                            valid_datas_index[data_index[0]][data_index[1] - 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0], data_index[1] - 1):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue

                # Perform a diagonal cross to get the valid data
                # around the missing data
                if data_index[0] < 32 and data_index[1] > 0:
                    if grid[data_index[0] + 1, data_index[1] - 1] == 1:
                        if valid_datas_index[data_index[0] + 1][data_index[1] - 1] == 0:
                            valid_datas_index[data_index[0] + 1][data_index[1] - 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] + 1,
                                                 data_index[1] - 1):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
                if data_index[0] > 0 and data_index[1] > 0:
                    if grid[data_index[0] - 1, data_index[1] - 1] == 1:
                        if valid_datas_index[data_index[0] - 1][data_index[1] - 1] == 0:
                            valid_datas_index[data_index[0] - 1][data_index[1] - 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] - 1,
                                                 data_index[1] - 1):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
                if data_index[1] < 32 and data_index[0] > 0:
                    if grid[data_index[0] - 1, data_index[1] + 1] == 1:
                        if valid_datas_index[data_index[0] - 1][data_index[1] + 1] == 0:
                            valid_datas_index[data_index[0] - 1][data_index[1] + 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] - 1,
                                                 data_index[1] + 1):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue
                if data_index[0] < 32 and data_index[1] < 32:
                    if grid[data_index[0] + 1, data_index[1] + 1] == 1:
                        if valid_datas_index[data_index[0] + 1][data_index[1] + 1] == 0:
                            valid_datas_index[data_index[0] + 1][data_index[1] + 1] = 1
                            number_of_data_for_valtest_for_a_grid += 1
                        if cross_check_for_whole(grid, valid_datas_index,
                                                 data_index[0] + 1,
                                                 data_index[1] + 1):
                            number_of_data_for_valtest_for_a_grid += 1
                            continue

    return valid_datas_index, number_of_data_for_valtest_for_a_grid


def get_datas_arround_missing_datas(grids, expected_grids_validation_test_size):
    validation_test_data_for_each_grid = {}
    valtest_data_num_per_grid = {}

    for grid_name, grid in grids.items():
        if grid_name == "KALININGRAD_GUSEV_2018-04-07_2018-04-19":
            validation_test_data_for_each_grid[grid_name] = copy.deepcopy(
                grids[grid_name]
            )
        else:
            validation_test_data_for_each_grid[grid_name] = np.zeros((33, 33),
                                                                     dtype=np.uint)
            valtest_data_num_per_grid[grid_name] = 0
            missing_datas_indexes, missing_size = get_missing_data_indexes(grid)

            print(grid_name, "has", missing_size, "missing data")

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


def split_train_val_test(validation_test_percent, data_path):

    csv_path = osp.join(data_path, 'image_series.csv')
    series = pd.read_csv(csv_path)

    grids, grids_size = grid_exploration(series)

    expected_grids_validation_test_size, potential_total_val_test_data_size = val_test_proportion_for_each_grid(
        validation_test_percent, grids_size)

    real_grids_validation_test, number_of_data_for_valtest_for_each_grids = get_datas_arround_missing_datas(
        grids, expected_grids_validation_test_size)

    number_of_data_for_valtest_for_each_grids["KALININGRAD_GUSEV_2018-04-07_2018-04-19"] = grids_size["KALININGRAD_GUSEV_2018-04-07_2018-04-19"]

    for grids_name, size in grids_size.items():
        print("===")
        print(grids_name)
        print("size =", size)
        print("expected test_val_size =", expected_grids_validation_test_size[grids_name])
        print("real_grids_validation_test_size =",
              number_of_data_for_valtest_for_each_grids[grids_name])
        print("===")

    print("========")
    print(
        "Based on each grid size expected total_val_test_data_size (KALININGRAD_GUSEV_2018-04-07_2018-04-19 is fully taken) =",
        potential_total_val_test_data_size)

    real_test_val_dataset_total_size = 0
    for _, dataset_size in number_of_data_for_valtest_for_each_grids.items():
        real_test_val_dataset_total_size += dataset_size

    print("========")
    print("Real total_val_test_data_size =", real_test_val_dataset_total_size)
    print("========")

    train_dataset = remove_dataset_from_another(grids, real_grids_validation_test)

    train_dataset.pop("KALININGRAD_GUSEV_2018-04-07_2018-04-19")

    return real_grids_validation_test, real_test_val_dataset_total_size, grids_size, train_dataset


def average_absolute_differences(line, data_path):
    s1_path = osp.join(data_path, 's1')  # contains VV and VH

    t_vv = tif.imread(osp.join(s1_path, line[3]))[..., 0]  # VV
    t_vh = tif.imread(osp.join(s1_path, line[3]))[..., 1]  # VH

    t1_vv = tif.imread(osp.join(s1_path, line[2]))[..., 0]  # VV
    t1_vh = tif.imread(osp.join(s1_path, line[2]))[..., 1]  # VH

    vv_differences = np.mean(np.absolute(t_vv - t1_vv))
    vh_differences = np.mean(np.absolute(t_vh - t1_vh))

    # print(vv_differences, vh_differences)

    return vv_differences, vh_differences


def get_vh_vv_differences(differences):

    s1_difference_dataset = {}
    for o, difference in enumerate(differences):
        conca_array = None
        no_in_dataset = 0
        in_dataset = 0

        """
        Segment correpond to the limit value beyond witch the difference is considered HIGH, first fort VV and second for VH
        """
        if o == 0:
            segment = 1.9
        else:
            segment = 1.6

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

        if o == 0:
            s1_type = "VV"
        else:
            s1_type = "VH"

        segment_end = np.round(conca_array.max(), decimals=2)

        proportion_in = np.round(100 / (no_in_dataset + in_dataset) * in_dataset, decimals=2)

        proportion_no_in = np.round(100 / (no_in_dataset + in_dataset) * no_in_dataset, decimals=2)

        print(f"{s1_type} difference entre 0 et {segment} = {no_in_dataset} => {proportion_no_in} %")
        print(f"{s1_type} difference entre {segment} et {segment_end} = {in_dataset} => {proportion_in} %")

        # goss_show(conca_array, f"{s1_type}_differences entre t et t-1")

    # Show the difference from the grid view
    # grid_show(differences[0], "VV differences t et t-1")
    # grid_show(differences[1], "VH differences t et t-1")

    return s1_difference_dataset


def get_s1_differences_dataset(dataset_to_check, data_path):

    csv_path = osp.join(data_path, 'image_series.csv')
    series = pd.read_csv(csv_path)

    vv_differences = {}
    vh_differences = {}

    for line in series.itertuples():
        name = line[1]
        split = name.split('-')
        head, row, col = '-'.join(split[:5]), int(split[7]), int(split[8].split('.')[0])
        if head in dataset_to_check:
            if dataset_to_check[head][row, col] == 1:
                if head not in vv_differences or head not in vv_differences:
                    vv_differences[head] = np.zeros((33, 33))
                    vh_differences[head] = np.zeros((33, 33))
                vv_differences[head][row, col], vh_differences[head][row,
                                                                     col] = average_absolute_differences(line, data_path)

    s1_difference_dataset = get_vh_vv_differences([vv_differences, vh_differences])

    return s1_difference_dataset


def remove_dataset_from_another(dataset_to_clean, dataset_to_remove):
    for key, _ in dataset_to_clean.items():
        for i in range(33):
            for j in range(33):
                if key in dataset_to_remove:
                    if dataset_to_remove[key][i, j] == 1:
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


def get_datasets_for_metrics(val_test_dataset, val_test_dataset_total_size, grids_size, data_path):

    # Generalisation dataset for the generalisation metrics
    kalingrad_copy = copy.copy(val_test_dataset["KALININGRAD_GUSEV_2018-04-07_2018-04-19"])
    generalisation_dataset = {
        "KALININGRAD_GUSEV_2018-04-07_2018-04-19": kalingrad_copy}

    generalisation_size = get_dataset_size(generalisation_dataset)
    print("Generalisation dataset size =", generalisation_size)

    # Data set for the netrics that give us the performance when the t et t-1
    # s1 data are very different
    s1_difference_dataset = get_s1_differences_dataset(val_test_dataset, data_path)

    return generalisation_dataset, s1_difference_dataset, val_test_dataset


def check_no_duplication(train_data_set,
                         val_generalisation_dataset,
                         test_generalisation_dataset,
                         val_s1_difference_dataset,
                         test_s1_difference_dataset,
                         val_regular_dataset,
                         test_regular_dataset):

    for key, _ in train_data_set.items():
        for i in range(32):
            for j in range(32):
                if train_data_set[key][i, j] == 1:
                    if key in val_generalisation_dataset:
                        if val_generalisation_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and val_generalisation_dataset at {key} index {i}{j}")
                    if key in test_generalisation_dataset:
                        if test_generalisation_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and test_generalisation_dataset dataset at {key} index {i}{j}")
                    if key in val_s1_difference_dataset:
                        if val_s1_difference_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and val_s1_difference_dataset dataset at {key} index {i}{j}")
                    if key in test_s1_difference_dataset:
                        if test_s1_difference_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and test_s1_difference_dataset at {key} index {i}{j}")
                    if key in val_regular_dataset:
                        if val_regular_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and val_regular_dataset dataset at {key} index {i}{j}")
                    if key in test_regular_dataset:
                        if test_regular_dataset[key][i, j] == 1:
                            raise ValueError(
                                f"You have commun data between your train and test_regular_dataset dataset at {key} index {i}{j}")

    print("========")
    print("You do not have duplication between your train dataset and your test/validation datasets")
    print("========")


def check_no_duplication_val_test(dataset_1, dataset_2):
    for key, _ in dataset_1.items():
        for i in range(33):
            for j in range(33):

                if dataset_1[key][i, j] == 1 and dataset_2[key][i, j] == 1:
                    raise ValueError(
                        "You have commun data between your validation and testing dataset")


def final_checkers(train_data_set,
                   val_generalisation_dataset,
                   test_generalisation_dataset,
                   val_s1_difference_dataset,
                   test_s1_difference_dataset,
                   val_regular_dataset,
                   test_regular_dataset,
                   data_path):

    train_difference_dataset = get_s1_differences_dataset(train_data_set, data_path)

    # grid_show(train_difference_dataset, "TRAIN DIFF DATASET")

    print("========")
    print("Train difference dataset size =", get_dataset_size(train_difference_dataset))

    check_no_duplication(
        train_data_set,
        val_generalisation_dataset,
        test_generalisation_dataset,
        val_s1_difference_dataset,
        test_s1_difference_dataset,
        val_regular_dataset,
        test_regular_dataset)

    check_no_duplication_val_test(val_generalisation_dataset, test_generalisation_dataset)
    check_no_duplication_val_test(val_s1_difference_dataset, test_s1_difference_dataset)
    check_no_duplication_val_test(val_regular_dataset, test_regular_dataset)

    print("You dont have duplicate date between your validations and tests datasets")
    print("========")


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


def split_val_test(generalisation_dataset, s1_difference_dataset, regular_dataset):
    val_generalisation_dataset = {}
    test_generalisation_dataset = {}
    val_s1_difference_dataset = {}
    test_s1_difference_dataset = {}
    val_regular_dataset = {}
    test_regular_dataset = {}

    val_generalisation_dataset, test_generalisation_dataset = get_one_of_two_datas(
        generalisation_dataset)

    val_s1_difference_dataset, test_s1_difference_dataset = get_one_of_two_datas(
        s1_difference_dataset)

    val_regular_dataset, test_regular_dataset = get_one_of_two_datas(
        regular_dataset)

    print("========")
    print("test_generalisation_dataset size", get_dataset_size(test_generalisation_dataset))
    print("val_generalisation_dataset size", get_dataset_size(val_generalisation_dataset))

    print("test_s1_difference_dataset size", get_dataset_size(test_s1_difference_dataset))
    print("val_s1_difference_dataset size", get_dataset_size(val_s1_difference_dataset))

    print("test_regular_dataset size", get_dataset_size(test_regular_dataset))
    print("val_regular_dataset size", get_dataset_size(val_regular_dataset))
    print("========")

    return val_generalisation_dataset, test_generalisation_dataset, val_s1_difference_dataset, test_s1_difference_dataset, val_regular_dataset, test_regular_dataset


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


def create_csv(datasets):

    dir = f"{os.path.join('../../', 'csv')}"

    if os.path.exists(dir):
        shutil.rmtree(dir)

    os.mkdir(dir)

    for dataset_name, _ in datasets.items():

        with open(os.path.join(dir, dataset_name), 'w', newline='') as csvfile:
            # Création de l'objet writer CSV
            writer = csv.writer(csvfile)

            writer.writerow(["0", "1", "2"])

            for key, _ in datasets[dataset_name].items():
                for i in range(33):
                    for j in range(33):
                        if datasets[dataset_name][key][i, j] == 1:
                            writer.writerow(
                                [f"{key}-0-0-{i}-{j}.tiff", f"{key}-1-0-{i}-{j}.tiff", f"{key}-2-0-{i}-{j}.tiff"])


def size_check(file):

    dir = f"{os.path.join('../../', 'csv')}"

    csv_path = osp.join(dir, file)

    series = pd.read_csv(csv_path)

    grids, _ = grid_exploration(series)

    return get_dataset_size(grids)


def get_all_grids(array):

    dir = f"{os.path.join('../../', 'csv')}"

    grids_array = {}
    for el in array:

        csv_path = osp.join(dir, el)

        series = pd.read_csv(csv_path)

        grids, _ = grid_exploration(series)

        grids_array[el] = grids

    return grids_array


def test_csv(data_path):
    print("===============\n")
    print("FROM HERE START THE CSV DATA CHECKER\n")
    print("===============")

    print("Check size for train dataset taken from CSV",
          size_check("train_regular.csv"))

    print("Check size for val generalisation dataset taken from CSV = ", size_check(
        "validation_generalisation.csv"))
    print("Check size for test generalisation dataset taken from CSV = ", size_check(
        "test_generalisation.csv"))

    print("Check size for val s1 difference dataset taken from CSV = ", size_check(
        "validation_s1_difference.csv"))
    print("Check size for test s1 difference dataset taken from CSV = ", size_check(
        "test_s1_difference.csv"))

    print("Check size for val regular dataset taken from CSV = ", size_check(
        "validation_regular.csv"))
    print("Check size for test regular dataset taken from CSV = ", size_check(
        "test_regular.csv"))

    grids_array = get_all_grids(["train_regular.csv",
                                 "validation_generalisation.csv",
                                 "test_generalisation.csv",
                                 "validation_s1_difference.csv",
                                 "test_s1_difference.csv",
                                 "validation_regular.csv",
                                 "test_regular.csv"])

    final_checkers(grids_array["train_regular.csv"],
                   grids_array["validation_generalisation.csv"],
                   grids_array["test_generalisation.csv"],
                   grids_array["validation_s1_difference.csv"],
                   grids_array["test_s1_difference.csv"],
                   grids_array["validation_regular.csv"],
                   grids_array["test_regular.csv"],
                   data_path)


if __name__ == '__main__':
    DATA_PATH = '../../../data'
    VALIDATION_TEST_PERCENT = 20  # Size of the VALIDATION and TEST dataset in Percent
    GRID_SHOW = True

    val_test_dataset, val_test_dataset_total_size, grids_size, train_data_set = split_train_val_test(
        VALIDATION_TEST_PERCENT, DATA_PATH)

    generalisation_dataset, s1_difference_dataset, regular_dataset = get_datasets_for_metrics(
        val_test_dataset, val_test_dataset_total_size, grids_size, DATA_PATH)

    print("========")
    print("Train dataset size =", get_dataset_size(train_data_set))
    print("Generalisation dataset size =", get_dataset_size(generalisation_dataset))
    print("Difference dataset size =", get_dataset_size(s1_difference_dataset))
    print("Regular dataset size = ", get_dataset_size(regular_dataset))
    print("========")

    val_generalisation_dataset, test_generalisation_dataset, val_s1_difference_dataset, test_s1_difference_dataset, val_regular_dataset, test_regular_dataset = split_val_test(
        generalisation_dataset, s1_difference_dataset, regular_dataset)

    clean_empty_key([val_generalisation_dataset,
                     test_generalisation_dataset,
                     val_s1_difference_dataset,
                     test_s1_difference_dataset,
                     val_regular_dataset,
                     test_regular_dataset])

    final_checkers(
        train_data_set,
        val_generalisation_dataset,
        test_generalisation_dataset,
        val_s1_difference_dataset,
        test_s1_difference_dataset,
        val_regular_dataset,
        test_regular_dataset,
        DATA_PATH)

    create_csv({"train_regular.csv": train_data_set,
               "validation_generalisation.csv": val_generalisation_dataset,
                "test_generalisation.csv": test_generalisation_dataset,
                "validation_s1_difference.csv": val_s1_difference_dataset,
                "test_s1_difference.csv": test_s1_difference_dataset,
                "validation_regular.csv": val_regular_dataset,
                "test_regular.csv": test_regular_dataset})

    test_csv(DATA_PATH)

    if GRID_SHOW:
        grid_show(train_data_set, "Train")
        grid_show(val_generalisation_dataset, "Val/ gen")
        grid_show(test_generalisation_dataset, "Test gen")
        grid_show(val_s1_difference_dataset, "Val dif")
        grid_show(test_s1_difference_dataset, "Test dif")
        grid_show(val_regular_dataset, "Val reg")
        grid_show(test_regular_dataset, "Test reg")

    plt.ion()
    plt.show()
    plt.pause(0.001)
    input("\nPress [enter] to continue and close all the windows.")
