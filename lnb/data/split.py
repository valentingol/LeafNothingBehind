"""Train/validation/test data splitter functions."""
import argparse
import copy
import csv
import os
import os.path as osp
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lnb.data.splitter_utils import (
    cloudy_pixels_prop,
    get_around_missing,
    grid_exploration,
    grid_show,
)


def select(
    selecter: int,
    head: int,
    row: int,
    col: int,
    train_data_set: dict,
    val_dataset: dict,
    test_dataset: dict,
    n_prop: int,
) -> int:
    """Select the dataset to add the pixel to."""
    temp = selecter
    rand = random.randint(0, 99)
    if rand < 100 - n_prop:
        selecter = 1

    if selecter in (0, 1):
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
    # NOTE: this should never happen
    return -1


def split_cloudy_set(cloudy_pixels: dict, n_prop: int) -> Tuple[dict, dict, dict]:
    """Split cloudy data in train/valid/test sets."""
    train_dataset: dict = {}
    val_dataset: dict = {}
    test_dataset: dict = {}

    selecter = [2, 2, 2]

    for head, _ in cloudy_pixels.items():
        for row in range(33):
            for col in range(33):
                cloudiest = cloudy_pixels[head][row][col]

                # Ensure that the distribution of double clouded images
                # (t-1 and t-2) is equal between the 3 datasets
                if cloudiest[0] != -1 and cloudiest[1] != -1:
                    selecter[0] = select(
                        selecter[0],
                        head,
                        row,
                        col,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        n_prop,
                    )
                # Ensure that the distribution of highly clouded images
                # (t-1 or t-2) is equal between the 3 datasets
                elif cloudiest[0] == 1 or cloudiest[1] == 1:
                    selecter[1] = select(
                        selecter[1],
                        head,
                        row,
                        col,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        n_prop,
                    )
                # Ensure that the distribution of the images that
                # are not very cloudy (t-1 or t-2) is equal between
                # the 3 datasets
                elif cloudiest[0] == 0 or cloudiest[1] == 0:
                    selecter[2] = select(
                        selecter[2],
                        head,
                        row,
                        col,
                        train_dataset,
                        val_dataset,
                        test_dataset,
                        n_prop,
                    )

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_cloudy(
    cloudy_pixels: dict,
    pro_array: list,
    n_prop: int,
) -> Tuple[dict, dict, dict]:
    """Get cloudy train/valid/test datasets."""
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

    return split_cloudy_set(cloudy_pixels, n_prop)


def get_cloudy_dataset(
    dataset_to_check: dict,
    from_cloudy_percentage: int,
    series: pd.DataFrame,
    data_path: str,
    n_prop: int,
) -> Tuple[dict, dict, dict]:
    """Get cloudy datasets."""
    cloudy_pixels = {}

    for line in series.itertuples():
        name = line[1]

        split = name.split("-")
        head, row, col = "-".join(split[:5]), int(split[7]), int(split[8].split(".")[0])
        if head in dataset_to_check and dataset_to_check[head][row, col] == 1:
            if head not in cloudy_pixels:
                cloudy_pixels[head] = np.full((33, 33, 2), -1)
            (
                cloudy_pixels[head][row, col],
                proportions_array,
            ) = cloudy_pixels_prop(line, from_cloudy_percentage, data_path)

    train_cloudy, val_cloudy, test_cloudy = get_train_val_test_cloudy(
        cloudy_pixels,
        proportions_array,
        n_prop,
    )

    return train_cloudy, val_cloudy, test_cloudy


def remove_dataset_from_another(
    dataset_to_clean: dict,
    dataset_to_remove: dict,
) -> dict:
    """Remove data already in one dataset from another dataset."""
    for key, _ in dataset_to_clean.items():
        for i in range(33):
            for j in range(33):
                if key in dataset_to_remove and dataset_to_remove[key][i, j] == 1:
                    dataset_to_clean[key][i, j] = 0

    return dataset_to_clean


def get_dataset_size(grids: dict) -> int:
    """Get dataset size."""
    size = 0
    for key, _ in grids.items():
        for i in range(33):
            for j in range(33):
                if grids[key][i, j] == 1:
                    size += 1
    return size


def check_no_duplication(  # noqa: C901
    train_data_set: dict,
    val_regular_dataset: dict,
    test_regular_dataset: dict,
    val_cloudy: dict,
    test_cloudy: dict,
) -> None:
    """Check for no duplication between all the sets.

    Raises
    ------
    ValueError
        If there is duplication between the datasets
    """
    for key, _ in train_data_set.items():
        for i in range(32):
            for j in range(32):
                if train_data_set[key][i, j] == 1:
                    if (key in val_regular_dataset
                            and val_regular_dataset[key][i, j] == 1):
                        raise ValueError(
                            f"You have commun data between your train and "
                            f"val_regular_dataset dataset at {key} index {i}{j}.",
                        )
                    if (key in test_regular_dataset
                            and test_regular_dataset[key][i, j] == 1):
                        raise ValueError(
                            f"You have commun data between your train and "
                            f"test_regular_dataset dataset at {key} index {i}{j}.",
                        )
                    if key in val_cloudy and val_cloudy[key][i, j] == 1:
                        raise ValueError(
                            f"You have commun data between your train and "
                            f"val_cloudy dataset at {key} index {i}{j}.",
                        )
                    if key in test_cloudy and test_cloudy[key][i, j] == 1:
                        raise ValueError(
                            f"You have commun data between your train and "
                            f"test_cloudy dataset at {key} index {i}{j}.",
                        )


def check_no_duplication_val_test(dataset_1: dict, dataset_2: dict) -> None:
    """Check for no duplication between val and test.

    Raises
    ------
    ValueError
        If there is duplication between the datasets.
    """
    for key, _ in dataset_1.items():
        for i in range(33):
            for j in range(33):
                if (key in dataset_2
                        and dataset_1[key][i, j] == 1
                        and dataset_2[key][i, j] == 1):
                    raise ValueError(
                        "You have commun data between your "
                        "validation and testing dataset.",
                    )


def final_checkers(
    train_data_set: dict,
    val_regular_dataset: dict,
    test_regular_dataset: dict,
    val_cloudy: dict,
    test_cloudy: dict,
) -> None:
    """Perform the final check for duplication in the datasets."""
    check_no_duplication(
        train_data_set,
        val_regular_dataset,
        test_regular_dataset,
        val_cloudy,
        test_cloudy,
    )

    check_no_duplication_val_test(val_regular_dataset, test_regular_dataset)
    check_no_duplication_val_test(val_cloudy, test_cloudy)


def get_one_of_two_datas(dataset: dict) -> Tuple[dict, dict]:
    """Cut the dataset in two datasets equally."""
    cuted_dataset: dict = {}
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


def split_val_test(regular_dataset: dict) -> Tuple[dict, dict]:
    """Split the val/test dataset in two datasets."""
    val_regular_dataset: dict = {}
    test_regular_dataset: dict = {}
    val_regular_dataset, test_regular_dataset = get_one_of_two_datas(regular_dataset)
    return val_regular_dataset, test_regular_dataset


def clean_empty_key(array: List[dict]) -> None:
    """Clean the empty keys in the datasets."""
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


def create_csv(data_path: str, datasets: dict) -> None:
    """Create the csv files for the datasets."""
    for dataset_name, _ in datasets.items():
        with open(os.path.join(data_path, dataset_name), "w", encoding="utf-8",
                  newline="") as csvfile:
            # Creation of csv writer
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


def get_all_grids(data_path: str, array: List[str]) -> dict:
    """Get all grids from the csv files."""
    grids_array = {}
    for elem in array:
        csv_path = osp.join(data_path, elem)
        series = pd.read_csv(csv_path)
        grids, _ = grid_exploration(series)
        grids_array[elem] = grids

    return grids_array


def test_csv(data_path: str) -> None:
    """Check duplication of csv files."""
    grids_array = get_all_grids(
        data_path,
        [
            "train_regular.csv",
            "train_cloudy.csv",
            "validation_regular.csv",
            "test_regular.csv",
            "validation_cloudy.csv",
            "test_cloudy.csv",
        ],
    )

    final_checkers(
        grids_array["train_regular.csv"],
        grids_array["validation_regular.csv"],
        grids_array["test_regular.csv"],
        grids_array["validation_cloudy.csv"],
        grids_array["test_cloudy.csv"],
    )
    final_checkers(
        grids_array["train_cloudy.csv"],
        grids_array["validation_regular.csv"],
        grids_array["test_regular.csv"],
        grids_array["validation_cloudy.csv"],
        grids_array["test_cloudy.csv"],
    )


def join_dataset(dataset_1: dict, dataset_2: dict) -> dict:
    """Join two datasets."""
    # All in join in the dataset_2
    for key, _ in dataset_1.items():
        for i in range(33):
            for j in range(33):
                if dataset_1[key][i][j] == 1:
                    if key not in dataset_2:
                        dataset_2[key] = np.zeros((33, 33))
                    dataset_2[key][i][j] = 1
    return dataset_2


def main() -> None:
    """Split and create the datasets."""
    print("Data splitting in progress, it may take a minute...")
    grid_show_bool = False
    random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        default="../data",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--from_cloudy_percentage",
        type=int,
        required=True,
    )
    parser.add_argument("--prop_cloudy_val_test", type=int, required=False, default=20)
    parser.add_argument(
        "--cloudy_in_VT_regular",
        type=str,
        required=False,
        default="False",
    )
    parser.add_argument(
        "--cloudy_in_train_regular",
        type=str,
        required=False,
        default="True",
    )
    args = parser.parse_args()
    data_path = args.data_path

    from_cloudy_percentage = args.from_cloudy_percentage / 100

    prop_cloudy_val_test = args.prop_cloudy_val_test

    csv_path = osp.join(data_path, args.csv_name)
    series = pd.read_csv(csv_path)

    grids, grids_size = grid_exploration(series)

    train_cloudy_dataset, val_cloudy_dataset, test_cloudy_dataset = get_cloudy_dataset(
        grids,
        from_cloudy_percentage,
        series,
        data_path,
        prop_cloudy_val_test,
    )

    grids = remove_dataset_from_another(grids, train_cloudy_dataset)
    grids = remove_dataset_from_another(grids, val_cloudy_dataset)
    grids = remove_dataset_from_another(grids, test_cloudy_dataset)

    val_test_regular_dataset, _ = get_around_missing(
        grids,
        prop_cloudy_val_test,
        grids_size,
    )

    train_dataset = remove_dataset_from_another(grids, val_test_regular_dataset)
    val_regular_dataset, test_regular_dataset = split_val_test(val_test_regular_dataset)
    train_dataset = remove_dataset_from_another(grids, val_test_regular_dataset)

    if args.cloudy_in_train_regular in ("True", "true"):
        train_dataset = join_dataset(train_cloudy_dataset, train_dataset)
    if args.cloudy_in_VT_regular in ("True", "true"):
        val_regular_dataset = join_dataset(val_cloudy_dataset, val_regular_dataset)
        test_regular_dataset = join_dataset(test_cloudy_dataset, test_regular_dataset)

    print()
    print("Train dataset size:", get_dataset_size(train_dataset))
    print("Train cloudy dataset size:", get_dataset_size(train_cloudy_dataset))
    print()
    print("Val regular dataset size:", get_dataset_size(val_regular_dataset))
    print("Test regular dataset size:", get_dataset_size(test_regular_dataset))
    print()
    print("Val cloudy dataset size:", get_dataset_size(val_cloudy_dataset))
    print("Test cloudy dataset size:", get_dataset_size(test_cloudy_dataset))
    print("========")

    clean_empty_key([val_regular_dataset, test_regular_dataset])

    final_checkers(
        train_dataset,
        val_regular_dataset,
        test_regular_dataset,
        val_cloudy_dataset,
        test_cloudy_dataset,
    )

    final_checkers(
        train_cloudy_dataset,
        val_regular_dataset,
        test_regular_dataset,
        val_cloudy_dataset,
        test_cloudy_dataset,
    )

    create_csv(
        data_path,
        {
            "train_regular.csv": train_dataset,
            "train_cloudy.csv": train_cloudy_dataset,
            "validation_regular.csv": val_regular_dataset,
            "test_regular.csv": test_regular_dataset,
            "validation_cloudy.csv": val_cloudy_dataset,
            "test_cloudy.csv": test_cloudy_dataset,
        },
    )

    test_csv(data_path)

    if grid_show_bool:
        grid_show(train_dataset, "Train")
        grid_show(train_cloudy_dataset, "Train clo")
        grid_show(val_regular_dataset, "Val reg")
        grid_show(test_regular_dataset, "Test reg")
        grid_show(val_cloudy_dataset, "Val clo")
        grid_show(test_cloudy_dataset, "Test clo")

    print("Cloud data in train regular:", args.cloudy_in_train_regular)
    print("Cloud data in val/test_regular", args.cloudy_in_VT_regular)

    print(f"FROM CLOUDY PERCENTAGE : {from_cloudy_percentage * 100}% ")
    print(f"PROPORTION CLOUDY VAL TEST : {prop_cloudy_val_test}% ")
    print("Save CSV in :")
    print(f"{os.path.join(data_path, 'train_regular.csv')}")
    print(f"{os.path.join(data_path, 'train_cloudy.csv')}")
    print(f"{os.path.join(data_path, 'validation_regular.csv')}")
    print(f"{os.path.join(data_path, 'test_regular.csv')}")
    print(f"{os.path.join(data_path, 'validation_cloudy.csv')}")
    print(f"{os.path.join(data_path, 'test_cloudy.csv')}")

    if grid_show_bool:
        plt.ion()
        plt.show()
        plt.pause(0.001)
        input("\nPress [enter] to continue and close all the windows.")


if __name__ == "__main__":
    main()
