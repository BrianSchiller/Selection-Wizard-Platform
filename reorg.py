#!/usr/bin/env python3
"""Copy IOH performance and meta data files to directories per BBOB problem.

NOTE: Only run from the raw data directory with results from PBS cluster.
"""
from pathlib import Path
import shutil
import json
import argparse


def move_ma_csvs(algo_dirs: list[Path]) -> None:
    """Organise MA-BBOB directories by algorithm-dimension-budget combination.

    Args:
        algo_dirs:
            List of algorithm directories to move. Each of these directories
            should have a .csv with preprocessed data, and a .zip with the raw
            data over all considered MA-BBOB problems.
    """
    current_dir = Path.cwd()
    target_dir = current_dir.with_name(f"{current_dir.name}_organised")
    target_dir.mkdir(exist_ok=True)

    for algo_dir in algo_dirs:
        # Get the name as the directory name, without the _processed component
        combination_name = str(algo_dir.name).removesuffix("_processed")

        # Copy the data.csv file from the directory to <name>.csv
        source = algo_dir / "data.csv"
        destination = target_dir / Path(f"{combination_name}.csv")

        print(f"Copy {source}")
        print(f"to   {destination}")
        shutil.copy(source, destination)

    return


def move_per_dim_bud(algo_dirs: list[Path]) -> None:
    """Organise directories by algorithm-dimension-budget combination.

    Args:
        algo_dirs:
            List of algorithm directories to move. Each of these directories
            should have IOH style .json files and subdirectories for each BBOB
            problem.
    """
    for algo_dir in algo_dirs:
        # Get needed scenario metadata
        # (should be the same for all .json files in the directory)
        json_path = algo_dir / "IOHprofiler_f1_Sphere.json"

        with json_path.open() as metadata_file:
            metadata = json.load(metadata_file)

        dims = metadata["scenarios"][0]["dimension"]
        bud = metadata["scenarios"][0]["runs"][0]["evals"]

        # Copy dir to new location
        algo_name = algo_dir.name
        destination = Path(f"{algo_name}-{dims}-{bud}")
        shutil.copytree(algo_dir, destination)

        print(f"Copy {algo_dir}")
        print(f"to   {destination}")

    return


def move_per_function(algo_dirs: list[Path]) -> None:
    """Organise directories by problem.

    Args:
        algo_dirs:
            List of algorithm directories to move. Each of these directories
            should have a single IOH style .json file and subdirectory for a
            BBOB problem.
    """
    # Get function names per directory to be moved
    func_names = []

    for algo_dir in algo_dirs:
        func_list = list(algo_dir.glob("data_*"))
        name = func_list[0].name.removeprefix("data_")
        func_names.append(name)

        # Copy dir to new location
        func_dir = Path(name)
        func_dir.mkdir(exist_ok=True)
        destination = func_dir / algo_dir.name
        shutil.copytree(algo_dir, destination)

        print(f"Copy {name} to {destination}")
        print(f"From: {algo_dir}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data_style",
        choices=["per_problem", "per_dim_bud", "ma"],
        default="per_problem",
        type=str,
        help="How the data directory is organised. Can be:\n"
             "- per_problem which "
             "has numbered directories each with a single problem and data for"
             " all dimensionalities, like: data_seeds2/0/CMA/\n"
             "- per_dim_bud which has numbered directories each with a single "
             "algorithm and data for each problem for a specific dimension and"
             "budget combination like: data_seeds2_ngopt/0/MetaModel/\n"
             "- ma which has numbered directories each with a single algorithm"
             "and a preprocessed csv file like: "
             "data_seeds2_ma/2/Cobyla_D25_B1000_processed/data.csv")
    args = parser.parse_args()

    # Get PBS batch job directories, exclude log
    this_dir = Path(".")
    pbs_dirs = [child for child in this_dir.iterdir() if child.is_dir()
                and not str(child.name).startswith("log")]
    pbs_dirs.sort()

    # Get directories that need to be moved excluding __pycache__ and csvs
    algo_dirs = []

    for child in pbs_dirs:
        algo_paths = [subd for subd in child.iterdir()
                      if subd.is_dir() and not str(subd.name).startswith("__")
                      and not str(subd.name).startswith("csvs")]

        for sub_child in algo_paths:
            algo_dirs.append(sub_child)

    if args.data_style == "per_problem":
        move_per_function(algo_dirs)
    elif args.data_style == "per_dim_bud":
        move_per_dim_bud(algo_dirs)
    elif args.data_style == "ma":
        move_ma_csvs(algo_dirs)
