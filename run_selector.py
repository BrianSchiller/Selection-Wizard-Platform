#!/usr/bin/env python3
"""Run the data-driven selector based on performance data."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_rank_data() -> pd.DataFrame:
    """Load ranking data.

    Returns:
        Pandas DataFrame with columns: dimensions, budget, algorithm, points,
        rank,ngopt rank
    """
    csv_dir = Path("csvs")
    rank_csv = csv_dir / "score_rank_0.6.0.csv"
    rank_df = pd.read_csv(rank_csv)

    return rank_df


def validate_dimensionality(dims: int, dims_pos: int) -> int:
    """Validate the dimensionality is possible, adjust it if needed.

    Args:
        dims: Number of dimensions to select an algorithm for
        dims_pos: Possible dimensionalities for the data

    Returns:
        The input dims if it is valid, the closest valid value otherwise
    """
    dims_min = np.min(dims_pos)
    dims_max = np.max(dims_pos)

    # Validate dimensionality
    if dims < dims_min:
        print(f"WARNING: Given problem dimensionality {dims} is smaller than "
              f"the minimum available {dims_min}. Selecting an algorithm for "
              "the minimum available dimensionality instead.")
        dims = dims_min
    elif dims > dims_max:
        print(f"WARNING: Given problem dimensionality {dims} is larger than "
              f"the maximum available {dims_max}. Selecting an algorithm for "
              "the maximum available dimensionality instead.")
        dims = dims_max
    elif dims not in dims_pos:
        dims_new = dims_pos[np.argmin(np.abs(np.array(dims_pos)-dims))]
        print(f"WARNING: Given problem dimensionality {dims} not available. "
              "Selecting an algorithm for the closest available "
              f"dimensionality ({dims_new}) instead.")
        # Pick the closest possible
        dims = dims_new

    return dims


def validate_budget(budget: int, buds_pos: int) -> int:
    """Validate the budget is possible, adjust it if needed.

    Args:
        budget: Evaluation budget to select an algorithm for
        buds_pos: Possible budget values for the data

    Returns:
        The input budget if it is valid, the closest valid value otherwise
    """
    buds_min = np.min(buds_pos)
    buds_max = np.max(buds_pos)

    # Validate budget
    if budget < buds_min:
        print(f"WARNING: Given problem budget {budget} is smaller than "
              f"the minimum available {buds_min}. Selecting an algorithm for "
              "the minimum available budget instead.")
        budget = buds_min
    elif budget > buds_max:
        print(f"WARNING: Given problem budget {budget} is larger than "
              f"the maximum available {buds_max}. Selecting an algorithm for "
              "the maximum available budget instead.")
        budget = buds_max
    elif budget not in buds_pos:
        buds_new = buds_pos[np.argmin(np.abs(np.array(buds_pos)-budget))]
        print(f"WARNING: Given problem budget {budget} not available. "
              "Selecting an algorithm for the closest available "
              f"budget ({buds_new}) instead.")
        # Pick the closest possible
        budget = buds_new

    return budget


def select_algorithm(budget: int, dims: int) -> None:
    """Write the chosen algorithm to the terminal.

    Args:
        budget: Evaluation budget to select an algorithm for
        dims: Number of dimensions to select an algorithm for
    """
    rank_df = load_rank_data()
    algorithm = ""
    buds_pos = rank_df["budget"].unique()
    dims_pos = rank_df["dimensions"].unique()
    dims = validate_dimensionality(dims, dims_pos)
    budget = validate_budget(budget, buds_pos)

    # Retrieve the algorithm
    algos = rank_df.loc[
        (rank_df["dimensions"] == dims)
        & (rank_df["budget"] == budget)
        & (rank_df["rank"] == 1)]

    # Prefer the NGOptChoice in case of a tie
    if 0 in algos["ngopt rank"].values:
        algorithm = algos.loc[algos["ngopt rank"] == 0, "algorithm"].values[0]
    else:
        algorithm = algos["algorithm"].values[0]

    print(algorithm)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "bud_dim",
        default=argparse.SUPPRESS,
        type=int,
        nargs=2,
        help="Budget and dimensionality for which to select an algorithm.")

    args = parser.parse_args()

    select_algorithm(args.bud_dim[0], args.bud_dim[1])

    sys.exit()
