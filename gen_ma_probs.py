#!/usr/bin/env python3
"""Module to generate CSV files with MA-BBOB problems and algorithms to use."""
from pathlib import Path

import numpy as np
import pandas as pd

import constants as const
from experiment import NGOptChoice


def write_opt_locs_csv() -> None:
    """Write a CSV with optimum locations until 100 dimensions."""
    n_dims = 100
    opt_locs = np.random.uniform(
        size=(const.N_MA_PROBLEMS, n_dims),
        low=const.LOWER_BOUND, high=const.UPPER_BOUND)
    pd.DataFrame(opt_locs).to_csv("csvs/opt_locs.csv")

    return


def write_prob_combos_csv() -> None:
    """Write a CSV with BBOB problem pairs and weights.

    Problems should cover [1,24]. For each pair of problems three weight
    combinations have to be considered. Since mirrored pairs are equivalent,
    pairings only have to be considered in one direction. (I.e., F1W0.1-F2W0.9
    is the same as F2W0.9-F1W0.1, where W indicates the weight.)
    """
    col_names = ["prob_a", "prob_b", "weight_a", "weight_b"]
    weight_vals = [0.1, 0.5, 0.9]
    weights_rev = weight_vals[::-1]

    probs_a = []
    probs_b = []
    weights_a = []
    weights_b = []

    probs_considered = const.PROBS_CONSIDERED

    for prob_a in const.PROBS_CONSIDERED:
        # Remove the first problem in the list since it is already covered by
        # the outer loop. By doing this each iteration both pairing problems
        # with themselves and mirrored pairs are avoided.
        probs_considered = probs_considered[1:]

        for prob_b in probs_considered:
            if prob_a != prob_b:
                probs_a.extend([prob_a] * len(weight_vals))
                probs_b.extend([prob_b] * len(weight_vals))
                weights_a.extend(weight_vals)
                weights_b.extend(weights_rev)

    prob_combos = pd.DataFrame(zip(probs_a, probs_b, weights_a, weights_b),
                               columns=col_names)
    out_path = Path("csvs/ma_prob_combos.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prob_combos.to_csv(out_path, index=True)

    return


def write_algo_combos_csvs() -> None:
    """Write CSV file for algorithms to run per budget-dimension pair.

    For each budget-dimension combination, five algorithms are chosen. The
    algorithm NGOpt would use is always included. In addition, the top four
    ranking algorithms are included, where NGOpt's choice is excluded from the
    ranking.
    """
    # Load NGOpt choice data
    nevergrad_version = "0.6.0"
    hsv_file = Path("ngopt_choices/dims1-100evals1-10000_separator_"
                    f"{nevergrad_version}.hsv")
    ngopt = NGOptChoice(hsv_file)

    # Load scoring based ranking data
    ranking_file = Path(f"csvs/score_rank_{nevergrad_version}.csv")
    ranking_data = pd.read_csv(ranking_file)
    ranking_data.drop("points", axis=1, inplace=True)

    # For each budget and dimension
    budgets = [dims * 100 for dims in const.DIMS_CONSIDERED]
    algos_to_run = pd.DataFrame()

    for budget in budgets:
        for dims in const.DIMS_CONSIDERED:
            # Get data for this budget and dimension
            bud_dim_condition = ((ranking_data["dimensions"] == dims)
                                 & (ranking_data["budget"] == budget))
            bud_dim_df = ranking_data.loc[bud_dim_condition].copy()

            # Retrieve the NGOpt choice and its rank from the data
            # and remove it from the data structure
            ngopt_choice = ngopt.get_ngopt_choice(dims, budget)
            ngopt_algo = bud_dim_df.loc[
                bud_dim_df["algorithm"] == ngopt_choice].copy()
            ngopt_algo["ngopt rank old"] = 0
            bud_dim_df.drop(bud_dim_df[
                bud_dim_df["algorithm"] == ngopt_choice].index, inplace=True)

            # Retrieve the (remaining) top 4 algorithms and their ranks from
            # the data. Resolve ties by taking the algorithm that appears
            # first. Since the data is sorted, this may create some bias for
            # algorithms that appear earlier, but ties a quite rare.
            bud_dim_df.sort_values("rank", inplace=True)
            best_n = 4
            best_algos = bud_dim_df.head(best_n).copy()
            best_algos["ngopt rank old"] = best_algos["rank"]

            # Add NGOpt choice and best algorithms to the to run data frame
            algos_to_run = pd.concat([algos_to_run, best_algos, ngopt_algo])

    # Add column with algorithm ID from file mapping algorithm IDs and names
    algos_file = Path(f"csvs/ngopt_algos_{nevergrad_version}.csv")
    algo_names = pd.read_csv(algos_file)
    algo_ids = [
        algo_names.loc[algo_names["short name"] == algo_name].ID.values[0]
        for algo_name in algos_to_run["algorithm"]]
    algos_to_run["algo ID"] = algo_ids

    # Sort to prioritise NGOpt choice and higher ranked algorithms
    algos_to_run.sort_values(by=["ngopt rank old"], inplace=True)
    algos_to_run.drop(columns="ngopt rank old", inplace=True)

    # Write the CSV
    out_path = Path("csvs/ma_algos.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    algos_to_run.to_csv(out_path, index=False)

    return


if __name__ == "__main__":
    write_opt_locs_csv()
    write_prob_combos_csv()
    write_algo_combos_csvs()
