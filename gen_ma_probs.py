#!/usr/bin/env python3
"""Module to generate CSV files describing MA-BBOB problems to use."""
from pathlib import Path

import numpy as np
import pandas as pd

import constants as const


def write_opt_locs_csv() -> None:
    """Write a CSV with optimum locations until 100 dimensions."""
    opt_locs = np.random.uniform(
        size=(1656, 100), low=const.LOWER_BOUND, high=const.UPPER_BOUND)
    pd.DataFrame(opt_locs).to_csv("csvs/opt_locs.csv")

    return


def write_prob_combos_csv() -> None:
    """Write a CSV with BBOB problem pairs and weights."""
    col_names = ["prob_a", "prob_b", "weight_a", "weight_b"]
    weight_vals = [0.1, 0.5, 0.9]
    weights_rev = weight_vals[::-1]

    probs_a = []
    probs_b = []
    weights_a = []
    weights_b = []

    for prob_a in const.PROBS_CONSIDERED:
        for prob_b in const.PROBS_CONSIDERED:
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
    """Write CSV files for algorithms to run per budget-dimension pair."""
    # Load NGOpt choice data
    nevergrad_version = "0.6.0"
    hsv_file = Path("ngopt_choices/dims1-100evals1-10000_separator_"
                    f"{nevergrad_version}.hsv")
    ngopt = NGOptChoice(hsv_file)

    # Get NGOpt choices to run per budget and dimension combination
    budgets = [dims * 100 for dims in const.DIMS_CONSIDERED]
    # TODO: Retrieve NGOpt choice dataframe
    # TODO: Exclude shorter runs for the same dimension for algorithms that are
    # not budget dependent
    # TODO: Write MA-BBOB NGOpt choice CSV
#    file_name = f"ngopt_choices_{nevergrad_version}"
#    ngopt.write_ngopt_choices_csv(const.DIMS_CONSIDERED, budgets, file_name)

    # Load performance data for all budgets and dimensions
    exp = Experiment(args.data_dir,
                     args.per_budget_data_dir,
                     # dimensionalities=[100, 35],
                     ng_version=nevergrad_version)

    # TODO: Retrieve best algorithm per combination dataframe
    # TODO: Exclude shorter runs for the same dimension for algorithms that are
    # not budget dependent
    # TODO: Exclude runs already covered by NGOpt choices
    # TODO: Write MA-BBOB data_1 CSV

    # Load budget-dependent performance data
    comp_data_dir = Path("data_seeds2_bud_dep_organised")
    exp.load_comparison_data(comp_data_dir)
    # TODO: Retrieve best algorithm including budget-dependent NGOpt choices
    # dataframe
    # TODO: Exclude shorter runs for the same dimension for algorithms that are
    # not budget dependent
    # TODO: Write MA-BBOB NGOpt choice CSV (if there is actually any algorithm
    # that is not already covered by the previous two CSVs)

#    file_name = f"scores_{nevergrad_version}"
#    exp.write_ranking_csv(file_name)
#    matrix = exp.get_ranking_matrix(ngopt=ngopt)
#    file_name = f"best_comparison_{nevergrad_version}"
#    exp.write_performance_comparison_csv(file_name)

    # TODO: Repeat the retrieval and writing for MA-BBOB data_n

    return


if __name__ == "__main__":
    write_opt_locs_csv()
    write_prob_combos_csv()
