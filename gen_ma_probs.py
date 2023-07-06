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
                probs_a.append([prob_a] * len(weight_vals))
                probs_b.append([prob_b] * len(weight_vals))
                weights_a.append(weight_vals)
                weights_b.append(weights_rev)

    prob_combos = pd.DataFrame(zip(probs_a, probs_b, weights_a, weights_b),
                               columns=col_names)
    out_path = Path("csvs/ma_prob_combos.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prob_combos.to_csv(out_path, index=True)

    return
