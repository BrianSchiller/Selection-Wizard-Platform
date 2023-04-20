#!/usr/bin/env python3
"""Process, analyse, and plot performance data."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

import constants as const


def read_ioh_json(metadata_path: Path, dims: int) -> (str, str, Path):
    """Read a .json metadata file from experiment with IOH.

    Args:
        metadata_path: Path to IOH metadata file.
        dims: int indicating the dimensionality for which to return the data
          file.

    Returns:
        str algorithm name.
        str function name.
        Path to the data file.
    """
    # TODO: Check scenarios.runs.run_success -- How to pass on/use this info?
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    algo_name = metadata["algorithm"]["name"]
    func_name = metadata["function_name"]

    for scenario in metadata["scenarios"]:
        if scenario["dimension"] == dims:
            data_path = Path(scenario["path"])
            break

    data_path = metadata_path.parent / data_path

    return (algo_name, func_name, data_path)


def read_ioh_results() -> None:
    """Read a specified set of result files form experiments with IOH."""
    prob_runs = []
    algo_names = []
    func_names = []
    dims = 100
    problem_names = ["f1_Sphere", "f3_Rastrigin"]

    for problem_name in problem_names:
        runs = []

        for algo_id in range(0, 6):
            algo_dir = const.ALGS_CONSIDERED[algo_id]
            json_path = Path(
                f"data_seeds_organised/{problem_name}/{algo_dir}/"
                f"IOHprofiler_{problem_name}.json")
            (algo_name, func_name, data_path) = read_ioh_json(json_path, dims)
            runs.append(read_ioh_dat(data_path))
            algo_names.append(algo_name)

        prob_runs.append(runs)
        func_names.append(func_name)

    algo_names = list(dict.fromkeys(algo_names))  # Remove duplicates
    plot_median(prob_runs, algo_names, func_names)

    return


def read_ioh_dat(result_path: Path) -> pd.DataFrame:
    """Read a .dat result file from experiment with IOH.

    These files contain blocks of data representing one run each of the form:
      evaluations raw_y
      1 1.0022434918
      ...
      10000 0.0000000000
    The first line indicates the start of a new run, and which data columsn are
    included. Following this, each line represents data from one evaluation.
    evaluations indicates the evaluation number.
    raw_y indicates the best value so far, except for the last line. The last
    line holds the value of the last evaluation, even if it is not the best so
    far.

    Args:
        result_path: Path pointing to an IOH data file.
    Returns:
        pandas DataFrame with performance data. Columns are evaluations,
          rows are different runs, column names are evaluation numbers.
    """
    with result_path.open("r") as result_file:
        lines = result_file.readlines()
        run_id = 0
        runs = []
        eval_ids = []
        run = []

        for line in lines:
            if line.startswith("e"):  # For 'evaluations'
                if run_id != 0:
                    runs.append(run)
                run = []
                run_id = run_id + 1
            else:
                words = line.split()
                eval_number = int(words[0])
                performance = float(words[1])
                run.append([eval_number, performance])

                if eval_number not in eval_ids:
                    eval_ids.append(eval_number)
        runs.append(run)

    eval_ids.sort()
    runs_full = np.zeros((len(runs), len(eval_ids)))

    for run_id in range(0, len(runs)):
        range_start = 0

        for run_eval in runs[run_id]:
            for idx in range(range_start, len(eval_ids)):
                # Element 0 is the evaluation number
                if run_eval[0] == eval_ids[idx]:
                    # If it is the last index, and the performance is
                    # larger than before, use the last best-so-far value.
                    # Element 1 is the performance value
                    if (idx == len(eval_ids) - 1
                       and run_eval[1] > runs_full[run_id][idx - 1]):
                        runs_full[run_id][idx] = runs_full[run_id][idx - 1]
                    else:
                        runs_full[run_id][idx] = run_eval[1]
                        range_start = idx + 1
                        break
                else:
                    # If it does not exist the value is the same as the
                    # previous evaluation.
                    # All runs should have a value for the first evaluation,
                    # so this block should not be reached without a previous
                    # value existing (i.e., idx - 1 should always be safe).
                    runs_full[run_id][idx] = runs_full[run_id][idx - 1]
                    range_start = idx + 1

    all_runs = pd.DataFrame(runs_full, columns=eval_ids)

    return all_runs


def plot_median(func_algo_runs: list[list[pd.DataFrame]],
                algo_names: list[str],
                func_names: list[str]) -> None:
    """Plot the median performance over time.

    Args:
        func_algo_runs: list of functions containing a list of pandas DataFrame
          with performance data per algorithm. Columns are evaluations, rows
          are different runs, column names are evaluation numbers.
        algo_names: List of algorithm names.
        func_names: List of function names.
    """
    fig, axs = plt.subplots(2, 1, layout="constrained")
    fig.suptitle("Median performance")

    # Draw a subplot for each problem
    for ax, func_name, algo_runs in zip(axs.flat, func_names, func_algo_runs):
        ax.set(xlabel="Evaluations",
               ylabel="Performance (best-so-far)")

        # Draw a curve for each algorithm
        for runs, algo_name in zip(algo_runs, algo_names):
            medians = runs.median(axis=0)
            eval_ids = runs.columns.values.tolist()
            lbl = "_nolegend_" if func_name != func_names[0] else algo_name
            ax.plot(eval_ids, medians, label=lbl)

        ax.set_title(f"{func_name}")
        ax.label_outer()

    fig.legend(loc="outside lower center")
    fig.show()
    fig.savefig("plot.pdf")

    return


if __name__ == "__main__":
    DEFAULT_EVAL_BUDGET = 10000
    DEFAULT_N_REPETITIONS = 25
    DEFAULT_DIMS = [4, 5]
    DEFAULT_PROBLEMS = list(range(1, 3))
    DEFAULT_INSTANCES = [1]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--infile",
        default=argparse.SUPPRESS,
        type=Path,
        help="File to read.")
    args = parser.parse_args()

    read_ioh_results()
