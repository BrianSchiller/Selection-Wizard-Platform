#!/usr/bin/env python3
"""Process, analyse, and plot performance data."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys

import constants as const
from experiment import Experiment
from experiment import NGOptChoice
from experiment import analyse_ma_csvs
from experiment import ma_plot_all


def read_ioh_json(metadata_path: Path, dims: int, verbose: bool = False) -> (
        str, str, Path, list[int]):
    """Read a .json metadata file from an experiment with IOH.

    Args:
        metadata_path: Path to IOH metadata file.
        dims: int indicating the dimensionality for which to return the data
          file.
        verbose: If True print more detailed information.

    Returns:
        str algorithm name.
        str function name.
        Path to the data file or empty Path if no file is found.
        list of usually ints showing the success/failure status of runs for
          this dimensionality. 1 indicates a successful run, 0 a crashed run,
          -1 a missing run. Other values than these mean something is likely
          to be wrong, e.g., a crash that was not detected during execution can
          have a value like 4.6355715189945e-310. An empty list is returned if
          no file is found.
    """
    if verbose:
        print(f"Reading json file: {metadata_path}")

    expected_runs = 25

    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    algo_name = metadata["algorithm"]["name"]
    func_id = metadata["function_id"]
    func_name = f"f{func_id}_{metadata['function_name']}"

    for scenario in metadata["scenarios"]:
        if scenario["dimension"] == dims:
            data_path = Path(scenario["path"])

            # Record per run whether it was successful
            run_success = [-1] * expected_runs
            for run, idx in zip(scenario["runs"], range(0, expected_runs)):
                run_success[idx] = run["run_success"]
            n_success = sum(run_suc for run_suc in run_success if run_suc == 1)

            if n_success != expected_runs:
                print(f"Found {n_success} successful runs out of "
                      f"{len(scenario['runs'])} instead of "
                      f"{expected_runs} runs for function {func_name} with "
                      f"algorithm {algo_name} and dimensionality {dims}.")

            break

    # Check whether a path to the data was identified
    try:
        data_path = metadata_path.parent / data_path
    except UnboundLocalError:
        print(f"No data found for function {func_name} with algorithm "
              f"{algo_name} and dimensionality {dims}.")
        data_path = Path()
        run_success = list()

    return (algo_name, func_name, data_path, run_success)


def read_ioh_results(data_dir: Path, verbose: bool = False) -> None:
    """Read a specified set of result files from experiments with IOH.

    Args:
        data_dir: Path to the data directory.
            This directory should have subdirectories per problem, which in
            turn should have subdirectories per algorithm, which should be
            organised in IOH format. E.g. for directory data, algorithm CMA,
            and problem f1_Sphere it should look like:
            data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
            data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat
        verbose: If True print more detailed information.
    """
    for dims in const.DIMS_CONSIDERED:
        print(f"Reading data for {dims} dimensional problems...")

        prob_runs = []
        algo_names = []
        func_names = []

        for problem_name in const.PROB_NAMES:
            runs = []

            for algo_id in range(0, 6):
                algo_dir = const.get_short_algo_name(
                    const.ALGS_CONSIDERED[algo_id])
                json_path = Path(
                    f"{data_dir}/{problem_name}/{algo_dir}/"
                    f"IOHprofiler_{problem_name}.json")
                (algo_name, func_name, data_path, _) = read_ioh_json(
                    json_path, dims, verbose)

                # Handle missing data files
                if data_path.is_file():
                    runs.append(read_ioh_dat(data_path, verbose))
                else:
                    # Filler to avoid mismatch in number of elements
                    runs.append(pd.DataFrame())

                algo_names.append(algo_name)

            prob_runs.append(runs)
            func_names.append(func_name)

        algo_names = list(dict.fromkeys(algo_names))  # Remove duplicates
        plot_median(prob_runs, algo_names, func_names, dims)

    return


def check_run_is_valid(eval_number: int, expected_evals: int,
                       run_id: int) -> bool:
    """Check whether run has the right number of evaluations.

    Args:
        eval_number: int with the last evaluation number of the run.
        expected_evals: int with the expected number of evaluations in the run.
        run_id: int with the ID of the current run.
    Returns:
        bool True if eval_number and expected_evals match, False otherwise.
    """
    if eval_number == expected_evals:
        return True
    else:
        print(f"Run with ID {run_id} is partial with only "
              f"{eval_number} evaluations instead of "
              f"{expected_evals}.")
        return False


def read_ioh_dat(result_path: Path, verbose: bool = False) -> pd.DataFrame:
    """Read a .dat result file with runs from an experiment with IOH.

    These files contain blocks of data representing one run each of the form:
      evaluations raw_y
      1 1.0022434918
      ...
      10000 0.0000000000
    The first line indicates the start of a new run, and which data columns are
    included. Following this, each line represents data from one evaluation.
    evaluations indicates the evaluation number.
    raw_y indicates the best value so far, except for the last line. The last
    line holds the value of the last evaluation, even if it is not the best so
    far.

    Args:
        result_path: Path pointing to an IOH data file.
        verbose: If True print more detailed information.
    Returns:
        pandas DataFrame with performance data. Columns are evaluations,
            rows are different runs, column names are evaluation numbers.
            Rows of failed runs are None/NaN.
    """
    if verbose:
        print(f"Reading dat file: {result_path}")

    expected_evals = 10000

    with result_path.open("r") as result_file:
        lines = result_file.readlines()
        run_id = 0
        runs = []
        eval_ids = []
        run = []
        eval_ids_run = []
        eval_number = 0

        for line in lines:
            if line.startswith("e"):  # For 'evaluations'
                if run_id != 0:
                    # Confirm run is complete before adding it
                    if check_run_is_valid(eval_number, expected_evals, run_id):
                        runs.append(run)
                        eval_ids.extend(eval_ids_run)
                    else:
                        runs.append(None)
                run = []
                eval_ids_run = []
                run_id = run_id + 1
            else:
                words = line.split()
                eval_number = int(words[0])
                performance = float(words[1])
                run.append([eval_number, performance])

                if eval_number not in eval_ids:
                    eval_ids_run.append(eval_number)
        # Confirm run is complete before adding it
        if check_run_is_valid(eval_number, expected_evals, run_id):
            runs.append(run)
            eval_ids.extend(eval_ids_run)
        # Add None row if invalid
        else:
            runs.append(None)

    eval_ids.sort()
    runs_full = np.zeros((len(runs), len(eval_ids)))

    # Fill the eval values for each run to make them all the same length
    for run_id in range(0, len(runs)):
        range_start = 0

        # Handle failed runs
        if runs[run_id] is None:
            runs_full[run_id] = runs[run_id]
            continue

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


def get_ranking_matrix(data_dir: Path) -> pd.DataFrame:
    """Get a matrix algorithm rankings for dimensionalities versus budget.

    Args:
        data_dir: Path to the data directory.
            This directory should have subdirectories per problem, which in
            turn should have subdirectories per algorithm, which should be
            organised in IOH format. E.g. for directory data, algorithm CMA,
            and problem f1_Sphere it should look like:
            data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
            data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat

    Returns:
        DataFrame with rows representing different dimensionalities and columns
            representing different evaluation budgets.
    """
    n_best = 25
    budgets = [dims * 10 for dims in const.DIMS_CONSIDERED]  # use 10d budgets
    algo_matrix = pd.DataFrame()

    for budget in budgets:
        ranks = []

        for dims in const.DIMS_CONSIDERED:
            ranks.append(rank_algorithms(data_dir, dims, budget, n_best))

        algo_matrix[budget] = ranks

    return algo_matrix


def rank_algorithms(data_dir: Path,
                    dims: int,
                    budget: int,
                    n_best: int = 25) -> pd.DataFrame:
    """Rank algorithms based on their performance over multiple problems.

    Args:
        data_dir: Path to the data directory.
            This directory should have subdirectories per problem, which in
            turn should have subdirectories per algorithm, which should be
            organised in IOH format. E.g. for directory data, algorithm CMA,
            and problem f1_Sphere it should look like:
            data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
            data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat
        dims: int indicating the number of variable space dimensions.
        budget: int indicating for which number of evaluations to rank the
            algorithms.
        n_best: int indicating the top how many runs to look for.

    Returns:
        DataFrame with columns: algorithm, points
    """
    print(f"Ranking algorithms for {dims} dimensional problems...")

    algo_names = [const.get_short_algo_name(algo_name)
                  for algo_name in const.ALGS_CONSIDERED]
    algo_scores = pd.DataFrame({
        "algorithm": algo_names,
        "points": [0] * len(algo_names)})

    for problem_name in const.PROB_NAMES:
        algo_runs = []

        for algo_id in range(0, len(const.ALGS_CONSIDERED)):
            algo_dir = const.get_short_algo_name(
                const.ALGS_CONSIDERED[algo_id])
            json_path = Path(
                f"{data_dir}/{problem_name}/{algo_dir}/"
                f"IOHprofiler_{problem_name}.json")
            (algo_name, prob_name, data_path, _) = read_ioh_json(
                json_path, dims)

            # Handle missing data files
            if data_path.is_file():
                algo_runs.append(read_ioh_dat(data_path))
            else:
                # Filler to avoid mismatch in number of elements
                algo_runs.append(pd.DataFrame())

        best_algos = get_best_runs_of_prob(
            algo_runs, algo_names, budget, n_best)

        # Count occurrences of algorithm
        algo_scores_for_prob = best_algos["algorithm"].value_counts()

        # Add counts to the scores
        algo_scores = pd.merge(
            algo_scores, algo_scores_for_prob, how="left", on="algorithm")
        algo_scores["count"].fillna(0, inplace=True)
        algo_scores["points"] += algo_scores["count"]
        algo_scores.drop(columns=["count"], inplace=True)

    return algo_scores


def get_best_runs_of_prob(algo_runs: list[pd.DataFrame],
                          algo_names: list[str],
                          budget: int,
                          n_best: int) -> pd.DataFrame:
    """Return the n best runs for a problem, dimension, budget combination.

    Args:
        algo_runs: list of pandas DataFrame with performance data per
            algorithm. Columns are evaluations, rows are different runs, column
            names are evaluation numbers.
        algo_names: list of algorithm names.
        budget: int indicating for which number of evaluations to rank the
            algorithms.
        n_best: int indicating the top how many runs to look for.
    Returns:
        DataFrame with n_best rows of algorithm, run ID, and performance. Any
            rows beyond row n_best that have the same performance as row n_best
            are also returned.
    """
    n_runs = 25
    algorithms = []
    run_ids = []
    performances = []

    # Loop over algorithms in algo_runs
    for runs, algo_name in zip(algo_runs, algo_names):
        # Find which column contains the relevant performance value
        for eval_id in runs.columns:
            if int(eval_id) <= budget:
                eval_col = eval_id
            else:
                break

        # Add run properties to lists: algorithm, run ID, performance at budget
        algorithms.extend([algo_name] * n_runs)
        run_ids.extend(list(range(1, n_runs + 1)))
        performances.extend(runs[eval_col])

    # Create a DataFrame from the lists
    runs = pd.DataFrame({
        "algorithm": algorithms,
        "run ID": run_ids,
        "performance": performances})

    # Sort the DataFrame by performance
    runs.sort_values("performance", inplace=True)

    # Return a DataFrame with the n_best rows
    best_runs = runs.head(n_best)
    runs = runs.iloc[n_best:]

    # Also return rows beyond n_best that have equal performance to row n_best
    best_runs_plus = runs.loc[
        runs["performance"] == best_runs["performance"].iloc[-1]]
    best_runs = pd.concat([best_runs, best_runs_plus])

    return best_runs


def plot_median(func_algo_runs: list[list[pd.DataFrame]],
                algo_names: list[str],
                func_names: list[str],
                dims: int) -> None:
    """Plot the median performance over time.

    Args:
        func_algo_runs: list of functions containing a list of pandas DataFrame
          with performance data per algorithm. Columns are evaluations, rows
          are different runs, column names are evaluation numbers.
        algo_names: List of algorithm names.
        func_names: List of function names.
        dims: int indicating the number of variable space dimensions.
    """
    # TODO: Make subplot dimensions depend on the number of functions
    # TODO: How to split the functions over columns/rows?
    fig, axs = plt.subplots(6, 4, layout="constrained",
                            figsize=(12, 20), dpi=80)
    fig.suptitle(f"Median symlog performance for {dims} dimensions.")

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
            ax.set_yscale("symlog", linthresh=0.1)

        ax.set_title(f"{func_name}")

    fig.legend(loc="outside lower center")
    fig.show()
    out_path = f"plots/convergence/plot_D{dims}.pdf"
    fig.savefig(out_path)
    print(
        f"Median convergence plot for {dims} dimensions saved to: {out_path}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "data_dir",
        default=argparse.SUPPRESS,
        type=Path,
        help="Directory to analyse.")
    parser.add_argument(
        "per_budget_data_dir",
        default=None,
        type=Path,
        nargs="?",  # 0 or 1
        help="Directory of budget specific data to analyse additionally.")
    parser.add_argument(
        "--per-prob-set",
        required=False,
        action="store_true",
        help=("Do analysis of the BBOB results for different subsets of the "
              "BBOB problems. E.g., only the multimodal problems, or the "
              "problems most similar to MA-BBOB."))
    parser.add_argument(
        "--ma",
        required=False,
        action="store_true",
        help="Analyse data_dir as being MA-BBOB preprocessed data.")
    parser.add_argument(
        "--ma-vs",
        required=False,
        action="store_true",
        help=("If set in addition to --ma, compare only the NGOpt choice"
              " and the data choice instead of all algorithms."))
    parser.add_argument(
        "--ma-plot",
        required=False,
        action="store_true",
        help=("Generate all plot(s) for the MA-BBOB data. If no other --ma "
              "argument is given, data_dir should be the path to the ranked "
              "MA-BBOB csv file. Use --ma-vs to indicate which algorithms are"
              "compared (controls the output file names)."))
    parser.add_argument(
        "--ma-loss",
        required=False,
        default=None,
        type=Path,
        help=("Path to dataframe with loss data per "
              "dimension-budget-algorithm-problem combination. If given "
              "plot lineplots with loss of algorithms per dimension-budget."))

    args = parser.parse_args()

    prob_sets = [
        # "all",
        # "separable", "low_cond", "high_cond", "multi_glob", "multi_weak",
        # "multimodal",
        # "ma-like_5", "ma-like_4",
        "ma-like_3", "ma-like_2",
        # "ma-like_0"  # Same as all
        ]

    # Analyse MA-BBOB preprocessed data
    if args.ma is True:
        analyse_ma_csvs(args.data_dir, ngopt_vs_data=args.ma_vs,
                        plot=args.ma_plot)
        sys.exit()
    # Plot MA-BBOB results from preprocessed data
    elif args.ma_plot is True:
        ma_plot_all(args.data_dir, ngopt_vs_data=args.ma_vs,
                    perf_data=args.ma_loss)
        sys.exit()
    # Plot BBOB results for all problems
    else:
        # Load NGOpt choices
        nevergrad_version = "0.6.0"
        hsv_file = Path("ngopt_choices/dims1-100evals1-10000_separator_"
                        f"{nevergrad_version}.hsv")
        ngopt = NGOptChoice(hsv_file)

        # Load experiment data
        exp = Experiment(args.data_dir,
                         ng_version=nevergrad_version)

        # Plot heatmap for all problems
        file_name = f"grid_data_{nevergrad_version}"
        matrix = exp.get_scoring_matrix(ngopt=ngopt)
        exp.plot_heatmap_data(matrix, ngopt, file_name)

        # Also plot BBOB results per function group
        if args.per_prob_set is True:
            for prob_set in prob_sets:
                exp.set_problems(prob_set)
                exp.load_data(verbose=True)
                matrix = exp.get_scoring_matrix(ngopt=ngopt)
                exp.plot_heatmap_data(matrix, ngopt, file_name)

        sys.exit()

    # read_ioh_results(args.data_dir, verbose = False)

    nevergrad_version = "0.6.0"
    hsv_file = Path("ngopt_choices/dims1-100evals1-10000_separator_"
                    f"{nevergrad_version}.hsv")
    ngopt = NGOptChoice(hsv_file)
    # Look at all dimensions, but exclude the largest budget (10000) because
    # it was already included in the original experiments.
#    budgets = [dims * 100 for dims in const.DIMS_CONSIDERED if dims < 100]
#    file_name = f"ngopt_choices_{nevergrad_version}"
#    ngopt.write_ngopt_choices_csv(const.DIMS_CONSIDERED, budgets, file_name)
#    file_name = f"ngopt_algos_{nevergrad_version}"
#    ngopt.write_unique_ngopt_algos_csv(file_name)
#    exp = Experiment(args.data_dir,
#                     args.per_budget_data_dir,
#                     ng_version=nevergrad_version)
    exp = Experiment(args.data_dir,
                     args.per_budget_data_dir,
                     # dimensionalities=[100, 35],
                     ng_version=nevergrad_version)
#    comp_data_dir = Path("data_seeds2_bud_dep_organised")
#    exp.load_comparison_data(comp_data_dir)
#    file_name = f"score_rank_{nevergrad_version}"
#    exp.write_score_rank_csv(file_name, ngopt)
    file_name = f"medians_{nevergrad_version}"
    exp.write_medians_csv(file_name, with_ranks=True)
#    file_name = f"scores_{nevergrad_version}"
#    exp.write_scoring_csv(file_name)
#    matrix = exp.get_scoring_matrix(ngopt=ngopt)
#    file_name = f"grid_{nevergrad_version}"
#    exp.plot_hist_grid(matrix, ngopt, file_name)
#    file_name = f"grid_data_{nevergrad_version}"
#    exp.plot_heatmap_data(matrix, ngopt, file_name)
#    exp.plot_heatmap_ngopt(ngopt)
#    file_name = f"best_comparison_{nevergrad_version}"
#    exp.write_performance_comparison_csv(file_name)
#    file_name = f"grid_data_budget_specific_{nevergrad_version}"
#    exp.plot_heatmap_data(matrix, ngopt, file_name)
#    print("Relevant algorithms:")
#    print(*exp.get_relevant_ngopt_algos(ngopt), sep="\n")
