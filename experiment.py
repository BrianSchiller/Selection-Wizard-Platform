"""Module with class definitions to describe an experiment and its data."""
from __future__ import annotations

from pathlib import Path
from pathlib import PurePath
import json
import statistics
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
import numpy as np
import scipy.stats as ss
from cmcrameri import cm

import constants as const


def analyse_test_csvs(data_dir: Path, ngopt_vs_data: bool = False,
                      plot: bool = True, test_bbob: bool = False) -> None:
    """Read and analyse preprocessed .csv files with performance data.

    Args:
        data_dir: Path to the data directory. This should have .csv files per
            algorithm-dimension-budget combination. Each of these files should
            have the columns: problem, algorithm, dimensions, budget, seed,
            status, performance, and if BBOB also instance; and 600 or 828
            rows (one per problem-instance pair for BBOB test data, or one per
            MA-BBOB problem).
        ngopt_vs_data: If True, compare only the NGOpt choice and the data
            choice; if False, compare NGOpt choice and top 4 from the data.
        plot: If True, also generate all available plots after the analysis.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    csv_dir = Path("csvs")
    out_dir = csv_dir / ("bbob_test" if test_bbob else "ma-bbob")
    out_dir.mkdir(parents=True, exist_ok=True)
    ngopt_v_data = "_1v1" if ngopt_vs_data else ""

    # Get all .csv files in the data directory
    csv_files = [csv_file for csv_file in data_dir.iterdir()
                 if str(csv_file).endswith(".csv")]

    # Read the data and collect it into a single DataFrame
    csv_dfs = list()

    for csv_file in csv_files:
        print("Loading:", csv_file)
        csv_dfs.append(pd.read_csv(csv_file))

    perf_data = pd.concat(csv_dfs)
    perf_data.reset_index(drop=True, inplace=True)
    print("Data loaded")

    # Add columns for data we want to add
    perf_data["algo ID"] = None
    perf_data["rank"] = None
    perf_data["percent loss"] = None
    perf_data["log loss"] = None

    # Add algorithm IDs
    names_csv = csv_dir / "ngopt_algos_0.6.0.csv"
    names_df = pd.read_csv(names_csv)

    for _, algo in names_df.iterrows():
        algo_name = algo["short name"]
        algo_id = algo["ID"]
        perf_data.loc[(perf_data["algorithm"] == algo_name),
                      "algo ID"] = algo_id

    # Create variables for all problem-dimension-budget combinations
    dimensionalities = const.DIMS_CONSIDERED
    dim_multiplier = 100
    budgets = [dims * dim_multiplier for dims in dimensionalities]
    probs_csv = csv_dir / "ma_prob_names.csv"
    problems = (const.PROB_NAMES
                if test_bbob else pd.read_csv(probs_csv)["problem"].to_list())

    # Create a DataFrame to store points per dimension-budget-algorithm combo
    ma_algos_csv = csv_dir / "ma_algos.csv"
    ranking = pd.read_csv(ma_algos_csv)
    ranking["in data"] = False
    ranking["points test"] = 0
    ranking["rank test"] = None

    # If we only compare the NGOpt choice and the data choice, remove others
    if ngopt_vs_data:
        # The ngopt rank column has 0 for the NGOpt choice, and -1 for the data
        # choice, if no -1 exists for a dimension-budget combination, the data
        # choice is the same as the NGOpt choice
        ranking.drop(ranking[ranking["ngopt rank"] > 0].index, inplace=True)

    # Prepare and check output paths
    failed_csv_path = out_dir / f"ranking{ngopt_v_data}_failed.csv"
    perf_csv_path = out_dir / f"perf_data{ngopt_v_data}.csv"
    rank_csv_path = out_dir / f"ranking{ngopt_v_data}.csv"
    csv_paths = [failed_csv_path, perf_csv_path, rank_csv_path]
    csv_paths = [csv_path for csv_path in csv_paths if csv_path.is_file()]

    for csv_path in csv_paths:
        new_csv_path = csv_path.with_suffix(".old")
        print(f"Output file {csv_path} already exists, moving it to "
              f"{new_csv_path}")
        csv_path.rename(new_csv_path)

    assign_points_test(
        dimensionalities, budgets, problems, perf_data, ranking,
        perf_csv_path, failed_csv_path, rank_csv_path, test_bbob)

    if plot:
        test_plot_all(rank_csv_path, ngopt_vs_data, perf_csv_path, test_bbob)

    return


def assign_points_test(dimensionalities: list[int],
                       budgets: list[int],
                       problems: list[str],
                       perf_data: pd.DataFrame,
                       ranking: pd.DataFrame,
                       perf_csv_path: Path,
                       failed_csv_path: Path,
                       rank_csv_path: Path,
                       test_bbob: bool = False) -> None:
    """Assign points for test data.

    Args:
        dimensionalities: List of dimensionalities to consider.
        budgets: List of budgets to consider.
        problems: List of problems to consider.
        perf_data: Performance data.
        ranking: Ranking data.
        perf_csv_path: Output path for performance data for all combinations.
        failed_csv_path: Output path for failed runs.
        rank_csv_path: Output path for algorithm rankings.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    # Assign points per problem on each dimension-budget combination
    for dimension in dimensionalities:
        for budget in budgets:
            # Check all algorithms we expect for this dimension-budget
            # combination are there; remove extras; report missing ones.
            algos_real = perf_data.loc[
                (perf_data["dimensions"] == dimension)
                & (perf_data["budget"] == budget)]
            algos_need = ranking.loc[
                (ranking["dimensions"] == dimension)
                & (ranking["budget"] == budget)]

            for algorithm in algos_real["algorithm"].unique():
                if algorithm in algos_need["algorithm"].values:
                    # Set in data column to True
                    ranking.loc[(ranking["dimensions"] == dimension)
                                & (ranking["budget"] == budget)
                                & (ranking["algorithm"] == algorithm),
                                "in data"] = True
                else:
                    # Remove algorithm from pref_data DataFrame
                    print(f"Found unexpected algorithm {algorithm} for "
                          f"D{dimension}B{budget}, excluding it from analysis")
                    perf_data.drop(
                        perf_data[
                            (perf_data["dimensions"] == dimension)
                            & (perf_data["budget"] == budget)
                            & (perf_data["algorithm"] == algorithm)].index,
                        inplace=True)

            # Check whether any data remains for this dim-bud combination
            perf_algos = perf_data.loc[
                (perf_data["dimensions"] == dimension)
                & (perf_data["budget"] == budget)]

            if len(perf_algos.index) == 0:
                print(f"No results found for D{dimension}B{budget}, skipping!")

                continue

            indices = []
            ranks = []
            loss_percent = []
            loss_log = []

            for problem in problems:
                instances = const.TEST_INSTANCES if test_bbob else [1]

                for instance in instances:
                    if instance == 1:
                        perf_algos = perf_data.loc[
                            (perf_data["dimensions"] == dimension)
                            & (perf_data["budget"] == budget)
                            & (perf_data["problem"] == problem)]
                    else:
                        perf_algos = perf_data.loc[
                            (perf_data["dimensions"] == dimension)
                            & (perf_data["budget"] == budget)
                            & (perf_data["problem"] == problem)
                            & (perf_data["instance"] == instance)]

                    # Check for each run whether it was successful
                    failed = perf_algos.loc[perf_algos["status"] != 1]

                    # Add failed runs to csv
                    if len(failed.index) > 0:
                        failed.to_csv(
                            failed_csv_path, mode="a",
                            header=not Path(failed_csv_path).exists(),
                            index=False)

                    for _, run in failed.iterrows():
                        error = run["status"]
                        err_str = (f"Run FAILED with error code: {error} for "
                                   f"algorithm {run['algorithm']} on "
                                   f"D{dimension}B{budget} on problem "
                                   f"{problem}")
                        err_str = (f"{err_str}, instance {run['instance']}"
                                   if test_bbob else err_str)
                        print(err_str)

                    # Get performance and indices
                    perfs = perf_algos["performance"].values
                    indices.extend(list(perf_algos["performance"].index))

                    # Rank the algorithms by performance on this
                    # dimension-budget-problem combination
                    # The "min" method resolves ties by assigning the
                    # minimum of the ranks of all tied methods. E.g., if
                    # the best two are tied, they get the minimum of rank 1
                    # and 2 = 1.
                    ranks.extend(
                        ss.rankdata(perfs, method="min", nan_policy="omit"))

                    # Compute percentage loss to best (and handle case where
                    # best is 0)
                    perfs_1 = perfs + 1
                    best = min(perfs_1)
                    loss_percent.extend((perfs_1 - best) / best * 100)

                    # Compute log loss to best
                    minimum = 0.00000000001
                    perfs_min = np.maximum(perfs, minimum)
                    best = min(perfs_min)
                    loss_log.extend(np.log10(perfs_min) - np.log10(best))

            # Update DataFrame for this dimension-budget combination
            perf_data.loc[indices, "rank"] = ranks
            perf_data.loc[indices, "percent loss"] = loss_percent
            perf_data.loc[indices, "log loss"] = loss_log

            # Write data of this dimension-budget combination to csv including
            # ranks and loss
            perf_db = perf_data.loc[(perf_data["dimensions"] == dimension)
                                    & (perf_data["budget"] == budget)]
            perf_db.to_csv(
                perf_csv_path,
                mode="a",
                header=not perf_csv_path.exists(),
                index=False)

            # Assign one point per row where an algorithms has rank 1
            top_ranks = perf_data.loc[
                (perf_data["dimensions"] == dimension)
                & (perf_data["budget"] == budget)
                & (perf_data["rank"] == 1), "algorithm"].values

            algos, counts = np.unique(top_ranks, return_counts=True)

            for algo, count in zip(algos, counts):
                ranking.loc[(ranking["dimensions"] == dimension)
                            & (ranking["budget"] == budget)
                            & (ranking["algorithm"] == algo),
                            "points test"] = count

            # Rank the algorithms for this dimension-budget combination based
            # on which algorithm has the most points over all problems.
            points = ranking.loc[(ranking["dimensions"] == dimension)
                                 & (ranking["budget"] == budget),
                                 "points test"].values
            # First take the negative of the points, to assign
            # ranks in descending order since more points is
            # better.
            neg_points = [-1 * point for point in points]
            # The "min" method resolves ties by assigning the
            # minimum of the ranks of all tied methods. E.g., if
            # the best two are tied, they get the minimum of rank 1
            # and 2 = 1.
            ranks = ss.rankdata(neg_points, method="min")
            ranking.loc[(ranking["dimensions"] == dimension)
                        & (ranking["budget"] == budget),
                        "rank test"] = ranks

            dim_bud_ranks = ranking.loc[
                (ranking["dimensions"] == dimension)
                & (ranking["budget"] == budget)]
            print(dim_bud_ranks)

            # Add points and ranks to csv
            dim_bud_ranks.to_csv(
                rank_csv_path,
                mode="a",
                header=not Path(rank_csv_path).exists(),
                index=False)


def test_plot_all(ranking_csv: Path, ngopt_vs_data: bool,
                  perf_data: Path | pd.DataFrame = None,
                  test_bbob: bool = False) -> None:
    """Generate all plots for test data on MA-BBOB or BBOB.

    Args:
        ranking_csv: Path to a csv file with algorithms ranked based on their
            performance on the test problems for each dimension-budget
            combination.
        ngopt_vs_data: If True, change output file names to indicate that the
            comparison only considers the NGOpt choice and the data choice; if
            False, use regular file names for the comparison between the NGOpt
            choice and top 4 from the data.
        perf_data: Path to the performance data csv with loss values per
            dimension-budget-algorithm-problem combination, or a pd.DataFrame
            with the same data. If None, don't plot cumulative loss plots.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    file_name = "grid_test"

    if ngopt_vs_data:
        file_name = f"{file_name}_1v1"

    plot_heatmap_data_test(ranking_csv, file_name=file_name,
                           comp_approach=False, test_bbob=test_bbob)
    plot_heatmap_data_test(ranking_csv, file_name=file_name,
                           comp_approach=True, test_bbob=test_bbob)

    if perf_data is not None:
        plot_cum_loss_data_test(perf_data, ngopt_vs_data, log=True, grid=True,
                                test_bbob=test_bbob)
#       plot_cum_loss_data_test(perf_data, ngopt_vs_data, log=True, grid=False)
        plot_cum_loss_data_test(perf_data, ngopt_vs_data, log=False, grid=True,
                                test_bbob=test_bbob)
#       plot_cum_loss_data_test(perf_data, ngopt_vs_data, log=False,
#                               grid=False)

        # Plot loss/gain heatmaps comparing best on MA-BBOB with NGOpt/Data
        # choice. Only when not considering the 1v1 case, since we need the
        # complete data.
        if not ngopt_vs_data:
            for magnitude in range(0, 6):
                plot_loss_gain_heatmap_test(
                    perf_data, ranking_csv, log=True,
                    compare="data", magnitude=magnitude, test_bbob=test_bbob)
                plot_loss_gain_heatmap_test(
                    perf_data, ranking_csv, log=True,
                    compare="ngopt", magnitude=magnitude, test_bbob=test_bbob)

    return


def plot_heatmap_data_test(ranking_csv: Path,
                           file_name: str = "grid_test",
                           comp_approach: bool = False,
                           test_bbob: bool = False) -> None:
    """Plot a heatmap showing the best algorithm per budget-dimension pair.

    In case of a tie, if one of the top ranking algorithms matches with the
    choice of NGOpt, this one is shown. If none of the tied algorithms
    match NGOpt, the one that happens to be on top is shown.

    Args:
        ranking_csv: Path to a csv file with algorithms ranked based on their
            performance on the MA-BBOB problems for each dimension-budget
            combination.
        file_name: Name of the file to write to. Will be written in the
            plots/heatmap/ directory with a _d{multiplier}.pdf extension.
        comp_approach: If True, compare approaches rather than algorithms.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    approach = "approach" if comp_approach else "algorithm"
    prob_set = "BBOB test" if test_bbob else "MA-BBOB"

    print(f"Plot heatmap of test data showing best {approach} per "
          f"budget-dimension pair for {prob_set}.")

    # Load data from csv
    algo_df = pd.read_csv(ranking_csv)

    if comp_approach:
        best_matrix = get_best_approach_test(algo_df)
        algo_names = [
            "NGOpt", "Data", "VBS", "Same (All)",
            "Tie (three-way)", "Tie (NGOpt-Data)", "Tie (NGOpt-VBS)",
            "Tie (Data-VBS)", "Tie (VBS-VBS)", "Missing"]
        best_algos = best_matrix.values.flatten().tolist()
        ids_in_plot = [idx for idx, algo in enumerate(algo_names)
                       if algo in best_algos]
        algos_in_plot = [algo for algo in algo_names if algo in best_algos]
        colours = cm.lipariS.colors[1::]  # Skip the first colour (black)
        tmp = colours[10]
        colours[8] = tmp  # Replace colour 8, too similar to 6
        colours_in_plot = [colours[i] for i in ids_in_plot]
    else:
        best_matrix = get_best_algorithms_test(algo_df)

        algorithms = []
        algo_names = [const.ALGS_CONSIDERED[idx] for idx in const.ALGS_0_6_0]

        for algo_name in algo_names:
            algorithms.append(Algorithm(algo_name))

        algo_names = [algo.name_short for algo in algorithms]
        algo_ids = [algo.id for algo in algorithms]
        best_algos = best_matrix.values.flatten().tolist()

        if "Missing" in best_algos:
            algo_names.append("Missing")
            algo_ids.append(14)  # Colour not used for const.ALGS_0_6_0

        # Get indices for algorithms relevant for the plot
        ids_in_plot = [idx for idx, algo in zip(algo_ids, algo_names)
                       if algo in best_algos]
        algos_in_plot = [algo for algo in algo_names if algo in best_algos]
        colours = const.ALGO_COLOURS
        colours_in_plot = [colours[i] for i in ids_in_plot]

    # Dict mapping short names to ints
    algo_to_int = {algo: i for i, algo in enumerate(algos_in_plot)}
    algo_to_id = {algo: idx for idx, algo
                  in zip(ids_in_plot, algos_in_plot)}

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    ax = sns.heatmap(
        best_matrix.replace(algo_to_int), cmap=colours_in_plot,
        annot=best_matrix.replace(algo_to_id),
        annot_kws={"size": const.FONT_SIZE_ALGO_ID},
        square=True)
    ax.set(xlabel="evaluation budget", ylabel="dimensions")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", labelrotation=90)

    # Add algorithm names to colour bar
    colorbar = ax.collections[0].colorbar
    r = colorbar.vmax - colorbar.vmin
    n = len(algo_to_int)
    colorbar.set_ticks(
        [colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
    algos_in_plot = [f"{idx}. {algo}" for idx, algo
                     in zip(ids_in_plot, algos_in_plot)]
    algo_to_int = {algo: i for i, algo in enumerate(algos_in_plot)}
    colorbar.set_ticklabels(list(algo_to_int.keys()))

    # Plot and save the figure
    plt.tight_layout()
    plt.show()
    dim_multiplier = 100

    out_dir = Path("plots/heatmap/")
    out_dir = out_dir / ("bbob_test" if test_bbob else "ma-bbob")

    if comp_approach:
        out_path = out_dir / f"{file_name}_approach_d{dim_multiplier}.pdf"
    else:
        out_path = out_dir / f"{file_name}_algos_d{dim_multiplier}.pdf"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)

    return


def get_best_algorithms_test(algo_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the top ranked algorithms per budget-dimensionality pair.

    In case of a tie, if one of the top ranking algorithms matches with the
    choice of NGOpt, this one is shown. If none of the tied algorithms
    match NGOpt, the one that happens to be on top is shown.

    Args:
        algo_df: DataFrame with rows representing different combinations of
            dimensionalities, budgets and algorithms. Available columns are:
            dimensions, budget, algorithm, rank, ngopt rank, algo ID, in data,
            points test, rank test

    Returns:
        A DataFrame of short Algorithm names with rows representing
        different dimensionalities and columns representing different
        evaluation budgets.
    """
    best_matrix = pd.DataFrame()
    budgets = algo_df["budget"].unique()
    dimensionalities = algo_df["dimensions"].unique()

    for budget in budgets:
        dims_best = []

        for dims in dimensionalities:
            algo_scores = algo_df.loc[(algo_df["dimensions"] == dims)
                                      & (algo_df["budget"] == budget)]

            if len(algo_scores.index) == 0:
                dims_best.append("Missing")
                continue

            # Retrieve the NGOpt choice for this dimension-budget combination
            ngopt_results = algo_scores.loc[
                algo_scores["ngopt rank"] == 0, "algorithm"]

            if len(ngopt_results.index) > 0:
                ngopt_algo = algo_scores.loc[
                    algo_scores["ngopt rank"] == 0, "algorithm"].values[0]
            else:
                ngopt_algo = None

            # Retrieve all algorithms that are tied for first place
            algo_scores = algo_scores.loc[algo_scores["rank test"] == 1]

            # Prefer the NGOptChoice in case of a tie
            if ngopt_algo in algo_scores["algorithm"].values:
                dims_best.append(ngopt_algo)
            else:
                dims_best.append(algo_scores["algorithm"].values[0])

        best_matrix[budget] = dims_best

    best_matrix.index = dimensionalities

    return best_matrix


def get_best_algorithms_test_func(algo_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the top ranked algorithms per budget-function pair.

    In case of a tie, if one of the top ranking algorithms matches with the
    choice of NGOpt, this one is shown. If none of the tied algorithms
    match NGOpt, the one that happens to be on top is shown.

    Args:
        algo_df: DataFrame with rows representing different combinations of
            dimensionalities, budgets, problems, and algorithms. Available
            columns are: problem, algorithm, dimensions, budget, seed, status,
            performance, algo ID, rank, percent loss, log loss

    Returns:
        A DataFrame of short Algorithm names with rows representing
        different evaluation budgets and columns representing different
        functions.
    """
    best_matrix = pd.DataFrame()
    budgets = algo_df["budget"].unique()
    functions = ["f60_MA0_F1-W0.1_F2_W0.9",
                 "f129_MA69_F2-W0.1_F3_W0.9",
                 "f195_MA135_F3-W0.1_F4_W0.9",
                 "f258_MA198_F4-W0.1_F5_W0.9",
                 "f318_MA258_F5-W0.1_F6_W0.9",
                 "f375_MA315_F6-W0.1_F7_W0.9",
                 "f429_MA369_F7-W0.1_F8_W0.9",
                 "f480_MA420_F8-W0.1_F9_W0.9",
                 "f528_MA468_F9-W0.1_F10_W0.9",
                 "f573_MA513_F10-W0.1_F11_W0.9",
                 "f615_MA555_F11-W0.1_F12_W0.9",
                 "f654_MA594_F12-W0.1_F13_W0.9",
                 "f690_MA630_F13-W0.1_F14_W0.9",
                 "f723_MA663_F14-W0.1_F15_W0.9",
                 "f753_MA693_F15-W0.1_F16_W0.9",
                 "f780_MA720_F16-W0.1_F17_W0.9",
                 "f804_MA744_F17-W0.1_F18_W0.9",
                 "f825_MA765_F18-W0.1_F19_W0.9",
                 "f843_MA783_F19-W0.1_F20_W0.9",
                 "f858_MA798_F20-W0.1_F21_W0.9",
                 "f870_MA810_F21-W0.1_F22_W0.9",
                 "f879_MA819_F22-W0.1_F23_W0.9",
                 "f885_MA825_F23-W0.1_F24_W0.9"]

    dims = 20

    for func in functions:
        dims_best = []

        budgets = [200, 5000, 10000]
        for budget in budgets:
            algo_scores = algo_df.loc[(algo_df["dimensions"] == dims)
                                      & (algo_df["budget"] == budget)
                                      & (algo_df["problem"] == func)]

            # Retrieve all algorithms that are tied for first place
            algo_scores = algo_scores.loc[algo_scores["rank"] == 1]
            dims_best.append(algo_scores["algorithm"].values[0])
            print(algo_scores["algorithm"].values)

        best_matrix[func] = dims_best

    best_matrix.index = budgets

    return best_matrix


def get_best_approach_test(algo_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve the top ranked approach per budget-dimensionality pair.

    When considering only the NGOpt choice and the data choice, the following
    outcomes are included: NGOpt, Data, Tie (each chose a different algorithm,
    but they scored the same number of points), Same algorithm (both choose the
    same algorithm), Missing (no results available).

    If additional options are considered, these additional outcomes are
    included: VBS (Virtual Best Solver), Tie (three-way), Tie (NGOpt-Data),
    Tie (NGOpt-VBS), Tie (Data-VBS), Tie (VBS-VBS).

    Args:
        algo_df: DataFrame with rows representing different combinations of
            dimensionalities, budgets and algorithms. Available columns are:
            dimensions, budget, algorithm, rank, ngopt rank, algo ID, in data,
            points test, rank test

    Returns:
        A DataFrame of outcomes, with rows representing
        different dimensionalities and columns representing different
        evaluation budgets.
    """
    best_matrix = pd.DataFrame()
    budgets = algo_df["budget"].unique()
    dimensionalities = algo_df["dimensions"].unique()

    for budget in budgets:
        dims_best = []

        for dims in dimensionalities:
            algo_scores = algo_df.loc[(algo_df["dimensions"] == dims)
                                      & (algo_df["budget"] == budget)]

            # Handle missing data
            if len(algo_scores.index) == 0:
                dims_best.append("Missing")
                continue

            # Retrieve all algorithms that are tied for first place
            algo_wins = algo_scores.loc[algo_scores["rank test"] == 1]
            train_wins = algo_scores.loc[algo_scores["rank"] == 1]

            # In the "ngopt rank" column -1 indicates the data-driven choice,
            # 0 indicates the NGOpt choice, and all other values will be a
            # positive integer equal to the "rank" column.

            # If all of 0, -1, >0 appear in ngopt rank, it is a three way Tie
            if (0 in algo_wins["ngopt rank"].values
                    and -1 in algo_wins["ngopt rank"].values
                    and any(r > 0 for r in algo_wins["ngopt rank"].values)):
                dims_best.append("Tie (three-way)")
            # If both 0, -1 appear in ngopt rank, it is a ngopt-data Tie
            elif (0 in algo_wins["ngopt rank"].values
                    and -1 in algo_wins["ngopt rank"].values):
                dims_best.append("Tie (NGOpt-Data)")
            # If both 0, >0 appear in ngopt rank, it is a ngopt-VBS Tie
            elif (0 in algo_wins["ngopt rank"].values
                    and any(r > 0 for r in algo_wins["ngopt rank"].values)):
                dims_best.append("Tie (NGOpt-VBS)")
            # If both -1, >0 appear in ngopt rank, it is a data-VBS Tie
            elif (-1 in algo_wins["ngopt rank"].values
                    and any(r > 0 for r in algo_wins["ngopt rank"].values)):
                dims_best.append("Tie (Data-VBS)")
            # If algo_wins still has more than 1 entry, this is a VBS-VBS Tie
            elif len(algo_wins.index) > 1:
                dims_best.append("Tie (VBS-VBS)")
            # If ngopt rank is 0, rank is 1, and there was only one algorithm
            # ranked first on the training set, all use the Same algorithm.
            # rank is 1 only checks it is the same as Data, but since this is
            # the set of algorithms ranked 1 on test, it is also the VBS.
            elif (0 in algo_wins["ngopt rank"].values
                  and 1 in algo_wins["rank"].values
                  and len(train_wins.index) == 1):
                dims_best.append("Same (All)")
            # Otherwise, if ngopt rank is 0, this is a win for NGOpt (=VBS)
            elif 0 in algo_wins["ngopt rank"].values:
                dims_best.append("NGOpt")
            # Otherwise, if ngopt rank is -1, this is a win for Data (=VBS)
            elif -1 in algo_wins["ngopt rank"].values:
                dims_best.append("Data")
            # Otherwise, the VBS wins
            else:
                dims_best.append("VBS")

        best_matrix[budget] = dims_best

    best_matrix.index = dimensionalities

    return best_matrix


def plot_cum_loss_data_test(perf_data: Path | pd.DataFrame,
                            ngopt_vs_data: bool,
                            log: bool = True,
                            grid: bool = True,
                            test_bbob: bool = False) -> None:
    """Plot the cumulative percentage of problems over the loss.

    Args:
        perf_data: Path to the performance data csv with loss values per
            dimension-budget-algorithm-problem combination, or a pd.DataFrame
            with the same data. If it is None, do nothing.
        ngopt_vs_data: If True, change output file names to indicate that the
            comparison only considers the NGOpt choice and the data choice; if
            False, use regular file names for the comparison between the NGOpt
            choice and top 4 from the data.
        log: If True plot the log loss, otherwise print the percentage loss.
        grid: If True plot a grid which each dimension-budget combination as
            subplot, otherwise create separate plots for each.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    loss_type = "log" if log else "percentage"
    plot_type = "grid" if grid else "individual plot"
    comp_type = "NGOpt and data best" if ngopt_vs_data else "all algorithms"
    prob_set = "BBOB test" if test_bbob else "MA-BBOB"

    print(f"Plot cumulative percentage of problems over the {loss_type} loss "
          f"for test data, comparing {comp_type}, with each budget-dimension "
          f"pair in a(n) {plot_type} for {prob_set}.")

    # If perf_data is None, do nothing
    if perf_data is None:
        return

    # If perf_data is given as Path, first load the data
    if isinstance(perf_data, PurePath):
        perf_data = pd.read_csv(perf_data)

    ngopt_v_data = "_1v1" if ngopt_vs_data else ""

    if grid:
        plot_cum_loss_data_test_grid(perf_data, ngopt_v_data, log, test_bbob)
    else:
        plot_cum_loss_data_test_separate(
            perf_data, ngopt_v_data, log, test_bbob)

    return


def plot_cum_loss_data_test_grid(perf_data: pd.DataFrame,
                                 ngopt_v_data: str,
                                 log: bool = True,
                                 test_bbob: bool = False) -> None:
    """Plot the cumulative percentage of problems over the loss in a grid.

    Args:
        perf_data: pd.DataFrame with the performance data csv with loss values
            per dimension-budget-algorithm-problem combination.
        ngopt_v_data: String to use for the output Path. Either empty or 1v1 to
            indicate only the NGOpt and Data choices are compared.
        log: If True plot the log loss, otherwise print the percentage loss.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    # For each dimension-budget combination
    budgets = perf_data["budget"].unique()
    dimensionalities = perf_data["dimensions"].unique()
    algorithms = []
    algo_names = [const.ALGS_CONSIDERED[idx] for idx in const.ALGS_0_6_0]

    for algo_name in algo_names:
        algorithms.append(Algorithm(algo_name))

    rows = len(dimensionalities)
    cols = len(budgets)
    fig, axs = plt.subplots(rows, cols, layout="tight",
                            figsize=(cols*6.2, rows*5.6), dpi=80)
    bud_dims = [(bud, dim) for dim in dimensionalities
                for bud in budgets]

    for bud_dim, ax in zip(bud_dims, axs.flatten()):
        budget = bud_dim[0]
        dims = bud_dim[1]
        algos_data = perf_data.loc[(perf_data["dimensions"] == dims)
                                   & (perf_data["budget"] == budget)]
        algos = []

        # For each algorithm
        for algorithm in algos_data["algorithm"].unique():
            algos.append(algorithm)
            algo_data = algos_data.loc[
                algos_data["algorithm"] == algorithm].copy()
            # Order the losses on the 828 problems in ascending order
            loss_type = "log" if log else "percent"
            algo_data.sort_values(f"{loss_type} loss", inplace=True)
            losses = algo_data[f"{loss_type} loss"].tolist()

            # For every distinct loss value
            n_probs = len(losses)
            perc_probs = [None] * n_probs
            last_val = -1

            for idx, loss in reversed(list(enumerate(losses))):
                # If the loss value is the same, so is the percentage of
                # problems solved with this loss value
                if loss == last_val:
                    perc_probs[idx] = perc_probs[idx+1]
                else:
                    # Compute the percentage of problems with equal or
                    # lower loss
                    perc_probs[idx] = (idx + 1) / n_probs * 100
                    last_val = loss

            loss_label = "log loss" if log else "loss %"
            algo_loss = pd.DataFrame({loss_label: losses,
                                      "problems %": perc_probs,
                                      "algorithm": algorithm})

            # Get indices for algorithms relevant for the plot
            algos_in_plot = [algo.name_short for algo in algorithms
                             if algo.name_short in algos]
            algo_ids = [algo.id for algo in algorithms]
            ids_in_plot = [idx for idx, algo in zip(algo_ids, algorithms)
                           if algo.name_short in algos_in_plot]
            colours = const.ALGO_COLOURS
            colours_in_plot = {algo: colours[i] for algo, i
                               in zip(algos_in_plot, ids_in_plot)}

            # Plot the loss (x) against the percentage of problems (y)
            sns.lineplot(data=algo_loss, x=loss_label,
                         y="problems %", hue="algorithm",
                         palette=colours_in_plot, ax=ax)

        sns.move_legend(ax, "lower left", bbox_to_anchor=(0, -0.5, 1, 0.2))
        ax.set_title(f"Dimensions: {dims}, Budget: {budget}")
        if not log:
            ax.set_xscale("log")

    # Set grid lines
    for ax in axs.flatten():
        ax.grid(visible=True)

    out_dir = Path("plots/line/")
    out_dir = out_dir / ("bbob_test" if test_bbob else "ma-bbob")
    out_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{out_dir}/loss_{loss_type}{ngopt_v_data}_grid.pdf"
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()

    return


def plot_cum_loss_data_test_separate(perf_data: pd.DataFrame,
                                     ngopt_v_data: str,
                                     log: bool = True,
                                     test_bbob: bool = False) -> None:
    """Plot the cumulative percentage of problems over the loss per pair.

    Each pair has a budget and number of dimensions.

    Args:
        perf_data: pd.DataFrame with the performance data csv with loss values
            per dimension-budget-algorithm-problem combination.
        ngopt_v_data: String to use for the output Path. Either empty or 1v1 to
            indicate only the NGOpt and Data choices are compared.
        log: If True plot the log loss, otherwise print the percentage loss.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    # For each dimension-budget combination
    budgets = perf_data["budget"].unique()
    dimensionalities = perf_data["dimensions"].unique()
    algorithms = []
    algo_names = [const.ALGS_CONSIDERED[idx] for idx in const.ALGS_0_6_0]

    for algo_name in algo_names:
        algorithms.append(Algorithm(algo_name))

    # Plot each figure to a separate file:
    for budget in budgets:
        for dims in dimensionalities:
            algos_data = perf_data.loc[(perf_data["dimensions"] == dims)
                                       & (perf_data["budget"] == budget)]
            plt.figure()
            algos = []

            # For each algorithm
            for algorithm in algos_data["algorithm"].unique():
                algos.append(algorithm)
                algo_data = algos_data.loc[
                    algos_data["algorithm"] == algorithm].copy()
                # Order the losses on the 828 problems in ascending order
                loss_type = "log" if log else "percent"
                algo_data.sort_values(f"{loss_type} loss", inplace=True)
                losses = algo_data[f"{loss_type} loss"].tolist()

                # For every distinct loss value
                n_probs = len(losses)
                perc_probs = [None] * n_probs
                last_val = -1

                for idx, loss in reversed(list(enumerate(losses))):
                    # If the loss value is the same, so is the percentage of
                    # problems solved with this loss value
                    if loss == last_val:
                        perc_probs[idx] = perc_probs[idx+1]
                    else:
                        # Compute the percentage of problems with equal or
                        # lower loss
                        perc_probs[idx] = (idx + 1) / n_probs * 100
                        last_val = loss

                loss_label = "log loss" if log else "loss %"
                algo_loss = pd.DataFrame({loss_label: losses,
                                          "problems %": perc_probs,
                                          "algorithm": algorithm})

                # Get indices for algorithms relevant for the plot
                algos_in_plot = [algo.name_short for algo in algorithms
                                 if algo.name_short in algos]
                algo_ids = [algo.id for algo in algorithms]
                ids_in_plot = [idx for idx, algo in zip(algo_ids, algorithms)
                               if algo.name_short in algos_in_plot]
                colours = const.ALGO_COLOURS
                colours_in_plot = {algo: colours[i] for algo, i
                                   in zip(algos_in_plot, ids_in_plot)}

                # Plot the loss (x) against the percentage of problems (y)
                ax = sns.lineplot(data=algo_loss, x=loss_label,
                                  y="problems %", hue="algorithm",
                                  palette=colours_in_plot)

            sns.move_legend(ax, "lower left", bbox_to_anchor=(0, -0.5, 1, 0.2))
            ax.set_title(f"Dimensions: {dims}, Budget: {budget}")

            if not log:
                ax.set_xscale("log")

            # Set grid lines
            plt.grid()

            out_dir = Path("plots/line/")
            out_dir = out_dir / ("bbob_test" if test_bbob else "ma-bbob")
            out_dir.mkdir(parents=True, exist_ok=True)

            file_name = (f"{out_dir}/single/loss_{loss_type}{ngopt_v_data}"
                         f"_D{dims}B{budget}.pdf")
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()

    return


def plot_loss_gain_heatmap_test(perf_data: Path | pd.DataFrame,
                                rank_data: Path | pd.DataFrame,
                                log: bool = True,
                                compare: str = "data",
                                magnitude: float = 0,
                                test_bbob: bool = False) -> None:
    """Plot a loss/gain heatmap compared to the best algorithm at 0 loss.

    The loss and gain compared to the best algorithm are computed by taking the
    difference in the percentage of problems for which the algorithms have the
    chosen worst case loss magnitude. At magnitude zero the data/ngopt choice
    can only lose or tie with the best algorithm. At higher magnitudes the
    data/ngopt choice may also perform better than the best algorithm. Here the
    best algorithm is taken as the algorithm covering the highest
    number/percentage of problems at magnitude 0.

    Args:
        perf_data: pd.DataFrame with the performance data csv with loss values
            per dimension-budget-algorithm-problem combination.
        rank_data: DataFrame with rows representing different combinations of
            dimensionalities, budgets and algorithms. Available columns are:
            dimensions, budget, algorithm, rank, ngopt rank, algo ID, in data,
            points test, rank test
        log: If True plot the log loss, otherwise print the percentage loss.
        compare: The selector to compare with the best algorithm. Can be "data"
            or "ngopt", to compare to the choices they make.
        magnitude: The order of magnitude of loss to compare at.
        test_bbob: If True, adjust names and variables to handle everything as
            data from BBOB test instances. If False, handle everything as
            MA-BBOB data.
    """
    loss_type = "log" if log else "percentage"
    comp_type = "train data" if compare == "data" else "NGOpt"
    prob_set = "BBOB test" if test_bbob else "MA-BBOB"

    print(f"Plot a loss/gain heatmap for {comp_type} compared to the best "
          f"algorithm at 0 loss ({loss_type} loss) on the test data"
          f"for {prob_set}.")

    # If rank_data is given as Path, first load the ranking data from csv
    if isinstance(rank_data, PurePath):
        rank_data = pd.read_csv(rank_data)

    # If perf_data is given as Path, first load the performance data from csv
    if isinstance(perf_data, PurePath):
        perf_data = pd.read_csv(perf_data)

    budgets = rank_data["budget"].unique()
    dimensionalities = rank_data["dimensions"].unique()
    diff_matrix = pd.DataFrame()

    for budget in budgets:
        bud_diffs = []

        for dims in dimensionalities:
            # Retrieve the relevant algorithm names
            best_algo = rank_data.loc[(rank_data["dimensions"] == dims)
                                      & (rank_data["budget"] == budget)
                                      & (rank_data["rank test"] == 1)]
            best_algo = best_algo["algorithm"].values[0]
            comp_algo = rank_data.loc[(rank_data["dimensions"] == dims)
                                      & (rank_data["budget"] == budget)
                                      & (rank_data["ngopt rank"] <= 0)]

            if compare == "ngopt":
                comp_algo = comp_algo.loc[comp_algo["ngopt rank"] == 0]
                comp_algo = comp_algo["algorithm"].values[0]
            elif compare == "data":
                comp_algo = comp_algo.loc[comp_algo["ngopt rank"]
                                          == comp_algo["ngopt rank"].min()]
                comp_algo = comp_algo["algorithm"].values[0]
            else:
                print("ERROR: Invalid 'compare' argument given to "
                      "function plot_loss_gain_heatmap_test()")
                sys.exit(-1)

            # Determine the loss type
            loss_type = "log" if log else "percent"
            loss_type_str = f"{loss_type} loss"

            # Take the percentage of problems covered by the best algorithm
            best_data = perf_data.loc[
                (perf_data["dimensions"] == dims)
                & (perf_data["budget"] == budget)
                & (perf_data["algorithm"] == best_algo)].copy()
            # Divide by the total number of rows to include failed runs that
            # have an empty entry in the loss_type_str column.
            best_perc = (best_data.loc[
                best_data[loss_type_str] <= magnitude, loss_type_str].count()
                / len(best_data.index) * 100)

            # Take the percentage of problems covered by the ngopt/data choice
            comp_data = perf_data.loc[
                (perf_data["dimensions"] == dims)
                & (perf_data["budget"] == budget)
                & (perf_data["algorithm"] == comp_algo)].copy()
            comp_perc = (comp_data.loc[
                comp_data[loss_type_str] <= magnitude, loss_type_str].count()
                / len(comp_data.index) * 100)

            # Take the loss/gain of ngopt/data compared to the best algorithm
            difference = comp_perc - best_perc
            print(f"difference: {difference} for B{budget}D{dims} between best"
                  f" {best_algo} ({best_perc}) and {compare} {comp_algo} "
                  f"({comp_perc})")

            # Store the difference for plotting
            bud_diffs.append(difference)

        diff_matrix[budget] = bud_diffs

    diff_matrix.index = dimensionalities

    # Plot the heatmap based on the differences
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    # First define the limit
    vmax = 38.
    vmin = -vmax
    # Then adjust depending on whether we are in the 0 magnitude case
    vmax = vmax if magnitude > 0 else 0.
    vcenter = 0. if magnitude > 0 else vmin / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap = sns.color_palette("vlag_r", as_cmap=True)
    cmap2 = sns.light_palette(
        cmap.get_under(), input="rgba", reverse=True, as_cmap=True)
    cmap = cmap if magnitude > 0 else cmap2

    ax = sns.heatmap(
        diff_matrix,
        square=True,
        annot=True, annot_kws={"size": 6},
        cmap=cmap, norm=norm)
    ax.set(xlabel="evaluation budget", ylabel="dimensions")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_title(f"Difference in percentage of problems for a {loss_type} loss"
                 f" of {magnitude}\nof the {compare} choice compared to the "
                 "best algorithm at 0 loss.")
    # Plot and save the figure
    plt.tight_layout()
    plt.show()

    out_dir = Path("plots/heatmap/")
    out_dir = out_dir / ("bbob_test" if test_bbob else "ma-bbob")

    out_path = Path(
        f"{out_dir}/loss_gain_{loss_type}_loss_mag{magnitude}_{compare}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)

    return


class Experiment:
    """Holds an experiment and its properties."""

    def __init__(self: Experiment,
                 data_dir: Path = None,
                 per_budget_data_dir: Path = None,
                 dimensionalities: list[int] = const.DIMS_CONSIDERED,
                 ng_version: str = "0.6.0",
                 prob_set: str = "all") -> None:
        """Initialise the Experiment.

        Args:
            data_dir: Path to the data directory. By default,
                this directory should have subdirectories per problem, which in
                turn should have subdirectories per algorithm, which should be
                organised in IOH format. E.g. for directory data, algorithm
                CMA, and problem f1_Sphere it should look like:
                data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
                data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat
            per_budget_data_dir: If a per_budget_data_dir is provided, this
                directory should have
                subdirectories named by algorithm-dimensionality-budget. These
                should each be organised in IOH format, and contain both a
                .json and directory for each BBOB function, specific to the
                dimensionality and budget combination for this subdirectory.
            dimensionalities (optional): List of ints indicating which
                dimensionalities to handle for the Experiment.
            ng_version: Version of Nevergrad. This influences which algorithms
                are chosen by NGOpt, and therefore which algorithms are
                included in the analysis.
            prob_set: The problem set to use. Default is all. Other options are
                ma-like_5 - BBOB problems most similar to MA-BBOB (based on 2D)
                    Includes functions 1, 3, 5, 6, 7, 10, 13, 20, 22, 23
                ma-like_4 - BBOB problems near similar to MA-BBOB (based on 2D)
                    Includes functions ma-like_5 and 8, 18
                ma-like_3 - BBOB problems somewhat like MA-BBOB (based on 2D)
                    Includes functions ma-like_4 and 4, 9, 12, 14, 15, 16, 17
                ma-like_2 - BBOB problems fairly unlike MA-BBOB (based on 2D)
                    Includes functions ma-like_3 and 2, 11, 19, 21
                ma-like_0 - BBOB problems unlike MA-BBOB (based on 2D)
                    Includes functions ma-like_2 and 24
                separable - BBOB problems 1-5 (separable)
                low_cond - BBOB problems 6-9 (low or moderate conditioning)
                high_cond - BBOB problems 10-14 (high conditioning, unimodal)
                multi_glob - BBOB problems 15-19 (multimodal, global structure)
                multi_weak - BBOB problems 20-24 (multimodal, weak g structure)
                multimodal - BBOB problems 15-24 (multimodal)
                Further, there are also options for each individual function:
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12
                f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24
        """
        self.data_dir = data_dir
        self.algorithms = []
        self.dimensionalities = dimensionalities
        self.prob_scenarios = {}
        self.prob_scenarios_per_b = {}
        self.prob_scenarios_comp = {}
        self.dim_multiplier = 100
        self.budgets = [
            dims * self.dim_multiplier for dims in self.dimensionalities]
        self.per_budget_data_dir = per_budget_data_dir
        self.problems_all = {}

        # Start with all problems to load all data
        self.set_problems(prob_set="all")

        if ng_version == "0.5.0":
            algo_names = [
                const.ALGS_CONSIDERED[idx] for idx in const.ALGS_0_5_0]
        elif ng_version == "0.6.0":
            algo_names = [
                const.ALGS_CONSIDERED[idx] for idx in const.ALGS_0_6_0]

        for algo_name in algo_names:
            self.algorithms.append(Algorithm(algo_name))

        if self.data_dir is not None:
            self.load_data()

        # Take the problem set given by the caller
        self.set_problems(prob_set)

        if self.per_budget_data_dir is not None:
            self.load_per_budget_data()

        return

    def set_problems(self: Experiment,
                     prob_set: str) -> list[Problem]:
        """Set the set of BBOB problems to consider in the analysis.

        Args:
            prob_set: The problem set to use. Default is all. Other options are
                ma-like_5 - BBOB problems most similar to MA-BBOB (based on 2D)
                    Includes functions 1, 3, 5, 6, 7, 10, 13, 20, 22, 23
                ma-like_4 - BBOB problems near similar to MA-BBOB (based on 2D)
                    Includes functions ma-like_5 and 8, 18
                ma-like_3 - BBOB problems somewhat like MA-BBOB (based on 2D)
                    Includes functions ma-like_4 and 4, 9, 12, 14, 15, 16, 17
                ma-like_2 - BBOB problems fairly unlike MA-BBOB (based on 2D)
                    Includes functions ma-like_3 and 2, 11, 19, 21
                ma-like_0 - BBOB problems unlike MA-BBOB (based on 2D)
                    Includes functions ma-like_2 and 24
                separable - BBOB problems 1-5 (separable)
                low_cond - BBOB problems 6-9 (low or moderate conditioning)
                high_cond - BBOB problems 10-14 (high conditioning, unimodal)
                multi_glob - BBOB problems 15-19 (multimodal, global structure)
                multi_weak - BBOB problems 20-24 (multimodal, weak g structure)
                multimodal - BBOB problems 15-24 (multimodal)
                Further, there are also options for each individual function:
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12
                f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24

        Returns:
            A list of Problem objects.
        """
        self.prob_set = prob_set
        self.problems = []

        # All indexes are 1 less than the associated function numbers
        prob_sets = {
            "all": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23],
            "ma-like_5": [0, 2, 4, 5, 6, 9, 12, 19, 21, 22],
            "ma-like_4": [0, 2, 4, 5, 6, 9, 12, 19, 21, 22, 7, 17],
            "ma-like_3": [0, 2, 4, 5, 6, 9, 12, 19, 21, 22, 7, 17,
                          3, 8, 11, 13, 14, 15, 16],
            "ma-like_2": [0, 2, 4, 5, 6, 9, 12, 19, 21, 22, 7, 17,
                          3, 8, 11, 13, 14, 15, 16, 1, 10, 18, 20],
            "ma-like_0": [0, 2, 4, 5, 6, 9, 12, 19, 21, 22, 7, 17,
                          3, 8, 11, 13, 14, 15, 16, 1, 10, 18, 20, 7, 17],
            "separable": [0, 1, 2, 3, 4],
            "low_cond": [5, 6, 7, 8],
            "high_cond": [9, 10, 11, 12, 13],
            "multi_glob": [14, 15, 16, 17, 18],
            "multi_weak": [19, 20, 21, 22, 23],
            "multimodal": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            "f1": [0],
            "f2": [1],
            "f3": [2],
            "f4": [3],
            "f5": [4],
            "f6": [5],
            "f7": [6],
            "f8": [7],
            "f9": [8],
            "f10": [9],
            "f11": [10],
            "f12": [11],
            "f13": [12],
            "f14": [13],
            "f15": [14],
            "f16": [15],
            "f17": [16],
            "f18": [17],
            "f19": [18],
            "f20": [19],
            "f21": [20],
            "f22": [21],
            "f23": [22],
            "f24": [23]}

        prob_idxs = prob_sets[self.prob_set]
        prob_names = [const.PROB_NAMES[idx] for idx in prob_idxs]
        prob_ids = [const.PROBS_CONSIDERED[idx] for idx in prob_idxs]

        for prob_name, prob_id in zip(prob_names, prob_ids):
            # If the Problem was not previously created yet, do it now
            if prob_id not in self.problems_all:
                self.problems_all.update(
                    {prob_id: Problem(prob_name, prob_id)})

            self.problems.append(self.problems_all[prob_id])

        return

    def load_data(self: Experiment,
                  verbose: bool = False) -> None:
        """Read IOH result files from the data directory.

        Args:
            verbose: If True print more detailed information.
        """
        for problem in self.problems:
            a_scenarios = {}

            for algorithm in self.algorithms:
                d_scenarios = {}

                for dims in self.dimensionalities:
                    d_scenarios[dims] = Scenario(self.data_dir,
                                                 problem,
                                                 algorithm,
                                                 dims,
                                                 const.RUNS_PER_SCENARIO,
                                                 const.EVAL_BUDGET,
                                                 verbose=verbose)

                a_scenarios[algorithm] = d_scenarios

            self.prob_scenarios[problem] = a_scenarios

        print("Done loading data!")

        return

    def load_per_budget_data(self: Experiment) -> None:
        """Read IOH results from a directory with data for multiple budgets."""
        data_dirs = [child for child
                     in self.per_budget_data_dir.iterdir() if child.is_dir()]

        for data_dir in data_dirs:
            algo, dims, budget = data_dir.name.split("-")
            dims = int(dims)
            budget = int(budget)

            # Get the matching algorithm object
            algorithm = next(
                a for a in self.algorithms if a.name_short == algo)

            prob_dirs = [child for child
                         in data_dir.iterdir() if child.is_dir()]

            for prob_dir in prob_dirs:
                # Get the matching problem object
                prob = prob_dir.name.removeprefix("data_")
                problem = next(p for p in self.problems if p.name == prob)

                # Create sub-dicts that don't exist yet
                if problem not in self.prob_scenarios_per_b:
                    self.prob_scenarios_per_b[problem] = {}
                if algorithm not in self.prob_scenarios_per_b[problem]:
                    self.prob_scenarios_per_b[problem][algorithm] = {}
                if dims not in self.prob_scenarios_per_b[problem][algorithm]:
                    self.prob_scenarios_per_b[problem][algorithm][dims] = {}

                self.prob_scenarios_per_b[problem][algorithm][dims][budget] = (
                    Scenario(
                        self.per_budget_data_dir,
                        problem,
                        algorithm,
                        dims,
                        const.RUNS_PER_SCENARIO,
                        budget,
                        per_budget=True))

        print("Done loading per budget data!")

        return

    def load_comparison_data(self: Experiment,
                             data_dir: Path) -> None:
        """Read IOH results to compare with the main experiment.

        Args:
            data_dir: Path to the data directory. By default,
                this directory should have subdirectories per problem, which in
                turn should have subdirectories per algorithm, which should be
                organised in IOH format. E.g. for directory data, algorithm
                CMA, and problem f1_Sphere it should look like:
                data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
                data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat
        """
        prob_dirs = [child for child
                     in data_dir.iterdir() if child.is_dir()]
        self.comp_probs = []
        self.comp_algos = []
        self.comp_dims = []
        self.comp_buds = []

        for prob_dir in prob_dirs:
            # Get the matching problem object
            prob = prob_dir.name
            problem = next(p for p in self.problems if p.name == prob)
            self.comp_probs.append(problem)

            a_scenarios = {}

            # Get the algorithms
            algo_dirs = [child for child
                         in prob_dir.iterdir() if child.is_dir()]

            for algo_dir in algo_dirs:
                # Get the matching algorithm object
                algo = algo_dir.name
                # TODO: Handle StopIteration if algorithm is match (e.g., when
                # comparing data with different ng_version)
                algorithm = next(
                    a for a in self.algorithms if a.name_short == algo)
                self.comp_algos.append(algorithm)

                d_scenarios = {}

                # Get dimensionalities
                json_path = algo_dir / f"IOHprofiler_{problem.name}.json"

                with json_path.open() as metadata_file:
                    metadata = json.load(metadata_file)

                # Read dimensionalities and budgets from json file
                for scenario in metadata["scenarios"]:
                    dims = scenario["dimension"]
                    budget = scenario["runs"][0]["evals"]
                    self.comp_dims.append(dims)
                    self.comp_buds.append(budget)

                    d_scenarios[dims] = Scenario(data_dir,
                                                 problem,
                                                 algorithm,
                                                 dims,
                                                 const.RUNS_PER_SCENARIO,
                                                 budget)

                a_scenarios[algorithm] = d_scenarios

            self.prob_scenarios_comp[problem] = a_scenarios

        print(f"Done loading comparison data from: {data_dir}")

        return

    def _get_perfs(self: Experiment,
                   algo: Algorithm,
                   problem: Problem,
                   budget: int,
                   dims: int,
                   main: bool = True) -> list[float]:
        """Retrieve performance values per run.

        Args:
            algo: Algorithm object for which to get the data.
            problem: A Problem object for which to get the data.
            budget: int indicating for which number of evaluations to get the
                performances.
            dims: int indicating the dimensionality for which to get the data.
            main: bool indicating whether to use the main results set or the
                comparison results set.

        Returns:
            List of performance values per run at the specified budget. Failed
            runs are assigned a value of -999.
        """
        if main:
            scenario = self.prob_scenarios[problem][algo][dims]
        else:
            scenario = self.prob_scenarios_comp[problem][algo][dims]
        # Missing runs get a default value of -999
        perfs = [run.get_performance(budget) if run.status == 1 else -999
                 for run in scenario.runs]

        return perfs

    def write_performance_comparison_csv(
            self: Experiment,
            file_name: str = "best_comparison") -> None:
        """Write a CSV comparing performance per run for matching scenarios.

        The CSV file contains the columns:
        dimensions, budget, problem, algorithm, perf_a, perf_b, equal

        Args:
            file_name: Name of the file to write to. Will be written in the
                csvs/ directory with a .csv extension.
        """
        col_names = ["dimensions", "budget", "problem", "algorithm",
                     "perf_a", "perf_b", "equal"]
        all_comps = []

        budgets = [
            budget for budget in self.budgets if budget in self.comp_buds]
        dimensionalities = [
            dims for dims in self.dimensionalities if dims in self.comp_dims]
        problems = [
            problem for problem in self.problems if problem in self.comp_probs]
        algorithms = [algorithm for algorithm in self.algorithms
                      if algorithm in self.comp_algos]

        for budget in budgets:
            for dims in dimensionalities:
                for problem in problems:
                    for algorithm in algorithms:
                        perfs_a = self._get_perfs(
                            algorithm, problem, budget, dims, main=True)
                        perfs_b = self._get_perfs(
                            algorithm, problem, budget, dims, main=False)

                        # Add things to organised collection
                        dim = [dims] * const.RUNS_PER_SCENARIO
                        bud = [budget] * const.RUNS_PER_SCENARIO
                        pro = [problem.name] * const.RUNS_PER_SCENARIO
                        alg = [algorithm.name_short] * const.RUNS_PER_SCENARIO
                        equ = [a == b for a, b in zip(perfs_a, perfs_b)]

                        # Create and store DataFrame for this combination
                        comp = pd.DataFrame(
                            zip(dim, bud, pro, alg, perfs_a, perfs_b, equ),
                            columns=col_names)
                        all_comps.append(comp)

        csv = pd.concat(all_comps)
        out_path = Path(f"csvs/{file_name}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv.to_csv(out_path, index=False)

        return

    def get_relevant_ngopt_algos(self: Experiment,
                                 ngopt: NGOptChoice) -> list[Algorithm]:
        """Get the algorithms NGOpt uses for the dimensionalities and budgets.

        Args:
            ngopt: Instance of NGOptChoice to enable retrieving algorithm
                choice of NGOpt for plotted dimensionalities and budgets.

        Returns:
            A list of short algorithm names.
        """
        algorithms = set()

        for budget in self.budgets:
            for dims in self.dimensionalities:
                algorithms.add(ngopt.get_ngopt_choice(dims, budget))

        algo_list = list(algorithms)
        algo_list.sort()

        return algo_list

    def score_algorithms(self: Experiment,
                         dims: int,
                         budget: int,
                         n_best: int,
                         score_per_prob: bool = False,
                         ngopt: NGOptChoice = None,
                         bud_specific: bool = False) -> pd.DataFrame:
        """Score algorithms based on their performance over multiple problems.

        Scores are based on the number of runs in the top n_best runs over all
        algorithms per problem, where for each run in the top n_best, one point
        is assigned.

        Args:
            dims: int indicating the number of variable space dimensions.
            budget: int indicating for which number of evaluations to score the
                algorithms.
            n_best: int indicating the top how many runs to look for.
            score_per_prob: If True include a column per problem with the score
                on that problem.
            ngopt: Instance of NGOptChoice to enable retrieving budget specific
                data for the algorithm choice of NGOpt, if available.
            bud_specific: Use budget specific data if available and set to
                True.

        Returns:
            DataFrame with columns: algorithm, points. The algorithm column
            holds Algorithm names in short form.
        """
        print(f"Ranking algorithms for {dims} dimensional problems with budget"
              f" {budget} ...")

        algo_names = [algo.name_short for algo in self.algorithms]
        algo_scores = pd.DataFrame({
            "algorithm": algo_names,
            "points": [0] * len(algo_names)})

        for problem in self.problems:
            best_algos = self.get_best_runs_of_prob(
                problem, budget, n_best, dims, ngopt=ngopt,
                bud_specific=bud_specific)

            # Count occurrences of algorithm
            algo_scores_for_prob = best_algos["algorithm"].value_counts()

            # Add counts to the scores
            algo_scores = pd.merge(
                algo_scores, algo_scores_for_prob, how="left", on="algorithm")
            algo_scores["count"].fillna(0, inplace=True)
            algo_scores["points"] += algo_scores["count"]

            if score_per_prob:
                algo_scores.rename(columns={"count": problem}, inplace=True)
            else:
                algo_scores.drop(columns=["count"], inplace=True)

        return algo_scores

    def write_medians_csv(self: Experiment,
                          file_name: str = "medians",
                          with_ranks: bool = False,
                          bud_specific: bool = False) -> None:
        """Write a CSV file with the medians per algorithm.

        The CSV contains the columns:
        dimensions, budget, problem, algorithm, median

        Args:
            file_name: Name of the file to write to. Will be written in the
                csvs/ directory with a .csv extension.
            with_ranks: If True, also include a column with the rank of the
                algorithm for this problem, based on the scores.
            bud_specific: Use budget specific data if available and set to
                True.
        """
        col_names = ["dimensions", "budget", "problem", "algorithm", "median"]

        if with_ranks:
            n_best = 25
            col_names.extend(["median_rank", "points", "score_rank"])

        all_medians = []

        for budget in self.budgets:
            for dims in self.dimensionalities:
                if with_ranks:
                    prob_scores = self.score_algorithms(
                        dims, budget, n_best, score_per_prob=True,
                        bud_specific=bud_specific)

                for problem in self.problems:
                    medians = self.get_medians_of_prob(problem, budget, dims)

                    n_algos = len(self.algorithms)
                    dim = [dims] * n_algos
                    buds = [budget] * n_algos
                    probs = [problem.name] * n_algos
                    algos = medians["algorithm"].tolist()
                    meds = medians["median"].tolist()

                    if with_ranks:
                        algo_points = prob_scores[["algorithm", problem]]
                        points = algo_points.set_index("algorithm").loc[
                            algos, problem].values
                        # The "min" method resolves ties by assigning the
                        # minimum of the ranks of all tied methods. E.g., if
                        # the best two are tied, they get the minimum of rank 1
                        # and 2 = 1.
                        m_ranks = ss.rankdata(meds, method="min")
                        # First take the negative of the points, to assign
                        # ranks in descending order since more points is
                        # better.
                        neg_points = [-1 * point for point in points]
                        s_ranks = ss.rankdata(neg_points, method="min")

                        prob_meds = pd.DataFrame(
                            zip(dim, buds, probs, algos, meds, m_ranks, points,
                                s_ranks), columns=col_names)
                    else:
                        prob_meds = pd.DataFrame(
                            zip(dim, buds, probs, algos, meds),
                            columns=col_names)

                    all_medians.append(prob_meds)

        csv = pd.concat(all_medians)
        out_path = Path(f"csvs/{file_name}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv.to_csv(out_path, index=False)

        return

    def write_scoring_csv(self: Experiment,
                          file_name: str = "scores",
                          bud_specific: bool = False) -> None:
        """Write a CSV file with the algorithm scores.

        The CSV contains the columns:
        dimensions, budget, problem, algorithm, score

        Args:
            file_name: Name of the file to write to. Will be written in the
                csvs/ directory with a .csv extension.
            bud_specific: Use budget specific data if available and set to
                True.
        """
        n_best = 25
        col_names = ["dimensions", "budget", "problem", "algorithm", "points"]
        all_scores = []

        for budget in self.budgets:
            for dims in self.dimensionalities:
                prob_scores = self.score_algorithms(
                    dims, budget, n_best, score_per_prob=True,
                    bud_specific=bud_specific)

                for _, row in prob_scores.iterrows():
                    row.drop("points", inplace=True)
                    n_problems = (len(row) - 1)
                    dim = [dims] * n_problems
                    buds = [budget] * n_problems
                    algo = [row["algorithm"]] * n_problems
                    row.drop("algorithm", inplace=True)
                    probs = [prob.name for prob in row.keys()]
                    scores = list(map(int, row.values))
                    algo_scores = pd.DataFrame(
                        zip(dim, buds, probs, algo, scores),
                        columns=col_names)
                    all_scores.append(algo_scores)

        csv = pd.concat(all_scores)
        out_path = Path(f"csvs/{file_name}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv.to_csv(out_path, index=False)

        return

    def write_score_rank_csv(self: Experiment,
                             file_name: str = "score_rank",
                             ngopt: NGOptChoice = None,
                             bud_specific: bool = False) -> None:
        """Write the algorithm rank based on scores over all problems to CVS.

        The CSV contains the columns:
        dimensions, budget, algorithm, points, rank, ngopt rank

        Args:
            file_name: Name of the file to write to. Will be written in the
                csvs/ directory with a .csv extension.
            ngopt: Instance of NGOptChoice to enable retrieving budget specific
                data for the algorithm choice of NGOpt, if available.
            bud_specific: Use budget specific data if available and set to
                True.
        """
        algo_matrix = self.get_scoring_matrix(ngopt, bud_specific=bud_specific)
        best_matrix = self._get_best_algorithms(algo_matrix, ngopt)
        col_names = ["dimensions", "budget", "algorithm", "points", "rank",
                     "ngopt rank", "ID"]
        all_scores = []

        # Get algorithm IDs
        names_csv = Path("csvs/ngopt_algos_0.6.0.csv")
        names_df = pd.read_csv(names_csv)
        name_dict = {}

        for _, algo in names_df.iterrows():
            algo_name = algo["short name"]
            algo_id = algo["ID"]
            name_dict[algo_name] = algo_id

        for budget in self.budgets:
            for dims in self.dimensionalities:
                algo_scores = algo_matrix.loc[dims].at[budget]
                n_algos = len(algo_scores)

                dim = [dims] * n_algos
                buds = [budget] * n_algos
                algos = algo_scores["algorithm"]
                points = algo_scores["points"]
                # First take the negative of the points, to assign ranks in
                # descending order since more points is better.
                neg_points = [-1 * point for point in points]
                ranks = ss.rankdata(neg_points, method="min")
                ngopt_ranks = ranks
                algo_ids = [name_dict[algo] for algo in algos]
                algo_ranks = pd.DataFrame(
                    zip(dim, buds, algos, points, ranks,
                        ngopt_ranks, algo_ids),
                    columns=col_names)

                # Set "ngopt rank" to 0 for the NGOpt choice, and -1 for the
                # data choice if it is different from the NGOpt choice
                ngopt_choice = ngopt.get_ngopt_choice(dims, budget)
                data_choice = best_matrix.loc[dims, budget]

                algo_ranks.loc[
                    algo_ranks["algorithm"] == ngopt_choice, "ngopt rank"] = 0

                if ngopt_choice != data_choice:
                    algo_ranks.loc[algo_ranks["algorithm"] == data_choice,
                                   "ngopt rank"] = -1

                # Sort by algorithm ID for historical consistency
                algo_ranks.sort_values("ID", inplace=True)
                algo_ranks.drop(columns="ID", inplace=True)

                all_scores.append(algo_ranks)

        csv = pd.concat(all_scores)
        out_path = Path(f"csvs/{file_name}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv.to_csv(out_path, index=False)

        return

    def get_scoring_matrix(self: Experiment,
                           ngopt: NGOptChoice = None,
                           bud_specific: bool = False) -> pd.DataFrame:
        """Get a matrix of algorithm scores for dimensionalities versus budget.

        Args:
            ngopt: Instance of NGOptChoice to enable retrieving budget specific
                data for the algorithm choice of NGOpt, if available.
            bud_specific: Use budget specific data if available and set to
                True.

        Returns:
            DataFrame with rows representing different dimensionalities and
                columns representing different evaluation budgets.
        """
        n_best = 25
        algo_matrix = pd.DataFrame()

        for budget in self.budgets:
            scores = []

            for dims in self.dimensionalities:
                scores.append(
                    self.score_algorithms(dims, budget, n_best, ngopt=ngopt,
                                          bud_specific=bud_specific))

            algo_matrix[budget] = scores

        algo_matrix.index = self.dimensionalities

        return algo_matrix

    def _get_best_algorithms(self: Experiment,
                             algo_matrix: pd.DataFrame,
                             ngopt: NGOptChoice,
                             blank_ngopt: bool = False) -> pd.DataFrame:
        """Retrieve the top ranked algorithms per budget-dimensionality pair.

        In case of a tie, if one of the top ranking algorithms matches with the
        choice of NGOpt, this one is shown. If none of the tied algorithms
        match NGOpt, the one that happens to be on top is shown.

        Args:
            algo_matrix: DataFrame with rows representing different
                dimensionalities and columns representing different evaluation
                budgets. Each cell with algorithm scores in a DataFrame with
                columns: algorithm, points
            ngopt: Instance of NGOptChoice to enable retrieving algorithm
                choice of NGOpt for plotted dimensionalities and budgets.
            blank_ngopt: If True, set cells with the same choice as NGOpt
                to an empty str.

        Returns:
            A DataFrame of short Algorithm names with rows representing
            different dimensionalities and columns representing different
            evaluation budgets.
        """
        best_matrix = pd.DataFrame()

        for budget in self.budgets:
            dims_best = []

            for dims in self.dimensionalities:
                algo_scores = algo_matrix.loc[dims].at[budget]

                # Retrieve all algorithms that are tied for first place
                algo_scores.sort_values(
                    "points", inplace=True, ascending=False)
                algo_scores = algo_scores.loc[
                    algo_scores["points"] == algo_scores["points"].iloc[0]]
                ngopt_algo = ngopt.get_ngopt_choice(dims, budget)

                # Prefer the NGOptChoice in case of a tie
                if ngopt_algo in algo_scores["algorithm"].values:
                    best = "" if blank_ngopt else ngopt_algo
                else:
                    best = algo_scores["algorithm"].values[0]

                dims_best.append(best)

            best_matrix[budget] = dims_best

        best_matrix.index = self.dimensionalities

        return best_matrix

    def plot_heatmap_ngopt(self: Experiment,
                           ngopt: NGOptChoice,
                           file_name: str = "grid_ngopt") -> None:
        """Plot a heatmap showing the best algorithm per budget-dimension pair.

        In case of a tie, if one of the top ranking algorithms matches with the
        choice of NGOpt, this one is shown. If none of the tied algorithms
        match NGOpt, the one that happens to be on top is shown.

        Args:
            ngopt: Instance of NGOptChoice to enable retrieving algorithm
                choice of NGOpt for plotted dimensionalities and budgets.
            file_name: Name of the file to write to. Will be written in the
                plots/heatmap/ directory with a _d{multiplier}.pdf extension.
        """
        ngopt_algos = [
            ngopt.get_ngopt_choice(dims, bud)
            for dims in self.dimensionalities for bud in self.budgets]
        algorithms = [algo.name_short for algo in self.algorithms]
        algo_ids = [algo.id for algo in self.algorithms]
        best_matrix = ngopt.get_ngopt_choices(
            self.dimensionalities, self.budgets)

        # Get indices for algorithms relevant for the plot
        ids_in_plot = [idx for idx, algo in zip(algo_ids, algorithms)
                       if algo in ngopt_algos]
        algos_in_plot = [algo for algo in algorithms if algo in ngopt_algos]
        colours = const.ALGO_COLOURS
        colours_in_plot = [colours[i] for i in ids_in_plot]

        # Dict mapping short names to ints, reduce to relevant algorithms
        algo_to_id = {algo: idx for idx, algo
                      in zip(ids_in_plot, algos_in_plot)}
        algo_to_int = {algo: i for i, algo in enumerate(algos_in_plot)}

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10.2, 5.6))
        ax = sns.heatmap(
            best_matrix.replace(algo_to_int), cmap=colours_in_plot,
            annot=best_matrix.replace(algo_to_id),
            annot_kws={"size": const.FONT_SIZE_ALGO_ID},
            square=True)
        ax.set(xlabel="evaluation budget", ylabel="dimensions")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", labelrotation=90)

        # Add algorithm names to colour bar
        colorbar = ax.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        n = len(algo_to_int)
        colorbar.set_ticks(
            [colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        # Update algo_to_int to include algorithm IDs for the legend
        algos_in_plot = [f"{idx}. {algo}" for idx, algo
                         in zip(ids_in_plot, algos_in_plot)]
        algo_to_int = {algo: i for i, algo in enumerate(algos_in_plot)}
        colorbar.set_ticklabels(list(algo_to_int.keys()))

        # Plot and save the figure
        plt.tight_layout()
        plt.show()
        out_path = Path(
            f"plots/heatmap/{file_name}_d{self.dim_multiplier}.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)

    def plot_heatmap_data(self: Experiment,
                          algo_matrix: pd.DataFrame,
                          ngopt: NGOptChoice,
                          blank_ngopt: bool = False,
                          file_name: str = "grid_data") -> None:
        """Plot a heatmap showing the best algorithm per budget-dimension pair.

        In case of a tie, if one of the top ranking algorithms matches with the
        choice of NGOpt, this one is shown. If none of the tied algorithms
        match NGOpt, the one that happens to be on top is shown.

        Args:
            algo_matrix: DataFrame with rows representing different
                dimensionalities and columns representing different evaluation
                budgets. Each cell with algorithm scores in a DataFrame with
                columns: algorithm, points
            ngopt: Instance of NGOptChoice to enable retrieving algorithm
                choice of NGOpt for plotted dimensionalities and budgets.
            blank_ngopt: If True, leave cells with the same choice as NGOpt
                blank.
            file_name: Name of the file to write to. Will be written in the
                plots/heatmap/ directory with a _d{multiplier}.pdf extension.
        """
        best_matrix = self._get_best_algorithms(
            algo_matrix, ngopt, blank_ngopt)
        # If blank_ngopt is True, empty strings may exist, treat them as NaN
        best_matrix.replace("", np.nan, inplace=True)

        algorithms = [algo.name_short for algo in self.algorithms]
        algo_ids = [algo.id for algo in self.algorithms]
        best_algos = best_matrix.values.flatten().tolist()

        # Get indices for algorithms relevant for the plot
        ids_in_plot = [idx for idx, algo in zip(algo_ids, algorithms)
                       if algo in best_algos]
        algos_in_plot = [algo for algo in algorithms if algo in best_algos]
        colours = const.ALGO_COLOURS
        colours_in_plot = [colours[i] for i in ids_in_plot]

        # Dict mapping short names to ints
        algo_to_id = {algo: idx for idx, algo
                      in zip(ids_in_plot, algos_in_plot)}
        algo_to_int = {algo: i for i, algo in enumerate(algos_in_plot)}

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10.2, 5.6))
        ax = sns.heatmap(
            best_matrix.replace(algo_to_int), cmap=colours_in_plot,
            annot=best_matrix.replace(algo_to_id),
            annot_kws={"size": const.FONT_SIZE_ALGO_ID},
            square=True)
        ax.set(xlabel="evaluation budget", ylabel="dimensions")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", labelrotation=90)

        # Add algorithm names to colour bar
        colorbar = ax.collections[0].colorbar
        r = colorbar.vmax - colorbar.vmin
        n = len(algo_to_int)
        colorbar.set_ticks(
            [colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
        # Update algo_to_int to include algorithm IDs for the legend
        algos_in_plot = [f"{idx}. {algo}" for idx, algo
                         in zip(ids_in_plot, algos_in_plot)]
        algo_to_int = {algo: i for i, algo in enumerate(algos_in_plot)}
        colorbar.set_ticklabels(list(algo_to_int.keys()))

        # Plot and save the figure
        plt.tight_layout()
        plt.show()
        prob_set = f"_probs_{self.prob_set}" if self.prob_set != "all" else ""
        out_path = Path(
            f"plots/heatmap/{file_name}{prob_set}_d{self.dim_multiplier}.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)

    def plot_hist_grid(self: Experiment,
                       algo_matrix: pd.DataFrame,
                       ngopt: NGOptChoice,
                       file_name: str = "grid") -> None:
        """Plot a grid of histograms showing algorithm scores.

        Args:
            algo_matrix: DataFrame with rows representing different
                dimensionalities and columns representing different evaluation
                budgets. Each cell with algorithm scores is a DataFrame with
                columns: algorithm, points
            ngopt: Instance of NGOptChoice to enable retrieving algorithm
                choice of NGOpt for plotted dimensionalities and budgets.
            file_name: Name of the file to write to. Will be written in the
                plots/bar/ directory with a _d{multiplier}.pdf extension.
        """
        top_n = 5
        top_algos = set()

        algorithms = [algo.name_short for algo in self.algorithms]
        algo_ids = [algo.id for algo in self.algorithms]

        # Get indices for algorithms relevant for the plot
        algos_in_plot = [algo for algo in algorithms if
                         algo in list(algo_matrix.values[0][0]["algorithm"])]
        ids_in_plot = [idx for idx, algo in zip(algo_ids, algorithms)
                       if algo in algos_in_plot]
        colours = const.ALGO_COLOURS
        colours_in_plot = {algo: colours[i]
                           for algo, i in zip(algos_in_plot, ids_in_plot)}

        rows = len(self.dimensionalities)
        cols = len(self.budgets)
        fig, axs = plt.subplots(rows, cols, layout="constrained",
                                figsize=(cols*7.4, rows*5.6), dpi=80)
        bud_dims = [(bud, dim) for dim in self.dimensionalities
                    for bud in self.budgets]

        for bud_dim, ax in zip(bud_dims, axs.flatten()):
            ngopt_algo = ngopt.get_ngopt_choice(bud_dim[1], bud_dim[0])
            algo_scores = algo_matrix.loc[bud_dim[1]].at[bud_dim[0]]
            algo_scores.sort_values(
                "points", inplace=True, ascending=False)
            algo_scores = algo_scores.head(top_n)
            sns.barplot(x=np.arange(top_n),
                        y="points",
                        hue="algorithm",
                        data=algo_scores,
                        palette=colours_in_plot,
                        ax=ax)

            # Loop to show label for every bar
            for bars in ax.containers:
                ax.bar_label(bars)

            ax.set_facecolor(self._get_bg_colour(algo_scores, ngopt_algo))
            ax.set_title(f"Dimensions: {bud_dim[1]}, Budget: {bud_dim[0]}, "
                         f"NGOpt choice: {ngopt_algo}",
                         color=self._get_bg_colour(algo_scores, ngopt_algo),
                         fontsize=9)
            ax.axis("off")

            # Save distinct top 5 algorithm names
            top_algos.update(algo_scores["algorithm"].values)

        # Print distinct algorithms that appear in the top 5 of dim-bud pairs
        print("Algorithms that appear in a top 5:")
        print(*top_algos, sep="\n")

        plt.show()
        prob_set = f"_probs_{self.prob_set}" if self.prob_set != "all" else ""
        out_path = Path(
            f"plots/bar/{file_name}{prob_set}_d{self.dim_multiplier}.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, facecolor=fig.get_facecolor())

    def _get_bg_colour(self: Experiment,
                       algo_scores: pd.DataFrame,
                       ngopt_algo: str) -> str:
        """Get the background colour based on match of NGOpt choice and data.

        Args:
            algo_scores: Top 5 rows of DataFrame with cols: algorithm, points.
            ngopt_algo: Short name of the algorithm chosen by NGOpt for the
                dimensionality and budget combination.

        Returns:
            Colour name to use in plot as a str.
        """
        # Retrieve all algorithms that are tied for first place
        tied_algo_scores = algo_scores.loc[
            algo_scores["points"] == algo_scores["points"].iloc[0]]

        # If it matches an algorithms with the highest points, use green
        if ngopt_algo in tied_algo_scores["algorithm"].values:
            return "green"
        # If it is in the top 5, use orange
        elif ngopt_algo in algo_scores["algorithm"].values:
            return "orange"
        # Otherwise, use red
        else:
            return "red"

    def plot_hist(self: Experiment,
                  algo_scores: pd.DataFrame,
                  ngopt_algo: str,
                  dims: int,
                  budget: int) -> None:
        """Plot a histogram showing algorithm scores.

        Args:
            algo_scores: DataFrame with columns: algorithm, points.
            ngopt_algo: Short name of the algorithm chosen by NGOpt for the
                dimensionality and budget combination.
            dims: The dimensionality algorithms are scored for.
            budget: The evaluation budget algorithms are scored for.
        """
        top_n = 5
        algo_scores.sort_values("points", inplace=True, ascending=False)
        ax = sns.barplot(x=np.arange(top_n),
                         y=algo_scores["points"].head(top_n),
                         label=algo_scores["algorithm"].head(top_n))
        ax.bar_label(ax.containers[0])
        ax.set_title(f"Dimensions: {dims}, Budget: {budget}, "
                     f"NGOpt choice: {ngopt_algo}")
        plt.axis("off")
        plt.legend(fontsize=4)
        plt.show()
        out_path = Path("plots/bar/test.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)

    def get_medians_of_prob(self: Experiment,
                            problem: Problem,
                            budget: int,
                            dims: int) -> pd.DataFrame:
        """Return the median per algorithm for a problem-dimension-budget set.

        Args:
            problem: A Problem object for which to get the data.
            budget: int indicating for which number of evaluations to get the
                data.
            dims: int indicating the dimensionality for which to get the data.

        Returns:
            DataFrame with a row per algorithm with name and median
                performance.
        """
        algorithms = []
        medians = []

        # Retrieve median performance per algorithm, counting successful
        # runs only.
        for algorithm in self.algorithms:
            scenario = self.prob_scenarios[problem][algorithm][dims]
            algorithms.extend([scenario.algorithm.name_short])
            runs = [run.get_performance(budget) for run in scenario.runs
                    if run.status == 1]
            meds = statistics.median(runs)
            medians.append(meds)

        # Create a DataFrame from the lists
        algo_med = pd.DataFrame({
            "algorithm": algorithms,
            "median": medians})

        return algo_med

    def get_best_runs_of_prob(self: Experiment,
                              problem: Problem,
                              budget: int,
                              n_best: int,
                              dims: int,
                              ngopt: NGOptChoice = None,
                              bud_specific: bool = False) -> pd.DataFrame:
        """Return the n_best runs for a problem, dimension, budget combination.

        If we have budget-specific data available as well, and the bud_specific
        option is set to True, the budget-specific data is used instead of the
        full run data, if the algorithm is the NGOpt choice.

        Args:
            problem: A Problem object for which to get the data.
            budget: int indicating for which number of evaluations to rank the
                algorithms.
            n_best: int indicating the top how many runs to look for.
            dims: int indicating the dimensionality for which to get the data.
            ngopt: Instance of NGOptChoice to enable retrieving budget specific
                data for the algorithm choice of NGOpt, if available.
            bud_specific: Use budget specific data if available and set to
                True.

        Returns:
            DataFrame with n_best rows of algorithm, run ID, and performance.
                Any rows beyond row n_best that have the same performance as
                row n_best are also returned.
        """
        algorithms = []
        run_ids = []
        performances = []

        # Retrieve performance and metadata per algorithm, counting successful
        # runs only.
        for algo in self.algorithms:
            # If we have per budget data and the algorithm is the NGOpt choice,
            # we use the budget specific data here instead of the full run
            if (bud_specific
                    and self.per_budget_data_dir is not None
                    and ngopt is not None
                    and budget < const.EVAL_BUDGET and
                    ngopt.get_ngopt_choice(dims, budget) == algo.name_short):
                scenario = (
                    self.prob_scenarios_per_b[problem][algo][dims][budget])
            else:
                scenario = self.prob_scenarios[problem][algo][dims]

            n_runs_suc = sum(1 for run in scenario.runs if run.status == 1)
            algorithms.extend([scenario.algorithm.name_short] * n_runs_suc)
            run_ids.extend(
                [run.idx for run in scenario.runs if run.status == 1])
            performances.extend(
                [run.get_performance(budget) for run in scenario.runs
                 if run.status == 1])

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

        # Also return rows beyond n_best with equal performance to row n_best
        best_runs_plus = runs.loc[
            runs["performance"] == best_runs["performance"].iloc[-1]]
        best_runs = pd.concat([best_runs, best_runs_plus])

        return best_runs


class Problem:
    """Manages problem properties."""

    def __init__(self: Problem,
                 prob_name: str,
                 prob_id: int) -> None:
        """Initialise a Problem object.

        Args:
            prob_name: Name of the problem.
            prob_id: ID of the problem.
        """
        self.name = prob_name
        self.id = prob_id


class NGOptChoice:
    """Manages algorithm choices by NGOpt."""

    def __init__(self: NGOptChoice,
                 hsv_file: Path) -> None:
        """Initialise an NGOptChoice object.

        Args:
            hsv_file: Path to a # (hash) separated value file.
        """
        self.hsv_file = hsv_file
        self._load_data()

        return

    def _load_data(self: NGOptChoice) -> None:
        """Load NGOpt choices from file."""
        self.ngopt_choices = pd.read_csv(self.hsv_file, sep="#")

    def get_ngopt_choice(self: NGOptChoice,
                         dims: int,
                         budget: int,
                         short_name: bool = True) -> str:
        """Return the algorithm NGOpt chose for a dimensionality and budget.

        Args:
            dims: Dimensionality of the search space (number of variables).
            budget: The evaluation budget for which to get the NGOpt choice.
            short_name: Flag whether to return algorithm names in short form
                or not.

        Returns:
            The name of the algorithm NGOpt chose.
        """
        # Take the right dimensionality and remove too large budgets
        relevant_rows = self.ngopt_choices.loc[
            (self.ngopt_choices["dimensionality"] == dims)
            & (self.ngopt_choices["budget"] <= budget)]

        # Largest budget remaining is the correct row
        right_row = relevant_rows[
            relevant_rows["budget"] == relevant_rows["budget"].max()]

        # Retrieve the algorithm name
        algo_name = right_row.values[0][0]

        # Remove class coding from NGOpt versions
        algo_name = self._remove_algo_class_coding(algo_name)

        if short_name:
            return Algorithm(algo_name).name_short
        else:
            return algo_name

    def _remove_algo_class_coding(self: NGOptChoice,
                                  algo_name: str) -> str:
        """Return the algorithm name without class coding.

        Args:
            algo_name: Name of the algorithm.

        Returns:
            Name of the algorithm without class coding. If there is no class
            coding in the name, this is equivalent to the input.
        """
        algo_name = algo_name.replace(
            "<class 'nevergrad.optimization.optimizerlib.", "")
        algo_name = algo_name.replace("'>", "")

        return algo_name

    def get_ngopt_choices(self: NGOptChoice,
                          dimensionalities: list[int],
                          budgets: list[int],
                          short_name: bool = True) -> pd.DataFrame:
        """Return NGOpt's choices for given dimensionalities and budgets.

        Args:
            dimensionalities: Dimensionalities of the search space (number of
                variables).
            budget: The evaluation budgets for which to get the NGOpt choices.
            short_name: Flag whether to return algorithm names in short form
                or not.

        Returns:
            The names of the algorithms NGOpt chose in a DataFrame with
            dimensionalities as rows and budgets as columns.
        """
        algo_matrix = pd.DataFrame()

        for budget in budgets:
            algos = []

            for dims in dimensionalities:
                algos.append(self.get_ngopt_choice(dims, budget, short_name))

            algo_matrix[budget] = algos

        algo_matrix.index = dimensionalities

        return algo_matrix

    def write_unique_ngopt_algos_csv(self: NGOptChoice,
                                     file_name: str = "ngopt_algos") -> None:
        """Write unique NGOpt choices to CSV for all dimensionalities, budgets.

        The CSV file contains the columns: short name, full name, ID.

        Args:
            file_name: Name of the file to write to. Will be written in the
                csvs/ directory with a .csv extension.
        """
        algo_names = set(self.ngopt_choices["algorithm"].values)
        algo_names = [
            self._remove_algo_class_coding(name) for name in algo_names]
        col_names = ["ID", "short name", "full name"]
        algos = [Algorithm(algo) for algo in algo_names]
        algo_short_names = [algo.name_short for algo in algos]
        algo_ids = [algo.id for algo in algos]

        unique_algos = pd.DataFrame(
            zip(algo_ids, algo_short_names, algo_names), columns=col_names)
        unique_algos.sort_values("ID", inplace=True)

        out_path = Path(f"csvs/{file_name}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        unique_algos.to_csv(out_path, index=False)

        return

    def write_ngopt_choices_csv(self: NGOptChoice,
                                dimensionalities: list[int],
                                budgets: list[int],
                                file_name: str = "ngopt_choices") -> None:
        """Write NGOpt's choices to CSV for given dimensionalities and budgets.

        The CSV file contains the columns: dimensions, budget, algorithm.

        Args:
            dimensionalities: Dimensionalities of the search space (number of
                variables).
            budget: The evaluation budgets for which to get the NGOpt choices.
            file_name: Name of the file to write to. Will be written in the
                csvs/ directory with a .csv extension.
        """
        all_choices = []
        col_names = ["dimensions", "budget", "algorithm"]
        algo_matrix = self.get_ngopt_choices(dimensionalities, budgets,
                                             short_name=False)

        for dims, row in algo_matrix.iterrows():
            n_buds = len(row)
            dim = [dims] * n_buds
            algos = [Algorithm(algo).id for algo in row.values]
            algo_choices = pd.DataFrame(
                zip(dim, budgets, algos), columns=col_names)
            all_choices.append(algo_choices)

        csv = pd.concat(all_choices)
        out_path = Path(f"csvs/{file_name}.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        csv.to_csv(out_path, index=False)

        return


class Algorithm:
    """Manages algorithm properties."""

    def __init__(self: Algorithm,
                 name: str) -> None:
        """Initialise an Algorithm object.

        Args:
            name: Full name of the algorithm.
        """
        self.name = name
        self.name_short = self.get_short_algo_name()
        self.id = const.ALGS_CONSIDERED.index(self.name)

        return

    def get_short_algo_name(self: Algorithm) -> str:
        """Return a str with a short name for a given algorithm name.

        Returns:
            algo_name for algorithms that already have a short name, or a
                shortened str for algorithms that have a lengthy name.
        """
        short_name = self.name

        if self.name.startswith("ConfPortfolio"):
            scl09 = "scale=0.9"
            scl13 = "scale=1.3"
            scnd_scale = "NA"

            if scl09 in self.name:
                scnd_scale = scl09
            elif scl13 in self.name:
                scnd_scale = scl13

            n_ngopt = self.name.count("NGOpt14")
            short_name = (
                f"ConfPortfolio_scale2_{scnd_scale}_ngopt14s_{n_ngopt}")

        return short_name


class Run:
    """Manages run properties."""

    def __init__(self: Run,
                 idx: int,
                 seed: int,
                 status: int,
                 instance: int,
                 eval_ids: list[int],
                 perf_vals: list[int],
                 expected_evals: int) -> None:
        """Initialise a Run object.

        Args:
            idx: ID of the run.
            seed: Seed used for the algorithm in the run.
            status: Exit status of the run. 1 indicates a successful run, 0,
                -2, -3 a crashed run, -1 a missing run, -4 a incomplete run
                detected during data reading. Other values than these
                mean something is likely to be wrong, e.g., a crash that was
                not detected during execution can have a value like
                4.6355715189945e-310.
            instance: int indicating the instance used for the run.
            eval_ids: List of evaluation IDs where a performance improvement
                was found during the run. The last evaluation is always
                included for successful runs (i.e., this ID should be equal to
                expected_evals). Should be equal length to perf_vals.
            perf_vals: List of performance values matching the evaluation IDs
                from the eval_ids variable. Should be equal length to eval_ids.
            expected_evals: Expected number of evaluations for this run.
        """
        self.idx = idx
        self.seed = seed
        self.status = status
        self.instance = instance
        self.eval_ids = eval_ids
        self.perf_vals = perf_vals
        self.complete = self.check_run_is_valid(expected_evals)

        return

    def check_run_is_valid(self: Run,
                           expected_evals: int) -> bool:
        """Check whether run has the right number of evaluations.

        Args:
            expected_evals: int with the expected number of evaluations in the
                run.

        Returns:
            bool True if eval_number and expected_evals match, False otherwise.
        """
        if self.eval_ids[-1] == expected_evals:
            return True
        else:
            print(f"Run with ID {self.idx} is partial with only "
                  f"{self.eval_ids[-1]} evaluations instead of "
                  f"{expected_evals}.")
            self.status = -4

            return False

    def get_performance(self: Run,
                        budget: int) -> float:
        """Return the performance of this Run at a specific evaluation budget.

        Args:
            budget: The evaluation budget for which to get the performance.

        Returns:
            The performance at the specified evaluation budget as a float.
        """
        perf = self.perf_vals[0]

        for eval_id, perf_val in zip(self.eval_ids, self.perf_vals):
            if budget <= eval_id:
                return perf
            else:
                perf = perf_val


class Scenario:
    """Holds an experimental scenario and its properties."""

    def __init__(self: Scenario,
                 data_dir: Path,
                 problem: Problem,
                 algorithm: Algorithm,
                 dims: int,
                 n_runs: int,
                 n_evals: int,
                 n_instances: int = 1,
                 per_budget: bool = False,
                 json_file: Path = None,
                 verbose: bool = False) -> None:
        """Initialise the Scenario.

        Args:
            data_dir: data_dir: Path to the data directory.
                This directory should have subdirectories per problem, which in
                turn should have subdirectories per algorithm, which should be
                organised in IOH format. E.g. for directory data, algorithm
                CMA, and problem f1_Sphere it should look like:
                data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
                data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat
            problem: Problem used in the scenario.
            algorithm: Algorithm used in the scenario.
            dims: Dimensionality of the search space (number of variables).
            n_runs: Number of runs performed with these settings.
            n_evals: Number of evaluations per run.
            n_instances: Number of instances to run on.
            per_budget: If set to True, treat data_dir as being organised with
                subdirectories named by algorithm-dimensionality-budget. These
                should each be organised in IOH format, and contain both a
                .json and directory for each BBOB function, specific to the
                dimensionality and budget combination for this subdirectory.
            json_file: Path to an IOH .json file for the Scenario. data_dir is
                treated as the parent directory of the file if this is set.
            verbose: If True print more detailed information.
        """
        self.data_dir = data_dir
        self.problem = problem
        self.algorithm = algorithm
        self.dims = dims
        self.n_runs = n_runs
        self.n_evals = n_evals
        self.n_instances = n_instances
        self.n_cases = self.n_runs * self.n_instances
        self.runs = []

        if per_budget:
            json_path = Path(
                f"{self.data_dir}/"
                f"{self.algorithm.name_short}-{self.dims}-{self.n_evals}/"
                f"IOHprofiler_{self.problem.name}.json")
        elif json_file is not None:
            json_path = json_file
        else:
            json_path = Path(
                f"{self.data_dir}/{self.problem.name}/"
                f"{self.algorithm.name_short}/"
                f"IOHprofiler_{self.problem.name}.json")
        self._load_data(json_path)

    def _load_data(self: Scenario,
                   json_file: Path,
                   verbose: bool = False) -> None:
        """Load the data associated with this scenario.

        Args:
            json_file: Path to an IOH experiment metadata json file.
            verbose: If True print more detailed information.
        """
        result_path, run_seeds, run_statuses, instances = self._read_ioh_json(
            json_file, verbose)
        self._read_ioh_dat(
            result_path, run_seeds, run_statuses, instances, verbose)

    def _read_ioh_json(self: Scenario,
                       metadata_path: Path,
                       verbose: bool = False) -> (
                       Path, list[int], list[int], list[int]):
        """Read a .json metadata file from an experiment with IOH.

        Args:
            metadata_path: Path to IOH metadata file.
            verbose: If True print more detailed information.

        Returns:
            Path to the data file or empty Path if no file is found.
            list of ints indicating the seed used for the run.
            list of usually ints showing the success/failure status of runs for
                this dimensionality. 1 indicates a successful run, 0, -2, -3 a
                crashed run, -1 a missing run. Other values than these mean
                something is likely to be wrong, e.g., a crash that was not
                detected during execution can have a value like
                4.6355715189945e-310. An empty list is returned if no file is
                found.
            instances: list of ints indicating the instance used for the run.
        """
        if verbose:
            print(f"Reading json file: {metadata_path}")

        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)

        for scenario in metadata["scenarios"]:
            if scenario["dimension"] == self.dims:
                data_path = Path(scenario["path"])

                # Record per run the seed and whether it was successful
                run_success = [-1] * self.n_cases
                seeds = [-1] * self.n_cases
                instances = [-1] * self.n_cases

                for run, idx in zip(scenario["runs"], range(0, self.n_cases)):
                    run_success[idx] = run["run_success"]
                    seeds[idx] = run["algorithm_seed"]
                    instances[idx] = run["instance"]

                n_success = sum(
                    run_suc for run_suc in run_success if run_suc == 1)

                if n_success != self.n_cases:
                    print(f"Found {n_success} successful runs * instances out "
                          f"of {len(scenario['runs'])} instead of "
                          f"{self.n_cases} runs * instances for function "
                          f"{self.problem.name} with "
                          f"algorithm {self.algorithm.name_short} and "
                          f"dimensionality {self.dims}.")

                break
            elif verbose:
                print(f"No dimension match, file has {scenario['dimension']}, "
                      f"was looking for {self.dims}")

        # Check whether a path to the data was identified
        try:
            data_path = metadata_path.parent / data_path
        except UnboundLocalError:
            print(f"No data found for function {self.problem.name} with "
                  f"algorithm {self.algorithm.name_short} and dimensionality "
                  f"{self.dims}.")
            data_path = Path()
            run_success = list()

        return (data_path, seeds, run_success, instances)

    def _read_ioh_dat(self: Scenario,
                      result_path: Path,
                      seeds: list[int],
                      run_statuses: list[int],
                      instances: list[int],
                      verbose: bool = False) -> None:
        """Read a .dat result file with runs from an experiment with IOH.

        These files contain data blocks representing one run each of the form:
            evaluations raw_y
            1 1.0022434918
            ...
            10000 0.0000000000
        The first line indicates the start of a new run, and which data columns
        are included. Following this, each line represents data from one
        evaluation. evaluations indicates the evaluation number.
        raw_y indicates the best value so far, except for the last line. The
        last line holds the value of the last evaluation, even if it is not the
        best so far.

        Args:
            result_path: Path pointing to an IOH data file.
            seeds: list of ints indicating the seed used for the run
            run_statuses: list of run statuses to be stored with the runs read
                from the .dat file.
            instances: list of ints indicating the instance used for the run.
            verbose: If True print more detailed information.
        """
        if verbose:
            print(f"Reading dat file: {result_path}")

        with result_path.open("r") as result_file:
            lines = result_file.readlines()
            run_id = 0
            eval_ids = []
            perf_vals = []

            for line in lines:
                if line.startswith("e"):  # For 'evaluations'
                    if run_id != 0:
                        run = Run(run_id, seeds[run_id - 1],
                                  run_statuses[run_id - 1],
                                  instances[run_id - 1],
                                  eval_ids, perf_vals, self.n_evals)
                        self.runs.append(run)

                    eval_ids = []
                    perf_vals = []
                    run_id = run_id + 1
                else:
                    words = line.split()
                    eval_ids.append(int(words[0]))
                    perf_vals.append(float(words[1]))

            run = Run(run_id, seeds[run_id - 1],
                      run_statuses[run_id - 1],
                      instances[run_id - 1],
                      eval_ids, perf_vals, self.n_evals)
            self.runs.append(run)

        return
