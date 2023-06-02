"""Module with class definitions to describe an experiment and its data."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import constants as const


class Experiment:
    """Holds an experiment and its properties."""

    def __init__(self: Experiment,
                 data_dir: Path,
                 dimensionalities: list[int] = const.DIMS_CONSIDERED) -> None:
        """Initialise the Experiment.

        Args:
            data_dir: Path to the data directory.
                This directory should have subdirectories per problem, which in
                turn should have subdirectories per algorithm, which should be
                organised in IOH format. E.g. for directory data, algorithm
                CMA, and problem f1_Sphere it should look like:
                data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
                data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat
            dimensionalities (optional): List of ints indicating which
                dimensionalities to handle for the Experiment.
        """
        self.data_dir = data_dir
        self.problems = []
        self.algorithms = []
        self.dimensionalities = dimensionalities
        self.prob_scenarios = {}
        self.dim_multiplier = 100

        for prob_name, prob_id in zip(const.PROB_NAMES,
                                      const.PROBS_CONSIDERED):
            self.problems.append(Problem(prob_name, prob_id))

        for algo_name in const.ALGS_CONSIDERED:
            self.algorithms.append(Algorithm(algo_name))

        self.load_data()

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
                                                 const.EVAL_BUDGET)

                a_scenarios[algorithm] = d_scenarios

            self.prob_scenarios[problem] = a_scenarios

        return

    def rank_algorithms(self: Experiment,
                        dims: int,
                        budget: int,
                        n_best: int) -> pd.DataFrame:
        """Rank algorithms based on their performance over multiple problems.

        Args:
            dims: int indicating the number of variable space dimensions.
            budget: int indicating for which number of evaluations to rank the
                algorithms.
            n_best: int indicating the top how many runs to look for.

        Returns:
            DataFrame with columns: algorithm, points
        """
        print(f"Ranking algorithms for {dims} dimensional problems with budget"
              f" {budget} ...")

        algo_names = [algo.name_short for algo in self.algorithms]
        algo_scores = pd.DataFrame({
            "algorithm": algo_names,
            "points": [0] * len(algo_names)})

        for problem in self.problems:
            best_algos = self.get_best_runs_of_prob(
                problem, budget, n_best, dims)

            # Count occurences of algorithm
            algo_scores_for_prob = best_algos["algorithm"].value_counts()

            # Add counts to the scores
            algo_scores = pd.merge(
                algo_scores, algo_scores_for_prob, how="left", on="algorithm")
            algo_scores["count"].fillna(0, inplace=True)
            algo_scores["points"] += algo_scores["count"]
            algo_scores.drop(columns=["count"], inplace=True)

        return algo_scores

    def get_ranking_matrix(self: Experiment) -> pd.DataFrame:
        """Get a matrix algorithm rankings for dimensionalities versus budget.

        Returns:
            DataFrame with rows representing different dimensionalities and
                columns representing different evaluation budgets.
        """
        n_best = 25
        # use 10 * dims budgets
        budgets = [
            dims * self.dim_multiplier for dims in self.dimensionalities]
        algo_matrix = pd.DataFrame()

        for budget in budgets:
            ranks = []

            for dims in self.dimensionalities:
                ranks.append(self.rank_algorithms(dims, budget, n_best))

            algo_matrix[budget] = ranks

        algo_matrix.index = self.dimensionalities

        return algo_matrix

    def plot_hist_grid(self: Experiment,
                       algo_matrix: pd.DataFrame,
                       ngopt: NGOptChoice) -> None:
        """Plot a grid of histograms showing algorithm scores.

        Args:
            algo_matrix: DataFrame with rows representing different
                dimensionalities and columns representing different evaluation
                budgets. Each cell with algorithm scores in a DataFrame with
                columns: algorithm, points
            ngopt: Instance of NGOptChoice to enable retrieving algorithm
                choice of NGOpt for plotted dimensionalities and budgets.
        """
        top_n = 5
        budgets = [
            dims * self.dim_multiplier for dims in self.dimensionalities]
        algorithms = algo_matrix.values[0][0]["algorithm"]
        colours = sns.color_palette("colorblind", len(algorithms))
        palette = {algorithm: colour
                   for algorithm, colour in zip(algorithms, colours)}

        rows = len(self.dimensionalities)
        cols = len(budgets)
        fig, axs = plt.subplots(rows, cols, layout="constrained",
                                figsize=(cols*7.4, rows*5.6), dpi=80)
        bud_dims = [
            (bud, dim) for dim in self.dimensionalities for bud in budgets]

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
                        palette=palette,
                        ax=ax)

            # Loop to show label for every bar
            for bars in ax.containers:
                ax.bar_label(bars)

            ax.set_facecolor(self._get_bg_colour(algo_scores, ngopt_algo))
            ax.set_title(f"Dimensions: {bud_dim[1]}, Budget: {bud_dim[0]}, "
                         f"NGOpt choice: {ngopt_algo.name_short}",
                         color=self._get_bg_colour(algo_scores, ngopt_algo),
                         fontsize=9)
            ax.axis("off")

        plt.axis("off")
        plt.show()
        out_path = Path(f"plots/bar/grid_d{self.dim_multiplier}.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, facecolor=fig.get_facecolor())

    def _get_bg_colour(self: Experiment,
                       algo_scores: pd.DataFrame,
                       ngopt_algo: Algorithm) -> str:
        """Get the background colour based on match of NGOpt choice and data.

        Args:
            algo_scores: Top 5 rows of DataFrame with cols: algorithm, points.
            ngopt_algo: Algorithm chosen by NGOpt for the dimensionality and
                budget combination.

        Returns:
            Colour name to use in plot as a str.
        """
        # Retrieve all algorithms that are tied for first place
        tied_algo_scores = algo_scores.loc[
            algo_scores["points"] == algo_scores["points"].iloc[0]]

        # If it matches an algorithms with the highest points, use green
        if ngopt_algo.name_short in tied_algo_scores["algorithm"].values:
            return "green"
        # If it is in the top 5, use orange
        elif ngopt_algo.name_short in algo_scores["algorithm"].values:
            return "orange"
        # Otherwise, use red
        else:
            return "red"

    def plot_hist(self: Experiment,
                  algo_scores: pd.DataFrame,
                  ngopt_algo: Algorithm,
                  dims: int,
                  budget: int) -> None:
        """Plot a histogram showing algorithm scores.

        Args:
            algo_scores: DataFrame with columns: algorithm, points.
            ngopt_algo: Algorithm chosen by NGOpt for the dimensionality and
                budget combination.
            dims: The dimensionality algorithms are ranked for.
            budget: The evaluation budget algorithms are ranked for.
        """
        top_n = 5
        algo_scores.sort_values("points", inplace=True, ascending=False)
        ax = sns.barplot(x=np.arange(top_n),
                         y=algo_scores["points"].head(top_n),
                         label=algo_scores["algorithm"].head(top_n))
        ax.bar_label(ax.containers[0])
        ax.set_title(f"Dimensions: {dims}, Budget: {budget}, "
                     f"NGOpt choice: {ngopt_algo.name_short}")
        plt.axis("off")
        plt.legend(fontsize=4)
        plt.show()
        out_path = Path("plots/bar/test.pdf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)

    def get_best_runs_of_prob(self: Experiment,
                              problem: Problem,
                              budget: int,
                              n_best: int,
                              dims: int) -> pd.DataFrame:
        """Return the n best runs for a problem, dimension, budget combination.

        Args:
            problem: A Problem object for which to get the data.
            budget: int indicating for which number of evaluations to rank the
                algorithms.
            n_best: int indicating the top how many runs to look for.
            dims: int indicating the dimensionality for which to get the data.

        Returns:
            DataFrame with n_best rows of algorithm, run ID, and performance.
                Any rows beyond row n_best that have the same performance as
                row n_best are also returned.
        """
        algorithms = []
        run_ids = []
        performances = []

        # Retrieve performance and metadata per algorithm, counting succesful
        # runs only.
        for algorithm in self.algorithms:
            scenario = self.prob_scenarios[problem][algorithm][dims]
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
                         budget: int) -> Algorithm:
        """Return the algorithm NGOpt chose for a dimensionality and budget.

        Args:
            dims: Dimensionality of the search space (number of variables).
            budget: The evaluation budget for which to get the NGOpt choice.

        Returns:
            An Algorithm object with the algorithm NGOpt chose.
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

        return Algorithm(algo_name)


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
            eval_ids: List of evaluation IDs where a performance improvement
                was found during the run. The last evaluation is always
                included for succesful runs (i.e., this ID should be equal to
                expected_evals). Should be equal length to perf_vals.
            perf_vals: List of performance values matching the evaluation IDs
                from the eval_ids variable. Should be equal length to eval_ids.
            expected_evals: Expected number of evaluations for this run.
        """
        self.idx = idx
        self.seed = seed
        self.status = status
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
            verbose: If True print more detailed information.
        """
        self.data_dir = data_dir
        self.problem = problem
        self.algorithm = algorithm
        self.dims = dims
        self.n_runs = n_runs
        self.n_evals = n_evals
        self.runs = []

        json_file = Path(
            f"{self.data_dir}/{self.problem.name}/{self.algorithm.name_short}/"
            f"IOHprofiler_{self.problem.name}.json")
        self._load_data(json_file)

    def _load_data(self: Scenario,
                   json_file: Path,
                   verbose: bool = False) -> None:
        """Load the data associated with this scenario.

        Args:
            json_file: Path to an IOH experiment metadata json file.
            verbose: If True print more detailed information.
        """
        result_path, run_seeds, run_statuses = self._read_ioh_json(
            json_file, verbose)
        self._read_ioh_dat(result_path, run_seeds, run_statuses, verbose)

    def _read_ioh_json(self: Scenario,
                       metadata_path: Path,
                       verbose: bool = False) -> (Path, list[int], list[int]):
        """Read a .json metadata file from an experiment with IOH.

        Args:
            metadata_path: Path to IOH metadata file.
            verbose: If True print more detailed information.

        Returns:
            Path to the data file or empty Path if no file is found.
            list of ints indicating the seed used for the run
            list of usually ints showing the success/failure status of runs for
                this dimensionality. 1 indicates a successful run, 0, -2, -3 a
                crashed run, -1 a missing run. Other values than these mean
                something is likely to be wrong, e.g., a crash that was not
                detected during execution can have a value like
                4.6355715189945e-310. An empty list is returned if no file is
                found.
        """
        if verbose:
            print(f"Reading json file: {metadata_path}")

        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)

        for scenario in metadata["scenarios"]:
            if scenario["dimension"] == self.dims:
                data_path = Path(scenario["path"])

                # Record per run the seed and whether it was successful
                run_success = [-1] * self.n_runs
                seeds = [-1] * self.n_runs

                for run, idx in zip(scenario["runs"], range(0, self.n_runs)):
                    run_success[idx] = run["run_success"]
                    seeds[idx] = run["algorithm_seed"]

                n_success = sum(
                    run_suc for run_suc in run_success if run_suc == 1)

                if n_success != self.n_runs:
                    print(f"Found {n_success} successful runs out of "
                          f"{len(scenario['runs'])} instead of "
                          f"{self.n_runs} runs for function "
                          f"{self.problem.name} with "
                          f"algorithm {self.algorithm.name_short} and "
                          f"dimensionality {self.dims}.")

                break

        # Check whether a path to the data was identified
        try:
            data_path = metadata_path.parent / data_path
        except UnboundLocalError:
            print(f"No data found for function {self.problem.name} with "
                  f"algorithm {self.algorithm.name_short} and dimensionality "
                  f"{self.dims}.")
            data_path = Path()
            run_success = list()

        return (data_path, seeds, run_success)

    def _read_ioh_dat(self: Scenario,
                      result_path: Path,
                      seeds: list[int],
                      run_statuses: list[int],
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
                      eval_ids, perf_vals, self.n_evals)
            self.runs.append(run)

        return
