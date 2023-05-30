"""Module with class definitions to describe an experiment and its data."""
from __future__ import annotations

from pathlib import Path
import json

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
        p_scenarios = {}

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

            p_scenarios[problem] = a_scenarios

        return


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
                -2, -3 a crashed run, -1 a missing run. Other values than these
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
            return False


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
