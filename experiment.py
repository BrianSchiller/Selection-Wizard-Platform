"""Module with class definitions to describe an experiment and its data."""

import json

import constants as const


class Experiment:
    """Holds an experiment and its properties."""

    def __init__(self, data_dir: Path) -> None:
        """Initialise the Experiment.

        Args:
            data_dir: Path to the data directory.
                This directory should have subdirectories per problem, which in
                turn should have subdirectories per algorithm, which should be
                organised in IOH format. E.g. for directory data, algorithm CMA,
                and problem f1_Sphere it should look like:
                data/f1_Sphere/CMA/IOHprofiler_f1_Sphere.json
                data/f1_Sphere/CMA/data_f1_Sphere/IOHprofiler_f1_DIM10.dat
        """
        self.data_dir = data_dir
        self.problems = []
        self.algorithms = []

        for prob_name, prob_id in zip(const.PROB_NAMES, const.PROBS_CONSIDERED):
            self.problems.append(Problem(prob_name, prob_id))

        for algo_name in const.ALGS_CONSIDERED:
            self.algorithms.append(Algorithm(algo_name))

        return

    def load_data(self, verbose: bool = False) -> None:
        """Read IOH result files from the data directory.

        Args:
            verbose: If True print more detailed information.
        """
        for dims in const.DIMS_CONSIDERED:
            print(f"Reading data for {dims} dimensional problems...")

            prob_runs = []
            algo_names = []
            func_names = []

            for problem in self.problems:
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

        return


class Scenario:
    """Holds an experimental scenario and its properties."""

    def __init__(self,
                 data_dir: Path,
                 dims: int,
                 problem: Problem,
                 algorithm: Algorithm,
                 n_runs: int,
                 n_evals: int) -> None:
        """Initialise the Scenario."""
        self.json_file = json_file
        read_ioh_json(json_file)
        
    dimensions
    eval_budget
    problem
    algorithm
    runs
    json_file
    dat_file

class Problem:
    """Manages problem properties."""

    def __init__(self, prob_name: str, prob_id: int):
        """Initialise a Problem object."""
        self.name = prob_name
        self.id = prob_id

class Algorithm:
    """Manages algorithm properties."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.name_short = self.get_short_algo_name()

        return

    def get_short_algo_name(self) -> str:
        """Return a str with a short name for a given algorithm name.
    
        Returns:
            algo_name for algorithms that already have a short name, or a shortened
                str for algorithms that have a lengthy name.
        """
        short_name = algo_name
    
        if algo_name.startswith("ConfPortfolio"):
            scl09 = "scale=0.9"
            scl13 = "scale=1.3"
            scnd_scale = "NA"
    
            if scl09 in algo_name:
                scnd_scale = scl09
            elif scl13 in algo_name:
                scnd_scale = scl13
    
            n_ngopt = algo_name.count("NGOpt14")
            short_name = (
                f"ConfPortfolio_scale2_{scnd_scale}_ngopt14s_{n_ngopt}")
    
        return short_name


class Run:
    seed
    status
    evaluations: list[int]
    raw_y: list[int]

experiment.f1.cma.run
