"""Module to run Nevergrad algorithm implementations with IOH profiler."""
from __future__ import annotations
import ioh
import argparse
import math
import sys

import numpy as np
import pandas as pd

import nevergrad as ng
from nevergrad.optimization.optimizerlib import Cobyla  # noqa: F401
from nevergrad.optimization.optimizerlib import MetaModel  # noqa: F401
from nevergrad.optimization.optimizerlib import CMA  # noqa: F401
from nevergrad.optimization.optimizerlib import ParametrizedMetaModel  # noqa: F401, E501
from nevergrad.optimization.optimizerlib import CmaFmin2  # noqa: F401
from nevergrad.optimization.optimizerlib import ChainMetaModelPowell  # noqa: F401, E501
from nevergrad.optimization.optimizerlib import MetaModelOnePlusOne  # noqa: F401, E501
from nevergrad.optimization.optimizerlib import ConfPortfolio  # noqa: F401
from nevergrad.optimization.optimizerlib import Rescaled  # noqa: F401
from nevergrad.optimization.optimizerlib import NGOpt14  # noqa: F401

import constants as const


class NGEvaluator:
    """Algorithm wrapper to use nevergrad algorithms with IOHprofiler."""

    algorithm_seed = 1
    run_success = -1  # "UNKNOWN"

    def __init__(self: NGEvaluator, optimizer: str, eval_budget: int) -> None:
        """Initialise the NGEvaluator.

        Args:
            optimizer: str with the algorithm name.
            eval_budget: int with the evaluation budget.
        """
        self.alg = optimizer
        self.eval_budget = eval_budget

    def __call__(self: NGEvaluator,
                 func: ioh.iohcpp.problem.RealSingleObjective,
                 seed: int) -> None:
        """Run the NGEvaluator on the given problem.

        Sets run_success in IOH json output per run with status codes:
            -1  UNKNOWN
             1  SUCCESS
             0  CRASHED due to OverflowError
            -2  CRASHED due to np.linalg.LinAlgError
            -3  CRASHED due to other Exception

        Args:
            func: IOH function to run the algorithm on.
            seed: int to seed the algorithm random state.
        """
        lower_bound = -5
        upper_bound = 5
        self.algorithm_seed = seed
        np.random.seed(self.algorithm_seed)
        parametrization = ng.p.Array(init=np.random.uniform(
            lower_bound, upper_bound, (func.meta_data.n_variables,)))
        parametrization.set_bounds(lower_bound, upper_bound)
        optimizer = eval(f"{self.alg}")(
            parametrization=parametrization,
            budget=self.eval_budget)

        try:
            optimizer.minimize(func)
            self.run_success = 1  # "SUCCESS"
        except OverflowError as err:
            print(f"OverflowError, run of {self.alg} with seed "
                  f"{self.algorithm_seed} CRASHED with message: {err}",
                  file=sys.stderr)
            self.run_success = 0  # "CRASHED" OverflowError
        except np.linalg.LinAlgError as err:
            print(f"LinAlgError, run of {self.alg} with seed "
                  f"{self.algorithm_seed} CRASHED with message: {err}",
                  file=sys.stderr)
            self.run_success = -2  # "CRASHED" np.linalg.LinAlgError
        except Exception as err:
            print(f"Unknown error, run of {self.alg} with seed "
                  f"{self.algorithm_seed} CRASHED with message: {err}",
                  file=sys.stderr)
            self.run_success = -3  # "CRASHED" other Exception


def run_algos(algorithms: list[str],
              problems: list[int],
              eval_budget: int,
              dimensionalities: list[int],
              n_repetitions: int,
              instances: list[int] = None) -> None:
    """Run the given algorithms on the given problem set.

    Args:
        algorithms: list of names of algorithms to run.
        problems: list of problem IDs (int) to run the algorithms on.
        eval_budget: int with the evaluation budget per run.
        dimensionalities: list of dimensionalities (int) to run per problem.
        n_repetitions: int for the number of repetitions (runs) to do per case.
            A case is an algorithm-problem-instance-dimensionality combination.
            The repetition number is also used as seed for the run.
        instances: list of instance IDs (int) to run per problem.
    """
    problem_class = ioh.ProblemClass.REAL

    for algname in algorithms:
        algname_short = const.get_short_algo_name(algname)
        algorithm = NGEvaluator(algname, eval_budget)
        logger = ioh.logger.Analyzer(folder_name=algname_short,
                                     algorithm_name=algname_short)
        logger.add_run_attributes(algorithm,
                                  ["algorithm_seed", "run_success"])

        for problem in problems:
            for dimension in dimensionalities:
                for instance in instances:
                    function = ioh.get_problem(problem, instance=instance,
                                               dimension=dimension,
                                               problem_class=problem_class)
                    function.attach_logger(logger)

                    for seed in range(1, n_repetitions + 1):
                        algorithm(function, seed)
                        function.reset()

        logger.close()

    return


def pbs_index_to_args_ngopt(index: int) -> (int, int, str):
    """Convert a PBS index to a dimension, budget and algorithm combination.

    Args:
        index: The index of the PBS job to run. This should be in [0,272).

    Returns:
        An int with the dimensionality.
        An int with the budget.
        A str with the algorithm name.
    """
    csv_path = "csvs/ngopt_choices.csv"
    run_settings = pd.read_csv(csv_path)

    dimensionality = run_settings.at[index, "dimensions"]
    budget = run_settings.at[index, "budget"]
    algo_id = run_settings.at[index, "algorithm"]
    algorithm = const.ALGS_CONSIDERED[algo_id]

    return dimensionality, budget, algorithm


def pbs_index_to_args_all_dims(index: int) -> (str, int):
    """Convert a PBS index to an algorithm and problem combination.

    Args:
        index: The index of the PBS job to run. This should be in [0,936).

    Returns:
        A str with the algorithm name.
        An int with the problem ID.
    """
    n_algos = len(const.ALGS_CONSIDERED)
    n_probs = len(const.PROBS_CONSIDERED)

    algo_id = index % n_algos
    prob_id = math.floor(index / n_algos) % n_probs

    algorithm = const.ALGS_CONSIDERED[algo_id]
    problem = const.PROBS_CONSIDERED[prob_id]

    return algorithm, problem


if __name__ == "__main__":
    DEFAULT_EVAL_BUDGET = 10000
    DEFAULT_N_REPETITIONS = 25
    DEFAULT_DIMS = [2]
    DEFAULT_PROBLEMS = list(range(1, 2))
    DEFAULT_INSTANCES = [1]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--algorithms",
        default=argparse.SUPPRESS,
        nargs="+",
        type=str,
        help="Algorithms to run.")
    parser.add_argument(
        "--eval-budget",
        default=DEFAULT_EVAL_BUDGET,
        type=int,
        help="Budget in function evaluations.")
    parser.add_argument(
        "--n-repetitions",
        default=DEFAULT_N_REPETITIONS,
        type=int,
        help=("Number of repetitions for an algorithm-problem-dimension "
              "combination."))
    parser.add_argument(
        "--dimensionalities",
        default=DEFAULT_DIMS,
        nargs="+",
        type=int,
        help=("List of variable space dimensionalities to consider for the "
              "problem."))
    parser.add_argument(
        "--problems",
        default=DEFAULT_PROBLEMS,
        nargs="+",
        type=int,
        help="List of BBOB problems.")
    parser.add_argument(
        "--instances",
        default=DEFAULT_INSTANCES,
        nargs="+",
        type=int,
        help="List of BBOB problem instances.")
    parser.add_argument(
        "--pbs-index-all-dims",
        type=int,
        help="PBS index to convert to algorithm and problem IDs.")
    parser.add_argument(
        "--pbs-index-ngopt",
        type=int,
        help="PBS index to convert to dimensionality, budget, and algorithm.")

    args = parser.parse_args()

    if args.pbs_index_all_dims is not None:
        algorithm, problem = (
            pbs_index_to_args_all_dims(args.pbs_index_all_dims))
        run_algos([algorithm], [problem], DEFAULT_EVAL_BUDGET,
                  const.DIMS_CONSIDERED,
                  DEFAULT_N_REPETITIONS, DEFAULT_INSTANCES)
    elif args.pbs_index_ngopt is not None:
        dimensionality, budget, algorithm = (
            pbs_index_to_args_ngopt(args.pbs_index_ngopt))
        run_algos([algorithm], const.PROBS_CONSIDERED, budget,
                  [dimensionality], DEFAULT_N_REPETITIONS, DEFAULT_INSTANCES)
    else:
        run_algos(args.algorithms, args.problems, args.eval_budget,
                  args.dimensionalities,
                  args.n_repetitions, args.instances)
