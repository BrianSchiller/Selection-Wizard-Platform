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

# SCALE_FACTORS adapted from:
# https://github.com/Dvermetten/Many-affine-BBOB/blob/master/affine_barebones.py
SCALE_FACTORS = [
    11., 17.5, 12.3, 12.6, 11.5, 15.3, 12.1, 15.3, 15.2, 17.4, 13.4, 20.4,
    12.9, 10.4, 12.3, 10.3, 9.8, 10.6, 10., 14.7, 10.7, 10.8, 9., 12.1]


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


class ManyAffine():
    """Affine combinations of BBOB problems.

    Class adapted from:
    https://github.com/Dvermetten/Many-affine-BBOB/blob/master/affine_barebones.py
    """

    def __init__(self: ManyAffine,
                 weights: list[float],
                 instances: list[int],
                 opt_loc: list[float] | int = 1,
                 dim: int = 5) -> None:
        """Initialise the problem instance.

        Args:
            weights: Weights per BBOB problem to specify the desired affine
                combination. Should have
                length equal to 24 (the number of BBOB problems).
            instances: Which instance to use per BBOB problem. Should have
                length equal to 24 (the number of BBOB problems).
            opt_loc: Location of the optimum in the search space. If float, it
                Should have length equal to dim. If an int is given, the
                optimum from the BBOB problem with this ID is taken (with IDs
                shifted from 1-24 to 0-23).
            dim: Dimensionality of the problem.
        """
        self.weights = weights / np.sum(weights)
        # Consider BBOB problems [1,24]
        self.fcts = [ioh.get_problem(fid, int(iid), dim)
                     for fid, iid in zip(range(1, 25), instances)]
        self.opts = [f.optimum.y for f in self.fcts]
        self.scale_factors = SCALE_FACTORS

        if type(opt_loc) == int:
            self.opt_x = self.fcts[opt_loc].optimum.x
        else:
            self.opt_x = opt_loc

        return

    def __call__(self: ManyAffine,
                 x: list[float]) -> float:
        """Evaluate the objective function.

        Args:
            x: List of variables to evaluate on the function.

        Returns:
            Objective value.
        """
        raw_vals = np.array(
            [np.clip(f(x+f.optimum.x - self.opt_x)-o, 1e-12, 1e20)
             for f, o in zip(self.fcts, self.opts)])
        weighted = (np.log10(raw_vals)+8)/self.scale_factors * self.weights

        return 10**(10*np.sum(weighted)-8)


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


def pbs_index_to_args_bud_dep(index: int) -> (int, int, str, int):
    """Convert a PBS ID to a dimension, budget, algorithm, problem combination.

    Here the dimension, budget, and problem are kept fixed as:
    - dimension: 100
    - budget: 200
    - problem: 1 (f1_Sphere)

    This is used to get performance data at a smaller budget than the default
    to be able to do a simple check for which algorithms the given budget
    influences the optimisation strategy. I.e., for which algorithms
    performance after, e.g., 200 evaluations is different with a budget of 200
    than with a budget of 10000.

    Args:
        index: The index of the PBS job to run. This should be in [0,39).

    Returns:
        An int with the dimensionality.
        An int with the budget.
        A str with the algorithm name.
        An int with the problem ID.
    """
    dimensionality = 100  # largest dimensionality
    budget = 200  # 100 * the smallest dimensionality
    algorithm = const.ALGS_CONSIDERED[index]
    problem = 1  # f1_Sphere, easiest problem

    return dimensionality, budget, algorithm, problem


def pbs_index_to_args_ngopt(index: int) -> (int, int, str):
    """Convert a PBS index to a dimension, budget and algorithm combination.

    This is used to get performance data for algorithms chosen by NGOpt when
    using the specific budget for which NGOpt would use them, rather than
    measuring the performance at that budget based on runs with a larger
    budget (e.g., measuring performance at 200 evaluations from a run with
    10000 evalutions).

    Here budgets up to 9000 are considered, since 10000 is already covered by
    the main experiment.

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

    This is used to run experiments for all algorithm and problem combinations
    on a single budget (10000 evaluations).

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
    parser.add_argument(
        "--pbs-index-bud-dep",
        type=int,
        help="PBS ID to convert to dimension, budget, algorithm, problem.")

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
    elif args.pbs_index_bud_dep is not None:
        dimensionality, budget, algorithm, problem = (
            pbs_index_to_args_bud_dep(args.pbs_index_bud_dep))
        run_algos([algorithm], [problem], budget,
                  [dimensionality], DEFAULT_N_REPETITIONS, DEFAULT_INSTANCES)
    else:
        run_algos(args.algorithms, args.problems, args.eval_budget,
                  args.dimensionalities,
                  args.n_repetitions, args.instances)
