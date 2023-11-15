#!/usr/bin/env python3
import ioh
import nevergrad as ng
from nevergrad.optimization.optimizerlib import ParametrizedMetaModel  # noqa: F401, E501
from nevergrad.optimization.optimizerlib import CmaFmin2  # noqa: F401


class NGEvaluator:
    def __init__(self, optimizer: str, eval_budget: int) -> None:
        self.alg = optimizer
        self.eval_budget = eval_budget

    def __call__(self, func, seed) -> None:
        parametrization = ng.p.Array(
            shape=(func.meta_data.n_variables,)).set_bounds(-5, 5)
        parametrization.random_state.seed(seed)
        optimizer = eval(f"{self.alg}")(
            parametrization=parametrization,
            budget=self.eval_budget)
        print(optimizer.minimize(func))


def run_algos(algorithm: str, problem: int,
              eval_budget: int,
              dimension: int,
              instance: int, seed: int) -> None:
    algorithm = NGEvaluator(algorithm, eval_budget)
    logger = ioh.logger.Analyzer()
    function = ioh.get_problem(problem, instance=instance,
                               dimension=dimension, problem_class=ioh.ProblemClass.REAL)
    algorithm(function, seed)
    function.reset()
    logger.close()

    return


if __name__ == "__main__":
    eval_budget = 200
    dimension = 25
    problem = 11
    instance = 1
    algorithm = "ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)"
    seed = 1

    run_algos(algorithm, problem, eval_budget, dimension, instance, seed)
