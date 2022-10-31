#!/usr/bin/env python3
"""Module to retrieve the optimisers recommended by NGOpt."""

from enum import Enum
from typing import Union

import nevergrad as ng
from nevergrad.optimization.base import Optimizer
from nevergrad.optimization.optimizerlib import NGOpt
from nevergrad.optimization.optimizerlib import ConfPortfolio
from nevergrad.parametrization.data import Array


class NGOptVersion(str, Enum):
    """Enum with different NGOpt versions."""

    NGOpt8 = 'NGOpt8'
    NGOpt15 = 'NGOpt15'
    NGOpt21 = 'NGOpt21'
    NGOpt36 = 'NGOpt36'
    NGOpt39 = 'NGOpt39'


def square(x: float) -> float:
    """Test objective function."""
    return sum((x - 0.5) ** 2)


def get_optimiser(params: Array, budget: int, workers: int,
                  ngopt: NGOptVersion) -> NGOpt:
    """Return an optimiser based on the properties and NGOpt version."""
    if ngopt == NGOptVersion.NGOpt39:
        optimiser = ng.optimizers.NGOpt39(
            parametrization=(
                ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)),
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt36:
        optimiser = ng.optimizers.NGOpt36(
            parametrization=(
                ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)),
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt21:
        optimiser = ng.optimizers.NGOpt21(
            parametrization=(
                ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)),
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt15:
        optimiser = ng.optimizers.NGOpt15(
            parametrization=(
                ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)),
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt8:
        optimiser = ng.optimizers.NGOpt8(
            parametrization=(
                ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)),
            budget=eval_budget,
            num_workers=n_workers)

    return optimiser


def short_name(algorithm: Optimizer) -> Union[Optimizer, str]:
    """Return a short name for algorithms with many details."""
    if type(algorithm) is ConfPortfolio:
        algorithm = 'ConfPortfolio'

    return algorithm


def get_algorithm_for_ngopt(algorithm: Optimizer) -> Optimizer:
    """Get the recommended algorithm if it is currently an NGOpt version."""
    # For some reason type() does not work on NGOpt classes
    if algorithm is ng.optimization.optimizerlib.NGOpt36:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, NGOptVersion.NGOpt36)
        algorithm = optimiser._select_optimizer_cls()

    if algorithm is ng.optimization.optimizerlib.NGOpt21:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, NGOptVersion.NGOpt21)
        algorithm = optimiser._select_optimizer_cls()

    if algorithm is ng.optimization.optimizerlib.NGOpt15:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, NGOptVersion.NGOpt15)
        algorithm = optimiser._select_optimizer_cls()

    if algorithm is ng.optimization.optimizerlib.NGOpt8:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, NGOptVersion.NGOpt8)
        algorithm = optimiser._select_optimizer_cls()

    return algorithm


# Tracking variables
latest_algortihm = ''

# Fixed properties
n_workers = 1

# Properties to be varied
n_dims_min = 1
n_dims_max = 100
eval_budget_min = 1
eval_budget_max = 10000

# Print derived problem properties
optimiser = ng.optimizers.NGOpt39(
    parametrization=ng.p.Array(shape=(1, n_dims_min)).set_bounds(-1, 5),
    budget=eval_budget_min,
    num_workers=n_workers)
print('has_noise', optimiser.has_noise)
print('fully_continuous', optimiser.fully_continuous)
print('fully_bounded', ng.parametrization.parameter.helpers.Normalizer(
    optimiser.parametrization).fully_bounded)

for n_dimensions in range(n_dims_min, n_dims_max + 1):
    for eval_budget in range(eval_budget_min, eval_budget_max + 1):
        params = ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)
        optimiser = get_optimiser(
            params, eval_budget, n_workers, NGOptVersion.NGOpt39)
        algorithm = optimiser._select_optimizer_cls()
        algorithm = get_algorithm_for_ngopt(algorithm)
        algorithm = short_name(algorithm)

        if str(algorithm) != latest_algortihm:
            print(algorithm, n_dimensions, eval_budget)
            latest_algortihm = str(algorithm)
