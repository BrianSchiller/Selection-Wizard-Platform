#!/usr/bin/env python3
"""Module to retrieve the optimisers recommended by NGOpt."""

from enum import Enum

import nevergrad as ng
from nevergrad.optimization.optimizerlib import NGOpt
from nevergrad.parametrization.data import Array


class NGOptVersion(str, Enum):
    """Enum with different NGOpt versions."""

    NGOpt39 = 'NGOpt39'
    NGOpt21 = 'NGOpt21'
    NGOpt8 = 'NGOpt8'
    NGOpt36 = 'NGOpt36'
    NGOpt15 = 'NGOpt15'


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
    elif ngopt == NGOptVersion.NGOpt21:
        optimiser = ng.optimizers.NGOpt21(
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
    elif ngopt == NGOptVersion.NGOpt36:
        optimiser = ng.optimizers.NGOpt36(
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

    return optimiser


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

for n_dimensions in range(n_dims_min, n_dims_max):
    for eval_budget in range(eval_budget_min, eval_budget_max):
        params = ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)
        optimiser = get_optimiser(
            params, eval_budget, n_workers, NGOptVersion.NGOpt39)
        algorithm = optimiser._select_optimizer_cls()

        # For some reason type() does not work here...
        if algorithm is ng.optimization.optimizerlib.NGOpt21:
            optimiser = get_optimiser(
                params, eval_budget, n_workers, NGOptVersion.NGOpt21)
            algorithm = optimiser._select_optimizer_cls()

        if type(algorithm) is ng.optimization.optimizerlib.ConfPortfolio:
            algorithm = 'ConfPortfolio'

        # For some reason type() does not work here...
        if algorithm is ng.optimization.optimizerlib.NGOpt8:
            optimiser = get_optimiser(
                params, eval_budget, n_workers, NGOptVersion.NGOpt8)
            algorithm = optimiser._select_optimizer_cls()

        if type(algorithm) is ng.optimization.optimizerlib.ConfPortfolio:
            algorithm = 'ConfPortfolio'

        # For some reason type() does not work here...
        if algorithm is ng.optimization.optimizerlib.NGOpt36:
            optimiser = get_optimiser(
                params, eval_budget, n_workers, NGOptVersion.NGOpt36)
            algorithm = optimiser._select_optimizer_cls()

        if type(algorithm) is ng.optimization.optimizerlib.ConfPortfolio:
            algorithm = 'ConfPortfolio'

        # For some reason type() does not work here...
        if algorithm is ng.optimization.optimizerlib.NGOpt15:
            optimiser = get_optimiser(
                params, eval_budget, n_workers, NGOptVersion.NGOpt15)
            algorithm = optimiser._select_optimizer_cls()

        if type(algorithm) is ng.optimization.optimizerlib.ConfPortfolio:
            algorithm = 'ConfPortfolio'

        # TODO: Consider whether to in- or exclude ParametrizedMetaModel
        # results
        # TODO: Somehow loop over at least a subset of options? Or is it
        # sufficient to always check in order NGOpt large to small, then other
        # options?

        if str(algorithm) != latest_algortihm:
            print(algorithm, n_dimensions, eval_budget)
            latest_algortihm = str(algorithm)
