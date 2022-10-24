#!/usr/bin/env python3

from enum import Enum

import nevergrad as ng

class NGOptVersion(str, Enum):
    NGOpt39 = 'NGOpt39'
    NGOpt21 = 'NGOpt21'

def square(x):
    return sum((x - 0.5) ** 2)

def get_optimiser(params, budget, workers, NGOpt: NGOptVersion):
    if NGOpt == NGOptVersion.NGOpt39:
        optimiser = ng.optimizers.NGOpt39(
           parametrization=ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5),
           budget=eval_budget,
           num_workers=n_workers)
    elif NGOpt == NGOptVersion.NGOpt21:
        optimiser = ng.optimizers.NGOpt21(
           parametrization=ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5),
           budget=eval_budget,
           num_workers=n_workers)

    return optimiser

# Fixed properties
n_workers = 1

# Properties to be varied
n_dims_min = 1
n_dims_max = 101
eval_budget_min = 1000
eval_budget_max = 1005

# Print derived problem properties
optimiser = ng.optimizers.NGOpt39(
    parametrization=ng.p.Array(shape=(1, n_dims_min)).set_bounds(-1, 5),
    budget=n_dims_min,
    num_workers=n_workers)
print('has_noise', optimiser.has_noise)
print('fully_continuous', optimiser.fully_continuous)
print('fully_bounded', ng.parametrization.parameter.helpers.Normalizer(optimiser.parametrization).fully_bounded)

for n_dimensions in range(n_dims_min, n_dims_max):
    for eval_budget in range(eval_budget_min, eval_budget_max):
        params = ng.p.Array(shape=(1, n_dimensions)).set_bounds(-1, 5)
        optimiser = get_optimiser(params, eval_budget, n_workers, NGOptVersion.NGOpt39)
        algorithm = optimiser._select_optimizer_cls()

        # For some reason type() does not work here...
        if algorithm is ng.optimization.optimizerlib.NGOpt21:
            optimiser = get_optimiser(params, eval_budget, n_workers, NGOptVersion.NGOpt21)
            algorithm = optimiser._select_optimizer_cls()

        if type(algorithm) is ng.optimization.optimizerlib.ConfPortfolio:
            algorithm = 'ConfPortfolio'

        # TODO: Handle NGOpt8
        # TODO: Consider whether to in- or exclude ParametrizedMetaModel results
        # TODO: Somehow loop over at least a subset of options? Or is it sufficient to always check in order NGOpt large to small, then other options?

        print(algorithm, n_dimensions, eval_budget)
