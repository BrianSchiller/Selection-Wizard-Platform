#!/usr/bin/env python3
"""Module to retrieve the optimisers recommended by NGOpt."""

from enum import Enum
from typing import Union
from pathlib import Path

import nevergrad as ng
from nevergrad.optimization.base import Optimizer
from nevergrad.optimization.optimizerlib import NGOpt
from nevergrad.optimization.optimizerlib import ConfPortfolio
from nevergrad.parametrization.data import Array

import constants as const


class NGOptVersion(str, Enum):
    """Enum with different NGOpt versions."""

    NGOpt8 = "NGOpt8"
    NGOpt14 = "NGOpt14"
    NGOpt15 = "NGOpt15"
    NGOpt16 = "NGOpt16"
    NGOpt21 = "NGOpt21"
    NGOpt36 = "NGOpt36"
    NGOpt39 = "NGOpt39"


def get_optimiser(params: Array,
                  eval_budget: int,
                  n_workers: int,
                  n_dimensions: int,
                  ngopt: NGOptVersion) -> NGOpt:
    """Return an optimiser based on the properties and NGOpt version."""
    if ngopt == NGOptVersion.NGOpt39:
        optimiser = ng.optimizers.NGOpt39(
            parametrization=params,
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt36:
        optimiser = ng.optimizers.NGOpt36(
            parametrization=params,
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt21:
        optimiser = ng.optimizers.NGOpt21(
            parametrization=params,
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt16:
        optimiser = ng.optimizers.NGOpt16(
            parametrization=params,
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt15:
        optimiser = ng.optimizers.NGOpt15(
            parametrization=params,
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt14:
        optimiser = ng.optimizers.NGOpt14(
            parametrization=params,
            budget=eval_budget,
            num_workers=n_workers)
    elif ngopt == NGOptVersion.NGOpt8:
        optimiser = ng.optimizers.NGOpt8(
            parametrization=params,
            budget=eval_budget,
            num_workers=n_workers)

    return optimiser


def short_name(algorithm: Optimizer) -> Union[Optimizer, str]:
    """Return a short name for algorithms with many details."""
    if type(algorithm) is ConfPortfolio:
        algorithm = "ConfPortfolio"

    return algorithm


def get_algorithm_for_ngopt(algorithm: Optimizer,
                            params: Array,
                            eval_budget: int,
                            n_workers: int,
                            n_dimensions: int) -> Optimizer:
    """Get the recommended algorithm if it is currently an NGOpt version."""
    # For some reason type() does not work on NGOpt classes
    if algorithm is ng.optimization.optimizerlib.NGOpt36:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, n_dimensions, NGOptVersion.NGOpt36)
        algorithm = optimiser._select_optimizer_cls()

    if algorithm is ng.optimization.optimizerlib.NGOpt21:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, n_dimensions, NGOptVersion.NGOpt21)
        algorithm = optimiser._select_optimizer_cls()

    if algorithm is ng.optimization.optimizerlib.NGOpt16:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, n_dimensions, NGOptVersion.NGOpt16)
        algorithm = optimiser._select_optimizer_cls()

    if algorithm is ng.optimization.optimizerlib.NGOpt15:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, n_dimensions, NGOptVersion.NGOpt15)
        algorithm = optimiser._select_optimizer_cls()

    if algorithm is ng.optimization.optimizerlib.NGOpt8:
        optimiser = get_optimiser(
            params, eval_budget, n_workers, n_dimensions, NGOptVersion.NGOpt8)
        algorithm = optimiser._select_optimizer_cls()

    return algorithm


def output_ngopt_algos(output_properties: bool,
                       output_header: bool,
                       hsv_format: bool,
                       full_algo_name: bool,
                       out_file: None) -> None:
    """Print NGOpt algorithms to standard output or file."""
    # Tracking variables
    latest_algortihm = ""

    # Fixed properties
    n_workers = 1

    # Properties to be varied
    n_dims_min = 1
    n_dims_max = 100
    eval_budget_min = 1
    eval_budget_max = const.EVAL_BUDGET

    # Output derived problem properties
    if output_properties:
        optimiser = ng.optimizers.NGOpt39(
            parametrization=ng.p.Array(shape=(1, n_dims_min)).set_bounds(
                const.LOWER_BOUND, const.UPPER_BOUND),
            budget=eval_budget_min,
            num_workers=n_workers)
        print("has_noise", optimiser.has_noise, file=out_file)
        print("fully_continuous", optimiser.fully_continuous, file=out_file)
        print("fully_bounded", ng.parametrization.parameter.helpers.Normalizer(
            optimiser.parametrization).fully_bounded, file=out_file)

    # Output header
    if output_header:
        if hsv_format:
            print("Algorithm#dimensionality#evaluation budget", file=out_file)
        else:
            print("Algorithm, dimensionality, evaluation budget",
                  file=out_file)

    for n_dimensions in range(n_dims_min, n_dims_max + 1):
        params = ng.p.Array(shape=(1, n_dimensions)).set_bounds(
            const.LOWER_BOUND, const.UPPER_BOUND)

        for eval_budget in range(eval_budget_min, eval_budget_max + 1):
            optimiser = get_optimiser(
                params, eval_budget, n_workers, n_dimensions,
                NGOptVersion.NGOpt39)
            algorithm = optimiser._select_optimizer_cls()
            algorithm = get_algorithm_for_ngopt(algorithm, params, eval_budget,
                                                n_workers, n_dimensions)

            if not full_algo_name:
                algorithm = short_name(algorithm)

            if str(algorithm) != latest_algortihm:
                if hsv_format:
                    print(f"{algorithm}#{n_dimensions}#{eval_budget}",
                          file=out_file)
                else:
                    print(algorithm, n_dimensions, eval_budget, file=out_file)

                latest_algortihm = str(algorithm)
    return


# Output settings
output_properties = True
output_header = False
hsv_format = False
full_algo_name = False

# Don't distinguish between different ConfPortfolio's (use short names)
out_path = Path("ngopt_choices/0.6.0/dims1-100evals1-10000.txt")

with out_path.open("w") as out_file:
    output_ngopt_algos(output_properties, output_header, hsv_format,
                       full_algo_name, out_file)


# Distinguish between different ConfPortfolio's (use full names)
full_algo_name = True
out_path = Path("ngopt_choices/0.6.0/dims1-100evals1-10000_full.txt")

with out_path.open("w") as out_file:
    output_ngopt_algos(output_properties, output_header, hsv_format,
                       full_algo_name, out_file)
