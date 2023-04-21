"""Module to run Nevergrad algorithm implementations with IOH profiler."""
from __future__ import annotations
import ioh
import argparse
import math
import sys
import numpy as np

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

# 6 algorithms
# 33 ConfPortfolios
ALGS_CONSIDERED = [
    "CMA",
    "ChainMetaModelPowell",
    "Cobyla",
    "MetaModel",
    "MetaModelOnePlusOne",
    "ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)",
    "ConfPortfolio(optimizers=[NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14], warmup_ratio=0.7)",  # noqa: E501
    "ConfPortfolio(optimizers=[NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14], warmup_ratio=0.7)",  # noqa: E501
    "ConfPortfolio(optimizers=[NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14], warmup_ratio=0.7)",  # noqa: E501
    "ConfPortfolio(optimizers=[NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14], warmup_ratio=0.7)",  # noqa: E501
    "ConfPortfolio(optimizers=[NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14], warmup_ratio=0.7)",  # noqa: E501
    "ConfPortfolio(optimizers=[NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14], warmup_ratio=0.7)",  # noqa: E501
    "ConfPortfolio(optimizers=[NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14, NGOpt14], warmup_ratio=0.7)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001), Rescaled(base_optimizer=NGOpt14, scale=0.3486784401000001), Rescaled(base_optimizer=NGOpt14, scale=0.31381059609000006), Rescaled(base_optimizer=NGOpt14, scale=0.2824295364810001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001), Rescaled(base_optimizer=NGOpt14, scale=0.3486784401000001), Rescaled(base_optimizer=NGOpt14, scale=0.31381059609000006)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001), Rescaled(base_optimizer=NGOpt14, scale=0.3486784401000001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312), Rescaled(base_optimizer=NGOpt14, scale=39.37376385699291), Rescaled(base_optimizer=NGOpt14, scale=51.18589301409078), Rescaled(base_optimizer=NGOpt14, scale=66.54166091831802), Rescaled(base_optimizer=NGOpt14, scale=86.50415919381344), Rescaled(base_optimizer=NGOpt14, scale=112.45540695195746), Rescaled(base_optimizer=NGOpt14, scale=146.1920290375447), Rescaled(base_optimizer=NGOpt14, scale=190.04963774880812)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312), Rescaled(base_optimizer=NGOpt14, scale=39.37376385699291), Rescaled(base_optimizer=NGOpt14, scale=51.18589301409078), Rescaled(base_optimizer=NGOpt14, scale=66.54166091831802), Rescaled(base_optimizer=NGOpt14, scale=86.50415919381344), Rescaled(base_optimizer=NGOpt14, scale=112.45540695195746), Rescaled(base_optimizer=NGOpt14, scale=146.1920290375447)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312), Rescaled(base_optimizer=NGOpt14, scale=39.37376385699291), Rescaled(base_optimizer=NGOpt14, scale=51.18589301409078), Rescaled(base_optimizer=NGOpt14, scale=66.54166091831802), Rescaled(base_optimizer=NGOpt14, scale=86.50415919381344), Rescaled(base_optimizer=NGOpt14, scale=112.45540695195746)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312), Rescaled(base_optimizer=NGOpt14, scale=39.37376385699291), Rescaled(base_optimizer=NGOpt14, scale=51.18589301409078), Rescaled(base_optimizer=NGOpt14, scale=66.54166091831802), Rescaled(base_optimizer=NGOpt14, scale=86.50415919381344)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312), Rescaled(base_optimizer=NGOpt14, scale=39.37376385699291), Rescaled(base_optimizer=NGOpt14, scale=51.18589301409078), Rescaled(base_optimizer=NGOpt14, scale=66.54166091831802)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312), Rescaled(base_optimizer=NGOpt14, scale=39.37376385699291), Rescaled(base_optimizer=NGOpt14, scale=51.18589301409078)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312), Rescaled(base_optimizer=NGOpt14, scale=39.37376385699291)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101), Rescaled(base_optimizer=NGOpt14, scale=30.287510659225312)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005), Rescaled(base_optimizer=NGOpt14, scale=23.29808512248101)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005), Rescaled(base_optimizer=NGOpt14, scale=17.921603940370005)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003), Rescaled(base_optimizer=NGOpt14, scale=13.785849184900005)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003), Rescaled(base_optimizer=NGOpt14, scale=10.604499373000003)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002), Rescaled(base_optimizer=NGOpt14, scale=8.157307210000003)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001), Rescaled(base_optimizer=NGOpt14, scale=6.274851700000002)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005), Rescaled(base_optimizer=NGOpt14, scale=4.826809000000001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005), Rescaled(base_optimizer=NGOpt14, scale=3.7129300000000005)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197), Rescaled(base_optimizer=NGOpt14, scale=2.8561000000000005)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002), Rescaled(base_optimizer=NGOpt14, scale=2.197)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3), Rescaled(base_optimizer=NGOpt14, scale=1.6900000000000002)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=1.3)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0)], warmup_ratio=0.5)",  # noqa: E501
    ]
DIMS_CONSIDERED = [2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80,
                   90, 100]
PROBS_CONSIDERED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23, 24]


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
                 seed: int = None) -> None:
        """Run the NGEvaluator on the given problem.

        Args:
            func: IOH function to run the algorithm on.
            seed: int to seed the algorithm random state.
        """
        parametrization = ng.p.Array(
            shape=(func.meta_data.n_variables,)).set_bounds(-5, 5)
        if seed is not None:
            self.algorithm_seed = seed
            parametrization.random_state.seed(seed)
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
            self.run_success = 0  # "CRASHED"
        except np.linalg.LinAlgError as err:
            print(f"LinAlgError, run of {self.alg} with seed "
                  f"{self.algorithm_seed} CRASHED with message: {err}",
                  file=sys.stderr)
            self.run_success = 0  # "CRASHED"


def run_algos(algorithms: list[str],
              problems: list[int],
              eval_budget: int,
              dimensionalities: list[int],
              n_repetitions: int,
              instances: list[int] = None,
              use_seed: bool = False) -> None:
    """Run the given algorithms on the given problem set.

    Args:
        algorithms: list of names of algorithm to run.
        problems: list of problem IDs (int) to run the algorithms on.
        eval_budget: int with the evaluation budget per run.
        dimensionalities: list of dimensionalities (int) to run per problem.
        n_repetitions: int for the number of repetitions (runs) to do per case.
            A case is an algorithm-problem-instance-dimensionality combination.
        instances: list of instance IDs (int) to run per problem.
        use_seed: If True, use the repetition number as seed.
    """
    problem_type = "BBOB"

    for algname in algorithms:
        algname_short = algname

        if algname.startswith("ConfPortfolio"):
            scl09 = "scale=0.9"
            scl13 = "scale=1.3"
            scnd_scale = "NA"
            if scl09 in algname:
                scnd_scale = scl09
            elif scl13 in algname:
                scnd_scale = scl13
            n_ngopt = algname.count("NGOpt14")
            algname_short = (
                f"ConfPortfolio_scale2_{scnd_scale}_ngopt14s_{n_ngopt}")

        out_dir = f"{algname_short}_D{dimensionalities[0]}_P{problems[0]}"

        if use_seed:
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
                                                   problem_type=problem_type)
                        function.attach_logger(logger)
                        for seed in range(1, n_repetitions + 1):
                            algorithm(function, seed)
                            function.reset()
            logger.close()
        else:
            # Set the optimization algorithm
            exp = ioh.Experiment(
                algorithm=NGEvaluator(algname, eval_budget),
                # Problem definitions. I use function name here, but could also
                # use the ID (25 in this case)
                fids=problems, iids=instances, dims=dimensionalities,
                reps=n_repetitions,
                problem_type=problem_type,
                # Set paralellization level here if desired, or use this within
                # your own parallelization
                njobs=1, output_directory=out_dir,
                # Logging specifications
                logged=True, folder_name=f"{algname_short}",
                algorithm_name=f"{algname_short}",
                store_positions=False,
                # Only keep data as a single zip-file
                merge_output=True, zip_output=True, remove_data=True)
            exp()

    return


def pbs_index_to_args(index: int) -> (str, int, int):
    """Convert a PBS index to algorithm, dimension, and problem combination.

    Args:
        index: The index of the PBS job to run. This should be in [0,15912).

    Returns:
        A str with the algorithm name.
        An int with the dimensionality.
        An int with the problem ID.
    """
    n_algos = len(ALGS_CONSIDERED)
    n_dims = len(DIMS_CONSIDERED)
    n_probs = len(PROBS_CONSIDERED)

    algo_id = index % n_algos
    dims_id = math.floor(index / n_algos) % n_dims
    prob_id = math.floor(index / n_algos / n_dims) % n_probs

    algorithm = ALGS_CONSIDERED[algo_id]
    dimensionality = DIMS_CONSIDERED[dims_id]
    problem = PROBS_CONSIDERED[prob_id]

    return algorithm, dimensionality, problem


def pbs_index_to_args_all_dims(index: int) -> (str, int):
    """Convert a PBS index to algorithm and problem combination.

    Args:
        index: The index of the PBS job to run. This should be in [0,936).

    Returns:
        A str with the algorithm name.
        An int with the problem ID.
    """
    n_algos = len(ALGS_CONSIDERED)
    n_probs = len(PROBS_CONSIDERED)

    algo_id = index % n_algos
    prob_id = math.floor(index / n_algos) % n_probs

    algorithm = ALGS_CONSIDERED[algo_id]
    problem = PROBS_CONSIDERED[prob_id]

    return algorithm, problem


if __name__ == "__main__":
    DEFAULT_EVAL_BUDGET = 10000
    DEFAULT_N_REPETITIONS = 25
    DEFAULT_DIMS = [4, 5]
    DEFAULT_PROBLEMS = list(range(1, 3))
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
        "--pbs-index",
        type=int,
        help=("PBS index to convert to algorithm and problem IDs. Each of them"
              " is executed for all dimensions."))
    parser.add_argument(
        "--pbs-index-all-dims",
        type=int,
        help="PBS index to convert to algorithm, dimension, and problem IDs.")
    parser.add_argument(
        "--use-seed",
        action="store_true",
        help="Set to use the repetition number as algorithm seed.")

    args = parser.parse_args()

    if args.pbs_index is not None:
        algorithm, dimensionality, problem = pbs_index_to_args(args.pbs_index)
        run_algos([algorithm], [problem], DEFAULT_EVAL_BUDGET,
                  [dimensionality],
                  DEFAULT_N_REPETITIONS, DEFAULT_INSTANCES, args.use_seed)
    elif args.pbs_index_all_dims is not None:
        algorithm, problem = (
            pbs_index_to_args_all_dims(args.pbs_index_all_dims))
        run_algos([algorithm], [problem], DEFAULT_EVAL_BUDGET,
                  DIMS_CONSIDERED,
                  DEFAULT_N_REPETITIONS, DEFAULT_INSTANCES, args.use_seed)
    else:
        run_algos(args.algorithms, args.problems, args.eval_budget,
                  args.dimensionalities,
                  args.n_repetitions, args.instances,
                  args.use_seed)
