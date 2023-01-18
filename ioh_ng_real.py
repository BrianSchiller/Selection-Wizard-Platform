import ioh
import argparse
from typing import Union

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


class NGEvaluator:
    def __init__(self, optimizer, eval_budget: int) -> None:
        self.alg = optimizer
        self.eval_budget = eval_budget

    def __call__(self, func) -> None:
        parametrization = ng.p.Array(
            shape=(func.meta_data.n_variables,)).set_bounds(-5, 5)
        optimizer = eval(f"{self.alg}")(
            parametrization=parametrization,
            budget=self.eval_budget)
        optimizer.minimize(func)


def run_algos(algorithms: list[str], problems: list[Union[int, str]],
              eval_budget: int,
              dimensionalities: list[int], n_repetitions: int,
              instances: list[int] = None) -> None:
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

        # Set the optimization algorithm
        exp = ioh.Experiment(
            algorithm=NGEvaluator(algname, eval_budget),
            # Problem definitions. I use function name here, but could also use
            # the ID (25 in this case)
            fids=problems, iids=instances, dims=dimensionalities,
            reps=n_repetitions,
            problem_type="BBOB",
            # Set paralellization level here if desired, or use this within
            # your own parallelization
            njobs=1, output_directory="OUTPUT",
            # Logging specifications
            logged=True, folder_name=f"{algname_short}",
            algorithm_name=f"{algname_short}",
            store_positions=False,
            # Only keep data as a single zip-file
            merge_output=True, zip_output=True, remove_data=True)
        exp()

    return


if __name__ == "__main__":
    DEFAULT_EVAL_BUDGET = 10000
    DEFAULT_N_REPETITIONS = 25
    DEFAULT_DIMS = [4, 5]
    DEFAULT_PROBLEMS = list(range(1, 3))
    DEFAULT_INSTANCES = [1]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--algorithm",
        default=argparse.SUPPRESS,
        type=str,
        help="Algorithm to run.")
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
        type=Union[int, str],
        help="List of BBOB problems.")
    parser.add_argument(
        "--intances",
        default=DEFAULT_INSTANCES,
        nargs="+",
        type=int,
        help="List of BBOB problem instances.")

    args = parser.parse_args()

    run_algos(args.algorithms, args.problems, args.eval_budget,
              args.dimensionalities,
              args.n_repetitions, args.instances)
