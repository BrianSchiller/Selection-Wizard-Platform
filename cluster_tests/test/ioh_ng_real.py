import ioh
import argparse

import nevergrad as ng
from nevergrad.optimization.optimizerlib import Cobyla  # noqa: F401
from nevergrad.optimization.optimizerlib import MetaModel  # noqa: F401
from nevergrad.optimization.optimizerlib import CMA  # noqa: F401
from nevergrad.optimization.optimizerlib import ParametrizedMetaModel  # noqa: F401
from nevergrad.optimization.optimizerlib import CmaFmin2  # noqa: F401
from nevergrad.optimization.optimizerlib import ChainMetaModelPowell  # noqa: F401
from nevergrad.optimization.optimizerlib import MetaModelOnePlusOne  # noqa: F401
from nevergrad.optimization.optimizerlib import ConfPortfolio  # noqa: F401
from nevergrad.optimization.optimizerlib import Rescaled  # noqa: F401
from nevergrad.optimization.optimizerlib import NGOpt14  # noqa: F401

# 6 algorithms
# 33 ConfPortfolios
algs_considered = [
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


def run_algos(algs_considered: list[str], eval_budget: int):
    for algname in algs_considered:
        algname_short = algname

        if algname.startswith("ConfPortfolio"):
            SCL09 = "scale=0.9"
            SCL13 = "scale=1.3"
            scnd_scale = "NA"
            if SCL09 in algname:
                scnd_scale = SCL09
            elif SCL13 in algname:
                scnd_scale = SCL13
            n_scaled = algname.count("Rescaled")
            n_ngopt = algname.count("NGOpt14")
            algname_short = f"ConfPortfolio_scale2_{scnd_scale}_ngopt14s_{n_ngopt}"

        # Set the optimization algorithm
        exp = ioh.Experiment(
            algorithm=NGEvaluator(algname, eval_budget),
            # Problem definitions. I use function name here, but could also use
            # the ID (25 in this case)
            fids=range(1, 2), iids=range(1, 2), dims=[5], reps=2,
            problem_type="BBOB",
            # Set paralellization level here if desired, or use this within your
            # own parallelization
            njobs=1, output_directory="OUTPUT",
            # Logging specifications
            logged=True, folder_name=f"{algname_short}",
            algorithm_name=f"{algname_short}",
            store_positions=False,
            # Only keep data as a single zip-file
            merge_output=True, zip_output=True, remove_data=True)
        exp()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'algorithm',
        type=str,
        help='Algorithm to run.')
    args = parser.parse_args()

    EVAL_BUDGET = 100000

    algs_considered = [args.algorithm]
    run_algos(algs_considered, EVAL_BUDGET)

