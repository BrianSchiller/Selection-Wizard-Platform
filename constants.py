#!/usr/bin/env python3
"""Constant variables for use in multiple files."""

RUNS_PER_SCENARIO = 25
EVAL_BUDGET = 10000

# 6 algorithms
# 33 ConfPortfolios in nevergrad 0.5.0
# 28 ConfPortfolios in nevergrad 0.6.0
ALGS_0_5_0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
              13, 14, 15, 16, 17,
              18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
              34, 35, 36, 37, 38]
ALGS_0_6_0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
              18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
              34, 35, 36, 37, 38]
ALGS_CONSIDERED = [
    # Algorithms relevant for both versions:
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

    # ConfPortfolios not relevant for nevergrad 0.6.0:
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001), Rescaled(base_optimizer=NGOpt14, scale=0.3486784401000001), Rescaled(base_optimizer=NGOpt14, scale=0.31381059609000006), Rescaled(base_optimizer=NGOpt14, scale=0.2824295364810001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001), Rescaled(base_optimizer=NGOpt14, scale=0.3486784401000001), Rescaled(base_optimizer=NGOpt14, scale=0.31381059609000006)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001), Rescaled(base_optimizer=NGOpt14, scale=0.3486784401000001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001), Rescaled(base_optimizer=NGOpt14, scale=0.3874204890000001)], warmup_ratio=0.5)",  # noqa: E501
    "ConfPortfolio(optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.0), Rescaled(base_optimizer=NGOpt14, scale=0.9), Rescaled(base_optimizer=NGOpt14, scale=0.81), Rescaled(base_optimizer=NGOpt14, scale=0.7290000000000001), Rescaled(base_optimizer=NGOpt14, scale=0.6561), Rescaled(base_optimizer=NGOpt14, scale=0.5904900000000001), Rescaled(base_optimizer=NGOpt14, scale=0.531441), Rescaled(base_optimizer=NGOpt14, scale=0.4782969000000001), Rescaled(base_optimizer=NGOpt14, scale=0.4304672100000001)], warmup_ratio=0.5)",  # noqa: E501

    # Continuation of algorithms relevant for both versions:
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
PROB_NAMES = ["f1_Sphere",
              "f2_Ellipsoid",
              "f3_Rastrigin",
              "f4_BuecheRastrigin",
              "f5_LinearSlope",
              "f6_AttractiveSector",
              "f7_StepEllipsoid",
              "f8_Rosenbrock",
              "f9_RosenbrockRotated",
              "f10_EllipsoidRotated",
              "f11_Discus",
              "f12_BentCigar",
              "f13_SharpRidge",
              "f14_DifferentPowers",
              "f15_RastriginRotated",
              "f16_Weierstrass",
              "f17_Schaffers10",
              "f18_Schaffers1000",
              "f19_GriewankRosenBrock",
              "f20_Schwefel",
              "f21_Gallagher101",
              "f22_Gallagher21",
              "f23_Katsuura",
              "f24_LunacekBiRastrigin"]


def get_short_algo_name(algo_name: str) -> str:
    """Return a str with a short name for a given algorithm name.

    Args:
        algo_name: str containing the algorithm name.

    Returns:
        algo_name for algorithms that already have a short name, or a shortened
            str for algorithms that have a lengthy name.
    """
    short_name = algo_name

    if algo_name.startswith("ConfPortfolio"):
        scl09 = "scale=0.9"
        scl13 = "scale=1.3"
        scnd_scale = "NA"

        if scl09 in algo_name:
            scnd_scale = scl09
        elif scl13 in algo_name:
            scnd_scale = scl13

        n_ngopt = algo_name.count("NGOpt14")
        short_name = (
            f"ConfPortfolio_scale2_{scnd_scale}_ngopt14s_{n_ngopt}")

    return short_name
