#!/usr/bin/env python3
"""Constant variables for use in multiple files."""

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
