#!/usr/bin/env python3

import nevergrad as ng

def square(x):
    return sum((x - 0.5) ** 2)

optimiser = ng.optimizers.NGOpt(parametrization=2, budget=100)
algo = optimiser._select_optimizer_cls()
print(algo)
