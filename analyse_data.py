#!/usr/bin/env python3
"""Process, analyse, and plot performance data."""

import argparse
from pathlib import Path
import sys
import os

from experiment import Experiment
from experiment import NGOptChoice
from experiment import analyse_test_csvs
from experiment import test_plot_all


if __name__ == "__main__":
    # directory = "Output/Eval_20240701_13-10-50"

    # subdirectories = [os.path.join(directory, name) for name in os.listdir(directory)
    #                   if os.path.isdir(os.path.join(directory, name))]

    # for directory in subdirectories:
    #     analyse_test_csvs(Path(directory), test_bbob=True)

    analyse_test_csvs(Path("Output/Eval_20240701_13-10-50"), test_bbob=True)
