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
    parser = argparse.ArgumentParser(description='Run algorithms on IOH Benchmarks.')
    parser.add_argument('--general', type=str, help='Whether to run on Slurm', required=False, default=False)
    args = parser.parse_args()

    analyse_test_csvs(Path("Output/Final_Train"), test_bbob=True, general=args.general)
