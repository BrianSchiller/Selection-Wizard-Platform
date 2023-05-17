#!/usr/bin/env python3
"""Copy IOH performance and meta data files to directories per BBOB problem.

NOTE: Only run from the raw data directory with results from PBS cluster.
"""
from pathlib import Path
import shutil

if __name__ == "__main__":
    # Get PBS batch job directories
    this_dir = Path(".")
    pbs_dirs = [child for child in this_dir.iterdir() if child.is_dir()]
    pbs_dirs.sort()

    # Get directories that need to be moved excluding __pycache__
    algo_dirs = []

    for child in pbs_dirs:
        algo_paths = [subd for subd in child.iterdir()
                      if subd.is_dir() and not str(subd.name).startswith("__")]

        for sub_child in algo_paths:
            algo_dirs.append(sub_child)

    # Get function names per directory to be moved
    func_names = []
    for algo_dir in algo_dirs:
        func_list = list(algo_dir.glob("data_*"))
        name = func_list[0].name.removeprefix("data_")
        func_names.append(name)

        # Copy dir to new location
        func_dir = Path(name)
        func_dir.mkdir(exist_ok=True)
        destination = func_dir / algo_dir.name
        shutil.copytree(algo_dir, destination)
        print(f"Copy {name} to {destination}")
        print(f"From: {algo_dir}")
