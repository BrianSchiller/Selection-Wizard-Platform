#!/usr/bin/env python3
"""Module to run Nevergrad algorithm implementations with IOH profiler."""
from __future__ import annotations  # For self annotations
import ioh
import argparse
import math
import sys
from pathlib import Path
import os
import shutil
import re
import datetime
import json

import numpy as np
import pandas as pd

import nevergrad as ng
# from nevergrad.optimization.optimizerlib import Cobyla  # noqa: F401
# from nevergrad.optimization.optimizerlib import MetaModel  # noqa: F401
# from nevergrad.optimization.optimizerlib import CMA  # noqa: F401
from nevergrad.optimization.optimizerlib import ParametrizedMetaModel  # noqa: F401, E501
from nevergrad.optimization.optimizerlib import CmaFmin2  # noqa: F401
# from nevergrad.optimization.optimizerlib import ChainMetaModelPowell  # noqa: F401, E501
# from nevergrad.optimization.optimizerlib import MetaModelOnePlusOne  # noqa: F401, E501
from nevergrad.optimization.optimizerlib import ConfPortfolio  # noqa: F401
from nevergrad.optimization.optimizerlib import Rescaled  # noqa: F401
from nevergrad.optimization.optimizerlib import NGOpt14  # noqa: F401
from nevergrad.optimization.optimizerlib import NGOptBase
from nevergrad.optimization.base import OptCls, ConfiguredOptimizer


import constants as const
from experiment import Problem
from experiment import Algorithm
from experiment import Scenario
import run_selector
from models import MetaModelFmin2, MetaModel, MetaModelOnePlusOne, ChainMetaModelPowell, CMA, Cobyla
from configurations import get_config

# SCALE_FACTORS adapted from:
# https://github.com/Dvermetten/Many-affine-BBOB/blob/1c144ff5fda2e68227bd56ccdb7d55ec696bdfcf/affine_barebones.py#L4
SCALE_FACTORS = [
    11., 17.5, 12.3, 12.6, 11.5, 15.3, 12.1, 15.3, 15.2, 17.4, 13.4, 20.4,
    12.9, 10.4, 12.3, 10.3, 9.8, 10.6, 10., 14.7, 10.7, 10.8, 9., 12.1]


class DDS(NGOptBase):
    """Data-driven selector."""

    def _select_optimizer_cls(self: DDS) -> OptCls:
        """Return an optimizer class."""
        return eval(run_selector.select_algorithm(self.budget, self.dimension))


class NGEvaluator:
    """Algorithm wrapper to use nevergrad algorithms with IOHprofiler."""

    algorithm_seed = 1
    run_success = -1  # "UNKNOWN"

    def __init__(self: NGEvaluator, optimizer: str | ConfiguredOptimizer, eval_budget: int) -> None:
        """Initialise the NGEvaluator.

        Args:
            optimizer: str with the algorithm name.
            eval_budget: int with the evaluation budget.
        """
        self.alg = optimizer
        self.eval_budget = eval_budget

    def __call__(self: NGEvaluator,
                 func: ioh.iohcpp.problem.RealSingleObjective,
                 seed: int) -> None:
        """Run the NGEvaluator on the given problem.

        Sets run_success in IOH json output per run with status codes:
            -1  UNKNOWN
             1  SUCCESS
             0  CRASHED due to OverflowError
            -2  CRASHED due to np.linalg.LinAlgError
            -3  CRASHED due to other Exception

        Args:
            func: IOH function to run the algorithm on.
            seed: int to seed the algorithm random state.
        """
        lower_bound = const.LOWER_BOUND
        upper_bound = const.UPPER_BOUND
        self.algorithm_seed = seed
        np.random.seed(self.algorithm_seed)
        # Generate an initial solution within the bounds uniformly at random
        parametrization = ng.p.Array(init=np.random.uniform(
            lower_bound, upper_bound, (func.meta_data.n_variables,)))
        parametrization.set_bounds(lower_bound, upper_bound)
        
        if type(self.alg) == str:
            optimizer = eval(f"{self.alg}")(
                parametrization=parametrization,
                budget=self.eval_budget)
        else:
            optimizer = self.alg(parametrization=parametrization, budget = self.eval_budget)

        try:
            optimizer.minimize(func)
            self.run_success = 1  # "SUCCESS"
        except OverflowError as err:
            print(f"OverflowError, run of {self.alg} with seed "
                  f"{self.algorithm_seed} CRASHED with message: {err}",
                  file=sys.stderr)
            self.run_success = 0  # "CRASHED" OverflowError
        except np.linalg.LinAlgError as err:
            print(f"LinAlgError, run of {self.alg} with seed "
                  f"{self.algorithm_seed} CRASHED with message: {err}",
                  file=sys.stderr)
            self.run_success = -2  # "CRASHED" np.linalg.LinAlgError
        except Exception as err:
            print(f"Unknown error, run of {self.alg} with seed "
                  f"{self.algorithm_seed} CRASHED with message: {err}",
                  file=sys.stderr)
            self.run_success = -3  # "CRASHED" other Exception


def run_algos(algorithms: list[str | ConfiguredOptimizer],
              problems: list[int],
              eval_budget: int,
              dimension: int,
              n_repetitions: int,
              instances: list[int] = None,
              process_intermediate_data: bool = False,
              output_dir: Path = None) -> None:
    """Run the given algorithms on the given problem set.

    Args:
        algorithms: list of names of algorithms to run.
        problems: list of problem IDs (int) to run the algorithms on.
        eval_budget: int with the evaluation budget per run.
        dimensionalities: list of dimensionalities (int) to run per problem.
        n_repetitions: int for the number of repetitions (runs) to do per case.
            A case is an algorithm-problem-instance-dimensionality combination.
            The repetition number is also used as seed for the run.
        instances: list of instance IDs (int) to run per problem.
        process_intermediate_data: If True, process and store intermediate data
            during execution.
    """
    problem_class = ioh.ProblemClass.REAL
    n_instances = len(instances)

    for alg in algorithms:
        if type(alg) == str:
            alg_name = const.get_short_algo_name(alg)
        else:
            alg_name = alg.name
        algorithm = NGEvaluator(alg.get_optimizer(), eval_budget)

        if not process_intermediate_data:
            logger = ioh.logger.Analyzer(folder_name=f"{output_dir}/{alg_name}",
                                         algorithm_name=alg_name)
            logger.add_run_attributes(algorithm,
                                      ["algorithm_seed", "run_success"])

        for problem in problems:
            if process_intermediate_data:
                dir_name = f"{output_dir}/B{eval_budget}_D{dimension}/{alg_name}"
                logger = ioh.logger.Analyzer(folder_name=dir_name,
                                            algorithm_name=alg_name)
                logger.add_run_attributes(
                    algorithm, ["algorithm_seed", "run_success"])

            for instance in instances:
                function = ioh.get_problem(problem, instance=instance,
                                        dimension=dimension,
                                        problem_class=problem_class)
                function.attach_logger(logger)

                for seed in range(1, n_repetitions + 1):
                    algorithm(function, seed)
                    function.reset()

            # Process all .json files in the output directory
            if process_intermediate_data:
                # Flush the logger to ensure files exist before processing
                logger.close()
                process_data(Path(dir_name), problem, alg_name, dimension,
                            n_repetitions, eval_budget, n_instances)

        logger.close()
        print(f"Finished: {alg.name}")

    return


def process_data(dir_name: Path,
                 prob_id: int,
                 algo_name: str,
                 dims: int,
                 n_runs: int,
                 n_evals: int,
                 n_instances: int) -> None:
    """Extract final run performance and add data to .zip file.

    Args:
        dir_name: Path to the output directory.
        prob_id: Problem ID.
        algo_name: Full algorithm name.
        dims: Dimensionality of the search space (number of variables).
        n_runs: Number of runs performed with these settings.
        n_evals: Number of evaluations per run.
        n_instances: Number of instances per problem.
    """
    # Get all .json files in the output directory
    json_files = [json_file for json_file in dir_name.iterdir()
                  if str(json_file).endswith(".json")]

    for json_file in json_files:
        # Problem is the file without preceding IOHprofiler_ and trailing .json
        problem_name = json_file.stem.removeprefix("IOHprofiler_")
        problem = Problem(problem_name, prob_id)
        algorithm = Algorithm(algo_name)
        scenario = Scenario(dir_name, problem, algorithm, dims, n_runs,
                            n_evals, n_instances, json_file=json_file)

        for run in scenario.runs:
            # Get the performance data and metadata from the scenario
            # Desired data: problem, algorithm, dimensions, budget, seed,
            # instance, run status, performance
            seed = run.seed
            instance = run.instance
            status = run.status
            performance = run.get_performance(n_evals)

            # Write (add) the performance and meta data to a .csv
            dir_name_out = dir_name.with_name(f"{dir_name.name}_processed")
            csv_path = dir_name_out / "data.csv"

            if not csv_path.exists():
                dir_name_out.mkdir(parents=True, exist_ok=True)
                csv_header = ("problem,algorithm,dimensions,budget,seed,"
                              "instance,status,performance")

                with csv_path.open("w") as csv_file:
                    csv_file.write(csv_header)

            csv_row = (f"\n{problem_name},{algorithm.name_short},{dims},"
                       f"{n_evals},{seed},{instance},{status},{performance}")

            with csv_path.open("a") as csv_file:
                csv_file.write(csv_row)

        # Add the raw data to a .zip
        # Escape the used paths to avoid issues with special characters.
        # Particularly the parentheses in:
        # ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
        json_path = Path(re.escape(str(json_file)))
        data_path = Path(re.escape(str(dir_name))) / f"data_{problem_name}"
        zip_path = Path(re.escape(str(dir_name_out))) / "data.zip"

        # Options: -r recursive, -q quiet, -g grow (append to existing zip)
        if zip_path.exists():
            os.system(f"zip -r -q -g {zip_path} {data_path} {json_path}")
        else:
            os.system(f"zip -r -q {zip_path} {data_path} {json_path}")

        # Remove the uncompressed data
        shutil.rmtree(dir_name)

    return

def write_scenario_file(output_dir):
    # Check if the file already exists (Prevent slurm creating multiple scenario files)
    scenario_file_path = os.path.join(output_dir, "scenario.json")
    if not os.path.isfile(scenario_file_path):
        data = {
            "Budget": const.BUDGETS_CONSIDERED,
            "Dimensions": const.DIMS_CONSIDERED,
            "Problems": const.PROBS_CONSIDERED,
            "Repetitions": const.REPETITIONS,
            "Instances": const.TEST_INSTANCES
        }
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/scenario.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

def create_job_script(budget, dimensions, instances, repetitions, name, general):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=K_D{'_'.join(map(str, dimensions))}_B{budget}
#SBATCH --output={name}/B{budget}_D{'_'.join(map(str, dimensions))}/slurm.out
#SBATCH --error={name}/B{budget}_D{'_'.join(map(str, dimensions))}/slurm.err
#SBATCH --time={const.TIME}
#SBATCH --partition={const.PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3000M

# Activate virtual environment
module load Python/3.11
source /storage/work/schiller/venvs/Selection/bin/activate

# Run the experiment
python run_ng_on_ioh.py  --name {name} --dimensions "{json.dumps(dimensions)}" --budget {budget} --instances "{json.dumps(instances)}" --repetitions {repetitions} --general {general}
"""
    return script_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run algorithms on IOH Benchmarks.')
    parser.add_argument('--name', type=str, help='Name of the result folder', required=False)
    parser.add_argument('--slurm', type=str, help='Whether to run on Slurm', required=False, default=False)
    parser.add_argument('--dimensions', type=str, help='Dimensions to run on (slurm)', required=False, default=None)
    parser.add_argument('--budget', type=str, help='Budgets to run on (slurm)', required=False, default=None)
    parser.add_argument('--instances', type=str, help='Instances to run on (slurm)', required=False, default=None)
    parser.add_argument('--repetitions', type=str, help='How often to run on each instance', required=False, default=None)
    parser.add_argument('--general', type=str, help='Whether to run general configs instead of specific ones', required=False, default=False)
    args = parser.parse_args()

    if args.dimensions is not None:
        dimensionalities = [json.loads(args.dimensions)]
    else:
        dimensionalities = const.DIMS_CONSIDERED

    if args.budget is not None:
        budgets = [int(args.budget)]
    else:
        budgets = const.BUDGETS_CONSIDERED

    if args.instances is not None:
        instances = json.loads(args.instances)
    else:
        instances = const.TEST_INSTANCES

    if args.repetitions is not None:
        repetitions = int(args.repetitions)
    else:
        repetitions = const.REPETITIONS

    # Basically, when general is set we only test generalised algorithms over multiple dimensions
    # Ugly implementation, but this is how it is
    if args.general == "True":
        general = True
    else:
        general = False

    # Prepare Output Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if args.name is not None:
        output_dir = Path(args.name)
        if args.slurm:
            output_dir = Path(f"Output/{args.name}_{timestamp}")
    else:
        output_dir = Path(f"Output/run_{timestamp}")
    write_scenario_file(output_dir)

    # Default Algorithms
    if args.slurm == False:
        Cobyla_Def = Cobyla() 
        MetaModelOnePlusOne_Def = MetaModelOnePlusOne(get_config("MetaModelOnePlusOne", None, None, True), "MetaModelOnePlusOne") 
        MetaModelFmin2_Def = MetaModelFmin2(get_config("MetaModelFmin2", None, None, True), "MetaModelFmin2") 

    for dimensions in dimensionalities:
        # Popsize of default depends on dimension
        # No need if only running general algorithms
        if not general:
            CMA_Def = CMA(get_config("CMA", dimensions, None, True), "CMA") 
            ChainMetaModelPowell_Def = ChainMetaModelPowell(get_config("ChainMetaModelPowell", dimensions, None, True), "ChainMetaModelPowell") 
            MetaModel_Def = MetaModel(get_config("MetaModel", dimensions, None, True), "MetaModel") 

        for budget in budgets:
            if args.slurm == False:

                CMA_Conf = CMA(get_config("CMA", dimensions, budget), "CMA_Conf") 
                ChainMetaModelPowell_Conf = ChainMetaModelPowell(get_config("ChainMetaModelPowell", dimensions, budget), "ChainMetaModelPowell_Conf") 
                MetaModel_Conf = MetaModel(get_config("MetaModel", dimensions, budget), "MetaModel_Conf") 
                MetaModelOnePlusOne_Conf = MetaModelOnePlusOne(get_config("MetaModelOnePlusOne", dimensions, budget), "MetaModelOnePlusOne_Conf") 
                MetaModelFmin2_Conf = MetaModelFmin2(get_config("MetaModelFmin2", dimensions, budget), "MetaModelFmin2_Conf") 

                if not general:
                    Algorithms =[
                        Cobyla_Def,
                        CMA_Def,
                        ChainMetaModelPowell_Def,
                        MetaModel_Def,
                        MetaModelOnePlusOne_Def,
                        MetaModelFmin2_Def,
                        CMA_Conf,
                        ChainMetaModelPowell_Conf,
                        MetaModel_Conf,
                        MetaModelOnePlusOne_Conf,
                        MetaModelFmin2_Conf
                    ]

                if general:
                    CMA_Conf.name = "CMA_Gen"
                    ChainMetaModelPowell_Conf.name = "ChainMetaModelPowell_Gen"
                    MetaModel_Conf.name = "MetaModel_Gen"
                    MetaModelOnePlusOne_Conf.name = "MetaModelOnePlusOne_Gen"
                    MetaModelFmin2_Conf.name = "MetaModelFmin2_Gen"
                    Algorithms =[
                        CMA_Conf,
                        ChainMetaModelPowell_Conf,
                        MetaModel_Conf,
                        MetaModelOnePlusOne_Conf,
                        MetaModelFmin2_Conf
                    ]

                for dimension in dimensions:
                    run_algos(Algorithms, const.PROBS_CONSIDERED, budget, dimension, repetitions, instances, True, output_dir)
                    time = datetime.datetime.now().strftime("%H-%M-%S")
                    print(f"({time}) Finished Budget: {budget}, Dimension: {dimension}")

            else: 
                job_script = create_job_script(budget, dimensions, instances, repetitions, output_dir, general)
                job_script_dir = output_dir / f"B{budget}_D{'_'.join(map(str, dimensions))}"
                os.makedirs(job_script_dir, exist_ok=True)
                job_script_path = job_script_dir / "slurm.sh"

                with open(job_script_path, 'w') as file:
                    file.write(job_script)
                
                # Submit the job script
                os.system(f"sbatch {job_script_path}")
