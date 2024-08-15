from pathlib import Path
from pathlib import PurePath
import json
import statistics
import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from nevergrad.optimization.base import OptCls, ConfiguredOptimizer

import constants as const
from run_ng_on_ioh import run_algos
from models import MetaModelFmin2, MetaModel, MetaModelOnePlusOne, ChainMetaModelPowell, CMA, Cobyla
from configurations import get_config

def get_optimiser(setting: tuple[int, int], optimiser: str) -> ConfiguredOptimizer:
    if optimiser.endswith("Conf"):
        optimiser = optimiser.replace("_Conf", "")
        config = get_config(optimiser, [setting[0]], setting[1])
        version = "_Conf"
    elif optimiser.endswith("Gen"):
        optimiser = optimiser.replace("_Gen", "")
        print(setting)
        if setting[0] == 235:
            config = get_config(optimiser, [2,3,5], setting[1])
        elif setting[0] == 1015:
            config = get_config(optimiser, [10,15], setting[1])
        elif setting[0] == 2351015:
            config = get_config(optimiser, [2,3,5,10,15], setting[1])
        else:
            print(f"Cant find config for {optimiser} with dimensions {setting[0]}")
        version = "_Gen"
    else:
        config = get_config(optimiser, [setting[0]], None, True)
        version = ""

    if optimiser == "CMA":
        return CMA(config, f"CMA{version}")
    if optimiser == "MetaModelOnePlusOne":
        return MetaModelOnePlusOne(config, f"MetaModelOnePlusOne{version}")
    if optimiser == "MetaModelFmin2":
        return MetaModelFmin2(config, f"MetaModelFmin2{version}")
    if optimiser == "ChainMetaModelPowell":
        return ChainMetaModelPowell(config, f"ChainMetaModelPowell{version}")
    if optimiser == "MetaModel":
        return MetaModel(config, f"MetaModel{version}")
    if optimiser == "Cobyla":
        return Cobyla()
    return None

def run_optimiser(def_alg: str, conf_alg: str, dim: int, bud: int, output: Path, gen_alg: str = None):
    def_alg = get_optimiser((dim, bud), def_alg)
    conf_alg = get_optimiser((dim, bud), conf_alg)
    algs = [def_alg, conf_alg]
    
    if gen_alg is not None:
        alg, _dim = gen_alg.split(".")
        gen_alg = get_optimiser((int(_dim), bud), alg)
        algs.append(gen_alg)

    run_algos(algs, const.PROBS_CONSIDERED, bud, dim, 1, const.COMPARE_INSTANCES, True, output)

    print("Finished!")
    

def create_job_script(budget, dimension, def_alg, conf_alg, output, gen_alg = None):
    gen_alg_param = f" --gen_alg {gen_alg}" if gen_alg else ""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=C_{dimension}_{budget}
#SBATCH --output={output}/B{budget}_D{dimension}/slurm.out
#SBATCH --error={output}/B{budget}_D{dimension}/slurm.err
#SBATCH --time={const.TIME}
#SBATCH --partition={const.PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3000M

# Activate virtual environment
module load Python/3.11
source /storage/work/schiller/venvs/Selection/bin/activate

# Run the experiment
python compare.py  --slurm True --dimension {dimension} --budget {budget} --def_alg {def_alg} --conf_alg {conf_alg} --output {output} {gen_alg_param}
"""

    return script_content


def compare_selectors(dir: Path, eval: bool = False, general: bool = False):
    # Construct default selector
    df = pd.read_csv(dir / "ranking_def.csv")
    df_def = df[df['rank test new'] == 1]
    def_selector = {(row['budget'], row['dimensions']): row['algorithm'] for index, row in df_def.iterrows()}
    # Construct configured selector
    df = pd.read_csv(dir / "ranking_conf.csv")
    df_conf = df[df['rank test new'] == 1]
    conf_selector = {(row['budget'], row['dimensions']): row['algorithm'] for index, row in df_conf.iterrows()}

    gen_selector = None
    if general:
        df = pd.read_csv(dir / "ranking_gen.csv")
        df_gen = df[df['rank test new'] == 1]
        gen_selector = {(row['budget'], row['dimensions']): row['algorithm'] for index, row in df_gen.iterrows()}

    output = Path(f"Comparison/{dir.name}")

    if eval == False:
    # Run Selectors
        for key in def_selector:
            if not general:
                job_script = create_job_script(key[0], key[1], def_selector[key], conf_selector[key], output)
            else:
                job_script = create_job_script(key[0], key[1], def_selector[key], conf_selector[key], output, gen_selector[key])
            job_script_dir = output / f"B{key[0]}_D{key[1]}"
            os.makedirs(job_script_dir, exist_ok=True)
            job_script_path = job_script_dir / "slurm.sh"
            with open(job_script_path, 'w') as file:
                file.write(job_script)
            
            os.system(f"sbatch {job_script_path}")
    else:
        analyse_result(def_selector, conf_selector, output, gen_selector)

def transform_performance(performance):
    performance = np.where(performance < 1e-10, 1e-10, performance)
    return np.log10(performance)

def analyse_result(def_selector, conf_selector, output, gen_selector = None):
    performance = {}
    points = {}
    for key in def_selector:
        df_def = pd.read_csv(output / f"B{key[0]}_D{key[1]}/{def_selector[key]}_processed/data.csv")
        df_conf = pd.read_csv(output / f"B{key[0]}_D{key[1]}/{conf_selector[key]}_processed/data.csv")
        
        # Calculate points
        if gen_selector is None:
            points_def = (df_def['performance'] <= df_conf['performance']).sum()
            points_conf = (df_conf['performance'] <= df_def['performance']).sum()
            points[key] = {"Def": points_def, "Conf": points_conf}
        else:
            df_gen = pd.read_csv(output / f"B{key[0]}_D{key[1]}/{gen_selector[key]}_processed/data.csv")
            points_def = ((df_def['performance'] <= df_conf['performance']) & 
                      (df_def['performance'] <= df_gen['performance'])).sum()
            points_conf = ((df_conf['performance'] <= df_def['performance']) & 
                        (df_conf['performance'] <= df_gen['performance'])).sum()
            points_gen = ((df_gen['performance'] <= df_def['performance']) & 
                        (df_gen['performance'] <= df_conf['performance'])).sum()
            points[key] = {"Def": points_def, "Gen": points_gen, "Conf": points_conf}


        # Calculate performance
        df_def["performance_trans"] = transform_performance(df_def["performance"])
        df_conf["performance_trans"] = transform_performance(df_conf["performance"])
        perf_def = df_def["performance_trans"].sum()
        perf_conf = df_conf["performance_trans"].sum()
        performance[key] = {"Def": perf_def, "Conf": perf_conf}

        if gen_selector is not None:
            df_gen["performance_trans"] = transform_performance(df_gen["performance"])
            perf_gen = df_gen["performance_trans"].sum()
            performance[key] = {"Def": perf_def, "Conf": perf_conf, "Gen": perf_gen}
            
    
    plot_heatmap(points, performance, output)
    plot_bar_graphs(points, performance, output)

    
def plot_heatmap(points, performance, output):
    points_diff = {key: val['Conf'] - val['Def'] for key, val in points.items()}
    performance_diff = {key: val['Conf'] - val['Def'] for key, val in performance.items()}

    # Convert dictionaries to DataFrames
    points_diff_df = pd.DataFrame.from_dict(points_diff, orient='index', columns=['Difference'])
    performance_diff_df = pd.DataFrame.from_dict(performance_diff, orient='index', columns=['Difference'])

    # Pivot the DataFrame for heatmap
    points_diff_pivot = points_diff_df.pivot_table(index=points_diff_df.index.map(lambda x: x[0]), columns=points_diff_df.index.map(lambda x: x[1]), values='Difference')
    performance_diff_pivot = performance_diff_df.pivot_table(index=performance_diff_df.index.map(lambda x: x[0]), columns=performance_diff_df.index.map(lambda x: x[1]), values='Difference')

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Heatmap for Points Difference
    sns.heatmap(points_diff_pivot, annot=True, fmt='d', cmap='coolwarm', center=0, cbar_kws={'label': 'Points Difference'}, ax=axes[0])
    axes[0].set_title('Points Difference Heatmap')

    # Heatmap for Performance Difference
    sns.heatmap(performance_diff_pivot, annot=True, fmt='.2f', cmap='coolwarm', center=0, cbar_kws={'label': 'Performance Difference'}, ax=axes[1])
    axes[1].set_title('Performance Difference Heatmap')

    plt.tight_layout()
    plt.savefig(output / "heatmap.pdf")


def plot_bar_graphs(points, performance, output):
     # Convert dictionaries to DataFrames
    points_data = [{'Budget': k[0], 'Dimension': k[1], 'Algorithm': 'Def', 'Points': v['Def']} 
                   for k, v in points.items()] + \
                  [{'Budget': k[0], 'Dimension': k[1], 'Algorithm': 'Conf', 'Points': v['Conf']} 
                   for k, v in points.items()] + \
                  [{'Budget': k[0], 'Dimension': k[1], 'Algorithm': 'Gen', 'Points': v['Gen']} 
                   for k, v in points.items() if 'Gen' in v]
    
    performance_data = [{'Budget': k[0], 'Dimension': k[1], 'Algorithm': 'Def', 'Performance': v['Def']} 
                        for k, v in performance.items()] + \
                       [{'Budget': k[0], 'Dimension': k[1], 'Algorithm': 'Conf', 'Performance': v['Conf']} 
                        for k, v in performance.items()] + \
                       [{'Budget': k[0], 'Dimension': k[1], 'Algorithm': 'Gen', 'Performance': v.get('Gen', 0)} 
                        for k, v in performance.items() if 'Gen' in v]

    points_df = pd.DataFrame(points_data)
    performance_df = pd.DataFrame(performance_data)

    # Function to plot bar graphs for each dimension and budget
    def plot_for_dimension(df, value_col, title, ylabel):
        dimensions = sorted(df['Dimension'].unique())
        budgets = sorted(df['Budget'].unique())
        num_dimensions = len(dimensions)
        num_budgets = len(budgets)
        
        fig, axes = plt.subplots(num_dimensions, num_budgets, figsize=(5*num_budgets, 5*num_dimensions), sharey=True, squeeze=False)
        bar_width = 0.8
        colors = sns.color_palette('Set1', 3)

        for dim_idx, dimension in enumerate(dimensions):
            for bud_idx, budget in enumerate(budgets):
                ax = axes[dim_idx, bud_idx]
                subset = df[(df['Dimension'] == dimension) & (df['Budget'] == budget)]
                
                bar_positions = [i - bar_width/2 for i in range(len(subset))]
                bars = ax.bar(bar_positions, subset[value_col], 
                              width=bar_width, label=subset['Algorithm'], 
                              color=[colors[i] for i in subset['Algorithm'].map({'Conf': 0, 'Def': 1, "Gen": 2})])
                
                # Display values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

                ax.set_xticks(bar_positions)
                ax.set_xticklabels(subset['Algorithm'])
                ax.set_title(f'Dim {dimension}, Budget {budget}')
                ax.set_xlabel('Algorithm')
                ax.set_ylabel(ylabel)
                # ax.legend(title='Algorithm', loc='upper right')
                if bud_idx == 0:
                    ax.set_ylabel(ylabel)
        
        plt.tight_layout()
        plt.savefig(output / f"{ylabel.lower()}_bar.pdf")
        plt.close()

    # Plot for points
    plot_for_dimension(points_df, 'Points', 'Points Comparison by Dimension and Budget', 'Points')
    # Plot for performance
    plot_for_dimension(performance_df, 'Performance', 'Performance Comparison by Dimension and Budget', 'Performance')
    
    print("Bar graphs can be found here: ", output)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run algorithms on IOH Benchmarks.')
    parser.add_argument('--general', type=str, help='Whether to include general configs', required=False, default=None)
    parser.add_argument('--eval', type=str, help='Whether to evaluate the results', required=False, default=False)
    # Not to be set by users
    # (Here slurm means that it was called by slurm, not that it calls slurm)
    parser.add_argument('--slurm', type=str, help='Whether to run on Slurm', required=False, default=False)
    parser.add_argument('--dimension', type=int, help='Dimensions to run on (slurm)', required=False, default=None)
    parser.add_argument('--budget', type=int, help='Budgets to run on (slurm)', required=False, default=None)
    parser.add_argument('--def_alg', type=str, help='Default alg to run', required=False, default=None)
    parser.add_argument('--conf_alg', type=str, help='Conf alg to run', required=False, default=None)
    parser.add_argument('--output', type=str, help='Output directory', required=False, default=None)
    parser.add_argument('--gen_alg', type=str, help='Output directory', required=False, default=None)
    args = parser.parse_args()

    if args.slurm == False:
        compare_selectors(Path("Output/_Compare"), args.eval, args.general)
    else:
        if args.gen_alg is None:
            run_optimiser(args.def_alg, args.conf_alg, args.dimension, args.budget, args.output)
        else:
            #TODO: A way to run the algorithm generalised for dim 2, 3, 5, 10, 15
            if args.dimension < 10:
                dim = "235"
            else:
                dim = "1015"
            gen_alg = f"{args.gen_alg}.{dim}"
            run_optimiser(args.def_alg, args.conf_alg, args.dimension, args.budget, args.output, gen_alg)


