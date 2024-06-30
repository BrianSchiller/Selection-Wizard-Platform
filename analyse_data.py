#!/usr/bin/env python3
"""Process, analyse, and plot performance data."""

import argparse
from pathlib import Path
import sys

from experiment import Experiment
from experiment import NGOptChoice
from experiment import analyse_test_csvs
from experiment import test_plot_all


if __name__ == "__main__":

    analyse_test_csvs(Path("Output/20240624_18-12-34"), test_bbob=True)

    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     "data_dir",
    #     default=argparse.SUPPRESS,
    #     type=Path,
    #     help="Directory to analyse.")
    # parser.add_argument(
    #     "per_budget_data_dir",
    #     default=None,
    #     type=Path,
    #     nargs="?",  # 0 or 1
    #     help="Directory of budget specific data to analyse additionally.")
    # parser.add_argument(
    #     "--per-prob-set",
    #     required=False,
    #     action="store_true",
    #     help=("Do analysis of the BBOB results for different subsets of the "
    #           "BBOB problems. E.g., only the multimodal problems, or the "
    #           "problems most similar to MA-BBOB."))
    # parser.add_argument(
    #     "--ma",
    #     required=False,
    #     action="store_true",
    #     help="Analyse data_dir as being MA-BBOB preprocessed data.")
    # parser.add_argument(
    #     "--test-bbob",
    #     required=False,
    #     action="store_true",
    #     help="Analyse data_dir as being BBOB preprocessed test data.")
    # parser.add_argument(
    #     "--test-vs",
    #     required=False,
    #     action="store_true",
    #     help=("If set in addition to --ma or --test-bbob, compare only the"
    #           " NGOpt choice and the data choice instead of all algorithms."))
    # parser.add_argument(
    #     "--ma-plot",
    #     required=False,
    #     action="store_true",
    #     help=("Generate all plot(s) for the MA-BBOB data. If no other --ma "
    #           "argument is given, data_dir should be the path to the ranked "
    #           "MA-BBOB csv file. Use --test-vs to indicate which algorithms "
    #           "are compared (controls the output file names)."))
    # parser.add_argument(
    #     "--test-plot",
    #     required=False,
    #     action="store_true",
    #     help=("Use alone or with --test-bbob. Generate all plot(s) for the "
    #           "BBOB test data. If no other --test"
    #           "argument is given, data_dir should be the path to the ranked "
    #           "BBOB test csv file. Use --test-vs to indicate which algorithms"
    #           " are compared (controls the output file names)."))
    # parser.add_argument(
    #     "--test-loss",
    #     required=False,
    #     default=None,
    #     type=Path,
    #     help=("Use with --ma or --test-bbob. Path to dataframe with loss data "
    #           "per dimension-budget-algorithm-problem combination. If given "
    #           "plot lineplots with loss of algorithms per dimension-budget."))
    # parser.add_argument(
    #     "--all-budgets",
    #     required=False,
    #     action="store_true",
    #     help=("Compute algorithm ranks for all budgets and the available "
    #           "dimensionalities on the BBOB training data, and write these to "
    #           "a csv file."))

    # args = parser.parse_args()

    # prob_sets = [
    #     # "all",
    #     "separable", "low_cond", "high_cond", "multi_glob", "multi_weak",
    #     "multimodal",
    #     "ma-like_5", "ma-like_4",
    #     "ma-like_3", "ma-like_2",
    #     # "ma-like_0",  # Same as all
    #     "f1", "f2", "f3", "f4", "f5", "f6",
    #     "f7", "f8", "f9", "f10", "f11", "f12",
    #     "f13", "f14", "f15", "f16", "f17", "f18",
    #     "f19", "f20", "f21", "f22", "f23", "f24"]

    # # Analyse MA-BBOB preprocessed data
    # if args.ma is True:
    #     analyse_test_csvs(args.data_dir, ngopt_vs_data=args.test_vs,
    #                       plot=args.ma_plot, test_bbob=False)
    #     sys.exit()
    # # Plot MA-BBOB results from preprocessed data
    # elif args.ma_plot is True:
    #     test_plot_all(args.data_dir, ngopt_vs_data=args.test_vs,
    #                   perf_data=args.test_loss, test_bbob=False)
    #     sys.exit()
    # # Analyse test BBOB preprocessed data
    # elif args.test_bbob is True:
    #     analyse_test_csvs(args.data_dir, ngopt_vs_data=args.test_vs,
    #                       plot=args.test_plot, test_bbob=args.test_bbob)
    #     sys.exit()
    # # Plot test BBOB results from preprocessed data
    # elif args.test_plot is True:
    #     test_plot_all(args.data_dir, ngopt_vs_data=args.test_vs,
    #                   perf_data=args.test_loss, test_bbob=args.test_plot)
    #     sys.exit()
    # # Analyse BBOB training data and create csv ranking for all buds and dims
    # elif args.all_budgets is True:
    #     print("Creating a ranking csv for all available budgets + dimensios.")
    #     print("WARNING: This takes a very long time!")
    #     nevergrad_version = "0.6.0"

    #     # Load NGOpt choices
    #     hsv_file = Path("ngopt_choices/dims1-100evals1-10000_separator_"
    #                     f"{nevergrad_version}.hsv")
    #     ngopt = NGOptChoice(hsv_file)

    #     # Load experiment data considering all possible budgets and dimensions
    #     exp = Experiment(args.data_dir,
    #                      args.per_budget_data_dir,
    #                      ng_version=nevergrad_version, prob_set="all",
    #                      budgets=list(range(1, 10001)))

    #     # Write a CSV file with points and ranks of algorithms per dimension-
    #     # budget combination
    #     file_name = f"score_rank_all_buds_{nevergrad_version}"
    #     exp.write_score_rank_csv(file_name, ngopt)

    #     sys.exit()
    # # Plot BBOB results for all problems
    # else:
    #     nevergrad_version = "0.6.0"

    #     # Plot heatmap based on NGOpt14 choices using a dummy experiment
    #     hsv_file = Path("ngopt_choices/ngopt14_dims1-100evals1-10000_separator"
    #                     f"_{nevergrad_version}.hsv")
    #     ngopt = NGOptChoice(hsv_file)
    #     file_name = f"grid_ngopt14_{nevergrad_version}"
    #     exp = Experiment(None, None, ng_version=nevergrad_version)  # Dummy
    #     exp.plot_heatmap_ngopt(ngopt, file_name)

    #     # Load NGOpt choices
    #     hsv_file = Path("ngopt_choices/dims1-100evals1-10000_separator_"
    #                     f"{nevergrad_version}.hsv")
    #     ngopt = NGOptChoice(hsv_file)

    #     # Write NGOpt algorithm names, short names, ID mapping
    #     file_name = f"ngopt_algos_{nevergrad_version}"
    #     ngopt.write_unique_ngopt_algos_csv(file_name)

        # Load experiment data
        # exp = Experiment(args.data_dir,
        #                  args.per_budget_data_dir,
        #                  ng_version=nevergrad_version, prob_set="all")

    #     # Plot heatmap based on budget-specific BBOB data for all problems
    #     # (Only the algorithms chosen by NGOpt have budget-specific runs.)
    #     if args.per_budget_data_dir is not None:
    #         matrix = exp.get_scoring_matrix(ngopt=ngopt, bud_specific=True)
    #         file_name = f"grid_data_budget_specific_{nevergrad_version}"
    #         exp.plot_heatmap_data(matrix, ngopt, file_name)

    #     # Plot heatmap based on NGOpt choices
    #     file_name = f"grid_ngopt_{nevergrad_version}"
    #     exp.plot_heatmap_ngopt(ngopt, file_name)

    #     # Plot heatmap based on BBOB data for all problems
    #     file_name = f"grid_data_{nevergrad_version}"
    #     matrix = exp.get_scoring_matrix(ngopt=ngopt)
    #     exp.plot_heatmap_data(matrix, ngopt, file_name)
    #     # Also plot a version with cells left blank if it is the same as NGOpt
    #     file_name = f"{file_name}"
    #     exp.plot_heatmap_data(
    #         matrix, ngopt, blank_ngopt=True, file_name=file_name)

    #     # Write a CSV file with points and ranks of algorithms per dimension-
    #     # budget combination
    #     file_name = f"score_rank_{nevergrad_version}"
    #     exp.write_score_rank_csv(file_name, ngopt)

    #     # Also plot heatmaps on BBOB data per function and function group
    #     if args.per_prob_set is True:
    #         for prob_set in prob_sets:
    #             exp.set_problems(prob_set)
    #             matrix = exp.get_scoring_matrix(ngopt=ngopt)
    #             exp.plot_heatmap_data(matrix, ngopt, file_name)

    #     sys.exit()

    # nevergrad_version = "0.6.0"
    # hsv_file = Path("ngopt_choices/dims1-100evals1-10000_separator_"
    #                 f"{nevergrad_version}.hsv")
    # ngopt = NGOptChoice(hsv_file)
    # Look at all dimensions, but exclude the largest budget (10000) because
    # it was already included in the original experiments.
#    budgets = [dims * 100 for dims in const.DIMS_CONSIDERED if dims < 100]
#    file_name = f"ngopt_choices_{nevergrad_version}"
#    ngopt.write_ngopt_choices_csv(const.DIMS_CONSIDERED, budgets, file_name)
#    file_name = f"ngopt_algos_{nevergrad_version}"
#    ngopt.write_unique_ngopt_algos_csv(file_name)
#    exp = Experiment(args.data_dir,
#                     args.per_budget_data_dir,
#                     ng_version=nevergrad_version)
    # exp = Experiment(args.data_dir,
    #                  args.per_budget_data_dir,
    #                  # dimensionalities=[100, 35],
    #                  ng_version=nevergrad_version)
#    comp_data_dir = Path("data_seeds2_bud_dep_organised")
#    exp.load_comparison_data(comp_data_dir)
#    file_name = f"score_rank_{nevergrad_version}"
#    exp.write_score_rank_csv(file_name, ngopt)
    # file_name = f"medians_{nevergrad_version}"
    # exp.write_medians_csv(file_name, with_ranks=True)
#    file_name = f"scores_{nevergrad_version}"
#    exp.write_scoring_csv(file_name)
#    matrix = exp.get_scoring_matrix(ngopt=ngopt)
#    file_name = f"grid_{nevergrad_version}"
#    exp.plot_hist_grid(matrix, ngopt, file_name)
#    file_name = f"grid_data_{nevergrad_version}"
#    exp.plot_heatmap_data(matrix, ngopt, file_name)
#    file_name = f"best_comparison_{nevergrad_version}"
#    exp.write_performance_comparison_csv(file_name)
#    file_name = f"grid_data_budget_specific_{nevergrad_version}"
#    exp.plot_heatmap_data(matrix, ngopt, file_name)
#    print("Relevant algorithms:")
#    print(*exp.get_relevant_ngopt_algos(ngopt), sep="\n")
