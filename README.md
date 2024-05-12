# AR4Opt: Algorithm recommendation for optimisation

## Installation with pip
0. Optionally first create an environment with `python3 -m venv venv` and activate it with `source venv/bin/activate`
1. Run `pip install -r requirements.txt`

## Installation with conda
> On the MeSU-Beta PBS cluster, first run `module load conda3-2020.02`

### Install the conda environment
`conda env create -f environment.yml`

### Activate conda environment
`source activate ar4opt`

### Deactivate conda environment
`conda deactivate`

### Remove conda environment
`conda env remove -n ar4opt`

# Generate NGOpt choices data
To analyse and compare results with what NGOpt does, some data is needed about which algorithms NGOpt chooses for different scenarios. This is how this data is obtained:

1. Run `./get_algos.py` (fairly inefficient and may take, e.g., half an hour)  
Output:  
    * Plain text file with which algorithms are used when by NGOpt39 in Nevergrad 0.6.0 for dimensions 1 to 100 and budgets 1 to 10000 for continuous problems without noise. All portfolio methods are shortened to 'ConfPortfolio' (and thus not distinguishable): `ngopt_choices/0.6.0/dims1-100evals1-10000.txt`  
    * As above, but all portfolio methods have their full name (and are thus distinguishable): `ngopt_choices/0.6.0/dims1-100evals1-10000_full.txt`  
    * As above, but output in '.hsv' format (hex separated values): `ngopt_choices/dims1-100evals1-10000_separator_0.6.0.hsv`  
    * As above, but for NGOpt14 instead of NGOpt39: `ngopt_choices/ngopt14_dims1-100evals1-10000_separator_0.6.0.hsv`  
    * Heatmap plot of which algorithm NGOpt14 uses for each dimension (in 1 to 100) and budget (in 1 to 10000): `plots/heatmap/grid_ngopt14_0.6.0_d100.pdf`

# Analyse data

## Analyse BBOB results
1. Analyse BBOB result data to plot a heatmap of the best algorithm per dimension and budget combination:  
`./analyse_data.py data_seeds2_organised/`  
Add the flag `--all-budgets` to instead generate only `csvs/score_rank_all_buds_0.6.0.csv` which uses all budgets in the BBOB results (1-10000) instead of the 17 selected for the further experiments and the paper. NOTE: The code for this flag is not optimised and takes a long time to run, e.g., ~10 hours.  
Output:  
    * Heatmap plot showing algorithms that are selected by the data-driven method: `plots/heatmap/grid_data_0.6.0_d100.pdf`  
    * Heatmap plot showing algorithms that are selected by NGOpt39 in Nevergrad version 0.6.0: `plots/heatmap/grid_ngopt_0.6.0_d100.pdf`  
    * CSV with algorithms used by NGOpt39 in Nevergrad version 0.6.0, including the ID, short name, and full name we use internally: `csvs/ngopt_algos_0.6.0.csv`
    * CSV with score and ranking of the considered algorithms per considered budget-dimension combination: `csvs/score_rank_0.6.0.csv`  

2. Add the path to budget-specific runs for the choices of NGOpt to also plot a heatmap taking this into account with:  
`./analyse_data.py data_seeds2_organised/ data_seeds2_ngopt_organised/`  
Output:  
    * `plots/heatmap/grid_data_budget_specific_0.6.0_d100.pdf`  

3. Add `--per-prob-set` to also plot heatmaps for different subsets of/individual problems.  
Output:  
    * `plots/heatmap/grid_data_0.6.0_probs_separable_d100.pdf` - Separable functions
    * `plots/heatmap/grid_data_0.6.0_probs_low_cond_d100.pdf` - Low or moderate conditioning
    * `plots/heatmap/grid_data_0.6.0_probs_high_cond_d100.pdf` - High conditioning and unimodal
    * `plots/heatmap/grid_data_0.6.0_probs_multi_glob_d100.pdf` - Multi-modal with adequate global structure
    * `plots/heatmap/grid_data_0.6.0_probs_multi_weak_d100.pdf` - Multi-modal with weak global structure
    * `plots/heatmap/grid_data_0.6.0_probs_multimodal_d100.pdf` - Both sets of multi-modal functions

    * `plots/heatmap/grid_data_0.6.0_probs_ma-like_5_d100.pdf` - Functions most similar to MA-BBOB based on ELA features in two dimensions: 1, 3, 5, 6, 7, 10, 13, 20, 22, 23
    * `plots/heatmap/grid_data_0.6.0_probs_ma-like_4_d100.pdf` - The above, and functions nearly similar: 8, 18
    * `plots/heatmap/grid_data_0.6.0_probs_ma-like_3_d100.pdf` - The above, and functions somewhat alike: 4, 9, 12, 14, 15, 16, 17.
    * `plots/heatmap/grid_data_0.6.0_probs_ma-like_2_d100.pdf` - The above, and functions fairly unalike: 2, 11, 19, 21

    * `plots/heatmap/grid_data_0.6.0_probs_f1_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f2_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f3_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f4_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f5_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f6_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f7_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f8_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f9_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f10_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f11_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f12_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f13_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f14_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f15_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f16_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f17_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f18_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f19_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f20_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f21_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f22_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f23_d100.pdf`
    * `plots/heatmap/grid_data_0.6.0_probs_f24_d100.pdf`

## Prepare MA-BBOB experiments
1. Analyse the BBOB results as described above  
2. Run `gen_ma_probs.py`  

## Analyse MA-BBOB preprocessed results
1. Analyse MA-BBOB preprocessed data comparing all algorithms:  
`./analyse_data.py data_seeds2_ma_complete_organised/ --ma`  
Add `--ma-plot` to immediately generate all possible plots after analysis.  
Output: `csvs/ma-bbob/ranking.csv`, `csvs/ma-bbob/ranking_failed.csv` (only if any runs, partially, failed)

2. Analyse MA-BBOB preprocessed data comparing only the NGOpt choice and the data based choice:  
`./analyse_data.py data_seeds2_ma_complete_organised/ --ma --test-vs`  
Add `--ma-plot` to immediately generate all possible plots after analysis.  
Output: `csvs/ma-bbob/ranking_1v1.csv`, `csvs/ma-bbob/ranking_1v1_failed.csv` (only if any runs, partially, failed)

3. Plot MA-BBOB results comparing all algorithms:  
`./analyse_data.py csvs/ma-bbob/ranking.csv --ma-plot --test-loss csvs/ma-bbob/perf_data.csv`  
Output: `plots/heatmap/ma-bbob/grid_test_algos_d100.pdf`, `plots/heatmap/ma-bbob/grid_test_approach_d100.pdf`, `plots/line/ma-bbob/loss_log_grid.pdf`, `plots/line/ma-bbob/loss_percent_grid.pdf`, individual plots per dimension-budget combination under `plots/line/ma-bbob/single/`

4. Plot MA-BBOB results comparing only the NGOpt choice and the data based choice:  
`./analyse_data.py csvs/ma-bbob/ranking_1v1.csv --ma-plot --test-vs --test-loss csvs/ma-bbob/perf_data_1v1.csv`  
Output: `plots/heatmap/ma-bbob/grid_test_1v1_algos_d100.pdf`, `plots/heatmap/ma-bbob/grid_test_1v1_approach_d100.pdf`, `plots/line/ma-bbob/loss_log_1v1_grid.pdf`, `plots/line/ma-bbob/loss_percent_1v1_grid.pdf`, individual plots per dimension-budget combination under `plots/line/ma-bbob/single/`

## Analyse BBOB test instances preprocessed results
1. Analyse BBOB test preprocessed data comparing all algorithms:  
`./analyse_data.py data_seeds2_bbob_test_organised/ --test-bbob`  
Add `--test-plot` to immediately generate all possible plots after analysis.  
Output: `csvs/bbob_test/ranking.csv`, `csvs/bbob_test/ranking_failed.csv` (only if any runs, partially, failed)

2. Analyse BBOB test preprocessed data comparing only the NGOpt choice and the data based choice:  
`./analyse_data.py data_seeds2_bbob_test_organised/ --test-bbob --test-vs`  
Add `--test-plot` to immediately generate all possible plots after analysis.  
Output: `csvs/bbob_test/ranking_1v1.csv`, `csvs/bbob_test/ranking_1v1_failed.csv` (only if any runs, partially, failed)

3. Plot BBOB test results comparing all algorithms:  
`./analyse_data.py csvs/bbob_test/ranking.csv --test-plot --test-loss csvs/bbob_test/perf_data.csv`  
Output: `plots/heatmap/bbob_test/grid_test_algos_d100.pdf`, `plots/heatmap/bbob_test/grid_test_approach_d100.pdf`, `plots/line/bbob_test/loss_log_grid.pdf`, `plots/line/bbob_test/loss_percent_grid.pdf`, individual plots per dimension-budget combination under `plots/line/bbob_test/single/`

4. Plot BBOB test results comparing only the NGOpt choice and the data based choice:  
`./analyse_data.py csvs/bbob_test/ranking_1v1.csv --test-plot --test-vs --test-loss csvs/bbob_test/perf_data_1v1.csv`  
Output: `plots/heatmap/bbob_test/grid_test_1v1_algos_d100.pdf`, `plots/heatmap/bbob_test/grid_test_1v1_approach_d100.pdf`, `plots/line/bbob_test/loss_log_1v1_grid.pdf`, `plots/line/bbob_test/loss_percent_1v1_grid.pdf`, individual plots per dimension-budget combination under `plots/line/bbob_test/single/`

# Run the data-driven selector
1. Make sure `csvs/score_rank_0.6.0.csv` (and/or `csvs/score_rank_all_buds_0.6.0.csv`) and `csvs/ngopt_algos_0.6.0.csv` exist (i.e., run the first command under "Analyse BBOB results").  
2. Call the selector as `./run_selector.py <budget> <dimensions>`, e.g., `./run_selector.py 2500 15`. If the given budget or dimensionality does not exist in the ranking data csv file, the closest available are used instead. Use the flag `--full` to use the full range of budgets (1-10000, this only works if the `csvs/score_rank_all_buds_0.6.0.csv` file exists).  
3. Run: `./run_ng_on_ioh.py --algorithms DDS` (with other options as desired) to run the selector on the problems included in the experiments.  

# Collect data for other algorithms

## Other algorithms implemented in Nevergrad
1. Import the algorithm in `run_ng_on_ioh.py`. E.g., for TwoPointDE add a line to the imports like: `from nevergrad.optimization.optimizerlib import TwoPointsDE`.
2. Run: `./run_ng_on_ioh.py --algorithms TwoPointsDE` (with other options as desired). NOTE: Some algorithms may need to be wrapped in `"` quotes when calling them. E.g., `ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)` should be called as `./run_ng_on_ioh.py --algorithms "ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)"`.
