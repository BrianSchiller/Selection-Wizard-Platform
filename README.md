# AR4Opt: Algorithm recommendation for optimisation

## Installation with pip
Run `pip install -r requirements.txt`

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

# Analyse data

## Analyse MA-BBOB preprocessed results
1. Analyse MA-BBOB preprocessed data comparing all algorithms: `./analyse_data.py data_seeds2_ma_organised/ --ma`
Output: `csvs/ma_ranking.csv`, `csvs/ma_ranking_failed.csv` (only if any runs, partially, failed)
2. Analyse MA-BBOB preprocessed data comparing only the NGOpt choice and the data based choice: `./analyse_data.py data_seeds2_ma_organised/ --ma --ma-vs`
Output: `csvs/ma_ranking_1v1.csv`, `csvs/ma_ranking_1v1_failed.csv` (only if any runs, partially, failed)
3. Plot MA-BBOB results comparing all algorithms: `./analyse_data csvs/ma_ranking.csv --ma-plot`
Output: `plots/heatmap/grid_test_algos_d100.pdf`, `plots/heatmap/grid_test_approach_d100.pdf`
4. Plot MA-BBOB results comparing only the NGOpt choice and the data based choice: `./analyse_data csvs/ma_ranking_1v1.csv --ma-plot --ma-vs`
Output: `plots/heatmap/grid_test_1v1_algos_d100.pdf`, `plots/heatmap/grid_test_1v1_approach_d100.pdf`
