#!/bin/bash
###
###
#SBATCH --job-name=ma-bbob.sh
#SBATCH --output=Tmp/ma-bbob.sh.txt
#SBATCH --error=Tmp/ma-bbob.sh.err
#SBATCH --array=0-288%289
#SBATCH --mem-per-cpu=3000
#SBATCH --exclude=ethnode[01-06,08-13,15-20,22-28,30-32]
#SBATCH --partition=graceADA
###
params=( $(seq 0 288 ) )

# To run the MA-BBOB problems for NGOpt choices:
srun -N1 -n1 --mem-per-cpu=3000 python3 ioh_ng_real.py --pbs-index-bud-dep ${params[$SLURM_ARRAY_TASK_ID]} 1> test.out 2> test.err
