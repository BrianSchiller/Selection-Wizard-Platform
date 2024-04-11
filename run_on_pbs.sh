#!/bin/bash

# To submit the job:
# qsub run_on_pbs.sh

# To watch to the job status:
# qstat [-u <username>]

# Set options
# -q Destination of the job. Destination names a queue, a server or a queue at a server
# -l Resource list
## select=:ncpus=:mpiprocs= indicate the number of nodes:cores:mpiprocs
## Use multiple of 2 with a maximum of 24 on 'ncpus' parameter, one node has 24 cores max
## With the 'select=3:ncpus=10:mpiprocs=10' option you get 30 cores on 3 nodes
## If you use select=1:ncpus=30 your job will NEVER run because no node has 30 cores.
## walltime= indicate the wall clock time limit in hh:mm:ss
# -N job name

#PBS -q beta
#PBS -l select=1:ncpus=1
#PBS -l walltime=36:00:00
#PBS -N ar4opt

# Job array from 0 to 1, in steps of 1
# For MeSU-Beta to run main experiment:
# First 0-467
# Second 468-935
# Remove space after hash to run:
# PBS -J 0-935:1

# To run NGOpt choices:
# Remove space after hash to run:
# PBS -J 0-271:1

# To run budget dependence test:
# Remove space after hash to run:
# PBS -J 0-38:1

# To run best algorithms and NGOpt choice on MA-BBOB problems or new BBOB instances:
# First 0-499
# Second 500-999
# Third 1000-1444
# Remove space after hash to run:
#PBS -J 0-1444:1

# Load modules
#. /etc/profile.d/modules.sh
module purge
module load python/3.10
module load conda3-2020.02
source activate ar4opt

# Move to working direcotry
cd $PBS_O_WORKDIR
OUTPUT='output'
mkdir -p $OUTPUT

# Prepare scratch directory space
SCRATCH=/scratchbeta/$USER/test_scratch_space
PROJECT='ar4opt_seed'
mkdir -p $SCRATCH/$PROJECT

cd $SCRATCH/$PROJECT

# Prepare run specific directory
mkdir $PBS_ARRAY_INDEX

# Copy input files to scratch
cp $PBS_O_WORKDIR/run_ng_on_ioh.py $SCRATCH/$PROJECT/$PBS_ARRAY_INDEX
cp $PBS_O_WORKDIR/constants.py $SCRATCH/$PROJECT/$PBS_ARRAY_INDEX
# For NGOpt choices and best algorithms on MA-BBOB problems:
cp -r $PBS_O_WORKDIR/csvs/ $SCRATCH/$PROJECT/$PBS_ARRAY_INDEX
# For best algorithms on MA-BBOB problems:
cp $PBS_O_WORKDIR/experiment.py $SCRATCH/$PROJECT/$PBS_ARRAY_INDEX

# Execute
cd $PBS_ARRAY_INDEX

# To run the main experiment:
#python3 run_ng_on_ioh.py --pbs-index-all-dims $PBS_ARRAY_INDEX 1> test.out 2> test.err
# To run NGOpt choices:
#python3 run_ng_on_ioh.py --pbs-index-ngopt $PBS_ARRAY_INDEX 1> test.out 2> test.err
# To run budget dependence test:
#python3 run_ng_on_ioh.py --pbs-index-bud-dep $PBS_ARRAY_INDEX 1> test.out 2> test.err
# To run best algorithms and NGOpt choice on MA-BBOB problems:
#python3 run_ng_on_ioh.py --pbs-index-ma $PBS_ARRAY_INDEX 1> test.out 2> test.err
# To run best algorithms and NGOpt choice on new BBOB instances:
python3 run_ng_on_ioh.py --pbs-index-test $PBS_ARRAY_INDEX 1> test.out 2> test.err

cd ..

# Move output
cp -rp $PBS_ARRAY_INDEX $PBS_O_WORKDIR/$OUTPUT

# Clean temporary directory
rm -rf "$SCRATCH/$PROJECT/$PBS_ARRAY_INDEX"
