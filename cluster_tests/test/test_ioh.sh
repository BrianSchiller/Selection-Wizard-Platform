#!/bin/bash

# Set options
# -q Destination of the job. destination names a queue, a server or a queue at a server
# -l Resource list
## select=:ncpus=:mpiprocs= indicate the number of nodes:cores:mpiprocs
## Use multiple of 2 with a maximum of 24 on 'ncpus' parameter, one node has 24 cores max
## With the 'select=3:ncpus=10:mpiprocs=10' option you get 30 cores on 3 nodes
## If you use select=1:ncpus=30 your job will NEVER run because no node has 30 cores.
## walltime= indicate the wall clock time limit in hh:mm:ss
# -N job name

#PBS -q beta
#PBS -l select=1:ncpus=2
#PBS -l walltime=0:01:00
#PBS -N testjob

# Job array from 0 to 1, in steps of 1
#PBS -J 0-1:1

# Load modules
#. /etc/profile.d/modules.sh
module purge
module load python/3.10
module load conda3-2020.02
source activate env

# Move to working direcotry
cd $PBS_O_WORKDIR
OUTPUT='output'
mkdir -p $OUTPUT

# Prepare scratch directory space
SCRATCH=/scratchbeta/$USER/test_scratch_space
PROJECT='test'
mkdir -p $SCRATCH/$PROJECT

cd $SCRATCH/$PROJECT

# Prepare run specific directory
mkdir $PBS_ARRAY_INDEX

# Copy input files to scratch
cp $PBS_O_WORKDIR/ioh_ng_real.py $SCRATCH/$PROJECT/$PBS_ARRAY_INDEX

# Execute
cd $PBS_ARRAY_INDEX

if [ $PBS_ARRAY_INDEX -eq 0 ];
then
    python3 ioh_ng_real.py CMA 1> test.out 2> test.err
else
    python3 ioh_ng_real.py Cobyla 1> test.out 2> test.err
fi

cd ..

# Move output
cp -rp $PBS_ARRAY_INDEX $PBS_O_WORKDIR/$OUTPUT

# Clean temporary directory
rm -rf "$SCRATCH/$PROJECT/$PBS_ARRAY_INDEX"
