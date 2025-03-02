#!/bin/bash
#PBS -P w47
#PBS -N prf70
#PBS -l mem=960GB
#PBS -l storage=scratch/w47
#PBS -l jobfs=960GB
#PBS -l ncpus=480
#PBS -q normal
#PBS -l walltime=05:00:00
#PBS -j oe


echo ">>starting:" `date`


cd "$PBS_O_WORKDIR" || exit $?

ulimit -s unlimited
module load gcc openmpi

mpirun -np 480 ./PeleLMeX3d.gnu.MPI.ex input.3d


echo ">>ending:  " `date`
