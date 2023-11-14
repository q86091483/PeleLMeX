#!/bin/bash
#SBATCH --job-name=rjicf
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=40
#SBATCH --time=10:00:00
#SBATCH --output=mpi_output_%j.txt
#SBATCH --mail-type=FAIL

cd $SLURM_SUBMIT_DIR

srun -n 400 PeleLMeX3d.intel-classic.MPI.ex input.3d

echo ">>ending:  " `date`
