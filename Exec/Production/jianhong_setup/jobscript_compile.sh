#!/bin/bash -l

#SBATCH --account=w47-gpu    # your account
#SBATCH --partition=gpu-dev # Using the gpu partition
#SBATCH --ntasks=8                 # Total number of tasks
#SBATCH --ntasks-per-node=8        # Set this for 1 mpi task per compute device

#SBATCH --gpus-per-task=1          # How many HIP compute devices to allocate to a  task
#SBATCH --gpu-bind=closest         # Bind each MPI task to the nearest GPU
#SBATCH --exclusive                # Use this to request all the resources on a node
#SBATCH --time=02:30:00
#SBATCH --output=outputfile.log
#SBATCH --error=errorfile.log

module load craype-x86-trento 
module swap PrgEnv-gnu PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
source ~/.bashrc
pele-dev

export MPICH_GPU_SUPPORT_ENABLED=1 # Enable GPU support with MPI

export OMP_NUM_THREADS=8    #cpus-per-task is set to 8 by default
export OMP_PLACES=cores     #To bind to cores 
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.
 
# Temporal workaround for avoiding Slingshot issues on shared nodes:
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

# Compile the software
##make -j 20 DIM=3 USE_HIP=TRUE USE_MPI=TRUE PELE_USE_MAGMA=TRUE Chemistry_Model=dodecane_lu_qss TPLrealclean
##make -j 20 DIM=3 USE_HIP=TRUE USE_MPI=TRUE PELE_USE_MAGMA=TRUE TPL
make clean
make -j 8 USE_HIP=TRUE USE_MPI=TRUE DIM=3 PELE_USE_MAGMA=TRUE

# Run a job with task placement and $BIND_OPTIONS
# srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS ./PeleLMeX3d.hip.x86-trento.MPI.HIP.ex inputs.3d_DodecaneQSS_maxgrid32
# srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS ./PeleLMeX3d.hip.x86-trento.MPI.HIP.ex inputs.3d_DodecaneQSS_maxgrid32_not_managed_memory
#srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS ./PeleLMeX3d.hip.x86-trento.MPI.HIP.ex inputs.3d_DodecaneQSS_baseline

#ARGS="inputs.3d_DodecaneQSS_baseline"
#ARGS="inputs.3d_DodecaneQSS_baseline amr.max_grid_size=64" # get memory error with this setting
#ARGS="inputs.3d_DodecaneQSS_baseline amr.max_grid_size=32 amrex.the_arena_is_managed=1" # memory error with this setting
#ARGS="inputs.3d_DodecaneQSS_baseline amr.max_grid_size=32 amrex.the_arena_is_managed=1 amrex.the_arena_init_size=8589934592" # much slower with this setign


#srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS ./PeleLMeX3d.hip.x86-trento.MPI.HIP.ex ${ARGS}






