TOP = /home/b/bsavard/zisen347/codes/exascale
PELELMEX_HOME      ?= ${TOP}/PeleLMeX
AMREX_HOME         ?= ${TOP}/amrex
PELE_PHYSICS_HOME  ?= ${TOP}/PelePhysics
AMREX_HYDRO_HOME   ?= ${TOP}/AMReX-Hydro
PELEMP_HOME        ?= ${TOP}/PeleMP
SUNDIALS_HOME=/home/b/bsavard/zisen347/opt/archive/sundials-6.7.0
HYPRE_DIR ?= /home/zisen347/opt/hypre-2.28.0

# AMReX
DIM = 3
COMP = gnu
PRECISION = DOUBLE
USE_EB = FALSE
USE_HYPRE = FALSE

# Profiling
PROFILE = FALSE
TINY_PROFILE = FALSE
COMM_PROFILE = FALSE
TRACE_PROFILE = FALSE
MEM_PROFILE = FALSE
USE_GPROF = FALSE

# Performance
USE_MPI = TRUE
USE_OMP = FALSE
USE_CUDA = FALSE
USE_HIP = FALSE
USE_SYCL = FALSE

# Debugging
DEBUG = FALSE
FSANITIZER = FALSE
THREAD_SANITIZER = FALSE

# PelePhysics
Chemistry_Model = BurkeH2
Eos_Model = Fuego
Transport_Model = Simple

include $(PELELMEX_HOME)/Exec/Make.PeleLMeX
