MACH ?= niagara

ifeq ($(MACH), niagara)
#TOP = /scratch/b/bsavard/zisen347/exascale_latest/PeleLMeX
TOP = /scratch/b/bsavard/zisen347/exascale_eb
PELE_HOME          ?= ${TOP}/PeleLMeX
AMREX_HOME         ?= ${TOP}/amrex
AMREX_HYDRO_HOME   ?= ${TOP}/AMReX-Hydro
PELE_PHYSICS_HOME  ?= ${TOP}/PelePhysics
SUNDIALS_HONE      ?= ${TOP}/PelePhysics/Submodules/sundials
#SUNDIALS_HOME = /Users/zisen/Codes/exascale_eb/PelePhysics/ThirdParty/INSTALL/clang
HYPRE_DIR ?= /home/zisen347/opt/hypre-2.28.0
endif

#ifeq ($(MACH), niagara)
#TOP = /Users/zisen/Codes/exascale_eb
#PELE_HOME      		?= /home/b/bsavard/zisen347/codes/exascale_eb/PeleLMeX
#AMREX_HOME        ?= /home/b/bsavard/zisen347/codes/exascale_eb/amrex
#AMREX_HYDRO_HOME  ?= /home/b/bsavard/zisen347/codes/exascale_eb/AMReX-Hydro
#PELE_PHYSICS_HOME ?= /home/b/bsavard/zisen347/codes/exascale_eb/PelePhysics
#HYPRE_DIR ?= /home/zisen347/opt/hypre-2.28.0
#SUNDIALS_HOME = /Users/zisen/Codes/exascale_eb/PelePhysics/ThirdParty/INSTALL/clang
#SUNDIALS_HONE = ${TOP}/Submodules/PelePhysics/Submodules/sundials
#endif

# Augxiliary fields
USE_AUX = TRUE
USE_MIXF = TRUE
USE_AGE = TRUE
USE_AGEPV = FALSE

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

# PeleLMeX
USE_PLASMA = FALSE

# PelePhysics
Chemistry_Model = BurkeH2
Reactions_Model = Fuego
Eos_Model = Fuego
Transport_Model = Simple

include $(PELE_HOME)/Exec/Make.PeleLMeX
