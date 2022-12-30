TOP = /home/snadakka/Code_Installations/PeleLMeX_LDI_Push/Submodules/
AMREX_HOME         = ${TOP}/amrex
PELELMEX_HOME      = /home/snadakka/Code_Installations/PeleLMeX_LDI_Push
PELE_PHYSICS_HOME  = ${TOP}/PelePhysics
AMREX_HYDRO_HOME   = ${TOP}/AMReX-Hydro
PELEMP_HOME        = ${TOP}/PeleMP


# AMReX
DIM             = 3
DEBUG           = FALSE
PRECISION       = DOUBLE
VERBOSE         = FALSE
TINY_PROFILE    = FALSE
USE_EB          = TRUE
USE_HYPRE       = FALSE

# Compilation
COMP            = gnu
USE_MPI         = TRUE
USE_OMP         = FALSE
USE_CUDA        = FALSE
USE_HIP         = FALSE

# PeleLMeX
CEXE_header    += EBUserDefined.H

# PeleMP
#USE_PARTICLES   = TRUE
#SPRAY_FUEL_NUM  = 1
#CEXE_sources   += SprayParticlesInitInsert.cpp
#PELELM_USE_SPRAY = TRUE

# PelePhysics
Chemistry_Model = dodmethair_4sp
#Chemistry_Model = dodecane_lu_qss 
Eos_Model	= Fuego
Transport_Model = Simple

include $(PELELMEX_HOME)/Utils/Make.PeleLMeX
