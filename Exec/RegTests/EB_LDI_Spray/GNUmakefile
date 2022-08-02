TOP = ../../../..
AMREX_HOME         ?= ${TOP}/amrex
PELELMEX_HOME      ?= ${TOP}/PeleLMeX
PELE_PHYSICS_HOME  ?= ${TOP}/PelePhysics
AMREX_HYDRO_HOME   ?= ${TOP}/AMReX-Hydro
#HYPRE_HOME         = /ccs/home/lesclapez/hypre/install


# AMReX
DIM             = 3
DEBUG           = FALSE
PRECISION       = DOUBLE
VERBOSE         = FALSE
TINY_PROFILE    = FALSE
USE_EB		    = TRUE
USE_HYPRE	    = FALSE

# Compilation
COMP            = gnu
USE_MPI         = TRUE
USE_OMP         = FALSE
USE_CUDA        = TRUE
USE_HIP         = FALSE

# PeleLMeX
CEXE_header    += EBUserDefined.H

PELE_USE_MAGMA  = FALSE

# PeleMP
USE_PARTICLES   = TRUE
SPRAY_FUEL_NUM  = 1
CEXE_sources   += SprayParticlesInitInsert.cpp

# PelePhysics
#Chemistry_Model = dodmethair_4sp
Chemistry_Model = dodecane_lu_qss
Eos_Model	    = Fuego
Transport_Model = Simple

include $(PELELMEX_HOME)/Utils/Make.PeleLMeX
