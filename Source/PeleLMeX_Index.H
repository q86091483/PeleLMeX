#ifndef PELELM_INDEX_H
#define PELELM_INDEX_H

#define VELX 0
#define VELY 1
#define VELZ 2
#define DENSITY AMREX_SPACEDIM
#define FIRSTSPEC (DENSITY + 1)
#define RHOH (FIRSTSPEC + NUM_SPECIES)
#define TEMP (RHOH + 1)
#define RHORT (TEMP + 1)
#ifdef PELE_USE_EFIELD
#define NE (RHORT + 1)
#define PHIV (NE + 1)
#define NUMEFIELD 2
#else
#define NUMEFIELD 0
#endif
#ifdef PELELM_USE_SOOT
#define FIRSTSOOT (RHORT + NUMEFIELD + 1)
#define NUMSOOTVAR (NUM_SOOT_MOMENTS + 1)
#else
#define NUMSOOTVAR 0
#endif

#define FIRSTAUX (RHORT + NUMEFIELD + NUMSOOTVAR + 1)

#ifdef PELE_USE_AUX
  // Age variable
  #ifdef PELE_USE_AGE
    #define AGE FIRSTAUX
    #define NUMAGE 1
  #else
    #define AGE 0
    #define NUMAGE 0
  #endif
  // Age progress variable
  #ifdef PELE_USE_AGEPV
    #define AGEPV (FIRSTAUX + NUMAGE)
    #define NUMAGEPV 1
  #else
    #define AGEPV 0
    #define NUMAGEPV 0
  #endif
  #define NUMAUX (NUMAGE + NUMAGEPV)
#else
  #define NUMAUX 0
#endif

#define NVAR (FIRSTAUX + NUMAUX)

#endif