#ifndef PELELM_PROB_PARM_H_
#define PELELM_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
   //Geometric conditions
   amrex::Real d_venturi 		= 0.013;
   amrex::Real d_swirlerexit 	= 0.02032;
   amrex::Real d_dump 			= 0.076;

   amrex::Real l_venturi = 0.0008382;
   amrex::Real l_swirlercup = 0.0043688;
   amrex::Real l_dump = 0.20;

   bool activate_fuel_injection = false;
   amrex::Real fuel_injection_max_radius=0.5;
   amrex::Real Y_O2_FUEL=0.065963;
   amrex::Real Y_N2_FUEL=0.434037;
   amrex::Real Y_C12H26_FUEL=0.5;

   // Chamber conditions
   amrex::Real P_mean        = 101325.0;
   amrex::Real T_mean        = 300.0;

};
#endif
