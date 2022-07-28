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

   // Chamber conditions
   amrex::Real P_mean        = 101325.0;
   amrex::Real T_mean        = 300.0;

   // Spray
   amrex::Real ps_halfangle = 35.0_rt;
   amrex::Real ps_r0        = 0.0005_rt;
   amrex::Real ps_rmsvel    = 5.0_rt;

   amrex::Real part_temp    = 300.0;
   amrex::Real mass_flow_rate = 0.003;

   amrex::Real jet_start_time = 0.0;
   amrex::Real jet_end_time = 1.0;

   amrex::Real floating_injection_mass = 0.0;
   amrex::Real part_dia_saved = -1.0;
   amrex::Real part_dia_min = 1.0e-7;

};
#endif
