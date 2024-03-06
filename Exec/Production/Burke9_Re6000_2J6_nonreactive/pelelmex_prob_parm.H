#ifndef PELELM_PROB_PARM_H
#define PELELM_PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
  amrex::Real P_mean = 1013250.0_rt; 	
  amrex::Real V_j = 80.5_rt;
  amrex::Real V_cf = 25.0_rt;
  amrex::Real jet_rad = 1.0e-3_rt;
  amrex::Real jet_temp = 400.0_rt;
  amrex::Real global_eq_ratio = 0.6_rt;
  amrex::Real ox_temp = 750._rt;
  amrex::Real X_O2 = 2.1e-1_rt;
  amrex::Real X_N2 = 7.9e-1_rt;
  amrex::Real pertmag_cf = 0.0_rt;
  amrex::Real pertmag_jet = 0.0_rt;
  amrex::Real jet_purity = 0.65_rt;
  amrex::Real bl_thickness = 2.0e-4_rt;

  amrex::Real init_time = 1.0e-6_rt;
  int double_jet = 0;

  int jet_dir = 2;
  int cf_dir = 0;
  int other_dir = 1;

};
#endif
