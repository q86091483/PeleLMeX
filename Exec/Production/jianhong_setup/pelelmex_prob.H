#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_Index.H>
#include <pelelmex_prob_parm.H>
#include <PMF.H>
#include <PMFData.H>
#include <PelePhysics.H>
#include <math.h>



// initialise
AMREX_GPU_DEVICE // is this a macro
AMREX_FORCE_INLINE
void
pelelmex_initdata(
  int i,
  int j,
  int k,
  int is_incompressible,
  amrex::Array4<amrex::Real> const& state,
  amrex::Array4<amrex::Real> const& aux,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* pmf_data)
{
  // Initialisation of some varibales for the cell, the geometry and the initial
  // values
  const amrex::Real* prob_lo = geomdata.ProbLo(); // jl- get the low bound of the domain, i.e. [x_low, y_low, z_low]
  const amrex::Real* prob_hi = geomdata.ProbHi(); // jl- get the high bound of the domain, i.e. [x_high, y_high, z_high]
  const amrex::Real* dx = geomdata.CellSize();    // jl- get the cell size i.e. [delta_x, delta_y, delta_z] 

  AMREX_D_TERM(const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0]; 
               , const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
               , const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];); // jl- this is used to compute the coordinate of the [i,j,k] cell, i.e. compute [x,y,z]

  AMREX_D_TERM(const amrex::Real Lx = prob_hi[0] - prob_lo[0];
               , const amrex::Real Ly = prob_hi[1] - prob_lo[1];
               , const amrex::Real Lz = prob_hi[2] - prob_lo[2];); // jl- this is used to compute the legth at all three directions [Lx, Ly, Lz]

  auto eos = pele::physics::PhysicsType::eos(); // jl- this is the equation of state, the type is auto
  //amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0}; //jl- GpuArray can also be used in CPU code.

  amrex::Real molefrac[NUM_SPECIES] = {0.0}; // jl- define array with NUM_SPECIES elements,
  amrex::Real massfrac[NUM_SPECIES] = {0.0}; // jl- define array with NUM_SPECIES elements

  // Set the pressure
  amrex::Real P_cgs = prob_parm.P_mean * 10.0; // Conversion to CGS //what is CGS and why it is needed//i.e. based on cm, g, and sec, the pressure unit in cgs is called 'barye', with 1 Pa = 10 barye. 

  amrex::Real mf; // mixture fraction
 
  amrex::Real mf_max;
  mf_max = 1.972222 * pow(z, 2.0) - 2.886 * z + 1.0;
//  if (z>0.9)
//  {
//  	mf_max = 0.0;
//  }
  mf_max = std::max(mf_max, 0.0); 
  amrex::Real ml_thickness = prob_parm.ml_thickness;
  amrex::Real D = prob_parm.D_fu;
  amrex::Real scaling_factor = 4.0;
  amrex::Real radsq;
  amrex::Real rad;
  radsq = pow(x,2)+pow(y,2);
  rad = sqrt(radsq); 

  
  mf = 0.5 + 0.5 * tanh((D/2.0) / (scaling_factor * ml_thickness) * ((D/2.0) / rad - rad / (D/2.0)));
  mf = mf*mf_max; 
   
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
  pele::physics::PMF::pmf(pmf_data, mf, mf, pmf_vals);

  for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = pmf_vals[3+n];
  }
  state(i,j,k,TEMP) = pmf_vals[0];

  //Density
  amrex::Real rho_cgs = 0.0;
  eos.PYT2R(P_cgs, massfrac, state(i,j,k,TEMP), rho_cgs);
  state(i,j,k,DENSITY) = rho_cgs * 1.0e3;

  amrex::Real h_cgs = 0.0;
  eos.TY2H(state(i,j,k,TEMP), massfrac, h_cgs);
  state(i,j,k,RHOH) = h_cgs * 1.0e-4 * state(i,j,k,DENSITY);

  for (int n = 0; n < NUM_SPECIES; n++) {
    state(i,j,k,FIRSTSPEC+n) = massfrac[n] * state(i,j,k,DENSITY);
  }

  amrex::Real x_velocity = 0.0;
  amrex::Real y_velocity = 0.0;
  amrex::Real z_velocity = prob_parm.V_init;
  AMREX_D_TERM(state(i, j, k, VELX) = x_velocity;
               , state(i, j, k, VELY) = y_velocity;
               , state(i, j, k, VELZ) = z_velocity);
  
}

//----------------------------------------------------------------------------------------------------
// Set the BC for inflows
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real x[AMREX_SPACEDIM],
  const int m_nAux,
  amrex::Real s_ext[NVAR],
  const int idir,
  const int sgn,
  const amrex::Real time,
  amrex::GeometryData const& geomdata,
  ProbParm const& prob_parm,
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();

  auto eos = pele::physics::PhysicsType::eos();
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};

  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real X_air[NUM_SPECIES] = {0.0};
  amrex::Real X_fuel[NUM_SPECIES] = {0.0};
  amrex::Real Y_air[NUM_SPECIES] = {0.0};
  amrex::Real Y_fuel[NUM_SPECIES] = {0.0};
  amrex::Real rho_cgs, P_cgs, RhoH_temp;
  amrex::Real radsq = 0.0, rad = 0.0;

  for (int n = 0; n < NVAR; n++) {
    s_ext[n] = 0.0;
  }
  
  amrex::Real mf = 0.0;
  amrex::Real D = prob_parm.D_fu;
  amrex::Real scaling_factor = 4.0;
  amrex::Real ml_thickness = prob_parm.ml_thickness;
 
  amrex::Real V_ox = prob_parm.V_ox;
  amrex::Real V_fu = prob_parm.V_fu; 

  amrex::Real T_ox = prob_parm.T_ox;
  amrex::Real T_fu = prob_parm.T_fu; 


  Y_air[O2_ID] = 0.233;
  Y_air[N2_ID] = 0.767;
 
  Y_fuel[prob_parm.fuelID] = 1.0;

  eos.Y2X(Y_air, X_air);
  eos.Y2X(Y_fuel, X_fuel);
    
  if ((idir == 2) && (sgn == 1)) {
	s_ext[VELX] = 0.0;
    s_ext[VELY] = 0.0;	
	radsq = pow(x[0], 2) + pow(x[1],2);
	rad = sqrt(radsq);
	mf = 0.5 + 0.5 * tanh((D/2.0) / (scaling_factor * ml_thickness) * ((D/2.0) / rad - rad / (D/2.0)));
 
	s_ext[VELZ] = (1.0 - mf) * V_ox + mf * V_fu;	
	s_ext[TEMP] = (1.0 - mf) * T_ox + mf * T_fu;
    P_cgs = prob_parm.P_mean * 10.0; // CGS->MKS conversion

    // Y
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = mf * Y_fuel[n] + (1.0 - mf) * Y_air[n];
    }
    eos.Y2X(massfrac, molefrac);

    eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
    s_ext[DENSITY] = rho_cgs * 1.0e3; // jl- why it is to be cgs 

    eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
    s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
    }
  }

}
//----------------------------------------------------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
zero_visc(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& beta,
  amrex::GeometryData const& geomdata,
  amrex::Box const& domainBox,
  const int dir,
  const int beta_comp,
  const int nComp)
{
  amrex::ignore_unused(
    i, j, k, beta, geomdata, domainBox, dir, beta_comp, nComp);

  // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
  // otherwise this routine could be called for other face diffusivity (Temp,
  // velocity, ...)
  if ((beta_comp == 0) && (nComp == NUM_SPECIES)) {
    if (dir == 2 && k <= domainBox.smallEnd(dir)) {
      // In the zone 2, we have an obstacle (fuel pipe). Thus, the flow is cut.
      const amrex::Real* prob_lo = geomdata.ProbLo();
      const amrex::Real* dx = geomdata.CellSize();
      const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
	  const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
	  const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
	
      amrex::Real radSq;
	  amrex::Real rad;
      radSq = pow(x,2)+pow(y,2);
      rad = sqrt(radSq); 
 

      if (rad>0.1716) // Hard coded size of the burner lip
                                         // (we can't use void get_zone as
                                         // Prob_parm is not called in this
                                         // function
      {
        for (int n = beta_comp; n < beta_comp + nComp; n++) {
          beta(i, j, k, n) = 0.0;
        }
      }
    }
  }


// the following lines added to include zero thermal conductivity  
   else if((beta_comp == NUM_SPECIES) &&(nComp == 1)){

    if (dir == 2 && k <= domainBox.smallEnd(dir)) {

      const amrex::Real* prob_lo = geomdata.ProbLo();
      const amrex::Real* dx = geomdata.CellSize();
      const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
	  const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
	  const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
	
      amrex::Real radSq;
	  amrex::Real rad;
      radSq = pow(x,2)+pow(y,2);
      rad = sqrt(radSq); 
 

      if (rad > 0.1716){
        beta(i,j,k,0) = 0.0;
      }
    }
  }

}



#endif

