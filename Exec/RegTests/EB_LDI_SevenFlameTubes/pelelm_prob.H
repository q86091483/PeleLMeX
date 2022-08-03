#ifndef PELELM_PROB_H_
#define PELELM_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLM_Index.H>
#include <pelelm_prob_parm.H>
#include <PMF.H>
#include <PMFData.H>
#include <PelePhysics.H>

// -----------------------------------------------------------
// Search for the closest index in an array to a given value
// using the bisection technique.
// INPUTS/OUTPUTS:
// xtable(0:n-1) => array to search in (ascending order)
// n             => array size
// x             => x location
// idxlo        <=> output st. xtable(idxlo) <= x < xtable(idxlo+1)
// -----------------------------------------------------------
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
locate(const amrex::Real* xtable, const int n, const amrex::Real& x, int& idxlo)
{
  // If x is out of bounds, return boundary index
  if (x >= xtable[n - 1]) {
    idxlo = n - 1;
    return;
  }
  if (x <= xtable[0]) {
    idxlo = 0;
    return;
  }

  // Do the bisection
  idxlo = 0;
  int idxhi = n - 1;
  bool notdone = true;
  while (notdone) {
    if (idxhi - idxlo <= 1) {
      notdone = false;
    } else {
      const int idxmid = (idxhi + idxlo) / 2;
      if (x >= xtable[idxmid]) {
        idxlo = idxmid;
      } else {
        idxhi = idxmid;
      }   
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void pelelm_initdata(int i, int j, int k,
                     int is_incompressible,
                     amrex::Array4<amrex::Real> const& state,
                     amrex::Array4<amrex::Real> const& /*aux*/,
                     amrex::GeometryData const& geomdata,
                     ProbParm const& prob_parm,
                     pele::physics::PMF::PmfData::DataContainer const * pmf_data)
{
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx      = geomdata.CellSize();

    amrex::Real x[3] = {
      prob_lo[0] + static_cast<amrex::Real>(i + 0.5) * dx[0],
      prob_lo[1] + static_cast<amrex::Real>(j + 0.5) * dx[1],
      prob_lo[2] + static_cast<amrex::Real>(k + 0.5) * dx[2]};

    AMREX_D_TERM(const amrex::Real Lx = prob_hi[0] - prob_lo[0];,
                 const amrex::Real Ly = prob_hi[1] - prob_lo[1];,
                 const amrex::Real Lz = prob_hi[2] - prob_lo[2]);

    AMREX_D_TERM(const amrex::Real xc = prob_lo[0] + 0.5 * Lx;,
                 const amrex::Real yc = prob_lo[1] + 0.5 * Ly;,
                 const amrex::Real zc = prob_lo[2] + 0.5 * Lz;);

    constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real massfrac[NUM_SPECIES] = {0.0};

    // Species mass
    massfrac[O2_ID] = 0.233;
    massfrac[N2_ID] = 0.767;
    state(i,j,k,TEMP) = prob_parm.T_mean;

    state(i,j,k,VELX) = 0.0;
    state(i,j,k,VELY) = 0.0;
    state(i,j,k,VELZ) = 0.0;

    amrex::Real P_cgs = prob_parm.P_mean*10;

    // Density
    amrex::Real rho_cgs = 0.0;
    eos.PYT2R(P_cgs, massfrac, state(i,j,k,TEMP), rho_cgs);
    state(i,j,k,DENSITY) = rho_cgs * 1.0e3;

    // Enthalpy
    amrex::Real h_cgs = 0.0;
    eos.TY2H(state(i,j,k,TEMP), massfrac, h_cgs);
    state(i,j,k,RHOH) = h_cgs * 1.0e-4 * state(i,j,k,DENSITY);

    for (int n = 0; n < NUM_SPECIES; n++) {
           state(i,j,k,FIRSTSPEC+n) = massfrac[n] * state(i,j,k,DENSITY);
        }

}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
time_eta(
  amrex::Real time, amrex::Real start, amrex::Real duration, amrex::Real tau)
{
  amrex::Real eta_time;
  eta_time = 0.5 * (std::tanh((time - start) / (duration * tau)) *
                    std::tanh((-time + start + duration) / (duration * tau)) +
                    1.0);
  eta_time = std::min(std::max(eta_time, 0.0), 1.0);
  return eta_time;
}

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
  pele::physics::PMF::PmfData::DataContainer const *pmf_data)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;
  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real Vax_max = 143.0;
  amrex::Real Vt_max = 102.0;
  amrex::Real Vr_max = 14.0;
  amrex::Real Vt=0,Vr=0,Vax=0;
  amrex::Real rad=0;		//Normalised radius
  amrex::Real theta=0;
  amrex::Real rad2,rad3,rad4,rad5,rad6;
  amrex::Real radius_cup=0.0065;

  //Center locations of all cups
  amrex::Real cen_y_swirler_0,cen_z_swirler_0;
  amrex::Real cen_y_swirler_1,cen_z_swirler_1;
  amrex::Real cen_y_swirler_2,cen_z_swirler_2;
  amrex::Real cen_y_swirler_3,cen_z_swirler_3;
  amrex::Real cen_y_swirler_4,cen_z_swirler_4;
  amrex::Real cen_y_swirler_5,cen_z_swirler_5;
  amrex::Real cen_y_swirler_6,cen_z_swirler_6;

  cen_y_swirler_0 = 0.0; cen_z_swirler_0 = 0.0;
  cen_y_swirler_1 = -0.011904858869; cen_z_swirler_1 = 0.020620105463;
  cen_y_swirler_2 = -0.02380996; cen_z_swirler_2 = 0.0;
  cen_y_swirler_3 = -0.011904858869; cen_z_swirler_3 = -0.020620105463;
  cen_y_swirler_4 = 0.011904858869; cen_z_swirler_4 = -0.020620105463;
  cen_y_swirler_5 = 0.02380996; cen_z_swirler_5 = 0.0;
  cen_y_swirler_6 = 0.011904858869; cen_z_swirler_6 = 0.020620105463;
  
  // Chamber composition
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  massfrac[O2_ID] = 0.233;
  massfrac[N2_ID] = 0.767;

  amrex::Real p_cgs = prob_parm.P_mean * 10.0;
  amrex::Real rho_cgs = 0.0;
  amrex::Real RhoH_temp = 0.0;
  s_ext[TEMP] = prob_parm.T_mean;
  eos.PYT2R(p_cgs, massfrac, s_ext[TEMP], rho_cgs);
  s_ext[DENSITY] = rho_cgs * 1.0e3;
  eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
  s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];   // CGS -> MKS conversion
  for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC+n] = massfrac[n] * s_ext[DENSITY];
    }

  //Velocity density
  if(sqrt((x[1]-cen_y_swirler_0)*(x[1]-cen_y_swirler_0)+(x[2]-cen_z_swirler_0)*(x[2]-cen_z_swirler_0))<=radius_cup)
  {
	  rad = sqrt((x[1]-cen_y_swirler_0)*(x[1]-cen_y_swirler_0)+(x[2]-cen_z_swirler_0)*(x[2]-cen_z_swirler_0))/0.0065;
  }
  else if(sqrt((x[1]-cen_y_swirler_1)*(x[1]-cen_y_swirler_1)+(x[2]-cen_z_swirler_1)*(x[2]-cen_z_swirler_1))<=radius_cup)
  {
	  rad = sqrt((x[1]-cen_y_swirler_1)*(x[1]-cen_y_swirler_1)+(x[2]-cen_z_swirler_1)*(x[2]-cen_z_swirler_1))/0.0065;
  }
  else if(sqrt((x[1]-cen_y_swirler_2)*(x[1]-cen_y_swirler_2)+(x[2]-cen_z_swirler_2)*(x[2]-cen_z_swirler_2))<=radius_cup)
 {
         rad = sqrt((x[1]-cen_y_swirler_2)*(x[1]-cen_y_swirler_2)+(x[2]-cen_z_swirler_2)*(x[2]-cen_z_swirler_2))/0.0065;
 }
  else if(sqrt((x[1]-cen_y_swirler_3)*(x[1]-cen_y_swirler_3)+(x[2]-cen_z_swirler_3)*(x[2]-cen_z_swirler_3))<=radius_cup)
 {
         rad = sqrt((x[1]-cen_y_swirler_3)*(x[1]-cen_y_swirler_3)+(x[2]-cen_z_swirler_3)*(x[2]-cen_z_swirler_3))/0.0065;
 }
  else if(sqrt((x[1]-cen_y_swirler_4)*(x[1]-cen_y_swirler_4)+(x[2]-cen_z_swirler_4)*(x[2]-cen_z_swirler_4))<=radius_cup)
 {
         rad = sqrt((x[1]-cen_y_swirler_4)*(x[1]-cen_y_swirler_4)+(x[2]-cen_z_swirler_4)*(x[2]-cen_z_swirler_4))/0.0065;
 }
  else if(sqrt((x[1]-cen_y_swirler_5)*(x[1]-cen_y_swirler_5)+(x[2]-cen_z_swirler_5)*(x[2]-cen_z_swirler_5))<=radius_cup)
 {
         rad = sqrt((x[1]-cen_y_swirler_5)*(x[1]-cen_y_swirler_5)+(x[2]-cen_z_swirler_5)*(x[2]-cen_z_swirler_5))/0.0065;
 }
  else if(sqrt((x[1]-cen_y_swirler_6)*(x[1]-cen_y_swirler_6)+(x[2]-cen_z_swirler_6)*(x[2]-cen_z_swirler_6))<=radius_cup)
 {
         rad = sqrt((x[1]-cen_y_swirler_6)*(x[1]-cen_y_swirler_6)+(x[2]-cen_z_swirler_6)*(x[2]-cen_z_swirler_6))/0.0065;
 }

  if(rad<=1.0)
  {
  
  rad2=rad*rad;
  rad3=rad2*rad;
  rad4=rad2*rad2;

  //Vax using piecewise polynomials
  if(rad<0.272546154)
  {
   Vax = -87.333*rad4 + 70.653*rad3 - 12.225*rad2+ 1.0613*rad - 0.2365;
  }
  else if(rad>=0.272546154 && rad<0.83056)
  {
   Vax = -4.1043*rad4 + 13.292*rad3 - 16.291*rad2 + 9.5823*rad - 1.5582;
  }
  else if(rad>=0.83056 && rad<=1.0)
  {
   Vax = -1017.7*rad4 + 3734.4*rad3 - 5124.4*rad2 + 3117.6*rad - 708.9;
  }
  else
  {
   //amrex::Print()<<"\nrad outof boudna vax "<<rad;
  }

  //Vt using piecewise polynomial
  if(rad<0.3023)
  {
    Vt =  -161.32*rad4 + 127.4*rad3 - 30.622*rad2 + 4.0507*rad - 0.1307;
  }
  else if(rad>=0.3023 && rad<0.9489)
  {
   Vt = -4.1079*rad4+ 13.818*rad3 - 17.652*rad2 + 10.342*rad - 1.3959 ;
  }
  else if(rad>=0.9489 && rad<=1)
  {
    Vt = -1.7043*rad+2.6172 ;
  }
  else
  {
   // amrex::Print()<<"\nrad out of bounds "<<rad;
  }
  
  //Vr using piecewise polynomial
  if(rad<0.33912)
  {
   Vr = -194.09*rad4 + 176.94*rad3 - 47.218*rad2 + 6.2139*rad - 0.1896;
  }
  else if(rad>=0.33912 && rad<0.877387692)
  {
   Vr = -28.831*rad4 + 80.084*rad3 - 90.188*rad2 + 42.115*rad - 5.8326 ; 
  }
  else if(rad>=0.877387692 && rad<=1.0)
  {
   Vr = -49175*rad4+ 179772*rad3 - 246359*rad2 + 149975*rad - 34219;
  }
  else
  {
   // amrex::Print()<<"\nError rad out of bounds vr "<<rad;
  }
  
  theta = atan2(x[1],x[2]);
  Vt *= Vt_max;
  Vr *= Vr_max;
}
  s_ext[VELX] = Vax_max*Vax;
  s_ext[VELZ] = cos(theta)*Vr-sin(theta)*Vt;
  s_ext[VELY] = sin(theta)*Vr+cos(theta)*Vt;;

}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
zero_visc (int i, int j, int k,
           amrex::Array4<amrex::Real> const& beta,
           amrex::GeometryData const& geomdata,
           amrex::Box const& domainBox,
           const int  dir,
           const int  beta_comp,
           const int  nComp)
{
     amrex::ignore_unused(i,j,k,beta,geomdata,domainBox,dir,beta_comp,nComp);
    // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
    // otherwise this routine could be called for other face diffusivity (Temp, velocity, ...)
}
#endif
