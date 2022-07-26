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
  amrex::Real Vax_max = 100.38;
  amrex::Real Vt_max = 69.1004;
  amrex::Real Vr_max = 10.953;
  amrex::Real Vt,Vr,Vax;
  amrex::Real rad=0;		//Normalised radius
  amrex::Real theta=0;
  amrex::Real rad2,rad3,rad4,rad5,rad6;

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
  rad = sqrt(x[1]*x[1]+x[2]*x[2])/0.0065;
  rad2=rad*rad;
  rad3=rad2*rad;
  rad4=rad2*rad2;
  //Vax using piecewise polynomials
  if(rad<0.305992308)
  {   
   Vax = 44.141*rad4 - 0.9797*rad3 - 1.7264*rad2 + 0.3147*rad - 0.2413;
  }
  else if(rad>=0.305992308 && rad<0.880469231)
  {
   Vax = -5.8675*rad4 + 18.068*rad3 - 21.319*rad2 + 12.122*rad - 2.123;
  }
  else
  {
   Vax = -66.92*rad3 + 199.57*rad2 - 196.03*rad + 64.397;
  }
  
  //Vt using piecewise polynomial
  if(rad<0.328516923)
  {
    Vt = 14.28*rad4 + 19.301*rad3 - 8.7313*rad2 + 1.4342*rad - 0.0517 ;
  }
  else if(rad>=0.328516923 && rad<0.929504615)
  {
   Vt = -0.1285*rad4 + 4.1168*rad3 - 9.8697*rad2 + 8.4315*rad - 1.5255;
  }
  else 
  {
    Vt = -780.4*rad3 + 2190.7*rad2 - 2049.4*rad + 639.92;
  }
  
  //Vr using piecewise polynomial
  if(rad<0.362524615)
  {
    Vr = -193.86*rad4 + 193.68*rad3 - 55.821*rad2 + 7.0488*rad - 0.1963;
  }
  else if(rad>=0.362524615 && rad<0.929504615)
  {
    Vr = -248.86*rad4*rad + 745.35*rad4 - 865.67*rad3 + 476.15*rad2 - 123.03*rad + 12.842; 
  }
  else
  {
    Vr = -72889*rad3 + 208825*rad2 - 199395*rad + 63452; 
  }
  
  theta = atan2(x[1],x[2]);
  Vt *= Vt_max;
  Vr *= Vr_max;
  
  // Add to the turb perturbation
  if ( rad < 0.4 ) { // Keep turbulence in the swirling jet only
      s_ext[VELX] = Vax_max*Vax;
      s_ext[VELZ] = cos(theta)*Vr-sin(theta)*Vt;
      s_ext[VELY] = sin(theta)*Vr+cos(theta)*Vt;;
  } else {
      s_ext[VELX] += Vax_max*Vax;
      s_ext[VELZ] += cos(theta)*Vr-sin(theta)*Vt;
      s_ext[VELY] += sin(theta)*Vr+cos(theta)*Vt;;
  }

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
