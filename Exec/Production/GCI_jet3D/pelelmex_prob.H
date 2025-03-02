#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_Index.H>
#include <PelePhysics.H>
#include <pelelmex_prob_parm.H>
#include <PMFData.H>
#include <PMF.H>

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
void get_nist_enthalpy(const std::string& name, const amrex::Real& temp, amrex::Real& res_h) 
{
      //-------------------------------------------------------------------------
      //
      // This is implemented in Fortran by Armin Wehrfritz.
      // Modified by Zisen Li for C++
      //
      // Calculate the enthalpy of the fuel based on the liquid properties
      // obtained from NIST. This thus accounts for the heat of evaporation.
      // Currently only implemented for n-dodecane, n-heptane and iso-octane.
      //
      // IN:
      //   std::string              name: fuel name
      //   const amrex::Real        &temp: temperature
      //
      // OUT:
      //   amrex::real              &res_h: enthalpy
      //
      // 1.  Liquid n-dodecane data from NIST:
      //     https://webbook.nist.gov/cgi/cbook.cgi?ID=C112403&Mask=2#Thermo-Condensed
      //
      //     W                      (kg/kmol)   = 170.3348
      //     DeltaH_liq             (kJ/mol)    = -352.1 +/- 1.4
      //     C_{p,liq} @ T=298.15 K (J/(mol.K)) = 376.00
      //
      // 2.  Liquid n-heptane data from NIST:
      //     https://webbook.nist.gov/cgi/cbook.cgi?ID=C142825&Mask=2#Thermo-Condensed
      //
      //     W                      (kg/kmol)   = 100.2019
      //     DeltaH_liq             (kJ/mol)    = -224.4 +/- 0.79
      //     C_{p,liq} @ T=298.15 K (J/(mol.K)) = 224.64
      //
      // 3.  Liquid iso-octane data from NIST:
      //     https://webbook.nist.gov/cgi/cbook.cgi?ID=C540841&Mask=2#Thermo-Condensed
      //
      //     W                      (kg/kmol)   = 114.2285
      //     DeltaH_liq             (kJ/mol)    = -259.3 +/- 1.3
      //     C_{p,liq} @ T=298.15 K (J/(mol.K)) = 242.49
      //-------------------------------------------------------------------------

      amrex::Real temp_nist_ref = 298.15_rt;
      amrex::Real h, inv_h_ref, W, dh_liq, cp_liq;

      if (name == "NC12H26") {
        W          = 170.3348E-3; // kg/mol
        dh_liq     = -352.1E3;    // J/mol
        cp_liq     = 376.00;      // J/((mol.K))
      } else if (name == "NC7H16") {
        W          = 100.2019E-3; // kg/mol
        dh_liq     = -224.4E3;    // J/mol
        cp_liq     = 224.64;      // J/(mol.K)
      } else if (name == "IC8H18") {
        W          = 114.228E-3; // kg/mol
        dh_liq     = -259.3E3;    // J/mol
        cp_liq     = 242.49;      // J/(mol.K)
      }

      dh_liq = dh_liq/W;          // J/kg
      cp_liq = cp_liq/W;          // J/(kg.K)
      res_h = dh_liq + cp_liq*(temp - temp_nist_ref);

      return;
}

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
void getTurbVel(amrex::Real t, amrex::Real x, amrex::Real y, amrex::Real z,
                amrex::GeometryData const& geomdata,
                ProbParm const& prob_parm,
                amrex::GpuArray<amrex::Real, 3>& uinterp)
{
    // Inlet Velocity fluctuation
    // Fill in the velocities
    amrex::Real xinterp[3] = {0.0};
    xinterp[0] = x;
    xinterp[1] = y;
    xinterp[2] = z;

    // Domain end
    const amrex::Real* prob_lo = geomdata.ProbLo();

    // Interpolation factors
    amrex::Real mod[3] = {0.0};
    int idx[3] = {0};
    int idxp1[3] = {0};
    amrex::Real slp[3] = {0.0};

    int idim;

    // x
    idim = 0;
    mod[idim] = x;
    locate(prob_parm.xin, prob_parm.nxin, mod[idim], idx[idim]);
    idxp1[idim] = (idx[idim] + 1) % prob_parm.nxin;
    slp[idim] = (mod[idim] - prob_parm.xin[idx[idim]]) / prob_parm.dxin[idx[idim]];
 
    // y
    idim = 1;
    mod[idim] = y;
    locate(prob_parm.yin, prob_parm.nyin, mod[idim], idx[idim]);
    idxp1[idim] = (idx[idim] + 1) % prob_parm.nyin;
    slp[idim] = (mod[idim] - prob_parm.yin[idx[idim]]) / prob_parm.dyin[idx[idim]];

    // z
    idim = 2;
    if (prob_parm.do_turbInlet == 1) {
      mod[idim] = prob_parm.Lxin - std::fmod(prob_parm.V_j * t, prob_parm.Lxin);
      mod[idim] = std::fmod(x - prob_lo[0] + mod[idim], prob_parm.Lxin) + prob_parm.xin[0];
    } else if (prob_parm.do_turbInlet == 2) {
      mod[idim] = z;
    }
    locate(prob_parm.zin, prob_parm.nzin, mod[idim], idx[idim]);
    idxp1[idim] = (idx[idim] + 1) % prob_parm.nzin;
    slp[idim] = (mod[idim] - prob_parm.zin[idx[idim]]) / prob_parm.dzin[idx[idim]];

    // Zisen - debug
    amrex::Real rad = std::sqrt(x*x + y*y);
    //amrex::Print() << x << ", " << y << ", " << z << std::endl;
    //if (rad < 1.7E-4) {
    //amrex::AllPrint() << "mod[z]: " << mod[idim] << ", idx: " << idx[idim] << ", idxp1: " << idxp1[idim] << ", " << slp[idim] << std::endl;
    //amrex::AllPrint() << "zin: [" << prob_parm.zin[0] << ", " << prob_parm.zin[482] << "]" << std::endl; 
    //}


    int nxin = prob_parm.nxin;
    int n1 = prob_parm.nxin;
    int n2 = prob_parm.nyin;

    const amrex::Real f0 = (1 - slp[0]) * (1 - slp[1]) * (1 - slp[2]);
    const amrex::Real f1 = slp[0] * (1 - slp[1]) * (1 - slp[2]);
    const amrex::Real f2 = (1 - slp[0]) * slp[1] * (1 - slp[2]);
    const amrex::Real f3 = (1 - slp[0]) * (1 - slp[1]) * slp[2];
    const amrex::Real f4 = slp[0] * (1 - slp[1]) * slp[2];
    const amrex::Real f5 = (1 - slp[0]) * slp[1] * slp[2];
    const amrex::Real f6 = slp[0] * slp[1] * (1 - slp[2]);
    const amrex::Real f7 = slp[0] * slp[1] * slp[2];
    uinterp[0] =   prob_parm.uin[  idx[0] + n1 * (  idx[1] + n2 * idx[2])] * f0
                 + prob_parm.uin[idxp1[0] + n1 * (  idx[1] + n2 * idx[2])] * f1
                 + prob_parm.uin[  idx[0] + n1 * (idxp1[1] + n2 * idx[2])] * f2
                 + prob_parm.uin[  idx[0] + n1 * (  idx[1] + n2 * idxp1[2])] * f3
                 + prob_parm.uin[idxp1[0] + n1 * (  idx[1] + n2 * idxp1[2])] * f4
                 + prob_parm.uin[  idx[0] + n1 * (idxp1[1] + n2 * idxp1[2])] * f5
                 + prob_parm.uin[idxp1[0] + n1 * (idxp1[1] + n2 * idx[2])] * f6
                 + prob_parm.uin[idxp1[0] + n1 * (idxp1[1] + n2 * idxp1[2])] * f7; 
    uinterp[1] =   prob_parm.vin[  idx[0] + n1 * (  idx[1] + n2 * idx[2])] * f0
                 + prob_parm.vin[idxp1[0] + n1 * (  idx[1] + n2 * idx[2])] * f1
                 + prob_parm.vin[  idx[0] + n1 * (idxp1[1] + n2 * idx[2])] * f2
                 + prob_parm.vin[  idx[0] + n1 * (  idx[1] + n2 * idxp1[2])] * f3
                 + prob_parm.vin[idxp1[0] + n1 * (  idx[1] + n2 * idxp1[2])] * f4
                 + prob_parm.vin[  idx[0] + n1 * (idxp1[1] + n2 * idxp1[2])] * f5
                 + prob_parm.vin[idxp1[0] + n1 * (idxp1[1] + n2 * idx[2])] * f6
                 + prob_parm.vin[idxp1[0] + n1 * (idxp1[1] + n2 * idxp1[2])] * f7;
    uinterp[2] =   prob_parm.win[  idx[0] + n1 * (  idx[1] + n2 * idx[2])] * f0
                 + prob_parm.win[idxp1[0] + n1 * (  idx[1] + n2 * idx[2])] * f1
                 + prob_parm.win[  idx[0] + n1 * (idxp1[1] + n2 * idx[2])] * f2
                 + prob_parm.win[  idx[0] + n1 * (  idx[1] + n2 * idxp1[2])] * f3
                 + prob_parm.win[idxp1[0] + n1 * (  idx[1] + n2 * idxp1[2])] * f4
                 + prob_parm.win[  idx[0] + n1 * (idxp1[1] + n2 * idxp1[2])] * f5
                 + prob_parm.win[idxp1[0] + n1 * (idxp1[1] + n2 * idx[2])] * f6
                 + prob_parm.win[idxp1[0] + n1 * (idxp1[1] + n2 * idxp1[2])] * f7;
}
    
    
AMREX_GPU_DEVICE
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
	// Get table
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};

	// Get MPI info
  int myproc = amrex::ParallelDescriptor::MyProc();
  int nprocs = amrex::ParallelDescriptor::NProcs();

	// Get geometry
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();

  AMREX_D_TERM(const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
               , const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
               , const amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];);

  AMREX_D_TERM(const amrex::Real Lx = prob_hi[0] - prob_lo[0];
               , const amrex::Real Ly = prob_hi[1] - prob_lo[1];
               , const amrex::Real Lz = prob_hi[2] - prob_lo[2]);

  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

  amrex::Real V_cf = prob_parm.V_cf;
  amrex::Real V_j = prob_parm.V_j;
  amrex::Real jet_rad = prob_parm.jet_rad;
  amrex::Real Zc = prob_parm.Zc;

	// Get eos model
  auto eos = pele::physics::PhysicsType::eos();

  // Mixture fraction
  amrex::Real atwCHON[4] = {0.0};
  pele::physics::eos::atomic_weightsCHON<pele::physics::PhysicsType::eos_type>(atwCHON);
  //for (int i = 0; i < 4; i++) {
  //  amrex::Print() << "atwCHON[" <<i << "]: " << atwCHON[i] << std::endl;
  //}
  int ecompCHON[NUM_SPECIES * 4];
  pele::physics::eos::element_compositionCHON<
    pele::physics::PhysicsType::eos_type>(ecompCHON);
  //amrex::Print() << "Number of atm (C,H,O,N) in species (molecular weight):" << std::endl;
  amrex::Real mwt[NUM_SPECIES];
  eos.molecular_weight(mwt);
  //for (int i = 0; i < NUM_SPECIES; ++i) {
  //  for (int k = 0; k < 4; k++) {
  //    amrex::Print() << ecompCHON[i*4+k] << ", ";
  //  }
  //  amrex::Print() << mwt[i];
  //  amrex::Print() << std::endl;
  //}
  amrex::Real Beta_mix[4] = {0.0};
  Beta_mix[0] = (atwCHON[0] != 0.0) ? 2.0 / atwCHON[0] : 0.0;
  Beta_mix[1] = (atwCHON[1] != 0.0) ? 1.0 / (2.0 * atwCHON[1]) : 0.0;
  Beta_mix[2] = (atwCHON[2] != 0.0) ? -1.0 / atwCHON[2] : 0.0;
  Beta_mix[3] = 0.0;

	// Get XF, XO, YF, YO
  amrex::Real YF[NUM_SPECIES] = {0.0};
  amrex::Real XF[NUM_SPECIES] = {0.0};
  amrex::Real YO[NUM_SPECIES] = {0.0};
  amrex::Real XO[NUM_SPECIES] = {0.0};
  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};

  XF[IC8H18_ID] = prob_parm.prf / 100.;
  XF[NC7H16_ID] = 1 - XF[IC8H18_ID];
  eos.X2Y(XF, YF);

  XO[O2_ID] = prob_parm.X_O2;
  XO[N2_ID] = 1 - XO[O2_ID];
  eos.X2Y(XO, YO);

  // Radius
  amrex::Real rad = sqrt(pow(x, 2) + pow(y, 2));

  // Initial mixture fraction
  amrex::Real factor = 0.0;
  factor = 0.5 *
      (1.0 - tanh((rad - jet_rad) / prob_parm.bl_thickness));
  factor = 0.0;
  // Initial Y
  for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = Zc * factor * YF[n] + (1.0 - Zc * factor) * YO[n];
  }
  eos.Y2X(massfrac, molefrac);

  // Initial u, v, w
  amrex::Real x_velocity = 0.0;
  amrex::Real y_velocity = 0.0;
  amrex::Real z_velocity = 0.0;

#if (AMREX_SPACEDIM == 2)
  x_velocity = 0.0;
  y_velocity = V_cf;
#elif (AMREX_SPACEDIM == 3)
  x_velocity = 0.0;
  y_velocity = 0.0;
  z_velocity = V_j * factor + V_cf * (1 - factor);

  if (prob_parm.do_turbInlet == 1) {
    amrex::GpuArray<amrex::Real, 3> uinterp = {0.0};
    amrex::Real tcur = 0.0;
    getTurbVel(tcur, x, y, tcur, geomdata, prob_parm, uinterp);
    x_velocity += uinterp[0];
    y_velocity += uinterp[1];
    z_velocity += uinterp[2];
  }
#endif

  AMREX_D_TERM(state(i, j, k, VELX) = x_velocity;
               , state(i, j, k, VELY) = y_velocity;
               , state(i, j, k, VELZ) = z_velocity);

	// Initial P
  amrex::Real P_cgs = prob_parm.P_mean * 10.0;

  // Initial T
  amrex::Real Hjet_cgs, Hair_cgs, Hmix_cgs;
  eos.TY2H(prob_parm.jet_temp, YF, Hjet_cgs);
  eos.TY2H(prob_parm.ox_temp,  YO, Hair_cgs);

  Hmix_cgs = Hjet_cgs * Zc * factor +
             Hair_cgs * (1 - Zc * factor);

  amrex::Real Tmix = 0.0; // Without initial value for Tmix, there would be nan.
  {
    eos.HY2T(Hmix_cgs, massfrac, Tmix);
    state(i, j, k, TEMP) = Tmix;
  }

  // Initial rho
  amrex::Real rho_cgs;
  {
    eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
    state(i, j, k, DENSITY) = rho_cgs * 1.0e3;
  }

  // Initial H
  amrex::Real h_cgs = 0.0;
  eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);
  state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY); // CGS -> MKS conversion

  // Initial Y
  for (int n = 0; n < NUM_SPECIES; n++) {
    state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
  }

  // Initial auxiliary field
#if defined(PELE_USE_AUX) && (NUMAUX > 0)
  #if defined(PELE_USE_MIXF) && (NUMMIXF > 0)
    amrex::Real spec_Bilger_fact[NUM_SPECIES] = {0.0};
    amrex::Real Zfu = 0.0;
    amrex::Real Zox = 0.0;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      spec_Bilger_fact[n] = 0.0;
      for (int m = 0; m < 4; m++) {
        spec_Bilger_fact[n] +=
          Beta_mix[m] * (ecompCHON[n * 4 + m] * atwCHON[m] / mwt[n]);
      }
      Zfu += spec_Bilger_fact[n] * YF[n];
      Zox += spec_Bilger_fact[n] * YO[n];

      // amrex::Print() << "In bcnormal, spec_Bilger_fact[ " << n << "] = " << spec_Bilger_fact[n] << std::endl;
    }
    // amrex::Print() << "Zfu: " << Zfu << ", Zox: " << Zox << std::endl;

    amrex::Real mixfrac = 0.0;
    for (int n = 0; n < NUM_SPECIES; n++) {
      mixfrac += massfrac[n] * spec_Bilger_fact[n];
    }
    mixfrac = (mixfrac - Zox) / (Zfu - Zox);

    state(i, j, k, MIXF) = mixfrac * state(i, j, k, DENSITY);
    //amrex::Print() << "Init data. " <<  state(i, j, k, MIXF) << "Zfu: " << Zfu << ", Zox: " << Zox << std::endl;
    if (NUMMIXF > 1) {
      state(i, j, k, MIXF+1) = state(i,j,k,DENSITY) * (1 - mixfrac);
    }
  #endif

  #if defined(PELE_USE_AGE) && (NUMAGE > 0)
    state(i, j, k, AGE) = ((z - prob_lo[2]) / V_j) * state(i, j, k, MIXF);
    #if (NUMAGE > 1)
      state(i, j, k, AGE+1) = ((z - prob_lo[2]) / V_cf) * state(i, j, k, MIXF+1);
    #endif
  #endif

  #if defined(PELE_USE_AGEPV) && (NUMAGEPV > 0)
    for (int n = 0; n < NUMAGEPV; n++) {
      state(i, j, k, AGEPV+n) = 0.0;
    }
    state(i, j, k, AGEPV+1) = ((z - prob_lo[2]) / V_cf) * state(i, j, k, MIXF+1);
  #endif

#endif
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
  pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
{
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* prob_hi = geomdata.ProbHi();
  const amrex::Real* dx = geomdata.CellSize();
  constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};

  amrex::Real V_j = prob_parm.V_j;
  amrex::Real V_cf = prob_parm.V_cf;
  amrex::Real jet_rad = prob_parm.jet_rad;
  amrex::Real Zc = prob_parm.Zc;
  int cf_dir 	= prob_parm.cf_dir;
  int jet_dir 	= prob_parm.jet_dir;
  int other_dir = prob_parm.other_dir;

  amrex::Real X_air[NUM_SPECIES] = {0.0};
  amrex::Real X_jet[NUM_SPECIES] = {0.0};
  amrex::Real Y_air[NUM_SPECIES] = {0.0};
  amrex::Real Y_jet[NUM_SPECIES] = {0.0};
  amrex::Real U_air[AMREX_SPACEDIM] = {0.0};
  amrex::Real U_jet[AMREX_SPACEDIM] = {0.0};
  amrex::Real rho_air = 0.0;
  amrex::Real rho_jet = 0.0;
  amrex::Real radsq = 0.0;
  amrex::Real rad = 0.0;
  amrex::Real factor = 0.0;
  amrex::Real area = 0.0;
  amrex::Real jet_vel = 0.0;

  auto eos = pele::physics::PhysicsType::eos();
  amrex::Real atwCHON[4] = {0.0};
  pele::physics::eos::atomic_weightsCHON<pele::physics::PhysicsType::eos_type>(atwCHON);
  int ecompCHON[NUM_SPECIES * 4];
  pele::physics::eos::element_compositionCHON<
    pele::physics::PhysicsType::eos_type>(ecompCHON);
  amrex::Real mwt[NUM_SPECIES];
  eos.molecular_weight(mwt);
  amrex::Real Beta_mix[4] = {0.0};
  Beta_mix[0] = (atwCHON[0] != 0.0) ? 2.0 / atwCHON[0] : 0.0;
  Beta_mix[1] = (atwCHON[1] != 0.0) ? 1.0 / (2.0 * atwCHON[1]) : 0.0;
  Beta_mix[2] = (atwCHON[2] != 0.0) ? -1.0 / atwCHON[2] : 0.0;
  Beta_mix[3] = 0.0;

  // Initialize the jet and air molefracs
  X_air[O2_ID] = prob_parm.X_O2;
  X_air[N2_ID] = 1 - X_air[O2_ID];
  X_jet[IC8H18_ID] = prob_parm.prf / 100.;
  X_jet[NC7H16_ID] = 1 - X_jet[IC8H18_ID];

  // Get air/jet mass fraction and densities
  // Mass fraction
  eos.X2Y(X_air, Y_air);
  eos.X2Y(X_jet, Y_jet);
  // Density
  eos.PYT2R(prob_parm.P_mean * 10.0, Y_air, prob_parm.ox_temp, rho_air);
  eos.PYT2R(prob_parm.P_mean * 10.0, Y_jet, prob_parm.jet_temp, rho_jet);
  rho_air *= 1.0e3; // CGS -> MKS conversion
  rho_jet *= 1.0e3;

  // Initialize jet velocity BCs
#if (AMREX_SPACEDIM == 2)
  amrex::Abort("GCI jet is not implemented in 2D");
#elif (AMREX_SPACEDIM == 3)
  area = (prob_hi[other_dir] - prob_lo[other_dir]) *
         (prob_hi[jet_dir] - prob_lo[jet_dir]);

  U_jet[jet_dir] = V_j;
  U_air[jet_dir] = V_cf;
#endif

  int njet = 1;
  amrex::Vector<amrex::Real> jet_xs = {0.0};
  amrex::Vector<amrex::Real> jet_ys = {0.0};
  amrex::Real radsq_min = 1E30;

  if (idir == jet_dir and sgn == 1) {

    amrex::Real P_cgs, Hjet_cgs, Hair_cgs, Hmix_cgs;
    P_cgs = prob_parm.P_mean * 10.0;

    // Radius
    for (int i = 0; i<njet; i++) {
      radsq = pow(x[other_dir]-jet_ys[i], 2) + pow(x[cf_dir]-jet_xs[i], 2);
      radsq_min = std::min(radsq_min, radsq);
    }
    radsq = radsq_min;
    rad = sqrt(radsq);

    // Mixture fraction
    factor = 0.5 *
      (1.0 - tanh((rad - jet_rad) / prob_parm.bl_thickness));

    // Y
    for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = Zc * factor * Y_jet[n] +
                    (1.0 - Zc * factor) * Y_air[n];
    }
    eos.Y2X(massfrac, molefrac);

    // H
    eos.TY2H(prob_parm.jet_temp, Y_jet, Hjet_cgs);
    eos.TY2H(prob_parm.ox_temp,  Y_air, Hair_cgs);
    if (false) { // correct fuel enthalpy based on NIST enthalpy
      amrex::GpuArray<amrex::Real, NUM_SPECIES> hfu_spec = {0.0_rt};
      eos.T2Hi(prob_parm.jet_temp, hfu_spec.data());
      get_nist_enthalpy("NC7H16", prob_parm.jet_temp, hfu_spec[NC7H16_ID]); 
      get_nist_enthalpy("IC8H18", prob_parm.jet_temp,  hfu_spec[IC8H18_ID]); 
      Hjet_cgs = 0.0;
      for (int i = 0; i < NUM_SPECIES; i++) {
        Hjet_cgs += Y_jet[i] * hfu_spec[i]; 
      }
      Hjet_cgs *= 1E4; // MKS -> CGS
    }

    Hmix_cgs = Hjet_cgs * Zc * factor +
               Hair_cgs * (1 - Zc * factor);

    // T
    amrex::Real Tmix = 0.0; // Without initial value for Tmix, there would be nan.
    {
      eos.HY2T(Hmix_cgs, massfrac, Tmix);
      s_ext[TEMP] = Tmix;
    }

    // rho
    amrex::Real rho_cgs;
    {
      eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
      s_ext[DENSITY] = rho_cgs * 1.0e3;
    }

    // rhoH
    eos.TY2H(s_ext[TEMP], massfrac, Hmix_cgs);
    s_ext[RHOH] = Hmix_cgs * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
    }

    // U, V, W
    amrex::Real time_factor = 0.0;
    if (time < prob_parm.init_time) {
      time_factor = (time / prob_parm.init_time) * (time / prob_parm.init_time);
    } else {
      time_factor = 1.0;
    }
    time_factor = 1.0;
    s_ext[VELX] = 0.0;
    s_ext[VELY] = 0.0;
    s_ext[VELZ] = 0.0;
    // If add HIT turbulence to velocity field
    amrex::GpuArray<amrex::Real, 3> uinterp = {0.0};
    if (prob_parm.do_turbInlet > 0) {
      amrex::Real tcur = time;
      if (prob_parm.do_turbInlet == 1) {
        getTurbVel(tcur, x[0], x[1], x[2], geomdata, prob_parm, uinterp);
      } else if (prob_parm.do_turbInlet == 2) {
        getTurbVel(tcur, x[0], x[1], tcur, geomdata, prob_parm, uinterp);
      }
    }
    s_ext[VELX] = time_factor * factor*(U_jet[0]+uinterp[0]) + (1-factor) * U_air[0];
    s_ext[VELY] = time_factor * factor*(U_jet[1]+uinterp[1]) + (1-factor) * U_air[1];
    s_ext[VELZ] = time_factor * factor*(U_jet[2]+uinterp[2]) + (1-factor) * U_air[2];
  
  }

#if defined(PELE_USE_AUX) && (NUMAUX>0)
  if (idir == jet_dir and sgn == 1) {

#if defined(PELE_USE_MIXF) && (NUMMIXF > 0)
    amrex::Real spec_Bilger_fact[NUM_SPECIES] = {0.0};
    amrex::Real Zfu = 0.0;
    amrex::Real Zox = 0.0;
    for (int n = 0; n < NUM_SPECIES; ++n) {
      spec_Bilger_fact[n] = 0.0;
      for (int m = 0; m < 4; m++) {
        spec_Bilger_fact[n] +=
        Beta_mix[m] * (ecompCHON[n * 4 + m] * atwCHON[m] / mwt[n]);
      }
      Zfu += spec_Bilger_fact[n] * Y_jet[n];
      Zox += spec_Bilger_fact[n] * Y_air[n];

      //amrex::Print() << "spec_Bilger_fact[: " << n << "] = " << spec_Bilger_fact[n] << std::endl;
    }
    //amrex::Print() << "Zfu: " << Zfu << ", Zox: " << Zox << ", rho: " << s_ext[DENSITY] << std::endl;

    amrex::Real mixfrac = 0.0;
    for (int n = 0; n < NUM_SPECIES; n++) {
      mixfrac += s_ext[FIRSTSPEC+n]/s_ext[DENSITY] * spec_Bilger_fact[n];
      }
    mixfrac = (mixfrac - Zox) / (Zfu - Zox);

    s_ext[MIXF] = mixfrac * s_ext[DENSITY];
    if (s_ext[MIXF] < 0) s_ext[MIXF] = 0;
#if (NUMMIXF > 1)
    s_ext[MIXF+1] = (1 - mixfrac) * s_ext[DENSITY];
    if (s_ext[MIXF+1] < 0) s_ext[MIXF+1] = 0;
#endif // #if (NUMMIXF > 1)
#endif // #if (NUMMIXF > 0)

#if defined(PELE_USE_AGE) && (NUMAGE > 0)
    for (int n = 0; n < NUMAGE; n++) {
      s_ext[AGE+n] = 0.0;
    }
#endif

#if defined(PELE_USE_AGE) && (NUMAGEPV > 0)
    for (int n = 0; n < NUMAGEPV; n++) {
      s_ext[AGEPV+n] = 0.0;
    }
#endif
  } // if (idir == inlet direction ...)

#endif

} // end bcnormal

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
zero_visc(int i, int j, int k,
          amrex::Array4<amrex::Real> const& beta,
          amrex::GeometryData const& geomdata,
          amrex::Box const& dBox,
          const int dir,
          const int beta_comp,
          const int nComp)
{
  //amrex::ignore_unused(
  //  i, j, k, beta, geomdata, domainBox, dir, beta_comp, nComp);
  // We treat species when beta_comp == 0 and nComp == NUM_SPECIES
  // otherwise this routine could be called for other face diffusivity (Temp,
  // velocity, ...)
  //
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx      = geomdata.CellSize();
  const amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
  const amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
  const amrex::Real z = prob_lo[2] + k*dx[2];

  int jet_dir = 2;
  int njet = 2;

  amrex::Vector<amrex::Real> jet_xs = {0., 0.};
  amrex::Vector<amrex::Real> jet_ys = {-1.0E-3, +1.0E-3};
  amrex::Real jet_rad = 2.5E-4;

  int is_in_jet = 0;
  if (dir == jet_dir and k == 0) {


    for(int l=0; l<njet; l++) {
      amrex::Real r = std::sqrt((std::pow(x - jet_xs[l],2)+std::pow(y - jet_ys[l],2)));
      if (r < jet_rad) {
        is_in_jet = 1;
      }
    }

    // Set diffusion coefficient to 0 to mimic a Neumann condition
    if (is_in_jet == 0){
      for (int n=0; n < nComp; n++) {
        beta(i,j,k,n) = 0.;
      }
    }
  }
}
#endif
