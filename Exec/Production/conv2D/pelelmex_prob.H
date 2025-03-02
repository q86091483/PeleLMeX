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
#include <math.h>

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
void getTurbVel(amrex::Real t, amrex::Real x, amrex::Real y, amrex::Real z,
                amrex::GeometryData const& geomdata,
                ProbParm const& prob_parm,
                amrex::Vector<amrex::Real>& uinterp)
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
		if (prob_parm.do_turbInlet == 1) {
    	mod[idim] = prob_parm.Lxin - std::fmod(prob_parm.V_cf * t, prob_parm.Lxin);
    	mod[idim] = std::fmod(x - prob_lo[0] + mod[idim], prob_parm.Lxin) + prob_parm.xin[0];
		} else if (prob_parm.do_turbInlet == 2) {
      mod[idim] = x;
    }
    locate(prob_parm.xin, prob_parm.nxin, mod[idim], idx[idim]);
    idxp1[idim] = (idx[idim] + 1) % prob_parm.nxin;
    slp[idim] = (mod[idim] - prob_parm.xin[idx[idim]]) / prob_parm.dxin[idx[idim]];
		//amrex::AllPrint() << "Here: " << mod[idim] << ", " << idxp1[idim] << ", " << slp[idim] << std::endl;
		
    // y
    idim = 1;
    mod[idim] = y;
    locate(prob_parm.yin, prob_parm.nyin, mod[idim], idx[idim]);
    idxp1[idim] = (idx[idim] + 1) % prob_parm.nyin;
    slp[idim] = (mod[idim] - prob_parm.yin[idx[idim]]) / prob_parm.dyin[idx[idim]];
    // z
    idim = 2;
    mod[idim] = z;
    locate(prob_parm.zin, prob_parm.nzin, mod[idim], idx[idim]);
    idxp1[idim] = (idx[idim] + 1) % prob_parm.nzin;
    idxp1[idim] = idx[idim] + 1;
    slp[idim] = (mod[idim] - prob_parm.zin[idx[idim]]) / prob_parm.dzin[idx[idim]];

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
  int myproc = amrex::ParallelDescriptor::MyProc();
  int nprocs = amrex::ParallelDescriptor::NProcs();

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

  auto eos = pele::physics::PhysicsType::eos();

  // Mixture fraction
 	amrex::Real atwCHON[4] = {0.0};
  pele::physics::eos::atomic_weightsCHON<pele::physics::PhysicsType::eos_type>(atwCHON);
  for (int n = 0; n < 4; n++) {
    amrex::Print() << "atwCHON[" <<n << "]: " << atwCHON[n] << std::endl;
  }
  int ecompCHON[NUM_SPECIES * 4];
  pele::physics::eos::element_compositionCHON<
    pele::physics::PhysicsType::eos_type>(ecompCHON);
  amrex::Print() << "Number of atm (C,H,O,N) in species (molecular weight):" << std::endl;
  amrex::Real mwt[NUM_SPECIES];
  eos.molecular_weight(mwt);
  for (int n = 0; n < NUM_SPECIES; ++n) {
    for (int m = 0; m < 4; m++) {
      amrex::Print() << ecompCHON[n*4+m] << ", ";
    }
    amrex::Print() << mwt[n];
    amrex::Print() << std::endl;
  }
  amrex::Real Beta_mix[4] = {0.0};
  Beta_mix[0] = (atwCHON[0] != 0.0) ? 2.0 / atwCHON[0] : 0.0;
  Beta_mix[1] = (atwCHON[1] != 0.0) ? 1.0 / (2.0 * atwCHON[1]) : 0.0;
  Beta_mix[2] = (atwCHON[2] != 0.0) ? -1.0 / atwCHON[2] : 0.0;
  Beta_mix[3] = 0.0;

  if (prob_parm.do_turbInlet == 1) {
    amrex::Vector<amrex::Real> uinterp(3);
    amrex::Real tcur = 0.0;
    //getTurbVel(tcur, x, y, z, geomdata, prob_parm, uinterp);
    //state(i,j,k,VELX) = uinterp[0];
    //state(i,j,k,VELY) = uinterp[1];
    //state(i,j,k,VELZ) = uinterp[2];
  }

  // Read table
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
  pele::physics::PMF::pmf(pmf_data, x, x, pmf_vals);

  // Declaration - stream  variables
	amrex::Real XF[NUM_SPECIES] = {0.0};
  amrex::Real XO[NUM_SPECIES] = {0.0};
  amrex::Real YF[NUM_SPECIES] = {0.0};
  amrex::Real YO[NUM_SPECIES] = {0.0};
  amrex::Real TF, TO;

  // Declaration - local variables
  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real rho_cgs, Hmix, Hmix_cgs, Tmix;
  amrex::Real HF_cgs, HO_cgs, P_cgs;
  amrex::Real x_velocity, y_velocity, z_velocity;

  // Stream pressure 
  P_cgs = prob_parm.P_mean * 10.0;

  // Stream Y & X
	XO[O2_ID] = 0.21;
  XO[N2_ID] = 0.79;
  eos.X2Y(XO, YO);

	XF[H2_ID] = 1.0;
  XF[N2_ID] = 0.0;
  eos.X2Y(XF, YF);

	// Stream T
	TF = 300.;
  TO = 300.;

  // Stream H
  eos.TY2H(TF, YF, HF_cgs);
  eos.TY2H(TO, YO, HO_cgs);

  // Stream U & V
  AMREX_D_TERM(const amrex::Real UF = 25.0;
               , const amrex::Real VF = 0.0;
               , const amrex::Real WF = 0.0);
  AMREX_D_TERM(const amrex::Real UO = 25.0;
               , const amrex::Real VO = 0.0;
               , const amrex::Real WO = 0.0);

  // Local Z
  amrex::Real mixf = 0.0; 
  mixf = 0.5 *
             (1.0 - tanh((y - 0.0) / prob_parm.bl_thickness));


	// Local Y 
  for (int n = 0; n < NUM_SPECIES; n++) {
	  massfrac[n] = mixf * YF[n] + (1 - mixf) * YO[n];
  }
  eos.Y2X(massfrac, molefrac);

	// Local H
	Hmix_cgs = mixf * HF_cgs + (1 - mixf) * HO_cgs;
  Hmix = Hmix_cgs * 1.0e-4;

  // Local T
  eos.HY2T(Hmix_cgs, massfrac, Tmix);

	// Local rho
	eos.PYT2R(P_cgs, massfrac, Tmix, rho_cgs);

  // Local rhoH
  eos.TY2H(Tmix, massfrac, Hmix_cgs);

  // Local u, v, w
#if (AMREX_SPACEDIM == 2)
  x_velocity = mixf * UF + (1 - mixf) * UO;
  y_velocity = mixf * VF + (1 - mixf) * VO;
#elif (AMREX_SPACEDIM == 3) 
  x_velocity = mixf * UF + (1 - mixf) * UO;
  y_velocity = mixf * VF + (1 - mixf) * VO;
  z_velocity = mixf * WF + (1 - mixf) * WO;
#endif

  // Set variables
#if (AMREX_SPACEDIM == 2)
  state(i, j, k, VELX) = x_velocity;
  state(i, j, k, VELY) = y_velocity;
#elif (AMREX_SPACEDIM == 3) 
  state(i, j, k, VELX) = x_velocity;
  state(i, j, k, VELY) = y_velocity;
  state(i, j, k, VELZ) = z_velocity;
  //amrex::Print() << "u, v, w" << x_velocity << ", " << y_velocity << ", " << z_velocity << std::endl;
#endif

  state(i, j, k, DENSITY) = rho_cgs * 1.0e3;
	//amrex::Print() << "DENSITY: " << state(i, j, k, DENSITY) << std::endl;

  for (int n = 0; n < NUM_SPECIES; n++) {
    state(i, j, k, FIRSTSPEC+n) = massfrac[n] * state(i, j, k, DENSITY);
		//amrex::Print() << "Y: " << state(i, j, k, FIRSTSPEC+n) / state(i,j,k,DENSITY) << std::endl;
  }

  state(i, j, k, TEMP) = Tmix;
	//amrex::Print() << "TEMPERATURE: " << state(i, j, k, TEMP) << std::endl;

  state(i, j, k, RHOH) = Hmix * state(i, j, k, DENSITY);
	//amrex::Print() << "RHOH: " << state(i, j, k, RHOH) << std::endl;

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
    if (NUMMIXF > 1) {
			state(i, j, k, MIXF+1) = state(i,j,k,DENSITY) * (1 - mixfrac);
    }
  #endif
	#if defined(PELE_USE_AGE) && (NUMAGE > 0)
		for (int n = 0; n < NUMAGE; n++) {
			state(i, j, k, AGE+n) = 0.0; 
		}
  #endif
#endif

	//amrex::Print() << "FIRSTAUX " << FIRSTAUX << std::endl;
	//amrex::Print() << "NUMAUX " << NUMAUX << std::endl;
	//amrex::Print() << "MIXF " << MIXF << std::endl;
	//amrex::Print() << "NUMMIXF " << NUMMIXF << std::endl;
	//amrex::Print() << "AGE " << AGE << std::endl;
	//amrex::Print() << "NUMAGE " << NUMAGE << std::endl;
	//amrex::Print() << "AGEPV " << AGEPV << std::endl;
	//amrex::Print() << "NUMAGEPV " << NUMAGEPV << std::endl;
	//amrex::Print() << "NVAR " << NVAR << std::endl;

  //if (amrex::ParallelDescriptor::IOProcessor()) {
  //  amrex::Print() << "Finished init." << std::endl;
  //}
  //if (massfrac[H2_ID]>1.0) {
  //  amrex::AllPrint() << "YH2" << massfrac[H2_ID] << std::endl;
  //}
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

  auto eos = pele::physics::PhysicsType::eos();
	
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};

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

  // Declaration - stream  variables
	amrex::Real XF[NUM_SPECIES] = {0.0};
  amrex::Real XO[NUM_SPECIES] = {0.0};
  amrex::Real YF[NUM_SPECIES] = {0.0};
  amrex::Real YO[NUM_SPECIES] = {0.0};
  amrex::Real TF, TO;

  // Declaration - local variables
  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real rho_cgs, Hmix, Hmix_cgs, Tmix;
  amrex::Real HF_cgs, HO_cgs, P_cgs;
  amrex::Real x_velocity, y_velocity, z_velocity;

  // Stream pressure 
  P_cgs = prob_parm.P_mean * 10.0;

  // Stream Y & X
	XO[O2_ID] = 0.21;
  XO[N2_ID] = 0.79;
  eos.X2Y(XO, YO);

	XF[H2_ID] = 1.0;
  XF[N2_ID] = 0.0;
  eos.X2Y(XF, YF);

	// Stream T
	TF = 300.;
  TO = 300.;

  // Stream H
  eos.TY2H(TF, YF, HF_cgs);
  eos.TY2H(TO, YO, HO_cgs);

  // Stream U & V
  AMREX_D_TERM(const amrex::Real UF = 25.0;
               , const amrex::Real VF = 0.0;
               , const amrex::Real WF = 0.0);
  AMREX_D_TERM(const amrex::Real UO = 25.0;
               , const amrex::Real VO = 0.0;
               , const amrex::Real WO = 0.0);

  // Local Z
  amrex::Real mixf = 0.0; 
  mixf = 0.5 *
             (1.0 - tanh((x[1] - 0.0) / prob_parm.bl_thickness));

	// Local Y 
  for (int n = 0; n < NUM_SPECIES; n++) {
	  massfrac[n] = mixf * YF[n] + (1 - mixf) * YO[n];
  }
  eos.Y2X(massfrac, molefrac);

	// Local H
	Hmix_cgs = mixf * HF_cgs + (1 - mixf) * HO_cgs;
  Hmix = Hmix_cgs * 1.0e-4;

  // Local T
  eos.HY2T(Hmix_cgs, massfrac, Tmix);

	// Local rho
	eos.PYT2R(P_cgs, massfrac, Tmix, rho_cgs);

  // Local rhoH
  eos.TY2H(Tmix, massfrac, Hmix_cgs);

  // Local u, v, w
#if (AMREX_SPACEDIM == 2)
  x_velocity = mixf * UF + (1 - mixf) * UO;
  y_velocity = mixf * VF + (1 - mixf) * VO;
#elif (AMREX_SPACEDIM == 3) 
  x_velocity = mixf * UF + (1 - mixf) * UO;
  y_velocity = mixf * VF + (1 - mixf) * VO;
  z_velocity = mixf * WF + (1 - mixf) * WO;
#endif

  // Set variables
  if ((idir == 0) and (sgn == 1)) {

    // Density
    s_ext[DENSITY] = rho_cgs * 1.0e3;

    // Mass fraction
    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
    }

    // Temperature
    s_ext[TEMP] = Tmix;

    //amrex::Real RhoH_temp;
    // Enthalpy
    //eos.TY2H(s_ext[TEMP], Y_air, RhoH_temp);
    s_ext[RHOH] = Hmix * 1.0e-4 * s_ext[DENSITY]; //RhoH_temp * 1.0e-4 * s_ext[DENSITY];
  	//amrex::Print() << "RhoH : " <<s_ext[RHOH] << std::endl;

    // Velocity
#if (AMREX_SPACEDIM == 2)
  	s_ext[VELX] = x_velocity;
  	s_ext(VELY) = y_velocity;
#elif (AMREX_SPACEDIM == 3) 
  	s_ext[VELX] = x_velocity;
  	s_ext[VELY] = y_velocity;
  	s_ext[VELZ] = z_velocity;
#endif

#if defined(PELE_USE_AUX) && (NUMAUX>0)
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

  		//amrex::Print() << "spec_Bilger_fact[: " << n << "] = " << spec_Bilger_fact[n] << std::endl;
  	}
  	//amrex::Print() << "Zfu: " << Zfu << ", Zox: " << Zox << std::endl;

		amrex::Real mixfrac = 0.0;
    for (int n = 0; n < NUM_SPECIES; n++) {
			mixfrac += massfrac[n] * spec_Bilger_fact[n];
    }
		mixfrac = (mixfrac - Zox) / (Zfu - Zox);

		s_ext[MIXF] = mixfrac * s_ext[DENSITY]; 
    if (NUMMIXF > 1) {
			s_ext[MIXF+1] = s_ext[DENSITY] * (1 - mixfrac);
    }
  #endif
	#if defined(PELE_USE_AGE) && (NUMAGE > 0)
		for (int n = 0; n < NUMAGE; n++) {
			s_ext[AGE+n] = 0.0; 
		}
  #endif
#endif

  } // if idir==0 and sgn=+1

  if ((idir == 1) and (sgn == -1)) {
		s_ext[TEMP] = 300.;
  } // if upper wall

  if ((idir == 1) and (sgn == +1)) {
		s_ext[TEMP] = 300.;
  } // if lower wall


}

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
  //const amrex::Real* prob_lo = geomdata.ProbLo();
  //const amrex::Real* dx      = geomdata.CellSize();
  //const amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
  //const amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
  //const amrex::Real z = prob_lo[2] + k*dx[2];

  //int jet_dir = 2;
  //int njet = 2;


  //const amrex::Real jet_rad = 2.25E-4;
  //amrex::Vector<amrex::Real> jet_xs = {0., 0.};
  //amrex::Vector<amrex::Real> jet_ys = {-(3.0+1.0)*jet_rad, +(3.0+1.0)*jet_rad};

  //int is_in_jet = 0;
  //if (dir == jet_dir and k == 0) {


  //  for(int l=0; l<njet; l++) {
  //    amrex::Real r = std::sqrt((std::pow(x - jet_xs[l],2)+std::pow(y - jet_ys[l],2)));
  //    if (r < jet_rad) {
  //      is_in_jet = 1;
  //    }
  //  }

    // Set diffusion coefficient to 0 to mimic a Neumann condition
  //  if (is_in_jet == 0){
  //    for (int n=0; n < nComp; n++) {
  //      beta(i,j,k,n) = 0.;
  //    }
  //  }
  //}
}
#endif
