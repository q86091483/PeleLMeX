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
  amrex::Real molefrac[NUM_SPECIES] = {0.0};
  amrex::Real massfrac[NUM_SPECIES] = {0.0};
  amrex::Real YF[NUM_SPECIES] = {0.0};
  amrex::Real XF[NUM_SPECIES] = {0.0};
  amrex::Real YO[NUM_SPECIES] = {0.0};
  amrex::Real XO[NUM_SPECIES] = {0.0};

  molefrac[O2_ID] = prob_parm.X_O2;
  molefrac[N2_ID] = prob_parm.X_N2;
  eos.X2Y(molefrac, massfrac);

  XF[H2_ID] = prob_parm.jet_purity;
  XF[N2_ID] = 1-prob_parm.jet_purity;
  eos.X2Y(XF, YF);

  XO[O2_ID] = prob_parm.X_O2;
  XO[N2_ID] = prob_parm.X_N2;
  eos.X2Y(XO, YO);

  state(i, j, k, TEMP) = prob_parm.ox_temp;

  // Set the initial velocity field
  amrex::Real x_velocity = 0.0;
  amrex::Real y_velocity = 0.0;
  amrex::Real z_velocity = 0.0;
  amrex::Real V_cf = prob_parm.V_cf;

  // This part of code is not used.
  //std::vector<int> dims(3);
  //std::vector<amrex::Real> x_;
  //std::vector<amrex::Real> y_;
  //std::vector<amrex::Real> z_;
  //std::vector<amrex::Real> mf_;
  //if (amrex::ParallelDescriptor::IOProcessor()) {
  //   std::ifstream fin("init_JICF_mixfrac.dat", std::ios::binary);
     // Dimensions
  //   fin.read(reinterpret_cast<char*>(&dims[0]), 3*sizeof(int));
     // Coordinates
  //   x_.resize(dims[0]);
  //   fin.read(reinterpret_cast<char*>(&x_[0]), dims[0]*sizeof(amrex::Real));
  //   y_.resize(dims[1]);
  //   fin.read(reinterpret_cast<char*>(&y_[0]), dims[1]*sizeof(amrex::Real));
  //   z_.resize(dims[2]);
  //   fin.read(reinterpret_cast<char*>(&z_[0]), dims[2]*sizeof(amrex::Real));
  //   mf_.resize(dims[0]*dims[1]*dims[2]);
  //   fin.read(reinterpret_cast<char*>(&mf_[0]), dims[0]*dims[1]*dims[2]*sizeof(amrex::Real));
  //   fin.close();
  //   amrex::Print() << dims[0] << dims [2] << std::endl;
  //}
  //amrex::ParallelDescriptor::Barrier();
  //

#if (AMREX_SPACEDIM == 2)
  x_velocity = 0.0;
  y_velocity = 0.0;
#elif (AMREX_SPACEDIM == 3)
  x_velocity = V_cf + prob_parm.pertmag_cf * V_cf *
                        (sin(2 * Pi * 4 * x / Lx) * sin(2 * Pi * 5 * y / Ly) *
                           sin(2 * Pi * 3 * z / Lz) +
                         1.017 * sin(2 * Pi * 3 * x / Lx) *
                           sin(2 * Pi * 7 * y / Ly) * sin(2 * Pi * 5 * z / Lz));
  y_velocity = prob_parm.pertmag_cf * V_cf *
               (sin(2 * Pi * 5 * x / Lx) * sin(2 * Pi * 3 * y / Ly) *
                  sin(2 * Pi * 7 * z / Lz) +
                1.017 * sin(2 * Pi * 9 * x / Lx) * sin(2 * Pi * 4 * y / Ly) *
                  sin(2 * Pi * 2 * z / Lz));
  z_velocity = prob_parm.pertmag_cf * V_cf *
               (sin(2 * Pi * 2 * x / Lx) * sin(2 * Pi * 6 * y / Ly) *
                  sin(2 * Pi * 5 * z / Lz) +
                1.017 * sin(2 * Pi * 7 * x / Lx) * sin(2 * Pi * 4 * y / Ly) *
                  sin(2 * Pi * 3 * z / Lz));
#endif

  AMREX_D_TERM(state(i, j, k, VELX) = x_velocity;
               , state(i, j, k, VELY) = y_velocity;
               , state(i, j, k, VELZ) = z_velocity);

  amrex::Real P_cgs = prob_parm.P_mean * 10.0;

  // Density
  amrex::Real rho_cgs = 0.0;
  eos.PYT2R(P_cgs, massfrac, state(i, j, k, TEMP), rho_cgs);
  state(i, j, k, DENSITY) = rho_cgs * 1.0e3;

  // Enthalpy
  amrex::Real h_cgs = 0.0;
  eos.TY2H(state(i, j, k, TEMP), massfrac, h_cgs);

  state(i, j, k, RHOH) = h_cgs * 1.0e-4 * state(i, j, k, DENSITY);

  // Species mass
  for (int n = 0; n < NUM_SPECIES; n++) {
    state(i, j, k, FIRSTSPEC + n) = massfrac[n] * state(i, j, k, DENSITY);
  }
  
  // Initialize mixture fraction field by Holderman's model (PECS1993)
  amrex::Real xc, H, b, zc, wp, wn, theta_mix, theta_c, theta_norm, theta_ref,
              cn, rcn, cp, rcp, zdis, bot;
  amrex::Real mf;
  xc = x - (-prob_parm.jet_rad - prob_parm.jet_rad);
  H = prob_hi[prob_parm.jet_dir] - prob_lo[prob_parm.jet_dir];
  b = 0.0002 * pow(xc/H, 0.1);
  zc = H * 0.15 * pow(xc/H, 0.25) * exp(-b);
  wp = H * 0.18 * pow(xc/H, 0.25);
  wn = H * 0.12 * pow(xc/H, 0.25);
  theta_mix = 0.01712; // 0.14027;
  if ((xc/H) < 1.0) {
    theta_c = 1 - 0.2 * pow(xc/H, 2);
  } else{
    theta_c = theta_mix + (1-theta_mix) * pow(0.6/(xc/H), 0.5);
  }
  cn = 0.2 * pow(xc/H, -0.6) * exp(4.0 * pow(xc/H, 2.0));
  rcn = 1 - exp(-cn);
  cp = 0.5 * pow(xc/H, 4.0);
  rcp = 1 - exp(-cp);
  zdis = z - zc;
  if (xc<=0) {
     mf = 0.0;
  } else {
     bot = -log(2.0) * pow(zdis, 2.0);
     if (zdis > 0) {
        bot = bot / pow(wp+1E-30, 2.0);
        theta_norm = exp(bot);
        theta_ref = theta_c * rcp;
     } else {
        bot = bot / pow(wn+1E-30, 2.0);
        theta_norm = exp(bot);
        theta_ref = theta_c * rcn;
     }
     mf = theta_ref + theta_norm * (theta_c - theta_ref);
     mf = mf * 0.1;
  }

  // Read table
  amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
  pele::physics::PMF::pmf(pmf_data, mf, mf, pmf_vals);

  for (int n = 0; n < NUM_SPECIES; n++) {
      massfrac[n] = pmf_vals[3+n];
  }
  state(i,j,k,TEMP) = pmf_vals[0];

  //int nbin = 30;
  //int ibin = 0;
  //amrex::Real dZ = 1 / amrex::Real(nbin);
  //amrex::Vector<amrex::Real> Tz = 
  //{ 750., 1143., 1487., 1791., 2060., 2293., 2431., 2301., 2164., 2033., 
  // 1910., 1793., 1682., 1576., 1476., 1380., 1289., 1202., 1119., 1039., 
  //  962.,  889.,  818.,  751.,  686.,  623.,  563.,  506.,  452.,  400.};
  //ibin =  floor(mf / dZ); 
  //ibin = std::min(nbin-2, ibin);
  //ibin = std::max(0, ibin);
  //amrex::Real r0, z0, z1;
  //z0 = ibin * dZ; 
  //r0 = (mf - z0) / dZ;
  //state(i,j,k,TEMP) = Tz[ibin]*(1-r0) + Tz[ibin+1]*r0;

  eos.PYT2R(P_cgs, massfrac, state(i,j,k,TEMP), rho_cgs);
  state(i,j,k,DENSITY) = rho_cgs * 1.0e3;

  eos.TY2H(state(i,j,k,TEMP), massfrac, h_cgs);
  state(i,j,k,RHOH) = h_cgs * 1.0e-4 * state(i,j,k,DENSITY);

  for (int n = 0; n < NUM_SPECIES; n++) {
    state(i,j,k,FIRSTSPEC+n) = massfrac[n] * state(i,j,k,DENSITY);
  }
  //if (amrex::ParallelDescriptor::IOProcessor()) {
  //  amrex::Print() << "Finished init." << std::endl;
  //}
  if (massfrac[H2_ID]>1.0) {
    amrex::AllPrint() << "YH2" << massfrac[H2_ID] << std::endl;
  }
  if (mf>1.0 || mf < 0.0) {
    amrex::AllPrint() << "mf " << mf << " " << massfrac[H2_ID]<< std::endl;
  }

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
  amrex::Real factor = 0.0;
  amrex::Real area = 0.0;
  amrex::Real jet_vel = 0.0;

  auto eos = pele::physics::PhysicsType::eos();

  // Initialize the jet and air molefracs
  X_air[O2_ID] = prob_parm.X_O2;
  X_air[N2_ID] = prob_parm.X_N2;
  X_jet[H2_ID] = prob_parm.jet_purity;
  X_jet[N2_ID] = 1.0 - prob_parm.jet_purity;

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
  amrex::Abort("JIC not implemented in 2D");
#elif (AMREX_SPACEDIM == 3)
  area = (prob_hi[other_dir] - prob_lo[other_dir]) *
         (prob_hi[jet_dir] - prob_lo[jet_dir]);

  //jet_vel = (prob_parm.global_eq_ratio * rho_air * area * V_cf * Y_air[O2_ID]) /
  //          (8 * Pi * jet_rad * jet_rad * rho_jet * Y_jet[H2_ID]);

  if (cf_dir == 0) {
    U_air[0] = V_cf;
  } else if (cf_dir == 1) {
    U_air[1] = V_cf;
  } else {
    U_air[2] = V_cf;
  }

  if (jet_dir == 0) {
    U_jet[0] = V_j;
  } else if (jet_dir == 1) {
    U_jet[1] = V_j;
  } else {
    U_jet[2] = V_j;
  }
#endif

  int njet = 2;
  amrex::Vector<amrex::Real> jet_xs = {0., 0.};
  amrex::Vector<amrex::Real> jet_ys = {-1.4E-3, 1.4E-3};
  amrex::Real radsq_min = 1E30;

  if (idir == jet_dir and (prob_parm.double_jet == 1 or sgn == 1)) {

    for (int i = 0; i<njet; i++) {
      radsq = pow(x[other_dir]-jet_ys[i], 2) + pow(x[cf_dir]-jet_xs[i], 2);
      radsq_min = std::min(radsq_min, radsq);
    }
    radsq = radsq_min; 

    factor =
      0.5 *
      (1.0 - tanh((sqrt(radsq) - jet_rad) / (prob_parm.bl_thickness / 4.0)));

    for (int n = 0; n < NUM_SPECIES; n++) {
      molefrac[n] = factor * X_jet[n] + (1.0 - factor) * X_air[n];
    }
    eos.X2Y(molefrac, massfrac);

    amrex::Real rho_cgs, P_cgs, RhoH_temp;
    P_cgs = prob_parm.P_mean * 10.0;

    s_ext[TEMP] =
      factor * prob_parm.jet_temp + (1.0 - factor) * prob_parm.ox_temp;

    eos.PYT2R(P_cgs, massfrac, s_ext[TEMP], rho_cgs);
    s_ext[DENSITY] = rho_cgs * 1.0e3;

    eos.TY2H(s_ext[TEMP], massfrac, RhoH_temp);
    s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY]; // CGS -> MKS conversion

    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = massfrac[n] * s_ext[DENSITY];
    }

    // Velocity
    amrex::Real time_factor = 0.0;
    if (time < prob_parm.init_time) {
      time_factor = (time / prob_parm.init_time) * (time / prob_parm.init_time);
    } else {
      time_factor = 1.0;
    }
    s_ext[VELX] =
      (1 - 0.5 * prob_parm.double_jet) * sgn * time_factor * factor * U_jet[0];
    s_ext[VELY] =
      (1 - 0.5 * prob_parm.double_jet) * sgn * time_factor * factor * U_jet[1];
    s_ext[VELZ] =
      (1 - 0.5 * prob_parm.double_jet) * sgn * time_factor * factor * U_jet[2];
  }

  if ((idir == cf_dir) and (sgn == 1)) {
    // Density
    s_ext[DENSITY] = rho_air;
    // Mass fraction
    for (int n = 0; n < NUM_SPECIES; n++) {
      s_ext[FIRSTSPEC + n] = Y_air[n] * s_ext[DENSITY];
    }
    // Temperature
    s_ext[TEMP] = prob_parm.ox_temp;
    amrex::Real RhoH_temp;
    // Enthalpy
    eos.TY2H(s_ext[TEMP], Y_air, RhoH_temp);
    s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];
    // Velocity
    s_ext[VELX] = U_air[0];
    s_ext[VELY] = U_air[1];
    s_ext[VELZ] = U_air[2];
  }
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
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx      = geomdata.CellSize();
  const amrex::Real z = prob_lo[2] + (k+0.5)*dx[2];
  const amrex::Real y = prob_lo[1] + j*dx[1];
  const amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];

  int cf_dir = 2;
  int njet = 2;

  amrex::Vector<amrex::Real> jet_xs = {0., 0.};
  amrex::Vector<amrex::Real> jet_ys = {-1.4E-3, 1.4E-3};
  amrex::Real jet_rad = 4E-4;

  int is_in_jet = 0;
  if (dir == cf_dir and k == 0) {


    for(int l=0; l<njet; l++) {
      amrex::Real r = std::sqrt((std::pow(x - jet_xs[l],2)+std::pow(z - jet_ys[l],2)));
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
