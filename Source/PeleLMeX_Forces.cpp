#include "AMReX_MFIter.H"
#include "PeleLMeX_Index.H"
#include "mechanism.H"
#include <PeleLMeX.H>
#include <PeleLMeX_K.H>

using namespace amrex;

// Return velocity forces scaled by rhoInv
// including grapP term if add_gradP = 1
// including divTau if input Vector not empty
void
PeleLM::getVelForces(
  const TimeStamp& a_time,
  const Vector<MultiFab*>& a_divTau,
  const Vector<MultiFab*>& a_velForce,
  int nGrowForce,
  int add_gradP)
{
  BL_PROFILE("PeleLMeX::getVelForces()");
  int has_divTau = static_cast<int>(!a_divTau.empty());

  for (int lev = 0; lev <= finest_level; ++lev) {
    if (has_divTau != 0) {
      getVelForces(a_time, lev, a_divTau[lev], a_velForce[lev], add_gradP);
    } else {
      getVelForces(a_time, lev, nullptr, a_velForce[lev], add_gradP);
    }
  }

  // FillPatch forces
  if (nGrowForce > 0) {
    fillpatch_forces(m_cur_time, a_velForce, nGrowForce);
  }
}

void
PeleLM::getVelForces(
  const TimeStamp& a_time,
  int lev,
  MultiFab* a_divTau,
  MultiFab* a_velForce,
  int add_gradP)
{

  // Get level data
  // TODO: the 1 here bypass getting halftime vel and return oldtime vel
  auto* ldata_p = getLevelDataPtr(lev, a_time, 1);

  // Get gradp: if m_t_old < 0.0, we are during initialization -> only NewTime
  // data initialized at this point
  auto* ldataGP_p = (m_t_old[lev] < 0.0) ? getLevelDataPtr(lev, AmrNewTime)
                                         : getLevelDataPtr(lev, AmrOldTime);

  Real time = getTime(lev, a_time);

  int has_divTau = static_cast<int>(a_divTau != nullptr);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(*a_velForce, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const auto& bx = mfi.tilebox();
    FArrayBox DummyFab(bx, 1);
    const auto& vel_arr = ldata_p->state.const_array(mfi, VELX);
    const auto& rho_arr = (m_incompressible) != 0
                            ? DummyFab.array()
                            : ldata_p->state.const_array(mfi, DENSITY);
    const auto& rhoY_arr = (m_incompressible) != 0
                             ? DummyFab.array()
                             : ldata_p->state.const_array(mfi, FIRSTSPEC);
    const auto& rhoh_arr = (m_incompressible) != 0
                             ? DummyFab.array()
                             : ldata_p->state.const_array(mfi, RHOH);
    const auto& temp_arr = (m_incompressible) != 0
                             ? DummyFab.array()
                             : ldata_p->state.const_array(mfi, TEMP);
    const auto& extmom_arr = m_extSource[lev]->const_array(mfi, VELX);
    const auto& extrho_arr = m_extSource[lev]->const_array(mfi, DENSITY);
    const auto& force_arr = a_velForce->array(mfi);

    // Get other forces (gravity, ...)
    getVelForces(
      lev, bx, time, force_arr, vel_arr, rho_arr, rhoY_arr, rhoh_arr, temp_arr,
      extmom_arr, extrho_arr);

#ifdef PELE_USE_EFIELD
    const auto& phiV_arr = ldata_p->state.const_array(mfi, PHIV);
    const auto& ne_arr = ldata_p->state.const_array(mfi, NE);
    addLorentzVelForces(lev, bx, time, force_arr, rhoY_arr, phiV_arr, ne_arr);
#endif

    // Add pressure gradient and viscous forces (if req.) and scale by density.
    int is_incomp = m_incompressible;
    Real incomp_rho_inv = 1.0 / m_rho;
    if ((add_gradP != 0) || (has_divTau != 0)) {
      const auto& gp_arr =
        (add_gradP) != 0 ? ldataGP_p->gp.const_array(mfi) : DummyFab.array();
      const auto& divTau_arr =
        (has_divTau) != 0 ? a_divTau->const_array(mfi) : DummyFab.array();
      amrex::ParallelFor(
        bx,
        [incomp_rho_inv, is_incomp, add_gradP, has_divTau, rho_arr, gp_arr,
         divTau_arr, force_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (is_incomp != 0) {
            for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
              if (add_gradP != 0) {
                force_arr(i, j, k, idim) -= gp_arr(i, j, k, idim);
              }
              if (has_divTau != 0) {
                force_arr(i, j, k, idim) += divTau_arr(i, j, k, idim);
              }
              force_arr(i, j, k, idim) *= incomp_rho_inv;
            }
          } else {
            for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
              if (add_gradP != 0) {
                force_arr(i, j, k, idim) -= gp_arr(i, j, k, idim);
              }
              if (has_divTau != 0) {
                force_arr(i, j, k, idim) += divTau_arr(i, j, k, idim);
              }
              force_arr(i, j, k, idim) /= rho_arr(i, j, k);
            }
          }
        });
    } else {
      amrex::ParallelFor(
        bx, [incomp_rho_inv, is_incomp, rho_arr,
             force_arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (is_incomp != 0) {
            for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
              force_arr(i, j, k, idim) *= incomp_rho_inv;
            }
          } else {
            for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {
              force_arr(i, j, k, idim) /= rho_arr(i, j, k);
            }
          }
        });
    }
  }
}

void
PeleLM::getVelForces(
  int lev,
  const Box& bx,
  const Real& a_time,
  Array4<Real> const& force,
  Array4<const Real> const& vel,
  Array4<const Real> const& rho,
  Array4<const Real> const& rhoY,
  Array4<const Real> const& rhoh,
  Array4<const Real> const& temp,
  Array4<const Real> const& extMom,
  Array4<const Real> const& extRho)
{
  const auto dx = geom[lev].CellSizeArray();

  // Get non-static info for the pseudo gravity forcing
  int pseudo_gravity = m_ctrl_pseudoGravity;
  const Real dV_control = m_ctrl_dV;

  int is_incomp = m_incompressible;
  Real rho_incomp = m_rho;

  amrex::ParallelFor(
    bx,
    [=, grav = m_gravity, gp0 = m_background_gp,
     ps_dir = m_ctrl_flameDir] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
      makeVelForce(
        i, j, k, is_incomp, rho_incomp, pseudo_gravity, ps_dir, a_time, grav,
        gp0, dV_control, dx, vel, rho, rhoY, rhoh, temp, extMom, extRho, force);
    });
}

// ZS
void PeleLM::computeSource(amrex::Vector<std::unique_ptr<amrex::MultiFab>>& source) {

  for (int lev = 0; lev <= finest_level; lev++) {

    const auto geomdata = geom[lev].data();
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();

    for (MFIter mfi(*(source[lev]), TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      auto const& extRhoH = m_extSource[lev]->const_array(mfi, RHOH); 
      amrex::ParallelFor(
        bx, 
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
          amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
          amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
          amrex::Real Lx = prob_hi[0] - prob_lo[0];
          amrex::Real Ly = prob_hi[1] - prob_lo[1];
          amrex::Real Lz = prob_hi[2] - prob_lo[2];
        }
      ); // ParallelFor
    } // mfi
    
  } // lev

} // computeSource;

void PeleLM::imposeHighT() {
  for (int lev = 0; lev <= finest_level; lev++) {
    auto* ldata_p = getLevelDataPtr(lev, AmrNewTime);
    const auto geomdata = geom[lev].data();
    const amrex::Real* prob_lo = geomdata.ProbLo();
    const amrex::Real* prob_hi = geomdata.ProbHi();
    const amrex::Real* dx = geomdata.CellSize();

    for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      auto const& state = ldata_p->state.array(mfi);
      amrex::ParallelFor(
        bx, 
        [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
          amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
          amrex::Real z = prob_lo[2] + (k + 0.5) * dx[2];
          amrex::Real Lx = prob_hi[0] - prob_lo[0];
          amrex::Real Ly = prob_hi[1] - prob_lo[1];
          amrex::Real Lz = prob_hi[2] - prob_lo[2];

          amrex::Real rad = std::sqrt(std::pow(x,2.0) + std::pow(z,2.0));
          amrex::Real rad_T = 2.0E-4;

          if (rad <= rad_T) {
            amrex::Real eta = 0.0;
            eta = 0.5 * (1.0 - tanh((rad - rad_T) / (1E-4 / 4.0)));
            
            amrex::Real massfrac[NUM_SPECIES] = {0.0};
            for (int n = 0; n < NUM_SPECIES; n++) {
              massfrac[n] = state(i,j,k,FIRSTSPEC+n) / state(i,j,k,DENSITY);
            }
            massfrac[H2_ID] = 6.86665300e-05;
            massfrac[O2_ID] = 6.65301167e-02;
            massfrac[H2O_ID] = 1.78018215e-01; 
            massfrac[H_ID] = 4.23935063e-06;
            massfrac[O_ID] = 2.75149017e-04;
            massfrac[OH_ID] = 3.49664589e-03;
            massfrac[HO2_ID] = 1.17798132e-05;
            massfrac[H2O2_ID] = 1.18889448e-06;
            massfrac[N2_ID] = 7.51593999e-01;

            state(i,j,k,TEMP) = eta*2334 + (1-eta)*state(i,j,k,TEMP);
            amrex::Real rho_cgs;
            auto eos = pele::physics::PhysicsType::eos();
            eos.PYT2R(101325*10.0*10.0, massfrac, state(i,j,k,TEMP), rho_cgs);
            state(i,j,k,DENSITY) = rho_cgs * 1.0e3;

            amrex::Real RhoH_temp;
            eos.TY2H(state(i,j,k,TEMP), massfrac, RhoH_temp);
            state(i,j,k,RHOH) = RhoH_temp * 1.0e-4 * state(i,j,k,DENSITY);

            for (int n = 0; n < NUM_SPECIES; n++) {
              state(i,j,k,FIRSTSPEC+n) = massfrac[n] * state(i,j,k,DENSITY);
            }

          } // rad <= rad_T

          } // lambda
      ); // ParallelFor
    } // mfi
  } // lev
} // imposeHighT
