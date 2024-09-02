#include <PeleLMeX.H>
#include <PeleLMeX_K.H>
#ifdef PELE_USE_EFIELD
#include <PeleLMeX_EF_Constants.H>
#endif

using namespace amrex;

void
PeleLM::advanceChemistry(std::unique_ptr<AdvanceAdvData>& advData)
{
  BL_PROFILE("PeleLMeX::advanceChemistry()");

  for (int lev = finest_level; lev >= 0; --lev) {
    if (lev != finest_level) {
      advanceChemistryBAChem(lev, m_dt, advData->Forcing[lev]);
    } else {
      // If we defined a new BA for chem on finest level, use that instead of
      // the default one
      if (m_max_grid_size_chem.min() > 0) {
        advanceChemistryBAChem(lev, m_dt, advData->Forcing[lev]);
      } else {
        advanceChemistry(lev, m_dt, advData->Forcing[lev]);
      }
    }
  }
}

// This advanceChemistry is called on the finest level
// It works with the AmrCore BoxArray and do not involve ParallelCopy
void
PeleLM::advanceChemistry(int lev, const Real& a_dt, MultiFab& a_extForcing)
{
  BL_PROFILE("PeleLMeX::advanceChemistry_Lev" + std::to_string(lev) + "()");

  auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
  auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
  auto* ldataR_p = getLevelDataReactPtr(lev);

  // Setup EB-covered cells mask
  iMultiFab mask(grids[lev], dmap[lev], 1, 0);
#ifdef AMREX_USE_EB
  getCoveredIMask(lev, mask);
#else
  mask.setVal(1);
#endif

  MFItInfo mfi_info;
  if (Gpu::notInLaunchRegion()) {
    mfi_info.EnableTiling().SetDynamic(true);
  }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(ldataNew_p->state, mfi_info); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    auto const& rhoY_o = ldataOld_p->state.const_array(mfi, FIRSTSPEC);
    auto const& rhoH_o = ldataOld_p->state.const_array(mfi, RHOH);
    auto const& temp_o = ldataOld_p->state.const_array(mfi, TEMP);
    auto const& rhoY_n = ldataNew_p->state.array(mfi, FIRSTSPEC);
    auto const& rhoH_n = ldataNew_p->state.array(mfi, RHOH);
    auto const& temp_n = ldataNew_p->state.array(mfi, TEMP);
    auto const& extF_rhoY = a_extForcing.array(mfi, 0);
    auto const& extF_rhoH = a_extForcing.array(mfi, NUM_SPECIES);
    auto const& fcl = ldataR_p->functC.array(mfi);
    auto const& mask_arr = mask.array(mfi);

    // Reset new to old and convert MKS -> CGS
    ParallelFor(
      bx, [rhoY_o, rhoH_o, temp_o, rhoY_n, rhoH_n, temp_n, extF_rhoY,
           extF_rhoH] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        for (int n = 0; n < NUM_SPECIES; n++) {
          rhoY_n(i, j, k, n) = rhoY_o(i, j, k, n) * 1.0e-3;
          extF_rhoY(i, j, k, n) *= 1.0e-3;
        }
        temp_n(i, j, k) = temp_o(i, j, k);
        rhoH_n(i, j, k) = rhoH_o(i, j, k) * 10.0;
        extF_rhoH(i, j, k) *= 10.0;
      });

    // Reset new to old for auxiliary variables and convert MKS -> CGS
#if (NUMAUX > 0)
    auto const& rhoAux_o = ldataOld_p->state.array(mfi, FIRSTAUX);
    auto const& rhoAux_n = ldataNew_p->state.array(mfi, FIRSTAUX);
    auto const& extF_rhoAux = a_extForcing.array(mfi, NUM_SPECIES+1);
    ParallelFor(
      bx, [rhoAux_o, rhoAux_n, extF_rhoAux]
        AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#if (NUMMIXF > 0)
        for (int n = 0; n < NUMMIXF; n++) {
          rhoAux_n(i,j,k,MIXF_IN_AUX+n) = rhoAux_o(i,j,k,MIXF_IN_AUX+n) * 1.0e-3;
          extF_rhoAux(i, j, k, MIXF_IN_AUX+n) *= 1.0e-3;
        }
#if (NUMAGE > 0)
        for (int n = 0; n < NUMAGE; n++) {
          rhoAux_n(i,j,k,AGE_IN_AUX+n) = rhoAux_o(i,j,k,AGE_IN_AUX+n) * 1.0e-3;
          extF_rhoAux(i, j, k, AGE_IN_AUX+n) *= 1.0e-3;
        }
#endif // #if (NUMAGE > 0)
#if (NUMAGEPV > 0)
        for (int n = 0; n < NUMAGEPV; n++) {
          rhoAux_n(i,j,k,AGEPV_IN_AUX+n) = rhoAux_o(i,j,k,AGEPV_IN_AUX+n) * 1.0e-3;
          extF_rhoAux(i, j, k, AGEPV_IN_AUX+n) *= 1.0e-3;
        }
#endif // #if (NUMAGEPV > 0)
#endif // #if (NUMMIX > 0)
      }); // ParallelFor
#endif // if (NUMAUX > 0)

#ifdef PELE_USE_EFIELD
    // Pass nE -> rhoY_e & FnE -> FrhoY_e
    auto const& nE_o = ldataOld_p->state.const_array(mfi, NE);
    auto const& FnE = a_extForcing.array(mfi, NUM_SPECIES + 1);
    auto const& rhoYe_n = ldataNew_p->state.array(mfi, FIRSTSPEC + E_ID);
    auto const& FrhoYe = a_extForcing.array(mfi, E_ID);
    auto eos = pele::physics::PhysicsType::eos();
    Real mwt[NUM_SPECIES] = {0.0};
    eos.molecular_weight(mwt);
    ParallelFor(
      bx, [mwt, nE_o, FnE, rhoYe_n,
           FrhoYe] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        rhoYe_n(i, j, k) = nE_o(i, j, k) / Na * mwt[E_ID] * 1.0e-6;
        FrhoYe(i, j, k) = FnE(i, j, k) / Na * mwt[E_ID] * 1.0e-6;
      });
#endif

    Real dt_incr = a_dt;
    Real time_chem = 0;
    /* Solve */
    m_reactor->react(
      bx, rhoY_n, extF_rhoY, temp_n, rhoH_n, extF_rhoH,
#if (NUMAUX > 0)
      rhoAux_n, extF_rhoAux,
#endif
      fcl, mask_arr, dt_incr, time_chem
#ifdef AMREX_USE_GPU
      ,
      amrex::Gpu::gpuStream()
#endif
    );

    // Convert CGS -> MKS
    ParallelFor(
      bx, [rhoY_n, rhoH_n, extF_rhoY,
           extF_rhoH] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        for (int n = 0; n < NUM_SPECIES; n++) {
          rhoY_n(i, j, k, n) *= 1.0e3;
          extF_rhoY(i, j, k, n) *= 1.0e3;
        }
        rhoH_n(i, j, k) *= 0.1;
        extF_rhoH(i, j, k) *= 0.1;
      });

    // Convert CGS -> MKS for auxiliary variables
#if (NUMAUX > 0)
    ParallelFor(
      bx, [rhoAux_n, extF_rhoAux]
        AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#if (NUMMIXF > 0)
        for (int n = 0; n < NUMMIXF; n++) {
          rhoAux_n(i,j,k,MIXF_IN_AUX+n) *= 1.0e3;
          extF_rhoAux(i, j, k, MIXF_IN_AUX+n) *= 1.0e3;
        }
#if (NUMAGE > 0)
        for (int n = 0; n < NUMAGE; n++) {
          rhoAux_n(i,j,k,AGE_IN_AUX+n) *= 1.0e3;
          extF_rhoAux(i, j, k, AGE_IN_AUX+n) *= 1.0e3;
        }
#endif // #if (NUMAGE > 0)
#if (NUMAGEPV > 0)
        for (int n = 0; n < NUMAGEPV; n++) {
          rhoAux_n(i,j,k,AGEPV_IN_AUX+n) *= 1.0e3;
          extF_rhoAux(i, j, k, AGEPV_IN_AUX+n) *= 1.0e3;
        }
#endif // #if (NUMAGEPV > 0)
#endif // #if (NUMMIX > 0)
      }); // ParallelFor
#endif // if (NUMAUX > 0)

#ifdef PELE_USE_EFIELD
    // rhoY_e -> nE and set rhoY_e to zero
    auto const& nE_n = ldataNew_p->state.array(mfi, NE);
    Real invmwt[NUM_SPECIES] = {0.0};
    eos.inv_molecular_weight(invmwt);
    ParallelFor(
      bx, [invmwt, nE_n, rhoYe_n,
           extF_rhoY] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        nE_n(i, j, k) = rhoYe_n(i, j, k) * Na * invmwt[E_ID] * 1.0e3;
        rhoYe_n(i, j, k) = 0.0;
        extF_rhoY(i, j, k, E_ID) = 0.0;
      });
#endif

#ifdef AMREX_USE_GPU
    Gpu::Device::streamSynchronize();
#endif
  }

  // Set reaction term
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(ldataNew_p->state, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const Box& bx = mfi.tilebox();
    auto const& rhoY_o = ldataOld_p->state.const_array(mfi, FIRSTSPEC);
    auto const& rhoY_n = ldataNew_p->state.const_array(mfi, FIRSTSPEC);
    auto const& extF_rhoY = a_extForcing.const_array(mfi, 0);
    auto const& rhoYdot = ldataR_p->I_R.array(mfi, 0);
    Real dt_inv = 1.0 / a_dt;
    ParallelFor(
      bx, NUM_SPECIES,
      [rhoY_o, rhoY_n, extF_rhoY, rhoYdot,
       dt_inv] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
        rhoYdot(i, j, k, n) =
          -(rhoY_o(i, j, k, n) - rhoY_n(i, j, k, n)) * dt_inv -
          extF_rhoY(i, j, k, n);
      });

#ifdef PELE_USE_EFIELD
    auto const& nE_o = ldataOld_p->state.const_array(mfi, NE);
    auto const& nE_n = ldataNew_p->state.const_array(mfi, NE);
    auto const& FnE = a_extForcing.const_array(mfi, NUM_SPECIES + 1);
    auto const& nEdot = ldataR_p->I_R.array(mfi, NUM_SPECIES);
    ParallelFor(
      bx, [nE_o, nE_n, FnE, nEdot,
           dt_inv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        nEdot(i, j, k) =
          -(nE_o(i, j, k) - nE_n(i, j, k)) * dt_inv - FnE(i, j, k);
      });
#endif
  }
}

// This advanceChemistry works with BoxArrays built such that each box
// is either covered or uncovered and chem. integrator is called only
// on uncovered boxes.
void
PeleLM::advanceChemistryBAChem(
  int lev, const Real& a_dt, MultiFab& a_extForcing)
{
  BL_PROFILE("PeleLMeX::advanceChemistry_Lev" + std::to_string(lev) + "()");

  auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
  auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
  auto* ldataR_p = getLevelDataReactPtr(lev);

  // Set chemistry MFs based on baChem and dmapChem
  MultiFab chemState(*m_baChem[lev], *m_dmapChem[lev], NUM_SPECIES + 3 + NUMAUX, 0);
  MultiFab chemForcing(*m_baChem[lev], *m_dmapChem[lev], nCompForcing(), 0);
  MultiFab functC(*m_baChem[lev], *m_dmapChem[lev], 1, 0);
#ifdef PELE_USE_EFIELD
  MultiFab chemnE(*m_baChem[lev], *m_dmapChem[lev], 1, 0);
#endif

  // Setup EB covered cells mask
  iMultiFab mask(*m_baChem[lev], *m_dmapChem[lev], 1, 0);
#ifdef AMREX_USE_EB
  getCoveredIMask(lev, mask);
#else
  mask.setVal(1);
#endif

  // ParallelCopy into chem MFs
  chemState.ParallelCopy(ldataOld_p->state, FIRSTSPEC, 0, NUM_SPECIES + 3 + NUMAUX);
  chemForcing.ParallelCopy(a_extForcing, 0, 0, nCompForcing());
#ifdef PELE_USE_EFIELD
  chemnE.ParallelCopy(ldataOld_p->state, NE, 0, 1);
#endif

  MFItInfo mfi_info;
  if (Gpu::notInLaunchRegion()) {
    mfi_info.EnableTiling().SetDynamic(true);
  }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(chemState, mfi_info); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    auto const& rhoY_o = chemState.array(mfi, 0);
    auto const& rhoH_o = chemState.array(mfi, NUM_SPECIES);
    auto const& temp_o = chemState.array(mfi, NUM_SPECIES + 1);
    auto const& extF_rhoY = chemForcing.array(mfi, 0);
    auto const& extF_rhoH = chemForcing.array(mfi, NUM_SPECIES);
    auto const& fcl = functC.array(mfi);
    auto const& mask_arr = mask.array(mfi);

    // Convert MKS -> CGS
    ParallelFor(
      bx, [rhoY_o, rhoH_o, extF_rhoY, extF_rhoH]
        AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        for (int n = 0; n < NUM_SPECIES; n++) {
          rhoY_o(i, j, k, n) *= 1.0e-3;
          extF_rhoY(i, j, k, n) *= 1.0e-3;
        }
        rhoH_o(i, j, k) *= 10.0;
        extF_rhoH(i, j, k) *= 10.0;
      });

    // Convert MKS -> CGS for auxiliary variables
#if (NUMAUX > 0)
    auto const& rhoAux_o = chemState.array(mfi, NUM_SPECIES + 3);
    auto const& extF_rhoAux = chemForcing.array(mfi, NUM_SPECIES + 1);
    ParallelFor(
      bx, [rhoAux_o, extF_rhoAux]
        AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#if (NUMMIXF > 0)
        for (int n = 0; n < NUMMIXF; n++) {
          rhoAux_o(i, j, k, MIXF_IN_AUX+n) *= 1.0e-3;
          extF_rhoAux(i, j, k, MIXF_IN_AUX+n) *= 1.0e-3;
        }
#if (NUMAGE > 0)
        for (int n = 0; n < NUMAGE; n++) {
          rhoAux_o(i, j, k, AGE_IN_AUX+n) *= 1.0e-3;
          extF_rhoAux(i, j, k, AGE_IN_AUX+n) *= 1.0e-3;
        }
#endif // #if (NUMAGE > 0)
#if (NUMAGEPV > 0)
        for (int n = 0; n < NUMAGEPV; n++) {
          rhoAux_o(i, j, k, AGEPV_IN_AUX+n) *= 1.0e-3;
          extF_rhoAux(i, j, k, AGEPV_IN_AUX+n) *= 1.0e-3;
        }
#endif // #if (NUMAGEPV > 0)
#endif // #if (NUMMIX > 0)
      }); // ParallelFor
#endif // #if (NUM_AUX > 0)

#ifdef PELE_USE_EFIELD
    // Pass nE -> rhoY_e & FnE -> FrhoY_e
    auto const& nE_o = chemnE.array(mfi);
    auto const& FnE = chemForcing.array(mfi, NUM_SPECIES + 1);
    auto const& rhoYe_o = chemState.array(mfi, E_ID);
    auto const& FrhoYe = chemForcing.array(mfi, E_ID);
    auto eos = pele::physics::PhysicsType::eos();
    Real mwt[NUM_SPECIES] = {0.0};
    eos.molecular_weight(mwt);
    ParallelFor(
      bx, [mwt, nE_o, FnE, rhoYe_o,
           FrhoYe] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        rhoYe_o(i, j, k) = nE_o(i, j, k) / Na * mwt[E_ID] * 1.0e-6;
        FrhoYe(i, j, k) = FnE(i, j, k) / Na * mwt[E_ID] * 1.0e-6;
      });
#endif

    // Do reaction only on uncovered box
    int do_reactionBox = m_baChemFlag[lev][mfi.index()];

    if (do_reactionBox != 0) {
      // Do reaction as usual using PelePhysics chemistry integrator
      Real dt_incr = a_dt;
      Real time_chem = 0;
      /* Solve */
      m_reactor->react(
        bx, rhoY_o, extF_rhoY, temp_o, rhoH_o, extF_rhoH,
#if (NUMAUX > 0)
        rhoAux_o, extF_rhoAux,
#endif
        fcl, mask_arr, dt_incr, time_chem
#ifdef AMREX_USE_GPU
        ,
        amrex::Gpu::gpuStream()
#endif
      );
    } else {
      // Just set the function call to 0.0
      ParallelFor(bx, [fcl] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        fcl(i, j, k) = 0.0;
      });
    }

    // Convert CGS -> MKS
    ParallelFor(
      bx, [rhoY_o, rhoH_o] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        for (int n = 0; n < NUM_SPECIES; n++) {
          rhoY_o(i, j, k, n) *= 1.0e3;
        }
        rhoH_o(i, j, k) *= 0.1;
      });

    // Convert CGS -> MKS for auxiliary variables
#if (NUMAUX > 0)
    ParallelFor(
      bx, [rhoAux_o, extF_rhoAux]
        AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
#if (NUMMIXF > 0)
        for (int n = 0; n < NUMMIXF; n++) {
          rhoAux_o(i, j, k, MIXF_IN_AUX+n) *= 1.0e3;
          extF_rhoAux(i, j, k, MIXF_IN_AUX+n) *= 1.0e3;
        }
#if (NUMAGE > 0)
        for (int n = 0; n < NUMAGE; n++) {
          rhoAux_o(i, j, k, AGE_IN_AUX+n) *= 1.0e3;
          extF_rhoAux(i, j, k, AGE_IN_AUX+n) *= 1.0e3;
        }
#endif // #if (NUMAGE > 0)
#if (NUMAGEPV > 0)
        for (int n = 0; n < NUMAGEPV; n++) {
          rhoAux_o(i, j, k, AGEPV_IN_AUX+n) *= 1.0e3;
          extF_rhoAux(i, j, k, AGEPV_IN_AUX+n) *= 1.0e3;
        }
#endif // #if (NUMAGEPV > 0)
#endif // #if (NUMMIX > 0)
      }); // ParallelFor
#endif // #if (NUM_AUX > 0)


#ifdef PELE_USE_EFIELD
    // rhoY_e -> nE and set rhoY_e to zero
    Real invmwt[NUM_SPECIES] = {0.0};
    eos.inv_molecular_weight(invmwt);
    ParallelFor(
      bx,
      [invmwt, nE_o, rhoYe_o] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        nE_o(i, j, k) = rhoYe_o(i, j, k) * Na * invmwt[E_ID] * 1.0e3;
        rhoYe_o(i, j, k) = 0.0;
      });
#endif

#ifdef AMREX_USE_GPU
    Gpu::Device::streamSynchronize();
#endif
  } // mfi

  // ParallelCopy into newstate MFs
  // Get the entire new state
  MultiFab StateTemp(grids[lev], dmap[lev], NUM_SPECIES + 3, 0);
  StateTemp.ParallelCopy(chemState, 0, 0, NUM_SPECIES + 3);
  ldataR_p->functC.ParallelCopy(functC, 0, 0, 1);
#ifdef PELE_USE_EFIELD
  MultiFab nETemp(grids[lev], dmap[lev], 1, 0);
  nETemp.ParallelCopy(chemnE, 0, 0, 1);
#endif

  // Pass from temp state MF to leveldata and set reaction term
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(ldataNew_p->state, amrex::TilingIfNotGPU()); mfi.isValid();
       ++mfi) {
    const Box& bx = mfi.tilebox();
    auto const& state_arr = StateTemp.const_array(mfi);
    auto const& rhoY_o = ldataOld_p->state.const_array(mfi, FIRSTSPEC);
    auto const& rhoY_n = ldataNew_p->state.array(mfi, FIRSTSPEC);
    auto const& rhoH_n = ldataNew_p->state.array(mfi, RHOH);
    auto const& temp_n = ldataNew_p->state.array(mfi, TEMP);
    auto const& extF_rhoY = a_extForcing.const_array(mfi, 0);
    auto const& rhoYdot = ldataR_p->I_R.array(mfi, 0);
    Real dt_inv = 1.0 / a_dt;
    ParallelFor(
      bx, [state_arr, rhoY_o, rhoY_n, rhoH_n, temp_n, extF_rhoY, rhoYdot,
           dt_inv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        // Pass into leveldata_new
        for (int n = 0; n < NUM_SPECIES; n++) {
          rhoY_n(i, j, k, n) = state_arr(i, j, k, n);
        }
        rhoH_n(i, j, k) = state_arr(i, j, k, NUM_SPECIES);
        temp_n(i, j, k) = state_arr(i, j, k, NUM_SPECIES + 1);
        // Compute I_R
        for (int n = 0; n < NUM_SPECIES; n++) {
          rhoYdot(i, j, k, n) =
            -(rhoY_o(i, j, k, n) - rhoY_n(i, j, k, n)) * dt_inv -
            extF_rhoY(i, j, k, n);
        }
      });

#ifdef PELE_USE_EFIELD
    auto const& nE_arr = nETemp.const_array(mfi);
    auto const& nE_o = ldataOld_p->state.const_array(mfi, NE);
    auto const& nE_n = ldataNew_p->state.array(mfi, NE);
    auto const& FnE = a_extForcing.const_array(mfi, NUM_SPECIES + 1);
    auto const& nEdot = ldataR_p->I_R.array(mfi, NUM_SPECIES);
    ParallelFor(
      bx, [nE_arr, nE_o, nE_n, FnE, nEdot,
           dt_inv] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        // Pass into leveldata_new
        nE_n(i, j, k) = nE_arr(i, j, k);
        // Compute I_R
        nEdot(i, j, k) =
          -(nE_o(i, j, k) - nE_n(i, j, k)) * dt_inv - FnE(i, j, k);
      });
#endif
  }
}

void
PeleLM::computeInstantaneousReactionRate(
  const Vector<MultiFab*>& I_R, const TimeStamp& a_time)
{
  for (int lev = 0; lev <= finest_level; ++lev) {
#ifdef PELE_USE_EFIELD
    computeInstantaneousReactionRateEF(lev, a_time, I_R[lev]);
#else
    computeInstantaneousReactionRate(lev, a_time, I_R[lev]);
#endif
  }
}

void
PeleLM::computeInstantaneousReactionRate(
  int lev, const TimeStamp& a_time, MultiFab* a_I_R)
{
  BL_PROFILE("PeleLMeX::computeInstantaneousReactionRate()");
  auto* ldata_p = getLevelDataPtr(lev, a_time);

#ifdef AMREX_USE_EB
  auto const& ebfact = EBFactory(lev);
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(ldata_p->state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
    const Box& bx = mfi.tilebox();
    auto const& rhoY = ldata_p->state.const_array(mfi, FIRSTSPEC);
    auto const& rhoH = ldata_p->state.const_array(mfi, RHOH);
    auto const& T = ldata_p->state.const_array(mfi, TEMP);
    auto const& rhoYdot = a_I_R->array(mfi);

#ifdef AMREX_USE_EB
    auto const& flagfab = ebfact.getMultiEBCellFlagFab()[mfi];
    auto const& flag = flagfab.const_array();
    if (flagfab.getType(bx) == FabType::covered) { // Covered boxes
      amrex::ParallelFor(
        bx, NUM_SPECIES,
        [rhoYdot] AMREX_GPU_DEVICE(int i, int j, int k, int n) noexcept {
          rhoYdot(i, j, k, n) = 0.0;
        });
    } else if (flagfab.getType(bx) != FabType::regular) { // EB containing boxes
      amrex::ParallelFor(
        bx, [rhoY, rhoH, T, rhoYdot,
             flag] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          if (flag(i, j, k).isCovered()) {
            for (int n = 0; n < NUM_SPECIES; n++) {
              rhoYdot(i, j, k, n) = 0.0;
            }
          } else {
            reactionRateRhoY(i, j, k, rhoY, rhoH, T, rhoYdot);
          }
        });
    } else
#endif
    {
      amrex::ParallelFor(
        bx, [rhoY, rhoH, T,
             rhoYdot] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          reactionRateRhoY(i, j, k, rhoY, rhoH, T, rhoYdot);
        });
    }
  }
}

void
PeleLM::getScalarReactForce(
  std::unique_ptr<AdvanceAdvData>& advData,
  std::unique_ptr<AdvanceDiffData>& diffData)
{
  // The differentialDiffusionUpdate just provided the {np1,kp1} AD state
  // -> use it to build the external forcing for the chemistry
  for (int lev = 0; lev <= finest_level; ++lev) {

    // Get t^{n} t^{np1} data pointer
    auto* ldataOld_p = getLevelDataPtr(lev, AmrOldTime);
    auto* ldataNew_p = getLevelDataPtr(lev, AmrNewTime);
    auto* ldataR_p = getLevelDataReactPtr(lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(advData->Forcing[lev], TilingIfNotGPU()); mfi.isValid();
         ++mfi) {
      const Box& bx = mfi.tilebox();
      auto const& rhoY_o = ldataOld_p->state.const_array(mfi, FIRSTSPEC);
      auto const& rhoH_o = ldataOld_p->state.const_array(mfi, RHOH);
      auto const& rhoY_n = ldataNew_p->state.const_array(mfi, FIRSTSPEC);
      auto const& rhoH_n = ldataNew_p->state.const_array(mfi, RHOH);
      auto const& react = ldataR_p->I_R.const_array(mfi, 0);
      auto const& extF_rhoY = advData->Forcing[lev].array(mfi, 0);
      auto const& extF_rhoH = advData->Forcing[lev].array(mfi, NUM_SPECIES);

      auto const& a_of_s = advData->AofS[lev].const_array(mfi, 0);
      auto const& dn = diffData->Dn[lev].const_array(mfi, 0);
      auto const& dnp1 = diffData->Dnp1[lev].const_array(mfi, 0);
      auto const& fAux = advData->Forcing[lev].array(mfi, NUM_SPECIES+1);
      auto const& new_arr = ldataNew_p->state.array(mfi, 0);
      auto const& old_arr = ldataOld_p->state.array(mfi, 0);

      amrex::Real dtinv = 1.0 / m_dt;
      amrex::Real Zox_lcl = Zox;
      amrex::Real Zfu_lcl = Zfu;
      amrex::GpuArray<amrex::Real, NUM_SPECIES> fact_Bilger;
      for (int n = 0; n < NUM_SPECIES; ++n) {
        fact_Bilger[n] = spec_Bilger_fact[n];
      }

      amrex::ParallelFor(
        bx, [rhoY_o, rhoH_o, rhoY_n, rhoH_n, react, extF_rhoY, extF_rhoH,
             a_of_s, fAux, dn, dnp1, new_arr, old_arr, fact_Bilger, Zox_lcl, Zfu_lcl,
             dtinv, dt = m_dt] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          for (int n = 0; n < NUM_SPECIES; n++) {
            extF_rhoY(i, j, k, n) =
              (rhoY_n(i, j, k, n) - rhoY_o(i, j, k, n)) * dtinv -
              react(i, j, k, n);
          }
          extF_rhoH(i, j, k) = (rhoH_n(i, j, k) - rhoH_o(i, j, k)) * dtinv;

#if (defined PELE_USE_AUX) && (NUMAUX > 0)
#if (defined PELE_USE_MIXF) && (NUMMIXF > 0)
          amrex::Real rhs_mixf = 0.0;
          amrex::Real rhs_age = 0.0;
          for (int m = 0; m < NUM_SPECIES; m++) {
            rhs_mixf += 0.5 * (dn(i, j, k, m) + dnp1(i, j, k, m)) *
              fact_Bilger[m] / (Zfu_lcl - Zox_lcl);
          }
          fAux(i, j, k, 0) = a_of_s(i, j, k, MIXF) + 1.0 * rhs_mixf;
          new_arr(i, j, k, MIXF+0) = old_arr(i, j, k, MIXF+0)
            + dt * fAux(i, j, k, 0);
#if (NUMMIXF > 1)
          //fAux(i, j, k, 1) = a_of_s(i, j, k, MIXF+1) - 1.0 * rhs_mixf;
#endif
        }); // ParallelFor
#endif
#endif
    }
  }
}

void
PeleLM::getHeatRelease(int a_lev, MultiFab* a_HR)
{
  auto* ldataNew_p = getLevelDataPtr(a_lev, AmrNewTime);
  auto* ldataR_p = getLevelDataReactPtr(a_lev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  {
    for (MFIter mfi(*a_HR, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
      const Box& bx = mfi.tilebox();
      FArrayBox EnthFab(bx, NUM_SPECIES, The_Async_Arena());
      auto const& react = ldataR_p->I_R.const_array(mfi, 0);
      auto const& T = ldataNew_p->state.const_array(mfi, TEMP);
      auto const& Hi = EnthFab.array();
      auto const& HRR = a_HR->array(mfi);
      amrex::ParallelFor(
        bx, [T, Hi, HRR, react] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
          getHGivenT(i, j, k, T, Hi);
          HRR(i, j, k) = 0.0;
          for (int n = 0; n < NUM_SPECIES; n++) {
            HRR(i, j, k) -= Hi(i, j, k, n) * react(i, j, k, n);
          }
        });
    }
  }
}
