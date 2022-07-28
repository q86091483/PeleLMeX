#include "SprayParticles.H"
#include <AMReX_Particles.H>
#include "pelelm_prob.H"

using namespace amrex;

bool
SprayParticleContainer::injectParticles(Real time,
                                        Real dt,
                                        int nstep,
                                        int lev,
                                        int finest_level,
                                        ProbParm &prob_parm)
{
  if (lev != 0)
      return false;

  if (time < prob_parm.jet_start_time || time > prob_parm.jet_end_time) {
      return false;
  }

  // Only the IO processor will do injection.
  if ( amrex::ParallelDescriptor::MyProc() == 0) {

      Real Pi_six = M_PI / 6.;

      // Get fuel species physical data
      const int pstateVel = m_sprayIndx.pstateVel;
      const int pstateT = m_sprayIndx.pstateT;
      const int pstateDia = m_sprayIndx.pstateDia;
      const int pstateY = m_sprayIndx.pstateY;
      const SprayData* fdat = m_sprayData;
      Real rho_part = fdat->rho[0];

      // Check how much mass we need to inject, adding leftover from last injection
      Real mass_flow_rate = prob_parm.mass_flow_rate;
      Real injection_mass = prob_parm.floating_injection_mass + mass_flow_rate * dt;

      // Check if we can inject the last particle we draw
      if ( prob_parm.part_dia_saved > 0.0 ) { 
          Real part_mass = Pi_six * rho_part * std::pow(prob_parm.part_dia_saved, 3);
          if ( injection_mass < part_mass ) { // We can't do it, just add to mass counter and get out
              prob_parm.floating_injection_mass += mass_flow_rate * dt;
              return true;
          }
      }

      // Geometry data
      const Geometry& geom = this->m_gdb->Geom(lev);
      const auto plo = geom.ProbLoArray();
      const auto phi = geom.ProbHiArray();
      const auto dx = geom.CellSize();
      RealVect dom_len(AMREX_D_DECL(geom.ProbLength(0),
                                    geom.ProbLength(1),
                                    geom.ProbLength(2)));
      AMREX_D_TERM(const Real splitx = plo[0] + 0.5 * dom_len[0];,
                   const Real splity = plo[1] + 0.5 * dom_len[1];,
                   const Real splitz = plo[2] + 0.5 * dom_len[2]);

      //#######################################################
      // Based a Sanjose et al. 2011
      //#######################################################

      // Pressure-swirl has a core of air and a swirling fuel around it.
      // Compute the ratio of air area to total injector area
      // Using Rizk & Levebre 1985
      Real theta = prob_parm.ps_halfangle * M_PI / 180.0;
      Real ratio = ( 1.0 - std::cos(theta)*std::cos(theta) ) / ( 1.0 + std::cos(theta)*std::cos(theta) );
      Real r0 = prob_parm.ps_r0;
      Real ra = std::sqrt(ratio*r0*r0);
      Real r_mean = 0.5 * ( r0 + ra );

      // Also get extremas angles of the cone
      Real theta_min = std::abs(std::atan(std::tan(theta) * 2.0 * ra / r_mean));
      Real theta_max = std::abs(std::atan(std::tan(theta) * 2.0 * r0 / r_mean));

      // Get axial speed from mass conservation
      Real jet_vel = mass_flow_rate / ( rho_part * M_PI * r0 * r0 * (1.0 - ratio));

      // This absolutely must be included with any injection or insertion
      // function or significant issues will arise
      if (jet_vel * dt / dx[0] > 0.5) {
        Real max_vel = dx[0] * 0.5 / dt;
        if (ParallelDescriptor::IOProcessor()) {
          std::string warn_msg =
            "Injection velocity of " + std::to_string(jet_vel) +
            " is reduced to maximum " + std::to_string(max_vel);
          amrex::Warning(warn_msg);
        }
        m_injectVel = jet_vel;
        jet_vel = max_vel;
      }


      // Injection on the x- domain face. Get Area/Length of injection region
      Real jetArea = M_PI*r0*r0;

      // Particle data
      const Real num_ppp = fdat->num_ppp;
      Real part_temp = prob_parm.part_temp;

      // Create a distribution
      /*
      std::unique_ptr<DistBase> dropDist; 
      std::string dist_type = "Weibull";
      dropDist = DistBase::create(dist_type);
      dropDist->init("RosinRammler");
      */

      // Host container
      amrex::ParticleLocData pld;  
      std::map<std::pair<int, int>, amrex::Gpu::HostVector<ParticleType>> host_particles;

      // Injection center
      RealVect cur_jet_cent {AMREX_D_DECL(plo[0],splity,splitz)};  // Center of x- face

      // Inject mass until we have the desired amount
      amrex::Real total_mass = 0.;
      while (total_mass < injection_mass) {

          // Pick a random radial location and get corresponding half angle
          // See Sanjose et al. for geometrical relation between real injector plane and injection
          // location plane -> here we assume they are the same, so inp_p_alpha is ~ 0.0
          Real inj_p_rad_random = amrex::Random();
          Real inj_p_rad = ra + inj_p_rad_random * (r0 - ra);
          Real inj_p_theta = theta_min + inj_p_rad_random * (theta_max - theta_min);
          Real inj_p_alpha = std::asin( inj_p_rad / inj_p_rad );  // Odd ...

          // Get the random position on the azimuth [0:2*pi]
          Real inj_p_angle_random = amrex::Random() * 2.0 * M_PI;
          RealVect part_loc(AMREX_D_DECL(cur_jet_cent[0],
                                         cur_jet_cent[1] + std::cos(inj_p_angle_random) * inj_p_rad,
                                         cur_jet_cent[2] + std::sin(inj_p_angle_random) * inj_p_rad));

          ParticleType p;
          p.id() = ParticleType::NextID();
          p.cpu() = ParallelDescriptor::MyProc();

          // Particle velocity in local cylindrical coordinates
          AMREX_D_TERM(Real ux_vel = jet_vel;,
                       Real ur_vel = std::cos(inj_p_alpha) * std::tan(inj_p_theta) * jet_vel;,
                       Real ut_vel = std::sin(inj_p_alpha) * std::tan(inj_p_theta) * jet_vel);
 
          // Particle velocity in cartesian coordinates
          AMREX_D_TERM(Real x_vel = jet_vel;,
                       Real y_vel = -std::sin(inj_p_angle_random) * ut_vel + std::cos(inj_p_angle_random) * ur_vel;,
                       Real z_vel =  std::cos(inj_p_angle_random) * ut_vel + std::sin(inj_p_angle_random) * ur_vel);
          
          // Particle velocity rms
          AMREX_D_TERM(Real x_vel_rms = jet_vel * (2.0 * amrex::Random() - 1.0) * prob_parm.ps_rmsvel ;,
                       Real y_vel_rms = jet_vel * (2.0 * amrex::Random() - 1.0) * prob_parm.ps_rmsvel ;,
                       Real z_vel_rms = jet_vel * (2.0 * amrex::Random() - 1.0) * prob_parm.ps_rmsvel );

          // Move into RealVect
          RealVect part_vel(AMREX_D_DECL(x_vel+x_vel_rms,
                                         y_vel+y_vel_rms,
                                         z_vel+z_vel_rms));

          // Set up a new particle
          AMREX_D_TERM(p.rdata(pstateVel) = part_vel[0];,
                       p.rdata(pstateVel + 1) = part_vel[1];,
                       p.rdata(pstateVel + 2) = part_vel[2];);

          // Get a particle size
          Real cur_dia = 0.0;
          if ( prob_parm.part_dia_saved > 0.0 ) { // We have a stored particles, use it
             cur_dia = prob_parm.part_dia_saved;
             prob_parm.part_dia_saved = -1.0;
          } else {    // Draw a particle
             cur_dia = 0.000035;
          }

          // Add particles as if they have advanced some random portion of dt
          Real pmov = amrex::Random();
          for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            p.pos(dir) = part_loc[dir] + pmov * dt * part_vel[dir];
          }

          p.rdata(pstateT) = part_temp;
          p.rdata(pstateDia) = cur_dia;
          for (int sp = 0; sp < SPRAY_FUEL_NUM; ++sp) {
              p.rdata(pstateY + sp) = 1.0;
          }
          
          // Put particle in place
          bool where = Where(p, pld);
          if (!where) {
              amrex::Abort("Bad injection particle");
          }
          std::pair<int, int> ind(pld.m_grid, pld.m_tile);
         
          host_particles[ind].push_back(p);

          Real pmass = Pi_six * rho_part * std::pow(cur_dia, 3);
          total_mass += num_ppp * pmass;
      }
      
      // Move particles to level holder
      for (auto& kv : host_particles) {
        auto grid = kv.first.first;
        auto tile = kv.first.second;
        const auto& src_tile = kv.second;
        auto& dst_tile = GetParticles(lev)[std::make_pair(grid, tile)];
        auto old_size = dst_tile.GetArrayOfStructs().size();
        auto new_size = old_size + src_tile.size();
        dst_tile.resize(new_size);
        // Copy the AoS part of the host particles to the GPU
        amrex::Gpu::copy( amrex::Gpu::hostToDevice, src_tile.begin(), src_tile.end(),
                          dst_tile.GetArrayOfStructs().begin() + old_size);
      }
  }

  // Redistribute is done outside of this function
  return true;
}

void
SprayParticleContainer::InitSprayParticles(ProbParm const& prob_parm)
{
  // This ensures the initial time step size stays reasonable
  // Use a rought guess of the ratio
  const SprayData* fdat = m_sprayData;
  Real rho_part = fdat->rho[0];
  Real jet_vel = prob_parm.mass_flow_rate / ( rho_part * M_PI * prob_parm.ps_r0 * prob_parm.ps_r0 * (1.0 - 0.2));
  m_injectVel = jet_vel;
  // Start without any particles
  return;
}
