#ifndef PELELM_PROB_H
#define PELELM_PROB_H

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_SPACE.H>

#include <PeleLMeX_ProblemSpecificFunctions.H>
#include <PeleLMeX_Index.H>
#include <PMF.H>
#include <PMFData.H>
#include <PelePhysics.H>
#include <math.h>

struct MyProbParm : public ProbParmDefault
{
  amrex::Real P_mean = 1013250.0; 	
  amrex::Real V_j = 80.5;
  amrex::Real V_cf = 25.0;
  amrex::Real jet_rad = 1.0e-3;
  amrex::Real jet_temp = 400.0;
  amrex::Real global_eq_ratio = 0.6;
  amrex::Real ox_temp = 750.;
  amrex::Real X_O2 = 2.1e-1;
  amrex::Real X_N2 = 7.9e-1;
  amrex::Real pertmag_cf = 0.0;
  amrex::Real pertmag_jet = 0.0;
  amrex::Real jet_purity = 1.0;
  amrex::Real bl_thickness = 4.5E-5;

  amrex::Real init_time = 1.0e-6;
  int double_jet = 0;

  int jet_dir = 2;
  int cf_dir = 0;
  int other_dir = 1;

  // Turbulent channel inlet at the low end of x direction
  int do_turbInlet = 0;
  int nxin = 0;  // Stretched coordinate
  int nyin = 0;  // uniform
  int nzin = 0;  // uniform
  amrex::Real Lxin = 0.0;
  amrex::Real Lyin = 0.0;
  amrex::Real Lzin = 0.0;
  amrex::Real *uin = nullptr;
  amrex::Real *vin = nullptr;
  amrex::Real *win = nullptr;
  amrex::Real *xin = nullptr;
  amrex::Real *yin = nullptr;
  amrex::Real *zin = nullptr;
  amrex::Real *dxin = nullptr;
  amrex::Real *dyin = nullptr;
  amrex::Real *dzin = nullptr;
};

struct MyProblemSpecificFunctions : public DefaultProblemSpecificFunctions
{

	AMREX_GPU_DEVICE
	AMREX_FORCE_INLINE
	static void initdata(
  	int i,
  	int j,
  	int k,
  	int is_incompressible,
  	amrex::Array4<amrex::Real> const& state,
  	amrex::Array4<amrex::Real> const& aux,
  	amrex::GeometryData const& geomdata,
  	MyProbParm const& prob_parm,
  	pele::physics::PMF::PmfData::DataContainer const* pmf_data)
	{
  	//int myproc = amrex::ParallelDescriptor::MyProc();
  	//int nprocs = amrex::ParallelDescriptor::NProcs();

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

  	XF[H2_ID] = 1.0; //prob_parm.jet_purity;
  	XF[N2_ID] = 0.0; //1-prob_parm.jet_purity;
  	eos.X2Y(XF, YF);

  	XO[O2_ID] = prob_parm.X_O2;
  	XO[N2_ID] = prob_parm.X_N2;
  	eos.X2Y(XO, YO);

		// Initial T
  	state(i, j, k, TEMP) = prob_parm.ox_temp;

  	// Initial Velocity
  	amrex::Real x_velocity = 37.5;
  	amrex::Real y_velocity = 0.0;
  	amrex::Real z_velocity = 0.0;
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

  	// Initialize mixture fraction and other fields by Holderman's model (PECS1993)
		bool use_holderman_model = false;
		if (use_holderman_model) {
  		amrex::Real mf;
  		amrex::Real xc, H, b, zc, wp, wn, theta_mix, theta_c, theta_norm, theta_ref,
              cn, rcn, cp, rcp, zdis, bot;
  		xc = x + 4 * prob_parm.jet_rad;
  		H = prob_hi[prob_parm.jet_dir] - prob_lo[prob_parm.jet_dir];
  		b = 0.002 * pow(xc/H, 0.1);
  		zc = H * 0.35 * pow(xc/H, 0.45) * exp(-b);
  		wp = H * 0.58 * pow(xc/H, 0.38);
  		wn = H * 0.12 * pow(xc/H, 0.25);
  		theta_mix = 0.0180444;
  		if ((xc/H) < 1.0) {
    		theta_c = 1 - 0.2 * pow(xc/H, 2);
  		} else{
    		theta_c = theta_mix + (1-theta_mix) * pow(0.6/(xc/H), 0.5);
  		}
  		cn = 0.2 * pow(xc/H, -0.6) * exp(4.0 * pow(xc/H, 2.0));
  		rcn = 1 - exp(-cn);
  		cp = 0.5 * pow(xc/H, 1.0);
  		rcp = 1 - exp(-cp);
  		zdis = z - zc;
  		mf = 0.0;
  		if (xc > 0) {
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
     		mf = mf * 0.075;
		 		//mf = 0.0;
      }

  		// Read table
  		amrex::GpuArray<amrex::Real, NUM_SPECIES + 4> pmf_vals = {0.0};
  		pele::physics::PMF::pmf(pmf_data, mf, mf, pmf_vals);

  		for (int n = 0; n < NUM_SPECIES; n++) {
      	massfrac[n] = pmf_vals[3+n];
  		}
  		state(i,j,k,TEMP) = pmf_vals[0];

  		eos.PYT2R(P_cgs, massfrac, state(i,j,k,TEMP), rho_cgs);
  		state(i,j,k,DENSITY) = rho_cgs * 1.0e3;

  		eos.TY2H(state(i,j,k,TEMP), massfrac, h_cgs);
  		state(i,j,k,RHOH) = h_cgs * 1.0e-4 * state(i,j,k,DENSITY);

  		for (int n = 0; n < NUM_SPECIES; n++) {
    		state(i,j,k,FIRSTSPEC+n) = massfrac[n] * state(i,j,k,DENSITY);
  		}
  	} // end if use_holderman_model

		// Auxiliaray field
#if defined(PELE_USE_AUX) && (NUMAUX > 0)
	#if defined(PELE_USE_MIXF) && (NUMMIXF > 0)
	  amrex::Real mixfrac = 0.0;
		state(i, j, k, MIXF) = mixfrac * state(i, j, k, DENSITY);
    if (NUMMIXF > 1) {
			state(i, j, k, MIXF+1) = state(i,j,k,DENSITY) * (1 - mixfrac);
    }
  #endif

	#if defined(PELE_USE_AGE) && (NUMAGE > 0)
		for (int n = 0; n < NUMAGE; n++) {
			state(i, j, k, AGE+n) = 0.0;
		}
    state(i, j, k, AGE+1) = ((x - prob_lo[0]) / 37.5) * state(i, j, k, MIXF+1);
  #endif

	#if defined(PELE_USE_AGEPV) && (NUMAGEPV > 0)
		for (int n = 0; n < NUMAGEPV; n++) {
			state(i, j, k, AGEPV+n) = 0.0;
		}
    state(i, j, k, AGEPV+1) = ((x - prob_lo[0]) / 37.5) * state(i, j, k, MIXF+1);
  #endif
#endif

	} // init_data

	AMREX_GPU_DEVICE
	AMREX_FORCE_INLINE
	static void bcnormal(
  	const amrex::Real x[AMREX_SPACEDIM],
  	const amrex::CellData<amrex::Real>& /*s_in*/,
  	amrex::Real s_ext[NVAR],
  	const int idir,
  	const int sgn,
  	const amrex::Real time,
  	amrex::GeometryData const& geomdata,
  	MyProbParm const& prob_parm,
  	pele::physics::PMF::PmfData::DataContainer const* /*pmf_data*/)
	{
  	const amrex::Real* prob_lo = geomdata.ProbLo();
  	const amrex::Real* prob_hi = geomdata.ProbHi();
  	const amrex::Real* dx = geomdata.CellSize();
  	constexpr amrex::Real Pi = 3.14159265358979323846264338327950288;

  	auto eos = pele::physics::PhysicsType::eos();

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
  	amrex::Real radsq = 0.0, rad = 0.0;
  	amrex::Real factor = 0.0;
  	amrex::Real area = 0.0;
  	amrex::Real jet_vel = 0.0;

  	X_air[O2_ID] = prob_parm.X_O2;
  	X_air[N2_ID] = 1-prob_parm.X_O2;
  	X_jet[H2_ID] = 1.0; //prob_parm.jet_purity;
  	X_jet[N2_ID] = 0.0; //1.0 - prob_parm.jet_purity;

  	eos.X2Y(X_air, Y_air);
  	eos.X2Y(X_jet, Y_jet);

  	eos.PYT2R(prob_parm.P_mean * 10.0, Y_air, prob_parm.ox_temp, rho_air);
  	eos.PYT2R(prob_parm.P_mean * 10.0, Y_jet, prob_parm.jet_temp, rho_jet);
  	rho_air *= 1.0e3; // CGS -> MKS conversion
  	rho_jet *= 1.0e3;

	#if (AMREX_SPACEDIM == 2)
  	amrex::Abort("JIC not implemented in 2D");
	#elif (AMREX_SPACEDIM == 3)
  	area = (prob_hi[other_dir] - prob_lo[other_dir]) *
         	 (prob_hi[jet_dir] - prob_lo[jet_dir]);
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
  	amrex::Vector<amrex::Real> jet_ys = {-(3.0+1.0)*jet_rad, +(3.0+1.0)*jet_rad};
  	amrex::Real radsq_min = 1E30;

  	if (idir == jet_dir and (prob_parm.double_jet == 1 or sgn == 1)) {

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
             (1.0 - tanh((sqrt(radsq) - jet_rad) / prob_parm.bl_thickness));

    	// Y
    	for (int n = 0; n < NUM_SPECIES; n++) {
      	massfrac[n] = factor * Y_jet[n] + (1.0 - factor) * Y_air[n];
    	}
    	eos.Y2X(massfrac, molefrac);

    	// H
    	eos.TY2H(prob_parm.jet_temp, Y_jet, Hjet_cgs);
    	eos.TY2H(prob_parm.ox_temp,  Y_air, Hair_cgs);
    	Hmix_cgs = Hjet_cgs * factor + Hair_cgs * (1-factor);

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
    	s_ext[VELX] = 0.0;
    	s_ext[VELX] =
      	(1 - 0.5 * prob_parm.double_jet) * sgn * time_factor * factor * U_jet[0];
    	s_ext[VELY] = 0.0;
    	s_ext[VELY] =
      	(1 - 0.5 * prob_parm.double_jet) * sgn * time_factor * factor * U_jet[1];
    	s_ext[VELZ] = 0.0;
    	s_ext[VELZ] =
      	(1 - 0.5 * prob_parm.double_jet) * sgn * time_factor * factor * U_jet[2];

    	int myproc = amrex::ParallelDescriptor::MyProc();
    	if (rad <= 2.0*jet_rad and
           (isnan(s_ext[RHOH]) or isnan(s_ext[TEMP]) or isnan(s_ext[DENSITY]) or
	    	isnan(s_ext[VELX]) or isnan(s_ext[VELY]) or isnan(s_ext[VELZ]))  )
    	{
				amrex::Print(myproc) << "At proc " << myproc
                           << ", (x,y,z,r) = (" << x[0] << ", " << x[1] << ", " << x[2] << ", "
                           << rad << ")" << std::endl;
    		for (int i = 0; i<NUM_SPECIES; i++) {
            amrex::Print(myproc) << "  - massfrac[" << i << "]: " << massfrac[i] << std::endl;
				}
				amrex::Print(myproc) << "  - Z        = " << factor << std::endl;
    		amrex::Print(myproc) << "  - Hmix_cgs = " << Hmix_cgs << ", "
                           << isnan(Hmix_cgs) << ", " << Hjet_cgs << ", " << Hair_cgs << std::endl;
    		amrex::Print(myproc) << "  - Tmix     = " << Tmix << std::endl;
    		amrex::Print(myproc) << "  - rho_cgs  = " << rho_cgs << std::endl;
				amrex::Print(myproc) << std::endl;
    	}
  	} // if jet inlet

  	if ((idir == cf_dir) and (sgn == 1)) {

			factor = 0.0; 

   		s_ext[DENSITY] = rho_air;

    	for (int n = 0; n < NUM_SPECIES; n++) {
      	s_ext[FIRSTSPEC + n] = Y_air[n] * s_ext[DENSITY];
    	}

    	s_ext[TEMP] = prob_parm.ox_temp;

    	amrex::Real RhoH_temp = 0.0;
    	eos.TY2H(s_ext[TEMP], Y_air, RhoH_temp);
    	s_ext[RHOH] = RhoH_temp * 1.0e-4 * s_ext[DENSITY];

    	s_ext[VELX] = U_air[0];
    	s_ext[VELY] = U_air[1];
    	s_ext[VELZ] = U_air[2];

    	//if (prob_parm.do_turbInlet > 0) {
      //	amrex::Vector<amrex::Real> uinterp(3);
      // 	amrex::Real tcur = time;
      // 	if (prob_parm.do_turbInlet == 1) {
      // 		getTurbVel(tcur, x[0], x[1], x[2], geomdata, prob_parm, uinterp);
      // 	} else if (prob_parm.do_turbInlet == 2) {
      // 		getTurbVel(tcur, tcur, x[1], x[2], geomdata, prob_parm, uinterp);
      // 	}
      // 	s_ext[VELX] = uinterp[0];
      // 	s_ext[VELY] = uinterp[1];
      //  s_ext[VELZ] = uinterp[2];
      //}
    } // if crossflow inlet

  	if ((idir == jet_dir) and (sgn == -1)) {
			s_ext[TEMP] = prob_parm.ox_temp;
  	} // if upper isothermal wall


		if ((idir == 2) and (sgn == 1) and false) {
			amrex::AllPrint() << "idir: " << idir << std::endl;
			amrex::AllPrint() << "mixf: " << factor << std::endl;
			amrex::AllPrint() << "rad: " << rad << std::endl;
			amrex::AllPrint() << "sgn: " << sgn << std::endl;
			amrex::AllPrint() << "TEMP: " << s_ext[TEMP] << std::endl;
			amrex::AllPrint() << "VELX: " << s_ext[VELX] << std::endl;
			amrex::AllPrint() << "VELY: " << s_ext[VELY] << std::endl;
			amrex::AllPrint() << "VELZ: " << s_ext[VELZ] << std::endl;
	  	amrex::AllPrint() << "RHOH: " << s_ext[RHOH] << std::endl;
	  	amrex::AllPrint() << "DEN: " << s_ext[DENSITY] << std::endl;
	  	amrex::AllPrint() << "jet_dir: " << jet_dir << std::endl;
	  	amrex::AllPrint() << "cf_dir: " << cf_dir << std::endl;
	  	amrex::AllPrint() << "other_dir: " << other_dir << std::endl;

			for (int n = 0; n < NUM_SPECIES; n++) {
				amrex::AllPrint() << "Y(" << n << "): " << s_ext[FIRSTSPEC+n] / s_ext[DENSITY] << std::endl;
    	}
		}

#if defined(PELE_USE_AUX) && (NUMAUX>0)
		if ( ((idir == cf_dir) and (sgn == 1)) or
		   ((idir == jet_dir) and (prob_parm.double_jet == 1 or sgn == 1))) {

#if defined(PELE_USE_MIXF) && (NUMMIXF > 0)
			amrex::Real mixfrac = factor;

    	if ( (idir == jet_dir) and (prob_parm.double_jet == 1 or sgn == 1) ) {
				s_ext[MIXF] = mixfrac * s_ext[DENSITY];
			} else if ((idir == cf_dir) and (sgn == 1)) {
				s_ext[MIXF] = 0.0;
    	}
#if (NUMMIXF > 1)
    	if ( (idir == jet_dir) and (prob_parm.double_jet == 1 or sgn == 1) ) {
				s_ext[MIXF+1] = (1 - mixfrac) * s_ext[DENSITY];
			} else if ((idir == cf_dir) and (sgn == 1)) {
				s_ext[MIXF+1] = (1 - mixfrac) * s_ext[DENSITY];
    	}
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

		}  // if (idir == jet_dir or cf_dir ...)
#endif // #if (NUMAUX > 0)

	} // bcnormal

	AMREX_GPU_DEVICE
	AMREX_FORCE_INLINE
	static void zero_visc(
		int i, 
		int j, 
		int k,
    amrex::Array4<amrex::Real> const& beta,
    amrex::GeometryData const& geomdata,
    amrex::Box const& dBox,
    const int dir,
    const int beta_comp,
    const int nComp,
    MyProbParm const& /*prob_parm*/)
	{
  	const amrex::Real* prob_lo = geomdata.ProbLo();
  	const amrex::Real* dx      = geomdata.CellSize();
  	const amrex::Real x = prob_lo[0] + (i+0.5)*dx[0];
  	const amrex::Real y = prob_lo[1] + (j+0.5)*dx[1];
  	const amrex::Real z = prob_lo[2] + k*dx[2];

  	int jet_dir = 2;
  	int njet = 2;

  	const amrex::Real jet_rad = 2.25E-4;
  	const amrex::Real bl_thickness = 4.5E-5;
  	amrex::Vector<amrex::Real> jet_xs = {0., 0.};
  	amrex::Vector<amrex::Real> jet_ys = {-(3.0+1.0)*jet_rad, +(3.0+1.0)*jet_rad};

  	int is_in_jet = 0;
  	if (dir == jet_dir and k == 0 and nComp == NUM_SPECIES) {

    	amrex::Real factor = 1E30;
    	for(int l=0; l<njet; l++) {
      	amrex::Real r = std::sqrt((std::pow(x - jet_xs[l],2)+std::pow(y - jet_ys[l],2)));
      	factor = std::min(factor, 0.5 * (1.0 - tanh((r - jet_rad) / bl_thickness)));
      	if (r < jet_rad) {
        	is_in_jet = 1;
      	}
    	}

    	// Set diffusion coefficient to 0 to mimic a Neumann condition
    	if (is_in_jet == 0 or true){
      	for (int n=0; n < nComp; n++) {
        	beta(i,j,k,n) = beta(i,j,k,n) * factor;
      	}
    	}
  	}
	} // end of zero_visc

};

using ProblemSpecificFunctions = MyProblemSpecificFunctions;
using ProbParm = MyProbParm;
#endif
