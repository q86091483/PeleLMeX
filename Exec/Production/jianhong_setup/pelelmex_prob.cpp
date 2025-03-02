#include <PeleLMeX.H>
#include <AMReX_ParmParse.H>

void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  std::string type;

  // Definition of the Geometry
  pp.query("Xmax", prob_parm->Xmax);
  pp.query("Xf", prob_parm->Xf);
  pp.query("Xe", prob_parm->Xe);
  pp.query("Xc", prob_parm->Xc);
  pp.query("Ymax", prob_parm->Ymax);
  pp.query("Zmax", prob_parm->Zmax);

  // Definition of the Inflow conditions
  pp.query("Yin", prob_parm->Yin);

  pp.query("V_fu", prob_parm->V_fu);
  pp.query("V_ox", prob_parm->V_ox);
  pp.query("V_init", prob_parm->V_init);
  pp.query("D_fu", prob_parm->D_fu);
//  pp.query("V_air", prob_parm->V_air);
  pp.query("V_obst", prob_parm->V_obst);
  pp.query("ml_thickness", prob_parm->ml_thickness);
  

  pp.query("T_fu", prob_parm->T_fu);
  pp.query("T_ox", prob_parm->T_ox);
//  pp.query("T_air", prob_parm->T_air);
  pp.query("T_obst", prob_parm->T_obst);

  pp.query("X_O2", prob_parm->X_O2);

  // Definition of the Initial conditions
  pp.query("P_mean", prob_parm->P_mean);

  // Definition of the Ignition zone
  pp.query("do_ignition", prob_parm->do_ignition);
  pp.query("ign_rad", prob_parm->ign_rad);
  pp.query("ign_T", prob_parm->ign_T);
  pp.query("ign_thick", prob_parm->ign_thick);

  // Definition of the fuel parameters
  pp.query("dilution", prob_parm->dilution);

#ifdef PELE_USE_EFIELD
  pp.query("PhiV_y_hi", PeleLM::prob_parm->phiV_hiy);
  pp.query("PhiV_y_lo", PeleLM::prob_parm->phiV_loy);
#endif

  PeleLM::prob_parm->fuelID = H2_ID;  
  PeleLM::pmf_data.initialize();

}
