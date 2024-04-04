#include <PeleLMeX.H>
#include <PMFData.H>
#include <AMReX_ParallelDescriptor.H>

AMREX_FORCE_INLINE
std::string
read_file(std::ifstream& in) 
{
  return static_cast<std::stringstream const&>(
           std::stringstream() << in.rdbuf())
    .str();
}

void read_coord_csv(
  const std::string& iname,
  const size_t nx,
  amrex::Vector<amrex::Real>& data)
{
  std::ifstream infile(iname, std::ios::in);
  const std::string memfile = read_file(infile);
  if (not infile.is_open()) {
    amrex::Abort("Unable to open input file " + iname);
  }
  infile.close();
  std::istringstream iss(memfile);

  // Read the file
  size_t nlines = 0;
  std::string firstline;
  std::string line;
  std::getline(iss, firstline); // skip header
  while (getline(iss, line)) {
    ++nlines;
  }

  // Quick sanity check
  if (nlines != nx) {
    amrex::Abort(
      "Number of lines in the input file (= " + std::to_string(nlines) +
      ") does not match the input resolution (=" + std::to_string(nx) + ")");
  }

  // Read the data from the file
  iss.clear();
  iss.seekg(0, std::ios::beg);
  std::getline(iss, firstline); // skip header
  int cnt = 0;
  while (std::getline(iss, line)) {
    std::istringstream linestream(line);
    std::string value;
    while (getline(linestream, value, ',')) {
      std::istringstream sinput(value);
      sinput >> data[cnt];
      cnt++;
    }
  }
}


void read_csv(
  const std::string& iname,
  const size_t nx,
  const size_t ny,
  const size_t nz,
  amrex::Vector<amrex::Real>& data)
{
  std::ifstream infile(iname, std::ios::in);
  const std::string memfile = read_file(infile);
  if (not infile.is_open()) {
    amrex::Abort("Unable to open input file " + iname);
  }
  infile.close();
  std::istringstream iss(memfile);

  // Read the file
  size_t nlines = 0;
  std::string firstline;
  std::string line;
  std::getline(iss, firstline); // skip header
  while (getline(iss, line)) {
    ++nlines;
  }

  // Quick sanity check
  if (nlines != nx * ny * nz) {
    amrex::Abort(
      "Number of lines in the input file (= " + std::to_string(nlines) +
      ") does not match the input resolution (=" + std::to_string(nx) + ")");
  }

  // Read the data from the file
  iss.clear();
  iss.seekg(0, std::ios::beg);
  std::getline(iss, firstline); // skip header
  int cnt = 0;

  while (std::getline(iss, line)) {
    std::istringstream linestream(line);
    std::string value;

    while (getline(linestream, value, ',')) {
      std::istringstream sinput(value);
      sinput >> data[cnt];
      cnt++;
    }
  }
}


void
PeleLM::readProbParm()
{
  amrex::ParmParse pp("prob");

  // Read local prob_parm_l 
  ProbParm prob_parm_l;
  
  int do_turbInlet;
  pp.query("do_turbInlet", do_turbInlet);

  int nxin, nyin, nzin;
  pp.query("nxin", nxin); 
  pp.query("nyin", nyin); 
  pp.query("nzin", nzin); 

  std::string file_x, file_y, file_z, file_uvw;
  pp.query("file_x", file_x);
  pp.query("file_y", file_y);
  pp.query("file_z", file_z);
  pp.query("file_uvw", file_uvw);

  amrex::Vector<amrex::Real> xin(nxin); /* this needs to be double */
  amrex::Vector<amrex::Real> yin(nyin); 
  amrex::Vector<amrex::Real> zin(nzin); 
  amrex::Vector<amrex::Real> dxin(nxin); 
  amrex::Vector<amrex::Real> dyin(nyin); 
  amrex::Vector<amrex::Real> dzin(nzin); 
  amrex::Vector<amrex::Real> uvwin(nxin * nyin * nzin * 6); 

  read_coord_csv(file_x, nxin, xin);
  read_coord_csv(file_y, nyin, yin);
  read_coord_csv(file_z, nzin, zin);
  read_csv(file_uvw, nxin, nyin, nzin, uvwin);

	for (int i = 0; i < nxin-1; ++i) {
		dxin[i] = xin[i+1] - xin[i]; 
  }
  dxin[nxin-1] = xin[1] - xin[0];

	for (int i = 0; i < nyin-1; ++i) {
		dyin[i] = yin[i+1] - yin[i]; 
  }
  dyin[nyin-1] = yin[1] - yin[0];

	for (int i = 0; i < nzin-1; ++i) {
		dzin[i] = zin[i+1] - zin[i]; 
  }
  dzin[nzin-1] = zin[1] - zin[0]; // Note that z is the height and not periodic

  // Pass data to local prob_parm
  prob_parm_l.do_turbInlet = do_turbInlet;
  prob_parm_l.nxin = nxin;
  prob_parm_l.nyin = nyin;
  prob_parm_l.nzin = nzin;
  prob_parm_l.Lxin   = xin[nxin - 1] - xin[0];
  prob_parm_l.Lyin   = yin[nyin - 1] - yin[0];
  prob_parm_l.Lzin   = zin[nzin - 1] - zin[0];

  prob_parm_l.xin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*sizeof(amrex::Real));
  prob_parm_l.yin = (amrex::Real*) amrex::The_Arena()->alloc(nyin*sizeof(amrex::Real));
  prob_parm_l.zin = (amrex::Real*) amrex::The_Arena()->alloc(nzin*sizeof(amrex::Real));

  prob_parm_l.dxin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*sizeof(amrex::Real));
  prob_parm_l.dyin = (amrex::Real*) amrex::The_Arena()->alloc(nyin*sizeof(amrex::Real));
  prob_parm_l.dzin = (amrex::Real*) amrex::The_Arena()->alloc(nzin*sizeof(amrex::Real));

  prob_parm_l.uin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*nyin*nzin*sizeof(amrex::Real));
  prob_parm_l.vin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*nyin*nzin*sizeof(amrex::Real));
  prob_parm_l.win = (amrex::Real*) amrex::The_Arena()->alloc(nxin*nyin*nzin*sizeof(amrex::Real));
  for (int i = 0; i < nxin; ++i) {
    prob_parm_l.xin[i] = xin[i];
    prob_parm_l.dxin[i] = dxin[i];
  }
  for (int i = 0; i < nyin; ++i) {
    prob_parm_l.yin[i] = yin[i];
    prob_parm_l.dyin[i] = dyin[i];
  }
  for (int i = 0; i < nzin; ++i) {
    prob_parm_l.zin[i] = zin[i];
    prob_parm_l.dzin[i] = dzin[i];
  }
  for (int i = 0; i < nxin*nyin*nzin; ++i) {
    prob_parm_l.uin[i] = uvwin[3 + i * 6];
    prob_parm_l.vin[i] = uvwin[4 + i * 6];
    prob_parm_l.win[i] = uvwin[5 + i * 6];
  }

  // Initialize PeleLM::prob_parm container
  PeleLM::prob_parm->do_turbInlet = prob_parm_l.do_turbInlet;
  PeleLM::prob_parm->nxin = prob_parm_l.nxin;
  PeleLM::prob_parm->nyin = prob_parm_l.nyin;
  PeleLM::prob_parm->nzin = prob_parm_l.nzin;
  PeleLM::prob_parm->Lxin = prob_parm_l.Lxin;
  PeleLM::prob_parm->Lyin = prob_parm_l.Lyin;
  PeleLM::prob_parm->Lzin = prob_parm_l.Lzin;

  // Copy into PeleLM::prob_parm: CPU only for now
  PeleLM::prob_parm->xin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*sizeof(amrex::Real));
  PeleLM::prob_parm->yin = (amrex::Real*) amrex::The_Arena()->alloc(nyin*sizeof(amrex::Real));
  PeleLM::prob_parm->zin = (amrex::Real*) amrex::The_Arena()->alloc(nzin*sizeof(amrex::Real)); 
  PeleLM::prob_parm->dxin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*sizeof(amrex::Real));
  PeleLM::prob_parm->dyin = (amrex::Real*) amrex::The_Arena()->alloc(nyin*sizeof(amrex::Real));
  PeleLM::prob_parm->dzin = (amrex::Real*) amrex::The_Arena()->alloc(nzin*sizeof(amrex::Real)); 
  PeleLM::prob_parm->uin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*nyin*nzin*sizeof(amrex::Real));
  PeleLM::prob_parm->vin = (amrex::Real*) amrex::The_Arena()->alloc(nxin*nyin*nzin*sizeof(amrex::Real));
  PeleLM::prob_parm->win = (amrex::Real*) amrex::The_Arena()->alloc(nxin*nyin*nzin*sizeof(amrex::Real));
  std::memcpy(&PeleLM::prob_parm->xin,&prob_parm_l.xin,sizeof(prob_parm_l.xin));
  std::memcpy(&PeleLM::prob_parm->yin,&prob_parm_l.yin,sizeof(prob_parm_l.yin));
  std::memcpy(&PeleLM::prob_parm->zin,&prob_parm_l.zin,sizeof(prob_parm_l.zin));

  std::memcpy(&PeleLM::prob_parm->dxin,&prob_parm_l.dxin,sizeof(prob_parm_l.dxin));
  std::memcpy(&PeleLM::prob_parm->dyin,&prob_parm_l.dyin,sizeof(prob_parm_l.dyin));
  std::memcpy(&PeleLM::prob_parm->dzin,&prob_parm_l.dzin,sizeof(prob_parm_l.dzin));

  //amrex::Print() << " x " << std::endl;
	//for (int i=0; i<nxin; ++i) {
  //  amrex::Print() << PeleLM::prob_parm->dxin[i] << std::endl;
  //}
  //amrex::Print() << " y " << std::endl;
	//for (int i=0; i<nyin; ++i) {
  //  amrex::Print() << PeleLM::prob_parm->dyin[i] << std::endl;
  //}
  //amrex::Print() << " z " << std::endl;
	//for (int i=0; i<nzin; ++i) {
  //  amrex::Print() << PeleLM::prob_parm->dzin[i] << std::endl;
  //}

  std::memcpy(&PeleLM::prob_parm->uin,&prob_parm_l.uin,sizeof(prob_parm_l.uin));
  std::memcpy(&PeleLM::prob_parm->vin,&prob_parm_l.vin,sizeof(prob_parm_l.vin));
  std::memcpy(&PeleLM::prob_parm->win,&prob_parm_l.win,sizeof(prob_parm_l.win));

  //int myproc = amrex::ParallelDescriptor::MyProc();
  //int NProcs = amrex::ParallelDescriptor::NProcs();
  //for (int i=0; i<NProcs; i++) {
  //  amrex::ParallelDescriptor::Barrier();
  //  if (myproc==i) {
  //    for (int m=0; m<nxin; m++)
  //      amrex::AllPrint() << myproc << "  " << PeleLM::prob_parm->uin[m] << "\n"; 
  //  }
  //}

  std::string type;
  pp.query("P_mean", PeleLM::prob_parm->P_mean);
  pp.query("V_j", PeleLM::prob_parm->V_j);
  pp.query("V_cf", PeleLM::prob_parm->V_cf);
  pp.query("jet_rad", PeleLM::prob_parm->jet_rad);
  pp.query("jet_temp", PeleLM::prob_parm->jet_temp);
  pp.query("global_eq_ratio", PeleLM::prob_parm->global_eq_ratio);
  pp.query("ox_temp", PeleLM::prob_parm->ox_temp);
  pp.query("X_O2", PeleLM::prob_parm->X_O2);
  pp.query("X_N2", PeleLM::prob_parm->X_N2);
  pp.query("pertmag_cf", PeleLM::prob_parm->pertmag_cf);
  pp.query("pertmag_jet", PeleLM::prob_parm->pertmag_jet);
  pp.query("jet_purity", PeleLM::prob_parm->jet_purity);
  pp.query("bl_thickness", PeleLM::prob_parm->bl_thickness);
  pp.query("init_time", PeleLM::prob_parm->init_time);
  pp.query("double_jet", PeleLM::prob_parm->double_jet);
  pp.query("jet_dir", PeleLM::prob_parm->jet_dir);
  pp.query("cf_dir", PeleLM::prob_parm->cf_dir);

  PeleLM::pmf_data.initialize(); 
}
