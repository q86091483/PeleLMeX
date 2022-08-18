#ifndef _EBUSERDEFINED_H_
#define _EBUSERDEFINED_H_

using namespace amrex;

#ifdef AMREX_USE_EB
#include <AMReX_ParmParse.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
void
EBUserDefined(const Geometry& geom,
              const int required_coarsening_level,
              const int max_coarsening_level)
{
    // ParmParse your geometry parameters
    // --> fixed geometry of the challenge problem
    //Geometric parameters. Default values
    amrex::Real d_venturi	= 0.013;
    amrex::Real d_swirlerexit	= 0.02032;
    amrex::Real d_dump		= 0.076;

    amrex::Real l_venturi	= 0.0008382;
    amrex::Real l_swirlercup	= 0.0043688;
    amrex::Real l_dump		= 0.20;

    amrex::Real cen_x_swirler_0,cen_y_swirler_0,cen_z_swirler_0;
    amrex::Real cen_x_swirler_1,cen_y_swirler_1,cen_z_swirler_1;
    amrex::Real cen_x_swirler_2,cen_y_swirler_2,cen_z_swirler_2;
    amrex::Real cen_x_swirler_3,cen_y_swirler_3,cen_z_swirler_3;
    amrex::Real cen_x_swirler_4,cen_y_swirler_4,cen_z_swirler_4;
    amrex::Real cen_x_swirler_5,cen_y_swirler_5,cen_z_swirler_5;
    amrex::Real cen_x_swirler_6,cen_y_swirler_6,cen_z_swirler_6;

    cen_x_swirler_0 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_0 = 0.0; cen_z_swirler_0 = 0.0;
    cen_x_swirler_1 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_1 = -0.011904858869; cen_z_swirler_1 = 0.020620105463;
    cen_x_swirler_2 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_2 = -0.02380996; cen_z_swirler_2 = 0.0;
    cen_x_swirler_3 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_3 = -0.011904858869; cen_z_swirler_3 = -0.020620105463;
    cen_x_swirler_4 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_4 = 0.011904858869; cen_z_swirler_4 = -0.020620105463;
    cen_x_swirler_5 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_5 = 0.02380996; cen_z_swirler_5 = 0.0;
    cen_x_swirler_6 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_6 = 0.011904858869; cen_z_swirler_6 = 0.020620105463;
 

    amrex::ParmParse pp("eb2");
    pp.query("d_venturi", d_venturi);
    pp.query("d_swirlerexit", d_swirlerexit);
    pp.query("d_dump", d_dump);

    pp.query("l_venturi", l_venturi);
    pp.query("l_swirlercup", l_swirlercup);
    pp.query("l_dump", l_dump);

    const amrex::Real buffer = 1000.0;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> plo = geom.ProbLoArray();
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> phi = geom.ProbHiArray();
    amrex::Real nx=(d_swirlerexit-d_venturi)/2.0;
    amrex::Real ny=l_swirlercup;
    amrex::Real nmag=sqrt(nx*nx+ny*ny);
    nx=nx/nmag;
    ny=ny/nmag;

    amrex::EB2::CylinderIF Dump(0.5*d_dump,l_dump, 0, {AMREX_D_DECL(l_venturi+l_swirlercup+l_dump*0.5, 0, 0)}, true);

    //Venturi Cylinders
    amrex::EB2::CylinderIF VenturiThroat_0(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_0,cen_y_swirler_0,cen_z_swirler_0)}, true);
    amrex::EB2::CylinderIF VenturiThroat_1(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_1,cen_y_swirler_1,cen_z_swirler_1)}, true);
    amrex::EB2::CylinderIF VenturiThroat_2(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_2,cen_y_swirler_2,cen_z_swirler_2)}, true);
    amrex::EB2::CylinderIF VenturiThroat_3(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_3,cen_y_swirler_3,cen_z_swirler_3)}, true);
    amrex::EB2::CylinderIF VenturiThroat_4(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_4,cen_y_swirler_4,cen_z_swirler_4)}, true);
    amrex::EB2::CylinderIF VenturiThroat_5(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_5,cen_y_swirler_5,cen_z_swirler_5)}, true);
    amrex::EB2::CylinderIF VenturiThroat_6(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_6,cen_y_swirler_6,cen_z_swirler_6)}, true);

    //Central Swirler Flare
    amrex::EB2::PlaneIF nozzle_plane_0({AMREX_D_DECL(0.5*d_venturi, 0.0, 0)},{AMREX_D_DECL(ny,-nx,0.0)},true);
    amrex::EB2::PlaneIF nozzle_truncplane_0({AMREX_D_DECL(l_venturi+l_swirlercup+0.00002, 0.0, 0)},{AMREX_D_DECL(1,0,0.0)},true);
    auto nozzle_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(nozzle_plane_0), 90 * M_PI / 180, 1),{AMREX_D_DECL(l_venturi,0, 0)});
    auto nozzle_trunc_0 = amrex::EB2::makeUnion(nozzle_0,nozzle_truncplane_0);
    amrex::EB2::PlaneIF exit_plane({AMREX_D_DECL(0.0, 0.0, 0.0)},{AMREX_D_DECL(0.866025,0.5,0.0)},true);
    auto exit_conv = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(exit_plane), 90 * M_PI / 180, 1),{AMREX_D_DECL(1.3*phi[0],0, 0)});
    //Copy and translate flares
    auto nozzle_trunc_1 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_1,cen_z_swirler_1)});
    auto nozzle_trunc_2 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_2,cen_z_swirler_2)});
    auto nozzle_trunc_3 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_3,cen_z_swirler_3)});
    auto nozzle_trunc_4 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_4,cen_z_swirler_4)});
    auto nozzle_trunc_5 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_5,cen_z_swirler_5)});
    auto nozzle_trunc_6 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_6,cen_z_swirler_6)});

    auto polys1 =amrex::EB2::makeIntersection(Dump,VenturiThroat_0,nozzle_trunc_0,
		VenturiThroat_1,nozzle_trunc_1,
		VenturiThroat_2,nozzle_trunc_2,
		VenturiThroat_3,nozzle_trunc_3,
		VenturiThroat_4,nozzle_trunc_4,
		VenturiThroat_5,nozzle_trunc_5,
		VenturiThroat_6,nozzle_trunc_6
                );
    auto polys = amrex::EB2::makeUnion(polys1,exit_conv);
    //auto polys =amrex::EB2::makeIntersection(Dump,VenturiThroat_0,VenturiThroat_1,VenturiThroat_2,VenturiThroat_3,VenturiThroat_4,VenturiThroat_5,VenturiThroat_6);
    auto gshop = amrex::EB2::makeShop(polys);
    amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
}
#endif
#endif
