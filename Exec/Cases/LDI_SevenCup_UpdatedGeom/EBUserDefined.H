#ifndef _EBUSERDEFINED_H_
#define _EBUSERDEFINED_H_

using namespace amrex;

#ifdef AMREX_USE_EB
#include <AMReX_ParmParse.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
/*
void
EBUserDefined_Working(const Geometry& geom,
              const int required_coarsening_level,
              const int max_coarsening_level)
{
    // ParmParse your geometry parameters
    // --> fixed geometry of the challenge problem
    //Geometric parameters. Default values
    amrex::Real d_venturi	= 0.013;
    amrex::Real d_swirlerexit	= 0.02032;
    amrex::Real d_dump		= 0.076;
    amrex::Real d_downstreamvane= 0.0221488;
    amrex::Real d_fuelstembeg   = 0.0067635;
    //amrex::Real d_fuelstemend   = 0.0067635;//0.0019261;
    amrex::Real d_fuelstemend   = 0.0019261;


    amrex::Real l_venturi	= 0.0008382;
    amrex::Real l_swirlercup	= 0.0043688;
    amrex::Real l_downstreamvane= 0.0050552;
    amrex::Real l_fuelstem      = 0.0022605;
    amrex::Real l_dump		= 0.20;
    amrex::Real l_fuelstem_offset = 0.0018014;

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

    amrex::Real nx1 = (d_downstreamvane-d_venturi)/2.0;
    amrex::Real ny1 = l_downstreamvane;
    amrex::Real nmag1=sqrt(nx1*nx1+ny1*ny1);
    nx1=nx1/nmag1;
    ny1=ny1/nmag1;
    nx1=cos(0.735249121);
    ny1=-sin(0.735249121);

    amrex::Real nx2 = (d_fuelstembeg-d_fuelstemend)/2.0;
    amrex::Real ny2 = l_fuelstem;
    amrex::Real nmag2=sqrt(nx2*nx2+ny2*ny2);
    nx2=nx2/nmag2;
    ny2=ny2/nmag2;

    ny2=-sin(0.8191944884);
    nx2=cos(0.8191944884);

    amrex::Real bufferdist=0.0009933;

    amrex::Print()<<"\nCos 45 = "<<nx2;


    //Dump region
    amrex::EB2::CylinderIF Dump(0.5*d_dump,l_dump, 0, {AMREX_D_DECL(l_venturi+l_swirlercup+l_dump*0.5, 0, 0)}, true);
    amrex::EB2::PlaneIF exit_plane({AMREX_D_DECL(0.0, 0.0, 0.0)},{AMREX_D_DECL(0.866025,0.5,0.0)},true);
    auto exit_conv = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(exit_plane), 90 * M_PI / 180, 1),{AMREX_D_DECL(1.3*phi[0],0, 0)});

    //Venturi Cylinders
    amrex::EB2::CylinderIF VenturiThroat_0(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_0,cen_y_swirler_0,cen_z_swirler_0)}, true);
    amrex::EB2::CylinderIF VenturiThroat_00(0.25*d_venturi, 1*(l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_0,cen_y_swirler_0,cen_z_swirler_0)}, true);
    //amrex::EB2::CylinderIF VenturiThroat_0(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_0,cen_y_swirler_0,cen_z_swirler_0)}, true);
    amrex::EB2::CylinderIF VenturiThroat_1(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_1,cen_y_swirler_1,cen_z_swirler_1)}, true);
    amrex::EB2::CylinderIF VenturiThroat_2(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_2,cen_y_swirler_2,cen_z_swirler_2)}, true);
    amrex::EB2::CylinderIF VenturiThroat_3(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_3,cen_y_swirler_3,cen_z_swirler_3)}, true);
    amrex::EB2::CylinderIF VenturiThroat_4(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_4,cen_y_swirler_4,cen_z_swirler_4)}, true);
    amrex::EB2::CylinderIF VenturiThroat_5(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_5,cen_y_swirler_5,cen_z_swirler_5)}, true);
    amrex::EB2::CylinderIF VenturiThroat_6(0.5*d_venturi, 1*(l_venturi+l_swirlercup), 0, {AMREX_D_DECL(cen_x_swirler_6,cen_y_swirler_6,cen_z_swirler_6)}, true);

    //Divergent part of venturi
    amrex::EB2::PlaneIF nozzle_plane_0({AMREX_D_DECL(0.5*d_venturi, 0.0, 0)},{AMREX_D_DECL(ny,-nx,0.0)},true);
    amrex::EB2::PlaneIF nozzle_truncplane_0({AMREX_D_DECL(l_venturi-0.00002, 0.0, 0)},{AMREX_D_DECL(-1,0,0.0)},true);
    amrex::EB2::PlaneIF nozzle_truncplane_e({AMREX_D_DECL(l_venturi+l_swirlercup+0.00002, 0.0, 0)},{AMREX_D_DECL(1,0,0.0)},true);
    auto nozzle_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(nozzle_plane_0), 90 * M_PI / 180, 1),{AMREX_D_DECL(l_venturi,0, 0)});
    auto nozzle_trunc_0 = amrex::EB2::makeUnion(nozzle_0,nozzle_truncplane_e,nozzle_truncplane_0);

    //Copy and translate divergent venturi
    auto nozzle_trunc_1 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_1,cen_z_swirler_1)});
    auto nozzle_trunc_2 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_2,cen_z_swirler_2)});
    auto nozzle_trunc_3 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_3,cen_z_swirler_3)});
    auto nozzle_trunc_4 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_4,cen_z_swirler_4)});
    auto nozzle_trunc_5 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_5,cen_z_swirler_5)});
    auto nozzle_trunc_6 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(0,cen_y_swirler_6,cen_z_swirler_6)});

    //Convergent part of venturi
    amrex::EB2::PlaneIF convnozzle_plane_0({AMREX_D_DECL(0.5*d_venturi, 0.0, 0)},{AMREX_D_DECL(nx1,ny1,0.0)},true);
    auto nozzlevanedownstream=amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(convnozzle_plane_0), 0 * M_PI / 180, 1),{AMREX_D_DECL(0,0, 0)});
    amrex::EB2::PlaneIF convnozzle_truncplane_0({AMREX_D_DECL(0.0, 0.0, 0)},{AMREX_D_DECL(0,0,-1.0)},true);
    amrex::EB2::PlaneIF convnozzle_truncplane_1({AMREX_D_DECL(0.0, 0.0, l_downstreamvane )},{AMREX_D_DECL(0,0,1.0)},true);
    auto convnozzle_trunc_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::makeUnion(nozzlevanedownstream,convnozzle_truncplane_0,convnozzle_truncplane_1),-90*M_PI/180.0,1),{AMREX_D_DECL(0,0, 0)});

    //FuelStem
    amrex::EB2::PlaneIF fuelstem_plane_0({AMREX_D_DECL(0.5*d_fuelstemend, 0.0, 0)},{AMREX_D_DECL(nx2,ny2,0.0)},true);
    auto fuelstem=amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(fuelstem_plane_0), 0 * M_PI / 180, 1),{AMREX_D_DECL(0,0, 0)});
    amrex::EB2::PlaneIF fuelstem_truncplane_0({AMREX_D_DECL(0, 0.0, 0)},{AMREX_D_DECL(0,0,-1.0)},true);
    amrex::EB2::PlaneIF fuelstem_truncplane_1({AMREX_D_DECL(0, 0.0,l_fuelstem+bufferdist)},{AMREX_D_DECL(0,0,1.0)},true);
    auto fuelstem_trunc_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::makeUnion(fuelstem,fuelstem_truncplane_0,fuelstem_truncplane_1),-90*M_PI/180.0,1),{AMREX_D_DECL(-l_fuelstem_offset,0, 0)});
    auto fuelstem_trunc_0_f = amrex::EB2::makeIntersection(convnozzle_trunc_0,fuelstem_trunc_0);

    //Test simple geometries to get the f'ing geometry right
    /*amrex::EB2::PlaneIF fuelstem_plane_0({AMREX_D_DECL(0.5*d_venturi, 0.0, 0)},{AMREX_D_DECL(1,0,0.0)},true);
    amrex::EB2::PlaneIF fuelstem_plane_1({AMREX_D_DECL(0.25*d_venturi, 0.0, 0)},{AMREX_D_DECL(-1,0,0.0)},true);
    amrex::EB2::PlaneIF fuelstem_end({AMREX_D_DECL(0.25*d_venturi, 0.1, 0)},{AMREX_D_DECL(0,1,0.0)},true);
    amrex::EB2::PlaneIF fuelstem_beg({AMREX_D_DECL(0.25*d_venturi, 0.05, 0)},{AMREX_D_DECL(0,-1,0.0)},true);

    auto rect= amrex::EB2::makeUnion(fuelstem_plane_0,fuelstem_plane_1,fuelstem_end,fuelstem_beg);
    auto fun = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(rect), 90 * M_PI / 180, 1),{AMREX_D_DECL(0.0, 0, 0)});*/

    //amrex::EB2::CylinderIF fs_0(0.5*d_fuelstemend*4, 1*(l_fuelstem), 0, {AMREX_D_DECL(-l_downstreamvane,cen_y_swirler_0,cen_z_swirler_0)}, true);


    //Test for one swirler
    //auto polys = amrex::EB2::makeUnion(VenturiThroat_0,amrex::EB2::makeComplement(fuelstem_trunc_0));
    //auto polys =amrex::EB2::makeIntersection(VenturiThroat_0,fuelstem);
    //auto polys = convnozzle_trunc_0;
    //auto polys = fuelstem_trunc_0,convnozzle_trunc_0);//_trunc_0;
/*
    auto swirler_0 = amrex::EB2::makeComplement(amrex::EB2::makeIntersection(fuelstem_trunc_0,amrex::EB2::makeComplement(convnozzle_trunc_0)));
    auto swirler_1 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(0,cen_y_swirler_1,cen_z_swirler_1)});
    auto swirler_2 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(0,cen_y_swirler_2,cen_z_swirler_2)});
    auto swirler_3 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(0,cen_y_swirler_3,cen_z_swirler_3)});
    auto swirler_4 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(0,cen_y_swirler_4,cen_z_swirler_4)});
    auto swirler_5 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(0,cen_y_swirler_5,cen_z_swirler_5)});
    auto swirler_6 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(0,cen_y_swirler_6,cen_z_swirler_6)});

    auto polys1 = amrex::EB2::makeIntersection(
    											swirler_0,VenturiThroat_0,nozzle_trunc_0,
												swirler_1,VenturiThroat_1,nozzle_trunc_1,
												swirler_2,VenturiThroat_2,nozzle_trunc_2,
												swirler_3,VenturiThroat_3,nozzle_trunc_3,
												swirler_4,VenturiThroat_4,nozzle_trunc_4,
												swirler_5,VenturiThroat_5,nozzle_trunc_5,
												swirler_6,VenturiThroat_6,nozzle_trunc_6,
												Dump);

    auto polys = amrex::EB2::makeUnion(polys1,exit_conv);
    auto gshop = amrex::EB2::makeShop(polys);
    amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
}*/


void
EBUserDefined(const Geometry& geom,
              const int required_coarsening_level,
              const int max_coarsening_level)
{
    // ParmParse your geometry parameters
    // --> fixed geometry of the challenge problem
    //Geometric parameters. Default values
    amrex::Real d_venturi		= 0.013;
    amrex::Real d_swirlerexit	= 0.02032;
    amrex::Real d_dump			= 0.076;
    amrex::Real d_downstreamvane= 0.0221488;
    amrex::Real d_fuelstembeg   = 0.0067635;
    amrex::Real d_fuelstemend   = 0.0019261;
    

    amrex::Real l_venturi			= 0.0008382;
    amrex::Real l_swirlercup		= 0.0043688;
    amrex::Real l_downstreamvane	= 0.0050552;
    amrex::Real l_fuelstem      	= 0.0022605;
    amrex::Real l_dump				= 0.20;
    amrex::Real l_fuelstem_offset 	= 0.0018014;

    amrex::Real cen_x_swirler_0,cen_y_swirler_0,cen_z_swirler_0;
    amrex::Real cen_x_swirler_1,cen_y_swirler_1,cen_z_swirler_1;
    amrex::Real cen_x_swirler_2,cen_y_swirler_2,cen_z_swirler_2;
    amrex::Real cen_x_swirler_3,cen_y_swirler_3,cen_z_swirler_3;
    amrex::Real cen_x_swirler_4,cen_y_swirler_4,cen_z_swirler_4;
    amrex::Real cen_x_swirler_5,cen_y_swirler_5,cen_z_swirler_5;
    amrex::Real cen_x_swirler_6,cen_y_swirler_6,cen_z_swirler_6;

    cen_z_swirler_0 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_0 = 0.0; cen_x_swirler_0 = 0.0;
    cen_z_swirler_1 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_1 = -0.011904858869; cen_x_swirler_1 = 0.020620105463;
    cen_z_swirler_2 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_2 = -0.02380996; cen_x_swirler_2 = 0.0;
    cen_z_swirler_3 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_3 = -0.011904858869; cen_x_swirler_3 = -0.020620105463;
    cen_z_swirler_4 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_4 = 0.011904858869; cen_x_swirler_4 = -0.020620105463;
    cen_z_swirler_5 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_5 = 0.02380996; cen_x_swirler_5 = 0.0;
    cen_z_swirler_6 = l_venturi*0.499+l_swirlercup*0.499;  cen_y_swirler_6 = 0.011904858869; cen_x_swirler_6 = 0.020620105463;
 

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

    amrex::Real nx1 = (d_downstreamvane-d_venturi)/2.0;
    amrex::Real ny1 = l_downstreamvane;
    amrex::Real nmag1=sqrt(nx1*nx1+ny1*ny1);
    nx1=nx1/nmag1;
    ny1=ny1/nmag1;
    nx1=cos(0.735249121);
    ny1=-sin(0.735249121);

    amrex::Real nx2 = (d_fuelstembeg-d_fuelstemend)/2.0;
    amrex::Real ny2 = l_fuelstem;
    amrex::Real nmag2=sqrt(nx2*nx2+ny2*ny2);
    nx2=nx2/nmag2;
    ny2=ny2/nmag2;

    ny2=-sin(0.8191944884);
    nx2=cos(0.8191944884);

    amrex::Real bufferdist=0.0009933;

    const int axis_direction=2;
    const int x_dir=0;
    const int y_dir=0;
    const int z_dir=0;


    //Dump region
    amrex::EB2::CylinderIF Dump(0.5*d_dump,l_dump, axis_direction, {AMREX_D_DECL(0,0,l_venturi+l_swirlercup+l_dump*0.5)}, true);
    amrex::EB2::PlaneIF exit_plane({AMREX_D_DECL(0.0, 0.0, 0.0)},{AMREX_D_DECL(0.866025,0.5,0.0)},true);
    auto exit_conv = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(exit_plane), 0 * M_PI / 180, 1),{AMREX_D_DECL(0, 0,1.3*phi[2])});


    amrex::EB2::CylinderIF VenturiThroat_0(0.5*d_venturi, 1*(l_venturi+l_swirlercup), axis_direction, {AMREX_D_DECL(cen_x_swirler_0,cen_y_swirler_0,cen_z_swirler_0)}, true);
    amrex::EB2::CylinderIF VenturiThroat_1(0.5*d_venturi, 1*(l_venturi+l_swirlercup), axis_direction, {AMREX_D_DECL(cen_x_swirler_1,cen_y_swirler_1,cen_z_swirler_1)}, true);
    amrex::EB2::CylinderIF VenturiThroat_2(0.5*d_venturi, 1*(l_venturi+l_swirlercup), axis_direction, {AMREX_D_DECL(cen_x_swirler_2,cen_y_swirler_2,cen_z_swirler_2)}, true);
    amrex::EB2::CylinderIF VenturiThroat_3(0.5*d_venturi, 1*(l_venturi+l_swirlercup), axis_direction, {AMREX_D_DECL(cen_x_swirler_3,cen_y_swirler_3,cen_z_swirler_3)}, true);
    amrex::EB2::CylinderIF VenturiThroat_4(0.5*d_venturi, 1*(l_venturi+l_swirlercup), axis_direction, {AMREX_D_DECL(cen_x_swirler_4,cen_y_swirler_4,cen_z_swirler_4)}, true);
    amrex::EB2::CylinderIF VenturiThroat_5(0.5*d_venturi, 1*(l_venturi+l_swirlercup), axis_direction, {AMREX_D_DECL(cen_x_swirler_5,cen_y_swirler_5,cen_z_swirler_5)}, true);
    amrex::EB2::CylinderIF VenturiThroat_6(0.5*d_venturi, 1*(l_venturi+l_swirlercup), axis_direction, {AMREX_D_DECL(cen_x_swirler_6,cen_y_swirler_6,cen_z_swirler_6)}, true);

    //Divergent part of venturi
    amrex::EB2::PlaneIF nozzle_plane_0({AMREX_D_DECL(0.5*d_venturi, 0.0, 0)},{AMREX_D_DECL(ny,-nx,0.0)},true);
    amrex::EB2::PlaneIF nozzle_truncplane_0({AMREX_D_DECL(0.0, 0,l_venturi-0.00002)},{AMREX_D_DECL(0,0.0,-1)},true);
    amrex::EB2::PlaneIF nozzle_truncplane_e({AMREX_D_DECL(0.0, 0, l_venturi+l_swirlercup+0.00002)},{AMREX_D_DECL(0,0.0,1)},true);
    auto nozzle_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(nozzle_plane_0), 0 * M_PI / 180, 1),{AMREX_D_DECL(0, 0, l_venturi)});
    auto nozzle_trunc_0 = amrex::EB2::makeUnion(nozzle_0,nozzle_truncplane_e,nozzle_truncplane_0);

    //Convergent part of venturi
    amrex::EB2::PlaneIF convnozzle_plane_0({AMREX_D_DECL(0.5*d_venturi, 0.0, 0)},{AMREX_D_DECL(nx1,ny1,0.0)},true);
    auto nozzlevanedownstream=amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(convnozzle_plane_0), 0 * M_PI / 180, 1),{AMREX_D_DECL(0,0, 0)});
    amrex::EB2::PlaneIF convnozzle_truncplane_0({AMREX_D_DECL(0.0, 0.0, 0)},{AMREX_D_DECL(0,0.0,-1)},true);
    amrex::EB2::PlaneIF convnozzle_truncplane_1({AMREX_D_DECL(0,0, l_downstreamvane )},{AMREX_D_DECL(0,0,1)},true);
    auto convnozzle_trunc_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::makeUnion(nozzlevanedownstream,convnozzle_truncplane_0,convnozzle_truncplane_1),180*M_PI/180.0,1),{AMREX_D_DECL(0,0, 0)});

    //FuelStem
    amrex::EB2::PlaneIF fuelstem_plane_0({AMREX_D_DECL(0.5*d_fuelstemend, 0.0, 0)},{AMREX_D_DECL(nx2,ny2,0.0)},true);
    auto fuelstem=amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(fuelstem_plane_0), 0 * M_PI / 180, 1),{AMREX_D_DECL(0,0, 0)});
    amrex::EB2::PlaneIF fuelstem_truncplane_0({AMREX_D_DECL(0, 0.0, 0)},{AMREX_D_DECL(0,0,-1.0)},true);
    amrex::EB2::PlaneIF fuelstem_truncplane_1({AMREX_D_DECL(0, 0.0,l_fuelstem+bufferdist)},{AMREX_D_DECL(0,0,1.0)},true);
    auto fuelstem_trunc_0 = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::makeUnion(fuelstem,fuelstem_truncplane_0,fuelstem_truncplane_1),180*M_PI/180.0,1),{AMREX_D_DECL(0, 0,-l_fuelstem_offset)});
    auto fuelstem_trunc_0_f = amrex::EB2::makeIntersection(convnozzle_trunc_0,fuelstem_trunc_0);

    auto swirler_0 = amrex::EB2::makeComplement(amrex::EB2::makeIntersection(fuelstem_trunc_0,amrex::EB2::makeComplement(convnozzle_trunc_0)));
    auto swirler_1 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(cen_x_swirler_1,cen_y_swirler_1,0)});
    auto swirler_2 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(cen_x_swirler_2,cen_y_swirler_2,0)});
    auto swirler_3 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(cen_x_swirler_3,cen_y_swirler_3,0)});
    auto swirler_4 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(cen_x_swirler_4,cen_y_swirler_4,0)});
    auto swirler_5 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(cen_x_swirler_5,cen_y_swirler_5,0)});
    auto swirler_6 = amrex::EB2::translate(swirler_0,{AMREX_D_DECL(cen_x_swirler_6,cen_y_swirler_6,0)});

    auto nozzle_trunc_1 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(cen_x_swirler_1,cen_y_swirler_1,0)});
    auto nozzle_trunc_2 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(cen_x_swirler_2,cen_y_swirler_2,0)});
    auto nozzle_trunc_3 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(cen_x_swirler_3,cen_y_swirler_3,0)});
    auto nozzle_trunc_4 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(cen_x_swirler_4,cen_y_swirler_4,0)});
    auto nozzle_trunc_5 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(cen_x_swirler_5,cen_y_swirler_5,0)});
    auto nozzle_trunc_6 = amrex::EB2::translate(nozzle_trunc_0,{AMREX_D_DECL(cen_x_swirler_6,cen_y_swirler_6,0)});

    auto polys1 = amrex::EB2::makeIntersection(
        											swirler_0,VenturiThroat_0,nozzle_trunc_0,
    												swirler_1,VenturiThroat_1,nozzle_trunc_1,
    												swirler_2,VenturiThroat_2,nozzle_trunc_2,
    												swirler_3,VenturiThroat_3,nozzle_trunc_3,
    												swirler_4,VenturiThroat_4,nozzle_trunc_4,
    												swirler_5,VenturiThroat_5,nozzle_trunc_5,
    												swirler_6,VenturiThroat_6,nozzle_trunc_6,
    												Dump);
    auto polys = amrex::EB2::makeUnion(polys1,exit_conv);

    //auto polys = amrex::EB2::makeUnion(polys1,exit_conv);
    auto gshop = amrex::EB2::makeShop(polys);
    amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
}
#endif
#endif
