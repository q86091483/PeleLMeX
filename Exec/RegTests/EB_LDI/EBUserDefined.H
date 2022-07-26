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
    amrex::EB2::CylinderIF VenturiThroat(0.5*d_venturi, l_venturi+l_swirlercup, 0, {AMREX_D_DECL(l_venturi*0.499+l_swirlercup*0.499,0, 0)}, true);
    amrex::EB2::PlaneIF nozzle_plane({AMREX_D_DECL(0.5*d_venturi, 0.0, 0)},{AMREX_D_DECL(ny,-nx,0.0)},true);
    amrex::EB2::PlaneIF nozzle_truncplane({AMREX_D_DECL(l_venturi+l_swirlercup+0.00002, 0.0, 0)},{AMREX_D_DECL(1,0,0.0)},true);
    auto nozzle = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(nozzle_plane), 90 * M_PI / 180, 1),{AMREX_D_DECL(0,0, 0)});
    amrex::EB2::PlaneIF exit_plane({AMREX_D_DECL(0.0, 0.0, 0.0)},{AMREX_D_DECL(1.0,1.0,0.0)},true);
    auto exit_conv = amrex::EB2::translate(amrex::EB2::rotate(amrex::EB2::lathe(exit_plane), 90 * M_PI / 180, 1),{AMREX_D_DECL(1.12*phi[0],0, 0)});
    //auto nozzle = amrex::EB2::rotate(amrex::EB2::translate(amrex::EB2::lathe(nozzle_plane),{AMREX_D_DECL(0,0,(l_venturi+l_swirlercup))}),90*M_PI / 180, 1);

    auto nozzle_trunc = amrex::EB2::makeUnion(nozzle,nozzle_truncplane);
    //auto polys =amrex::EB2::makeIntersection(Dump,VenturiThroat,nozzle_trunc);
    //auto polys =amrex::EB2::makeIntersection(VenturiThroat,nozzle_trunc);
    //auto polys =amrex::EB2::makeIntersection(Dump,VenturiThroat,nozzle);
    //auto polys =amrex::EB2::makeIntersection(Dump,VenturiThroat);
    //auto polys =amrex::EB2::makeIntersection(Dump,nozzle_trunc);
    //auto polys = VenturiThroat;

    auto polys_1 = amrex::EB2::makeIntersection(Dump,VenturiThroat,nozzle_trunc);
    auto polys = amrex::EB2::makeUnion(polys_1,exit_conv);
    auto gshop = amrex::EB2::makeShop(polys);
    amrex::EB2::Build(gshop, geom, required_coarsening_level, max_coarsening_level);
}
#endif
#endif
