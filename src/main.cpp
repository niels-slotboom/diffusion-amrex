#include <AMReX.H>
#include <AMReX_Print.H>

#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_RealBox.H"

#include <AMReX_PlotFileUtil.H>

#include <filesystem>
namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    // Initialize AMReX (handles MPI setup, GPU device selection, etc.)
    amrex::Initialize(argc, argv);

    // amrex::Print() prints only from the I/O processor (rank 0)
    amrex::Print() << "Hello, World from AMReX!" << std::endl;

    // set up the (integer) Box `domain` that gives the base resolution
    int n_cell = 64;
    amrex::IntVect dom_lo(0, 0, 0);
    amrex::IntVect dom_hi(n_cell - 1, n_cell - 1, n_cell - 1);

    amrex::Box domain(dom_lo, dom_hi);

    // set up the coordinates that box corresponds to in a RealBox
    amrex::RealBox real_box({-2.0, -2.0, -2.0}, {2.0, 2.0, 2.0});

    int coord = 0;

    int is_periodic[3] = {
        1,
        1,
        1,
    };

    // glue integer box and real box into a geometry object
    amrex::Geometry geom(domain, &real_box, coord, is_periodic);

    // produe a BoxArray to store all the smaller boxes the large box is chopped
    // up into
    amrex::BoxArray ba(domain);
    ba.maxSize(32);

    // initialise the DistributionMapping for MPI distribution (trivial on one
    // device)
    amrex::DistributionMapping dm(ba);

    // make the MultiFab object
    int ncomp = 1; // how many components there are to be stored per gridpoint
    int ngrow = 1; // how many ghost cells to grow/expand the boxes by
    amrex::MultiFab mf(ba, dm, ncomp, ngrow);
    mf.setVal(0.0); // init to 0

    // fill with some interesting data
    const auto dx = geom.CellSizeArray();

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box &box = mfi.validbox();
        const amrex::Array4<amrex::Real> &arr = mf.array(mfi);

        const auto dx = geom.CellSizeArray();
        const auto lo = geom.ProbLoArray();

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            amrex::Real x = lo[0] + (i + 0.5) * dx[0];
            amrex::Real y = lo[1] + (j + 0.5) * dx[1];
            amrex::Real z = lo[2] + (k + 0.5) * dx[2];

            amrex::Real rsq = x * x + y * y + z * z + y * z;
            amrex::Real sigma = 0.5;

            arr(i, j, k, 0) =
                x * y * y * std::exp(-rsq / (2.0 * sigma * sigma));
        });
    }

    // export to disk for viewer
    fs::path rawOutputPath("raw");

    amrex::Vector<std::string> var_names = {"f(x,y,z)"};
    amrex::WriteSingleLevelPlotfile(rawOutputPath / "export", mf, var_names,
                                    geom, 0.0, 0);

    // Clean up resources
    amrex::Finalize();
    return 0;
}