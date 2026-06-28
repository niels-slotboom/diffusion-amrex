#include <AMReX.H>
#include <AMReX_Print.H>

int main(int argc, char *argv[]) {
    // Initialize AMReX (handles MPI setup, GPU device selection, etc.)
    amrex::Initialize(argc, argv);

    // amrex::Print() prints only from the I/O processor (rank 0)
    amrex::Print() << "Hello, World from AMReX!" << std::endl;

    // Clean up resources
    amrex::Finalize();
    return 0;
}