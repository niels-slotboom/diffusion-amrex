cmake_minimum_required(VERSION 3.15)

project(DiffusionAMReX CXX CUDA)

# Set output directory to bin/ in the project root
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Set your preferred compiler flags/options
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release)

# 1. Fetch AMReX
include(FetchContent)
FetchContent_Declare(
    amrex
    GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
    GIT_TAG        development # Or a specific release tag
)

# Set AMReX options before making available
set(AMREX_SPACEDIM 3 CACHE STRING "" FORCE)
set(AMREX_GPU_BACKEND CUDA CACHE STRING "" FORCE) # Since you use CUDA
set(AMREX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(amrex)

# 2. Glob all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "src/*.hpp")

# 3. Create your target
add_executable(diffusion-amrex ${SOURCES} ${HEADERS})

set_source_files_properties(src/main.cpp PROPERTIES LANGUAGE CUDA)

# 4. Include directories and link
target_include_directories(diffusion-amrex PRIVATE src)
target_link_libraries(diffusion-amrex PRIVATE amrex)