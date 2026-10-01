# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2025-2026 Arkady Gonoskov

# Builds the compiled core of twincher: the Python extension module twincher._core.
# Normally invoked through pip (scikit-build-core), see pyproject.toml and CONTRIBUTING.md.

cmake_minimum_required(VERSION 3.24)

# Compile CUDA code for the GPU(s) of the build machine unless architectures are given
# explicitly, e.g. via the environment variable CUDAARCHS="86;120" or
# -DCMAKE_CUDA_ARCHITECTURES="86;120".
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(twincher LANGUAGES CXX)

# GPU backend: AUTO builds it if a CUDA compiler (nvcc) is found, and otherwise builds a
# CPU-only package with a warning. ON makes the CUDA compiler mandatory, OFF disables it.
set(TWINCHER_CUDA AUTO CACHE STRING "Build the CUDA backend (AUTO, ON or OFF)")
set_property(CACHE TWINCHER_CUDA PROPERTY STRINGS AUTO ON OFF)
set(TWINCHER_WITH_CUDA OFF)
if(NOT TWINCHER_CUDA STREQUAL "OFF")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(TWINCHER_WITH_CUDA ON)
    elseif(TWINCHER_CUDA STREQUAL "ON")
        message(FATAL_ERROR "TWINCHER_CUDA=ON, but no CUDA compiler (nvcc) was found.")
    else()
        message(WARNING
            "No CUDA compiler (nvcc) was found: twincher is built without GPU support. "
            "To enable it, make nvcc available (e.g. add the bin directory of the CUDA toolkit "
            "to PATH, or set CUDACXX) and reinstall.")
    endif()
endif()
message(STATUS "twincher: CUDA backend ${TWINCHER_WITH_CUDA}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# compile_commands.json in the build directory, for code editors (IntelliSense) and tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TWINCHER_NATIVE_ARCH
    "Optimize host code for the CPU of the build machine (-march=native); the result is not portable" OFF)
option(TWINCHER_BUILD_BENCHMARKS "Build C++ benchmark executables" OFF)

# Version string reported by info(); scikit-build-core provides it from pyproject.toml
if(DEFINED SKBUILD_PROJECT_VERSION)
    set(TWINCHER_VERSION "${SKBUILD_PROJECT_VERSION}")
else()
    set(TWINCHER_VERSION "dev")
endif()

if(NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
endif()
if(TWINCHER_NATIVE_ARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

# ---------------------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
if(NOT DEFINED SKBUILD AND NOT DEFINED pybind11_DIR)
    # plain CMake build: use pybind11 installed in the active Python environment
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
find_package(pybind11 CONFIG REQUIRED)
# OpenMP parallelizes the CPU backend; without it, the CPU backend runs in a single thread
find_package(OpenMP COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP was not found: the CPU backend of twincher will run in a single thread.")
    if(NOT MSVC)
        add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>)
    endif()
endif()

# ---------------------------------------------------------------------------------------
# CUDA backend (static library linked into the extension module)
# ---------------------------------------------------------------------------------------
if(TWINCHER_WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(twincher_cuda STATIC src/shuttle_gpu.cu)
    target_include_directories(twincher_cuda
        PUBLIC  ${PROJECT_SOURCE_DIR}/include
        PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl
    )
    # The static CUDA runtime makes the module independent of the location/version of
    # libcudart.so at run time; only the NVIDIA driver is needed.
    target_link_libraries(twincher_cuda PUBLIC CUDA::cudart_static)
    target_compile_definitions(twincher_cuda PUBLIC TWINCHER_WITH_CUDA)
    set_target_properties(twincher_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    # Useful options for development and profiling of CUDA kernels:
    #   -lineinfo              source line information for Nsight Compute
    #   -Xptxas -v             report register and shared memory usage
    #   --maxrregcount=60      limit the number of registers per thread
    # target_compile_options(twincher_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

# ---------------------------------------------------------------------------------------
# Python extension module twincher._core
# ---------------------------------------------------------------------------------------
pybind11_add_module(_core src/bindings.cpp src/common.cpp)
target_include_directories(_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()
if(TWINCHER_WITH_CUDA)
    target_link_libraries(_core PRIVATE twincher_cuda)
endif()
target_compile_definitions(_core PRIVATE TWINCHER_VERSION="${TWINCHER_VERSION}")
install(TARGETS _core LIBRARY DESTINATION twincher)

# ---------------------------------------------------------------------------------------
# Benchmarks (optional, not installed)
# ---------------------------------------------------------------------------------------
if(TWINCHER_BUILD_BENCHMARKS)
    add_executable(bench_cpu benchmarks/bench_cpu.cpp)
    target_include_directories(bench_cpu PRIVATE ${PROJECT_SOURCE_DIR}/include)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(bench_cpu PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
