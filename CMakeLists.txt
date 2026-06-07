# --- CMake Configuration ---
cmake_minimum_required(VERSION 3.15)

# --- Project Definition ---
project(smc_lmb_tracker LANGUAGES CXX)

# --- C++ Standard Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Python / pybind11 ---
set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# --- Project Dependencies ---
find_package(Eigen3 REQUIRED)
find_package(OpenMP)

option(LMB_ENGINE_ENABLE_VALIDATION "Keep hot-path validation checks in Release builds" OFF)

# --- Python Module Target ---
pybind11_add_module(lmb_engine
    src/main.cpp
    src/adaptive_birth_model.cpp
    src/smc_lmb_tracker.cpp
    src/in_orbit_sensor_model.cpp
    src/assignment.cpp
    src/metrics.cpp
    src/two_body_propagator.cpp
)

# --- Include Directories ---
target_include_directories(lmb_engine PUBLIC src)

# --- Link Libraries ---
target_link_libraries(lmb_engine PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(lmb_engine PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_definitions(lmb_engine PRIVATE
  $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
  $<$<CONFIG:Release>:EIGEN_DONT_PARALLELIZE>
  $<$<BOOL:${LMB_ENGINE_ENABLE_VALIDATION}>:LMB_ENGINE_ENABLE_VALIDATION>
)

set_property(TARGET lmb_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)

# --- Set Output Directory for Python Package ---
set_target_properties(lmb_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
)
