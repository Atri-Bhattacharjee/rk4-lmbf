# --- CMake Configuration ---
cmake_minimum_required(VERSION 3.15)

# --- Project Definition ---
project(smc_lmb_tracker LANGUAGES CXX)

# --- C++ Standard Configuration ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Project Dependencies ---
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# --- Python Module Target ---
# Create the Python module target named 'lmb_engine' that will be importable in Python
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
target_include_directories(lmb_engine PUBLIC src external/munkres-cpp)
target_include_directories(lmb_engine PRIVATE external/astro/include)

# --- Link Libraries ---
target_link_libraries(lmb_engine PRIVATE Eigen3::Eigen)

# --- Add Astrodynamics Library ---
add_subdirectory(external/astro)

# --- Set Output Directory for Python Package ---
set_target_properties(lmb_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
)