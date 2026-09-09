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
option(LMB_ENGINE_SANITIZE "Build the extension with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)

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

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lmb_engine PRIVATE -Wall -Wextra)
endif()

target_compile_definitions(lmb_engine PRIVATE
  $<$<CONFIG:Release>:EIGEN_NO_DEBUG>
  $<$<CONFIG:Release>:EIGEN_DONT_PARALLELIZE>
  $<$<BOOL:${LMB_ENGINE_ENABLE_VALIDATION}>:LMB_ENGINE_ENABLE_VALIDATION>
)

# --- Sanitizers ---
# The extension is instrumented but the host interpreter is not, so the ASan runtime has to be
# LD_PRELOADed at run time. scripts/asan-test.sh does that. Output goes to a dedicated
# python/lmb_engine/Asan directory so an instrumented build never shadows Release or Debug.
if(LMB_ENGINE_SANITIZE)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "LMB_ENGINE_SANITIZE requires GCC or Clang")
  endif()
  set(LMB_SANITIZE_FLAGS
    -fsanitize=address
    -fsanitize=undefined
    -fno-sanitize-recover=undefined
    -fno-omit-frame-pointer
    -g
  )
  target_compile_options(lmb_engine PRIVATE ${LMB_SANITIZE_FLAGS})
  target_link_options(lmb_engine PRIVATE -fsanitize=address -fsanitize=undefined)
endif()

if(NOT LMB_ENGINE_SANITIZE)
  set_property(TARGET lmb_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# --- Set Output Directory for Python Package ---
if(LMB_ENGINE_SANITIZE)
  set(LMB_ENGINE_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/python/lmb_engine/Asan")
  set_target_properties(lmb_engine PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${LMB_ENGINE_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${LMB_ENGINE_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY_DEBUG "${LMB_ENGINE_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY_RELEASE "${LMB_ENGINE_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${LMB_ENGINE_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_DEBUG "${LMB_ENGINE_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${LMB_ENGINE_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${LMB_ENGINE_OUTPUT_DIR}"
  )
else()
  set_target_properties(lmb_engine PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
      LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
      RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/python/lmb_engine/Debug"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/python/lmb_engine/Release"
  )
endif()
