cmake_minimum_required(VERSION 3.15)

# Project name and version
project(MatrixLib VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build position independent code for static libs used by shared modules (pybind11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add library
add_library(matrixlib
    src/matrixlib.cpp
)

# Ensure matrixlib is built as PIC to allow linking into shared modules
set_target_properties(matrixlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set library properties
target_include_directories(matrixlib
    PUBLIC 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        ${PROJECT_SOURCE_DIR}/src
)

# Build examples if enabled
if(BUILD_EXAMPLES)
    add_executable(example examples/example_main.cpp)
    target_link_libraries(example PRIVATE matrixlib)
    set_target_properties(example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/examples
    )
endif()

# Python bindings with pybind11
# Build a Python extension module: matrixlib_py
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(matrixlib_py
    python/pybind_module.cpp
)
target_link_libraries(matrixlib_py PRIVATE matrixlib)
set_target_properties(matrixlib_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python_pkg/LinAlgKit
)

# Build tests if enabled
if(BUILD_TESTS)
    enable_testing()
    
    # Find or download GoogleTest
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    
    add_executable(matrix_tests tests/test_matrix.cpp)
    target_link_libraries(matrix_tests
        PRIVATE
            matrixlib
            GTest::gtest
            GTest::gtest_main
    )
    
    # Add test (use resolved path for robustness)
    add_test(NAME matrix_tests COMMAND $<TARGET_FILE:matrix_tests>)
    
    set_target_properties(matrix_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )
endif()

# Build benchmarks if enabled
if(BUILD_BENCHMARKS)
    # Find or download Google Benchmark
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.7.1
    )
    
    set(BENCHMARK_ENABLE_TESTING OFF)
    FetchContent_MakeAvailable(benchmark)
    
    add_executable(matrix_benchmarks benchmarks/benchmark_matrix.cpp)
    target_link_libraries(matrix_benchmarks
        PRIVATE
            matrixlib
            benchmark::benchmark
    )
    
    set_target_properties(matrix_benchmarks PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
    )
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS matrixlib
    EXPORT MatrixLibTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT MatrixLibTargets
    FILE MatrixLibTargets.cmake
    NAMESPACE MatrixLib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatrixLib
)

# Package configuration
# include(CMakePackageConfigHelpers)
# configure_package_config_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatrixLibConfig.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/MatrixLibConfig.cmake
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatrixLib
# )


# Generate the version file
# write_basic_package_version_file(
#     ${CMAKE_CURRENT_BINARY_DIR}/MatrixLibConfigVersion.cmake
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY SameMajorVersion
# )

# Install the config and version files
# install(FILES
#     ${CMAKE_CURRENT_BINARY_DIR}/MatrixLibConfig.cmake
#     ${CMAKE_CURRENT_BINARY_DIR}/MatrixLibConfigVersion.cmake
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MatrixLib
# )
