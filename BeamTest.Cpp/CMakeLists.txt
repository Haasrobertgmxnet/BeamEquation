cmake_minimum_required(VERSION 3.20)
project(BeamTest.Cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg toolchain support
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/../vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

# Find Torch and Catch2 via vcpkg
find_package(Torch REQUIRED)
find_package(Catch2 CONFIG REQUIRED)

# Include your Beam source (adjust path as necessary)
add_subdirectory(../Beam.Cpp ${CMAKE_BINARY_DIR}/Beam.Cpp_build)

# Define test executable
add_executable(BeamTest.Cpp test_beam.cpp)

# Link libraries
target_link_libraries(BeamTest.Cpp PRIVATE ${TORCH_LIBRARIES}
    Catch2::Catch2WithMain
)

# Add include paths if needed
target_include_directories(BeamTest.Cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Beam.Cpp/include
)

# Register with CTest (optional)
include(CTest)
enable_testing()
add_test(NAME BeamTest.Cpp COMMAND BeamTest.Cpp)
