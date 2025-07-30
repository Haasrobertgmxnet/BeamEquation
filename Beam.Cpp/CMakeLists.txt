cmake_minimum_required(VERSION 3.16)
project(Beam.Cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug und Release Konfigurationen definieren
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

# Setze vcpkg toolchain explizit
set(CMAKE_TOOLCHAIN_FILE "C:/Users/haasr/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

# Finde Torch
find_package(Torch REQUIRED)

# Erstelle executable
add_executable(Beam.Cpp main.cpp)

# Include-Verzeichnis hinzufuegen
target_include_directories(Beam.Cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/build)

# Link libraries
target_link_libraries(Beam.Cpp ${TORCH_LIBRARIES})

# Compiler flags
target_compile_features(Beam.Cpp PRIVATE cxx_std_23)

# Debug-Konfiguration
target_compile_options(Beam.Cpp PRIVATE
    $<$<CONFIG:Debug>:
        $<$<CXX_COMPILER_ID:MSVC>:/Od /Zi /MDd>
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g>
    >
)

# Release-Konfiguration
target_compile_options(Beam.Cpp PRIVATE
    $<$<CONFIG:Release>:
        $<$<CXX_COMPILER_ID:MSVC>:/O2 /DNDEBUG /MD>
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -DNDEBUG>
    >
)

# Windows spezifisch
if(MSVC)
    target_compile_options(Beam.Cpp PRIVATE /bigobj)
endif()