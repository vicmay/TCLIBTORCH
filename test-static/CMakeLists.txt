cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorchtcl_test CXX)

# Set C++17 for LibTorch compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set LibTorch paths
set(LIBTORCH_PATH "/home/user/CascadeProjects/LIBTORCH/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})

# Find Torch package
set(TORCH_USE_STATIC_LIBS ON)
find_package(Torch REQUIRED)

# Find TCL
find_package(TCL REQUIRED)

# Add compiler flags
if(NOT MSVC)
    add_compile_options(-Wall -Wextra)
endif()

# Create the test executable
add_executable(test_static main.cpp)

# Include directories
target_include_directories(test_static PRIVATE
    ${TCL_INCLUDE_PATH}
    ${TORCH_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(test_static PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/liblibtorchtcl.a
    ${TCL_LIBRARY}
    "${TORCH_LIBRARIES}"
    -static-libgcc
    -static-libstdc++
)
