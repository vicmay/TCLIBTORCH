cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(libtorchtcl CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set CUDA architecture for your GTX 860M (compute capability 5.0)
# This must be set before finding Torch package
set(CMAKE_CUDA_ARCHITECTURES "50")
set(TORCH_CUDA_ARCH_LIST "5.0")

# Disable nvToolsExt to avoid dependency issues
set(USE_NVTX OFF CACHE BOOL "Disable NVTX support")

# Set LibTorch path
set(LIBTORCH_PATH "${CMAKE_CURRENT_LIST_DIR}/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_PATH})

# Find TCL
find_package(TCL REQUIRED)
message(STATUS "Found TCL: ${TCL_INCLUDE_PATH}")

# Find LibTorch (CUDA-enabled build)
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${TORCH_INCLUDE_DIRS}")

# Find NCCL (now installed)
find_library(NCCL_LIBRARY nccl PATHS /usr/lib/x86_64-linux-gnu)
find_path(NCCL_INCLUDE_DIR nccl.h PATHS /usr/include)

if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
    message(STATUS "NCCL include: ${NCCL_INCLUDE_DIR}")
    set(NCCL_FOUND TRUE)
else()
    message(STATUS "NCCL not found")
    set(NCCL_FOUND FALSE)
endif()

# Add compiler flags
if(NOT MSVC)
    add_compile_options(-Wall -Wextra -fPIC -Wno-unused-parameter -Wno-unused-variable -Wno-unused-but-set-variable)
endif()

# Add CUDA support flags with compatibility fixes for Ubuntu 25.04
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -DWITH_CUDA")

# Fix CUDA compatibility issues with Ubuntu 25.04
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# Use system CUDA toolkit paths
set(CUDA_TOOLKIT_ROOT_DIR "/usr/lib/cuda")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/lib/cuda/include")
# Simplify flags for better compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options=-fno-strict-aliasing")

# Set CUDA compiler to use system-installed version (Ubuntu 25.04 compatible)
set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")

# Add system include directories to avoid header conflicts
include_directories(SYSTEM /usr/lib/cuda/include)
include_directories(SYSTEM /usr/include)

# Add NCCL support if found
if(NCCL_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWITH_NCCL")
endif()

# Create the module with all source files
add_library(libtorchtcl SHARED 
    src/libtorchtcl.cpp
    src/helpers.cpp
    src/neural_device_management.cpp
    src/tensor_core_functions.cpp
    src/training_workflow.cpp
    src/additional_optimizers.cpp
    src/loss_functions.cpp
    src/learning_rate_schedulers.cpp
    src/advanced_layers.cpp
    src/advanced_tensor_ops.cpp
    src/basic_tensor_ops.cpp
    src/signal_processing.cpp
    src/basic_layers.cpp
    src/basic_optimizers.cpp
    src/model_io.cpp
    src/cuda_utils.cpp
    src/linear_algebra.cpp
    src/recurrent_layers.cpp
    src/amp_precision.cpp
src/advanced_tensor_operations.cpp
src/model_checkpointing.cpp
src/distributed_training.cpp
src/tensor_creation_ops.cpp
src/mathematical_operations.cpp
src/activation_functions.cpp
src/extended_convolution_layers.cpp
src/extended_pooling_layers.cpp
src/extended_loss_functions.cpp
src/extended_optimizers.cpp
src/tensor_manipulation_extensions.cpp
src/tensor_info_ops.cpp
src/vision_operations.cpp
src/linear_algebra_extensions.cpp
src/transformer_components.cpp
src/embedding_layers.cpp
src/sparse_tensor_operations.cpp
src/quantization_operations.cpp
src/padding_layers.cpp
src/random_number_generation.cpp
src/automatic_differentiation.cpp
src/memory_performance.cpp
src/advanced_signal_processing.cpp
src/distributed_operations.cpp
src/parameter_parsing.cpp

)

# Include directories
target_include_directories(libtorchtcl PRIVATE
    ${TCL_INCLUDE_PATH}
    ${TORCH_INCLUDE_DIRS}
)

# Add NCCL include if found
if(NCCL_FOUND)
    target_include_directories(libtorchtcl PRIVATE ${NCCL_INCLUDE_DIR})
endif()

# Link libraries but filter out problematic ones
set(FILTERED_TORCH_LIBRARIES "")
foreach(lib ${TORCH_LIBRARIES})
    if(NOT lib MATCHES "nvtoolsext")
        list(APPEND FILTERED_TORCH_LIBRARIES ${lib})
    endif()
endforeach()

target_link_libraries(libtorchtcl 
    ${TCL_LIBRARY}
    ${FILTERED_TORCH_LIBRARIES}
)

# Add NCCL library if found
if(NCCL_FOUND)
    target_link_libraries(libtorchtcl ${NCCL_LIBRARY})
endif()

# Set library properties
set_target_properties(libtorchtcl PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    INSTALL_RPATH "${LIBTORCH_PATH}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print configuration for debugging
message(STATUS "TCL_LIBRARY: ${TCL_LIBRARY}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
if(NCCL_FOUND)
    message(STATUS "Using CUDA-enabled LibTorch build with NCCL support")
else()
    message(STATUS "Using CUDA-enabled LibTorch build")
endif()
