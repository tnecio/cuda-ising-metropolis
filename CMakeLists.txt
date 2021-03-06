cmake_minimum_required(VERSION 3.17)
project(Ising CUDA CXX)
enable_testing()

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDAToolkit REQUIRED)

add_executable(Ising main.cpp simulation.cpp simulation.h simulation_cpu.cpp simulation_cpu.h simulation_gpu.h simulation_gpu.cu)
add_executable(IsingGPU main.cpp simulation.cpp simulation.h simulation_cpu.cpp simulation_cpu.h simulation_gpu.h simulation_gpu.cu)
target_compile_definitions(IsingGPU PRIVATE GPU_ISING=1)

target_link_libraries(Ising PUBLIC CUDA::curand)
target_link_libraries(IsingGPU PUBLIC CUDA::curand)

set_target_properties(
        Ising
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(
        IsingGPU
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(tests)
