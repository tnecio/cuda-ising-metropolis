cmake_minimum_required(VERSION 3.17)
project(Ising CUDA CXX)
enable_testing()

set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(CUDAToolkit REQUIRED)

add_executable(Ising main.cpp simulation.cpp simulation.h main_cpu.cpp main_cpu.h main_gpu.cu main_gpu.cuh simulation_cpu.cpp simulation_cpu.h simulation_cpu_threads.cpp simulation_cpu_threads.h simulation_gpu.cpp simulation_gpu.h simulation_gpu_cuda.cu simulation_gpu_cuda.cuh)

target_link_libraries(Ising PUBLIC CUDA::curand)

set_target_properties(
        Ising
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_subdirectory(tests)
