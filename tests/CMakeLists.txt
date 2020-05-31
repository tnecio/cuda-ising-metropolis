add_executable(Mre mre_main.cpp mre.cuh mre.cu)

add_executable(Tests tests.cpp test_simulation.h test_simulation_cpu.h test_simulation_cpu_threads.h test_simulation_gpu.h test_simulation_gpu_cuda.h utils.h
        ../simulation.cpp ../simulation_cpu.cpp ../simulation_gpu.cpp ../simulation_gpu_cuda.cu ../simulation_cpu_threads.cpp)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(Mre PUBLIC CUDA::curand)

add_test(Test Tests)

target_link_libraries(Tests PUBLIC CUDA::curand)