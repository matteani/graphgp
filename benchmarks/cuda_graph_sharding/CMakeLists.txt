cmake_minimum_required(VERSION 3.24)

project(graphgp_cuda_graph_sharding LANGUAGES CXX)

option(GRAPH_SHARDING_ENABLE_CUDA "Build the CUDA/NCCL benchmark when a CUDA compiler is available" ON)
option(GRAPH_SHARDING_ENABLE_GPU_TESTS "Register GPU smoke tests with CTest" OFF)

# Raven's A100 nodes are Ampere (sm_80). Avoid "native" because CMake is
# commonly run on a GPU-less login node. Users can override this cache entry.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to compile")
endif()

add_library(graph_sharding_planner STATIC src/planner.cpp)
target_include_directories(graph_sharding_planner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(graph_sharding_planner PUBLIC cxx_std_17)
target_compile_options(graph_sharding_planner PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>)

include(CTest)
if(BUILD_TESTING)
  add_executable(graph_sharding_planner_tests tests/planner_tests.cpp)
  target_link_libraries(graph_sharding_planner_tests PRIVATE graph_sharding_planner)
  add_test(NAME graph_sharding_planner_tests COMMAND graph_sharding_planner_tests)
endif()

if(GRAPH_SHARDING_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_path(NCCL_INCLUDE_DIR nccl.h
      HINTS ENV NCCL_ROOT ENV NCCL_HOME
      PATH_SUFFIXES include)
    find_library(NCCL_LIBRARY nccl
      HINTS ENV NCCL_ROOT ENV NCCL_HOME
      PATH_SUFFIXES lib lib64)

    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
      add_executable(cuda_graph_sharding_bench src/main.cu)
      target_include_directories(cuda_graph_sharding_bench PRIVATE ${NCCL_INCLUDE_DIR})
      target_link_libraries(cuda_graph_sharding_bench PRIVATE
        graph_sharding_planner CUDA::cudart CUDA::curand ${NCCL_LIBRARY})
      if(TARGET CUDA::nvtx3)
        target_link_libraries(cuda_graph_sharding_bench PRIVATE CUDA::nvtx3)
        target_compile_definitions(cuda_graph_sharding_bench PRIVATE GRAPH_SHARDING_HAS_NVTX=1)
      endif()
      target_compile_features(cuda_graph_sharding_bench PRIVATE cxx_std_17)
      set_target_properties(cuda_graph_sharding_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED YES
        CUDA_SEPARABLE_COMPILATION ON)

      if(BUILD_TESTING AND GRAPH_SHARDING_ENABLE_GPU_TESTS)
        add_test(NAME cuda_graph_sharding_smoke
          COMMAND cuda_graph_sharding_bench
            --nodes 257 --parents 2 --levels 5 --gpus 1
            --communities 1 --locality 0.5 --warmup 0 --runs 1
            --strategy all)
        add_test(NAME cuda_graph_sharding_iterative_smoke
          COMMAND cuda_graph_sharding_bench
            --nodes 257 --parents 2 --levels 5 --gpus 1
            --communities 1 --locality 0.5 --warmup 1 --runs 2
            --evaluations-per-run 2 --mode iterative --strategy all)
        add_test(NAME cuda_graph_sharding_iterative_two_gpu
          COMMAND cuda_graph_sharding_bench
            --nodes 257 --parents 2 --levels 5 --gpus 2
            --communities 2 --locality 0.5 --warmup 1 --runs 2
            --evaluations-per-run 2 --mode iterative --strategy all)
        add_test(NAME cuda_graph_sharding_iterative_double
          COMMAND cuda_graph_sharding_bench
            --nodes 257 --parents 2 --levels 5 --gpus 1
            --communities 1 --locality 1 --warmup 1 --runs 2
            --evaluations-per-run 2 --mode iterative --strategy all --dtype double)
      endif()
    else()
      message(WARNING "CUDA was found but NCCL was not; set NCCL_ROOT to build cuda_graph_sharding_bench")
    endif()
  else()
    message(STATUS "No CUDA compiler found; building host planner tests only")
  endif()
endif()
