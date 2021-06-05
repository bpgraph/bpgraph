cmake_minimum_required (VERSION 3.8 FATAL_ERROR)
project (bpgraph LANGUAGES C CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

if (NOT CMAKE_CUDA_COMPILER)
	set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.0/bin/nvcc")
endif()

set(CUDA_NVCC_FLAGS -std=c++11 -L/usr/local/cuda-10.2/lib64 -lcudart -lcuda)
set(CMAKE_CXX_STANDARD 11)

if (CUDA_VERBOSE_PTXAS)
    set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CUDA_NVCC_FLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CUDA_NVCC_FLAGS} -O3 -Wall")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(GENCODE_SM30
    -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35
    -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)
set(GENCODE_SM37
    -gencode=arch=compute_37,code=sm_37 -gencode=arch=compute_37,code=compute_37)
set(GENCODE_SM50
    -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50)
set(GENCODE_SM60
    -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60)
set(GENCODE_SM61
    -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61)
set(GENCODE_SM70
    -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_70,code=compute_70)
set(GENCODE_SM71
    -gencode=arch=compute_71,code=sm_71 -gencode=arch=compute_71,code=compute_71)
set(GENCODE_SM75
    -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75)

option(CUDAMATRIX_GENCODE_SM30 "GENCODE_SM30" OFF)
option(CUDAMATRIX_GENCODE_SM35 "GENCODE_SM35" ON)
option(CUDAMATRIX_GENCODE_SM37 "GENCODE_SM37" OFF)
option(CUDAMATRIX_GENCODE_SM50 "GENCODE_SM50" OFF)
option(CUDAMATRIX_GENCODE_SM60 "GENCODE_SM60" OFF)
option(CUDAMATRIX_GENCODE_SM61 "GENCODE_SM61" OFF)
option(CUDAMATRIX_GENCODE_SM70 "GENCODE_SM70" OFF)
option(CUDAMATRIX_GENCODE_SM71 "GENCODE_SM71" OFF)
option(CUDAMATRIX_GENCODE_SM75 "GENCODE_SM75" OFF)

if (CUDAMATRIX_GENCODE_SM37)
    set(GENCODE ${GENCODE} ${GENCODE_SM37})
endif(CUDAMATRIX_GENCODE_SM37)

if (CUDAMATRIX_GENCODE_SM50)
    set(GENCODE ${GENCODE} ${GENCODE_SM50})
endif(CUDAMATRIX_GENCODE_SM50)

if (CUDAMATRIX_GENCODE_SM60)
    set(GENCODE ${GENCODE} ${GENCODE_SM60})
endif(CUDAMATRIX_GENCODE_SM60)

if (CUDAMATRIX_GENCODE_SM61)
    set(GENCODE ${GENCODE} ${GENCODE_SM61})
endif(CUDAMATRIX_GENCODE_SM61)

if (CUDAMATRIX_GENCODE_SM70)
    set(GENCODE ${GENCODE} ${GENCODE_SM70})
endif(CUDAMATRIX_GENCODE_SM70)

if(CUDAMATRIX_GENCODE_SM71)
    set(GENCODE ${GENCODE} ${GENCODE_SM71})
endif(CUDAMATRIX_GENCODE_SM71)

if(CUDAMATRIX_GENCODE_SM75)
    set(GENCODE ${GENCODE} ${GENCODE_SM75})
endif(CUDAMATRIX_GENCODE_SM75)

include_directories(/usr/local/cuda/include)

set (Boost_FIND_REQUIRED TRUE)
set (Boost_FIND_QUIETLY TRUE)
set (Boost_DEBUG FALSE)
set (Boost_USE_MULTITHREADED TRUE)
set (Boost_USE_STATIC_LIBS TRUE)
set (Boost_ADDITIONAL_VERSIONS "1.39" "1.39.0")

find_package(Boost COMPONENTS  program_options )
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

set(VCC_FLAGS -std=c++11 -lstdc++fs)
set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CC_FLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CC_FLAGS} -O3 -Wall")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


#include_directories(utils)

#add_subdirectory(test)
link_libraries(stdc++fs)
link_libraries(pthread)
#target_link_libraries(bpgraph stdc++fs)

aux_source_directory(. DIR_LIB_SRCS)
cuda_add_executable(bpgraph main.cu
    OPTIONS ${GENCODE} ${CUDA_VERBOSE_PTXAS})

set_target_properties(bpgraph
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#add_executable(bpgraph ${DIR_LIB_SRCS})
