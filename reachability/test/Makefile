CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc

SM_TARGETS   = -gencode=arch=compute_52,code=\"sm_52,compute_52\"
SM_DEF     = -DSM520

#SM_TARGETS   = -gencode=arch=compute_70,code=\"sm_70,compute_70\"
#SM_DEF     = -DSM700

GENCODE_SM50    := -gencode arch=compute_52,code=sm_52
#GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_FLAGS   := $(GENCODE_SM50)

NVCCFLAGS += --std=c++11 $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\#

SRC = src
BIN = bin
OBJ = obj

CUB_DIR = /home/zh/cub-1.8.0/

INCLUDES = -I$(CUB_DIR) -I$(CUB_DIR)test -I. -I$(INC)


$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(BIN)/%: $(OBJ)/%.o
	$(NVCC) $(SM_TARGETS) -lcurand $^ -o $@

setup:
	mkdir -p bin obj
