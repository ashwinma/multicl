# === Basics ===
CC       = gcc
CXX      = g++
LD       = g++
AR       = ar
RANLIB   = ranlib

CPPFLAGS += -I${SNUCLROOT}/inc -I$(SHOC_ROOT)/src/common -I${SHOC_ROOT}/config -I/opt/AMDAPP/include/
CFLAGS   += -g -O2
CXXFLAGS += -g -O2
NVCXXFLAGS += -g -O2
ARFLAGS  = rcv
LDFLAGS  = 
LIBS     = -L$(SHOC_ROOT)/lib  -lrt

USE_MPI         = yes
MPICXX          = /home/aaji/opt/bin/mpicxx

OCL_CPPFLAGS    += -I${SNUCLROOT}/inc -I${SHOC_ROOT}/src/opencl/common
OCL_LIBS        = -lOpenCL

NVCC            = /home/aaji/opt/cuda/bin/nvcc
CUDA_CXX        = /home/aaji/opt/cuda/bin/nvcc
CUDA_INC        = -I/home/aaji/opt/cuda/include
CUDA_CPPFLAGS   += -gencode=arch=compute_12,code=sm_12  -gencode=arch=compute_13,code=sm_13  -gencode=arch=compute_20,code=sm_20  -gencode=arch=compute_20,code=compute_20 -I${SHOC_ROOT}/src/cuda/include

USE_CUDA        = yes
ifeq ($(USE_CUDA),yes)
CUDA_LIBS       := $(shell /home/aaji/opt/cuda/bin/nvcc -dryrun bogus.cu 2>&1 | grep LIBRARIES | sed 's/^.*LIBRARIES=//')
else
CUDA_LIBS       =
endif



