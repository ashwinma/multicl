include Makefile.common

TARGET := $(SNUCLROOT)/lib/libsnucl_single.so
CXX    := g++

MORE_INCLUDES += -I$(SNUCLROOT)/gpgpu-sim/v3.x/libopencl -I$(CUDA_INSTALL_PATH)/include -I$(HOME)/opt/include
LIBRARY  += -lEmulatedOpenCL -lhwloc
LINK += -L$(SNUCLROOT)/gpgpu-sim/v3.x/lib/gcc-4.5.3/cuda-5000/release -L$(HOME)/opt/lib

CXX_FLAGS += -DLEGACY_PLATFORM -DSNUCL_DEBUG $(MORE_INCLUDES) -std=c++0x

SOURCES += $(wildcard legacy/*.cpp)

OBJ_DIR=objs/legacy

SUBDIRS=legacy
OBJ_SUBDIRS=$(patsubst %,$(OBJ_DIR)/%,$(SUBDIRS))
DEPS_DIR=$(patsubst %,$(OBJ_DIR)/.deps/%,$(SUBDIRS))

include Makefile.rules
