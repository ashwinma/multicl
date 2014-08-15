# Makefile for OpenCL version of miniMD
SHELL = /bin/sh
#.IGNORE:

# OpenCL directories.

OPENCL_HOME  = $(SNUCLROOT)
CL_INCLUDE = -I$(OPENCL_HOME)/inc
# AMD
#CL_INCLUDE = -I /opt/AMDAPP/include/
#CL_LIB = -L /opt/AMDAPP/lib/x86_64/

# Intel
#CL_INCLUDE = -I /opt/opencl/intel-2012/include/
#CL_LIB = -L /opt/opencl/intel-2012/lib64/ -L /opt/opencl/intel-2012/lib64/OpenCL/vendors/intel/

# NVIDIA
#CL_INCLUDE = -I /opt/cuda/toolkit/4.1.28/cuda/include/
#CL_LIB = -L /opt/opencl/intel-ocl-sdk-1.5/lib64/

# Necessary on my SNB machine to help things compile...
GCC_VERSION=4.4
GCC_FLAGS=-gcc-name=/opt/apps/gcc/4.5.3/bin/gcc -gxx-name=/opt/apps/gcc/4.5.3/bin/g++
#GCC_FLAGS=-gcc-name=/usr/bin/gcc-${GCC_VERSION} -gxx-name=/usr/bin/g++-${GCC_VERSION}

# System-specific settings.
CC =		mpicxx 
#CCFLAGS =	-O3 $(CL_INCLUDE) $(OVERRIDE)
CCFLAGS =	-O3 -DMPICH_IGNORE_CXX_SEEK  -DMDPREC=1 -DPRECMPI=MPI_FLOAT -DPREC_TIMER -I$(CL_INCLUDE)
LINK =		mpicxx
LINKFLAGS =	-O3 
#LINKFLAGS =	-O3 $(CL_LIB) $(OVERRIDE)
USRLIB = #-lOpenCL 
SYSLIB =	-lpthread -lOpenCL -lrt
SIZE =		size

# Link rule
$(EXE):	$(OBJ)
	$(LINK) $(LINKFLAGS) $(OBJ) $(USRLIB) $(SYSLIB) -o $(EXE)
	$(SIZE) $(EXE)

# Compilation rules
.cpp.o:
	$(CC) $(CCFLAGS) -c $<

# Individual dependencies
$(OBJ): $(INC)
