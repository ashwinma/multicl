LINKF      = disfd_comm.o disfd_sub.o disfd_subc.o timer.o disfd_sub_opencl.o #disfd_subcu.o
MPIF90     = mpif90 -O3 
MPICC      = mpicc -O3
FF90       = gfortran  -O3
#FF90       = pgf90  -O3
CC         = gcc -O3
#CC         = pgcc -O3
GCC        = gcc -O3
MPI_DIR    = ${MPICH_DIR}
CUDA_HOME  = ${CUDA_INSTALL_PATH}
OPENCL_HOME  = ${SNUCLROOT}

all: disfd vm4dfd_v3 read_dfd

timer.o: timer.c
	$(CC) -c $< -o $@

#disfd_subcu.o: disfd_sub.cu
#	$(CUDA_HOME)/bin/nvcc -gencode arch=compute_35,code=sm_35 -c $< -o $@

disfd_subc.o: disfd_sub.c
	$(CC) -I$(MPI_DIR)/include -c $< -o $@

disfd: disfd.f90 ${LINKF}
	$(MPIF90) -I$(MPI_DIR)/include -lOpenCL -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_DIR)/lib -lmpichf90 -o $@ $@.f90 ${LINKF} 

disfd_sub.o: disfd_sub.f90 disfd_comm.o 
	$(MPIF90) -I$(MPI_DIR)/include -c disfd_sub.f90

disfd_comm.o: disfd_comm.f90
	$(MPIF90) -I$(MPI_DIR)/include -c $< -o $@

vm4dfd_v3: vm4dfd_v3.f90
	$(FF90) -o $@ $< 

read_dfd: read_dfd.f90 xapiir.o sacio.o
	$(FF90) -o $@ $@.f90 xapiir.o sacio.o

xapiir.o: xapiir.f90
	$(FF90) -c $< -o $@
	
sacio.o: sacio.c
	$(CC) -c $< -o $@

disfd_sub_opencl.o: disfd_sub_opencl.c
	$(GCC) -o $@ -c $< -I$(CUDA_HOME)/include 
	
clean:
	rm -f *.mod *.o \
		disfd vm4dfd_v3 read_dfd
