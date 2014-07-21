LINKF     = disfd_comm.o disfd_sub.o disfd_subc.o timer.o 
MPIF90    = mpif90 -O3
MPICC     = mpicc -O3
FF90      = pgf90 -O3
#CC        = opencc -O3
CC        = pgcc -O3
MPI_DIR   = /home/kaixihou/mpich2-1.4.1-pgi-install
CUDA_HOME = /usr/local/cuda-5.0

all: disfd vm4dfd_v3 read_dfd

timer.o: timer.c
	$(CC) -c $< -o $@

#disfd_subcu.o: disfd_sub.cu
	#nvcc --ptxas-options="-v" -gencode arch=compute_20,code=sm_20 -c $< -o $@

disfd_subc.o: disfd_sub.c
	$(CC) -I$(MPI_DIR)/include/ -c $< -o $@

disfd: disfd.f90 ${LINKF} 
	$(MPIF90) -L$(CUDA_HOME)/lib64 -lcudart -L$(MPI_DIR)/lib -lmpichf90 -o $@ $@.f90 ${LINKF} -I$(MPI_DIR)/include/
	
disfd_sub.o: disfd_sub.f90 disfd_comm.o 
	$(MPIF90) -c disfd_sub.f90 -I$(MPI_DIR)/include/

disfd_comm.o: disfd_comm.f90
	#$(MPIF90)  -c $< -o $@ -L$(MPI_DIR)/lib
	$(MPIF90)  -c $< -o $@ -I$(MPI_DIR)/include/

vm4dfd_v3: vm4dfd_v3.f90
	$(FF90)  -o $@ $<

read_dfd: read_dfd.f90 xapiir.o sacio.o
	$(FF90) -o $@ $@.f90 xapiir.o sacio.o
	
xapiir.o: xapiir.f90
	$(FF90)  -c $< -o $@

sacio.o: sacio.c
	$(CC)  -c $< -o $@

clean:
	rm -f *.mod *.o \
		disfd vm4dfd_v3 read_dfd
