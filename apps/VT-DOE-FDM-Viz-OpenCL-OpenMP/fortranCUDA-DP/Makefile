linkf=disfd_comm.o disfd_sub.o  disfd_subcu.o disfd_sub_stress_kernels.o disfd_sub_velocity_kernels.o disfd_gpu_marshaling.o  disfd_subc.o 
MPIF90=mpif90 -ffree-line-length-none -ffixed-line-length-none -O3
MPICC=mpicc -O3
FF90=gfortran  -ffree-line-length-none -ffixed-line-length-none -O3
CC=gcc -O3
CUDA_DIR=/cuda-5.0
USR_LIB=/usr/lib64
MPI_DIR=/home/sriramc/install/mpich2-1.4.1
#NVCC_OPTS=-gencode arch=compute_35,code=sm_35 --ptxas-options="-v"
NVCC_OPTS=--ptxas-options="-v"
all : timer.o disfd vm4dfd_v3 read_dfd clean

timer.o: timer.c
	$(CC) -c timer.c

disfd_subcu.o: disfd_sub.cu 
	#$(MPICC) -c disfd_sub.c -o disfd_subc.o
	nvcc  $(NVCC_OPTS) -I $(CUDA_DIR)/include -I $(MPI_DIR)/include   -L $(MPI_DIR)/lib -lmpich -lmpichcxx -c disfd_sub.cu -o disfd_subcu.o


disfd_sub_stress_kernels.o: disfd_sub_stress_kernels.cu
	nvcc $(NVCC_OPTS) -c disfd_sub_stress_kernels.cu -o disfd_sub_stress_kernels.o
disfd_sub_velocity_kernels.o: disfd_sub_velocity_kernels.cu
	nvcc $(NVCC_OPTS) -c disfd_sub_velocity_kernels.cu -o disfd_sub_velocity_kernels.o
disfd_gpu_marshaling.o: disfd_gpu_marshaling.cu
	nvcc $(NVCC_OPTS) -c disfd_gpu_marshaling.cu -o disfd_gpu_marshaling.o

disfd_subc.o: disfd_sub.c
	$(CC) -I $(CUDA_DIR)/include -I $(MPI_DIR)/include  -c disfd_sub.c -o disfd_subc.o

disfd: disfd.f90 ${linkf} 
	$(MPIF90) -cpp -L $(MPI_DIR)/lib -lmpichf90  -L$(USR_LIB) -lstdc++ -L $(CUDA_DIR)/lib64 -lcuda -lcudart -o $@ $@.f90 ${linkf} timer.o
disfd_sub.o: disfd_sub.f90 disfd_comm.o 
	$(MPIF90) -cpp  -c disfd_sub.f90
disfd_comm.o: disfd_comm.f90
	$(MPIF90) -c disfd_comm.f90

vm4dfd_v3: vm4dfd_v3.f90
	$(FF90) -o $@ $@.f90 

read_dfd: read_dfd.f90 xapiir.o sacio.o
	$(FF90) -o $@ $@.f90 xapiir.o sacio.o
xapiir.o: xapiir.f90
	$(FF90) -c xapiir.f90
sacio.o: sacio.c
	$(CC) -c sacio.c
clean:
	rm -f *.mod *.o
