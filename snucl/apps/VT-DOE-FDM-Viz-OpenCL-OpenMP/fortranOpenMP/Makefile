#
# © 2013.  Virginia Polytechnic Institute & State University
# 
# This OpenMP-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
#
linkf=disfd_comm.o disfd_sub.o disfd_subc.o timw.o timer.o
MPIF90=mpif90 -ffree-line-length-none -ffixed-line-length-none -O3 -g -fopenmp 
MPICC=mpicc -g -O3
FF90=gfortran -ffree-line-length-none -ffixed-line-length-none -O3 -g -fopenmp
CC=gcc -fopenmp -g -O3

#User Defined path for MPI_DIR
MPI_INSTALL_PATH=${MPICH_DIR}
#MPI_INSTALL_PATH=/home/aaniket/mpich2-install
#MPI_INSTALL_PATH=/usr/... 

all : disfd vm4dfd_v3 read_dfd 

timer.o: timer.c
	$(CC) -c $< -o $@

disfd_subc.o: disfd_sub_openmp.c
	$(CC) -I$(MPI_INSTALL_PATH)/include -c disfd_sub_openmp.c -o disfd_subc.o

disfd: disfd.f90 ${linkf} 
	$(MPIF90) -L$(MPI_INSTALL_PATH)/lib -lmpichf90 -o $@ $@.f90 ${linkf}

disfd_sub.o: disfd_sub.f90 disfd_comm.o 
	$(MPIF90) -c disfd_sub.f90

disfd_comm.o: disfd_comm.f90
	$(MPIF90) -c disfd_comm.f90

timw.o: timw.c
	$(CC) -c timw.c

vm4dfd_v3: vm4dfd_v3.f90
	$(FF90) -o $@ $@.f90 

read_dfd: read_dfd.f90 xapiir.o sacio.o
	$(FF90) -o $@ $@.f90 xapiir.o sacio.o

xapiir.o: xapiir.f90
	$(FF90) -c xapiir.f90

sacio.o: sacio.c
	$(CC) -c sacio.c

clean:
	rm -f *.mod *.o \
		disfd vm4dfd_v3 read_dfd
