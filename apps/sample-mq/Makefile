MPI_DIR    = ${MPICH_DIR}
CUDA_HOME  = ${CUDA_INSTALL_PATH}
OPENCL_HOME  = ${SNUCLROOT}
MY_INCLUDES=-I$(HOME)/opt/include -I/usr/include -I$(MPI_DIR)/include -I. -I$(OPENCL_HOME)/inc -I$(CUDA_HOME)/include 
#CC=mpic++ -O3 -g $(MY_INCLUDES)
CC=g++ -O3 -g $(MY_INCLUDES)

single:
	$(CC) -o sample -g sample_opencl.c -DSAMPLE_SINGLE -I$(SNUCLROOT)/inc -lOpenCL -lpthread

cluster:
	$(CC) -o sample -g sample_opencl.c -DSAMPLE_CLUSTER -I$(SNUCLROOT)/inc -L$(SNUCLROOT)/lib -lsnucl_cluster

opencl:
	$(CC) -o sample -g sample_opencl.c -I$(SNUCLROOT)/inc -lOpenCL

cpu:
	$(CC) -o sample -g sample_opencl.c -DSAMPLE_CPU -I$(SNUCLROOT)/inc -lOpenCL

default: single

clean:
	rm ./sample
