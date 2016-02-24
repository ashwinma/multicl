CC=mpic++
#CC=g++

single:
	$(CC) -o sample sample.cpp -DSAMPLE_SINGLE -I$(SNUCLROOT)/inc -lOpenCL

cluster:
	$(CC) -o sample sample.cpp -DSAMPLE_CLUSTER -I$(SNUCLROOT)/inc -L$(SNUCLROOT)/lib -lsnucl_cluster

opencl:
	$(CC) -o sample sample.cpp -I$(SNUCLROOT)/inc -lOpenCL

cpu:
	$(CC) -o sample sample.cpp -DSAMPLE_CPU -I$(SNUCLROOT)/inc -lOpenCL

default: single
