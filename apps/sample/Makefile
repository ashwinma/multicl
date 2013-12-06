CC=mpic++
#CC=g++
cpu:
	$(CC) -o sample sample.cpp -DSAMPLE_CPU -I$(SNUCLROOT)/inc -lOpenCL

single:
	$(CC) -o sample sample.cpp -DSAMPLE_SINGLE -I$(SNUCLROOT)/inc -lOpenCL

cluster:
	$(CC) -o sample sample.cpp -DSAMPLE_CLUSTER -I$(SNUCLROOT)/inc -L$(SNUCLROOT)/lib -lsnucl_cluster

opencl:
	$(CC) -o sample sample.cpp -I$(SNUCLROOT)/inc -lOpenCL
