nvcc --ptxas-options=-v -L /usr/lib64 -lcuda -arch=sm_20 microbm_main.cu -o microbm_main
