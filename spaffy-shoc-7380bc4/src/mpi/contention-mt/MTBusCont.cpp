#include <stdio.h>
#include <stdlib.h>
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "cuda_runtime.h"

#include "misc_defs.h"

#define MPIACC_TESTS
#ifdef MPIACC_TESTS
#include "mpix.h"
#else
#include "mpi.h"
#endif
extern unsigned long long mpido;
extern unsigned long long cudado;
extern unsigned long long mpidone;
extern unsigned long long cudadone;
extern pthread_barrier_t mpitest_barrier;

using namespace std;
#ifdef MPIACC_TESTS
#define CUDA_CHECK_ERR(msg) \
	{ \
		cudaError_t err = cudaGetLastError(); \
		if(err != cudaSuccess) { \
			fprintf(stderr, "CUDA error: %s %s.\n", msg, cudaGetErrorString( err) ); \
			exit(EXIT_FAILURE); \
		} \
	}

#endif
// ****************************************************************************
// Function: MPITest 
//
// Purpose:
//   dumps results from GPU and MPI groups in sequence
//
// Arguments:
//   op: benchmark options
//   resultDB: the parallel results data base
//   numtasks: total tasks that will run the benchmark
//   myrank: my rank
//   mypair: my pair to communicate to
//   newcomm: the context for the ranks
//
// Returns: 
//
// Creation: July 08, 2009
//
// Modifications:
//
// ****************************************************************************
void MPITest(OptionParser &op, ResultDatabase &resultDB, int numtasks, int myrank,
                int mypair, MPI_Comm newcomm)
{
    int msgsize;
    int skip=10,i=0,j=0;
    int minmsg_sz = op.getOptionInt("MPIminmsg");
    int maxmsg_sz = op.getOptionInt("MPImaxmsg");
    string gpu_tests = op.getOptionString("gpuTests");
    int iterations = 2;
    //int iterations = op.getOptionInt("MPIiter");
    int npasses = op.getOptionInt("passes");
    char *recvbuf = NULL;
	char *sendbuf = NULL;
	char *sendptr, *recvptr;
    char sizeStr[256];
    double minlat=0, maxlat=0, avglat=0, latency, t_start, t_end;
    MPI_Status reqstat;
    MPI_Request req;

#ifdef MPIACC_TESTS
	cudaSetDevice(0);
	CUDA_CHECK_ERR("CUDA Set device");
	size_t mem_free;
	size_t mem_total;
	MPIX_Buffertype sendtype = MPIX_GPU_CUDA;
	MPIX_Buffertype recvtype = MPIX_GPU_CUDA;
	if(gpu_tests == "TEST_H2D_MPIACC_SEND" || gpu_tests == "TEST_D2H_MPIACC_SEND")
	{
		sendtype = MPIX_GPU_CUDA;
		recvtype = MPIX_CPU;
	}
	else if(gpu_tests == "TEST_H2D_MPIACC_RECV" || gpu_tests == "TEST_D2H_MPIACC_RECV")
	{
		sendtype = MPIX_CPU;
		recvtype = MPIX_GPU_CUDA;
	}
	if(cudado != 1)
	{
	    /*int rc = pthread_barrier_wait(&mpitest_barrier);
		if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
		{
			printf("Could not wait on barrier\n");
			exit(-1);
		}*/
	}
	if(sendtype != MPIX_CPU)
	{
		cudaMemGetInfo(&mem_free, &mem_total);
		CUDA_CHECK_ERR("CUDA Meminfo");
		printf("GPU memory available %d MB out of %d MB\n", mem_free/(1024*1024), mem_total/(1024*1024));
		CUDA_CHECK_ERR("CUDA Meminfo");
		cudaMalloc((void **)&sendbuf, 16);
		CUDA_CHECK_ERR("CUDA Malloc sendbuf 16");
		cudaMemGetInfo(&mem_free, &mem_total);
		CUDA_CHECK_ERR("CUDA Meminfo");
		printf("GPU memory available %d MB out of %d MB\n", mem_free/(1024*1024), mem_total/(1024*1024));
		cudaFree(sendbuf);
		CUDA_CHECK_ERR("CUDA Free 16");
		cudaMalloc((void **)&sendbuf, 1024 * 1024);
		CUDA_CHECK_ERR("CUDA Malloc sendbuf 1M");
		cudaMemGetInfo(&mem_free, &mem_total);
		CUDA_CHECK_ERR("CUDA Meminfo");
		printf("GPU memory available %d MB out of %d MB\n", mem_free/(1024*1024), mem_total/(1024*1024));
		cudaFree(sendbuf);
		CUDA_CHECK_ERR("CUDA Free 1M");
		printf("Trying to allocate %d MB\n", (maxmsg_sz*16)/(1024*1024));
		cudaMalloc((void **)&sendbuf, maxmsg_sz*16);
		CUDA_CHECK_ERR("CUDA Malloc sendbuf");
	}
	else
	{
    	sendbuf = (char *) malloc(maxmsg_sz*16);
	}
	if(recvtype != MPIX_CPU)
	{
		cudaMemGetInfo(&mem_free, &mem_total);
		CUDA_CHECK_ERR("CUDA Meminfo");
		printf("GPU memory available %d MB out of %d MB\n", mem_free/(1024*1024), mem_total/(1024*1024));
		CUDA_CHECK_ERR("CUDA Malloc before recvbuf");
		cudaMalloc((void **)&recvbuf, maxmsg_sz*16);
		CUDA_CHECK_ERR("CUDA Malloc recvbuf");
	}
	else
	{
    	recvbuf = (char *) malloc(maxmsg_sz*16);
	}
	cudaStream_t d_stream;
	cudaStreamCreate(&d_stream);
	CUDA_CHECK_ERR("CUDA Stream create");
#else
    recvbuf = (char *) malloc(maxmsg_sz*16);
    sendbuf = (char *) malloc(maxmsg_sz*16);
#endif
    if (recvbuf==NULL || sendbuf == NULL)
    {
        printf("\n%d:memory allocation in %s failed",myrank,__FUNCTION__);
        fflush(stdout);
        exit(1);
    }

#ifdef MPIACC_TESTS
	printf("[MPI-Task] Flag values (mpido, cudado): (%d, %d)\n", mpido, cudado);
	printf("[MPI-Task] Waiting for CUDA flag...\n");
	//while(cudado != 1);
	if(cudado != 1)
	{
	    int rc = pthread_barrier_wait(&mpitest_barrier);
		if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
		{
			printf("Could not wait on barrier\n");
			exit(-1);
	    }
	}
	printf("[MPI-Task] Set MPI flag...\n");
	//mpido = 1;
	if(mpido != 1)
	{
	}

	printf("[MPI-Task] All clear to start MPI task... \n");
#endif //MPIACC_TESTS
#ifndef LONG_GPU_RUNS
    for(int passes = 0; passes < npasses; passes++) {
#else
	//do {
    for(int passes = 0; passes < npasses; passes++) {
#endif
	for (msgsize = minmsg_sz; msgsize <= maxmsg_sz; 
         msgsize = (msgsize ? msgsize * 2 : msgsize + 1)) 
    {

        MPI_Barrier(newcomm);

        if (myrank < mypair) 
        {
            for (i = 0; i < iterations + skip; i++) 
            {
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf+msgsize*((j++)%16);
                recvptr = recvbuf+msgsize*((j++)%16);
#ifdef MPIACC_TESTS
				//cudaMemcpyAsync(recvptr, sendptr, msgsize, cudaMemcpyHostToDevice, d_stream);
				//CUDA_CHECK_ERR("CUDA memcpy async");
				MPIX_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, 0, 0, 0, 0);
                MPIX_Recv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, 0, 0, 0, 0);
#else
                MPI_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm);
                MPI_Recv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat);
#endif
            }
#ifdef MPIACC_TESTS
			//cudaStreamSynchronize(d_stream);
			//CUDA_CHECK_ERR("CUDA stream sync");
#endif
            t_end = MPI_Wtime();
        }
        else if (myrank > mypair) 
        {
            for (i = 0; i < iterations + skip; i++) 
            {
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf+msgsize*((j++)%16);
                recvptr = recvbuf+msgsize*((j++)%16);
#ifdef MPIACC_TESTS
				//cudaMemcpyAsync(recvptr, sendptr, msgsize, cudaMemcpyHostToDevice, d_stream);
				//CUDA_CHECK_ERR("CUDA memcpy async");
				MPIX_Recv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, 0, 0, 0, 0);
                MPIX_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, 0, 0, 0, 0);
#else
                MPI_Recv(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat);
                MPI_Send(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm);
#endif
            }
#ifdef MPIACC_TESTS
			//cudaStreamSynchronize(d_stream);
			//CUDA_CHECK_ERR("CUDA stream sync");
#endif
            t_end = MPI_Wtime();
        }
        else 
        {
            for (i = 0; i < iterations + skip; i++) 
            {
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf+msgsize*((j++)%16);
                recvptr = recvbuf+msgsize*((j++)%16);
#ifdef MPIACC_TESTS
				//cudaMemcpyAsync(recvptr, sendptr, msgsize, cudaMemcpyHostToDevice, d_stream);
				//CUDA_CHECK_ERR("CUDA memcpy async");
				MPIX_Irecv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &req, recvtype, 0, 0, 0, 0, 0);
                MPIX_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, 0, 0, 0, 0);
#else
                MPI_Irecv(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &req);
                MPI_Send(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm);
#endif
                MPI_Wait(&req, &reqstat);
            }
#ifdef MPIACC_TESTS
			//cudaStreamSynchronize(d_stream);
			//CUDA_CHECK_ERR("CUDA stream sync");
#endif
            t_end = MPI_Wtime();
        }

        latency = (t_end - t_start) * 1e6 / (2.0 * iterations);
        sprintf(sizeStr, "% 6dkB", msgsize);
        resultDB.AddResult("MPI Latency", sizeStr, "MicroSeconds", latency);
        printf("MPI Latency for %s: %g MicroSeconds\n", sizeStr, latency);


        //MPI_Reduce(&latency,&maxlat,1,MPI_DOUBLE, MPI_MAX, 0, newcomm);
        //MPI_Reduce(&latency,&minlat,1,MPI_DOUBLE, MPI_MIN, 0, newcomm);
        //MPI_Reduce(&latency,&avglat,1,MPI_DOUBLE, MPI_SUM, 0, newcomm);
        //MPI_Comm_size(newcomm,&j);
        //avglat/=j;
        j=0;
        //if (myrank == 0) 
        //{
        //    printf("\n%d\t%f\t%f\t%f",msgsize,minlat,avglat,maxlat);
        //    fflush(stdout);
        //}
    }
#ifdef LONG_GPU_RUNS
	}
	//} while(cudadone != 1);
#else
	}
	mpidone = 1;
#endif
	//mpido = 0;
#ifdef MPIACC_TESTS
	cudaStreamDestroy(d_stream);
	CUDA_CHECK_ERR("CUDA destroy stream");
	if(recvtype != MPIX_CPU)
	{
		cudaFree(recvbuf);
		CUDA_CHECK_ERR("CUDA Free recvbuf");
	}
	else
	{
		free(recvbuf);
	}
	if(sendtype != MPIX_CPU)
	{
		cudaFree(sendbuf);
		CUDA_CHECK_ERR("CUDA Free sendbuf");
	}
	else
	{
    	free(sendbuf);
	}
#else
    free(recvbuf);
    free(sendbuf);
#endif
}
