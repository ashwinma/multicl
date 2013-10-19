#include <stdio.h>
#include <stdlib.h>
#include "ResultDatabase.h"
#include "OptionParser.h"
#include "cuda_runtime.h"
#include <unistd.h>

#include "misc_defs.h"
#include "misc_mpi_defs.h"
#define MESSAGE_ALIGNMENT 64
#define MAX_MSG_SIZE (1<<22)
#define MYBUFSIZE (MAX_MSG_SIZE * 16 + MESSAGE_ALIGNMENT)

extern void *g_gpuhostbuf1;
extern void *g_gpuhostbuf2;
#define MPIACC_SEND
//#define MPICUDA_SYNCSEND
//#define MPICUDA_ASYNCSEND

#ifndef MPICUDA_ASYNCSEND
const int CMD_QUEUE_COUNT = 0;
#else
const int CMD_QUEUE_COUNT = 16;
#endif
const int foo_threshold = 128 * 1024;
const int PACKET_SIZE = 512 * 1024;

extern cudaStream_t g_gpustreams[CMD_QUEUE_COUNT];

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
#define MPI_CHECK_ERR(err, str) \
	if (err != MPI_SUCCESS) \
{ \
	fprintf(stderr, "MPI Error %d: %s\n", err, str); \
}
#ifdef MPIACC_TESTS
#define CUDA_CHECK_ERR(msg) \
	{ \
		cudaError_t err = cudaGetLastError(); \
		if(err != cudaSuccess) { \
			fprintf(stderr, "CUDA error: %s %s.\n", msg, cudaGetErrorString( err) ); \
			exit(EXIT_FAILURE); \
		} \
	}


int MPIX_Send_sync(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPIX_Buffertype buffertype, int cuda_devid, cudaStream_t cuda_stream, cl_mem ocl_mem, cl_context ocl_ctx, cl_device_id ocl_devid, cl_command_queue ocl_cmdq)
{
	char *s_hbuf = (char *)g_gpuhostbuf1; 
	char *r_hbuf = (char *)g_gpuhostbuf2; 
	int size = -1;
	int err = MPI_Type_size(datatype, &size);
	size *= count;
	cudaMemcpy(s_hbuf, buf, size, cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERR("Memcpy");
	err = MPI_Send(s_hbuf, size, datatype, dest, tag, comm);
	MPI_CHECK_ERR(err, "MPI Send Error by proc 0");
}

int MPIX_Recv_sync(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status, MPIX_Buffertype buffertype, int cuda_devid, cudaStream_t cuda_stream, cl_mem ocl_mem, cl_context ocl_ctx, cl_device_id ocl_devid, cl_command_queue ocl_cmdq)
{
	char *s_hbuf = (char *)g_gpuhostbuf1; 
	char *r_hbuf = (char *)g_gpuhostbuf2; 
	int size = -1;
	int err = MPI_Type_size(datatype, &size);
	size *= count;
	err = MPI_Recv(r_hbuf, size, datatype, source, tag, comm, status);
	MPI_CHECK_ERR(err, "MPI Recv Error by proc 1");
	//cudaMemcpy(buf, r_hbuf, size, cudaMemcpyHostToDevice);
	//CUDA_CHECK_ERR("Memcpy");
}

int MPIX_Send_async(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPIX_Buffertype buffertype, int cuda_devid, cudaStream_t cuda_stream, cl_mem ocl_mem, cl_context ocl_ctx, cl_device_id ocl_devid, cl_command_queue ocl_cmdq)
{	
    double t_start_test = 0.0, t_end_test = 0.0;
	char *s_hbuf = (char *)g_gpuhostbuf1; 
	char *r_hbuf = (char *)g_gpuhostbuf2; 
	int err;
	int stream_idx = 0;
	int packet_count = 0;
	int packet_idx = 0;
	cudaError_t cuda_err;
	int flag;

    MPI_Status reqstat;
    MPI_Status status[64 * 64];
    MPI_Status sstat[64 * 64];
    MPI_Status rstat[64 * 64];
    MPI_Request sreq[64 * 64];
    MPI_Request rreq[64 * 64];
	int size = -1;
	err = MPI_Type_size(datatype, &size);
	size *= count;
	packet_count = size/PACKET_SIZE;
		if(packet_count <= 0)
		{
			packet_count = 1;
		}
	if(size >= PACKET_SIZE)
	{
		/* Send */
		for(packet_idx = 0; packet_idx < packet_count; packet_idx++)
		{
			size_t offset = packet_idx * PACKET_SIZE;
			stream_idx = packet_idx % CMD_QUEUE_COUNT;
			cudaMemcpyAsync(s_hbuf + offset, buf + offset, PACKET_SIZE, cudaMemcpyDeviceToHost, g_gpustreams[stream_idx]);
			CUDA_CHECK_ERR("[MPI] Async D2H Copy");
		}
		for(packet_idx = 0; packet_idx < packet_count; packet_idx++)
		{
			//printf("packet ID: %d/%d\n", packet_idx, packet_count);
			size_t offset = packet_idx * PACKET_SIZE;
			stream_idx = packet_idx % CMD_QUEUE_COUNT;
			do
			{
				t_start_test = MPI_Wtime();
				cuda_err = cudaStreamQuery(g_gpustreams[stream_idx]);
				t_end_test = MPI_Wtime();
				double latency_test = (t_end_test - t_start_test) * 1e6 ;
				/*if(latency_test > 10)
				  {
				  fprintf(stdout, "Iter: %d; SyncTime for StreamID(%d/%d): %-*d%*.*f\n", i, stream_idx, packet_idx, 10, size, FIELD_WIDTH,
				  FLOAT_PRECISION, latency_test);
				  }*/

				if(packet_idx > 0)MPI_Test(&sreq[packet_idx - 1], &flag, &sstat[packet_idx - 1]);
			}while(cuda_err != cudaSuccess);
			MPI_Isend(s_hbuf + offset, PACKET_SIZE, datatype, dest, tag, comm, &sreq[packet_idx]);
		}
		//printf("Waiting for MPI\n");
		MPI_Waitall(packet_count, sreq, sstat);
	}
	else
	{
		//t_start_test = MPI_Wtime();
		cudaMemcpyAsync(s_hbuf, buf, size, cudaMemcpyDeviceToHost, g_gpustreams[0]);
		CUDA_CHECK_ERR("[MPI] Async D2H Copy");
		cudaStreamSynchronize(g_gpustreams[0]);
		//cudaDeviceSynchronize();
		CUDA_CHECK_ERR("GPU Device Synch");
		/*	t_end_test = MPI_Wtime();
			double latency_test = (t_end_test - t_start_test) * 1e6;
			fprintf(stdout, "Test-%-*d%*.*f\n", 10, size, FIELD_WIDTH,
			FLOAT_PRECISION, latency_test);
			fflush(stdout);*/
		err = MPI_Send(s_hbuf, size, datatype, dest, tag, comm);
		MPI_CHECK_ERR(err, "MPI Send Error by proc 0");
	}
}

int MPIX_Recv_async(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status, MPIX_Buffertype buffertype, int cuda_devid, cudaStream_t cuda_stream, cl_mem ocl_mem, cl_context ocl_ctx, cl_device_id ocl_devid, cl_command_queue ocl_cmdq)
{
    double t_start_test = 0.0, t_end_test = 0.0;
	char *s_hbuf = (char *)g_gpuhostbuf1; 
	char *r_hbuf = (char *)g_gpuhostbuf2; 
	int err;
	int stream_idx = 0;
	int packet_count = 0;
	int packet_idx = 0;
	cudaError_t cuda_err;
	int flag;

    MPI_Status reqstat;
    //MPI_Status status[64 * 64];
    MPI_Status sstat[64 * 64];
    MPI_Status rstat[64 * 64];
    MPI_Request sreq[64 * 64];
    MPI_Request rreq[64 * 64];
	int size = -1;
	err = MPI_Type_size(datatype, &size);
	size *= count;
	packet_count = size/PACKET_SIZE;
		if(packet_count <= 0)
		{
			packet_count = 1;
		}
	if(size >= PACKET_SIZE)
	{
		/* Recv */
		for(packet_idx = 0; packet_idx < packet_count; packet_idx++)
		{
			size_t offset = packet_idx * PACKET_SIZE;
			stream_idx = packet_idx % CMD_QUEUE_COUNT;
			MPI_Irecv(r_hbuf + offset, PACKET_SIZE, datatype, source, tag, comm, &rreq[packet_idx]);
		}
		for(packet_idx = 0; packet_idx < packet_count; packet_idx++)
		{
			size_t offset = packet_idx * PACKET_SIZE;
			stream_idx = packet_idx % CMD_QUEUE_COUNT;
			MPI_Wait(&rreq[packet_idx], &rstat[packet_idx]);
			//cudaMemcpyAsync(buf + offset, r_hbuf + offset, PACKET_SIZE, cudaMemcpyHostToDevice, g_gpustreams[stream_idx]);
			//CUDA_CHECK_ERR("Async H2D Copy");
		}
		for(stream_idx = 0; stream_idx < CMD_QUEUE_COUNT; stream_idx++)
		{
			//cudaStreamSynchronize(g_gpustreams[stream_idx]);
			//CUDA_CHECK_ERR("GPU Stream Synch");
		}
	}
	else
	{
		//t_start_test = MPI_Wtime();
		/*	t_end_test = MPI_Wtime();
			double latency_test = (t_end_test - t_start_test) * 1e6;
			fprintf(stdout, "Test-%-*d%*.*f\n", 10, size, FIELD_WIDTH,
			FLOAT_PRECISION, latency_test);
			fflush(stdout);*/
		err = MPI_Recv(r_hbuf, size, datatype, source, tag, comm, &reqstat);
		MPI_CHECK_ERR(err, "MPI Recv Error by proc 0");
		//cudaMemcpyAsync(buf, r_hbuf, size, cudaMemcpyHostToDevice, g_gpustreams[0]);
		//CUDA_CHECK_ERR("Async H2D Copy");
		//cudaStreamSynchronize(g_gpustreams[0]);
		//cudaDeviceSynchronize();
		//CUDA_CHECK_ERR("GPU Device Synch");
	}
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
    int skip=4,i=0,j=0;
    int minmsg_sz = op.getOptionInt("MPIminmsg");
    int maxmsg_sz = op.getOptionInt("MPImaxmsg");
    //int minmsg_sz = op.getOptionInt("MPIminmsg");
    //int maxmsg_sz = op.getOptionInt("MPIminmsg");
    string gpu_tests = op.getOptionString("gpuTests");
    int iterations = 10;
    //int iterations = op.getOptionInt("MPIiter");
    int npasses = op.getOptionInt("passes");
    char *recvbuf = NULL;
	char *sendbuf = NULL;
	char *sendptr, *recvptr;
    char sizeStr[256];
    double minlat=0, maxlat=0, avglat=0, latency, t_start, t_end, t_total_start, t_total_end;
    MPI_Status reqstat;
    MPI_Request req;

#ifdef MPIACC_TESTS
	int cur_cpu_core;
	MPIACCGetCPUCore_thread(&cur_cpu_core);
	MPIACCSetCPUCore_thread(0);
	//cudaSetDevice(0);
	CUDA_CHECK_ERR("CUDA Set device");
	size_t mem_free;
	size_t mem_total;
	MPIX_Buffertype sendtype = MPIX_CPU;
	MPIX_Buffertype recvtype = MPIX_CPU;
	if(gpu_tests == "TEST_H2D_MPIACC_SEND" || gpu_tests == "TEST_D2H_MPIACC_SEND")
	{
		sendtype = MPIX_GPU_CUDA;
		//sendtype = MPIX_CPU;
		recvtype = MPIX_CPU;
	}
	else if(gpu_tests == "TEST_H2D_MPIACC_RECV" || gpu_tests == "TEST_D2H_MPIACC_RECV")
	{
		sendtype = MPIX_CPU;
		//recvtype = MPIX_CPU;
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
		/*CUDA_CHECK_ERR("CUDA Meminfo");
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
		CUDA_CHECK_ERR("CUDA Free 1M");*/
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
	//cudaStream_t d_stream;
	//cudaStreamCreate(&d_stream);
	//CUDA_CHECK_ERR("CUDA Stream create");
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
	int cur_dev;
	cudaGetDevice(&cur_dev);
	CUDA_CHECK_ERR("Get CUDA Device");
    cout << "[MPI-Task] Running MPI-ACC benchmarks on device: " << cur_dev << "\n";
#endif //MPIACC_TESTS
	t_total_start = MPI_Wtime();
#ifndef LONG_GPU_RUNS
    for(int passes = 0; passes < npasses; passes++) {
#else
	//do {
    for(int passes = 0; passes < npasses; passes++) {
#endif
	msgsize = maxmsg_sz; 
	//for (msgsize = minmsg_sz; msgsize <= maxmsg_sz; 
    //     msgsize = (msgsize ? msgsize * 2 : msgsize + 1)) 
    {

        MPI_Barrier(newcomm);

        if (myrank < mypair) 
        {
            for (i = 0; i < iterations + skip; i++) 
            {
    //    		printf("MPI Send Iter: (%d, %d)\n", passes, i);
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf;//+msgsize*((j++)%16);
                recvptr = recvbuf;//+msgsize*((j++)%16);
#ifdef MPIACC_TESTS
				//cudaMemcpyAsync(recvptr, sendptr, msgsize, cudaMemcpyHostToDevice, d_stream);
				//CUDA_CHECK_ERR("CUDA memcpy async");
				#ifdef MPIACC_SEND
				MPIX_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
				MPIX_Recv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
				#elif defined(MPICUDA_SYNCSEND)
				MPIX_Send_sync(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
                MPIX_Recv_sync(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
				#elif defined(MPICUDA_ASYNCSEND)
				MPIX_Send_async(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
                MPIX_Recv_async(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
				#endif
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
    //    		printf("MPI Recv Iter: (%d, %d)\n", passes, i);
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf;//+msgsize*((j++)%16);
                recvptr = recvbuf;//+msgsize*((j++)%16);
#ifdef MPIACC_TESTS
				//cudaMemcpyAsync(recvptr, sendptr, msgsize, cudaMemcpyHostToDevice, d_stream);
				//CUDA_CHECK_ERR("CUDA memcpy async");
				#ifdef MPIACC_SEND
				MPIX_Recv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
                MPIX_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
				#elif defined(MPICUDA_SYNCSEND)
				MPIX_Recv_sync(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
                MPIX_Send_sync(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
				#elif defined(MPICUDA_ASYNCSEND)
				MPIX_Recv_async(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &reqstat, recvtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
                MPIX_Send_async(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
				#endif
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
			printf("Wrong Conditions\n");
            for (i = 0; i < iterations + skip; i++) 
            {
                if (i == skip) t_start = MPI_Wtime();
                sendptr = sendbuf+msgsize*((j++)%16);
                recvptr = recvbuf+msgsize*((j++)%16);
#ifdef MPIACC_TESTS
				//cudaMemcpyAsync(recvptr, sendptr, msgsize, cudaMemcpyHostToDevice, d_stream);
				//CUDA_CHECK_ERR("CUDA memcpy async");
				MPIX_Irecv(recvptr, msgsize, MPI_CHAR, mypair, 1, newcomm, 
                                &req, recvtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
                MPIX_Send(sendptr, msgsize, MPI_CHAR, mypair, 1, newcomm, sendtype, 0, (cudaStream_t)(void *)-1, 0, 0, 0, 0);
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
        //printf("MPI Latency for %s: %g MicroSeconds\n", sizeStr, latency);


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
	t_total_end = MPI_Wtime();
    double t_total = (t_total_end - t_total_start) * 1e6;
    printf("Total MPI Time: %g us\n", t_total);
	//mpido = 0;
#ifdef MPIACC_TESTS
	MPIACCSetCPUCore_thread(cur_cpu_core);
	//cudaStreamDestroy(d_stream);
	//CUDA_CHECK_ERR("CUDA destroy stream");
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
