#include <stdio.h>
#include "cudacommon.h"
#include "OptionParser.h"
#include "ResultDatabase.h"

#include "misc_defs.h"

extern unsigned long long mpido;
extern unsigned long long cudado;
extern unsigned long long mpidone;
extern unsigned long long cudadone;
extern pthread_barrier_t mpitest_barrier;
//extern cudaStream_t g_cuda_default_stream;

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific command line argument parsing.
//
//   -nopinned
//   This option controls whether page-locked or "pinned" memory is used.
//   The use of pinned memory typically results in higher bandwidth for data
//   transfer between host and device.
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
    op.addOption("nopinned", OPT_BOOL, "",
                 "disable usage of pinned (pagelocked) memory", 'p');
}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   Measures the bandwidth of the bus connecting the host processor to the
//   OpenCL device.  This benchmark repeatedly transfers data chunks of various
//   sizes across the bus to the OpenCL device, and calculates the bandwidth.
//
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Jeremy Meredith
// Creation: September 08, 2009
//
// Modifications:
//    Jeremy Meredith, Wed Dec  1 17:05:27 EST 2010
//    Added calculation of latency estimate.
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB,
                  OptionParser &op)
{
	//MPIACCGetGPUDevice(&cur_device);
	//MPIACCSetGPUDevice(1);

    const bool verbose = op.getOptionBool("verbose");
    const bool pinned = !op.getOptionBool("nopinned");

	int cur_cpu_core;
	MPIACCGetCPUCore_thread(&cur_cpu_core);
	MPIACCSetCPUCore_thread(1);

	int cur_device;
	cudaGetDevice(&cur_device);
	CHECK_CUDA_ERROR();
	//cudaSetDevice(0);
	CHECK_CUDA_ERROR();
    // Sizes are in kb
    //int nSizes  = 20;
    //int sizes[20] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,
	//	     32768,65536,131072,262144,524288};
    int nSizes  = 10;
#if 0
    int sizes[10] = {1024,2048,4096,8192,16384,
		     32768,65536,131072,262144,524288};
#else
    int sizes[10] = {1,2,4,8,16,
		     32,64,128,256,512};
#endif
    /*
	int maxmsg_sz = op.getOptionInt("MPImaxmsg");
    int nSizes  = 1;
	int sizes[1] = {maxmsg_sz/1024};
	*/
	int iterations = 10;
	int skip = 4;
    long long numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;

    // Create some host memory pattern
    float *hostMem = NULL; 
    if (pinned)
    {
        cudaMallocHost((void**)&hostMem, sizeof(float) * numMaxFloats);
        while (cudaGetLastError() != cudaSuccess)
        {
 	    // drop the size and try again
	    if (verbose) cout << " - dropping size allocating pinned mem\n";
	    --nSizes;
	    if (nSizes < 1)
	    {
		cerr << "Error: Couldn't allocated any pinned buffer\n";
		return;
	    }
	    numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
            cudaMallocHost((void**)&hostMem, sizeof(float) * numMaxFloats);
        }
    }
    else
    {
        hostMem = new float[numMaxFloats];
    }

    for (int i = 0; i < numMaxFloats; i++)
    {
        hostMem[i] = i % 77;
    }

    float *device;
    cudaMalloc((void**)&device, sizeof(float) * numMaxFloats);
    while (cudaGetLastError() != cudaSuccess)
    {
	// drop the size and try again
	if (verbose) cout << " - dropping size allocating device mem\n";
	--nSizes;
	if (nSizes < 1)
	{
	    cerr << "Error: Couldn't allocated any device buffer\n";
	    return;
	}
	numMaxFloats = 1024 * (sizes[nSizes-1]) / 4;
        cudaMalloc((void**)&device, sizeof(float) * numMaxFloats);
    }

    const unsigned int passes = op.getOptionInt("passes");
	cudaStream_t g_cuda_default_stream;
	cudaStreamCreate(&g_cuda_default_stream);
	CHECK_CUDA_ERROR();


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CHECK_CUDA_ERROR();

	if(cudado != 1)
	{
/*	    int rc = pthread_barrier_wait(&mpitest_barrier);
		if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
		{
			printf("Could not wait on barrier\n");
			exit(-1);
	    }*/
	}
	
	int cur_dev;
	cudaGetDevice(&cur_dev);
	CHECK_CUDA_ERROR();
    cout << "[CUDA-Task] Running CUDA benchmarks on device: " << cur_dev << "\n";
	int iter = 0;
    // Three passes, forward and backward both
#ifndef LONG_GPU_RUNS
	do {
    //for (int pass = 0; pass < passes*10565; pass++) {
#else
    for (int pass = 0; pass < passes; pass++)
    {
#endif
        //cout << "Running benchmarks, pass: " << iter << "\n";
        // store the times temporarily to estimate latency
        //float times[nSizes];
        // Step through sizes forward on even passes and backward on odd
		//int i = 0;
		//int i = 4;
		int i = 9;
        //for (int i = 0; i < nSizes; i++)
        {
            int sizeIndex;
            //if ((iter % 2) == 0)
                sizeIndex = i;
            //else
            //    sizeIndex = (nSizes - 1) - i;

            int nbytes = sizes[sizeIndex] * 1024;

            for (int idx = 0; idx < iterations + skip; idx++) 
			{
            if(idx == skip) 
            	cudaEventRecord(start, g_cuda_default_stream);
            if(pinned)
			{
				printf("CUDA Thread Stream (H2D): %p\n", g_cuda_default_stream);
				cudaMemcpyAsync(device, hostMem, nbytes, cudaMemcpyHostToDevice, g_cuda_default_stream);
			}
			else
            	cudaMemcpy(device, hostMem, nbytes, cudaMemcpyHostToDevice);
			}
            cudaEventRecord(stop, g_cuda_default_stream);
            cudaEventSynchronize(stop);
            float t = 0;
            cudaEventElapsedTime(&t, start, stop);
			t/=iterations;
            //times[sizeIndex] = t;

            // Convert to GB/sec
            if (verbose)
            {
                cerr << "size " << sizes[sizeIndex] << "k took " << t <<
                        " ms\n";
            }

            double speed = (double(sizes[sizeIndex]) * 1024. / (1000*1000)) / t;
            char sizeStr[256];
            sprintf(sizeStr, "% 7dkB", sizes[sizeIndex]);
            resultDB.AddResult("DownloadSpeed", sizeStr, "GB/sec", speed);
            resultDB.AddResult("DownloadTime", sizeStr, "ms", t);
        }
		iter++;
	//resultDB.AddResult("DownloadLatencyEstimate", "1-2kb", "ms", times[0]-(times[1]-times[0])/1.);
	//resultDB.AddResult("DownloadLatencyEstimate", "1-4kb", "ms", times[0]-(times[2]-times[0])/3.);
	//resultDB.AddResult("DownloadLatencyEstimate", "2-4kb", "ms", times[1]-(times[2]-times[1])/1.);
#ifdef LONG_GPU_RUNS
    }
	cudadone = 1;
#else
	//}
	} while(mpidone != 1);
#endif

    // Cleanup
	printf("Done with CUDA Tests...\n");
	cudaStreamDestroy(g_cuda_default_stream);
	CHECK_CUDA_ERROR();
    cudaFree((void*)device);
    CHECK_CUDA_ERROR();
    if (pinned)
    {
        cudaFreeHost((void*)hostMem);
        CHECK_CUDA_ERROR();
    }
    else
    {
        delete[] hostMem;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	//cudaSetDevice(cur_device);
    CHECK_CUDA_ERROR();
	MPIACCSetCPUCore_thread(cur_cpu_core);
}
