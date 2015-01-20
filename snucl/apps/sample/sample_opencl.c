//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
//
/**************************************************************************/
/* Author: Ashwin Aji                                                     */
/* Organization: Department of Computer Science, Virginia Tech            */
/*                                                                        */
/* All copyrights reserved.                                               */
/*                                                                        */
/**************************************************************************/
# include <stdlib.h>
# include <stdio.h>
# include <string.h>

#include "sample.h"
#include "RealTimer.h"
#define CHECK_ERROR(err, str) \
	if (err != CL_SUCCESS) \
{\
	fprintf(stderr, "Error in \"%s\", %d : %s\n", str, err, clCheckErrorString(err)); \
}

#define CHECK_NULL_ERROR(err, str) \
	if (err == NULL) \
{\
	fprintf(stderr, "Error creating memory objects in \"%s\"\n", str); \
}
#define SNUCL_PERF_MODEL_OPTIMIZATION

struct __dim3 {
	int x;
	int y;
	int z;
	
	__dim3() {}
	__dim3(int _x = 1, int _y = 1, int _z = 1) 
	{
		x = _x;
		y = _y;
		z = _z;
	}
};

typedef struct __dim3 dim3;

RealTimer kernelTimer[NUM_COMMAND_QUEUES];
#ifdef __cplusplus
extern "C" {
#endif
void logger_log_timing(const char *msg, double time) 
{
	fprintf(stdout, "[Sample] %s %g\n", msg, time);
}

void write_output(float *arr, int size, const char *filename)
{
	FILE *fp;
	if((fp = fopen(filename, "a")) == NULL)
	{
		fprintf(stderr, "File write error!\n");
	}
	int i;
	for(i = 0; i < size; i++)
	{
		fprintf(fp, "%f ", arr[i]);
		if( i%10 == 0)
			fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	fclose(fp);
}

struct timeval t1, t2;
//---------------------------

void record_time(double* time);


size_t LoadProgramSource(const char *filename1, const char **progSrc) 
{
	FILE *f1 = fopen(filename1, "r");
	fseek(f1, 0, SEEK_END);
	size_t len1 = (size_t) ftell(f1);

	*progSrc = (const char *) malloc(sizeof(char)*(len1));

	rewind(f1);
	fread((void *) *progSrc, len1, 1, f1);
	fclose(f1);
	return len1;
}

// opencl initialization
void init_cl(const size_t mem_size) 
{
	int i;
	cl_int errNum;
	cl_int err;
	// launch the device
	size_t progLen;
	const char *progSrc;
	cl_uint num_platforms;
	size_t            platform_name_size;
	char*             platform_name;
	cl_platform_id *platforms;
	const char *PLATFORM_NAME = "SnuCL Single";
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		Init(&kernelTimer[i]);
	}
	errNum = clGetPlatformIDs(0, NULL, &num_platforms);
	CHECK_ERROR(errNum, "Platform Count");
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, platforms, NULL);
	CHECK_ERROR(err, "Platform Count");

	for (i = 0; i < num_platforms; i++) {
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL,
				&platform_name_size);
		CHECK_ERROR(err, "Platform Info");

		platform_name = (char*)malloc(sizeof(char) * platform_name_size);
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_size,
				platform_name, NULL);
		CHECK_ERROR(err, "Platform Info");

		printf("Platform %d: %s\n", i, platform_name);
		if (strcmp(platform_name, PLATFORM_NAME) == 0)
		{
			printf("Choosing Platform %d: %s\n", i, platform_name);
			_cl_firstPlatform = platforms[i];
		}
		free(platform_name);
	}

	if (_cl_firstPlatform == NULL) {
		printf("%s platform is not found.\n", PLATFORM_NAME);
		//exit(EXIT_FAILURE);
	}
	/*errNum = clGetPlatformIDs(1, &_cl_firstPlatform, NULL);
	  if(errNum != CL_SUCCESS)
	  {
	  fprintf(stderr, "Failed to find any available OpenCL platform!\n");
	  }*/
	cl_uint num_devices;
	errNum = clGetDeviceIDs(_cl_firstPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
	if(errNum != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to find any available OpenCL device!\n");
	}
	_cl_devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
	errNum = clGetDeviceIDs(_cl_firstPlatform, CL_DEVICE_TYPE_ALL, num_devices, _cl_devices, NULL);
	if(errNum != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to find any available OpenCL device!\n");
	}
#ifdef SNUCL_PERF_MODEL_OPTIMIZATION
	cl_context_properties props[5] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) _cl_firstPlatform,
		CL_CONTEXT_SCHEDULER,
		//CL_CONTEXT_SCHEDULER_CODE_SEGMENTED_PERF_MODEL,
		//CL_CONTEXT_SCHEDULER_PERF_MODEL,
		CL_CONTEXT_SCHEDULER_FIRST_EPOCH_BASED_PERF_MODEL,
		//CL_CONTEXT_SCHEDULER_ALL_EPOCH_BASED_PERF_MODEL,
		0 };
	_cl_context = clCreateContext(props, num_devices, _cl_devices, NULL, NULL, &errNum);
#else
	_cl_context = clCreateContext(NULL, num_devices, _cl_devices, NULL, NULL, &errNum);
#endif
	if(errNum != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to create GPU context!\n");
	}

	//_cl_commandQueue = clCreateCommandQueue(_cl_context, _cl_firstDevice, CL_QUEUE_PROFILING_ENABLE, NULL);
#ifndef DISFD_SINGLE_COMMANDQ
	for(i = 0; i < NUM_COMMAND_QUEUES; i++)
	{
		int chosen_dev_id = 0; 
		//= (i + 2) % num_devices;
		if(i == 0) 
		{
			char *foo = getenv("SNUCL_DEV_0");
			if(foo != NULL)
				chosen_dev_id = atoi(foo);
		}
		else if (i == 1) 
		{
			char *foo = getenv("SNUCL_DEV_1");
			if(foo != NULL)
				chosen_dev_id = atoi(foo);
		}
		printf("[OpenCL] %dth command queue uses Dev ID %d\n", i, chosen_dev_id);
		_cl_commandQueues[i] = clCreateCommandQueue(_cl_context, 
				_cl_devices[chosen_dev_id], 
#ifdef SNUCL_PERF_MODEL_OPTIMIZATION
				CL_QUEUE_AUTO_DEVICE_SELECTION | 
				CL_QUEUE_ITERATIVE | 
				//CL_QUEUE_IO_INTENSIVE | 
				CL_QUEUE_COMPUTE_INTENSIVE | 
#endif
				CL_QUEUE_PROFILING_ENABLE, NULL);
		if(_cl_commandQueues[i] == NULL)
		{
			fprintf(stderr, "Failed to create commandQueue for the %dth device!\n", i);
		}
	}
#else
	int chosen_dev_id = 0;
	char *foo = getenv("SNUCL_DEV_0");
	if(foo != NULL)
		chosen_dev_id = atoi(foo);
	printf("[OpenCL] All command queue uses Dev ID %d\n", chosen_dev_id);
	_cl_commandQueues[0] = clCreateCommandQueue(_cl_context, _cl_devices[chosen_dev_id], 
#ifdef SNUCL_PERF_MODEL_OPTIMIZATION
			//CL_QUEUE_AUTO_DEVICE_SELECTION | 
			//CL_QUEUE_ITERATIVE | 
			//CL_QUEUE_IO_INTENSIVE | 
			//CL_QUEUE_COMPUTE_INTENSIVE | 
#endif
			CL_QUEUE_PROFILING_ENABLE, NULL);
	if(_cl_commandQueues[0] == NULL)
	{
		fprintf(stderr, "Failed to create commandQueue for the 0th device!\n");
	}
	errNum = clRetainCommandQueue(_cl_commandQueues[0]);
	if(errNum != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to retain cmd queue\n");
	}
	_cl_commandQueues[1] = _cl_commandQueues[0];
#endif
	progLen = LoadProgramSource("sample_kernel.cl", &progSrc);
	printf("[OpenCL] program loaded\n");
	_cl_program = clCreateProgramWithSource(_cl_context, 1, &progSrc, &progLen, NULL);
	if(_cl_program == NULL)
	{
		fprintf(stderr, "Failed to create CL program from source!\n");
	}

	free((void *)progSrc);
	// -cl-mad-enable
	// -cl-opt-disable
	errNum = clBuildProgram(_cl_program, 0, NULL, "-I.", NULL, NULL);
	printf("[OpenCL] program built!\n");
	//errNum = clBuildProgram(_cl_program, num_devices, _cl_devices, "", NULL, NULL);
	//errNum = clBuildProgram(_cl_program, 1, &_cl_firstDevice, "-cl-opt-disable", NULL, NULL);
	//if(errNum != CL_SUCCESS)
	{
		for(i = 0; i < num_devices; i++)
		{
			char buildLog[16384];
			cl_int err = clGetProgramBuildInfo(_cl_program, _cl_devices[i], CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
			fprintf(stderr, "Build Log for device %d:\n", i);
			buildLog[16383] = '\0';
			fprintf(stderr, "%s\n", buildLog);
			// clReleaseProgram(_cl_program);
		}
	}

	_marshal_kernel = clCreateKernel(_cl_program, "marshal_kernel", NULL);
	_compute_kernel = clCreateKernel(_cl_program, "compute_kernel", NULL);
	if(_marshal_kernel == NULL)
		fprintf(stderr, "Failed to create kernel marshal_kernel!\n");
	if(_compute_kernel == NULL)
		fprintf(stderr, "Failed to create kernel compute_kernel!\n");
	float *_buf_src = (float *)malloc(mem_size);
	int j = 0;
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
        for(j = 0; j < mem_size / sizeof(float); j++) {
            _buf_src[j] = 42.0f;
        }

		_bufs[i] = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, mem_size, NULL, NULL);
		CHECK_NULL_ERROR(_bufs[i], "_bufs");
    	errNum = clEnqueueWriteBuffer(_cl_commandQueues[i], _bufs[i], CL_TRUE, 0, mem_size, _buf_src, 0, NULL, NULL);
    	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, _buf");
	}
	free(_buf_src);

	printf("[OpenCL] device initialization success!\n");
}

// opencl release
void release_cl() 
{
	int i;
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) 
	{
		printf("Compute time for Q[%d]: %g\n", i, Elapsed(&kernelTimer[i]));
		clReleaseMemObject(_bufs[i]);
	}
	if(_compute_kernel)clReleaseKernel(_compute_kernel);
	if(_marshal_kernel)clReleaseKernel(_marshal_kernel);
	clReleaseProgram(_cl_program);
	//clReleaseCommandQueue(_cl_commandQueue);
	for(i = 0; i < NUM_COMMAND_QUEUES; i++)
	{
		clReleaseCommandQueue(_cl_commandQueues[i]);
	}
	clReleaseContext(_cl_context);
	free(_cl_devices);
	printf("[OpenCL] context all released!\n");
}

void compute(const int iter_scale, const size_t const_mem_size, const size_t mem_size)
{
	int blocks = 14 * 8;
	int threads = 128;
	size_t dimBlock[3] = {threads, 1, 1};
	size_t dimGrid1[3] = {blocks, 1, 1};
	cl_int errNum;
	int nIters = iter_scale;
	float v1 = 3.75;
	float v2 = 0.355;
	long int d_size = const_mem_size / sizeof(float);
	//OpenCL code
	int i = 0;
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		int arg_idx = 0;
		errNum = clSetKernelArg(_compute_kernel, arg_idx++, sizeof(cl_mem), &_bufs[i]);
		errNum = clSetKernelArg(_compute_kernel, arg_idx++, sizeof(int), &nIters);
		errNum = clSetKernelArg(_compute_kernel, arg_idx++, sizeof(float), &v1);
		errNum = clSetKernelArg(_compute_kernel, arg_idx++, sizeof(float), &v2);
		errNum = clSetKernelArg(_compute_kernel, arg_idx++, sizeof(long int), &d_size);
		if(errNum != CL_SUCCESS)
		{
			fprintf(stderr, "Error: setting kernel _compute_kernel arguments!\n");
		}
		localWorkSize[0] = dimBlock[0];
		localWorkSize[1] = dimBlock[1];
		localWorkSize[2] = dimBlock[2];
		globalWorkSize[0] = dimGrid1[0]*localWorkSize[0];
		globalWorkSize[1] = dimGrid1[1]*localWorkSize[1];
		globalWorkSize[2] = dimGrid1[2]*localWorkSize[2];
		errNum = clEnqueueNDRangeKernel(_cl_commandQueues[i], _compute_kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
		if(errNum != CL_SUCCESS)
		{
			fprintf(stderr, "Error: queuing kernel _compute_kernel for execution!\n");
		}
	}
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		Start(&kernelTimer[i]);
	}
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		errNum = clFinish(_cl_commandQueues[i]);
		if(errNum != CL_SUCCESS) {
			fprintf(stderr, "Error: finishing velocity for execution!\n");
		}
		Stop(&kernelTimer[i]);
	}
	printf("[OpenCL] compute success!\n");
	return;
}

int main () {
	int iter_scale = 16 * 1024;
	const size_t const_mem_size = 32 * 1024 * 1024;
	const size_t start_mem_size = 2 * 1024 * 1024;
	init_cl(const_mem_size);
	//for(iter_scale = 512; iter_scale <= 2048; iter_scale *= 2) {
	for(size_t mem_size = start_mem_size; mem_size <= 16 * start_mem_size; mem_size *= 2) {
		compute(iter_scale, start_mem_size, mem_size);
	}
	release_cl();
	return 0;
}
		
#ifdef __cplusplus
}
#endif
