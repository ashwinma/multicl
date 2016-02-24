//-------------------------------------------------------------------------//
//                                                                         //
//  This benchmark is an OpenCL version of the NPB EP code for multiple    //
//  devices. This OpenCL version is developed by the Center for Manycore   //
//  Programming at Seoul National University and derived from the MPI      //
//  Fortran versions in "NPB3.3-MPI" developed by NAS.                     //
//                                                                         //
//  Permission to use, copy, distribute and modify this software for any   //
//  purpose with or without fee is hereby granted. This software is        //
//  provided "as is" without express or implied warranty.                  //
//                                                                         //
//  Information on NPB 3.3, including the technical report, the original   //
//  specifications, source code, results and information on how to submit  //
//  new results, is available at:                                          //
//                                                                         //
//           http://www.nas.nasa.gov/Software/NPB/                         //
//                                                                         //
//  Send comments or suggestions for this OpenCL version to                //
//  cmp@aces.snu.ac.kr                                                     //
//                                                                         //
//          Center for Manycore Programming                                //
//          School of Computer Science and Engineering                     //
//          Seoul National University                                      //
//          Seoul 151-744, Korea                                           //
//                                                                         //
//          E-mail:  cmp@aces.snu.ac.kr                                    //
//                                                                         //
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Authors: Sangmin Seo, Jungwon Kim, Jun Lee, Gangwon Jo, Jeongho Nah,    //
//          and Jaejin Lee                                                 //
//-------------------------------------------------------------------------//

//--------------------------------------------------------------------
//      program EMBAR
//--------------------------------------------------------------------
//  This is the serial version of the APP Benchmark 1,
//  the "embarassingly parallel" benchmark.
//
//
//  M is the Log_2 of the number of complex pairs of uniform (0, 1) random
//  numbers.  MK is the Log_2 of the size of each batch of uniform random
//  numbers.  MK can be set for convenience on a given system, since it does
//  not affect the results.
//--------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "npbparams.h"
#include "ep.h"

#include "type.h"
#include "randdp.h"
#include "timers.h"
#include "print_results.h"

#include <CL/cl.h>
//#include <CL/cl_ext.h>
#include "cl_util.h"
#define TIMER_OPENCL    10
#define TIMER_KERNEL    11
#define TIMER_BUILD     12
#define TIMER_BUFFER    13
#define TIMER_RELEASE   14

#define TIMER_DETAIL

#ifdef TIMER_DETAIL
#define DTIMER_START(id)    if (timers_enabled) timer_start(id)
#define DTIMER_STOP(id)     if (timers_enabled) timer_stop(id)
#else
#define DTIMER_START(id)
#define DTIMER_STOP(id)
#endif

//#define USE_CHUNK_SCHEDULE
//#define MINIMD_SNUCL_OPTIMIZATIONS
//#define DEVICE_FISSION
#define SOCL_OPTIMIZATIONS
#ifdef SOCL_OPTIMIZATIONS
#include "socl.h"
#endif
//--------------------------------------------------------------------
// OpenCL part
//--------------------------------------------------------------------
static cl_device_type    device_type;
static cl_device_id      p_device;
static char             *device_name;
static cl_device_id     *devices;
static cl_uint           num_devices;
static cl_context        context;
static cl_command_queue *cmd_queue;
static cl_program       program;
static cl_kernel        *kernel;
static cl_int            err_code;
static cl_mem *pgq;
static cl_mem *pgsx;
static cl_mem *pgsy;

static int GROUP_SIZE;
static int *gq_size;
static int *gsx_size;
static int *gsy_size;
static int *g_chunk;
static int *g_offset; 
static size_t *globalWorkSize;

void setup_opencl(int argc, char *argv[]);
void release_opencl();

static int    np;
static double q[NQ];
static logical timers_enabled;

#ifdef CLUSTER
int __main(int argc, char** argv)
#else
int main(int argc, char *argv[]) 
#endif
{
  double Mops, t1, t2;
  double tsx, tsy, tm, an, tt, gc;
  double sx_verify_value, sy_verify_value, sx_err, sy_err;
  int    i, nit;
  int    k_offset, j, n;
  logical verified;

  char   size[16];

  FILE *fp;

  if ((fp = fopen("timer.flag", "r")) == NULL) {
    timers_enabled = false;
  } else {
    timers_enabled = true;
    fclose(fp);
  }
  timers_enabled = true;

  //--------------------------------------------------------------------
  //  Because the size of the problem is too large to store in a 32-bit
  //  integer for some classes, we put it into a string (for printing).
  //  Have to strip off the decimal point put in there by the floating
  //  point print statement (internal file)
  //--------------------------------------------------------------------

  sprintf(size, "%15.0lf", pow(2.0, M+1));
  j = 14;
  if (size[j] == '.') j--;
  size[j+1] = '\0';
  printf("\n\n NAS Parallel Benchmarks (NPB3.3-OCL-MD) - EP Benchmark\n");
  printf("\n Number of random numbers generated: %15s\n", size);

  verified = false;

  //--------------------------------------------------------------------
  //  Compute the number of "batches" of random number pairs generated 
  //  per processor. Adjust if the number of processors does not evenly 
  //  divide the total number
  //--------------------------------------------------------------------

  np = NN; 

  setup_opencl(argc, argv);

  timer_clear(0);
  timer_start(0);

  //--------------------------------------------------------------------
  //  Compute AN = A ^ (2 * NK) (mod 2^46).
  //--------------------------------------------------------------------

  t1 = A;

  for (i = 0; i < MK + 1; i++) {
    t2 = randlc(&t1, t1);
  }

  an = t1;
  tt = S;

  //--------------------------------------------------------------------
  //  Each instance of this loop may be performed independently. We compute
  //  the k offsets separately to take into account the fact that some nodes
  //  have more numbers to generate than others
  //--------------------------------------------------------------------

  k_offset = -1;

  DTIMER_START(TIMER_KERNEL);

  // Launch the kernel
  int q_size  = GROUP_SIZE * NQ * sizeof(cl_double);
  int sx_size = GROUP_SIZE * sizeof(cl_double);
  int sy_size = GROUP_SIZE * sizeof(cl_double);
  size_t localWorkSize = GROUP_SIZE;

  for (i = 0; i < num_devices; i++) {
    err_code  = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &pgq[i]);
    err_code |= clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &pgsx[i]);
    err_code |= clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &pgsy[i]);
    err_code |= clSetKernelArg(kernel[i], 3, sizeof(cl_int), &k_offset);
    err_code |= clSetKernelArg(kernel[i], 4, sizeof(cl_double), &an);
    err_code |= clSetKernelArg(kernel[i], 5, sizeof(cl_int), &g_offset[i]);
    err_code |= clSetKernelArg(kernel[i], 6, sizeof(cl_int), &g_chunk[i]);
    
    clu_CheckError(err_code, "clSetKernelArg()");
  }
    
  for (i = 0; i < num_devices; i++) {
  	printf("Before enqueueing to cmd_queue %d\n", i);
    err_code = clEnqueueNDRangeKernel(cmd_queue[i],
                                      kernel[i],
                                      1, NULL,
                                      &globalWorkSize[i], 
                                      &localWorkSize,
                                      0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueNDRangeKernel()");
  }

  double (**gq)[NQ] = (double (**)[NQ])malloc(sizeof(double (*)[NQ]) * num_devices);
  double **gsx = (double **)malloc(sizeof(double *) * num_devices);
  double **gsy = (double **)malloc(sizeof(double *) * num_devices);

  gc  = 0.0;
  tsx = 0.0;
  tsy = 0.0;

  for (i = 0; i < NQ; i++) {
    q[i] = 0.0;
  }

  timer_start(52);
  // Get the result
  for (i = 0; i < num_devices; i++) {
    gq[i] = (double (*)[NQ])malloc(gq_size[i]);
    gsx[i] = (double *)malloc(gsx_size[i]);
    gsy[i] = (double *)malloc(gsy_size[i]);

    err_code = clEnqueueReadBuffer(cmd_queue[i], 
                                   pgq[i],
                                   CL_FALSE,
                                   0, gq_size[i], 
                                   gq[i],
                                   0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueReadbuffer()");

    err_code = clEnqueueReadBuffer(cmd_queue[i], 
                                   pgsx[i],
                                   CL_FALSE,
                                   0, gsx_size[i], 
                                   gsx[i],
                                   0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueReadbuffer()");

    err_code = clEnqueueReadBuffer(cmd_queue[i],
                                   pgsy[i],
                                   CL_FALSE,
                                   0, gsy_size[i], 
                                   gsy[i],
                                   0, NULL, NULL);
    clu_CheckError(err_code, "clEnqueueReadbuffer()");
  }

  for (i = 0; i < num_devices; i++) {
    clFinish(cmd_queue[i]);
  }
  timer_stop(52);
  printf("COMM    :     %9.9lf\n", timer_read(52));

  for (n = 0; n < num_devices; n++) {
    for (i = 0; i < globalWorkSize[n]/localWorkSize; i++) {
      for (j = 0; j < NQ; j++ ){
        q[j] = q[j] + gq[n][i][j];
      }
      tsx = tsx + gsx[n][i];
      tsy = tsy + gsy[n][i];
    }
  }

  DTIMER_STOP(TIMER_KERNEL);

  for (i = 0; i < NQ; i++) {
    gc = gc + q[i];
  }

  timer_stop(0);
  tm = timer_read(0);

  nit = 0;
  verified = true;
  if (M == 24) {
    sx_verify_value = -3.247834652034740e+3;
    sy_verify_value = -6.958407078382297e+3;
  } else if (M == 25) {
    sx_verify_value = -2.863319731645753e+3;
    sy_verify_value = -6.320053679109499e+3;
  } else if (M == 28) {
    sx_verify_value = -4.295875165629892e+3;
    sy_verify_value = -1.580732573678431e+4;
  } else if (M == 30) {
    sx_verify_value =  4.033815542441498e+4;
    sy_verify_value = -2.660669192809235e+4;
  } else if (M == 32) {
    sx_verify_value =  4.764367927995374e+4;
    sy_verify_value = -8.084072988043731e+4;
  } else if (M == 36) {
    sx_verify_value =  1.982481200946593e+5;
    sy_verify_value = -1.020596636361769e+5;
  } else if (M == 40) {
    sx_verify_value = -5.319717441530e+05;
    sy_verify_value = -3.688834557731e+05;
  } else {
    verified = false;
  }

  if (verified) {
    sx_err = fabs((tsx - sx_verify_value) / sx_verify_value);
    sy_err = fabs((tsy - sy_verify_value) / sy_verify_value);
    verified = ((sx_err <= EPSILON) && (sy_err <= EPSILON));
  }

  Mops = pow(2.0, M+1) / tm / 1000000.0;

  printf("\nEP Benchmark Results:\n\n");
  printf("CPU Time =%10.4lf\n", tm);
  printf("N = 2^%5d\n", M);
  printf("No. Gaussian Pairs = %15.0lf\n", gc);
  printf("Sums = %25.15lE %25.15lE\n", tsx, tsy);
  printf("Counts: \n");
  for (i = 0; i < NQ; i++) {
    printf("%3d%15.0lf\n", i, q[i]);
  }

  c_print_results("EP", CLASS, M+1, 0, 0, nit,
      tm, Mops, 
      "Random numbers generated",
      verified, NPBVERSION, COMPILETIME, 
      CS1, CS2, CS3, CS4, CS5, CS6, CS7,
      clu_GetDeviceTypeName(device_type), device_name, num_devices);

  if (timers_enabled) {
    if (tm <= 0.0) tm = 1.0;
    tt = timer_read(0);
    printf("\nTotal time:     %9.3lf (%6.2lf)\n", tt, tt*100.0/tm);
  }

  for (i = 0; i < num_devices; i++) {
    free(gq[i]);
    free(gsx[i]);
    free(gsy[i]);
  }
  free(gq);
  free(gsx);
  free(gsy);
  release_opencl();

  return 0;
}


//---------------------------------------------------------------------
// Set up the OpenCL environment.
//---------------------------------------------------------------------
void setup_opencl(int argc, char *argv[])
{
  int i;
  cl_int err_code;
  char *source_dir = ".";
  int num_subs;
  int num_cus;
  cl_uint max_compute_units;

  if (argc > 1) source_dir = argv[1];

  if (timers_enabled) {
    timer_clear(TIMER_OPENCL);
    timer_clear(TIMER_KERNEL);
    timer_clear(TIMER_BUILD);
    timer_clear(TIMER_BUFFER);
    timer_clear(TIMER_RELEASE);

    timer_start(TIMER_OPENCL);
  }

  // 1. Find the default device type and get a device for the device type
  //    Then, create sub-devices from the parent device.
  //device_type = CL_DEVICE_TYPE_CPU;
  //device_type = CL_DEVICE_TYPE_GPU;
  device_type = CL_DEVICE_TYPE_ALL;
  if(argc <= 2) {
    printf("Device type argument missing!\n");
	exit(-1);
  }
  char *device_type_str = argv[2];
  if(strcmp(device_type_str, "CPU") == 0 || strcmp(device_type_str, "cpu") == 0) {
  	device_type = CL_DEVICE_TYPE_CPU;
  } else if(strcmp(device_type_str, "GPU") == 0 || strcmp(device_type_str, "gpu") == 0) {
  	device_type = CL_DEVICE_TYPE_GPU;
  } else if(strcmp(device_type_str, "ALL") == 0 || strcmp(device_type_str, "all") == 0) {
  	device_type = CL_DEVICE_TYPE_ALL;
  } else {
    printf("Unsupported device type!\n");
	exit(-1);
  }
   
  cl_platform_id platform;
//  err_code = clGetPlatformIDs(1, &platform, NULL);
//  clu_CheckError(err_code, "clGetPlatformIDs()");

#ifdef SOCL_OPTIMIZATIONS
	const char *PLATFORM_NAME = "SOCL Platform";
#else
	const char *PLATFORM_NAME = "SnuCL Single";
#endif
	cl_uint num_platforms;
	cl_platform_id *platforms;
	cl_int errNum = clGetPlatformIDs(0, NULL, &num_platforms);
	clu_CheckError(errNum, "Platform Count");
	printf("Number of platforms: %d\n", num_platforms);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	cl_int err = clGetPlatformIDs(num_platforms, platforms, NULL);
	clu_CheckError(err, "Platform Count");

	int platform_name_size;
	for (i = 0; i < num_platforms; i++) {
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL,
				&platform_name_size);
		clu_CheckError(err, "Platform Info");

		char *platform_name = (char*)malloc(sizeof(char) * platform_name_size);
		err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_size,
				platform_name, NULL);
		clu_CheckError(err, "Platform Info");

		printf("Platform %d: %s\n", i, platform_name);
		if (strcmp(platform_name, PLATFORM_NAME) == 0)
		{
			printf("Choosing Platform %d: %s\n", i, platform_name);
			platform = platforms[i];
		}
		free(platform_name);
	}

	if (platform == NULL) {
		printf("%s platform is not found.\n", PLATFORM_NAME);
		//exit(EXIT_FAILURE);
	}

#ifdef DEVICE_FISSION
  cl_device_id *gpu_devices = NULL;
  cl_device_id *out_cpu_devices = NULL;
  int num_gpu_devices = 0;
  int num_partitions = 0;

  if(device_type == CL_DEVICE_TYPE_GPU || device_type == CL_DEVICE_TYPE_ALL) {
	  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, 
			  NULL, &num_gpu_devices);
	  clu_CheckError(err_code, "clGetDeviceIDs()");
	  gpu_devices = (cl_device_id *)malloc(sizeof(cl_device_id) * 
			  num_gpu_devices);
	  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_gpu_devices,
			  gpu_devices, NULL);
	  clu_CheckError(err_code, "clGetDeviceIDs()");
  }
  if(device_type == CL_DEVICE_TYPE_CPU || device_type == CL_DEVICE_TYPE_ALL) {
	  num_partitions = 2;
	  cl_device_id cpu_device;
	  out_cpu_devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_partitions);

	  err_code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &cpu_device, NULL);
	  clu_CheckError(err_code, "clGetDeviceIDs()");
#if 0
	  cl_device_affinity_domain supported_affinity = 0;
	  err_code = clGetDeviceInfo(cpu_device,
			  CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
			  sizeof(cl_device_affinity_domain),
			  &supported_affinity,
			  NULL);
	  clu_CheckError(err_code, "clGetDeviceInfo");
	  printf("Supported Affinity: %x\n", supported_affinity);

#endif

	  /*cl_device_partition_property  partition_properties[3] = {
		  CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
		  CL_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE,
		  //		CL_DEVICE_AFFINITY_DOMAIN_NUMA,
		  0 };*/
	  cl_device_partition_property  partition_properties[3] = {
	  	  CL_DEVICE_PARTITION_EQUALLY, 8, 0 };
	  int ret_device_count;
	  err_code = clCreateSubDevices(cpu_device,
			  partition_properties,
			  num_partitions,
			  out_cpu_devices,
			  &ret_device_count);
	  clu_CheckError(err_code, "clCreateSubDevices()");
  }
  num_devices = num_gpu_devices + num_partitions;
  devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);

  int dev_id = 0;
  for(i = 0; i < num_partitions; i++) {
  	devices[dev_id++] = out_cpu_devices[i];
  }
  for(i = 0; i < num_gpu_devices; i++) {
  	devices[dev_id++] = gpu_devices[i];
  }
  if(gpu_devices) free(gpu_devices);
  if(out_cpu_devices) free(out_cpu_devices);
#else
  err_code = clGetDeviceIDs(platform, device_type, 0, NULL, &num_devices);
  clu_CheckError(err_code, "clGetDeviceIDs()");

  devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  err_code = clGetDeviceIDs(platform, device_type, num_devices, devices, NULL);
  clu_CheckError(err_code, "clGetDeviceIDs()");
#endif
  max_compute_units = 16;

  // 2. Create a context for devices
#ifdef MINIMD_SNUCL_OPTIMIZATIONS
	cl_context_properties props[5] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		CL_CONTEXT_SCHEDULER,
		//CL_CONTEXT_SCHEDULER_CODE_SEGMENTED_PERF_MODEL,
		//CL_CONTEXT_SCHEDULER_PERF_MODEL,
		CL_CONTEXT_SCHEDULER_FIRST_EPOCH_BASED_PERF_MODEL,
		//CL_CONTEXT_SCHEDULER_ALL_EPOCH_BASED_PERF_MODEL,
		0 };
  context = clCreateContext(props, 
#elif defined(SOCL_OPTIMIZATIONS)
	cl_context_properties props[5] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties)platform,
		CL_CONTEXT_SCHEDULER_SOCL,
		"dmda",
		0 };
  context = clCreateContext(props, 
#else
  context = clCreateContext(NULL, 
#endif
                            num_devices,
                            devices,
                            NULL, NULL, &err_code);
  clu_CheckError(err_code, "clCreateContext()");

  // 4. Build the program
  if (timers_enabled) timer_start(TIMER_BUILD);
  char *source_file = "ep_kernel.cl";
  char build_option[30];
#if 0
  if (device_type == CL_DEVICE_TYPE_CPU) {
    sprintf(build_option, "-DM=%d -I. -DUSE_CPU", M);
#ifdef USE_CHUNK_SCHEDULE
    GROUP_SIZE = 1;
#else
    GROUP_SIZE = 128;
#endif
  } else {
    sprintf(build_option, "-DM=%d -I.", M);
    GROUP_SIZE = 128;
  }
#endif
  sprintf(build_option, "-DM=%d -I.", M);
  GROUP_SIZE = 128;

  //program = clu_MakeProgram(context, devices, source_dir, source_file, build_option);
  program = clu_CreateProgram(context, source_dir, source_file);
  clu_MakeProgram(program, num_devices, devices, source_dir, build_option);

  if (timers_enabled) timer_stop(TIMER_BUILD);

  // 3. Create a command queue
  char *num_command_queues_str = getenv("SNU_NPB_COMMAND_QUEUES");
  cl_uint num_command_queues = 1;
  if(num_command_queues_str != NULL)
  	num_command_queues = atoi(num_command_queues_str);
  cmd_queue = (cl_command_queue*)malloc(sizeof(cl_command_queue)*num_command_queues);
  for (i = 0; i < num_command_queues; i++) {
#ifdef SOCL_OPTIMIZATIONS
	cmd_queue[i] = clCreateCommandQueue(context, NULL,
#else
	cmd_queue[i] = clCreateCommandQueue(context, devices[i % num_devices], 
#endif
    //cmd_queue[i] = clCreateCommandQueue(context, devices[num_devices - 1 - (i%num_devices)], 
#ifdef MINIMD_SNUCL_OPTIMIZATIONS
			CL_QUEUE_AUTO_DEVICE_SELECTION
			| CL_QUEUE_COMPUTE_INTENSIVE
			| CL_QUEUE_ITERATIVE,
#else
	0,
#endif
	&err_code);
    clu_CheckError(err_code, "clCreateCommandQueue()");
  }

  num_devices = num_command_queues;
  // 5. Create buffers
  if (timers_enabled) timer_start(TIMER_BUFFER);

  gq_size  = (int *)malloc(sizeof(int) * num_devices);
  gsx_size = (int *)malloc(sizeof(int) * num_devices);
  gsy_size = (int *)malloc(sizeof(int) * num_devices);
  g_chunk  = (int *)malloc(sizeof(int) * num_devices);
  g_offset = (int *)malloc(sizeof(int) * num_devices);
  globalWorkSize = (size_t *)malloc(sizeof(size_t) * num_devices);
  pgq  = (cl_mem *)malloc(sizeof(cl_mem) * num_devices);
  pgsx = (cl_mem *)malloc(sizeof(cl_mem) * num_devices);
  pgsy = (cl_mem *)malloc(sizeof(cl_mem) * num_devices);

#if 0
  int CPU_WORK = atoi(argv[1]);
  int cpu_cnt = num_devices / 2;
  int gpu_cnt = num_devices - cpu_cnt;
  int cpu_chunk = GROUP_SIZE * CPU_WORK;
  int gpu_chunk = (np - (cpu_cnt * cpu_chunk) + gpu_cnt - 1) / (gpu_cnt);

  printf("CPU_CHUNK:GPU_CHUNK[%d, %d]\n", cpu_chunk, gpu_chunk);

  int offset = 0;
  for (i = 0; i < num_devices-1; i++) {
    g_chunk[i] = i % 2 == 0 ? cpu_chunk : gpu_chunk;
    g_offset[i] = offset;
    offset += g_chunk[i];
  }
  g_chunk[i] = np - offset;
  g_offset[i] = offset;
#else
  int chunk = (np + num_devices - 1) / num_devices;

  int offset = 0;
  for (i = 0; i < num_devices-1; i++) {
    g_chunk[i] = chunk;
    g_offset[i] = offset;
    offset += g_chunk[i];
  }
  g_chunk[i] = np - offset;
  g_offset[i] = offset;
#endif

  for (i = 0; i < num_devices; i++) {
#ifdef USE_CHUNK_SCHEDULE
    globalWorkSize[i] = max_compute_units;
#else
    globalWorkSize[i] = clu_RoundWorkSize((size_t)g_chunk[i], GROUP_SIZE);
#endif

    gq_size[i]  = globalWorkSize[i] / GROUP_SIZE * NQ * sizeof(double);
    gsx_size[i] = globalWorkSize[i] / GROUP_SIZE * sizeof(double);
    gsy_size[i] = globalWorkSize[i] / GROUP_SIZE * sizeof(double);

    pgq[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                            gq_size[i], NULL, &err_code);
    clu_CheckError(err_code, "clCreateBuffer() for pgq");

    pgsx[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                             gsx_size[i], NULL, &err_code);
    clu_CheckError(err_code, "clCreateBuffer() for pgsx");

    pgsy[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                             gsy_size[i], NULL, &err_code);
    clu_CheckError(err_code, "clCreateBuffer() for pgsy");
  }

  if (timers_enabled) timer_stop(TIMER_BUFFER);

  // 6. Create a kernel
  kernel = (cl_kernel *)malloc(sizeof(cl_kernel) * num_devices);
  for (i = 0; i < num_devices; i++) {
    kernel[i] = clCreateKernel(program, "embar", &err_code);
    clu_CheckError(err_code, "clCreateKernel()");
  }
}


void release_opencl()
{
  int i;

  if (timers_enabled) timer_start(TIMER_RELEASE);

  free(gq_size);
  free(gsx_size);
  free(gsy_size);
  free(g_chunk);
  free(g_offset);
  free(globalWorkSize);

  for (i = 0; i < num_devices; i++) {
    clReleaseMemObject(pgq[i]);
    clReleaseMemObject(pgsx[i]);
    clReleaseMemObject(pgsy[i]);
    clReleaseKernel(kernel[i]);
    clReleaseCommandQueue(cmd_queue[i]);
  }
  free(pgq);
  free(pgsx);
  free(pgsy);
  free(kernel);
  free(cmd_queue);

  clReleaseContext(context);
  clReleaseProgram(program);

  free(devices);
  
  if (timers_enabled) timer_stop(TIMER_RELEASE);
  if (timers_enabled) timer_stop(TIMER_OPENCL);

  if (timers_enabled) {
    double tt;
    tt = timer_read(TIMER_OPENCL);
    printf("OpenCL    :     %9.3lf\n", tt);
    tt = timer_read(TIMER_BUILD);
    printf(" - Build  :     %9.3lf\n", tt);
    tt = timer_read(TIMER_KERNEL);
    printf(" - Kernel :     %9.3lf\n", tt);
    tt = timer_read(TIMER_BUFFER);
    printf(" - Buffer :     %9.3lf\n", tt);
    tt = timer_read(TIMER_RELEASE);
    printf(" - Release:     %9.3lf\n", tt);
  }
}

