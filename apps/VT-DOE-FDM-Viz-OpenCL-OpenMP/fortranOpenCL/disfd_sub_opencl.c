//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
//
/**************************************************************************/
/* Author: Kaixi Hou                                                      */
/* Organization: Department of Computer Science, Virginia Tech            */
/*                                                                        */
/* All copyrights reserved.                                               */
/*                                                                        */
/**************************************************************************/
# include <stdlib.h>
# include <stdio.h>

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/opencl.h>
#endif
#include "papi_interface.h"
#include "RealTimer.h"
#define CHECK_ERROR(err, str) \
	if (err != CL_SUCCESS) \
	{\
		fprintf(stderr, "Error in \"%s\", %d\n", str, err); \
	}

#define CHECK_NULL_ERROR(err, str) \
	if (err == NULL) \
	{\
		fprintf(stderr, "Error creating memory objects in \"%s\"\n", str); \
	}
//#define DISFD_DEBUG
//#define DISFD_PAPI
//#define DISFD_USE_OPTIMIZED
#define DISFD_H2D_SYNC_KERNEL
/***********************************************/
/* for debug: check the output                 */
/***********************************************/
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

#define NUM_COMMAND_QUEUES	2
cl_platform_id     _cl_firstPlatform;
cl_device_id       _cl_firstDevice;
cl_device_id       *_cl_devices;
cl_context         _cl_context;
//cl_command_queue   _cl_commandQueue;
cl_command_queue _cl_commandQueues[NUM_COMMAND_QUEUES];
cl_program         _cl_program;

size_t globalWorkSize[3];
size_t localWorkSize[3];

RealTimer kernelTimerStress[NUM_COMMAND_QUEUES];
RealTimer kernelTimerVelocity[NUM_COMMAND_QUEUES];
RealTimer h2dTimerStress;
RealTimer h2dTimerVelocity;
RealTimer d2hTimerStress;
RealTimer d2hTimerVelocity;

cl_kernel _cl_Kernel_velocity_inner_IC;
cl_kernel _cl_Kernel_velocity_inner_IIC;
cl_kernel _cl_Kernel_vel_PmlX_IC;
cl_kernel _cl_Kernel_vel_PmlY_IC;
cl_kernel _cl_Kernel_vel_PmlX_IIC;
cl_kernel _cl_Kernel_vel_PmlY_IIC;
cl_kernel _cl_Kernel_vel_PmlZ_IIC;
cl_kernel _cl_Kernel_stress_norm_xy_IC;
cl_kernel _cl_Kernel_stress_xz_yz_IC;
cl_kernel _cl_Kernel_stress_resetVars;
cl_kernel _cl_Kernel_stress_norm_PmlX_IC;
cl_kernel _cl_Kernel_stress_norm_PmlY_IC;
cl_kernel _cl_Kernel_stress_xy_PmlX_IC;
cl_kernel _cl_Kernel_stress_xy_PmlY_IC;
cl_kernel _cl_Kernel_stress_xz_PmlX_IC;
cl_kernel _cl_Kernel_stress_xz_PmlY_IC;
cl_kernel _cl_Kernel_stress_yz_PmlX_IC;
cl_kernel _cl_Kernel_stress_yz_PmlY_IC;
cl_kernel _cl_Kernel_stress_norm_xy_II;
cl_kernel _cl_Kernel_stress_xz_yz_IIC;
cl_kernel _cl_Kernel_stress_norm_PmlX_IIC;
cl_kernel _cl_Kernel_stress_norm_PmlY_II;
cl_kernel _cl_Kernel_stress_norm_PmlZ_IIC;
cl_kernel _cl_Kernel_stress_xy_PmlX_IIC;
cl_kernel _cl_Kernel_stress_xy_PmlY_IIC;
cl_kernel _cl_Kernel_stress_xy_PmlZ_II;
cl_kernel _cl_Kernel_stress_xz_PmlX_IIC;
cl_kernel _cl_Kernel_stress_xz_PmlY_IIC;
cl_kernel _cl_Kernel_stress_xz_PmlZ_IIC;
cl_kernel _cl_Kernel_stress_yz_PmlX_IIC;
cl_kernel _cl_Kernel_stress_yz_PmlY_IIC;
cl_kernel _cl_Kernel_stress_yz_PmlZ_IIC;

//device memory pointers
static cl_mem nd1_velD;
static cl_mem nd1_txyD;
static cl_mem nd1_txzD;
static cl_mem nd1_tyyD;
static cl_mem nd1_tyzD;
static cl_mem rhoD;
static cl_mem drvh1D;
static cl_mem drti1D;
static cl_mem drth1D;
static cl_mem damp1_xD;
static cl_mem damp1_yD;
static cl_mem idmat1D;
static cl_mem dxi1D;
static cl_mem dyi1D;
static cl_mem dzi1D;
static cl_mem dxh1D;
static cl_mem dyh1D;
static cl_mem dzh1D;
static cl_mem t1xxD;
static cl_mem t1xyD;
static cl_mem t1xzD;
static cl_mem t1yyD;
static cl_mem t1yzD;
static cl_mem t1zzD;
static cl_mem t1xx_pxD;
static cl_mem t1xy_pxD;
static cl_mem t1xz_pxD;
static cl_mem t1yy_pxD;
static cl_mem qt1xx_pxD;
static cl_mem qt1xy_pxD;
static cl_mem qt1xz_pxD;
static cl_mem qt1yy_pxD;
static cl_mem t1xx_pyD;
static cl_mem t1xy_pyD;
static cl_mem t1yy_pyD;
static cl_mem t1yz_pyD;
static cl_mem qt1xx_pyD;
static cl_mem qt1xy_pyD;
static cl_mem qt1yy_pyD;
static cl_mem qt1yz_pyD;
static cl_mem qt1xxD;
static cl_mem qt1xyD;
static cl_mem qt1xzD;
static cl_mem qt1yyD;
static cl_mem qt1yzD;
static cl_mem qt1zzD;
static cl_mem clamdaD;
static cl_mem cmuD;
static cl_mem epdtD;
static cl_mem qwpD;
static cl_mem qwsD;
static cl_mem qwt1D;
static cl_mem qwt2D;

static cl_mem v1xD;    //output
static cl_mem v1yD;
static cl_mem v1zD;
static cl_mem v1x_pxD;
static cl_mem v1y_pxD;
static cl_mem v1z_pxD;
static cl_mem v1x_pyD;
static cl_mem v1y_pyD;
static cl_mem v1z_pyD;

//for inner_II---------------------------------------------------------
static cl_mem nd2_velD;
static cl_mem nd2_txyD;  //int[18]
static cl_mem nd2_txzD;  //int[18]
static cl_mem nd2_tyyD;  //int[18]
static cl_mem nd2_tyzD;  //int[18]

static cl_mem drvh2D;
static cl_mem drti2D;
static cl_mem drth2D; 	//float[mw2_pml1,0:1]

static cl_mem idmat2D;
static cl_mem damp2_xD;
static cl_mem damp2_yD;
static cl_mem damp2_zD;
static cl_mem dxi2D;
static cl_mem dyi2D;
static cl_mem dzi2D;
static cl_mem dxh2D;
static cl_mem dyh2D;
static cl_mem dzh2D;
static cl_mem t2xxD;
static cl_mem t2xyD;
static cl_mem t2xzD;
static cl_mem t2yyD;
static cl_mem t2yzD;
static cl_mem t2zzD;
static cl_mem qt2xxD;
static cl_mem qt2xyD;
static cl_mem qt2xzD;
static cl_mem qt2yyD;
static cl_mem qt2yzD;
static cl_mem qt2zzD;

static cl_mem t2xx_pxD;
static cl_mem t2xy_pxD;
static cl_mem t2xz_pxD;
static cl_mem t2yy_pxD;
static cl_mem qt2xx_pxD;
static cl_mem qt2xy_pxD;
static cl_mem qt2xz_pxD;
static cl_mem qt2yy_pxD;

static cl_mem t2xx_pyD;
static cl_mem t2xy_pyD;
static cl_mem t2yy_pyD;
static cl_mem t2yz_pyD;
static cl_mem qt2xx_pyD;
static cl_mem qt2xy_pyD;
static cl_mem qt2yy_pyD;
static cl_mem qt2yz_pyD;

static cl_mem t2xx_pzD;
static cl_mem t2xz_pzD;
static cl_mem t2yz_pzD;
static cl_mem t2zz_pzD;
static cl_mem qt2xx_pzD;
static cl_mem qt2xz_pzD;
static cl_mem qt2yz_pzD;
static cl_mem qt2zz_pzD;

static cl_mem v2xD;		//output
static cl_mem v2yD;
static cl_mem v2zD;
static cl_mem v2x_pxD;
static cl_mem v2y_pxD;
static cl_mem v2z_pxD;
static cl_mem v2x_pyD;
static cl_mem v2y_pyD;
static cl_mem v2z_pyD;
static cl_mem v2x_pzD;
static cl_mem v2y_pzD;
static cl_mem v2z_pzD;

//debug----------------------
float totalTimeH2DV, totalTimeD2HV;
float totalTimeH2DS, totalTimeD2HS;
float totalTimeCompV, totalTimeCompS;
float tmpTime;
float onceTime;
float onceTime2;
struct timeval t1, t2;
int procID;
//---------------------------

size_t LoadProgramSource(const char *filename, const char **progSrc) 
{
    FILE *f = fopen(filename, "r");
    fseek(f, 0, SEEK_END);
    size_t len = (size_t) ftell(f);
    *progSrc = (const char *) malloc(sizeof(char)*len);
    rewind(f);
    fread((void *) *progSrc, len, 1, f);
    fclose(f);
    return len;
}

// opencl initialization
void init_cl(int *deviceID) 
{
	int i;
#ifdef DISFD_PAPI
	papi_init();
	printf("PAPI Init Done\n");
#endif
    cl_int errNum;
    cl_int err;
    // launch the device
    size_t progLen;
    const char *progSrc;
	int num_platforms;
  size_t            platform_name_size;
  char*             platform_name;
	cl_platform_id *platforms;
	const char *PLATFORM_NAME = "SnuCL Single";
    for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
    Init(&kernelTimerStress[i]);
    Init(&kernelTimerVelocity[i]);
	}
	Init(&h2dTimerStress);
	Init(&h2dTimerVelocity);
	Init(&d2hTimerStress);
	Init(&d2hTimerVelocity);
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
	int num_devices;
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
	cl_context_properties props[5] = {
		CL_CONTEXT_PLATFORM,
		_cl_firstPlatform,
		CL_CONTEXT_SCHEDULER,
		CL_CONTEXT_SCHEDULER_PERF_MODEL,
		0 };
    _cl_context = clCreateContext(props, num_devices, _cl_devices, NULL, NULL, &errNum);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Failed to create GPU context!\n");
    }

    //_cl_commandQueue = clCreateCommandQueue(_cl_context, _cl_firstDevice, CL_QUEUE_PROFILING_ENABLE, NULL);
#if 1
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
			CL_QUEUE_AUTO_DEVICE_SELECTION | 
			CL_QUEUE_ITERATIVE | 
			CL_QUEUE_VOLATILE_EPOCHS | 
			CL_QUEUE_PROFILING_ENABLE, NULL);
		if(_cl_commandQueues[i] == NULL)
		{
			fprintf(stderr, "Failed to create commandQueue for the %dth device!\n", i);
		}
	}
#else
	int chosen_dev_id = 0;
	_cl_commandQueues[0] = clCreateCommandQueue(_cl_context, _cl_devices[chosen_dev_id], 
			CL_QUEUE_AUTO_DEVICE_SELECTION | 
			CL_QUEUE_ITERATIVE | 
			CL_QUEUE_VOLATILE_EPOCHS | 
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
#ifdef DISFD_USE_OPTIMIZED
    progLen = LoadProgramSource("disfd_sub_opt.cl", &progSrc);
#else
    //progLen = LoadProgramSource("disfd_sub.cl.test", &progSrc);
    progLen = LoadProgramSource("disfd_sub.cl", &progSrc);
#endif
    _cl_program = clCreateProgramWithSource(_cl_context, 1, &progSrc, &progLen, NULL);
    if(_cl_program == NULL)
    {
        fprintf(stderr, "Failed to create CL program from source!\n");
    }
    
    free((void *)progSrc);
    // -cl-mad-enable
    // -cl-opt-disable
    errNum = clBuildProgram(_cl_program, 0, NULL, "", NULL, NULL);
    //errNum = clBuildProgram(_cl_program, num_devices, _cl_devices, "", NULL, NULL);
    //errNum = clBuildProgram(_cl_program, 1, &_cl_firstDevice, "-cl-opt-disable", NULL, NULL);
    if(errNum != CL_SUCCESS)
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

    // create the kernels
	_cl_Kernel_velocity_inner_IC   	= clCreateKernel(_cl_program,"velocity_inner_IC"   	,NULL);
	_cl_Kernel_velocity_inner_IIC  	= clCreateKernel(_cl_program,"velocity_inner_IIC"  	,NULL);
	_cl_Kernel_vel_PmlX_IC         	= clCreateKernel(_cl_program,"vel_PmlX_IC"         	,NULL);
	_cl_Kernel_vel_PmlY_IC         	= clCreateKernel(_cl_program,"vel_PmlY_IC"         	,NULL);
	_cl_Kernel_vel_PmlX_IIC        	= clCreateKernel(_cl_program,"vel_PmlX_IIC"        	,NULL);
	_cl_Kernel_vel_PmlY_IIC        	= clCreateKernel(_cl_program,"vel_PmlY_IIC"        	,NULL);
	_cl_Kernel_vel_PmlZ_IIC        	= clCreateKernel(_cl_program,"vel_PmlZ_IIC"        	,NULL);
	_cl_Kernel_stress_norm_xy_IC   	= clCreateKernel(_cl_program,"stress_norm_xy_IC"   	,NULL);
	_cl_Kernel_stress_xz_yz_IC    	= clCreateKernel(_cl_program,"stress_xz_yz_IC"     	,NULL);
	_cl_Kernel_stress_resetVars    	= clCreateKernel(_cl_program,"stress_resetVars"    	,NULL);
	_cl_Kernel_stress_norm_PmlX_IC 	= clCreateKernel(_cl_program,"stress_norm_PmlX_IC" 	,NULL);
	_cl_Kernel_stress_norm_PmlY_IC 	= clCreateKernel(_cl_program,"stress_norm_PmlY_IC" 	,NULL);
	_cl_Kernel_stress_xy_PmlX_IC   	= clCreateKernel(_cl_program,"stress_xy_PmlX_IC"   	,NULL);
	_cl_Kernel_stress_xy_PmlY_IC   	= clCreateKernel(_cl_program,"stress_xy_PmlY_IC"   	,NULL);
	_cl_Kernel_stress_xz_PmlX_IC   	= clCreateKernel(_cl_program,"stress_xz_PmlX_IC"   	,NULL);
	_cl_Kernel_stress_xz_PmlY_IC   	= clCreateKernel(_cl_program,"stress_xz_PmlY_IC"   	,NULL);
	_cl_Kernel_stress_yz_PmlX_IC   	= clCreateKernel(_cl_program,"stress_yz_PmlX_IC"   	,NULL);
	_cl_Kernel_stress_yz_PmlY_IC   	= clCreateKernel(_cl_program,"stress_yz_PmlY_IC"   	,NULL);
	_cl_Kernel_stress_norm_xy_II   	= clCreateKernel(_cl_program,"stress_norm_xy_II"   	,NULL);
	_cl_Kernel_stress_xz_yz_IIC    	= clCreateKernel(_cl_program,"stress_xz_yz_IIC"    	,NULL);
	_cl_Kernel_stress_norm_PmlX_IIC	= clCreateKernel(_cl_program,"stress_norm_PmlX_IIC"	,NULL);
	_cl_Kernel_stress_norm_PmlY_II 	= clCreateKernel(_cl_program,"stress_norm_PmlY_II" 	,NULL);
	_cl_Kernel_stress_norm_PmlZ_IIC	= clCreateKernel(_cl_program,"stress_norm_PmlZ_IIC"	,NULL);
	_cl_Kernel_stress_xy_PmlX_IIC  	= clCreateKernel(_cl_program,"stress_xy_PmlX_IIC"  	,NULL);
	_cl_Kernel_stress_xy_PmlY_IIC  	= clCreateKernel(_cl_program,"stress_xy_PmlY_IIC"  	,NULL);
	_cl_Kernel_stress_xy_PmlZ_II   	= clCreateKernel(_cl_program,"stress_xy_PmlZ_II"   	,NULL);
	_cl_Kernel_stress_xz_PmlX_IIC  	= clCreateKernel(_cl_program,"stress_xz_PmlX_IIC"  	,NULL);
	_cl_Kernel_stress_xz_PmlY_IIC  	= clCreateKernel(_cl_program,"stress_xz_PmlY_IIC"  	,NULL);
	_cl_Kernel_stress_xz_PmlZ_IIC  	= clCreateKernel(_cl_program,"stress_xz_PmlZ_IIC"  	,NULL);
	_cl_Kernel_stress_yz_PmlX_IIC  	= clCreateKernel(_cl_program,"stress_yz_PmlX_IIC"  	,NULL);
	_cl_Kernel_stress_yz_PmlY_IIC  	= clCreateKernel(_cl_program,"stress_yz_PmlY_IIC"  	,NULL);
	_cl_Kernel_stress_yz_PmlZ_IIC  	= clCreateKernel(_cl_program,"stress_yz_PmlZ_IIC"  	,NULL);

    if(_cl_Kernel_velocity_inner_IC	== NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_velocity_inner_IC!\n");
    if(_cl_Kernel_velocity_inner_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_velocity_inner_IIC!\n");
    if(_cl_Kernel_vel_PmlX_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_vel_PmlX_IC!\n");
    if(_cl_Kernel_vel_PmlY_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_vel_PmlY_IC!\n");
    if(_cl_Kernel_vel_PmlX_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_vel_PmlX_IIC!\n");
    if(_cl_Kernel_vel_PmlY_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_vel_PmlY_IIC!\n");
    if(_cl_Kernel_vel_PmlZ_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_vel_PmlZ_IIC!\n");
    
    if(_cl_Kernel_stress_norm_xy_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_norm_xy_IC!\n");
    if(_cl_Kernel_stress_xz_yz_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xz_yz_IC!\n");
    if(_cl_Kernel_stress_resetVars == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_resetVars!\n");
    if(_cl_Kernel_stress_norm_PmlX_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_norm_PmlX_IC!\n");
    if(_cl_Kernel_stress_norm_PmlY_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_norm_PmlY_IC!\n");
    if(_cl_Kernel_stress_xy_PmlX_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xy_PmlX_IC!\n");
    if(_cl_Kernel_stress_xy_PmlY_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xy_PmlY_IC!\n");
    if(_cl_Kernel_stress_xz_PmlX_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xz_PmlX_IC!\n");
    if(_cl_Kernel_stress_xz_PmlY_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xz_PmlY_IC!\n");
    if(_cl_Kernel_stress_yz_PmlX_IC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_yz_PmlX_IC!\n");
    if(_cl_Kernel_stress_norm_xy_II == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_norm_xy_II!\n");
    if(_cl_Kernel_stress_xz_yz_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xz_yz_IIC!\n");
    if(_cl_Kernel_stress_norm_PmlX_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_norm_PmlX_IIC!\n");
    if(_cl_Kernel_stress_norm_PmlY_II == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_norm_PmlY_II!\n");
    if(_cl_Kernel_stress_norm_PmlZ_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_norm_PmlZ_IIC!\n");
    if(_cl_Kernel_stress_xy_PmlX_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xy_PmlX_IIC!\n");
    if(_cl_Kernel_stress_xy_PmlY_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xy_PmlY_IIC!\n");
    if(_cl_Kernel_stress_xy_PmlZ_II == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xy_PmlZ_II!\n");
    if(_cl_Kernel_stress_xz_PmlX_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xz_PmlX_IIC!\n");
    if(_cl_Kernel_stress_xz_PmlY_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xz_PmlY_IIC!\n");
    if(_cl_Kernel_stress_xz_PmlZ_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_xz_PmlZ_IIC!\n");
    if(_cl_Kernel_stress_yz_PmlX_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_yz_PmlX_IIC!\n");
    if(_cl_Kernel_stress_yz_PmlY_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_yz_PmlY_IIC!\n");
    if(_cl_Kernel_stress_yz_PmlZ_IIC == NULL)
        fprintf(stderr, "Failed to create kernel _cl_Kernel_stress_yz_PmlZ_IIC!\n");

    //printf("[OpenCL] device initialization success!\n");
}

void free_device_memC_opencl(int *lbx, int *lby)
{
    // timing information
	printf("[OpenCL] id = %d, vel, H2D =, %.3f, D2H =, %.3f, comp =, %.3f\n", procID, totalTimeH2DV, totalTimeD2HV, totalTimeCompV);
	printf("[OpenCL] id = %d, str, H2D =, %.3f, D2H =, %.3f, comp =, %.3f\n", procID, totalTimeH2DS, totalTimeD2HS, totalTimeCompS);
//    printf("[Opencl] vel_once h2d = %.3f\n", onceTime);
//    printf("[Opencl] str_once h2d = %.3f\n", onceTime2);
	Print(&kernelTimerStress[0], "Stress Kernel 0 Time");
	Print(&kernelTimerVelocity[0], "Velocity Kernel 0 Time");
	Print(&kernelTimerStress[1], "Stress Kernel 1 Time");
	Print(&kernelTimerVelocity[1], "Velocity Kernel 1 Time");
	Print(&h2dTimerStress, "Stress H2D Time");
	Print(&h2dTimerVelocity, "Velocity H2D Time");
	Print(&d2hTimerStress, "Stress D2H Time");
	Print(&d2hTimerVelocity, "Velocity D2H Time");
#ifdef DISFD_PAPI
	papi_print_all_events();
	papi_stop_all_events();
#endif
	clReleaseMemObject(nd1_velD);
	clReleaseMemObject(nd1_txyD);
	clReleaseMemObject(nd1_txzD);
	clReleaseMemObject(nd1_tyyD);
	clReleaseMemObject(nd1_tyzD);
	clReleaseMemObject(rhoD);
	clReleaseMemObject(drvh1D);
	clReleaseMemObject(drti1D);
	clReleaseMemObject(drth1D);
	clReleaseMemObject(idmat1D);
	clReleaseMemObject(dxi1D);
	clReleaseMemObject(dyi1D);
	clReleaseMemObject(dzi1D);
	clReleaseMemObject(dxh1D);
	clReleaseMemObject(dyh1D);
	clReleaseMemObject(dzh1D);
	clReleaseMemObject(t1xxD);
	clReleaseMemObject(t1xyD);
	clReleaseMemObject(t1xzD);
	clReleaseMemObject(t1yyD);
	clReleaseMemObject(t1yzD);
	clReleaseMemObject(t1zzD);
	clReleaseMemObject(v1xD);    //output
	clReleaseMemObject(v1yD);
	clReleaseMemObject(v1zD);

	if (lbx[1] >= lbx[0])
	{
		clReleaseMemObject(damp1_xD);
		clReleaseMemObject(t1xx_pxD);
		clReleaseMemObject(t1xy_pxD);
		clReleaseMemObject(t1xz_pxD);
		clReleaseMemObject(t1yy_pxD);
		clReleaseMemObject(qt1xx_pxD);
		clReleaseMemObject(qt1xy_pxD);
		clReleaseMemObject(qt1xz_pxD);
		clReleaseMemObject(qt1yy_pxD);
		clReleaseMemObject(v1x_pxD);
		clReleaseMemObject(v1y_pxD);
		clReleaseMemObject(v1z_pxD);
	}

	if (lby[1] >= lby[0])
	{
		clReleaseMemObject(damp1_yD);
		clReleaseMemObject(t1xx_pyD);
		clReleaseMemObject(t1xy_pyD);
		clReleaseMemObject(t1yy_pyD);
		clReleaseMemObject(t1yz_pyD);
		clReleaseMemObject(qt1xx_pyD);
		clReleaseMemObject(qt1xy_pyD);
		clReleaseMemObject(qt1yy_pyD);
		clReleaseMemObject(qt1yz_pyD);
		clReleaseMemObject(v1x_pyD);
		clReleaseMemObject(v1y_pyD);
		clReleaseMemObject(v1z_pyD);
	}

	clReleaseMemObject(qt1xxD);
	clReleaseMemObject(qt1xyD);
	clReleaseMemObject(qt1xzD);
	clReleaseMemObject(qt1yyD);
	clReleaseMemObject(qt1yzD);
	clReleaseMemObject(qt1zzD);

	clReleaseMemObject(clamdaD);
	clReleaseMemObject(cmuD);
	clReleaseMemObject(epdtD);
	clReleaseMemObject(qwpD);
	clReleaseMemObject(qwsD);
	clReleaseMemObject(qwt1D);
	clReleaseMemObject(qwt2D);
//-------------------------------------
	clReleaseMemObject(nd2_velD);
	clReleaseMemObject(nd2_txyD);
	clReleaseMemObject(nd2_txzD);
	clReleaseMemObject(nd2_tyyD);
	clReleaseMemObject(nd2_tyzD);

	clReleaseMemObject(drvh2D);
	clReleaseMemObject(drti2D);
	clReleaseMemObject(drth2D);
	clReleaseMemObject(idmat2D);
	clReleaseMemObject(damp2_zD);
	clReleaseMemObject(dxi2D);
	clReleaseMemObject(dyi2D);
	clReleaseMemObject(dzi2D);
	clReleaseMemObject(dxh2D);
	clReleaseMemObject(dyh2D);
	clReleaseMemObject(dzh2D);
	clReleaseMemObject(t2xxD);
	clReleaseMemObject(t2xyD);
	clReleaseMemObject(t2xzD);
	clReleaseMemObject(t2yyD);
	clReleaseMemObject(t2yzD);
	clReleaseMemObject(t2zzD);

	clReleaseMemObject(qt2xxD);
	clReleaseMemObject(qt2xyD);
	clReleaseMemObject(qt2xzD);
	clReleaseMemObject(qt2yyD);
	clReleaseMemObject(qt2yzD);
	clReleaseMemObject(qt2zzD);

	if (lbx[1] >= lbx[0])
	{
		clReleaseMemObject(damp2_xD);

		clReleaseMemObject(t2xx_pxD);
		clReleaseMemObject(t2xy_pxD);
		clReleaseMemObject(t2xz_pxD);
		clReleaseMemObject(t2yy_pxD);
		clReleaseMemObject(qt2xx_pxD);
		clReleaseMemObject(qt2xy_pxD);
		clReleaseMemObject(qt2xz_pxD);
		clReleaseMemObject(qt2yy_pxD);

		clReleaseMemObject(v2x_pxD);
		clReleaseMemObject(v2y_pxD);
		clReleaseMemObject(v2z_pxD);
	}

	if (lby[1] >= lby[0])
	{
		clReleaseMemObject(damp2_yD);

		clReleaseMemObject(t2xx_pyD);
		clReleaseMemObject(t2xy_pyD);
		clReleaseMemObject(t2yy_pyD);
		clReleaseMemObject(t2yz_pyD);

		clReleaseMemObject(qt2xx_pyD);
		clReleaseMemObject(qt2xy_pyD);
		clReleaseMemObject(qt2yy_pyD);
		clReleaseMemObject(qt2yz_pyD);

		clReleaseMemObject(v2x_pyD);
		clReleaseMemObject(v2y_pyD);
		clReleaseMemObject(v2z_pyD);
	}

	clReleaseMemObject(t2xx_pzD);
	clReleaseMemObject(t2xz_pzD);
	clReleaseMemObject(t2yz_pzD);
	clReleaseMemObject(t2zz_pzD);

	clReleaseMemObject(qt2xx_pzD);
	clReleaseMemObject(qt2xz_pzD);
	clReleaseMemObject(qt2yz_pzD);
	clReleaseMemObject(qt2zz_pzD);

	clReleaseMemObject(v2xD);		//output
	clReleaseMemObject(v2yD);
	clReleaseMemObject(v2zD);

	clReleaseMemObject(v2x_pzD);
	clReleaseMemObject(v2y_pzD);
	clReleaseMemObject(v2z_pzD);
	
    //printf("[OpenCL] memory space is freed.\n");
	return;
}

// opencl release
void release_cl(int *deviceID) 
{
	int i;
    // release the opencl context
	clReleaseKernel(_cl_Kernel_velocity_inner_IC   );
	clReleaseKernel(_cl_Kernel_velocity_inner_IIC  );
	clReleaseKernel(_cl_Kernel_vel_PmlX_IC         );
	clReleaseKernel(_cl_Kernel_vel_PmlY_IC         );
	clReleaseKernel(_cl_Kernel_vel_PmlX_IIC        );
	clReleaseKernel(_cl_Kernel_vel_PmlY_IIC        );
	clReleaseKernel(_cl_Kernel_vel_PmlZ_IIC        );
	clReleaseKernel(_cl_Kernel_stress_norm_xy_IC   );
	clReleaseKernel(_cl_Kernel_stress_xz_yz_IC     );
	clReleaseKernel(_cl_Kernel_stress_resetVars    );
	clReleaseKernel(_cl_Kernel_stress_norm_PmlX_IC );
	clReleaseKernel(_cl_Kernel_stress_norm_PmlY_IC );
	clReleaseKernel(_cl_Kernel_stress_xy_PmlX_IC   );
	clReleaseKernel(_cl_Kernel_stress_xy_PmlY_IC   );
	clReleaseKernel(_cl_Kernel_stress_xz_PmlX_IC   );
	clReleaseKernel(_cl_Kernel_stress_xz_PmlY_IC   );
	clReleaseKernel(_cl_Kernel_stress_yz_PmlX_IC   );
	clReleaseKernel(_cl_Kernel_stress_yz_PmlY_IC   );
	clReleaseKernel(_cl_Kernel_stress_norm_xy_II   );
	clReleaseKernel(_cl_Kernel_stress_xz_yz_IIC    );
	clReleaseKernel(_cl_Kernel_stress_norm_PmlX_IIC);
	clReleaseKernel(_cl_Kernel_stress_norm_PmlY_II );
	clReleaseKernel(_cl_Kernel_stress_norm_PmlZ_IIC);
	clReleaseKernel(_cl_Kernel_stress_xy_PmlX_IIC  );
	clReleaseKernel(_cl_Kernel_stress_xy_PmlY_IIC  );
	clReleaseKernel(_cl_Kernel_stress_xy_PmlZ_II   );
	clReleaseKernel(_cl_Kernel_stress_xz_PmlX_IIC  );
	clReleaseKernel(_cl_Kernel_stress_xz_PmlY_IIC  );
	clReleaseKernel(_cl_Kernel_stress_xz_PmlZ_IIC  );
	clReleaseKernel(_cl_Kernel_stress_yz_PmlX_IIC  );
	clReleaseKernel(_cl_Kernel_stress_yz_PmlY_IIC  );
	clReleaseKernel(_cl_Kernel_stress_yz_PmlZ_IIC  );

	clReleaseProgram(_cl_program);
	//clReleaseCommandQueue(_cl_commandQueue);
	for(i = 0; i < NUM_COMMAND_QUEUES; i++)
	{
		clReleaseCommandQueue(_cl_commandQueues[i]);
	}
	clReleaseContext(_cl_context);
	free(_cl_devices);
    //printf("[OpenCL] context all released!\n");
}

// called from disfd.f90 only once
void allocate_gpu_memC_opencl(int   *lbx,
					int   *lby,
					int   *nmat,		//dimension #, int
					int	  *mw1_pml1,	//int
					int	  *mw2_pml1,	//int
					int	  *nxtop,		//int
					int	  *nytop,		//int
					int   *nztop,
					int	  *mw1_pml,		//int
					int   *mw2_pml,		//int
					int	  *nxbtm,		//int
					int	  *nybtm,		//int
					int	  *nzbtm,
					int   *nzbm1,
					int   *nll)
{
    //printf("[OpenCL] allocation..........");
	int nv2, nti, nth;
    // debug -----------
    // printf("lbx[1] = %d, lbx[0] = %d\n", lbx[1], lbx[0]);
    // printf("lby[1] = %d, lby[0] = %d\n", lby[1], lby[0]);
    // printf("nmat = %d\n", *nmat);
    // printf("mw1_pml1 = %d, mw2_pml1 = %d\n", *mw1_pml1, *mw2_pml1);
    // printf("mw1_pml = %d, mw2_pml = %d\n", *mw1_pml, *mw2_pml);
    // printf("nxtop = %d, nytop = %d, nztop = %d\n", *nxtop, *nytop, *nztop);
    // printf("nxbtm = %d, nybtm = %d, nzbtm = %d\n", *nxbtm, *nybtm, *nzbtm);
    // printf("nzbm1 = %d, nll = %d\n", *nzbm1, *nll);

	// timing ---------
    totalTimeH2DV = 0.0f;
	totalTimeD2HV = 0.0f;
	totalTimeH2DS = 0.0f;
	totalTimeD2HS = 0.0f;
	totalTimeCompV = 0.0f;
	totalTimeCompS = 0.0f;

	//for inner_I
    nd1_velD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    nd1_txyD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    nd1_txzD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    nd1_tyyD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    nd1_tyzD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    CHECK_NULL_ERROR(nd1_velD, "nd1_velD");
    CHECK_NULL_ERROR(nd1_txyD, "nd1_txyD");
    CHECK_NULL_ERROR(nd1_txzD, "nd1_txzD");
    CHECK_NULL_ERROR(nd1_tyyD, "nd1_tyyD");
    CHECK_NULL_ERROR(nd1_tyzD, "nd1_tyzD");
    
    rhoD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nmat), NULL, NULL);
    CHECK_NULL_ERROR(rhoD, "rhoD");

    drvh1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*mw1_pml1) * 2, NULL, NULL);
    drti1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*mw1_pml1) * 2, NULL, NULL);
    drth1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*mw1_pml1) * 2, NULL, NULL);
    CHECK_NULL_ERROR(drvh1D, "drvh1D");
    CHECK_NULL_ERROR(drti1D, "drti1D");
    CHECK_NULL_ERROR(drth1D, "drth1D");

    if (lbx[1] >= lbx[0])
	{
        damp1_xD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1), NULL, NULL);
        CHECK_NULL_ERROR(damp1_xD, "damp1_xD");
	}

	if (lby[1] >= lby[0])
	{
        damp1_yD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1), NULL, NULL);
        CHECK_NULL_ERROR(damp1_yD, "damp1_yD");
	}

    idmat1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1), NULL, NULL);
    CHECK_NULL_ERROR(idmat1D, "idmat1D");
    dxi1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nxtop), NULL, NULL);
    dyi1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nytop), NULL, NULL);
    dzi1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nztop + 1), NULL, NULL);
    dxh1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nxtop), NULL, NULL);
    dyh1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nytop), NULL, NULL);
    dzh1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nztop + 1), NULL, NULL);
    CHECK_NULL_ERROR(dxi1D, "dxi1D");
    CHECK_NULL_ERROR(dyi1D, "dyi1D");
    CHECK_NULL_ERROR(dzi1D, "dzi1D");
    CHECK_NULL_ERROR(dxh1D, "dxh1D");
    CHECK_NULL_ERROR(dyh1D, "dyh1D");
    CHECK_NULL_ERROR(dzh1D, "dzh1D");

    t1xxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), NULL, NULL);
    t1xyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), NULL, NULL);
    t1xzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), NULL, NULL);
    t1yyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), NULL, NULL);
    t1yzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), NULL, NULL);
    t1zzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), NULL, NULL);
    CHECK_NULL_ERROR(t1xxD, "t1xxD");
    CHECK_NULL_ERROR(t1xyD, "t1xyD");
    CHECK_NULL_ERROR(t1xzD, "t1xzD");
    CHECK_NULL_ERROR(t1yyD, "t1yyD");
    CHECK_NULL_ERROR(t1yzD, "t1yzD");
    CHECK_NULL_ERROR(t1zzD, "t1zzD");
	
	if (lbx[1] >= lbx[0])
	{
		nti = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1];
		nth = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + 1 - lbx[0];

		t1xx_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (nti) * (*nytop), NULL, NULL);
		t1xy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * nth * (*nytop), NULL, NULL);
		t1xz_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop+1) * nth * (*nytop), NULL, NULL);
		t1yy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * nti * (*nytop), NULL, NULL);
        CHECK_NULL_ERROR(t1xx_pxD, "t1xx_pxD");
        CHECK_NULL_ERROR(t1xy_pxD, "t1xy_pxD");
        CHECK_NULL_ERROR(t1xz_pxD, "t1xz_pxD");
        CHECK_NULL_ERROR(t1yy_pxD, "t1yy_pxD");

		qt1xx_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (nti) * (*nytop), NULL, NULL);
		qt1xy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * nth * (*nytop), NULL, NULL);
		qt1xz_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop+1) * nth * (*nytop), NULL, NULL);
		qt1yy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * nti * (*nytop), NULL, NULL);
        CHECK_NULL_ERROR(qt1xx_pxD, "qt1xx_pxD");
        CHECK_NULL_ERROR(qt1xy_pxD, "qt1xy_pxD");
        CHECK_NULL_ERROR(qt1xz_pxD, "qt1xz_pxD");
        CHECK_NULL_ERROR(qt1yy_pxD, "qt1yy_pxD");
	}

	if (lby[1] >= lby[0])
	{
		nti = (lby[1] - lby[0] + 1) * (*mw1_pml) + lby[1];
		nth = (lby[1] - lby[0] + 1) * (*mw1_pml) + 1 - lby[0];

		t1xx_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nti, NULL, NULL);
		t1xy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nth, NULL, NULL);
		t1yy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nti, NULL, NULL);
		t1yz_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop+1) * (*nxtop) * nth, NULL, NULL);
        CHECK_NULL_ERROR(t1xx_pyD, "t1xx_pyD");
        CHECK_NULL_ERROR(t1xy_pyD, "t1xy_pyD");
        CHECK_NULL_ERROR(t1yy_pyD, "t1yy_pyD");
        CHECK_NULL_ERROR(t1yz_pyD, "t1yz_pyD");

		qt1xx_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nti, NULL, NULL);
		qt1xy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nth, NULL, NULL);
		qt1yy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nti, NULL, NULL);
		qt1yz_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop+1) * (*nxtop) * nth, NULL, NULL);
        CHECK_NULL_ERROR(qt1xx_pyD, "qt1xx_pyD");
        CHECK_NULL_ERROR(qt1xy_pyD, "qt1xy_pyD");
        CHECK_NULL_ERROR(qt1yy_pyD, "qt1yy_pyD");
        CHECK_NULL_ERROR(qt1yz_pyD, "qt1yz_pyD");
	}

	qt1xxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), NULL, NULL);
	qt1xyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), NULL, NULL);
	qt1xzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop), NULL, NULL);
	qt1yyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), NULL, NULL);
	qt1yzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop), NULL, NULL);
	qt1zzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), NULL, NULL);
    CHECK_NULL_ERROR(qt1xxD, "qt1xxD");
    CHECK_NULL_ERROR(qt1xyD, "qt1xyD");
    CHECK_NULL_ERROR(qt1xzD, "qt1xzD");
    CHECK_NULL_ERROR(qt1yyD, "qt1yyD");
    CHECK_NULL_ERROR(qt1yzD, "qt1yzD");
    CHECK_NULL_ERROR(qt1zzD, "qt1zzD");

	clamdaD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nmat), NULL, NULL);
	cmuD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nmat), NULL, NULL);
	epdtD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nll), NULL, NULL);
	qwpD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nmat), NULL, NULL);
	qwsD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nmat), NULL, NULL);
	qwt1D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nll), NULL, NULL);
	qwt2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nll), NULL, NULL);
    CHECK_NULL_ERROR(clamdaD, "clamdaD");
    CHECK_NULL_ERROR(cmuD, "cmuD");
    CHECK_NULL_ERROR(epdtD, "epdtD");
    CHECK_NULL_ERROR(qwpD, "qwpD");
    CHECK_NULL_ERROR(qwsD, "qwsD");
    CHECK_NULL_ERROR(qwt1D, "qwt1D");
    CHECK_NULL_ERROR(qwt2D, "qwt2D");

    v1xD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), NULL, NULL);
    v1yD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), NULL, NULL);
    v1zD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), NULL, NULL);
    CHECK_NULL_ERROR(v1xD, "v1xD");
    CHECK_NULL_ERROR(v1yD, "v1yD");
    CHECK_NULL_ERROR(v1zD, "v1zD");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw1_pml);
        
        v1x_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * nv2 * (*nytop), NULL, NULL);
        v1y_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * nv2 * (*nytop), NULL, NULL);
        v1z_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * nv2 * (*nytop), NULL, NULL);
        CHECK_NULL_ERROR(v1x_pxD, "v1x_pxD");
        CHECK_NULL_ERROR(v1y_pxD, "v1y_pxD");
        CHECK_NULL_ERROR(v1z_pxD, "v1z_pxD");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw1_pml);
        v1x_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nv2, NULL, NULL);
        v1y_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nv2, NULL, NULL);
        v1z_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nztop) * (*nxtop) * nv2, NULL, NULL);
        CHECK_NULL_ERROR(v1x_pyD, "v1x_pyD");
        CHECK_NULL_ERROR(v1y_pyD, "v1y_pyD");
        CHECK_NULL_ERROR(v1z_pyD, "v1z_pyD");
	}

//for inner_II-----------------------------------------------------------------------------------------
    nd2_velD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    nd2_txyD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    nd2_txzD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL); 
    nd2_tyyD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    nd2_tyzD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * 18, NULL, NULL);
    CHECK_NULL_ERROR(nd2_velD, "nd2_velD");
    CHECK_NULL_ERROR(nd2_txyD, "nd2_txyD");
    CHECK_NULL_ERROR(nd2_txzD, "nd2_txzD");
    CHECK_NULL_ERROR(nd2_tyyD, "nd2_tyyD");
    CHECK_NULL_ERROR(nd2_tyzD, "nd2_tyzD");

    drvh2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*mw2_pml1) * 2, NULL, NULL);
    drti2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*mw2_pml1) * 2, NULL, NULL);
    drth2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*mw2_pml1) * 2, NULL, NULL);
    CHECK_NULL_ERROR(drvh2D, "drvh2D");
    CHECK_NULL_ERROR(drti2D, "drti2D");
    CHECK_NULL_ERROR(drth2D, "drth2D");

    idmat2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm + 1), NULL, NULL);
    CHECK_NULL_ERROR(idmat2D, "idmat2D");
	
	if (lbx[1] >= lbx[0])
	{
        damp2_xD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1), NULL, NULL);
        CHECK_NULL_ERROR(damp2_xD, "damp2_xD");
	}

	if (lby[1] >= lby[0])
	{
        damp2_yD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1), NULL, NULL);
        CHECK_NULL_ERROR(damp2_yD, "damp2_yD");
	}
    damp2_zD = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * (*nxbtm) * (*nybtm), NULL, NULL);
    CHECK_NULL_ERROR(damp2_zD, "damp2_zD");

    dxi2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nxbtm), NULL, NULL);
    dyi2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nybtm), NULL, NULL);
    dzi2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nzbtm), NULL, NULL);
    dxh2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nxbtm), NULL, NULL);
    dyh2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nybtm), NULL, NULL);
    dzh2D = clCreateBuffer(_cl_context, CL_MEM_READ_ONLY, sizeof(float) * 4 * (*nzbtm), NULL, NULL);
    CHECK_NULL_ERROR(dxi2D, "dxi2D");
    CHECK_NULL_ERROR(dyi2D, "dyi2D");
    CHECK_NULL_ERROR(dzi2D, "dzi2D");
    CHECK_NULL_ERROR(dxh2D, "dxh2D");
    CHECK_NULL_ERROR(dyh2D, "dyh2D");
    CHECK_NULL_ERROR(dxh2D, "dzh2D");

    t2xxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), NULL, NULL);
    t2xyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), NULL, NULL);
    t2xzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), NULL, NULL);
    t2yyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), NULL, NULL);
    t2yzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), NULL, NULL);
    t2zzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), NULL, NULL);
    CHECK_NULL_ERROR(t2xxD, "t2xxD");
    CHECK_NULL_ERROR(t2xyD, "t2xyD");
    CHECK_NULL_ERROR(t2xzD, "t2xzD");
    CHECK_NULL_ERROR(t2yyD, "t2yyD");
    CHECK_NULL_ERROR(t2yzD, "t2yzD");
    CHECK_NULL_ERROR(t2zzD, "t2zzD");

	qt2xxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2xyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2xzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2yyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2yzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2zzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), NULL, NULL);
    CHECK_NULL_ERROR(qt2xxD, "qt2xxD");
    CHECK_NULL_ERROR(qt2xyD, "qt2xyD");
    CHECK_NULL_ERROR(qt2xzD, "qt2xzD");
    CHECK_NULL_ERROR(qt2yyD, "qt2yyD");
    CHECK_NULL_ERROR(qt2yzD, "qt2yzD");
    CHECK_NULL_ERROR(qt2zzD, "qt2zzD");


	if (lbx[1] >= lbx[0])
	{
        nti = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + lbx[1];
        nth = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + 1 - lbx[0];

		t2xx_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nti * (*nybtm), NULL, NULL);
		t2xy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nth * (*nybtm), NULL, NULL);
		t2xz_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nth * (*nybtm), NULL, NULL);
		t2yy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nti * (*nybtm), NULL, NULL);
        CHECK_NULL_ERROR(t2xx_pxD, "t2xx_pxD");
        CHECK_NULL_ERROR(t2xy_pxD, "t2xy_pxD");
        CHECK_NULL_ERROR(t2xz_pxD, "t2xz_pxD");
        CHECK_NULL_ERROR(t2yy_pxD, "t2yy_pxD");

		qt2xx_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nti * (*nybtm), NULL, NULL);
		qt2xy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nth * (*nybtm), NULL, NULL);
		qt2xz_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nth * (*nybtm), NULL, NULL);
		qt2yy_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nti * (*nybtm), NULL, NULL);
        CHECK_NULL_ERROR(qt2xx_pxD, "qt2xx_pxD");
        CHECK_NULL_ERROR(qt2xy_pxD, "qt2xy_pxD");
        CHECK_NULL_ERROR(qt2xz_pxD, "qt2xz_pxD");
        CHECK_NULL_ERROR(qt2yy_pxD, "qt2yy_pxD");
	}

	if (lby[1] >= lby[0])
	{
        nti = (lby[1] - lby[0] + 1) * (*mw2_pml) + lby[1];
        nth = (lby[1] - lby[0] + 1) * (*mw2_pml) + 1 - lby[0];

		t2xx_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, NULL, NULL);
		t2xy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, NULL, NULL);
		t2yy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, NULL, NULL);
		t2yz_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, NULL, NULL);
        CHECK_NULL_ERROR(t2xx_pyD, "t2xx_pyD");
        CHECK_NULL_ERROR(t2xy_pyD, "t2xy_pyD");
        CHECK_NULL_ERROR(t2yy_pyD, "t2yy_pyD");
        CHECK_NULL_ERROR(t2yz_pyD, "t2yz_pyD");

		qt2xx_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, NULL, NULL);
		qt2xy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, NULL, NULL);
		qt2yy_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, NULL, NULL);
		qt2yz_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, NULL, NULL);
        CHECK_NULL_ERROR(qt2xx_pyD, "qt2xx_pyD");
        CHECK_NULL_ERROR(qt2xy_pyD, "qt2xy_pyD");
        CHECK_NULL_ERROR(qt2yy_pyD, "qt2yy_pyD");
        CHECK_NULL_ERROR(qt2yz_pyD, "qt2yz_pyD");
	}

	t2xx_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), NULL, NULL);
	t2xz_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), NULL, NULL);
	t2yz_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), NULL, NULL);
	t2zz_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), NULL, NULL);
    CHECK_NULL_ERROR(t2xx_pzD, "t2xx_pzD");
    CHECK_NULL_ERROR(t2xz_pzD, "t2xz_pzD");
    CHECK_NULL_ERROR(t2yz_pzD, "t2yz_pzD");
    CHECK_NULL_ERROR(t2zz_pzD, "t2zz_pzD");

	qt2xx_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2xz_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2yz_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), NULL, NULL);
	qt2zz_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), NULL, NULL);
    CHECK_NULL_ERROR(qt2xx_pzD, "qt2xx_pzD");
    CHECK_NULL_ERROR(qt2xz_pzD, "qt2xz_pzD");
    CHECK_NULL_ERROR(qt2yz_pzD, "qt2yz_pzD");
    CHECK_NULL_ERROR(qt2zz_pzD, "qt2zz_pzD");

	v2xD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), NULL, NULL);
	v2yD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), NULL, NULL);
	v2zD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), NULL, NULL);
    CHECK_NULL_ERROR(v2xD, "v2xD");
    CHECK_NULL_ERROR(v2yD, "v2yD");
    CHECK_NULL_ERROR(v2zD, "v2zD");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw2_pml);
        v2x_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), NULL, NULL);
        v2y_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), NULL, NULL);
        v2z_pxD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), NULL, NULL);
        CHECK_NULL_ERROR(v2x_pxD, "v2x_pxD");
        CHECK_NULL_ERROR(v2y_pxD, "v2y_pxD");
        CHECK_NULL_ERROR(v2z_pxD, "v2z_pxD");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw2_pml);
        v2x_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, NULL, NULL);
        v2y_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, NULL, NULL);
        v2z_pyD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, NULL, NULL);
        CHECK_NULL_ERROR(v2x_pyD, "v2x_pyD");
        CHECK_NULL_ERROR(v2y_pyD, "v2y_pyD");
        CHECK_NULL_ERROR(v2z_pyD, "v2z_pyD");
	}

    v2x_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), NULL, NULL);
    v2y_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), NULL, NULL);
    v2z_pzD = clCreateBuffer(_cl_context, CL_MEM_READ_WRITE, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), NULL, NULL);
    CHECK_NULL_ERROR(v2x_pzD, "v2x_pzD");
    CHECK_NULL_ERROR(v2y_pzD, "v2y_pzD");
    CHECK_NULL_ERROR(v2z_pzD, "v2z_pzD");

    //printf("done!\n");

	return;
}

// copy data h2d before iterations
void cpy_h2d_velocityInputsCOneTimecl(int   *lbx,
						  int   *lby,
						  int   *nd1_vel,
						  float *rho,
						  float *drvh1,
						  float *drti1,
						  float *damp1_x,
						  float *damp1_y,
						  int	*idmat1,
						  float *dxi1,
						  float *dyi1,
						  float *dzi1,
						  float *dxh1,
						  float *dyh1,
						  float *dzh1,
						  float *t1xx,
						  float *t1xy,
						  float *t1xz,
						  float *t1yy,
						  float *t1yz,
						  float *t1zz,
						  float *v1x_px,
						  float *v1y_px,
						  float *v1z_px,
						  float *v1x_py,
						  float *v1y_py,
						  float *v1z_py,
						  int	*nd2_vel,
						  float *drvh2,
						  float *drti2,
						  int	*idmat2,
						  float *damp2_x,
						  float *damp2_y,
						  float *damp2_z,
						  float *dxi2,
						  float *dyi2,
						  float *dzi2,
						  float *dxh2,
						  float *dyh2,
						  float *dzh2,
						  float *t2xx,
						  float *t2xy,
						  float *t2xz,
						  float *t2yy,
						  float *t2yz,
						  float *t2zz,
						  float *v2x_px,
						  float *v2y_px,
						  float *v2z_px,
						  float *v2x_py,
						  float *v2y_py,
						  float *v2z_py,
						  float *v2x_pz,
						  float *v2y_pz,
						  float *v2z_pz,
						  int   *nmat,		//dimension #, int
						  int	*mw1_pml1,	//int
						  int	*mw2_pml1,	//int
						  int	*nxtop,		//int
						  int	*nytop,		//int
						  int   *nztop,
						  int	*mw1_pml,	//int
						  int   *mw2_pml,	//int
						  int	*nxbtm,		//int
						  int	*nybtm,		//int
						  int	*nzbtm,
						  int   *nzbm1)
{
    //printf("[OpenCL] initial h2d cpy for velocity ........");
    // printf("lbx[0] = %d, lbx[1] = %d\n", lbx[0], lbx[1]);
    // printf("lby[0] = %d, lby[1] = %d\n", lby[0], lby[1]);

	cl_int errNum;
	int nv2;

	//for inner_I
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], nd1_velD, CL_TRUE, 0, sizeof(int) * 18, nd1_vel, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, nd1_vel");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], rhoD, CL_TRUE, 0, sizeof(float) * (*nmat), rho, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, rho");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], drvh1D, CL_TRUE, 0, sizeof(float) * (*mw1_pml1) * 2, drvh1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, drvh1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], drti1D, CL_TRUE, 0, sizeof(float) * (*mw1_pml1) * 2, drti1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, drti1");

	if (lbx[1] >= lbx[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], damp1_xD, CL_TRUE, 0, sizeof(float) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1), damp1_x, 0, NULL, NULL);
		CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], damp1_yD, CL_TRUE, 0, sizeof(float) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1), damp1_y, 0, NULL, NULL);
		CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, damp1_y");
	}

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], idmat1D, CL_TRUE, 0, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1), idmat1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, idmat1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dxi1D, CL_TRUE, 0, sizeof(float) * 4 * (*nxtop), dxi1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dxi1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dyi1D, CL_TRUE, 0, sizeof(float) * 4 * (*nytop), dyi1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dyi1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dzi1D, CL_TRUE, 0, sizeof(float) * 4 * (*nztop + 1), dzi1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dzi1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dxh1D, CL_TRUE, 0, sizeof(float) * 4 * (*nxtop), dxh1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dxh1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dyh1D, CL_TRUE, 0, sizeof(float) * 4 * (*nytop), dyh1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dyh1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dzh1D, CL_TRUE, 0, sizeof(float) * 4 * (*nztop + 1), dzh1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dzh1");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xxD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), t1xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), t1xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xzD, CL_TRUE, 0, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), t1xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), t1yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yzD, CL_TRUE, 0, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), t1yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1zzD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), t1zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1zz");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw1_pml);
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1x_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * nv2 * (*nytop), v1x_px, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1x_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1y_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * nv2 * (*nytop), v1y_px, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1y_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1z_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * nv2 * (*nytop), v1z_px, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw1_pml);
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1x_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nv2, v1x_py, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1x_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1y_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nv2, v1y_py, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1y_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1z_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nv2, v1z_py, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1z_py");
	}


	//for inner_II
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], nd2_velD, CL_TRUE, 0, sizeof(int) * 18, nd2_vel, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, nd2_vel");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], drvh2D, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * 2, drvh2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, drvh2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], drti2D, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * 2, drti2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, drti2");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], idmat2D, CL_TRUE, 0, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm +1), idmat2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], damp2_xD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1), damp2_x, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], damp2_yD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1), damp2_y, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, damp2_y");
	}
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], damp2_zD, CL_TRUE, 0, sizeof(float) * (*nxbtm) * (*nybtm), damp2_z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, damp2_z");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dxi2D, CL_TRUE, 0, sizeof(float) * 4 * (*nxbtm), dxi2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dxi2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dyi2D, CL_TRUE, 0, sizeof(float) * 4 * (*nybtm), dyi2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dyi2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dzi2D, CL_TRUE, 0, sizeof(float) * 4 * (*nzbtm), dzi2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dzi2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dxh2D, CL_TRUE, 0, sizeof(float) * 4 * (*nxbtm), dxh2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dxh2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dyh2D, CL_TRUE, 0, sizeof(float) * 4 * (*nybtm), dyh2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dyh2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dzh2D, CL_TRUE, 0, sizeof(float) * 4 * (*nzbtm), dzh2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, dzh2");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), t2xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), t2xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xzD, CL_TRUE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), t2xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), t2yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yzD, CL_TRUE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), t2yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2zzD, CL_TRUE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), t2zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2zz");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw2_pml);
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2x_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), v2x_px, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2x_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2y_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), v2y_px, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2y_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2z_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), v2z_px, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw2_pml);
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2x_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, v2x_py, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2x_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2y_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, v2y_py, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2y_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2z_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, v2z_py, 0, NULL, NULL);
		CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2z_py");
	}

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2x_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), v2x_pz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2x_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2y_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), v2y_pz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2y_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2z_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), v2z_pz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2z_pz");
    
    //printf("done!\n");
    return;
}

void cpy_h2d_stressInputsCOneTimecl(int   *lbx,
						  int   *lby,
						  int   *nd1_txy,
						  int   *nd1_txz,
						  int   *nd1_tyy,
						  int   *nd1_tyz,
						  float *drti1,
						  float *drth1,
						  float *damp1_x,
						  float *damp1_y,
						  int	*idmat1,
						  float *dxi1,
						  float *dyi1,
						  float *dzi1,
						  float *dxh1,
						  float *dyh1,
						  float *dzh1,
						  float *v1x,
						  float *v1y,
						  float *v1z,
						  float *t1xx_px,
						  float *t1xy_px,
						  float *t1xz_px,
						  float *t1yy_px,
						  float *qt1xx_px,
						  float *qt1xy_px,
						  float *qt1xz_px,
						  float *qt1yy_px,
						  float *t1xx_py,
						  float *t1xy_py,
						  float *t1yy_py,
						  float *t1yz_py,
						  float *qt1xx_py,
						  float *qt1xy_py,
						  float *qt1yy_py,
						  float *qt1yz_py,
						  float *qt1xx,
						  float *qt1xy,
						  float *qt1xz,
						  float *qt1yy,
						  float *qt1yz,
						  float *qt1zz,
						  float *clamda,
						  float *cmu,
						  float *epdt,
						  float *qwp,
						  float *qws,
						  float *qwt1,
						  float *qwt2,
						  int   *nd2_txy,
						  int   *nd2_txz,
						  int   *nd2_tyy,
						  int   *nd2_tyz,
						  float *drti2,
						  float *drth2,
						  int	*idmat2,
						  float *damp2_x,
						  float *damp2_y,
						  float *damp2_z,
						  float *dxi2,
						  float *dyi2,
						  float *dzi2,
						  float *dxh2,
						  float *dyh2,
						  float *dzh2,
						  float *v2x,
						  float *v2y,
						  float *v2z,
						  float *qt2xx,
						  float *qt2xy,
						  float *qt2xz,
						  float *qt2yy,
						  float *qt2yz,
						  float *qt2zz,
						  float *t2xx_px,
						  float *t2xy_px,
						  float *t2xz_px,
						  float *t2yy_px,
						  float *qt2xx_px,
						  float *qt2xy_px,
						  float *qt2xz_px,
						  float *qt2yy_px,
						  float *t2xx_py,
						  float *t2xy_py,
						  float *t2yy_py,
						  float *t2yz_py,
						  float *qt2xx_py,
						  float *qt2xy_py,
						  float *qt2yy_py,
						  float *qt2yz_py,
						  float *t2xx_pz,
						  float *t2xz_pz,
						  float *t2yz_pz,
						  float *t2zz_pz,
						  float *qt2xx_pz,
						  float *qt2xz_pz,
						  float *qt2yz_pz,
						  float *qt2zz_pz,
						  int   *nmat,		//dimension #, int
						  int	*mw1_pml1,	//int
						  int	*mw2_pml1,	//int
						  int	*nxtop,		//int
						  int	*nytop,		//int
						  int   *nztop,
						  int	*mw1_pml,	//int
						  int   *mw2_pml,	//int
						  int	*nxbtm,		//int
						  int	*nybtm,		//int
						  int	*nzbtm,
						  int   *nll)
{
    //printf("[OpenCL] initial h2d cpy for stress ........");
    // printf("lbx[0] = %d, lbx[1] = %d\n", lbx[0], lbx[1]);
    // printf("lby[0] = %d, lby[1] = %d\n", lby[0], lby[1]);

	cl_int errNum;
	int nti, nth;

	//for inner_I
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], nd1_txyD, CL_TRUE, 0, sizeof(int) * 18, nd1_txy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd1_txy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], nd1_txzD, CL_TRUE, 0, sizeof(int) * 18, nd1_txz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd1_txz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], nd1_tyyD, CL_TRUE, 0, sizeof(int) * 18, nd1_tyy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd1_tyy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], nd1_tyzD, CL_TRUE, 0, sizeof(int) * 18, nd1_tyz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd1_tyz");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], drti1D, CL_TRUE, 0, sizeof(float) * (*mw1_pml1) * 2, drti1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, drti1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], drth1D, CL_TRUE, 0, sizeof(float) * (*mw1_pml1) * 2, drth1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, drth1");

	if (lbx[1] >= lbx[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], damp1_xD, CL_TRUE, 0, sizeof(float) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1), damp1_x, 0, NULL, NULL);
		CHECK_ERROR(errNum, "InputDataCopyHostToDevice, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], damp1_yD, CL_TRUE, 0, sizeof(float) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1), damp1_y, 0, NULL, NULL);
		CHECK_ERROR(errNum, "InputDataCopyHostToDevice, damp1_y");
	}

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], idmat1D, CL_TRUE, 0, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1), idmat1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, idmat1");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dxi1D, CL_TRUE, 0, sizeof(float) * 4 * (*nxtop), dxi1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dxi1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dyi1D, CL_TRUE, 0, sizeof(float) * 4 * (*nytop), dyi1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dyi1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dzi1D, CL_TRUE, 0, sizeof(float) * 4 * (*nztop + 1), dzi1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dzi1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dxh1D, CL_TRUE, 0, sizeof(float) * 4 * (*nxtop), dxh1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dxh1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dyh1D, CL_TRUE, 0, sizeof(float) * 4 * (*nytop), dyh1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dyh1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], dzh1D, CL_TRUE, 0, sizeof(float) * 4 * (*nztop + 1), dzh1, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dzh1");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1xD, CL_TRUE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, v1x");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1yD, CL_TRUE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, v1y");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1zD, CL_TRUE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, v1z");

	if (lbx[1] >= lbx[0])
	{
		nti = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1];
		nth = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + 1 - lbx[0];

		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xx_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * (nti) * (*nytop), t1xx_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1xx_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xy_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * nth * (*nytop), t1xy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1xy_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xz_pxD, CL_TRUE, 0, sizeof(float) * (*nztop+1) * nth * (*nytop), t1xz_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1xz_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yy_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * nti * (*nytop), t1yy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1yy_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xx_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * (nti) * (*nytop), qt1xx_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xx_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xy_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * nth * (*nytop), qt1xy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xy_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xz_pxD, CL_TRUE, 0, sizeof(float) * (*nztop+1) * nth * (*nytop), qt1xz_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xz_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1yy_pxD, CL_TRUE, 0, sizeof(float) * (*nztop) * nti * (*nytop), qt1yy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1yy_px");
	}

	if (lby[1] >= lby[0])
	{
		nti = (lby[1] - lby[0] + 1) * (*mw1_pml) + lby[1];
		nth = (lby[1] - lby[0] + 1) * (*mw1_pml) + 1 - lby[0];
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xx_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nti, t1xx_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1xx_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xy_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nth, t1xy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1xy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yy_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nti, t1yy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1yy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yz_pyD, CL_TRUE, 0, sizeof(float) * (*nztop+1) * (*nxtop) * nth, t1yz_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t1yz_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xx_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nti, qt1xx_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xx_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xy_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nth, qt1xy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1yy_pyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * nti, qt1yy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1yy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1yz_pyD, CL_TRUE, 0, sizeof(float) * (*nztop+1) * (*nxtop) * nth, qt1yz_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1yz_py");
	}

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xxD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), qt1xx, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), qt1xy, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1xzD, CL_TRUE, 0, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop), qt1xz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1yyD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), qt1yy, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1yzD, CL_TRUE, 0, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop), qt1yz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qt1zzD, CL_TRUE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), qt1zz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt1zz");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], clamdaD, CL_TRUE, 0, sizeof(float) * (*nmat), clamda, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, clamda");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], cmuD, CL_TRUE, 0, sizeof(float) * (*nmat), cmu, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, cmu");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], epdtD, CL_TRUE, 0, sizeof(float) * (*nll), epdt, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, epdt");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qwpD, CL_TRUE, 0, sizeof(float) * (*nmat), qwp, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qwp");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qwsD, CL_TRUE, 0, sizeof(float) * (*nmat), qws, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qws");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qwt1D, CL_TRUE, 0, sizeof(float) * (*nll), qwt1, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qwt1");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], qwt2D, CL_TRUE, 0, sizeof(float) * (*nll), qwt2, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qwt2");

	//for inner_II
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], nd2_txyD, CL_TRUE, 0, sizeof(int) * 18, nd2_txy, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd2_txy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], nd2_txzD, CL_TRUE, 0, sizeof(int) * 18, nd2_txz, 0, NULL, NULL); 
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd2_txz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], nd2_tyyD, CL_TRUE, 0, sizeof(int) * 18, nd2_tyy, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd2_tyy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], nd2_tyzD, CL_TRUE, 0, sizeof(int) * 18, nd2_tyz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, nd2_tyz");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], drti2D, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * 2, drti2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, drti2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], drth2D, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * 2, drth2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, drth2");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], idmat2D, CL_TRUE, 0, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm +1), idmat2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], damp2_xD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1), damp2_x, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], damp2_yD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1), damp2_y, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, damp2_y");
	}
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], damp2_zD, CL_TRUE, 0, sizeof(float) * (*nxbtm) * (*nybtm), damp2_z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, damp2_z");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dxi2D, CL_TRUE, 0, sizeof(float) * 4 * (*nxbtm), dxi2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dxi2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dyi2D, CL_TRUE, 0, sizeof(float) * 4 * (*nybtm), dyi2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dyi2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dzi2D, CL_TRUE, 0, sizeof(float) * 4 * (*nzbtm), dzi2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dzi2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dxh2D, CL_TRUE, 0, sizeof(float) * 4 * (*nxbtm), dxh2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dxh2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dyh2D, CL_TRUE, 0, sizeof(float) * 4 * (*nybtm), dyh2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dyh2");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], dzh2D, CL_TRUE, 0, sizeof(float) * 4 * (*nzbtm), dzh2, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, dzh2");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2xD, CL_TRUE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, v2x");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2yD, CL_TRUE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, v2y");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2zD, CL_TRUE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, v2z");

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), qt2xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), qt2xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xzD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), qt2xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2yyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), qt2yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2yzD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), qt2yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2zzD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), qt2zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2zz");

	if (lbx[1] >= lbx[0])
	{
        nti = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + lbx[1];
        nth = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + 1 - lbx[0];
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xx_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nti * (*nybtm), t2xx_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2xx_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xy_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nth * (*nybtm), t2xy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2xy_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xz_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nth * (*nybtm), t2xz_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2xz_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yy_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nti * (*nybtm), t2yy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2yy_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xx_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nti * (*nybtm), qt2xx_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xx_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xy_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nth * (*nybtm), qt2xy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xy_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xz_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nth * (*nybtm), qt2xz_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xz_px");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2yy_pxD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * nti * (*nybtm), qt2yy_px, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2yy_px");
	}

	if (lby[1] >= lby[0])
	{
        nti = (lby[1] - lby[0] + 1) * (*mw2_pml) + lby[1];
        nth = (lby[1] - lby[0] + 1) * (*mw2_pml) + 1 - lby[0];
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xx_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, t2xx_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2xx_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xy_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, t2xy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2xy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yy_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, t2yy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2yy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yz_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, t2yz_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2yz_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xx_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, qt2xx_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xx_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xy_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, qt2xy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2yy_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, qt2yy_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2yy_py");
		errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2yz_pyD, CL_TRUE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, qt2yz_py, 0, NULL, NULL);
        CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2yz_py");
	}

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xx_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), t2xx_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2xx_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xz_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), t2xz_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2xz_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yz_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), t2yz_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2yz_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2zz_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), t2zz_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, t2zz_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xx_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), qt2xx_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xx_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2xz_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), qt2xz_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2xz_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2yz_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), qt2yz_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2yz_pz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], qt2zz_pzD, CL_TRUE, 0, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), qt2zz_pz, 0, NULL, NULL);
    CHECK_ERROR(errNum, "InputDataCopyHostToDevice, qt2zz_pz");

    //printf("done!\n");
	return;
}

void cpy_h2d_velocityInputsC_opencl(float *t1xx,
							float *t1xy,
							float *t1xz,
							float *t1yy,
							float *t1yz,
							float *t1zz,
							float *t2xx,
							float *t2xy,
							float *t2xz,
							float *t2yy,
							float *t2yz,
							float *t2zz,
							int	*nxtop,		
							int	*nytop,		
							int *nztop,
							int	*nxbtm,		
							int	*nybtm,		
							int	*nzbtm)
{
    //printf("[OpenCL] h2d cpy for input ........");

	cl_int errNum;
	int i;

	//for inner_I
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xxD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), t1xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xyD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), t1xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xzD, CL_FALSE, 0, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), t1xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yyD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), t1yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yzD, CL_FALSE, 0, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), t1yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1zzD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), t1zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t1zz");

	//for inner_II
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xxD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), t2xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xyD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), t2xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), t2xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yyD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), t2yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), t2yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2zzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), t2zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "InputDataCopyHostToDevice1, t2zz");
#ifdef DISFD_H2D_SYNC_KERNEL 
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		errNum = clFinish(_cl_commandQueues[i]);
		if(errNum != CL_SUCCESS) {
			fprintf(stderr, "Vel H2D Error!\n");
		}
	}
#endif
    //printf("done!\n");

	return;
}

void cpy_h2d_velocityOutputsC_opencl(float *v1x,
							  float *v1y,
							  float *v1z,
							  float *v2x,
							  float *v2y,
							  float *v2z,
							  int	*nxtop,	
							  int	*nytop,
							  int   *nztop,
							  int	*nxbtm,
							  int	*nybtm,
							  int	*nzbtm)
{
    //printf("[OpenCL] h2d cpy for output ........");

	cl_int errNum;
	int i;

	//for inner_I
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1xD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1x");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1yD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1y");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1zD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v1z");

	//for inner_II
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2xD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2x");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2yD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2y");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2zD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice1, v2z");
#ifdef DISFD_H2D_SYNC_KERNEL 
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		errNum = clFinish(_cl_commandQueues[i]);
		if(errNum != CL_SUCCESS) {
			fprintf(stderr, "Vel H2D Error!\n");
		}
	}
#endif
    //printf("done!\n");
	return;
}

void cpy_d2h_velocityOutputsC_opencl(float *v1x, 
							  float *v1y,
							  float *v1z,
							  float *v2x,
							  float *v2y,
							  float *v2z,
							  int	*nxtop,
							  int	*nytop,
							  int   *nztop,
							  int	*nxbtm,
							  int	*nybtm,
							  int	*nzbtm)
{
    //printf("[OpenCL] d2h cpy for output ........");

	cl_int errNum;
	int i;
	//for inner_I
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], v1xD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost1, v1x");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], v1yD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost1, v1y");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], v1zD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost1, v1z");

	//for inner_II
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], v2xD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost1, v2x");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], v2yD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost1, v2y");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], v2zD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost1, vzz");

	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		errNum = clFinish(_cl_commandQueues[i]);
		if(errNum != CL_SUCCESS) {
			fprintf(stderr, "Vel H2D Error!\n");
		}
	}
    //printf("done!\n");
	return;
}

void compute_velocityC_opencl(int *nztop, int *nztm1, float *ca, int *lbx,
			 int *lby, int *nd1_vel, float *rhoM, float *drvh1M, float *drti1M,
             float *damp1_xM, float *damp1_yM, int *idmat1M, float *dxi1M, float *dyi1M,
             float *dzi1M, float *dxh1M, float *dyh1M, float *dzh1M, float *t1xxM,
             float *t1xyM, float *t1xzM, float *t1yyM, float *t1yzM, float *t1zzM,
             void **v1xMp, void **v1yMp, void **v1zMp, float *v1x_pxM, float *v1y_pxM,
             float *v1z_pxM, float *v1x_pyM, float *v1y_pyM, float *v1z_pyM, 
             int *nzbm1, int *nd2_vel, float *drvh2M, float *drti2M, 
             int *idmat2M, float *damp2_xM, float *damp2_yM, float *damp2_zM,
             float *dxi2M, float *dyi2M, float *dzi2M, float *dxh2M, float *dyh2M,
             float *dzh2M, float *t2xxM, float *t2xyM, float *t2xzM, float *t2yyM,
             float *t2yzM, float *t2zzM, void **v2xMp, void **v2yMp, void **v2zMp,
             float *v2x_pxM, float *v2y_pxM, float *v2z_pxM, float *v2x_pyM, 
             float *v2y_pyM, float *v2z_pyM, float *v2x_pzM, float *v2y_pzM,
             float *v2z_pzM, int *nmat,	int *mw1_pml1, int *mw2_pml1, 
             int *nxtop, int *nytop, int *mw1_pml, int *mw2_pml,
             int *nxbtm, int *nybtm, int *nzbtm, int *myid)
{
	int i;
    //printf("[OpenCL] velocity computation:\n"); 
    cl_int errNum;
	//define the dimensions of different kernels
	int blockSizeX = 16;
	int blockSizeY = 16;

    float *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;

    // extract specific input/output pointers
    v1xM=(float *) *v1xMp;
    v1yM=(float *) *v1yMp;
    v1zM=(float *) *v1zMp;
    v2xM=(float *) *v2xMp;
    v2yM=(float *) *v2yMp;
    v2zM=(float *) *v2zMp;

    procID = *myid;

    gettimeofday(&t1, NULL);
	Start(&h2dTimerVelocity);
    cpy_h2d_velocityInputsC_opencl(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);

    cpy_h2d_velocityOutputsC_opencl(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	Stop(&h2dTimerVelocity);
    gettimeofday(&t2, NULL);
    tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    totalTimeH2DV += tmpTime;

    gettimeofday(&t1, NULL);
    for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
	Start(&kernelTimerVelocity[i]);
	}
#ifdef DISFD_PAPI
	papi_start_all_events();
#endif
	size_t dimBlock[3] = {blockSizeX, blockSizeY, 1};
#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX1 = (*nztm1 + 1)/blockSizeX + 1;
	int gridSizeY1 = (nd1_vel[9] - nd1_vel[8])/blockSizeY + 1;
#else
	int gridSizeX1 = (nd1_vel[3] - nd1_vel[2])/blockSizeX + 1;
	int gridSizeY1 = (nd1_vel[9] - nd1_vel[8])/blockSizeY + 1;
#endif 
	size_t dimGrid1[3] = {gridSizeX1, gridSizeY1, 1};
	//OpenCL code
	errNum = clSetKernelArg(_cl_Kernel_velocity_inner_IC, 0, sizeof(int), nztop);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 1, sizeof(int), nztm1);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 2, sizeof(float), ca);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 3, sizeof(cl_mem), &nd1_velD);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 4, sizeof(cl_mem), &rhoD);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 5, sizeof(cl_mem), &idmat1D);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 6, sizeof(cl_mem), &dxi1D);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 7, sizeof(cl_mem), &dyi1D);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 8, sizeof(cl_mem), &dzi1D);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 9, sizeof(cl_mem), &dxh1D);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 10, sizeof(cl_mem), &dyh1D);
    errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 11, sizeof(cl_mem), &dzh1D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 12, sizeof(cl_mem), &t1xxD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 13, sizeof(cl_mem), &t1xyD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 14, sizeof(cl_mem), &t1xzD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 15, sizeof(cl_mem), &t1yyD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 16, sizeof(cl_mem), &t1yzD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 17, sizeof(cl_mem), &t1zzD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 18, sizeof(int), nxtop);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 19, sizeof(int), nytop);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 20, sizeof(cl_mem), &v1xD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 21, sizeof(cl_mem), &v1yD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IC, 22, sizeof(cl_mem), &v1zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_velocity_inner_IC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid1[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid1[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid1[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_velocity_inner_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_velocity_inner_IC for execution!\n");
    }
    //errNum = clFinish(_cl_commandQueues[0]);
    //if(errNum != CL_SUCCESS) {
    //    fprintf(stderr, "Error: finishing velocity for execution!\n");
	//}

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX2 = (*nztop - 1)/blockSizeX + 1;
	int gridSizeY2 = (nd1_vel[5] - nd1_vel[0])/blockSizeX + 1;
#else
	int gridSizeX2 = (nd1_vel[5] - nd1_vel[0])/blockSizeX + 1;
	int gridSizeY2 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
	size_t dimGrid2[3] = {gridSizeX2, gridSizeY2, 1};
    //	printf("myid = %d, grid2 = (%d, %d)\n", *myid, gridSizeX2, gridSizeY2);

	if (lbx[1] >= lbx[0])
	{
		errNum = clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 0, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 1, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 2, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 3, sizeof(cl_mem), &nd1_velD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 4, sizeof(cl_mem), &rhoD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 5, sizeof(cl_mem), &drvh1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 6, sizeof(cl_mem), &drti1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 7, sizeof(cl_mem), &damp1_xD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 8, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 9, sizeof(cl_mem), &dxi1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 10, sizeof(cl_mem), &dyi1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 11, sizeof(cl_mem), &dzi1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 12, sizeof(cl_mem), &dxh1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 13, sizeof(cl_mem), &dyh1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 14, sizeof(cl_mem), &dzh1D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 15, sizeof(cl_mem), &t1xxD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 16, sizeof(cl_mem), &t1xyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 17, sizeof(cl_mem), &t1xzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 18, sizeof(cl_mem), &t1yyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 19, sizeof(cl_mem), &t1yzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 20, sizeof(cl_mem), &t1zzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 21, sizeof(int), mw1_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 22, sizeof(int), mw1_pml);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 23, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 24, sizeof(int), nytop);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 25, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 26, sizeof(cl_mem), &v1xD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 27, sizeof(cl_mem), &v1yD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 28, sizeof(cl_mem), &v1zD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 29, sizeof(cl_mem), &v1x_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 30, sizeof(cl_mem), &v1y_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IC, 31, sizeof(cl_mem), &v1z_pxD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_vel_PmlX_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid2[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid2[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid2[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_vel_PmlX_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_vel_PmlX_IC for execution!\n");
        }
    }

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX3 = (*nztop-1)/blockSizeX + 1;
	int gridSizeY3 = (nd1_vel[11] - nd1_vel[6])/blockSizeY + 1;

#else
	int gridSizeX3 = (lby[1] - lby[0])/blockSizeX + 1;
	int gridSizeY3 = (nd1_vel[11] - nd1_vel[6])/blockSizeY + 1;
#endif
	size_t dimGrid3[3] = {gridSizeX3, gridSizeY3, 1};
	
    //	printf("myid = %d, grid3 = (%d, %d)\n", *myid, gridSizeX3, gridSizeY3);
	if (lby[1] >= lby[0])
	{
		clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 0, sizeof(int), nztop);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 1, sizeof(float), ca);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 2, sizeof(int), &lby[0]);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 3, sizeof(int), &lby[1]);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 4, sizeof(cl_mem), &nd1_velD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 5, sizeof(cl_mem), &rhoD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 6, sizeof(cl_mem), &drvh1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 7, sizeof(cl_mem), &drti1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 8, sizeof(cl_mem), &idmat1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 9, sizeof(cl_mem), &damp1_yD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 10, sizeof(cl_mem), &dxi1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 11, sizeof(cl_mem), &dyi1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 12, sizeof(cl_mem), &dzi1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 13, sizeof(cl_mem), &dxh1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 14, sizeof(cl_mem), &dyh1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 15, sizeof(cl_mem), &dzh1D);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 16, sizeof(cl_mem), &t1xxD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 17, sizeof(cl_mem), &t1xyD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 18, sizeof(cl_mem), &t1xzD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 19, sizeof(cl_mem), &t1yyD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 20, sizeof(cl_mem), &t1yzD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 21, sizeof(cl_mem), &t1zzD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 22, sizeof(int), mw1_pml1);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 23, sizeof(int), mw1_pml);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 24, sizeof(int), nxtop);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 25, sizeof(int), nytop);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 26, sizeof(cl_mem), &v1xD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 27, sizeof(cl_mem), &v1yD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 28, sizeof(cl_mem), &v1zD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 29, sizeof(cl_mem), &v1x_pyD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 30, sizeof(cl_mem), &v1y_pyD);
        clSetKernelArg(_cl_Kernel_vel_PmlY_IC, 31, sizeof(cl_mem), &v1z_pyD);
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid3[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid3[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid3[2]*localWorkSize[2];
        clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_vel_PmlY_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    }

#ifdef DISFD_USE_OPTIMIZED
        int gridSizeX4 = (nd2_vel[15] - 2)/blockSizeX + 1;
	int gridSizeY4 = (nd2_vel[9] - nd2_vel[8])/blockSizeY + 1;	
#else
	int gridSizeX4 = (nd2_vel[3] - nd2_vel[2])/blockSizeX + 1;
	int gridSizeY4 = (nd2_vel[9] - nd2_vel[8])/blockSizeY + 1;
#endif
	size_t dimGrid4[3] = {gridSizeX4, gridSizeY4, 1};
    //	printf("myid = %d, grid4 = (%d, %d)\n", *myid, gridSizeX4, gridSizeY4);
	
	errNum = clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 0, sizeof(float), ca);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 1, sizeof(cl_mem), &nd2_velD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 2, sizeof(cl_mem), &rhoD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 3, sizeof(cl_mem), &dxi2D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 4, sizeof(cl_mem), &dyi2D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 5, sizeof(cl_mem), &dzi2D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 6, sizeof(cl_mem), &dxh2D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 7, sizeof(cl_mem), &dyh2D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 8, sizeof(cl_mem), &dzh2D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 9, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 10, sizeof(cl_mem), &t2xxD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 11, sizeof(cl_mem), &t2xyD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 12, sizeof(cl_mem), &t2xzD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 13, sizeof(cl_mem), &t2yyD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 14, sizeof(cl_mem), &t2yzD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 15, sizeof(cl_mem), &t2zzD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 16, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 17, sizeof(int), nybtm);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 18, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 19, sizeof(cl_mem), &v2xD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 20, sizeof(cl_mem), &v2yD);
	errNum |= clSetKernelArg(_cl_Kernel_velocity_inner_IIC, 21, sizeof(cl_mem), &v2zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_velocity_inner_IIC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid4[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid4[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid4[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_velocity_inner_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_velocity_inner_IIC for execution!\n");
    }

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX5 = (*nzbm1 - 1)/blockSizeX + 1;
        int gridSizeY5 = (nd2_vel[5] - nd2_vel[0])/blockSizeY + 1;
#else 
	int gridSizeX5 = (nd2_vel[5] - nd2_vel[0])/blockSizeX + 1;
	int gridSizeY5 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
	size_t dimGrid5[3] = {gridSizeX5, gridSizeY5, 1};
    //	printf("myid = %d, grid5 = (%d, %d)\n", *myid, gridSizeX5, gridSizeY5);
	
	if (lbx[1] >= lbx[0])
	{
		errNum = clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 0, sizeof(int), nzbm1);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 1, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 2, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 3, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 4, sizeof(cl_mem), &nd2_velD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 5, sizeof(cl_mem), &drvh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 6, sizeof(cl_mem), &drti2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 7, sizeof(cl_mem), &rhoD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 8, sizeof(cl_mem), &damp2_xD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 9, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 10, sizeof(cl_mem), &dxi2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 11, sizeof(cl_mem), &dyi2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 12, sizeof(cl_mem), &dzi2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 13, sizeof(cl_mem), &dxh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 14, sizeof(cl_mem), &dyh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 15, sizeof(cl_mem), &dzh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 16, sizeof(cl_mem), &t2xxD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 17, sizeof(cl_mem), &t2xyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 18, sizeof(cl_mem), &t2xzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 19, sizeof(cl_mem), &t2yyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 20, sizeof(cl_mem), &t2yzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 21, sizeof(cl_mem), &t2zzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 22, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 23, sizeof(int), mw2_pml);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 24, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 25, sizeof(int), nybtm);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 26, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 27, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 28, sizeof(cl_mem), &v2yD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 29, sizeof(cl_mem), &v2zD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 30, sizeof(cl_mem), &v2x_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 31, sizeof(cl_mem), &v2y_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlX_IIC, 32, sizeof(cl_mem), &v2z_pxD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_vel_PmlX_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid5[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid5[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid5[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_vel_PmlX_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_vel_PmlX_IIC for execution!\n");
        }
    }

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX6 = (*nzbm1 -1)/blockSizeX + 1;
	int gridSizeY6 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
#else
	int gridSizeX6 = (lby[1] - lby[0])/blockSizeX + 1;
	int gridSizeY6 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
#endif
	size_t dimGrid6[3] = {gridSizeX6, gridSizeY6, 1};
    //	printf("myid = %d, grid = (%d, %d)\n", *myid, gridSizeX6, gridSizeY6);

	if (lby[1] >= lby[0])
	{
		errNum = clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 0, sizeof(int), nzbm1);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 1, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 2, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 3, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 4, sizeof(cl_mem), &nd2_velD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 5, sizeof(cl_mem), &drvh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 6, sizeof(cl_mem), &drti2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 7, sizeof(cl_mem), &rhoD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 8, sizeof(cl_mem), &damp2_yD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 9, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 10, sizeof(cl_mem), &dxi2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 11, sizeof(cl_mem), &dyi2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 12, sizeof(cl_mem), &dzi2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 13, sizeof(cl_mem), &dxh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 14, sizeof(cl_mem), &dyh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 15, sizeof(cl_mem), &dzh2D);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 16, sizeof(cl_mem), &t2xxD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 17, sizeof(cl_mem), &t2xyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 18, sizeof(cl_mem), &t2xzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 19, sizeof(cl_mem), &t2yyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 20, sizeof(cl_mem), &t2yzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 21, sizeof(cl_mem), &t2zzD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 22, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 23, sizeof(int), mw2_pml);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 24, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 25, sizeof(int), nybtm);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 26, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 27, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 28, sizeof(cl_mem), &v2yD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 29, sizeof(cl_mem), &v2zD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 30, sizeof(cl_mem), &v2x_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 31, sizeof(cl_mem), &v2y_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_vel_PmlY_IIC, 32, sizeof(cl_mem), &v2z_pyD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_vel_PmlY_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid6[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid6[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid6[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_vel_PmlY_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_vel_PmlY_IIC for execution!\n");
        }
    }

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX7 = (*nzbm1 - 1)/blockSizeX + 1;
	int gridSizeY7 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
#else
	int gridSizeX7 = (nd2_vel[5] - nd2_vel[0])/blockSizeX + 1;
	int gridSizeY7 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
#endif
	size_t dimGrid7[3] = {gridSizeX7, gridSizeY7, 1};
    //	printf("myid = %d, grid7 = (%d, %d)\n", *myid, gridSizeX7, gridSizeY7);
	
	errNum = clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 0, sizeof(int), nzbm1);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 1, sizeof(float), ca);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 2, sizeof(cl_mem), &nd2_velD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 3, sizeof(cl_mem), &drvh2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 4, sizeof(cl_mem), &drti2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 5, sizeof(cl_mem), &rhoD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 6, sizeof(cl_mem), &damp2_zD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 7, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 8, sizeof(cl_mem), &dxi2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 9, sizeof(cl_mem), &dyi2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 10, sizeof(cl_mem), &dzi2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 11, sizeof(cl_mem), &dxh2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 12, sizeof(cl_mem), &dyh2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 13, sizeof(cl_mem), &dzh2D);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 14, sizeof(cl_mem), &t2xxD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 15, sizeof(cl_mem), &t2xyD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 16, sizeof(cl_mem), &t2xzD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 17, sizeof(cl_mem), &t2yyD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 18, sizeof(cl_mem), &t2yzD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 19, sizeof(cl_mem), &t2zzD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 20, sizeof(int), mw2_pml1);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 21, sizeof(int), mw2_pml);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 22, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 23, sizeof(int), nybtm);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 24, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 25, sizeof(cl_mem), &v2xD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 26, sizeof(cl_mem), &v2yD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 27, sizeof(cl_mem), &v2zD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 28, sizeof(cl_mem), &v2x_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 29, sizeof(cl_mem), &v2y_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_vel_PmlZ_IIC, 30, sizeof(cl_mem), &v2z_pzD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_vel_PmlZ_IIC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid7[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid7[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid7[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_vel_PmlZ_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_vel_PmlZ_IIC for execution!\n");
    }
    
    for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
    errNum = clFinish(_cl_commandQueues[i]);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: finishing stress velocity for execution!\n");
    }
	Stop(&kernelTimerVelocity[i]);
    }
#ifdef DISFD_PAPI
	papi_accum_all_events();
	//papi_stop_all_events();
	papi_print_all_events();
	//papi_reset_all_events();
#endif
	gettimeofday(&t2, NULL);
    tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    totalTimeCompV += tmpTime;

    gettimeofday(&t1, NULL);
	Start(&d2hTimerVelocity);
    cpy_d2h_velocityOutputsC_opencl(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop,	nytop, nztop, nxbtm, nybtm, nzbtm);
	Stop(&d2hTimerVelocity);
    gettimeofday(&t2, NULL);
    tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    totalTimeD2HV += tmpTime;

#ifdef DISFD_DEBUG
  int size = (*nztop + 2) * (*nxtop + 3) * (*nytop + 3); 
  write_output(v1xM, size, "OUTPUT_ARRAYS/v1xM.txt");
  write_output(v1yM, size, "OUTPUT_ARRAYS/v1yM.txt");
  write_output(v1zM, size, "OUTPUT_ARRAYS/v1zM.txt");
  size = (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3);
  write_output(v2xM, size, "OUTPUT_ARRAYS/v2xM.txt");
  write_output(v2yM, size, "OUTPUT_ARRAYS/v2yM.txt");
  write_output(v2zM, size, "OUTPUT_ARRAYS/v2zM.txt");
#endif
    return;
}

void cpy_h2d_stressInputsC_opencl(float *v1x,
						   float *v1y,
						   float *v1z,
						   float *v2x,
						   float *v2y,
						   float *v2z,
						   int	*nxtop,
						   int	*nytop,
						   int  *nztop,
						   int	*nxbtm,
						   int	*nybtm,
						   int	*nzbtm)
{
    //printf("[OpenCL] h2d cpy for input ........");
	int i;

	cl_int errNum;

	//for inner_I
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1xD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, v1x");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1yD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, v1y");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], v1zD, CL_FALSE, 0, sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), v1z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, v1z");

	//for inner_II
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2xD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2x, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, v2x");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2yD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2y, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, v2y");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], v2zD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), v2z, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, v2z");

#ifdef DISFD_H2D_SYNC_KERNEL 
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		errNum = clFinish(_cl_commandQueues[i]);
		if(errNum != CL_SUCCESS) {
			fprintf(stderr, "Vel H2D Error!\n");
		}
	}
#endif
    //printf("done!\n");
	return;
}

void cpy_h2d_stressOutputsC_opencl(float *t1xx,
						    float *t1xy,
						    float *t1xz,
						    float *t1yy,
						    float *t1yz,
						    float *t1zz,
						    float *t2xx,
						    float *t2xy,
						    float *t2xz,
						    float *t2yy,
						    float *t2yz,
						    float *t2zz,
						    int	  *nxtop,
						    int	  *nytop,
						    int   *nztop,
						    int	  *nxbtm,
						    int	  *nybtm,
						    int	  *nzbtm)
{
    //printf("[OpenCL] h2d cpy for output ........");
	cl_int errNum;
	int i;
	int nth, nti;

	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xxD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), t1xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t1xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xyD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), t1xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t1xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1xzD, CL_FALSE, 0, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), t1xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t1xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yyD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), t1yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t1yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1yzD, CL_FALSE, 0, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), t1yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t1yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[0], t1zzD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), t1zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t1zz");

	//for inner_II
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xxD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), t2xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t2xx");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xyD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), t2xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t2xy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2xzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), t2xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t2xz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yyD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), t2yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t2yy");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2yzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), t2yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t2yz");
	errNum = clEnqueueWriteBuffer(_cl_commandQueues[1], t2zzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), t2zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyHostToDevice, t2zz");

#ifdef DISFD_H2D_SYNC_KERNEL 
	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		errNum = clFinish(_cl_commandQueues[i]);
		if(errNum != CL_SUCCESS) {
			fprintf(stderr, "Vel H2D Error!\n");
		}
	}
#endif
    //printf("done!\n");

	return;
}

void cpy_d2h_stressOutputsC_opencl(float *t1xx,
						    float *t1xy,
						    float *t1xz,
						    float *t1yy,
						    float *t1yz,
						    float *t1zz,
						    float *t2xx,
						    float *t2xy,
						    float *t2xz,
						    float *t2yy,
						    float *t2yz,
						    float *t2zz,
						    int	  *nxtop,
						    int	  *nytop,
						    int   *nztop,
						    int	  *nxbtm,
						    int	  *nybtm,
						    int	  *nzbtm)
{
    //printf("[OpenCL] d2h cpy for output ........");
	cl_int errNum;
	int i;
    // printf("\nnxtop=%d, nytop=%d, nztop=%d\n", *nxtop, *nytop, *nztop);
    // printf("nxbtm=%d, nybtm=%d, nzbtm=%d\n", *nxbtm, *nybtm, *nzbtm);

    // printf("t1xxD: %d\n", sizeof(float) *  sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop));
    
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], t1xxD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), t1xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t1xx");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], t1xyD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), t1xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t1xy");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], t1xzD, CL_FALSE, 0, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), t1xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t1xz");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], t1yyD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), t1yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t1yy");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], t1yzD, CL_FALSE, 0, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), t1yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t1yz");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[0], t1zzD, CL_FALSE, 0, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), t1zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t1zz");

	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], t2xxD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), t2xx, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t2xx");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], t2xyD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), t2xy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t2xy");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], t2xzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), t2xz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t2xz");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], t2yyD, CL_FALSE, 0, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), t2yy, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t2yy");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], t2yzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), t2yz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t2yz");
	errNum = clEnqueueReadBuffer(_cl_commandQueues[1], t2zzD, CL_FALSE, 0, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), t2zz, 0, NULL, NULL);
	CHECK_ERROR(errNum, "outputDataCopyDeviceToHost, t2zz");

	for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
		errNum = clFinish(_cl_commandQueues[i]);
		if(errNum != CL_SUCCESS) {
			fprintf(stderr, "Vel H2D Error!\n");
		}
	}

    //printf("done!\n");
    
	return;
}

void compute_stressC_opencl(int *nxb1, int *nyb1, int *nx1p1, int *ny1p1, int *nxtop, int *nytop, int *nztop, int *mw1_pml,
                int *mw1_pml1, int *nmat, int *nll, int *lbx, int *lby, int *nd1_txy, int *nd1_txz,
                int *nd1_tyy, int *nd1_tyz, int *idmat1M, float *ca, float *drti1M, float *drth1M, float *damp1_xM, float *damp1_yM,
                float *clamdaM, float *cmuM, float *epdtM, float *qwpM, float *qwsM, float *qwt1M, float *qwt2M, float *dxh1M,
                float *dyh1M, float *dzh1M, float *dxi1M, float *dyi1M, float *dzi1M, float *t1xxM, float *t1xyM, float *t1xzM, 
                float *t1yyM, float *t1yzM, float *t1zzM, float *qt1xxM, float *qt1xyM, float *qt1xzM, float *qt1yyM, float *qt1yzM, 
                float *qt1zzM, float *t1xx_pxM, float *t1xy_pxM, float *t1xz_pxM, float *t1yy_pxM, float *qt1xx_pxM, float *qt1xy_pxM,
                float *qt1xz_pxM, float *qt1yy_pxM, float *t1xx_pyM, float *t1xy_pyM, float *t1yy_pyM, float *t1yz_pyM, float *qt1xx_pyM,
                float *qt1xy_pyM, float *qt1yy_pyM, float *qt1yz_pyM, void **v1xMp, void **v1yMp, void **v1zMp,
                int *nxb2, int *nyb2, int *nxbtm, int *nybtm, int *nzbtm, int *mw2_pml, int *mw2_pml1, int *nd2_txy, int *nd2_txz, 
                int *nd2_tyy, int *nd2_tyz, int *idmat2M, 
                float *drti2M, float *drth2M, float *damp2_xM, float *damp2_yM, float *damp2_zM, 
                float *t2xxM, float *t2xyM, float *t2xzM, float *t2yyM, float *t2yzM, float *t2zzM, 
                float *qt2xxM, float *qt2xyM, float *qt2xzM, float *qt2yyM, float *qt2yzM, float *qt2zzM, 
                float *dxh2M, float *dyh2M, float *dzh2M, float *dxi2M, float *dyi2M, float *dzi2M, 
                float *t2xx_pxM, float *t2xy_pxM, float *t2xz_pxM, float *t2yy_pxM, float *t2xx_pyM, float *t2xy_pyM,
                float *t2yy_pyM, float *t2yz_pyM, float *t2xx_pzM, float *t2xz_pzM, float *t2yz_pzM, float *t2zz_pzM,
                float *qt2xx_pxM, float *qt2xy_pxM, float *qt2xz_pxM, float *qt2yy_pxM, float *qt2xx_pyM, float *qt2xy_pyM, 
                float *qt2yy_pyM, float *qt2yz_pyM, float *qt2xx_pzM, float *qt2xz_pzM, float *qt2yz_pzM, float *qt2zz_pzM,
                void **v2xMp, void **v2yMp, void **v2zMp, int *myid)
{
    //printf("[OpenCL] stress computation:\n"); 
	int i;
    cl_int errNum;

	float *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;
	int blockSizeX = 16;
	int blockSizeY = 16;
	size_t dimBlock[3] = {blockSizeX, blockSizeY, 1};

	v1xM = (float *) *v1xMp;
	v1yM = (float *) *v1yMp;
	v1zM = (float *) *v1zMp;
	v2xM = (float *) *v2xMp;
	v2yM = (float *) *v2yMp;
	v2zM = (float *) *v2zMp;

    gettimeofday(&t1, NULL);
	Start(&h2dTimerStress);
	cpy_h2d_stressInputsC_opencl(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	cpy_h2d_stressOutputsC_opencl(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM,t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	Stop(&h2dTimerStress);
    gettimeofday(&t2, NULL);
    tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    totalTimeH2DS += tmpTime;

    gettimeofday(&t1, NULL);
    for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
	Start(&kernelTimerStress[i]);
    }
#ifdef DISFD_USE_OPTIMIZED
    int gridSizeX1 = (nd1_tyy[17] - nd1_tyy[12])/blockSizeX + 1;
    int gridSizeY1 = (nd1_tyy[9] - nd1_tyy[8])/blockSizeY + 1;
#else
	int gridSizeX1 = (nd1_tyy[3] - nd1_tyy[2])/blockSizeX + 1;
    int gridSizeY1 = (nd1_tyy[9] - nd1_tyy[8])/blockSizeY + 1;
#endif
	size_t dimGrid1[3] = {gridSizeX1, gridSizeY1, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 0, sizeof(int), nxb1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 1, sizeof(int), nyb1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 2, sizeof(int), nxtop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 3, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 4, sizeof(cl_mem), &nd1_tyyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 5, sizeof(cl_mem), &idmat1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 6, sizeof(float), ca);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 7, sizeof(cl_mem), &clamdaD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 8, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 9, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 10, sizeof(cl_mem), &qwpD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 11, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 12, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 13, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 14, sizeof(cl_mem), &dxh1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 15, sizeof(cl_mem), &dyh1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 16, sizeof(cl_mem), &dxi1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 17, sizeof(cl_mem), &dyi1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 18, sizeof(cl_mem), &dzi1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 19, sizeof(cl_mem), &t1xxD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 20, sizeof(cl_mem), &t1xyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 21, sizeof(cl_mem), &t1yyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 22, sizeof(cl_mem), &t1zzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 23, sizeof(cl_mem), &qt1xxD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 24, sizeof(cl_mem), &qt1xyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 25, sizeof(cl_mem), &qt1yyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 26, sizeof(cl_mem), &qt1zzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 27, sizeof(cl_mem), &v1xD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 28, sizeof(cl_mem), &v1yD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_IC, 29, sizeof(cl_mem), &v1zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_norm_xy_IC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid1[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid1[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid1[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_norm_xy_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_norm_xy_IC for execution!\n");
    }

#ifdef DISFD_USE_OPTIMIZED
    int gridSizeX2 = (nd1_tyz[17] - nd1_tyz[12])/blockSizeX + 1;
    int gridSizeY2 = (nd1_tyz[9] - nd1_tyz[8])/blockSizeY + 1;
#else
    int gridSizeX2 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeX + 1;
    int gridSizeY2 = (nd1_tyz[9] - nd1_tyz[8])/blockSizeY + 1;
#endif
	size_t dimGrid2[3] = {gridSizeX2, gridSizeY2, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 0, sizeof(int), nxb1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 1, sizeof(int), nyb1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 2, sizeof(int), nxtop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 3, sizeof(int), nytop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 4, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 5, sizeof(cl_mem), &nd1_tyzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 6, sizeof(cl_mem), &idmat1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 7, sizeof(float), ca);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 8, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 9, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 10, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 11, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 12, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 13, sizeof(cl_mem), &dxi1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 14, sizeof(cl_mem), &dyi1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 15, sizeof(cl_mem), &dzh1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 16, sizeof(cl_mem), &v1xD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 17, sizeof(cl_mem), &v1yD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 18, sizeof(cl_mem), &v1zD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 19, sizeof(cl_mem), &t1xzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 20, sizeof(cl_mem), &t1yzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 21, sizeof(cl_mem), &qt1xzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IC, 22, sizeof(cl_mem), &qt1yzD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xz_yz_IC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid2[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid2[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid2[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_xz_yz_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xz_yz_IC for execution!\n");
    }
    
	int gridSizeX3Temp1 = ((*ny1p1) + 1)/blockSizeX + 1;
	int gridSizeX3Temp2 = ((*nytop) - 1)/blockSizeX + 1;
    int gridSizeY3Temp1 = ((*nxtop) - 1)/blockSizeY + 1;
    int gridSizeY3Temp2 = ((*nx1p1) + 1)/blockSizeY + 1;
	int gridSizeX3 = (gridSizeX3Temp1 > gridSizeX3Temp2) ? gridSizeX3Temp1 : gridSizeX3Temp2;
	int gridSizeY3 = (gridSizeY3Temp1 > gridSizeY3Temp2) ? gridSizeY3Temp1 : gridSizeY3Temp2;
	size_t dimGrid3[3] = {gridSizeX3, gridSizeY3, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_resetVars, 0, sizeof(int), ny1p1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_resetVars, 1, sizeof(int), nx1p1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_resetVars, 2, sizeof(int), nxtop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_resetVars, 3, sizeof(int), nytop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_resetVars, 4, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_resetVars, 5, sizeof(cl_mem), &t1xzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_resetVars, 6, sizeof(cl_mem), &t1yzD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_resetVars arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid3[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid3[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid3[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_resetVars, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_resetVars for execution!\n");
    }

	if (lbx[1] >= lbx[0])
	{
#ifdef DISFD_USE_OPTIMIZED
        int gridSizeX4 = (nd1_tyy[17] - nd1_tyy[12])/blockSizeX + 1;
        int gridSizeY4 = (nd1_tyy[5] - nd1_tyy[0])/blockSizeY + 1;
#else
		int gridSizeX4 = (nd1_tyy[5] - nd1_tyy[0])/blockSizeX + 1;
		int gridSizeY4 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid4[3] = {gridSizeX4, gridSizeY4, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 2, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 3, sizeof(int), nytop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 5, sizeof(int), mw1_pml);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 6, sizeof(int), mw1_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 7, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 8, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 9, sizeof(cl_mem), &nd1_tyyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 10, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 11, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 12, sizeof(cl_mem), &drti1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 13, sizeof(cl_mem), &damp1_xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 14, sizeof(cl_mem), &clamdaD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 15, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 16, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 17, sizeof(cl_mem), &qwpD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 18, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 19, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 20, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 21, sizeof(cl_mem), &dzi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 22, sizeof(cl_mem), &dxh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 23, sizeof(cl_mem), &dyh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 24, sizeof(cl_mem), &v1xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 25, sizeof(cl_mem), &v1yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 26, sizeof(cl_mem), &v1zD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 27, sizeof(cl_mem), &t1xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 28, sizeof(cl_mem), &t1yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 29, sizeof(cl_mem), &t1zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 30, sizeof(cl_mem), &t1xx_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 31, sizeof(cl_mem), &t1yy_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 32, sizeof(cl_mem), &qt1xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 33, sizeof(cl_mem), &qt1yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 34, sizeof(cl_mem), &qt1zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 35, sizeof(cl_mem), &qt1xx_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IC, 36, sizeof(cl_mem), &qt1yy_pxD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_norm_PmlX_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid4[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid4[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid4[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_norm_PmlX_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_norm_PmlX_IC for execution!\n");
        }
    }
	if (lby[1] >= lby[0])
	{
#ifdef DISFD_USE_OPTIMIZED
        int gridSizeX5 = (nd1_tyy[17] - nd1_tyy[12])/blockSizeX + 1;
        int gridSizeY5 = (nd1_tyy[11] - nd1_tyy[6])/blockSizeY + 1;
#else
		int gridSizeX5 = (nd1_tyy[11] - nd1_tyy[6])/blockSizeX + 1;
		int gridSizeY5 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
		size_t dimGrid5[3] = {gridSizeX5, gridSizeY5, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 2, sizeof(int), mw1_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 3, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 5, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 6, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 7, sizeof(cl_mem), &nd1_tyyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 8, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 9, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 10, sizeof(cl_mem), &drti1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 11, sizeof(cl_mem), &damp1_yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 12, sizeof(cl_mem), &clamdaD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 13, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 14, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 15, sizeof(cl_mem), &qwpD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 16, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 17, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 18, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 19, sizeof(cl_mem), &dxh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 20, sizeof(cl_mem), &dyh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 21, sizeof(cl_mem), &dzi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 22, sizeof(cl_mem), &t1xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 23, sizeof(cl_mem), &t1yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 24, sizeof(cl_mem), &t1zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 25, sizeof(cl_mem), &qt1xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 26, sizeof(cl_mem), &qt1yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 27, sizeof(cl_mem), &qt1zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 28, sizeof(cl_mem), &t1xx_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 29, sizeof(cl_mem), &t1yy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 30, sizeof(cl_mem), &qt1xx_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 31, sizeof(cl_mem), &qt1yy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 32, sizeof(cl_mem), &v1xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 33, sizeof(cl_mem), &v1yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_IC, 34, sizeof(cl_mem), &v1zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_norm_PmlY_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid5[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid5[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid5[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_norm_PmlY_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_norm_PmlY_IC for execution!\n");
        }
	}

	if (lbx[1] >= lbx[0]) 
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX6 = (nd1_txy[17] - nd1_txy[12])/blockSizeX + 1;
		int gridSizeY6 = (nd1_txy[5] - nd1_txy[0])/blockSizeY + 1;
#else
		int gridSizeX6 = (nd1_txy[5] - nd1_txy[0])/blockSizeX + 1;
		int gridSizeY6 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid6[3] = {gridSizeX6, gridSizeY6, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 2, sizeof(int), mw1_pml);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 3, sizeof(int), mw1_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 4, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 5, sizeof(int), nytop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 6, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 7, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 8, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 9, sizeof(cl_mem), &nd1_txyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 10, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 11, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 12, sizeof(cl_mem), &drth1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 13, sizeof(cl_mem), &damp1_xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 14, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 15, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 16, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 17, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 18, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 19, sizeof(cl_mem), &dxi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 20, sizeof(cl_mem), &dyi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 21, sizeof(cl_mem), &t1xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 22, sizeof(cl_mem), &qt1xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 23, sizeof(cl_mem), &t1xy_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 24, sizeof(cl_mem), &qt1xy_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 25, sizeof(cl_mem), &v1xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IC, 26, sizeof(cl_mem), &v1yD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xy_PmlX_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid6[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid6[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid6[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_xy_PmlX_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xy_PmlX_IC for execution!\n");
        }
    }

	if (lby[1] >= lby[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX7 = (nd1_txy[17] - nd1_txy[12])/blockSizeX + 1;
		int gridSizeY7 = (nd1_txy[11] - nd1_txy[6])/blockSizeY + 1;
#else
		int gridSizeX7 = (nd1_txy[11] - nd1_txy[6])/blockSizeX + 1;
		int gridSizeY7 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
		size_t dimGrid7[3] = {gridSizeX7, gridSizeY7, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 2, sizeof(int), mw1_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 3, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 5, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 6, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 7, sizeof(cl_mem), &nd1_txyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 8, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 9, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 10, sizeof(cl_mem), &drth1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 11, sizeof(cl_mem), &damp1_yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 12, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 13, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 14, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 15, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 16, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 17, sizeof(cl_mem), &dxi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 18, sizeof(cl_mem), &dyi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 19, sizeof(cl_mem), &t1xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 20, sizeof(cl_mem), &qt1xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 21, sizeof(cl_mem), &t1xy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 22, sizeof(cl_mem), &qt1xy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 23, sizeof(cl_mem), &v1xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IC, 24, sizeof(cl_mem), &v1yD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xy_PmlY_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid7[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid7[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid7[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_xy_PmlY_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xy_PmlY_IC for execution!\n");
        }
	}

	if (lbx[1] >= lbx[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX8 = (nd1_txz[17] - nd1_txz[12])/blockSizeX + 1;
		int gridSizeY8 = (nd1_txz[5] - nd1_txz[0])/blockSizeX + 1;
#else
		int gridSizeX8 = (nd1_txz[5] - nd1_txz[0])/blockSizeX + 1;
		int gridSizeY8 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid8[3] = {gridSizeX8, gridSizeY8, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 2, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 3, sizeof(int), nytop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 5, sizeof(int), mw1_pml);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 6, sizeof(int), mw1_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 7, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 8, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 9, sizeof(cl_mem), &nd1_txzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 10, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 11, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 12, sizeof(cl_mem), &drth1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 13, sizeof(cl_mem), &damp1_xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 14, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 15, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 16, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 17, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 18, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 19, sizeof(cl_mem), &dxi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 20, sizeof(cl_mem), &dzh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 21, sizeof(cl_mem), &t1xzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 22, sizeof(cl_mem), &qt1xzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 23, sizeof(cl_mem), &t1xz_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 24, sizeof(cl_mem), &qt1xz_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 25, sizeof(cl_mem), &v1xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IC, 26, sizeof(cl_mem), &v1zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xz_PmlX_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid8[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid8[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid8[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_xz_PmlX_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xz_PmlX_IC for execution!\n");
        }
	}

	if (lby[1] >= lby[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX9 = (nd1_txz[17] - nd1_txz[12])/blockSizeX + 1;
		int gridSizeY9 = (nd1_txz[9] - nd1_txz[8])/blockSizeY + 1;
#else
		int gridSizeX9 = (nd1_txz[9] - nd1_txz[8])/blockSizeX + 1;
		int gridSizeY9 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
		size_t dimGrid9[3] = {gridSizeX9, gridSizeY9, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 2, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 3, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 4, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 5, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 6, sizeof(cl_mem), &nd1_txzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 7, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 8, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 9, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 10, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 11, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 12, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 13, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 14, sizeof(cl_mem), &dxi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 15, sizeof(cl_mem), &dzh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 16, sizeof(cl_mem), &t1xzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 17, sizeof(cl_mem), &qt1xzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 18, sizeof(cl_mem), &v1xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IC, 19, sizeof(cl_mem), &v1zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xz_PmlY_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid9[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid9[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid9[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_xz_PmlY_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xz_PmlY_IC for execution!\n");
        }
	}

	if (lbx[1] >= lbx[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX10 = (nd1_tyz[17] - nd1_tyz[12])/blockSizeX + 1;
		int gridSizeY10 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeY + 1;
#else
		int gridSizeX10 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeX + 1;
		int gridSizeY10 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid10[3] = {gridSizeX10, gridSizeY10, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 2, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 3, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 4, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 5, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 6, sizeof(cl_mem), &nd1_tyzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 7, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 8, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 9, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 10, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 11, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 12, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 13, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 14, sizeof(cl_mem), &dyi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 15, sizeof(cl_mem), &dzh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 16, sizeof(cl_mem), &t1yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 17, sizeof(cl_mem), &qt1yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 18, sizeof(cl_mem), &v1yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IC, 19, sizeof(cl_mem), &v1zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_yz_PmlX_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid10[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid10[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid10[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_yz_PmlX_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_yz_PmlX_IC for execution!\n");
        }
    }

	if (lby[1] >= lby[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX11 = (nd1_tyz[17] - nd1_tyz[12])/blockSizeX + 1;
		int gridSizeY11 = (nd1_tyz[11] - nd1_tyz[6])/blockSizeY + 1;
#else
		int gridSizeX11 = (nd1_tyz[11] - nd1_tyz[6])/blockSizeX + 1;
		int gridSizeY11 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
		size_t dimGrid11[3] = {gridSizeX11, gridSizeY11, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 0, sizeof(int), nxb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 1, sizeof(int), nyb1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 2, sizeof(int), mw1_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 3, sizeof(int), nxtop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 5, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 6, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 7, sizeof(cl_mem), &nd1_tyzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 8, sizeof(cl_mem), &idmat1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 9, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 10, sizeof(cl_mem), &drth1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 11, sizeof(cl_mem), &damp1_yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 12, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 13, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 14, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 15, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 16, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 17, sizeof(cl_mem), &dyi1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 18, sizeof(cl_mem), &dzh1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 19, sizeof(cl_mem), &t1yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 20, sizeof(cl_mem), &qt1yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 21, sizeof(cl_mem), &t1yz_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 22, sizeof(cl_mem), &qt1yz_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 23, sizeof(cl_mem), &v1yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IC, 24, sizeof(cl_mem), &v1zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_yz_PmlY_IC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid11[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid11[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid11[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[0], _cl_Kernel_stress_yz_PmlY_IC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_yz_PmlY_IC for execution!\n");
        }
	}
    
#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX12 = (nd2_tyy[15] - nd2_tyy[12])/blockSizeX + 1;
	int gridSizeY12 = (nd2_tyy[9] - nd2_tyy[8])/blockSizeY + 1;
#else
	int gridSizeX12 = (nd2_tyy[3] - nd2_tyy[2])/blockSizeX + 1;
	int gridSizeY12 = (nd2_tyy[9] - nd2_tyy[8])/blockSizeY + 1;
#endif
	size_t dimGrid12[3] = {gridSizeX12, gridSizeY12, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 0, sizeof(int), nxb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 1, sizeof(int), nyb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 2, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 3, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 4, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 5, sizeof(cl_mem), &nd2_tyyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 6, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 7, sizeof(cl_mem), &clamdaD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 8, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 9, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 10, sizeof(cl_mem), &qwpD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 11, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 12, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 13, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 14, sizeof(cl_mem), &t2xxD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 15, sizeof(cl_mem), &t2xyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 16, sizeof(cl_mem), &t2yyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 17, sizeof(cl_mem), &t2zzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 18, sizeof(cl_mem), &qt2xxD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 19, sizeof(cl_mem), &qt2xyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 20, sizeof(cl_mem), &qt2yyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 21, sizeof(cl_mem), &qt2zzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 22, sizeof(cl_mem), &dxh2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 23, sizeof(cl_mem), &dyh2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 24, sizeof(cl_mem), &dxi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 25, sizeof(cl_mem), &dyi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 26, sizeof(cl_mem), &dzi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 27, sizeof(cl_mem), &v2xD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 28, sizeof(cl_mem), &v2yD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_xy_II, 29, sizeof(cl_mem), &v2zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_norm_xy_II arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid12[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid12[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid12[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_norm_xy_II, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_norm_xy_II for execution!\n");
    }

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX13 = (nd2_tyz[15] - nd2_tyz[12])/blockSizeX + 1;
	int gridSizeY13 = (nd2_tyz[9] - nd2_tyz[8])/blockSizeY + 1;
#else
    int gridSizeX13 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeX + 1;
    int gridSizeY13 = (nd2_tyz[9] - nd2_tyz[8])/blockSizeY + 1;
#endif
    size_t dimGrid13[3] = {gridSizeX13, gridSizeY13, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 0, sizeof(int), nxb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 1, sizeof(int), nyb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 2, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 3, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 4, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 5, sizeof(cl_mem), &nd2_tyzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 6, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 7, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 8, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 9, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 10, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 11, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 12, sizeof(cl_mem), &dxi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 13, sizeof(cl_mem), &dyi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 14, sizeof(cl_mem), &dzh2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 15, sizeof(cl_mem), &t2xzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 16, sizeof(cl_mem), &t2yzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 17, sizeof(cl_mem), &qt2xzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 18, sizeof(cl_mem), &qt2yzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 19, sizeof(cl_mem), &v2xD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 20, sizeof(cl_mem), &v2yD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_yz_IIC, 21, sizeof(cl_mem), &v2zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xz_yz_IIC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid13[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid13[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid13[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_xz_yz_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xz_yz_IIC for execution!\n");
    }

	if (lbx[1] >= lbx[0])
	{
#ifdef DISFD_USE_OPTIMIZED
        int gridSizeX14 = (nd2_tyy[17] - nd2_tyy[12])/blockSizeX + 1;
        int gridSizeY14 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeY + 1;
#else
		int gridSizeX14 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeX + 1;
		int gridSizeY14 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid14[3] = {gridSizeX14, gridSizeY14, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 2, sizeof(int), mw2_pml);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 3, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 5, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 6, sizeof(int), nybtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 7, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 8, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 9, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 10, sizeof(cl_mem), &nd2_tyyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 11, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 12, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 13, sizeof(cl_mem), &drti2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 14, sizeof(cl_mem), &damp2_xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 15, sizeof(cl_mem), &clamdaD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 16, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 17, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 18, sizeof(cl_mem), &qwpD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 19, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 20, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 21, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 22, sizeof(cl_mem), &dxh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 23, sizeof(cl_mem), &dyh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 24, sizeof(cl_mem), &dzi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 25, sizeof(cl_mem), &t2xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 26, sizeof(cl_mem), &t2yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 27, sizeof(cl_mem), &t2zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 28, sizeof(cl_mem), &qt2xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 29, sizeof(cl_mem), &qt2yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 30, sizeof(cl_mem), &qt2zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 31, sizeof(cl_mem), &t2xx_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 32, sizeof(cl_mem), &t2yy_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 33, sizeof(cl_mem), &qt2xx_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 34, sizeof(cl_mem), &qt2yy_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 35, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 36, sizeof(cl_mem), &v2yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlX_IIC, 37, sizeof(cl_mem), &v2zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_norm_PmlX_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid14[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid14[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid14[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_norm_PmlX_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_norm_PmlX_IIC for execution!\n");
        }
	}

	if (lby[1] >= lby[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX15 = (nd2_tyy[17] - nd2_tyy[12])/blockSizeX + 1;
		int gridSizeY15 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeY + 1;
#else
		int gridSizeX15 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeX + 1;
		int gridSizeY15 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
		size_t dimGrid15[3] = {gridSizeX15, gridSizeY15, 1};
		errNum = clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 2, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 3, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 4, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 5, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 6, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 7, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 8, sizeof(cl_mem), &nd2_tyyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 9, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 10, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 11, sizeof(cl_mem), &drti2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 12, sizeof(cl_mem), &damp2_yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 13, sizeof(cl_mem), &clamdaD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 14, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 15, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 16, sizeof(cl_mem), &qwpD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 17, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 18, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 19, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 20, sizeof(cl_mem), &dxh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 21, sizeof(cl_mem), &dyh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 22, sizeof(cl_mem), &dzi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 23, sizeof(cl_mem), &t2xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 24, sizeof(cl_mem), &t2yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 25, sizeof(cl_mem), &t2zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 26, sizeof(cl_mem), &qt2xxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 27, sizeof(cl_mem), &qt2yyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 28, sizeof(cl_mem), &qt2zzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 29, sizeof(cl_mem), &t2xx_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 30, sizeof(cl_mem), &t2yy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 31, sizeof(cl_mem), &qt2xx_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 32, sizeof(cl_mem), &qt2yy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 33, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 34, sizeof(cl_mem), &v2yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlY_II, 35, sizeof(cl_mem), &v2zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_norm_PmlY_II arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid15[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid15[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid15[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_norm_PmlY_II, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_norm_PmlY_II for execution!\n");
        }
	}

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX16 = (nd2_tyy[17] - nd2_tyy[16])/blockSizeX + 1;
	int gridSizeY16 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeY + 1;
#else
	int gridSizeX16 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeX + 1;
	int gridSizeY16 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeY + 1;
#endif
	size_t dimGrid16[3] = {gridSizeX16, gridSizeY16, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 0, sizeof(int), nxb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 1, sizeof(int), nyb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 2, sizeof(int), mw2_pml);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 3, sizeof(int), mw2_pml1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 4, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 5, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 6, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 7, sizeof(cl_mem), &nd2_tyyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 8, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 9, sizeof(float), ca);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 10, sizeof(cl_mem), &damp2_zD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 11, sizeof(cl_mem), &drth2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 12, sizeof(cl_mem), &clamdaD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 13, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 14, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 15, sizeof(cl_mem), &qwpD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 16, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 17, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 18, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 19, sizeof(cl_mem), &dxh2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 20, sizeof(cl_mem), &dyh2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 21, sizeof(cl_mem), &dzi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 22, sizeof(cl_mem), &t2xxD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 23, sizeof(cl_mem), &t2yyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 24, sizeof(cl_mem), &t2zzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 25, sizeof(cl_mem), &qt2xxD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 26, sizeof(cl_mem), &qt2yyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 27, sizeof(cl_mem), &qt2zzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 28, sizeof(cl_mem), &t2xx_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 29, sizeof(cl_mem), &t2zz_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 30, sizeof(cl_mem), &qt2xx_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 31, sizeof(cl_mem), &qt2zz_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 32, sizeof(cl_mem), &v2xD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 33, sizeof(cl_mem), &v2yD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_norm_PmlZ_IIC, 34, sizeof(cl_mem), &v2zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_norm_PmlZ_IIC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid16[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid16[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid16[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_norm_PmlZ_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_norm_PmlZ_IIC for execution!\n");
    }

	if (lbx[1] >= lbx[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX17 = (nd2_txy[17] - nd2_txy[12])/blockSizeX + 1;
		int gridSizeY17 = (nd2_txy[5] - nd2_txy[0])/blockSizeY + 1;
#else
		int gridSizeX17 = (nd2_txy[5] - nd2_txy[0])/blockSizeX + 1;
		int gridSizeY17 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid17[3] = {gridSizeX17, gridSizeY17, 1};
		errNum = clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 2, sizeof(int), mw2_pml);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 3, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 4, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 5, sizeof(int), nybtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 6, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 7, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 8, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 9, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 10, sizeof(cl_mem), &nd2_txyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 11, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 12, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 13, sizeof(cl_mem), &drth2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 14, sizeof(cl_mem), &damp2_xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 15, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 16, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 17, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 18, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 19, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 20, sizeof(cl_mem), &dxi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 21, sizeof(cl_mem), &dyi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 22, sizeof(cl_mem), &t2xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 23, sizeof(cl_mem), &qt2xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 24, sizeof(cl_mem), &t2xy_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 25, sizeof(cl_mem), &qt2xy_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 26, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlX_IIC, 27, sizeof(cl_mem), &v2yD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xy_PmlX_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid17[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid17[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid17[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_xy_PmlX_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xy_PmlX_IIC for execution!\n");
        }
	}

	if (lby[1] >= lby[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX18 = (nd2_txy[17] - nd2_txy[12])/blockSizeX + 1;
		int gridSizeY18 = (nd2_txy[11] - nd2_txy[6])/blockSizeY + 1;
#else
		int gridSizeX18 = (nd2_txy[11] - nd2_txy[6])/blockSizeX + 1;
		int gridSizeY18 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
		size_t dimGrid18[3] = {gridSizeX18, gridSizeY18, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 2, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 3, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 4, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 5, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 6, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 7, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 8, sizeof(cl_mem), &nd2_txyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 9, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 10, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 11, sizeof(cl_mem), &drth2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 12, sizeof(cl_mem), &damp2_yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 13, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 14, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 15, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 16, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 17, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 18, sizeof(cl_mem), &dxi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 19, sizeof(cl_mem), &dyi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 20, sizeof(cl_mem), &t2xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 21, sizeof(cl_mem), &qt2xyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 22, sizeof(cl_mem), &t2xy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 23, sizeof(cl_mem), &qt2xy_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 24, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlY_IIC, 25, sizeof(cl_mem), &v2yD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xy_PmlY_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid18[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid18[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid18[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_xy_PmlY_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xy_PmlY_IIC for execution!\n");
        }
	}

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX19 = (nd2_txy[17] - nd2_txy[16])/blockSizeX + 1;
	int gridSizeY19 = (nd2_txy[9] - nd2_txy[8])/blockSizeY + 1;
#else
	int gridSizeX19 = (nd2_txy[3] - nd2_txy[2])/blockSizeX + 1;
	int gridSizeY19 = (nd2_txy[9] - nd2_txy[8])/blockSizeY + 1;
#endif
	size_t dimGrid19[3] = {gridSizeX19, gridSizeY19, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 0, sizeof(int), nxb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 1, sizeof(int), nyb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 2, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 3, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 4, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 5, sizeof(cl_mem), &nd2_txyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 6, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 7, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 8, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 9, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 10, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 11, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 12, sizeof(cl_mem), &dxi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 13, sizeof(cl_mem), &dyi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 14, sizeof(cl_mem), &t2xyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 15, sizeof(cl_mem), &qt2xyD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 16, sizeof(cl_mem), &v2xD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xy_PmlZ_II, 17, sizeof(cl_mem), &v2yD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xy_PmlZ_II arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid19[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid19[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid19[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_xy_PmlZ_II, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xy_PmlZ_II for execution!\n");
    }

    if (lbx[1] >= lbx[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX20 = (nd2_txz[17] - nd2_txz[12])/blockSizeX + 1;
		int gridSizeY20 = (nd2_txz[5] - nd2_txz[0])/blockSizeY + 1;
#else
		int gridSizeX20 = (nd2_txz[5] - nd2_txz[0])/blockSizeX + 1;
		int gridSizeY20 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid20[3] = {gridSizeX20, gridSizeY20, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 2, sizeof(int), mw2_pml);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 3, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 4, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 5, sizeof(int), nybtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 6, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 7, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 8, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 9, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 10, sizeof(cl_mem), &nd2_txzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 11, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 12, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 13, sizeof(cl_mem), &drth2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 14, sizeof(cl_mem), &damp2_xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 15, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 16, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 17, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 18, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 19, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 20, sizeof(cl_mem), &dxi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 21, sizeof(cl_mem), &dzh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 22, sizeof(cl_mem), &t2xzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 23, sizeof(cl_mem), &qt2xzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 24, sizeof(cl_mem), &t2xz_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 25, sizeof(cl_mem), &qt2xz_pxD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 26, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlX_IIC, 27, sizeof(cl_mem), &v2zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xz_PmlX_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid20[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid20[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid20[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_xz_PmlX_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xz_PmlX_IIC for execution!\n");
        }
	}

	if (lby[1] >= lby[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX21 = (nd2_txz[15] - nd2_txz[12])/blockSizeX + 1;
		int gridSizeY21 = (nd2_txz[9] - nd2_txz[8])/blockSizeY + 1;
#else
		int gridSizeX21 = (nd2_txz[9] - nd2_txz[8])/blockSizeX + 1;
		int gridSizeY21 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
		size_t dimGrid21[3] = {gridSizeX21, gridSizeY21, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 2, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 3, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 5, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 6, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 7, sizeof(cl_mem), &nd2_txzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 8, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 9, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 10, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 11, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 12, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 13, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 14, sizeof(cl_mem), &dxi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 15, sizeof(cl_mem), &dzh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 16, sizeof(cl_mem), &v2xD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 17, sizeof(cl_mem), &v2zD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 18, sizeof(cl_mem), &t2xzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlY_IIC, 19, sizeof(cl_mem), &qt2xzD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xz_PmlY_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid21[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid21[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid21[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_xz_PmlY_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xz_PmlY_IIC for execution!\n");
        }
    }

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX22 = (nd2_txz[17] - nd2_txz[16])/blockSizeX + 1;
	int gridSizeY22 = (nd2_txz[11] - nd2_txz[6])/blockSizeY + 1;
#else
    int gridSizeX22 = (nd2_txz[5] - nd2_txz[0])/blockSizeX + 1;
	int gridSizeY22 = (nd2_txz[11] - nd2_txz[6])/blockSizeY + 1;
#endif
	size_t dimGrid22[3] = {gridSizeX22, gridSizeY22, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 0, sizeof(int), nxb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 1, sizeof(int), nyb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 2, sizeof(int), mw2_pml1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 3, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 4, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 5, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 6, sizeof(cl_mem), &nd2_txzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 7, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 8, sizeof(float), ca);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 9, sizeof(cl_mem), &drti2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 10, sizeof(cl_mem), &damp2_zD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 11, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 12, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 13, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 14, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 15, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 16, sizeof(cl_mem), &dxi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 17, sizeof(cl_mem), &dzh2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 18, sizeof(cl_mem), &t2xzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 19, sizeof(cl_mem), &qt2xzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 20, sizeof(cl_mem), &t2xz_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 21, sizeof(cl_mem), &qt2xz_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 22, sizeof(cl_mem), &v2xD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_xz_PmlZ_IIC, 23, sizeof(cl_mem), &v2zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_xz_PmlZ_IIC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid22[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid22[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid22[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_xz_PmlZ_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_xz_PmlZ_IIC for execution!\n");
    }

	if (lbx[1] >= lbx[0])
	{
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX23 = (nd2_tyz[15] - nd2_tyz[12])/blockSizeX + 1;
		int gridSizeY23 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeY + 1;
#else
		int gridSizeX23 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeX + 1;
		int gridSizeY23 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
		size_t dimGrid23[3] = {gridSizeX23, gridSizeY23, 1};
		errNum = clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 2, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 3, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 4, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 5, sizeof(int), &lbx[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 6, sizeof(int), &lbx[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 7, sizeof(cl_mem), &nd2_tyzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 8, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 9, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 10, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 11, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 12, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 13, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 14, sizeof(cl_mem), &dyi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 15, sizeof(cl_mem), &dzh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 16, sizeof(cl_mem), &t2yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 17, sizeof(cl_mem), &qt2yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 18, sizeof(cl_mem), &v2yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlX_IIC, 19, sizeof(cl_mem), &v2zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_yz_PmlX_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid23[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid23[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid23[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_yz_PmlX_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_yz_PmlX_IIC for execution!\n");
        }
    }

    if (lby[1] >= lby[0])
    {
#ifdef DISFD_USE_OPTIMIZED
		int gridSizeX24 = (nd2_tyz[17] - nd2_tyz[12])/blockSizeX + 1;
		int gridSizeY24 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeY + 1;
#else
        int gridSizeX24 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeX + 1;
        int gridSizeY24 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
        size_t dimGrid24[3] = {gridSizeX24, gridSizeY24, 1};

		errNum = clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 0, sizeof(int), nxb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 1, sizeof(int), nyb2);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 2, sizeof(int), mw2_pml1);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 3, sizeof(int), nxbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 4, sizeof(int), nzbtm);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 5, sizeof(int), nztop);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 6, sizeof(int), &lby[0]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 7, sizeof(int), &lby[1]);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 8, sizeof(cl_mem), &nd2_tyzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 9, sizeof(cl_mem), &idmat2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 10, sizeof(float), ca);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 11, sizeof(cl_mem), &drth2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 12, sizeof(cl_mem), &damp2_yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 13, sizeof(cl_mem), &cmuD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 14, sizeof(cl_mem), &epdtD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 15, sizeof(cl_mem), &qwsD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 16, sizeof(cl_mem), &qwt1D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 17, sizeof(cl_mem), &qwt2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 18, sizeof(cl_mem), &dyi2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 19, sizeof(cl_mem), &dzh2D);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 20, sizeof(cl_mem), &t2yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 21, sizeof(cl_mem), &qt2yzD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 22, sizeof(cl_mem), &t2yz_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 23, sizeof(cl_mem), &qt2yz_pyD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 24, sizeof(cl_mem), &v2yD);
		errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlY_IIC, 25, sizeof(cl_mem), &v2zD);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_yz_PmlY_IIC arguments!\n");
        }
        localWorkSize[0] = dimBlock[0];
        localWorkSize[1] = dimBlock[1];
        localWorkSize[2] = dimBlock[2];
        globalWorkSize[0] = dimGrid24[0]*localWorkSize[0];
        globalWorkSize[1] = dimGrid24[1]*localWorkSize[1];
        globalWorkSize[2] = dimGrid24[2]*localWorkSize[2];
        errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_yz_PmlY_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
        if(errNum != CL_SUCCESS)
        {
            fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_yz_PmlY_IIC for execution!\n");
        }
	}

#ifdef DISFD_USE_OPTIMIZED
	int gridSizeX25 = (nd2_tyz[17] - nd2_tyz[16])/blockSizeX + 1;
	int gridSizeY25 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeY + 1;
#else
	int gridSizeX25 = (nd2_tyz[5] - nd2_tyz[0])/blockSizeX + 1;
	int gridSizeY25 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeY + 1;
#endif
	size_t dimGrid25[3] = {gridSizeX25, gridSizeY25, 1};

	errNum = clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 0, sizeof(int), nxb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 1, sizeof(int), nyb2);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 2, sizeof(int), mw2_pml1);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 3, sizeof(int), nxbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 4, sizeof(int), nzbtm);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 5, sizeof(int), nztop);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 6, sizeof(cl_mem), &nd2_tyzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 7, sizeof(cl_mem), &idmat2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 8, sizeof(float), ca);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 9, sizeof(cl_mem), &drti2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 10, sizeof(cl_mem), &damp2_zD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 11, sizeof(cl_mem), &cmuD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 12, sizeof(cl_mem), &epdtD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 13, sizeof(cl_mem), &qwsD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 14, sizeof(cl_mem), &qwt1D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 15, sizeof(cl_mem), &qwt2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 16, sizeof(cl_mem), &dyi2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 17, sizeof(cl_mem), &dzh2D);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 18, sizeof(cl_mem), &t2yzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 19, sizeof(cl_mem), &qt2yzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 20, sizeof(cl_mem), &t2yz_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 21, sizeof(cl_mem), &qt2yz_pzD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 22, sizeof(cl_mem), &v2yD);
	errNum |= clSetKernelArg(_cl_Kernel_stress_yz_PmlZ_IIC, 23, sizeof(cl_mem), &v2zD);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: setting kernel _cl_Kernel_stress_yz_PmlZ_IIC arguments!\n");
    }
    localWorkSize[0] = dimBlock[0];
    localWorkSize[1] = dimBlock[1];
    localWorkSize[2] = dimBlock[2];
    globalWorkSize[0] = dimGrid25[0]*localWorkSize[0];
    globalWorkSize[1] = dimGrid25[1]*localWorkSize[1];
    globalWorkSize[2] = dimGrid25[2]*localWorkSize[2];
    errNum = clEnqueueNDRangeKernel(_cl_commandQueues[1], _cl_Kernel_stress_yz_PmlZ_IIC, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: queuing kernel _cl_Kernel_stress_yz_PmlZ_IIC for execution!\n");
    }

    for(i = 0; i < NUM_COMMAND_QUEUES; i++) {
    errNum = clFinish(_cl_commandQueues[i]);
    if(errNum != CL_SUCCESS)
    {
        fprintf(stderr, "Error: finishing stress kernel for execution!\n");
    }
	Stop(&kernelTimerStress[i]);
	}
    
    // printf("[OpenCL] computating finished!\n");

    gettimeofday(&t2, NULL);
    tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    totalTimeCompS += tmpTime;

    gettimeofday(&t1, NULL);
	Start(&d2hTimerStress);
    cpy_d2h_stressOutputsC_opencl(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	Stop(&d2hTimerStress);
    gettimeofday(&t2, NULL);

    tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
    totalTimeD2HS += tmpTime;

#ifdef DISFD_DEBUG
    int size = (*nztop) * (*nxtop + 3) * (*nytop);
    write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM.txt");
    size = (*nztop) * (*nxtop + 3) * (*nytop + 3);
    write_output(t1xyM, size, "OUTPUT_ARRAYS/t1xyM.txt");
    size = (*nztop + 1) * (*nxtop + 3) * (*nytop);
    write_output(t1xzM, size, "OUTPUT_ARRAYS/t1xzM.txt");
    size = (*nztop) * (*nxtop) * (*nytop + 3);
    write_output(t1yyM, size, "OUTPUT_ARRAYS/t1yyM.txt");
    size = (*nztop + 1) * (*nxtop) * (*nytop + 3);
    write_output(t1yzM, size, "OUTPUT_ARRAYS/t1yzM.txt");
    size = (*nztop) * (*nxtop) * (*nytop);
    write_output(t1zzM, size, "OUTPUT_ARRAYS/t1zzM.txt");
    size = (*nzbtm) * (*nxbtm + 3) * (*nybtm);
    write_output(t2xxM, size, "OUTPUT_ARRAYS/t2xxM.txt");
    size = (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3);
    write_output(t2xyM, size, "OUTPUT_ARRAYS/t2xyM.txt");
    size = (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm);
    write_output(t2xzM, size, "OUTPUT_ARRAYS/t2xzM.txt");
    size = (*nzbtm) * (*nxbtm) * (*nybtm + 3);
    write_output(t2yyM, size, "OUTPUT_ARRAYS/t2yyM.txt");
    size = (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3);
    write_output(t2yzM, size, "OUTPUT_ARRAYS/t2yzM.txt");
    size = (*nzbtm + 1) * (*nxbtm) * (*nybtm);
    write_output(t2zzM, size, "OUTPUT_ARRAYS/t2zzM.txt");
#endif
    /************** correctness *********************/
/*   
    FILE *fp;
    int i;
    const char *filename = "v1x.txt";
    const char *filename1 = "v1y.txt";
    const char *filename2 = "v1z.txt";
    if((fp = fopen(filename, "w+")) == NULL)
    {
        fprintf(stderr, "File Write Error!\n");
    }
    for(i =0; i<(*nztop+2)*(*nxtop+3)*(*nytop+3); i++)
    {
        fprintf(fp,"%f ", v1xM[i]);

    }
    fprintf(fp, "\n");
    fclose(fp);
    if((fp = fopen(filename1, "w+")) == NULL)
    {
        fprintf(stderr, "File Write Error!\n");
    }
    for(i =0; i<(*nztop+2)*(*nxtop+3)*(*nytop+3); i++)
    {
        fprintf(fp,"%f ", v1yM[i]);

    }
    fprintf(fp, "\n");
    fclose(fp);
    if((fp = fopen(filename2, "w+")) == NULL)
    {
        fprintf(stderr, "File Write Error!\n");
    }
    for(i =0; i<(*nztop+2)*(*nxtop+3)*(*nytop+3); i++)
    {
        fprintf(fp,"%f ", v1zM[i]);

    }
    fprintf(fp, "\n");
    fclose(fp);

    const char *filename3 = "x_t1xx.txt";
    const char *filename4 = "x_t1xy.txt";
    const char *filename5 = "x_t1xz.txt";

    if((fp = fopen(filename3, "w+")) == NULL)
        fprintf(stderr, "File write error!\n");
  
    for(i = 0; i< (*nztop) * (*nxtop + 3) * (*nytop); i++ )
    {
        fprintf(fp, "%f ", t1xxM[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    if((fp = fopen(filename4, "w+")) == NULL)
        fprintf(stderr, "File write error!\n");

    for(i = 0; i< (*nztop) * (*nxtop + 3) * (*nytop+3); i++ )
    {
        fprintf(fp, "%f ", t1xyM[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    if((fp = fopen(filename5, "w+")) == NULL)
        fprintf(stderr, "File write error!\n");
    for(i = 0; i< (*nztop+1) * (*nxtop + 3) * (*nytop); i++ )
    {
        fprintf(fp, "%f ", t1xzM[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
*/
    return;
}

