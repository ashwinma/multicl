#ifndef __SAMPLE_H__
#define __SAMPLE_H__

#ifdef __APPLE__
# include <OpenCL/opencl.h>
#else
# include <CL/opencl.h>
#endif

#define NUM_COMMAND_QUEUES	2
cl_kernel _compute_kernel;
cl_kernel _marshal_kernel;
cl_context _context; 
cl_program _program;
cl_mem _bufs[NUM_COMMAND_QUEUES];
cl_platform_id     _cl_firstPlatform;
cl_device_id       _cl_firstDevice;
cl_device_id       *_cl_devices;
cl_context         _cl_context;
//cl_command_queue   _cl_commandQueue;
cl_command_queue _cl_commandQueues[NUM_COMMAND_QUEUES];
cl_event _cl_events[NUM_COMMAND_QUEUES];
cl_program         _cl_program;

size_t globalWorkSize[3];
size_t localWorkSize[3];

#endif //__SAMPLE_H__
