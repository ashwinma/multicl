#ifndef __MPIACC_CLH__
#define __MPIACC_CLH__

#include <CL/cl.h>
#ifdef __cplusplus
extern "C" {
#endif
#define NUM_COMMAND_QUEUES	2
#define MARSHAL_CMDQ	0
extern cl_platform_id     _cl_firstPlatform;
extern cl_device_id       _cl_firstDevice;
extern cl_device_id       *_cl_devices;
extern cl_context         _cl_context;
//extern cl_command_queue   _cl_commandQueue;
extern cl_command_queue   _cl_commandQueues[NUM_COMMAND_QUEUES];
extern cl_program         _cl_program;
extern cl_event 		  _cl_events[NUM_COMMAND_QUEUES];

#ifdef __cplusplus
}
#endif
#endif // __MPIACC_CLH__
