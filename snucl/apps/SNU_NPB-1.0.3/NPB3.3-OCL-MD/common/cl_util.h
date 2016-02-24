#ifndef CL_UTIL_H
#define CL_UTIL_H

#include <stdarg.h>
#include <CL/cl.h>

/****************************************************************************/
/* OpenCL Utility Functions                                                 */
/****************************************************************************/

/* Error Checking */
// Exit the host program with a message.
void clu_Exit(const char *format, ...);

// If err_code is not CU_SUCCESS, exit the host program with msg.
void clu_CheckErrorInternal(cl_int err_code, 
                            const char *msg,
                            const char *file,
                            int line);
#define clu_CheckError(e,m)  clu_CheckErrorInternal(e,m,__FILE__,__LINE__)
//#define clu_CheckError(e,m)

/* OpenCL Device */
// Find the device type from the environment variable OPENCL_DEVICE_TYPE. 
// - If the value of OPENCL_DEVICE_TYPE is "cpu", "gpu", or "accelerator",
//   return CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, or CL_DEVICE_ACCELERATOR,
//   respectively.
// - If it is not set or invalid, return CL_DEVICE_TYPE_DEFAULT.
cl_device_type clu_GetDefaultDeviceType();

// Return an available cl_device_id corresponding to device_type.
//cl_device_id clu_GetAvailableDevice(cl_device_type device_type);
cl_device_id *clu_GetAvailableDevices(cl_device_type device_type, cl_uint *device_count);

// Return the name of device. E.g., Geforce_GTX_480.
char *clu_GetDeviceName(cl_device_id device);

// Return the string of device_type. E.g., CL_DEVICE_TYPE_CPU.
const char *clu_GetDeviceTypeName(cl_device_type device_type);


/* Program Build */
// Load the source code from source_file and return the pointer of the source 
// code string. Length of source code is saved through source_len_ret pointer.
char *clu_LoadProgSource(const char *source_file, size_t *source_len_ret);

// Load the OpenCL program binary from binary_file and return the pointer of
// loaded binary. Size of binary is saved through binary_len_ret pointer.
unsigned char *clu_LoadProgBinary(const char *binary_file, 
                                  size_t *binary_len_ret);

#if 0
// Create a program and build the program.
cl_program clu_MakeProgram(cl_context context,
						   //cl_device_id device,
                           const cl_uint num_devices,
                           cl_device_id *devices,
                           char *source_dir,
                           char *source_file, 
                           char *build_option);

#endif
cl_program clu_CreateProgram(cl_context context,
                           char *source_dir,
                           char *source_file 
                           );
void clu_MakeProgram(cl_program program,
                           const cl_uint num_devices,
                           cl_device_id *devices,
                           char *source_dir,
                           char *build_option);
/* Misc */
// Return the size that is rounded up to the multiple of group_size.
size_t clu_RoundWorkSize(size_t work_size, size_t group_size);


#endif //CL_UTIL_H
