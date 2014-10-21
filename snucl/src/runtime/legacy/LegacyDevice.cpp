/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2011-2013 Seoul National University.                        */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE          */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   Center for Manycore Programming                                         */
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 151-744, Korea                         */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jungwon Kim, Sangmin Seo, Gangwon Jo, Jun Lee, Jeongho Nah,             */
/*   Jungho Park, Junghyun Kim, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

#include "legacy/LegacyDevice.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <CL/cl.h>
#include "CLCommand.h"
#include "CLDevice.h"
#include "CLKernel.h"
#include "CLMem.h"
#include "CLPlatform.h"
#include "CLProgram.h"
#include "CLSampler.h"
#include "ICD.h"
#include "Utils.h"
using namespace std;


#define LEGACY_VERSION_1_0 10
#define LEGACY_VERSION_1_1 11
#define LEGACY_VERSION_1_2 12

#ifdef LEGACY_PLATFORM
#define ICD_VENDOR_PATH "/home/aaji/etc/OpenCL/vendors"
#else
#define ICD_VENDOR_PATH "/etc/OpenCL/vendors"
#endif
#define ICD_EXTENSION ".icd"
#define ICD_SNUCL_CLUSTER_NAME "snucl_cluster.icd"
#define ICD_SNUCL_SINGLE_NAME "snucl_single.icd"

	/* Extension function access
	 *
	 * Returns the extension function address for the given function name,
	 * or NULL if a valid function can not be found.  The client must
	 * check to make sure the address is not NULL, before using or 
	 * calling the returned function address.
	 */
typedef CL_API_ENTRY void * (CL_API_CALL *pfn_clGetExtensionFunctionAddress)(const char * func_name) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *pfn_clIcdGetPlatformIDs)(
    cl_uint num_entries, 
    cl_platform_id *platforms, 
    cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0;

#define CHECK_ERROR(cond, err)                    \
  if (cond) {                                     \
    if(command) command->SetError(err);                       \
    SNUCL_ERROR("legacy vendor error %d\n", err); \
    return;                                       \
  }

#define UPDATE_ERROR(err)                         \
  if (err != CL_SUCCESS) {                        \
    if(command)command->SetError(err);            \
    SNUCL_ERROR("legacy vendor error with device %p : %d\n", device_id_, err); \
    return;                                       \
  }

#define VERIFY_ERROR(err)                         \
  if (err != CL_SUCCESS) {                        \
    SNUCL_ERROR("legacy vendor error %d\n", err); \
  }

LegacyDevice::LegacyDevice(void* library, struct _cl_icd_dispatch* dispatch, 
						   cl_context context,
                           cl_platform_id platform_id, cl_device_id device_id)
    : CLDevice(0) {
  gLegacyTimer.Init();
  library_ = library;
  dispatch_ = dispatch;
  platform_id_ = platform_id;
  device_id_ = device_id;

  context_ = context;
  //context_ = NULL;
  kernel_queue_ = NULL;
  mem_queue_ = NULL;
  misc_queue_ = NULL;

  cl_int err;

#define GET_LEGACY_DEVICE_INFO(param, type, value, def)             \
  err = dispatch_->clGetDeviceInfo(device_id_, param, sizeof(type), \
                                   &value, NULL);                   \
  if (err != CL_SUCCESS)                                            \
    value = def;

#define GET_LEGACY_DEVICE_INFO_A(param, type, value, length, def)            \
  err = dispatch_->clGetDeviceInfo(device_id_, param, sizeof(type) * length, \
                                   value, NULL);                             \
  if (err != CL_SUCCESS) {                                                   \
    for (int i = 0; i < length; i++)                                         \
      value[i] = def;                                                        \
  }

#define GET_LEGACY_DEVICE_INFO_S(param, value)                           \
  {                                                                      \
    size_t size;                                                         \
    err = dispatch_->clGetDeviceInfo(device_id_, param, 0, NULL, &size); \
    if (err != CL_SUCCESS || size == 0) {                                \
      value = NULL;                                                      \
    } else {                                                             \
      char* buffer = (char*)malloc(size);                                \
      err = dispatch_->clGetDeviceInfo(device_id_, param, size, buffer,  \
                                       NULL);                            \
      if (err != CL_SUCCESS) {                                           \
        free(buffer);                                                    \
        value = NULL;                                                    \
      } else {                                                           \
        value = buffer;                                                  \
      }                                                                  \
    }                                                                    \
  }

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_AVAILABLE, cl_bool, available_, CL_FALSE);
  GET_LEGACY_DEVICE_INFO_S(CL_DEVICE_PROFILE, profile_);
  GET_LEGACY_DEVICE_INFO_S(CL_DEVICE_VERSION, device_version_);
  if (profile_ == NULL || strcmp(profile_, "FULL_PROFILE") != 0)
    available_ = CL_FALSE;
  if (device_version_ != NULL) {
    if (strncmp(device_version_, "OpenCL 1.2", 10) == 0) {
      version_ = LEGACY_VERSION_1_2;
    } else if (strncmp(device_version_, "OpenCL 1.1", 10) == 0) {
      version_ = LEGACY_VERSION_1_1;
    } else if (strncmp(device_version_, "OpenCL 1.0", 10) == 0) {
      version_ = LEGACY_VERSION_1_0;
    } else {
      available_ = CL_FALSE;
    }
  } else {
    available_ = CL_FALSE;
  }

  /* TODO: Get the values for mem_bw_, lmem_bw_ and compute_throughput_
   * via microbenchmarks for this legacy device
   */
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_TYPE, cl_device_type, type_,
                         CL_DEVICE_TYPE_DEFAULT);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_VENDOR_ID, cl_uint, vendor_id_, 201111);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_COMPUTE_UNITS, cl_uint,
                         max_compute_units_, 1);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, cl_uint,
                         max_work_item_dimensions_, 3);
  GET_LEGACY_DEVICE_INFO_A(CL_DEVICE_MAX_WORK_ITEM_SIZES, size_t,
                           max_work_item_sizes_, max_work_item_dimensions_, 1);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_WORK_GROUP_SIZE, size_t,
                         max_work_group_size_, 1);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, cl_uint,
                         preferred_vector_width_char_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, cl_uint,
                         preferred_vector_width_short_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, cl_uint,
                         preferred_vector_width_int_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, cl_uint,
                         preferred_vector_width_long_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, cl_uint,
                         preferred_vector_width_float_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, cl_uint,
                         preferred_vector_width_double_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, cl_uint,
                         preferred_vector_width_half_, 0);
  if (version_ >= LEGACY_VERSION_1_1) {
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, cl_uint,
                           native_vector_width_char_, 0);
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, cl_uint,
                           native_vector_width_short_, 0);
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, cl_uint,
                           native_vector_width_int_, 0);
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, cl_uint,
                           native_vector_width_long_, 0);
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, cl_uint,
                           native_vector_width_float_, 0);
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, cl_uint,
                           native_vector_width_double_, 0);
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, cl_uint,
                           native_vector_width_half_, 0);
  } else {
    native_vector_width_char_ = 0;
    native_vector_width_short_ = 0;
    native_vector_width_int_ = 0;
    native_vector_width_long_ = 0;
    native_vector_width_float_ = 0;
    native_vector_width_double_ = 0;
    native_vector_width_half_ = 0;
  }
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_CLOCK_FREQUENCY, cl_uint,
                         max_clock_frequency_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_ADDRESS_BITS, cl_uint, address_bits_,
                         sizeof(size_t));

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_MEM_ALLOC_SIZE, cl_ulong,
                         max_mem_alloc_size_, 0);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE_SUPPORT, cl_bool, image_support_,
                         CL_FALSE);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_READ_IMAGE_ARGS, cl_uint,
                         max_read_image_args_, 8);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, cl_uint,
                         max_write_image_args_, 1);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE2D_MAX_WIDTH, size_t,
                         image2d_max_width_, 2048);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE2D_MAX_HEIGHT, size_t,
                         image2d_max_height_, 2048);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE3D_MAX_WIDTH, size_t,
                         image3d_max_width_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE3D_MAX_HEIGHT, size_t,
                         image3d_max_height_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE3D_MAX_DEPTH, size_t,
                         image3d_max_depth_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE, size_t,
                         image_max_buffer_size_, 2048);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_IMAGE_MAX_ARRAY_SIZE, size_t,
                         image_max_array_size_, 256);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_SAMPLERS, cl_uint, max_samplers_, 8);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_PARAMETER_SIZE, size_t,
                         max_parameter_size_, 256);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MEM_BASE_ADDR_ALIGN, cl_uint,
                         mem_base_addr_align_, 1024);
  if (version_ <= LEGACY_VERSION_1_1) {
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, cl_uint,
                           min_data_type_align_size_, 0);
  } else {
    min_data_type_align_size_ = 0;
  }

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config,
                         single_fp_config_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_SINGLE_FP_CONFIG, cl_device_fp_config,
                         double_fp_config_, 0);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                         cl_device_mem_cache_type, global_mem_cache_type_,
                         CL_NONE);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint,
                         global_mem_cacheline_size_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong,
                         global_mem_cache_size_, 0);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong,
                         global_mem_size_, 0);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, cl_ulong,
                         max_constant_buffer_size_, 1024);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_MAX_CONSTANT_ARGS, cl_uint,
                         max_constant_args_, 4);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_LOCAL_MEM_TYPE, cl_device_local_mem_type,
                         local_mem_type_, CL_GLOBAL);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_LOCAL_MEM_SIZE, cl_ulong, local_mem_size_,
                         1024);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_ERROR_CORRECTION_SUPPORT, cl_bool,
                         error_correction_support_, CL_FALSE);

  if (version_ >= LEGACY_VERSION_1_1) {
    GET_LEGACY_DEVICE_INFO(CL_DEVICE_HOST_UNIFIED_MEMORY, cl_bool,
                           host_unified_memory_, CL_FALSE);
  } else {
    host_unified_memory_ = CL_FALSE;
  }

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PROFILING_TIMER_RESOLUTION, size_t,
                         profiling_timer_resolution_, 1);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_ENDIAN_LITTLE, cl_bool, endian_little_,
                         CL_TRUE);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_COMPILER_AVAILABLE, cl_bool,
                         compiler_available_, CL_TRUE);
  GET_LEGACY_DEVICE_INFO(CL_DEVICE_LINKER_AVAILABLE, cl_bool,
                         linker_available_, CL_TRUE);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_EXECUTION_CAPABILITIES,
                         cl_device_exec_capabilities, execution_capabilities_,
                         CL_EXEC_KERNEL);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_QUEUE_PROPERTIES,
                         cl_command_queue_properties, queue_properties_, 0);

  built_in_kernels_ = "";

  GET_LEGACY_DEVICE_INFO_S(CL_DEVICE_NAME, name_);
  GET_LEGACY_DEVICE_INFO_S(CL_DEVICE_VENDOR, vendor_);
  GET_LEGACY_DEVICE_INFO_S(CL_DRIVER_VERSION, driver_version_);
  if (version_ >= LEGACY_VERSION_1_1) {
    GET_LEGACY_DEVICE_INFO_S(CL_DEVICE_OPENCL_C_VERSION, opencl_c_version_);
  } else {
    opencl_c_version_ = NULL;
  }
  GET_LEGACY_DEVICE_INFO_S(CL_DEVICE_EXTENSIONS, device_extensions_);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PRINTF_BUFFER_SIZE, size_t,
                         printf_buffer_size_, 1024);

  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PREFERRED_INTEROP_USER_SYNC, cl_bool,
                         preferred_interop_user_sync_, CL_FALSE);
/*
  if (version_ >= LEGACY_VERSION_1_2) {
	  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PARTITION_MAX_SUB_DEVICES, cl_uint,
			  partition_max_sub_devices_, 0);
	  err = dispatch_->clGetDeviceInfo(device_id_, CL_DEVICE_PARTITION_PROPERTIES,
			  0, NULL, &num_partition_properties_);
	  if (err != CL_SUCCESS)
		  num_partition_properties_ = 0;
	  num_partition_properties_ /= sizeof(cl_device_partition_properties);
	  GET_LEGACY_DEVICE_INFO_A(CL_DEVICE_PARTITION_PROPERTIES,
			  cl_device_partition_properties,
			  partition_properties_, num_partition_properties_,
			  0);
	  GET_LEGACY_DEVICE_INFO(CL_DEVICE_PARTITION_AFFINITY_DOMAIN,
			  cl_device_partition_affinity_domain,
			  affinity_domain_, 0);
  }
  else */
  {
	  partition_max_sub_devices_ = 1;
	  partition_max_compute_units_ = max_compute_units_;
	  num_partition_properties_ = 0;
	  affinity_domain_ = 0;
	  partition_type_ = NULL;
	  partition_type_len_ = 0;
  }

#undef GET_LEGACY_DEVICE_INFO
#undef GET_LEGACY_DEVICE_INFO_A
#undef GET_LEGACY_DEVICE_INFO_S

#if 0
  cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
                                         (cl_context_properties)platform_id_,
                                         0};
  context_ = dispatch_->clCreateContext(properties, 1, &device_id_, NULL, NULL,
                                        &err);
  if (err != CL_SUCCESS) {
    available_ = false;
  } else {
#else
  if(context_ != NULL) {  
#endif
	kernel_queue_ = dispatch_->clCreateCommandQueue(context_, device_id_, CL_QUEUE_PROFILING_ENABLE,
                                                    &err);
    if (err != CL_SUCCESS)
      available_ = false;
    mem_queue_ = dispatch_->clCreateCommandQueue(context_, device_id_, 0,
                                                 &err);
    if (err != CL_SUCCESS)
      available_ = false;
    misc_queue_ = dispatch_->clCreateCommandQueue(context_, device_id_, 0,
                                                  &err);
    if (err != CL_SUCCESS)
      available_ = false;
  }
  std::cout << "Done creating legacy device" << std::endl;
}

LegacyDevice::~LegacyDevice() {
  if(gLegacyTimer.Count() > 0)
	  std::cout << "[Device " << name_ << "] Dispatch timer: " << gLegacyTimer << std::endl;
  if (kernel_queue_)
    dispatch_->clReleaseCommandQueue(kernel_queue_);
  if (mem_queue_)
    dispatch_->clReleaseCommandQueue(mem_queue_);
  if (misc_queue_)
    dispatch_->clReleaseCommandQueue(misc_queue_);
  if (context_)
    dispatch_->clReleaseContext(context_);
  if (name_)
    free((char*)name_);
  if (vendor_)
    free((char*)vendor_);
  if (driver_version_)
    free((char*)driver_version_);
  if (profile_)
    free((char*)profile_);
  if (device_version_)
    free((char*)device_version_);
  if (opencl_c_version_)
    free((char*)opencl_c_version_);
  if (device_extensions_)
    free((char*)device_extensions_);
}

double LegacyDevice::WaitForKernel(CLCommand *command) {
  //cl_int err = dispatch_->clFinish(kernel_queue_);
  cl_int err = dispatch_->clWaitForEvents(1, &kernel_event_);
  
  cl_ulong start = 0, end = 0;
  dispatch_->clGetEventProfilingInfo(kernel_event_, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
  dispatch_->clGetEventProfilingInfo(kernel_event_, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  //END-START gives you hints on kind of “pure HW execution time”
  ////the resolution of the events is 1e-09 sec
  double g_NDRangePureExecTimeMs = (double)(end - start)*(double)(1e-06); 
  SNUCL_INFO("Test Kernel Event Time: %g msec\n", g_NDRangePureExecTimeMs);
  if (err != CL_SUCCESS) {                        
    if(command)command->SetError(err);            
    SNUCL_ERROR("legacy vendor error with device %p : %d\n", device_id_, err);
  }
  return g_NDRangePureExecTimeMs/1000;
}

void LegacyDevice::LaunchTestKernel(CLCommand* command, CLKernel* kernel,
                                cl_uint work_dim, size_t gwo[3], size_t gws[3],
                                size_t lws[3], size_t nwg[3],
                                map<cl_uint, CLKernelArg*>* kernel_args, 
								bool useTrainingKernel) {
  //SNUCL_INFO("Test run kernel: %s on Device ID %p (type %d)\n", kernel->name(), device_id_, type_);
  //printf("Device Type: %d Ptr: %p\n", type_, device_id_);
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_kernel legacy_kernel = NULL;
  if(useTrainingKernel)
	  legacy_kernel = (cl_kernel)kernel->GetDevSpecificTraining(this);
  else
  	  legacy_kernel = (cl_kernel)kernel->GetDevSpecific(this);
  CHECK_ERROR(legacy_kernel == NULL, CL_INVALID_PROGRAM_EXECUTABLE);
  CLKernelLaunchParams legacy_kernel_launch_params = kernel->GetDevSpecificLaunchConfiguration(this);
  cl_int err;
  for (map<cl_uint, CLKernelArg*>::iterator it = kernel_args->begin();
       it != kernel_args->end();
       ++it) {
    //SNUCL_INFO("Kernel Idx: %u, param: %p\n", it->first, it->second);
    cl_uint index = it->first;
    CLMem* mem = it->second->mem;
    CLSampler* sampler = it->second->sampler;
    if (mem != NULL) {
      cl_mem mem_dev = (cl_mem)mem->GetDevSpecific(this);
	  //SNUCL_INFO("Test kernel Device: %p, Mem: %p, dev specific: %p, size: %llu\n", device_id_, mem, mem_dev, mem->size());
      CHECK_ERROR(mem_dev == NULL, CL_INVALID_MEM_OBJECT);
      err = dispatch_->clSetKernelArg(legacy_kernel, index, sizeof(cl_mem),
                                      &mem_dev);
    } else if (sampler != NULL) {
      cl_sampler sampler_dev = (cl_sampler)sampler->GetDevSpecific(this);
      CHECK_ERROR(sampler_dev == NULL, CL_INVALID_SAMPLER);
      err = dispatch_->clSetKernelArg(legacy_kernel, index, sizeof(cl_sampler),
                                      &sampler_dev);
    } else if (it->second->local) {
      err = dispatch_->clSetKernelArg(legacy_kernel, index, it->second->size,
                                      NULL);
    } else {
      err = dispatch_->clSetKernelArg(legacy_kernel, index, it->second->size,
                                      it->second->value);
    }
    UPDATE_ERROR(err);
  }
  //cl_event event;
  if(kernel->HasDevSpecificLaunchConfiguration(this)) {
	  work_dim = legacy_kernel_launch_params.work_dim_;
//	  SNUCL_INFO("Just before launching test kernel with work_dim:%u\n", work_dim);
	  for(int i = 0; i < 3; i++) {
		  gwo[i] = legacy_kernel_launch_params.gwo_[i];
		  gws[i] = legacy_kernel_launch_params.gws_[i];
		  lws[i] = legacy_kernel_launch_params.lws_[i];
		  nwg[i] = legacy_kernel_launch_params.nwg_[i];
//		  SNUCL_INFO("GWO[%d]: %lu\n", i, gwo[i]);
//		  SNUCL_INFO("GWS[%d]: %lu\n", i, gws[i]);
//		  SNUCL_INFO("LWS[%d]: %lu\n", i, lws[i]);
//		  SNUCL_INFO("NWG[%d]: %lu\n", i, nwg[i]);
	  }
  }
  gLegacyTimer.Start();
  err = dispatch_->clEnqueueNDRangeKernel(kernel_queue_, legacy_kernel,
                                          work_dim, gwo, gws, lws, 0, NULL,
                                          &kernel_event_);
  UPDATE_ERROR(err);
  //err = dispatch_->clWaitForEvents(1, &event);
  gLegacyTimer.Stop();
  SNUCL_INFO("Test Kernel Launch Time: %g sec\n", gLegacyTimer.CurrentElapsed());
  UPDATE_ERROR(err);
}

void LegacyDevice::LaunchKernel(CLCommand* command, CLKernel* kernel,
                                cl_uint work_dim, size_t gwo[3], size_t gws[3],
                                size_t lws[3], size_t nwg[3],
                                map<cl_uint, CLKernelArg*>* kernel_args) {
//  SNUCL_INFO("run kernel: %s Device Type: %d Ptr: %p\n", 
  //		kernel->name(), type_, device_id_);
  gLegacyTimer.Start();
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_kernel legacy_kernel = (cl_kernel)kernel->GetDevSpecific(this);
  CHECK_ERROR(legacy_kernel == NULL, CL_INVALID_PROGRAM_EXECUTABLE);
  cl_int err;
  CLKernelLaunchParams legacy_kernel_launch_params = kernel->GetDevSpecificLaunchConfiguration(this);
  for (map<cl_uint, CLKernelArg*>::iterator it = kernel_args->begin();
       it != kernel_args->end();
       ++it) {
    cl_uint index = it->first;
    CLMem* mem = it->second->mem;
    CLSampler* sampler = it->second->sampler;
    if (mem != NULL) {
      cl_mem mem_dev = (cl_mem)mem->GetDevSpecific(this);
      CHECK_ERROR(mem_dev == NULL, CL_INVALID_MEM_OBJECT);
      err = dispatch_->clSetKernelArg(legacy_kernel, index, sizeof(cl_mem),
                                      &mem_dev);
    } else if (sampler != NULL) {
      cl_sampler sampler_dev = (cl_sampler)sampler->GetDevSpecific(this);
      CHECK_ERROR(sampler_dev == NULL, CL_INVALID_SAMPLER);
      err = dispatch_->clSetKernelArg(legacy_kernel, index, sizeof(cl_sampler),
                                      &sampler_dev);
    } else if (it->second->local) {
      err = dispatch_->clSetKernelArg(legacy_kernel, index, it->second->size,
                                      NULL);
    } else {
      err = dispatch_->clSetKernelArg(legacy_kernel, index, it->second->size,
                                      it->second->value);
    }
    UPDATE_ERROR(err);
  }
  if(kernel->HasDevSpecificLaunchConfiguration(this)) {
	//  SNUCL_INFO("Just before copying kernel %s config with work_dim:%u\n", kernel->name(), work_dim);
	  work_dim = legacy_kernel_launch_params.work_dim_;
	//  SNUCL_INFO("Just before launching test kernel with work_dim:%u\n", work_dim);
	  for(int i = 0; i < 3; i++) {
	/*	  SNUCL_INFO("Before GWO[%d]: %lu\n", i, gwo[i]);
		  SNUCL_INFO("Before GWS[%d]: %lu\n", i, gws[i]);
		  SNUCL_INFO("Before LWS[%d]: %lu\n", i, lws[i]);
		  SNUCL_INFO("Before NWG[%d]: %lu\n", i, nwg[i]);
*/
		  gwo[i] = legacy_kernel_launch_params.gwo_[i];
		  gws[i] = legacy_kernel_launch_params.gws_[i];
		  lws[i] = legacy_kernel_launch_params.lws_[i];
		  nwg[i] = legacy_kernel_launch_params.nwg_[i];
/*		  SNUCL_INFO("GWO[%d]: %lu\n", i, gwo[i]);
		  SNUCL_INFO("GWS[%d]: %lu\n", i, gws[i]);
		  SNUCL_INFO("LWS[%d]: %lu\n", i, lws[i]);
		  SNUCL_INFO("NWG[%d]: %lu\n", i, nwg[i]);
*/
	  }
  }
  cl_event event;
  err = dispatch_->clEnqueueNDRangeKernel(kernel_queue_, legacy_kernel,
                                          work_dim, gwo, gws, lws, 0, NULL,
                                          &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
  gLegacyTimer.Stop();
  //if(strcmp(kernel->name(), "cffts1") == 0)
  //SNUCL_INFO("[Device %p Type %d] Kernel %s Time: %g sec\n", 
  	//	device_id_, type_,
	//	kernel->name(), gLegacyTimer.CurrentElapsed());
}

void LegacyDevice::LaunchNativeKernel(CLCommand* command,
                                      void (*user_func)(void*),
                                      void* native_args, size_t size,
                                      cl_uint num_mem_objects,
                                      CLMem** mem_list,
                                      ptrdiff_t* mem_offsets) {
  cl_mem* mem_list_dev = NULL;
  const void** args_mem_loc = NULL;
  if (num_mem_objects > 0) {
    for (cl_uint i = 0; i < num_mem_objects; i++) {
      cl_mem mem_dev = (cl_mem)mem_list[i]->GetDevSpecific(this);
      CHECK_ERROR(mem_dev == NULL, CL_INVALID_MEM_OBJECT);
      memcpy((void*)((size_t)native_args + mem_offsets[i]), &mem_dev,
             sizeof(void*));
    }
    mem_list_dev = (cl_mem*)malloc(sizeof(cl_mem) * num_mem_objects);
    args_mem_loc = (const void**)malloc(sizeof(const void*) * num_mem_objects);
    for (cl_uint i = 0; i < num_mem_objects; i++) {
      mem_list_dev[i] = (cl_mem)mem_list[i]->GetDevSpecific(this);
      args_mem_loc[i] = (const void*)((size_t)native_args + mem_offsets[i]);
    }
  }
  cl_int err;
  cl_event event;
  err = dispatch_->clEnqueueNativeKernel(kernel_queue_, user_func, native_args,
                                         size, num_mem_objects, mem_list_dev,
                                         args_mem_loc, 0, NULL, &event);
  if (num_mem_objects > 0) {
    free(mem_list_dev);
    free(args_mem_loc);
  }
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
}

void LegacyDevice::ReadBuffer(CLCommand* command, CLMem* mem_src,
                              size_t off_src, size_t size, void* ptr) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
//  SNUCL_INFO("[D2H Device: %p] ReadBuffer CLMem: %p->CLMem: %p, offset: %lu, size: %lu host ptr: %p\n",
//					this, mem_src, mem_src_dev, off_src, size, ptr);
  err = dispatch_->clEnqueueReadBuffer(mem_queue_, mem_src_dev, CL_TRUE,
                                       off_src, size, ptr, 0, NULL, NULL);
  UPDATE_ERROR(err);
}

void LegacyDevice::WriteBuffer(CLCommand* command, CLMem* mem_dst,
                               size_t off_dst, size_t size, void* ptr) {
  //gLegacyTimer.Start();
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
//  SNUCL_INFO("[H2D Device: %p] WriteBuffer CLMem: %p->CLMem: %p, offset: %lu, size: %lu host ptr: %p\n",
//					this, mem_dst, mem_dst_dev, off_dst, size, ptr);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  err = dispatch_->clEnqueueWriteBuffer(mem_queue_, mem_dst_dev, CL_TRUE,
                                        off_dst, size, ptr, 0, NULL, NULL);
  //err = dispatch_->clFinish(mem_queue_);
 // SNUCL_INFO("Writing Buffer Ptr %p with dispatch ptr: %p\n", ptr, dispatch_);
  //gLegacyTimer.Stop();
  UPDATE_ERROR(err);
}

void LegacyDevice::CopyBuffer(CLCommand* command, CLMem* mem_src,
                              CLMem* mem_dst, 
							   cl_mem mem_src_dev_specific, cl_mem mem_dst_dev_specific, 
							  size_t off_src, size_t off_dst,
                              size_t size) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_src_dev = mem_src_dev_specific;
  cl_mem mem_dst_dev = mem_dst_dev_specific;
  #if 0
  CLDevice *nearSrc = mem_src->FrontLatest();
  CLDevice *nearDest = mem_dst->FrontLatest();
  cl_mem src_tmp = (cl_mem)mem_src->GetDevSpecific(this);
  cl_mem dst_tmp = (cl_mem)mem_dst->GetDevSpecific(this);
  if(nearSrc == nearDest) 
  {
  	SNUCL_INFO("[D2D Device: %p] CopyBuffer Src Device: %p Dest Device: %p, src ptr: %p src tmp: %p dst ptr: %p dst tmp: %p\n",
  			this, nearSrc, nearDest, 
			mem_src_dev_specific, mem_dst_dev_specific,
			src_tmp, dst_tmp);
//  	SNUCL_INFO("[D2D Device: %p] CopyBuffer Src Device: %p Dest Device: %p\n",
  //			this, nearSrc, nearDest);
  }
  #endif
//  if(!mem_src_dev) mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
//  if(!mem_dst_dev) mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  //SNUCL_INFO("[D2D Device: %p] CopyBuffer CLMem: %p->CLMem: %p CLMem: %p->CLMem: %p src offset: %lu dst offset: %lu size: %lu\n",
	//				this, mem_src, mem_src_dev, 
	//				mem_dst, mem_dst_dev, 
	//				off_src, off_dst, size);

//  SNUCL_INFO("[D2D Device: %p] CopyBuffer SrcCLMem: %p->CLMem: %p, src_offset: %lu, \
  //			  							  DstCLMem: %p->CLMem: %p, dst_offset: %lu, size: %lu\n", 
	//				this, mem_src, mem_src_dev, off_src, 
	//					  mem_dst, mem_dst_dev, off_dst, size);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  cl_event event;
  //gLegacyTimer.Start();
  err = dispatch_->clEnqueueCopyBuffer(mem_queue_, mem_src_dev, mem_dst_dev,
                                       off_src, off_dst, size, 0, NULL,
                                       &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
  //gLegacyTimer.Stop();
  //SNUCL_INFO("Copy Buffer Time: %g sec\n", gLegacyTimer.CurrentElapsed());
}

void LegacyDevice::ReadImage(CLCommand* command, CLMem* mem_src,
                             size_t src_origin[3], size_t region[3],
                             size_t dst_row_pitch, size_t dst_slice_pitch,
                             void* ptr) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  err = dispatch_->clEnqueueReadImage(mem_queue_, mem_src_dev, CL_TRUE,
                                      src_origin, region, dst_row_pitch,
                                      dst_slice_pitch, ptr, 0, NULL, NULL);
  UPDATE_ERROR(err);
}

void LegacyDevice::WriteImage(CLCommand* command, CLMem* mem_dst,
                              size_t dst_origin[3], size_t region[3],
                              size_t src_row_pitch, size_t src_slice_pitch,
                              void* ptr) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  err = dispatch_->clEnqueueWriteImage(mem_queue_, mem_dst_dev, CL_TRUE,
                                       dst_origin, region, src_row_pitch,
                                       src_slice_pitch, ptr, 0, NULL, NULL);
  UPDATE_ERROR(err);
}

void LegacyDevice::CopyImage(CLCommand* command, CLMem* mem_src,
                             CLMem* mem_dst, size_t src_origin[3],
                             size_t dst_origin[3], size_t region[3]) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  cl_event event;
  err = dispatch_->clEnqueueCopyImage(mem_queue_, mem_src_dev, mem_dst_dev,
                                      src_origin, dst_origin, region, 0, NULL,
                                      &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
}

void LegacyDevice::CopyImageToBuffer(CLCommand* command, CLMem* mem_src,
                                     CLMem* mem_dst, size_t src_origin[3],
                                     size_t region[3], size_t off_dst) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  cl_event event;
  err = dispatch_->clEnqueueCopyImageToBuffer(mem_queue_, mem_src_dev,
                                              mem_dst_dev, src_origin, region,
                                              off_dst, 0, NULL, &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
}

void LegacyDevice::CopyBufferToImage(CLCommand* command, CLMem* mem_src,
                                     CLMem* mem_dst, size_t off_src,
                                     size_t dst_origin[3], size_t region[3]) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  cl_mem mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  cl_event event;
  err = dispatch_->clEnqueueCopyBufferToImage(mem_queue_, mem_src_dev,
                                              mem_dst_dev, off_src, dst_origin,
                                              region, 0, NULL, &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
}

void LegacyDevice::ReadBufferRect(CLCommand* command, CLMem* mem_src,
                                  size_t src_origin[3], size_t dst_origin[3],
                                  size_t region[3], size_t src_row_pitch,
                                  size_t src_slice_pitch, size_t dst_row_pitch,
                                  size_t dst_slice_pitch, void* ptr) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  CHECK_ERROR(version_ < LEGACY_VERSION_1_1, CL_INVALID_OPERATION);
  cl_mem mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  err = dispatch_->clEnqueueReadBufferRect(mem_queue_, mem_src_dev, CL_TRUE,
                                           src_origin, dst_origin, region,
                                           src_row_pitch, src_slice_pitch,
                                           dst_row_pitch, dst_slice_pitch,
                                           ptr, 0, NULL, NULL);
  UPDATE_ERROR(err);
}

void LegacyDevice::WriteBufferRect(CLCommand* command, CLMem* mem_dst,
                                   size_t src_origin[3], size_t dst_origin[3],
                                   size_t region[3], size_t src_row_pitch,
                                   size_t src_slice_pitch,
                                   size_t dst_row_pitch,
                                   size_t dst_slice_pitch, void* ptr) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  CHECK_ERROR(version_ < LEGACY_VERSION_1_1, CL_INVALID_OPERATION);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  err = dispatch_->clEnqueueWriteBufferRect(mem_queue_, mem_dst_dev, CL_TRUE,
                                            dst_origin, src_origin, region,
                                            dst_row_pitch, dst_slice_pitch,
                                            src_row_pitch, src_slice_pitch,
                                            ptr, 0, NULL, NULL);
  UPDATE_ERROR(err);
}

void LegacyDevice::CopyBufferRect(CLCommand* command, CLMem* mem_src,
                                  CLMem* mem_dst, size_t src_origin[3],
                                  size_t dst_origin[3], size_t region[3],
                                  size_t src_row_pitch, size_t src_slice_pitch,
                                  size_t dst_row_pitch,
                                  size_t dst_slice_pitch) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  CHECK_ERROR(version_ < LEGACY_VERSION_1_1, CL_INVALID_OPERATION);
  cl_mem mem_src_dev = (cl_mem)mem_src->GetDevSpecific(this);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, CL_INVALID_MEM_OBJECT);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  cl_event event;
  err = dispatch_->clEnqueueCopyBufferRect(mem_queue_, mem_src_dev,
                                           mem_dst_dev, src_origin, dst_origin,
                                           region, src_row_pitch,
                                           src_slice_pitch, dst_row_pitch,
                                           dst_slice_pitch, 0, NULL, &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
}

void LegacyDevice::FillBuffer(CLCommand* command, CLMem* mem_dst,
                              void* pattern, size_t pattern_size,
                              size_t off_dst, size_t size) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  CHECK_ERROR(version_ < LEGACY_VERSION_1_2, CL_INVALID_OPERATION);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  cl_event event;
  err = dispatch_->clEnqueueFillBuffer(mem_queue_, mem_dst_dev, pattern,
                                       pattern_size, off_dst, size, 0, NULL,
                                       &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
}

void LegacyDevice::FillImage(CLCommand* command, CLMem* mem_dst,
                          void* fill_color, size_t dst_origin[3],
                          size_t region[3]) {
  CHECK_ERROR(available_ == CL_FALSE, CL_DEVICE_NOT_AVAILABLE);
  CHECK_ERROR(version_ < LEGACY_VERSION_1_2, CL_INVALID_OPERATION);
  cl_mem mem_dst_dev = (cl_mem)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_dst_dev == NULL, CL_INVALID_MEM_OBJECT);
  cl_int err;
  cl_event event;
  err = dispatch_->clEnqueueFillImage(mem_queue_, mem_dst_dev, fill_color,
                                      dst_origin, region, 0, NULL, &event);
  UPDATE_ERROR(err);
  err = dispatch_->clWaitForEvents(1, &event);
  UPDATE_ERROR(err);
}

void LegacyDevice::BuildProgram(CLCommand* command, CLProgram* program,
                                CLProgramSource* source,
                                CLProgramBinary* binary, 
								const char* options) {
  cl_int err;
  cl_program legacy_program;
  cl_program legacy_training_program = NULL;
  if (binary != NULL)
    legacy_program = CreateProgram(binary);
  else {
    legacy_program = CreateProgram(source);
    legacy_training_program = CreateTrainingProgram(source);
  }
  if (legacy_program != NULL) {
    err = dispatch_->clBuildProgram(legacy_program, 1, &device_id_, options,
                                    NULL, NULL);
	VERIFY_ERROR(err);
	if(legacy_training_program != NULL ) {
		err = dispatch_->clBuildProgram(legacy_training_program, 1, &device_id_, options,
				NULL, NULL);
		VERIFY_ERROR(err);
	}
  }
  if (legacy_program != NULL && err == CL_SUCCESS) {
    CLProgramBinary* result = ReadBinary(legacy_program);
    char* build_log = ReadBuildLog(legacy_program);
    CLProgramBinary* training_result = NULL;
	char* training_build_log = NULL;
	if(legacy_training_program) {
		training_result = ReadBinary(legacy_training_program);
    	//training_build_log = ReadBuildLog(legacy_training_program);
	}
    program->CompleteBuild(this, CL_BUILD_SUCCESS, build_log, result,
                           (void*)legacy_program, 
						   training_result,
						   (void*)legacy_training_program
						   );
    ReadKernelInfo(legacy_program, program);
  } else {
    char* build_log = ReadBuildLog(legacy_program);
	SNUCL_ERROR("Build Log: %s\n", build_log);
    program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL, 
						NULL,
						NULL,
						NULL);
    dispatch_->clReleaseProgram(legacy_program);
    dispatch_->clReleaseProgram(legacy_training_program);
  }
}

void LegacyDevice::CompileProgram(CLCommand* command, CLProgram* program,
                                  CLProgramSource* source, const char* options,
                                  size_t num_headers,
                                  CLProgramSource** headers) {
  if (version_ < LEGACY_VERSION_1_2) {
    char* build_log = (char*)malloc(sizeof(char) *
                                    (strlen("Version < OpenCL 1.2") + 1));
    strcpy(build_log, "Version < OpenCL 1.2");
    program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL);
    return;
  }

  cl_int err;
  cl_program legacy_program;
  cl_program* input_headers = NULL;
  const char** header_include_names = NULL;
  legacy_program = CreateProgram(source);
  if (num_headers > 0) {
    input_headers = (cl_program*)malloc(sizeof(cl_program) * num_headers);
    header_include_names = (const char**)malloc(sizeof(const char*) *
                                                num_headers);
    for (size_t i = 0; i < num_headers; i++) {
      input_headers[i] = CreateProgram(headers[i]);
      header_include_names[i] = headers[i]->header_name();
    }
  }
  if (legacy_program != NULL) {
    err = dispatch_->clCompileProgram(legacy_program, 1, &device_id_, options,
                                      num_headers, input_headers,
                                      header_include_names, NULL, NULL);
  }
  if (num_headers > 0) {
    for (size_t i = 0; i < num_headers; i++)
      dispatch_->clReleaseProgram(input_headers[i]);
    free(input_headers);
    free(header_include_names);
  }
  if (legacy_program != NULL && err == CL_SUCCESS) {
    CLProgramBinary* result = ReadBinary(legacy_program);
    char* build_log = ReadBuildLog(legacy_program);
    program->CompleteBuild(this, CL_BUILD_SUCCESS, build_log, result);
  } else {
    char* build_log = ReadBuildLog(legacy_program);
    program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL);
  }
  dispatch_->clReleaseProgram(legacy_program);
}

void LegacyDevice::LinkProgram(CLCommand* command, CLProgram* program,
                               size_t num_binaries, CLProgramBinary** binaries,
                               const char* options) {
  if (version_ < LEGACY_VERSION_1_2) {
    char* build_log = (char*)malloc(sizeof(char) *
                                    (strlen("Version < OpenCL 1.2") + 1));
    strcpy(build_log, "Version < OpenCL 1.2");
    program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL);
    return;
  }

  cl_int err;
  cl_program legacy_program;
  cl_program* input_programs;
  input_programs = (cl_program*)malloc(sizeof(cl_program) * num_binaries);
  for (size_t i = 0; i < num_binaries; i++)
    input_programs[i] = CreateProgram(binaries[i]);
  legacy_program = dispatch_->clLinkProgram(context_, 1, &device_id_, options,
                                            num_binaries, input_programs, NULL,
                                            NULL, &err);
  for (size_t i = 0; i < num_binaries; i++)
    dispatch_->clReleaseProgram(input_programs[i]);
  free(input_programs);
  if (err == CL_SUCCESS) {
    CLProgramBinary* result = ReadBinary(legacy_program);
    char* build_log = ReadBuildLog(legacy_program);
    program->CompleteBuild(this, CL_BUILD_SUCCESS, build_log, result,
                           (void*)legacy_program, NULL, NULL);
    ReadKernelInfo(legacy_program, program);
  } else {
    char* build_log = ReadBuildLog(legacy_program);
    program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL, NULL,
						NULL, NULL);
    dispatch_->clReleaseProgram(legacy_program);
  }
}

void* LegacyDevice::AllocMem(CLMem* mem) {
  cl_int err = CL_SUCCESS;
  cl_mem_flags flags = mem->flags() & (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY |
                                       CL_MEM_WRITE_ONLY);
  cl_mem m = NULL;
  if (mem->IsSubBuffer()) {
    if (version_ >= LEGACY_VERSION_1_1) {
      cl_mem parent = (cl_mem)mem->parent()->GetDevSpecific(this);
      cl_buffer_region region = {mem->offset(), mem->size()};
      m = dispatch_->clCreateSubBuffer(parent, flags,
                                       CL_BUFFER_CREATE_TYPE_REGION, &region,
                                       &err);
    }
  } else if (mem->IsImage()) {
    cl_image_format image_format = mem->image_format();
    if (version_ >= LEGACY_VERSION_1_2) {
      cl_image_desc image_desc = mem->image_desc();
      m = dispatch_->clCreateImage(context_, flags, &image_format, &image_desc,
                                   NULL, &err);
    } else {
      if (mem->type() == CL_MEM_OBJECT_IMAGE2D) {
        m = dispatch_->clCreateImage2D(context_, flags, &image_format,
                                       mem->image_width(), mem->image_height(),
                                       mem->image_row_pitch(), NULL, &err);
      } else if (mem->type() == CL_MEM_OBJECT_IMAGE3D) {
        m = dispatch_->clCreateImage3D(context_, flags, &image_format,
                                       mem->image_width(), mem->image_height(),
                                       mem->image_depth(),
                                       mem->image_row_pitch(),
                                       mem->image_slice_pitch(), NULL, &err);
      }
    }
  } else {
    m = dispatch_->clCreateBuffer(context_, flags, mem->size(), NULL, &err);
  	//SNUCL_INFO("Mem: %p --> Dev Specific Mem: %p\n", mem, m);
  }
  if (err != CL_SUCCESS)
    m = NULL;
  return (void*)m;
}

void *LegacyDevice::AllocHostMem(CLMem *mem)
{
	void *ptr = NULL;
	//if(mem->flags() & CL_MEM_ALLOC_HOST_PTR)
	{
		cl_int err;
		void *m = mem->GetDevSpecific(this);
		ptr = dispatch_->clEnqueueMapBuffer(mem_queue_, 
								(cl_mem)m, CL_TRUE, 
								CL_MAP_READ|CL_MAP_WRITE, 0, mem->size(), 0, NULL, NULL,
								&err);
  		//SNUCL_INFO("[Queue: %p] Mapping CLMem %p, dev specific obj %p and host ptr: %p\n", mem_queue_, mem, m, ptr);
		VERIFY_ERROR(err);
	}
	return ptr;
}

void LegacyDevice::FreeHostMem(CLMem* mem, void* dev_specific) {
  if (dev_specific != NULL) {
	cl_int err;
	void *m = mem->GetDevSpecific(this);
	//dispatch_->clFinish(mem_queue_);
  	//SNUCL_INFO("[Queue: %p] Unmapping aligned memory %p, dev specificobj %p and host ptr: %p\n", mem_queue_, mem, m, dev_specific);
    err = dispatch_->clEnqueueUnmapMemObject(mem_queue_, (cl_mem)m, dev_specific, 
				0, NULL, NULL);
	VERIFY_ERROR(err);
	err = dispatch_->clFinish(mem_queue_);
	VERIFY_ERROR(err);
  	//SNUCL_INFO("Unmapped aligned memory %p, dev specific obj %p and host ptr: %p\n", mem, m, dev_specific);
  }
}

void LegacyDevice::FreeMem(CLMem* mem, void* dev_specific) {
  if (dev_specific != NULL) {
    dispatch_->clReleaseMemObject((cl_mem)dev_specific);
  }
}

void* LegacyDevice::AllocSampler(CLSampler* sampler) {
  cl_int err;
  cl_sampler s = dispatch_->clCreateSampler(context_,
                                            sampler->normalized_coords(),
                                            sampler->addressing_mode(),
                                            sampler->filter_mode(), &err);
  if (err != CL_SUCCESS)
    s = NULL;
  return (void*)s;
}

void LegacyDevice::FreeSampler(CLSampler* sampler, void* dev_specific) {
  if (dev_specific != NULL) {
    dispatch_->clReleaseSampler((cl_sampler)dev_specific);
  }
}

void LegacyDevice::FreeExecutable(CLProgram* program, void* executable) {
  if (executable != NULL) {
    dispatch_->clReleaseProgram((cl_program)executable);
  }
}

void* LegacyDevice::AllocTrainingKernel(CLKernel* kernel) {
  cl_int err;
  cl_program program = (cl_program)kernel->program()->GetTrainingExecutable(this);
  cl_kernel k = dispatch_->clCreateKernel(program, kernel->name(), &err);
  if (err != CL_SUCCESS)
    k = NULL;
  return (void*)k;
}
/*
void LegacyDevice::FreeTrainingKernel(CLKernel* kernel, void* dev_specific_training) {
  if (dev_specific_training != NULL) {
    dispatch_->clReleaseKernel((cl_kernel)dev_specific_training);
  }
}
*/

void* LegacyDevice::AllocKernel(CLKernel* kernel) {
  cl_int err;
  cl_program program = (cl_program)kernel->program()->GetExecutable(this);
  cl_kernel k = dispatch_->clCreateKernel(program, kernel->name(), &err);
  if (err != CL_SUCCESS)
    k = NULL;
  return (void*)k;
}

void LegacyDevice::FreeKernel(CLKernel* kernel, void* dev_specific) {
  if (dev_specific != NULL) {
    dispatch_->clReleaseKernel((cl_kernel)dev_specific);
  }
}

cl_program LegacyDevice::CreateTrainingProgram(CLProgramSource* source) {
  cl_int err;
  cl_program program;
  const char* concat_source = source->concat_training_source();
  size_t concat_source_length = source->concat_training_source_length();
  program = dispatch_->clCreateProgramWithSource(context_, 1, &concat_source,
                                                 &concat_source_length, &err);
  if (err != CL_SUCCESS)
    return NULL;
  return program;
}

cl_program LegacyDevice::CreateProgram(CLProgramSource* source) {
  cl_int err;
  cl_program program;
  const char* concat_source = source->concat_source();
  size_t concat_source_length = source->concat_source_length();
  program = dispatch_->clCreateProgramWithSource(context_, 1, &concat_source,
                                                 &concat_source_length, &err);
  if (err != CL_SUCCESS)
    return NULL;
  return program;
}

cl_program LegacyDevice::CreateProgram(CLProgramBinary* binary) {
  cl_int err;
  cl_program program;
  const unsigned char* core_binary = binary->core();
  size_t size = binary->core_size();
  program = dispatch_->clCreateProgramWithBinary(context_, 1, &device_id_,
                                                 &size, &core_binary, NULL,
                                                 &err);
  if (err != CL_SUCCESS)
    return NULL;
  return program;
}

CLProgramBinary* LegacyDevice::ReadBinary(cl_program program) {
  cl_int err;
  unsigned char* core_binary;
  size_t size;
  err = dispatch_->clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                    sizeof(size_t), &size, NULL);
  if (err != CL_SUCCESS || size == 0)
    return NULL;
  core_binary = (unsigned char*)malloc(size);
  err = dispatch_->clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                                    sizeof(unsigned char*), &core_binary,
                                    NULL);
  if (err != CL_SUCCESS) {
    free(core_binary);
    return NULL;
  }
  CLProgramBinary* binary = new CLProgramBinary(
      this, CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT, core_binary, size);
  free(core_binary);
  return binary;
}

char* LegacyDevice::ReadBuildLog(cl_program program) {
  cl_int err;
  char* build_log;
  size_t size;
  err = dispatch_->clGetProgramBuildInfo(program, device_id_,
                                         CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
  if (err != CL_SUCCESS || size == 0)
    return NULL;
  build_log = (char*)malloc(size);
  err = dispatch_->clGetProgramBuildInfo(program, device_id_,
                                         CL_PROGRAM_BUILD_LOG, size, build_log,
                                         NULL);
  if (err != CL_SUCCESS) {
    free(build_log);
    return NULL;
  }
  return build_log;
}

void LegacyDevice::ReadKernelInfo(cl_program legacy_program,
                                  CLProgram* program) {
  if (!program->BeginRegisterKernelInfo())
    return;

  cl_int err;
  cl_kernel* kernels;
  cl_uint num_kernels;
  err = dispatch_->clCreateKernelsInProgram(legacy_program, 0, NULL,
                                            &num_kernels);
  if (err != CL_SUCCESS || num_kernels == 0) {
    program->FinishRegisterKernelInfo();
    return;
  }
  kernels = (cl_kernel*)malloc(sizeof(cl_kernel) * num_kernels);
  err = dispatch_->clCreateKernelsInProgram(legacy_program, num_kernels,
                                            kernels, NULL);
  if (err != CL_SUCCESS) {
    free(kernels);
    program->FinishRegisterKernelInfo();
    return;
  }

  size_t size;
  for (cl_uint i = 0; i < num_kernels; i++) {
    char* function_name;
    cl_uint num_args;
    char* attributes;
    size_t work_group_size;
    size_t compile_work_group_size[3];
    cl_ulong local_mem_size;
    size_t preferred_work_group_size_multiple;
    cl_ulong private_mem_size;
    err = dispatch_->clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME, 0,
                                     NULL, &size);
    if (err != CL_SUCCESS || size == 0)
      continue;
    function_name = (char*)malloc(size);
    err = dispatch_->clGetKernelInfo(kernels[i], CL_KERNEL_FUNCTION_NAME,
                                       size, function_name, NULL);
    if (err != CL_SUCCESS) {
      free(function_name);
      continue;
    }
    err = dispatch_->clGetKernelInfo(kernels[i], CL_KERNEL_NUM_ARGS,
                                     sizeof(cl_uint), &num_args, NULL);
    if (err != CL_SUCCESS)
      num_args = 0;
    err = dispatch_->clGetKernelInfo(kernels[i], CL_KERNEL_ATTRIBUTES, 0, NULL,
                                     &size);
    if (err != CL_SUCCESS || size == 0) {
      attributes = (char*)malloc(sizeof(char));
      attributes[0] = '\0';
    } else {
      attributes = (char*)malloc(size);
      err = dispatch_->clGetKernelInfo(kernels[i], CL_KERNEL_ATTRIBUTES, size,
                                       attributes, NULL);
      if (err != CL_SUCCESS)
        attributes[0] = '\0';
    }
    err = dispatch_->clGetKernelWorkGroupInfo(kernels[i], device_id_,
                                              CL_KERNEL_WORK_GROUP_SIZE,
                                              sizeof(size_t), &work_group_size,
                                              NULL);
    if (err != CL_SUCCESS)
      work_group_size = max_work_group_size_;
    err = dispatch_->clGetKernelWorkGroupInfo(
        kernels[i], device_id_, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
        sizeof(size_t) * 3, compile_work_group_size, NULL);
    if (err != CL_SUCCESS) {
      compile_work_group_size[0] = 0;
      compile_work_group_size[1] = 0;
      compile_work_group_size[2] = 0;
    }
    err = dispatch_->clGetKernelWorkGroupInfo(kernels[i], device_id_,
                                              CL_KERNEL_LOCAL_MEM_SIZE,
                                              sizeof(cl_ulong),
                                              &local_mem_size, NULL);
    if (err != CL_SUCCESS)
      local_mem_size = 0;
    if (version_ >= LEGACY_VERSION_1_1) {
      err = dispatch_->clGetKernelWorkGroupInfo(
          kernels[i], device_id_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
          sizeof(size_t), &preferred_work_group_size_multiple, NULL);
      if (err != CL_SUCCESS)
        preferred_work_group_size_multiple = 0;
      err = dispatch_->clGetKernelWorkGroupInfo(kernels[i], device_id_,
                                                CL_KERNEL_PRIVATE_MEM_SIZE,
                                                sizeof(cl_ulong),
                                                &private_mem_size, NULL);
      if (err != CL_SUCCESS)
        private_mem_size = 0;
    } else {
      preferred_work_group_size_multiple = 0;
      private_mem_size = 0;
    }
    program->RegisterKernelInfo(function_name, num_args, attributes, this,
                                work_group_size, compile_work_group_size,
                                local_mem_size,
                                preferred_work_group_size_multiple,
                                private_mem_size, -1);
    if (version_ >= LEGACY_VERSION_1_2) {
      for (cl_uint j = 0; j < num_args; j++) {
        cl_kernel_arg_address_qualifier arg_address_qualifier;
        cl_kernel_arg_access_qualifier arg_access_qualifier;
        char* arg_type_name;
        cl_kernel_arg_type_qualifier arg_type_qualifier;
        char* arg_name;
        err = dispatch_->clGetKernelArgInfo(
            kernels[i], j, CL_KERNEL_ARG_ADDRESS_QUALIFIER,
            sizeof(cl_kernel_arg_address_qualifier), &arg_address_qualifier,
            NULL);
        if (err != CL_SUCCESS)
          arg_address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
        err = dispatch_->clGetKernelArgInfo(
            kernels[i], j, CL_KERNEL_ARG_ACCESS_QUALIFIER,
            sizeof(cl_kernel_arg_access_qualifier), &arg_access_qualifier,
            NULL);
        if (err != CL_SUCCESS)
          arg_access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
        err = dispatch_->clGetKernelArgInfo(kernels[i], j,
                                            CL_KERNEL_ARG_TYPE_NAME, 0, NULL,
                                            &size);
        if (err != CL_SUCCESS || size == 0) {
          arg_type_name = (char*)malloc(sizeof(char));
          arg_type_name[0] = '\0';
        } else {
          arg_type_name = (char*)malloc(size);
          err = dispatch_->clGetKernelArgInfo(kernels[i], j,
                                              CL_KERNEL_ARG_TYPE_NAME, size,
                                              arg_type_name, NULL);
          if (err != CL_SUCCESS)
            arg_type_name[0] = '\0';
        }
        err = dispatch_->clGetKernelArgInfo(
            kernels[i], j, CL_KERNEL_ARG_TYPE_QUALIFIER,
            sizeof(cl_kernel_arg_type_qualifier), &arg_type_qualifier, NULL);
        if (err != CL_SUCCESS)
          arg_type_qualifier = CL_KERNEL_ARG_TYPE_NONE;
        err = dispatch_->clGetKernelArgInfo(kernels[i], j, CL_KERNEL_ARG_NAME,
                                            0, NULL, &size);
        if (err != CL_SUCCESS || size == 0) {
          arg_name = (char*)malloc(sizeof(char));
          arg_name[0] = '\0';
        } else {
          arg_name = (char*)malloc(size);
          err = dispatch_->clGetKernelArgInfo(kernels[i], j,
                                              CL_KERNEL_ARG_NAME, size,
                                              arg_name, NULL);
          if (err != CL_SUCCESS)
            arg_name[0] = '\0';
        }
        program->RegisterKernelArgInfo(function_name, j, arg_address_qualifier,
                                       arg_access_qualifier, arg_type_name,
                                       arg_type_qualifier, arg_name);
        free(arg_type_name);
        free(arg_name);
      }
    }
    free(function_name);
    free(attributes);
  }

  for (cl_uint i = 0; i < num_kernels; i++)
    dispatch_->clReleaseKernel(kernels[i]);
  free(kernels);
  program->FinishRegisterKernelInfo();
}

void LegacyDevice::CreateDevices() {
  DIR* dir = opendir(ICD_VENDOR_PATH);
  fprintf(stderr, "ICD Path:%s, File ptr: %p\n", ICD_VENDOR_PATH, dir);
  if (dir == NULL)
    return;
  struct dirent* dir_entry;
  while ((dir_entry = readdir(dir)) != NULL) {
    switch (dir_entry->d_type) {
      case DT_UNKNOWN:
      case DT_REG:
      case DT_LNK: {
        char* library_name = IcdGetLibraryName(dir_entry->d_name);
		fprintf(stderr, "ICD Library Name: %s\n", library_name);
        if (library_name != NULL &&
            strcmp(library_name, ICD_SNUCL_CLUSTER_NAME) != 0 &&
            strcmp(library_name, ICD_SNUCL_SINGLE_NAME) != 0) {
          IcdVendorAdd(library_name);
          free(library_name);
        }
        break;
      }
      default: break;
    }
  }
  closedir(dir);
}

void LegacyDevice::IcdVendorAdd(const char* library_name) {
  void* library = dlopen(library_name, RTLD_NOW);
  fprintf(stderr, "[Device] 1; library %s: %p\n", library_name, library);
  if (library == NULL)
    return;
  cl_uint num_platforms;
  cl_platform_id* platforms = IcdGetPlatforms(library, &num_platforms);
  if (platforms != NULL) {
    for (cl_uint i = 0; i < num_platforms; i++) {
      if (platforms[i] == NULL)
        continue;
      IcdPlatformAdd(library_name, platforms[i]);
    }
    free(platforms);
  }
  dlclose(library);
}

void LegacyDevice::IcdPlatformAdd(const char* library_name,
                                  cl_platform_id platform) {
  cl_uint num_devices;
  fprintf(stderr, "[Device] 2\n");
  cl_device_id* devices = IcdGetDevices(platform, &num_devices);
  if (devices != NULL) {
  cl_int err = CL_SUCCESS;
  cl_context context = NULL;
  cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,
                                         (cl_context_properties)platform,
                                         0};
  context = platform->dispatch->clCreateContext(properties, num_devices, 
  				devices, NULL, NULL, &err);
	//context = NULL;
	if(err != CL_SUCCESS) 
	{
		context = NULL;
		SNUCL_ERROR("Context creation error for lib: %s\n", library_name);
	}
    for (cl_uint i = 0; i < num_devices; i++) {
  	  //fprintf(stderr, "[Device] DLOpening %s\n", library_name);
      void* library = dlopen(library_name, RTLD_NOW);
  	  //fprintf(stderr, "[Device] Platform Dispatch %p\n", platform->dispatch);
	  // Create a common context and pass it to the devices from same platform?
      LegacyDevice* device = new LegacyDevice(library, platform->dispatch,
											  context,
                                              platform, devices[i]);
  	  fprintf(stderr, "[Device] 3\n");
    }
    free(devices);
  }
}

char* LegacyDevice::IcdGetLibraryName(const char* entry_name) {
  size_t entry_length = strlen(entry_name);
  size_t extension_length = strlen(ICD_EXTENSION);
  if (entry_length < extension_length)
    return NULL;
  if (strcmp(entry_name + (entry_length - extension_length),
             ICD_EXTENSION) != 0)
    return NULL;

  char* file_name = (char*)malloc(strlen(ICD_VENDOR_PATH) + entry_length + 2);
  if (file_name == NULL)
    return NULL;
  sprintf(file_name, "%s/%s", ICD_VENDOR_PATH, entry_name);
  FILE* fp = fopen(file_name, "r");
  free(file_name);
  if (fp == NULL)
    return NULL;

  fseek(fp, 0, SEEK_END);
  size_t buffer_size = (size_t)ftell(fp);
  char* buffer = (char*)malloc(buffer_size + 1);
  if (buffer != NULL) {
    memset(buffer, 0, buffer_size + 1);
    fseek(fp, 0, SEEK_SET);
    if (fread(buffer, 1, buffer_size, fp) == buffer_size) {
      if (buffer[buffer_size - 1] == '\n')
        buffer[buffer_size - 1] = '\0';
    } else {
      free(buffer);
      buffer = NULL;
    }
  }
  fclose(fp);
  return buffer;
}

cl_platform_id* LegacyDevice::IcdGetPlatforms(void* library,
                                              cl_uint* num_platforms) {
  pfn_clGetExtensionFunctionAddress p_extFuncAddress = 
  		(pfn_clGetExtensionFunctionAddress)dlsym(library, "clGetExtensionFunctionAddress");
  fprintf(stderr, "[Device] Getting Extension Function Address: %p\n", p_extFuncAddress);
  pfn_clIcdGetPlatformIDs p_clIcdGetPlatformIDs =
  			(pfn_clIcdGetPlatformIDs) p_extFuncAddress("clIcdGetPlatformIDsKHR");
/*
  pfn_clIcdGetPlatformIDs p_clIcdGetPlatformIDs =
      (pfn_clIcdGetPlatformIDs)dlsym(library, "clIcdGetPlatformIDsKHR");*/
  fprintf(stderr, "[Device] Getting ICD platform: %p\n", p_clIcdGetPlatformIDs);
  if (p_clIcdGetPlatformIDs == NULL)
    return NULL;

  fprintf(stderr, "[Device] Got ICD platform ID\n");
  cl_int err;
  err = p_clIcdGetPlatformIDs(0, NULL, num_platforms);
  fprintf(stderr, "[Device] Got platform ID\n");
  if (err != CL_SUCCESS)
    return NULL;
  cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) *
                                                      (*num_platforms));
  if (platforms != NULL) {
    memset(platforms, 0, sizeof(cl_platform_id) * (*num_platforms));
    err = p_clIcdGetPlatformIDs(*num_platforms, platforms, NULL);
    fprintf(stderr, "[Device] Got  ICD platform ID 2\n");
    if (err != CL_SUCCESS) {
      free(platforms);
      platforms = NULL;
    }
  }
  return platforms;
}

cl_device_id* LegacyDevice::IcdGetDevices(cl_platform_id platform,
                                          cl_uint* num_devices) {
  cl_int err;
  err = platform->dispatch->clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0,
                                           NULL, num_devices);
  //fprintf(stderr, "[Device] 2.5\n");
  if (err != CL_SUCCESS)
    return NULL;
  cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) *
                                                (*num_devices));
  if (devices != NULL) {
    err = platform->dispatch->clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                             *num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
      free(devices);
      devices = NULL;
    }
  }
  fprintf(stderr, "[Device] Number of devices: %lu\n", *num_devices);
  return devices;
}
