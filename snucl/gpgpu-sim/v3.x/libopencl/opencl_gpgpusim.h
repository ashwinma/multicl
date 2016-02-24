#ifndef __OPENCL_GPGPUSIM__
#define __OPENCL_GPGPUSIM__
/* 
 * opencl_runtime_api.cc
 *
 * Copyright © 2009 by Tor M. Aamodt and the University of British Columbia, 
 * Vancouver, BC V6T 1Z4, All Rights Reserved.
 * 
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and 
 * benchmarks/template/ are derived from the CUDA SDK available from 
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from 
 * src/intersim/ are derived from Booksim (a simulator provided with the 
 * textbook "Principles and Practices of Interconnection Networks" available 
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by 
 * the corresponding legal terms and conditions set forth separately (original 
 * copyright notices are left in files from these sources and where we have 
 * modified a file our copyright notice appears before the original copyright 
 * notice).  
 * 
 * Using this version of GPGPU-Sim requires a complete installation of CUDA 
 * which is distributed seperately by NVIDIA under separate terms and 
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 * 
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.  
 *  
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 * 
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung, 
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia, 
 * Vancouver, BC V6T 1Z4
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#ifdef OPENGL_SUPPORT
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#endif

#define __CUDA_RUNTIME_API_H__
#include "host_defines.h"
#include "builtin_types.h"
#include "__cudaFatFormat.h"
#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/ptx_loader.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/gpgpusim_entrypoint.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/gpgpu-sim/shader.h"

//#   define __my_func__    __PRETTY_FUNCTION__
# if defined __cplusplus ? __GNUC_PREREQ (2, 6) : __GNUC_PREREQ (2, 4)
#   define __my_func__    __func__
# else
#  if defined __STDC_VERSION__ && __STDC_VERSION__ >= 199901L
#   define __my_func__    __my_func__
#  else
#   define __my_func__    ((__const char *) 0)
#  endif
# endif

#define CL_USE_DEPRECATED_OPENCL_1_0_APIS
#include <CL/cl.h>

#include <map>
#include <string>

typedef struct _emu_cl_platform_id *    emu_cl_platform_id;
typedef struct _emu_cl_device_id *      emu_cl_device_id;
typedef struct _emu_cl_context *        emu_cl_context;
typedef struct _emu_cl_command_queue *  emu_cl_command_queue;
typedef struct _emu_cl_mem *            emu_cl_mem;
typedef struct _emu_cl_program *        emu_cl_program;
typedef struct _emu_cl_kernel *         emu_cl_kernel;
//typedef struct _emu_cl_event *          emu_cl_event;
//typedef struct _emu_cl_sampler *        emu_cl_sampler;
// GPGPU-sim folks have not implemented anything related to the below two
typedef struct _cl_event *          emu_cl_event;
typedef struct _cl_sampler *        emu_cl_sampler;

struct _emu_cl_event {};
struct _emu_cl_sampler {};

struct _emu_cl_context {
   _emu_cl_context() {}
   _emu_cl_context( emu_cl_device_id gpu );
   emu_cl_device_id get_first_device();
   emu_cl_mem CreateBuffer(
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret );
   emu_cl_mem lookup_mem( emu_cl_mem m );
private:
   unsigned m_uid;
   emu_cl_device_id m_gpu;
   static unsigned sm_context_uid;

   std::map<void* /*host_ptr*/,emu_cl_mem> m_hostptr_to_cl_mem;
   std::map<emu_cl_mem/*device ptr*/,emu_cl_mem> m_devptr_to_cl_mem;
};

struct _emu_cl_device_id {
   _emu_cl_device_id() {}
   _emu_cl_device_id(gpgpu_sim* gpu) {m_id = 0; m_next = NULL; m_gpgpu=gpu;}
   struct _emu_cl_device_id *next() { return m_next; }
   gpgpu_sim *the_device() const { return m_gpgpu; }
private:
   unsigned m_id;
   gpgpu_sim *m_gpgpu;
   struct _emu_cl_device_id *m_next;
};

struct _emu_cl_command_queue 
{
   _emu_cl_command_queue() {}
   _emu_cl_command_queue( emu_cl_context context, emu_cl_device_id device, cl_command_queue_properties properties ) 
   {
      m_valid = true;
      m_context = context;
      m_device = device;
      m_properties = properties;
   }
   bool is_valid() { return m_valid; }
   emu_cl_context get_context() { return m_context; }
   emu_cl_device_id get_device() { return m_device; }
   cl_command_queue_properties get_properties() { return m_properties; }
private:
   bool m_valid;
   emu_cl_context                     m_context;
   emu_cl_device_id                   m_device;
   cl_command_queue_properties    m_properties;
};

struct _emu_cl_mem {
   _emu_cl_mem() {}
   _emu_cl_mem( cl_mem_flags flags, size_t size , void *host_ptr, cl_int *errcode_ret, emu_cl_device_id gpu );
   emu_cl_mem device_ptr();
   void* host_ptr();
   bool is_on_host() { return m_is_on_host; }
private:
   bool m_is_on_host;
   size_t m_device_ptr;
   void *m_host_ptr;
   cl_mem_flags m_flags; 
   size_t m_size;
};

struct pgm_info {
   std::string   m_source;
   std::string   m_asm;
   class symbol_table *m_symtab;
   std::map<std::string,function_info*> m_kernels;
};

struct _emu_cl_program {
   _emu_cl_program() {}
   _emu_cl_program( emu_cl_context context,
                cl_uint           count, 
             const char **     strings,   
             const size_t *    lengths );
   void Build(const char *options);
   emu_cl_kernel CreateKernel( const char *kernel_name, cl_int *errcode_ret );
   emu_cl_context get_context() { return m_context; }
   char *get_ptx();
   size_t get_ptx_size();

private:
   emu_cl_context m_context;
   std::map<cl_uint,pgm_info> m_pgm;
   static unsigned m_kernels_compiled;
};

struct _emu_cl_kernel {
   _emu_cl_kernel() {}
   _emu_cl_kernel( emu_cl_program prog, const char* kernel_name, class function_info *kernel_impl );
   void SetKernelArg(
      cl_uint      arg_index,
      size_t       arg_size,
      const void * arg_value );
   cl_int bind_args( gpgpu_ptx_sim_arg_list_t &arg_list );
   std::string name() const { return m_kernel_name; }
   size_t get_workgroup_size(emu_cl_device_id device);
   emu_cl_program get_program() { return m_prog; }
   class function_info *get_implementation() { return m_kernel_impl; }
private:
   unsigned m_uid;
   static unsigned sm_context_uid;
   emu_cl_program m_prog;

   std::string m_kernel_name;

   struct arg_info {
      size_t m_arg_size;
      const void *m_arg_value;
   };
   
   std::map<unsigned, arg_info> m_args;
   class function_info *m_kernel_impl;
};

struct _emu_cl_platform_id {
   static const unsigned m_uid = 0;
   void emu_clFoo(); 
};

static pgm_info *sg_info;

void gpgpusim_exit();
void gpgpusim_opencl_warning( const char* func, unsigned line, const char *desc );
void gpgpusim_opencl_error( const char* func, unsigned line, const char *desc );
void register_ptx_function( const char *name, function_info *impl );
void ptxinfo_addinfo();
class _emu_cl_device_id *GPGPUSim_Init();
void opencl_not_implemented( const char* func, unsigned line );
void opencl_not_finished( const char* func, unsigned line );
extern CL_API_ENTRY emu_cl_context CL_API_CALL
emu_clCreateContextFromType(const cl_context_properties * properties,
                        cl_device_type          device_type,
                        void (*pfn_notify)(const char *, const void *, size_t, void *),
                        void *                  user_data,
                        cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0;

/***************************** Unimplemented shell functions *******************************************/
extern CL_API_ENTRY emu_cl_program CL_API_CALL
emu_clCreateProgramWithBinary(emu_cl_context                     /* context */,
                          cl_uint                        /* num_devices */,
                          const emu_cl_device_id *           /* device_list */,
                          const size_t *                 /* lengths */,
                          const unsigned char **         /* binaries */,
                          cl_int *                       /* binary_status */,
                          cl_int *                       /* errcode_ret */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetEventProfilingInfo(emu_cl_event            /* event */,
                        cl_profiling_info   /* param_name */,
                        size_t              /* param_value_size */,
                        void *              /* param_value */,
                        size_t *            /* param_value_size_ret */) CL_API_SUFFIX__VERSION_1_0;
/*******************************************************************************************************/

extern CL_API_ENTRY emu_cl_context CL_API_CALL
emu_clCreateContext(  const cl_context_properties * properties,
                  cl_uint num_devices,
                  const emu_cl_device_id *devices,
                  void (*pfn_notify)(const char *, const void *, size_t, void *),
                  void *                  user_data,
                  cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetContextInfo(emu_cl_context         context, 
                 cl_context_info    param_name, 
                 size_t             param_value_size, 
                 void *             param_value, 
                 size_t *           param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0;


extern CL_API_ENTRY emu_cl_command_queue CL_API_CALL
emu_clCreateCommandQueue(emu_cl_context                     context, 
                     emu_cl_device_id                   device, 
                     cl_command_queue_properties    properties,
                     cl_int *                       errcode_ret) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY emu_cl_mem CL_API_CALL
emu_clCreateBuffer(emu_cl_context   context,
               cl_mem_flags flags,
               size_t       size ,
               void *       host_ptr,
               cl_int *     errcode_ret ) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY emu_cl_program CL_API_CALL
emu_clCreateProgramWithSource(emu_cl_context        context,
                          cl_uint           count,
                          const char **     strings,
                          const size_t *    lengths,
                          cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clBuildProgram(emu_cl_program           program,
               cl_uint              num_devices,
               const emu_cl_device_id * device_list,
               const char *         options, 
               void (*pfn_notify)(emu_cl_program /* program */, void * /* user_data */),
               void *               user_data ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY emu_cl_kernel CL_API_CALL
emu_clCreateKernel(emu_cl_program      program,
               const char *    kernel_name,
               cl_int *        errcode_ret) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clSetKernelArg(emu_cl_kernel    kernel,
               cl_uint      arg_index,
               size_t       arg_size,
               const void * arg_value ) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clEnqueueNDRangeKernel(emu_cl_command_queue command_queue,
                       emu_cl_kernel        kernel,
                       cl_uint          work_dim,
                       const size_t *   global_work_offset,
                       const size_t *   global_work_size,
                       const size_t *   local_work_size,
                       cl_uint          num_events_in_wait_list,
                       const emu_cl_event * event_wait_list,
                       emu_cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clEnqueueReadBuffer(emu_cl_command_queue    command_queue,
                    emu_cl_mem              buffer,
                    cl_bool             blocking_read,
                    size_t              offset,
                    size_t              cb, 
                    void *              ptr,
                    cl_uint             num_events_in_wait_list,
                    const emu_cl_event *    event_wait_list,
                    emu_cl_event *          event ) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clEnqueueWriteBuffer(emu_cl_command_queue   command_queue, 
                     emu_cl_mem             buffer, 
                     cl_bool            blocking_write, 
                     size_t             offset, 
                     size_t             cb, 
                     const void *       ptr, 
                     cl_uint            num_events_in_wait_list, 
                     const emu_cl_event *   event_wait_list, 
                     emu_cl_event *         event ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clReleaseMemObject(emu_cl_mem /* memobj */) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clReleaseKernel(emu_cl_kernel   /* kernel */) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clReleaseProgram(emu_cl_program /* program */) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clReleaseCommandQueue(emu_cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clReleaseContext(emu_cl_context /* context */) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetPlatformIDs(cl_uint num_entries, emu_cl_platform_id *platforms, cl_uint *num_platforms ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL 
emu_clGetPlatformInfo(emu_cl_platform_id   platform, 
                  cl_platform_info param_name,
                  size_t           param_value_size, 
                  void *           param_value,
                  size_t *         param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetDeviceIDs(emu_cl_platform_id   platform,
               cl_device_type   device_type, 
               cl_uint          num_entries, 
               emu_cl_device_id *   devices, 
               cl_uint *        num_devices ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetDeviceInfo(emu_cl_device_id    device,
                cl_device_info  param_name, 
                size_t          param_value_size, 
                void *          param_value,
                size_t *        param_value_size_ret) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clFinish(emu_cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetProgramInfo(emu_cl_program         program,
                 cl_program_info    param_name,
                 size_t             param_value_size,
                 void *             param_value,
                 size_t *           param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clEnqueueCopyBuffer(emu_cl_command_queue    command_queue, 
                    emu_cl_mem              src_buffer,
                    emu_cl_mem              dst_buffer, 
                    size_t              src_offset,
                    size_t              dst_offset,
                    size_t              cb, 
                    cl_uint             num_events_in_wait_list,
                    const emu_cl_event *    event_wait_list,
                    emu_cl_event *          event ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetKernelWorkGroupInfo(emu_cl_kernel                  kernel,
                         emu_cl_device_id               device,
                         cl_kernel_work_group_info  param_name,
                         size_t                     param_value_size,
                         void *                     param_value,
                         size_t *                   param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0;
extern CL_API_ENTRY cl_int CL_API_CALL
emu_clWaitForEvents(cl_uint             /* num_events */,
                const emu_cl_event *    /* event_list */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clReleaseEvent(emu_cl_event /* event */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetCommandQueueInfo(emu_cl_command_queue      command_queue,
                      cl_command_queue_info param_name,
                      size_t                param_value_size,
                      void *                param_value,
                      size_t *              param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clFlush(emu_cl_command_queue /* command_queue */) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clGetSupportedImageFormats(emu_cl_context           context,
                           cl_mem_flags         flags,
                           cl_mem_object_type   image_type,
                           cl_uint              num_entries,
                           cl_image_format *    image_formats,
                           cl_uint *            num_image_formats) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY void * CL_API_CALL
emu_clEnqueueMapBuffer(emu_cl_command_queue command_queue,
                   emu_cl_mem           buffer,
                   cl_bool          blocking_map, 
                   cl_map_flags     map_flags,
                   size_t           offset,
                   size_t           cb,
                   cl_uint          num_events_in_wait_list,
                   const emu_cl_event * event_wait_list,
                   emu_cl_event *       event,
                   cl_int *         errcode_ret ) CL_API_SUFFIX__VERSION_1_0;

extern CL_API_ENTRY cl_int CL_API_CALL
emu_clSetCommandQueueProperty( emu_cl_command_queue command_queue,
                              cl_command_queue_properties properties,
                              cl_bool enable,
                              cl_command_queue_properties *old_properties
                           ) CL_API_SUFFIX__VERSION_1_0;



 #endif // __OPENCL_GPGPUSIM__
