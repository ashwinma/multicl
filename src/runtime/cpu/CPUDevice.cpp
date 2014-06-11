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

#include "cpu/CPUDevice.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <limits.h>
#include <malloc.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include "CLCommand.h"
#include "CLDevice.h"
#include "CLKernel.h"
#include "CLMem.h"
#include "CLPlatform.h"
#include "CLProgram.h"
#include "CLSampler.h"
#include "Utils.h"
#include "cpu/CPUComputeUnit.h"

using namespace std;

#define CPU_SCHEDULE_DYNAMIC

#define SNUCL_BUILD_DIR "snucl_kernels"
#define SNUCL_CPU_BUILD_DIR "snucl_kernels/cpu"

enum CPUSamplerNormalizedCoords {
  CLK_NORMALIZED_COORDS_TRUE = 0x0100,
  CLK_NORMALIZED_COORDS_FALSE = 0x0080
};

enum CPUSamplerAddressingModes {
  CLK_ADDRESS_CLAMP_TO_EDGE = 0x0010, 
  CLK_ADDRESS_CLAMP = 0x0008, 
  CLK_ADDRESS_NONE = 0x0004, 
  CLK_ADDRESS_REPEAT = 0x0002, 
  CLK_ADDRESS_MIRRORED_REPEAT = 0x0001
};

enum CPUSamplerFilterModes {
  CLK_FILTER_NEAREST = 0x0040, 
  CLK_FILTER_LINEAR = 0x0020
};

CPUDevice::CPUDevice(int num_cores)
    : CLDevice(0) {
  srand(time(NULL));
  num_cores_ = num_cores;
  compute_units_ = (CPUComputeUnit**)malloc(sizeof(CPUComputeUnit*) *
                                            num_cores_);
  for (int i = 0; i < num_cores_; i++)
    compute_units_[i] = new CPUComputeUnit(this, i);

  type_ = CL_DEVICE_TYPE_CPU;
  vendor_id_ = 201110;
  max_compute_units_ = num_cores_;
  max_work_item_dimensions_ = 3;
  max_work_item_sizes_[0] = max_work_item_sizes_[1] =
      max_work_item_sizes_[2] = 4096;
  max_work_group_size_ = 4096;

  preferred_vector_width_char_ = 16 / sizeof(char);
  preferred_vector_width_short_ = 16 / sizeof(short);
  preferred_vector_width_int_ = 16 / sizeof(int);
  preferred_vector_width_long_ = 16 / sizeof(long);
  preferred_vector_width_float_ = 16 / sizeof(float);
  preferred_vector_width_double_ = 0;
  preferred_vector_width_half_ = 0;
  native_vector_width_char_ = 16 / sizeof(char);
  native_vector_width_short_ = 16 / sizeof(short);
  native_vector_width_int_ = 16 / sizeof(int);
  native_vector_width_long_ = 16 / sizeof(long);
  native_vector_width_float_ = 16 / sizeof(float);
  native_vector_width_double_ = 0;
  native_vector_width_half_ = 0;
  {
    char buf[64];
    PipeRead("/bin/cat /proc/cpuinfo | grep '^cpu MHz' | head -1", buf,
             sizeof(buf) - 1);
    float frequency;
    sscanf(buf, "cpu MHz %*s %f", &frequency);
    max_clock_frequency_ = (cl_uint)frequency;
  }
  address_bits_ = 32;

  max_mem_alloc_size_ = 128 * 1024 * 1024ULL;

  image_support_ = CL_TRUE;
  max_read_image_args_ = 128;
  max_write_image_args_ = 128;
  image2d_max_width_ = 8192;
  image2d_max_height_ = 8192;
  image3d_max_width_ = 2048;
  image3d_max_height_ = 2048;
  image3d_max_depth_ = 2048;
  image_max_buffer_size_ = 65536;
  image_max_array_size_ = 65536;
  max_samplers_ = 16;

  max_parameter_size_ = 1024;
  mem_base_addr_align_ = 1024;
  min_data_type_align_size_ = sizeof(cl_long16);

  single_fp_config_ = CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST |
                      CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA;
                 // | CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT | CL_FP_SOFT_FLOAT;
  double_fp_config_ = CL_FP_DENORM | CL_FP_INF_NAN | CL_FP_ROUND_TO_NEAREST |
                      CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_FMA;

  global_mem_cache_type_ = CL_READ_WRITE_CACHE;
  global_mem_cacheline_size_ = 128;
  global_mem_cache_size_ = 4 * 1024;
  global_mem_size_ = 128 * 1024 * 1024ULL;

  max_constant_buffer_size_ = 64 * 1024;
  max_constant_args_ = 8;

  local_mem_type_ = CL_GLOBAL;
  local_mem_size_ = 4 * 1024 * 1024;
  error_correction_support_ = CL_FALSE;

  host_unified_memory_ = CL_TRUE;

  profiling_timer_resolution_ = 1;

  endian_little_ = CL_TRUE;
  available_ = CL_TRUE;

  compiler_available_ = CL_TRUE;
  linker_available_ = CL_TRUE;

  execution_capabilities_ = CL_EXEC_KERNEL | CL_EXEC_NATIVE_KERNEL;
  queue_properties_ = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
                      CL_QUEUE_PROFILING_ENABLE;

  built_in_kernels_ = "";

  {
    char* name = (char*)malloc(sizeof(char) * 64);
    gethostname(name, sizeof(char) * 63);
    name_ = name;
  }
  {
    char buf[64];
    PipeRead("/bin/cat /proc/cpuinfo | grep '^vendor_id' | head -1", buf,
             sizeof(buf) - 1);
    char* vendor = (char*)malloc(sizeof(char) * 64);
    sscanf(buf, "vendor_id %*s %s", vendor);
    vendor_ = vendor;
  }
  driver_version_ = "SnuCL 1.2";
  profile_ = "FULL_PROFILE";
  device_version_ = "OpenCL 1.2 rev01";
  opencl_c_version_ = "OpenCL 1.2 rev01";
  device_extensions_ = "cl_khr_global_int32_base_atomics "
                       "cl_khr_global_int32_extended_atomics "
                       "cl_khr_local_int32_base_atomics "
                       "cl_khr_local_int32_extended_atomics "
                       "cl_khr_byte_addressable_store "
                       "cl_khr_fp64";

  printf_buffer_size_ = 1024 * 1024;

  preferred_interop_user_sync_ = CL_FALSE;

//  partition_max_sub_devices_ = num_cores_;
//  partition_max_compute_units_ = num_cores_;
//  partition_properties[0] = CL_DEVICE_PARTITION_EQUALLY; 
//  partition_properties[1] = CL_DEVICE_PARTITION_BY_COUNTS;
  partition_max_sub_devices_ = 1;
  partition_max_compute_units_ = num_cores_;
  num_partition_properties_ = 0;
  affinity_domain_ = 0; 
  partition_type_ = NULL;
  partition_type_len_ = 0;

  kernel_dir_ = (char*)malloc(sizeof(char) * 128);
  sprintf(kernel_dir_, "%s/%s", SNUCL_CPU_BUILD_DIR, name_);
}

CPUDevice::~CPUDevice() {
  for (int i = 0; i < num_cores_; i++)
    delete compute_units_[i];
  free(compute_units_);

  free((char*)name_);
  free((char*)vendor_);
  free(kernel_dir_);
}

void CPUDevice::LaunchKernel(CLCommand* command, CLKernel* kernel,
                             cl_uint work_dim, size_t gwo[3], size_t gws[3],
                             size_t lws[3], size_t nwg[3],
                             map<cl_uint, CLKernelArg*>* kernel_args) {
  CPUKernelParam param;
  size_t args_size = SetKernelParam(&param, kernel, work_dim, gwo, gws, lws,
                                    nwg, kernel_args);
  void* handle = kernel->program()->GetExecutable(this);
  CPUWorkGroupAssignment* wga =
      (CPUWorkGroupAssignment*)malloc(sizeof(CPUWorkGroupAssignment) *
                                      num_cores_);
  for (int i = 0; i < num_cores_; i++) {
    wga[i].handle = handle;
    wga[i].param = &param;
    wga[i].args_size = args_size;
  }

#ifdef CPU_SCHEDULE_DYNAMIC
  ScheduleDynamic(nwg, wga);
#else
  ScheduleStatic(nwg, wga);
#endif

  for (int i = 0; i < num_cores_; i++)
    compute_units_[i]->Sync();
  free(wga);
}

void CPUDevice::LaunchNativeKernel(CLCommand* command,
                                   void (*user_func)(void*), void* native_args,
                                   size_t size, cl_uint num_mem_objects,
                                   CLMem** mem_list, ptrdiff_t* mem_offsets) {
  for (cl_uint i = 0; i < num_mem_objects; i++) {
    CLMem* mem = mem_list[i];
    void* ptr = mem->GetDevSpecific(this);
    memcpy((void*)((size_t)native_args + mem_offsets[i]), &ptr, sizeof(void*));
  }
  user_func(native_args);
}

void CPUDevice::ReadBuffer(CLCommand* command, CLMem* mem_src, size_t off_src,
                           size_t size, void* ptr) {
  void* mem_src_dev = mem_src->GetDevSpecific(this);
  if (ptr != (void*)((size_t)mem_src_dev + off_src))
    memcpy(ptr, (void*)((size_t)mem_src_dev + off_src), size);
}

void CPUDevice::WriteBuffer(CLCommand* command, CLMem* mem_dst, size_t off_dst,
                            size_t size, void* ptr) {
  void* mem_dst_dev = mem_dst->GetDevSpecific(this);
  if ((void*)((size_t)mem_dst_dev + off_dst) != ptr)
    memcpy((void*)((size_t)mem_dst_dev + off_dst), ptr, size);
}

void CPUDevice::CopyBuffer(CLCommand* command, CLMem* mem_src, CLMem* mem_dst,
                           size_t off_src, size_t off_dst, size_t size) {
  void* mem_src_dev = mem_src->GetDevSpecific(this);
  void* mem_dst_dev = mem_dst->GetDevSpecific(this);
  if ((void*)((size_t)mem_dst_dev + off_dst) !=
      (void*)((size_t)mem_src_dev + off_src))
    memcpy((void*)((size_t)mem_dst_dev + off_dst),
           (void*)((size_t)mem_src_dev + off_src), size);
}

void CPUDevice::ReadImage(CLCommand* command, CLMem* mem_src,
                          size_t src_origin[3], size_t region[3],
                          size_t dst_row_pitch, size_t dst_slice_pitch,
                          void* ptr) {
  ReadImageCommon(mem_src, src_origin, region, dst_row_pitch, dst_slice_pitch,
                  ptr);
}

void CPUDevice::WriteImage(CLCommand* command, CLMem* mem_dst,
                           size_t dst_origin[3], size_t region[3],
                           size_t src_row_pitch, size_t src_slice_pitch,
                           void* ptr) {
  WriteImageCommon(mem_dst, dst_origin, region, src_row_pitch, src_slice_pitch,
                   ptr);
}

void CPUDevice::CopyImage(CLCommand* command, CLMem* mem_src, CLMem* mem_dst,
                          size_t src_origin[3], size_t dst_origin[3],
                          size_t region[3]) {
  void* mem_src_dev = mem_src->GetDevSpecific(this);
  void* mem_dst_dev = mem_dst->GetDevSpecific(this);

  size_t s_image_row_pitch = mem_src->image_row_pitch();
  size_t s_image_slice_pitch = mem_src->image_slice_pitch();

  cl_mem_object_type d_image_type = mem_dst->type();
  size_t d_image_element_size = mem_dst->image_element_size();
  size_t d_image_row_pitch = mem_dst->image_row_pitch();
  size_t d_image_slice_pitch = mem_dst->image_slice_pitch();

  switch (d_image_type) {
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      CopyRegion(mem_src_dev, mem_dst_dev, 3, src_origin, dst_origin, region,
                 d_image_element_size, s_image_row_pitch, s_image_slice_pitch,
                 d_image_row_pitch, d_image_slice_pitch);
      break;
    case CL_MEM_OBJECT_IMAGE2D:
      CopyRegion(mem_src_dev, mem_dst_dev, 2, src_origin, dst_origin, region,
                 d_image_element_size, s_image_row_pitch, 0, d_image_row_pitch,
                 0);
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      CopyRegion(mem_src_dev, mem_dst_dev, 2, src_origin, dst_origin, region,
                 d_image_element_size, s_image_slice_pitch, 0,
                 d_image_slice_pitch, 0);
      break;
    case CL_MEM_OBJECT_IMAGE1D:
      CopyRegion(mem_src_dev, mem_dst_dev, 1, src_origin, dst_origin, region,
                 d_image_element_size, 0, 0, 0, 0);
      break;
    default: break;
  }
}

void CPUDevice::CopyImageToBuffer(CLCommand* command, CLMem* mem_src,
                                  CLMem* mem_dst, size_t src_origin[3],
                                  size_t region[3], size_t off_dst) {
  ReadImageCommon(mem_src, src_origin, region, 0, 0,
                  (void*)((size_t)mem_dst->GetDevSpecific(this) + off_dst));
}

void CPUDevice::CopyBufferToImage(CLCommand* command, CLMem* mem_src,
                                  CLMem* mem_dst, size_t off_src,
                                  size_t dst_origin[3], size_t region[3]) {
  WriteImageCommon(mem_dst, dst_origin, region, 0, 0,
                   (void*)((size_t)mem_src->GetDevSpecific(this) + off_src));
}

void CPUDevice::ReadBufferRect(CLCommand* command, CLMem* mem_src,
                               size_t src_origin[3], size_t dst_origin[3],
                               size_t region[3], size_t src_row_pitch,
                               size_t src_slice_pitch, size_t dst_row_pitch,
                               size_t dst_slice_pitch, void* ptr) {
  CopyRegion(mem_src->GetDevSpecific(this), ptr, 3, src_origin, dst_origin,
             region, 1, src_row_pitch, src_slice_pitch, dst_row_pitch,
             dst_slice_pitch);
}

void CPUDevice::WriteBufferRect(CLCommand* command, CLMem* mem_dst,
                                size_t src_origin[3], size_t dst_origin[3],
                                size_t region[3], size_t src_row_pitch,
                                size_t src_slice_pitch, size_t dst_row_pitch,
                                size_t dst_slice_pitch, void* ptr) {
  CopyRegion(ptr, mem_dst->GetDevSpecific(this), 3, src_origin, dst_origin,
             region, 1, src_row_pitch, src_slice_pitch, dst_row_pitch,
             dst_slice_pitch);
}

void CPUDevice::CopyBufferRect(CLCommand* command, CLMem* mem_src,
                               CLMem* mem_dst, size_t src_origin[3],
                               size_t dst_origin[3], size_t region[3],
                               size_t src_row_pitch, size_t src_slice_pitch,
                               size_t dst_row_pitch, size_t dst_slice_pitch) {
  CopyRegion(mem_src->GetDevSpecific(this), mem_dst->GetDevSpecific(this), 3,
             src_origin, dst_origin, region, 1, src_row_pitch, src_slice_pitch,
             dst_row_pitch, dst_slice_pitch);
}

void CPUDevice::FillBuffer(CLCommand* command, CLMem* mem_dst, void* pattern,
                           size_t pattern_size, size_t off_dst, size_t size) {
  char* mem_dst_dev = (char*)mem_dst->GetDevSpecific(this);
  size_t index = off_dst;
  while (index + pattern_size <= off_dst + size) {
    memcpy(mem_dst_dev + index, pattern, pattern_size);
    index += pattern_size;
  }
}

void CPUDevice::FillImage(CLCommand* command, CLMem* mem_dst,
                          void* fill_color, size_t dst_origin[3],
                          size_t region[3]) {
  void* mem_dst_dev = mem_dst->GetDevSpecific(this);

  cl_mem_object_type image_type = mem_dst->type();
  size_t image_element_size = mem_dst->image_element_size();
  size_t image_channels = mem_dst->image_channels();
  size_t image_row_pitch = mem_dst->image_row_pitch();
  size_t image_slice_pitch = mem_dst->image_slice_pitch();
  cl_image_format image_format = mem_dst->image_format();
  cl_channel_type image_channel_data_type =
      image_format.image_channel_data_type;

  size_t pattern_size;
  switch (image_channel_data_type) {
    case CL_SIGNED_INT8:
    case CL_SIGNED_INT16:
    case CL_SIGNED_INT32:
      pattern_size = image_channels * sizeof(cl_int);
      break;
    case CL_UNSIGNED_INT8:
    case CL_UNSIGNED_INT16:
    case CL_UNSIGNED_INT32:
      pattern_size = image_channels * sizeof(cl_uint);
      break;
    default:
      pattern_size = image_channels * sizeof(cl_float);
      break;
  }
  void* packed_color = malloc(pattern_size);
  switch (image_channel_data_type) {
    case CL_SIGNED_INT8:
    case CL_SIGNED_INT16:
    case CL_SIGNED_INT32:
      PackImagePixel((cl_int*)fill_color, packed_color, image_channels,
                     &image_format);
      break;
    case CL_UNSIGNED_INT8:
    case CL_UNSIGNED_INT16:
    case CL_UNSIGNED_INT32:
      PackImagePixel((cl_uint*)fill_color, packed_color, image_channels,
                     &image_format);
      break;
    default:
      PackImagePixel((cl_float*)fill_color, packed_color, image_channels,
                     &image_format);
      break;
  }

  switch (image_type) {
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY: {
      size_t zmax = dst_origin[2] + region[2];
      size_t ymax = dst_origin[1] + region[1];
      size_t xmax = dst_origin[0] + region[0];
      for (size_t z = dst_origin[2]; z < zmax; z++) {
        for (size_t y = dst_origin[1]; y < ymax; y++) {
          for (size_t x = dst_origin[0]; x < xmax; x++) {
            size_t off = z * image_slice_pitch + y * image_row_pitch +
                         x * image_element_size;
            memcpy((char*)mem_dst_dev + off, packed_color, image_element_size);
          }
        }
      }
      break;
    }
    case CL_MEM_OBJECT_IMAGE2D: {
      size_t ymax = dst_origin[1] + region[1];
      size_t xmax = dst_origin[0] + region[0];
      for (size_t y = dst_origin[1]; y < ymax; y++) {
        for (size_t x = dst_origin[0]; x < xmax; x++) {
          size_t off = y * image_row_pitch + x * image_element_size;
          memcpy((char*)mem_dst_dev + off, packed_color, image_element_size);
        }
      }
      break;
    }
    case CL_MEM_OBJECT_IMAGE1D_ARRAY: {
      size_t ymax = dst_origin[1] + region[1];
      size_t xmax = dst_origin[0] + region[0];
      for (size_t y = dst_origin[1]; y < ymax; y++) {
        for (size_t x = dst_origin[0]; x < xmax; x++) {
          size_t off = y * image_slice_pitch + x * image_element_size;
          memcpy((char*)mem_dst_dev + off, packed_color, image_element_size);
        }
      }
      break;
    }
    case CL_MEM_OBJECT_IMAGE1D: {
      size_t xmax = dst_origin[0] + region[0];
      for (size_t x = dst_origin[0]; x < xmax; x++) {
        size_t off = x * image_element_size;
        memcpy((char*)mem_dst_dev + off, packed_color, image_element_size);
      }
      break;
    }
    default: break;
  }
}

void CPUDevice::BuildProgram(CLCommand* command, CLProgram* program,
                             CLProgramSource* source, CLProgramBinary* binary,
                             const char* options) {
  char object_index[11];
  char executable_index[11];
  bool compile_required = false;
  bool link_required = false;

  CheckKernelDir();
  if (binary != NULL) {
    switch (binary->type()) {
      case CL_PROGRAM_BINARY_TYPE_EXECUTABLE:
        GenerateFileIndex(executable_index);
        WriteExecutable(binary, executable_index);
        break;
      case CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT:
        GenerateFileIndex(object_index);
        WriteCompiledObject(binary, object_index);
        link_required = true;
        break;
      default:
        SNUCL_ERROR("Unsupported binary type");
        break;
    }
  } else { // source != NULL
    GenerateFileIndex(object_index);
    WriteSource(source, object_index);
    compile_required = true;
    link_required = true;
  }
  if (compile_required) {
    RunCompiler(object_index, options);
    if (!CheckCompileResult(object_index)) {
      char* build_log = ReadCompileLog(object_index);
      program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL, NULL);
      return;
    }
  }
  if (link_required) {
    char* object_index_ptr = object_index;
    GenerateFileIndex(executable_index);
    RunLinker(executable_index, 1, &object_index_ptr, options);
    if (!CheckLinkResult(executable_index)) {
      char* build_log = ReadLinkLog(executable_index);
      program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL, NULL);
      return;
    }
  }

  CLProgramBinary* result = ReadExecutable(executable_index);
  void* executable = OpenExecutable(executable_index);
  program->CompleteBuild(this, CL_BUILD_SUCCESS, NULL, result, executable);
  ReadKernelInfo(executable_index, program);
}

void CPUDevice::CompileProgram(CLCommand* command, CLProgram* program,
                               CLProgramSource* source, const char* options,
                               size_t num_headers, CLProgramSource** headers) {
  char file_index[11];

  CheckKernelDir();
  GenerateFileIndex(file_index);
  WriteSource(source, file_index);
  for (size_t i = 0; i < num_headers; i++)
    WriteHeader(headers[i]);

  string new_options = AppendHeaderOptions(options, num_headers, headers);
  RunCompiler(file_index, (new_options.empty() ? NULL : new_options.c_str()));
  if (!CheckCompileResult(file_index)) {
    char* build_log = ReadCompileLog(file_index);
    program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL);
  }
  CLProgramBinary* result = ReadCompiledObject(file_index);
  program->CompleteBuild(this, CL_BUILD_SUCCESS, NULL, result);
}

void CPUDevice::LinkProgram(CLCommand* command, CLProgram* program,
                            size_t num_binaries, CLProgramBinary** binaries,
                            const char* options) {
  char** object_indices;
  char executable_index[11];

  CheckKernelDir();
  object_indices = (char**)malloc(sizeof(char*) * num_binaries);
  for (size_t i = 0; i < num_binaries; i++) {
    object_indices[i] = (char*)malloc(sizeof(char) * 11);
    if (binaries[i]->type() == CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT) {
      GenerateFileIndex(object_indices[i]);
      WriteCompiledObject(binaries[i], object_indices[i]);
    } else {
      SNUCL_ERROR("Unsupported binary type");
    }
  }
  GenerateFileIndex(executable_index);
  RunLinker(executable_index, num_binaries, object_indices, options);
  for (size_t i = 0; i < num_binaries; i++)
    free(object_indices[i]);
  free(object_indices);
  if (!CheckLinkResult(executable_index)) {
    char* build_log = ReadLinkLog(executable_index);
    program->CompleteBuild(this, CL_BUILD_ERROR, build_log, NULL, NULL);
  }

  CLProgramBinary* result = ReadExecutable(executable_index);
  void* executable = OpenExecutable(executable_index);
  program->CompleteBuild(this, CL_BUILD_SUCCESS, NULL, result, executable);
  ReadKernelInfo(executable_index, program);
}

void* CPUDevice::AllocMem(CLMem* mem) {
  mem->AllocHostPtr();
  void* m = mem->GetHostPtr();
  if (mem->IsImage()) {
    CPUImageParam* image = (CPUImageParam*)malloc(sizeof(CPUImageParam));
    image->ptr = m;
    image->size = mem->size();
    image->image_format = mem->image_format();
    image->image_desc = mem->image_desc();
    image->image_row_pitch = mem->image_row_pitch();
    image->image_slice_pitch = mem->image_slice_pitch();
    image->image_elem_size = mem->image_element_size();
    image->image_channels = mem->image_channels();
    m = (void*)image;
  }
  return m;
}

void* CPUDevice::AllocHostMem(CLMem *mem)
{
	void *m = mem->GetHostPtr();
	if(m == NULL)
		m = AllocMem(mem);
	return m;
}

void FreeHostMem(CLMem* mem, void* dev_specific)
{
	FreeMem(mem, dev_specific);
}

void CPUDevice::FreeMem(CLMem* mem, void* dev_specific) {
  if (mem->IsImage()) {
    CPUImageParam* image = (CPUImageParam*)dev_specific;
    dev_specific = image->ptr;
    free(image);
  }
  // dev_specific is the host pointer and will be freed by the CLMem::~CLMem()
}

void CPUDevice::FreeExecutable(CLProgram* program, void* executable) {
  dlclose(executable);
}

size_t CPUDevice::SetKernelParam(CPUKernelParam* param, CLKernel* kernel,
                                 cl_uint work_dim, size_t gwo[3],
                                 size_t gws[3], size_t lws[3], size_t nwg[3],
                                 map<cl_uint, CLKernelArg*>* kernel_args) {
  param->launch_id = 0;
  param->work_dim = work_dim;
  memcpy(param->gwo, gwo, sizeof(size_t) * 3);
  memcpy(param->lws, lws, sizeof(size_t) * 3);
  param->orig_grid[0] = gws[0] / lws[0];
  param->orig_grid[1] = gws[1] / lws[1];
  param->orig_grid[2] = gws[2] / lws[2];
  param->kernel_idx = kernel->snucl_index();
  param->file_index = 0;

  size_t args_size = 0;
  for (map<cl_uint, CLKernelArg*>::iterator it = kernel_args->begin();
       it != kernel_args->end();
       ++it) {
    CLMem* mem = it->second->mem;
    CLSampler* sampler = it->second->sampler;
    if (mem != NULL) {
      void* ptr = mem->GetDevSpecific(this);
      memcpy(param->args + args_size, (void*)&ptr, sizeof(ptr));
      args_size += sizeof(ptr);
    } else if (sampler != NULL) {
      int value = 0;
      if (sampler->normalized_coords())
        value |= CLK_NORMALIZED_COORDS_TRUE;
      else
        value |= CLK_NORMALIZED_COORDS_FALSE;

      switch (sampler->addressing_mode()) {
        case CL_ADDRESS_CLAMP_TO_EDGE:
          value |= CLK_ADDRESS_CLAMP_TO_EDGE;
          break;
        case CL_ADDRESS_CLAMP:
          value |= CLK_ADDRESS_CLAMP;
          break;
        case CL_ADDRESS_NONE:
          value |= CLK_ADDRESS_NONE;
          break;
        case CL_ADDRESS_REPEAT:
          value |= CLK_ADDRESS_REPEAT;
          break;
        case CL_ADDRESS_MIRRORED_REPEAT:
          value |= CLK_ADDRESS_MIRRORED_REPEAT;
          break;
        default:
          break;
      }

      switch (sampler->filter_mode()) {
        case CL_FILTER_NEAREST:
          value |= CLK_FILTER_NEAREST;
          break;
        case CL_FILTER_LINEAR:
          value |= CLK_FILTER_LINEAR;
          break;
        default:
          break;
      }

      memcpy(param->args + args_size, (void*)&value, sizeof(value));
      args_size += sizeof(value);
    } else if (it->second->local) {
      uint32_t local_size = (uint32_t)it->second->size;
      memcpy(param->args + args_size, &local_size, sizeof(local_size));
      args_size += sizeof(local_size);
    } else {
      memcpy(param->args + args_size, it->second->value, it->second->size);
      args_size += it->second->size;
    }
  }
  return args_size;
}

#define IDIV(a, b) (((a) % (b)) == 0 ? ((a) / (b)) : (((a) / (b)) + 1))

void CPUDevice::ScheduleStatic(size_t nwg[3], CPUWorkGroupAssignment* wga) {
  size_t wg_total = nwg[0] * nwg[1] * nwg[2];
  size_t wg_chunk = IDIV(wg_total, num_cores_);
  size_t wg_start = 0;
  size_t wg_remain = wg_total;

  if (wg_remain > 0) {
    for (int i = 0; i < num_cores_; i++) {
      if (wg_remain == 0) break;
      if (wg_chunk > wg_remain)
        wg_chunk = wg_remain;

      wga[i].wg_id_start = wg_start;
      wga[i].wg_id_end = wg_start + wg_chunk;
      wg_remain -= wg_chunk;
      wg_start += wg_chunk;
      compute_units_[i]->Launch(&wga[i]);
      if (wg_remain == 0) break;
    }
  }
}

void CPUDevice::ScheduleDynamic(size_t nwg[3], CPUWorkGroupAssignment* wga) {
  size_t wg_total = nwg[0] * nwg[1] * nwg[2];
  size_t wg_chunk;
  size_t wg_start = 0;
  size_t wg_remain = wg_total;

  while (wg_remain > 0) {
    for (int i = 0; i < num_cores_; i++) {
      if (compute_units_[i]->IsIdle()) {
        wg_chunk = IDIV(wg_remain, 2 * num_cores_);
        if (wg_chunk > wg_remain)
          wg_chunk = wg_remain;

        wga[i].wg_id_start = wg_start;
        wga[i].wg_id_end = wg_start + wg_chunk;
        wg_remain -= wg_chunk;
        wg_start += wg_chunk;
        compute_units_[i]->Launch(&wga[i]);
        if (wg_remain == 0) break;
      }
    }
  }
}

void CPUDevice::ReadImageCommon(CLMem* mem_src, size_t src_origin[3],
                                size_t region[3], size_t dst_row_pitch,
                                size_t dst_slice_pitch, void* ptr) {
  cl_mem_object_type image_type = mem_src->type();
  size_t element_size = mem_src->image_element_size();
  size_t src_row_pitch = mem_src->image_row_pitch();
  size_t src_slice_pitch = mem_src->image_slice_pitch();
  size_t dst_origin[3] = {0, 0, 0};
  size_t dimension = 3;
  switch (image_type) {
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      dimension = 3;
      break;
    case CL_MEM_OBJECT_IMAGE2D:
      dimension = 2;
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      dimension = 2;
      dst_row_pitch = dst_slice_pitch;
      break;
    case CL_MEM_OBJECT_IMAGE1D:
      dimension = 1;
      break;
  }
  CopyRegion(mem_src->GetDevSpecific(this), ptr, dimension, src_origin,
             dst_origin, region, element_size, src_row_pitch, src_slice_pitch,
             dst_row_pitch, dst_slice_pitch);
}

void CPUDevice::WriteImageCommon(CLMem* mem_dst, size_t dst_origin[3],
                                 size_t region[3], size_t src_row_pitch,
                                 size_t src_slice_pitch, void* ptr) {
  cl_mem_object_type image_type = mem_dst->type();
  size_t element_size = mem_dst->image_element_size();
  size_t dst_row_pitch = mem_dst->image_row_pitch();
  size_t dst_slice_pitch = mem_dst->image_slice_pitch();
  size_t src_origin[3] = {0, 0, 0};
  size_t dimension = 3;
  switch (image_type) {
    case CL_MEM_OBJECT_IMAGE3D:
    case CL_MEM_OBJECT_IMAGE2D_ARRAY:
      dimension = 3;
      break;
    case CL_MEM_OBJECT_IMAGE2D:
      dimension = 2;
      break;
    case CL_MEM_OBJECT_IMAGE1D_ARRAY:
      dimension = 2;
      dst_row_pitch = dst_slice_pitch;
      break;
    case CL_MEM_OBJECT_IMAGE1D:
      dimension = 1;
      break;
  }
  CopyRegion(ptr, mem_dst->GetDevSpecific(this), dimension, src_origin,
             dst_origin, region, element_size, src_row_pitch, src_slice_pitch,
             dst_row_pitch, dst_slice_pitch);
}

#define SWIZZLE_VECTOR(vec, type)                     \
  if (image_format->image_channel_order == CL_BGRA) { \
    type temp = vec[0];                               \
    vec[0] = vec[2];                                  \
    vec[2] = temp;                                    \
  }

#define SATURATE(value, min, max) \
  ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))

#define NORMALIZE_UNSIGNED(value, max)                         \
  ((value) < 0 ? 0 :                                           \
   ((value) > 1.0f ? (max) : FloatToInt_rte((value) * (max))))

void CPUDevice::PackImagePixel(cl_int* src, void* dst, size_t image_channels,
                               const cl_image_format* image_format) {
  SWIZZLE_VECTOR(src, cl_int);
  switch (image_format->image_channel_data_type) {
    case CL_SIGNED_INT8: {
      cl_char* p = (cl_char*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = (cl_char)SATURATE(src[i], -128, 127);
      break;
    }
    case CL_SIGNED_INT16: {
      cl_short* p = (cl_short*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = (cl_short)SATURATE(src[i], -32768, 32767);
      break;
    }
    case CL_SIGNED_INT32: {
      cl_int* p = (cl_int*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = src[i];
      break;
    }
    default: break;
  }
}

void CPUDevice::PackImagePixel(cl_uint* src, void* dst, size_t image_channels,
                               const cl_image_format* image_format) {
  SWIZZLE_VECTOR(src, cl_uint);
  switch (image_format->image_channel_data_type) {
    case CL_UNSIGNED_INT8: {
      cl_uchar* p = (cl_uchar*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = (cl_uchar)SATURATE(src[i], 0, 255);
      break;
    }
    case CL_UNSIGNED_INT16: {
      cl_ushort* p = (cl_ushort*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = (cl_ushort)SATURATE(src[i], 0, 65535);
      break;
    }
    case CL_UNSIGNED_INT32: {
      cl_uint* p = (cl_uint*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = src[i];
      break;
    }
    default: break;
  }
}

int CPUDevice::FloatToInt_rte(float value) {
  if (value >= -(float)INT_MIN)
    return INT_MAX;
  if (value <= (float)INT_MIN)
    return INT_MIN;

  static const float const_1p23 = ldexp(1.0f, 23);
  static const float const_m1p23 = ldexp(-1.0f, 23);
  if (fabsf(value) < const_1p23) {
    float magic = (value >= 0.0f ? const_1p23 : const_m1p23);
    value += magic;
    value -= magic;
  }
  return (int)value;
}

cl_ushort CPUDevice::FloatToHalf_rte(float value) {
  union {
    uint32_t u;
    float f;
  } fv, ft;
  ft.f = value;
  uint16_t sign = (ft.u >> 16) & 0x8000;
  uint16_t exponent = (ft.u >> 23) & 0xFF;
  uint32_t mantissa = ft.u & 0x007FFFFF;
  fv.u = fv.u & 0x7FFFFFFF;
  if (exponent == 0xFF) {
    if (mantissa == 0)
      return (sign | 0x7C00);
    else
      return (sign | 0x7E00 | (mantissa >> 13));
  }
  if (fv.f >= 0x1.FFEp15f)
    return (sign | 0x7C00);
  if (fv.f <= 0x1.0p-25f)
    return sign;
  if (fv.f < 0x1.8p-24f)
    return (sign | 1);
  if (fv.f < 0x1.0p-14f) {
    fv.f = fv.f * 0x1.0p-125f;
    return (sign | fv.u);
  }
  ft.f = value * 0x1.0p13f;
  ft.u = ft.u & 0x7F800000;
  ft.f = ((fv.f + ft.f) - ft.f) * 0x1.0p-112f;
  return (sign | (uint16_t)(ft.u >> 13));
}


void CPUDevice::PackImagePixel(cl_float* src, void* dst, size_t image_channels,
                               const cl_image_format* image_format) {
  SWIZZLE_VECTOR(src, cl_float);
  switch (image_format->image_channel_data_type) {
    case CL_UNORM_INT8: {
      cl_uchar* p = (cl_uchar*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = (cl_uchar)NORMALIZE_UNSIGNED(src[i], 255.0f);
      break;
    }
    case CL_UNORM_INT16: {
      cl_ushort* p = (cl_ushort*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = (cl_ushort)NORMALIZE_UNSIGNED(src[i], 65535.0f);
      break;
    }
    case CL_HALF_FLOAT: {
      cl_ushort* p = (cl_ushort *)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = FloatToHalf_rte(src[i]);
      break;
    }
    case CL_FLOAT: {
      cl_float* p = (cl_float*)dst;
      for (size_t i = 0; i < image_channels; i++)
        p[i] = src[i];
      break;
    }
    default: break;
  }
}

void CPUDevice::CheckKernelDir() {
  mkdir(SNUCL_BUILD_DIR, 0755);
  mkdir(SNUCL_CPU_BUILD_DIR, 0755);
  mkdir(kernel_dir_, 0755);
}

void CPUDevice::GenerateFileIndex(char file_index[11]) {
  char file_name[128];
  while (true) {
    file_index[0] = rand() % 10 + '0';
    file_index[1] = rand() % 10 + '0';
    file_index[2] = rand() % 10 + '0';
    file_index[3] = rand() % 26 + 'a';
    file_index[4] = rand() % 26 + 'A';
    file_index[5] = rand() % 26 + 'a';
    file_index[6] = rand() % 10 + '0';
    file_index[7] = rand() % 10 + '0';
    file_index[8] = rand() % 26 + 'A';
    file_index[9] = rand() % 10 + '0';
    file_index[10] = '\0';
    sprintf(file_name, "%s/__cl_kernel_%s.so", kernel_dir_, file_index);
    if (access(file_name, F_OK) == 0) continue;
    sprintf(file_name, "%s/__cl_kernel_%s.cpp", kernel_dir_, file_index);
    if (access(file_name, F_OK) == 0) continue;
    sprintf(file_name, "%s/__cl_kernel_%s.cl", kernel_dir_, file_index);
    if (access(file_name, F_OK) == 0) continue;
    break;
  }
}

void CPUDevice::WriteSource(CLProgramSource* source, char file_index[11]) {
  char file_name[128];
  sprintf(file_name, "%s/__cl_kernel_%s.cl", kernel_dir_, file_index);
  FILE* fp = fopen(file_name, "w");
  fprintf(fp, "%s", source->concat_source());
  fclose(fp);
}

void CPUDevice::WriteHeader(CLProgramSource* source) {
  char header_dir[128];
  sprintf(header_dir, "%s/headers", kernel_dir_);
  mkdir(header_dir, 0755);

  string header_name(source->header_name());
  string header_subdir(header_dir);
  size_t s, e;
  s = 0;
  while ((e = header_name.find('/', s)) != header_name.npos) {
    header_subdir.append("/");
    header_subdir.append(header_name, s, e - s);
    mkdir(header_subdir.c_str(), 0755);
    s = e + 1;
  }

  header_name = string(header_dir) + "/" + header_name;
  FILE* fp = fopen(header_name.c_str(), "w");
  fprintf(fp, "%s", source->concat_source());
  fclose(fp);
}

void CPUDevice::WriteCompiledObject(CLProgramBinary* binary,
                                    char file_index[11]) {
  char file_name_obj[128];
  char file_name_cpp[128];
  char file_name_info[128];
  sprintf(file_name_obj, "%s/__cl_kernel_%s.o", kernel_dir_, file_index);
  sprintf(file_name_cpp, "%s/__cl_kernel_%s.cpp", kernel_dir_, file_index);
  sprintf(file_name_info, "%s/__cl_kernel_info_%s.so", kernel_dir_,
          file_index);

  const unsigned char* core_binary = binary->core();
  size_t size = binary->core_size();
  size_t size_prefix = sizeof(size_t) * 3;
  if (size < size_prefix)
    return;

  size_t size_obj, size_cpp, size_info;
  memcpy(&size_obj, core_binary, sizeof(size_t));
  memcpy(&size_cpp, core_binary + sizeof(size_t), sizeof(size_t));
  memcpy(&size_info, core_binary + sizeof(size_t) * 2, sizeof(size_t));
  if (size != size_prefix + size_obj + size_cpp + size_info)
    return;

  FILE* fp_obj = fopen(file_name_obj, "wb");
  FILE* fp_cpp = fopen(file_name_cpp, "w");
  FILE* fp_info = fopen(file_name_info, "wb");
  fwrite(core_binary + size_prefix, 1, size_obj, fp_obj);
  fwrite(core_binary + size_prefix + size_obj, 1, size_cpp, fp_cpp);
  fwrite(core_binary + size_prefix + size_obj + size_cpp, 1, size_info,
         fp_info);
  fclose(fp_obj);
  fclose(fp_cpp);
  fclose(fp_info);
}

void CPUDevice::WriteExecutable(CLProgramBinary* binary, char file_index[11]) {
  char file_name_so[128];
  char file_name_info[128];
  sprintf(file_name_so, "%s/__cl_kernel_%s.so", kernel_dir_, file_index);
  sprintf(file_name_info, "%s/__cl_kernel_info_%s.so", kernel_dir_,
          file_index);

  const unsigned char* core_binary = binary->core();
  size_t size = binary->core_size();
  size_t size_prefix = sizeof(size_t) * 2;
  if (size < size_prefix)
    return;

  size_t size_so, size_info;
  memcpy(&size_so, core_binary, sizeof(size_t));
  memcpy(&size_info, core_binary + sizeof(size_t), sizeof(size_t));
  if (size != size_prefix + size_so + size_info)
    return;

  FILE* fp_so = fopen(file_name_so, "wb");
  FILE* fp_info = fopen(file_name_info, "wb");
  fwrite(core_binary + size_prefix, 1, size_so, fp_so);
  fwrite(core_binary + size_prefix + size_so, 1, size_info, fp_info);
  fclose(fp_so);
  fclose(fp_info);
}

string CPUDevice::AppendHeaderOptions(const char* options, size_t num_headers,
                                      CLProgramSource** headers) {
  std::string new_options;
  if (num_headers > 0) {
    char header_dir[128];
    sprintf(header_dir, "%s/headers", kernel_dir_);
    new_options.append("-I");
    new_options.append(header_dir);
    new_options.append(" ");
    for (size_t i = 0; i < num_headers; i++) {
      string header_name(headers[i]->header_name());
      size_t p = header_name.rfind('/');
      if (p != header_name.npos) {
        new_options.append("-I");
        new_options.append(header_dir);
        new_options.append("/");
        new_options.append(header_name.substr(0, p));
        new_options.append(" ");
      }
    }
  }
  if (options != NULL)
    new_options.append(options);
  return new_options;
}

void CPUDevice::RunCompiler(char file_index[11], const char* options) {
  size_t option_length = 0;
  if (options)
    option_length = strlen(options);
  char* command = (char*)malloc(sizeof(char) * (option_length + 64));
  if (options)
    sprintf(command, "t-cpu-compile.sh %s %s", file_index, options);
  else
    sprintf(command, "t-cpu-compile.sh %s", file_index);
  system(command);
  free(command);
}

void CPUDevice::RunLinker(char output_index[11], size_t num_inputs,
                          char** input_indices, const char* options) {
  size_t option_length = 0;
  if (options)
    option_length = strlen(options);
  char* command =
      (char*)malloc(sizeof(char) * (option_length + num_inputs * 11 + 64));
  sprintf(command, "t-cpu-link.sh %s %d %d", output_index, num_cores_,
          num_inputs);
  for (size_t i = 0; i < num_inputs; i++)
    sprintf(command, "%s %s", command, input_indices[i]);
  if (options)
    sprintf(command, "%s %s", command, options);
  system(command);
  free(command);
}

bool CPUDevice::CheckCompileResult(char file_index[11]) {
  char file_name[128];
  sprintf(file_name, "%s/__cl_kernel_%s.o", kernel_dir_, file_index);
  if (access(file_name, F_OK) != 0) return false;
  sprintf(file_name, "%s/__cl_kernel_%s.cpp", kernel_dir_, file_index);
  if (access(file_name, F_OK) != 0) return false;
  return true;
}

bool CPUDevice::CheckLinkResult(char file_index[11]) {
  char file_name[128];
  sprintf(file_name, "%s/__cl_kernel_%s.so", kernel_dir_, file_index);
  if (access(file_name, F_OK) != 0) return false;
  return true;
}

char* CPUDevice::ReadCompileLog(char file_index[11]) {
  char* log = NULL;
  char file_name[128];
  sprintf(file_name, "%s/__cl_kernel_%s.log", kernel_dir_, file_index);
  FILE* fp = fopen(file_name, "r");
  struct stat file_stat;
  fstat(fileno(fp), &file_stat);
  size_t file_size = file_stat.st_size;
  if (file_size > 0) {
    log = (char*)malloc(file_size + 1);
    file_size = fread(log, 1, file_size, fp);
    log[file_size] = '\0';
  }
  fclose(fp);
  return log;
}

char* CPUDevice::ReadLinkLog(char file_index[11]) {
  char* log = (char*)malloc(sizeof(char) * 11);
  strcpy(log, "Link error");
  return log;
}

CLProgramBinary* CPUDevice::ReadCompiledObject(char file_index[11]) {
  char file_name_obj[128];
  char file_name_cpp[128];
  char file_name_info[128];
  sprintf(file_name_obj, "%s/__cl_kernel_%s.o", kernel_dir_, file_index);
  sprintf(file_name_cpp, "%s/__cl_kernel_%s.cpp", kernel_dir_, file_index);
  sprintf(file_name_info, "%s/__cl_kernel_info_%s.so", kernel_dir_,
          file_index);

  FILE* fp_obj = fopen(file_name_obj, "rb");
  FILE* fp_cpp = fopen(file_name_cpp, "r");
  FILE* fp_info = fopen(file_name_info, "rb");

  struct stat file_stat;
  fstat(fileno(fp_obj), &file_stat);
  size_t size_obj = file_stat.st_size;
  fstat(fileno(fp_cpp), &file_stat);
  size_t size_cpp = file_stat.st_size;
  fstat(fileno(fp_info), &file_stat);
  size_t size_info = file_stat.st_size;
  size_t size_prefix = sizeof(size_t) * 3;
  size_t size = size_prefix + size_obj + size_cpp + size_info;

  unsigned char* core_binary = 
      (unsigned char*)malloc(sizeof(unsigned char) * size);

  memcpy(core_binary, &size_obj, sizeof(size_t));
  memcpy(core_binary + sizeof(size_t), &size_cpp, sizeof(size_t));
  memcpy(core_binary + sizeof(size_t) * 2, &size_info, sizeof(size_t));
  fread((char*)core_binary + size_prefix, 1, size_obj, fp_obj);
  fread((char*)core_binary + size_prefix + size_obj, 1, size_cpp, fp_cpp);
  fread((char*)core_binary + size_prefix + size_obj + size_cpp, 1, size_info,
        fp_info);
  fclose(fp_obj);
  fclose(fp_cpp);
  fclose(fp_info);

  CLProgramBinary* binary =
      new CLProgramBinary(this, CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT,
                          core_binary, size);
  free(core_binary);
  return binary;
}

CLProgramBinary* CPUDevice::ReadExecutable(char file_index[11]) {
  char file_name_so[128];
  char file_name_info[128];
  sprintf(file_name_so, "%s/__cl_kernel_%s.so", kernel_dir_, file_index);
  sprintf(file_name_info, "%s/__cl_kernel_info_%s.so", kernel_dir_,
          file_index);

  FILE* fp_so = fopen(file_name_so, "rb");
  FILE* fp_info = fopen(file_name_info, "rb");

  struct stat file_stat;
  fstat(fileno(fp_so), &file_stat);
  size_t size_so = file_stat.st_size;
  fstat(fileno(fp_info), &file_stat);
  size_t size_info = file_stat.st_size;
  size_t size_prefix = sizeof(size_t) * 2;
  size_t size = size_prefix + size_so + size_info;

  unsigned char* core_binary =
      (unsigned char*)malloc(sizeof(unsigned char) * size);

  memcpy(core_binary, &size_so, sizeof(size_t));
  memcpy(core_binary + sizeof(size_t), &size_info, sizeof(size_t));
  fread((char*)core_binary + size_prefix, 1, size_so, fp_so);
  fread((char*)core_binary + size_prefix + size_so, 1, size_info, fp_info);
  fclose(fp_so);
  fclose(fp_info);

  CLProgramBinary* binary =
      new CLProgramBinary(this, CL_PROGRAM_BINARY_TYPE_EXECUTABLE, core_binary,
                          size);
  free(core_binary);
  return binary;
}

void* CPUDevice::OpenExecutable(char file_index[11]) {
  char file_name[128];
  sprintf(file_name, "%s/__cl_kernel_%s.so", kernel_dir_, file_index);
  return dlopen(file_name, RTLD_NOW);
}

void CPUDevice::ReadKernelInfo(char file_index[11], CLProgram* program) {
  if (!program->BeginRegisterKernelInfo())
    return;

  char file_name[128];
  sprintf(file_name, "%s/__cl_kernel_info_%s.so", kernel_dir_, file_index);
  void* handle = dlopen(file_name, RTLD_NOW);

  unsigned int* _cl_kernel_num =
      (unsigned int*)dlsym(handle, "_cl_kernel_num");
  char** _cl_kernel_names = (char**)dlsym(handle, "_cl_kernel_names");
  unsigned int* _cl_kernel_num_args =
      (unsigned int*)dlsym(handle, "_cl_kernel_num_args");
  char** _cl_kernel_attributes =
      (char**)dlsym(handle, "_cl_kernel_attributes");
  unsigned int (*_cl_kernel_reqd_work_group_size)[3] =
      (unsigned int(*)[3])dlsym(handle, "_cl_kernel_reqd_work_group_size");
  unsigned long long* _cl_kernel_local_mem_size =
      (unsigned long long*)dlsym(handle, "_cl_kernel_local_mem_size");
  unsigned long long* _cl_kernel_private_mem_size =
      (unsigned long long*)dlsym(handle, "_cl_kernel_private_mem_size");
  char** _cl_kernel_arg_address_qualifier =
      (char**)dlsym(handle, "_cl_kernel_arg_address_qualifier");
  char** _cl_kernel_arg_access_qualifier =
      (char**)dlsym(handle, "_cl_kernel_arg_access_qualifier");
  char** _cl_kernel_arg_type_name =
      (char**)dlsym(handle, "_cl_kernel_arg_type_name");
  unsigned int* _cl_kernel_arg_type_qualifier =
      (unsigned int*)dlsym(handle, "_cl_kernel_arg_type_qualifier");
  char** _cl_kernel_arg_name = (char**)dlsym(handle, "_cl_kernel_arg_name");

  unsigned int arg_index = 0;
  for (unsigned int i = 0; i < *_cl_kernel_num; i++) {
    size_t compile_work_group_size[3] = {
        _cl_kernel_reqd_work_group_size[i][0],
        _cl_kernel_reqd_work_group_size[i][1],
        _cl_kernel_reqd_work_group_size[i][2]};
    program->RegisterKernelInfo(_cl_kernel_names[i], _cl_kernel_num_args[i],
                                _cl_kernel_attributes[i], this,
                                max_work_group_size_,
                                compile_work_group_size,
                                _cl_kernel_local_mem_size[i], 64,
                                _cl_kernel_private_mem_size[i], i);
    for (unsigned int j = 0; j < _cl_kernel_num_args[i]; j++) {
      cl_kernel_arg_address_qualifier arg_address_qualifier;
      switch (_cl_kernel_arg_address_qualifier[i][j]) {
        case '0':
          arg_address_qualifier = CL_KERNEL_ARG_ADDRESS_PRIVATE;
          break;
        case '1':
          arg_address_qualifier = CL_KERNEL_ARG_ADDRESS_LOCAL;
          break;
        case '2':
          arg_address_qualifier = CL_KERNEL_ARG_ADDRESS_CONSTANT;
          break;
        case '3':
          arg_address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL;
          break;
        default:
          SNUCL_ERROR("Invalid argument address qualifier");
          break;
      }
      cl_kernel_arg_access_qualifier arg_access_qualifier;
      switch (_cl_kernel_arg_access_qualifier[i][j]) {
        case '0':
          arg_access_qualifier = CL_KERNEL_ARG_ACCESS_NONE;
          break;
        case '1':
          arg_access_qualifier = CL_KERNEL_ARG_ACCESS_READ_WRITE;
          break;
        case '2':
          arg_access_qualifier = CL_KERNEL_ARG_ACCESS_WRITE_ONLY;
          break;
        case '3':
          arg_access_qualifier = CL_KERNEL_ARG_ACCESS_READ_ONLY;
          break;
        default:
          SNUCL_ERROR("Invalid argument access qualifier");
          break;
      }
      program->RegisterKernelArgInfo(_cl_kernel_names[i], j,
                                     arg_address_qualifier,
                                     arg_access_qualifier,
                                     _cl_kernel_arg_type_name[arg_index],
                                     _cl_kernel_arg_type_qualifier[arg_index],
                                     _cl_kernel_arg_name[arg_index]);
      arg_index++;
    }
  }
  dlclose(handle);
  program->FinishRegisterKernelInfo();
}

void CPUDevice::CreateDevices() {
  char buf[8];
  PipeRead("/bin/cat /sys/devices/system/cpu/possible", buf, sizeof(buf) - 1);
  int from, to;
  sscanf(buf, "%d-%d", &from, &to);
  int num_cores = to - from + 1;
  SNUCL_INFO("%d cores are detected", num_cores);

  CPUDevice* device = new CPUDevice(num_cores);
}
