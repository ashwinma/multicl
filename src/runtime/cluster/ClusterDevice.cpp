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

#include "cluster/ClusterDevice.h"
#include <cstring>
#include <map>
#include <vector>
#include <malloc.h>
#include <mpi.h>
#include <stdint.h>
#include <CL/cl.h>
#include "CLCommand.h"
#include "CLDevice.h"
#include "CLEvent.h"
#include "CLKernel.h"
#include "CLMem.h"
#include "CLPlatform.h"
#include "CLProgram.h"
#include "CLSampler.h"
#include "Utils.h"
#include "cluster/ClusterMessage.h"

using namespace std;

#define CLUSTER_VERSION_1_0 10
#define CLUSTER_VERSION_1_1 11
#define CLUSTER_VERSION_1_2 12

#define CHECK_VERSION(required)                        \
  if (available_ == CL_FALSE || version_ < required) { \
    command->SetError(CL_DEVICE_NOT_AVAILABLE);        \
  }

ClusterDevice::ClusterDevice(int node_id, size_t device_id,
                             cl_device_type type)
    : CLDevice(node_id) {
  device_id_ = device_id;
  type_ = type;

  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_DEVICE_INFO);
  request.WriteULong(device_id);
  request.Send(node_id, CLUSTER_TAG_COMMAND);

  ClusterRecvMessage response;
  response.Recv(node_id, CLUSTER_TAG_DEVICE_INFO);
  DeserializeDeviceInfo(response.ptr());

  if (strcmp(profile_, "FULL_PROFILE") != 0)
    available_ = CL_FALSE;

  if (strncmp(device_version_, "OpenCL 1.2", 10) == 0)
    version_ = CLUSTER_VERSION_1_2;
  else if (strncmp(device_version_, "OpenCL 1.1", 10) == 0)
    version_ = CLUSTER_VERSION_1_1;
  else if (strncmp(device_version_, "OpenCL 1.0", 10) == 0)
    version_ = CLUSTER_VERSION_1_0;
  else
    available_ = CL_FALSE;
}

ClusterDevice::~ClusterDevice() {
  free((char*)built_in_kernels_);
  free((char*)name_);
  free((char*)vendor_);
  free((char*)driver_version_);
  free((char*)profile_);
  free((char*)device_version_);
  free((char*)opencl_c_version_);
  free((char*)device_extensions_);
}

void ClusterDevice::LaunchKernel(CLCommand* command, CLKernel* kernel,
                                 cl_uint work_dim, size_t gwo[3],
                                 size_t gws[3], size_t lws[3], size_t nwg[3],
                                 map<cl_uint, CLKernelArg*>* kernel_args) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_LAUNCH_KERNEL);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  SendKernel(request, kernel);
  request.WriteUInt(kernel_args->size());
  for (cl_uint i = 0; i < kernel_args->size(); i++) {
    CLKernelArg* arg = (*kernel_args)[i];
    if (arg->mem != NULL) {
      request.WriteInt(0);
      SendMem(request, arg->mem);
    } else if (arg->sampler != NULL) {
      request.WriteInt(1);
      SendSampler(request, arg->sampler);
    } else if (arg->local) {
      request.WriteInt(2);
      request.WriteULong(arg->size);
    } else {
      request.WriteInt(3);
      request.WriteULong(arg->size);
      request.WriteBuffer(arg->value, arg->size);
    }
  }
  request.WriteUInt(work_dim);
  request.WriteULong(gwo[0]);
  request.WriteULong(gwo[1]);
  request.WriteULong(gwo[2]);
  request.WriteULong(gws[0]);
  request.WriteULong(gws[1]);
  request.WriteULong(gws[2]);
  request.WriteULong(lws[0]);
  request.WriteULong(lws[1]);
  request.WriteULong(lws[2]);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::LaunchNativeKernel(CLCommand* command,
                                       void (*user_func)(void*),
                                       void* native_args, size_t size,
                                       cl_uint num_mem_objects,
                                       CLMem** mem_list,
                                       ptrdiff_t* mem_offsets) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_LAUNCH_NATIVE_KERNEL);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  request.WriteULong((size_t)user_func);
  request.WriteULong(size);
  request.WriteBuffer(native_args, size);
  request.WriteUInt(num_mem_objects);
  for (cl_uint i = 0; i < num_mem_objects; i++)
    SendMem(request, mem_list[i]);
  for (cl_uint i = 0; i < num_mem_objects; i++)
    request.WriteULong(mem_offsets[i]);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::ReadBuffer(CLCommand* command, CLMem* mem_src,
                               size_t off_src, size_t size, void* ptr) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_SEND_BUFFER);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  if (command->destination_node() >= 0)
    request.WriteInt(command->destination_node());
  else
    request.WriteInt(0);
  SendMem(request, mem_src);
  request.WriteULong(off_src);
  request.WriteULong(size);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  if (command->destination_node() < 0) {
    WaitingData(command, event_id, ptr, size);
  }
}

void ClusterDevice::WriteBuffer(CLCommand* command, CLMem* mem_dst,
                                size_t off_dst, size_t size, void* ptr) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_RECV_BUFFER);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  if (command->source_node() >= 0)
    request.WriteInt(command->source_node());
  else
    request.WriteInt(0);
  SendMem(request, mem_dst);
  request.WriteULong(off_dst);
  request.WriteULong(size);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  if (command->source_node() < 0) {
    MPI_Send(ptr, size, MPI_BYTE, node_id_, CLUSTER_TAG_SEND_BODY(event_id),
             MPI_COMM_WORLD);
  }
  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::CopyBuffer(CLCommand* command, CLMem* mem_src,
                               CLMem* mem_dst, size_t off_src, size_t off_dst,
                               size_t size) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_COPY_BUFFER);
  request.WriteULong(event_id);
  ClusterDevice* source_device =
      dynamic_cast<ClusterDevice*>(command->source_device());
  if (source_device != NULL)
    request.WriteULong(source_device->device_id_);
  else
    request.WriteULong(device_id_);
  ClusterDevice* destination_device =
      dynamic_cast<ClusterDevice*>(command->destination_device());
  if (destination_device != NULL)
    request.WriteULong(destination_device->device_id_);
  else
    request.WriteULong(device_id_);
  SendMem(request, mem_src);
  SendMem(request, mem_dst);
  request.WriteULong(off_src);
  request.WriteULong(off_dst);
  request.WriteULong(size);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::ReadImage(CLCommand* command, CLMem* mem_src,
                              size_t src_origin[3], size_t region[3],
                              size_t dst_row_pitch, size_t dst_slice_pitch,
                              void* ptr) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_SEND_IMAGE);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  if (command->destination_node() >= 0)
    request.WriteInt(command->destination_node());
  else
    request.WriteInt(0);
  SendMem(request, mem_src);
  request.WriteULong(src_origin[0]);
  request.WriteULong(src_origin[1]);
  request.WriteULong(src_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  if (command->destination_node() < 0) {
    size_t zero_origin[3] = {0, 0, 0};
    if (mem_src->type() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
      dst_row_pitch = dst_slice_pitch;
      dst_slice_pitch = 0;
    }
    WaitingDataAndUnpack(command, event_id, ptr, zero_origin, region,
                         mem_src->image_element_size(), dst_row_pitch,
                         dst_slice_pitch);
  }
}

void ClusterDevice::WriteImage(CLCommand* command, CLMem* mem_dst,
                               size_t dst_origin[3], size_t region[3],
                               size_t src_row_pitch, size_t src_slice_pitch,
                               void* ptr) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_RECV_IMAGE);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  if (command->source_node() >= 0)
    request.WriteInt(command->source_node());
  else
    request.WriteInt(0);
  SendMem(request, mem_dst);
  request.WriteULong(dst_origin[0]);
  request.WriteULong(dst_origin[1]);
  request.WriteULong(dst_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  if (command->source_node() < 0) {
    size_t size = mem_dst->GetRegionSize(region);
    void* packed_ptr = memalign(4096, size);
    size_t zero_origin[3] = {0, 0, 0};
    if (mem_dst->type() == CL_MEM_OBJECT_IMAGE1D_ARRAY) {
      src_row_pitch = src_slice_pitch;
      src_slice_pitch = 0;
    }
    CopyRegion(ptr, packed_ptr, 3, zero_origin, dst_origin, region,
               mem_dst->image_element_size(), src_row_pitch, src_slice_pitch,
               0, 0);
    MPI_Send(packed_ptr, size, MPI_BYTE, node_id_,
             CLUSTER_TAG_SEND_BODY(event_id), MPI_COMM_WORLD);
    free(packed_ptr);
  }
  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::CopyImage(CLCommand* command, CLMem* mem_src,
                              CLMem* mem_dst, size_t src_origin[3],
                              size_t dst_origin[3], size_t region[3]) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_COPY_IMAGE);
  request.WriteULong(event_id);
  ClusterDevice* source_device =
      dynamic_cast<ClusterDevice*>(command->source_device());
  if (source_device != NULL)
    request.WriteULong(source_device->device_id_);
  else
    request.WriteULong(device_id_);
  ClusterDevice* destination_device =
      dynamic_cast<ClusterDevice*>(command->destination_device());
  if (destination_device != NULL)
    request.WriteULong(destination_device->device_id_);
  else
    request.WriteULong(device_id_);
  SendMem(request, mem_src);
  SendMem(request, mem_dst);
  request.WriteULong(src_origin[0]);
  request.WriteULong(src_origin[1]);
  request.WriteULong(src_origin[2]);
  request.WriteULong(dst_origin[0]);
  request.WriteULong(dst_origin[1]);
  request.WriteULong(dst_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::CopyImageToBuffer(CLCommand* command, CLMem* mem_src,
                                      CLMem* mem_dst, size_t src_origin[3],
                                      size_t region[3], size_t off_dst) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_COPY_IMAGE_TO_BUFFER);
  request.WriteULong(event_id);
  ClusterDevice* source_device =
      dynamic_cast<ClusterDevice*>(command->source_device());
  if (source_device != NULL)
    request.WriteULong(source_device->device_id_);
  else
    request.WriteULong(device_id_);
  ClusterDevice* destination_device =
      dynamic_cast<ClusterDevice*>(command->destination_device());
  if (destination_device != NULL)
    request.WriteULong(destination_device->device_id_);
  else
    request.WriteULong(device_id_);
  SendMem(request, mem_src);
  SendMem(request, mem_dst);
  request.WriteULong(src_origin[0]);
  request.WriteULong(src_origin[1]);
  request.WriteULong(src_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.WriteULong(off_dst);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::CopyBufferToImage(CLCommand* command, CLMem* mem_src,
                                      CLMem* mem_dst, size_t off_src,
                                      size_t dst_origin[3], size_t region[3]) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_COPY_BUFFER_TO_IMAGE);
  request.WriteULong(event_id);
  ClusterDevice* source_device =
      dynamic_cast<ClusterDevice*>(command->source_device());
  if (source_device != NULL)
    request.WriteULong(source_device->device_id_);
  else
    request.WriteULong(device_id_);
  ClusterDevice* destination_device =
      dynamic_cast<ClusterDevice*>(command->destination_device());
  if (destination_device != NULL)
    request.WriteULong(destination_device->device_id_);
  else
    request.WriteULong(device_id_);
  SendMem(request, mem_src);
  SendMem(request, mem_dst);
  request.WriteULong(off_src);
  request.WriteULong(dst_origin[0]);
  request.WriteULong(dst_origin[1]);
  request.WriteULong(dst_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::ReadBufferRect(CLCommand* command, CLMem* mem_src,
                                   size_t src_origin[3], size_t dst_origin[3],
                                   size_t region[3], size_t src_row_pitch,
                                   size_t src_slice_pitch,
                                   size_t dst_row_pitch,
                                   size_t dst_slice_pitch, void* ptr) {
  CHECK_VERSION(CLUSTER_VERSION_1_1);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_SEND_BUFFER_RECT);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  if (command->destination_node() >= 0)
    request.WriteInt(command->destination_node());
  else
    request.WriteInt(0);
  SendMem(request, mem_src);
  request.WriteULong(src_origin[0]);
  request.WriteULong(src_origin[1]);
  request.WriteULong(src_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.WriteULong(src_row_pitch);
  request.WriteULong(src_slice_pitch);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  if (command->destination_node() < 0) {
    WaitingDataAndUnpack(command, event_id, ptr, dst_origin, region, 1,
                         dst_row_pitch, dst_slice_pitch);
  }
}

void ClusterDevice::WriteBufferRect(CLCommand* command, CLMem* mem_dst,
                                    size_t src_origin[3], size_t dst_origin[3],
                                    size_t region[3], size_t src_row_pitch,
                                    size_t src_slice_pitch,
                                    size_t dst_row_pitch,
                                    size_t dst_slice_pitch, void* ptr) {
  CHECK_VERSION(CLUSTER_VERSION_1_1);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_RECV_BUFFER_RECT);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  if (command->source_node() >= 0)
    request.WriteInt(command->source_node());
  else
    request.WriteInt(0);
  SendMem(request, mem_dst);
  request.WriteULong(dst_origin[0]);
  request.WriteULong(dst_origin[1]);
  request.WriteULong(dst_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.WriteULong(dst_row_pitch);
  request.WriteULong(dst_slice_pitch);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  if (command->source_node() < 0) {
    size_t size = mem_dst->GetRegionSize(region);
    void* packed_ptr = memalign(4096, size);
    size_t zero_origin[3] = {0, 0, 0};
    CopyRegion(ptr, packed_ptr, 3, src_origin, zero_origin, region, 1,
               src_row_pitch, src_slice_pitch, 0, 0);
    MPI_Send(packed_ptr, size, MPI_BYTE, node_id_,
             CLUSTER_TAG_SEND_BODY(event_id), MPI_COMM_WORLD);
    free(packed_ptr);
  }
  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::CopyBufferRect(CLCommand* command, CLMem* mem_src,
                                   CLMem* mem_dst, size_t src_origin[3],
                                   size_t dst_origin[3], size_t region[3],
                                   size_t src_row_pitch,
                                   size_t src_slice_pitch,
                                   size_t dst_row_pitch,
                                   size_t dst_slice_pitch) {
  CHECK_VERSION(CLUSTER_VERSION_1_1);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_COPY_BUFFER_RECT);
  request.WriteULong(event_id);
  ClusterDevice* source_device =
      dynamic_cast<ClusterDevice*>(command->source_device());
  if (source_device != NULL)
    request.WriteULong(source_device->device_id_);
  else
    request.WriteULong(device_id_);
  ClusterDevice* destination_device =
      dynamic_cast<ClusterDevice*>(command->destination_device());
  if (destination_device != NULL)
    request.WriteULong(destination_device->device_id_);
  else
    request.WriteULong(device_id_);
  SendMem(request, mem_src);
  SendMem(request, mem_dst);
  request.WriteULong(src_origin[0]);
  request.WriteULong(src_origin[1]);
  request.WriteULong(src_origin[2]);
  request.WriteULong(dst_origin[0]);
  request.WriteULong(dst_origin[1]);
  request.WriteULong(dst_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.WriteULong(src_row_pitch);
  request.WriteULong(src_slice_pitch);
  request.WriteULong(dst_row_pitch);
  request.WriteULong(dst_slice_pitch);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::FillBuffer(CLCommand* command, CLMem* mem_dst,
                               void* pattern, size_t pattern_size,
                               size_t off_dst, size_t size) {
  CHECK_VERSION(CLUSTER_VERSION_1_2);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_FILL_BUFFER);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  SendMem(request, mem_dst);
  request.WriteULong(pattern_size);
  request.WriteBuffer(pattern, pattern_size);
  request.WriteULong(off_dst);
  request.WriteULong(size);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::FillImage(CLCommand* command, CLMem* mem_dst,
                              void* fill_color, size_t dst_origin[3],
                              size_t region[3]) {
  CHECK_VERSION(CLUSTER_VERSION_1_2);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_FILL_IMAGE);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  SendMem(request, mem_dst);
  size_t pattern_size;
  switch (mem_dst->image_format().image_channel_data_type) {
    case CL_SIGNED_INT8:
    case CL_SIGNED_INT16:
    case CL_SIGNED_INT32:
      pattern_size = mem_dst->image_channels() * sizeof(cl_int);
      break;
    case CL_UNSIGNED_INT8:
    case CL_UNSIGNED_INT16:
    case CL_UNSIGNED_INT32:
      pattern_size = mem_dst->image_channels() * sizeof(cl_uint);
      break;
    default:
      pattern_size = mem_dst->image_channels() * sizeof(cl_float);
      break;
  }
  request.WriteULong(pattern_size);
  request.WriteBuffer(fill_color, pattern_size);
  request.WriteULong(dst_origin[0]);
  request.WriteULong(dst_origin[1]);
  request.WriteULong(dst_origin[2]);
  request.WriteULong(region[0]);
  request.WriteULong(region[1]);
  request.WriteULong(region[2]);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::BuildProgram(CLCommand* command, CLProgram* program,
                                 CLProgramSource* source,
                                 CLProgramBinary* binary,
                                 const char* options) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request, detail_request(false);
  request.WriteInt(CLUSTER_REQUEST_BUILD_PROGRAM);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  SendProgram(request, program);
  bool has_source = (source != NULL);
  request.WriteBool(has_source);
  if (has_source)
    detail_request.WriteString(source->concat_source());
  bool has_binary = (binary != NULL);
  request.WriteBool(has_binary);
  if (has_binary) {
    detail_request.WriteULong(binary->full_size());
    detail_request.WriteBuffer(binary->full(), binary->full_size());
  }
  bool has_option = (options != NULL);
  request.WriteBool(has_option);
  if (has_option)
    detail_request.WriteString(options);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);
  detail_request.Send(node_id_, CLUSTER_TAG_SEND_BODY(event_id));

  WaitingBuildResult(command, event_id, program, true);
}

void ClusterDevice::CompileProgram(CLCommand* command, CLProgram* program,
                                   CLProgramSource* source,
                                   const char* options, size_t num_headers,
                                   CLProgramSource** headers) {
  CHECK_VERSION(CLUSTER_VERSION_1_2);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request, detail_request(false);
  request.WriteInt(CLUSTER_REQUEST_COMPILE_PROGRAM);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  SendProgram(request, program);
  detail_request.WriteString(source->concat_source());
  bool has_option = (options != NULL);
  request.WriteBool(has_option);
  if (has_option)
    detail_request.WriteString(options);
  request.WriteULong(num_headers);
  for (size_t i = 0; i < num_headers; i++) {
    detail_request.WriteString(headers[i]->concat_source());
    detail_request.WriteString(headers[i]->header_name());
  }
  request.Send(node_id_, CLUSTER_TAG_COMMAND);
  detail_request.Send(node_id_, CLUSTER_TAG_SEND_BODY(event_id));

  WaitingBuildResult(command, event_id, program, false);
}

void ClusterDevice::LinkProgram(CLCommand* command, CLProgram* program,
                                size_t num_binaries,
                                CLProgramBinary** binaries,
                                const char* options) {
  CHECK_VERSION(CLUSTER_VERSION_1_2);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request, detail_request(false);
  request.WriteInt(CLUSTER_REQUEST_LINK_PROGRAM);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  SendProgram(request, program);
  request.WriteULong(num_binaries);
  for (size_t i = 0; i < num_binaries; i++) {
    detail_request.WriteULong(binaries[i]->full_size());
    detail_request.WriteBuffer(binaries[i]->full(), binaries[i]->full_size());
  }
  bool has_option = (options != NULL);
  request.WriteBool(has_option);
  if (has_option)
    request.WriteString(options);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);
  detail_request.Send(node_id_, CLUSTER_TAG_SEND_BODY(event_id));

  WaitingBuildResult(command, event_id, program, true);
}

void ClusterDevice::AlltoAllBuffer(CLCommand* command, CLMem* mem_src,
                                   CLMem* mem_dst, size_t off_src,
                                   size_t off_dst, size_t size) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_ALLTOALL);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  SendMem(request, mem_src);
  SendMem(request, mem_dst);
  request.WriteULong(off_src);
  request.WriteULong(off_dst);
  request.WriteULong(size);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

void ClusterDevice::BroadcastBuffer(CLCommand* command, CLMem* mem_src,
                                    CLMem* mem_dst, size_t off_src,
                                    size_t off_dst, size_t size) {
  CHECK_VERSION(CLUSTER_VERSION_1_0);

  unsigned long event_id = command->event_id();
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_ALLTOALL);
  request.WriteULong(event_id);
  request.WriteULong(device_id_);
  ClusterDevice* source_device =
      dynamic_cast<ClusterDevice*>(command->source_device());
  request.WriteInt(source_device->node_id_);
  if (source_device->node_id_ == node_id_) {
    request.WriteULong(source_device->device_id_);
    SendMem(request, mem_src);
  }
  SendMem(request, mem_dst);
  request.WriteULong(off_src);
  request.WriteULong(off_dst);
  request.WriteULong(size);
  request.Send(node_id_, CLUSTER_TAG_COMMAND);

  WaitingPlainResponse(command, event_id);
}

bool ClusterDevice::IsComplete(CLCommand* command) {
  if (waiting_response_.count(command) > 0) {
    int tag = waiting_response_[command];
    int flag;
    MPI_Iprobe(node_id_, tag, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      void (*func)(int, int, void*) = finalize_function_[command];
      void* data = finalize_data_[command];
      if (func != NULL)
        func(node_id_, tag, data);
      if (data != NULL)
        free(data);
      waiting_response_.erase(command);
      finalize_function_.erase(command);
      finalize_data_.erase(command);
      return true;
    } else {
      return false;
    }
  } else {
    return true;
  }
}

void* ClusterDevice::AllocMem(CLMem* mem) {
  return (void*)1;
}

void ClusterDevice::FreeMem(CLMem* mem, void* dev_specific) {
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_FREE_MEM);
  request.WriteULong(mem->id());
  request.Send(node_id_, CLUSTER_TAG_COMMAND);
}

void* ClusterDevice::AllocSampler(CLSampler* sampler) {
  return (void*)1;
}

void ClusterDevice::FreeSampler(CLSampler* sampler, void* dev_specific) {
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_FREE_SAMPLER);
  request.WriteULong(sampler->id());
  request.Send(node_id_, CLUSTER_TAG_COMMAND);
}

void ClusterDevice::FreeExecutable(CLProgram* program, void* executable) {
}

void* ClusterDevice::AllocKernel(CLKernel* kernel) {
  return (void*)1;
}

void ClusterDevice::FreeKernel(CLKernel* kernel, void* dev_specific) {
  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_FREE_KERNEL);
  request.WriteULong(kernel->id());
  request.Send(node_id_, CLUSTER_TAG_COMMAND);
}

void ClusterDevice::SendMem(ClusterSendMessage& request, CLMem* mem) {
  if (mem->HasDevSpecific(this)) {
    request.WriteULong(mem->id());
  } else {
    request.WriteULong(0);
    request.WriteULong(mem->id());
    request.WriteULong(mem->flags());
    request.WriteULong(mem->size());
    request.WriteBool(mem->IsImage());
    request.WriteBool(mem->IsSubBuffer());
    if (mem->IsImage()) {
      cl_image_format format = mem->image_format();
      cl_image_desc desc = mem->image_desc();
      request.WriteUInt(format.image_channel_order);
      request.WriteUInt(format.image_channel_data_type);
      request.WriteUInt(desc.image_type);
      request.WriteULong(desc.image_width);
      request.WriteULong(desc.image_height);
      request.WriteULong(desc.image_depth);
      request.WriteULong(desc.image_array_size);
      request.WriteULong(desc.image_row_pitch);
      request.WriteULong(desc.image_slice_pitch);
      request.WriteUInt(desc.num_mip_levels);
      request.WriteUInt(desc.num_samples);
    } else if (mem->IsSubBuffer()) {
      SendMem(request, mem->parent());
      request.WriteULong(mem->offset());
    }
    mem->GetDevSpecific(this);
  }
}

void ClusterDevice::SendSampler(ClusterSendMessage& request,
                                CLSampler* sampler) {
  if (sampler->HasDevSpecific(this)) {
    request.WriteULong(sampler->id());
  } else {
    request.WriteULong(0);
    request.WriteULong(sampler->id());
    request.WriteUInt(sampler->normalized_coords());
    request.WriteUInt(sampler->addressing_mode());
    request.WriteUInt(sampler->filter_mode());
    sampler->GetDevSpecific(this);
  }
}

void ClusterDevice::SendProgram(ClusterSendMessage& request,
                                CLProgram* program) {
  request.WriteULong(program->id());
}

void ClusterDevice::SendKernel(ClusterSendMessage& request, CLKernel* kernel) {
  if (kernel->HasDevSpecific(this)) {
    request.WriteULong(kernel->id());
  } else {
    request.WriteULong(0);
    request.WriteULong(kernel->program()->id());
    request.WriteULong(kernel->id());
    request.WriteString(kernel->name());
    kernel->GetDevSpecific(this);
  }
}

static void RecvPlainResponse(int node, int tag, void* ptr) {
  uint64_t value;
  MPI_Recv(&value, sizeof(uint64_t), MPI_BYTE, node, tag, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
}

typedef struct _RecvDataInfo {
  void* ptr;
  size_t size;
} RecvDataInfo;

static void RecvData(int node, int tag, void* ptr) {
  RecvDataInfo* info = (RecvDataInfo*)ptr;
  MPI_Recv(info->ptr, (int)info->size, MPI_BYTE, node, tag, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
}

typedef struct _RecvDataAndUnpackInfo {
  void* ptr;
  size_t dst_origin[3];
  size_t region[3];
  size_t element_size;
  size_t dst_row_pitch;
  size_t dst_slice_pitch;
} RecvDataAndUnpackInfo;

static void RecvDataAndUnpack(int node, int tag, void* ptr) {
  RecvDataAndUnpackInfo* info = (RecvDataAndUnpackInfo*)ptr;
  size_t size = info->region[0] * info->region[1] * info->region[2] *
                info->element_size;
  void* packed_ptr = memalign(4096, size);
  MPI_Recv(packed_ptr, (int)size, MPI_BYTE, node, tag, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  size_t zero_origin[3] = {0, 0, 0};
  CopyRegion(packed_ptr, info->ptr, 3, zero_origin, info->dst_origin,
             info->region, info->element_size, 0, 0, info->dst_row_pitch,
             info->dst_slice_pitch);
  free(packed_ptr);
}

typedef struct _RecvBuildResultInfo {
  CLProgram* program;
  CLDevice* device;
  bool has_executable;
} RecvBuildResultInfo;

static void RecvBuildResult(int node, int tag, void* ptr) {
  RecvBuildResultInfo* info = (RecvBuildResultInfo*)ptr;
  CLProgram* program = info->program;
  CLDevice* device = info->device;

  ClusterRecvMessage response(false);
  response.Recv(node, tag);

  cl_build_status build_status = response.ReadInt();
  char* build_log = response.ReadString();
  char* copied_build_log = (char*)malloc(sizeof(char) *
                                         (strlen(build_log) + 1));
  strcpy(copied_build_log, build_log);
  if (build_status == CL_BUILD_SUCCESS) {
    size_t full_binary_size = response.ReadULong();
    unsigned char* full_binary =
        (unsigned char*)response.ReadBuffer(full_binary_size);
    CLProgramBinary* binary = new CLProgramBinary(device, full_binary,
                                                  full_binary_size);
    program->CompleteBuild(device, build_status, copied_build_log, binary,
                           (info->has_executable ? (void*)1 : NULL));
    if (info->has_executable) {
      size_t serialized_info_size = response.ReadULong();
      if (serialized_info_size > 0) {
        void* serialized_info = response.ReadBuffer(serialized_info_size);
        program->DeserializeKernelInfo(serialized_info, device);
      }
    }
  } else {
    program->CompleteBuild(device, build_status, copied_build_log, NULL, NULL);
  }
}

void ClusterDevice::WaitingPlainResponse(CLCommand* command,
                                         unsigned long event_id) {
  waiting_response_[command] = CLUSTER_TAG_EVENT_WAIT(event_id);
  finalize_function_[command] = RecvPlainResponse;
  finalize_data_[command] = NULL;
}

void ClusterDevice::WaitingData(CLCommand* command, unsigned long event_id,
                                void* ptr, size_t size) {
  waiting_response_[command] = CLUSTER_TAG_RECV_BODY(event_id);
  RecvDataInfo* info = (RecvDataInfo*)malloc(sizeof(RecvDataInfo));
  info->ptr = ptr;
  info->size = size;
  finalize_function_[command] = RecvData;
  finalize_data_[command] = info;
}

void ClusterDevice::WaitingDataAndUnpack(CLCommand* command,
                                         unsigned long event_id, void* ptr,
                                         size_t dst_origin[3],
                                         size_t region[3], size_t element_size,
                                         size_t dst_row_pitch,
                                         size_t dst_slice_pitch) {
  waiting_response_[command] = CLUSTER_TAG_RECV_BODY(event_id);
  RecvDataAndUnpackInfo* info =
      (RecvDataAndUnpackInfo*)malloc(sizeof(RecvDataAndUnpackInfo));
  info->ptr = ptr;
  memcpy(info->dst_origin, dst_origin, sizeof(size_t) * 3);
  memcpy(info->region, region, sizeof(size_t) * 3);
  info->element_size = element_size;
  info->dst_row_pitch = dst_row_pitch;
  info->dst_slice_pitch = dst_slice_pitch;
  finalize_function_[command] = RecvDataAndUnpack;
  finalize_data_[command] = info;
}

void ClusterDevice::WaitingBuildResult(CLCommand* command,
                                       unsigned long event_id,
                                       CLProgram* program,
                                       bool has_executable) {
  waiting_response_[command] = CLUSTER_TAG_RECV_BODY(event_id);
  RecvBuildResultInfo* info =
      (RecvBuildResultInfo*)malloc(sizeof(RecvBuildResultInfo));
  info->program = program;
  info->device = this;
  info->has_executable = has_executable;
  finalize_function_[command] = RecvBuildResult;
  finalize_data_[command] = info;
}

void ClusterDevice::CreateDevices() {
  int num_nodes;
  int total_devices = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  ClusterSendMessage request;
  request.WriteInt(CLUSTER_REQUEST_NODE_INFO);
  for (int node_id = 1; node_id < num_nodes; node_id++)
    request.Send(node_id, CLUSTER_TAG_COMMAND);

  ClusterRecvMessage response;
  for (int node_id = 1; node_id < num_nodes; node_id++) {
    response.Recv(node_id, CLUSTER_TAG_NODE_INFO);
    size_t num_devices = response.ReadULong();
    total_devices += num_devices;
    //SNUCL_INFO("[Host] Devices in Node %d : %d\n", node_id, num_devices);
    for (size_t device_id = 0; device_id < num_devices; device_id++) {
      cl_device_type type = response.ReadULong();
      ClusterDevice* device = new ClusterDevice(node_id, device_id, type);
    }
  }
  SNUCL_INFO("[Host] Total devices : %u\n", total_devices);
}
