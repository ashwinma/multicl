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

#include "cluster/ClusterDriver.h"
#include <cstdlib>
#include <map>
#include <vector>
#include <malloc.h>
#include <mpi.h>
#include <stdint.h>
#include <CL/cl.h>
#include "CLCommand.h"
#include "CLContext.h"
#include "CLDevice.h"
#include "CLEvent.h"
#include "CLKernel.h"
#include "CLMem.h"
#include "CLPlatform.h"
#include "CLProgram.h"
#include "CLSampler.h"
#include "cluster/ClusterMessage.h"

using namespace std;

/*
 * Three custom commands: Send, Recv, and Resp.
 */

typedef struct _CustomSendData {
  void* ptr;
  size_t size;
  int node_id;
  unsigned long event_id;
  CLEvent* event;
  map<MPI_Request, CLEvent*>* comm_event_map;
} CustomSendData;

static void CustomSendHandler(void* ptr) {
  CustomSendData* data = (CustomSendData*)ptr;
  MPI_Request request;
  MPI_Isend(data->ptr, (int)data->size, MPI_BYTE, data->node_id,
            CLUSTER_TAG_SEND_BODY(data->event_id), MPI_COMM_WORLD, &request);
  (*(data->comm_event_map))[request] = data->event;
  free(data);
}

typedef struct _CustomRecvData {
  void* ptr;
  size_t size;
  int node_id;
  unsigned long event_id;
  CLEvent* event;
  map<MPI_Request, CLEvent*>* comm_event_map;
} CustomRecvData;

static void CustomRecvHandler(void* ptr) {
  CustomRecvData* data = (CustomRecvData*)ptr;
  MPI_Request request;
  MPI_Irecv(data->ptr, (int)data->size, MPI_BYTE, data->node_id,
            CLUSTER_TAG_RECV_BODY(data->event_id), MPI_COMM_WORLD, &request);
  (*(data->comm_event_map))[request] = data->event;
  free(data);
}

typedef struct _CustomAlltoAllData {
  void* read_ptr;
  void* write_ptr;
  size_t size;
  MPI_Comm MPI_COMM_NODE;
} CustomAlltoAllData;

static void CustomAlltoAllHandler(void* ptr) {
  CustomAlltoAllData* data = (CustomAlltoAllData*)ptr;
  MPI_Alltoall(data->read_ptr, (int)data->size, MPI_BYTE, data->write_ptr,
               (int)data->size, MPI_BYTE, data->MPI_COMM_NODE);
  free(data);
}

typedef struct _CustomBroadcastData {
  void* ptr;
  size_t size;
  int src_node_id;
  MPI_Comm MPI_COMM_NODE;
} CustomBroadcastData;

static void CustomBroadcastHandler(void* ptr) {
  CustomBroadcastData* data = (CustomBroadcastData*)ptr;
  MPI_Bcast(data->ptr, (int)data->size, MPI_BYTE, data->src_node_id - 1,
            data->MPI_COMM_NODE);
  free(data);
}

static void CustomFreeHandler(void* ptr) {
  free(ptr);
}

ClusterDriver singleton_driver;

ClusterDriver::ClusterDriver() {
  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  MPI_Group MPI_GROUP_WORLD, MPI_GROUP_NODE;
  MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
  int* ranks = (int*)malloc(sizeof(int) * size_);
  for (int i = 0; i < size_; i++)
    ranks[i] = i;
  MPI_Group_incl(MPI_GROUP_WORLD, size_ - 1, ranks + 1, &MPI_GROUP_NODE);
  MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_NODE, &MPI_COMM_NODE);
  free(ranks);

  if (rank_ != 0) {
    RunCompute();
    MPI_Finalize();
    exit(0);
  }
}

ClusterDriver::~ClusterDriver() {
  if (rank_ == 0) {
    ClusterSendMessage msg;
    msg.WriteInt(CLUSTER_REQUEST_EXIT);
    for (int i = 1; i < size_; i++)
      msg.Send(i, CLUSTER_TAG_COMMAND);
    MPI_Finalize();
  }
}

int ClusterDriver::RunCompute() {
  platform_ = CLPlatform::GetPlatform();
  platform_->GetDevices(devices_);
  context_ = new CLContext(devices_, 0, NULL);
  running_ = true;

  ClusterRecvMessage msg;
  MPI_Request mpi_request;
  int flag;
  while (running_) {
    mpi_request = msg.Irecv(MPI_ANY_SOURCE, CLUSTER_TAG_COMMAND);
    do {
      CheckWaitEvents();
      MPI_Test(&mpi_request, &flag, MPI_STATUS_IGNORE);
    } while (!flag);
    HandleCommand(msg);
  }

  delete context_;
  return 0;
}

void ClusterDriver::HandleCommand(ClusterRecvMessage& request) {
  int header = request.ReadInt();
  switch (header) {
    case CLUSTER_REQUEST_EXIT:
      running_ = false;
      break;
    case CLUSTER_REQUEST_NODE_INFO:
      HandleNodeInfo(request);
      break;
    case CLUSTER_REQUEST_DEVICE_INFO:
      HandleDeviceInfo(request);
      break;
    case CLUSTER_REQUEST_LAUNCH_KERNEL:
      HandleLaunchKernel(request);
      break;
    case CLUSTER_REQUEST_LAUNCH_NATIVE_KERNEL:
      HandleLaunchNativeKernel(request);
      break;
    case CLUSTER_REQUEST_SEND_BUFFER:
      HandleSendBuffer(request);
      break;
    case CLUSTER_REQUEST_RECV_BUFFER:
      HandleRecvBuffer(request);
      break;
    case CLUSTER_REQUEST_COPY_BUFFER:
      HandleCopyBuffer(request);
      break;
    case CLUSTER_REQUEST_SEND_IMAGE:
      HandleSendImage(request);
      break;
    case CLUSTER_REQUEST_RECV_IMAGE:
      HandleRecvImage(request);
      break;
    case CLUSTER_REQUEST_COPY_IMAGE:
      HandleCopyImage(request);
      break;
    case CLUSTER_REQUEST_COPY_IMAGE_TO_BUFFER:
      HandleCopyImageToBuffer(request);
      break;
    case CLUSTER_REQUEST_COPY_BUFFER_TO_IMAGE:
      HandleCopyBufferToImage(request);
      break;
    case CLUSTER_REQUEST_SEND_BUFFER_RECT:
      HandleSendBufferRect(request);
      break;
    case CLUSTER_REQUEST_RECV_BUFFER_RECT:
      HandleRecvBufferRect(request);
      break;
    case CLUSTER_REQUEST_COPY_BUFFER_RECT:
      HandleCopyBufferRect(request);
      break;
    case CLUSTER_REQUEST_FILL_BUFFER:
      HandleFillBuffer(request);
      break;
    case CLUSTER_REQUEST_FILL_IMAGE:
      HandleFillImage(request);
      break;
    case CLUSTER_REQUEST_BUILD_PROGRAM:
      HandleBuildProgram(request);
      break;
    case CLUSTER_REQUEST_COMPILE_PROGRAM:
      HandleCompileProgram(request);
      break;
    case CLUSTER_REQUEST_LINK_PROGRAM:
      HandleLinkProgram(request);
      break;
    case CLUSTER_REQUEST_ALLTOALL:
      HandleAlltoAll(request);
      break;
    case CLUSTER_REQUEST_BROADCAST:
      HandleBroadcast(request);
      break;
    case CLUSTER_REQUEST_FREE_MEM:
      HandleFreeMem(request);
      break;
    case CLUSTER_REQUEST_FREE_SAMPLER:
      HandleFreeSampler(request);
      break;
    case CLUSTER_REQUEST_FREE_PROGRAM:
      HandleFreeProgram(request);
      break;
    case CLUSTER_REQUEST_FREE_KERNEL:
      HandleFreeKernel(request);
      break;
  }
}

void ClusterDriver::CheckWaitEvents() {
  for (map<MPI_Request, CLEvent*>::iterator it = comm_event_.begin();
       it != comm_event_.end();
       ++it) {
    MPI_Request mpi_request = it->first;
    CLEvent* event = it->second;
    int flag;
    MPI_Test(&mpi_request, &flag, MPI_STATUS_IGNORE);
    if (flag) {
      event->SetStatus(CL_COMPLETE);
      event->Release();
      comm_event_.erase(it);
      break;
    }
  }

  for (map<CLEvent*, CLCommand*>::iterator it = sub_command_.begin();
       it != sub_command_.end();
       ++it) {
    CLEvent* event = it->first;
    CLCommand* command = it->second;
    if (event->IsComplete()) {
      event->Release();
      if (command->type() == CL_COMMAND_CUSTOM ||
          command->type() == CL_COMMAND_NOP) {
        command->Execute();
        command->SetAsComplete();
        delete command;
      } else {
        command->device()->EnqueueReadyQueue(command);
      }
      sub_command_.erase(it);
      break;
    }
  }

  for (map<CLEvent*, unsigned long>::iterator it = sub_response_.begin();
       it != sub_response_.end();
       ++it) {
    CLEvent* event = it->first;
    uint64_t event_id = it->second;
    if (event->IsComplete()) {
      event->Release();
      MPI_Send(&event_id, sizeof(uint64_t), MPI_BYTE, 0,
               CLUSTER_TAG_EVENT_WAIT(event_id), MPI_COMM_WORLD);
      sub_response_.erase(it);
      break;
    }
  }
}

void ClusterDriver::HandleNodeInfo(ClusterRecvMessage& request) {
  ClusterSendMessage response;
  response.WriteULong(devices_.size());
  for (vector<CLDevice*>::iterator it = devices_.begin();
       it != devices_.end();
       ++it) {
    response.WriteULong((*it)->type());
  }
  response.Send(0, CLUSTER_TAG_NODE_INFO);
}

void ClusterDriver::HandleDeviceInfo(ClusterRecvMessage& request) {
  size_t device_id = request.ReadULong();
  CLDevice* device = devices_[device_id];

  ClusterSendMessage response;
  device->SerializeDeviceInfo(response.ptr());
  response.Send(0, CLUSTER_TAG_DEVICE_INFO);
}

void ClusterDriver::HandleLaunchKernel(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  CLKernel* kernel = RecvKernel(request);
  cl_uint num_args = request.ReadUInt();
  for (cl_uint i = 0; i < num_args; i++) {
    int arg_type = request.ReadInt();
    if (arg_type == 0) {
      CLMem* mem = RecvMem(request);
      cl_mem arg = mem->st_obj();
      kernel->SetKernelArg(i, sizeof(cl_mem), &arg);
    } else if (arg_type == 1) {
      CLSampler* sampler = RecvSampler(request);
      cl_sampler arg = sampler->st_obj();
      kernel->SetKernelArg(i, sizeof(cl_sampler), &arg);
    } else if (arg_type == 2) {
      size_t size = request.ReadULong();
      kernel->SetKernelArg(i, size, NULL);
    } else {
      size_t size = request.ReadULong();
      void* value = request.ReadBuffer(size);
      kernel->SetKernelArg(i, size, value);
    }
  }
  cl_uint work_dim = request.ReadUInt();
  size_t gwo[3], gws[3], lws[3], nwg[3];
  gwo[0] = request.ReadULong();
  gwo[1] = request.ReadULong();
  gwo[2] = request.ReadULong();
  gws[0] = request.ReadULong();
  gws[1] = request.ReadULong();
  gws[2] = request.ReadULong();
  lws[0] = request.ReadULong();
  lws[1] = request.ReadULong();
  lws[2] = request.ReadULong();

  CLCommand* command = CLCommand::CreateNDRangeKernel(
      context_, devices_[device_id], NULL, kernel, work_dim, gwo, gws, lws);
  SetSubResponse(command, event_id);
  devices_[device_id]->EnqueueReadyQueue(command);
}

void ClusterDriver::HandleLaunchNativeKernel(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  void (*user_func)(void*) = (void (*)(void*))request.ReadULong();
  size_t cb_args = request.ReadULong();
  void* args = request.ReadBuffer(cb_args);
  cl_uint num_mem_objects = request.ReadUInt();
  cl_mem* mem_list = NULL;
  const void** args_mem_loc = NULL;
  if (num_mem_objects > 0) {
    mem_list = (cl_mem*)malloc(sizeof(cl_mem) * num_mem_objects);
    for (cl_uint i = 0; i < num_mem_objects; i++)
      mem_list[i] = RecvMem(request)->st_obj();
    args_mem_loc = (const void**)malloc(sizeof(const void*) * num_mem_objects);
    for (cl_uint i = 0; i < num_mem_objects; i++) {
      ptrdiff_t offset = request.ReadULong();
      args_mem_loc[i] = (const void*)((size_t)args + offset);
    }
  }

  CLCommand* command = CLCommand::CreateNativeKernel(
      context_, devices_[device_id], NULL, user_func, args, cb_args,
      num_mem_objects, mem_list, args_mem_loc);
  if (num_mem_objects > 0) {
    free(mem_list);
    free(args_mem_loc);
  }
  SetSubResponse(command, event_id);
  devices_[device_id]->EnqueueReadyQueue(command);
}

void ClusterDriver::HandleSendBuffer(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  int node_id = request.ReadInt();
  CLMem* mem = RecvMem(request);
  size_t offset = request.ReadULong();
  size_t size = request.ReadULong();

  void* ptr = (void*)((size_t)mem->GetHostPtr() + offset);
  CLCommand* read_command = CLCommand::CreateReadBuffer(
      context_, devices_[device_id], NULL, mem, offset, size, ptr);
  CLEvent* send_event = new CLEvent(context_);
  CLCommand* send_command = CreateCustomSendCommand(
      device_id, ptr, size, node_id, event_id, send_event);
  SetSubCommand(read_command, send_command);
  // Send does not require a response message
  devices_[device_id]->EnqueueReadyQueue(read_command);
}

void ClusterDriver::HandleRecvBuffer(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  int node_id = request.ReadInt();
  CLMem* mem = RecvMem(request);
  size_t offset = request.ReadULong();
  size_t size = request.ReadULong();

  void* ptr = (void*)((size_t)mem->GetHostPtr() + offset);
  CLCommand* nop_command = CLCommand::CreateNop(
      context_, devices_[device_id], NULL);
  CLEvent* recv_event = new CLEvent(context_);
  CLCommand* recv_command = CreateCustomRecvCommand(
      device_id, ptr, size, node_id, event_id, recv_event);
  CLCommand* write_command = CLCommand::CreateWriteBuffer(
      context_, devices_[device_id], NULL, mem, offset, size, ptr);
  SetSubCommand(nop_command, recv_command);
  SetSubCommand(recv_event, write_command);
  SetSubResponse(write_command, event_id);
  devices_[device_id]->EnqueueReadyQueue(nop_command);
}

void ClusterDriver::HandleCopyBuffer(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t src_device_id = request.ReadULong();
  size_t dst_device_id = request.ReadULong();
  CLMem* src_mem = RecvMem(request);
  CLMem* dst_mem = RecvMem(request);
  size_t src_offset = request.ReadULong();
  size_t dst_offset = request.ReadULong();
  size_t size = request.ReadULong();

  if (src_device_id == dst_device_id) {
    CLCommand* command = CLCommand::CreateCopyBuffer(
        context_, devices_[dst_device_id], NULL, src_mem, dst_mem, src_offset,
        dst_offset, size);
    SetSubResponse(command, event_id);
    devices_[dst_device_id]->EnqueueReadyQueue(command);
  } else {
    void* ptr = (void*)((size_t)src_mem->GetHostPtr() + src_offset);
    CLCommand* read_command = CLCommand::CreateReadBuffer(
        context_, devices_[src_device_id], NULL, src_mem, src_offset, size,
        ptr);
    CLCommand* write_command = CLCommand::CreateWriteBuffer(
        context_, devices_[dst_device_id], NULL, dst_mem, dst_offset, size,
        ptr);
    SetSubCommand(read_command, write_command);
    SetSubResponse(write_command, event_id);
    devices_[src_device_id]->EnqueueReadyQueue(read_command);
  }
}

void ClusterDriver::HandleSendImage(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  int node_id = request.ReadInt();
  CLMem* mem = RecvMem(request);
  size_t origin[3], region[3];
  origin[0] = request.ReadULong();
  origin[1] = request.ReadULong();
  origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();

  size_t row_pitch = region[0] * mem->image_element_size();
  size_t slice_pitch = region[1] * row_pitch;
  size_t size = region[2] * slice_pitch;
  void* ptr = memalign(4096, size);
  CLCommand* read_command = CLCommand::CreateReadImage(
      context_, devices_[device_id], NULL, mem, origin, region, row_pitch,
      slice_pitch, ptr);
  CLEvent* send_event = new CLEvent(context_);
  CLCommand* send_command = CreateCustomSendCommand(
      device_id, ptr, size, node_id, event_id, send_event);
  CLCommand* free_command = CreateCustomFreeCommand(device_id, ptr);
  SetSubCommand(read_command, send_command);
  SetSubCommand(send_event, free_command);
  // Send does not require a response message
  devices_[device_id]->EnqueueReadyQueue(read_command);
}

void ClusterDriver::HandleRecvImage(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  int node_id = request.ReadInt();
  CLMem* mem = RecvMem(request);
  size_t origin[3], region[3];
  origin[0] = request.ReadULong();
  origin[1] = request.ReadULong();
  origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();

  size_t row_pitch = region[0] * mem->image_element_size();
  size_t slice_pitch = region[1] * row_pitch;
  size_t size = region[2] * slice_pitch;
  void* ptr = memalign(4096, size);
  CLCommand* nop_command = CLCommand::CreateNop(
      context_, devices_[device_id], NULL);
  CLEvent* recv_event = new CLEvent(context_);
  CLCommand* recv_command = CreateCustomRecvCommand(
      device_id, ptr, size, node_id, event_id, recv_event);
  CLCommand* write_command = CLCommand::CreateWriteImage(
      context_, devices_[device_id], NULL, mem, origin, region, row_pitch,
      slice_pitch, ptr);
  CLCommand* free_command = CreateCustomFreeCommand(device_id, ptr);
  SetSubCommand(nop_command, recv_command);
  SetSubCommand(recv_event, write_command);
  SetSubCommand(write_command, free_command);
  SetSubResponse(free_command, event_id);
  devices_[device_id]->EnqueueReadyQueue(nop_command);
}

void ClusterDriver::HandleCopyImage(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t src_device_id = request.ReadULong();
  size_t dst_device_id = request.ReadULong();
  CLMem* src_mem = RecvMem(request);
  CLMem* dst_mem = RecvMem(request);
  size_t src_origin[3], dst_origin[3], region[3];
  src_origin[0] = request.ReadULong();
  src_origin[1] = request.ReadULong();
  src_origin[2] = request.ReadULong();
  dst_origin[0] = request.ReadULong();
  dst_origin[1] = request.ReadULong();
  dst_origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();

  if (src_device_id == dst_device_id) {
    CLCommand* command = CLCommand::CreateCopyImage(
        context_, devices_[dst_device_id], NULL, src_mem, dst_mem, src_origin,
        dst_origin, region);
    SetSubResponse(command, event_id);
    devices_[dst_device_id]->EnqueueReadyQueue(command);
  } else {
    size_t row_pitch = region[0] * src_mem->image_element_size();
    size_t slice_pitch = region[1] * row_pitch;
    size_t size = region[2] * slice_pitch;
    void* ptr = memalign(4096, size);
    CLCommand* read_command = CLCommand::CreateReadImage(
        context_, devices_[src_device_id], NULL, src_mem, src_origin, region,
        row_pitch, slice_pitch, ptr);
    CLCommand* write_command = CLCommand::CreateWriteImage(
        context_, devices_[dst_device_id], NULL, dst_mem, dst_origin, region,
        row_pitch, slice_pitch, ptr);
    CLCommand* free_command = CreateCustomFreeCommand(dst_device_id, ptr);
    SetSubCommand(read_command, write_command);
    SetSubCommand(write_command, free_command);
    SetSubResponse(free_command, event_id);
    devices_[src_device_id]->EnqueueReadyQueue(read_command);
  }
}

void ClusterDriver::HandleCopyImageToBuffer(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t src_device_id = request.ReadULong();
  size_t dst_device_id = request.ReadULong();
  CLMem* src_mem = RecvMem(request);
  CLMem* dst_mem = RecvMem(request);
  size_t src_origin[3], region[3];
  src_origin[0] = request.ReadULong();
  src_origin[1] = request.ReadULong();
  src_origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();
  size_t dst_offset = request.ReadULong();

  if (src_device_id == dst_device_id) {
    CLCommand* command = CLCommand::CreateCopyImageToBuffer(
        context_, devices_[dst_device_id], NULL, src_mem, dst_mem, src_origin,
        region, dst_offset);
    SetSubResponse(command, event_id);
    devices_[dst_device_id]->EnqueueReadyQueue(command);
  } else {
    size_t row_pitch = region[0] * src_mem->image_element_size();
    size_t slice_pitch = region[1] * row_pitch;
    size_t size = region[2] * slice_pitch;
    void* ptr = (void*)((size_t)dst_mem->GetHostPtr() + dst_offset);
    CLCommand* read_command = CLCommand::CreateReadImage(
        context_, devices_[src_device_id], NULL, src_mem, src_origin, region,
        row_pitch, slice_pitch, ptr);
    CLCommand* write_command = CLCommand::CreateWriteBuffer(
        context_, devices_[dst_device_id], NULL, dst_mem, dst_offset, size,
        ptr);
    SetSubCommand(read_command, write_command);
    SetSubResponse(write_command, event_id);
    devices_[src_device_id]->EnqueueReadyQueue(read_command);
  }
}

void ClusterDriver::HandleCopyBufferToImage(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t src_device_id = request.ReadULong();
  size_t dst_device_id = request.ReadULong();
  CLMem* src_mem = RecvMem(request);
  CLMem* dst_mem = RecvMem(request);
  size_t src_offset = request.ReadULong();
  size_t dst_origin[3], region[3];
  dst_origin[0] = request.ReadULong();
  dst_origin[1] = request.ReadULong();
  dst_origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();

  if (src_device_id == dst_device_id) {
    CLCommand* command = CLCommand::CreateCopyBufferToImage(
        context_, devices_[dst_device_id], NULL, src_mem, dst_mem, src_offset,
        dst_origin, region);
    SetSubResponse(command, event_id);
    devices_[dst_device_id]->EnqueueReadyQueue(command);
  } else {
    size_t row_pitch = region[0] * dst_mem->image_element_size();
    size_t slice_pitch = region[1] * row_pitch;
    size_t size = region[2] * slice_pitch;
    void* ptr = (void*)((size_t)src_mem->GetHostPtr() + src_offset);
    CLCommand* read_command = CLCommand::CreateReadBuffer(
        context_, devices_[src_device_id], NULL, src_mem, src_offset, size,
        ptr);
    CLCommand* write_command = CLCommand::CreateWriteImage(
         context_, devices_[dst_device_id], NULL, dst_mem, dst_origin, region,
         row_pitch, slice_pitch, ptr);
    SetSubCommand(read_command, write_command);
    SetSubResponse(write_command, event_id);
    devices_[src_device_id]->EnqueueReadyQueue(read_command);
  }
}

void ClusterDriver::HandleSendBufferRect(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  int node_id = request.ReadInt();
  CLMem* mem = RecvMem(request);
  size_t buffer_origin[3], region[3];
  buffer_origin[0] = request.ReadULong();
  buffer_origin[1] = request.ReadULong();
  buffer_origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();
  size_t row_pitch = request.ReadULong();
  size_t slice_pitch = request.ReadULong();

  size_t host_origin[3] = {0, 0, 0};
  size_t size = region[0] * region[1] * region[2];
  void* ptr = memalign(4096, size);
  CLCommand* read_command = CLCommand::CreateReadBufferRect(
      context_, devices_[device_id], NULL, mem, buffer_origin, host_origin,
      region, row_pitch, slice_pitch, 0, 0, ptr);
  CLEvent* send_event = new CLEvent(context_);
  CLCommand* send_command = CreateCustomSendCommand(
      device_id, ptr, size, node_id, event_id, send_event);
  CLCommand* free_command = CreateCustomFreeCommand(device_id, ptr);
  SetSubCommand(read_command, send_command);
  SetSubCommand(send_event, free_command);
  // Send does not require a response message
  devices_[device_id]->EnqueueReadyQueue(read_command);
}

void ClusterDriver::HandleRecvBufferRect(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  int node_id = request.ReadInt();
  CLMem* mem = RecvMem(request);
  size_t buffer_origin[3], region[3];
  buffer_origin[0] = request.ReadULong();
  buffer_origin[1] = request.ReadULong();
  buffer_origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();
  size_t row_pitch = request.ReadULong();
  size_t slice_pitch = request.ReadULong();

  size_t host_origin[3] = {0, 0, 0};
  size_t size = region[0] * region[1] * region[2];
  void* ptr = memalign(4096, size);
  CLCommand* nop_command = CLCommand::CreateNop(
      context_, devices_[device_id], NULL);
  CLEvent* recv_event = new CLEvent(context_);
  CLCommand* recv_command = CreateCustomRecvCommand(
      device_id, ptr, size, node_id, event_id, recv_event);
  CLCommand* write_command = CLCommand::CreateWriteBufferRect(
      context_, devices_[device_id], NULL, mem, buffer_origin, host_origin,
      region, row_pitch, slice_pitch, 0, 0, ptr);
  CLCommand* free_command = CreateCustomFreeCommand(device_id, ptr);
  SetSubCommand(nop_command, recv_command);
  SetSubCommand(recv_event, write_command);
  SetSubCommand(write_command, free_command);
  SetSubResponse(free_command, event_id);
  devices_[device_id]->EnqueueReadyQueue(nop_command);
}

void ClusterDriver::HandleCopyBufferRect(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t src_device_id = request.ReadULong();
  size_t dst_device_id = request.ReadULong();
  CLMem* src_mem = RecvMem(request);
  CLMem* dst_mem = RecvMem(request);
  size_t src_origin[3], dst_origin[3], region[3];
  src_origin[0] = request.ReadULong();
  src_origin[1] = request.ReadULong();
  src_origin[2] = request.ReadULong();
  dst_origin[0] = request.ReadULong();
  dst_origin[1] = request.ReadULong();
  dst_origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();
  size_t src_row_pitch = request.ReadULong();
  size_t src_slice_pitch = request.ReadULong();
  size_t dst_row_pitch = request.ReadULong();
  size_t dst_slice_pitch = request.ReadULong();

  if (src_device_id == dst_device_id) {
    CLCommand* command = CLCommand::CreateCopyBufferRect(
        context_, devices_[dst_device_id], NULL, src_mem, dst_mem, src_origin,
        dst_origin, region, src_row_pitch, src_slice_pitch, dst_row_pitch,
        dst_slice_pitch);
    SetSubResponse(command, event_id);
    devices_[dst_device_id]->EnqueueReadyQueue(command);
  } else {
    size_t host_origin[3] = {0, 0, 0};
    size_t size = region[0] * region[1] * region[2];
    void* ptr = memalign(4096, size);
    CLCommand* read_command = CLCommand::CreateReadBufferRect(
        context_, devices_[src_device_id], NULL, src_mem, src_origin,
        host_origin, region, src_row_pitch, src_slice_pitch, 0, 0, ptr);
    CLCommand* write_command = CLCommand::CreateWriteBufferRect(
        context_, devices_[dst_device_id], NULL, dst_mem, dst_origin,
        host_origin, region, dst_row_pitch, dst_slice_pitch, 0, 0, ptr);
    CLCommand* free_command = CreateCustomFreeCommand(dst_device_id, ptr);
    SetSubCommand(read_command, write_command);
    SetSubCommand(write_command, free_command);
    SetSubResponse(free_command, event_id);
    devices_[src_device_id]->EnqueueReadyQueue(read_command);
  }
}

void ClusterDriver::HandleFillBuffer(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  CLMem* mem = RecvMem(request);
  size_t pattern_size = request.ReadULong();
  void* pattern = request.ReadBuffer(pattern_size);
  size_t offset = request.ReadULong();
  size_t size = request.ReadULong();

  CLCommand* command = CLCommand::CreateFillBuffer(
      context_, devices_[device_id], NULL, mem, pattern, pattern_size, offset,
      size);
  SetSubResponse(command, event_id);
  devices_[device_id]->EnqueueReadyQueue(command);
}

void ClusterDriver::HandleFillImage(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  CLMem* mem = RecvMem(request);
  size_t pattern_size = request.ReadULong();
  void* fill_color = malloc(pattern_size);
  request.ReadBuffer(fill_color, pattern_size);
  size_t origin[3], region[3];
  origin[0] = request.ReadULong();
  origin[1] = request.ReadULong();
  origin[2] = request.ReadULong();
  region[0] = request.ReadULong();
  region[1] = request.ReadULong();
  region[2] = request.ReadULong();

  CLCommand* fill_command = CLCommand::CreateFillImage(
      context_, devices_[device_id], NULL, mem, fill_color, origin, region);
  CLCommand* free_command = CreateCustomFreeCommand(device_id, fill_color);
  SetSubCommand(fill_command, free_command);
  SetSubResponse(free_command, event_id);
  devices_[device_id]->EnqueueReadyQueue(fill_command);
}

void ClusterDriver::HandleBuildProgram(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  ClusterRecvMessage detail_request(false);
  detail_request.Recv(0, CLUSTER_TAG_RECV_BODY(event_id));

  size_t device_id = request.ReadULong();
  CLProgram* program = RecvProgram(request);
  CLProgramSource* source = NULL;
  bool has_source = request.ReadBool();
  if (has_source) {
    char* source_string = detail_request.ReadString();
    source = new CLProgramSource();
    source->AddSource(source_string);
  }
  CLProgramBinary* binary = NULL;
  bool has_binary = request.ReadBool();
  if (has_binary) {
    size_t full_binary_size = detail_request.ReadULong();
    unsigned char* full_binary =
        (unsigned char*)detail_request.ReadBuffer(full_binary_size);
    binary = new CLProgramBinary(devices_[device_id], full_binary,
                                 full_binary_size);
  }
  char* options = NULL;
  bool has_option = request.ReadBool();
  if (has_option)
    options = detail_request.ReadString();

  if (program->EnterBuild(devices_[device_id])) {
    devices_[device_id]->BuildProgram(NULL, program, source, binary, options);
  }
  if (source != NULL)
    delete source;
  if (binary != NULL)
    delete binary;

  ClusterSendMessage response(false);
  SendBuildResult(response, program, devices_[device_id], true);
  response.Send(0, CLUSTER_TAG_SEND_BODY(event_id));
}

void ClusterDriver::HandleCompileProgram(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  ClusterRecvMessage detail_request(false);
  detail_request.Recv(0, CLUSTER_TAG_RECV_BODY(event_id));

  size_t device_id = request.ReadULong();
  CLProgram* program = RecvProgram(request);
  char* source_string = detail_request.ReadString();
  CLProgramSource* source = new CLProgramSource();
  source->AddSource(source_string);
  char* options = NULL;
  bool has_option = request.ReadBool();
  if (has_option)
    options = detail_request.ReadString();
  size_t num_headers = request.ReadULong();
  CLProgramSource** headers = NULL;
  if (num_headers > 0) {
    headers = (CLProgramSource**)malloc(sizeof(CLProgramSource*) *
                                        num_headers);
    for (size_t i = 0; i < num_headers; i++) {
      char* header_string = detail_request.ReadString();
      char* header_name = detail_request.ReadString();
      headers[i] = new CLProgramSource();
      headers[i]->AddSource(header_string);
      headers[i]->SetHeaderName(header_name);
    }
  }

  if (program->EnterBuild(devices_[device_id])) {
    devices_[device_id]->CompileProgram(NULL, program, source, options,
                                        num_headers, headers);
  }
  delete source;
  if (num_headers > 0) {
    for (size_t i = 0; i < num_headers; i++)
      delete headers[i];
    free(headers);
  }

  ClusterSendMessage response(false);
  SendBuildResult(response, program, devices_[device_id], false);
  response.Send(0, CLUSTER_TAG_SEND_BODY(event_id));
}

void ClusterDriver::HandleLinkProgram(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  ClusterRecvMessage detail_request(false);
  detail_request.Recv(0, CLUSTER_TAG_RECV_BODY(event_id));

  size_t device_id = request.ReadULong();
  CLProgram* program = RecvProgram(request);
  size_t num_binaries = request.ReadULong();
  CLProgramBinary** binaries =
      (CLProgramBinary**)malloc(sizeof(CLProgramBinary*) * num_binaries);
  for (size_t i = 0; i < num_binaries; i++) {
    size_t full_binary_size = detail_request.ReadULong();
    unsigned char* full_binary =
        (unsigned char*)detail_request.ReadBuffer(full_binary_size);
    binaries[i] = new CLProgramBinary(devices_[device_id], full_binary,
                                      full_binary_size);
  }
  char* options = NULL;
  bool has_option = request.ReadBool();
  if (has_option)
    options = detail_request.ReadString();

  if (program->EnterBuild(devices_[device_id])) {
    devices_[device_id]->LinkProgram(NULL, program, num_binaries, binaries,
                                     options);
  }
  for (size_t i = 0; i < num_binaries; i++)
    delete binaries[i];
  free(binaries);

  ClusterSendMessage response(false);
  SendBuildResult(response, program, devices_[device_id], true);
  response.Send(0, CLUSTER_TAG_SEND_BODY(event_id));
}

void ClusterDriver::HandleAlltoAll(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  CLMem* src_mem = RecvMem(request);
  CLMem* dst_mem = RecvMem(request);
  size_t src_offset = request.ReadULong();
  size_t dst_offset = request.ReadULong();
  size_t size = request.ReadULong();

  void* read_ptr = (void*)((size_t)src_mem->GetHostPtr() + src_offset);
  void* write_ptr = (void*)((size_t)dst_mem->GetHostPtr() + dst_offset);
  CLCommand* read_command = CLCommand::CreateReadBuffer(
      context_, devices_[device_id], NULL, src_mem, src_offset, size,
      read_ptr);
  CLCommand* alltoall_command = CreateCustomAlltoAllCommand(
      device_id, read_ptr, write_ptr, size);
  CLCommand* write_command = CLCommand::CreateWriteBuffer(
      context_, devices_[device_id], NULL, dst_mem, dst_offset, size,
      write_ptr);
  SetSubCommand(read_command, alltoall_command);
  SetSubCommand(alltoall_command, write_command);
  SetSubResponse(write_command, event_id);
  devices_[device_id]->EnqueueReadyQueue(read_command);
}

void ClusterDriver::HandleBroadcast(ClusterRecvMessage& request) {
  unsigned long event_id = request.ReadULong();
  size_t device_id = request.ReadULong();
  int src_node_id = request.ReadInt();
  size_t src_device_id = 0;
  CLMem* src_mem = NULL;
  if (src_node_id == rank_) {
    src_device_id = request.ReadULong();
    src_mem = RecvMem(request);
  }
  CLMem* dst_mem = RecvMem(request);
  size_t src_offset = request.ReadULong();
  size_t dst_offset = request.ReadULong();
  size_t size = request.ReadULong();

  void* ptr = (void*)((size_t)dst_mem->GetHostPtr() + dst_offset);
  CLCommand* read_command;
  if (src_node_id == rank_) {
    read_command = CLCommand::CreateReadBuffer(
        context_, devices_[src_device_id], NULL, src_mem, src_offset, size,
        ptr);
  } else {
    read_command = CLCommand::CreateNop(
        context_, devices_[device_id], NULL);
  }
  CLCommand* broadcast_command = CreateCustomBroadcastCommand(
      device_id, ptr, size, src_node_id);
  CLCommand* write_command = CLCommand::CreateWriteBuffer(
      context_, devices_[device_id], NULL, dst_mem, dst_offset, size, ptr);
  SetSubCommand(read_command, broadcast_command);
  SetSubCommand(broadcast_command, write_command);
  SetSubResponse(write_command, event_id);
  read_command->device()->EnqueueReadyQueue(read_command);
}

void ClusterDriver::HandleFreeMem(ClusterRecvMessage& request) {
  unsigned long mem_id = request.ReadULong();
  if (mems_.count(mem_id) > 0) {
    CLMem* mem = mems_[mem_id];
    mems_.erase(mem_id);
    mem->Release();
  }
}

void ClusterDriver::HandleFreeSampler(ClusterRecvMessage& request) {
  unsigned long sampler_id = request.ReadULong();
  if (samplers_.count(sampler_id) > 0) {
    CLSampler* sampler = samplers_[sampler_id];
    samplers_.erase(sampler_id);
    sampler->Release();
  }
}

void ClusterDriver::HandleFreeProgram(ClusterRecvMessage& request) {
  unsigned long program_id = request.ReadULong();
  if (programs_.count(program_id) > 0) {
    CLProgram* program = programs_[program_id];
    programs_.erase(program_id);
    program->Release();
  }
}

void ClusterDriver::HandleFreeKernel(ClusterRecvMessage& request) {
  unsigned long kernel_id = request.ReadULong();
  if (kernels_.count(kernel_id) > 0) {
    CLKernel* kernel = kernels_[kernel_id];
    kernels_.erase(kernel_id);
    kernel->Release();
  }
}

CLMem* ClusterDriver::RecvMem(ClusterRecvMessage& request) {
  unsigned long mem_id = request.ReadULong();
  if (mem_id > 0)
    return mems_[mem_id];

  mem_id = request.ReadULong();
  cl_mem_flags flags = request.ReadULong();
  size_t size = request.ReadULong();
  bool is_image = request.ReadBool();
  bool is_sub_buffer = request.ReadBool();
  flags &= (CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE);
  flags |= CL_MEM_ALLOC_HOST_PTR;

  CLMem* mem;
  if (is_image) {
    cl_image_format format;
    cl_image_desc desc;
    format.image_channel_order = request.ReadUInt();
    format.image_channel_data_type = request.ReadUInt();
    desc.image_type = request.ReadUInt();
    desc.image_width = request.ReadULong();
    desc.image_height = request.ReadULong();
    desc.image_depth = request.ReadULong();
    desc.image_array_size = request.ReadULong();
    desc.image_row_pitch = request.ReadULong();
    desc.image_slice_pitch = request.ReadULong();
    desc.num_mip_levels = request.ReadUInt();
    desc.num_samples = request.ReadUInt();
    mem = CLMem::CreateImage(context_, flags, &format, &desc, NULL, NULL);
  } else if (is_sub_buffer) {
    CLMem* parent = RecvMem(request);
    size_t offset = request.ReadULong();
    mem = CLMem::CreateSubBuffer(parent, flags, offset, size, NULL);
  } else {
    mem = CLMem::CreateBuffer(context_, flags, size, NULL, NULL);
  }
  mems_[mem_id] = mem;
  return mem;
}

CLSampler* ClusterDriver::RecvSampler(ClusterRecvMessage& request) {
  unsigned long sampler_id = request.ReadULong();
  if (sampler_id > 0)
    return samplers_[sampler_id];

  sampler_id = request.ReadULong();
  cl_bool normalized_coords = request.ReadUInt();
  cl_addressing_mode addressing_mode = request.ReadUInt();
  cl_filter_mode filter_mode = request.ReadUInt();
  CLSampler* sampler = new CLSampler(context_, normalized_coords,
                                     addressing_mode, filter_mode);
  samplers_[sampler_id] = sampler;
  return sampler;
}

CLProgram* ClusterDriver::RecvProgram(ClusterRecvMessage& request) {
  unsigned long program_id = request.ReadULong();
  if (programs_.count(program_id) > 0)
    return programs_[program_id];

  cl_uint num_devices = (cl_uint)devices_.size();
  cl_device_id* device_list = (cl_device_id*)malloc(sizeof(cl_device_id) *
                                                    num_devices);
  for (cl_uint i = 0; i < num_devices; i++)
    device_list[i] = devices_[i]->st_obj();
  cl_int err;
  CLProgram* program = CLProgram::CreateProgramWithNothing(context_,
                                                           num_devices,
                                                           device_list, &err);
  programs_[program_id] = program;
  return program;
}

CLKernel* ClusterDriver::RecvKernel(ClusterRecvMessage& request) {
  unsigned long kernel_id = request.ReadULong();
  if (kernel_id > 0)
    return kernels_[kernel_id];

  unsigned long program_id = request.ReadULong();
  kernel_id = request.ReadULong();
  char* kernel_name = request.ReadString();
  CLProgram* program = programs_[program_id];
  CLKernel* kernel = program->CreateKernel(kernel_name, NULL);
  kernels_[kernel_id] = kernel;
  return kernel;
}

void ClusterDriver::SendBuildResult(ClusterSendMessage& response,
                                    CLProgram* program, CLDevice* device,
                                    bool has_kernel_info) {
  cl_build_status build_status = program->GetBuildStatus(device);
  response.WriteInt(build_status);
  const char* build_log = program->GetBuildLog(device);
  if (build_log == NULL)
    response.WriteString("");
  else
    response.WriteString(build_log);
  if (build_status == CL_BUILD_SUCCESS) {
    CLProgramBinary* binary = program->GetBinary(device);
    response.WriteULong(binary->full_size());
    response.WriteBuffer(binary->full(), binary->full_size());
    if (has_kernel_info) {
      size_t serialized_info_size;
      void* serialized_info =
          program->SerializeKernelInfo(device, &serialized_info_size);
      if (serialized_info != NULL) {
        response.WriteULong(serialized_info_size);
        response.WriteBuffer(serialized_info, serialized_info_size);
        free(serialized_info);
      } else {
        response.WriteULong(0);
      }
    }
  }
}

CLCommand* ClusterDriver::CreateCustomSendCommand(size_t device_id, void* ptr,
                                                  size_t size, int node_id,
                                                  unsigned long event_id,
                                                  CLEvent* event) {
  CustomSendData* data = (CustomSendData*)malloc(sizeof(CustomSendData));
  data->ptr = ptr;
  data->size = size;
  data->node_id = node_id;
  data->event_id = event_id;
  data->event = event;
  data->comm_event_map = &comm_event_;
  return CLCommand::CreateCustom(context_, devices_[device_id], NULL,
                                 CustomSendHandler, data);
}

CLCommand* ClusterDriver::CreateCustomRecvCommand(size_t device_id, void* ptr,
                                                  size_t size, int node_id,
                                                  unsigned long event_id,
                                                  CLEvent* event) {
  CustomRecvData* data = (CustomRecvData*)malloc(sizeof(CustomRecvData));
  data->ptr = ptr;
  data->size = size;
  data->node_id = node_id;
  data->event_id = event_id;
  data->event = event;
  data->comm_event_map = &comm_event_;
  return CLCommand::CreateCustom(context_, devices_[device_id], NULL,
                                 CustomRecvHandler, data);
}

CLCommand* ClusterDriver::CreateCustomAlltoAllCommand(size_t device_id,
                                                      void* read_ptr,
                                                      void* write_ptr,
                                                      size_t size) {
  CustomAlltoAllData* data =
      (CustomAlltoAllData*)malloc(sizeof(CustomAlltoAllData));
  data->read_ptr = read_ptr;
  data->write_ptr = write_ptr;
  data->size = size;
  data->MPI_COMM_NODE = MPI_COMM_NODE;
  return CLCommand::CreateCustom(context_, devices_[device_id], NULL,
                                 CustomAlltoAllHandler, data);
}

CLCommand* ClusterDriver::CreateCustomBroadcastCommand(size_t device_id,
                                                       void* ptr, size_t size,
                                                       int src_node_id) {
  CustomBroadcastData* data =
      (CustomBroadcastData*)malloc(sizeof(CustomBroadcastData));
  data->ptr = ptr;
  data->size = size;
  data->src_node_id = src_node_id;
  data->MPI_COMM_NODE = MPI_COMM_NODE;
  return CLCommand::CreateCustom(context_, devices_[device_id], NULL,
                                 CustomBroadcastHandler, data);
}

CLCommand* ClusterDriver::CreateCustomFreeCommand(size_t device_id,
                                                  void* ptr) {
  return CLCommand::CreateCustom(context_, devices_[device_id], NULL,
                                 CustomFreeHandler, ptr);
}

void ClusterDriver::SetSubCommand(CLCommand* prior, CLCommand* posterior) {
  sub_command_[prior->ExportEvent()] = posterior;
}

void ClusterDriver::SetSubCommand(CLEvent* prior, CLCommand* posterior) {
  prior->Retain();
  sub_command_[prior] = posterior;
}

void ClusterDriver::SetSubResponse(CLCommand* prior, unsigned long event_id) {
  sub_response_[prior->ExportEvent()] = event_id;
}

void ClusterDriver::SetSubResponse(CLEvent* prior, unsigned long event_id) {
  prior->Retain();
  sub_response_[prior] = event_id;
}
