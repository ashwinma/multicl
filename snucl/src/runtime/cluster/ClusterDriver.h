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

#ifndef __SNUCL__CLUSTER_DRIVER_H
#define __SNUCL__CLUSTER_DRIVER_H

#include <map>
#include <vector>
#include <mpi.h>

class CLCommand;
class CLContext;
class CLDevice;
class CLEvent;
class CLKernel;
class CLMem;
class CLPlatform;
class CLProgram;
class CLSampler;
class ClusterRecvMessage;
class ClusterSendMessage;

class ClusterDriver {
 public:
  ClusterDriver();
  ~ClusterDriver();

 private:
  int RunCompute();

  void HandleCommand(ClusterRecvMessage& msg);
  void CheckWaitEvents();

  void HandleNodeInfo(ClusterRecvMessage& request);
  void HandleDeviceInfo(ClusterRecvMessage& request);
  void HandleLaunchKernel(ClusterRecvMessage& request);
  void HandleLaunchNativeKernel(ClusterRecvMessage& request);
  void HandleSendBuffer(ClusterRecvMessage& request);
  void HandleRecvBuffer(ClusterRecvMessage& request);
  void HandleCopyBuffer(ClusterRecvMessage& request);
  void HandleSendImage(ClusterRecvMessage& request);
  void HandleRecvImage(ClusterRecvMessage& request);
  void HandleCopyImage(ClusterRecvMessage& request);
  void HandleCopyImageToBuffer(ClusterRecvMessage& request);
  void HandleCopyBufferToImage(ClusterRecvMessage& request);
  void HandleSendBufferRect(ClusterRecvMessage& request);
  void HandleRecvBufferRect(ClusterRecvMessage& request);
  void HandleCopyBufferRect(ClusterRecvMessage& request);
  void HandleFillBuffer(ClusterRecvMessage& request);
  void HandleFillImage(ClusterRecvMessage& request);
  void HandleBuildProgram(ClusterRecvMessage& request);
  void HandleCompileProgram(ClusterRecvMessage& request);
  void HandleLinkProgram(ClusterRecvMessage& request);
  void HandleAlltoAll(ClusterRecvMessage& request);
  void HandleBroadcast(ClusterRecvMessage& request);
  void HandleFreeMem(ClusterRecvMessage& request);
  void HandleFreeSampler(ClusterRecvMessage& request);
  void HandleFreeProgram(ClusterRecvMessage& request);
  void HandleFreeKernel(ClusterRecvMessage& request);

  CLMem* RecvMem(ClusterRecvMessage& request);
  CLSampler* RecvSampler(ClusterRecvMessage& request);
  CLProgram* RecvProgram(ClusterRecvMessage& request);
  CLKernel* RecvKernel(ClusterRecvMessage& request);

  void SendBuildResult(ClusterSendMessage& response, CLProgram* program,
                       CLDevice* device, bool has_kernel_info);

  CLCommand* CreateCustomSendCommand(size_t device_id, void* ptr, size_t size,
                                     int node_id, unsigned long event_id,
                                     CLEvent* event);
  CLCommand* CreateCustomRecvCommand(size_t device_id, void* ptr, size_t size,
                                    int node_id, unsigned long event_id,
                                    CLEvent* event);
  CLCommand* CreateCustomAlltoAllCommand(size_t device_id, void* read_ptr,
                                         void* write_ptr, size_t size);
  CLCommand* CreateCustomBroadcastCommand(size_t device_id, void* ptr,
                                          size_t size, int src_node_id);
  CLCommand* CreateCustomFreeCommand(size_t device_id, void* ptr);

  void SetSubCommand(CLCommand* prior, CLCommand* posterior);
  void SetSubCommand(CLEvent* prior, CLCommand* posterior);
  void SetSubResponse(CLCommand* prior, unsigned long event_id);
  void SetSubResponse(CLEvent* prior, unsigned long event_id);

  int rank_;
  int size_;
  MPI_Comm MPI_COMM_NODE;

  CLPlatform* platform_;
  std::vector<CLDevice*> devices_;
  CLContext* context_;

  std::map<unsigned long, CLMem*> mems_;
  std::map<unsigned long, CLSampler*> samplers_;
  std::map<unsigned long, CLProgram*> programs_;
  std::map<unsigned long, CLKernel*> kernels_;

  std::map<MPI_Request, CLEvent*> comm_event_;
  std::map<CLEvent*, CLCommand*> sub_command_;
  std::map<CLEvent*, unsigned long> sub_response_;

  bool running_;
};

#endif // __SNUCL__CLUSTER_DRIVER_H
