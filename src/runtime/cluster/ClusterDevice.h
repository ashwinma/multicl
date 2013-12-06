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

#ifndef __SNUCL__CLUSTER_DEVICE_H
#define __SNUCL__CLUSTER_DEVICE_H

#include <map>
#include <vector>
#include <CL/cl.h>
#include "CLDevice.h"
#include "CLKernel.h"
#include "ICD.h"

class CLCommand;
class CLKernel;
class CLMem;
class CLPlatform;
class CLProgram;
class CLProgramBinary;
class CLProgramSource;
class ClusterSendMessage;

class ClusterDevice: public CLDevice {
 public:
  ClusterDevice(int node_id, size_t device_id, cl_device_type type);
  ~ClusterDevice();

  virtual void LaunchKernel(CLCommand* command, CLKernel* kernel,
                            cl_uint work_dim, size_t gwo[3], size_t gws[3],
                            size_t lws[3], size_t nwg[3],
                            std::map<cl_uint, CLKernelArg*>* kernel_args);
  virtual void LaunchNativeKernel(CLCommand* command, void (*user_func)(void*),
                                  void* native_args, size_t size,
                                  cl_uint num_mem_objects, CLMem** mem_list,
                                  ptrdiff_t* mem_offsets);
  virtual void ReadBuffer(CLCommand* command, CLMem* mem_src, size_t off_src,
                          size_t size, void* ptr);
  virtual void WriteBuffer(CLCommand* command, CLMem* mem_dst, size_t off_dst,
                           size_t size, void* ptr);
  virtual void CopyBuffer(CLCommand* command, CLMem* mem_src, CLMem* mem_dst,
                          size_t off_src, size_t off_dst, size_t size);
  virtual void ReadImage(CLCommand* command, CLMem* mem_src,
                         size_t src_origin[3], size_t region[3],
                         size_t dst_row_pitch, size_t dst_slice_pitch,
                         void* ptr);
  virtual void WriteImage(CLCommand* command, CLMem* mem_dst,
                          size_t dst_origin[3], size_t region[3],
                          size_t src_row_pitch, size_t src_slice_pitch,
                          void* ptr);
  virtual void CopyImage(CLCommand* command, CLMem* mem_src, CLMem* mem_dst,
                         size_t src_origin[3], size_t dst_origin[3],
                         size_t region[3]);
  virtual void CopyImageToBuffer(CLCommand* command, CLMem* mem_src,
                                 CLMem* mem_dst, size_t src_origin[3],
                                 size_t region[3], size_t off_dst);
  virtual void CopyBufferToImage(CLCommand* command, CLMem* mem_src,
                                 CLMem* mem_dst, size_t off_src,
                                 size_t dst_origin[3], size_t region[3]);
  virtual void ReadBufferRect(CLCommand* command, CLMem* mem_src,
                              size_t src_origin[3], size_t dst_origin[3],
                              size_t region[3], size_t src_row_pitch,
                              size_t src_slice_pitch, size_t dst_row_pitch,
                              size_t dst_slice_pitch, void* ptr);
  virtual void WriteBufferRect(CLCommand* command, CLMem* mem_dst,
                               size_t src_origin[3], size_t dst_origin[3],
                               size_t region[3], size_t src_row_pitch,
                               size_t src_slice_pitch, size_t dst_row_pitch,
                               size_t dst_slice_pitch, void* ptr);
  virtual void CopyBufferRect(CLCommand* command, CLMem* mem_src,
                              CLMem* mem_dst, size_t src_origin[3],
                              size_t dst_origin[3], size_t region[3],
                              size_t src_row_pitch, size_t src_slice_pitch,
                              size_t dst_row_pitch, size_t dst_slice_pitch);
  virtual void FillBuffer(CLCommand* command, CLMem* mem_dst, void* pattern,
                          size_t pattern_size, size_t off_dst, size_t size);
  virtual void FillImage(CLCommand* command, CLMem* mem_dst, void* fill_color,
                         size_t dst_origin[3], size_t region[3]);

  virtual void BuildProgram(CLCommand* command, CLProgram* program,
                            CLProgramSource* source, CLProgramBinary* binary,
                            const char* options);
  virtual void CompileProgram(CLCommand* command, CLProgram* program,
                              CLProgramSource* source, const char* options,
                              size_t num_headers, CLProgramSource** headers);
  virtual void LinkProgram(CLCommand* command, CLProgram* program,
                           size_t num_binaries, CLProgramBinary** binaries,
                           const char* options);

  virtual void AlltoAllBuffer(CLCommand* command, CLMem* mem_src,
                              CLMem* mem_dst, size_t off_src, size_t off_dst,
                              size_t size);
  virtual void BroadcastBuffer(CLCommand* command, CLMem* mem_src,
                               CLMem* mem_dst, size_t off_src, size_t off_dst,
                               size_t size);

  virtual bool IsComplete(CLCommand* command);

  virtual void* AllocMem(CLMem* mem);
  virtual void FreeMem(CLMem* mem, void* dev_specific);
  virtual void* AllocSampler(CLSampler* sampler);
  virtual void FreeSampler(CLSampler* sampler, void* dev_specific);

  virtual void FreeExecutable(CLProgram* program, void* executable);
  virtual void* AllocKernel(CLKernel* kernel);
  virtual void FreeKernel(CLKernel* kernel, void* dev_specific);

 private:
  void SendMem(ClusterSendMessage& request, CLMem* mem);
  void SendSampler(ClusterSendMessage& request, CLSampler* sampler);
  void SendProgram(ClusterSendMessage& request, CLProgram* program);
  void SendKernel(ClusterSendMessage& request, CLKernel* kernel);

  void WaitingPlainResponse(CLCommand* command, unsigned long event_id);
  void WaitingData(CLCommand* command, unsigned long event_id, void* ptr,
                   size_t size);
  void WaitingDataAndUnpack(CLCommand* command, unsigned long event_id,
                            void* ptr, size_t dst_origin[3], size_t region[3],
                            size_t element_size, size_t dst_row_pitch,
                            size_t dst_slice_pitch);
  void WaitingBuildResult(CLCommand* command, unsigned long event_id,
                          CLProgram* program, bool has_executable);

  size_t device_id_;
  int version_;

  std::map<CLCommand*, int> waiting_response_;
  std::map<CLCommand*, void (*)(int, int, void*)> finalize_function_;
  std::map<CLCommand*, void*> finalize_data_;

 public:
  static void CreateDevices();
};

#endif // __SNUCL__CLUSTER_DEVICE_H
