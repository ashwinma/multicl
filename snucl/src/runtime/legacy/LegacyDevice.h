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

#ifndef __SNUCL__LEGACY_DEVICE_H
#define __SNUCL__LEGACY_DEVICE_H

#include <map>
#include <vector>
#include <CL/cl.h>
#include "CLDevice.h"
#include "CLKernel.h"
#include "ICD.h"
#include "RealTimer.h"

class CLCommand;
class CLKernel;
class CLMem;
class CLPlatform;
class CLProgram;
class CLProgramBinary;
class CLProgramSource;

class LegacyDevice: public CLDevice {
 public:
  LegacyDevice(void* library, struct _cl_icd_dispatch* dispatch,
						   cl_context context,
               cl_platform_id platform_id, cl_device_id device_id);
  ~LegacyDevice();

  virtual void LaunchTestKernel(CLCommand* command, CLKernel* kernel,
                            cl_uint work_dim, size_t gwo[3], size_t gws[3],
                            size_t lws[3], size_t nwg[3],
                            std::map<cl_uint, CLKernelArg*>* kernel_args, 
							bool useTrainingKernel);
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
							   cl_mem mem_src_dev_specific, cl_mem mem_dst_dev_specific, 
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

  virtual void *AllocHostMem(CLMem *mem);
  virtual void* AllocMem(CLMem* mem);
  virtual void FreeMem(CLMem* mem, void* dev_specific);
  virtual void FreeHostMem(CLMem* mem, void* dev_specific);
  virtual void* AllocSampler(CLSampler* sampler);
  virtual void FreeSampler(CLSampler* sampler, void* dev_specific);

  virtual void FreeExecutable(CLProgram* program, void* executable);
  virtual void* AllocKernel(CLKernel* kernel);
  virtual void* AllocTrainingKernel(CLKernel* kernel);
  virtual void FreeKernel(CLKernel* kernel, void* dev_specific);

  virtual cl_context context() const { return context_; }
 private:
  cl_program CreateProgram(CLProgramSource* source);
  cl_program CreateTrainingProgram(CLProgramSource* source);
  cl_program CreateProgram(CLProgramBinary* binary);
  CLProgramBinary* ReadBinary(cl_program program);
  char* ReadBuildLog(cl_program program);
  void ReadKernelInfo(cl_program legacy_program, CLProgram* program);

  void* library_;
  struct _cl_icd_dispatch* dispatch_;
  cl_platform_id platform_id_;
  cl_device_id device_id_;
  cl_context context_;
  cl_command_queue kernel_queue_;
  cl_command_queue mem_queue_;
  cl_command_queue misc_queue_;
  int version_;

 public:
  static void CreateDevices();

 private:
  static void IcdVendorAdd(const char* library_name);
  static void IcdPlatformAdd(const char* library_name,
                             cl_platform_id platform);
  static char* IcdGetLibraryName(const char* entry_name);
  static cl_platform_id* IcdGetPlatforms(void* library,
                                         cl_uint* num_platforms);
  static cl_device_id* IcdGetDevices(cl_platform_id platform,
                                     cl_uint* num_devices);
};

#endif // __SNUCL__LEGACY_DEVICE_H
