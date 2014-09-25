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

#ifndef __SNUCL__CPU_DEVICE_H
#define __SNUCL__CPU_DEVICE_H

#include <map>
#include <string>
#include <CL/cl.h>
#include "CLDevice.h"
#include "CLKernel.h"
#include "cpu/CPUComputeUnit.h"

class CLCommand;
class CLKernel;
class CLMem;
class CLPlatform;
class CLProgram;
class CLProgramBinary;
class CLProgramSource;

class CPUDevice: public CLDevice {
 public:
  CPUDevice(int num_cores);
  ~CPUDevice();

  virtual void LaunchTestKernel(CLCommand* command, CLKernel* kernel,
                            cl_uint work_dim, size_t gwo[3], size_t gws[3],
                            size_t lws[3], size_t nwg[3],
                            std::map<cl_uint, CLKernelArg*>* kernel_args,
							bool useTrainingKernel) {}
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

  virtual cl_context context() const { return NULL; }
  virtual void FreeExecutable(CLProgram* program, void* executable);

 private:
  size_t SetKernelParam(CPUKernelParam* param, CLKernel* kernel,
                        cl_uint work_dim, size_t gwo[3], size_t gws[3],
                        size_t lws[3], size_t nwg[3],
                        std::map<cl_uint, CLKernelArg*>* kernel_args);
  void ScheduleStatic(size_t nwg[3], CPUWorkGroupAssignment* wga);
  void ScheduleDynamic(size_t nwg[3], CPUWorkGroupAssignment* wga);

  void ReadImageCommon(CLMem* mem_src, size_t src_origin[3], size_t region[3],
                       size_t dst_row_pitch, size_t dst_slice_pitch,
                       void* ptr);
  void WriteImageCommon(CLMem* mem_dst, size_t dst_origin[3], size_t region[3],
                        size_t src_row_pitch, size_t src_slice_pitch,
                        void* ptr);
  void PackImagePixel(cl_int* src, void* dst, size_t image_channels,
                      const cl_image_format* image_format);
  void PackImagePixel(cl_uint* src, void* dst, size_t image_channels,
                      const cl_image_format* image_format);
  int FloatToInt_rte(float value);
  cl_ushort FloatToHalf_rte(float value);
  void PackImagePixel(cl_float* src, void* dst, size_t image_channels,
                      const cl_image_format* image_format);

  void CheckKernelDir();
  void GenerateFileIndex(char file_index[11]);
  void WriteSource(CLProgramSource* source, char file_index[11]);
  void WriteHeader(CLProgramSource* source);
  void WriteCompiledObject(CLProgramBinary* binary, char file_index[11]);
  void WriteExecutable(CLProgramBinary* binary, char file_index[11]);
  std::string AppendHeaderOptions(const char* options, size_t num_headers,
                                  CLProgramSource** headers);
  void RunCompiler(char file_index[11], const char* options);
  void RunLinker(char output_index[11], size_t num_inputs,
                 char** input_indices, const char* options);
  bool CheckCompileResult(char file_index[11]);
  bool CheckLinkResult(char file_index[11]);
  char* ReadCompileLog(char file_index[11]);
  char* ReadLinkLog(char file_index[11]);
  CLProgramBinary* ReadCompiledObject(char file_index[11]);
  CLProgramBinary* ReadExecutable(char file_index[11]);
  void* OpenExecutable(char file_index[11]);
  void ReadKernelInfo(char file_index[11], CLProgram* program);

 private:
  int num_cores_;
  CPUComputeUnit** compute_units_;
  char* kernel_dir_;

 public:
  static void CreateDevices();
};

#endif // __SNUCL__CPU_DEVICE_H
