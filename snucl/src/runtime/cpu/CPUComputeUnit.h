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

#ifndef __SNUCL__CPU_COMPUTE_UNIT_H
#define __SNUCL__CPU_COMPUTE_UNIT_H

#include <deque>
#include <pthread.h>
#include <semaphore.h>
#include <CL/cl.h>

class CLDevice;

typedef struct _CPUKernelParam {
  char args[1024];
  unsigned int launch_id;
  unsigned int work_dim;
  size_t gwo[3];
  size_t lws[3];
  size_t orig_grid[3];
  size_t wg_id_begin[3];
  size_t wg_id_size[3];
  size_t wg_id_start;
  size_t wg_id_end;
  int kernel_idx;
  int file_index;
} CPUKernelParam;

typedef struct _CPUImageParam {
  void* ptr;
  size_t size;
  cl_image_format image_format;
  cl_image_desc image_desc;
  size_t image_row_pitch;
  size_t image_slice_pitch;
  size_t image_elem_size;
  size_t image_channels;
} CPUImageParam;

typedef struct _CPUWorkGroupAssignment {
  void* handle;
  CPUKernelParam* param;
  size_t args_size;
  size_t wg_id_start;
  size_t wg_id_end;
} CPUWorkGroupAssignment;

class CPUComputeUnit {
 public:
  CPUComputeUnit(CLDevice* device, int id);
  ~CPUComputeUnit();

  bool IsIdle() const;
  void Launch(CPUWorkGroupAssignment* wga);
  void Sync();

 private:
  void Run();

  CLDevice* device_;
  int id_;

  pthread_t thread_;
  bool thread_running_;

  std::deque<CPUWorkGroupAssignment*> work_queue_;
  bool work_to_do_;

  pthread_mutex_t mutex_work_queue_;
  pthread_cond_t cond_work_queue_;

 public:
  static void* ThreadFunc(void* argp);
};

#endif // __SNUCL__CPU_COMPUTE_UNIT_H
