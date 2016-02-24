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

#include "cpu/CPUComputeUnit.h"
#include <cstring>
#include <deque>
#include <dlfcn.h>
#include <pthread.h>
#include <semaphore.h>
#include <CL/cl.h>

using namespace std;

CPUComputeUnit::CPUComputeUnit(CLDevice* device, int id) {
  device_ = device;
  id_ = id;

  pthread_mutex_init(&mutex_work_queue_, NULL);
  pthread_cond_init(&cond_work_queue_, NULL);

  thread_running_ = true;
  pthread_create(&thread_, NULL, &CPUComputeUnit::ThreadFunc, this);
}

CPUComputeUnit::~CPUComputeUnit() {
  thread_running_ = false;
  pthread_mutex_lock(&mutex_work_queue_);
  pthread_cond_signal(&cond_work_queue_);
  pthread_mutex_unlock(&mutex_work_queue_);
  pthread_join(thread_, NULL);

  pthread_mutex_destroy(&mutex_work_queue_);
  pthread_cond_destroy(&cond_work_queue_);
}

bool CPUComputeUnit::IsIdle() const {
  return !work_to_do_;
}

void CPUComputeUnit::Launch(CPUWorkGroupAssignment* wga) {
  pthread_mutex_lock(&mutex_work_queue_);
  work_queue_.push_back(wga);
  work_to_do_ = true;
  pthread_cond_signal(&cond_work_queue_);
  pthread_mutex_unlock(&mutex_work_queue_);
}

void CPUComputeUnit::Sync() {
  pthread_mutex_lock(&mutex_work_queue_);
  if (work_to_do_)
    pthread_cond_wait(&cond_work_queue_, &mutex_work_queue_);
  pthread_mutex_unlock(&mutex_work_queue_);
}

void CPUComputeUnit::Run() {
  deque<CPUWorkGroupAssignment*> pong_queue;
  while (thread_running_) {
    pthread_mutex_lock(&mutex_work_queue_);
    if (!work_to_do_)
      pthread_cond_wait(&cond_work_queue_, &mutex_work_queue_);
    pong_queue.swap(work_queue_);
    pthread_mutex_unlock(&mutex_work_queue_);

    while (!pong_queue.empty()) {
      CPUWorkGroupAssignment* wga = pong_queue.front();
      int (*entry)(int id, void* argp);
      *(void**)(&entry) = dlsym(wga->handle, "dev_entry");

      CPUKernelParam* param = wga->param;
      CPUKernelParam local_param;
      memcpy(local_param.args, param->args, wga->args_size);
      local_param.launch_id = param->launch_id;
      local_param.work_dim = param->work_dim;
      memcpy(local_param.gwo, param->gwo, sizeof(size_t) * 3);
      memcpy(local_param.lws, param->lws, sizeof(size_t) * 3);
      memcpy(local_param.orig_grid, param->orig_grid, sizeof(size_t) * 3);
      memcpy(local_param.wg_id_begin, param->wg_id_begin, sizeof(size_t) * 3);
      memcpy(local_param.wg_id_size, param->wg_id_size, sizeof(size_t) * 3);
      local_param.wg_id_start = wga->wg_id_start;
      local_param.wg_id_end = wga->wg_id_end;
      local_param.kernel_idx = param->kernel_idx;
      local_param.file_index = param->file_index;
      (*entry)(id_, &local_param);

      pong_queue.pop_front();
    }

    pthread_mutex_lock(&mutex_work_queue_);
    if (work_queue_.empty()) {
      work_to_do_ = false;
      pthread_cond_signal(&cond_work_queue_);
    }
    pthread_mutex_unlock(&mutex_work_queue_);
  }
}

void* CPUComputeUnit::ThreadFunc(void* argp) {
  ((CPUComputeUnit*)argp)->Run();
  return NULL;
}
