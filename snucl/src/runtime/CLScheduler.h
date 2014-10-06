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

#ifndef __SNUCL__CL_SCHEDULER_H
#define __SNUCL__CL_SCHEDULER_H

#include <vector>
#include <map>
#include <pthread.h>
#include <semaphore.h>
#include "RealTimer.h"

class CLCommand;
class CLCommandQueue;
class CLPlatform;

class CLScheduler {
 public:
  CLScheduler(CLPlatform* platform, bool busy_waiting);
  ~CLScheduler();

  void Start();
  void Stop();
  void Invoke();
  void Progress(bool special_event = false);
  void AddCommandQueue(CLCommandQueue* queue);
  void RemoveCommandQueue(CLCommandQueue* queue);

 private:
  //typedef std::vector<std::vector<double> > queuePerfVector;
  //std::map<std::string, queuePerfVector> epochPerformances_; 

  Global::RealTimer gSchedTimer;
  void Run();

  CLPlatform* platform_;
  bool busy_waiting_;
  std::vector<CLCommandQueue*> queues_;
  bool queues_updated_;

  pthread_t thread_;
  bool thread_running_;
  sem_t sem_schedule_;

  pthread_mutex_t mutex_queues_;
  pthread_cond_t cond_queues_remove_;

  static void* ThreadFunc(void* argp);
};

#endif // __SNUCL__CL_SCHEDULER_H

