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

#include "CLScheduler.h"
#include <algorithm>
#include <vector>
#include <pthread.h>
#include <semaphore.h>
#include <CL/cl.h>
#include "CLCommand.h"
#include "CLCommandQueue.h"
#include "CLDevice.h"
#include "CLEvent.h"
#include "CLPlatform.h"

using namespace std;

CLScheduler::CLScheduler(CLPlatform* platform, bool busy_waiting) {
  platform_ = platform;
  busy_waiting_ = busy_waiting;
  queues_updated_ = false;
  thread_ = (pthread_t)NULL;
  thread_running_ = false;
  gSchedTimer.Init();
  if (!busy_waiting_)
    sem_init(&sem_schedule_, 0, 0);
  pthread_mutex_init(&mutex_queues_, NULL);
  pthread_cond_init(&cond_queues_remove_, NULL);
}

CLScheduler::~CLScheduler() {
  std::cout << "Scheduler Time: " << gSchedTimer << std::endl;
  Stop();
  if (!busy_waiting_)
    sem_destroy(&sem_schedule_);
  pthread_mutex_destroy(&mutex_queues_);
  pthread_cond_destroy(&cond_queues_remove_);
}

void CLScheduler::Start() {
  if (!thread_) {
    thread_running_ = true;
    pthread_create(&thread_, NULL, &CLScheduler::ThreadFunc, this);
  }
}

void CLScheduler::Stop() {
  if (thread_) {
    thread_running_ = false;
    Invoke();
    pthread_join(thread_, NULL);
	SNUCL_INFO("CLScheduler joined main thread", 0);
    thread_ = (pthread_t)NULL;
  }
}

void CLScheduler::Invoke() {
  if (!busy_waiting_)
  {
  	//gSchedTimer.Start();
    sem_post(&sem_schedule_);
  }
}

void CLScheduler::AddCommandQueue(CLCommandQueue* queue) {
  pthread_mutex_lock(&mutex_queues_);
  queues_updated_ = true;
  queues_.push_back(queue);
  pthread_mutex_unlock(&mutex_queues_);
}

void CLScheduler::RemoveCommandQueue(CLCommandQueue* queue) {
  pthread_mutex_lock(&mutex_queues_);
  vector<CLCommandQueue*>::iterator it = find(queues_.begin(), queues_.end(),
                                              queue);
  if (it != queues_.end()) {
    queues_updated_ = true;
    *it = NULL;
    if (!busy_waiting_)
      sem_post(&sem_schedule_);
    pthread_cond_wait(&cond_queues_remove_, &mutex_queues_);
  }
  pthread_mutex_unlock(&mutex_queues_);
}

void CLScheduler::Run() {
  vector<CLCommandQueue*> target_queues;

  while (thread_running_) {
    if (!busy_waiting_)
	{
      sem_wait(&sem_schedule_);
	  //if(!gSchedTimer.IsRunning()) gSchedTimer.Start();
	}

    if (queues_updated_) {
      pthread_mutex_lock(&mutex_queues_);
      pthread_cond_broadcast(&cond_queues_remove_);
      queues_.erase(remove(queues_.begin(), queues_.end(),
                           (CLCommandQueue*)NULL),
                    queues_.end());
      target_queues = queues_;
      queues_updated_ = false;
      pthread_mutex_unlock(&mutex_queues_);
    }

    for (vector<CLCommandQueue*>::iterator it = target_queues.begin();
         it != target_queues.end();
         ++it) {
      CLCommandQueue* queue = *it;
      CLCommand* command = queue->Peek();
	  //bool resolved = true;
	  /*if(command != NULL)
	  {
	  if(!gSchedTimer.IsRunning()) gSchedTimer.Start();
	  resolved = command->ResolveConsistency();
  	  gSchedTimer.Stop();
	  }*/
      //if (command != NULL && resolved) {
      if (command != NULL && command->ResolveConsistency()) {
	  	// command should have the appropriate device already chosen at this point
        if(command->IsAlreadyCompleted())
		{
			command->SetAsComplete();
			delete command;
		}
		else
		{
		command->Submit();
		}
		queue->Dequeue(command);
      }
    }
  }
  //gSchedTimer.Stop();
}

void* CLScheduler::ThreadFunc(void* argp) {
	pthread_t thread = pthread_self();
#if 1
#if 1	
	hwloc_topology_t topology = CLPlatform::GetPlatform()->HWLOCTopology();
    int n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
    int n_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
    int n_cpu_sockets = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NODE); 
	int n_pus_per_socket = n_pus / n_cpu_sockets;
	hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
	hwloc_bitmap_zero(cpuset);
	hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);
	hwloc_obj_t cpuset_obj = hwloc_get_next_obj_covering_cpuset_by_type(topology, cpuset, HWLOC_OBJ_NODE, NULL);
	int socket_id = 0;
    for(socket_id = 0; socket_id < n_cpu_sockets; socket_id++)
    {
	  hwloc_obj_t node_obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_NODE, socket_id);
	  if(node_obj == cpuset_obj)
	  	break;
	}
	SNUCL_INFO("Setting Scheduler Thread to Core: %d\n", socket_id * n_pus_per_socket + n_pus_per_socket - 1);
	cpu_set_t c_set;
	CPU_ZERO(&c_set);
	CPU_SET(socket_id * n_pus_per_socket + n_pus_per_socket - 1, &c_set);
	pthread_setaffinity_np(thread, sizeof(cpu_set_t), &c_set);
	//cpuset_obj = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NODE, cpuset_obj);
	//hwloc_set_thread_cpubind(topology, thread, cpuset_obj->cpuset, HWLOC_CPUBIND_THREAD); 
	hwloc_bitmap_free(cpuset);
#else
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(5, &cpuset);
	pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
#endif
#endif
	((CLScheduler*)argp)->Run();
	return NULL;
}
