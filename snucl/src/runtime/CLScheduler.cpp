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

void CLScheduler::Progress() {
	//return;
	if(queues_.size() <= 0) return;
	/* Analyze the pending cmds in all the target queues and 
	 * then invoke the scheduler */
	// RR Scheduler 
	// Check if all cmd queues are of same ctx
	CLContext *ctx = queues_[0]->context();
	std::vector<CLDevice *> devices = ctx->devices();
	size_t num_devices = devices.size();
	if(num_devices <= 1) return;
	size_t num_queues = queues_.size();
#if 0
	int chosen_dev_id = 0; // start RR from dev 0 of the context;
	for (vector<CLCommandQueue*>::iterator it = queues_.begin();
			it != queues_.end();
			++it) {
		// Do this only if the cmdq props say so
		CLCommandQueue* queue = *it;
		if(queue->context() != ctx) {
			SNUCL_ERROR("Scheduler is managing queues from different \
					contexts!", 0);
			exit(1);
		}
		queue->set_device(devices[chosen_dev_id]);
		chosen_dev_id = (chosen_dev_id + 1) % num_devices;
	}
#else
	// performance modeling
	/* this function should do the following 
	 * identify best device for the given SET of commands
	 * that are 'enqueued'
	 */
	std::vector<std::vector<double> > est_times;
	est_times.resize(num_queues);
	std::vector<bool> queue_has_kernel(queues_.size());
	// collect perf data
	Global::RealTimer q_perf_estimation_timer;
	q_perf_estimation_timer.Init();
	Global::RealTimer q_cmd_clone_timer;
	q_cmd_clone_timer.Init();
	for(int q_id = 0; q_id < queues_.size(); q_id++) 
	{
		queue_has_kernel[q_id] = false;
		est_times[q_id].resize(num_devices);
		CLCommandQueue *queue = queues_[q_id];
		if(queue->IsAutoDeviceSelection() != true) continue;
		std::list<CLCommand *> cmds = queue->commands();
		SNUCL_INFO("Command count for q_id: %d: %lu\n", q_id, cmds.size());
		//if(queue->get_perf_model_done()) continue; // MEGA HACK....CHANGE LOGIC ASAP
		std::string epochString;
		for (list<CLCommand*>::iterator cmd_it = cmds.begin();
				cmd_it != cmds.end();
				++cmd_it) 
		{
			if((*cmd_it)->type() != CL_COMMAND_NDRANGE_KERNEL) continue;
			//SNUCL_INFO("Cloning cmd (%p)\n", (*cmd_it));
			epochString += (*cmd_it)->kernel()->name();
		}
		if(queue->isEpochRecorded(epochString)) continue;
		//if(epochPerformances_.find(epochString) != epochPerformances_.end()) continue;
		//FIXME: epochPerformances_ should be a member of queue
	//	for(int device_id = 0; device_id < num_devices; device_id++) {
	//		SNUCL_INFO("Device %d: %p\n", device_id, devices[device_id]);
	//	}
		// iterate over all the commands in queue
		for (list<CLCommand*>::iterator cmd_it = cmds.begin();
				cmd_it != cmds.end();
				++cmd_it) 
		{
			//CLCommand* cmd = *cmd_it;
			SNUCL_INFO("Estimating Cost of Command Type %x for all devices\n", (*cmd_it)->type());
			if((*cmd_it)->type() != CL_COMMAND_NDRANGE_KERNEL) continue;
			//SNUCL_INFO("Cloning cmd (%p)\n", (*cmd_it));
			if(!q_perf_estimation_timer.IsRunning())q_perf_estimation_timer.Start();
			q_cmd_clone_timer.Start();
			CLCommand* cmd = (*cmd_it)->Clone();
			q_cmd_clone_timer.Stop();
			//SNUCL_INFO("Cloned cmd (%p->%p)\n", (*cmd_it), cmd);
			// estimate cost of cmd on device_id
			double est_cost = 0.0;
			for(int device_id = 0; device_id < num_devices; device_id++) 
			{
				est_times[q_id][device_id] = 0.0;
				CLDevice *dev = devices[device_id];
				if(cmd->type() == CL_COMMAND_NDRANGE_KERNEL)
				{
					est_cost = cmd->EstimatedCost(dev);
					queue_has_kernel[q_id] = true; 
				}
				SNUCL_INFO("Estimated Cost of Command Type %x for device %p: %g\n", cmd->type(), dev, est_cost);
				est_times[q_id][device_id] += est_cost;

				// if device_id was the 'chosen' device, what would
				// be the estimated cost? Approach for different cmd
				// types: 
				// a) kernel: 
				// 	1)run one work group on this device and
				// 	collect the performance. 
				// 	2) extrapolate to the entire device
				// b) read/write: get bw numbers from H2D and D2H benchmark
				// numbers; extrapolate if exact data size is not
				// there?
				// 	1) map: treat it as a special case read
				// c) ignore the rest of the commands for performance
				// estimation (is this the weakness of this appraoch?)
			}
			delete cmd;
		}
		if(q_perf_estimation_timer.IsRunning())
		{
			q_perf_estimation_timer.Stop();
			SNUCL_INFO("[Q%d] Performance Estimation Overhead: %gs; \
				Cmd cloning overhead: %gs\n", 
				q_id, q_perf_estimation_timer.CurrentElapsed(),
				q_cmd_clone_timer.Elapsed());
			q_cmd_clone_timer.Reset();
			q_perf_estimation_timer.Reset();
		}
		SNUCL_INFO("Epoch String for queue %p: %s\n", queue, epochString.c_str());
		queue->recordEpoch(epochString, est_times[q_id]);
		//epochPerformances_[epochString] = est_times;
	}
	// choose queue->device mapping
	for(int q_id = 0; q_id < queues_.size(); q_id++) {

		CLCommandQueue *queue = queues_[q_id];
		if(queue_has_kernel[q_id] != true || 
				queue->IsAutoDeviceSelection() != true) continue;
		std::list<CLCommand *> cmds = queue->commands();
		if(cmds.size() <= 0) continue;
		int chosen_dev_id = -1;
		int cur_dev_id = -1;
		for(int device_id = 0; device_id < num_devices; device_id++)
		{
			if(queues_[q_id]->device() == devices[device_id])
				cur_dev_id = device_id;
		}

		chosen_dev_id = cur_dev_id;
		double q_est_time = est_times[q_id][cur_dev_id];
		for(int device_id = 0; device_id < num_devices; device_id++)
		{
			SNUCL_INFO("Estimated Cost for Queue %p for device %p: %g\n", queues_[q_id], 
					devices[device_id], est_times[q_id][device_id]);
			if(est_times[q_id][device_id] < q_est_time)
			{
				chosen_dev_id = device_id;
				SNUCL_INFO("Device changing from %d to %d\n", cur_dev_id, chosen_dev_id);
				q_est_time = est_times[q_id][device_id];
			}
		}
		if(cur_dev_id != chosen_dev_id)
		{
			queues_[q_id]->set_device(devices[chosen_dev_id]);
		}
		// update estimated times to include already chosen time
		for(int q_idx = 0; q_idx < queues_.size(); q_idx++)
		{
			if(q_idx != q_id)
				est_times[q_idx][chosen_dev_id] += q_est_time;
		}
		//queue->set_perf_model_done(true);
	}
	// "best" device must have been selected by now. 
	for(int q_id = 0; q_id < queues_.size(); q_id++) {
		if(est_times[q_id].size() > 0)
			est_times[q_id].resize(0);
	}
	if(est_times.size() > 0)
		est_times.resize(0);
#endif
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
