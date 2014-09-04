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

#include "CLCommandQueue.h"
#include <list>
#include <stdio.h>
#include <pthread.h>
#include <CL/cl.h>
#include "CLCommand.h"
#include "CLContext.h"
#include "CLDevice.h"
#include "CLEvent.h"
#include "CLObject.h"
#include "Structs.h"
#include "Utils.h"
#include "CLPlatform.h"

using namespace std;

#define COMMAND_QUEUE_SIZE 4096
/*
CLCommandQueue::CLCommandQueue(CLContext *context, CLDevice* device,
                               cl_queue_properties *properties) {
  context_ = context;
  context_->Retain();
  device_ = device;
  properties_ = properties;
  //type_ = (properties_ & 0xFFF0) ? CL_QUEUE_AUTO_DEVICE_SELECTION : CL_QUEUE_FIXED_DEVICE_SELECTION;
  // Q: Automatic device selection before adding this to the device? 
  // A: Only if we have all the information (nearest). If we don't have 
  // all the info (kernel) then wait for the device selection
  //SNUCL_INFO("(Before) Device in Cmdqueue Creation: %p\n", device_);
  device_ = SelectBestDevice(context, device, properties);
  //SNUCL_INFO("(After) Device in Cmdqueue Creation: %p\n", device_);
  device_->AddCommandQueue(this);
  gQueueTimer.Init();
}
*/
CLCommandQueue::CLCommandQueue(CLContext *context, CLDevice* device,
                               cl_command_queue_properties properties) {
  perfModDone_ = false;
  context_ = context;
  context_->Retain();
  device_ = device;
  properties_ = properties;
  //type_ = (properties_ & 0xFFF0) ? CL_QUEUE_AUTO_DEVICE_SELECTION : CL_QUEUE_FIXED_DEVICE_SELECTION;
  // Q: Automatic device selection before adding this to the device? 
  // A: Only if we have all the information (nearest). If we don't have 
  // all the info (kernel) then wait for the device selection
  //SNUCL_INFO("(Before) Device in Cmdqueue Creation: %p\n", device_);
  CLDevice *new_device = SelectBestDevice(context, device, properties);
  cl_int err = set_device(new_device);
  device_->AddCommandQueue(this);
  gQueueTimer.Init();
}

CLCommandQueue::~CLCommandQueue() {
  device_->RemoveCommandQueue(this);
  context_->Release();
}

CLDevice *CLCommandQueue::SelectBestDevice(CLContext *context, CLDevice* device, 
                               cl_command_queue_properties properties) {
	cl_command_queue_properties prop_mask = properties & 0xFFF0;
	CLDevice* new_device = NULL;
	static bool has_printed = false;
	cl_device_type device_type;
	cl_uint vendor_id;
	device->GetDeviceInfo(CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	device->GetDeviceInfo(CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL);

	unsigned int chosen_device_id = 0;
	unsigned int chosen_host_id = 0;
	const std::vector<CLDevice*> devices = context->devices();
	const std::vector<hwloc_obj_t> hosts = context->hosts();
	for(cl_uint i = 0; i < devices.size(); i++)
	{
		//SNUCL_INFO("Command's Device: %p <--> Other Device: %p\n", device_, devices[i]);
		if(device == devices[i])
		{
			chosen_device_id = i;
			break;
		}
	}

	//SNUCL_INFO("Given Properties: %x Prop Mask: %x (out of %x %x)\n", properties, prop_mask, CL_QUEUE_DEVICE_SELECT_NEAREST, CL_QUEUE_DEVICE_SELECT_BEST_COMPUTE);
	if(prop_mask == CL_QUEUE_DEVICE_SELECT_NEAREST)
	{
		if(!has_printed)
		{
			SNUCL_INFO("(Before) Cmdqueue with %u type device with ID %u (%u/%u)\n", 
						device_type, vendor_id, chosen_device_id, devices.size());
		}
	  	//gCommandTimer.Start();
		// Find current host cpuset index/hwloc_obj_t
		CLPlatform *platform = CLPlatform::GetPlatform();
		hwloc_topology_t topology = platform->HWLOCTopology();
		hwloc_cpuset_t cpuset;
		cpuset = hwloc_bitmap_alloc();
		hwloc_bitmap_zero(cpuset);
		hwloc_get_cpubind(topology, cpuset, HWLOC_CPUBIND_THREAD);
		hwloc_obj_t cpuset_obj = hwloc_get_next_obj_covering_cpuset_by_type(topology, cpuset, HWLOC_OBJ_NODE, NULL);
		assert(cpuset_obj != NULL);

		std::vector<CLContext::perf_order_vector> d2h_distances = context->d2h_distances();
		for(unsigned int idx = 0; idx < hosts.size(); idx++)
		{
			//SNUCL_INFO("Comparing cpuset obj: %p Hosts hwloc ptr[%d]: %p\n", cpuset_obj, idx, hosts[idx]);
			//SNUCL_INFO("Chosen ones...host ID: %u/%u and device ID: %u/%u\n", chosen_host_id, hosts.size(), chosen_device_id, devices.size());
			if(hosts[idx] == cpuset_obj)
			{
				// choose this to find distance between this cpuset and all devices
				chosen_host_id = idx;
				// Nearest device will be at d2h_distances[idx][0];
				chosen_device_id = d2h_distances[idx][0].second;
				//SNUCL_INFO("Chosen ones...host ID: %u/%u and device ID: %u/%u\n", chosen_host_id, hosts.size(), chosen_device_id, devices.size());
			}
		}

		hwloc_bitmap_free(cpuset);

	  	//gCommandTimer.Stop();
		if(!has_printed)
		{
			SNUCL_INFO("(After) Cmdqueue with %u type device with ID %u (%u/%u)\n", 
						device_type, vendor_id, chosen_device_id, devices.size());
			has_printed = true;
		}
	}
	else if(prop_mask == CL_QUEUE_DEVICE_SELECT_BEST_COMPUTE)
	{
		CLContext::perf_order_vector devs_compute = context->devices_compute_perf();
		chosen_device_id = devs_compute[0].second;
	}
	else //if(prop_mask == CL_QUEUE_DEVICE_SELECT_MANUAL)
	{
		// No change to device_
	}
	// set the chosen device
	new_device = devices[chosen_device_id];
	new_device->GetDeviceInfo(CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	new_device->GetDeviceInfo(CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL);
	for(cl_uint i = 0; i < devices.size(); i++)
	{
		if(new_device == devices[i])
		{
			chosen_device_id = i;
			break;
		}
	}
	return new_device;
}
#if 0
bool CLCommandQueue::isEpochRecorded(std::string epoch) {
	if(epochPerformances_.find(epoch) != epochPerformances_.end())
		return true;
	return false;
}

std::vector<double> CLCommandQueue::getEpochCosts(std::string epoch) {
	if(isEpochRecorded(epoch))
		return epochPerformances_[epoch];
	return std::vector<double>(0);
}

void CLCommandQueue::recordEpoch(std::string epoch, std::vector<double> performances) {
	epochPerformances_[epoch] = performances;
}
#endif
cl_int CLCommandQueue::GetCommandQueueInfo(cl_command_queue_info param_name,
                                           size_t param_value_size,
                                           void* param_value,
                                           size_t* param_value_size_ret) {
  switch (param_name) {
    GET_OBJECT_INFO_T(CL_QUEUE_CONTEXT, cl_context, context_->st_obj());
    GET_OBJECT_INFO_T(CL_QUEUE_DEVICE, cl_device_id, device_->st_obj());
    GET_OBJECT_INFO_T(CL_QUEUE_REFERENCE_COUNT, cl_uint, ref_cnt());
    GET_OBJECT_INFO(CL_QUEUE_PROPERTIES, cl_command_queue_properties,
                    properties_);
    GET_OBJECT_INFO(CL_QUEUE_TYPE, cl_command_queue_type, type_);
    default: return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

void CLCommandQueue::PrintInfo()
{
  //if(IsProfiled())
  //	std::cout << "Queue Timer: " << gQueueTimer << std::endl;
}

void CLCommandQueue::InvokeScheduler() {
  device_->InvokeScheduler();
}
#if 0
size_t CLCommandQueue::CheckQueueProperties(
    const cl_queue_properties* properties, 
	cl_command_queue_properties *command_queue_properties,
	cl_int* err) {
  if (properties == NULL) return 0;

  size_t idx = 0;
  bool set_command_queue_properties = false;
  bool set_sched_policies = false;
  while (properties[idx] > 0) {
    if (properties[idx] == CL_QUEUE_PROPERTIES) {
      if (set_command_queue_properties) {
        *err = CL_INVALID_PROPERTY;
        return 0;
      }
      set_command_queue_properties = true;
      *command_queue_properties = (cl_command_queue_properties)properties[idx + 1];
      idx += 2;
    } else if (properties[idx] == CL_QUEUE_SCHEDULING_POLICIES) {
	  if(set_sched_policies) {
	  	*err = CL_INVALID_PROPERTY;
		return 0;
	  }
	  /* Iterate over the list until we run out of scheduling policies */
	  /*do {
	  }while(properties[idx + 1] != CL_QUEUE_PROPERTIES ||
	  		 properties[idx + 1] != CL_QUEUE_SCHEDULING_POLICIES ||
			 properties[idx + 1] != 0);
	  idx += howmuch;
	 */
	 set_sched_policies = true;
    } else {
      *err = CL_INVALID_PROPERTY;
      return 0;
    }
  }
  return idx + 1;
}
#endif
CLCommandQueue* CLCommandQueue::CreateCommandQueue(
    CLContext* context, CLDevice* device,
    cl_command_queue_properties properties, cl_int* err) {
  CLCommandQueue* queue;
  if (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    queue = new CLOutOfOrderCommandQueue(context, device, properties);
  else
    queue = new CLInOrderCommandQueue(context, device, properties);
  if (queue == NULL) {
    *err = CL_OUT_OF_HOST_MEMORY;
    return NULL;
  }
  //SNUCL_INFO("Command Queue Properties: %x\n", properties);
  return queue;
}

cl_int CLCommandQueue::set_device(CLDevice *d) {
  	if(d == NULL || !context_->IsValidDevice(d))
	{
		// ERROR
		return CL_INVALID_DEVICE;
	}
	if(d != device_)
	{
		SNUCL_INFO("CmdQ: %p Dev Changed (%p->%p)\n",
				this, device_, d);
		device_ = d; 
	}
	return CL_SUCCESS;
}

void CLInOrderCommandQueue::Flush() {
    device_->ProgressScheduler();	
	commands_.clear();
	InvokeScheduler();
}

CLInOrderCommandQueue::CLInOrderCommandQueue(
    CLContext* context, CLDevice* device,
    cl_command_queue_properties properties)
    : CLCommandQueue(context, device, properties),
      queue_(COMMAND_QUEUE_SIZE) {
  last_event_ = NULL;
}

CLInOrderCommandQueue::~CLInOrderCommandQueue() {
  //SNUCL_INFO("Destructor of Inorder CLCommandQueue %p\n", this);
  if (last_event_)
    last_event_->Release();
}

CLCommand* CLInOrderCommandQueue::Peek() {
  if (queue_.Size() == 0) return NULL;
  CLCommand* command;
  if (queue_.Peek(&command) && command->IsExecutable())
    return command;
  else
    return NULL;
}

void CLInOrderCommandQueue::Enqueue(CLCommand* command) {
  if(command->type() == CL_COMMAND_WRITE_BUFFER && IsProfiled())
  {
	  //gQueueTimer.Start();
  }
  if (last_event_ != NULL) {
    command->AddWaitEvent(last_event_);
    last_event_->Release();
  }
  last_event_ = command->ExportEvent();

  commands_.push_back(command);
  while (!queue_.Enqueue(command)) {}
  //SNUCL_INFO("Enqueued into queue: %p, q_size: %lu, cmds_size: %lu\n", 
  	//			this, queue_.Size(), commands_.size());
#if 0
  /*if(command->type() == CL_COMMAND_WRITE_BUFFER)
  {
	 gQueueTimer.Stop();
	 std::cout << "After dequeue" << std::endl;
  }*/
  InvokeScheduler();
#endif
}

void CLInOrderCommandQueue::Dequeue(CLCommand* command) {
  CLCommand* dequeued_command;
  queue_.Dequeue(&dequeued_command);
  //queue_vec_.delete(command);??
#ifdef SNUCL_DEBUG
  if (command != dequeued_command)
    SNUCL_ERROR("Invalid dequeue request", 0);
#endif // SNUCL_DEBUG
  if(command->type() == CL_COMMAND_WRITE_BUFFER && IsProfiled())
  {
	 //gQueueTimer.Stop();
	 //std::cout << "After dequeue" << std::endl;
  }
}

CLOutOfOrderCommandQueue::CLOutOfOrderCommandQueue(
    CLContext* context, CLDevice* device,
    cl_command_queue_properties properties)
    : CLCommandQueue(context, device, properties) {
  pthread_mutex_init(&mutex_commands_, NULL);
}

CLOutOfOrderCommandQueue::~CLOutOfOrderCommandQueue() {
  pthread_mutex_destroy(&mutex_commands_);
}

CLCommand* CLOutOfOrderCommandQueue::Peek() {
  if (commands_.empty()) return NULL;

  CLCommand* result = NULL;
  pthread_mutex_lock(&mutex_commands_);
  for (list<CLCommand*>::iterator it = commands_.begin();
       it != commands_.end();
       ++it) {
    CLCommand* command = *it;
    if (!command->IsExecutable()) continue;

    if (command->type() == CL_COMMAND_MARKER ||
        command->type() == CL_COMMAND_BARRIER) {
      if (it == commands_.begin())
        result = command;
    } else {
      result = command;
    }
    break;
  }
  pthread_mutex_unlock(&mutex_commands_);
  return result;
}

void CLOutOfOrderCommandQueue::Enqueue(CLCommand* command) {
  pthread_mutex_lock(&mutex_commands_);
  commands_.push_back(command);
  pthread_mutex_unlock(&mutex_commands_);
  InvokeScheduler();
}

void CLOutOfOrderCommandQueue::Dequeue(CLCommand* command) {
  pthread_mutex_lock(&mutex_commands_);
  commands_.remove(command);
  pthread_mutex_unlock(&mutex_commands_);
}
