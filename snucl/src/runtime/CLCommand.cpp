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

#include "CLCommand.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <malloc.h>
#include <stdio.h>
#include <CL/cl.h>
#include "Callbacks.h"
#include "CLCommandQueue.h"
#include "CLContext.h"
#include "CLDevice.h"
#include "CLEvent.h"
#include "CLKernel.h"
#include "CLMem.h"
#include "CLProgram.h"
#include "CLPlatform.h"
#include "CLSampler.h"
#include "Structs.h"

using namespace std;

class CLPlatform;

CLCommand::CLCommand(CLContext* context, CLDevice* device,
                     CLCommandQueue* queue, cl_command_type type) {
  gCommandTimer.Init();
  sched_type_ = SNUCL_SCHED_MANUAL;
  alreadyCompleted_ = false;
  type_ = type;
  queue_ = queue;
  if (queue_ != NULL) {
    queue_->Retain();
    context = queue_->context();
    device = queue_->device();
  }
  context_ = context;
  context_->Retain();
  device_ = device;
  if (queue_ != NULL) {
    event_ = new CLEvent(queue, this);
  } else {
    event_ = new CLEvent(context, this);
  }
  wait_events_complete_ = false;
  wait_events_good_ = true;
  consistency_resolved_ = false;
  error_ = CL_SUCCESS;

  dev_src_ = NULL;
  dev_dst_ = NULL;
  node_src_ = -1;
  node_dst_ = -1;
  event_id_ = event_->id();

  mem_src_ = NULL;
  mem_dst_ = NULL;
  pattern_ = NULL;
  kernel_ = NULL;
  kernel_args_ = NULL;
  native_args_ = NULL;
  mem_list_ = NULL;
  mem_offsets_ = NULL;
  temp_buf_ = NULL;
  program_ = NULL;
  headers_ = NULL;
  link_binaries_ = NULL;

  custom_function_ = NULL;
  custom_data_ = NULL;
}

CLCommand::~CLCommand() {
  //std::cout << "Command Timer: " << gCommandTimer << std::endl;
  if (queue_) queue_->Release();
  context_->Release();
  for (vector<CLEvent*>::iterator it = wait_events_.begin();
       it != wait_events_.end();
       ++it) {
    (*it)->Release();
  }
  //SNUCL_INFO("[TID: %p] Destructor of Command: %p\n", pthread_self(), this); 
  if (mem_src_) mem_src_->Release();
  if (mem_dst_) mem_dst_->Release();
  if (pattern_) free(pattern_);
  if (kernel_) kernel_->Release();
  if (kernel_args_) {
    for (map<cl_uint, CLKernelArg*>::iterator it = kernel_args_->begin();
         it != kernel_args_->end();
         ++it) {
      CLKernelArg* arg = it->second;
      if (arg->mem) arg->mem->Release();
      if (arg->sampler) arg->sampler->Release();
      free(arg);
    }
    delete kernel_args_;
  }
  if (native_args_) free(native_args_);
  if (mem_list_) {
    for (uint i = 0; i < num_mem_objects_; i++)
      mem_list_[i]->Release();
    free(mem_list_);
  }
  if (mem_offsets_) free(mem_offsets_);
  if (temp_buf_) 
  {
  	//if(alloc_host_) 
//		device_->FreeHostMem(mem_dst_, temp_buf_);
	//else 
	//	free(temp_buf_);
  }
  if (program_) program_->Release();
  if (headers_) {
    for (size_t i = 0; i < size_; i++)
      delete headers_[i];
    free(headers_);
  }
  if (link_binaries_) {
    for (size_t i = 0; i < size_; i++)
      delete link_binaries_[i];
    free(link_binaries_);
  }
  event_->Release();
}

double CLCommand::EstimatedCost(CLDevice *target_device) {
	Global::RealTimer cmd_timer;
	cmd_timer.Init();
	cmd_timer.Start();
	switch(type_)
	{
		case CL_COMMAND_NDRANGE_KERNEL:
			/*for(std::map<cl_uint, CLKernelArg*>::iterator it=kernel_args_->begin(); 
						it!=kernel_args_->end(); ++it) {
    			//SNUCL_INFO("Kernel Idx: %u, param: %p\n", it->first, it->second);
				CLMem *mem = it->second->mem;
				if(mem != NULL) {
					
				}
			}*/
			/*for(all kernel params)
			{
				create buffer on device if param is a dev buf
				should we init it//?
			}*/
      		target_device->LaunchTestKernel(this, kernel_, work_dim_, gwo_, gws_, lws_, nwg_,
                            kernel_args_);
			/*for(all kernel params)
			{
				destroy buffer params one by one
			}*/
			break;
		//case CL_COMMAND_READ_BUFFER:
      	//	target_device->ReadBuffer(this, mem_src_, off_src_, size_, ptr_);
		//	break;
		//case CL_COMMAND_WRITE_BUFFER:
      	//	target_device->WriteBuffer(this, mem_dst_, off_dst_, size_, ptr_);
		//	break;
			//case CL_COMMAND_COPY_BUFFER:
			//case CL_COMMAND_MAP_IMAGE:
			//case CL_COMMAND_UNMAP_MEM_OBJECT:
		default:
			// ignore this command for cost
			// estimation
			break;
	}
	cmd_timer.Stop();
	return cmd_timer.CurrentElapsed();
}

void CLCommand::SetWaitList(cl_uint num_events_in_wait_list,
                            const cl_event* event_wait_list) {
  wait_events_.reserve(num_events_in_wait_list);
  for (cl_uint i = 0; i < num_events_in_wait_list; i++) {
    CLEvent* e = event_wait_list[i]->c_obj;
    e->Retain();
    wait_events_.push_back(e);
  }
}

void CLCommand::AddWaitEvent(CLEvent* event) {
  event->Retain();
  wait_events_.push_back(event);
  wait_events_complete_ = false;
}

bool CLCommand::IsExecutable() {
  if (wait_events_complete_)
    return true;

  for (vector<CLEvent*>::iterator it = wait_events_.begin();
       it != wait_events_.end();
       ++it) {
    if (!(*it)->IsComplete())
      return false;
  }

  for (vector<CLEvent*>::iterator it = wait_events_.begin();
       it != wait_events_.end();
       ++it) {
    CLEvent* event = *it;
    if (event->IsError())
      wait_events_good_ = false;
    event->Release();
  }
  wait_events_.clear();
  wait_events_complete_ = true;
  return true;
}

void CLCommand::Submit() {
  event_->SetStatus(CL_SUBMITTED);
  device_->EnqueueReadyQueue(this);
}

void CLCommand::SetError(cl_int error) {
  error_ = error;
}

void CLCommand::SetAsRunning() {
  event_->SetStatus(CL_RUNNING);
}

void CLCommand::SetAsComplete() {
  if (error_ != CL_SUCCESS && error_ < 0) {
    event_->SetStatus(error_);
  } else {
	device_->RemoveCommand();
    event_->SetStatus(CL_COMPLETE);
  }
}

CLEvent* CLCommand::ExportEvent() {
  event_->Retain();
  return event_;
}

void CLCommand::AnnotateSourceDevice(CLDevice* device) {
  dev_src_ = device;
}

void CLCommand::AnnotateDestinationDevice(CLDevice* device) {
  dev_dst_ = device;
}

void CLCommand::AnnotateSourceNode(int node) {
  node_src_ = node;
}

void CLCommand::AnnotateDestinationNode(int node) {
  node_dst_ = node;
}

bool CLCommand::IsPartialCommand() const {
  return event_id_ != event_->id();
}

void CLCommand::SetAsPartialCommand(CLCommand* root) {
  event_id_ = root->event_id_;
}

void CLCommand::Execute() {
  switch (type_) {
    case CL_COMMAND_NDRANGE_KERNEL:
    case CL_COMMAND_TASK:
      device_->LaunchKernel(this, kernel_, work_dim_, gwo_, gws_, lws_, nwg_,
                            kernel_args_);
      break;
    case CL_COMMAND_NATIVE_KERNEL:
      device_->LaunchNativeKernel(this, user_func_, native_args_, size_,
                                  num_mem_objects_, mem_list_, mem_offsets_);
      break;
    case CL_COMMAND_READ_BUFFER:
      device_->ReadBuffer(this, mem_src_, off_src_, size_, ptr_);
      break;
    case CL_COMMAND_WRITE_BUFFER:
	  // if read only buffer, copy buffer to all devices in the context
	  if(mem_dst_->flags() & CL_MEM_READ_ONLY) 
	  {
		  vector<CLDevice *> devices = context_->devices();
		  for (vector<CLDevice*>::iterator it = devices.begin();
				  it != devices.end(); ++it) {
			  (*it)->WriteBuffer(this, mem_dst_, off_dst_, size_, ptr_);
		  }
	  } 
	  else {
		  device_->WriteBuffer(this, mem_dst_, off_dst_, size_, ptr_);
	  }
      break;
    case CL_COMMAND_COPY_BUFFER:
      device_->CopyBuffer(this, mem_src_, mem_dst_, mem_src_dev_specific_, mem_dst_dev_specific_, off_src_, off_dst_, size_);
      //device_->CopyBuffer(this, mem_src_, mem_dst_, off_src_, off_dst_, size_);
      break;
    case CL_COMMAND_READ_IMAGE:
      device_->ReadImage(this, mem_src_, src_origin_, region_, dst_row_pitch_,
                         dst_slice_pitch_, ptr_);
      break;
    case CL_COMMAND_WRITE_IMAGE:
      device_->WriteImage(this, mem_dst_, dst_origin_, region_, src_row_pitch_,
                          src_slice_pitch_, ptr_);
      break;
    case CL_COMMAND_COPY_IMAGE:
      device_->CopyImage(this, mem_src_, mem_dst_, src_origin_, dst_origin_,
                         region_);
      break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      device_->CopyImageToBuffer(this, mem_src_, mem_dst_, src_origin_,
                                 region_, off_dst_);
      break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      device_->CopyBufferToImage(this, mem_src_, mem_dst_, off_src_,
                                 dst_origin_, region_);
      break;
    case CL_COMMAND_MAP_BUFFER:
      device_->MapBuffer(this, mem_src_, map_flags_, off_src_, size_, ptr_);
      break;
    case CL_COMMAND_MAP_IMAGE:
      device_->MapImage(this, mem_src_, map_flags_, src_origin_, region_,
                        ptr_);
      break;
    case CL_COMMAND_UNMAP_MEM_OBJECT:
      device_->UnmapMemObject(this, mem_src_, ptr_);
      break;
    case CL_COMMAND_MARKER:
      break;
    case CL_COMMAND_READ_BUFFER_RECT:
      device_->ReadBufferRect(this, mem_src_, src_origin_, dst_origin_,
                              region_, src_row_pitch_, src_slice_pitch_,
                              dst_row_pitch_, dst_slice_pitch_, ptr_);
      break;
    case CL_COMMAND_WRITE_BUFFER_RECT:
      device_->WriteBufferRect(this, mem_dst_, src_origin_, dst_origin_,
                               region_, src_row_pitch_, src_slice_pitch_,
                               dst_row_pitch_, dst_slice_pitch_, ptr_);
      break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      device_->CopyBufferRect(this, mem_src_, mem_dst_, src_origin_,
                              dst_origin_, region_, src_row_pitch_,
                              src_slice_pitch_, dst_row_pitch_,
                              dst_slice_pitch_);
      break;
    case CL_COMMAND_BARRIER:
      break;
    case CL_COMMAND_MIGRATE_MEM_OBJECTS:
      device_->MigrateMemObjects(this, num_mem_objects_, mem_list_,
                                 migration_flags_);
      break;
    case CL_COMMAND_FILL_BUFFER:
      device_->FillBuffer(this, mem_dst_, pattern_, pattern_size_, off_dst_,
                          size_);
      break;
    case CL_COMMAND_FILL_IMAGE:
      device_->FillImage(this, mem_dst_, ptr_, dst_origin_, region_);
      break;
    case CL_COMMAND_BUILD_PROGRAM:
      device_->BuildProgram(this, program_, source_, binary_, options_);
      break;
    case CL_COMMAND_COMPILE_PROGRAM:
      device_->CompileProgram(this, program_, source_, options_, size_,
                              headers_);
      break;
    case CL_COMMAND_LINK_PROGRAM:
      device_->LinkProgram(this, program_, size_, link_binaries_, options_);
      break;
    case CL_COMMAND_WAIT_FOR_EVENTS:
      break;
    case CL_COMMAND_CUSTOM:
      custom_function_(custom_data_);
      break;
    case CL_COMMAND_NOP:
      break;
    case CL_COMMAND_ALLTOALL_BUFFER:
      device_->AlltoAllBuffer(this, mem_src_, mem_dst_, off_src_, off_dst_,
                              size_);
      break;
    case CL_COMMAND_BROADCAST_BUFFER:
      device_->BroadcastBuffer(this, mem_src_, mem_dst_, off_src_, off_dst_,
                               size_);
      break;
    default:
      SNUCL_ERROR("Unsupported command [%x]", type_);
      break;
  }
  device_->AddCommand();
}

bool CLCommand::IsAlreadyCompleted() {
	return alreadyCompleted_;
}

bool CLCommand::ResolveConsistency() {
  if(queue_ != NULL)
  {
  	device_ = queue_->device();
  }
#if 0
  switch(type_)
  {
  	case CL_COMMAND_NDRANGE_KERNEL:
	case CL_COMMAND_TASK:
	case CL_COMMAND_WRITE_BUFFER:
	case CL_COMMAND_WRITE_IMAGE:
	case CL_COMMAND_WRITE_BUFFER_RECT:
	case CL_COMMAND_FILL_BUFFER:
	case CL_COMMAND_FILL_IMAGE:
	case CL_COMMAND_READ_BUFFER:
	case CL_COMMAND_READ_IMAGE:
	case CL_COMMAND_READ_BUFFER_RECT:
  		ResolveDeviceCharacteristics();
	    break;
	default: 
		break;
  }
  switch(type_)
  {
  	case CL_COMMAND_NDRANGE_KERNEL:
	case CL_COMMAND_TASK:
  		ResolveDeviceOfLaunchKernel();
		break;
	case CL_COMMAND_WRITE_BUFFER:
	case CL_COMMAND_WRITE_IMAGE:
	case CL_COMMAND_WRITE_BUFFER_RECT:
	case CL_COMMAND_FILL_BUFFER:
	case CL_COMMAND_FILL_IMAGE:
		ResolveDeviceOfWriteMem();
		break;
	case CL_COMMAND_READ_BUFFER:
	case CL_COMMAND_READ_IMAGE:
	case CL_COMMAND_READ_BUFFER_RECT:
		ResolveDeviceOfWriteMem();
	    //ResolveDeviceOfReadMem();
		//Is there a specific use case to have different algos for Read
		//vs. Write?
	    break;
	default: 
		break;
  }
#endif
  bool resolved = consistency_resolved_;
  if (!resolved) {
    switch (type_) {
      case CL_COMMAND_NDRANGE_KERNEL:
      case CL_COMMAND_TASK:
        resolved = ResolveConsistencyOfLaunchKernel();
        break;
      case CL_COMMAND_NATIVE_KERNEL:
        resolved = ResolveConsistencyOfLaunchNativeKernel();
        break;
      case CL_COMMAND_READ_BUFFER:
      case CL_COMMAND_READ_IMAGE:
      case CL_COMMAND_READ_BUFFER_RECT:
        resolved = ResolveConsistencyOfReadMem();
        break;
      case CL_COMMAND_WRITE_BUFFER:
      case CL_COMMAND_WRITE_IMAGE:
      case CL_COMMAND_WRITE_BUFFER_RECT:
      case CL_COMMAND_FILL_BUFFER:
      case CL_COMMAND_FILL_IMAGE:
        resolved = ResolveConsistencyOfWriteMem();
        break;
      case CL_COMMAND_COPY_BUFFER:
      case CL_COMMAND_COPY_IMAGE:
      case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      case CL_COMMAND_COPY_BUFFER_RECT:
        resolved = ResolveConsistencyOfCopyMem();
        break;
      case CL_COMMAND_MAP_BUFFER:
      case CL_COMMAND_MAP_IMAGE:
        resolved = ResolveConsistencyOfMap();
        break;
      case CL_COMMAND_UNMAP_MEM_OBJECT:
        resolved = ResolveConsistencyOfUnmap();
        break;
      case CL_COMMAND_BROADCAST_BUFFER:
        resolved = ResolveConsistencyOfBroadcast();
        break;
      case CL_COMMAND_ALLTOALL_BUFFER:
        resolved = ResolveConsistencyOfAlltoAll();
        break;
      default:
        resolved = true;
        break;
    }
  }
  if (resolved) {
    switch (type_) {
      case CL_COMMAND_NDRANGE_KERNEL:
      case CL_COMMAND_TASK:
        UpdateConsistencyOfLaunchKernel();
        break;
      case CL_COMMAND_NATIVE_KERNEL:
        UpdateConsistencyOfLaunchNativeKernel();
        break;
      case CL_COMMAND_READ_BUFFER:
      case CL_COMMAND_READ_IMAGE:
      case CL_COMMAND_READ_BUFFER_RECT:
        UpdateConsistencyOfReadMem();
        break;
      case CL_COMMAND_WRITE_BUFFER:
      case CL_COMMAND_WRITE_IMAGE:
      case CL_COMMAND_WRITE_BUFFER_RECT:
      case CL_COMMAND_FILL_BUFFER:
      case CL_COMMAND_FILL_IMAGE:
        UpdateConsistencyOfWriteMem();
        break;
      case CL_COMMAND_COPY_BUFFER:
      case CL_COMMAND_COPY_IMAGE:
      case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      case CL_COMMAND_COPY_BUFFER_RECT:
        UpdateConsistencyOfCopyMem();
        break;
      case CL_COMMAND_MAP_BUFFER:
      case CL_COMMAND_MAP_IMAGE:
        UpdateConsistencyOfMap();
        break;
      case CL_COMMAND_UNMAP_MEM_OBJECT:
        UpdateConsistencyOfUnmap();
        break;
      case CL_COMMAND_BROADCAST_BUFFER:
        UpdateConsistencyOfBroadcast();
        break;
      case CL_COMMAND_ALLTOALL_BUFFER:
        UpdateConsistencyOfAlltoAll();
        break;
      default:
        break;
    }
  }
  return resolved;
}

void CLCommand::ResolveDeviceCharacteristics()
{
	cl_command_queue_properties queue_props;
	queue_->GetCommandQueueInfo(CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties),
								&queue_props, NULL);
	//SNUCL_INFO("Command Queue Properties for this command: %x\n", queue_props);
	switch(queue_props & 0xFFF0)
	{
		case CL_QUEUE_DEVICE_SELECT_BEST_COMPUTE:
			sched_type_ = SNUCL_SCHED_MAX_COMPUTE;
			break;
		case CL_QUEUE_DEVICE_SELECT_BEST_MEM:
			sched_type_ = SNUCL_SCHED_MAX_MEMORY;
			break;
		case CL_QUEUE_DEVICE_SELECT_BEST_LMEM:
			sched_type_ = SNUCL_SCHED_MAX_LMEMORY;
			break;
		case CL_QUEUE_DEVICE_SELECT_PERF_MODEL:
			sched_type_ = SNUCL_SCHED_PERF_MODEL;
			break;
		case CL_QUEUE_DEVICE_SELECT_NEAREST:
			sched_type_ = SNUCL_SCHED_CLOSEST;
			break;
		case CL_QUEUE_DEVICE_SELECT_LEAST_LOADED:
			sched_type_ = SNUCL_SCHED_LEAST_LOADED;
			break;
		case CL_QUEUE_DEVICE_SELECT_MANUAL:
		default:
			sched_type_ = SNUCL_SCHED_MANUAL;
			break;
	}
	//SNUCL_INFO("Scheduler Type: %x\n", sched_type_);
}

int CLCommand::ResolveDeviceOfLaunchKernel() {
	cl_device_type device_type;
	cl_uint vendor_id;
	cl_uint device_id;
	device_->GetDeviceInfo(CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	device_->GetDeviceInfo(CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL);

	const std::vector<CLDevice*> devices = context_->devices();
	for(cl_uint i = 0; i < devices.size(); i++)
	{
		//SNUCL_INFO("Command's Device: %p <--> Other Device: %p\n", device_, devices[i]);
		if(device_ == devices[i])
		{
			device_id = i;
			break;
		}
	}
	int chosen_device_id = 0;
	//SNUCL_INFO("(Before) Submitting %u command type to %u type device with ID %u (%u/%u)\n", type_, device_type, vendor_id, device_id, devices.size());
	// If this is a kernel execution: (yes)
	// 1) Extract the memory objects from the kernel (will do)
	// 2) Find their latest device? (will do)
	// 3) Do the copy to the chosen device? (snucl does this)
	
	// perfmodel_->EnqueueReadyQueue(this);
	// perfmodel_->rankDevices(program, kernel, in_device_list,
	// out_device_list);
	// Performance model here?

	if(sched_type_ == SNUCL_SCHED_CLOSEST)
	{
		float min_avg_distance = 10565.0f;
		float avg_distance = 0.0f;
		for(cl_uint i = 0; i < devices.size(); i++)
		{
			int total_distance = 0;
			int mem_args_count = 0;
			CLDevice *cur_device = devices[i];
			for (map<cl_uint, CLKernelArg*>::iterator it = kernel_args_->begin();
					it != kernel_args_->end();
					++it) {
				CLKernelArg* arg = it->second;
				if (arg->mem != NULL)
				{
					int cur_distance = 0;
					if (arg->mem->HasLatest(cur_device) || arg->mem->EmptyLatest())
						cur_distance = 0;
					else
					{
						CLDevice* source = arg->mem->GetNearestLatest(cur_device);
						cur_distance = cur_device->GetDistance(source);
					}
					total_distance += cur_distance;
					mem_args_count++;
				}
			}
			//SNUCL_INFO("Total distance from device %d: %d\n", i, total_distance);
			if(mem_args_count > 0)
				avg_distance = total_distance / mem_args_count;
			else
				avg_distance = 0.0f;

			if(avg_distance < min_avg_distance)
			{
				// Choose this device
				//device_ = devices[i];
				chosen_device_id = i;
				min_avg_distance = avg_distance;
			}
		}
	}
	// set the chosen device
	device_ = devices[chosen_device_id];
	device_->GetDeviceInfo(CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	device_->GetDeviceInfo(CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL);
	for(cl_uint i = 0; i < devices.size(); i++)
	{
		if(device_ == devices[i])
		{
			device_id = i;
			break;
		}
	}

	//SNUCL_INFO("(After) Submitted %u command type to %u type device with ID %u (%u/%u)\n", type_, device_type, vendor_id, device_id, devices.size());
	return 0;
}

int CLCommand::ResolveDeviceOfWriteMem() {
	static bool has_printed = false;
	cl_device_type device_type;
	cl_uint vendor_id;
	device_->GetDeviceInfo(CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	device_->GetDeviceInfo(CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL);

	unsigned int chosen_device_id = 0;
	unsigned int chosen_host_id = 0;
	const std::vector<CLDevice*> devices = context_->devices();
	const std::vector<hwloc_obj_t> hosts = context_->hosts();
	for(cl_uint i = 0; i < devices.size(); i++)
	{
		//SNUCL_INFO("Command's Device: %p <--> Other Device: %p\n", device_, devices[i]);
		if(device_ == devices[i])
		{
			chosen_device_id = i;
			break;
		}
	}
	/*if(!has_printed)
	{
		SNUCL_INFO("(Before) Submitting %u command type to %u type device with ID %u (%u/%u)\n", type_, device_type, vendor_id, chosen_device_id, devices.size());
	}*/

	if(sched_type_ == SNUCL_SCHED_CLOSEST)
	{
		if(!has_printed)
		{
			SNUCL_INFO("(Before) Submitting %u command type to %u type device with ID %u (%u/%u)\n", type_, device_type, vendor_id, chosen_device_id, devices.size());
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

		std::vector<CLContext::perf_order_vector> d2h_distances = context_->d2h_distances();
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
			SNUCL_INFO("(After) Submitted %u command type to %u type device with ID %u (%u/%u)\n", type_, device_type, vendor_id, chosen_device_id, devices.size());
			has_printed = true;
		}
	}
	else if(sched_type_ == SNUCL_SCHED_MAX_COMPUTE)
	{
		CLContext::perf_order_vector devs_compute = context_->devices_compute_perf();
		chosen_device_id = devs_compute[0].second;
	}
	else if(sched_type_ == SNUCL_SCHED_MANUAL)
	{
		// No change to device_
	}
	// set the chosen device
	device_ = devices[chosen_device_id];
	device_->GetDeviceInfo(CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
	device_->GetDeviceInfo(CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &vendor_id, NULL);
	for(cl_uint i = 0; i < devices.size(); i++)
	{
		if(device_ == devices[i])
		{
			chosen_device_id = i;
			break;
		}
	}
	/*if(!has_printed)
	{
		SNUCL_INFO("(After) Submitted %u command type to %u type device with ID %u (%u/%u)\n", type_, device_type, vendor_id, chosen_device_id, devices.size());
		has_printed = true;
	}*/
	return 0;
}


bool CLCommand::ResolveConsistencyOfLaunchKernel() {
  bool already_resolved = true;
  //SNUCL_INFO("Resolving consistency of kernel\n", 0);
  for (map<cl_uint, CLKernelArg*>::iterator it = kernel_args_->begin();
       it != kernel_args_->end();
       ++it) {
    CLKernelArg* arg = it->second;
    if (arg->mem != NULL)
	{
      already_resolved &= LocateMemOnDevice(arg->mem);
	}
  }
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfLaunchNativeKernel() {
  bool already_resolved = true;
  for (cl_uint i = 0; i < num_mem_objects_; i++)
    already_resolved &= LocateMemOnDevice(mem_list_[i]);
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfReadMem() {
  //SNUCL_INFO("(Before) Device in Resolve Read consistency: %p\n", device_);
  //gCommandTimer.Start();
  CLDevice *before = device_;
  //bool already_resolved = LocateMemOnDevice(mem_src_);
  bool already_resolved = ChangeDeviceToReadMem(mem_src_, device_);
  //gCommandTimer.Stop();
  if(before != device_)
  {
  	SNUCL_INFO("D2H Device changed from %p to %p\n", before, device_);
  }
  
  if(queue_ && queue_->IsAutoDeviceSelection()) 
  {
  	// FIXME: Change only if the queue properties say so
  //	cl_int err = queue_->set_latest_device(device_);
//	if(err != CL_SUCCESS) SNUCL_ERROR("Invalid Device Set!\n", 0);
  //	SNUCL_INFO("(After) Device in Resolve Read consistency: %p\n", queue_->device());
  	if(size_ == 0)
	{
		alreadyCompleted_ = true;
	}
  }
  
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfWriteMem() {
  bool already_resolved = true;
  bool write_all = false;
  switch (type_) {
    case CL_COMMAND_WRITE_BUFFER:
      write_all = (off_dst_ == 0 && size_ == mem_dst_->size());
      break;
    case CL_COMMAND_WRITE_IMAGE: {
      size_t* region = mem_dst_->GetImageRegion();
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 && region_[0] == region[0] &&
                   region_[1] == region[1] && region_[2] == region[2]);
      break;
    }
    case CL_COMMAND_WRITE_BUFFER_RECT:
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 &&
                   region_[0] * region_[1] * region_[2] == mem_dst_->size() &&
                   (dst_row_pitch_ == 0 || dst_row_pitch_ == region_[0]) &&
                   (dst_slice_pitch_ == 0 ||
                    dst_slice_pitch_ == region_[0] * region_[1]));
      break;
    case CL_COMMAND_FILL_BUFFER:
    case CL_COMMAND_FILL_IMAGE:
      write_all = true;
      break;
    default:
      SNUCL_ERROR("Unsupported command [%x]", type_);
      break;
  }
  if (!write_all)
    already_resolved &= LocateMemOnDevice(mem_dst_);
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfCopyMem() {
  CLDevice* source = device_;
  if (!ChangeDeviceToReadMem(mem_src_, source))
    return false;

  bool already_resolved = true;
  bool write_all = false;
  switch (type_) {
    case CL_COMMAND_COPY_BUFFER:
      write_all = (off_dst_ == 0 && size_ == mem_dst_->size());
      break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      write_all = (off_dst_ == 0 &&
                   region_[0] * region_[1] * region_[2] == mem_dst_->size());
      break;
    case CL_COMMAND_COPY_IMAGE:
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE: {
      size_t* region = mem_dst_->GetImageRegion();
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 && region_[0] == region[0] &&
                   region_[1] == region[1] && region_[2] == region[2]);
      break;
    }
    case CL_COMMAND_COPY_BUFFER_RECT:
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 &&
                   region_[0] * region_[1] * region_[2] == mem_dst_->size() &&
                   (dst_row_pitch_ == 0 || dst_row_pitch_ == region_[0]) &&
                   (dst_slice_pitch_ == 0 ||
                    dst_slice_pitch_ == region_[0] * region_[1]));
      break;
    default:
      SNUCL_ERROR("Unsupported command [%x]", type_);
      break;
  }
  if (!write_all)
    already_resolved &= LocateMemOnDevice(mem_dst_);

  bool use_read, use_write, use_copy, use_send, use_recv, use_rcopy;
  bool alloc_ptr, use_host_ptr;
  GetCopyPattern(source, device_, use_read, use_write, use_copy, use_send,
                 use_recv, use_rcopy, alloc_ptr, use_host_ptr /* unused */);

  if(use_copy)
  {
  	cl_mem mem_src_dev_specific_ptr = (cl_mem)mem_src_->GetDevSpecific(source);
  	cl_mem mem_dst_dev_specific_ptr = (cl_mem)mem_dst_->GetDevSpecific(device_);
	mem_src_dev_specific_ = mem_src_dev_specific_ptr;//->c_obj;
	mem_dst_dev_specific_ = mem_dst_dev_specific_ptr;//->c_obj;
  }
  void* ptr = NULL;
  if (alloc_ptr) {
    size_t size;
    switch (type_) {
      case CL_COMMAND_COPY_BUFFER:
        size = size_;
        break;
      case CL_COMMAND_COPY_IMAGE:
      case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      case CL_COMMAND_COPY_BUFFER_RECT:
        size = mem_dst_->GetRegionSize(region_);
        break;
      case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
        size = mem_src_->GetRegionSize(region_);
        break;
      default:
        SNUCL_ERROR("Unsupported command [%x]", type_);
        break;
    }
    //ptr = memalign(4096, size);
	//ptr = device_->AllocHostMem(mem_dst_);
	ptr = mem_dst_->GetDevSpecificHostPtr(device_);
	//SNUCL_INFO("CopyMem Mapped Host Ptr: %p\n", ptr);
  }

  CLCommand* read = NULL;
  switch (type_) {
    case CL_COMMAND_COPY_BUFFER:
      if (use_read || use_send) {
        read = CreateReadBuffer(context_, source, NULL, mem_src_, off_src_,
                                size_, ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_BUFFER;
        ptr_ = ptr;
      }
      break;
    case CL_COMMAND_COPY_IMAGE:
      if (use_read || use_send) {
        read = CreateReadImage(context_, source, NULL, mem_src_, src_origin_,
                               region_, 0, 0, ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_IMAGE;
        src_row_pitch_ = 0;
        src_slice_pitch_ = 0;
        ptr_ = ptr;
      }
      break;
    case CL_COMMAND_COPY_IMAGE_TO_BUFFER:
      if (use_read || use_send) {
        read = CreateReadImage(context_, source, NULL, mem_src_, src_origin_,
                               region_, 0, 0, ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_BUFFER;
        size_ = mem_src_->GetRegionSize(region_);
        ptr_ = ptr;
      }
      break;
    case CL_COMMAND_COPY_BUFFER_TO_IMAGE:
      if (use_read || use_send) {
        read = CreateReadBuffer(context_, source, NULL, mem_src_, off_src_,
                                mem_src_->GetRegionSize(region_), ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_IMAGE;
        src_row_pitch_ = 0;
        src_slice_pitch_ = 0;
        ptr_ = ptr;
      }
      break;
    case CL_COMMAND_COPY_BUFFER_RECT:
      if (use_read || use_send) {
        size_t host_origin[3] = {0, 0, 0};
        read = CreateReadBufferRect(context_, source, NULL, mem_src_,
                                    src_origin_, host_origin, region_,
                                    src_row_pitch_, src_slice_pitch_, 0, 0,
                                    ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_BUFFER_RECT;
        src_origin_[0] = 0;
        src_origin_[1] = 0;
        src_origin_[2] = 0;
        src_row_pitch_ = 0;
        src_slice_pitch_ = 0;
        ptr_ = ptr;
      }
      break;
    default:
      SNUCL_ERROR("Unsupported command [%x]", type_);
      break;
  }
  if (use_send) {
    read->AnnotateDestinationNode(device_->node_id());
    read->SetAsPartialCommand(this);
  }
  if (use_recv)
    AnnotateSourceNode(source->node_id());
  if (use_rcopy) {
    AnnotateSourceDevice(source);
    AnnotateDestinationDevice(device_);
  }
  if (alloc_ptr)
    temp_buf_ = ptr;

  if (use_read && use_write) {
    CLEvent* last_event = read->ExportEvent();
    AddWaitEvent(last_event);
    last_event->Release();
    already_resolved = false;
  }
  if (read != NULL)
    read->Submit();
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfMap() {
  bool already_resolved = LocateMemOnDevice(mem_src_);
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfUnmap() {
  bool already_resolved = LocateMemOnDevice(mem_src_);
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfBroadcast() {
  bool already_resolved = true;
  bool write_all = (off_dst_ == 0 && size_ == mem_dst_->size());
  if (!(off_dst_ == 0 && size_ == mem_dst_->size()))
    already_resolved &= LocateMemOnDevice(mem_dst_);
  // To be reimplemented
  AnnotateSourceDevice(mem_src_->FrontLatest());
  consistency_resolved_ = true;
  return already_resolved;
}

bool CLCommand::ResolveConsistencyOfAlltoAll() {
  bool already_resolved = true;
  already_resolved &= LocateMemOnDevice(mem_src_);
  already_resolved &= LocateMemOnDevice(mem_dst_);
  consistency_resolved_ = true;
  return already_resolved;
}

void CLCommand::UpdateConsistencyOfLaunchKernel() {
  for (map<cl_uint, CLKernelArg*>::iterator it = kernel_args_->begin();
       it != kernel_args_->end();
       ++it) {
    CLKernelArg* arg = it->second;
    if (arg->mem != NULL)
      AccessMemOnDevice(arg->mem, arg->mem->IsWritable());
  }
}

void CLCommand::UpdateConsistencyOfLaunchNativeKernel() {
  for (cl_uint i = 0; i < num_mem_objects_; i++)
    AccessMemOnDevice(mem_list_[i], true);
}

void CLCommand::UpdateConsistencyOfReadMem() {
  AccessMemOnDevice(mem_src_, false);
}

void CLCommand::UpdateConsistencyOfWriteMem() {
  AccessMemOnDevice(mem_dst_, true);
}

void CLCommand::UpdateConsistencyOfCopyMem() {
  AccessMemOnDevice(mem_dst_, true);
}

void CLCommand::UpdateConsistencyOfMap() {
  AccessMemOnDevice(mem_src_, false);
}

void CLCommand::UpdateConsistencyOfUnmap() {
  AccessMemOnDevice(mem_src_, true);
}

void CLCommand::UpdateConsistencyOfBroadcast() {
  AccessMemOnDevice(mem_dst_, true);
}

void CLCommand::UpdateConsistencyOfAlltoAll() {
  AccessMemOnDevice(mem_dst_, true);
}

void CLCommand::GetCopyPattern(CLDevice* dev_src, CLDevice* dev_dst,
                               bool& use_read, bool& use_write, bool& use_copy,
                               bool& use_send, bool& use_recv, bool& use_rcopy,
                               bool& alloc_ptr, bool& use_host_ptr) {
  /*
   * GetCopyPattern() decides a method to copy a memory object in dev_src to a
   * memory object in dev_dst. Possible results are as follows:
   *
   * (1) dev_dst -> Copy -> dev_dst (dev_src == dev_dst)
   * (2) host pointer -> Write -> dev_dst
   * (3) dev_src -> Read -> a temporary buffer in host -> Write -> dev_dst
   * (4) dev_src -> ClusterDriver -> dev_dst
   * (5) dev_src -> Read -> MPI_Send -> MPI_Recv -> Write -> dev_dst
   *
   * No.  Required Commands                     Intermediate Buffer
   *      read  write  copy  send  recv  rcopy  alloc  use_host
   * (1)               TRUE
   * (2)        TRUE                                   TRUE
   * (3)  TRUE  TRUE                            TRUE
   * (4)                                 TRUE
   * (5)                     TRUE  TRUE
   */

  use_read = false;
  use_write = false;
  use_copy = false;
  use_send = false;
  use_recv = false;
  use_rcopy = false;
  alloc_ptr = false;
  use_host_ptr = false;

  if((dev_src->context() == dev_dst->context())
  				&& (dev_src->context() != NULL))
  {
	  use_copy = true;
  }
  else
  {
  if (dev_src == dev_dst) { // (1)
    use_copy = true;
  } else if (dev_src == LATEST_HOST) { // (2)
    use_write = true;
    use_host_ptr = true;
  } else if (dev_src->node_id() == dev_dst->node_id()) {
    if (dev_src->node_id() == 0) { // (3)
      use_read = true;
      use_write = true;
      alloc_ptr = true;
    } else { // (4)
      use_rcopy = true;
    }
  } else { // (5)
    use_send = true;
    use_recv = true;
  }
  }

  //SNUCL_INFO("Clone Mem: read: %d, write: %d, alloc ptr: %d copy: %d, use host ptr: %d\n", 
  //	use_read, use_write, alloc_ptr, 
 // 	use_copy, use_host_ptr);
}

static void CL_CALLBACK IssueCommandCallback(cl_event event, cl_int status,
                                             void* user_data) {
  CLCommand* command = (CLCommand*)user_data;
  command->Submit();
}

CLEvent* CLCommand::CloneMem(CLDevice* dev_src, CLDevice* dev_dst,
                             CLMem* mem) {
  bool use_read, use_write, use_copy, use_send, use_recv, use_rcopy;
  bool alloc_ptr, use_host_ptr;
  GetCopyPattern(dev_src, dev_dst, use_read, use_write, use_copy, use_send,
                 use_recv, use_rcopy, alloc_ptr, use_host_ptr);

  void* ptr = NULL;
  if(use_copy)
  {
  	cl_mem mem_src_dev_specific_ptr = (cl_mem)mem->GetDevSpecific(dev_src);
  	cl_mem mem_dst_dev_specific_ptr = (cl_mem)mem->GetDevSpecific(dev_dst);
	mem_src_dev_specific_ = mem_src_dev_specific_ptr;//->c_obj;
	mem_dst_dev_specific_ = mem_dst_dev_specific_ptr;//->c_obj;
	//cl_mem foo = mem_src_dev_specific_->st_obj();
  }
  //SNUCL_INFO("Cloning Mem: %p\n", mem);
  if (alloc_ptr)
  {
    //gCommandTimer.Start();
    //ptr = memalign(4096, mem->size());
	ptr = mem->GetDevSpecificHostPtr(dev_dst);
	//ptr = dev_dst->AllocHostMem(mem);
	//SNUCL_INFO("Mapped Host Ptr: %p\n", ptr);
    //gCommandTimer.Stop();
	//SNUCL_INFO("Mapped Host Ptr Time: %g sec\n", gCommandTimer.CurrentElapsed());
  }
  if (use_host_ptr)
    ptr = mem->GetHostPtr();

  CLCommand* read = NULL;
  CLCommand* write = NULL;
  CLCommand* copy = NULL;
  if (mem->IsImage()) {
    size_t origin[3] = {0, 0, 0};
    size_t* region = mem->GetImageRegion();
    if (use_read || use_send)
      read = CreateReadImage(context_, dev_src, NULL, mem, origin, region, 0,
                             0, ptr);
    if (use_write || use_recv)
      write = CreateWriteImage(context_, dev_dst, NULL, mem, origin, region, 0,
                               0, ptr);
    if (use_copy || use_rcopy)
      copy = CreateCopyImage(context_, dev_dst, NULL, mem, mem, origin, origin,
                             region);
  } else {
    if (use_read || use_send)
      read = CreateReadBuffer(context_, dev_src, NULL, mem, 0, mem->size(),
                              ptr);
    if (use_write || use_recv)
      write = CreateWriteBuffer(context_, dev_dst, NULL, mem, 0, mem->size(),
                                ptr);
    if (use_copy || use_rcopy)
      copy = CreateCopyBuffer(context_, dev_dst, NULL, mem, mem, mem_src_dev_specific_, mem_dst_dev_specific_, 0, 0,
                              mem->size());
  }
  if (use_send) {
    read->AnnotateDestinationNode(dev_dst->node_id());
    read->SetAsPartialCommand(write);
  }
  if (use_recv)
    write->AnnotateSourceNode(dev_src->node_id());
  if (use_rcopy) {
    copy->AnnotateSourceDevice(dev_src);
    copy->AnnotateDestinationDevice(dev_dst);
  }
  if (alloc_ptr)
    write->temp_buf_ = ptr;

  CLEvent* last_event = NULL;
  if (copy != NULL)
    last_event = copy->ExportEvent();
  else
    last_event = write->ExportEvent();

  if (use_read && use_write) {
    read->event_->AddCallback(new EventCallback(IssueCommandCallback, write,
                                                CL_COMPLETE));
    write = NULL;
  }
  if (read != NULL)
    read->Submit();
  if (write != NULL)
    write->Submit();
  if (copy != NULL)
    copy->Submit();
  mem->AddLatest(dev_dst);

  return last_event;
}

bool CLCommand::LocateMemOnDevice(CLMem* mem) {
  if (mem->HasLatest(device_) || mem->EmptyLatest())
    return true;
	gCommandTimer.Start();
  CLDevice* source = mem->GetNearestLatest(device_);
  CLEvent* last_event = CloneMem(source, device_, mem);
  AddWaitEvent(last_event);
  last_event->Release();
	gCommandTimer.Stop();
	SNUCL_INFO("Mem moved from %p->%p Time: %g sec\n", 
		source, device_, gCommandTimer.CurrentElapsed());
	//gCommandTimer.PrintCurrent("Resolve Mem Location Overhead");
  return false;
}

void CLCommand::AccessMemOnDevice(CLMem* mem, bool write) {
  if (write)
    mem->SetLatest(device_);
  else
    mem->AddLatest(device_);
}

bool CLCommand::ChangeDeviceToReadMem(CLMem* mem, CLDevice*& device) {
  if (mem->HasLatest(device) || mem->EmptyLatest())
    return true;
  CLDevice* source = mem->GetNearestLatest(device);
  if (source == LATEST_HOST) {
    CLEvent* last_event = CloneMem(source, device, mem);
    AddWaitEvent(last_event);
    last_event->Release();
    return false;
  }
  device = source;
  return true;
}

CLCommand*
CLCommand::Clone() {
	CLCommand* command = new CLCommand(context(), device(),
			queue_, type());
	if (command == NULL) return NULL;
	command->kernel_ = kernel_;
	command->kernel_->Retain();
	command->work_dim_ = work_dim_;
	for (cl_uint i = 0; i < work_dim_; i++) {
		command->gwo_[i] = (gwo_ != NULL) ? gwo_[i] : 0;
		command->gws_[i] = gws_[i];
		command->lws_[i] = (lws_ != NULL) ? lws_[i] : ((gws_[i] % 4 == 0) ? 4 : 1);
		command->nwg_[i] = command->gws_[i] / command->lws_[i];
	}
	for (cl_uint i = work_dim_; i < 3; i++) {
		command->gwo_[i] = 0;
		command->gws_[i] = 1;
		command->lws_[i] = 1;
		command->nwg_[i] = 1;
	}
	command->kernel_args_ = kernel_->DuplicateArgs();
	return command;
}

CLCommand*
CLCommand::CreateReadBuffer(CLContext* context, CLDevice* device,
                            CLCommandQueue* queue, CLMem* buffer,
                            size_t offset, size_t size, void* ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_READ_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = buffer;
  command->mem_src_->Retain();
  command->off_src_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
  return command;
}

CLCommand*
CLCommand::CreateReadBufferRect(CLContext* context, CLDevice* device,
                                CLCommandQueue* queue, CLMem* buffer,
                                const size_t* buffer_origin,
                                const size_t* host_origin,
                                const size_t* region,
                                size_t buffer_row_pitch,
                                size_t buffer_slice_pitch,
                                size_t host_row_pitch, size_t host_slice_pitch,
                                void* ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_READ_BUFFER_RECT);
  if (command == NULL) return NULL;
  command->mem_src_ = buffer;
  command->mem_src_->Retain();
  memcpy(command->src_origin_, buffer_origin, sizeof(size_t) * 3);
  memcpy(command->dst_origin_, host_origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  command->src_row_pitch_ = buffer_row_pitch;
  command->src_slice_pitch_ = buffer_slice_pitch;
  command->dst_row_pitch_ = host_row_pitch;
  command->dst_slice_pitch_ = host_slice_pitch;
  command->ptr_ = ptr;
  return command;
}

CLCommand*
CLCommand::CreateWriteBuffer(CLContext* context, CLDevice* device,
                             CLCommandQueue* queue, CLMem* buffer,
                             size_t offset, size_t size, void* ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_WRITE_BUFFER);
  if (command == NULL) return NULL;
  command->mem_dst_ = buffer;
  command->mem_dst_->Retain();
  command->off_dst_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
  return command;
}

CLCommand*
CLCommand::CreateWriteBufferRect(CLContext* context, CLDevice* device,
                                 CLCommandQueue* queue, CLMem* buffer,
                                 const size_t* buffer_origin,
                                 const size_t* host_origin,
                                 const size_t* region,
                                 size_t buffer_row_pitch,
                                 size_t buffer_slice_pitch,
                                 size_t host_row_pitch,
                                 size_t host_slice_pitch, void* ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_WRITE_BUFFER_RECT);
  if (command == NULL) return NULL;
  command->mem_dst_ = buffer;
  command->mem_dst_->Retain();
  memcpy(command->src_origin_, host_origin, sizeof(size_t) * 3);
  memcpy(command->dst_origin_, buffer_origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  command->src_row_pitch_ = host_row_pitch;
  command->src_slice_pitch_ = host_slice_pitch;
  command->dst_row_pitch_ = buffer_row_pitch;
  command->dst_slice_pitch_ = buffer_slice_pitch;
  command->ptr_ = ptr;
  return command;
}

CLCommand*
CLCommand::CreateFillBuffer(CLContext* context, CLDevice* device,
                            CLCommandQueue* queue, CLMem* buffer,
                            const void* pattern, size_t pattern_size,
                            size_t offset, size_t size) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_FILL_BUFFER);
  if (command == NULL) return NULL;
  command->mem_dst_ = buffer;
  command->mem_dst_->Retain();
  command->pattern_ = malloc(pattern_size);
  memcpy(command->pattern_, pattern, pattern_size);
  command->pattern_size_ = pattern_size;
  command->off_dst_ = offset;
  command->size_ = size;
  return command;
}

CLCommand*
CLCommand::CreateCopyBuffer(CLContext* context, CLDevice* device,
                            CLCommandQueue* queue, CLMem* src_buffer,
                            CLMem* dst_buffer, 
							cl_mem src_buffer_dev_specific,
							cl_mem dst_buffer_dev_specific,
							size_t src_offset,
                            size_t dst_offset, size_t size) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_COPY_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = src_buffer;
  command->mem_src_->Retain();
  command->mem_dst_ = dst_buffer;
  command->mem_dst_->Retain();
  command->mem_src_dev_specific_ = src_buffer_dev_specific;
  command->mem_dst_dev_specific_ = dst_buffer_dev_specific;
  command->off_src_ = src_offset;
  command->off_dst_ = dst_offset;
  command->size_ = size;
  return command;
}

CLCommand*
CLCommand::CreateCopyBufferRect(CLContext* context, CLDevice* device,
                                CLCommandQueue* queue, CLMem* src_buffer,
                                CLMem* dst_buffer, const size_t* src_origin,
                                const size_t* dst_origin, const size_t* region,
                                size_t src_row_pitch, size_t src_slice_pitch,
                                size_t dst_row_pitch, size_t dst_slice_pitch) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_COPY_BUFFER_RECT);
  if (command == NULL) return NULL;
  command->mem_src_ = src_buffer;
  command->mem_src_->Retain();
  command->mem_dst_ = dst_buffer;
  command->mem_dst_->Retain();
  memcpy(command->src_origin_, src_origin, sizeof(size_t) * 3);
  memcpy(command->dst_origin_, dst_origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  command->src_row_pitch_ = src_row_pitch;
  command->src_slice_pitch_ = src_slice_pitch;
  command->dst_row_pitch_ = dst_row_pitch;
  command->dst_slice_pitch_ = dst_slice_pitch;
  return command;
}

CLCommand*
CLCommand::CreateReadImage(CLContext* context, CLDevice* device,
                           CLCommandQueue* queue, CLMem* image,
                           const size_t* origin, const size_t* region,
                           size_t row_pitch, size_t slice_pitch, void* ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_READ_IMAGE);
  if (command == NULL) return NULL;
  command->mem_src_ = image;
  command->mem_src_->Retain();
  memcpy(command->src_origin_, origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  command->dst_row_pitch_ = row_pitch;
  command->dst_slice_pitch_ = slice_pitch;
  command->ptr_ = ptr;
  return command;
}

CLCommand*
CLCommand::CreateWriteImage(CLContext* context, CLDevice* device,
                            CLCommandQueue* queue, CLMem* image,
                            const size_t* origin, const size_t* region,
                            size_t row_pitch, size_t slice_pitch, void* ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_WRITE_IMAGE);
  if (command == NULL) return NULL;
  command->mem_dst_ = image;
  command->mem_dst_->Retain();
  memcpy(command->dst_origin_, origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  command->src_row_pitch_ = row_pitch;
  command->src_slice_pitch_ = slice_pitch;
  command->ptr_ = ptr;
  return command;
}

CLCommand*
CLCommand::CreateFillImage(CLContext* context, CLDevice* device,
                           CLCommandQueue* queue, CLMem* image,
                           const void* fill_color, const size_t* origin,
                           const size_t* region) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_FILL_IMAGE);
  if (command == NULL) return NULL;
  command->mem_dst_ = image;
  command->mem_dst_->Retain();
  command->ptr_ = (void*)fill_color;
  memcpy(command->dst_origin_, origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  return command;
}

CLCommand*
CLCommand::CreateCopyImage(CLContext* context, CLDevice* device,
                           CLCommandQueue* queue, CLMem* src_image,
                           CLMem* dst_image, const size_t* src_origin,
                           const size_t* dst_origin, const size_t* region) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_COPY_IMAGE);
  if (command == NULL) return NULL;
  command->mem_src_ = src_image;
  command->mem_src_->Retain();
  command->mem_dst_ = dst_image;
  command->mem_dst_->Retain();
  memcpy(command->src_origin_, src_origin, sizeof(size_t) * 3);
  memcpy(command->dst_origin_, dst_origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  return command;
}

CLCommand*
CLCommand::CreateCopyImageToBuffer(CLContext* context, CLDevice* device,
                                   CLCommandQueue* queue, CLMem* src_image,
                                   CLMem* dst_buffer, const size_t* src_origin,
                                   const size_t* region, size_t dst_offset) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_COPY_IMAGE_TO_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = src_image;
  command->mem_src_->Retain();
  command->mem_dst_ = dst_buffer;
  command->mem_dst_->Retain();
  memcpy(command->src_origin_, src_origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  command->off_dst_ = dst_offset;
  return command;
}

CLCommand*
CLCommand::CreateCopyBufferToImage(CLContext* context, CLDevice* device,
                                   CLCommandQueue* queue, CLMem* src_buffer,
                                   CLMem* dst_image, size_t src_offset,
                                   const size_t* dst_origin,
                                   const size_t* region) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_COPY_BUFFER_TO_IMAGE);
  if (command == NULL) return NULL;
  command->mem_src_ = src_buffer;
  command->mem_src_->Retain();
  command->mem_dst_ = dst_image;
  command->mem_dst_->Retain();
  command->off_src_ = src_offset;
  memcpy(command->dst_origin_, dst_origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  return command;
}

CLCommand*
CLCommand::CreateMapBuffer(CLContext* context, CLDevice* device,
                           CLCommandQueue* queue, CLMem* buffer,
                           cl_map_flags map_flags, size_t offset, size_t size,
                           void* mapped_ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_MAP_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = buffer;
  command->mem_src_->Retain();
  command->map_flags_ = map_flags;
  command->off_src_ = offset;
  command->size_ = size;
  command->ptr_ = mapped_ptr;
  return command;
}

CLCommand*
CLCommand::CreateMapImage(CLContext* context, CLDevice* device,
                          CLCommandQueue* queue, CLMem* image,
                          cl_map_flags map_flags, const size_t* origin,
                          const size_t* region, void* mapped_ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_MAP_IMAGE);
  if (command == NULL) return NULL;
  command->mem_src_ = image;
  command->mem_src_->Retain();
  command->map_flags_ = map_flags;
  memcpy(command->src_origin_, origin, sizeof(size_t) * 3);
  memcpy(command->region_, region, sizeof(size_t) * 3);
  command->ptr_ = mapped_ptr;
  return command;
}

CLCommand*
CLCommand::CreateUnmapMemObject(CLContext* context, CLDevice* device,
                                CLCommandQueue* queue, CLMem* mem,
                                void* mapped_ptr) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_UNMAP_MEM_OBJECT);
  if (command == NULL) return NULL;
  command->mem_src_ = mem;
  command->mem_src_->Retain();
  command->ptr_ = mapped_ptr;
  return command;
}

CLCommand*
CLCommand::CreateMigrateMemObjects(CLContext* context, CLDevice* device,
                                   CLCommandQueue* queue,
                                   cl_uint num_mem_objects,
                                   const cl_mem* mem_list,
                                   cl_mem_migration_flags flags) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_MIGRATE_MEM_OBJECTS);
  if (command == NULL) return NULL;
  command->num_mem_objects_ = num_mem_objects;
  command->mem_list_ = (CLMem**)malloc(sizeof(CLMem*) * num_mem_objects);
  for (uint i = 0; i < num_mem_objects; i++) {
    command->mem_list_[i] = mem_list[i]->c_obj;
    command->mem_list_[i]->Retain();
  }
  command->migration_flags_ = flags;
}

CLCommand*
CLCommand::CreateNDRangeKernel(CLContext* context, CLDevice* device,
                               CLCommandQueue* queue, CLKernel* kernel,
                               cl_uint work_dim,
                               const size_t* global_work_offset,
                               const size_t* global_work_size,
                               const size_t* local_work_size) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_NDRANGE_KERNEL);
  if (command == NULL) return NULL;
  command->kernel_ = kernel;
  command->kernel_->Retain();
  command->work_dim_ = work_dim;
  for (cl_uint i = 0; i < work_dim; i++) {
    command->gwo_[i] = (global_work_offset != NULL) ? global_work_offset[i] :
                                                      0;
    command->gws_[i] = global_work_size[i];
    command->lws_[i] = (local_work_size != NULL) ? local_work_size[i] :
                           ((global_work_size[i] % 4 == 0) ? 4 : 1);
    command->nwg_[i] = command->gws_[i] / command->lws_[i];
  }
  for (cl_uint i = work_dim; i < 3; i++) {
    command->gwo_[i] = 0;
    command->gws_[i] = 1;
    command->lws_[i] = 1;
    command->nwg_[i] = 1;
  }
  command->kernel_args_ = kernel->ExportArgs();
  return command;
}

CLCommand*
CLCommand::CreateNativeKernel(CLContext* context, CLDevice* device,
                              CLCommandQueue* queue, void (*user_func)(void*),
                              void* args, size_t cb_args,
                              cl_uint num_mem_objects, const cl_mem* mem_list,
                              const void** args_mem_loc) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_NATIVE_KERNEL);
  if (command == NULL) return NULL;
  command->user_func_ = user_func;
  if (args != NULL) {
    command->native_args_ = malloc(cb_args);
    memcpy(command->native_args_, args, cb_args);
    command->size_ = cb_args;
  } else {
    command->size_ = 0;
  }
  if (num_mem_objects > 0) {
    command->num_mem_objects_ = num_mem_objects;
    command->mem_list_ = (CLMem**)malloc(sizeof(CLMem*) * num_mem_objects);
    command->mem_offsets_ = (ptrdiff_t*)malloc(sizeof(ptrdiff_t) *
                                               num_mem_objects);
    for (cl_uint i = 0; i < num_mem_objects; i++) {
      command->mem_list_[i] = mem_list[i]->c_obj;
      command->mem_list_[i]->Retain();
      command->mem_offsets_[i] = (size_t)args_mem_loc[i] - (size_t)args;
    }
  }
  return command;
}

CLCommand*
CLCommand::CreateMarker(CLContext* context, CLDevice* device,
                        CLCommandQueue* queue) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_MARKER);
  return command;
}

CLCommand*
CLCommand::CreateBarrier(CLContext* context, CLDevice* device,
                         CLCommandQueue* queue) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_BARRIER);
  return command;
}

CLCommand*
CLCommand::CreateWaitForEvents(CLContext* context, CLDevice* device,
                               CLCommandQueue* queue) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_WAIT_FOR_EVENTS);
  return command;
}

CLCommand*
CLCommand::CreateBuildProgram(CLDevice* device, CLProgram* program,
                              CLProgramSource* source, CLProgramBinary* binary,
                              const char* options) {
  CLCommand* command = new CLCommand(program->context(), device, NULL,
                                     CL_COMMAND_BUILD_PROGRAM);
  if (command == NULL) return NULL;
  command->program_ = program;
  command->program_->Retain();
  command->source_ = source;
  command->binary_ = binary;
  command->options_ = options;
  return command;
}

CLCommand*
CLCommand::CreateCompileProgram(CLDevice* device, CLProgram* program,
                                CLProgramSource* source, const char* options,
                                vector<CLProgramSource*>& headers) {
  CLCommand* command = new CLCommand(program->context(), device, NULL,
                                     CL_COMMAND_COMPILE_PROGRAM);
  if (command == NULL) return NULL;
  command->program_ = program;
  command->program_->Retain();
  command->source_ = source;
  command->options_ = options;
  if (headers.empty()) {
    command->size_ = 0;
  } else {
    command->size_ = headers.size();
    command->headers_ = (CLProgramSource**)malloc(sizeof(CLProgramSource*) *
                                                  headers.size());
    for (size_t i = 0; i < headers.size(); i++)
      command->headers_[i] = headers[i];
  }
  return command;
}

CLCommand*
CLCommand::CreateLinkProgram(CLDevice* device, CLProgram* program,
                             vector<CLProgramBinary*>& binaries,
                             const char* options) {
  CLCommand* command = new CLCommand(program->context(), device, NULL,
                                     CL_COMMAND_LINK_PROGRAM);
  if (command == NULL) return NULL;
  command->program_ = program;
  command->program_->Retain();
  command->size_ = binaries.size();
  command->link_binaries_ =
      (CLProgramBinary**)malloc(sizeof(CLProgramBinary*) * binaries.size());
  for (size_t i = 0; i < binaries.size(); i++)
    command->link_binaries_[i] = binaries[i];
  command->options_ = options;
  return command;
}

CLCommand*
CLCommand::CreateCustom(CLContext* context, CLDevice* device,
                        CLCommandQueue* queue, void (*custom_function)(void*),
                        void* custom_data) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_CUSTOM);
  if (command == NULL) return NULL;
  command->custom_function_ = custom_function;
  command->custom_data_ = custom_data;
  return command;
}

CLCommand*
CLCommand::CreateNop(CLContext* context, CLDevice* device,
                     CLCommandQueue* queue) {
  CLCommand* command = new CLCommand(context, device, queue, CL_COMMAND_NOP);
  return command;
}

CLCommand*
CLCommand::CreateBroadcastBuffer(CLContext* context, CLDevice* device,
                                 CLCommandQueue* queue, CLMem* src_buffer,
                                 CLMem* dst_buffer, size_t src_offset,
                                 size_t dst_offset, size_t cb) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_BROADCAST_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = src_buffer;
  command->mem_dst_ = dst_buffer;
  command->off_src_ = src_offset;
  command->off_dst_ = dst_offset;
  command->size_ = cb;
  return command;
}

CLCommand*
CLCommand::CreateAlltoAllBuffer(CLContext* context, CLDevice* device,
                                CLCommandQueue* queue, CLMem* src_buffer,
                                CLMem* dst_buffer, size_t src_offset,
                                size_t dst_offset, size_t cb) {
  CLCommand* command = new CLCommand(context, device, queue,
                                     CL_COMMAND_ALLTOALL_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = src_buffer;
  command->mem_dst_ = dst_buffer;
  command->off_src_ = src_offset;
  command->off_dst_ = dst_offset;
  command->size_ = cb;
  return command;
}
