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

#include "CLPlatform.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <vector>
#include <pthread.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include "Callbacks.h"
#include "CLContext.h"
#include "CLDevice.h"
#include "CLDispatch.h"
#include "CLIssuer.h"
#include "CLObject.h"
#include "CLScheduler.h"
#include "Structs.h"
#include "Utils.h"
#include <stdio.h>
#ifdef CPU_PLATFORM
#include "cpu/CPUDevice.h"
#define SCHEDULER_SINGLE
#define ISSUER_PER_DEVICE_BLOCKING
#endif

#ifdef LEGACY_PLATFORM
#include "legacy/LegacyDevice.h"
#define SCHEDULER_SINGLE
#define ISSUER_PER_DEVICE_BLOCKING
#endif

#ifdef CLUSTER_PLATFORM
#include <mpi.h>
#include "legacy/LegacyDevice.h"
#include "cluster/ClusterDevice.h"
#define SCHEDULER_CLUSTER
#define ISSUER_CLUSTER
#endif

using namespace std;

CLPlatform::CLPlatform() {
  profile_ = "FULL_PROFILE";
  version_ = "OpenCL 1.2 rev01";
#if defined(CPU_PLATFORM)
  name_ = "SnuCL CPU";
#elif defined(LEGACY_PLATFORM)
  name_ = "SnuCL Single";
#elif defined(CLUSTER_PLATFORM)
  name_ = "SnuCL Cluster";
#else
  name_ = "SnuCL";
#endif
  vendor_ = "Seoul National University";
  extensions_ = "";
  suffix_ = "SnuCL";
#if defined(CPU_PLATFORM)
  default_device_type_ = CL_DEVICE_TYPE_CPU;
#elif defined(LEGACY_PLATFORM)
  default_device_type_ = CL_DEVICE_TYPE_GPU;
#else
  default_device_type_ = CL_DEVICE_TYPE_GPU;
#endif
#ifdef CLUSTER_PLATFORM
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  is_host_ = (rank == 0);
#endif
  hwloc_topology_init(&topology_);
  hwloc_topology_set_flags(topology_, HWLOC_TOPOLOGY_FLAG_IO_DEVICES);
  hwloc_topology_load(topology_);
  hwloc_initialized_ = 1;

  pthread_mutex_init(&mutex_devices_, NULL);
  pthread_mutex_init(&mutex_issuers_, NULL);
}

CLPlatform::~CLPlatform() {
  if (hwloc_initialized_ == 1)
  {
  	hwloc_topology_destroy(topology_);
	hwloc_initialized_ = 0;
  }
  hosts_.clear();
  unsigned int device_id = 0;
  unsigned int host_id = 0;
  for(host_id = 0; host_id < hosts_.size(); host_id++)
  {
  	devices_hosts_distances_[host_id].clear();
  }
  devices_hosts_distances_.clear();
  for(device_id = 0; device_id < devices_.size(); device_id++)
  {
  	devices_devices_distances_[device_id].clear();
  }
  devices_devices_distances_.clear();
  devices_compute_perf_.clear();
  devices_memory_perf_.clear();
  devices_lmemory_perf_.clear();

  for (vector<CLDevice*>::iterator it = devices_.begin();
       it != devices_.end();
       ++it) {
    delete (*it);
  }
  for (vector<CLScheduler*>::iterator it = schedulers_.begin();
       it != schedulers_.end();
       ++it) {
    delete (*it);
  }
  for (vector<CLIssuer*>::iterator it = issuers_.begin();
       it != issuers_.end();
       ++it) {
    delete (*it);
  }
  pthread_mutex_destroy(&mutex_devices_);
  pthread_mutex_destroy(&mutex_issuers_);
}

void CLPlatform::Init() {
#if defined(SCHEDULER_SINGLE)
  InitSchedulers(1, false);
#elif defined(SCHEDULER_MULTIPLE)
  InitSchedulers(4, true);
#elif defined(SCHEDULER_CLUSTER)
  if (is_host_) // SCHEDULER_MULTIPLE
    InitSchedulers(4, true);
  else // SCHEDULER_SINGLE
    InitSchedulers(1, false);
#endif

#if defined(ISSUER_GLOBAL)
  AddIssuer(new CLIssuer(false));
#elif defined(ISSUER_CLUSTER)
  if (is_host_) // ISSUER_GLOBAL
    AddIssuer(new CLIssuer(false));
#endif

#if defined(CPU_PLATFORM)
  CPUDevice::CreateDevices();
#elif defined(LEGACY_PLATFORM)
  LegacyDevice::CreateDevices();
#elif defined(CLUSTER_PLATFORM)
  if (is_host_)
    ClusterDevice::CreateDevices();
  else
    LegacyDevice::CreateDevices();
#endif

#ifdef SNUCL_DEBUG
  unsigned int num_cpu = 0;
  unsigned int num_gpu = 0;
  unsigned int num_accelerator = 0;
  unsigned int num_custom = 0;

  unsigned int device_id = 0;

  for (vector<CLDevice*>::const_iterator it = devices_.begin();
       it != devices_.end();
       ++it) {
    CLDevice* device = *it;
    switch (device->type()) {
      case CL_DEVICE_TYPE_CPU:
        num_cpu++;
        break;
      case CL_DEVICE_TYPE_GPU:
        num_gpu++;
        break;
      case CL_DEVICE_TYPE_ACCELERATOR:
        num_accelerator++;
        break;
      case CL_DEVICE_TYPE_CUSTOM:
        num_custom++;
        break;
      default:
        SNUCL_ERROR("Unsupported device type [%lx]", device->type());
    }
  }

  InitDeviceMetrics();
  SNUCL_INFO("SnuCL platform has been initialized.", 0);
  SNUCL_INFO("Total %u devices (%u CPUs, %u GPUs, %u accelerators, "
             "and %u custom devices) are in the platform.",
             devices_.size(), num_cpu, num_gpu, num_accelerator, num_custom);
#endif // SNUCL_DEBUG
}

void CLPlatform::InitDeviceMetrics()
{
  // Find number of "hosts" for the system.
  // For now, hosts = number of CPU sockets (as per hwloc)
  // FIXME: Verify if the above logic works for all modes of SnuCL, 
  // i.e. Cluster, CPU and Single
  int n_cpu_sockets = hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_NODE); 
  printf("Number of CPU sockets discovered: %d\n", n_cpu_sockets);
  hosts_.clear();
  hosts_.resize(n_cpu_sockets);
  unsigned int host_id = 0;
  SNUCL_INFO("(Much before) Number of hosts: %u\n", hosts_.size());
  for(host_id = 0; host_id < n_cpu_sockets; host_id++)
  {
	hosts_[host_id] = hwloc_get_obj_by_type(topology_, HWLOC_OBJ_NODE, host_id);
  	//hosts_[host_id] = cpuset;
    SNUCL_INFO("(During) Number of hosts: %u\n", hosts_.size());
	SNUCL_INFO("Hosts hwloc ptr[%d]: %p\n", host_id, hosts_[host_id]);
  }
  // FIXME!!!! devices_hosts_distances_ should be a HxD matrix where each
  // row corresponds to a host CPUset and each column represents a device
  devices_hosts_distances_.resize(hosts_.size());
  for(host_id = 0; host_id < hosts_.size(); host_id++)
  {
  	devices_hosts_distances_[host_id].resize(devices_.size());
  	for(int next_device_id = 0; next_device_id < devices_.size(); next_device_id++)
	{
		// TODO: Arbitrary values are filled now, but need to be 
		// replaced with actual bandwidth numbers
		// bandwidth = getH2DBandwidth(hosts_[host_id], devices_[next_device_id])
		int num_entities = devices_.size();
  		devices_hosts_distances_[host_id][next_device_id] = (host_id + next_device_id) % num_entities;
	}
  }

  unsigned int device_id = 0;
  devices_devices_distances_.resize(devices_.size());
  device_id = 0;
  // We need separate loops because we will be initializing in a triangular fashion
  for(device_id = 0; device_id < devices_.size(); device_id++)
  {
  	devices_devices_distances_[device_id].resize(devices_.size());
  }
  for(device_id = 0; device_id < devices_.size(); device_id++)
  {
  	devices_devices_distances_[device_id][device_id] = numeric_limits<double>::max();
  	for(int next_device_id = device_id+1; next_device_id < devices_.size(); next_device_id++)
	{
		// TODO: Arbitrary values are filled now, but need to be 
		// replaced with actual bandwidth numbers
  		devices_devices_distances_[device_id][next_device_id] = 10;
  		devices_devices_distances_[next_device_id][device_id] = 1;
	}
  }

  // TODO: Move the below individual metrics to be withing CLDevice
  // itself. They should all be populated as and when the device obj
  // get created.
  devices_compute_perf_.resize(devices_.size());
  for(device_id = 0; device_id < devices_.size(); device_id++)
  {
  	devices_compute_perf_[device_id] = devices_.size() - device_id;
  }

  devices_memory_perf_.resize(devices_.size());
  for(device_id = 0; device_id < devices_.size(); device_id++)
  {
  	devices_memory_perf_[device_id] = device_id;
  }

  devices_lmemory_perf_.resize(devices_.size());
  for(device_id = 0; device_id < devices_.size(); device_id++)
  {
  	devices_lmemory_perf_[device_id] = device_id;
  }
}

cl_int CLPlatform::GetPlatformInfo(cl_platform_info param_name,
                                   size_t param_value_size,
                                   void* param_value,
                                   size_t* param_value_size_ret) {
  switch (param_name) {
    GET_OBJECT_INFO_S(CL_PLATFORM_PROFILE, profile_);
    GET_OBJECT_INFO_S(CL_PLATFORM_VERSION, version_);
    GET_OBJECT_INFO_S(CL_PLATFORM_NAME, name_);
    GET_OBJECT_INFO_S(CL_PLATFORM_VENDOR, vendor_);
    GET_OBJECT_INFO_S(CL_PLATFORM_EXTENSIONS, extensions_);
    GET_OBJECT_INFO_S(CL_PLATFORM_ICD_SUFFIX_KHR, suffix_);
    default: return CL_INVALID_VALUE;
  }
  return CL_SUCCESS;
}

cl_int CLPlatform::GetDeviceIDs(cl_device_type device_type,
                                cl_uint num_entries, cl_device_id* devices,
                                cl_uint* num_devices) {
  if (device_type == CL_DEVICE_TYPE_DEFAULT)
    device_type = default_device_type_;
  if (device_type == CL_DEVICE_TYPE_ALL)
    device_type ^= CL_DEVICE_TYPE_CUSTOM;

  vector<CLDevice*> all_devices;
  pthread_mutex_lock(&mutex_devices_);
  all_devices = devices_;
  pthread_mutex_unlock(&mutex_devices_);

  cl_uint num_devices_ret = 0;
  for (vector<CLDevice*>::iterator it = all_devices.begin();
       it != all_devices.end();
       ++it) {
    CLDevice* device = *it;
    if (device->IsSubDevice()) continue;

    if (device->type() & device_type) {
      if (devices != NULL && num_devices_ret < num_entries) {
        devices[num_devices_ret] = device->st_obj();
      }
      num_devices_ret++;
    }
  }
  if (num_devices) {
    if (num_entries > 0 && num_devices_ret > num_entries)
      *num_devices = num_entries;
    else
      *num_devices = num_devices_ret;
  }

  if (num_devices_ret == 0)
    return CL_DEVICE_NOT_FOUND;
  else
    return CL_SUCCESS;
}

CLContext* CLPlatform::CreateContextFromDevices(
    const cl_context_properties* properties, cl_uint num_devices,
    const cl_device_id* devices, ContextErrorNotificationCallback* callback,
    cl_int* err) {
  *err = CL_SUCCESS;
  size_t num_properties = CheckContextProperties(properties, err);
  if (*err != CL_SUCCESS) return NULL;

  vector<CLDevice*> selected_devices;
  selected_devices.reserve(num_devices);

  vector<size_t> selected_devices_indices;
  selected_devices_indices.reserve(num_devices);

  for (cl_uint i = 0; i < num_devices; i++) {
    if (devices[i] == NULL) {
      *err = CL_INVALID_DEVICE;
      return NULL;
    }
    if (!devices[i]->c_obj->IsAvailable()) {
      *err = CL_DEVICE_NOT_AVAILABLE;
      return NULL;
    }
    selected_devices.push_back(devices[i]->c_obj);
	// TODO
	// Find the index of device[i]->c_obj in devices_
	std::vector<CLDevice*>::iterator this_device_iter = std::find(devices_.begin(), devices_.end(), devices[i]->c_obj);
	CLDevice *this_device = *this_device_iter;
	if(this_device_iter == devices_.end())
	{
      *err = CL_INVALID_DEVICE;
      return NULL;
	}
	size_t this_device_index = 0; 
	this_device_index = std::distance(devices_.begin(), this_device_iter);
	fprintf(stderr, "Adding device to ctx: ID: %d, Type: %d\n", this_device_index, this_device->type());
	selected_devices_indices.push_back(this_device_index);
  }

  SNUCL_INFO("(Before) Number of hosts: %u\n", hosts_.size());
  CLContext* context = new CLContext(selected_devices, num_properties,
                                     properties,
									 hosts_,
									 devices_devices_distances_,
									 devices_hosts_distances_,
									 devices_compute_perf_,
									 devices_memory_perf_,
									 devices_lmemory_perf_,
									 selected_devices_indices);
  if (context == NULL) {
    *err = CL_OUT_OF_HOST_MEMORY;
    return NULL;
  }

  if (callback != NULL)
    context->SetErrorNotificationCallback(callback);
  
  return context;
}

CLContext* CLPlatform::CreateContextFromType(
    const cl_context_properties* properties, cl_device_type device_type,
    ContextErrorNotificationCallback* callback, cl_int* err) {
  *err = CL_SUCCESS;
  size_t num_properties = CheckContextProperties(properties, err);
  if (*err != CL_SUCCESS) return NULL;

  if (device_type == CL_DEVICE_TYPE_DEFAULT)
    device_type = default_device_type_;
  if (device_type == CL_DEVICE_TYPE_ALL)
    device_type ^= CL_DEVICE_TYPE_CUSTOM;

  vector<CLDevice*> all_devices;
  pthread_mutex_lock(&mutex_devices_);
  all_devices = devices_;
  pthread_mutex_unlock(&mutex_devices_);

  vector<CLDevice*> selected_devices;
  selected_devices.reserve(all_devices.size());

  vector<size_t> selected_devices_indices;
  selected_devices_indices.reserve(all_devices.size());

  size_t device_exist = false;
  for (vector<CLDevice*>::iterator it = all_devices.begin();
       it != all_devices.end();
       ++it) {
    CLDevice* device = *it;
    if (device->IsSubDevice()) continue;

	if (device->type() & device_type) {
		device_exist = true;
		if (device->IsAvailable())
			selected_devices.push_back(device);
		// TODO
		// Find the index of device[i]->c_obj in all_devices
		std::vector<CLDevice*>::iterator this_device_iter = std::find(all_devices.begin(), all_devices.end(), device);
		CLDevice *this_device = *this_device_iter;
		if(this_device_iter == all_devices.end())
		{
			*err = CL_INVALID_DEVICE;
			return NULL;
		}
		size_t this_device_index = 0; 
		this_device_index = std::distance(all_devices.begin(), this_device_iter);
		fprintf(stderr, "Adding device to ctx: ID: %d, Type: %d\n", this_device_index, this_device->type());
		selected_devices_indices.push_back(this_device_index);
	}
  }

  if (!device_exist) {
    *err = CL_DEVICE_NOT_FOUND;
    return NULL;
  }
  if (selected_devices.empty()) {
    *err = CL_DEVICE_NOT_AVAILABLE;
    return NULL;
  }

  SNUCL_INFO("(Before) Number of hosts: %u\n", hosts_.size());
  CLContext* context = new CLContext(selected_devices, num_properties,
                                     properties,
									 hosts_,
									 devices_devices_distances_,
									 devices_hosts_distances_,
									 devices_compute_perf_,
									 devices_memory_perf_,
									 devices_lmemory_perf_,
									 selected_devices_indices);
  if (context == NULL) {
    *err = CL_OUT_OF_HOST_MEMORY;
    return NULL;
  }

  if (callback != NULL)
    context->SetErrorNotificationCallback(callback);
  return context;
}

void CLPlatform::GetDevices(std::vector<CLDevice*>& devices) {
  pthread_mutex_lock(&mutex_devices_);
  devices = devices_;
  pthread_mutex_unlock(&mutex_devices_);
}

CLDevice* CLPlatform::GetFirstDevice() {
  pthread_mutex_lock(&mutex_devices_);
  CLDevice* device = devices_.front();
  pthread_mutex_unlock(&mutex_devices_);
  return device;
}

void CLPlatform::AddDevice(CLDevice* device) {
  pthread_mutex_lock(&mutex_devices_);
  devices_.push_back(device);
  pthread_mutex_unlock(&mutex_devices_);
#if defined(ISSUER_PER_DEVICE_BLOCKING)
  AddIssuer(new CLIssuer(device, true));
#elif defined(ISSUER_PER_DEVICE_NON_BLOCKING)
  AddIssuer(new CLIssuer(device, false));
#elif defined(ISSUER_GLOBAL)
  AddDeviceToFirstIssuer(device);
#elif defined(ISSUER_CLUSTER)
  if (is_host_) // ISSUER_GLOBAL
    AddDeviceToFirstIssuer(device);
  else // ISSUER_PER_DEVICE_BLOCKING
    AddIssuer(new CLIssuer(device, true));
#endif
}

void CLPlatform::RemoveDevice(CLDevice* device) {
#if defined(ISSUER_PER_DEVICE_BLOCKING) || \
    defined(ISSUER_PER_DEVICE_NON_BLOCKING)
  RemoveIssuerOfDevice(device);
#elif defined(ISSUER_GLOBAL)
  RemoveDeviceFromFirstIssuer(device);
#elif defined(ISSUER_CLUSTER)
  if (is_host_) // ISSUER_GLOBAL
    RemoveDeviceFromFirstIssuer(device);
  else // ISSUER_PER_DEVICE_BLOCKING
    RemoveIssuerOfDevice(device);
#endif
  pthread_mutex_lock(&mutex_devices_);
  devices_.erase(remove(devices_.begin(), devices_.end(), device),
                 devices_.end());
  pthread_mutex_unlock(&mutex_devices_);
}

CLScheduler* CLPlatform::AllocIdleScheduler() {
  // Round-robin manner
  static size_t next = 0;
  CLScheduler* scheduler = schedulers_[next];
  next = (next + 1) % schedulers_.size();
  return scheduler;
}

void CLPlatform::InvokeAllSchedulers() {
  for (vector<CLScheduler*>::iterator it = schedulers_.begin();
       it != schedulers_.end();
       ++it) {
    (*it)->Invoke();
  }
}

size_t CLPlatform::CheckContextProperties(
    const cl_context_properties* properties, cl_int* err) {
  if (properties == NULL) return 0;

  size_t idx = 0;
  bool set_platform = false;
  bool set_sync = false;
  while (properties[idx] > 0) {
    if (properties[idx] == CL_CONTEXT_PLATFORM) {
      if (set_platform) {
        *err = CL_INVALID_PROPERTY;
        return 0;
      }
      set_platform = true;
      if ((cl_platform_id)properties[idx + 1] != st_obj()) {
        *err = CL_INVALID_PLATFORM;
        return 0;
      }
      idx += 2;
    } else if (properties[idx] == CL_CONTEXT_INTEROP_USER_SYNC) {
      if (set_sync) {
        *err = CL_INVALID_PROPERTY;
        return 0;
      }
      set_sync = true;
    } else {
      *err = CL_INVALID_PROPERTY;
      return 0;
    }
  }
  return idx + 1;
}

void CLPlatform::InitSchedulers(unsigned int num_schedulers,
                                bool busy_waiting) {
  schedulers_.reserve(num_schedulers);
  for (int i = 0; i < num_schedulers; i++) {
    CLScheduler* scheduler = new CLScheduler(this, busy_waiting);
    schedulers_.push_back(scheduler);
    scheduler->Start();
  }
}

void CLPlatform::AddIssuer(CLIssuer* issuer) {
  pthread_mutex_lock(&mutex_issuers_);
  issuers_.push_back(issuer);
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->Start();
}

void CLPlatform::RemoveIssuerOfDevice(CLDevice* device) {
  pthread_mutex_lock(&mutex_issuers_);
  CLIssuer* issuer = NULL;
  for (vector<CLIssuer*>::iterator it = issuers_.begin();
       it != issuers_.end();
       ++it) {
    if ((*it)->GetFirstDevice() == device) {
      issuer = *it;
      issuers_.erase(it);
      break;
    }
  }
  pthread_mutex_unlock(&mutex_issuers_);
  if (issuer != NULL)
    delete issuer;
}

void CLPlatform::AddDeviceToFirstIssuer(CLDevice* device) {
  pthread_mutex_lock(&mutex_issuers_);
  CLIssuer* issuer = issuers_.front();
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->AddDevice(device);
}

void CLPlatform::RemoveDeviceFromFirstIssuer(CLDevice* device) {
  pthread_mutex_lock(&mutex_issuers_);
  CLIssuer* issuer = issuers_.front();
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->RemoveDevice(device);
}

CLPlatform* CLPlatform::singleton_ = NULL;
//CLPlatform CLPlatform::singleton_obj;

CLPlatform* CLPlatform::GetPlatform() {
  static CLPlatform singleton_obj;
  //singleton_ = &singleton_obj;
  //return singleton_;
  if (singleton_ == NULL) {
    //singleton_ = new CLPlatform();
    singleton_ = &singleton_obj;
    singleton_->Init();
  }
  return singleton_;
}
