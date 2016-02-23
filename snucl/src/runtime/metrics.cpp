#include "metrics.h"
#include "RealTimer.h"
#include <CLEvent.h>
#include <CLCommand.h>
#include <CLMem.h>
#define MEMCOPY_ITERATIONS  100
#define OPENCL_CHECK_ERR(err, str) \
	if (err != CL_SUCCESS) \
{ \
	fprintf(stderr, "CL Error %d: %s: %s\n", err, getOpenCLErrorCodeStr(err), str); \
}

cl_int CLFinish(CLCommandQueue *q)
{
  if (q == NULL) return CL_INVALID_COMMAND_QUEUE;
  CLCommand* command = CLCommand::CreateMarker(NULL, NULL, q);
  if (command == NULL) return CL_OUT_OF_HOST_MEMORY;

  CLEvent* blocking = command->ExportEvent();
  q->Enqueue(command);
  blocking->Wait();
  blocking->Release();

  return CL_SUCCESS;
}

cl_int CLCopyBuffer(CLCommandQueue *q, 
					CLMem *cmSrcDevData, CLMem* cmDestDevData,
					CLDevice *src_dev, CLDevice *dest_dev,
					const size_t src_offset, const size_t dest_offset,
					const size_t memSize,
					cl_bool blocked_cmd) 
{
	cl_int err = CL_SUCCESS;
	if (memSize == 0 || !cmSrcDevData->IsWithinRange(src_offset, memSize)
					|| !cmDestDevData->IsWithinRange(dest_offset, memSize))
	{
		err = CL_INVALID_VALUE;
	}
	else
	{
  		SNUCL_INFO("[Before Copy] CopyBuffer Src Device: %p Dest Device: %p\n",
  			cmSrcDevData->FrontLatest(), cmDestDevData->FrontLatest());
  		CLCommand* command = CLCommand::CreateCopyBuffer(
      		NULL, NULL, q, cmSrcDevData, cmDestDevData, 
			// NULL, NULL, 
			(cl_mem)cmSrcDevData->GetDevSpecific(src_dev),
			(cl_mem)cmDestDevData->GetDevSpecific(dest_dev),
			src_offset, dest_offset, memSize);
		if (command == NULL) {
			err = CL_OUT_OF_HOST_MEMORY;
		}
		else {
			CLEvent *blocking = NULL;
			if(blocked_cmd == CL_TRUE) blocking = command->ExportEvent();
			q->Enqueue(command);
			if(blocked_cmd == CL_TRUE)
			{
				cl_int ret = blocking->Wait();
				blocking->Release();
				if (ret < 0)
					err = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
				else err = CL_SUCCESS;
			}
		}
	}
	return err;
}

cl_int CLWriteBuffer(CLCommandQueue *q, CLMem *cmDevData, 
					const size_t offset, const size_t memSize,
					void *h_data, cl_bool blocked_cmd) 
{
	cl_int err = CL_SUCCESS;
	if (memSize == 0 || !cmDevData->IsWithinRange(offset, memSize))
	{
		err = CL_INVALID_VALUE;
	}
	else
	{
		CLCommand* command = CLCommand::CreateWriteBuffer(
				NULL, NULL, q, cmDevData, offset, memSize, h_data);
		if (command == NULL) {
			err = CL_OUT_OF_HOST_MEMORY;
		}
		else {
			CLEvent *blocking = NULL;
			if(blocked_cmd == CL_TRUE) blocking = command->ExportEvent();
			q->Enqueue(command);
			if(blocked_cmd == CL_TRUE)
			{
				cl_int ret = blocking->Wait();
				blocking->Release();
				if (ret < 0)
					err = CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST;
				else err = CL_SUCCESS;
			}
		}
	}
	return err;
}

 /*  Purpose: Provides meaningful error messages for error codes
 */
const char* getOpenCLErrorCodeStr(int input)
{
	int errorCode = input;
	switch(errorCode)
	{
		case CL_DEVICE_NOT_FOUND:
			return "CL_DEVICE_NOT_FOUND";
		case CL_DEVICE_NOT_AVAILABLE:
			return "CL_DEVICE_NOT_AVAILABLE";
		case CL_COMPILER_NOT_AVAILABLE:
			return "CL_COMPILER_NOT_AVAILABLE";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case CL_OUT_OF_RESOURCES:
			return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY:
			return "CL_OUT_OF_HOST_MEMORY";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case CL_MEM_COPY_OVERLAP:
			return "CL_MEM_COPY_OVERLAP";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "CL_IMAGE_FORMAT_MISMATCH";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case CL_BUILD_PROGRAM_FAILURE:
			return "CL_BUILD_PROGRAM_FAILURE";
		case CL_MAP_FAILURE:
			return "CL_MAP_FAILURE";
			//case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			//    return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
			//case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			//    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case CL_INVALID_VALUE:
			return "CL_INVALID_VALUE";
		case CL_INVALID_DEVICE_TYPE:
			return "CL_INVALID_DEVICE_TYPE";
		case CL_INVALID_PLATFORM:
			return "CL_INVALID_PLATFORM";
		case CL_INVALID_DEVICE:
			return "CL_INVALID_DEVICE";
		case CL_INVALID_CONTEXT:
			return "CL_INVALID_CONTEXT";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "CL_INVALID_QUEUE_PROPERTIES";
		case CL_INVALID_COMMAND_QUEUE:
			return "CL_INVALID_COMMAND_QUEUE";
		case CL_INVALID_HOST_PTR:
			return "CL_INVALID_HOST_PTR";
		case CL_INVALID_MEM_OBJECT:
			return "CL_INVALID_MEM_OBJECT";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case CL_INVALID_IMAGE_SIZE:
			return "CL_INVALID_IMAGE_SIZE";
		case CL_INVALID_SAMPLER:
			return "CL_INVALID_SAMPLER";
		case CL_INVALID_BINARY:
			return "CL_INVALID_BINARY";
		case CL_INVALID_BUILD_OPTIONS:
			return "CL_INVALID_BUILD_OPTIONS";
		case CL_INVALID_PROGRAM:
			return "CL_INVALID_PROGRAM";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
		case CL_INVALID_KERNEL_NAME:
			return "CL_INVALID_KERNEL_NAME";
		case CL_INVALID_KERNEL_DEFINITION:
			return "CL_INVALID_KERNEL_DEFINITION";
		case CL_INVALID_KERNEL:
			return "CL_INVALID_KERNEL";
		case CL_INVALID_ARG_INDEX:
			return "CL_INVALID_ARG_INDEX";
		case CL_INVALID_ARG_VALUE:
			return "CL_INVALID_ARG_VALUE";
		case CL_INVALID_ARG_SIZE:
			return "CL_INVALID_ARG_SIZE";
		case CL_INVALID_KERNEL_ARGS:
			return "CL_INVALID_KERNEL_ARGS";
		case CL_INVALID_WORK_DIMENSION:
			return "CL_INVALID_WORK_DIMENSION";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "CL_INVALID_WORK_GROUP_SIZE";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "CL_INVALID_WORK_ITEM_SIZE";
		case CL_INVALID_GLOBAL_OFFSET:
			return "CL_INVALID_GLOBAL_OFFSET";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case CL_INVALID_EVENT:
			return "CL_INVALID_EVENT";
		case CL_INVALID_OPERATION:
			return "CL_INVALID_OPERATION";
		case CL_INVALID_GL_OBJECT:
			return "CL_INVALID_GL_OBJECT";
		case CL_INVALID_BUFFER_SIZE:
			return "CL_INVALID_BUFFER_SIZE";
		case CL_INVALID_MIP_LEVEL:
			return "CL_INVALID_MIP_LEVEL";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
		//case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR:
		//	return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		//case CL_PLATFORM_NOT_FOUND_KHR:
		//	return "CL_PLATFORM_NOT_FOUND_KHR";
			//case CL_INVALID_PROPERTY_EXT:
			//    return "CL_INVALID_PROPERTY_EXT";
			//case CL_DEVICE_PARTITION_FAILED_EXT:
			//    return "CL_DEVICE_PARTITION_FAILED_EXT";
			//case CL_INVALID_PARTITION_COUNT_EXT:
			//    return "CL_INVALID_PARTITION_COUNT_EXT";
		default:
			return "unknown error code";
	}
	return "unknown error code";
}

H2DMetricsManager::H2DMetricsManager(const char *filename, hwloc_topology_t topology, 
						//const std::vector<hwloc_obj_t> &hosts, const std::vector<CLDevice *> &devices, 
						const int force_test) : _topology(topology)//, _hosts(hosts), _devices(devices)
{
	// if force_test == 1, then run tests and
	// populate the BW file
	_filename.assign(getenv("HOME"));
	_filename += "/.mpiacc.d/";
	_filename += filename;
	if(force_test == 1)
	{
		testAndWriteH2DMetrics();
	}
	//_filename = new char[1024];
	//memset(_filename, 1024, 0);
	//strcpy(_filename, filename);
	//_ifile_stream.open(filename);
	//_file_stream.open(filename, ios_base::in | ios_base::out | ios_base::app);
}

D2DMetricsManager::D2DMetricsManager(const char *filename, hwloc_topology_t topology, 
						//const std::vector<hwloc_obj_t> &hosts, const std::vector<CLDevice *> &devices, 
						const int force_test) : _topology(topology)//, _hosts(hosts), _devices(devices)
{
	// if force_test == 1, then run tests and
	// populate the BW file
	_filename.assign(getenv("HOME"));
	_filename += "/.mpiacc.d/";
	_filename += filename;
	if(force_test == 1)
	{
		testAndWriteD2DMetrics();
	}
	//_filename = new char[1024];
	//memset(_filename, 1024, 0);
	//strcpy(_filename, filename);
	//_ifile_stream.open(filename);
	//_file_stream.open(filename, ios_base::in | ios_base::out | ios_base::app);
}

#define _PLATFORM_NAME "SnuCL Single"
//#define _PLATFORM_NAME "NVIDIA CUDA"
//#define _PLATFORM_NAME "AMD Accelerated Parallel Processing"
void MetricsManager::clInit()
{
	cl_int err;
	cl_uint num_platforms;
	/*int chosen_platform_id = -1;
    size_t            platform_name_size;
    char*             platform_name;
	// Retrieve an OpenCL platform
	err = clGetPlatformIDs(0, NULL, &num_platforms);
	OPENCL_CHECK_ERR(err, "Get Platform IDs");
	printf("%u _platforms are detected.\n", num_platforms);

	_platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	err = clGetPlatformIDs(num_platforms, _platforms, NULL);
	OPENCL_CHECK_ERR(err, "Get Platform IDs");

	int i = 0;
	for (i = 0; i < num_platforms; i++) {
		err = clGetPlatformInfo(_platforms[i], CL_PLATFORM_NAME, 0, NULL,
				&platform_name_size);
		OPENCL_CHECK_ERR(err, "Get Platform Info");

		platform_name = (char*)malloc(sizeof(char) * platform_name_size);
		err = clGetPlatformInfo(_platforms[i], CL_PLATFORM_NAME, platform_name_size,
				platform_name, NULL);
		OPENCL_CHECK_ERR(err, "Get Platform Info");

		printf("Platform %d: %s\n", i, platform_name);
		if (strcmp(platform_name, _PLATFORM_NAME) == 0)
		{
			printf("Choosing Platform %d: %s\n", i, platform_name);
			chosen_platform_id = i;
		}
		free(platform_name);
	}

	if (chosen_platform_id == -1) {
		printf("%s platform is not found.\n", _PLATFORM_NAME);
		//exit(EXIT_FAILURE);
	}

	// Connect to a compute device
	err = clGetDeviceIDs(_platforms[chosen_platform_id], CL_DEVICE_TYPE_ALL, 0, NULL, &_num_devices);
	OPENCL_CHECK_ERR(err, "Failed to create a device group!");
	*/
	CLPlatform::GetPlatform()->GetDevices(_Devices);
	_num_devices = _Devices.size();
	_devices = (cl_device_id *)malloc(sizeof(cl_device_id) * _num_devices);
	//err = clGetDeviceIDs(_platforms[chosen_platform_id], CL_DEVICE_TYPE_ALL, _num_devices, _devices, NULL);
	//OPENCL_CHECK_ERR(err, "Failed to create a device group!");
	SNUCL_INFO("Total available devices: %d\n", _Devices.size());
	//printf("Total available devices: %d\n", _num_devices);
	for(int i = 0; i < _num_devices; i++)
	{
		_devices[i] = _Devices[i]->st_obj();
	}

	// Create a compute _context
  	_Context = CLPlatform::GetPlatform()->CreateContextFromDevices(
      NULL, _num_devices, _devices, NULL, &err);
	//_context = clCreateContext(0, _num_devices, _devices, NULL, NULL, &err);
	OPENCL_CHECK_ERR(err, "Failed to create a compute _context!");

	_Queues.resize(_num_devices);
	//_queues = (cl_command_queue *)malloc(sizeof(cl_command_queue) * _num_devices);
	for(int i = 0; i < _num_devices; i++)
	{
  	    _Queues[i] = CLCommandQueue::CreateCommandQueue(_Context, _Devices[i], 0, &err);
		//_queues[i] = clCreateCommandQueue(_context, _devices[i], 0, &err);
		OPENCL_CHECK_ERR(err, "Failed to create a command queue!");
	}
	SNUCL_INFO("Done CLInit!\n", 0);
}

void MetricsManager::clFinalize()
{
	for(int i = 0; i < _num_devices; i++)
	{
		//clReleaseCommandQueue(_queues[i]);
  		_Queues[i]->Release();
	}
	free(_devices);
	//free(_platforms);
	//free(_queues);
	_Queues.clear();
	//clReleaseContext(_context);
  	_Context->Release();
}

void H2DMetricsManager::testAndWriteH2DMetrics()
{
	SNUCL_INFO("About to run tests and write the H2D Bandwidths", 0);
	clInit();
	if(!_ofile_stream.is_open())
	{
		SNUCL_INFO("About to open File: %s\n", _filename.c_str());
		_ofile_stream.open(_filename.c_str());
		//_ofile_stream.open(_filename.c_str());
		if(!_ofile_stream)
		{
			// FIXME: Throw some error
			SNUCL_ERROR("cannot open metrics file for write\n", 0);
		}
	}
	Global::RealTimer gH2DMetricTimer;
  	gH2DMetricTimer.Init();
	// how many host sockets?
	hwloc_topology_t topology = CLPlatform::GetPlatform()->HWLOCTopology();
	std::vector<CLDevice *> devices;
	CLPlatform::GetPlatform()->GetDevices(devices);
	int n_devices = devices.size();
	pthread_t thread = pthread_self();
	int n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
	int n_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
	int n_sockets = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NODE); 
	int n_cores_per_socket = n_pus / n_sockets;
	cpu_set_t cpuset;
	cpu_set_t cur_cpuset;
	cl_int err = CL_SUCCESS;
	// Cache the current host thread mapping and restore at end of
	// this function
	pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cur_cpuset);
	const unsigned int defaultMemSize = (64 * (1 << 20));

	const size_t MIN_MEM_SIZE = (256 * 1024);
	const size_t MAX_MEM_SIZE = (1024* 1024);
	const int MULTIPLIER = 2;
	int num_memSizes = 0;
	for(size_t memSize = MIN_MEM_SIZE; memSize <= MAX_MEM_SIZE; memSize *= MULTIPLIER)
	{
		num_memSizes++;
	}

	SNUCL_INFO("About to write into File: %s\n", _filename.c_str());
	_ofile_stream << n_sockets << " " << n_devices << " " << num_memSizes << std::endl; 
	for(int host_id = 0; host_id < n_sockets; host_id++)
	{
		_ofile_stream << host_id << std::endl; 
		CPU_ZERO(&cpuset);
		CPU_SET(host_id * n_cores_per_socket, &cpuset);
		pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
		unsigned char *h_data = NULL;
		//std::vector<unsigned char *> h_data_ptrs(n_devices);
		//for(int device_id = n_devices - 1; device_id >= 0; device_id--)
		const size_t name_size = 1024;
		char device_name[name_size];
		std::vector<double> bandwidths(n_devices);
		std::vector<double> latencies(n_devices);
		for(size_t memSize = MIN_MEM_SIZE; memSize <= MAX_MEM_SIZE; memSize *= MULTIPLIER)
		{
			bandwidths.clear();
			bandwidths.resize(n_devices);
			latencies.clear();
			latencies.resize(n_devices);
			// run a bandwidth loop
			for(int device_id = 0; device_id < n_devices; device_id++)
			{
				// Create host side cl_mem here local to socket host_id
				//		cl_mem cmPinnedData = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, defaultMemSize, NULL, &err);
				CLMem* cmPinnedData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, defaultMemSize, NULL, &err);
				OPENCL_CHECK_ERR(err, "Creating Host Buffer");
				// allocate device memory 
				//cl_mem cmDevData = clCreateBuffer(_context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				CLMem* cmDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				OPENCL_CHECK_ERR(err, "Creating Device Buffer");
				// Get a mapped pointer
				//	unsigned char *h_data = (unsigned char*)clEnqueueMapBuffer(_queues[device_id], cmPinnedData, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, defaultMemSize, 0, NULL, NULL, &err);
				h_data = (unsigned char *)(cmPinnedData->MapAsBuffer(CL_MAP_WRITE | CL_MAP_READ, 0, defaultMemSize, _Queues[device_id]));
				if (h_data == NULL) {
					err = CL_OUT_OF_HOST_MEMORY;
				}
				OPENCL_CHECK_ERR(err, "Mapping Buffer");
				//	h_data_ptrs[device_id] = h_data;
				//SNUCL_INFO("Mapped data ref count: %d\n", cmPinnedData->ref_cnt());
				//initialize 
				for(unsigned int i = 0; i < defaultMemSize/sizeof(unsigned char); i++)
				{
					h_data[i] = (unsigned char)(i & 0xff);
				}
				//memset((void *)h_data, defaultMemSize, 0);
				//SNUCL_INFO("After Init Host Mem: %p Host Ptr: %p\n", cmPinnedData, h_data);
				// Sync queue to host
				//clFinish(_queues[device_id]);
				//CLFinish(_Queues[device_id]);
				const int iter_skip = 5;
				double bandwidthInMBs = 0.0;
				double elapsedTimeInSec = 0.0;
				int max_partitions = defaultMemSize / memSize;
				if(max_partitions < 1) max_partitions = 1;
				// begin time measurement
				int cur_partition = 0;
				for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
				{
					if(i == iter_skip) gH2DMetricTimer.Start();
					//err = clEnqueueWriteBuffer(_queues[device_id], cmDevData, CL_FALSE, cur_partition * memSize, memSize, h_data + (cur_partition * memSize), 0, NULL, NULL);
					err = CLWriteBuffer(_Queues[device_id], cmDevData, cur_partition * memSize, 
							memSize, (void *)(h_data + (cur_partition * memSize)), CL_FALSE);
					OPENCL_CHECK_ERR(err, "H2D Transfer");
					cur_partition = (cur_partition + 1) % max_partitions;
				}
				//err = clFinish(_queues[device_id]);
				err = CLFinish(_Queues[device_id]);
				OPENCL_CHECK_ERR(err, "Finish H2D Transfer");
				// end time measurement
				gH2DMetricTimer.Stop();
				elapsedTimeInSec = gH2DMetricTimer.Elapsed();
				bandwidthInMBs = ((double)memSize * (double)(MEMCOPY_ITERATIONS - iter_skip))/(elapsedTimeInSec * (double)(1 << 20));
				_Devices[device_id]->GetDeviceInfo(CL_DEVICE_NAME, name_size, device_name, NULL);
				printf("[H%d->D%d %s] Async Size %u Time(us) %g, Time per iter(us) %g, BW(MB/s) %g\n", host_id, device_id, device_name, memSize, elapsedTimeInSec, memSize/bandwidthInMBs, bandwidthInMBs);
				bandwidths[device_id] = bandwidthInMBs;
				//calculate bandwidth in MB/s

				// begin time measurement
				gH2DMetricTimer.Reset();
				CLCommand* command = CLCommand::CreateUnmapMemObject(
						NULL, NULL, _Queues[device_id], cmPinnedData, (void *)h_data);
				if (command == NULL) err = CL_OUT_OF_HOST_MEMORY;
				else {
					CLEvent* blocking = command->ExportEvent();
					_Queues[device_id]->Enqueue(command);
					blocking->Wait();
					blocking->Release();
					//SNUCL_INFO("Unmapped data ref count: %d\n", cmPinnedData->ref_cnt());
					err = CL_SUCCESS;
				}
				OPENCL_CHECK_ERR(err, "Unmap Buffer");
				//cmPinnedData->Release();
				h_data = NULL;
				// clean up cl_mem and other misc objects
				//clReleaseMemObject(cmDevData);
				cmDevData->Release();
				// clean up host side memory
				//clReleaseMemObject(cmPinnedData);
				cmPinnedData->Release();

			}

			// run a latency loop
			for(int device_id = 0; device_id < n_devices; device_id++)
			{
				// Create host side cl_mem here local to socket host_id
				//		cl_mem cmPinnedData = clCreateBuffer(_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, defaultMemSize, NULL, &err);
				CLMem* cmPinnedData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, defaultMemSize, NULL, &err);
				OPENCL_CHECK_ERR(err, "Creating Host Buffer");
				// allocate device memory 
				//cl_mem cmDevData = clCreateBuffer(_context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				CLMem* cmDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				OPENCL_CHECK_ERR(err, "Creating Device Buffer");
				// Get a mapped pointer
				//	unsigned char *h_data = (unsigned char*)clEnqueueMapBuffer(_queues[device_id], cmPinnedData, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, defaultMemSize, 0, NULL, NULL, &err);
				h_data = (unsigned char *)(cmPinnedData->MapAsBuffer(CL_MAP_WRITE | CL_MAP_READ, 0, defaultMemSize, _Queues[device_id]));
				if (h_data == NULL) {
					err = CL_OUT_OF_HOST_MEMORY;
				}
				OPENCL_CHECK_ERR(err, "Mapping Buffer");
				//	h_data_ptrs[device_id] = h_data;
				//SNUCL_INFO("Mapped data ref count: %d\n", cmPinnedData->ref_cnt());
				//initialize 
				for(unsigned int i = 0; i < defaultMemSize/sizeof(unsigned char); i++)
				{
					h_data[i] = (unsigned char)(i & 0xff);
				}
				//memset((void *)h_data, defaultMemSize, 0);
				//SNUCL_INFO("After Init Host Mem: %p Host Ptr: %p\n", cmPinnedData, h_data);
				// Sync queue to host
				//clFinish(_queues[device_id]);
				//CLFinish(_Queues[device_id]);
				const int iter_skip = 5;
				double bandwidthInMBs = 0.0;
				double elapsedTimeInSec = 0.0;
				int max_partitions = defaultMemSize / memSize;
				if(max_partitions < 1) max_partitions = 1;
				// run a bandwidth loop
				// begin time measurement
				int cur_partition = 0;
				for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
				{
					if(i == iter_skip) gH2DMetricTimer.Start();
					//err = clEnqueueWriteBuffer(_queues[device_id], cmDevData, CL_TRUE, cur_partition * memSize, memSize, h_data + (cur_partition * memSize), 0, NULL, NULL);
					err = CLWriteBuffer(_Queues[device_id], cmDevData, cur_partition * memSize, 
							memSize, (void *)(h_data + (cur_partition * memSize)), CL_TRUE);
					OPENCL_CHECK_ERR(err, "H2D Transfer");
					cur_partition = (cur_partition + 1) % max_partitions;
				}
				//err = clFinish(_queues[device_id]);
				err = CLFinish(_Queues[device_id]);
				OPENCL_CHECK_ERR(err, "Finish H2D Transfer");
				gH2DMetricTimer.Stop();
				// end time measurement
				elapsedTimeInSec = gH2DMetricTimer.Elapsed();
				bandwidthInMBs = ((double)memSize * (double)(MEMCOPY_ITERATIONS - iter_skip))/(elapsedTimeInSec * (double)(1 << 20));
				_Devices[device_id]->GetDeviceInfo(CL_DEVICE_NAME, name_size, device_name, NULL);
				printf("[H%d->D%d %s] Sync Size %u Time(us) %g, Time per iter(us) %g, BW(MB/s) %g\n", host_id, device_id, device_name, memSize, elapsedTimeInSec, memSize/bandwidthInMBs, bandwidthInMBs);
				latencies[device_id] = memSize/bandwidthInMBs;
				gH2DMetricTimer.Reset();
				//calculate bandwidth in MB/s
				// store it in a new row in the file

				//err = clEnqueueUnmapMemObject(_queues[device_id], cmPinnedData, (void*)h_data, 0, NULL, NULL);
				CLCommand* command = CLCommand::CreateUnmapMemObject(
						NULL, NULL, _Queues[device_id], cmPinnedData, (void *)h_data);
				if (command == NULL) err = CL_OUT_OF_HOST_MEMORY;
				else {
					CLEvent* blocking = command->ExportEvent();
					_Queues[device_id]->Enqueue(command);
					blocking->Wait();
					blocking->Release();
					//SNUCL_INFO("Unmapped data ref count: %d\n", cmPinnedData->ref_cnt());
					err = CL_SUCCESS;
				}
				OPENCL_CHECK_ERR(err, "Unmap Buffer");
				//cmPinnedData->Release();
				h_data = NULL;
				// clean up cl_mem and other misc objects
				//clReleaseMemObject(cmDevData);
				cmDevData->Release();
				// clean up host side memory
				//clReleaseMemObject(cmPinnedData);
				cmPinnedData->Release();
			}
			//h_data_ptrs.clear();
			/*// clean up cl_mem and other misc objects
			//clReleaseMemObject(cmDevData);
			cmDevData->Release();
			// clean up host side memory
			//clReleaseMemObject(cmPinnedData);
			cmPinnedData->Release();*/

			for(int device_id = 0; device_id < n_devices; device_id++)
			{
				_ofile_stream << device_id << " " 
					<< memSize << " "
					<< bandwidths[device_id] << " " 
					<< latencies[device_id] << std::endl;
			}
		}
	}
	pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cur_cpuset);

	clFinalize();
	_ofile_stream.close();
}

void H2DMetricsManager::readH2DMetrics()
{
	SNUCL_INFO("About to read the H2D Bandwidths from file %s\n", _filename.c_str());
	
	if(!_ifile_stream.is_open())
	{
		_ifile_stream.open(_filename.c_str());
		if(!_ifile_stream)
		{
			// if file not found force tests and create file
			testAndWriteH2DMetrics();
			// Try opening it again now
			_ifile_stream.open(_filename.c_str());
			if(!_ifile_stream)
			{
				SNUCL_ERROR("Still cannot open metrics file for reading: %s\n", _filename.c_str());
			}
		}
	}
	std::string metric_line;
	std::vector<std::string> tokens;
	int host_count, device_count, memsize_count;
	int host_id;
	int line_num = 0;
	metrics_vector d2h_metrics_vector;
	d2h_metrics_vector.clear();
	while(std::getline(_ifile_stream, metric_line))
	{
		//SNUCL_INFO("Reading Line: %s\n", metric_line.c_str());
		if(!tokens.empty()) tokens.clear();
		tokens = mysplit(metric_line, ' ');
		//SNUCL_INFO("Tokens Size: %d\n", tokens.size());
		if(line_num == 0)
		{
			host_count = atoi(tokens[0].c_str());
			device_count = atoi(tokens[1].c_str());
			memsize_count = atoi(tokens[2].c_str());
		}
		else
		{
			if(tokens.size() == 1) // host ID only
			{
				host_id = atoi(tokens[0].c_str());
				if(!d2h_metrics_vector.empty())
				{
					_d2h_metrics_matrix.push_back(d2h_metrics_vector);
				}
				d2h_metrics_vector.clear();
			}
			else
			{
				//insert tokens as tuple elements
				//SNUCL_INFO("Token0: %s Token1: %s Token2: %s Token3: %s\n", tokens[0].c_str(), tokens[1].c_str(), tokens[2].c_str(), tokens[3].c_str());
				metrics_tuple mt = std::make_tuple(
									atoi(tokens[0].c_str()),
									atol(tokens[1].c_str()),
									atof(tokens[2].c_str()),
									atof(tokens[3].c_str()));
				d2h_metrics_vector.push_back(mt);
			}
		}
		line_num++;
	}
	if(!d2h_metrics_vector.empty())
		_d2h_metrics_matrix.push_back(d2h_metrics_vector);
	d2h_metrics_vector.clear();

	// if contents are empty/badly formatted, then error
	_ifile_stream.close();
}

double H2DMetricsManager::getH2DBandwidth(const int host_id, const int device_id, const size_t mem_size, const metric_type mt)
{
	SNUCL_INFO("About to get the individual H2D Bandwidths\n", 0);
	double ret_val = 0.0;
	if(_d2h_metrics_matrix.empty())
	{
		SNUCL_INFO("Metrics matrix is empty. Reconstructing...\n", 0);
		readH2DMetrics();
	}
	metrics_vector d2h_metrics_vector = _d2h_metrics_matrix[host_id];
	for(size_t i = 0; i < d2h_metrics_vector.size(); i++)
	{
		if(std::get<0>(d2h_metrics_vector[i]) == device_id 
			&& std::get<1>(d2h_metrics_vector[i]) == mem_size)
		{
			switch(mt)
			{
				case SNUCL_BANDWIDTH:
					ret_val = std::get<2>(d2h_metrics_vector[i]);
	//				SNUCL_INFO("Getting BW from Metrics matrix %g\n", ret_val);
					break;
				case SNUCL_LATENCY:
					ret_val = std::get<3>(d2h_metrics_vector[i]);
					//SNUCL_INFO("Getting Latency from Metrics matrix %g\n", ret_val);
					break;
			}
		}
	}
	return ret_val;
	// Retrieve the hwloc/opencl objects based on the host
	// and device IDs
	//
	// Check if tests need to be run again
	//
	// If yes, run the tests and record in the given file
	//
	// If no, just read values from the data structutes and retrieve the
	// bandwidth
}

void D2DMetricsManager::testAndWriteD2DMetrics()
{
	SNUCL_INFO("About to run tests and write the D2D Bandwidths", 0);
	clInit();
	if(!_ofile_stream.is_open())
	{
		SNUCL_INFO("About to open File: %s\n", _filename.c_str());
		_ofile_stream.open(_filename.c_str());
		//_ofile_stream.open(_filename.c_str());
		if(!_ofile_stream)
		{
			// FIXME: Throw some error
			SNUCL_ERROR("cannot open metrics file for write\n", 0);
		}
	}
	Global::RealTimer gD2DMetricTimer;
  	gD2DMetricTimer.Init();
	// how many host sockets?
	hwloc_topology_t topology = CLPlatform::GetPlatform()->HWLOCTopology();
	std::vector<CLDevice *> devices;
	CLPlatform::GetPlatform()->GetDevices(devices);
	int n_devices = devices.size();
	pthread_t thread = pthread_self();
	int n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
	int n_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
	int n_sockets = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_SOCKET); 
	int n_cores_per_socket = n_pus / n_sockets;
	cpu_set_t cpuset;
	cpu_set_t cur_cpuset;
	cl_int err = CL_SUCCESS;
	// Cache the current host thread mapping and restore at end of
	// this function
	pthread_getaffinity_np(thread, sizeof(cpu_set_t), &cur_cpuset);
	const unsigned int defaultMemSize = (64 * (1 << 20));

	const size_t MIN_MEM_SIZE = (256 * 1024);
	const size_t MAX_MEM_SIZE = (1 * 1024 * 1024);
	const int MULTIPLIER = 2;
	int num_memSizes = 0;
	for(size_t memSize = MIN_MEM_SIZE; memSize <= MAX_MEM_SIZE; memSize *= MULTIPLIER)
	{
		num_memSizes++;
	}

	SNUCL_INFO("About to write into File: %s\n", _filename.c_str());
	_ofile_stream << n_devices << " " << num_memSizes << std::endl; 
	for(int src_dev_id = 0; src_dev_id < n_devices; src_dev_id++)
	{
		_ofile_stream << src_dev_id << std::endl; 
		unsigned char *h_data = NULL;
		//std::vector<unsigned char *> h_data_ptrs(n_devices);
		const size_t name_size = 1024;
		char device_name[name_size];
		std::vector<double> bandwidths(n_devices);
		std::vector<double> latencies(n_devices);
		// allocate device memory 
		//cl_mem cmDevData = clCreateBuffer(_context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
		CLMem* cmSrcDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
		OPENCL_CHECK_ERR(err, "Creating Device Buffer");
		// Get a host pointer
		h_data = (unsigned char *)malloc(sizeof(unsigned char) * defaultMemSize);
		if (h_data == NULL) {
			err = CL_OUT_OF_HOST_MEMORY;
		}
		OPENCL_CHECK_ERR(err, "Creating Host Data");
		//	h_data_ptrs[dest_dev_id] = h_data;
		//initialize 
		for(unsigned int i = 0; i < defaultMemSize/sizeof(unsigned char); i++)
		{
			h_data[i] = (unsigned char)(i & 0xff);
		}
		//memset((void *)h_data, defaultMemSize, 0);
		//SNUCL_INFO("After Init Src Mem: %p Host Ptr: %p\n", cmSrcDevData, h_data);
		devices[src_dev_id]->WriteBuffer(NULL, cmSrcDevData, 0, defaultMemSize * sizeof(unsigned char), h_data); 
		cmSrcDevData->SetLatest(devices[src_dev_id]);
		for(size_t memSize = MIN_MEM_SIZE; memSize <= MAX_MEM_SIZE; memSize *= MULTIPLIER)
		{
		//CLMem* cmSrcDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, memSize, NULL, &err);
		//OPENCL_CHECK_ERR(err, "Creating Device Buffer");
		//devices[src_dev_id]->WriteBuffer(NULL, cmSrcDevData, 0, memSize * sizeof(unsigned char), h_data); 
		//cmSrcDevData->SetLatest(devices[src_dev_id]);
			bandwidths.clear();
			bandwidths.resize(n_devices);
			latencies.clear();
			latencies.resize(n_devices);
			// run a bandwidth loop
			for(int dest_dev_id = 0; dest_dev_id < n_devices; dest_dev_id++)
			{
				const int iter_skip = 5;
				double bandwidthInMBs = 0.0;
				double elapsedTimeInSec = 0.0;
				int max_partitions = defaultMemSize / memSize;
				if(max_partitions < 1) max_partitions = 1;
				// allocate dest device memory 
				//cl_mem cmDevData = clCreateBuffer(_context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				//CLMem* cmDestDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				CLMem* cmDestDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, memSize, NULL, &err);
				OPENCL_CHECK_ERR(err, "Creating Device Buffer");
				//devices[dest_dev_id]->WriteBuffer(NULL, cmDestDevData, 0, defaultMemSize * sizeof(unsigned char), h_data); 
				devices[dest_dev_id]->WriteBuffer(NULL, cmDestDevData, 0, memSize * sizeof(unsigned char), h_data); 
		        cmDestDevData->SetLatest(devices[dest_dev_id]);
				//cmSrcDevData->SetLatest(devices[src_dev_id]);
				// begin time measurement
				int cur_partition = 0;
				for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
				{
					if(i == iter_skip) gD2DMetricTimer.Start();
					//err = clEnqueueWriteBuffer(_queues[dest_dev_id], cmDevData, CL_FALSE, cur_partition * memSize, memSize, h_data + (cur_partition * memSize), 0, NULL, NULL);
					//err = CLCopyBuffer(_Queues[src_dev_id], 
					err = CLCopyBuffer(_Queues[dest_dev_id], 
						cmSrcDevData, cmDestDevData,
						devices[src_dev_id], devices[dest_dev_id],
						//cur_partition * memSize, cur_partition * memSize,
						0, 0,
						memSize, CL_FALSE);
					OPENCL_CHECK_ERR(err, "D2D Transfer");
					cur_partition = (cur_partition + 1) % max_partitions;
				}
				//err = clFinish(_queues[dest_dev_id]);
				//err = CLFinish(_Queues[src_dev_id]);
				err = CLFinish(_Queues[dest_dev_id]);
				OPENCL_CHECK_ERR(err, "Finish D2D Transfer");
				// end time measurement
				gD2DMetricTimer.Stop();
				elapsedTimeInSec = gD2DMetricTimer.Elapsed();
				bandwidthInMBs = ((double)memSize * (double)(MEMCOPY_ITERATIONS - iter_skip))/(elapsedTimeInSec * (double)(1 << 20));
				_Devices[dest_dev_id]->GetDeviceInfo(CL_DEVICE_NAME, name_size, device_name, NULL);
				printf("[D%d->D%d %s] Async Size %u Time(us) %g, Time per iter(us) %g, BW(MB/s) %g\n", src_dev_id, dest_dev_id, device_name, memSize, elapsedTimeInSec, memSize/bandwidthInMBs, bandwidthInMBs);
				bandwidths[dest_dev_id] = bandwidthInMBs;
				//calculate bandwidth in MB/s

				// begin time measurement
				gD2DMetricTimer.Reset();
				//clReleaseMemObject(cmDevData);
				cmDestDevData->Release();
			}

			//cmSrcDevData->SetLatest(devices[src_dev_id]);
			// run a latency loop
			for(int dest_dev_id = 0; dest_dev_id < n_devices; dest_dev_id++)
			{
				const int iter_skip = 5;
				double bandwidthInMBs = 0.0;
				double elapsedTimeInSec = 0.0;
				int max_partitions = defaultMemSize / memSize;
				if(max_partitions < 1) max_partitions = 1;
				// allocate device memory 
				//cl_mem cmDevData = clCreateBuffer(_context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				//CLMem* cmDestDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, defaultMemSize, NULL, &err);
				CLMem* cmDestDevData = CLMem::CreateBuffer(_Context, CL_MEM_READ_WRITE, memSize, NULL, &err);
				OPENCL_CHECK_ERR(err, "Creating Device Buffer");
				//devices[dest_dev_id]->WriteBuffer(NULL, cmDestDevData, 0, defaultMemSize * sizeof(unsigned char), h_data); 
				devices[dest_dev_id]->WriteBuffer(NULL, cmDestDevData, 0, memSize * sizeof(unsigned char), h_data); 
		        cmDestDevData->SetLatest(devices[dest_dev_id]);
			//	cmSrcDevData->SetLatest(devices[src_dev_id]);
				// run a bandwidth loop
				// begin time measurement
				int cur_partition = 0;
				for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
				{
					if(i == iter_skip) gD2DMetricTimer.Start();
					//err = clEnqueueWriteBuffer(_queues[dest_dev_id], cmDevData, CL_TRUE, cur_partition * memSize, memSize, h_data + (cur_partition * memSize), 0, NULL, NULL);
					//err = CLCopyBuffer(_Queues[src_dev_id], 
					err = CLCopyBuffer(_Queues[dest_dev_id], 
						cmSrcDevData, cmDestDevData,
						devices[src_dev_id], devices[dest_dev_id],
						//cur_partition * memSize, cur_partition * memSize,
						0, 0,
						memSize, CL_TRUE);
					OPENCL_CHECK_ERR(err, "D2D Transfer");
					cur_partition = (cur_partition + 1) % max_partitions;
				}
				gD2DMetricTimer.Stop();
				// end time measurement
				elapsedTimeInSec = gD2DMetricTimer.Elapsed();
				//err = clFinish(_queues[dest_dev_id]);
				//err = CLFinish(_Queues[src_dev_id]);
				err = CLFinish(_Queues[dest_dev_id]);
				OPENCL_CHECK_ERR(err, "Finish D2D Transfer");
				bandwidthInMBs = ((double)memSize * (double)(MEMCOPY_ITERATIONS - iter_skip))/(elapsedTimeInSec * (double)(1 << 20));
				_Devices[dest_dev_id]->GetDeviceInfo(CL_DEVICE_NAME, name_size, device_name, NULL);
				printf("[D%d->D%d %s] Sync Size %u Time(us) %g, Time per iter(us) %g, BW(MB/s) %g\n", src_dev_id, dest_dev_id, device_name, memSize, elapsedTimeInSec, memSize/bandwidthInMBs, bandwidthInMBs);
				latencies[dest_dev_id] = memSize/bandwidthInMBs;
				gD2DMetricTimer.Reset();
				//calculate bandwidth in MB/s
				// store it in a new row in the file

				cmDestDevData->Release();
			}
			//h_data_ptrs.clear();
			/*// clean up cl_mem and other misc objects
			//clReleaseMemObject(cmDevData);
			cmDevData->Release();
			// clean up host side memory
			//clReleaseMemObject(cmPinnedData);
			cmPinnedData->Release();*/

			for(int dest_dev_id = 0; dest_dev_id < n_devices; dest_dev_id++)
			{
				_ofile_stream << dest_dev_id << " " 
					<< memSize << " "
					<< bandwidths[dest_dev_id] << " " 
					<< latencies[dest_dev_id] << std::endl;
			}
		//cmSrcDevData->Release();
		}
		cmSrcDevData->Release();
		//cmDestDevData->Release();
		free(h_data);
	}
	pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cur_cpuset);

	clFinalize();
	_ofile_stream.close();
}

void D2DMetricsManager::readD2DMetrics()
{
	SNUCL_INFO("About to read the D2D Bandwidths from file %s\n", _filename.c_str());
	
	if(!_ifile_stream.is_open())
	{
		_ifile_stream.open(_filename.c_str());
		if(!_ifile_stream)
		{
			// if file not found force tests and create file
			testAndWriteD2DMetrics();
			// Try opening it again now
			_ifile_stream.open(_filename.c_str());
			if(!_ifile_stream)
			{
				SNUCL_ERROR("Still cannot open metrics file for reading: %s\n", _filename.c_str());
			}
		}
	}
	std::string metric_line;
	std::vector<std::string> tokens;
	int device_count, memsize_count;
	int device_id;
	int line_num = 0;
	metrics_vector d2d_metrics_vector;
	d2d_metrics_vector.clear();
	while(std::getline(_ifile_stream, metric_line))
	{
		//SNUCL_INFO("Reading Line: %s\n", metric_line.c_str());
		if(!tokens.empty()) tokens.clear();
		tokens = mysplit(metric_line, ' ');
		//SNUCL_INFO("Tokens Size: %d\n", tokens.size());
		if(line_num == 0)
		{
			//host_count = atoi(tokens[0].c_str());
			device_count = atoi(tokens[0].c_str());
			memsize_count = atoi(tokens[1].c_str());
		}
		else
		{
			if(tokens.size() == 1) // device ID only
			{
				device_id = atoi(tokens[0].c_str());
				if(!d2d_metrics_vector.empty())
				{
					_d2d_metrics_matrix.push_back(d2d_metrics_vector);
				}
				d2d_metrics_vector.clear();
			}
			else
			{
				//insert tokens as tuple elements
				metrics_tuple mt = std::make_tuple(
									atoi(tokens[0].c_str()),
									atol(tokens[1].c_str()),
									atof(tokens[2].c_str()),
									atof(tokens[3].c_str()));
				d2d_metrics_vector.push_back(mt);
			}
		}
		line_num++;
	}
	if(!d2d_metrics_vector.empty())
		_d2d_metrics_matrix.push_back(d2d_metrics_vector);
	d2d_metrics_vector.clear();

	// if contents are empty/badly formatted, then error
	_ifile_stream.close();
}

double D2DMetricsManager::getD2DBandwidth(const int src_dev_id, const int dest_dev_id, const size_t mem_size, const metric_type mt)
{
	SNUCL_INFO("About to get the individual D2D Bandwidths\n", 0);
	double ret_val = 0.0;
	if(_d2d_metrics_matrix.empty())
	{
		SNUCL_INFO("Metrics matrix is empty. Reconstructing...\n", 0);
		readD2DMetrics();
	}
	metrics_vector d2d_metrics_vector = _d2d_metrics_matrix[src_dev_id];
	for(size_t i = 0; i < d2d_metrics_vector.size(); i++)
	{
		if(std::get<0>(d2d_metrics_vector[i]) == dest_dev_id 
			&& std::get<1>(d2d_metrics_vector[i]) == mem_size)
		{
			switch(mt)
			{
				case SNUCL_BANDWIDTH:
					ret_val = std::get<2>(d2d_metrics_vector[i]);
	//				SNUCL_INFO("Getting BW from Metrics matrix %g\n", ret_val);
					break;
				case SNUCL_LATENCY:
					ret_val = std::get<3>(d2d_metrics_vector[i]);
					//SNUCL_INFO("Getting Latency from Metrics matrix %g\n", ret_val);
					break;
			}
		}
	}
	return ret_val;
	// Retrieve the hwloc/opencl objects based on the host
	// and device IDs
	//
	// Check if tests need to be run again
	//
	// If yes, run the tests and record in the given file
	//
	// If no, just read values from the data structutes and retrieve the
	// bandwidth
}
