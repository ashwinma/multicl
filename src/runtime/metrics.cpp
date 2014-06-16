#include "metrics.h"

H2DMetricsManager::H2DMetricsManager(const char *filename, hwloc_topology_t topology, 
						//const std::vector<hwloc_obj_t> &hosts, const std::vector<CLDevice *> &devices, 
						const int force_test) : _topology(topology)//, _hosts(hosts), _devices(devices)
{
	// if force_test == 1, then run tests and
	// populate the BW file
	if(force_test == 1)
	{
		testAndWriteH2DMetrics();
	}
	_filename.assign(getenv("HOME"));
	_filename += "/.mpiacc.d/";
	_filename += filename;
	//_filename = new char[1024];
	//memset(_filename, 1024, 0);
	//strcpy(_filename, filename);
	//_ifile_stream.open(filename);
	//_file_stream.open(filename, ios_base::in | ios_base::out | ios_base::app);
}

void H2DMetricsManager::testAndWriteH2DMetrics()
{
	SNUCL_INFO("About to run tests and write the H2D Bandwidths", 0);
	// how many host sockets?
	hwloc_topology_t topology = CLPlatform::GetPlatform()->HWLOCTopology();
	std::vector<CLDevice *> devices;
	CLPlatform::GetPlatform()->GetDevices(&devices);
	int n_devices = devices.size();
	pthread_t thread = pthread_self();
	int n_pus = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
	int n_cores = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_CORE);
	int n_sockets = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_SOCKET); 
	int n_cores_per_socket = n_pus / n_sockets;
	cpu_set_t cpuset;
	for(int host_id = 0; host_id < n_sockets; host_id++)
	{
		CPU_ZERO(&cpuset);
		CPU_SET(host_id * n_cores_per_socket, &cpuset);
		pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
		for(int device_id = 0; device_id < n_devices; device_id++)
		{
			size_t memSize = (512 * 1024);
			//for(size_t memSize = MIN_MEM_SIZE; memSize <= MAX_MEM_SIZE; memSize *= MULTIPLIER)
			{
				// run a bandwidth loop
				// run a latency loop
				// store it in a new row in the file
			}
		}
	}
}

void H2DMetricsManager::readH2DMetrics()
{
	SNUCL_INFO("About to read the H2D Bandwidths from file %s", _filename.c_str());
	
	if(!_ifile_stream.is_open())
	{
		_ifile_stream.open(_filename.c_str());
		if(!_ifile_stream)
		{
			// if file not found force tests and create file
			testAndWriteH2DMetrics();
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
		SNUCL_INFO("Reading Line: %s\n", metric_line.c_str());
		if(!tokens.empty()) tokens.clear();
		tokens = mysplit(metric_line, ' ');
		SNUCL_INFO("Tokens Size: %d\n", tokens.size());
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
}

double H2DMetricsManager::getH2DBandwidth(const int host_id, const int device_id, const size_t mem_size, const metric_type mt)
{
	SNUCL_INFO("About to get the individual H2D Bandwidths", 0);
	double ret_val = 0.0;
	if(_d2h_metrics_matrix.empty())
	{
		SNUCL_INFO("Metrics matrix is empty. Reconstructing...", 0);
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
					SNUCL_INFO("Getting BW from Metrics matrix %g\n", ret_val);
					break;
				case SNUCL_LATENCY:
					ret_val = std::get<3>(d2h_metrics_vector[i]);
					SNUCL_INFO("Getting Latency from Metrics matrix %g\n", ret_val);
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
