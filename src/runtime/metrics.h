#ifndef __SNUCL__METRICS_H
#define __SNUCL__METRICS_H
#include <fstream>
#include <tuple>
#include <vector>
#include <string>
#include "hwloc.h"
#include <CLDevice.h>
class H2DMetricsManager
{
	public:
		enum metric_type {SNUCL_BANDWIDTH = 0, SNUCL_LATENCY};
		typedef std::tuple<unsigned int, size_t, double, double> metrics_tuple;
		typedef std::vector<metrics_tuple> metrics_vector;
		
		H2DMetricsManager(const char *filename, hwloc_topology_t topology, 
						//const std::vector<hwloc_obj_t> &hosts, const std::vector<CLDevice *> &devices, 
						const int force_test = 0);

		~H2DMetricsManager()
		{
			//delete []_filename;
			//_filename = NULL;
			//_file_stream.close();
		}

	void readH2DMetrics();
	void testAndWriteH2DMetrics();
	double getH2DBandwidth(const int host_id, const int device_id, const size_t mem_size, const metric_type = SNUCL_BANDWIDTH);
	private:
		std::string _filename;
		//char *_filename;
		std::fstream _file_stream;
		std::ifstream _ifile_stream;
		std::ofstream _ofile_stream;
		hwloc_topology_t _topology; // we could just use hwloc_obj_ts
		std::vector<CLDevice *> _devices;
		std::vector<hwloc_obj_t> _hosts;
  /* 2D vector storing distances between CPUsets and OpenCL
   * devices. Each row represents one CPUset. */
  		std::vector<metrics_vector> _d2h_metrics_matrix;
};


#endif //__SNUCL__METRICS_H

