/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National 
   Laboratories ( http://www.mantevo.org ). The primary 
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier 
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you 
   can redistribute it and/or modify it under the terms of the GNU Lesser 
   General Public License as published by the Free Software Foundation; 
   either version 3 of the License, or (at your option) any later 
   version.
  
   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
   Lesser General Public License for more details.
    
   You should have received a copy of the GNU Lesser General Public 
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov). 

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#ifndef OPENCL_WRAPPER
#define OPENCL_WRAPPER

#include <cstdio>
#include <cstdarg>
#define __CR printf("Got to line %i in '%s'\n",__LINE__,__FILE__);

#include <CL/opencl.h>

class OpenCLWrapper{
public:
    cl_uint num_platforms;
    cl_platform_id* platformIDs;
    cl_platform_id myPlatform;

    cl_uint num_devices;
    cl_device_id* deviceIDs;
    cl_device_id myDevice;
    cl_context myGPUContext;

    cl_uint num_subdevices;
    cl_device_id* subdeviceIDs;

    cl_uint num_queues;
    cl_command_queue* queues;
    cl_command_queue defaultQueue;

    cl_int ciErrNum;

    cl_mem buffer;
    cl_uint buffersize;

    cl_uint total_dev_mem;

    cl_uint blockdim;

    bool is_cpu;

    OpenCLWrapper();
	~OpenCLWrapper();

    int Init(int argc, char** argv,int me,int ppn,int* devicelist, int platformid=0, int subdevice=-1);

    cl_mem AllocDevData(unsigned nbytes);
    void FreeDevData(cl_mem dev_data,unsigned nbytes);

    void* AllocPinnedHostData(unsigned nbytes);
    void FreePinnedHostData(void* host_data);

    int UploadData(void* host_data, cl_mem dev_data, unsigned nbytes, size_t offset = 0);
    int DownloadData(void* host_data, cl_mem dev_data, unsigned nbytes, size_t offset = 0);
    int UploadDataAsync(void* host_data, cl_mem dev_data, unsigned nbytes, unsigned int stream, size_t offset = 0);
    int DownloadDataAsync(void* host_data, cl_mem dev_data, unsigned nbytes, unsigned int stream, size_t offset = 0);

    cl_mem AllocDevDataImageFloat4(unsigned elements, int &imagesize);
    int CopyBufferToImageFloat4(cl_mem src_buffer, cl_mem dst_image);

    int Memset(cl_mem dev_data, unsigned nbytes, int value);

    cl_mem Buffer() {return buffer;};
    cl_uint BufferSize() {return buffersize;};
    cl_mem BufferGrow(cl_uint newsize);
    cl_mem BufferResize(cl_uint newsize);

    int ReadKernelSource(const char* file);
    int CompileProgram(char* options = NULL);
    int CreateKernel(const char* kernelname);
    int LaunchKernel(int kernel_num, int threads, int nargs, ...);
    int LaunchKernel(const char* kernel_name, int threads, int nargs, ...);

	void Error(cl_int ciErrNUM,const char message[],const char file[],int line);

private:
	cl_uint num_pinned_host_buffers;
	cl_uint max_pinned_host_buffers;
	cl_mem* pinned_host_buffers;
	void** pinned_host_buffers_pointers;

	cl_uint num_kernel_sources;
	cl_uint max_kernels;
	char** kernel_source;

	cl_program program;

	cl_uint num_kernels;
	cl_kernel* kernels;
	char** kernel_names;
};


#endif /* OPENCL_WRAPPER*/
