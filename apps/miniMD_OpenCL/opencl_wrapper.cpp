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

#include "opencl_wrapper.h"
#include <cstring>
#include <cmath>
#include <cstdlib>

OpenCLWrapper::OpenCLWrapper()
{
	num_platforms = 0;
	platformIDs = NULL;

	num_devices = 0;
	deviceIDs = NULL;

	num_subdevices = 0;
	subdeviceIDs = NULL;

	num_queues = 0;
	queues = NULL;

	buffer = NULL;
	buffersize = 0;

	ciErrNum = 0;

	num_pinned_host_buffers = 0;
	max_pinned_host_buffers = 32;
	pinned_host_buffers = new cl_mem[max_pinned_host_buffers];
	pinned_host_buffers_pointers = new void*[max_pinned_host_buffers];

	max_kernels = 100;
	num_kernel_sources = 0;
	kernel_source = new char*[max_kernels];

	num_kernels = 0;
	kernels = new cl_kernel[max_kernels];
	kernel_names = new char*[max_kernels];

	blockdim = 192;

	is_cpu = false;
}

OpenCLWrapper::~OpenCLWrapper()
{

}

int OpenCLWrapper::Init(int argc, char** argv,int me,int ppn,int* devicelist,int platformid,int subdevice)
{

	ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS)
    {
    	Error(ciErrNum,"GetPlatformIDs", __FILE__, __LINE__);
    	return ciErrNum;
    }
    if(num_platforms == 0)
    {
    	printf("No OpenCL platform found - exiting\n");
    	return ciErrNum;
    }
    platformIDs = new cl_platform_id[num_platforms];
    ciErrNum = clGetPlatformIDs (num_platforms, platformIDs, NULL);

    myPlatform = platformIDs[platformid];

    ciErrNum = clGetDeviceIDs (myPlatform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (ciErrNum != CL_SUCCESS)
    {
    	Error(ciErrNum,"Get_Num_Devices", __FILE__, __LINE__);
    	return ciErrNum;
    }


    deviceIDs = new cl_device_id[num_devices];
    ciErrNum = clGetDeviceIDs (myPlatform, CL_DEVICE_TYPE_ALL, num_devices, deviceIDs, &num_devices);
    if (ciErrNum != CL_SUCCESS)
    {
    	Error(ciErrNum,"Get_DeviceIDs", __FILE__, __LINE__);
    	return ciErrNum;
    }

    myDevice = deviceIDs[me%ppn];


	cl_device_type device_type;
    clGetDeviceInfo(myDevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
    if(device_type & CL_DEVICE_TYPE_CPU) is_cpu = true;

    char device_string[1024];
    clGetDeviceInfo(myDevice, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
    printf("Using Device: %i %s (a %s)\n",me%ppn,device_string,is_cpu?"CPU":"GPU");

    subdeviceIDs = new cl_device_id[128];
    if(is_cpu&&subdevice>-1)
    {
        cl_device_partition_property_ext   partition_properties[] =
    			{ CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT, CL_AFFINITY_DOMAIN_NUMA_EXT,
    				CL_PROPERTIES_LIST_END_EXT };
    	//ciErrNum = clCreateSubDevicesEXT(myDevice,partition_properties, 128, subdeviceIDs, &num_subdevices );
    	Error(ciErrNum,"CreateSubDevices", __FILE__, __LINE__);
    	myDevice=subdeviceIDs[subdevice];
    	printf("Attaching to subdevice %i of %i\n",subdevice,num_subdevices);
    }


    myGPUContext = clCreateContext(0, 1, &myDevice, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    {
    	Error(ciErrNum,"Get_Context", __FILE__, __LINE__);
    	return ciErrNum;
    }


    num_queues = 1;
    queues = new cl_command_queue[num_queues];
    for(unsigned int i=0;i<num_queues;i++)
      queues[i] = clCreateCommandQueue(myGPUContext, myDevice, CL_QUEUE_PROFILING_ENABLE, NULL);
    defaultQueue = queues[0];

	return 0;
}


cl_mem OpenCLWrapper::AllocDevData(unsigned nbytes)
{
  cl_mem dev_data = clCreateBuffer(myGPUContext, CL_MEM_READ_WRITE, nbytes, NULL, &ciErrNum);
  if (ciErrNum != CL_SUCCESS)
    Error(ciErrNum,"AllocDevData",__FILE__,__LINE__);
  total_dev_mem += nbytes;
  return dev_data;
}

void OpenCLWrapper::FreeDevData(cl_mem dev_data,unsigned nbytes)
{
   	if(dev_data) clReleaseMemObject(dev_data);
   	total_dev_mem -= nbytes;
}

void* OpenCLWrapper::AllocPinnedHostData(unsigned nbytes)
{
	//get pinned memory object (hopefully, since its not guaranteed)
	pinned_host_buffers[num_pinned_host_buffers] =
			clCreateBuffer(myGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, nbytes, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"AllocPinnedHostData",__FILE__,__LINE__);

    //get normal host pointer to that pinned memory object
    pinned_host_buffers_pointers[num_pinned_host_buffers] =
    		clEnqueueMapBuffer(defaultQueue, pinned_host_buffers[num_pinned_host_buffers],
    				CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, nbytes, 0, NULL, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
     	Error(ciErrNum,"AllocPinnedHostData",__FILE__,__LINE__);

    num_pinned_host_buffers++;

    //increase list of pinned memory objects if necessary
    if(num_pinned_host_buffers == max_pinned_host_buffers)
    {
    	max_pinned_host_buffers*=2;
    	cl_mem* tmp_bufs = new cl_mem[max_pinned_host_buffers];
    	void** tmp_buf_ps = new void*[max_pinned_host_buffers];
    	for(unsigned int i = 0; i< num_pinned_host_buffers; i++)
    	{
    		tmp_bufs[i] = pinned_host_buffers[i];
    		tmp_buf_ps[i] = pinned_host_buffers_pointers[i];
    		delete [] pinned_host_buffers;
    		delete [] pinned_host_buffers_pointers;
    		pinned_host_buffers = tmp_bufs;
    		pinned_host_buffers_pointers = tmp_buf_ps;
    	}
    }
    return pinned_host_buffers_pointers[num_pinned_host_buffers-1];
}

void OpenCLWrapper::FreePinnedHostData(void* host_data)
{
	for(unsigned int i = 0; i < num_pinned_host_buffers; i++)
	{
		if(host_data == pinned_host_buffers_pointers[i])
		{
			clEnqueueUnmapMemObject(defaultQueue, pinned_host_buffers[i], host_data, 0, NULL, NULL);
			clReleaseMemObject(pinned_host_buffers[i]);
			pinned_host_buffers[i] = pinned_host_buffers[num_pinned_host_buffers-1];
			pinned_host_buffers_pointers[i] = pinned_host_buffers_pointers[num_pinned_host_buffers-1];
			num_pinned_host_buffers--;
		}
	}
}

cl_mem OpenCLWrapper::AllocDevDataImageFloat4(unsigned elements, int &imagesizei)
{
	if(imagesizei<1)
	{
		double sqr_elements=sqrt(1.0*elements);
		imagesizei = 1;
		while(imagesizei<sqr_elements) imagesizei*=2;
	}
	size_t imagesize=imagesizei;
	cl_mem image=clCreateImage2D(myGPUContext, CL_MEM_READ_ONLY, (cl_image_format[]){CL_RGBA, CL_FLOAT},imagesize,imagesize, 0, NULL, &ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"UploadData",__FILE__,__LINE__);
    return image;
}

int OpenCLWrapper::UploadData(void* host_data, cl_mem dev_data, unsigned nbytes, size_t offset)
{
	//if(is_cpu)
	//ciErrNum = clEnqueueUnmapMemObject(defaultQueue, dev_data, host_data, 0, NULL, NULL);
	//else
  	ciErrNum = clEnqueueWriteBuffer(defaultQueue,dev_data,CL_TRUE, offset,nbytes,host_data,0,NULL,NULL);

    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"UploadData",__FILE__,__LINE__);
	return ciErrNum;
}

int OpenCLWrapper::DownloadData(void* host_data, cl_mem dev_data, unsigned nbytes, size_t offset)
{
	//if(is_cpu)
	//	clEnqueueMapBuffer(defaultQueue, dev_data,
	//		CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, nbytes, 0, NULL, NULL, &ciErrNum);
	//else
  	ciErrNum = clEnqueueReadBuffer(defaultQueue,dev_data,CL_TRUE, offset,nbytes,host_data,0,NULL,NULL);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"DownloadData",__FILE__,__LINE__);
	return ciErrNum;
}

int OpenCLWrapper::UploadDataAsync(void* host_data, cl_mem dev_data, unsigned nbytes, unsigned int stream, size_t offset)
{
	if(stream>=num_queues)
	{
		Error(99,"Upload Async Invalid Stream number",__FILE__,__LINE__);
		return 99;
	}
  	ciErrNum = clEnqueueWriteBuffer(queues[stream],dev_data,CL_FALSE, offset,nbytes,host_data,0,NULL,NULL);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"UploadData",__FILE__,__LINE__);
	return ciErrNum;
}

int OpenCLWrapper::DownloadDataAsync(void* host_data, cl_mem dev_data, unsigned nbytes, unsigned int stream, size_t offset)
{
	if(stream>=num_queues)
	{
		Error(99,"Download Async Invalid Stream number",__FILE__,__LINE__);
		return 99;
	}
  	ciErrNum = clEnqueueReadBuffer(queues[stream],dev_data,CL_FALSE, offset,nbytes,host_data,0,NULL,NULL);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"DownloadData",__FILE__,__LINE__);
	return ciErrNum;
}

int OpenCLWrapper::CopyBufferToImageFloat4(cl_mem src_buffer, cl_mem dst_image)
{
	size_t region[3];
	ciErrNum = clGetImageInfo(dst_image,CL_IMAGE_WIDTH,sizeof(size_t),&region[0],NULL);
	ciErrNum |= clGetImageInfo(dst_image,CL_IMAGE_HEIGHT,sizeof(size_t),&region[1],NULL);

	region[2]=1;
	   if (ciErrNum != CL_SUCCESS)
	    	Error(ciErrNum,"DownloadData",__FILE__,__LINE__);
	ciErrNum = clEnqueueCopyBufferToImage(defaultQueue,src_buffer,dst_image,0,(size_t[]){0,0,0},region,0,NULL,NULL);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"DownloadData",__FILE__,__LINE__);
	return ciErrNum;
}

int OpenCLWrapper::Memset(cl_mem dev_data, unsigned nbytes, int value)
{

	return 0;
}

cl_mem OpenCLWrapper::BufferResize(cl_uint newsize)
{
	if(buffer) FreeDevData(buffer,buffersize);
	buffer = AllocDevData(newsize);
	buffersize = newsize;
	return buffer;
}

cl_mem OpenCLWrapper::BufferGrow(cl_uint newsize)
{
	if(newsize<=buffersize) return buffer;

	if(buffer) FreeDevData(buffer,buffersize);
	buffer = AllocDevData(newsize);
	buffersize = newsize;
	return buffer;
}

int OpenCLWrapper::ReadKernelSource(const char* file)
{
  FILE *fp;
  unsigned long int length;
  char *buf;

  fp=fopen(file,"rb");
  if(fp==NULL) {printf("Can't find kernel source %s\n",file);return -1;}
  fseek(fp,0,SEEK_END);
  length = ftell(fp);

  fseek(fp,0,SEEK_SET);

  buf = new char[length+1];
  fread(buf,length,1,fp);
  buf[length]=0;
  fclose(fp);

  kernel_source[num_kernel_sources] = buf;

  num_kernel_sources++;

  if(num_kernel_sources==max_kernels)
  {
	  char** tmp_kernel_source = new char*[max_kernels*2];
	  cl_kernel* tmp_kernels = new cl_kernel[max_kernels*2];
	  char** tmp_kernel_names = new char*[max_kernels*2];
	  for(unsigned int i=0;i<max_kernels;i++)
	  {
		  tmp_kernel_source[i] = kernel_source[i];
		  tmp_kernels[i] = kernels[i];
		  tmp_kernel_names[i] = kernel_names[i];
	  }
	  max_kernels*=2;

	  delete [] kernel_source;
	  kernel_source = tmp_kernel_source;
	  delete [] kernels;
	  kernels = tmp_kernels;
	  delete [] kernel_names;
	  kernel_names = tmp_kernel_names;
  }

  return num_kernel_sources-1;
}

int OpenCLWrapper::CompileProgram(char* options)
{
	program = clCreateProgramWithSource(myGPUContext,num_kernel_sources,(const char**) kernel_source,NULL,&ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"CreateProgram",__FILE__,__LINE__);
	ciErrNum = clBuildProgram(program,1,&myDevice,options,NULL,NULL);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"BuildProgram",__FILE__,__LINE__);
    if(ciErrNum==-11)
    {
      cl_build_status build_status;
      clGetProgramBuildInfo(program, myDevice, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
      if (build_status == CL_SUCCESS)
                    return ciErrNum;

      char *build_log;
      size_t ret_val_size;
      clGetProgramBuildInfo(program, myDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
      build_log = new char[ret_val_size+1];
      clGetProgramBuildInfo(program, myDevice, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

      // to be carefully, terminate with \0
      // there's no information in the reference whether the string is 0 terminated or not
      build_log[ret_val_size] = '\0';

      //std::cout << "BUILD LOG: " << std::endl;
      printf("%s", build_log);
      exit(0);
    }
	return ciErrNum;
}

int OpenCLWrapper::CreateKernel(const char* kernel_name)
{
	char* tmp_kernel_name = new char[strlen(kernel_name)+1];
	strcpy(tmp_kernel_name,kernel_name);
	kernel_names[num_kernels] = tmp_kernel_name;
	kernels[num_kernels] = clCreateKernel(program,tmp_kernel_name,&ciErrNum);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"CreateKernel",__FILE__,__LINE__);
    num_kernels++;
	return num_kernels-1;
}

int OpenCLWrapper::LaunchKernel(int kernel_num, int glob_threads, int nargs, ...)
{
	va_list args;
	va_start(args,nargs);
	ciErrNum = 0;
	for(int i=0; i<nargs; i++)
	{
		void* arg = va_arg(args,void*);
		unsigned int size = va_arg(args,unsigned int);
		ciErrNum = clSetKernelArg(kernels[kernel_num],i,size,arg);
	    if (ciErrNum != CL_SUCCESS)
	    	Error(ciErrNum,"SetKernelArgs",__FILE__,__LINE__);
	}
	va_end(args);

	size_t grid[1];
	grid[0] = ((glob_threads+blockdim-1)/blockdim)*blockdim;

	size_t block[1];
	block[0] = blockdim;
	ciErrNum = clEnqueueNDRangeKernel(defaultQueue,kernels[kernel_num],1,NULL,grid,block,0,NULL,NULL);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"LaunchKernel",__FILE__,__LINE__);
	return ciErrNum;
}

int OpenCLWrapper::LaunchKernel(const char* kernel_name, int glob_threads, int nargs, ...)
{
	int kernel_num=-1;
	for(int i=0;i<num_kernels;i++)
		if(strcmp(kernel_name,kernel_names[i])==0) kernel_num=i;
	if(kernel_num==-1) {printf("Kernel name not found. Exiting."); exit(0);}
	va_list args;
	va_start(args,nargs);
	ciErrNum = 0;
	for(int i=0; i<nargs; i++)
	{
		void* arg = va_arg(args,void*);
		unsigned int size = va_arg(args,unsigned int);
		ciErrNum = clSetKernelArg(kernels[kernel_num],i,size,arg);
	    if (ciErrNum != CL_SUCCESS)
	    	Error(ciErrNum,"SetKernelArgs",__FILE__,__LINE__);
	}
	va_end(args);

	size_t grid[1];
	grid[0] = ((glob_threads+blockdim-1)/blockdim)*blockdim;

	size_t block[1];
	block[0] = blockdim;
	ciErrNum = clEnqueueNDRangeKernel(defaultQueue,kernels[kernel_num],1,NULL,grid,block,0,NULL,NULL);
    if (ciErrNum != CL_SUCCESS)
    	Error(ciErrNum,"LaunchKernel",__FILE__,__LINE__);
	return ciErrNum;
}

void OpenCLWrapper::Error(cl_int ciErrNUM,const char message[],const char file[],int line)
{
	   static const char* errorString[] = {
	        "CL_SUCCESS",
	        "CL_DEVICE_NOT_FOUND",
	        "CL_DEVICE_NOT_AVAILABLE",
	        "CL_COMPILER_NOT_AVAILABLE",
	        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
	        "CL_OUT_OF_RESOURCES",
	        "CL_OUT_OF_HOST_MEMORY",
	        "CL_PROFILING_INFO_NOT_AVAILABLE",
	        "CL_MEM_COPY_OVERLAP",
	        "CL_IMAGE_FORMAT_MISMATCH",
	        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
	        "CL_BUILD_PROGRAM_FAILURE",
	        "CL_MAP_FAILURE",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "",
	        "CL_INVALID_VALUE",
	        "CL_INVALID_DEVICE_TYPE",
	        "CL_INVALID_PLATFORM",
	        "CL_INVALID_DEVICE",
	        "CL_INVALID_CONTEXT",
	        "CL_INVALID_QUEUE_PROPERTIES",
	        "CL_INVALID_COMMAND_QUEUE",
	        "CL_INVALID_HOST_PTR",
	        "CL_INVALID_MEM_OBJECT",
	        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
	        "CL_INVALID_IMAGE_SIZE",
	        "CL_INVALID_SAMPLER",
	        "CL_INVALID_BINARY",
	        "CL_INVALID_BUILD_OPTIONS",
	        "CL_INVALID_PROGRAM",
	        "CL_INVALID_PROGRAM_EXECUTABLE",
	        "CL_INVALID_KERNEL_NAME",
	        "CL_INVALID_KERNEL_DEFINITION",
	        "CL_INVALID_KERNEL",
	        "CL_INVALID_ARG_INDEX",
	        "CL_INVALID_ARG_VALUE",
	        "CL_INVALID_ARG_SIZE",
	        "CL_INVALID_KERNEL_ARGS",
	        "CL_INVALID_WORK_DIMENSION",
	        "CL_INVALID_WORK_GROUP_SIZE",
	        "CL_INVALID_WORK_ITEM_SIZE",
	        "CL_INVALID_GLOBAL_OFFSET",
	        "CL_INVALID_EVENT_WAIT_LIST",
	        "CL_INVALID_EVENT",
	        "CL_INVALID_OPERATION",
	        "CL_INVALID_GL_OBJECT",
	        "CL_INVALID_BUFFER_SIZE",
	        "CL_INVALID_MIP_LEVEL",
	        "CL_INVALID_GLOBAL_WORK_SIZE",
	    };

	    const int errorCount = sizeof(errorString) / sizeof(errorString[0]);
	    const int index = -ciErrNum;

	    if(index >= 0 && index < errorCount) printf("OpenCL Error: %i %s %s in file '%s' at line %i\n",ciErrNum,errorString[index],message,file,line);
	    else printf("OpenCL unspecified Error %i\n",ciErrNum);
}

