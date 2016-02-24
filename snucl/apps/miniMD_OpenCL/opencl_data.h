/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator 

   Original Version:
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov 

   See the README file in the top-level LAMMPS directory. 

   ----------------------------------------------------------------------- 

   USER-CUDA Package and associated modifications:
   https://sourceforge.net/projects/lammpscuda/ 

   Christian Trott, christian.trott@tu-ilmenau.de
   Lars Winterfeld, lars.winterfeld@tu-ilmenau.de
   Theoretical Physics II, University of Technology Ilmenau, Germany 

   See the README file in the USER-CUDA directory. 

   This software is distributed under the GNU General Public License.
------------------------------------------------------------------------- */

#ifndef _OPENCL_DATA_H_
#define _OPENCL_DATA_H_


enum copy_mode {x, xx, xy, yx, xyz, xzy}; // yxz, yzx, zxy, zyx not yet implemented since they were not needed yet
//xx==x in atom_vec x is a member therefore copymode x produces compile errors
#include "opencl_wrapper.h"
#include <ctime>

#include <cstdio>
#include <typeinfo>

template <typename host_type, typename dev_type, copy_mode mode>
class cOpenCLData
{
	protected:
	OpenCLWrapper* wrapper;
	void** buffer;
	int* buf_size;
	host_type* host_data;
	cl_mem dev_data;
	cl_mem dev_image;
	dev_type* temp_data;
	unsigned int dim[3];
	unsigned nbytes;
	bool is_continues;
	bool pinned;
	bool owns_host_data;
	

	public:
	unsigned nbytes_device;
	int imagesize;
	cOpenCLData(OpenCLWrapper* ocl_wrapper, unsigned dim_x, unsigned dim_y=0, unsigned dim_z=0, bool is_pinned=false, bool use_image=false);
	cOpenCLData(OpenCLWrapper* ocl_wrapper, host_type* host_data, unsigned dim_x, unsigned dim_y=0, unsigned dim_z=0, bool is_pinned=false, bool use_image=false);
	~cOpenCLData();
	cl_mem devData() {return dev_data;};
	cl_mem* devDataRef() {return &dev_data;};
	cl_mem* devImageRef() {return &dev_image;};
	void setHostData(host_type* host_data);
	host_type* hostData() { return host_data;};
	unsigned int* getDim() {return dim;};

	void syncImage() {};//wrapper->CopyBufferToImageFloat4(dev_data,dev_image);};

	void upload();
	void uploadAsync(unsigned int stream);
	void download();
	void downloadAsync(unsigned int stream);
	void memsetDevice(int value);
	int devSize() {return nbytes;}
};



template <typename host_type, typename dev_type, copy_mode mode>
cOpenCLData<host_type, dev_type, mode>
::cOpenCLData(OpenCLWrapper* ocl_wrapper, unsigned dim_x, unsigned dim_y, unsigned dim_z, bool is_pinned, bool use_image)
{
	wrapper = ocl_wrapper;
	pinned = is_pinned;
	is_continues = true;
	owns_host_data = true;
	temp_data = NULL;
  buffer = NULL;
  buf_size = 0;

	imagesize = 0;

	unsigned ndev;
	if((mode == x)||(mode==xx))
	{
		ndev = dim_x;
		dim[0] = dim_x;
		dim[1] = 0;
		dim[2] = 0;
		is_continues = true;
	}
	else if(mode == xy || mode == yx )
	{
		ndev = dim_x * dim_y;
		dim[0] = dim_x;
		dim[1] = dim_y;
		dim[2] = 0;
	}
	else
	{
		ndev = dim_x * dim_y * dim_z;
		dim[0] = dim_x;
		dim[1] = dim_y;
		dim[2] = dim_z;
	}

	nbytes = ndev * sizeof(dev_type);
	if(nbytes<=0)
	{
		this->host_data=NULL;
		temp_data=NULL;
		dev_data=NULL;
		return;
	}

	nbytes_device = nbytes;

	if(use_image)
	{
	  imagesize = 1;
	  while(imagesize*imagesize*sizeof(cl_float4)<nbytes) imagesize*=2;
	  nbytes_device = imagesize*imagesize*sizeof(cl_float4);
	}

	dev_data = wrapper->AllocDevData(nbytes_device);

	host_type* host_tmp;
	if(pinned) 	host_tmp = (host_type*) wrapper->AllocPinnedHostData(ndev*sizeof(host_type));
	else		host_tmp = new host_type[ndev];
	if((mode==x)||(mode==xx))
		host_data = host_tmp;
	if((mode==xy)||(mode==yx))
	{
		host_type** host_tmpx = new host_type*[dim[0]];
		for(unsigned int i=0;i<dim[0];i++)
			host_tmpx[i] = &host_tmp[i*dim[1]];
		host_data = (host_type*) host_tmpx;
	}
	if((mode==xyz)||(mode==xzy))
	{
		host_type*** host_tmpx = new host_type**[dim[0]];
		for(unsigned int i=0;i<dim[0];i++)
		{
			host_tmpx[i] = new host_type*[dim[1]];
			for(unsigned int j=0;j<dim[1];j++)
				host_tmpx[i][j] = &host_tmp[(i*dim[1]+j)*dim[2]];
		}
		host_data = (host_type*) host_tmpx;
	}

	if(((mode!=x)&&(mode!=xx)&&(!((mode==xy)&&is_continues))&&(!((mode==xyz)&&is_continues))) || (typeid(host_type) != typeid(dev_type)))
	{
		if(not pinned)
		temp_data = new dev_type[ndev];
		else
		{
			temp_data = (dev_type*) wrapper->AllocPinnedHostData(ndev*sizeof(dev_type));
		}
	}

	//dev_image = wrapper->AllocDevDataImageFloat4(0,imagesize);
}

template <typename host_type, typename dev_type, copy_mode mode>
cOpenCLData<host_type, dev_type, mode>
::cOpenCLData(OpenCLWrapper* ocl_wrapper, host_type* host_data, unsigned dim_x, unsigned dim_y, unsigned dim_z, bool is_pinned, bool use_image)
{
	wrapper = ocl_wrapper;
	pinned = is_pinned;
	is_continues = false;
	owns_host_data = false;
  temp_data = NULL;
  buffer = NULL;
  buf_size = 0;

	imagesize = 0;

	this->host_data = host_data;
	unsigned ndev;
	if((mode == x)||(mode==xx))
	{
		ndev = dim_x;
		dim[0] = dim_x;
		dim[1] = 0;
		dim[2] = 0;
		is_continues = true;
	}
	else if(mode == xy || mode == yx )
	{
		ndev = dim_x * dim_y;
		dim[0] = dim_x;
		dim[1] = dim_y;
		dim[2] = 0;
	}
	else
	{
		ndev = dim_x * dim_y * dim_z;
		dim[0] = dim_x;
		dim[1] = dim_y;
		dim[2] = dim_z;
	}
	
	nbytes = ndev * sizeof(dev_type);
	if(nbytes<=0)
	{
		this->host_data=NULL;
		temp_data=NULL;
		dev_data=NULL;
		return;
	}
	
	nbytes_device = nbytes;

	if(use_image)
	{
	  imagesize = 1;
	  while(imagesize*imagesize*sizeof(cl_float4)<nbytes) imagesize*=2;
	  nbytes_device = imagesize*imagesize*sizeof(cl_float4);
	}

	dev_data = wrapper->AllocDevData(nbytes_device);
	if(((mode!=x)&&(mode!=xx)) || (typeid(host_type) != typeid(dev_type)))
	{
		if(not pinned)
		temp_data = new dev_type[ndev];
		else
		{
			temp_data = (dev_type*) wrapper->AllocPinnedHostData(ndev*sizeof(dev_type));
		}
	}

	//dev_image = wrapper->AllocDevDataImageFloat4(0,imagesize);
}

template <typename host_type, typename dev_type, copy_mode mode>
cOpenCLData<host_type, dev_type, mode>
::~cOpenCLData()
{
	if(owns_host_data)
	{
		host_type* host_tmp;
		if((mode==x)||(mode==xx)) host_tmp=host_data;
		if((mode==xy)||(mode==yx))
		{
			host_tmp=&((host_type**)host_data)[0][0];
			delete [] (host_type**)host_data;
		}
		if((mode==xyz)||(mode==xzy))
		{
			host_tmp=&((host_type***)host_data)[0][0][0];
			for(unsigned int i=0;i<dim[0];i++)
			delete [] ((host_type***)host_data)[i];
			delete [] (host_type***)host_data;
		}
		if(not pinned) delete [] host_tmp;
		else wrapper->FreePinnedHostData(host_tmp);
	}
	if(temp_data)
	{
		if(not pinned)
		delete [] temp_data;
		else
		{
			wrapper->FreePinnedHostData((void*)temp_data);
		}
	}
	if((dev_data)&&(nbytes>0))
	wrapper->FreeDevData(dev_data,nbytes_device);
}

template <typename host_type, typename dev_type, copy_mode mode>
void cOpenCLData<host_type, dev_type, mode>
::setHostData(host_type* host_data)
{
	this->host_data = host_data;
}

template <typename host_type, typename dev_type, copy_mode mode>
void cOpenCLData<host_type, dev_type, mode>
::upload()
{
	switch(mode)
	{
		case x:
		{
			if(typeid(host_type) == typeid(dev_type))
				wrapper->UploadData(host_data, dev_data, nbytes);
			else
			{
 			  for(unsigned i=0; i<dim[0]; ++i) temp_data[i] = static_cast<dev_type>(host_data[i]);
			  wrapper->UploadData(temp_data, dev_data, nbytes);
			}
			break;
		}
		
		case xx:
		{
			if(typeid(host_type) == typeid(dev_type))
				wrapper->UploadData(host_data,  dev_data, nbytes);
			else
			{
				for(unsigned i=0; i< dim[0]; ++i) temp_data[i] = static_cast<dev_type>(host_data[i]);
				wrapper->UploadData(temp_data,  dev_data, nbytes);
			}
			break;
		}

		case xy:
		{
			for(unsigned i=0; i<dim[0]; ++i)
			{
				dev_type* temp = &temp_data[i * dim[1]];
				for(unsigned j=0; j< dim[1]; ++j)
				{
					temp[j] = static_cast<dev_type>((reinterpret_cast<host_type**>(host_data))[i][j]);
				}
			}
			wrapper->UploadData(temp_data, dev_data, nbytes);
			break;
		}
		
		case yx:
		{
			for(unsigned j=0; j< dim[1]; ++j)
			{
				dev_type* temp = &temp_data[j * dim[0]];
				for(unsigned i=0; i< dim[0]; ++i)
				{
					temp[i] = static_cast<dev_type>(reinterpret_cast<host_type**>(host_data)[i][j]);
				}
			}
			wrapper->UploadData(temp_data,  dev_data, nbytes);
			break;
		}	
		case xyz:
		{
			for(unsigned i=0; i < dim[0]; ++i)
			for(unsigned j=0; j < dim[1]; ++j)
			{
				dev_type* temp = &temp_data[(i * dim[1] + j) * dim[2]];
				for(unsigned k=0; k < dim[2]; ++k)
				{
					temp[k] = static_cast<dev_type>(reinterpret_cast<host_type***>(host_data)[i][j][k]);
				}
			}
			wrapper->UploadData(temp_data,  dev_data, nbytes);
			break;
		}	

		case xzy:
		{
			for(unsigned i=0; i< dim[0]; ++i)
			for(unsigned k=0; k< dim[2]; ++k)
			{
				dev_type* temp = &temp_data[(i* dim[2]+k)* dim[1]];
				for(unsigned j=0; j< dim[1]; ++j)
				{
					temp[j] = static_cast<dev_type>(reinterpret_cast<host_type***>(host_data)[i][j][k]);
				}
			}
			wrapper->UploadData(temp_data,  dev_data, nbytes);
			break;
		}	
	}
}

template <typename host_type, typename dev_type, copy_mode mode>
void cOpenCLData<host_type, dev_type, mode>
::uploadAsync(unsigned int stream)
{
	switch(mode)
	{
		case x:
		{
			if(typeid(host_type) == typeid(dev_type))
				wrapper->UploadDataAsync(host_data, dev_data, nbytes, stream);
			else
			{
 			  for(unsigned i=0; i<dim[0]; ++i) temp_data[i] = static_cast<dev_type>(host_data[i]);
			  wrapper->UploadDataAsync(temp_data, dev_data, nbytes, stream);
			}
			break;
		}
		
		case xx:
		{
			if(typeid(host_type) == typeid(dev_type))
				wrapper->UploadDataAsync(host_data, dev_data, nbytes, stream);
			else
			{
				for(unsigned i=0; i< dim[0]; ++i) temp_data[i] = static_cast<dev_type>(host_data[i]);
				wrapper->UploadDataAsync(temp_data, dev_data, nbytes, stream);
			}
			break;
		}

		case xy:
		{
			for(unsigned i=0; i<dim[0]; ++i)
			{
				dev_type* temp = &temp_data[i * dim[1]];
				for(unsigned j=0; j< dim[1]; ++j)
				{
					temp[j] = static_cast<dev_type>((reinterpret_cast<host_type**>(host_data))[i][j]);
				}
			}
			wrapper->UploadDataAsync(temp_data, dev_data, nbytes, stream);
			break;
		}
		
		case yx:
		{
			for(unsigned j=0; j< dim[1]; ++j)
			{
				dev_type* temp = &temp_data[j * dim[0]];
				for(unsigned i=0; i< dim[0]; ++i)
				{
					temp[i] = static_cast<dev_type>(reinterpret_cast<host_type**>(host_data)[i][j]);
				}
			}
			wrapper->UploadDataAsync(temp_data, dev_data, nbytes, stream);
			break;
		}	
		case xyz:
		{
			for(unsigned i=0; i < dim[0]; ++i)
			for(unsigned j=0; j < dim[1]; ++j)
			{
				dev_type* temp = &temp_data[(i * dim[1] + j) * dim[2]];
				for(unsigned k=0; k < dim[2]; ++k)
				{
					temp[k] = static_cast<dev_type>(reinterpret_cast<host_type***>(host_data)[i][j][k]);
				}
			}
			wrapper->UploadDataAsync(temp_data, dev_data, nbytes, stream);
			break;
		}	

		case xzy:
		{
			for(unsigned i=0; i< dim[0]; ++i)
			for(unsigned k=0; k< dim[2]; ++k)
			{
				dev_type* temp = &temp_data[(i* dim[2]+k)* dim[1]];
				for(unsigned j=0; j< dim[1]; ++j)
				{
					temp[j] = static_cast<dev_type>(reinterpret_cast<host_type***>(host_data)[i][j][k]);
				}
			}
			wrapper->UploadDataAsync(temp_data, dev_data, nbytes, stream);
			break;
		}	
	}
}

template <typename host_type, typename dev_type, copy_mode mode>
void cOpenCLData<host_type, dev_type, mode>
::download()
{
	switch(mode)
	{
		case x:
		{
			if(typeid(host_type) == typeid(dev_type))
				wrapper->DownloadData(host_data,  dev_data, nbytes);
			else
			{
				wrapper->DownloadData(temp_data,  dev_data, nbytes);
 				for(unsigned i=0; i< dim[0]; ++i) host_data[i] = static_cast<host_type>(temp_data[i]);
			}
			break;
		}
		
		case xx:
		{
			if(typeid(host_type) == typeid(dev_type))
				wrapper->DownloadData(host_data,  dev_data, nbytes);
			else
			{
				wrapper->DownloadData(temp_data,  dev_data, nbytes);
				for(unsigned i=0; i< dim[0]; ++i) host_data[i] = static_cast<host_type>(temp_data[i]);
			}
			break;
		}
		
		case xy:
		{
			wrapper->DownloadData(temp_data,  dev_data, nbytes);
			for(unsigned i=0; i< dim[0]; ++i)
			{
				dev_type* temp = &temp_data[i *  dim[1]];
				for(unsigned j=0; j< dim[1]; ++j)
				{
					reinterpret_cast<host_type**>(host_data)[i][j] = static_cast<host_type>(temp[j]);
				}
			}
			break;
		}
		
		case yx:
		{
			wrapper->DownloadData(temp_data,  dev_data, nbytes);
			for(unsigned j=0; j< dim[1]; ++j)
			{
				dev_type* temp = &temp_data[j* dim[0]];
				for(unsigned i=0; i< dim[0]; ++i)
				{
					reinterpret_cast<host_type**>(host_data)[i][j] = static_cast<host_type>(temp[i]);
				}
			}
			break;
		}

		case xyz:
		{
			wrapper->DownloadData(temp_data,  dev_data, nbytes);
			for(unsigned i=0; i< dim[0]; ++i)
			for(unsigned j=0; j< dim[1]; ++j)
			{
				dev_type* temp = &temp_data[(i *  dim[1]+j)* dim[2]];
				for(unsigned k=0; k< dim[2]; ++k)
				{
					reinterpret_cast<host_type***>(host_data)[i][j][k] = static_cast<host_type>(temp[k]);
				}
			}
			break;
		}

		case xzy:
		{
			wrapper->DownloadData(temp_data,  dev_data, nbytes);
			for(unsigned i=0; i< dim[0]; ++i)
			for(unsigned k=0; k< dim[2]; ++k)
			{
				dev_type* temp = &temp_data[(i *  dim[2]+k)* dim[1]];
				for(unsigned j=0; j< dim[1]; ++j)
				{
					reinterpret_cast<host_type***>(host_data)[i][j][k] = static_cast<host_type>(temp[j]);
				}
			}
			break;
		}
	}
}

template <typename host_type, typename dev_type, copy_mode mode>
void cOpenCLData<host_type, dev_type, mode>
::downloadAsync(unsigned int stream)
{
	switch(mode)
	{
		case x:
		{
			if(typeid(host_type) == typeid(dev_type))
			{
				wrapper->DownloadDataAsync((void*) host_data,  dev_data, nbytes, stream);
				//OpenCLWrapper_SyncStream(stream);
			}		
			else
			{
				wrapper->DownloadDataAsync((void*) temp_data,  dev_data, nbytes, stream);
				//OpenCLWrapper_SyncStream(stream);
				for(unsigned i=0; i< dim[0]; ++i) host_data[i] = static_cast<host_type>(temp_data[i]);
			}
			break;
		}
		
		case xx:
		{
			if(typeid(host_type) == typeid(dev_type))
			{
				wrapper->DownloadDataAsync((void*) host_data,  dev_data, nbytes, stream);
			    //OpenCLWrapper_SyncStream(stream);
			}
			else
			{
				wrapper->DownloadDataAsync((void*) temp_data,  dev_data, nbytes, stream);
 			    //OpenCLWrapper_SyncStream(stream);
				for(unsigned i=0; i< dim[0]; ++i) host_data[i] = static_cast<host_type>(temp_data[i]);
			}
			break;
		}

		case xy:
		{
			wrapper->DownloadDataAsync((void*) temp_data,  dev_data, nbytes, stream);
			//OpenCLWrapper_SyncStream(stream);
			for(unsigned i=0; i< dim[0]; ++i)
			{
				dev_type* temp = &temp_data[i *  dim[1]];
				for(unsigned j=0; j< dim[1]; ++j)
				{
					reinterpret_cast<host_type**>(host_data)[i][j] = static_cast<host_type>(temp[j]);
				}
			}
			break;
		}
		
		case yx:
		{
			wrapper->DownloadDataAsync((void*) temp_data,  dev_data, nbytes, stream);
			//OpenCLWrapper_SyncStream(stream);
			for(unsigned j=0; j< dim[1]; ++j)
			{
				dev_type* temp = &temp_data[j* dim[0]];
				for(unsigned i=0; i< dim[0]; ++i)
				{
					reinterpret_cast<host_type**>(host_data)[i][j] = static_cast<host_type>(temp[i]);
				}
			}
			break;
		}
	}
}


template <typename host_type, typename dev_type, copy_mode mode>
void cOpenCLData<host_type, dev_type, mode>
::memsetDevice(int value)
{
   wrapper->Memset( dev_data,value, nbytes);
}

#endif // _OPENCL_DATA_H_
