#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

extern "C"
{
	__global__ void testKernel(int* addr, unsigned short param1, char param2)
	{
		addr[0] = param1 + param2;
	}
}

char* muGetErrorString(CUresult result);

void muEC(int position) //checks and outputs error position and error string
{
	cudaError_t errcode = cudaGetLastError();
	if(errcode==cudaSuccess)
	{
		printf("No error at position %i\n", position);
		return;
	}
	printf("Error position: %i\nCode:%s\n", position, cudaGetErrorString(errcode));
}

void muRC(int position, CUresult result)
{
	if(result!=0)
		printf("Error at %i:%s\n", position, muGetErrorString(result));
}

char* muGetErrorString(CUresult result)
{
	switch(result)
	{
	case 0:		return "Success";
	case 1:		return "Invalid value";
	case 2:		return "Out of memory";
	case 3:		return "Not Initialized";
	case 4:		return "Deinitialized";

	case 100:	return "No device";
	case 101:	return "Invalid device";

	case 200:	return "Invalid image";
	case 201:	return "Invalid context";
	case 202:	return "Context already current";
	case 205:	return "Map failed";
	case 206:	return "Unmap failed";
	case 207:	return "Array is mapped";
	case 208:	return "Already mapped";
	case 209:	return "No binary for GPU";
	case 210:	return "Already acquired";
	case 211:	return "Not mapped";

	case 300:	return "Invalid source";
	case 301:	return "File not found";

	case 400:	return "Invalid handle";
	case 500:	return "Not found";
	case 600:	return "Not ready";

	case 700:	return "Launch failed";
	case 701:	return "Launch out of resources";
	case 702:	return "Launch timeout";
	case 703:	return "Launch incompatible texturing";

	case 999:	return "Unknown";
	};
	return "Unknown";
}


int main( int argc, char** argv) 
{
	if(argc<3)
	{
		puts("arguments: cubinname kernelname");
		return;
	}
       
	//Thread count
	int tcount = 1;
	if(argc>=4)
	{
		tcount = atoi(argv[3]);
	}

        
        int length = 3*tcount;
	long* cpu_output=new long[length];
	int size = sizeof(long)*length;
	
	//int* cpu_output=new int[length];
	//int size = sizeof(int)*length;
	int interval = 1;
	
        bool odd = true;
	bool even = true;
	if(argc>=5)
	{
		int choice = atoi(argv[4]);
		if(choice==1)
			even = false;
		else if(choice==2)
			odd = false;
	}
	CUdeviceptr gpu_output;
	CUdevice device;
	CUcontext context;

	muRC(100, cuInit(0));
	muRC(95, cuDeviceGet(&device, 0));
	muRC(92, cuCtxCreate(&context, CU_CTX_SCHED_SPIN, device));
	muRC(90, cuMemAlloc(&gpu_output, size));

	//------------Loading the cubin---------
	CUmodule module;
	CUfunction kernel;
	CUresult result = cuModuleLoad(&module, argv[1]);
	muRC(0 , result);
	result = cuModuleGetFunction(&kernel, module, argv[2]);
	muRC(1, result); 
	int param = 0x1010;
        /*
         * cuParamSetSize
         * Sets through numbytes the total size in bytes needed by the function
         * parameters of the kernel corresponding to hfunc.
         * Parameters:
         * hfunc- Kernel to set parameter size for
         * numbytes - Size of parameter list in bytes
         */
	//muRC(2, cuParamSetSize(kernel, 20));
	muRC(2, cuParamSetSize(kernel, 8));

        /* 
         * cuParamSetv
         * Copies an arbitrary amount of data (specified in numbytes) from ptr
         * into the parameter space of the kernel corresponding 
         * to hfunc. offset is a byte offset.
         *   Parameters:
         *   hfunc   - Kernel to add data to
         *   offset  - Offset to add data to argument list
         *   ptr     - Pointer to arbitrary data
         *   numbytes    - Size of data to copy in bytes
         */
	muRC(3, cuParamSetv(kernel, 0, &gpu_output, size));
	//muRC(3, cuParamSetv(kernel, 16, &param, 4));

        /*
         * cuFuncSetBlockShape:
         * Specifies the x, y, and z dimensions of the thread blocks that are
         * created when the kernel given by hfunc is launched.
         * Parameters:
         * hfunc    - Kernel to specify dimensions of
         * x    - X dimension
         * y    - Y dimension
         * z    - Z dimension
         */
	muRC(4, cuFuncSetBlockShape(kernel, tcount,1,1));

	//------------- Launching the kernel from cubin-------	
	muRC(5, cuLaunch(kernel));

	muRC(6, cuMemcpyDtoH(cpu_output, gpu_output, size));
	muRC(7, cuCtxSynchronize());
	printf("length=%i\n", length);
	printf("tcount=%i\n", tcount);

	//---------- Printing the tiing information from each thread.
	printf("Thread \t Time \t Start time \t End time \t Result \n");
	for (int i=0; i<tcount; i++) {
        	//printf("%d %d %d \n", cpu_output[i*3], cpu_output[i*3+1], cpu_output[i*3+2]);
        	printf("%d \t %d \t %d \t %d \t %d \n", i, cpu_output[i*3 + 1] - cpu_output[i*3], cpu_output[i*3], cpu_output[i*3 +1], cpu_output[i*3+2]);
	}
        /*
	for(int i=0; i<length/interval; i++)
	{
		if(i%2==0)
		{
			if(!even) continue;
		}
		else
		{
			if(!odd) continue;
		}
		for(int j=0; j<interval; j++)
			printf("i=%i, j=%i, output=%i\n", i, j, cpu_output[i*interval+j]);
		if(interval!=1)
			puts("");
	}
        */
	muRC(8, cuModuleUnload(module));
	muRC(9, cuMemFree(gpu_output));
	muRC(10, cuCtxDestroy(context));
	delete[] cpu_output;
	return 0;
}
