//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
//
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string>
#include <string.h>
//#include "mpi.h"
#include "timer.h"
#include "switches.h"

//device memory pointers
static int   *nd1_velD;
static int   *nd1_txyD;
static int   *nd1_txzD;
static int   *nd1_tyyD;
static int   *nd1_tyzD;
static double *rhoD;
static double *drvh1D;
static double *drti1D;
static double *drth1D;
static double *damp1_xD;
static double *damp1_yD;
static int   *idmat1D;
static double *dxi1D;
static double *dyi1D;
static double *dzi1D;
static double *dxh1D;
static double *dyh1D;
static double *dzh1D;
static double *t1xxD;
static double *t1xyD;
static double *t1xzD;
static double *t1yyD;
static double *t1yzD;
static double *t1zzD;
static double *t1xx_pxD;
static double *t1xy_pxD;
static double *t1xz_pxD;
static double *t1yy_pxD;
static double *qt1xx_pxD;
static double *qt1xy_pxD;
static double *qt1xz_pxD;
static double *qt1yy_pxD;
static double *t1xx_pyD;
static double *t1xy_pyD;
static double *t1yy_pyD;
static double *t1yz_pyD;
static double *qt1xx_pyD;
static double *qt1xy_pyD;
static double *qt1yy_pyD;
static double *qt1yz_pyD;
static double *qt1xxD;
static double *qt1xyD;
static double *qt1xzD;
static double *qt1yyD;
static double *qt1yzD;
static double *qt1zzD;
static double *clamdaD;
static double *cmuD;
static double *epdtD;
static double *qwpD;
static double *qwsD;
static double *qwt1D;
static double *qwt2D;

double *v1xD;    //output
double *v1yD;
double *v1zD;
static double *v1x_pxD;
static double *v1y_pxD;
static double *v1z_pxD;
static double *v1x_pyD;
static double *v1y_pyD;
static double *v1z_pyD;

//for inner_II---------------------------------------------------------
static int	 *nd2_velD;
static int   *nd2_txyD;  //int[18]
static int   *nd2_txzD;  //int[18]
static int   *nd2_tyyD;  //int[18]
static int   *nd2_tyzD;  //int[18]

static double *drvh2D;
static double *drti2D;
static double *drth2D; 	//double[mw2_pml1,0:1]

static int	 *idmat2D;
static double *damp2_xD;
static double *damp2_yD;
static double *damp2_zD;
static double *dxi2D;
static double *dyi2D;
static double *dzi2D;
static double *dxh2D;
static double *dyh2D;
static double *dzh2D;
static double *t2xxD;
static double *t2xyD;
static double *t2xzD;
static double *t2yyD;
static double *t2yzD;
static double *t2zzD;
static double *qt2xxD;
static double *qt2xyD;
static double *qt2xzD;
static double *qt2yyD;
static double *qt2yzD;
static double *qt2zzD;

static double *t2xx_pxD;
static double *t2xy_pxD;
static double *t2xz_pxD;
static double *t2yy_pxD;
static double *qt2xx_pxD;
static double *qt2xy_pxD;
static double *qt2xz_pxD;
static double *qt2yy_pxD;

static double *t2xx_pyD;
static double *t2xy_pyD;
static double *t2yy_pyD;
static double *t2yz_pyD;
static double *qt2xx_pyD;
static double *qt2xy_pyD;
static double *qt2yy_pyD;
static double *qt2yz_pyD;

static double *t2xx_pzD;
static double *t2xz_pzD;
static double *t2yz_pzD;
static double *t2zz_pzD;
static double *qt2xx_pzD;
static double *qt2xz_pzD;
static double *qt2yz_pzD;
static double *qt2zz_pzD;

double *v2xD;		//output
double *v2yD;
double *v2zD;
static double *v2x_pxD;
static double *v2y_pxD;
static double *v2z_pxD;
static double *v2x_pyD;
static double *v2y_pyD;
static double *v2z_pyD;
static double *v2x_pzD;
static double *v2y_pzD;
static double *v2z_pzD;

// MPI-ACC pointers
 double *sdx51D;
 double *sdx52D;
 double *sdx41D;
 double *sdx42D;
 double *sdy51D;
 double *sdy52D;
 double *sdy41D;
 double *sdy42D;
 double *rcx51D;
 double *rcx52D;
 double *rcx41D;
 double *rcx42D;
 double *rcy51D;
 double *rcy52D;
 double *rcy41D;
 double *rcy42D;

double *sdx1D;
double *sdy1D;
double *sdx2D;
double *sdy2D;
double *rcx1D;
double *rcy1D;
double *rcx2D;
double *rcy2D;

static double* cixD;
static double* ciyD;
static double* chxD;
static double* chyD;

static int *index_xyz_sourceD;
static double *fampD;
static double *sparam2D;
static double *risetD;
static double *ruptmD;
static double *sutmArrD;

double* temp_gpu_arr; // used by get_gpu_arrC
int rank;

#ifdef __cplusplus
extern "C" {
#endif
void record_time(double* time);

// A GPU Array metadata Structure
struct GPUArray_Metadata {
    char* arr_name;
    int dimensions;
    int i;
    int j;
    int k;
    double* dptr_1D;
    double* dptr_2D;
    double* dptr_3D;
} 
v1xD_meta, v2xD_meta, v1yD_meta, v2yD_meta, v1zD_meta, v2zD_meta,
t1xxD_meta, t1xyD_meta, t1xzD_meta, t1yyD_meta, t1yzD_meta, t1zzD_meta,
t2xxD_meta, t2xyD_meta, t2xzD_meta, t2yyD_meta, t2yzD_meta, t2zzD_meta,
sdx1D_meta, sdx2D_meta, sdy1D_meta, sdy2D_meta,
rcx1D_meta, rcx2D_meta, rcy1D_meta, rcy2D_meta,
sdx41D_meta, sdx42D_meta, sdy41D_meta, sdy42D_meta,
sdx51D_meta, sdx52D_meta, sdy51D_meta, sdy52D_meta,
rcx41D_meta, rcx42D_meta, rcy41D_meta, rcy42D_meta,
rcx51D_meta, rcx52D_meta, rcy51D_meta, rcy52D_meta;


/* 
    Initialises all the GPU array metadata
    IMPORTANT : MUST BE IN SYNC WITH Array_Info metadata in disfd_comm.f90
*/
void init_gpuarr_metadata (int nxtop, int nytop, int nztop, int nxbtm, int nybtm, int nzbtm) 
{
    v1xD_meta = (struct GPUArray_Metadata){"v1x", 3,nztop+2, nxtop+3,  nytop+3, NULL, NULL, v1xD}; 
    v1yD_meta = (struct GPUArray_Metadata){"v1y", 3,nztop+2, nxtop+3,  nytop+3, NULL, NULL, v1yD}; 
    v1zD_meta = (struct GPUArray_Metadata){"v1z", 3,nztop+2, nxtop+3,  nytop+3, NULL, NULL, v1zD};
    v2xD_meta = (struct GPUArray_Metadata){"v2x", 3,nzbtm+1, nxbtm+3,  nybtm+3, NULL, NULL, v2xD}; 
    v2yD_meta = (struct GPUArray_Metadata){"v2y", 3,nzbtm+1, nxbtm+3,  nybtm+3, NULL, NULL, v2yD};
    v2zD_meta = (struct GPUArray_Metadata){"v2z", 3,nzbtm+1, nxbtm+3,  nybtm+3, NULL, NULL, v2zD}; 

    t1xxD_meta = (struct GPUArray_Metadata){"t1xx", 3, nztop, nxtop+3, nytop, NULL, NULL  ,  t1xxD};
    t1xyD_meta = (struct GPUArray_Metadata){"t1xy", 3, nztop, nxtop+3, nytop+3, NULL, NULL,  t1xyD};
    t1xzD_meta = (struct GPUArray_Metadata){"t1xz", 3, nztop+1, nxtop+3, nytop, NULL, NULL,  t1xzD};
    t1yyD_meta = (struct GPUArray_Metadata){"t1yy", 3, nztop, nxtop, nytop+3, NULL, NULL  ,  t1yyD};
    t1yzD_meta = (struct GPUArray_Metadata){"t1yz", 3, nztop+1, nxtop, nytop+3, NULL, NULL,  t1yzD};
    t1zzD_meta = (struct GPUArray_Metadata){"t1zz", 3, nztop, nxtop, nytop, NULL, NULL    ,  t1zzD};

    t2xxD_meta = (struct GPUArray_Metadata){"t2xx", 3, nzbtm, nxbtm+3, nybtm, NULL, NULL  ,  t2xxD};
    t2xyD_meta = (struct GPUArray_Metadata){"t2xy", 3, nzbtm, nxbtm+3, nybtm+3, NULL, NULL,  t2xyD};
    t2xzD_meta = (struct GPUArray_Metadata){"t2xz", 3, nzbtm+1, nxbtm+3, nybtm, NULL, NULL,  t2xzD};
    t2yyD_meta = (struct GPUArray_Metadata){"t2yy", 3, nzbtm, nxbtm, nybtm+3, NULL, NULL  ,  t2yyD};
    t2yzD_meta = (struct GPUArray_Metadata){"t2yz", 3, nzbtm+1, nxbtm, nybtm+3, NULL, NULL,  t2yzD};
    t2zzD_meta = (struct GPUArray_Metadata){"t2zz", 3, nzbtm+1, nxbtm, nybtm, NULL, NULL  ,  t2zzD};

    sdx1D_meta = (struct GPUArray_Metadata){"sdx1", 1, nytop+6, 0, 0, sdx1D,  NULL, NULL}; 
    sdx2D_meta = (struct GPUArray_Metadata){"sdx2", 1, nytop+6, 0, 0, sdx2D,  NULL, NULL}; 
    sdy1D_meta = (struct GPUArray_Metadata){"sdy1", 1, nxtop+6, 0, 0, sdy1D,  NULL, NULL}; 
    sdy2D_meta = (struct GPUArray_Metadata){"sdy2", 1, nxtop+6, 0, 0, sdy2D,  NULL, NULL}; 
    rcx1D_meta = (struct GPUArray_Metadata){"rcx1", 1, nytop+6, 0, 0, rcx1D,  NULL, NULL}; 
    rcx2D_meta = (struct GPUArray_Metadata){"rcx2", 1, nytop+6, 0, 0, rcx2D,  NULL, NULL}; 
    rcy1D_meta = (struct GPUArray_Metadata){"rcy1", 1, nxtop+6, 0, 0, rcy1D,  NULL, NULL}; 
    rcy2D_meta = (struct GPUArray_Metadata){"rcy2", 1, nxtop+6, 0, 0, rcy2D,  NULL, NULL}; 

    sdx41D_meta = (struct GPUArray_Metadata){"sdx41", 3, nztop, nytop, 4, NULL, NULL, sdx41D};
    sdx42D_meta = (struct GPUArray_Metadata){"sdx42", 3, nzbtm, nybtm, 4, NULL, NULL, sdx42D};
    sdy41D_meta = (struct GPUArray_Metadata){"sdy41", 3, nztop, nxtop, 4, NULL, NULL, sdy41D};
    sdy42D_meta = (struct GPUArray_Metadata){"sdy42", 3, nzbtm, nxbtm, 4, NULL, NULL, sdy42D};
    sdx51D_meta = (struct GPUArray_Metadata){"sdx51", 3, nztop, nytop, 5, NULL, NULL, sdx51D};
    sdx52D_meta = (struct GPUArray_Metadata){"sdx52", 3, nzbtm, nybtm, 5, NULL, NULL, sdx52D};
    sdy51D_meta = (struct GPUArray_Metadata){"sdy51", 3, nztop, nxtop, 5, NULL, NULL, sdy51D};
    sdy52D_meta = (struct GPUArray_Metadata){"sdy52", 3, nzbtm, nxbtm, 5, NULL, NULL, sdy52D};

    rcx41D_meta = (struct GPUArray_Metadata){"rcx41", 3, nztop, nytop, 4, NULL, NULL, rcx41D};
    rcx42D_meta = (struct GPUArray_Metadata){"rcx42", 3, nzbtm, nybtm, 4, NULL, NULL, rcx42D};
    rcy41D_meta = (struct GPUArray_Metadata){"rcy41", 3, nztop, nxtop, 4, NULL, NULL, rcy41D};
    rcy42D_meta = (struct GPUArray_Metadata){"rcy42", 3, nzbtm, nxbtm, 4, NULL, NULL, rcy42D};
    rcx51D_meta = (struct GPUArray_Metadata){"rcx51", 3, nztop, nytop, 5, NULL, NULL, rcx51D};
    rcx52D_meta = (struct GPUArray_Metadata){"rcx52", 3, nzbtm, nybtm, 5, NULL, NULL, rcx52D};
    rcy51D_meta = (struct GPUArray_Metadata){"rcy51", 3, nztop, nxtop, 5, NULL, NULL, rcy51D};
    rcy52D_meta = (struct GPUArray_Metadata){"rcy52", 3, nzbtm, nxbtm, 5, NULL, NULL, rcy52D};
}
#ifdef __cplusplus
}
#endif

#define CHECK_ERROR(err, str) \
	if (err != cudaSuccess) \
	{\
		printf("Error in \"%s\", %s\n", str, cudaGetErrorString(err)); \
	}

//debug----------------------
double totalTimeH2DV, totalTimeD2HV;
double totalTimeH2DS, totalTimeD2HS;
double totalTimeCompV, totalTimeCompS;
double tmpTime;
struct timeval t1, t2;
int procID;


/* Logging Framwork stores logs of a node in a node-{rank}.log file
This includes functions to log whole arrays, strings, timing information
in a same node-specific log file
*/
class LoggingFramework {
    FILE* log; // the logFile
    std::string logname; // logFileName, determined in ctors
    
    /* 
       Downloads an array 'buf' from GPU and
       writes it to the 'log' file with bufname

       NOTE that the memory alignment of the array is coloumn-major (fortran)
       Arrays on GPU retain their Fortran memory alignment as they were 
       created in Fortran subroutines
    */
    void logGPUMem_file(FILE* log, double* buf, int len, char* bufname) {        
        cudaError_t cudaRes;
        double* temp = (double*) malloc (sizeof(double)* len);
        //printf("bufname: %s buf: %d, len: %d \n", bufname, buf, len);
        cudaRes = cudaMemcpy (temp, buf, sizeof(double)*len, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
        CHECK_ERROR (cudaRes, "Error in Temp copying for logging: ");
        CHECK_ERROR (cudaRes, bufname);
        for (long i=0; i<len; i++) {
            if(temp[i]!=0)
                fprintf( log, "%d:%f, ", i, temp[i]);
        }
        fprintf(log, "\n");
        free(temp);
    }


    public:
    LoggingFramework() {
        
    }
    
    /*
        initilaises the LoggingFramework and
        creates a new "node-{rank}.log" file
    */
    LoggingFramework (int rank) {
       char rank_str[2];
       sprintf(rank_str, "%d", rank);
       std::string s1("node-"), s2(rank_str), s3(".log");
       logname = (s1+ s2 + s3); 
       log = fopen(logname.c_str(), "w"); // creates a new or overwrites existing
       fclose(log);
    }

    /*
        initilaises the LoggingFramework and
        creates a new "logID-{rank}.log" file
    */
    LoggingFramework (int rank, char* logID) {
       char rank_str[2];
       sprintf(rank_str, "%d", rank);
       std::string s1(logID), s2(rank_str), s3(".log");
       logname = (s1+ s2 + s3); 
       printf("logfilename: %s\n", logname.c_str());
       log = fopen(logname.c_str(), "w");
       fclose(log);
    }

    /* 
        Calls the logGPUMem_file function with appropriate params
        which downloads 1D array from GPU and writes to log
    */
    void logGPUMem_1D(double* buf, int i, char* bufname) {
        log = fopen(logname.c_str(), "a");
        fprintf(log, "%s(len=%d): ", bufname, i);
        logGPUMem_file(log, buf, i,bufname);
        fclose(log);
    }

     /* 
        Calls the logGPUMem_file function with appropriate params
        which downloads 2D array from GPU and writes to log
    */
    void logGPUMem_2D(double* buf, int i, int j, char* bufname) {
        log = fopen(logname.c_str(), "a");
        fprintf(log, "Fortran shape %s(%d,%d): ", bufname, i, j); 
        logGPUMem_file(log, buf, i*j,bufname);
        fclose(log);
    }

    /* 
        Calls the logGPUMem_file function with appropriate params
        which downloads 3D array from GPU and writes to log
    */
    void logGPUMem_3D(double* buf, int i, int j, int k, char* bufname) {
        log = fopen(logname.c_str(), "a");
        fprintf(log, "Fortran shape %s(%d,%d,%d): ", bufname, i, j ,k);
        logGPUMem_file(log, buf, i*j*k,bufname);
        fclose(log);
    }

    /*
        Writes a string to log
    */
    void log_string(char* tag) {
        log = fopen(logname.c_str(), "a");
        fprintf(log, "%s\n", tag);
        printf("%s\n", tag);
        fclose(log);    
    }

    /*
        Writes timing information to log
    */
    void log_timing(char* tag, double time) {
        //log = fopen(logname.c_str(), "a");
        //fprintf(log, "TIME %s:%lf\n", tag, time);
        printf("TIME %s: %lf\n", tag, time);
        //fclose(log);    
    }
    void log_timing1(char* tag, double time) {
        //log = fopen(logname.c_str(), "a");
        //fprintf(log, "TIME %s:%lf\n", tag, time);
        printf("TIME %s: %lf\n", tag, time);
        //fclose(log);    
    }

    /*
        Writes MPI communication information to log
    */
    void log_comm(char* tag, int dest, int size) {
        //log = fopen(logname.c_str(), "a");
        //fprintf(log, "COMM: %s of size %d(B) to/from %d\n", tag, size*sizeof(double), dest);
        printf("COMM: %s of size %d(B) sent to/from %d\n", tag, size*sizeof(double), dest);
        //fclose(log);    
    }


    /*
        Calls appropriate logger function
        depending upon the Array metadata
    */
    void logGPUArrInfo (struct GPUArray_Metadata gpu_arr_info) {
        switch(gpu_arr_info.dimensions) {
            case 1: 
                logGPUMem_1D(gpu_arr_info.dptr_1D, gpu_arr_info.i, gpu_arr_info.arr_name);
                break;
            case 2:
                logGPUMem_2D(gpu_arr_info.dptr_2D, gpu_arr_info.i, gpu_arr_info.j, gpu_arr_info.arr_name);
                break;
            case 3:
                logGPUMem_3D(gpu_arr_info.dptr_3D, gpu_arr_info.i, gpu_arr_info.j, gpu_arr_info.k, gpu_arr_info.arr_name);
                break;
        }
    }
} logger;

//--------------------------------


#define MaxThreadsPerBlock 1024
#define NTHREADS 8

__global__ void velocity_inner_IC(int	nztop,
					  int 	nztm1,
					  double ca,
					  int	*nd1_vel,
					  double *rhoM,
					  int   *idmat1M,
					  double *dxi1M,
					  double *dyi1M,
					  double *dzi1M,
					  double *dxh1M,
					  double *dyh1M,
					  double *dzh1M,
					  double *t1xxM,
					  double *t1xyM,
					  double *t1xzM,
					  double *t1yyM,
					  double *t1yzM,
					  double *t1zzM,
					  int	nxtop,		//dimension #
					  int	nytop,
					  double *v1xM,		//output
					  double *v1yM,
					  double *v1zM);

__global__ void velocity_inner_IIC(double ca,
					   int	 *nd2_vel,
					   double *rhoM,
					   double *dxi2,
					   double *dyi2,
					   double *dzi2,
					   double *dxh2,
					   double *dyh2,
					   double *dzh2,
					   int 	 *idmat2,
					   double *t2xx,
					   double *t2xy,
					   double *t2xz,
					   double *t2yy,
					   double *t2yz,
					   double *t2zz,
					   int   nxbtm,	//dimension #s
					   int   nybtm,
					   int   nzbtm,
					   double *v2x,		//output
					   double *v2y,
					   double *v2z);

__global__ void vel_PmlX_IC(double ca,
				int   lbx0,
				int	  lbx1,
				int	  *nd1_vel,
				double *rhoM,
				double *drvh1,
				double *drti1,
				double *damp1_x,
				int	  *idmat1,
				double *dxi1,
				double *dyi1,
				double *dzi1,
				double *dxh1,
				double *dyh1,
				double *dzh1,
				double *t1xx,
				double *t1xy,
				double *t1xz,
				double *t1yy,
				double *t1yz,
				double *t1zz,
				int   mw1_pml1,	//dimension #
			    int   mw1_pml,
			    int   nxtop,
			    int   nytop,
			    int   nztop,
				double *v1x,		//output
				double *v1y,
				double *v1z,
				double *v1x_px,
				double *v1y_px,
				double *v1z_px);

__global__ void vel_PmlY_IC(int  nztop,
				double ca,
				int	  lby0,
				int   lby1,
				int   *nd1_vel,
				double *rhoM,
				double *drvh1,
				double *drti1,
				int   *idmat1,
				double *damp1_y,
				double *dxi1,
				double *dyi1,
				double *dzi1,
				double *dxh1,
				double *dyh1,
				double *dzh1,
				double *t1xx,
				double *t1xy,
				double *t1xz,
				double *t1yy,
				double *t1yz,
				double *t1zz,
				int   mw1_pml1, //dimension #s
				int   mw1_pml,
				int   nxtop,
				int   nytop,
				double *v1x,		//output
				double *v1y,
				double *v1z,
				double *v1x_py,
				double *v1y_py,
				double *v1z_py);

__global__ void vel_PmlX_IIC(int   nzbm1,
				 double ca,
				 int   lbx0,
				 int   lbx1,
				 int   *nd2_vel,
				 double *drvh2,
				 double *drti2,
				 double *rhoM,
				 double *damp2_x,
				 int   *idmat2,
				 double *dxi2,
				 double *dyi2,
				 double *dzi2,
				 double *dxh2,
				 double *dyh2,
				 double *dzh2,
				 double *t2xx,
				 double *t2xy,
				 double *t2xz,
				 double *t2yy,
				 double *t2yz,
				 double *t2zz,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2x,	//output
				 double *v2y,
				 double *v2z,
				 double *v2x_px,
				 double *v2y_px,
				 double *v2z_px);

__global__ void vel_PmlY_IIC(int   nzbm1,
				 double ca,
				 int   lby0,
				 int   lby1,
				 int   *nd2_vel,
				 double *drvh2,
				 double *drti2,
				 double *rhoM,
				 double *damp2_y,
				 int   *idmat2,
				 double *dxi2,
				 double *dyi2,
				 double *dzi2,
				 double *dxh2,
				 double *dyh2,
				 double *dzh2,
				 double *t2xx,
				 double *t2xy,
				 double *t2xz,
				 double *t2yy,
				 double *t2yz,
				 double *t2zz,
				 int   mw2_pml1,
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2x,		//output
				 double *v2y,
				 double *v2z,
				 double *v2x_py,
				 double *v2y_py,
				 double *v2z_py);

__global__ void vel_PmlZ_IIC(int   nzbm1,
				 double ca,
				 int   *nd2_vel,
				 double *drvh2,
				 double *drti2,
				 double *rhoM,
				 double *damp2_z,
				 int   *idmat2,
				 double *dxi2,
				 double *dyi2,
				 double *dzi2,
				 double *dxh2,
				 double *dyh2,
				 double *dzh2,
				 double *t2xx,
				 double *t2xy,
				 double *t2xz,
				 double *t2yy,
				 double *t2yz,
				 double *t2zz,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2x,		//output
				 double *v2y,
				 double *v2z,
				 double *v2x_pz,
				 double *v2y_pz,
				 double *v2z_pz);

__global__ void stress_norm_xy_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   int *nd1_tyy,
					   int *idmat1M,
					   double ca,
					   double *clamdaM,
					   double *cmuM,
					   double *epdtM,
					   double *qwpM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dxh1M,
					   double *dyh1M,
					   double *dxi1M,
					   double *dyi1M,
					   double *dzi1M,
					   double *t1xxM,
					   double *t1xyM,
					   double *t1yyM,
					   double *t1zzM,
					   double *qt1xxM,
					   double *qt1xyM,
					   double *qt1yyM,
					   double *qt1zzM,
					   double *v1xM,
					   double *v1yM,
					   double *v1zM);
__global__ void  stress_xz_yz_IC(int nxb1,
					  int nyb1,
					  int nxtop,
					  int nytop,
					  int nztop,
					  int *nd1_tyz,
					  int *idmat1M,
					  double ca,
					  double *cmuM,
					  double *epdtM,
					  double *qwsM,
					  double *qwt1M,
					  double *qwt2M,
					  double *dxi1M,
					  double *dyi1M,
					  double *dzh1M,
					  double *v1xM,
					  double *v1yM,
					  double *v1zM,
					  double *t1xzM,
					  double *t1yzM,
					  double *qt1xzM,
					  double *qt1yzM);

__global__ void stress_resetVars(int ny1p1,
					  int nx1p1,
					  int nxtop,
					  int nytop,
					  int nztop,
					  double *t1xzM,
					  double *t1yzM);

__global__ void stress_norm_PmlX_IC(int nxb1,
						 int nyb1,
						 int nxtop,
						 int nytop,
						 int nztop,
						 int mw1_pml,
						 int mw1_pml1,
						 int lbx0,
						 int lbx1,
						 int *nd1_tyy,
						 int *idmat1M,
						 double ca,
						 double *drti1M,
						 double *damp1_xM,
						 double *clamdaM,
						 double *cmuM,
						 double *epdtM,
						 double *qwpM,
						 double *qwsM,
						 double *qwt1M,
						 double *qwt2M,
						 double *dzi1M,
						 double *dxh1M,
						 double *dyh1M,
						 double *v1xM,
						 double *v1yM,
						 double *v1zM,
						 double *t1xxM,
						 double *t1yyM,
						 double *t1zzM,
						 double *t1xx_pxM,
						 double *t1yy_pxM,
						 double *qt1xxM,
						 double *qt1yyM,
						 double *qt1zzM,
						 double *qt1xx_pxM,
						 double *qt1yy_pxM);

__global__ void stress_norm_PmlY_IC(int nxb1,
						 int nyb1,
						 int mw1_pml1,
						 int nxtop,
						 int nztop,
						 int lby0,
						 int lby1,
						 int *nd1_tyy,
						 int *idmat1M,
						 double ca,
						 double *drti1M,
						 double *damp1_yM,
						 double *clamdaM,
						 double *cmuM,
						 double *epdtM,
						 double *qwpM,
						 double *qwsM,
						 double *qwt1M,
						 double *qwt2M,
						 double *dxh1M,
						 double *dyh1M,
						 double *dzi1M,
						 double *t1xxM,
						 double *t1yyM,
						 double *t1zzM,
						 double *qt1xxM,
						 double *qt1yyM,
						 double *qt1zzM,
						 double *t1xx_pyM,
						 double *t1yy_pyM,
						 double *qt1xx_pyM,
						 double *qt1yy_pyM,
						 double *v1xM,
						 double *v1yM,
						 double *v1zM);

__global__ void stress_xy_PmlX_IC(int nxb1,
					   int nyb1,
					   int mw1_pml,
					   int mw1_pml1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int lbx0,
					   int lbx1,
					   int *nd1_txy,
					   int *idmat1M,
					   double ca,
					   double *drth1M,
					   double *damp1_xM,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dxi1M,
					   double *dyi1M,
					   double *t1xyM,
					   double *qt1xyM,
					   double *t1xy_pxM,
					   double *qt1xy_pxM,
					   double *v1xM,
					   double *v1yM);

__global__ void stress_xy_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd1_txy,
					   int *idmat1M,
					   double ca,
					   double *drth1M,
					   double *damp1_yM,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dxi1M,
					   double *dyi1M,
					   double *t1xyM,
					   double *qt1xyM,
					   double *t1xy_pyM,
					   double *qt1xy_pyM,
					   double *v1xM,
					   double *v1yM);

__global__ void stress_xz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int mw1_pml,
					   int mw1_pml1,
					   int lbx0,
					   int lbx1,
					   int *nd1_txz,
					   int *idmat1M,
					   double ca,
					   double *drth1M,
					   double *damp1_xM,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dxi1M,
					   double *dzh1M,
					   double *t1xzM,
					   double *qt1xzM,
					   double *t1xz_pxM,
					   double *qt1xz_pxM,
					   double *v1xM,
					   double *v1zM);

__global__ void stress_xz_PmlY_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd1_txz,
					   int *idmat1M,
					   double ca,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dxi1M,
					   double *dzh1M,
					   double *t1xzM,
					   double *qt1xzM,
					   double *v1xM,
					   double *v1zM);

__global__ void stress_yz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nztop,
					   int nxtop,
					   int lbx0,
					   int lbx1,
					   int *nd1_tyz,
					   int *idmat1M,
					   double ca,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dyi1M,
					   double *dzh1M,
					   double *t1yzM,
					   double *qt1yzM,
					   double *v1yM,
					   double *v1zM);

__global__ void stress_yz_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd1_tyz,
					   int *idmat1M,
					   double ca,
					   double *drth1M,
					   double *damp1_yM,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dyi1M,
					   double *dzh1M,
					   double *t1yzM,
					   double *qt1yzM,
					   double *t1yz_pyM,
					   double *qt1yz_pyM,
					   double *v1yM,
					   double *v1zM);

__global__ void stress_norm_xy_II(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int *nd2_tyy,
					   int *idmat2M,
					   double *clamdaM,
					   double *cmuM,
					   double *epdtM,
					   double *qwpM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *t2xxM,
					   double *t2xyM,
					   double *t2yyM,
					   double *t2zzM,
					   double *qt2xxM,
					   double *qt2xyM,
					   double *qt2yyM,
					   double *qt2zzM,
					   double *dxh2M,
					   double *dyh2M,
					   double *dxi2M,
					   double *dyi2M,
					   double *dzi2M,
					   double *v2xM,
					   double *v2yM,
					   double *v2zM);

//call stress_xz_yz_II
__global__ void stress_xz_yz_IIC(int nxb2,
					  int nyb2,
					  int nztop,
					  int nxbtm,
					  int nzbtm,
					  int *nd2_tyz,
					  int *idmat2M,
					  double *cmuM,
					  double *epdtM,
					  double *qwsM,
					  double *qwt1M,
					  double *qwt2M,
					  double *dxi2M,
					  double *dyi2M,
					  double *dzh2M,
					  double *t2xzM,
					  double *t2yzM,
					  double *qt2xzM,
					  double *qt2yzM,
					  double *v2xM,
					  double *v2yM,
					  double *v2zM);

//call stress_norm_PmlX_II
__global__ void stress_norm_PmlX_IIC(int nxb2,
						  int nyb2,
						  int mw2_pml,
						  int mw2_pml1,
						  int nztop,
						  int nxbtm,
						  int nybtm,
						  int nzbtm,
						  int lbx0,
						  int lbx1,
						  int *nd2_tyy,
						  int *idmat2M,
						  double ca,
						  double *drti2M,
						  double *damp2_xM,
						  double *clamdaM,
						  double *cmuM,
						  double *epdtM,
						  double *qwpM,
						  double *qwsM,
						  double *qwt1M,
						  double *qwt2M,
						  double *dxh2M,
						  double *dyh2M,
						  double *dzi2M,
						  double *t2xxM,
						  double *t2yyM,
						  double *t2zzM,
						  double *qt2xxM,
						  double *qt2yyM,
						  double *qt2zzM,
						  double *t2xx_pxM,
						  double *t2yy_pxM,
						  double *qt2xx_pxM,
						  double *qt2yy_pxM,
						  double *v2xM,
						  double *v2yM,
						  double *v2zM);

__global__ void stress_norm_PmlY_II(int nxb2,
						 int nyb2,
						 int nztop,
						 int nxbtm,
						 int nzbtm,
						 int mw2_pml1,
						 int lby0,
						 int lby1,
						 int *nd2_tyy,
						 int *idmat2M,
						 double ca,
						 double *drti2M,
						 double *damp2_yM,
						 double *clamdaM,
						 double *cmuM,
						 double *epdtM,
						 double *qwpM,
						 double *qwsM,
						 double *qwt1M,
						 double *qwt2M,
						 double *dxh2M,
						 double *dyh2M,
						 double *dzi2M,
						 double *t2xxM,
						 double *t2yyM,
						 double *t2zzM,
						 double *qt2xxM,
						 double *qt2yyM,
						 double *qt2zzM,
						 double *t2xx_pyM,
						 double *t2yy_pyM,
						 double *qt2xx_pyM,
						 double *qt2yy_pyM,
						 double *v2xM,
						 double *v2yM,
						 double *v2zM);

__global__ void stress_norm_PmlZ_IIC(int nxb2,
						  int nyb2,
						  int mw2_pml,
						  int mw2_pml1,
						  int nztop,
						  int nxbtm,
						  int nzbtm,
						  int *nd2_tyy,
						  int *idmat2M,
						  double ca,
						  double *damp2_zM,
						  double *drth2M,
						  double *clamdaM,
						  double *cmuM,
						  double *epdtM,
						  double *qwpM,
						  double *qwsM,
						  double *qwt1M,
						  double *qwt2M,
						  double *dxh2M,
						  double *dyh2M,
						  double *dzi2M,
						  double *t2xxM,
						  double *t2yyM,
						  double *t2zzM,
						  double *qt2xxM,
						  double *qt2yyM,
						  double *qt2zzM,
						  double *t2xx_pzM,
						  double *t2zz_pzM,
						  double *qt2xx_pzM,
						  double *qt2zz_pzM,
						  double *v2xM,
						  double *v2yM,
						  double *v2zM);

__global__ void stress_xy_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int lbx0,
						int lbx1,
						int *nd2_txy,
						int *idmat2M,
						double ca,
						double *drth2M,
						double *damp2_xM,
						double *cmuM,
						double *epdtM,
						double *qwsM,
						double *qwt1M,
						double *qwt2M,
						double *dxi2M,
						double *dyi2M,
						double *t2xyM,
						double *qt2xyM,
						double *t2xy_pxM,
						double *qt2xy_pxM,
						double *v2xM,
						double *v2yM);

__global__ void stress_xy_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nztop,
						int nxbtm,
						int nzbtm,
						int lby0,
						int lby1,
						int *nd2_txy,
						int *idmat2M,
						double ca,
						double *drth2M,
						double *damp2_yM,
						double *cmuM,
						double *epdtM,
						double *qwsM,
						double *qwt1M,
						double *qwt2M,
						double *dxi2M,
						double *dyi2M,
						double *t2xyM,
						double *qt2xyM,
						double *t2xy_pyM,
						double *qt2xy_pyM,
						double *v2xM,
						double *v2yM);

__global__ void stress_xy_PmlZ_II(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int *nd2_txy,
					   int *idmat2M,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dxi2M,
					   double *dyi2M,
					   double *t2xyM,
					   double *qt2xyM,
					   double *v2xM,
					   double *v2yM);

__global__ void stress_xz_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int lbx0,
						int lbx1,
						int *nd2_txz,
						int *idmat2M,
						double ca,
						double *drth2M,
						double *damp2_xM,
						double *cmuM,
						double *epdtM,
						double *qwsM,
						double *qwt1M,
						double *qwt2M,
						double *dxi2M,
						double *dzh2M,
						double *t2xzM,
						double *qt2xzM,
						double *t2xz_pxM,
						double *qt2xz_pxM,
						double *v2xM,
						double *v2zM);

__global__ void stress_xz_PmlY_IIC(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd2_txz,
					   int *idmat2M,
					   double *cmuM,
					   double *epdtM,
					   double *qwsM,
					   double *qwt1M,
					   double *qwt2M,
					   double *dxi2M,
					   double *dzh2M,
					   double *v2xM,
					   double *v2zM,
					   double *t2xzM,
					   double *qt2xzM);

__global__ void stress_xz_PmlZ_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *nd2_txz,
						int *idmat2M,
						double ca,
						double *drti2M,
						double *damp2_zM,
						double *cmuM,
						double *epdtM,
						double *qwsM,
						double *qwt1M,
						double *qwt2M,
						double *dxi2M,
						double *dzh2M,
						double *t2xzM,
						double *qt2xzM,
						double *t2xz_pzM,
						double *qt2xz_pzM,
						double *v2xM,
						double *v2zM);

//call stress_yz_PmlX_II
__global__ void stress_yz_PmlX_IIC(int nxb2,
						int nyb2,
						int nxbtm,
						int nzbtm,
						int nztop,
						int lbx0,
						int lbx1,
						int *nd2_tyz,
						int *idmat2M,
						double *cmuM,
						double *epdtM,
						double *qwsM,
						double *qwt1M,
						double *qwt2M,
						double *dyi2M,
						double *dzh2M,
						double *t2yzM,
						double *qt2yzM,
						double *v2yM,
						double *v2zM);

__global__ void stress_yz_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int lby0,
						int lby1,
						int *nd2_tyz,
						int *idmat2M,
						double ca,
						double *drth2M,
						double *damp2_yM,
						double *cmuM,
						double *epdtM,
						double *qwsM,
						double *qwt1M,
						double *qwt2M,
						double *dyi2M,
						double *dzh2M,
						double *t2yzM,
						double *qt2yzM,
						double *t2yz_pyM,
						double *qt2yz_pyM,
						double *v2yM,
						double *v2zM);

__global__ void stress_yz_PmlZ_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *nd2_tyz,
						int *idmat2M,
						double ca,
						double *drti2M,
						double *damp2_zM,
						double *cmuM,
						double *epdtM,
						double *qwsM,
						double *qwt1M,
						double *qwt2M,
						double *dyi2M,
						double *dzh2M,
						double *t2yzM,
						double *qt2yzM,
						double *t2yz_pzM,
						double *qt2yz_pzM,
						double *v2yM,
						double *v2zM);


__global__ void vel1_dummy (int nxtop, int nztop, int nytop,  double* v1x, double* v1y, double* v1z);
__global__ void vel2_dummy (int nxbtm, int nzbtm, int nybtm, double* v2x, double* v2y, double* v2z);

__global__ void vel_sdx51 (double* sdx51M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop) ;
__global__ void vel_sdx52 (double* sdx52M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void vel_sdx41 (double* sdx41M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop, int nxtm1) ;
__global__ void vel_sdx42 (double* sdx42M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm, int nxbm1) ;
__global__ void vel_sdy51 (double* sdy51M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop) ;
__global__ void vel_sdy52 (double* sdy52M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void vel_sdy41 (double* sdy41M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop, int nytm1) ;
__global__ void vel_sdy42 (double* sdy42M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm, int nybm1) ;
__global__ void vel_rcx51 (double* rcx51M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop, int nx1p1, int nx1p2) ;
__global__ void vel_rcx52 (double* rcx52M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm, int nx2p1, int nx2p2) ;
__global__ void vel_rcx41 (double* rcx41M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop) ;
__global__ void vel_rcx42 (double* rcx42M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void vel_rcy51 (double* rcy51M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop, int ny1p1, int ny1p2) ;
__global__ void vel_rcy52 (double* rcy52M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm, int ny2p1, int ny2p2) ;
__global__ void vel_rcy41 (double* rcy41M, double* v1xM, double* v1yM, double* v1zM, int nytop, int nztop, int nxtop) ;
__global__ void vel_rcy42 (double* rcy42M, double* v2xM, double* v2yM, double* v2zM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void vel_sdx1(double* sdx1D, double* v2xM, double* v2yM, double* v2zM, int nxbtm, int nzbtm, int ny2p1, int ny2p2) ;
__global__ void vel_sdy1(double* sdy1D, double* v2xM, double* v2yM, double* v2zM, int nxbtm, int nzbtm, int nx2p1, int nx2p2) ;
__global__ void vel_sdx2(double* sdx2D, double* v2xM, double* v2yM, double* v2zM, int nxbm1, int nxbtm, int nzbtm, int ny2p1, int ny2p2) ;
__global__ void vel_sdy2(double* sdy2D, double* v2xM, double* v2yM, double* v2zM, int nybm1, int nybtm,int nxbtm, int nzbtm,  int nx2p1, int nx2p2) ;
__global__ void vel_rcx1(double* rcx1D, double* v2xM, double* v2yM, double* v2zM, int nxbtm, int nzbtm, int ny2p1, int ny2p2) ;
__global__ void vel_rcy1(double* rcy1D, double* v2xM, double* v2yM, double* v2zM, int nxbtm, int nzbtm, int nx2p1, int nx2p2) ;
__global__ void vel_rcx2(double* rcx2D, double* v2xM, double* v2yM, double* v2zM, int nxbtm, int nzbtm,  int nx2p1, int nx2p2, int ny2p1, int ny2p2) ;
__global__ void vel_rcy2(double* rcy2D, double* v2xM, double* v2yM, double* v2zM, int nxbtm, int nzbtm, int nx2p1, int nx2p2, int ny2p1, int ny2p2) ;
__global__ void vel1_interpl_3vbtm( int ny1p2, int ny2p2, int nz1p1, int nyvx, 
                int nxbm1, int nxbtm, int nzbtm, int nxtop, int nztop, int neighb2,
                double* chxM, double* v1xM, double* v2xM, double* rcx2D);
__global__ void vel3_interpl_3vbtm( int ny1p2, int nz1p1, int nyvx1, 
                int nxbm1, int nxbtm, int nzbtm, int nxtop, int nztop, 
                double* ciyM, double* v1xM);
__global__ void vel4_interpl_3vbtm( int nx1p2, int ny2p2, int nz1p1, int nxvy, 
                int nybm1, int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop, 
                double* chyM, double* v1yM, double * v2yM);
__global__ void vel5_interpl_3vbtm( int nx1p2, int nx2p2, int nz1p1, int nxvy, 
                int nybm1, int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop, 
                double* chyM, double* v1yM, double* v2yM, double* rcy2D);
__global__ void vel6_interpl_3vbtm( int nx1p2, int nz1p1, int nxvy1, 
                int nybm1, int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop, 
                double* cixM, double* v1yM);
__global__ void vel7_interpl_3vbtm( int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop, 
                double* ciyM, double* sdx1D,  double* rcx1D);
__global__ void vel8_interpl_3vbtm( int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop, 
                double* ciyM, double* sdx2D,  double* rcx2D);
__global__ void vel9_interpl_3vbtm( int nz1p1, int nx1p2, int ny2p1, int nxvy, 
                int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop, 
                int neighb4, double* ciyM, double* rcy1D, double* rcy2D, double* v1zM, double* v2zM);
__global__ void vel11_interpl_3vbtm( int nx1p2, int nx2p1, int ny1p1, int nz1p1, int nxvy1,
		int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop, 
                double* cixM, double* sdx1D, double* sdx2D,  double* v1zM);
__global__ void vel13_interpl_3vbtm(double* v1xM, double* v2xM, 
				    int nxtop, int nytop, int nztop,
				    int nxbtm, int nybtm, int nzbtm);
__global__ void vel14_interpl_3vbtm(double* v1yM, double* v2yM, 
				    int nxbtm, int nybtm, int nzbtm,
				    int nxtop, int nytop, int nztop);
__global__ void vel15_interpl_3vbtm(double* v1zM, double* v2zM, 
				    int nxbtm, int nybtm, int nzbtm,
				    int nxtop, int nytop, int nztop);


//__global__ void vel2_interpl_3vbtm(int nx1p2, int nx2p2, int nx2p1, int ny1p1,  int ny1p2, int ny2p1, int ny2p2, 
//               int nz1p1, int nxvy, int nxvy1, int nyvx, int nyvx1, int nxtop, int nytop, int nztop,
//               int nxbm1, int nxbtm, int nybtm, int nybm1, int nzbtm, int neighb1, int neighb2, 
//                int neighb3, int neighb4,
//                double* cixM, double* ciyM, double* chxM, double* chyM,
//                double* v1xM, double* v1yM, double* v1zM,
//                double* v2xM, double* v2yM, double* v2zM,
//               double* sdx1, double* sdx2, double* sdx1D, double* sdx2D,
//                double* rcx1D, double* rcx2D, double* rcy1D, double* rcy2D);

__global__ void vel1_vxy_image_layer (double* v1xM, double* v1zM, int* nd1_velD,
				double* dxi1M,  double* dzh1M,
				int i, double dzdx,
				int nxbtm, int nybtm, int nzbtm, 
				int nxtop, int nytop, int nztop) ;
__global__ void vel2_vxy_image_layer (double* v1xM, double* v1zM, int* nd1_velD,
				double* dxi1M,  double* dzh1M,
				int iix, double dzdt,
				int nxbtm, int nybtm, int nzbtm, 
				int nxtop, int nytop, int nztop); 
__global__ void vel3_vxy_image_layer (double* v1yM, double* v1zM, int* nd1_velD,
				double* dyi1M,  double* dzh1M,
				int j, double dzdy,
				int nxbtm, int nybtm, int nzbtm, 
				int nxtop, int nytop, int nztop); 
__global__ void vel4_vxy_image_layer (double* v1yM, double* v1zM, int* nd1_velD,
				double* dyi1M,  double* dzh1M,
				int jjy, double dzdt,
				int nxbtm, int nybtm, int nzbtm, 
				int nxtop, int nytop, int nztop); 

//__global__ void vel_vxy_image_layer1 ( double* v1xM, double* v1yM, double* v1zM, int* nd1_velD, 
//                                   double* dxi1M, double* dyi1M, double* dzh1M,
//                                    int nxtm1, int nytm1, int nxtop, int nytop, int nztop,
//                                    int neighb1, int neighb2, int neighb3, int neighb4);
__global__ void vel_vxy_image_layer_sdx( double* sdx1D, double* sdx2D, double* v1xM, double* v1yM, int nxtop, int nytop, int  nztop);
__global__ void vel_vxy_image_layer_sdy( double* sdy1D, double* sdy2D, double* v1xM, double* v1yM, int nxtop, int nytop, int nztop);
__global__ void vel_vxy_image_layer_rcx( double* rcx1D, double* rcx2D, double* v1xM, double* v1yM, int nxtop, int nytop, int nztop, int nx1p1);
__global__ void vel_vxy_image_layer_rcy( double* , double* , double* v1xM, double* v1yM, int nxtop, int nytop, int nztop, int ny1p1);


__global__ void vel_add_dcs (double* t1xxM, double* t1xyM, double* t1xzM, double* t1yyM, 
            double* t1yzM, double* t1zzM, double* t2xxM, double* t2yyM, double* t2xyM,
            double* t2xzM, double* t2yzM, double* t2zzM,
            int nfadd, int* index_xyz_sourceD, int ixsX, int ixsY, int ixsZ, 
            double* fampD, int fampX, int fampY,
            double* ruptmD, int ruptmX, double* risetD, int risetX, 
            double* sparam2D, int sparam2X,
            double* sutmArrD, int nzrg11, int nzrg12, int nzrg13, int nzrg14, 
            int nxtop, int nytop, int nztop,
            int nxbtm, int nybtm, int nzbtm);

__global__ void stress_sdx41 (double* sdx41M, double* t1xxM, double* t1xyM, double* t1xzM, int nytop, int nztop, int nxtop) ;
__global__ void stress_sdx42 (double* sdx42M, double* t2xxM, double* t2xyM, double* t2xzM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void stress_sdx51 (double* sdx51M, double* t1xxM, double* t1xyM, double* t1xzM, int nytop, int nztop, int nxtop, int nxtm1) ;
__global__ void stress_sdx52 (double* sdx42M, double* t2xxM, double* t2xyM, double* t2xzM, int nybtm, int nzbtm, int nxbtm, int nxbm1) ;
__global__ void stress_sdy41 (double* sdy41M, double* t1yyM, double* t1xyM, double* t1yzM, int nytop, int nztop, int nxtop) ;
__global__ void stress_sdy42 (double* sdy42M, double* t2yyM, double* t2xyM, double* t2yzM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void stress_sdy51 (double* sdy51M, double* t1yyM, double* t1xyM, double* t1yzM, int nytop, int nztop, int nxtop, int nytm1) ;
__global__ void stress_sdy52 (double* sdy52M, double* t2yyM, double* t2xyM, double* t2yzM, int nybtm, int nzbtm, int nxbtm, int nybm1) ;
__global__ void stress_rcx41 (double* rcx41M, double* t1xxM, double* t1xyM, double* t1xzM, int nytop, int nztop, int nxtop, int nx1p1, int nx1p2) ;
__global__ void stress_rcx42 (double* rcx42M, double* t2xxM, double* t2xyM, double* t2xzM, int nybtm, int nzbtm, int nxbtm, int nx2p1, int nx2p2) ;
__global__ void stress_rcx51 (double* rcx51M, double* t1xxM, double* t1xyM, double* t1xzM, int nytop, int nztop, int nxtop) ;
__global__ void stress_rcx52 (double* rcx42M, double* t2xxM, double* t2xyM, double* t2xzM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void stress_rcy41 (double* rcy41M, double* t1yyM, double* t1xyM, double* t1yzM, int nytop, int nztop, int nxtop, int ny1p1, int ny1p2) ;
__global__ void stress_rcy42 (double* rcy42M, double* t2yyM, double* t2xyM, double* t2yzM, int nybtm, int nzbtm, int nxbtm, int ny2p1, int ny2p2) ;
__global__ void stress_rcy51 (double* rcy51M, double* t1yyM, double* t1xyM, double* t1yzM, int nytop, int nztop, int nxtop) ;
__global__ void stress_rcy52 (double* rcy52M, double* t2yyM, double* t2xyM, double* t2yzM, int nybtm, int nzbtm, int nxbtm) ;
__global__ void stress_interp1 (int ntx1, int nz1p1, 
                                int nxbtm, int nybtm, int nzbtm,
                                int  nxtop, int nytop, int nztop,
                                double* t1xzM, double* t2xzM );
__global__ void stress_interp2 (int nty1, int nz1p1, 
                                int nxbtm, int nybtm, int nzbtm,
                                int  nxtop, int nytop, int nztop,
                                double* t1yzM, double* t2yzM );
__global__ void stress_interp3 ( int nxbtm, int nybtm, int nzbtm,
                                int  nxtop, int nytop, int nztop,
                                double* t1zzM, double* t2zzM );

__global__ void stress_interp_stress (double* t1xzM, double* t1yzM, double* t1zzM,
                                double* t2xzM, double* t2yzM, double* t2zzM,
                                int neighb1, int neighb2, int neighb3, int neighb4,
                                int nxbm1, int nybm1, 
                                int nxbtm, int nybtm, int nzbtm,
                                int nxtop, int nytop, int nztop, int nz1p1);



#ifdef __cplusplus
extern "C" {
#endif

extern void compute_velocityCDebug( int* nztop,  int* nztm1,  double* ca, int* lbx,
			 int *lby, int *nd1_vel, double *rhoM, double *drvh1M, double *drti1M,
			 double *damp1_xM, double *damp1_yM, int *idmat1M,double *dxi1M, double *dyi1M,
			 double *dzi1M, double *dxh1M, double *dyh1M, double *dzh1M, double *t1xxM,
			 double *t1xyM, double *t1xzM, double *t1yyM, double *t1yzM, double *t1zzM,
			 void **v1xMp, void **v1yMp, void **v1zMp, double *v1x_pxM, double *v1y_pxM,
			 double *v1z_pxM, double *v1x_pyM, double *v1y_pyM,  double *v1z_pyM,
			  int *nzbm1,  int *nd2_vel, double *drvh2M, double *drti2M,
			 int *idmat2M, double *damp2_xM, double *damp2_yM, double *damp2_zM,
			 double *dxi2M, double *dyi2M, double *dzi2M, double *dxh2M, double *dyh2M,
			 double *dzh2M, double *t2xxM, double *t2xyM, double *t2xzM, double *t2yyM,
			 double *t2yzM, double *t2zzM, void **v2xMp, void **v2yMp, void **v2zMp,
			 double *v2x_pxM, double *v2y_pxM, double *v2z_pxM, double *v2x_pyM,
			 double *v2y_pyM, double *v2z_pyM, double *v2x_pzM, double *v2y_pzM,
			 double *v2z_pzM,  int *nmat, int *mw1_pml1,  int *mw2_pml1,
			  int *nxtop,  int *nytop,  int *mw1_pml,  int *mw2_pml,
			  int *nxbtm,  int *nybtm,  int *nzbtm);

extern void compute_stressCDebug(int *nxb1, int *nyb1, int *nx1p1, int *ny1p1, int *nxtop, int *nytop, int *nztop, int *mw1_pml,
			int *mw1_pml1, int *lbx, int *lby, int *nd1_txy, int *nd1_txz,
			int *nd1_tyy, int *nd1_tyz, int *idmat1M, double *ca, double *drti1M, double *drth1M, double *damp1_xM, double *damp1_yM,
			double *clamdaM, double *cmuM, double *epdtM, double *qwpM, double *qwsM, double *qwt1M, double *qwt2M, double *dxh1M,
			double *dyh1M, double *dzh1M, double *dxi1M, double *dyi1M, double *dzi1M, double *t1xxM, double *t1xyM, double *t1xzM, 
			double *t1yyM, double *t1yzM, double *t1zzM, double *qt1xxM, double *qt1xyM, double *qt1xzM, double *qt1yyM, double *qt1yzM, 
			double *qt1zzM, double *t1xx_pxM, double *t1xy_pxM, double *t1xz_pxM, double *t1yy_pxM, double *qt1xx_pxM, double *qt1xy_pxM,
			double *qt1xz_pxM, double *qt1yy_pxM, double *t1xx_pyM, double *t1xy_pyM, double *t1yy_pyM, double *t1yz_pyM, double *qt1xx_pyM,
			double *qt1xy_pyM, double *qt1yy_pyM, double *qt1yz_pyM, void **v1xMp, void **v1yMp, void **v1zMp,
			int *nxb2, int *nyb2, int *nxbtm, int *nybtm, int *nzbtm, int *mw2_pml, int *mw2_pml1, int *nd2_txy, int *nd2_txz, 
			int *nd2_tyy, int *nd2_tyz, int *idmat2M, 
			double *drti2M, double *drth2M, double *damp2_xM, double *damp2_yM, double *damp2_zM, 
			double *t2xxM, double *t2xyM, double *t2xzM, double *t2yyM, double *t2yzM, double *t2zzM, 
			double *qt2xxM, double *qt2xyM, double *qt2xzM, double *qt2yyM, double *qt2yzM, double *qt2zzM, 
			double *dxh2M, double *dyh2M, double *dzh2M, double *dxi2M, double *dyi2M, double *dzi2M, 
			double *t2xx_pxM, double *t2xy_pxM, double *t2xz_pxM, double *t2yy_pxM, double *t2xx_pyM, double *t2xy_pyM,
			double *t2yy_pyM, double *t2yz_pyM, double *t2xx_pzM, double *t2xz_pzM, double *t2yz_pzM, double *t2zz_pzM,
			double *qt2xx_pxM, double *qt2xy_pxM, double *qt2xz_pxM, double *qt2yy_pxM, double *qt2xx_pyM, double *qt2xy_pyM, 
			double *qt2yy_pyM, double *qt2yz_pyM, double *qt2xx_pzM, double *qt2xz_pzM, double *qt2yz_pzM, double *qt2zz_pzM,
			void **v2xMp, void **v2yMp, void **v2zMp, int *myid);


/* 
    Get GPU Array Metadata from the  array id
    The id's for Arrays is set in disfd_comm.f90
*/
GPUArray_Metadata getGPUArray_Metadata(int id) {
    switch(id) {
       case  10: return sdx1D_meta; 
       case  11: return sdy1D_meta; 
       case  12: return sdx2D_meta; 
       case  13: return sdy2D_meta; 
       case  14: return rcx1D_meta; 
       case  15: return rcy1D_meta; 
       case  16: return rcx2D_meta; 
       case  17: return rcy2D_meta; 
       case  40: return v1xD_meta; 
       case  41: return v1yD_meta; 
       case  42: return v1zD_meta; 
       case  43: return v2xD_meta; 
       case  44: return v2yD_meta; 
       case  45: return v2zD_meta; 
       case  46: return t1xxD_meta; 
       case  47: return t1xyD_meta; 
       case  48: return t1xzD_meta; 
       case  49: return t1yyD_meta; 
       case  50: return t1yzD_meta; 
       case  51: return t1zzD_meta; 
       case  52: return t2xxD_meta; 
       case  53: return t2xyD_meta; 
       case  54: return t2xzD_meta; 
       case  55: return t2yyD_meta; 
       case  56: return t2yzD_meta; 
       case  57: return t2zzD_meta; 
       case  80: return sdx51D_meta; 
       case  81: return sdy51D_meta; 
       case  82: return sdx52D_meta; 
       case  83: return sdy52D_meta; 
       case  84: return sdx41D_meta; 
       case  85: return sdy41D_meta; 
       case  86: return sdx42D_meta; 
       case  87: return sdy42D_meta; 
       case  88: return rcx51D_meta; 
       case  89: return rcy51D_meta; 
       case  90: return rcx52D_meta; 
       case  91: return rcy52D_meta; 
       case  92: return rcx41D_meta; 
       case  93: return rcy41D_meta; 
       case  94: return rcx42D_meta; 
       case  95: return rcy42D_meta; 
       default : printf("Wrong pointer id\n"); exit(0);
    }    
}

/* 
    Get GPU Array pointer from its id
    The id's for Arrays is set in disfd_comm.f90
*/
double* getDevicePointer(int id) {
    GPUArray_Metadata gpumeta= getGPUArray_Metadata(id);
    switch(gpumeta.dimensions) {
        case 1: return gpumeta.dptr_1D; 
        case 2: return gpumeta.dptr_2D; 
        case 3: return gpumeta.dptr_3D; 
        default : printf("Wrong array dimensions\n"); exit(0);
    }

}

/* 
   -------------------------------------------------------------
   Following functions ending with 'C'
   are called from fortran subroutines
*/
void set_deviceC(int *deviceID)
{
	cudaSetDevice(*deviceID);
}

void getsizeof_double(int* size) {
    *size = sizeof(double);    
}

void set_rank_C (int myid) {
    rank = myid;
}

/*
    Followig functions call respective functions 
    in the LoggingFramework class.
    Used for logging/debugging purposes
*/

void init_logger_C(int myid) {
    set_rank_C(myid);
    LoggingFramework lf(rank);    
    logger = lf;
}

void log_stringC(char* tag) {
    logger.log_string(tag);
}
void log_timingC(char* tag, double time) {
    logger.log_timing1(tag, time);
}
void log_commC(char* tag, int dest, int size) {
    logger.log_comm(tag, dest, size);
}

void logGPUMem_C (int arr_id) {
    logger.logGPUArrInfo(getGPUArray_Metadata(arr_id));
}

void get_arr_shape1D_C(int arr_id, int* shape_i) {
    *shape_i = getGPUArray_Metadata(arr_id).i;
}
void get_arr_shape2D_C(int arr_id, int* shape_i, int* shape_j) {
    *shape_i = getGPUArray_Metadata(arr_id).i;
    *shape_j = getGPUArray_Metadata(arr_id).j;
}
void get_arr_shape3D_C(int arr_id, int* shape_i, int* shape_j, int* shape_k) {
    *shape_i = getGPUArray_Metadata(arr_id).i;
    *shape_j = getGPUArray_Metadata(arr_id).j;
    *shape_k = getGPUArray_Metadata(arr_id).k;
}
void get_arr_nameC(int arr_id, char* arr_name) {
    strcpy(arr_name, getGPUArray_Metadata(arr_id).arr_name);
}

void upload_to_gpuC(double* arr_f, int arr_id) {
    int i, len;
    double* dptr;
    GPUArray_Metadata arr_meta = getGPUArray_Metadata(arr_id);
    switch(arr_meta.dimensions) {
        case 3: 
            len = (arr_meta.i)*(arr_meta.j)*(arr_meta.k); 
            dptr = arr_meta.dptr_3D;
            break;
        case 2: 
            len = (arr_meta.i)*(arr_meta.j); 
            dptr = arr_meta.dptr_2D;
            break;
        case 1: 
            len = (arr_meta.i); 
            dptr = arr_meta.dptr_1D;
            break;
        default:
            printf("Wrong dimensions of arr\n"); exit(0);
    }
    
    cudaError_t cudaRes;
    cudaRes = cudaMemcpy (dptr, arr_f, sizeof(double)*len, cudaMemcpyHostToDevice);
    CHECK_ERROR (cudaRes, "Error in Temp copying for logging: ");
}

void dwnld_gpu_arrC (int arr_id,  void** array) {
    int i, len;
    double* dptr;
    GPUArray_Metadata arr_meta = getGPUArray_Metadata(arr_id);
    switch(arr_meta.dimensions) {
        case 3: 
            len = (arr_meta.i)*(arr_meta.j)*(arr_meta.k); 
            dptr = arr_meta.dptr_3D;
            break;
        case 2: 
            len = (arr_meta.i)*(arr_meta.j); 
            dptr = arr_meta.dptr_2D;
            break;
        case 1: 
            len = (arr_meta.i); 
            dptr = arr_meta.dptr_1D;
            break;
        default:
            printf("Wrong dimensions of arr\n"); exit(0);
    }
    
    cudaError_t cudaRes;
    temp_gpu_arr = (double*) malloc (sizeof(double)* len );
    cudaRes = cudaMemcpy (temp_gpu_arr, dptr, sizeof(double)*len, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CHECK_ERROR (cudaRes, "Error in Temp copying for logging: ");
    *array = temp_gpu_arr;
}

void validate_array (double* array, GPUArray_Metadata arr_meta) {
    int i, len;
    double* dptr;
    double* temp_arr;
    switch(arr_meta.dimensions) {
        case 3: 
            len = (arr_meta.i)*(arr_meta.j)*(arr_meta.k); 
            dptr = arr_meta.dptr_3D;
            break;
        case 2: 
            len = (arr_meta.i)*(arr_meta.j); 
            dptr = arr_meta.dptr_2D;
            break;
        case 1: 
            len = (arr_meta.i); 
            dptr = arr_meta.dptr_1D;
            break;
        default:
            printf("Wrong dimensions of arr\n"); exit(0);
    }
    
    cudaError_t cudaRes;
    temp_arr = (double*) malloc (sizeof(double)* len );
    cudaRes = cudaMemcpy (temp_arr, dptr, sizeof(double)*len, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    CHECK_ERROR (cudaRes, "Error in Temp copying for logging: ");
    
    int same=1;
    for(int i=0; i<len; i++) {
	if (abs(temp_arr[i]-array[i]) > 0.0001) {
		printf("Validation Failed for %s at %d : host= %f, device= %f \n", arr_meta.arr_name, i, array[i], temp_arr[i] );
		same=0;
	}
    }
    if (same == 1) { 
	printf("Array %s same. First Two Elements: %f,%f & %f,%f\n", arr_meta.arr_name, array[0], array[1], temp_arr[0], temp_arr[1]);
    }
}

void free_gpu_arrC(double* array) {
    free(array);
}
void print_arrayC (int* arr, int length, char* nameS) {
    
	printf ("%s: ", nameS);
	for (int i=0; i<length ; i++) {	
		printf ("%d-%d", i, arr[i]);
		if (i != length-1)
			printf(", ");
	}
	printf ("\n");
}
//===========================================================================
/*
    Called from disfd.f90 as one time data transfer to GPU before
    the time series loop starts
*/    

void one_time_data_vel (int* index_xyz_source, int ixsX, int ixsY, int ixsZ,
                            double* famp, int fampX, int fampY,
                             double* ruptm, int ruptmX, 
                             double* riset, int risetX,
                             double* sparam2, int sparam2X ) 
{
    cudaError_t cudaRes;

    //cudaRes = cudaMalloc((int **)&nd1_velD, sizeof(int) * 18);
    //CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_vel");
    cudaRes = cudaMalloc((void**)&index_xyz_sourceD, sizeof(int) * (ixsX) * (ixsY) * (ixsZ));
    CHECK_ERROR(cudaRes, "Allocate Device Memory1, index_xyz_source");
    cudaRes = cudaMalloc((void**)&fampD, sizeof(double) * (fampX) * (fampY));
    CHECK_ERROR(cudaRes, "Allocate Device Memory1, fampX");
    cudaRes = cudaMalloc((void**)&ruptmD, sizeof(double) * (ruptmX));
    CHECK_ERROR(cudaRes, "Allocate Device Memory1, ruptmX");
    cudaRes = cudaMalloc((void**)&risetD, sizeof(double) * (risetX));
    CHECK_ERROR(cudaRes, "Allocate Device Memory1, risetX");
    cudaRes = cudaMalloc((void**)&sparam2D, sizeof(double) * (sparam2X));
    CHECK_ERROR(cudaRes, "Allocate Device Memory1, sparam2X");

    cudaRes = cudaMemcpy(index_xyz_sourceD, index_xyz_source, sizeof(int) * (ixsX) * (ixsY) *(ixsZ), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, index_xyz_source");
    cudaRes = cudaMemcpy(fampD, famp, sizeof(double) * (fampX) * (fampY), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, fampx");
    cudaRes = cudaMemcpy(ruptmD, ruptm, sizeof(double) * (ruptmX), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, ruptm");
    cudaRes = cudaMemcpy(risetD, riset, sizeof(double) * (risetX), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, riset");
    cudaRes = cudaMemcpy(sparam2D, sparam2, sizeof(double) * (sparam2X), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, sparam2");
}


void allocate_gpu_memC(int   *lbx,
					int   *lby,
					int   *nmat,		//dimension #, int
					int	  *mw1_pml1,	//int
					int	  *mw2_pml1,	//int
					int	  *nxtop,		//int
					int	  *nytop,		//int
					int   *nztop,
					int	  *mw1_pml,		//int
					int   *mw2_pml,		//int
					int	  *nxbtm,		//int
					int	  *nybtm,		//int
					int	  *nzbtm,
					int   *nzbm1,
					int   *nll)
{
	int nv2, nti, nth;
	cudaError_t cudaRes;

  printf("lbx[1] = %d, lbx[0] = %d\n", lbx[1], lbx[0]);
  printf("lby[1] = %d, lby[0] = %d\n", lby[1], lby[0]);
  printf("nmat = %d\n", *nmat);
  printf("mw1_pml1 = %d, mw2_pml1 = %d\n", *mw1_pml1, *mw2_pml1);
  printf("mw1_pml = %d, mw2_pml = %d\n", *mw1_pml, *mw2_pml);
  printf("nxtop = %d, nytop = %d, nztop = %d\n", *nxtop, *nytop, *nztop);
  printf("nxbtm = %d, nybtm = %d, nzbtm = %d\n", *nxbtm, *nybtm, *nzbtm);
  printf("nzbm1 = %d, nll = %d\n", *nzbm1, *nll);
	//debug-----------------
	totalTimeH2DV = 0.0f;
	totalTimeD2HV = 0.0f;
	totalTimeH2DS = 0.0f;
	totalTimeD2HS = 0.0f;
	totalTimeCompV = 0.0f;
	totalTimeCompS = 0.0f;

	//for inner_I
	cudaRes = cudaMalloc((int **)&nd1_velD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_vel");
	cudaRes = cudaMalloc((void **)&nd1_txyD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_txy");
	cudaRes = cudaMalloc((void **)&nd1_txzD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_txz");
	cudaRes = cudaMalloc((void **)&nd1_tyyD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_tyy");
	cudaRes = cudaMalloc((void **)&nd1_tyzD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_tyz");

	cudaRes = cudaMalloc((void **)&rhoD, sizeof(double) * (*nmat));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, rho");
	cudaRes = cudaMalloc((void **)&drvh1D, sizeof(double) * (*mw1_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, drvh1");
	cudaRes = cudaMalloc((void **)&drti1D, sizeof(double) * (*mw1_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, drti1");
	cudaRes = cudaMalloc((void **)&drth1D, sizeof(double) * (*mw1_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, drth1");

	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMalloc((void **)&damp1_xD, sizeof(double) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMalloc((void **)&damp1_yD, sizeof(double) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, damp1_y");
	}

	cudaRes = cudaMalloc((void **)&idmat1D, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, idmat1");
	cudaRes = cudaMalloc((void **)&dxi1D, sizeof(double) * 4 * (*nxtop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dxi1");
	cudaRes = cudaMalloc((void **)&dyi1D, sizeof(double) * 4 * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dyi1");
	cudaRes = cudaMalloc((void **)&dzi1D, sizeof(double) * 4 * (*nztop + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dzi1");
	cudaRes = cudaMalloc((void **)&dxh1D, sizeof(double) * 4 * (*nxtop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dxh1");
	cudaRes = cudaMalloc((void **)&dyh1D, sizeof(double) * 4 * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dyh1");
	cudaRes = cudaMalloc((void **)&dzh1D, sizeof(double) * 4 * (*nztop + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dzh1");

	cudaRes = cudaMalloc((void **)&t1xxD, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xx");
	cudaRes = cudaMalloc((void **)&t1xyD, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xy");
	cudaRes = cudaMalloc((void **)&t1xzD, sizeof(double) * (*nztop + 1) * (*nxtop + 3) * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xz");
	cudaRes = cudaMalloc((void **)&t1yyD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1yy");
	cudaRes = cudaMalloc((void **)&t1yzD, sizeof(double) * (*nztop + 1) * (*nxtop) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1yz");
	cudaRes = cudaMalloc((void **)&t1zzD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1zz");
	
	if (lbx[1] >= lbx[0])
	{
		nti = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1];
		nth = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + 1 - lbx[0];
		cudaMalloc((void **)&t1xx_pxD,  sizeof(double) * (*nztop) * (nti) * (*nytop));
		cudaMalloc((void **)&t1xy_pxD,  sizeof(double) * (*nztop) * nth * (*nytop));
		cudaMalloc((void **)&t1xz_pxD,  sizeof(double) * (*nztop+1) * nth * (*nytop));
		cudaMalloc((void **)&t1yy_pxD,  sizeof(double) * (*nztop) * nti * (*nytop));

		cudaMalloc((void **)&qt1xx_pxD, sizeof(double) * (*nztop) * (nti) * (*nytop));
		cudaMalloc((void **)&qt1xy_pxD, sizeof(double) * (*nztop) * nth * (*nytop));
		cudaMalloc((void **)&qt1xz_pxD, sizeof(double) * (*nztop+1) * nth * (*nytop));
		cudaMalloc((void **)&qt1yy_pxD, sizeof(double) * (*nztop) * nti * (*nytop));
	}

	if (lby[1] >= lby[0])
	{
		nti = (lby[1] - lby[0] + 1) * (*mw1_pml) + lby[1];
		nth = (lby[1] - lby[0] + 1) * (*mw1_pml) + 1 - lby[0];
		cudaMalloc((void **)&t1xx_pyD,  sizeof(double) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&t1xy_pyD,  sizeof(double) * (*nztop) * (*nxtop) * nth);
		cudaMalloc((void **)&t1yy_pyD,  sizeof(double) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&t1yz_pyD,  sizeof(double) * (*nztop+1) * (*nxtop) * nth);
		cudaMalloc((void **)&qt1xx_pyD, sizeof(double) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&qt1xy_pyD, sizeof(double) * (*nztop) * (*nxtop) * nth);
		cudaMalloc((void **)&qt1yy_pyD, sizeof(double) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&qt1yz_pyD, sizeof(double) * (*nztop+1) * (*nxtop) * nth);
	}

	cudaMalloc((void **)&qt1xxD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1xyD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1xzD, sizeof(double) * (*nztop+1) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1yyD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1yzD, sizeof(double) * (*nztop+1) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1zzD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop));

	cudaMalloc((void **)&clamdaD, sizeof(double) * (*nmat));
	cudaMalloc((void **)&cmuD,    sizeof(double) * (*nmat));
	cudaMalloc((void **)&epdtD,   sizeof(double) * (*nll));
	cudaMalloc((void **)&qwpD,    sizeof(double) * (*nmat));
	cudaMalloc((void **)&qwsD,    sizeof(double) * (*nmat));
	cudaMalloc((void **)&qwt1D,   sizeof(double) * (*nll));
	cudaMalloc((void **)&qwt2D,   sizeof(double) * (*nll));

	cudaRes = cudaMalloc((void **)&v1xD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1x");
	cudaRes = cudaMalloc((void **)&v1yD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1y");
	cudaRes = cudaMalloc((void **)&v1zD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1z");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw1_pml);
		cudaRes = cudaMalloc((void **)&v1x_pxD, sizeof(double) * (*nztop) * nv2 * (*nytop));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1x_px");
		cudaRes = cudaMalloc((void **)&v1y_pxD, sizeof(double) * (*nztop) * nv2 * (*nytop));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1y_px");
		cudaRes = cudaMalloc((void **)&v1z_pxD, sizeof(double) * (*nztop) * nv2 * (*nytop));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw1_pml);
		cudaRes = cudaMalloc((void **)&v1x_pyD, sizeof(double) * (*nztop) * (*nxtop) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1x_py");
		cudaRes = cudaMalloc((void **)&v1y_pyD, sizeof(double) * (*nztop) * (*nxtop) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1y_py");
		cudaRes = cudaMalloc((void **)&v1z_pyD, sizeof(double) * (*nztop) * (*nxtop) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1z_py");
	}

//for inner_II-----------------------------------------------------------------------------------------
	cudaRes = cudaMalloc((void **)&nd2_velD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, nd2_vel");
	cudaRes = cudaMalloc((void **)&nd2_txyD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, nd2_txy");
	cudaRes = cudaMalloc((void **)&nd2_txzD, sizeof(int) * 18); 
	CHECK_ERROR(cudaRes, "Allocate Device Memory, nd2_txz");
	cudaRes = cudaMalloc((void **)&nd2_tyyD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, nd2_tyy");
	cudaRes = cudaMalloc((void **)&nd2_tyzD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, nd2_tyz");
	cudaRes = cudaMalloc((void **)&drvh2D, sizeof(double) * (*mw2_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, drvh2");
	cudaRes = cudaMalloc((void **)&drti2D, sizeof(double) * (*mw2_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, drti2");
	cudaRes = cudaMalloc((void **)&drth2D, sizeof(double) * (*mw2_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, drth2");

	cudaRes = cudaMalloc((void **)&idmat2D, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMalloc((void **)&damp2_xD, sizeof(double) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMalloc((void **)&damp2_yD, sizeof(double) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, damp2_y");
	}
	cudaRes = cudaMalloc((void **)&damp2_zD, sizeof(double) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, damp2_z");

	cudaRes = cudaMalloc((void **)&dxi2D, sizeof(double) * 4 * (*nxbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dxi2");
	cudaRes = cudaMalloc((void **)&dyi2D, sizeof(double) * 4 * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dyi2");
	cudaRes = cudaMalloc((void **)&dzi2D, sizeof(double) * 4 * (*nzbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dzi2");
	cudaRes = cudaMalloc((void **)&dxh2D, sizeof(double) * 4 * (*nxbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dxh2");
	cudaRes = cudaMalloc((void **)&dyh2D, sizeof(double) * 4 * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dyh2");
	cudaRes = cudaMalloc((void **)&dzh2D, sizeof(double) * 4 * (*nzbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dzh2");

	cudaRes = cudaMalloc((void **)&t2xxD, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2xx");
	cudaRes = cudaMalloc((void **)&t2xyD, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2xy");
	cudaRes = cudaMalloc((void **)&t2xzD, sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2xz");
	cudaRes = cudaMalloc((void **)&t2yyD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2yy");
	cudaRes = cudaMalloc((void **)&t2yzD, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2yz");
	cudaRes = cudaMalloc((void **)&t2zzD, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2zz");

	cudaMalloc((void **)&qt2xxD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2xyD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2xzD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2yyD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2yzD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2zzD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm));


	if (lbx[1] >= lbx[0])
	{
        nti = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + lbx[1];
        nth = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + 1 - lbx[0];
		cudaMalloc((void **)&t2xx_pxD, sizeof(double) * (*nzbtm) * nti * (*nybtm));
		cudaMalloc((void **)&t2xy_pxD, sizeof(double) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&t2xz_pxD, sizeof(double) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&t2yy_pxD, sizeof(double) * (*nzbtm) * nti * (*nybtm));

		cudaMalloc((void **)&qt2xx_pxD, sizeof(double) * (*nzbtm) * nti * (*nybtm));
		cudaMalloc((void **)&qt2xy_pxD, sizeof(double) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&qt2xz_pxD, sizeof(double) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&qt2yy_pxD, sizeof(double) * (*nzbtm) * nti * (*nybtm));
	}

	if (lby[1] >= lby[0])
	{
        nti = (lby[1] - lby[0] + 1) * (*mw2_pml) + lby[1];
        nth = (lby[1] - lby[0] + 1) * (*mw2_pml) + 1 - lby[0];
		cudaMalloc((void **)&t2xx_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&t2xy_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nth);
		cudaMalloc((void **)&t2yy_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&t2yz_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nth);

		cudaMalloc((void **)&qt2xx_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&qt2xy_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nth);
		cudaMalloc((void **)&qt2yy_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&qt2yz_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nth);
	}

	cudaMalloc((void **)&t2xx_pzD, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&t2xz_pzD, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&t2yz_pzD, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&t2zz_pzD, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm));

	cudaMalloc((void **)&qt2xx_pzD, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2xz_pzD, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2yz_pzD, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2zz_pzD, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm));

	cudaMalloc((void **)&v2xD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3));
	cudaMalloc((void **)&v2yD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3));
	cudaMalloc((void **)&v2zD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3));

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw2_pml);
		cudaRes = cudaMalloc((void **)&v2x_pxD, sizeof(double) * (*nzbtm) * nv2 * (*nybtm));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2x_px");
		cudaRes = cudaMalloc((void **)&v2y_pxD, sizeof(double) * (*nzbtm) * nv2 * (*nybtm));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2y_px");
		cudaRes = cudaMalloc((void **)&v2z_pxD, sizeof(double) * (*nzbtm) * nv2 * (*nybtm));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw2_pml);
		cudaRes = cudaMalloc((void **)&v2x_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2x_py");
		cudaRes = cudaMalloc((void **)&v2y_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2y_py");
		cudaRes = cudaMalloc((void **)&v2z_pyD, sizeof(double) * (*nzbtm) * (*nxbtm) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2z_py");
	}

	cudaRes = cudaMalloc((void **)&v2x_pzD, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, v2x_pz");
	cudaRes = cudaMalloc((void **)&v2y_pzD, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, v2y_pz");
	cudaRes = cudaMalloc((void **)&v2z_pzD, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, v2z_pz");

// MPI-ACC
	cudaRes = cudaMalloc((void **)&sdx51D, sizeof(double) * (*nztop) * (*nytop) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdx51D");
	cudaRes = cudaMalloc((void **)&sdx52D, sizeof(double) * (*nzbtm) * (*nybtm) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdx52D");
	cudaRes = cudaMalloc((void **)&sdx41D, sizeof(double) * (*nztop) * (*nytop) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdx41D");
	cudaRes = cudaMalloc((void **)&sdx42D, sizeof(double) * (*nzbtm) * (*nybtm) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdx42D");

	cudaRes = cudaMalloc((void **)&sdy51D, sizeof(double) * (*nztop) * (*nxtop) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdy51D");
	cudaRes = cudaMalloc((void **)&sdy52D, sizeof(double) * (*nzbtm) * (*nxbtm) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdy52D");
	cudaRes = cudaMalloc((void **)&sdy41D, sizeof(double) * (*nztop) * (*nxtop) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdy41D");
	cudaRes = cudaMalloc((void **)&sdy42D, sizeof(double) * (*nzbtm) * (*nxbtm) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdy42D");

	cudaRes = cudaMalloc((void **)&rcx51D, sizeof(double) * (*nztop) * (*nytop) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcx51D");
	cudaRes = cudaMalloc((void **)&rcx52D, sizeof(double) * (*nzbtm) * (*nybtm) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcx52D");
	cudaRes = cudaMalloc((void **)&rcx41D, sizeof(double) * (*nztop) * (*nytop) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcx41D");
	cudaRes = cudaMalloc((void **)&rcx42D, sizeof(double) * (*nzbtm) * (*nybtm) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcx42D");

	cudaRes = cudaMalloc((void **)&rcy51D, sizeof(double) * (*nztop) * (*nxtop) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcy51D");
	cudaRes = cudaMalloc((void **)&rcy52D, sizeof(double) * (*nzbtm) * (*nxbtm) * (5));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcy52D");
	cudaRes = cudaMalloc((void **)&rcy41D, sizeof(double) * (*nztop) * (*nxtop) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcy41D");
	cudaRes = cudaMalloc((void **)&rcy42D, sizeof(double) * (*nzbtm) * (*nxbtm) * (4));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcy42D");

        cudaRes = cudaMalloc((void **)&sdy1D, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdx1D");
        cudaRes = cudaMalloc((void **)&sdy2D, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdy2D");
        cudaRes = cudaMalloc((void **)&rcy1D, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcy1D");
        cudaRes = cudaMalloc((void **)&rcy2D, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcy2D");

        cudaRes = cudaMalloc((void **)&sdx1D, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdx1D");
        cudaRes = cudaMalloc((void **)&sdx2D, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, sdx2D");
        cudaRes = cudaMalloc((double **)&rcx1D, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcx1D");
        cudaRes = cudaMalloc((double **)&rcx2D, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, rcx2D");

       //memset them to 0
        cudaRes = cudaMemset(sdx51D, 0, sizeof(double) * (*nztop) * (*nytop) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdx51D");
	cudaRes = cudaMemset(sdx52D, 0, sizeof(double) * (*nzbtm) * (*nybtm) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdx52D");
	cudaRes = cudaMemset(sdx41D, 0, sizeof(double) * (*nztop) * (*nytop) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdx41D");
	cudaRes = cudaMemset(sdx42D, 0, sizeof(double) * (*nzbtm) * (*nybtm) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdx42D");

	cudaRes = cudaMemset(sdy51D, 0, sizeof(double) * (*nztop) * (*nxtop) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdy51D");
	cudaRes = cudaMemset(sdy52D, 0, sizeof(double) * (*nzbtm) * (*nxbtm) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdy52D");
	cudaRes = cudaMemset(sdy41D, 0, sizeof(double) * (*nztop) * (*nxtop) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdy41D");
	cudaRes = cudaMemset(sdy42D, 0, sizeof(double) * (*nzbtm) * (*nxbtm) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdy42D");

	cudaRes = cudaMemset(rcx51D, 0, sizeof(double) * (*nztop) * (*nytop) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcx51D");
	cudaRes = cudaMemset(rcx52D, 0, sizeof(double) * (*nzbtm) * (*nybtm) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcx52D");
	cudaRes = cudaMemset(rcx41D, 0, sizeof(double) * (*nztop) * (*nytop) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcx41D");
	cudaRes = cudaMemset(rcx42D, 0, sizeof(double) * (*nzbtm) * (*nybtm) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcx42D");

	cudaRes = cudaMemset(rcy51D, 0, sizeof(double) * (*nztop) * (*nxtop) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcy51D");
	cudaRes = cudaMemset(rcy52D, 0, sizeof(double) * (*nzbtm) * (*nxbtm) * (5));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcy52D");
	cudaRes = cudaMemset(rcy41D, 0, sizeof(double) * (*nztop) * (*nxtop) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcy41D");
	cudaRes = cudaMemset(rcy42D, 0, sizeof(double) * (*nzbtm) * (*nxbtm) * (4));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcy42D");

        cudaRes = cudaMemset(sdy1D, 0, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdx1D");
        cudaRes = cudaMemset(sdy2D, 0, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdy2D");
        cudaRes = cudaMemset(rcy1D, 0, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcy1D");
        cudaRes = cudaMemset(rcy2D, 0, sizeof(double) * (*nxtop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcy2D");

        cudaRes = cudaMemset(sdx1D, 0, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdx1D");
        cudaRes = cudaMemset(sdx2D, 0, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, sdx2D");
        cudaRes = cudaMemset(rcx1D, 0, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcx1D");
        cudaRes = cudaMemset(rcx2D, 0, sizeof(double) * (*nytop + 6));
	CHECK_ERROR(cudaRes, "CUDA-Memset, rcx2D");

        //
        // ch/i x/y
        cudaRes = cudaMalloc((void **)&cixD, sizeof(double) * (*nxbtm +6+1)*8);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, cixD");
        cudaRes = cudaMalloc((void **)&ciyD, sizeof(double) * (*nybtm +6+1)*8);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, ciyD");
        cudaRes = cudaMalloc((void **)&chxD, sizeof(double) * (*nxbtm +6+1)*8);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, chxD");
        cudaRes = cudaMalloc((void **)&chyD, sizeof(double) * (*nybtm +6+1)*8);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, chyD");


	return;
}
	
void cpy_h2d_velocityInputsCOneTime(int   *lbx,
						  int   *lby,
						  int   *nd1_vel,
						  double *rho,
						  double *drvh1,
						  double *drti1,
						  double *damp1_x,
						  double *damp1_y,
						  int	*idmat1,
						  double *dxi1,
						  double *dyi1,
						  double *dzi1,
						  double *dxh1,
						  double *dyh1,
						  double *dzh1,
						  double *t1xx,
						  double *t1xy,
						  double *t1xz,
						  double *t1yy,
						  double *t1yz,
						  double *t1zz,
						  double *v1x_px,
						  double *v1y_px,
						  double *v1z_px,
						  double *v1x_py,
						  double *v1y_py,
						  double *v1z_py,
						  int	*nd2_vel,
						  double *drvh2,
						  double *drti2,
						  int	*idmat2,
						  double *damp2_x,
						  double *damp2_y,
						  double *damp2_z,
						  double *dxi2,
						  double *dyi2,
						  double *dzi2,
						  double *dxh2,
						  double *dyh2,
						  double *dzh2,
						  double *t2xx,
						  double *t2xy,
						  double *t2xz,
						  double *t2yy,
						  double *t2yz,
						  double *t2zz,
						  double *v2x_px,
						  double *v2y_px,
						  double *v2z_px,
						  double *v2x_py,
						  double *v2y_py,
						  double *v2z_py,
						  double *v2x_pz,
						  double *v2y_pz,
						  double *v2z_pz,
                                                  double* cix,
                                                  double* ciy,
                                                  double* chx,
                                                  double* chy,
						  int   *nmat,		//dimension #, int
						  int	*mw1_pml1,	//int
						  int	*mw2_pml1,	//int
						  int	*nxtop,		//int
						  int	*nytop,		//int
						  int   *nztop,
						  int	*mw1_pml,	//int
						  int   *mw2_pml,	//int
						  int	*nxbtm,		//int
						  int	*nybtm,		//int
						  int	*nzbtm,
						  int   *nzbm1)
{
	cudaError_t cudaRes;
	int nv2;

	//for inner_I
	cudaRes = cudaMemcpy(nd1_velD, nd1_vel, sizeof(int) * 18, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, nd1_vel");

	cudaRes = cudaMemcpy(rhoD, rho, sizeof(double) * (*nmat), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, rho");
	cudaRes = cudaMemcpy(drvh1D, drvh1, sizeof(double) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drvh1");
	cudaRes = cudaMemcpy(drti1D, drti1, sizeof(double) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drti1");

	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp1_xD, damp1_x, 
					sizeof(double) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp1_yD, damp1_y, 
					sizeof(double) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp1_y");
	}

	cudaRes = cudaMemcpy(idmat1D, idmat1, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, idmat1");
	cudaRes = cudaMemcpy(dxi1D, dxi1, sizeof(double) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxi1");
	cudaRes = cudaMemcpy(dyi1D, dyi1, sizeof(double) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyi1");
	cudaRes = cudaMemcpy(dzi1D, dzi1, sizeof(double) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzi1");
	cudaRes = cudaMemcpy(dxh1D, dxh1, sizeof(double) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxh1");
	cudaRes = cudaMemcpy(dyh1D, dyh1, sizeof(double) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyh1");
	cudaRes = cudaMemcpy(dzh1D, dzh1, sizeof(double) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzh1");

	cudaRes = cudaMemcpy(t1xxD, t1xx, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xx");
	cudaRes = cudaMemcpy(t1xyD, t1xy, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xy");
	cudaRes = cudaMemcpy(t1xzD, t1xz, sizeof(double) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xz");
	cudaRes = cudaMemcpy(t1yyD, t1yy, sizeof(double) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yy");
	cudaRes = cudaMemcpy(t1yzD, t1yz, sizeof(double) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yz");
	cudaRes = cudaMemcpy(t1zzD, t1zz, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1zz");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw1_pml);
		cudaRes = cudaMemcpy(v1x_pxD, v1x_px, sizeof(double) * (*nztop) * nv2 * (*nytop), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1x_px");
		cudaRes = cudaMemcpy(v1y_pxD, v1y_px, sizeof(double) * (*nztop) * nv2 * (*nytop), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1y_px");
		cudaRes = cudaMemcpy(v1z_pxD, v1z_px, sizeof(double) * (*nztop) * nv2 * (*nytop), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw1_pml);
		cudaRes = cudaMemcpy(v1x_pyD, v1x_py, sizeof(double) * (*nztop) * (*nxtop) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1x_py");
		cudaRes = cudaMemcpy(v1y_pyD, v1y_py, sizeof(double) * (*nztop) * (*nxtop) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1y_py");
		cudaRes = cudaMemcpy(v1z_pyD, v1z_py, sizeof(double) * (*nztop) * (*nxtop) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1z_py");
	}


	//for inner_II
	cudaRes = cudaMemcpy(nd2_velD, nd2_vel, sizeof(int) * 18, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, nd2_vel");

	cudaRes = cudaMemcpy(drvh2D, drvh2, sizeof(double) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drvh2");
	cudaRes = cudaMemcpy(drti2D, drti2, sizeof(double) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drti2");

	cudaRes = cudaMemcpy(idmat2D, idmat2, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm +1),  cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp2_xD, damp2_x, 
					sizeof(double) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp2_yD, damp2_y, 
					sizeof(double) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp2_y");
	}
	cudaRes = cudaMemcpy(damp2_zD, damp2_z, sizeof(double) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp2_z");

	cudaRes = cudaMemcpy(dxi2D, dxi2, sizeof(double) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxi2");
	cudaRes = cudaMemcpy(dyi2D, dyi2, sizeof(double) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyi2");
	cudaRes = cudaMemcpy(dzi2D, dzi2, sizeof(double) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzi2");
	cudaRes = cudaMemcpy(dxh2D, dxh2, sizeof(double) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxh2");
	cudaRes = cudaMemcpy(dyh2D, dyh2, sizeof(double) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyh2");
	cudaRes = cudaMemcpy(dzh2D, dzh2, sizeof(double) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzh2");

	cudaRes = cudaMemcpy(t2xxD, t2xx, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xx");
	cudaRes = cudaMemcpy(t2xyD, t2xy, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xy");
	cudaRes = cudaMemcpy(t2xzD, t2xz, sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xz");
	cudaRes = cudaMemcpy(t2yyD, t2yy, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yy");
	cudaRes = cudaMemcpy(t2yzD, t2yz, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yz");
	cudaRes = cudaMemcpy(t2zzD, t2zz, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2zz");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw2_pml);
		cudaRes = cudaMemcpy(v2x_pxD, v2x_px, sizeof(double) * (*nzbtm) * nv2 * (*nybtm), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x_px");
		cudaRes = cudaMemcpy(v2y_pxD, v2y_px, sizeof(double) * (*nzbtm) * nv2 * (*nybtm), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y_px");
		cudaRes = cudaMemcpy(v2z_pxD, v2z_px, sizeof(double) * (*nzbtm) * nv2 * (*nybtm), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw2_pml);
		cudaRes = cudaMemcpy(v2x_pyD, v2x_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x_py");
		cudaRes = cudaMemcpy(v2y_pyD, v2y_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y_py");
		cudaRes = cudaMemcpy(v2z_pyD, v2z_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z_py");
	}

	cudaRes = cudaMemcpy(v2x_pzD, v2x_pz, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x_pz");
	cudaRes = cudaMemcpy(v2y_pzD, v2y_pz, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y_pz");
	cudaRes = cudaMemcpy(v2z_pzD, v2z_pz, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z_pz");

        // COMMON
	cudaRes = cudaMemcpy(cixD, cix, sizeof(double) * (*nxbtm + 6 +1) * (8), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, cix");
	cudaRes = cudaMemcpy(ciyD, ciy, sizeof(double) * (*nybtm + 6 +1) * (8), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, cix");
	cudaRes = cudaMemcpy(chxD, chx, sizeof(double) * (*nxbtm + 6 +1) * (8), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, cix");
	cudaRes = cudaMemcpy(chyD, chy, sizeof(double) * (*nybtm + 6 +1) * (8), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, cix");

	return;
}

void cpy_h2d_velocityInputsC(double *t1xx,
							double *t1xy,
							double *t1xz,
							double *t1yy,
							double *t1yz,
							double *t1zz,
							double *t2xx,
							double *t2xy,
							double *t2xz,
							double *t2yy,
							double *t2yz,
							double *t2zz,
							int	*nxtop,		
							int	*nytop,		
							int *nztop,
							int	*nxbtm,		
							int	*nybtm,		
							int	*nzbtm)
{
	cudaError_t cudaRes;

	//for inner_I
	cudaRes = cudaMemcpy(t1xxD, t1xx, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xx");
	cudaRes = cudaMemcpy(t1xyD, t1xy, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xy");
	cudaRes = cudaMemcpy(t1xzD, t1xz, sizeof(double) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xz");
	cudaRes = cudaMemcpy(t1yyD, t1yy, sizeof(double) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yy");
	cudaRes = cudaMemcpy(t1yzD, t1yz, sizeof(double) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yz");
	cudaRes = cudaMemcpy(t1zzD, t1zz, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1zz");

	//for inner_II
	cudaRes = cudaMemcpy(t2xxD, t2xx, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xx");
	cudaRes = cudaMemcpy(t2xyD, t2xy, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xy");
	cudaRes = cudaMemcpy(t2xzD, t2xz, sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xz");
	cudaRes = cudaMemcpy(t2yyD, t2yy, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yy");
	cudaRes = cudaMemcpy(t2yzD, t2yz, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yz");
	cudaRes = cudaMemcpy(t2zzD, t2zz, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2zz");

	return;
}

//=====================================================================
void cpy_h2d_stressInputsCOneTime(int   *lbx,
						  int   *lby,
						  int   *nd1_txy,
						  int   *nd1_txz,
						  int   *nd1_tyy,
						  int   *nd1_tyz,
						  double *drti1,
						  double *drth1,
						  double *damp1_x,
						  double *damp1_y,
						  int	*idmat1,
						  double *dxi1,
						  double *dyi1,
						  double *dzi1,
						  double *dxh1,
						  double *dyh1,
						  double *dzh1,
						  double *v1x,
						  double *v1y,
						  double *v1z,
						  double *t1xx_px,
						  double *t1xy_px,
						  double *t1xz_px,
						  double *t1yy_px,
						  double *qt1xx_px,
						  double *qt1xy_px,
						  double *qt1xz_px,
						  double *qt1yy_px,
						  double *t1xx_py,
						  double *t1xy_py,
						  double *t1yy_py,
						  double *t1yz_py,
						  double *qt1xx_py,
						  double *qt1xy_py,
						  double *qt1yy_py,
						  double *qt1yz_py,
						  double *qt1xx,
						  double *qt1xy,
						  double *qt1xz,
						  double *qt1yy,
						  double *qt1yz,
						  double *qt1zz,
						  double *clamda,
						  double *cmu,
						  double *epdt,
						  double *qwp,
						  double *qws,
						  double *qwt1,
						  double *qwt2,
						  int   *nd2_txy,
						  int   *nd2_txz,
						  int   *nd2_tyy,
						  int   *nd2_tyz,
						  double *drti2,
						  double *drth2,
						  int	*idmat2,
						  double *damp2_x,
						  double *damp2_y,
						  double *damp2_z,
						  double *dxi2,
						  double *dyi2,
						  double *dzi2,
						  double *dxh2,
						  double *dyh2,
						  double *dzh2,
						  double *v2x,
						  double *v2y,
						  double *v2z,
						  double *qt2xx,
						  double *qt2xy,
						  double *qt2xz,
						  double *qt2yy,
						  double *qt2yz,
						  double *qt2zz,
						  double *t2xx_px,
						  double *t2xy_px,
						  double *t2xz_px,
						  double *t2yy_px,
						  double *qt2xx_px,
						  double *qt2xy_px,
						  double *qt2xz_px,
						  double *qt2yy_px,
						  double *t2xx_py,
						  double *t2xy_py,
						  double *t2yy_py,
						  double *t2yz_py,
						  double *qt2xx_py,
						  double *qt2xy_py,
						  double *qt2yy_py,
						  double *qt2yz_py,
						  double *t2xx_pz,
						  double *t2xz_pz,
						  double *t2yz_pz,
						  double *t2zz_pz,
						  double *qt2xx_pz,
						  double *qt2xz_pz,
						  double *qt2yz_pz,
						  double *qt2zz_pz,
						  int   *nmat,		//dimension #, int
						  int	*mw1_pml1,	//int
						  int	*mw2_pml1,	//int
						  int	*nxtop,		//int
						  int	*nytop,		//int
						  int   *nztop,
						  int	*mw1_pml,	//int
						  int   *mw2_pml,	//int
						  int	*nxbtm,		//int
						  int	*nybtm,		//int
						  int	*nzbtm,
						  int   *nll)
{
	cudaError_t cudaRes;
	int nti, nth;

	//for inner_I
	cudaRes = cudaMemcpy(nd1_txyD, nd1_txy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd1_txzD, nd1_txz, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd1_tyyD, nd1_tyy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd1_tyzD, nd1_tyz, sizeof(int) * 18, cudaMemcpyHostToDevice);

	cudaRes = cudaMemcpy(drti1D, drti1, sizeof(double) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drti1");
	cudaRes = cudaMemcpy(drth1D, drth1, sizeof(double) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drth1");

	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp1_xD, damp1_x, 
					sizeof(double) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp1_yD, damp1_y, 
					sizeof(double) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp1_y");
	}

	cudaRes = cudaMemcpy(idmat1D, idmat1, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, idmat1");
	cudaRes = cudaMemcpy(dxi1D, dxi1, sizeof(double) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxi1");
	cudaRes = cudaMemcpy(dyi1D, dyi1, sizeof(double) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyi1");
	cudaRes = cudaMemcpy(dzi1D, dzi1, sizeof(double) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzi1");
	cudaRes = cudaMemcpy(dxh1D, dxh1, sizeof(double) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxh1");
	cudaRes = cudaMemcpy(dyh1D, dyh1, sizeof(double) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyh1");
	cudaRes = cudaMemcpy(dzh1D, dzh1, sizeof(double) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzh1");
        

	if (lbx[1] >= lbx[0])
	{
		nti = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1];
		nth = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + 1 - lbx[0];
		cudaMemcpy(t1xx_pxD, t1xx_px, sizeof(double) * (*nztop) * (nti) * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(t1xy_pxD, t1xy_px, sizeof(double) * (*nztop) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(t1xz_pxD, t1xz_px, sizeof(double) * (*nztop+1) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(t1yy_pxD, t1yy_px, sizeof(double) * (*nztop) * nti * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xx_pxD, qt1xx_px, sizeof(double) * (*nztop) * (nti) * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xy_pxD, qt1xy_px, sizeof(double) * (*nztop) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xz_pxD, qt1xz_px, sizeof(double) * (*nztop+1) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1yy_pxD, qt1yy_px, sizeof(double) * (*nztop) * nti * (*nytop), cudaMemcpyHostToDevice);
	}

	if (lby[1] >= lby[0])
	{
		nti = (lby[1] - lby[0] + 1) * (*mw1_pml) + lby[1];
		nth = (lby[1] - lby[0] + 1) * (*mw1_pml) + 1 - lby[0];
		cudaMemcpy(t1xx_pyD,  t1xx_py,  sizeof(double) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t1xy_pyD,  t1xy_py,  sizeof(double) * (*nztop) * (*nxtop) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(t1yy_pyD,  t1yy_py,  sizeof(double) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t1yz_pyD,  t1yz_py,  sizeof(double) * (*nztop+1) * (*nxtop) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xx_pyD, qt1xx_py, sizeof(double) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xy_pyD, qt1xy_py, sizeof(double) * (*nztop) * (*nxtop) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1yy_pyD, qt1yy_py, sizeof(double) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1yz_pyD, qt1yz_py, sizeof(double) * (*nztop+1) * (*nxtop) * nth, cudaMemcpyHostToDevice);
	}

	cudaMemcpy(qt1xxD, qt1xx, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1xyD, qt1xy, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1xzD, qt1xz, sizeof(double) * (*nztop+1) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1yyD, qt1yy, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1yzD, qt1yz, sizeof(double) * (*nztop+1) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1zzD, qt1zz, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);

	cudaMemcpy(clamdaD, clamda, sizeof(double) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(cmuD,    cmu,    sizeof(double) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(epdtD,   epdt,   sizeof(double) * (*nll),  cudaMemcpyHostToDevice);
	cudaMemcpy(qwpD,    qwp,    sizeof(double) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(qwsD,    qws,    sizeof(double) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(qwt1D,   qwt1,   sizeof(double) * (*nll),  cudaMemcpyHostToDevice);
	cudaMemcpy(qwt2D,   qwt2,   sizeof(double) * (*nll),  cudaMemcpyHostToDevice);

        cudaRes=  cudaMemcpy(v1yD, v1y,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, v1y stress");
        cudaRes = cudaMemcpy(v1zD, v1z,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, v1z stress");
        cudaRes = cudaMemcpy(v1xD, v1x,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, v1x stress");

	//for inner_II
	cudaRes = cudaMemcpy(nd2_txyD, nd2_txy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd2_txzD, nd2_txz, sizeof(int) * 18, cudaMemcpyHostToDevice); 
	cudaRes = cudaMemcpy(nd2_tyyD, nd2_tyy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd2_tyzD, nd2_tyz, sizeof(int) * 18, cudaMemcpyHostToDevice);

	cudaRes = cudaMemcpy(drti2D, drti2, sizeof(double) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drti2");
	cudaRes = cudaMemcpy(drth2D, drth2, sizeof(double) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drth2");

	cudaRes = cudaMemcpy(idmat2D, idmat2, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm +1),  cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp2_xD, damp2_x, 
					sizeof(double) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp2_yD, damp2_y, 
					sizeof(double) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp2_y");
	}
	cudaRes = cudaMemcpy(damp2_zD, damp2_z, sizeof(double) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp2_z");

	cudaRes = cudaMemcpy(dxi2D, dxi2, sizeof(double) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxi2");
	cudaRes = cudaMemcpy(dyi2D, dyi2, sizeof(double) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyi2");
	cudaRes = cudaMemcpy(dzi2D, dzi2, sizeof(double) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzi2");
	cudaRes = cudaMemcpy(dxh2D, dxh2, sizeof(double) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxh2");
	cudaRes = cudaMemcpy(dyh2D, dyh2, sizeof(double) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyh2");
	cudaRes = cudaMemcpy(dzh2D, dzh2, sizeof(double) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzh2");

	cudaMemcpy(v2xD, v2x,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2yD, v2y,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2zD, v2z,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);

	cudaMemcpy(qt2xxD, qt2xx, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xyD, qt2xy, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xzD, qt2xz, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2yyD, qt2yy, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2yzD, qt2yz, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2zzD, qt2zz, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);

	if (lbx[1] >= lbx[0])
	{
        nti = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + lbx[1];
        nth = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + 1 - lbx[0];
		cudaMemcpy(t2xx_pxD, t2xx_px, sizeof(double) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(t2xy_pxD, t2xy_px, sizeof(double) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(t2xz_pxD, t2xz_px, sizeof(double) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(t2yy_pxD, t2yy_px, sizeof(double) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xx_pxD, qt2xx_px, sizeof(double) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xy_pxD, qt2xy_px, sizeof(double) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xz_pxD, qt2xz_px, sizeof(double) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2yy_pxD, qt2yy_px, sizeof(double) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
	}

	if (lby[1] >= lby[0])
	{
        nti = (lby[1] - lby[0] + 1) * (*mw2_pml) + lby[1];
        nth = (lby[1] - lby[0] + 1) * (*mw2_pml) + 1 - lby[0];
		cudaMemcpy(t2xx_pyD, t2xx_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t2xy_pyD, t2xy_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(t2yy_pyD, t2yy_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t2yz_pyD, t2yz_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xx_pyD, qt2xx_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xy_pyD, qt2xy_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2yy_pyD, qt2yy_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2yz_pyD, qt2yz_py, sizeof(double) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
	}

	cudaMemcpy(t2xx_pzD, t2xx_pz, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(t2xz_pzD, t2xz_pz, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(t2yz_pzD, t2yz_pz, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(t2zz_pzD, t2zz_pz, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xx_pzD, qt2xx_pz, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xz_pzD, qt2xz_pz, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2yz_pzD, qt2yz_pz, sizeof(double) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2zz_pzD, qt2zz_pz, sizeof(double) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);

	return;
}

void cpy_h2d_stressInputsC(double *v1x,
						   double *v1y,
						   double *v1z,
						   double *v2x,
						   double *v2y,
						   double *v2z,
						   int	*nxtop,
						   int	*nytop,
						   int  *nztop,
						   int	*nxbtm,
						   int	*nybtm,
						   int	*nzbtm)
{
	cudaError_t cudaRes;

	//for inner_I
	cudaMemcpy(v1xD, v1x,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v1yD, v1y,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v1zD, v1z,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);

	//for inner_II
	cudaMemcpy(v2xD, v2x,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2yD, v2y,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2zD, v2z,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);

	return;
}

//=====================================================================
void cpy_h2d_velocityOutputsC(double *v1x,
							  double *v1y,
							  double *v1z,
							  double *v2x,
							  double *v2y,
							  double *v2z,
							  int	*nxtop,	
							  int	*nytop,
							  int   *nztop,
							  int	*nxbtm,
							  int	*nybtm,
							  int	*nzbtm)
{
	cudaError_t cudaRes;

	//for inner_I
	cudaRes = cudaMemcpy(v1xD, v1x,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1x");
	cudaRes = cudaMemcpy(v1yD, v1y,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1y");
	cudaRes = cudaMemcpy(v1zD, v1z,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1z");

	//for inner_II
	cudaRes = cudaMemcpy(v2xD, v2x,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x");
	cudaRes = cudaMemcpy(v2yD, v2y,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y");
	cudaRes = cudaMemcpy(v2zD, v2z,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z");
	
	return;
}

//=====================================================================
void cpy_d2h_velocityOutputsC(double *v1x, 
							  double *v1y,
							  double *v1z,
							  double *v2x,
							  double *v2y,
							  double *v2z,
							  int	*nxtop,
							  int	*nytop,
							  int   *nztop,
							  int	*nxbtm,
							  int	*nybtm,
							  int	*nzbtm)
{
	cudaError_t cudaRes;

	//for inner_I
	cudaRes = cudaMemcpy(v1x, v1xD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1x");
	cudaRes = cudaMemcpy(v1y, v1yD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1y");
	cudaRes = cudaMemcpy(v1z, v1zD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1z");

	//for inner_II
	cudaRes = cudaMemcpy(v2x, v2xD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2x");
	cudaRes = cudaMemcpy(v2y, v2yD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2y");
	cudaRes = cudaMemcpy(v2z, v2zD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, vzz");

	return;
}

void cpy_h2d_stressOutputsC(double *t1xx,
						    double *t1xy,
						    double *t1xz,
						    double *t1yy,
						    double *t1yz,
						    double *t1zz,
						    double *t2xx,
						    double *t2xy,
						    double *t2xz,
						    double *t2yy,
						    double *t2yz,
						    double *t2zz,
						    int	  *nxtop,
						    int	  *nytop,
						    int   *nztop,
						    int	  *nxbtm,
						    int	  *nybtm,
						    int	  *nzbtm)
{
	cudaError_t cudaRes;
	int nth, nti;

	cudaRes = cudaMemcpy(t1xxD, t1xx, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1xx");
	cudaRes = cudaMemcpy(t1xyD, t1xy, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1xy");
	cudaRes = cudaMemcpy(t1xzD, t1xz, sizeof(double) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1xz");
	cudaRes = cudaMemcpy(t1yyD, t1yy, sizeof(double) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1yy");
	cudaRes = cudaMemcpy(t1yzD, t1yz, sizeof(double) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1yz");
	cudaRes = cudaMemcpy(t1zzD, t1zz, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1zz");

	//for inner_II
	cudaRes = cudaMemcpy(t2xxD, t2xx, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2xx");
	cudaRes = cudaMemcpy(t2xyD, t2xy, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2xy");
	cudaRes = cudaMemcpy(t2xzD, t2xz, sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2xz");
	cudaRes = cudaMemcpy(t2yyD, t2yy, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2yy");
	cudaRes = cudaMemcpy(t2yzD, t2yz, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2yz");
	cudaRes = cudaMemcpy(t2zzD, t2zz, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2zz");

	return;
}

void cpy_d2h_stressOutputsC(double *t1xx,
						    double *t1xy,
						    double *t1xz,
						    double *t1yy,
						    double *t1yz,
						    double *t1zz,
						    double *t2xx,
						    double *t2xy,
						    double *t2xz,
						    double *t2yy,
						    double *t2yz,
						    double *t2zz,
						    int	  *nxtop,
						    int	  *nytop,
						    int   *nztop,
						    int	  *nxbtm,
						    int	  *nybtm,
						    int	  *nzbtm)
{
	cudaError_t cudaRes;

	cudaRes = cudaMemcpy(t1xx, t1xxD, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1xx");
	cudaRes = cudaMemcpy(t1xy, t1xyD, sizeof(double) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1xy");
	cudaRes = cudaMemcpy(t1xz, t1xzD, sizeof(double) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1xz");
	cudaRes = cudaMemcpy(t1yy, t1yyD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1yy");
	cudaRes = cudaMemcpy(t1yz, t1yzD, sizeof(double) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1yz");
	cudaRes = cudaMemcpy(t1zz, t1zzD, sizeof(double) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1zz");

	cudaRes = cudaMemcpy(t2xx, t2xxD, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2xx");
	cudaRes = cudaMemcpy(t2xy, t2xyD, sizeof(double) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2xy");
	cudaRes = cudaMemcpy(t2xz, t2xzD, sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2xz");
	cudaRes = cudaMemcpy(t2yy, t2yyD, sizeof(double) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2yy");
	cudaRes = cudaMemcpy(t2yz, t2yzD, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2yz");
	cudaRes = cudaMemcpy(t2zz, t2zzD, sizeof(double) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2zz");

	return;
}

void free_device_memC(int *lbx, int *lby)
{
	//-------------------------------------------------
        cudaFree(index_xyz_sourceD);
        cudaFree(fampD);
        cudaFree(ruptmD);
        cudaFree(risetD);
        cudaFree(sparam2D);

	cudaFree(nd1_velD);
	cudaFree(nd1_txyD);
	cudaFree(nd1_txzD);
	cudaFree(nd1_tyyD);
	cudaFree(nd1_tyzD);
	cudaFree(rhoD);
	cudaFree(drvh1D);
	cudaFree(drti1D);
	cudaFree(drth1D);
	cudaFree(idmat1D);
	cudaFree(dxi1D);
	cudaFree(dyi1D);
	cudaFree(dzi1D);
	cudaFree(dxh1D);
	cudaFree(dyh1D);
	cudaFree(dzh1D);
	cudaFree(t1xxD);
	cudaFree(t1xyD);
	cudaFree(t1xzD);
	cudaFree(t1yyD);
	cudaFree(t1yzD);
	cudaFree(t1zzD);
	cudaFree(v1xD);    //output
	cudaFree(v1yD);
	cudaFree(v1zD);

	if (lbx[1] >= lbx[0])
	{
		cudaFree(damp1_xD);
		cudaFree(t1xx_pxD);
		cudaFree(t1xy_pxD);
		cudaFree(t1xz_pxD);
		cudaFree(t1yy_pxD);
		cudaFree(qt1xx_pxD);
		cudaFree(qt1xy_pxD);
		cudaFree(qt1xz_pxD);
		cudaFree(qt1yy_pxD);
		cudaFree(v1x_pxD);
		cudaFree(v1y_pxD);
		cudaFree(v1z_pxD);
	}

	if (lby[1] >= lby[0])
	{
		cudaFree(damp1_yD);
		cudaFree(t1xx_pyD);
		cudaFree(t1xy_pyD);
		cudaFree(t1yy_pyD);
		cudaFree(t1yz_pyD);
		cudaFree(qt1xx_pyD);
		cudaFree(qt1xy_pyD);
		cudaFree(qt1yy_pyD);
		cudaFree(qt1yz_pyD);
		cudaFree(v1x_pyD);
		cudaFree(v1y_pyD);
		cudaFree(v1z_pyD);
	}

	cudaFree(qt1xxD);
	cudaFree(qt1xyD);
	cudaFree(qt1xzD);
	cudaFree(qt1yyD);
	cudaFree(qt1yzD);
	cudaFree(qt1zzD);

	cudaFree(clamdaD);
	cudaFree(cmuD);
	cudaFree(epdtD);
	cudaFree(qwpD);
	cudaFree(qwsD);
	cudaFree(qwt1D);
	cudaFree(qwt2D);
//-------------------------------------
	cudaFree(nd2_velD);
	cudaFree(nd2_txyD);
	cudaFree(nd2_txzD);
	cudaFree(nd2_tyyD);
	cudaFree(nd2_tyzD);

	cudaFree(drvh2D);
	cudaFree(drti2D);
	cudaFree(drth2D);
	cudaFree(idmat2D);
	cudaFree(damp2_zD);
	cudaFree(dxi2D);
	cudaFree(dyi2D);
	cudaFree(dzi2D);
	cudaFree(dxh2D);
	cudaFree(dyh2D);
	cudaFree(dzh2D);
	cudaFree(t2xxD);
	cudaFree(t2xyD);
	cudaFree(t2xzD);
	cudaFree(t2yyD);
	cudaFree(t2yzD);
	cudaFree(t2zzD);

	cudaFree(qt2xxD);
	cudaFree(qt2xyD);
	cudaFree(qt2xzD);
	cudaFree(qt2yyD);
	cudaFree(qt2yzD);
	cudaFree(qt2zzD);

	if (lbx[1] >= lbx[0])
	{
		cudaFree(damp2_xD);

		cudaFree(t2xx_pxD);
		cudaFree(t2xy_pxD);
		cudaFree(t2xz_pxD);
		cudaFree(t2yy_pxD);
		cudaFree(qt2xx_pxD);
		cudaFree(qt2xy_pxD);
		cudaFree(qt2xz_pxD);
		cudaFree(qt2yy_pxD);

		cudaFree(v2x_pxD);
		cudaFree(v2y_pxD);
		cudaFree(v2z_pxD);
	}

	if (lby[1] >= lby[0])
	{
		cudaFree(damp2_yD);

		cudaFree(t2xx_pyD);
		cudaFree(t2xy_pyD);
		cudaFree(t2yy_pyD);
		cudaFree(t2yz_pyD);

		cudaFree(qt2xx_pyD);
		cudaFree(qt2xy_pyD);
		cudaFree(qt2yy_pyD);
		cudaFree(qt2yz_pyD);

		cudaFree(v2x_pyD);
		cudaFree(v2y_pyD);
		cudaFree(v2z_pyD);
	}

	cudaFree(t2xx_pzD);
	cudaFree(t2xz_pzD);
	cudaFree(t2yz_pzD);
	cudaFree(t2zz_pzD);

	cudaFree(qt2xx_pzD);
	cudaFree(qt2xz_pzD);
	cudaFree(qt2yz_pzD);
	cudaFree(qt2zz_pzD);

	cudaFree(v2xD);		//output
	cudaFree(v2yD);
	cudaFree(v2zD);

	cudaFree(v2x_pzD);
	cudaFree(v2y_pzD);
	cudaFree(v2z_pzD);
	
	return;
}

/* 
    Following functions are responsible for data marshalling on GPU
    after velocity computation.
    If MPI-ACC is not used, the marshalled arrays are transferred
    back to CPU for MPI communication.
*/
void sdx51_vel (double** sdx51, int* nytop, int* nztop, int* nxtop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;
    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdx51<<< dimGrid, dimBlock >>> (sdx51D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdx51_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx51_vel", kernel_time );

#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx51, sdx51D, sizeof(double) * (*nztop)* (*nytop)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx51_vel", cpy_time );
#endif   
}
void sdx52_vel (double** sdx52, int* nybtm, int* nzbtm, int* nxbtm) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdx52<<< dimGrid, dimBlock >>> (sdx52D, v2xD, v2yD, v2zD,  *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdy52_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx52_vel", kernel_time );


#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx52, sdx52D, sizeof(double) * (*nzbtm)* (*nybtm)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx52_vel", cpy_time );
#endif    
}
void sdx41_vel (double** sdx41, int* nxtop, int* nytop, int* nztop, int* nxtm1) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdx41<<< dimGrid, dimBlock >>> (sdx41D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop, *nxtm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdx41_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx41_vel", kernel_time );

#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx41, sdx41D, sizeof(double) * (*nztop)* (*nytop)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx41_vel", cpy_time );
#endif    
}

void sdx42_vel (double** sdx42, int* nxbtm, int* nybtm, int* nzbtm, int* nxbm1){ 
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdx42<<< dimGrid, dimBlock >>> (sdx42D, v2xD, v2yD, v2zD, *nybtm, *nzbtm, *nxbtm, *nxbm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdx42_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx42_vel", kernel_time );

#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx42, sdx42D, sizeof(double) * (*nzbtm)* (*nybtm)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx42_vel", cpy_time );
#endif    
}

// --- sdy
void sdy51_vel (double** sdy51, int* nxtop, int* nytop, int* nztop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    struct timeval start, end;
    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdy51<<< dimGrid, dimBlock >>> (sdy51D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop);
    cudaRes=cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel failed sdy51_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy51_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy51D_meta) ;
#endif    
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy51, sdy51D, sizeof(double) * (*nztop)* (*nxtop)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy51_vel", cpy_time );
#endif    
}
void sdy52_vel (double** sdy52, int* nxbtm, int* nybtm, int* nzbtm) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdy52<<< dimGrid, dimBlock >>> (sdy52D, v2xD, v2yD, v2zD,  *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdy52_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy52_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy52D_meta); 
#endif    
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy52, sdy52D, sizeof(double)*(*nzbtm)* (*nxbtm)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy52_vel", cpy_time );
#endif    
}
void sdy41_vel (double** sdy41, int* nxtop, int* nytop, int* nztop, int* nytm1) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdy41<<< dimGrid, dimBlock >>> (sdy41D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop, *nytm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdy41_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy41_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy41D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy41, sdy41D, sizeof(double) * (*nztop)* (*nxtop)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy41_vel", cpy_time );
#endif    
}

void sdy42_vel (double** sdy42, int* nxbtm, int* nybtm, int* nzbtm, int* nybm1) 
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_sdy42<<< dimGrid, dimBlock >>> (sdy42D, v2xD, v2yD, v2zD, *nybtm, *nzbtm, *nxbtm, *nybm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdy42_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy42_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy42D_meta); 
#endif    
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy42, sdy42D, sizeof(double) * (*nzbtm)* (*nxbtm)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy42_vel", cpy_time );
#endif    
}
//--- sdy's
// -- RCX/Y

void rcx51_vel (double** rcx51, int* nxtop, int* nytop, int* nztop, int* nx1p1, int* nx1p2) {
     double tstart, tend;
    double cpy_time=0, kernel_time=0;

   struct timeval start, end;
    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx51D, *rcx51, sizeof(double) * (*nztop)* (*nytop)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx51_vel", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcx51<<< dimGrid, dimBlock >>> (rcx51D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop, *nx1p1, *nx1p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcx51_vel");
     record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx51_vel", kernel_time );
   
#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx51D_meta); 
#endif    
}

void rcx52_vel (double** rcx52, int* nxbtm, int* nybtm, int* nzbtm, int* nx2p1, int* nx2p2) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx52D, *rcx52, sizeof(double) * (*nzbtm)* (*nybtm)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevicE, rcx52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx52_vel", cpy_time );
#endif    
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcx52<<< dimGrid, dimBlock >>> (rcx52D, v2xD, v2yD, v2zD,  *nybtm, *nzbtm, *nxbtm, *nx2p1, *nx2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcx52_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx52_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx52D_meta); 
#endif    
}

void rcx41_vel (double** rcx41, int* nxtop, int* nytop, int* nztop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx41D, *rcx41, sizeof(double) * (*nztop)* (*nytop)* 4, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, rcx41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx41_vel", cpy_time );
#endif    
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcx41<<< dimGrid, dimBlock >>> (rcx41D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcx41_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx41_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx41D_meta); 
#endif
}

void rcx42_vel (double** rcx42, int* nxbtm, int* nybtm, int* nzbtm) { 
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx42D, *rcx42, sizeof(double) * (*nzbtm)* (*nybtm)* 4, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx42_vel", cpy_time );
#endif    
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcx42<<< dimGrid, dimBlock >>> (rcx42D, v2xD, v2yD, v2zD, *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcx42_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx42_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx42D_meta); 
#endif
}
// --- rcy
void rcy51_vel (double** rcy51, int* nxtop, int* nytop, int* nztop, int* ny1p1,int* ny1p2) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    struct timeval start, end;
    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy51D, *rcy51, sizeof(double) * (*nztop)* (*nxtop)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy51_vel", cpy_time );
#endif    
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcy51<<< dimGrid, dimBlock >>> (rcy51D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop, *ny1p1, *ny1p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcy51_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy51_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy51D_meta); 
#endif
}

void rcy52_vel (double** rcy52, int* nxbtm, int* nybtm, int* nzbtm, int* ny2p1, int* ny2p2) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy52D, *rcy52, sizeof(double) * (*nzbtm)* (*nxbtm)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy52_vel", cpy_time );
#endif    
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcy52<<< dimGrid, dimBlock >>> (rcy52D, v2xD, v2yD, v2zD,  *nybtm, *nzbtm, *nxbtm, *ny2p1, *ny2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcy52_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy52_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy52D_meta); 
#endif    
}
void rcy41_vel (double** rcy41, int* nxtop, int* nytop, int* nztop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy41D, *rcy41, sizeof(double) * (*nztop)* (*nxtop)* 4, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy41_vel", cpy_time );
#endif    
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcy41<<< dimGrid, dimBlock >>> (rcy41D, v1xD, v1yD, v1zD, *nytop, *nztop, *nxtop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcy41_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy41_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy41D_meta); 
#endif    
}
void rcy42_vel (double** rcy42, int* nxbtm, int* nybtm, int* nzbtm) { 
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*rcy42, rcy42D, sizeof(double) * (*nzbtm)* (*nxbtm)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy42_vel", cpy_time );
#endif    
    int blockSizeX = 32;        
    int blockSizeY = 32;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    vel_rcy42<<< dimGrid, dimBlock >>> (rcy42D, v2xD, v2yD, v2zD, *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcy42_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy42_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy42D_meta);
#endif    
}
// velocity_interp functions

// sdx/y/1/2
void sdx1_vel(double* sdx1, int* nxtop, int* nytop, int* nxbtm, int* nzbtm, int* ny2p1, int* ny2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;
    cudaError_t cudaRes;

    record_time(&tstart);
    vel_sdx1<<< 1, 1>>> (sdx1D, v2xD, v2yD, v2zD, *nxbtm, *nzbtm, *ny2p1, *ny2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdx1_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx41_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx1D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdx1, sdx1D, sizeof(double) * (*nytop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx1D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx1_vel", cpy_time );
#endif    
}

void sdy1_vel(double* sdy1, int* nxtop, int* nytop, int* nxbtm, int* nzbtm, int* nx2p1, int* nx2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;

    record_time(&tstart);
    vel_sdy1<<< 1, 1>>> (sdy1D, v2xD, v2yD, v2zD, *nxbtm, *nzbtm, *nx2p1, *nx2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdy1_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy1_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy1D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdy1, sdy1D, sizeof(double) * (*nxtop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy1D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy1_vel", cpy_time );
#endif    
}

void sdx2_vel(double* sdx2, int* nxtop, int* nytop, int* nxbm1, int* nxbtm, int* nzbtm, int* ny2p1, int* ny2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;

    record_time(&tstart);
    vel_sdx2<<< 1, 1>>> (sdx2D, v2xD, v2yD, v2zD, *nxbm1, *nxbtm, *nzbtm, *ny2p1, *ny2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdx2_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx2_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx2D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdx2, sdx2D, sizeof(double) * (*nytop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost1, sdx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx2_vel", cpy_time );
#endif    
}

void sdy2_vel(double* sdy2, int* nxtop, int* nytop, int* nybm1, int* nybtm, int* nxbtm, int* nzbtm, int* nx2p1, int* nx2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;

    record_time(&tstart);
    vel_sdy2<<< 1, 1>>> (sdy2D, v2xD, v2yD, v2zD, *nybm1, *nybtm, *nxbtm, *nzbtm, *nx2p1, *nx2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, sdy2_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy2_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy2D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdy2, sdy2D, sizeof(double) * (*nxtop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy2_vel", cpy_time );
#endif    
}

// rcx/y/1/2
void rcx1_vel(double* rcx1, int* nxtop, int* nytop, int* nxbtm, int* nzbtm, int* ny2p1, int* ny2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy( rcx1D, rcx1,  sizeof(double) * (*nytop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx1D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx1_vel", cpy_time );
#endif    
    record_time(&tstart);
    vel_rcx1<<< 1, 1>>> (rcx1D, v2xD, v2yD, v2zD, *nxbtm, *nzbtm, *ny2p1, *ny2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcx1_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx1_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx1D_meta); 
#endif
}

void rcy1_vel(double* rcy1, int* nxtop, int* nytop, int*nxbtm, int* nzbtm,  int* nx2p1, int* nx2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy1D, rcy1, sizeof(double) * (*nxtop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy1D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy1_vel", cpy_time );
#endif    

    record_time(&tstart);
    vel_rcy1<<< 1, 1>>> (rcy1D, v2xD, v2yD, v2zD, *nxbtm, *nzbtm, *nx2p1, *nx2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcy1_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy1_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy1D_meta); 
#endif
}

void rcx2_vel(double* rcx2, int* nxtop, int* nytop, int* nxbtm, int* nzbtm, int* nx2p1, int* nx2p2, int* ny2p1, int* ny2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx2D, rcx2, sizeof(double) * (*nytop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx2_vel", cpy_time );
#endif    

    record_time(&tstart);
    vel_rcx2<<< 1, 1>>> (rcx2D, v2xD, v2yD, v2zD, *nxbtm, *nzbtm, *nx2p1, *nx2p2, *ny2p1, *ny2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcx2_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx2_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx2D_meta); 
#endif
}

void rcy2_vel(double* rcy2, int* nxtop, int* nytop, int* nxbtm, int*  nzbtm, int* nx2p1, int* nx2p2, int* ny2p1, int* ny2p2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy2D, rcy2, sizeof(double) * (*nxtop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy2_vel", cpy_time );
#endif    

    record_time(&tstart);
    vel_rcy2<<< 1, 1>>> (rcy2D, v2xD, v2yD, v2zD, *nxbtm, *nzbtm,  *nx2p1, *nx2p2, *ny2p1, *ny2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "Kernel Launch, rcy2_vel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy2_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy2D_meta); 
#endif
}


// --- Vel_interpl_3vbtm
void interpl_3vbtm_vel1(int* ny1p2, int* ny2p2, int* nz1p1, int* nyvx, 
                int* nxbm1, int* nxbtm, int* nzbtm, int* nxtop, int* nztop,  int* neighb2, double* rcx2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nyvx + 1)/32)+1 , ((int)(*nxbm1/32)+1));
    cudaFuncSetCacheConfig(vel1_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel1_interpl_3vbtm<<<blocks,threadsPerBlock>>> (*ny1p2, *ny2p2, *nz1p1, *nyvx,
                                 *nxbm1, *nxbtm, *nzbtm, *nxtop, *nztop, *neighb2, 
                                  chxD, v1xD, v2xD, rcx2D);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel1_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel1_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
#endif
}

void interpl_3vbtm_vel3(int* ny1p2, int* nz1p1, int* nyvx1, int* nxbm1, 
                int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nxtop)/32)+1 , ((int)(*nyvx1/32)+1));
    cudaFuncSetCacheConfig(vel3_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel3_interpl_3vbtm<<<blocks,threadsPerBlock>>> ( *ny1p2, *nz1p1, *nyvx1, 
                *nxbm1, *nxbtm, *nzbtm, *nxtop, *nztop, 
                ciyD, v1xD);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel3_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel3_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
#endif
}

void interpl_3vbtm_vel4(int* nx1p2, int* ny2p2, int* nz1p1, int* nxvy, int* nybm1, 
                int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nxvy+1)/32)+1 , ((int)(*nybm1/32)+1));
    cudaFuncSetCacheConfig(vel4_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel4_interpl_3vbtm<<<blocks,threadsPerBlock>>> ( *nx1p2, *ny2p2, *nz1p1, *nxvy, 
                *nybm1, *nxbtm, *nybtm, *nzbtm, *nxtop, *nytop, *nztop, 
                chyD, v1yD, v2yD);

    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel4_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel4_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1yD_meta); 
#endif
}

void interpl_3vbtm_vel5(int* nx1p2, int* nx2p2, int* nz1p1, int* nxvy, int* nybm1, 
                int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(512);
    dim3 blocks(((int)(*nxvy+1)/512)+1);
    cudaFuncSetCacheConfig(vel5_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel5_interpl_3vbtm<<<blocks,threadsPerBlock>>> ( *nx1p2, *nx2p2, *nz1p1, *nxvy, 
                *nybm1, *nxbtm, *nybtm, *nzbtm, *nxtop, *nytop, *nztop, 
                chyD, v1yD, v2yD, rcy2D);

    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel5_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel5_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1yD_meta); 
#endif
}

void interpl_3vbtm_vel6(int* nx1p2,  int* nz1p1, int* nxvy1, int* nybm1, 
                int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nytop)/32)+1,((int)((*nxvy1)/32)+1));
    cudaFuncSetCacheConfig(vel6_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel6_interpl_3vbtm<<<blocks,threadsPerBlock>>>( *nx1p2, *nz1p1, *nxvy1, 
                *nybm1, *nxbtm, *nybtm, *nzbtm, *nxtop, *nytop,*nztop, 
                cixD, v1yD);

    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel6_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel6_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1yD_meta); 
#endif
}

void interpl_3vbtm_vel7 (int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop, double* sdx1)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(512);
    dim3 blocks(((int)(*nybtm)/512)+1);
    cudaFuncSetCacheConfig(vel7_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel7_interpl_3vbtm<<<blocks,threadsPerBlock>>>( *nxbtm, *nybtm, *nzbtm, *nxtop, *nytop, *nztop, 
                ciyD, sdx1D, rcx1D);

    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel7_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel7_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx1D_meta); 
#endif
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdx1, sdx1D, sizeof(double) * (*nytop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost3, sdx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel7_interpl_3vbtm", cpy_time );
#endif    

}
void interpl_3vbtm_vel8 (int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop, double* sdx2)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(512);
    dim3 blocks(((int)(*nybtm)/512)+1);
    cudaFuncSetCacheConfig(vel8_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel8_interpl_3vbtm<<<blocks,threadsPerBlock>>> (*nxbtm, *nybtm, *nzbtm, 
                                                *nxtop, *nytop, *nztop, 
                                                ciyD, sdx2D, rcx2D);

    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel8_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel8_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx2D_meta); 
#endif
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdx2, sdx2D, sizeof(double) * (*nytop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost2, sdx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel8_interpl_3vbtm", cpy_time );
#endif    

}

void interpl_3vbtm_vel9(int* nx1p2, int* ny2p1, int* nz1p1, int* nxvy, int* nybm1, 
                int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop, int* neighb4)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nxvy+1)/32)+1,((int)((*nybtm)/32)+1));
    cudaFuncSetCacheConfig(vel9_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel9_interpl_3vbtm<<<blocks,threadsPerBlock>>> ( *nz1p1, *nx1p2, *ny2p1, *nxvy, 
                *nxbtm, *nybtm, *nzbtm, *nxtop, *nytop, *nztop, 
                *neighb4, ciyD, rcy1D, rcy2D, v1zD, v2zD);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel9_interpl_3vbtm kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel9_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1zD_meta); 
#endif
}

void interpl_3vbtm_vel11(int* nx1p2, int* nx2p1, int* ny1p1, int* nz1p1, int* nxvy1, 
                int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*ny1p1+1)/32)+1,((int)((*nxvy1)/32)+1));
    cudaFuncSetCacheConfig(vel11_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel11_interpl_3vbtm<<<blocks,threadsPerBlock>>>  (*nx1p2, *nx2p1, *ny1p1, *nz1p1, *nxvy1,
		*nxbtm, *nybtm, *nzbtm, *nxtop, *nytop, *nztop, 
                cixD, sdx1D, sdx2D, v1zD);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel11_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel11_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1zD_meta); 
#endif
}

void interpl_3vbtm_vel13(int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nybtm)/32)+1,((int)((*nxbtm)/32)+1));
    cudaFuncSetCacheConfig(vel13_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel13_interpl_3vbtm<<<blocks,threadsPerBlock>>> (v1xD, v2xD,
				    *nxbtm, *nybtm, *nzbtm,
                                    *nxtop, *nytop, *nztop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel13_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel13_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v2xD_meta); 
#endif
}

void interpl_3vbtm_vel14(int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nybtm)/32)+1,((int)((*nxbtm)/32)+1));
    cudaFuncSetCacheConfig(vel14_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel14_interpl_3vbtm<<<blocks,threadsPerBlock>>> (v1yD, v2yD,
				    *nxbtm, *nybtm, *nzbtm,
                                    *nxtop, *nytop, *nztop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel14_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel14_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v2yD_meta); 
#endif
}

void interpl_3vbtm_vel15(int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*nybtm)/32)+1,((int)((*nxbtm)/32)+1));
    cudaFuncSetCacheConfig(vel15_interpl_3vbtm, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel15_interpl_3vbtm<<<blocks,threadsPerBlock>>> (v1zD, v2zD,
				    *nxbtm, *nybtm, *nzbtm,
                                    *nxtop, *nytop, *nztop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel15_interpl_kernel fails, interpl kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel15_interpl_3vbtm", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v2zD_meta); 
#endif
}

/*
void vxy_image_layer_vel1(double* v1x, double* v1y, double* v1z,
                            double* v2x, double* v2y, double* v2z,
                            int neighb1, int neighb2, int neighb3, int neighb4,
                            int* nxtm1, int* nytm1, int* nxtop, int* nytop, int* nztop,
                            int* nxbtm, int* nybtm ,int* nzbtm)
{
    
    vel_vxy_image_layer1<<<1,1>>>(v1xD, v1yD, v1zD, nd1_velD, dxi1D, dyi1D, dzh1D,
                                    *nxtm1, *nytm1, *nxtop, *nytop, *nztop, 
                                    neighb1, neighb2, neighb3, neighb4);
    cudaError_t cudaRes;
    cudaRes = cudaThreadSynchronize();
    CHECK_ERROR(cudaRes, "vxy_image_layer kernel error");
#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
    logger.logGPUArrInfo(v1yD_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    //for inner_I
    cudaRes = cudaMemcpy(v1x, v1xD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1x");
    cudaRes = cudaMemcpy(v1y, v1yD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1y");
    cudaRes = cudaMemcpy(v1z, v1zD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1z");

    //for inner_II
    cudaRes = cudaMemcpy(v2x, v2xD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2x");
    cudaRes = cudaMemcpy(v2y, v2yD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2y");
    cudaRes = cudaMemcpy(v2z, v2zD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, vzz");
#endif

}
*/
void vxy_image_layer_vel1(int* nd1_vel, int i, double dzdx, int nxbtm, int nybtm, int nzbtm, 
                            int nxtop, int nytop, int nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    dim3 threadsPerBlock(1024);
    dim3 blocks(((int)(nd1_vel[5]-nd1_vel[0]+1)/1024)+1);
    cudaFuncSetCacheConfig(vel1_vxy_image_layer, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel1_vxy_image_layer<<<blocks,threadsPerBlock>>> (v1xD, v1zD,
                                    nd1_velD, dxi1D, dzh1D, 
                                    i, dzdx,
    				    nxbtm, nybtm, nzbtm,
                                    nxtop, nytop, nztop);

    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel1_vxy_image_layer fails, kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel1_vxy_image_layer", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
#endif
}

void vxy_image_layer_vel2(int* nd1_vel, double* v1x, int* iix, double* dzdt, int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(nd1_vel[5]-nd1_vel[0]+1)/32)+1, 
                ((int)(*iix-nd1_vel[6]+1)/32)+1);
    cudaFuncSetCacheConfig(vel2_vxy_image_layer, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel2_vxy_image_layer<<<blocks,threadsPerBlock>>> (v1xD, v1zD, nd1_velD,
				dxi1D, dzh1D,
				*iix, *dzdt,
				*nxbtm, *nybtm, *nzbtm, 
				*nxtop, *nytop, *nztop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel2_vxy_image_layer fails, kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel2_vxy_image_layer", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(v1x, v1xD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1x");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel2_vxy_image_layer", cpy_time );
#endif
}

void vxy_image_layer_vel3(int* nd1_vel, int* j, double* dzdy, int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(1024);
    dim3 blocks(((int)(nd1_vel[11]-nd1_vel[6]+1)/1024)+1);
    cudaFuncSetCacheConfig(vel3_vxy_image_layer, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel3_vxy_image_layer<<<blocks,threadsPerBlock>>> (v1yD, v1zD,
                                    nd1_velD, dyi1D, dzh1D, 
                                    *j, *dzdy,
				    *nxbtm, *nybtm, *nzbtm,
                                    *nxtop, *nytop, *nztop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel3_vxy_image_layer fails, kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel3_vxy_image_layer", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1yD_meta); 
#endif
}

void vxy_image_layer_vel4(int* nd1_vel, double* v1y, int* jjy, double* dzdt, int* nxbtm, int* nybtm, int* nzbtm, int* nxtop, int* nytop, int* nztop)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int i=0;
    dim3 threadsPerBlock(32,32);
    dim3 blocks(((int)(*jjy - nd1_vel[0]+1)/32)+1, 
                ((int)(nd1_vel[11]-nd1_vel[6]+1)/32)+1);
    cudaFuncSetCacheConfig(vel4_vxy_image_layer, cudaFuncCachePreferL1);

    record_time(&tstart);
    vel4_vxy_image_layer<<<blocks,threadsPerBlock>>> (v1yD, v1zD, nd1_velD,
				dyi1D, dzh1D,
				*jjy, *dzdt,
				*nxbtm, *nybtm, *nzbtm, 
				*nxtop, *nytop, *nztop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vel4_vxy_image_layer fails, kernel");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel4_vxy_image_layer", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1yD_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(v1y, v1yD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1y");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel4_vxy_image_layer", cpy_time );
#endif
}



/*
//vxy_image_layer
void vxy_image_layer_vel1(double* v1x, double* v1y, double* v1z,
                            double* v2x, double* v2y, double* v2z,
                            int neighb1, int neighb2, int neighb3, int neighb4,
                            int* nxtm1, int* nytm1, int* nxtop, int* nytop, int* nztop,
                            int* nxbtm, int* nybtm ,int* nzbtm)
{
    vel_vxy_image_layer1<<<1,1>>>(v1xD, v1yD, v1zD, nd1_velD, dxi1D, dyi1D, dzh1D,
                                    *nxtm1, *nytm1, *nxtop, *nytop, *nztop, 
                                    neighb1, neighb2, neighb3, neighb4);
    cudaError_t cudaRes;
    cudaRes = cudaThreadSynchronize();
    CHECK_ERROR(cudaRes, "vxy_image_layer kernel error");
#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
    logger.logGPUArrInfo(v1yD_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    //for inner_I
    cudaRes = cudaMemcpy(v1x, v1xD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1x");
    cudaRes = cudaMemcpy(v1y, v1yD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1y");
    cudaRes = cudaMemcpy(v1z, v1zD,  sizeof(double) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1z");

    //for inner_II
    cudaRes = cudaMemcpy(v2x, v2xD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2x");
    cudaRes = cudaMemcpy(v2y, v2yD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2y");
    cudaRes = cudaMemcpy(v2z, v2zD,  sizeof(double) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, vzz");
#endif

}
*/
void vxy_image_layer_sdx_vel(double* sdx1, double* sdx2, int* nxtop, int* nytop, int* nztop ) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    dim3 threadsPerBlock(MaxThreadsPerBlock);
    dim3 numblocks((*nytop/MaxThreadsPerBlock)+1);

    record_time(&tstart);
    vel_vxy_image_layer_sdx<<<numblocks,threadsPerBlock>>>(sdx1D, sdx2D, v1xD, v1yD, *nxtop,
    *nytop, *nztop); 
    cudaRes = cudaThreadSynchronize();
    CHECK_ERROR(cudaRes, "vxy_image_layer_sdx kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel_vxy_image_layer_sdx", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx1D_meta); 
    logger.logGPUArrInfo(sdx2D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdx1, sdx1D, sizeof(double) * (*nytop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx1D");
    cudaRes = cudaMemcpy(sdx2, sdx2D, sizeof(double) * (*nytop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost4, sdx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel_vxy_image_layer_sdx", cpy_time );
#endif    
}

void vxy_image_layer_sdy_vel(double* sdy1, double* sdy2, int* nxtop, int* nytop,int* nztop ) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    dim3 threadsPerBlock(MaxThreadsPerBlock);
    dim3 numblocks((*nxtop/MaxThreadsPerBlock)+1);

    record_time(&tstart);
    vel_vxy_image_layer_sdy <<<numblocks,threadsPerBlock>>>(sdy1D, sdy2D, v1xD, v1yD, *nxtop,
    *nytop, *nztop); 
    cudaError_t cudaRes;
    cudaRes = cudaThreadSynchronize();
    CHECK_ERROR(cudaRes, "vxy_image_layer_sdy kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel_vxy_image_layer_sdy", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy1D_meta); 
    logger.logGPUArrInfo(sdy2D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(sdy1, sdy1D, sizeof(double) * (*nxtop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx1D");
    cudaRes = cudaMemcpy(sdy2, sdy2D, sizeof(double) * (*nxtop+6), cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost5, sdx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel_vxy_image_layer_sdy", cpy_time );
#endif    
}

void vxy_image_layer_rcx_vel(double* rcx1, double* rcx2, int* nxtop, int* nytop, int* nztop,  int* nx1p1) {

    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx1D, rcx1, sizeof(double) * (*nytop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyhostToDevice, rcx1D");
    cudaRes = cudaMemcpy( rcx2D, rcx2, sizeof(double) * (*nytop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, rcx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel_vxy_image_layer_rcx", cpy_time );
#endif   
    dim3 threadsPerBlock(MaxThreadsPerBlock);
    dim3 numblocks((*nytop/MaxThreadsPerBlock)+1);

    record_time(&tstart);
    vel_vxy_image_layer_rcx<<<numblocks,threadsPerBlock>>>(rcx1D, rcx2D, v1xD, v1yD, *nxtop,
    *nytop, *nztop, *nx1p1); 
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vxy_image_layer_rcx kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel_vxy_image_layer_rcx", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
    logger.logGPUArrInfo(v1yD_meta); 
#endif    
}

void vxy_image_layer_rcx2_vel(double* rcx1, double* rcx2, int* nxtop, int* nytop, int* nztop, int* ny1p1) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx1D, rcx1, sizeof(double) * (*nytop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx1D");
    cudaRes = cudaMemcpy(rcx2D, rcx2, sizeof(double) * (*nytop+6), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx2D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY vel_vxy_image_layer_rcx2", cpy_time );
#endif  

    dim3 threadsPerBlock(MaxThreadsPerBlock);
    dim3 numblocks((*nxtop/MaxThreadsPerBlock)+1);
    //printf("in vxy_image_layer : nxtop=%d, nytop=%d\n", nxtop, nytop);

    record_time(&tstart);
    vel_vxy_image_layer_rcy <<<numblocks,threadsPerBlock>>>(rcx1D, rcx2D, v1xD, v1yD, *nxtop, *nytop, *nztop, *ny1p1); 
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "vxy_image_layer_rcx2 kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL vel_vxy_image_layer_rcx2", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(v1xD_meta); 
    logger.logGPUArrInfo(v1yD_meta); 
#endif    
}

void add_dcs_vel(double* sutmArr, int nfadd, int ixsX, int ixsY, int ixsZ, 
    int fampX, int fampY, int ruptmX, int risetX, int sparam2X,  int nzrg11, int nzrg12, int nzrg13, int nzrg14, 
    int nxtop, int nytop, int nztop, int nxbtm, int nybtm, int nzbtm)
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;

    cudaFree(sutmArrD);
    cudaRes = cudaMalloc((void**)&sutmArrD, sizeof(double) * (nfadd));
    cudaRes = cudaMemcpy(sutmArrD, sutmArr, sizeof(double) * (nfadd), cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sutmArrD");
    
    dim3 threadsPerBlock(MaxThreadsPerBlock);
    dim3 numblocks((nfadd/MaxThreadsPerBlock)+1);
    record_time(&tstart);
    vel_add_dcs<<<numblocks, threadsPerBlock>>>( t1xxD, t1xyD, t1xzD, t1yyD, t1yzD, t1zzD, 
            t2xxD, t2yyD, t2xyD, t2xzD, t2yzD, t2zzD, 
            nfadd, index_xyz_sourceD,  ixsX, ixsY, ixsZ, 
            fampD, fampX, fampY, 
            ruptmD, ruptmX, risetD, risetX, sparam2D, sparam2X, sutmArrD,
            nzrg11, nzrg12, nzrg13, nzrg14, 
            nxtop, nytop, nztop,
            nxbtm, nybtm , nzbtm);
    cudaRes = cudaThreadSynchronize();
    CHECK_ERROR(cudaRes, "add_dcs_vel kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL add_dcs_vel", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(t1xxD_meta); 
    logger.logGPUArrInfo(t1yyD_meta); 
    logger.logGPUArrInfo(t1zzD_meta); 
    logger.logGPUArrInfo(t2xxD_meta); 
    logger.logGPUArrInfo(t2yyD_meta); 
    logger.logGPUArrInfo(t2zzD_meta); 
    logger.logGPUArrInfo(t1xzD_meta); 
    logger.logGPUArrInfo(t2xzD_meta); 
    logger.logGPUArrInfo(t1yzD_meta); 
    logger.logGPUArrInfo(t2yzD_meta); 
#endif
}

// STRESS  Computation:
void sdx41_stress (double** sdx41, int* nxtop, int* nytop, int* nztop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdx41<<< dimGrid, dimBlock >>> (sdx41D, t1xxD, t1xyD, t1xzD, *nytop, *nztop, *nxtop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdx41_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx41_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx41D_meta);
#endif    
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx41, sdx41D, sizeof(double) * (*nztop)* (*nytop)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx41_stress", cpy_time );
#endif    
}

void sdx42_stress (double** sdx42, int* nxbtm, int* nybtm, int* nzbtm) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdx42<<< dimGrid, dimBlock >>> (sdx42D, t2xxD, t2xyD, t2xzD, *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdx42_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx42_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx42D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx42, sdx42D, sizeof(double) * (*nzbtm)* (*nybtm)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx42_stress", cpy_time );
#endif    
}
void sdx51_stress (double** sdx51, int* nxtop, int* nytop, int* nztop, int* nxtm1) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdx51<<< dimGrid, dimBlock >>> (sdx51D, t1xxD, t1xyD, t1xzD, *nytop, *nztop, *nxtop, *nxtm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdx51_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx51_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx51D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx51, sdx51D, sizeof(double) * (*nztop)* (*nytop)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx51_stress", cpy_time );
#endif    
}

void sdx52_stress (double** sdx52, int* nxbtm, int* nybtm, int* nzbtm, int* nxbm1) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdx52<<< dimGrid, dimBlock >>> (sdx52D, t2xxD, t2xyD, t2xzD, *nybtm, *nzbtm, *nxbtm, *nxbm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdx52_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdx52_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdx52D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdx52, sdx52D, sizeof(double) * (*nzbtm)* (*nybtm)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdx52_stress", cpy_time );
#endif    
}
void sdy41_stress (double** sdy41, int* nxtop, int* nytop, int* nztop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdy41<<< dimGrid, dimBlock >>> (sdy41D, t1yyD, t1xyD, t1yzD, *nytop, *nztop, *nxtop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdy41_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy41_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy41D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy41, sdy41D, sizeof(double) * (*nztop)* (*nxtop)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy41_stress", cpy_time );
#endif    
}

void sdy42_stress (double** sdy42, int* nxbtm, int* nybtm, int* nzbtm) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdy42<<< dimGrid, dimBlock >>> (sdy42D, t2yyD, t2xyD, t2yzD, *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdy42_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy42_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy42D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy42, sdy42D, sizeof(double) * (*nzbtm)* (*nxbtm)* 4, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdx42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy42_stress", cpy_time );
#endif    
}
void sdy51_stress (double** sdy51, int* nxtop, int* nytop, int* nztop, int* nytm1) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdy51<<< dimGrid, dimBlock >>> (sdy51D, t1yyD, t1xyD, t1yzD, *nytop, *nztop, *nxtop, *nytm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdy51_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy51_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy51D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy51, sdy51D, sizeof(double) * (*nztop)* (*nxtop)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy51_stress", cpy_time );
#endif    
}

void sdy52_stress (double** sdy52, int* nxbtm, int* nybtm, int* nzbtm, int* nybm1) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_sdy52<<< dimGrid, dimBlock >>> (sdy52D, t2yyD, t2xyD, t2yzD, *nybtm, *nzbtm, *nxbtm, *nybm1);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "sdy52_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL sdy52_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(sdy52D_meta); 
#endif
#if USE_MPIX == 0 && VALIDATE_MODE == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(*sdy52, sdy52D, sizeof(double) * (*nzbtm)* (*nxbtm)* 5, cudaMemcpyDeviceToHost);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, sdy52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY sdy52_stress", cpy_time );
#endif    
}
//rc
void rcx41_stress (double** rcx41, int* nxtop, int* nytop, int* nztop, int* nx1p1, int* nx1p2) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx41D, *rcx41, sizeof(double) * (*nztop)* (*nytop)* 4, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx41_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcx41<<< dimGrid, dimBlock >>> (rcx41D, t1xxD, t1xyD, t1xzD, *nytop, *nztop, *nxtop, *nx1p1, *nx1p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcx41_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx41_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx41D_meta); 
#endif
}

void rcx42_stress (double** rcx42, int* nxbtm, int* nybtm, int* nzbtm, int* nx2p1, int* nx2p2) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx42D, *rcx42, sizeof(double) * (*nzbtm)* (*nybtm)* 4, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx42_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcx42<<< dimGrid, dimBlock >>> (rcx42D, t2xxD, t2xyD, t2xzD, *nybtm, *nzbtm, *nxbtm, *nx2p1, *nx2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcx42_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx42_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx42D_meta); 
#endif
}
void rcx51_stress (double** rcx51, int* nxtop, int* nytop, int* nztop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx51D, *rcx51, sizeof(double) * (*nztop)* (*nytop)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx51_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nytop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcx51<<< dimGrid, dimBlock >>> (rcx51D, t1xxD, t1xyD, t1xzD, *nytop, *nztop, *nxtop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcx51_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx51_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx51D_meta); 
#endif
}

void rcx52_stress (double** rcx52, int* nxbtm, int* nybtm, int* nzbtm) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcx52D, *rcx52, sizeof(double) * (*nzbtm)* (*nybtm)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, rcx52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx52_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nybtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcx52<<< dimGrid, dimBlock >>> (rcx52D, t2xxD, t2xyD, t2xzD, *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcx52_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx52_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx52D_meta); 
#endif
}

void rcy41_stress (double** rcy41, int* nxtop, int* nytop, int* nztop, int* ny1p1, int* ny1p2) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy41D, *rcy41, sizeof(double) * (*nztop)* (*nxtop)* 4, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy41D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy41_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcy41<<< dimGrid, dimBlock >>> (rcy41D, t1yyD, t1xyD, t1yzD, *nytop, *nztop, *nxtop, *ny1p1, *ny1p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcy41_stress kernel error");

    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy41_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy41D_meta); 
#endif
}

void rcy42_stress (double** rcy42, int* nxbtm, int* nybtm, int* nzbtm, int* ny2p1, int* ny2p2) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy42D, *rcy42, sizeof(double) * (*nzbtm)* (*nxbtm)* 4, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcx42D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy42_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcy42<<< dimGrid, dimBlock >>> (rcy42D, t2yyD, t2xyD, t2yzD, *nybtm, *nzbtm, *nxbtm, *ny2p1, *ny2p2);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcy42_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy42_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy41D_meta); 
#endif
}
void rcy51_stress (double** rcy51, int* nxtop, int* nytop, int* nztop) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy51D, *rcy51, sizeof(double) * (*nztop)* (*nxtop)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy51D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcy51_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nztop)/blockSizeX + 1;
    int gridY = (*nxtop)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcy51<<< dimGrid, dimBlock >>> (rcy51D, t1yyD, t1xyD, t1yzD, *nytop, *nztop, *nxtop);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcy51_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcy51_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcy41D_meta); 
#endif
}

void rcy52_stress (double** rcy52, int* nxbtm, int* nybtm, int* nzbtm) {
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;    
#if USE_MPIX == 0
    record_time(&tstart);
    cudaRes = cudaMemcpy(rcy52D, *rcy52, sizeof(double) * (*nzbtm)* (*nxbtm)* 5, cudaMemcpyHostToDevice);
    CHECK_ERROR(cudaRes, "InputDataCopyDeviceToHost, rcy52D");
    record_time(&tend);
    cpy_time = tend-tstart;
    logger.log_timing("CPY rcx52_stress", cpy_time );
#endif
    int blockSizeX = NTHREADS;        
    int blockSizeY = NTHREADS;        
    dim3 dimBlock(blockSizeX, blockSizeY);
    int gridX = (*nzbtm)/blockSizeX + 1;
    int gridY = (*nxbtm)/blockSizeY + 1;
    dim3 dimGrid(gridX, gridY);

    record_time(&tstart);
    stress_rcy52<<< dimGrid, dimBlock >>> (rcy52D, t2yyD, t2xyD, t2yzD, *nybtm, *nzbtm, *nxbtm);
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "rcx52_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL rcx52_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(rcx52D_meta); 
#endif    
}
void interp_stress (int neighb1, int neighb2, int neighb3, int neighb4, 
    int nxbm1, int nybm1, int nxbtm , int nybtm , int nzbtm, int nxtop, int nytop,  int nztop, int nz1p1) 
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    record_time(&tstart);
    stress_interp_stress<<<1,1>>> (t1xzD, t1yzD, t1zzD, t2xzD, t2yzD, t2zzD,
                                   neighb1, neighb2, neighb3, neighb4, 
                                   nxbm1, nybm1,
                                   nxbtm, nybtm, nzbtm,
                                   nxtop, nytop, nztop, nz1p1 );     
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "interp_stress kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL interp_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(t2xzD_meta); 
    logger.logGPUArrInfo(t2yzD_meta); 
    logger.logGPUArrInfo(t2zzD_meta); 
#endif    
}

void interp_stress1 ( int ntx1, int nz1p1, int nxbtm , int nybtm , int nzbtm, 
                    int nxtop, int nytop,  int nztop) 
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    dim3 threads(32,32);
    dim3 blocks ( ((int)nybtm/32)+1, ((int)ntx1/32)+1);
    record_time(&tstart);
    stress_interp1<<< blocks, threads>>>(ntx1, nz1p1, 
                                nxbtm, nybtm, nzbtm,
                                nxtop, nytop, nztop,
                                t1xzD, t2xzD); 
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "stress_inerp1 kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL interp1_stress", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(t2xzD_meta); 
#endif    
}

void interp_stress2 ( int nty1, int nz1p1, int nxbtm , int nybtm , int nzbtm, 
                    int nxtop, int nytop,  int nztop) 
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    dim3 threads(32,32);
    dim3 blocks ( ((int)nty1/32)+1, ((int)nxbtm/32)+1);
    record_time(&tstart);
    stress_interp2<<< blocks, threads>>>(nty1, nz1p1, 
                                nxbtm, nybtm, nzbtm,
                                nxtop, nytop, nztop,
                                t1yzD, t2yzD); 
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "stress_inerp2 kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL stress_interp2", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(t2yzD_meta); 
#endif    
}

void interp_stress3 ( int nxbtm , int nybtm , int nzbtm, 
                    int nxtop, int nytop,  int nztop) 
{
    double tstart, tend;
    double cpy_time=0, kernel_time=0;

    cudaError_t cudaRes;
    dim3 threads(32,32);
    dim3 blocks ( ((int)nybtm/32)+1, ((int)nxbtm/32)+1);
    record_time(&tstart);
    stress_interp3<<< blocks, threads>>>(nxbtm, nybtm, nzbtm,
                                nxtop, nytop, nztop,
                                t1zzD, t2zzD); 
    cudaRes = cudaDeviceSynchronize();
    CHECK_ERROR(cudaRes, "stress_inerp3 kernel error");
    record_time(&tend);
    kernel_time = tend-tstart;
    logger.log_timing("KERNEL stress_interp3", kernel_time );

#if LOGGING_ENABLED == 1    
    logger.logGPUArrInfo(t2zzD_meta); 
#endif    
}


// -- </ MPI-ACC >
void compute_velocityCDummy(int* nxtop, int* nztop, int* nytop, int* nxbtm, int* nzbtm, int* nybtm,
                            double** v1xMh, double** v2xMh, double** v1yMh, double** v2yMh,
                            double** v1zMh, double** v2zMh)
{

    cpy_h2d_velocityOutputsC(*v1xMh, *v1yMh, *v1zMh, *v2xMh, *v2yMh, *v2zMh, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
    int blockDimX = 32;
    int blockDimY = 32;
    dim3 dimBlock(blockDimX, blockDimY);
    int gridX = ((*nztop+2)/blockDimX) +1;
    int gridY = ((*nxtop+3)/blockDimY) +1;
    dim3 dimGrid(gridX, gridY);
    vel1_dummy <<<dimGrid, dimBlock >>> (*nxtop, *nztop, *nytop, v1xD, v1yD, v1zD);

    gridX = ((*nzbtm+1)/blockDimX) +1;
    gridY = ((*nxbtm+3)/blockDimY) +1;
    dim3 dimGrid2(gridX, gridY);
    vel2_dummy <<<dimGrid2, dimBlock >>> (*nxbtm, *nzbtm, *nybtm, v2xD, v2yD, v2zD);
	
    cpy_d2h_velocityOutputsC(*v1xMh, *v1yMh, *v1zMh, *v2xMh, *v2yMh, *v2zMh, nxtop,	nytop, nztop, nxbtm, nybtm, nzbtm);
}


void compute_velocityC(int *nztop, int *nztm1, double *ca, int *lbx,
			 int *lby, int *nd1_vel, double *rhoM, double *drvh1M, double *drti1M,
             double *damp1_xM, double *damp1_yM, int *idmat1M, double *dxi1M, double *dyi1M,
             double *dzi1M, double *dxh1M, double *dyh1M, double *dzh1M, double *t1xxM,
             double *t1xyM, double *t1xzM, double *t1yyM, double *t1yzM, double *t1zzM,
             void **v1xMp, void **v1yMp, void **v1zMp, double *v1x_pxM, double *v1y_pxM,
             double *v1z_pxM, double *v1x_pyM, double *v1y_pyM, double *v1z_pyM, 
             int *nzbm1, int *nd2_vel, double *drvh2M, double *drti2M, 
             int *idmat2M, double *damp2_xM, double *damp2_yM, double *damp2_zM,
             double *dxi2M, double *dyi2M, double *dzi2M, double *dxh2M, double *dyh2M,
             double *dzh2M, double *t2xxM, double *t2xyM, double *t2xzM, double *t2yyM,
             double *t2yzM, double *t2zzM, void **v2xMp, void **v2yMp, void **v2zMp,
             double *v2x_pxM, double *v2y_pxM, double *v2z_pxM, double *v2x_pyM, 
             double *v2y_pyM, double *v2z_pyM, double *v2x_pzM, double *v2y_pzM,
             double *v2z_pzM, int *nmat,	int *mw1_pml1, int *mw2_pml1, 
             int *nxtop, int *nytop, int *mw1_pml, int *mw2_pml,
             int *nxbtm, int *nybtm, int *nzbtm, int *myid)
{


    double tstart, tend;

    cudaError_t cudaRes;
	//define the dimensions of different kernels
	int blockSizeX = 32;
	int blockSizeY = 32;

  double *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;

  // extract specific input/output pointers
  v1xM=(double *) *v1xMp;
  v1yM=(double *) *v1yMp;
  v1zM=(double *) *v1zMp;
  v2xM=(double *) *v2xMp;
  v2yM=(double *) *v2yMp;
  v2zM=(double *) *v2zMp;

  procID = *myid;
 
  /*record_time(&tstart);
  cpy_h2d_velocityInputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, 
  			  t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);

  cpy_h2d_velocityOutputsC(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
  record_time(&tend);
  logger.log_timing("Velocity H2D Transfer", tend - tstart);
  */	
  // debug
  //print_arrayC(nd1_vel, 18, "nd1_vel");
  //print_arrayC(lbx, 2, "lbx");
  //print_arrayC(lby, 2, "lby");
  //print_arrayC(nd2_vel, 18, "nd2_vel");

  record_time(&tstart);

	dim3 dimBlock(blockSizeX, blockSizeY);

#if USE_Optimized_velocity_inner_IC == 1
	int gridSizeX1 = (*nztm1 + 1)/blockSizeX + 1;
	int gridSizeY1 = (nd1_vel[9] - nd1_vel[8])/blockSizeY + 1;
	dim3 dimGrid1(gridSizeX1, gridSizeY1);
#elif USE_Optimized_velocity_inner_IC == 0
	int gridSizeX1 = (nd1_vel[3] - nd1_vel[2])/blockSizeX + 1;
	int gridSizeY1 = (nd1_vel[9] - nd1_vel[8])/blockSizeY + 1;
	dim3 dimGrid1(gridSizeX1, gridSizeY1);
#endif 

#if USE_Optimized_velocity_inner_IC ==1 || USE_Optimized_velocity_inner_IC ==0
	velocity_inner_IC<<<dimGrid1, dimBlock>>>(*nztop,
					 *nztm1,
					 *ca,
					 nd1_velD,
					 rhoD,
					 idmat1D,
					 dxi1D,
					 dyi1D,
					 dzi1D,
					 dxh1D,
					 dyh1D,
					 dzh1D,
					 t1xxD,
					 t1xyD,
					 t1xzD,
					 t1yyD,
					 t1yzD,
					 t1zzD,
					 *nxtop,	//dimension #
					 *nytop,
					 v1xD,		//output
					 v1yD,
					 v1zD);

          cudaRes = cudaThreadSynchronize();
          CHECK_ERROR(cudaRes, "Velocity_  inner ic kernel computation error");
#endif

# if USE_Optimized_vel_PmlX_IC ==1
	int gridSizeX2 = (*nztop - 1)/blockSizeX + 1;
	int gridSizeY2 = (nd1_vel[5] - nd1_vel[0])/blockSizeX + 1;
# elif USE_Optimized_vel_PmlX_IC ==0
	int gridSizeX2 = (nd1_vel[5] - nd1_vel[0])/blockSizeX + 1;
	int gridSizeY2 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
#if USE_Optimized_vel_PmlX_IC ==1 || USE_Optimized_vel_PmlX_IC == 0        
	dim3 dimGrid2(gridSizeX2, gridSizeY2);
	if (lbx[1] >= lbx[0])
	{
		vel_PmlX_IC<<<dimGrid2, dimBlock>>>(*ca,
			   lbx[0],
			   lbx[1],
			   nd1_velD,
			   rhoD,
			   drvh1D,
			   drti1D,
			   damp1_xD,
			   idmat1D,
			   dxi1D,
			   dyi1D,
			   dzi1D,
			   dxh1D,
			   dyh1D,
			   dzh1D,
			   t1xxD,
			   t1xyD,
			   t1xzD,
			   t1yyD,
			   t1yzD,
			   t1zzD,
			   *mw1_pml1,	//dimension #
			   *mw1_pml,
			   *nxtop,
			   *nytop,
			   *nztop,
			   v1xD,		//output
			   v1yD,
			   v1zD,
			   v1x_pxD,
			   v1y_pxD,
			   v1z_pxD);
          cudaRes = cudaDeviceSynchronize();
          CHECK_ERROR(cudaRes, "Velocity_ pmlx ic kernel computation error");


	}
#endif        

#if USE_Optimized_vel_PmlY_IC == 1
	int gridSizeX3 = (*nztop-1)/blockSizeX + 1;
	int gridSizeY3 = (nd1_vel[11] - nd1_vel[6])/blockSizeY + 1;

#elif USE_Optimized_vel_PmlY_IC == 0
	int gridSizeX3 = (lby[1] - lby[0])/blockSizeX + 1;
	int gridSizeY3 = (nd1_vel[11] - nd1_vel[6])/blockSizeY + 1;
#endif
        
#if USE_Optimized_vel_PmlY_IC == 1 || USE_Optimized_vel_PmlY_IC == 0
	dim3 dimGrid3(gridSizeX3, gridSizeY3);
	
	if (lby[1] >= lby[0])
	{
		vel_PmlY_IC<<<dimGrid3, dimBlock>>>(*nztop,
			   *ca,
			   lby[0],
			   lby[1],
			   nd1_velD,
			   rhoD,
			   drvh1D,
			   drti1D,
			   idmat1D,
			   damp1_yD,
			   dxi1D,
			   dyi1D,
			   dzi1D,
			   dxh1D,
			   dyh1D,
			   dzh1D,
			   t1xxD,
			   t1xyD,
			   t1xzD,
			   t1yyD,
			   t1yzD,
			   t1zzD,
			   *mw1_pml1,	//dimension #s
			   *mw1_pml,
			   *nxtop,
			   *nytop,
			   v1xD,		//output
			   v1yD,
			   v1zD,
			   v1x_pyD,
			   v1y_pyD,
			   v1z_pyD);
          cudaRes = cudaDeviceSynchronize();
          CHECK_ERROR(cudaRes, "Velocity_ pmly ic kernel computation error");
	}
#endif

#if USE_Optimized_velocity_inner_IIC == 1
        int gridSizeX4 = (nd2_vel[15] - 2)/blockSizeX + 1;
	int gridSizeY4 = (nd2_vel[9] - nd2_vel[8])/blockSizeY + 1;	
#elif USE_Optimized_velocity_inner_IIC == 0
        int gridSizeX4 = (nd2_vel[3] - nd2_vel[2])/blockSizeX + 1;
	int gridSizeY4 = (nd2_vel[9] - nd2_vel[8])/blockSizeY + 1;
#endif

#if USE_Optimized_velocity_inner_IIC ==1 || USE_Optimized_velocity_inner_IIC==0

	dim3 dimGrid4(gridSizeX4, gridSizeY4);

	velocity_inner_IIC<<<dimGrid4, dimBlock>>>(*ca,
					  nd2_velD,
					  rhoD,
					  dxi2D,
					  dyi2D,
					  dzi2D,
					  dxh2D,
					  dyh2D,
					  dzh2D,
					  idmat2D,
					  t2xxD,
					  t2xyD,
					  t2xzD,
					  t2yyD,
					  t2yzD,
					  t2zzD,
					  *nxbtm,
					  *nybtm,
					  *nzbtm,
					  v2xD,		//output
					  v2yD,
					  v2zD);
        cudaRes = cudaDeviceSynchronize();
        CHECK_ERROR(cudaRes, "Velocity_inner_IIC kernel computation error");
#endif

#if USE_Optimized_vel_PmlX_IIC == 1
	int gridSizeX5 = (*nzbm1 - 1)/blockSizeX + 1;
        int gridSizeY5 = (nd2_vel[5] - nd2_vel[0])/blockSizeY + 1;
#elif USE_Optimized_vel_PmlX_IIC == 0
	int gridSizeX5 = (nd2_vel[5] - nd2_vel[0])/blockSizeX + 1;
	int gridSizeY5 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
#if USE_Optimized_vel_PmlX_IIC ==1 || USE_Optimized_vel_PmlX_IIC ==0        
	dim3 dimGrid5(gridSizeX5, gridSizeY5);
	
	if (lbx[1] >= lbx[0])
	{
		vel_PmlX_IIC<<<dimGrid5, dimBlock>>>(*nzbm1,
				*ca,
				lbx[0],
				lbx[1],
				nd2_velD,
				drvh2D,
				drti2D,
				rhoD,
				damp2_xD,
				idmat2D,
				dxi2D,
				dyi2D,
				dzi2D,
				dxh2D,
				dyh2D,
				dzh2D,
				t2xxD,
				t2xyD,
				t2xzD,
				t2yyD,
				t2yzD,
				t2zzD,
				*mw2_pml1,	//dimension #s
				*mw2_pml,
				*nxbtm,
				*nybtm,
				*nzbtm,
				v2xD,	//output
				v2yD,
				v2zD,
				v2x_pxD,
				v2y_pxD,
				v2z_pxD);
          cudaRes = cudaDeviceSynchronize();
          CHECK_ERROR(cudaRes, "Vel_PmlX_IIC kernel computation error");

	}
#endif

#if USE_Optimized_vel_PmlY_IIC == 1
	int gridSizeX6 = (*nzbm1 -1)/blockSizeX + 1;
	int gridSizeY6 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
#elif USE_Optimized_vel_PmlY_IIC == 0
	int gridSizeX6 = (lby[1] - lby[0])/blockSizeX + 1;
	int gridSizeY6 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
#endif

#if USE_Optimized_vel_PmlY_IIC == 0 || USE_Optimized_vel_PmlY_IIC ==1
	dim3 dimGrid6(gridSizeX6, gridSizeY6);

	if (lby[1] >= lby[0])
	{
		vel_PmlY_IIC<<<dimGrid6, dimBlock>>>(*nzbm1,
				*ca,
				lby[0],
				lby[1],
				nd2_velD,
				drvh2D,
				drti2D,
				rhoD,
				damp2_yD,
				idmat2D,
				dxi2D,
				dyi2D,
				dzi2D,
				dxh2D,
				dyh2D,
				dzh2D,
				t2xxD,
				t2xyD,
				t2xzD,
				t2yyD,
				t2yzD,
				t2zzD,
				*mw2_pml1,	//dimension #s
				*mw2_pml,
				*nxbtm,
				*nybtm,
				*nzbtm,
				v2xD,		//output
				v2yD,
				v2zD,
				v2x_pyD,
				v2y_pyD,
				v2z_pyD);
          cudaRes = cudaDeviceSynchronize();
          CHECK_ERROR(cudaRes, "Vel_PmlY_IIC kernel computation error");
	}
#endif        

#if USE_Optimized_vel_PmlZ_IIC == 1
	int gridSizeX7 = (*nzbm1 - 1)/blockSizeX + 1;
	int gridSizeY7 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;

#elif USE_Optimized_vel_PmlZ_IIC == 0
	int gridSizeX7 = (nd2_vel[5] - nd2_vel[0])/blockSizeX + 1;
	int gridSizeY7 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
#endif

#if USE_Optimized_vel_PmlZ_IIC == 0 || USE_Optimized_vel_PmlZ_IIC == 1
	dim3 dimGrid7(gridSizeX7, gridSizeY7);

	vel_PmlZ_IIC<<<dimGrid7, dimBlock>>>(*nzbm1,
				*ca,
				nd2_velD,
				drvh2D,
				drti2D,
				rhoD,
				damp2_zD,
				idmat2D,
				dxi2D,
				dyi2D,
				dzi2D,
				dxh2D,
				dyh2D,
				dzh2D,
				t2xxD,
				t2xyD,
				t2xzD,
				t2yyD,
				t2yzD,
				t2zzD,
				*mw2_pml1,	//dimension #s
				*mw2_pml,
				*nxbtm,
				*nybtm,
				*nzbtm,
				v2xD,		//output
				v2yD,
				v2zD,
				v2x_pzD,
				v2y_pzD,
				v2z_pzD);

#endif
  cudaRes = cudaDeviceSynchronize();
  CHECK_ERROR(cudaRes, "Velocity kernel computation error");
  record_time(&tend);
  logger.log_timing("Velocity Computation on GPU", tend-tstart );

#if COMPUTATION_CORRECTNESS_MODE == 1 && VALIDATE_MODE == 1
	record_time(&tstart);
	compute_velocityCDebug( nztop,  nztm1,  ca, lbx,
			lby, nd1_vel, rhoM, drvh1M, drti1M,
			 damp1_xM, damp1_yM, idmat1M, dxi1M,  dyi1M,
			 dzi1M, dxh1M, dyh1M, dzh1M, t1xxM,
			 t1xyM, t1xzM, t1yyM, t1yzM, t1zzM,
			 v1xMp,  v1yMp,  v1zMp,  v1x_pxM,  v1y_pxM,
			 v1z_pxM, v1x_pyM,  v1y_pyM,   v1z_pyM,
			 nzbm1,   nd2_vel,  drvh2M,  drti2M,
			  idmat2M,damp2_xM,  damp2_yM,  damp2_zM,
			  dxi2M,  dyi2M,  dzi2M,  dxh2M,  dyh2M,
			  dzh2M,  t2xxM,  t2xyM,  t2xzM,  t2yyM,
			  t2yzM,  t2zzM,  v2xMp,  v2yMp,  v2zMp,
			  v2x_pxM,  v2y_pxM,  v2z_pxM,  v2x_pyM,
			  v2y_pyM,  v2z_pyM,  v2x_pzM,  v2y_pzM,
			  v2z_pzM,   nmat,  mw1_pml1,   mw2_pml1,
			   nxtop,   nytop,   mw1_pml,   mw2_pml,
			   nxbtm,   nybtm,   nzbtm);
	record_time(&tend);
	logger.log_timing("Velocity Computation on CPU", tend- tstart);
	validate_array(v1xM, v1xD_meta);	
	validate_array(v1yM, v1yD_meta);	
	validate_array(v1zM, v1zD_meta);	
	validate_array(v2xM, v2xD_meta);	
	validate_array(v2yM, v2yD_meta);	
	validate_array(v2zM, v2zD_meta);	
/* int *nztop,  int *nztm1,  double *ca, int *lbx,
	 int *lby, int *nd1_vel, double *rhoM, double *drvh1M, double *drti1M,
	 double *damp1_xM, double *damp1_yM, int *idmat1M,double *dxi1M, double *dyi1M,
	 double *dzi1M, double *dxh1M, double *dyh1M, double *dzh1M, double *t1xxM,
	 double *t1xyM, double *t1xzM, double *t1yyM, double *t1yzM, double *t1zzM,
	 void **v1xMp, void **v1yMp, void **v1zMp, double *v1x_pxM, double *v1y_pxM,
	 double *v1z_pxM, double *v1x_pyM, double *v1y_pyM,  double *v1z_pyM,
	  int *nzbm1,  int *nd2_vel, double *drvh2M, double *drti2M,
	 int *idmat2M, double *damp2_xM, double *damp2_yM, double *damp2_zM,
	 double *dxi2M, double *dyi2M, double *dzi2M, double *dxh2M, double *dyh2M,
	 double *dzh2M, double *t2xxM, double *t2xyM, double *t2xzM, double *t2yyM,
	 double *t2yzM, double *t2zzM, void **v2xMp, void **v2yMp, void **v2zMp,
	 double *v2x_pxM, double *v2y_pxM, double *v2z_pxM, double *v2x_pyM,
	 double *v2y_pyM, double *v2z_pyM, double *v2x_pzM, double *v2y_pzM,
	 double *v2z_pzM,  int *nmat, int *mw1_pml1,  int *mw2_pml1,
	  int *nxtop,  int *nytop,  int *mw1_pml,  int *mw2_pml,
	  int *nxbtm,  int *nybtm,  int *nzbtm);
*/

#endif

#if VALIDATE_MODE == 1
  record_time(&tstart);
  cpy_d2h_velocityOutputsC(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, 
                                    nxtop,nytop, nztop, nxbtm, nybtm, nzbtm);
  record_time(&tend);
  logger.log_timing(" Data Transfer POST Velocity Computation", tend-tstart );

#endif        
  return;
}

void compute_stressC(int *nxb1, int *nyb1, int *nx1p1, int *ny1p1, int *nxtop, int *nytop, int *nztop, int *mw1_pml,
		int *mw1_pml1, int *nmat, int *nll, int *lbx, int *lby, int *nd1_txy, int *nd1_txz,
		int *nd1_tyy, int *nd1_tyz, int *idmat1M, double *ca, double *drti1M, double *drth1M, double *damp1_xM, double *damp1_yM,
		double *clamdaM, double *cmuM, double *epdtM, double *qwpM, double *qwsM, double *qwt1M, double *qwt2M, double *dxh1M,
		double *dyh1M, double *dzh1M, double *dxi1M, double *dyi1M, double *dzi1M, double *t1xxM, double *t1xyM, double *t1xzM, 
		double *t1yyM, double *t1yzM, double *t1zzM, double *qt1xxM, double *qt1xyM, double *qt1xzM, double *qt1yyM, double *qt1yzM, 
		double *qt1zzM, double *t1xx_pxM, double *t1xy_pxM, double *t1xz_pxM, double *t1yy_pxM, double *qt1xx_pxM, double *qt1xy_pxM,
		double *qt1xz_pxM, double *qt1yy_pxM, double *t1xx_pyM, double *t1xy_pyM, double *t1yy_pyM, double *t1yz_pyM, double *qt1xx_pyM,
		double *qt1xy_pyM, double *qt1yy_pyM, double *qt1yz_pyM, void **v1xMp, void **v1yMp, void **v1zMp,
		int *nxb2, int *nyb2, int *nxbtm, int *nybtm, int *nzbtm, int *mw2_pml, int *mw2_pml1, int *nd2_txy, int *nd2_txz, 
		int *nd2_tyy, int *nd2_tyz, int *idmat2M, 
		double *drti2M, double *drth2M, double *damp2_xM, double *damp2_yM, double *damp2_zM, 
		double *t2xxM, double *t2xyM, double *t2xzM, double *t2yyM, double *t2yzM, double *t2zzM, 
		double *qt2xxM, double *qt2xyM, double *qt2xzM, double *qt2yyM, double *qt2yzM, double *qt2zzM, 
		double *dxh2M, double *dyh2M, double *dzh2M, double *dxi2M, double *dyi2M, double *dzi2M, 
		double *t2xx_pxM, double *t2xy_pxM, double *t2xz_pxM, double *t2yy_pxM, double *t2xx_pyM, double *t2xy_pyM,
		double *t2yy_pyM, double *t2yz_pyM, double *t2xx_pzM, double *t2xz_pzM, double *t2yz_pzM, double *t2zz_pzM,
		double *qt2xx_pxM, double *qt2xy_pxM, double *qt2xz_pxM, double *qt2yy_pxM, double *qt2xx_pyM, double *qt2xy_pyM, 
		double *qt2yy_pyM, double *qt2yz_pyM, double *qt2xx_pzM, double *qt2xz_pzM, double *qt2yz_pzM, double *qt2zz_pzM,
		void **v2xMp, void **v2yMp, void **v2zMp, int *myid)
{
        double tstart, tend;	
	double *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;
	int blockSizeX = 32;
	int blockSizeY = 32;
	dim3 dimBlock(blockSizeX, blockSizeY);
	cudaError_t cudaRes;

	v1xM = (double *) *v1xMp;
	v1yM = (double *) *v1yMp;
	v1zM = (double *) *v1zMp;
	v2xM = (double *) *v2xMp;
	v2yM = (double *) *v2yMp;
	v2zM = (double *) *v2zMp;

/*
	record_time(&tstart);
	cpy_h2d_stressInputsC(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);

	cpy_h2d_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, 
			t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	record_time(&tend);
	logger.log_timing("Stress Computation H2D Transfer", tend-tstart );
*/

/* debug
  print_arrayC(nd1_tyy, 18, "nd1_tyy");
  print_arrayC(nd1_tyz, 18, "nd1_tyz");
  print_arrayC(lbx, 2, "lbx");
  print_arrayC(lby, 2, "lby");
  print_arrayC(nd1_txy, 18, "nd1_txy");
  print_arrayC(nd1_txz, 18, "nd1_txz");
  print_arrayC(nd2_txy, 18, "nd2_txy");
  print_arrayC(nd2_txz, 18, "nd2_txz");
*/	

    record_time(&tstart);

#if USE_Optimized_stress_norm_xy_IC == 1
    int gridSizeX1 = (nd1_tyy[17] - nd1_tyy[12])/blockSizeX + 1;
    int gridSizeY1 = (nd1_tyy[9] - nd1_tyy[8])/blockSizeY + 1;
#elif USE_Optimized_stress_norm_xy_IC == 0
    int gridSizeX1 = (nd1_tyy[3] - nd1_tyy[2])/blockSizeX + 1;
    int gridSizeY1 = (nd1_tyy[9] - nd1_tyy[8])/blockSizeY + 1;
#endif    

#if USE_Optimized_stress_norm_xy_IC == 0 || USE_Optimized_stress_norm_xy_IC == 1
	dim3 dimGrid1(gridSizeX1, gridSizeY1);

	stress_norm_xy_IC<<<dimGrid1, dimBlock>>>(*nxb1,
					  *nyb1,
					  *nxtop,
					  *nztop,
					  nd1_tyyD,
					  idmat1D,
					  *ca,
					  clamdaD,
					  cmuD,
					  epdtD,
					  qwpD,
					  qwsD,
					  qwt1D,
					  qwt2D,
					  dxh1D,
					  dyh1D,
					  dxi1D,
					  dyi1D,
					  dzi1D,
					  t1xxD,
					  t1xyD,
					  t1yyD,
					  t1zzD,
					  qt1xxD,
					  qt1xyD,
					  qt1yyD,
					  qt1zzD,
					  v1xD,
					  v1yD,
					  v1zD);
#endif        

#if USE_Optimized_stress_xz_yz_IC ==1
    int gridSizeX2 = (nd1_tyz[17] - nd1_tyz[12])/blockSizeX + 1;
    int gridSizeY2 = (nd1_tyz[9] - nd1_tyz[8])/blockSizeY + 1;
#elif USE_Optimized_stress_xz_yz_IC ==0
    int gridSizeX2 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeX + 1;
    int gridSizeY2 = (nd1_tyz[9] - nd1_tyz[8])/blockSizeY + 1;
#endif     

#if USE_Optimized_stress_xz_yz_IC == 1 || USE_Optimized_stress_xz_yz_IC ==0
        dim3 dimGrid2(gridSizeX2, gridSizeY2);

	stress_xz_yz_IC<<<dimGrid2, dimBlock>>>(*nxb1,
					  *nyb1,
					  *nxtop,
					  *nytop,
					  *nztop,
					  nd1_tyzD,
					  idmat1D,
					  *ca,
					  cmuD,
					  epdtD,
					  qwsD,
					  qwt1D,
					  qwt2D,
					  dxi1D,
					  dyi1D,
					  dzh1D,
					  v1xD,
					  v1yD,
					  v1zD,
					  t1xzD,
					  t1yzD,
					  qt1xzD,
					  qt1yzD);

#endif
	int gridSizeX3Temp1 = ((*ny1p1) + 1)/blockSizeX + 1;
	int gridSizeX3Temp2 = ((*nytop) - 1)/blockSizeX + 1;
    int gridSizeY3Temp1 = ((*nxtop) - 1)/blockSizeY + 1;
    int gridSizeY3Temp2 = ((*nx1p1) + 1)/blockSizeY + 1;
	int gridSizeX3 = (gridSizeX3Temp1 > gridSizeX3Temp2) ? gridSizeX3Temp1 : gridSizeX3Temp2;
	int gridSizeY3 = (gridSizeY3Temp1 > gridSizeY3Temp2) ? gridSizeY3Temp1 : gridSizeY3Temp2;
	dim3 dimGrid3(gridSizeX3, gridSizeY3);

	stress_resetVars<<<dimGrid3, dimBlock>>>(*ny1p1,
                     *nx1p1,
                     *nxtop,
                     *nytop,
					 *nztop,
                     t1xzD,
                     t1yzD);


	if (lbx[1] >= lbx[0])
	{
#if USE_Optimized_stress_norm_PmlX_IC == 1                
        int gridSizeX4 = (nd1_tyy[17] - nd1_tyy[12])/blockSizeX + 1;
        int gridSizeY4 = (nd1_tyy[5] - nd1_tyy[0])/blockSizeY + 1;
#elif USE_Optimized_stress_norm_PmlX_IC == 0                
		int gridSizeX4 = (nd1_tyy[5] - nd1_tyy[0])/blockSizeX + 1;
		int gridSizeY4 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif                
#if USE_Optimized_stress_norm_PmlX_IC ==1||USE_Optimized_stress_norm_PmlX_IC ==0
		dim3 dimGrid4(gridSizeX4, gridSizeY4);

		stress_norm_PmlX_IC<<<dimGrid4, dimBlock>>>(*nxb1,
							 *nyb1,
							 *nxtop,
							 *nytop,
							 *nztop,
							 *mw1_pml,
							 *mw1_pml1,
							 lbx[0],
							 lbx[1],
							 nd1_tyyD,
							 idmat1D,
							 *ca,
							 drti1D,
							 damp1_xD,
							 clamdaD,
							 cmuD,
							 epdtD,
							 qwpD,
							 qwsD,
							 qwt1D,
							 qwt2D,
							 dzi1D,
							 dxh1D,
							 dyh1D,
							 v1xD,
							 v1yD,
							 v1zD,
							 t1xxD,
							 t1yyD,
							 t1zzD,
							 t1xx_pxD,
							 t1yy_pxD,
							 qt1xxD,
							 qt1yyD,
							 qt1zzD,
							 qt1xx_pxD,
							 qt1yy_pxD);

#endif
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_norm_PmlY_IC == 1            
                int gridSizeX5 = (nd1_tyy[17] - nd1_tyy[12])/blockSizeX + 1;
                int gridSizeY5 = (nd1_tyy[11] - nd1_tyy[6])/blockSizeY + 1;
#elif USE_Optimized_stress_norm_PmlY_IC ==0            
		int gridSizeX5 = (nd1_tyy[11] - nd1_tyy[6])/blockSizeX + 1;
		int gridSizeY5 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
#if USE_Optimized_stress_norm_PmlY_IC == 1 || USE_Optimized_stress_norm_PmlY_IC ==0                 
		dim3 dimGrid5(gridSizeX5, gridSizeY5);

		stress_norm_PmlY_IC<<<dimGrid5, dimBlock>>>(*nxb1,
						 *nyb1,
						 *mw1_pml1,
						 *nxtop,
						 *nztop,
						 lby[0],
						 lby[1],
						 nd1_tyyD,
						 idmat1D,
						 *ca,
						 drti1D,
						 damp1_yD,
						 clamdaD,
						 cmuD,
						 epdtD,
						 qwpD,
						 qwsD,
						 qwt1D,
						 qwt2D,
						 dxh1D,
						 dyh1D,
						 dzi1D,
						 t1xxD,
						 t1yyD,
						 t1zzD,
						 qt1xxD,
						 qt1yyD,
						 qt1zzD,
						 t1xx_pyD,
						 t1yy_pyD,
						 qt1xx_pyD,
						 qt1yy_pyD,
						 v1xD,
						 v1yD,
						 v1zD);
#endif
	}

	if (lbx[1] >= lbx[0]) 
	{
#if USE_Optimized_stress_xy_PmlX_IC ==1            
		int gridSizeX6 = (nd1_txy[17] - nd1_txy[12])/blockSizeX + 1;
		int gridSizeY6 = (nd1_txy[5] - nd1_txy[0])/blockSizeY + 1;
#elif USE_Optimized_stress_xy_PmlX_IC ==0            
		int gridSizeX6 = (nd1_txy[5] - nd1_txy[0])/blockSizeX + 1;
		int gridSizeY6 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif                
#if USE_Optimized_stress_xy_PmlX_IC ==0 || USE_Optimized_stress_xy_PmlX_IC == 1                
		dim3 dimGrid6(gridSizeX6, gridSizeY6);

		stress_xy_PmlX_IC<<<dimGrid6, dimBlock>>>(*nxb1,
					   *nyb1,
					   *mw1_pml,
					   *mw1_pml1,
					   *nxtop,
					   *nytop,
					   *nztop,
					   lbx[0],
					   lbx[1],
					   nd1_txyD,
					   idmat1D,
					   *ca,
					   drth1D,
					   damp1_xD,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dxi1D,
					   dyi1D,
					   t1xyD,
					   qt1xyD,
					   t1xy_pxD,
					   qt1xy_pxD,
					   v1xD,
					   v1yD);
#endif                
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_xy_PmlY_IC == 1
		int gridSizeX7 = (nd1_txy[17] - nd1_txy[12])/blockSizeX + 1;
		int gridSizeY7 = (nd1_txy[11] - nd1_txy[6])/blockSizeY + 1;
#elif USE_Optimized_stress_xy_PmlY_IC == 0            
		int gridSizeX7 = (nd1_txy[11] - nd1_txy[6])/blockSizeX + 1;
		int gridSizeY7 = (lby[1] - lby[0])/blockSizeY + 1;
#endif

#if USE_Optimized_stress_xy_PmlY_IC == 1 || USE_Optimized_stress_xy_PmlY_IC == 0
		dim3 dimGrid7(gridSizeX7, gridSizeY7);

		stress_xy_PmlY_IC<<<dimGrid7, dimBlock>>>(*nxb1,
					   *nyb1,
					   *mw1_pml1,
					   *nxtop,
					   *nztop,
					   lby[0],
					   lby[1],
					   nd1_txyD,
					   idmat1D,
					   *ca,
					   drth1D,
					   damp1_yD,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dxi1D,
					   dyi1D,
					   t1xyD,
					   qt1xyD,
					   t1xy_pyD,
					   qt1xy_pyD,
					   v1xD,
					   v1yD);
#endif              
	}

	if (lbx[1] >= lbx[0])
	{
#if USE_Optimized_stress_xz_PmlX_IC == 1            
		int gridSizeX8 = (nd1_txz[17] - nd1_txz[12])/blockSizeX + 1;
		int gridSizeY8 = (nd1_txz[5] - nd1_txz[0])/blockSizeX + 1;
#elif USE_Optimized_stress_xz_PmlX_IC == 0            
		int gridSizeX8 = (nd1_txz[5] - nd1_txz[0])/blockSizeX + 1;
		int gridSizeY8 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif 
#if USE_Optimized_stress_xz_PmlX_IC == 1 || USE_Optimized_stress_xz_PmlX_IC == 0                
		dim3 dimGrid8(gridSizeX8, gridSizeY8);
		stress_xz_PmlX_IC<<<dimGrid8, dimBlock>>>(*nxb1,
					   *nyb1,
					   *nxtop,
					   *nytop,
					   *nztop,
					   *mw1_pml,
					   *mw1_pml1,
					   lbx[0],
					   lbx[1],
					   nd1_txzD,
					   idmat1D,
					   *ca,
					   drth1D,
					   damp1_xD,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dxi1D,
					   dzh1D,
					   t1xzD,
					   qt1xzD,
					   t1xz_pxD,
					   qt1xz_pxD,
					   v1xD,
					   v1zD);

#endif              
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_xz_PmlY_IC == 1            
		int gridSizeX9 = (nd1_txz[17] - nd1_txz[12])/blockSizeX + 1;
		int gridSizeY9 = (nd1_txz[9] - nd1_txz[8])/blockSizeY + 1;
#elif USE_Optimized_stress_xz_PmlY_IC == 0      
		int gridSizeX9 = (nd1_txz[9] - nd1_txz[8])/blockSizeX + 1;
		int gridSizeY9 = (lby[1] - lby[0])/blockSizeY + 1;
#endif                
#if USE_Optimized_stress_xz_PmlY_IC ==1 || USE_Optimized_stress_xz_PmlY_IC == 0                
		dim3 dimGrid9(gridSizeX9, gridSizeY9);

		stress_xz_PmlY_IC<<<dimGrid9, dimBlock>>>(*nxb1,
					   *nyb1,
					   *nxtop,
					   *nztop,
					   lby[0],
					   lby[1],
					   nd1_txzD,
					   idmat1D,
					   *ca,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dxi1D,
					   dzh1D,
					   t1xzD,
					   qt1xzD,
					   v1xD,
					   v1zD);
#endif
	}

	if (lbx[1] >= lbx[0])
	{
#if USE_Optimized_stress_yz_PmlX_IC == 1            
		int gridSizeX10 = (nd1_tyz[17] - nd1_tyz[12])/blockSizeX + 1;
		int gridSizeY10 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeY + 1;
#elif USE_Optimized_stress_yz_PmlX_IC == 0            
		int gridSizeX10 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeX + 1;
		int gridSizeY10 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif                
                
#if USE_Optimized_stress_yz_PmlX_IC == 0 || USE_Optimized_stress_yz_PmlX_IC ==1            
		dim3 dimGrid10(gridSizeX10, gridSizeY10);

		stress_yz_PmlX_IC<<<dimGrid10, dimBlock>>>(*nxb1,
					   *nyb1,
					   *nztop,
					   *nxtop,
					   lbx[0],
					   lbx[1],
					   nd1_tyzD,
					   idmat1D,
					   *ca,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dyi1D,
					   dzh1D,
					   t1yzD,
					   qt1yzD,
					   v1yD,
					   v1zD);
#endif              
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_yz_PmlY_IC ==1            
		int gridSizeX11 = (nd1_tyz[17] - nd1_tyz[12])/blockSizeX + 1;
		int gridSizeY11 = (nd1_tyz[11] - nd1_tyz[6])/blockSizeY + 1;
#elif USE_Optimized_stress_yz_PmlY_IC == 0            
		int gridSizeX11 = (nd1_tyz[11] - nd1_tyz[6])/blockSizeX + 1;
		int gridSizeY11 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
#if USE_Optimized_stress_yz_PmlY_IC == 1 || USE_Optimized_stress_yz_PmlY_IC == 0            
		dim3 dimGrid11(gridSizeX11, gridSizeY11);

		stress_yz_PmlY_IC<<<dimGrid11,dimBlock>>>(*nxb1,
					   *nyb1,
					   *mw1_pml1,
					   *nxtop,
					   *nztop,
					   lby[0],
					   lby[1],
					   nd1_tyzD,
					   idmat1D,
					   *ca,
					   drth1D,
					   damp1_yD,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dyi1D,
					   dzh1D,
					   t1yzD,
					   qt1yzD,
					   t1yz_pyD,
					   qt1yz_pyD,
					   v1yD,
					   v1zD);
#endif        
	}

#if USE_Optimized_stress_norm_xy_II == 1
	int gridSizeX12 = (nd2_tyy[15] - nd2_tyy[12])/blockSizeX + 1;
	int gridSizeY12 = (nd2_tyy[9] - nd2_tyy[8])/blockSizeY + 1;
#elif USE_Optimized_stress_norm_xy_II == 0
	int gridSizeX12 = (nd2_tyy[3] - nd2_tyy[2])/blockSizeX + 1;
	int gridSizeY12 = (nd2_tyy[9] - nd2_tyy[8])/blockSizeY + 1;
#endif        

#if USE_Optimized_stress_norm_xy_II ==0 || USE_Optimized_stress_norm_xy_II == 1
	dim3 dimGrid12(gridSizeX12, gridSizeY12);

	stress_norm_xy_II<<<dimGrid12, dimBlock>>>(*nxb2,
					   *nyb2,
					   *nxbtm,
					   *nzbtm,
					   *nztop,
					   nd2_tyyD,
					   idmat2D,
					   clamdaD,
					   cmuD,
					   epdtD,
					   qwpD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   t2xxD,
					   t2xyD,
					   t2yyD,
					   t2zzD,
					   qt2xxD,
					   qt2xyD,
					   qt2yyD,
					   qt2zzD,
					   dxh2D,
					   dyh2D,
					   dxi2D,
					   dyi2D,
					   dzi2D,
					   v2xD,
					   v2yD,
					   v2zD);
#endif
 
#if USE_Optimized_stress_xz_yz_IIC  == 1
	int gridSizeX13 = (nd2_tyz[15] - nd2_tyz[12])/blockSizeX + 1;
	int gridSizeY13 = (nd2_tyz[9] - nd2_tyz[8])/blockSizeY + 1;
#elif USE_Optimized_stress_xz_yz_IIC  == 0
	int gridSizeX13 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeX + 1;
	int gridSizeY13 = (nd2_tyz[9] - nd2_tyz[8])/blockSizeY + 1;
#endif        

#if USE_Optimized_stress_xz_yz_IIC == 0 || USE_Optimized_stress_xz_yz_IIC == 1
	dim3 dimGrid13(gridSizeX13, gridSizeY13);

	stress_xz_yz_IIC<<<dimGrid13, dimBlock>>>(*nxb2,
					  *nyb2,
					  *nztop,
					  *nxbtm,
					  *nzbtm,
					  nd2_tyzD,
					  idmat2D,
					  cmuD,
					  epdtD,
					  qwsD,
					  qwt1D,
					  qwt2D,
					  dxi2D,
					  dyi2D,
					  dzh2D,
					  t2xzD,
					  t2yzD,
					  qt2xzD,
					  qt2yzD,
					  v2xD,
					  v2yD,
					  v2zD);
#endif

	if (lbx[1] >= lbx[0])
	{
#if USE_Optimized_stress_norm_PmlX_IIC == 1
                int gridSizeX14 = (nd2_tyy[17] - nd2_tyy[12])/blockSizeX + 1;
                int gridSizeY14 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeY + 1;
#elif USE_Optimized_stress_norm_PmlX_IIC == 0
                int gridSizeX14 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeX + 1;
		int gridSizeY14 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
#if USE_Optimized_stress_norm_PmlX_IIC==0||USE_Optimized_stress_norm_PmlX_IIC==1
		dim3 dimGrid14(gridSizeX14, gridSizeY14);

		stress_norm_PmlX_IIC<<<dimGrid14, dimBlock>>>(*nxb2,
						  *nyb2,
						  *mw2_pml,
						  *mw2_pml1,
						  *nztop,
						  *nxbtm,
						  *nybtm,
						  *nzbtm,
						  lbx[0],
						  lbx[1],
						  nd2_tyyD,
						  idmat2D,
						  *ca,
						  drti2D,
						  damp2_xD,
						  clamdaD,
						  cmuD,
						  epdtD,
						  qwpD,
						  qwsD,
						  qwt1D,
						  qwt2D,
						  dxh2D,
						  dyh2D,
						  dzi2D,
						  t2xxD,
						  t2yyD,
						  t2zzD,
						  qt2xxD,
						  qt2yyD,
						  qt2zzD,
						  t2xx_pxD,
						  t2yy_pxD,
						  qt2xx_pxD,
						  qt2yy_pxD,
						  v2xD,
						  v2yD,
						  v2zD);

#endif              
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_norm_PmlY_II == 1            
		int gridSizeX15 = (nd2_tyy[17] - nd2_tyy[12])/blockSizeX + 1;
		int gridSizeY15 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeY + 1;
#elif USE_Optimized_stress_norm_PmlY_II == 0
		int gridSizeX15 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeX + 1;
		int gridSizeY15 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
#if USE_Optimized_stress_norm_PmlY_II==0||USE_Optimized_stress_norm_PmlY_II == 1                
		dim3 dimGrid15(gridSizeX15, gridSizeY15);

		stress_norm_PmlY_II<<<dimGrid15, dimBlock>>>(*nxb2,
						 *nyb2,
						 *nztop,
						 *nxbtm,
						 *nzbtm,
						 *mw2_pml1,
						 lby[0],
						 lby[1],
						 nd2_tyyD,
						 idmat2D,
						 *ca,
						 drti2D,
						 damp2_yD,
						 clamdaD,
						 cmuD,
						 epdtD,
						 qwpD,
						 qwsD,
						 qwt1D,
						 qwt2D,
						 dxh2D,
						 dyh2D,
						 dzi2D,
						 t2xxD,
						 t2yyD,
						 t2zzD,
						 qt2xxD,
						 qt2yyD,
						 qt2zzD,
						 t2xx_pyD,
						 t2yy_pyD,
						 qt2xx_pyD,
						 qt2yy_pyD,
						 v2xD,
						 v2yD,
						 v2zD);
#endif                
	}
#if USE_Optimized_stress_norm_PmlZ_IIC == 1
	int gridSizeX16 = (nd2_tyy[17] - nd2_tyy[16])/blockSizeX + 1;
	int gridSizeY16 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeY + 1;
#elif USE_Optimized_stress_norm_PmlZ_IIC == 0
	int gridSizeX16 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeX + 1;
	int gridSizeY16 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeY + 1;
#endif
#if USE_Optimized_stress_norm_PmlZ_IIC == 1|| USE_Optimized_stress_norm_PmlZ_IIC==0        
	dim3 dimGrid16(gridSizeX16, gridSizeY16);

	 stress_norm_PmlZ_IIC<<<dimGrid16, dimBlock>>>(*nxb2,
						  *nyb2,
						  *mw2_pml,
						  *mw2_pml1,
						  *nztop,
						  *nxbtm,
						  *nzbtm,
						  nd2_tyyD,
						  idmat2D,
						  *ca,
						  damp2_zD,
						  drth2D,
						  clamdaD,
						  cmuD,
						  epdtD,
						  qwpD,
						  qwsD,
						  qwt1D,
						  qwt2D,
						  dxh2D,
						  dyh2D,
						  dzi2D,
						  t2xxD,
						  t2yyD,
						  t2zzD,
						  qt2xxD,
						  qt2yyD,
						  qt2zzD,
						  t2xx_pzD,
						  t2zz_pzD,
						  qt2xx_pzD,
						  qt2zz_pzD,
						  v2xD,
						  v2yD,
						  v2zD);
#endif              

	if (lbx[1] >= lbx[0])
	{
#if USE_Optimized_stress_xy_PmlX_IIC == 1            
		int gridSizeX17 = (nd2_txy[17] - nd2_txy[12])/blockSizeX + 1;
		int gridSizeY17 = (nd2_txy[5] - nd2_txy[0])/blockSizeY + 1;
#elif USE_Optimized_stress_xy_PmlX_IIC == 0            
		int gridSizeX17 = (nd2_txy[5] - nd2_txy[0])/blockSizeX + 1;
		int gridSizeY17 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif
#if USE_Optimized_stress_xy_PmlX_IIC == 0 || USE_Optimized_stress_xy_PmlX_IIC==1
		dim3 dimGrid17(gridSizeX17, gridSizeY17);

		stress_xy_PmlX_IIC<<<dimGrid17, dimBlock>>>(*nxb2,
						*nyb2,
						*mw2_pml,
						*mw2_pml1,
						*nxbtm,
						*nybtm,
						*nzbtm,
						*nztop,
						lbx[0],
						lbx[1],
						nd2_txyD,
						idmat2D,
						*ca,
						drth2D,
						damp2_xD,
						cmuD,
						epdtD,
						qwsD,
						qwt1D,
						qwt2D,
						dxi2D,
						dyi2D,
						t2xyD,
						qt2xyD,
						t2xy_pxD,
						qt2xy_pxD,
						v2xD,
						v2yD);
#endif              
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_xy_PmlY_IIC == 1            
		int gridSizeX18 = (nd2_txy[17] - nd2_txy[12])/blockSizeX + 1;
		int gridSizeY18 = (nd2_txy[11] - nd2_txy[6])/blockSizeY + 1;
#elif USE_Optimized_stress_xy_PmlY_IIC == 0
		int gridSizeX18 = (nd2_txy[11] - nd2_txy[6])/blockSizeX + 1;
		int gridSizeY18 = (lby[1] - lby[0])/blockSizeY + 1;
#endif
#if USE_Optimized_stress_xy_PmlY_IIC ==1 || USE_Optimized_stress_xy_PmlY_IIC ==0                
		dim3 dimGrid18(gridSizeX18, gridSizeY18);

		stress_xy_PmlY_IIC<<<dimGrid18, dimBlock>>>(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nztop,
						*nxbtm,
						*nzbtm,
						lby[0],
						lby[1],
						nd2_txyD,
						idmat2D,
						*ca,
						drth2D,
						damp2_yD,
						cmuD,
						epdtD,
						qwsD,
						qwt1D,
						qwt2D,
						dxi2D,
						dyi2D,
						t2xyD,
						qt2xyD,
						t2xy_pyD,
						qt2xy_pyD,
						v2xD,
						v2yD);
#endif                
	}

#if USE_Optimized_stress_xy_PmlZ_II == 1
	int gridSizeX19 = (nd2_txy[17] - nd2_txy[16])/blockSizeX + 1;
	int gridSizeY19 = (nd2_txy[9] - nd2_txy[8])/blockSizeY + 1;
#elif USE_Optimized_stress_xy_PmlZ_II == 0
	int gridSizeX19 = (nd2_txy[3] - nd2_txy[2])/blockSizeX + 1;
	int gridSizeY19 = (nd2_txy[9] - nd2_txy[8])/blockSizeY + 1;
#endif       

#if USE_Optimized_stress_xy_PmlZ_II == 1|| USE_Optimized_stress_xy_PmlZ_II ==0
	dim3 dimGrid19(gridSizeX19, gridSizeY19);

	stress_xy_PmlZ_II<<<dimGrid19, dimBlock>>>(*nxb2,
					   *nyb2,
					   *nxbtm,
					   *nzbtm,
					   *nztop,
					   nd2_txyD,
					   idmat2D,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dxi2D,
					   dyi2D,
					   t2xyD,
					   qt2xyD,
					   v2xD,
					   v2yD);
#endif        

	if (lbx[1] >= lbx[0])
	{
#if USE_Optimized_stress_xz_PmlX_IIC == 1
		int gridSizeX20 = (nd2_txz[17] - nd2_txz[12])/blockSizeX + 1;
		int gridSizeY20 = (nd2_txz[5] - nd2_txz[0])/blockSizeY + 1;
#elif USE_Optimized_stress_xz_PmlX_IIC == 0
		int gridSizeX20 = (nd2_txz[5] - nd2_txz[0])/blockSizeX + 1;
		int gridSizeY20 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif               
#if USE_Optimized_stress_xz_PmlX_IIC == 0 || USE_Optimized_stress_xz_PmlX_IIC==1                
		dim3 dimGrid20(gridSizeX20, gridSizeY20);

	 	stress_xz_PmlX_IIC<<<dimGrid20, dimBlock>>>(*nxb2,
						*nyb2,
						*mw2_pml,
						*mw2_pml1,
						*nxbtm,
						*nybtm,
						*nzbtm,
						*nztop,
						lbx[0],
						lbx[1],
						nd2_txzD,
						idmat2D,
						*ca,
						drth2D,
						damp2_xD,
						cmuD,
						epdtD,
						qwsD,
						qwt1D,
						qwt2D,
						dxi2D,
						dzh2D,
						t2xzD,
						qt2xzD,
						t2xz_pxD,
						qt2xz_pxD,
						v2xD,
						v2zD);
#endif                
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_xz_PmlY_IIC == 1           
		int gridSizeX21 = (nd2_txz[15] - nd2_txz[12])/blockSizeX + 1;
		int gridSizeY21 = (nd2_txz[9] - nd2_txz[8])/blockSizeY + 1;
#elif USE_Optimized_stress_xz_PmlY_IIC == 0            
		int gridSizeX21 = (nd2_txz[9] - nd2_txz[8])/blockSizeX + 1;
		int gridSizeY21 = (lby[1] - lby[0])/blockSizeY + 1;
#endif                
#if USE_Optimized_stress_xz_PmlY_IIC ==1 || USE_Optimized_stress_xz_PmlY_IIC ==0
		dim3 dimGrid21(gridSizeX21, gridSizeY21);

		stress_xz_PmlY_IIC<<<dimGrid21, dimBlock>>>(*nxb2,
					   *nyb2,
					   *nxbtm,
					   *nzbtm,
					   *nztop,
					   lby[0],
					   lby[1],
					   nd2_txzD,
					   idmat2D,
					   cmuD,
					   epdtD,
					   qwsD,
					   qwt1D,
					   qwt2D,
					   dxi2D,
					   dzh2D,
					   v2xD,
					   v2zD,
					   t2xzD,
					   qt2xzD);
#endif                
	}
#if USE_Optimized_stress_xz_PmlZ_IIC == 1
	int gridSizeX22 = (nd2_txz[17] - nd2_txz[16])/blockSizeX + 1;
	int gridSizeY22 = (nd2_txz[11] - nd2_txz[6])/blockSizeY + 1;
#elif USE_Optimized_stress_xz_PmlZ_IIC == 0
	int gridSizeX22 = (nd2_txz[5] - nd2_txz[0])/blockSizeX + 1;
	int gridSizeY22 = (nd2_txz[11] - nd2_txz[6])/blockSizeY + 1;
#endif        
#if USE_Optimized_stress_xz_PmlZ_IIC == 1|| USE_Optimized_stress_xz_PmlZ_IIC ==0
	dim3 dimGrid22(gridSizeX22, gridSizeY22);

	stress_xz_PmlZ_IIC<<<dimGrid22, dimBlock>>>(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nxbtm,
						*nzbtm,
						*nztop,
						nd2_txzD,
						idmat2D,
						*ca,
						drti2D,
						damp2_zD,
						cmuD,
						epdtD,
						qwsD,
						qwt1D,
						qwt2D,
						dxi2D,
						dzh2D,
						t2xzD,
						qt2xzD,
						t2xz_pzD,
						qt2xz_pzD,
						v2xD,
						v2zD);
#endif              

	if (lbx[1] >= lbx[0])
	{
#if USE_Optimized_stress_yz_PmlX_IIC == 1            
		int gridSizeX23 = (nd2_tyz[15] - nd2_tyz[12])/blockSizeX + 1;
		int gridSizeY23 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeY + 1;
#elif USE_Optimized_stress_yz_PmlX_IIC == 0            
		int gridSizeX23 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeX + 1;
		int gridSizeY23 = (lbx[1] - lbx[0])/blockSizeY + 1;
#endif                
#if USE_Optimized_stress_yz_PmlX_IIC ==0 || USE_Optimized_stress_yz_PmlX_IIC==1		
                dim3 dimGrid23(gridSizeX23, gridSizeY23);
		stress_yz_PmlX_IIC<<<dimGrid23, dimBlock>>>(*nxb2,
						*nyb2,
						*nxbtm,
						*nzbtm,
						*nztop,
						lbx[0],
						lbx[1],
						nd2_tyzD,
						idmat2D,
						cmuD,
						epdtD,
						qwsD,
						qwt1D,
						qwt2D,
						dyi2D,
						dzh2D,
						t2yzD,
						qt2yzD,
						v2yD,
						v2zD);
#endif                
	}

	if (lby[1] >= lby[0])
	{
#if USE_Optimized_stress_yz_PmlY_IIC == 1
		int gridSizeX24 = (nd2_tyz[17] - nd2_tyz[12])/blockSizeX + 1;
		int gridSizeY24 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeY + 1;
#elif USE_Optimized_stress_yz_PmlY_IIC == 0
		int gridSizeX24 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeX + 1;
		int gridSizeY24 = (lby[1] - lby[0])/blockSizeY + 1;
#endif              
#if USE_Optimized_stress_yz_PmlY_IIC ==1 || USE_Optimized_stress_yz_PmlY_IIC==0                
		dim3 dimGrid24(gridSizeX24, gridSizeY24);

		stress_yz_PmlY_IIC<<<dimGrid24, dimBlock>>>(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nxbtm,
						*nzbtm,
						*nztop,
						lby[0],
						lby[1],
						nd2_tyzD,
						idmat2D,
						*ca,
						drth2D,
						damp2_yD,
						cmuD,
						epdtD,
						qwsD,
						qwt1D,
						qwt2D,
						dyi2D,
						dzh2D,
						t2yzD,
						qt2yzD,
						t2yz_pyD,
						qt2yz_pyD,
						v2yD,
						v2zD);
#endif              
	}

#if USE_Optimized_stress_yz_PmlZ_IIC ==1
	int gridSizeX25 = (nd2_tyz[17] - nd2_tyz[16])/blockSizeX + 1;
	int gridSizeY25 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeY + 1;
#elif USE_Optimized_stress_yz_PmlZ_IIC ==0
	int gridSizeX25 = (nd2_tyz[5] - nd2_tyz[0])/blockSizeX + 1;
	int gridSizeY25 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeY + 1;
#endif       

#if USE_Optimized_stress_yz_PmlZ_IIC ==0 || USE_Optimized_stress_yz_PmlZ_IIC == 1
	dim3 dimGrid25(gridSizeX25, gridSizeY25);

	stress_yz_PmlZ_IIC<<<dimGrid25, dimBlock>>>(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nxbtm,
						*nzbtm,
						*nztop,
						nd2_tyzD,
						idmat2D,
						*ca,
						drti2D,
						damp2_zD,
						cmuD,
						epdtD,
						qwsD,
						qwt1D,
						qwt2D,
						dyi2D,
						dzh2D,
						t2yzD,
						qt2yzD,
						t2yz_pzD,
						qt2yz_pzD,
						v2yD,
						v2zD);
#endif
  cudaRes = cudaDeviceSynchronize();
  CHECK_ERROR(cudaRes, "Stress kernel computation error");

  record_time(&tend);
  logger.log_timing("Stress Computation", tend-tstart );

#if VALIDATE_MODE==1
	record_time(&tstart);
	cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, 
			t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	record_time(&tend);
  	logger.log_timing("Data Transfer POST Stress Computation", tend-tstart );
#endif        
	return;
}

#ifdef __cplusplus
}
#endif

