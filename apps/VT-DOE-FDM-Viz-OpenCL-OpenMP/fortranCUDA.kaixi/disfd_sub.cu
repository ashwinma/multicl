#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
/***********************************************/
/* for debug: check the output                 */
/***********************************************/
void write_output(float *arr, int size, const char *filename)
{
    FILE *fp;
    if((fp = fopen(filename, "w+")) == NULL)
    {
        fprintf(stderr, "File write error!\n");
    }
    int i;
    for(i = 0; i < size; i++)
    {
        fprintf(fp, "%f ", arr[i]);
        if( i%10 == 0)
            fprintf(fp, "\n");
    }
    fprintf(fp, "\n");
    fclose(fp);
}
//device memory pointers
static int   *nd1_velD;
static int   *nd1_txyD;
static int   *nd1_txzD;
static int   *nd1_tyyD;
static int   *nd1_tyzD;
static float *rhoD;
static float *drvh1D;
static float *drti1D;
static float *drth1D;
static float *damp1_xD;
static float *damp1_yD;
static int   *idmat1D;
static float *dxi1D;
static float *dyi1D;
static float *dzi1D;
static float *dxh1D;
static float *dyh1D;
static float *dzh1D;
static float *t1xxD;
static float *t1xyD;
static float *t1xzD;
static float *t1yyD;
static float *t1yzD;
static float *t1zzD;
static float *t1xx_pxD;
static float *t1xy_pxD;
static float *t1xz_pxD;
static float *t1yy_pxD;
static float *qt1xx_pxD;
static float *qt1xy_pxD;
static float *qt1xz_pxD;
static float *qt1yy_pxD;
static float *t1xx_pyD;
static float *t1xy_pyD;
static float *t1yy_pyD;
static float *t1yz_pyD;
static float *qt1xx_pyD;
static float *qt1xy_pyD;
static float *qt1yy_pyD;
static float *qt1yz_pyD;
static float *qt1xxD;
static float *qt1xyD;
static float *qt1xzD;
static float *qt1yyD;
static float *qt1yzD;
static float *qt1zzD;
static float *clamdaD;
static float *cmuD;
static float *epdtD;
static float *qwpD;
static float *qwsD;
static float *qwt1D;
static float *qwt2D;

static float *v1xD;    //output
static float *v1yD;
static float *v1zD;
static float *v1x_pxD;
static float *v1y_pxD;
static float *v1z_pxD;
static float *v1x_pyD;
static float *v1y_pyD;
static float *v1z_pyD;

//for inner_II---------------------------------------------------------
static int	 *nd2_velD;
static int   *nd2_txyD;  //int[18]
static int   *nd2_txzD;  //int[18]
static int   *nd2_tyyD;  //int[18]
static int   *nd2_tyzD;  //int[18]

static float *drvh2D;
static float *drti2D;
static float *drth2D; 	//float[mw2_pml1,0:1]

static int	 *idmat2D;
static float *damp2_xD;
static float *damp2_yD;
static float *damp2_zD;
static float *dxi2D;
static float *dyi2D;
static float *dzi2D;
static float *dxh2D;
static float *dyh2D;
static float *dzh2D;
static float *t2xxD;
static float *t2xyD;
static float *t2xzD;
static float *t2yyD;
static float *t2yzD;
static float *t2zzD;
static float *qt2xxD;
static float *qt2xyD;
static float *qt2xzD;
static float *qt2yyD;
static float *qt2yzD;
static float *qt2zzD;

static float *t2xx_pxD;
static float *t2xy_pxD;
static float *t2xz_pxD;
static float *t2yy_pxD;
static float *qt2xx_pxD;
static float *qt2xy_pxD;
static float *qt2xz_pxD;
static float *qt2yy_pxD;

static float *t2xx_pyD;
static float *t2xy_pyD;
static float *t2yy_pyD;
static float *t2yz_pyD;
static float *qt2xx_pyD;
static float *qt2xy_pyD;
static float *qt2yy_pyD;
static float *qt2yz_pyD;

static float *t2xx_pzD;
static float *t2xz_pzD;
static float *t2yz_pzD;
static float *t2zz_pzD;
static float *qt2xx_pzD;
static float *qt2xz_pzD;
static float *qt2yz_pzD;
static float *qt2zz_pzD;

static float *v2xD;		//output
static float *v2yD;
static float *v2zD;
static float *v2x_pxD;
static float *v2y_pxD;
static float *v2z_pxD;
static float *v2x_pyD;
static float *v2y_pyD;
static float *v2z_pyD;
static float *v2x_pzD;
static float *v2y_pzD;
static float *v2z_pzD;

#define CHECK_ERROR(err, str) \
	if (err != cudaSuccess) \
	{\
		printf("Error in \"%s\", %s\n", str, cudaGetErrorString(err)); \
	}

//debug----------------------
double  totalTimeH2DV, totalTimeD2HV;
double  totalTimeH2DS, totalTimeD2HS;
double  totalTimeCompV, totalTimeCompS;
double  tmpTime;
struct timeval t1, t2;
int procID;
//--------------------------------

//!XSC--------------------------------------------------------------------
#define drvh1(i, j) drvh1M[(i) - 1 + (j) * mw1_pml1]
#define drti1(i, j) drti1M[(i) - 1 + (j) * mw1_pml1]
#define drth1(i, j) drth1M[(i) - 1 + (j) * mw1_pml1]

#define damp1_x(i, j, k) damp1_xM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) - lbx0) * nytop)]
#define damp1_y(i, j, k) damp1_yM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) - lby0) * nxtop)]

#define idmat1(i, j, k) idmat1M[(i) + (nztop + 2) * ((j) - 1 + ((k) - 1) * (nxtop + 1))]

#define v1x(i, j, k) v1xM[(i) + (nztop + 2) * ((j) + 1 + (k) * (nxtop + 3))]
#define v1y(i, j, k) v1yM[(i) + (nztop + 2) * ((j) + ((k) + 1) * (nxtop + 3))]
#define v1z(i, j, k) v1zM[(i) + (nztop + 2) * ((j) + (k) * (nxtop + 3))]

//nv2x=(lbx(2) - lbx(1) + 1) * mw1_pml
#define v1x_px(i, j, k) v1x_pxM[(i) - 1 + nztop * ((j) - 1 + nv2x * ((k) - 1))]
#define v1y_px(i, j, k) v1y_pxM[(i) - 1 + nztop * ((j) - 1 + nv2x * ((k) - 1))]
#define v1z_px(i, j, k) v1z_pxM[(i) - 1 + nztop * ((j) - 1 + nv2x * ((k) - 1))]

#define v1x_py(i, j, k) v1x_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define v1y_py(i, j, k) v1y_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define v1z_py(i, j, k) v1z_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]

#define dxi1(i, j) dxi1M[((j) - 1) * 4 + (i) - 1]
#define dyi1(i, j) dyi1M[((j) - 1) * 4 + (i) - 1]
#define dzi1(i, j) dzi1M[((j) - 1) * 4 + (i) - 1]
#define dxh1(i, j) dxh1M[((j) - 1) * 4 + (i) - 1]
#define dyh1(i, j) dyh1M[((j) - 1) * 4 + (i) - 1]
#define dzh1(i, j) dzh1M[((j) - 1) * 4 + (i) - 1]

#define t1xx(i, j, k) t1xxM[(i) - 1 + nztop * ((j) + ((k) - 1) * (nxtop + 3))]
#define t1xy(i, j, k) t1xyM[(i) - 1 + nztop * ((j) + 1 + ((k) + 1) * (nxtop + 3))]
#define t1xz(i, j, k) t1xzM[(i) - 1 + (nztop + 1) * ((j) + 1 + ((k) - 1) * (nxtop + 3))]
#define t1yy(i, j, k) t1yyM[(i) - 1 + nztop * (((j) - 1) + (k) * nxtop)]
#define t1yz(i, j, k) t1yzM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) + 1) * nxtop)]
#define t1zz(i, j, k) t1zzM[(i) - 1 + nztop * ((j) - 1 + ((k) - 1) * nxtop)]

//nti = (lbx(2) - lbx(1) + 1) * mw1_pml + lbx(2)
//nth = (lbx(2) - lbx(1) + 1) * mw1_pml + 1 - lbx(1)
#define t1xx_px(i, j, k)   t1xx_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]
#define t1xy_px(i, j, k)   t1xy_pxM[(i) - 1 + nztop * ((j) - 1 + nth * ((k) - 1))]
#define t1xz_px(i, j, k)   t1xz_pxM[(i) - 1 + (nztop + 1) * ((j) - 1 + nth * ((k) - 1))]
#define t1yy_px(i, j, k)   t1yy_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]
#define qt1xx_px(i, j, k) qt1xx_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]
#define qt1xy_px(i, j, k) qt1xy_pxM[(i) - 1 + nztop * ((j) - 1 + nth * ((k) - 1))]
#define qt1xz_px(i, j, k) qt1xz_pxM[(i) - 1 + (nztop + 1) * ((j) - 1 + nth * ((k) - 1))]
#define qt1yy_px(i, j, k) qt1yy_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]

//nti = (lby(2) - lby(1) + 1) * mw1_pml + lby(2)
//nth = (lby(2) - lby(1) + 1) * mw1_pml + 1 - lby(1)
#define t1xx_py(i, j, k)   t1xx_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define t1xy_py(i, j, k)   t1xy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define t1yy_py(i, j, k)   t1yy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define t1yz_py(i, j, k)   t1yz_pyM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xx_py(i, j, k) qt1xx_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xy_py(i, j, k) qt1xy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yy_py(i, j, k) qt1yy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yz_py(i, j, k) qt1yz_pyM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]

#define qt1xx(i, j, k) qt1xxM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xy(i, j, k) qt1xyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xz(i, j, k) qt1xzM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yy(i, j, k) qt1yyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yz(i, j, k) qt1yzM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1zz(i, j, k) qt1zzM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]

#define rho(i)     rhoM[(i) - 1]
#define clamda(i) clamdaM[(i) - 1]
#define cmu(i)     cmuM[(i) - 1]
#define epdt(i)    epdtM[(i) - 1]
#define qwp(i)     qwpM[(i) - 1]
#define qws(i)     qwsM[(i) - 1]
#define qwt1(i)    qwt1M[(i) - 1]
#define qwt2(i)    qwt2M[(i) - 1]

//for inner_II
#define drvh2(i, j) drvh2M[(i) - 1 + (j) * mw2_pml1]
#define drti2(i, j) drti2M[(i) - 1 + (j) * mw2_pml1]
#define drth2(i, j) drth2M[(i) - 1 + (j) * mw2_pml1]

#define idmat2(i, j, k) idmat2M[(i) + (nzbtm + 1) * ((j) - 1 + ((k) - 1) * (nxbtm + 1))]

#define damp2_x(i, j, k) damp2_xM[(i) - 1 + nzbtm * ((j) - 1 + ((k) - lbx0) * nybtm)]
#define damp2_y(i, j, k) damp2_yM[(i) - 1 + nzbtm * ((j) - 1 + ((k) - lby0) * nxbtm)]
#define damp2_z(i, j) 	 damp2_zM[(i) - 1 + nxbtm * ((j) - 1)]

#define dxi2(i, j) dxi2M[(i) - 1 + 4 * ((j) - 1)]
#define dyi2(i, j) dyi2M[(i) - 1 + 4 * ((j) - 1)]
#define dzi2(i, j) dzi2M[(i) - 1 + 4 * ((j) - 1)]

#define dxh2(i, j) dxh2M[(i) - 1 + 4 * ((j) - 1)]
#define dyh2(i, j) dyh2M[(i) - 1 + 4 * ((j) - 1)]
#define dzh2(i, j) dzh2M[(i) - 1 + 4 * ((j) - 1)]

#define t2xx(i, j, k) t2xxM[(i) - 1 + nzbtm * ((j) + ((k) - 1) * (nxbtm + 3))]
#define t2xy(i, j, k) t2xyM[(i) - 1 + nzbtm * ((j) + 1 + ((k) + 1) * (nxbtm + 3))]
#define t2xz(i, j, k) t2xzM[(i) + (nzbtm + 1) * ((j) + 1 + ((k) - 1) * (nxbtm + 3))]
#define t2yy(i, j, k) t2yyM[(i) - 1 + nzbtm * (((j) - 1) + (k) * nxbtm)]
#define t2yz(i, j, k) t2yzM[(i) + (nzbtm + 1) * ((j) - 1 + ((k) + 1) * nxbtm)]
#define t2zz(i, j, k) t2zzM[(i) + (nzbtm + 1) * ((j) - 1 + ((k) - 1) * nxbtm)]
#define qt2xx(i, j, k) qt2xxM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xy(i, j, k) qt2xyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xz(i, j, k) qt2xzM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yy(i, j, k) qt2yyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yz(i, j, k) qt2yzM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2zz(i, j, k) qt2zzM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

//nti = (lbx(2) - lbx(1) + 1) * mw2_pml + lbx(2)
//nth = (lbx(2) - lbx(1) + 1) * mw2_pml + 1 - lbx(1)
#define t2xx_px(i, j, k) t2xx_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]
#define t2xy_px(i, j, k) t2xy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define t2xz_px(i, j, k) t2xz_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define t2yy_px(i, j, k) t2yy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]

#define t2xx_py(i, j, k) t2xx_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2xy_py(i, j, k) t2xy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2yy_py(i, j, k) t2yy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2yz_py(i, j, k) t2yz_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

#define t2xx_pz(i, j, k) t2xx_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2xz_pz(i, j, k) t2xz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2yz_pz(i, j, k) t2yz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2zz_pz(i, j, k) t2zz_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]

#define qt2xx_px(i, j, k) qt2xx_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]
#define qt2xy_px(i, j, k) qt2xy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define qt2xz_px(i, j, k) qt2xz_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define qt2yy_px(i, j, k) qt2yy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]

#define qt2xx_py(i, j, k) qt2xx_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xy_py(i, j, k) qt2xy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yy_py(i, j, k) qt2yy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yz_py(i, j, k) qt2yz_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

#define qt2xx_pz(i, j, k) qt2xx_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xz_pz(i, j, k) qt2xz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yz_pz(i, j, k) qt2yz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2zz_pz(i, j, k) qt2zz_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]

#define v2x(i, j, k) v2xM[(i) + (nzbtm + 1) * ((j) + 1 + (nxbtm + 3) * (k))]
#define v2y(i, j, k) v2yM[(i) + (nzbtm + 1) * ((j) + (nxbtm + 3) * ((k) + 1))]
#define v2z(i, j, k) v2zM[(i) + (nzbtm + 1) * ((j) + (nxbtm + 3) * (k))]

//nv2y = (lbx(2) - lbx(1) + 1) * mw2_pml
#define v2x_px(i, j, k) v2x_pxM[(i) - 1 + nzbtm * ((j) - 1 + nv2y * ((k) - 1))]
#define v2y_px(i, j, k) v2y_pxM[(i) - 1 + nzbtm * ((j) - 1 + nv2y * ((k) - 1))]
#define v2z_px(i, j, k) v2z_pxM[(i) - 1 + nzbtm * ((j) - 1 + nv2y * ((k) - 1))]

#define v2x_py(i, j, k) v2x_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2y_py(i, j, k) v2y_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2z_py(i, j, k) v2z_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

#define v2x_pz(i, j, k) v2x_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2y_pz(i, j, k) v2y_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2z_pz(i, j, k) v2z_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]

__global__ void velocity_inner_IC(int	nztop,
					  int 	nztm1,
					  float ca,
					  int	*nd1_vel,
					  float *rhoM,
					  int   *idmat1M,
					  float *dxi1M,
					  float *dyi1M,
					  float *dzi1M,
					  float *dxh1M,
					  float *dyh1M,
					  float *dzh1M,
					  float *t1xxM,
					  float *t1xyM,
					  float *t1xzM,
					  float *t1yyM,
					  float *t1yzM,
					  float *t1zzM,
					  int	nxtop,		//dimension #
					  int	nytop,
					  float *v1xM,		//output
					  float *v1yM,
					  float *v1zM);

__global__ void velocity_inner_IIC(float ca,
					   int	 *nd2_vel,
					   float *rhoM,
					   float *dxi2,
					   float *dyi2,
					   float *dzi2,
					   float *dxh2,
					   float *dyh2,
					   float *dzh2,
					   int 	 *idmat2,
					   float *t2xx,
					   float *t2xy,
					   float *t2xz,
					   float *t2yy,
					   float *t2yz,
					   float *t2zz,
					   int   nxbtm,	//dimension #s
					   int   nybtm,
					   int   nzbtm,
					   float *v2x,		//output
					   float *v2y,
					   float *v2z);

__global__ void vel_PmlX_IC(float ca,
				int   lbx0,
				int	  lbx1,
				int	  *nd1_vel,
				float *rhoM,
				float *drvh1,
				float *drti1,
				float *damp1_x,
				int	  *idmat1,
				float *dxi1,
				float *dyi1,
				float *dzi1,
				float *dxh1,
				float *dyh1,
				float *dzh1,
				float *t1xx,
				float *t1xy,
				float *t1xz,
				float *t1yy,
				float *t1yz,
				float *t1zz,
				int   mw1_pml1,	//dimension #
			    int   mw1_pml,
			    int   nxtop,
			    int   nytop,
			    int   nztop,
				float *v1x,		//output
				float *v1y,
				float *v1z,
				float *v1x_px,
				float *v1y_px,
				float *v1z_px);

__global__ void vel_PmlY_IC(int  nztop,
				float ca,
				int	  lby0,
				int   lby1,
				int   *nd1_vel,
				float *rhoM,
				float *drvh1,
				float *drti1,
				int   *idmat1,
				float *damp1_y,
				float *dxi1,
				float *dyi1,
				float *dzi1,
				float *dxh1,
				float *dyh1,
				float *dzh1,
				float *t1xx,
				float *t1xy,
				float *t1xz,
				float *t1yy,
				float *t1yz,
				float *t1zz,
				int   mw1_pml1, //dimension #s
				int   mw1_pml,
				int   nxtop,
				int   nytop,
				float *v1x,		//output
				float *v1y,
				float *v1z,
				float *v1x_py,
				float *v1y_py,
				float *v1z_py);

__global__ void vel_PmlX_IIC(int   nzbm1,
				 float ca,
				 int   lbx0,
				 int   lbx1,
				 int   *nd2_vel,
				 float *drvh2,
				 float *drti2,
				 float *rhoM,
				 float *damp2_x,
				 int   *idmat2,
				 float *dxi2,
				 float *dyi2,
				 float *dzi2,
				 float *dxh2,
				 float *dyh2,
				 float *dzh2,
				 float *t2xx,
				 float *t2xy,
				 float *t2xz,
				 float *t2yy,
				 float *t2yz,
				 float *t2zz,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 float *v2x,	//output
				 float *v2y,
				 float *v2z,
				 float *v2x_px,
				 float *v2y_px,
				 float *v2z_px);

__global__ void vel_PmlY_IIC(int   nzbm1,
				 float ca,
				 int   lby0,
				 int   lby1,
				 int   *nd2_vel,
				 float *drvh2,
				 float *drti2,
				 float *rhoM,
				 float *damp2_y,
				 int   *idmat2,
				 float *dxi2,
				 float *dyi2,
				 float *dzi2,
				 float *dxh2,
				 float *dyh2,
				 float *dzh2,
				 float *t2xx,
				 float *t2xy,
				 float *t2xz,
				 float *t2yy,
				 float *t2yz,
				 float *t2zz,
				 int   mw2_pml1,
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 float *v2x,		//output
				 float *v2y,
				 float *v2z,
				 float *v2x_py,
				 float *v2y_py,
				 float *v2z_py);

__global__ void vel_PmlZ_IIC(int   nzbm1,
				 float ca,
				 int   *nd2_vel,
				 float *drvh2,
				 float *drti2,
				 float *rhoM,
				 float *damp2_z,
				 int   *idmat2,
				 float *dxi2,
				 float *dyi2,
				 float *dzi2,
				 float *dxh2,
				 float *dyh2,
				 float *dzh2,
				 float *t2xx,
				 float *t2xy,
				 float *t2xz,
				 float *t2yy,
				 float *t2yz,
				 float *t2zz,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 float *v2x,		//output
				 float *v2y,
				 float *v2z,
				 float *v2x_pz,
				 float *v2y_pz,
				 float *v2z_pz);
#ifdef __cplusplus
extern "C" {
#endif
extern void compute_velocityCDebug( int *nztop,  int *nztm1,  float *ca, int *lbx,
			 int *lby, int *nd1_vel, float *rhoM, float *drvh1M, float *drti1M,
			 float *damp1_xM, float *damp1_yM, int *idmat1M,float *dxi1M, float *dyi1M,
			 float *dzi1M, float *dxh1M, float *dyh1M, float *dzh1M, float *t1xxM,
			 float *t1xyM, float *t1xzM, float *t1yyM, float *t1yzM, float *t1zzM,
			 void **v1xMp, void **v1yMp, void **v1zMp, float *v1x_pxM, float *v1y_pxM,
			 float *v1z_pxM, float *v1x_pyM, float *v1y_pyM,  float *v1z_pyM,
			  int *nzbm1,  int *nd2_vel, float *drvh2M, float *drti2M,
			 int *idmat2M, float *damp2_xM, float *damp2_yM, float *damp2_zM,
			 float *dxi2M, float *dyi2M, float *dzi2M, float *dxh2M, float *dyh2M,
			 float *dzh2M, float *t2xxM, float *t2xyM, float *t2xzM, float *t2yyM,
			 float *t2yzM, float *t2zzM, void **v2xMp, void **v2yMp, void **v2zMp,
			 float *v2x_pxM, float *v2y_pxM, float *v2z_pxM, float *v2x_pyM,
			 float *v2y_pyM, float *v2z_pyM, float *v2x_pzM, float *v2y_pzM,
			 float *v2z_pzM,  int *nmat, int *mw1_pml1,  int *mw2_pml1,
			  int *nxtop,  int *nytop,  int *mw1_pml,  int *mw2_pml,
			  int *nxbtm,  int *nybtm,  int *nzbtm);

extern void compute_stressCDebug(int *nxb1, int *nyb1, int *nx1p1, int *ny1p1, int *nxtop, int *nytop, int *nztop, int *mw1_pml,
			int *mw1_pml1, int *lbx, int *lby, int *nd1_txy, int *nd1_txz,
			int *nd1_tyy, int *nd1_tyz, int *idmat1M, float *ca, float *drti1M, float *drth1M, float *damp1_xM, float *damp1_yM,
			float *clamdaM, float *cmuM, float *epdtM, float *qwpM, float *qwsM, float *qwt1M, float *qwt2M, float *dxh1M,
			float *dyh1M, float *dzh1M, float *dxi1M, float *dyi1M, float *dzi1M, float *t1xxM, float *t1xyM, float *t1xzM, 
			float *t1yyM, float *t1yzM, float *t1zzM, float *qt1xxM, float *qt1xyM, float *qt1xzM, float *qt1yyM, float *qt1yzM, 
			float *qt1zzM, float *t1xx_pxM, float *t1xy_pxM, float *t1xz_pxM, float *t1yy_pxM, float *qt1xx_pxM, float *qt1xy_pxM,
			float *qt1xz_pxM, float *qt1yy_pxM, float *t1xx_pyM, float *t1xy_pyM, float *t1yy_pyM, float *t1yz_pyM, float *qt1xx_pyM,
			float *qt1xy_pyM, float *qt1yy_pyM, float *qt1yz_pyM, void **v1xMp, void **v1yMp, void **v1zMp,
			int *nxb2, int *nyb2, int *nxbtm, int *nybtm, int *nzbtm, int *mw2_pml, int *mw2_pml1, int *nd2_txy, int *nd2_txz, 
			int *nd2_tyy, int *nd2_tyz, int *idmat2M, 
			float *drti2M, float *drth2M, float *damp2_xM, float *damp2_yM, float *damp2_zM, 
			float *t2xxM, float *t2xyM, float *t2xzM, float *t2yyM, float *t2yzM, float *t2zzM, 
			float *qt2xxM, float *qt2xyM, float *qt2xzM, float *qt2yyM, float *qt2yzM, float *qt2zzM, 
			float *dxh2M, float *dyh2M, float *dzh2M, float *dxi2M, float *dyi2M, float *dzi2M, 
			float *t2xx_pxM, float *t2xy_pxM, float *t2xz_pxM, float *t2yy_pxM, float *t2xx_pyM, float *t2xy_pyM,
			float *t2yy_pyM, float *t2yz_pyM, float *t2xx_pzM, float *t2xz_pzM, float *t2yz_pzM, float *t2zz_pzM,
			float *qt2xx_pxM, float *qt2xy_pxM, float *qt2xz_pxM, float *qt2yy_pxM, float *qt2xx_pyM, float *qt2xy_pyM, 
			float *qt2yy_pyM, float *qt2yz_pyM, float *qt2xx_pzM, float *qt2xz_pzM, float *qt2yz_pzM, float *qt2zz_pzM,
			void **v2xMp, void **v2yMp, void **v2zMp, int *myid);


void set_deviceC(int *deviceID)
{
	cudaSetDevice(*deviceID);
    //printf("[CUDA] device set success!\n");
}

//===========================================================================
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
    //printf("[CUDA] allocation ...............");
	int nv2, nti, nth;
	cudaError_t cudaRes;

//  printf("lbx[1] = %d, lbx[0] = %d\n", lbx[1], lbx[0]);
//  printf("lby[1] = %d, lby[0] = %d\n", lby[1], lby[0]);
//  printf("nmat = %d\n", *nmat);
//  printf("mw1_pml1 = %d, mw2_pml1 = %d\n", *mw1_pml1, *mw2_pml1);
//  printf("mw1_pml = %d, mw2_pml = %d\n", *mw1_pml, *mw2_pml);
//  printf("nxtop = %d, nytop = %d, nztop = %d\n", *nxtop, *nytop, *nztop);
//  printf("nxbtm = %d, nybtm = %d, nzbtm = %d\n", *nxbtm, *nybtm, *nzbtm);
//  printf("nzbm1 = %d, nll = %d\n", *nzbm1, *nll);
	//debug-----------------
	totalTimeH2DV = 0.0f;
	totalTimeD2HV = 0.0f;
	totalTimeH2DS = 0.0f;
	totalTimeD2HS = 0.0f;
	totalTimeCompV = 0.0f;
	totalTimeCompS = 0.0f;

	//for inner_I
	cudaRes = cudaMalloc((void **)&nd1_velD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_vel");
	cudaRes = cudaMalloc((void **)&nd1_txyD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_txy");
	cudaRes = cudaMalloc((void **)&nd1_txzD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_txz");
	cudaRes = cudaMalloc((void **)&nd1_tyyD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_tyy");
	cudaRes = cudaMalloc((void **)&nd1_tyzD, sizeof(int) * 18);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, nd1_tyz");

	cudaRes = cudaMalloc((void **)&rhoD, sizeof(float) * (*nmat));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, rho");
	cudaRes = cudaMalloc((void **)&drvh1D, sizeof(float) * (*mw1_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, drvh1");
	cudaRes = cudaMalloc((void **)&drti1D, sizeof(float) * (*mw1_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, drti1");
	cudaRes = cudaMalloc((void **)&drth1D, sizeof(float) * (*mw1_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, drth1");

	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMalloc((void **)&damp1_xD, sizeof(float) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMalloc((void **)&damp1_yD, sizeof(float) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, damp1_y");
	}

	cudaRes = cudaMalloc((void **)&idmat1D, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, idmat1");
	cudaRes = cudaMalloc((void **)&dxi1D, sizeof(float) * 4 * (*nxtop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dxi1");
	cudaRes = cudaMalloc((void **)&dyi1D, sizeof(float) * 4 * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dyi1");
	cudaRes = cudaMalloc((void **)&dzi1D, sizeof(float) * 4 * (*nztop + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dzi1");
	cudaRes = cudaMalloc((void **)&dxh1D, sizeof(float) * 4 * (*nxtop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dxh1");
	cudaRes = cudaMalloc((void **)&dyh1D, sizeof(float) * 4 * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dyh1");
	cudaRes = cudaMalloc((void **)&dzh1D, sizeof(float) * 4 * (*nztop + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, dzh1");

	cudaRes = cudaMalloc((void **)&t1xxD, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xx");
	cudaRes = cudaMalloc((void **)&t1xyD, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xy");
	cudaRes = cudaMalloc((void **)&t1xzD, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xz");
	cudaRes = cudaMalloc((void **)&t1yyD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1yy");
	cudaRes = cudaMalloc((void **)&t1yzD, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1yz");
	cudaRes = cudaMalloc((void **)&t1zzD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1zz");
	
	if (lbx[1] >= lbx[0])
	{
		nti = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1];
		nth = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + 1 - lbx[0];
		cudaMalloc((void **)&t1xx_pxD,  sizeof(float) * (*nztop) * (nti) * (*nytop));
		cudaMalloc((void **)&t1xy_pxD,  sizeof(float) * (*nztop) * nth * (*nytop));
		cudaMalloc((void **)&t1xz_pxD,  sizeof(float) * (*nztop+1) * nth * (*nytop));
		cudaMalloc((void **)&t1yy_pxD,  sizeof(float) * (*nztop) * nti * (*nytop));

		cudaMalloc((void **)&qt1xx_pxD, sizeof(float) * (*nztop) * (nti) * (*nytop));
		cudaMalloc((void **)&qt1xy_pxD, sizeof(float) * (*nztop) * nth * (*nytop));
		cudaMalloc((void **)&qt1xz_pxD, sizeof(float) * (*nztop+1) * nth * (*nytop));
		cudaMalloc((void **)&qt1yy_pxD, sizeof(float) * (*nztop) * nti * (*nytop));
	}

	if (lby[1] >= lby[0])
	{
		nti = (lby[1] - lby[0] + 1) * (*mw1_pml) + lby[1];
		nth = (lby[1] - lby[0] + 1) * (*mw1_pml) + 1 - lby[0];
		cudaMalloc((void **)&t1xx_pyD,  sizeof(float) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&t1xy_pyD,  sizeof(float) * (*nztop) * (*nxtop) * nth);
		cudaMalloc((void **)&t1yy_pyD,  sizeof(float) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&t1yz_pyD,  sizeof(float) * (*nztop+1) * (*nxtop) * nth);
		cudaMalloc((void **)&qt1xx_pyD, sizeof(float) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&qt1xy_pyD, sizeof(float) * (*nztop) * (*nxtop) * nth);
		cudaMalloc((void **)&qt1yy_pyD, sizeof(float) * (*nztop) * (*nxtop) * nti);
		cudaMalloc((void **)&qt1yz_pyD, sizeof(float) * (*nztop+1) * (*nxtop) * nth);
	}

	cudaMalloc((void **)&qt1xxD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1xyD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1xzD, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1yyD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1yzD, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop));
	cudaMalloc((void **)&qt1zzD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop));

	cudaMalloc((void **)&clamdaD, sizeof(float) * (*nmat));
	cudaMalloc((void **)&cmuD,    sizeof(float) * (*nmat));
	cudaMalloc((void **)&epdtD,   sizeof(float) * (*nll));
	cudaMalloc((void **)&qwpD,    sizeof(float) * (*nmat));
	cudaMalloc((void **)&qwsD,    sizeof(float) * (*nmat));
	cudaMalloc((void **)&qwt1D,   sizeof(float) * (*nll));
	cudaMalloc((void **)&qwt2D,   sizeof(float) * (*nll));

	cudaRes = cudaMalloc((void **)&v1xD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1x");
	cudaRes = cudaMalloc((void **)&v1yD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1y");
	cudaRes = cudaMalloc((void **)&v1zD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1z");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw1_pml);
		cudaRes = cudaMalloc((void **)&v1x_pxD, sizeof(float) * (*nztop) * nv2 * (*nytop));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1x_px");
		cudaRes = cudaMalloc((void **)&v1y_pxD, sizeof(float) * (*nztop) * nv2 * (*nytop));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1y_px");
		cudaRes = cudaMalloc((void **)&v1z_pxD, sizeof(float) * (*nztop) * nv2 * (*nytop));
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw1_pml);
		cudaRes = cudaMalloc((void **)&v1x_pyD, sizeof(float) * (*nztop) * (*nxtop) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1x_py");
		cudaRes = cudaMalloc((void **)&v1y_pyD, sizeof(float) * (*nztop) * (*nxtop) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1y_py");
		cudaRes = cudaMalloc((void **)&v1z_pyD, sizeof(float) * (*nztop) * (*nxtop) * nv2);
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
	cudaRes = cudaMalloc((void **)&drvh2D, sizeof(float) * (*mw2_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, drvh2");
	cudaRes = cudaMalloc((void **)&drti2D, sizeof(float) * (*mw2_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, drti2");
	cudaRes = cudaMalloc((void **)&drth2D, sizeof(float) * (*mw2_pml1) * 2);
	CHECK_ERROR(cudaRes, "Allocate Device Memory, drth2");

	cudaRes = cudaMalloc((void **)&idmat2D, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm + 1));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMalloc((void **)&damp2_xD, sizeof(float) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMalloc((void **)&damp2_yD, sizeof(float) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, damp2_y");
	}
	cudaRes = cudaMalloc((void **)&damp2_zD, sizeof(float) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, damp2_z");

	cudaRes = cudaMalloc((void **)&dxi2D, sizeof(float) * 4 * (*nxbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dxi2");
	cudaRes = cudaMalloc((void **)&dyi2D, sizeof(float) * 4 * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dyi2");
	cudaRes = cudaMalloc((void **)&dzi2D, sizeof(float) * 4 * (*nzbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dzi2");
	cudaRes = cudaMalloc((void **)&dxh2D, sizeof(float) * 4 * (*nxbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dxh2");
	cudaRes = cudaMalloc((void **)&dyh2D, sizeof(float) * 4 * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dyh2");
	cudaRes = cudaMalloc((void **)&dzh2D, sizeof(float) * 4 * (*nzbtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, dzh2");

	cudaRes = cudaMalloc((void **)&t2xxD, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2xx");
	cudaRes = cudaMalloc((void **)&t2xyD, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2xy");
	cudaRes = cudaMalloc((void **)&t2xzD, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2xz");
	cudaRes = cudaMalloc((void **)&t2yyD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2yy");
	cudaRes = cudaMalloc((void **)&t2yzD, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2yz");
	cudaRes = cudaMalloc((void **)&t2zzD, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, t2zz");

	cudaMalloc((void **)&qt2xxD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2xyD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2xzD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2yyD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2yzD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2zzD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm));


	if (lbx[1] >= lbx[0])
	{
        nti = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + lbx[1];
        nth = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + 1 - lbx[0];
		cudaMalloc((void **)&t2xx_pxD, sizeof(float) * (*nzbtm) * nti * (*nybtm));
		cudaMalloc((void **)&t2xy_pxD, sizeof(float) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&t2xz_pxD, sizeof(float) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&t2yy_pxD, sizeof(float) * (*nzbtm) * nti * (*nybtm));

		cudaMalloc((void **)&qt2xx_pxD, sizeof(float) * (*nzbtm) * nti * (*nybtm));
		cudaMalloc((void **)&qt2xy_pxD, sizeof(float) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&qt2xz_pxD, sizeof(float) * (*nzbtm) * nth * (*nybtm));
		cudaMalloc((void **)&qt2yy_pxD, sizeof(float) * (*nzbtm) * nti * (*nybtm));
	}

	if (lby[1] >= lby[0])
	{
        nti = (lby[1] - lby[0] + 1) * (*mw2_pml) + lby[1];
        nth = (lby[1] - lby[0] + 1) * (*mw2_pml) + 1 - lby[0];
		cudaMalloc((void **)&t2xx_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&t2xy_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nth);
		cudaMalloc((void **)&t2yy_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&t2yz_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nth);

		cudaMalloc((void **)&qt2xx_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&qt2xy_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nth);
		cudaMalloc((void **)&qt2yy_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nti);
		cudaMalloc((void **)&qt2yz_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nth);
	}

	cudaMalloc((void **)&t2xx_pzD, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&t2xz_pzD, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&t2yz_pzD, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&t2zz_pzD, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm));

	cudaMalloc((void **)&qt2xx_pzD, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2xz_pzD, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2yz_pzD, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm));
	cudaMalloc((void **)&qt2zz_pzD, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm));

	cudaMalloc((void **)&v2xD,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3));
	cudaMalloc((void **)&v2yD,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3));
	cudaMalloc((void **)&v2zD,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3));

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw2_pml);
		cudaRes = cudaMalloc((void **)&v2x_pxD, sizeof(float) * (*nzbtm) * nv2 * (*nybtm));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2x_px");
		cudaRes = cudaMalloc((void **)&v2y_pxD, sizeof(float) * (*nzbtm) * nv2 * (*nybtm));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2y_px");
		cudaRes = cudaMalloc((void **)&v2z_pxD, sizeof(float) * (*nzbtm) * nv2 * (*nybtm));
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw2_pml);
		cudaRes = cudaMalloc((void **)&v2x_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2x_py");
		cudaRes = cudaMalloc((void **)&v2y_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2y_py");
		cudaRes = cudaMalloc((void **)&v2z_pyD, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2);
		CHECK_ERROR(cudaRes, "Allocate Device Memory, v2z_py");
	}

	cudaRes = cudaMalloc((void **)&v2x_pzD, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, v2x_pz");
	cudaRes = cudaMalloc((void **)&v2y_pzD, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, v2y_pz");
	cudaRes = cudaMalloc((void **)&v2z_pzD, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm));
	CHECK_ERROR(cudaRes, "Allocate Device Memory, v2z_pz");
    //printf("done!\n");

	return;
}

void cpy_h2d_velocityInputsCOneTime(int   *lbx,
						  int   *lby,
						  int   *nd1_vel,
						  float *rho,
						  float *drvh1,
						  float *drti1,
						  float *damp1_x,
						  float *damp1_y,
						  int	*idmat1,
						  float *dxi1,
						  float *dyi1,
						  float *dzi1,
						  float *dxh1,
						  float *dyh1,
						  float *dzh1,
						  float *t1xx,
						  float *t1xy,
						  float *t1xz,
						  float *t1yy,
						  float *t1yz,
						  float *t1zz,
						  float *v1x_px,
						  float *v1y_px,
						  float *v1z_px,
						  float *v1x_py,
						  float *v1y_py,
						  float *v1z_py,
						  int	*nd2_vel,
						  float *drvh2,
						  float *drti2,
						  int	*idmat2,
						  float *damp2_x,
						  float *damp2_y,
						  float *damp2_z,
						  float *dxi2,
						  float *dyi2,
						  float *dzi2,
						  float *dxh2,
						  float *dyh2,
						  float *dzh2,
						  float *t2xx,
						  float *t2xy,
						  float *t2xz,
						  float *t2yy,
						  float *t2yz,
						  float *t2zz,
						  float *v2x_px,
						  float *v2y_px,
						  float *v2z_px,
						  float *v2x_py,
						  float *v2y_py,
						  float *v2z_py,
						  float *v2x_pz,
						  float *v2y_pz,
						  float *v2z_pz,
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
    //printf("[CUDA] initial h2d cpy for velocity ........");
	cudaError_t cudaRes;
	int nv2;

    // int i;
    // for(i=0; i<(*nzbtm) * (*nxbtm + 3) * (*nybtm); i++)
    // {
        // printf("%f ", t2xy[i]);
    // }
    // printf("\n");

	//for inner_I
	cudaRes = cudaMemcpy(nd1_velD, nd1_vel, sizeof(int) * 18, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, nd1_vel");

	cudaRes = cudaMemcpy(rhoD, rho, sizeof(float) * (*nmat), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, rho");
	cudaRes = cudaMemcpy(drvh1D, drvh1, sizeof(float) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drvh1");
	cudaRes = cudaMemcpy(drti1D, drti1, sizeof(float) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drti1");

	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp1_xD, damp1_x, 
					sizeof(float) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp1_yD, damp1_y, 
					sizeof(float) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp1_y");
	}

	cudaRes = cudaMemcpy(idmat1D, idmat1, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, idmat1");
	cudaRes = cudaMemcpy(dxi1D, dxi1, sizeof(float) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxi1");
	cudaRes = cudaMemcpy(dyi1D, dyi1, sizeof(float) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyi1");
	cudaRes = cudaMemcpy(dzi1D, dzi1, sizeof(float) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzi1");
	cudaRes = cudaMemcpy(dxh1D, dxh1, sizeof(float) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxh1");
	cudaRes = cudaMemcpy(dyh1D, dyh1, sizeof(float) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyh1");
	cudaRes = cudaMemcpy(dzh1D, dzh1, sizeof(float) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzh1");

	cudaRes = cudaMemcpy(t1xxD, t1xx, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xx");
	cudaRes = cudaMemcpy(t1xyD, t1xy, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xy");
	cudaRes = cudaMemcpy(t1xzD, t1xz, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xz");
	cudaRes = cudaMemcpy(t1yyD, t1yy, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yy");
	cudaRes = cudaMemcpy(t1yzD, t1yz, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yz");
	cudaRes = cudaMemcpy(t1zzD, t1zz, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1zz");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw1_pml);
		cudaRes = cudaMemcpy(v1x_pxD, v1x_px, sizeof(float) * (*nztop) * nv2 * (*nytop), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1x_px");
		cudaRes = cudaMemcpy(v1y_pxD, v1y_px, sizeof(float) * (*nztop) * nv2 * (*nytop), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1y_px");
		cudaRes = cudaMemcpy(v1z_pxD, v1z_px, sizeof(float) * (*nztop) * nv2 * (*nytop), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw1_pml);
		cudaRes = cudaMemcpy(v1x_pyD, v1x_py, sizeof(float) * (*nztop) * (*nxtop) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1x_py");
		cudaRes = cudaMemcpy(v1y_pyD, v1y_py, sizeof(float) * (*nztop) * (*nxtop) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1y_py");
		cudaRes = cudaMemcpy(v1z_pyD, v1z_py, sizeof(float) * (*nztop) * (*nxtop) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1z_py");
	}


	//for inner_II
	cudaRes = cudaMemcpy(nd2_velD, nd2_vel, sizeof(int) * 18, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, nd2_vel");

	cudaRes = cudaMemcpy(drvh2D, drvh2, sizeof(float) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drvh2");
	cudaRes = cudaMemcpy(drti2D, drti2, sizeof(float) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, drti2");

	cudaRes = cudaMemcpy(idmat2D, idmat2, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm +1),  cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp2_xD, damp2_x, 
					sizeof(float) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp2_yD, damp2_y, 
					sizeof(float) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp2_y");
	}
	cudaRes = cudaMemcpy(damp2_zD, damp2_z, sizeof(float) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, damp2_z");

	cudaRes = cudaMemcpy(dxi2D, dxi2, sizeof(float) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxi2");
	cudaRes = cudaMemcpy(dyi2D, dyi2, sizeof(float) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyi2");
	cudaRes = cudaMemcpy(dzi2D, dzi2, sizeof(float) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzi2");
	cudaRes = cudaMemcpy(dxh2D, dxh2, sizeof(float) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dxh2");
	cudaRes = cudaMemcpy(dyh2D, dyh2, sizeof(float) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dyh2");
	cudaRes = cudaMemcpy(dzh2D, dzh2, sizeof(float) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, dzh2");

	cudaRes = cudaMemcpy(t2xxD, t2xx, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xx");
	cudaRes = cudaMemcpy(t2xyD, t2xy, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xy");
	cudaRes = cudaMemcpy(t2xzD, t2xz, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xz");
	cudaRes = cudaMemcpy(t2yyD, t2yy, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yy");
	cudaRes = cudaMemcpy(t2yzD, t2yz, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yz");
	cudaRes = cudaMemcpy(t2zzD, t2zz, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2zz");

	if (lbx[1] >= lbx[0])
	{
		nv2 = (lbx[1] - lbx[0] + 1) * (*mw2_pml);
		cudaRes = cudaMemcpy(v2x_pxD, v2x_px, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x_px");
		cudaRes = cudaMemcpy(v2y_pxD, v2y_px, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y_px");
		cudaRes = cudaMemcpy(v2z_pxD, v2z_px, sizeof(float) * (*nzbtm) * nv2 * (*nybtm), cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z_px");
	}

	if (lby[1] >= lby[0])
	{
		nv2 = (lby[1] - lby[0] + 1) * (*mw2_pml);
		cudaRes = cudaMemcpy(v2x_pyD, v2x_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x_py");
		cudaRes = cudaMemcpy(v2y_pyD, v2y_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y_py");
		cudaRes = cudaMemcpy(v2z_pyD, v2z_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nv2, cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z_py");
	}

	cudaRes = cudaMemcpy(v2x_pzD, v2x_pz, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x_pz");
	cudaRes = cudaMemcpy(v2y_pzD, v2y_pz, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y_pz");
	cudaRes = cudaMemcpy(v2z_pzD, v2z_pz, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z_pz");
    //printf("done!\n");

	return;
}

void cpy_h2d_velocityInputsC(float *t1xx,
							float *t1xy,
							float *t1xz,
							float *t1yy,
							float *t1yz,
							float *t1zz,
							float *t2xx,
							float *t2xy,
							float *t2xz,
							float *t2yy,
							float *t2yz,
							float *t2zz,
							int	*nxtop,		
							int	*nytop,		
							int *nztop,
							int	*nxbtm,		
							int	*nybtm,		
							int	*nzbtm)
{
    //printf("[CUDA] h2d cpy for input ..........");
	cudaError_t cudaRes;

	//for inner_I
	cudaRes = cudaMemcpy(t1xxD, t1xx, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xx");
	cudaRes = cudaMemcpy(t1xyD, t1xy, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xy");
	cudaRes = cudaMemcpy(t1xzD, t1xz, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1xz");
	cudaRes = cudaMemcpy(t1yyD, t1yy, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yy");
	cudaRes = cudaMemcpy(t1yzD, t1yz, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1yz");
	cudaRes = cudaMemcpy(t1zzD, t1zz, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t1zz");

	//for inner_II
	cudaRes = cudaMemcpy(t2xxD, t2xx, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xx");
	cudaRes = cudaMemcpy(t2xyD, t2xy, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xy");
	cudaRes = cudaMemcpy(t2xzD, t2xz, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2xz");
	cudaRes = cudaMemcpy(t2yyD, t2yy, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yy");
	cudaRes = cudaMemcpy(t2yzD, t2yz, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2yz");
	cudaRes = cudaMemcpy(t2zzD, t2zz, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice1, t2zz");

    //printf("done!\n");
	return;
}

//=====================================================================
void cpy_h2d_stressInputsCOneTime(int   *lbx,
						  int   *lby,
						  int   *nd1_txy,
						  int   *nd1_txz,
						  int   *nd1_tyy,
						  int   *nd1_tyz,
						  float *drti1,
						  float *drth1,
						  float *damp1_x,
						  float *damp1_y,
						  int	*idmat1,
						  float *dxi1,
						  float *dyi1,
						  float *dzi1,
						  float *dxh1,
						  float *dyh1,
						  float *dzh1,
						  float *v1x,
						  float *v1y,
						  float *v1z,
						  float *t1xx_px,
						  float *t1xy_px,
						  float *t1xz_px,
						  float *t1yy_px,
						  float *qt1xx_px,
						  float *qt1xy_px,
						  float *qt1xz_px,
						  float *qt1yy_px,
						  float *t1xx_py,
						  float *t1xy_py,
						  float *t1yy_py,
						  float *t1yz_py,
						  float *qt1xx_py,
						  float *qt1xy_py,
						  float *qt1yy_py,
						  float *qt1yz_py,
						  float *qt1xx,
						  float *qt1xy,
						  float *qt1xz,
						  float *qt1yy,
						  float *qt1yz,
						  float *qt1zz,
						  float *clamda,
						  float *cmu,
						  float *epdt,
						  float *qwp,
						  float *qws,
						  float *qwt1,
						  float *qwt2,
						  int   *nd2_txy,
						  int   *nd2_txz,
						  int   *nd2_tyy,
						  int   *nd2_tyz,
						  float *drti2,
						  float *drth2,
						  int	*idmat2,
						  float *damp2_x,
						  float *damp2_y,
						  float *damp2_z,
						  float *dxi2,
						  float *dyi2,
						  float *dzi2,
						  float *dxh2,
						  float *dyh2,
						  float *dzh2,
						  float *v2x,
						  float *v2y,
						  float *v2z,
						  float *qt2xx,
						  float *qt2xy,
						  float *qt2xz,
						  float *qt2yy,
						  float *qt2yz,
						  float *qt2zz,
						  float *t2xx_px,
						  float *t2xy_px,
						  float *t2xz_px,
						  float *t2yy_px,
						  float *qt2xx_px,
						  float *qt2xy_px,
						  float *qt2xz_px,
						  float *qt2yy_px,
						  float *t2xx_py,
						  float *t2xy_py,
						  float *t2yy_py,
						  float *t2yz_py,
						  float *qt2xx_py,
						  float *qt2xy_py,
						  float *qt2yy_py,
						  float *qt2yz_py,
						  float *t2xx_pz,
						  float *t2xz_pz,
						  float *t2yz_pz,
						  float *t2zz_pz,
						  float *qt2xx_pz,
						  float *qt2xz_pz,
						  float *qt2yz_pz,
						  float *qt2zz_pz,
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
    //printf("[CUDA] initial h2d cpy for stress ...........");
	cudaError_t cudaRes;
	int nti, nth;

	//for inner_I
	cudaRes = cudaMemcpy(nd1_txyD, nd1_txy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd1_txzD, nd1_txz, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd1_tyyD, nd1_tyy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd1_tyzD, nd1_tyz, sizeof(int) * 18, cudaMemcpyHostToDevice);

	cudaRes = cudaMemcpy(drti1D, drti1, sizeof(float) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drti1");
	cudaRes = cudaMemcpy(drth1D, drth1, sizeof(float) * (*mw1_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drth1");

	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp1_xD, damp1_x, 
					sizeof(float) * (*nztop + 1) * (*nytop) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp1_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp1_yD, damp1_y, 
					sizeof(float) * (*nztop + 1) * (*nxtop) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
		CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp1_y");
	}

	cudaRes = cudaMemcpy(idmat1D, idmat1, sizeof(int) * (*nztop + 2) * (*nxtop + 1) * (*nytop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, idmat1");
	cudaRes = cudaMemcpy(dxi1D, dxi1, sizeof(float) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxi1");
	cudaRes = cudaMemcpy(dyi1D, dyi1, sizeof(float) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyi1");
	cudaRes = cudaMemcpy(dzi1D, dzi1, sizeof(float) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzi1");
	cudaRes = cudaMemcpy(dxh1D, dxh1, sizeof(float) * 4 * (*nxtop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxh1");
	cudaRes = cudaMemcpy(dyh1D, dyh1, sizeof(float) * 4 * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyh1");
	cudaRes = cudaMemcpy(dzh1D, dzh1, sizeof(float) * 4 * (*nztop + 1), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzh1");

	cudaMemcpy(v1xD, v1x,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v1yD, v1y,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v1zD, v1z,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);

	if (lbx[1] >= lbx[0])
	{
		nti = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1];
		nth = (lbx[1] - lbx[0] + 1) * (*mw1_pml) + 1 - lbx[0];
		cudaMemcpy(t1xx_pxD, t1xx_px, sizeof(float) * (*nztop) * (nti) * (*nytop), cudaMemcpyHostToDevice);
		//debug
		//write_output(t1xx_px, (*nztop) * (nti) * (*nytop), "OUTPUT_ARRAYS/t1xx_px_cuda.txt");
		
		cudaMemcpy(t1xy_pxD, t1xy_px, sizeof(float) * (*nztop) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(t1xz_pxD, t1xz_px, sizeof(float) * (*nztop+1) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(t1yy_pxD, t1yy_px, sizeof(float) * (*nztop) * nti * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xx_pxD, qt1xx_px, sizeof(float) * (*nztop) * (nti) * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xy_pxD, qt1xy_px, sizeof(float) * (*nztop) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xz_pxD, qt1xz_px, sizeof(float) * (*nztop+1) * nth * (*nytop), cudaMemcpyHostToDevice);
		cudaMemcpy(qt1yy_pxD, qt1yy_px, sizeof(float) * (*nztop) * nti * (*nytop), cudaMemcpyHostToDevice);
	}

	if (lby[1] >= lby[0])
	{
		nti = (lby[1] - lby[0] + 1) * (*mw1_pml) + lby[1];
		nth = (lby[1] - lby[0] + 1) * (*mw1_pml) + 1 - lby[0];
		cudaMemcpy(t1xx_pyD,  t1xx_py,  sizeof(float) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t1xy_pyD,  t1xy_py,  sizeof(float) * (*nztop) * (*nxtop) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(t1yy_pyD,  t1yy_py,  sizeof(float) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t1yz_pyD,  t1yz_py,  sizeof(float) * (*nztop+1) * (*nxtop) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xx_pyD, qt1xx_py, sizeof(float) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1xy_pyD, qt1xy_py, sizeof(float) * (*nztop) * (*nxtop) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1yy_pyD, qt1yy_py, sizeof(float) * (*nztop) * (*nxtop) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt1yz_pyD, qt1yz_py, sizeof(float) * (*nztop+1) * (*nxtop) * nth, cudaMemcpyHostToDevice);
	}

	cudaMemcpy(qt1xxD, qt1xx, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1xyD, qt1xy, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1xzD, qt1xz, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1yyD, qt1yy, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1yzD, qt1yz, sizeof(float) * (*nztop+1) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	cudaMemcpy(qt1zzD, qt1zz, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);

	cudaMemcpy(clamdaD, clamda, sizeof(float) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(cmuD,    cmu,    sizeof(float) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(epdtD,   epdt,   sizeof(float) * (*nll),  cudaMemcpyHostToDevice);
	cudaMemcpy(qwpD,    qwp,    sizeof(float) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(qwsD,    qws,    sizeof(float) * (*nmat), cudaMemcpyHostToDevice);
	cudaMemcpy(qwt1D,   qwt1,   sizeof(float) * (*nll),  cudaMemcpyHostToDevice);
	cudaMemcpy(qwt2D,   qwt2,   sizeof(float) * (*nll),  cudaMemcpyHostToDevice);

	//for inner_II
	cudaRes = cudaMemcpy(nd2_txyD, nd2_txy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd2_txzD, nd2_txz, sizeof(int) * 18, cudaMemcpyHostToDevice); 
	cudaRes = cudaMemcpy(nd2_tyyD, nd2_tyy, sizeof(int) * 18, cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(nd2_tyzD, nd2_tyz, sizeof(int) * 18, cudaMemcpyHostToDevice);

	cudaRes = cudaMemcpy(drti2D, drti2, sizeof(float) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drti2");
	cudaRes = cudaMemcpy(drth2D, drth2, sizeof(float) * (*mw2_pml1) * 2, cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, drth2");

	cudaRes = cudaMemcpy(idmat2D, idmat2, sizeof(int) * (*nzbtm + 1) * (*nxbtm + 1) * (*nybtm +1),  cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, idmat2");
	
	if (lbx[1] >= lbx[0])
	{
		cudaRes = cudaMemcpy(damp2_xD, damp2_x, 
					sizeof(float) * (*nzbtm) * (*nybtm) * (lbx[1] - lbx[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp2_x");
	}

	if (lby[1] >= lby[0])
	{
		cudaRes = cudaMemcpy(damp2_yD, damp2_y, 
					sizeof(float) * (*nzbtm) * (*nxbtm) * (lby[1] - lby[0] + 1),
					cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp2_y");
	}
	cudaRes = cudaMemcpy(damp2_zD, damp2_z, sizeof(float) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, damp2_z");

	cudaRes = cudaMemcpy(dxi2D, dxi2, sizeof(float) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxi2");
	cudaRes = cudaMemcpy(dyi2D, dyi2, sizeof(float) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyi2");
	cudaRes = cudaMemcpy(dzi2D, dzi2, sizeof(float) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzi2");
	cudaRes = cudaMemcpy(dxh2D, dxh2, sizeof(float) * 4 * (*nxbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dxh2");
	cudaRes = cudaMemcpy(dyh2D, dyh2, sizeof(float) * 4 * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dyh2");
	cudaRes = cudaMemcpy(dzh2D, dzh2, sizeof(float) * 4 * (*nzbtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "InputDataCopyHostToDevice, dzh2");

	cudaMemcpy(v2xD, v2x,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2yD, v2y,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2zD, v2z,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);

	cudaMemcpy(qt2xxD, qt2xx, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xyD, qt2xy, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xzD, qt2xz, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2yyD, qt2yy, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2yzD, qt2yz, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2zzD, qt2zz, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);

	if (lbx[1] >= lbx[0])
	{
        nti = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + lbx[1];
        nth = (lbx[1] - lbx[0] + 1) * (*mw2_pml) + 1 - lbx[0];
		cudaMemcpy(t2xx_pxD, t2xx_px, sizeof(float) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(t2xy_pxD, t2xy_px, sizeof(float) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(t2xz_pxD, t2xz_px, sizeof(float) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(t2yy_pxD, t2yy_px, sizeof(float) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xx_pxD, qt2xx_px, sizeof(float) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xy_pxD, qt2xy_px, sizeof(float) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xz_pxD, qt2xz_px, sizeof(float) * (*nzbtm) * nth * (*nybtm), cudaMemcpyHostToDevice);
		cudaMemcpy(qt2yy_pxD, qt2yy_px, sizeof(float) * (*nzbtm) * nti * (*nybtm), cudaMemcpyHostToDevice);
	}

	if (lby[1] >= lby[0])
	{
        nti = (lby[1] - lby[0] + 1) * (*mw2_pml) + lby[1];
        nth = (lby[1] - lby[0] + 1) * (*mw2_pml) + 1 - lby[0];
		cudaMemcpy(t2xx_pyD, t2xx_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t2xy_pyD, t2xy_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(t2yy_pyD, t2yy_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(t2yz_pyD, t2yz_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xx_pyD, qt2xx_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2xy_pyD, qt2xy_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2yy_pyD, qt2yy_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nti, cudaMemcpyHostToDevice);
		cudaMemcpy(qt2yz_pyD, qt2yz_py, sizeof(float) * (*nzbtm) * (*nxbtm) * nth, cudaMemcpyHostToDevice);
	}

	cudaMemcpy(t2xx_pzD, t2xx_pz, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(t2xz_pzD, t2xz_pz, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(t2yz_pzD, t2yz_pz, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(t2zz_pzD, t2zz_pz, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xx_pzD, qt2xx_pz, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2xz_pzD, qt2xz_pz, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2yz_pzD, qt2yz_pz, sizeof(float) * (*mw2_pml1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	cudaMemcpy(qt2zz_pzD, qt2zz_pz, sizeof(float) * (*mw2_pml) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
    //printf("done!\n");

	return;
}

void cpy_h2d_stressInputsC(float *v1x,
						   float *v1y,
						   float *v1z,
						   float *v2x,
						   float *v2y,
						   float *v2z,
						   int	*nxtop,
						   int	*nytop,
						   int  *nztop,
						   int	*nxbtm,
						   int	*nybtm,
						   int	*nzbtm)
{
    //printf("[CUDA] h2d cpy for input ..............");
	cudaError_t cudaRes;

	//for inner_I
	cudaMemcpy(v1xD, v1x,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v1yD, v1y,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v1zD, v1z,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);

	//for inner_II
	cudaMemcpy(v2xD, v2x,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2yD, v2y,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	cudaMemcpy(v2zD, v2z,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
    //printf("done!\n");

	return;
}

//=====================================================================
void cpy_h2d_velocityOutputsC(float *v1x,
							  float *v1y,
							  float *v1z,
							  float *v2x,
							  float *v2y,
							  float *v2z,
							  int	*nxtop,	
							  int	*nytop,
							  int   *nztop,
							  int	*nxbtm,
							  int	*nybtm,
							  int	*nzbtm)
{
    //printf("[CUDA] h2d cpy for output .........");
	cudaError_t cudaRes;

	//for inner_I
	cudaRes = cudaMemcpy(v1xD, v1x,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1x");
	cudaRes = cudaMemcpy(v1yD, v1y,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1y");
	cudaRes = cudaMemcpy(v1zD, v1z,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v1z");

	//for inner_II
	cudaRes = cudaMemcpy(v2xD, v2x,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2x");
	cudaRes = cudaMemcpy(v2yD, v2y,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2y");
	cudaRes = cudaMemcpy(v2zD, v2z,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice1, v2z");
    //printf("done!\n");
	
	return;
}

//=====================================================================
void cpy_d2h_velocityOutputsC(float *v1x, 
							  float *v1y,
							  float *v1z,
							  float *v2x,
							  float *v2y,
							  float *v2z,
							  int	*nxtop,
							  int	*nytop,
							  int   *nztop,
							  int	*nxbtm,
							  int	*nybtm,
							  int	*nzbtm)
{
    //printf("[CUDA] d2h cpy for output .........");
	cudaError_t cudaRes;

	//for inner_I
	cudaRes = cudaMemcpy(v1x, v1xD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1x");
	cudaRes = cudaMemcpy(v1y, v1yD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1y");
	cudaRes = cudaMemcpy(v1z, v1zD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v1z");

	//for inner_II
	cudaRes = cudaMemcpy(v2x, v2xD,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2x");
	cudaRes = cudaMemcpy(v2y, v2yD,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, v2y");
	cudaRes = cudaMemcpy(v2z, v2zD,  sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost1, vzz");
    //printf("done!\n");

	return;
}

void cpy_h2d_stressOutputsC(float *t1xx,
						    float *t1xy,
						    float *t1xz,
						    float *t1yy,
						    float *t1yz,
						    float *t1zz,
						    float *t2xx,
						    float *t2xy,
						    float *t2xz,
						    float *t2yy,
						    float *t2yz,
						    float *t2zz,
						    int	  *nxtop,
						    int	  *nytop,
						    int   *nztop,
						    int	  *nxbtm,
						    int	  *nybtm,
						    int	  *nzbtm)
{
    //printf("[CUDA] h2d cpy for output ..............");
	cudaError_t cudaRes;
	int nth, nti;

	cudaRes = cudaMemcpy(t1xxD, t1xx, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1xx");
	cudaRes = cudaMemcpy(t1xyD, t1xy, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1xy");
	cudaRes = cudaMemcpy(t1xzD, t1xz, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1xz");
	cudaRes = cudaMemcpy(t1yyD, t1yy, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1yy");
	cudaRes = cudaMemcpy(t1yzD, t1yz, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1yz");
	cudaRes = cudaMemcpy(t1zzD, t1zz, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t1zz");

	//for inner_II
	cudaRes = cudaMemcpy(t2xxD, t2xx, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2xx");
	cudaRes = cudaMemcpy(t2xyD, t2xy, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2xy");
	cudaRes = cudaMemcpy(t2xzD, t2xz, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2xz");
	cudaRes = cudaMemcpy(t2yyD, t2yy, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2yy");
	cudaRes = cudaMemcpy(t2yzD, t2yz, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2yz");
	cudaRes = cudaMemcpy(t2zzD, t2zz, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyHostToDevice);
	CHECK_ERROR(cudaRes, "outputDataCopyHostToDevice, t2zz");
    //printf("done!\n");

	return;
}

void cpy_d2h_stressOutputsC(float *t1xx,
						    float *t1xy,
						    float *t1xz,
						    float *t1yy,
						    float *t1yz,
						    float *t1zz,
						    float *t2xx,
						    float *t2xy,
						    float *t2xz,
						    float *t2yy,
						    float *t2yz,
						    float *t2zz,
						    int	  *nxtop,
						    int	  *nytop,
						    int   *nztop,
						    int	  *nxbtm,
						    int	  *nybtm,
						    int	  *nzbtm)
{
    //printf("[CUDA] stress cpy d2h for output .....");
    // printf("\nnxtop=%d, nytop=%d, nztop=%d\n", *nxtop, *nytop, *nztop);
    // printf("nxbtm=%d, nybtm=%d, nzbtm=%d\n", *nxbtm, *nybtm, *nzbtm);
    
    cudaError_t cudaRes;

	cudaRes = cudaMemcpy(t1xx, t1xxD, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1xx");
	cudaRes = cudaMemcpy(t1xy, t1xyD, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1xy");
	cudaRes = cudaMemcpy(t1xz, t1xzD, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1xz");
	cudaRes = cudaMemcpy(t1yy, t1yyD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1yy");
	cudaRes = cudaMemcpy(t1yz, t1yzD, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1yz");
	cudaRes = cudaMemcpy(t1zz, t1zzD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t1zz");

	cudaRes = cudaMemcpy(t2xx, t2xxD, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2xx");
	cudaRes = cudaMemcpy(t2xy, t2xyD, sizeof(float) * (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2xy");
	cudaRes = cudaMemcpy(t2xz, t2xzD, sizeof(float) * (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2xz");
	cudaRes = cudaMemcpy(t2yy, t2yyD, sizeof(float) * (*nzbtm) * (*nxbtm) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2yy");
	cudaRes = cudaMemcpy(t2yz, t2yzD, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2yz");
	cudaRes = cudaMemcpy(t2zz, t2zzD, sizeof(float) * (*nzbtm + 1) * (*nxbtm) * (*nybtm), cudaMemcpyDeviceToHost);
	CHECK_ERROR(cudaRes, "outputDataCopyDeviceToHost, t2zz");

    //printf("done!\n");
    // int i;
    // for(i=0; i<(*nzbtm) * (*nxbtm + 3) * (*nybtm); i++)
    // {
        // //printf("%f ", t2xx[i]);
    // }
    // printf("\n");

	return;
}

void free_device_memC(int *lbx, int *lby)
{
	//debug---------------------------------------------------
	printf("[CUDA] id = %d, vel, H2D =, %.3f, D2H =, %.3f, comp =, %lf\n", procID, totalTimeH2DV, totalTimeD2HV, totalTimeCompV);
	printf("[CUDA] id = %d, str, H2D =, %.3f, D2H =, %.3f, comp =, %lf\n", procID, totalTimeH2DS, totalTimeD2HS, totalTimeCompS);
	//-------------------------------------------------
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
    //printf("[CUDA] memory space is freed.\n");
	
	return;
}

void compute_velocityC(int *nztop, int *nztm1, float *ca, int *lbx,
			 int *lby, int *nd1_vel, float *rhoM, float *drvh1M, float *drti1M,
             float *damp1_xM, float *damp1_yM, int *idmat1M, float *dxi1M, float *dyi1M,
             float *dzi1M, float *dxh1M, float *dyh1M, float *dzh1M, float *t1xxM,
             float *t1xyM, float *t1xzM, float *t1yyM, float *t1yzM, float *t1zzM,
             void **v1xMp, void **v1yMp, void **v1zMp, float *v1x_pxM, float *v1y_pxM,
             float *v1z_pxM, float *v1x_pyM, float *v1y_pyM, float *v1z_pyM, 
             int *nzbm1, int *nd2_vel, float *drvh2M, float *drti2M, 
             int *idmat2M, float *damp2_xM, float *damp2_yM, float *damp2_zM,
             float *dxi2M, float *dyi2M, float *dzi2M, float *dxh2M, float *dyh2M,
             float *dzh2M, float *t2xxM, float *t2xyM, float *t2xzM, float *t2yyM,
             float *t2yzM, float *t2zzM, void **v2xMp, void **v2yMp, void **v2zMp,
             float *v2x_pxM, float *v2y_pxM, float *v2z_pxM, float *v2x_pyM, 
             float *v2y_pyM, float *v2z_pyM, float *v2x_pzM, float *v2y_pzM,
             float *v2z_pzM, int *nmat,	int *mw1_pml1, int *mw2_pml1, 
             int *nxtop, int *nytop, int *mw1_pml, int *mw2_pml,
             int *nxbtm, int *nybtm, int *nzbtm, int *myid)
{
	
    //printf("[CUDA] velocity computation:\n");
	//difine the dimensions of different kernels
	int blockSizeX = 8;
	int blockSizeY = 8;

	float *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;

	// extract specific input/output pointers
	v1xM=(float *) *v1xMp;
	v1yM=(float *) *v1yMp;
	v1zM=(float *) *v1zMp;
	v2xM=(float *) *v2xMp;
	v2yM=(float *) *v2yMp;
	v2zM=(float *) *v2zMp;

	procID = *myid;
 
	gettimeofday(&t1, NULL);
	cpy_h2d_velocityInputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, 
				t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);


	cpy_h2d_velocityOutputsC(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);

	gettimeofday(&t2, NULL);
	tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
	totalTimeH2DV += tmpTime;

	gettimeofday(&t1, NULL);
	dim3 dimBlock(blockSizeX, blockSizeY);
	int gridSizeX1 = (nd1_vel[3] - nd1_vel[2])/blockSizeX + 1;
	int gridSizeY1 = (nd1_vel[9] - nd1_vel[8])/blockSizeY + 1;
	dim3 dimGrid1(gridSizeX1, gridSizeY1);
//	printf("myid = %d, grid1 = (%d, %d)\n", *myid, gridSizeX1, gridSizeY1);

	//CUDA code
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
	// printf("velocity_inner_IC called!\n");
	

	int gridSizeX2 = (nd1_vel[5] - nd1_vel[0])/blockSizeX + 1;
	int gridSizeY2 = (lbx[1] - lbx[0])/blockSizeY + 1;
	dim3 dimGrid2(gridSizeX2, gridSizeY2);
//	printf("myid = %d, grid2 = (%d, %d)\n", *myid, gridSizeX2, gridSizeY2);

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
		// printf("vel_PmlX_IC called!\n");	   
	}

	int gridSizeX3 = (lby[1] - lby[0])/blockSizeX + 1;
	int gridSizeY3 = (nd1_vel[11] - nd1_vel[6])/blockSizeY + 1;
	dim3 dimGrid3(gridSizeX3, gridSizeY3);
	
//	printf("myid = %d, grid3 = (%d, %d)\n", *myid, gridSizeX3, gridSizeY3);
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
		// printf("vel_PmlY_IC called!\n");
	}


	
	int gridSizeX4 = (nd2_vel[3] - nd2_vel[2])/blockSizeX + 1;
	int gridSizeY4 = (nd2_vel[9] - nd2_vel[8])/blockSizeY + 1;
	dim3 dimGrid4(gridSizeX4, gridSizeY4);
//	printf("myid = %d, grid4 = (%d, %d)\n", *myid, gridSizeX4, gridSizeY4);
	
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
	// printf("velocity_inner_IIC called!\n");

	int gridSizeX5 = (nd2_vel[5] - nd2_vel[0])/blockSizeX + 1;
	int gridSizeY5 = (lbx[1] - lbx[0])/blockSizeY + 1;
	dim3 dimGrid5(gridSizeX5, gridSizeY5);
//	printf("myid = %d, grid5 = (%d, %d)\n", *myid, gridSizeX5, gridSizeY5);
	
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
		// printf("vel_PmlX_IIC called!\n");
	}

	int gridSizeX6 = (lby[1] - lby[0])/blockSizeX + 1;
	int gridSizeY6 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
	dim3 dimGrid6(gridSizeX6, gridSizeY6);
//	printf("myid = %d, grid = (%d, %d)\n", *myid, gridSizeX6, gridSizeY6);

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
		// printf("vel_PmlY_IIC called!\n");
	}

	
	int gridSizeX7 = (nd2_vel[5] - nd2_vel[0])/blockSizeX + 1;
	int gridSizeY7 = (nd2_vel[11] - nd2_vel[6])/blockSizeY + 1;
	dim3 dimGrid7(gridSizeX7, gridSizeY7);
//	printf("myid = %d, grid7 = (%d, %d)\n", *myid, gridSizeX7, gridSizeY7);
	
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
	// printf("vel_PmlZ_IIC called!\n");


  cudaThreadSynchronize();
  gettimeofday(&t2, NULL);
  tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
  totalTimeCompV += tmpTime;

  gettimeofday(&t1, NULL);
	cpy_d2h_velocityOutputsC(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop,	nytop, nztop, nxbtm, nybtm, nzbtm);
  gettimeofday(&t2, NULL);
  tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
  totalTimeD2HV += tmpTime;
  
  // for debug
  // int size = (*nztop + 2) * (*nxtop + 3) * (*nytop + 3); 
  // write_output(v1xM, size, "OUTPUT_ARRAYS/v1xM.txt");
  // write_output(v1yM, size, "OUTPUT_ARRAYS/v1yM.txt");
  // write_output(v1zM, size, "OUTPUT_ARRAYS/v1zM.txt");
  // size = (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm + 3);
  // write_output(v2xM, size, "OUTPUT_ARRAYS/v2xM.txt");
  // write_output(v2yM, size, "OUTPUT_ARRAYS/v2yM.txt");
  // write_output(v2zM, size, "OUTPUT_ARRAYS/v2zM.txt");
  
 
    return;
}

#ifdef __cplusplus
}
#endif

__global__ void velocity_inner_IC(int	nztop,
					  int 	nztm1,
					  float ca,
					  int	*nd1_vel,
					  float *rhoM,
					  int   *idmat1M,
					  float *dxi1M,
					  float *dyi1M,
					  float *dzi1M,
					  float *dxh1M,
					  float *dyh1M,
					  float *dzh1M,
					  float *t1xxM,
					  float *t1xyM,
					  float *t1xzM,
					  float *t1yyM,
					  float *t1yzM,
					  float *t1zzM,
					  int	nxtop,		//dimension #
					  int	nytop,
					  float *v1xM,		//output
					  float *v1yM,
					  float *v1zM)
{
	int i, j, k, k3;
	float dtxz, dtyz, dtzz;
	j = blockIdx.x * blockDim.x + threadIdx.x + nd1_vel[2];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd1_vel[8];

	if (j > nd1_vel[3] || i > nd1_vel[9])
	{
		return;
	}

	for (k3 = 1; k3 <= 3; k3++)
	{
		k=k3;
		if(k3==3) k=nztop;
		if(k==1)
		{
			dtxz=(dzi1(3,k)-dzi1(1,k))*t1xz(2,i,j)+dzi1(4,k)*t1xz(3,i,j);
			dtyz=(dzi1(3,k)-dzi1(1,k))*t1yz(2,i,j)+dzi1(4,k)*t1yz(3,i,j);
			dtzz=dzh1(3,k)/ca*(35./8.*t1zz(k,i,j) -35./24.*t1zz(k+1,i,j)+
					 21./40.*t1zz(k+2,i,j)-5./56.*t1zz(k+3,i,j));
		}
		else if(k==2)
		{
			dtxz=dzi1(2,k)*t1xz(2,i,j)+dzi1(3,k)*t1xz(3,i,j)+dzi1(4,k)*t1xz(4,i,j);
			dtyz=dzi1(2,k)*t1yz(2,i,j)+dzi1(3,k)*t1yz(3,i,j)+dzi1(4,k)*t1yz(4,i,j);
			dtzz=dzh1(3,k)/ca*(-31./24.*t1zz(k-1,i,j) +29./24.*t1zz(k,i,j)-
					  3./40.*t1zz(k+1,i,j)+1./168.*t1zz(k+2,i,j));
		}
		else
		{
			dtxz=dzi1(2,k)/ca*(t1xz(k,i,j)-t1xz(k+1,i,j));
			dtyz=dzi1(2,k)/ca*(t1yz(k,i,j)-t1yz(k+1,i,j));
			dtzz=dzh1(2,k)/ca*(t1zz(k-1,i,j)-t1zz(k,i,j));
		}
		
	 	v1x(k,i,j)=v1x(k,i,j)+ 
			0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i+1,j)))*
			(dxi1(1,i)*t1xx(k,i-1,j)+dxi1(2,i)*t1xx(k,i,  j)+ 
			dxi1(3,i)*t1xx(k,i+1,j)+dxi1(4,i)*t1xx(k,i+2,j)+ 
			dyh1(1,j)*t1xy(k,i,j-2)+dyh1(2,j)*t1xy(k,i,j-1)+ 
			dyh1(3,j)*t1xy(k,i,j  )+dyh1(4,j)*t1xy(k,i,j+1)+dtxz);

		v1y(k,i,j)=v1y(k,i,j)+
			0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i,j+1)))* 
			(dxh1(1,i)*t1xy(k,i-2,j)+dxh1(2,i)*t1xy(k,i-1,j)+ 
			dxh1(3,i)*t1xy(k,i,  j)+dxh1(4,i)*t1xy(k,i+1,j)+ 
			dyi1(1,j)*t1yy(k,i,j-1)+dyi1(2,j)*t1yy(k,i,j  )+ 
			dyi1(3,j)*t1yy(k,i,j+1)+dyi1(4,j)*t1yy(k,i,j+2)+dtyz);

		v1z(k,i,j)=v1z(k,i,j)+
			0.5*(rho(idmat1(k,i,j))+rho(idmat1(k-1,i,j)))*
			(dxh1(1,i)*t1xz(k,i-2,j)+dxh1(2,i)*t1xz(k,i-1,j)+
			dxh1(3,i)*t1xz(k,i,  j)+dxh1(4,i)*t1xz(k,i+1,j)+
			dyh1(1,j)*t1yz(k,i,j-2)+dyh1(2,j)*t1yz(k,i,j-1)+
			dyh1(3,j)*t1yz(k,i,j  )+dyh1(4,j)*t1yz(k,i,j+1)+dtzz);
	}
	
	for (k = 3; k <=nztm1; k++)
	{
		v1x(k,i,j)=v1x(k,i,j)+ 
			0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i+1,j)))* 
			(dxi1(1,i)*t1xx(k,i-1,j)+dxi1(2,i)*t1xx(k,i,  j)+ 
			dxi1(3,i)*t1xx(k,i+1,j)+dxi1(4,i)*t1xx(k,i+2,j)+ 
			dyh1(1,j)*t1xy(k,i,j-2)+dyh1(2,j)*t1xy(k,i,j-1)+ 
			dyh1(3,j)*t1xy(k,i,j  )+dyh1(4,j)*t1xy(k,i,j+1)+ 
			dzi1(1,k)*t1xz(k-1,i,j)+dzi1(2,k)*t1xz(k,  i,j)+ 
			dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j));

		v1y(k,i,j)=v1y(k,i,j)+ 
			0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i,j+1)))* 
			(dxh1(1,i)*t1xy(k,i-2,j)+dxh1(2,i)*t1xy(k,i-1,j)+ 
			dxh1(3,i)*t1xy(k,i,  j)+dxh1(4,i)*t1xy(k,i+1,j)+ 
			dyi1(1,j)*t1yy(k,i,j-1)+dyi1(2,j)*t1yy(k,i,j  )+ 
			dyi1(3,j)*t1yy(k,i,j+1)+dyi1(4,j)*t1yy(k,i,j+2)+ 
			dzi1(1,k)*t1yz(k-1,i,j)+dzi1(2,k)*t1yz(k  ,i,j)+ 
			dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j));

		v1z(k,i,j)=v1z(k,i,j)+ 
			0.5*(rho(idmat1(k,i,j))+rho(idmat1(k-1,i,j)))* 
			(dxh1(1,i)*t1xz(k,i-2,j)+dxh1(2,i)*t1xz(k,i-1,j)+
			dxh1(3,i)*t1xz(k,i,  j)+dxh1(4,i)*t1xz(k,i+1,j)+ 
			dyh1(1,j)*t1yz(k,i,j-2)+dyh1(2,j)*t1yz(k,i,j-1)+ 
			dyh1(3,j)*t1yz(k,i,j  )+dyh1(4,j)*t1yz(k,i,j+1)+ 
			dzh1(1,k)*t1zz(k-2,i,j)+dzh1(2,k)*t1zz(k-1,i,j)+ 
			dzh1(3,k)*t1zz(k  ,i,j)+dzh1(4,k)*t1zz(k+1,i,j));
	}

 	return;
}


//-----------------------------------------------------------------------
__global__ void velocity_inner_IIC(float ca,
					   int	 *nd2_vel,
					   float *rhoM,
					   float *dxi2M,
					   float *dyi2M,
					   float *dzi2M,
					   float *dxh2M,
					   float *dyh2M,
					   float *dzh2M,
					   int 	 *idmat2M,
					   float *t2xxM,
					   float *t2xyM,
					   float *t2xzM,
					   float *t2yyM,
					   float *t2yzM,
					   float *t2zzM,
					   int   nxbtm,	//dimension #s
					   int   nybtm,
					   int   nzbtm,
					   float *v2xM,		//output
					   float *v2yM,
					   float *v2zM)
{
	int i, j, k;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_vel[2];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_vel[8];

	if (j > nd2_vel[3] || i > nd2_vel[9])
	{
		return;
	}
	
	//for (j = nd2_vel(3); j <= nd2_vel(4); j++)
	//for (j = nd2_vel[2]; j <= nd2_vel[3]; j++)
	//{
		//for (i = nd2_vel[8]; i <= nd2_vel[9]; i++)
		//{
	k=1;
	v2x(k,i,j)=v2x(k,i,j)+
		0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i+1,j)))*
		(dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+
		dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j)+
		dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+
		dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1)+
		dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j  )+
		dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j));

	v2y(k,i,j)=v2y(k,i,j)+ 
		0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i,j+1)))* 
		(dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+
		dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j)+
		dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+ 
		dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2)+ 
		dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+ 
		dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j));

	v2z(k,i,j)=v2z(k,i,j)+
		0.5*(rho(idmat2(k,i,j))+rho(idmat2(k-1,i,j)))*
		(dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+ 
		dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j)+ 
		dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+ 
		dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1)+ 
		dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j)));
	
	//for (k = 2; k <= nd2_vel(16); k++)
	for (k = 2; k <= nd2_vel[15]; k++)
	{
		v2x(k,i,j)=v2x(k,i,j)+ 
			0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i+1,j)))* 
			(dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+ 
			dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j)+ 
			dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+ 
			dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1)+ 
			dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j  )+ 
			dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j));

		v2y(k,i,j)=v2y(k,i,j)+ 
			0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i,j+1)))* 
			(dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+ 
			dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j)+ 
			dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+ 
			dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2)+ 
			dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+ 
			dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j));

		v2z(k,i,j)=v2z(k,i,j)+
			0.5*(rho(idmat2(k,i,j))+rho(idmat2(k-1,i,j)))* 
			(dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+ 
			dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j)+ 
			dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+ 
			dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1)+ 
			dzh2(1,k)*t2zz(k-2,i,j)+dzh2(2,k)*t2zz(k-1,i,j)+ 
			dzh2(3,k)*t2zz(k,  i,j)+dzh2(4,k)*t2zz(k+1,i,j));
	}
		//}
	//}
	
	return;
}


//-----------------------------------------------------------------------
__global__ void vel_PmlX_IC(float ca,
				int   lbx0,
				int   lbx1,
				int	  *nd1_vel,
				float *rhoM,
				float *drvh1M,
				float *drti1M,
				float *damp1_xM,
				int	  *idmat1M,
				float *dxi1M,
				float *dyi1M,
				float *dzi1M,
				float *dxh1M,
				float *dyh1M,
				float *dzh1M,
				float *t1xxM,
				float *t1xyM,
				float *t1xzM,
				float *t1yyM,
				float *t1yzM,
				float *t1zzM,
				int   mw1_pml1,	//dimension #
			    int   mw1_pml,
			    int   nxtop,
			    int   nytop,
			    int   nztop,
				float *v1xM,		//output
				float *v1yM,
				float *v1zM,
				float *v1x_pxM,
				float *v1y_pxM,
				float *v1z_pxM)
{
// !Compute the velocities in region of PML-x-I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
	int i,j,k,lb,ib,kb;
	float rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
        vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz,dtxy,dtyy,dtzy;

	j  = blockIdx.x * blockDim.x + threadIdx.x + nd1_vel[0];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;


	//int nv2x=(lbx(2) - lbx(1) + 1) * mw1_pml;
	int nv2x=(lbx1 - lbx0 + 1) * mw1_pml;
	
	//if ( lbx(1)>lbx(2) ) return;
	if (lbx0 > lbx1) 
	{
		return;
	}

	if (j > nd1_vel[5] || lb > lbx1)
	{
		return;
	}

	//calculate the value of ib
	ib = 0;
	for (k = lbx0; k < lb; k++)
	{
		for (i = nd1_vel[6+4*k]; i <= nd1_vel[7+4*k]; i++)
		{
			ib++;
		}
	}

	//for (j = nd1_vel(1); j <= nd1_vel(6); j++)
	//for (j = nd1_vel[0]; j <= nd1_vel[5]; j++)
	//{
		//ib=0;
		//for (lb = lbx(1); lb <= lbx(2); lb++)
		//for (lb = lbx[0]; lb <= lbx[1]; lb++)
		//{

	kb=0;
	//for (i = nd1_vel(7+4*lb); i <= nd1_vel(8+4*lb); i++)
	for (i = nd1_vel[6+4*lb]; i <= nd1_vel[7+4*lb]; i++)
	{
		kb=kb+1;
		ib=ib+1;
		rth=drvh1(kb,lb);
		rti=drti1(kb,lb);

		for (k = 1; k <= nztop; k++)
		{
			damp0=damp1_x(k,j,lb);
			dmpx2=1./(1.+rth*damp0);
			dmpx1=dmpx2*2.-1.;
			dmpyz2=1./(1.+rti*damp0);
			dmpyz1=dmpyz2*2.-1.;
			ro1=rho(idmat1(k,i,j));
			rox=0.5*(ro1+rho(idmat1(k,i+1,j)));
			roy=0.5*(ro1+rho(idmat1(k,i,j+1)));
			roz=0.5*(ro1+rho(idmat1(k-1,i,j)));
			vtmpx=v1x(k,i,j)-v1x_px(k,ib,j);
			vtmpy=v1y(k,i,j)-v1y_px(k,ib,j);
			vtmpz=v1z(k,i,j)-v1z_px(k,ib,j);

 			//if(j>nd1_vel(2) && j<nd1_vel(5))
 			if(j>nd1_vel[1] && j<nd1_vel[4])
			{
				dtxy=dyh1(1,j)*t1xy(k,i,j-2)+dyh1(2,j)*t1xy(k,i,j-1)+
					 dyh1(3,j)*t1xy(k,i,j  )+dyh1(4,j)*t1xy(k,i,j+1);
				dtyy=dyi1(1,j)*t1yy(k,i,j-1)+dyi1(2,j)*t1yy(k,i,j  )+
					 dyi1(3,j)*t1yy(k,i,j+1)+dyi1(4,j)*t1yy(k,i,j+2);
				dtzy=dyh1(1,j)*t1yz(k,i,j-2)+dyh1(2,j)*t1yz(k,i,j-1)+
					 dyh1(3,j)*t1yz(k,i,j  )+dyh1(4,j)*t1yz(k,i,j+1);
				if(k==1)
				{
				  dtxz=(dzi1(3,k)-dzi1(1,k))*t1xz(2,i,j)+dzi1(4,k)*t1xz(3,i,j);
				  dtyz=(dzi1(3,k)-dzi1(1,k))*t1yz(2,i,j)+dzi1(4,k)*t1yz(3,i,j);
				  dtzz=dzh1(3,k)/ca*(35./8.*t1zz(k,i,j)-35./24.*t1zz(k+1,i,j)+
					   21./40.*t1zz(k+2,i,j)-5./56.*t1zz(k+3,i,j));
				}
				else if(k==2)
				{
				  dtxz=dzi1(2,k)*t1xz(k,i,j)+ 
					   dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j);
				  dtyz=dzi1(2,k)*t1yz(k,i,j)+
					   dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j);
				  dtzz=dzh1(3,k)/ca*(-31./24.*t1zz(k-1,i,j)+29./24.*t1zz(k,i,j)-
					   3./40.*t1zz(k+1,i,j)+1./168.*t1zz(k+2,i,j));
				}
				else if(k==nztop)
				{
				  dtxz=dzi1(2,k)/ca*(t1xz(k,i,j)-t1xz(k+1,i,j));
				  dtyz=dzi1(2,k)/ca*(t1yz(k,i,j)-t1yz(k+1,i,j));
				  dtzz=dzh1(2,k)/ca*(t1zz(k-1,i,j)-t1zz(k,i,j));
				}
				else
				{
				  dtxz=dzi1(1,k)*t1xz(k-1,i,j)+dzi1(2,k)*t1xz(k,  i,j)+
					   dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j);
				  dtyz=dzi1(1,k)*t1yz(k-1,i,j)+dzi1(2,k)*t1yz(k  ,i,j)+
					   dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j);
				  dtzz=dzh1(1,k)*t1zz(k-2,i,j)+dzh1(2,k)*t1zz(k-1,i,j)+
					   dzh1(3,k)*t1zz(k  ,i,j)+dzh1(4,k)*t1zz(k+1,i,j);
				}

				vtmpx=vtmpx+(dtxy+dtxz)*rox;
				vtmpy=vtmpy+(dtyy+dtyz)*roy;
				vtmpz=vtmpz+(dtzy+dtzz)*roz;
			}

			v1x_px(k,ib,j)=v1x_px(k,ib,j)*dmpx1+dmpx2*rox*
						   dxi1(2,i)/ca*(t1xx(k,i,j)-t1xx(k,i+1,j));
			v1x(k,i,j)=vtmpx+v1x_px(k,ib,j);

			v1y_px(k,ib,j)=v1y_px(k,ib,j)*dmpyz1+dmpyz2*roy*
						   dxh1(2,i)/ca*(t1xy(k,i-1,j)-t1xy(k,i,j));
			v1y(k,i,j)=vtmpy+v1y_px(k,ib,j);

			v1z_px(k,ib,j)=v1z_px(k,ib,j)*dmpyz1+dmpyz2*roz*
						   dxh1(2,i)/ca*(t1xz(k,i-1,j)-t1xz(k,i,j));
			v1z(k,i,j)=vtmpz+v1z_px(k,ib,j);
		}
	}
		//}
	//}
	
	return;
}


//-----------------------------------------------------------------------
__global__ void vel_PmlY_IC(int  nztop,
				float ca,
				int	  lby0,
				int   lby1,
				int   *nd1_vel,
				float *rhoM,
				float *drvh1M,
				float *drti1M,
				int   *idmat1M,
				float *damp1_yM,
				float *dxi1M,
				float *dyi1M,
				float *dzi1M,
				float *dxh1M,
				float *dyh1M,
				float *dzh1M,
				float *t1xxM,
				float *t1xyM,
				float *t1xzM,
				float *t1yyM,
				float *t1yzM,
				float *t1zzM,
				int   mw1_pml1, //dimension #s
				int   mw1_pml,
				int   nxtop,
				int   nytop,
				float *v1xM,		//output
				float *v1yM,
				float *v1zM,
				float *v1x_pyM,
				float *v1y_pyM,
				float *v1z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	float rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
        dtxz,dtyz,dtzz,vtmpx,vtmpy,vtmpz;

	//if( lby(1)>lby(2) ) 
	if( lby0>lby1 ) 
 		return;

	lb = blockDim.x * blockIdx.x + threadIdx.x + lby0;
	i  = blockDim.y * blockIdx.y + threadIdx.y + nd1_vel[6];

	if (lb > lby1 || i > nd1_vel[11])
	{
		return;
	}

	jbIni = 0;
	for (k = lby0; k < lb; k++)
	{
		for (j = nd1_vel[4*k]; j <= nd1_vel[1+4*k]; j++)
		{
			jbIni++;
		}
	}

	jb = jbIni;
	kb = 0;

	//for (lb = lby(1); lb <= lby(2); lb++)
	//for (lb = lby0; lb <= lby1; lb++)
	//{
	//	kb=0;

	//	//for (i = nd1_vel(7); i <= nd1_vel(12); i++)
	//	for (i = nd1_vel[6]; i <= nd1_vel[11]; i++)
	//	{
			//for (j = nd1_vel(1+4*lb); j <= nd1_vel(2+4*lb); j++)
	for (j = nd1_vel[4*lb]; j <= nd1_vel[1+4*lb]; j++)
	{
		kb=kb+1;
		jb=jb+1;
		rth=drvh1(kb,lb);
		rti=drti1(kb,lb);

		for (k = 1; k <= nztop; k++)
		{
			damp0=damp1_y(k,i,lb);
			dmpy2=1./(1.+rth*damp0);
			dmpy1=dmpy2*2.-1.;
			dmpxz2=1./(1.+rti*damp0);
			dmpxz1=dmpxz2*2.-1.;
			ro1=rho(idmat1(k,i,j));
			rox=0.5*(ro1+rho(idmat1(k,i+1,j)));
			roy=0.5*(ro1+rho(idmat1(k,i,j+1)));
			roz=0.5*(ro1+rho(idmat1(k-1,i,j)));

			if(k==1)
			{
				dtxz=(dzi1(3,k)-dzi1(1,k))*t1xz(2,i,j)+dzi1(4,k)*t1xz(3,i,j);
				dtyz=(dzi1(3,k)-dzi1(1,k))*t1yz(2,i,j)+dzi1(4,k)*t1yz(3,i,j);
				dtzz=dzh1(3,k)/ca*(35./8.*t1zz(k,i,j)-35./24.*t1zz(k+1,i,j)+
					21./40.*t1zz(k+2,i,j)-5./56.*t1zz(k+3,i,j));
			}
			else if(k==2)
			{
				dtxz=dzi1(2,k)*t1xz(k,i,j)+ 
					dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j);
				dtyz=dzi1(2,k)*t1yz(k,i,j)+
					dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j);
				dtzz=dzh1(3,k)/ca*(-31./24.*t1zz(k-1,i,j)+29./24.*t1zz(k,i,j)-
					3./40.*t1zz(k+1,i,j)+1./168.*t1zz(k+2,i,j));
			}
			else if(k==nztop)
			{
				dtxz=dzi1(2,k)/ca*(t1xz(k,i,j)-t1xz(k+1,i,j));
				dtyz=dzi1(2,k)/ca*(t1yz(k,i,j)-t1yz(k+1,i,j));
				dtzz=dzh1(2,k)/ca*(t1zz(k-1,i,j)-t1zz(k,i,j));
			}
			else
			{
				dtxz=dzi1(1,k)*t1xz(k-1,i,j)+dzi1(2,k)*t1xz(k,  i,j)+
					dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j);
				dtyz=dzi1(1,k)*t1yz(k-1,i,j)+dzi1(2,k)*t1yz(k  ,i,j)+
					dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j);
				dtzz=dzh1(1,k)*t1zz(k-2,i,j)+dzh1(2,k)*t1zz(k-1,i,j)+
					dzh1(3,k)*t1zz(k  ,i,j)+dzh1(4,k)*t1zz(k+1,i,j);
			}
			vtmpx=v1x(k,i,j)-v1x_py(k,i,jb)+dtxz*rox;
			vtmpy=v1y(k,i,j)-v1y_py(k,i,jb)+dtyz*roy;
			vtmpz=v1z(k,i,j)-v1z_py(k,i,jb)+dtzz*roz;

			//if(i>nd1_vel(8) && i<nd1_vel(11))
			if(i>nd1_vel[7] && i<nd1_vel[10])
			{
				vtmpx=vtmpx+
					rox*(dxi1(1,i)*t1xx(k,i-1,j)+dxi1(2,i)*t1xx(k,i,  j)+
					dxi1(3,i)*t1xx(k,i+1,j)+dxi1(4,i)*t1xx(k,i+2,j));
				vtmpy=vtmpy+ 
					roy*(dxh1(1,i)*t1xy(k,i-2,j)+dxh1(2,i)*t1xy(k,i-1,j)+
					dxh1(3,i)*t1xy(k,i,  j)+dxh1(4,i)*t1xy(k,i+1,j));
				vtmpz=vtmpz+
					roz*(dxh1(1,i)*t1xz(k,i-2,j)+dxh1(2,i)*t1xz(k,i-1,j)+
					dxh1(3,i)*t1xz(k,i,  j)+dxh1(4,i)*t1xz(k,i+1,j));
			}

			v1x_py(k,i,jb)=v1x_py(k,i,jb)*dmpxz1+dmpxz2*
				rox*dyh1(2,j)/ca*(t1xy(k,i,j-1)-t1xy(k,i,j));
			v1x(k,i,j)=vtmpx+v1x_py(k,i,jb);

			v1y_py(k,i,jb)=v1y_py(k,i,jb)*dmpy1+dmpy2*
				roy*dyi1(2,j)/ca*(t1yy(k,i,j)-t1yy(k,i,j+1));
			v1y(k,i,j)=vtmpy+v1y_py(k,i,jb);

			v1z_py(k,i,jb)=v1z_py(k,i,jb)*dmpxz1+dmpxz2*
				roz*dyh1(2,j)/ca*(t1yz(k,i,j-1)-t1yz(k,i,j));
			v1z(k,i,j)=vtmpz+v1z_py(k,i,jb);
		}
	}
	//}
	//}

 	return;
}


//-----------------------------------------------------------------------
__global__ void vel_PmlX_IIC(int   nzbm1,
				 float ca,
				 int   lbx0,
				 int   lbx1,
				 int   *nd2_vel,
				 float *drvh2M,
				 float *drti2M,
				 float *rhoM,
				 float *damp2_xM,
				 int   *idmat2M,
				 float *dxi2M,
				 float *dyi2M,
				 float *dzi2M,
				 float *dxh2M,
				 float *dyh2M,
				 float *dzh2M,
				 float *t2xxM,
				 float *t2xyM,
				 float *t2xzM,
				 float *t2yyM,
				 float *t2yzM,
				 float *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 float *v2xM,	//output
				 float *v2yM,
				 float *v2zM,
				 float *v2x_pxM,
				 float *v2y_pxM,
				 float *v2z_pxM)
{
	int i,j,k,lb,ib,kb;
	float rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
		vtmpx,vtmpy,vtmpz,dtxy,dtyy,dtzy,dtxz,dtyz,dtzz;

	//int nv2y = (lbx(2) - lbx(1) + 1) * mw2_pml;
	int nv2y = (lbx1 - lbx0 + 1) * mw2_pml;

	//if ( lbx(1)>lbx(2) ) return;
	if ( lbx0>lbx1 ) return;

	j  = blockIdx.x * blockDim.x + threadIdx.x + nd2_vel[0];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

	if (j > nd2_vel[5] || lb > lbx1)
	{
		return;
	}

	ib = 0;
	for (k = lbx0; k < lb; k++)
	{
		for (i = nd2_vel[6+4*k]; i <= nd2_vel[7+4*k]; i++)
		{
			ib++;
		}
	}
	
	//for (j = nd2_vel(1); j <= nd2_vel(6); j++)
	//for (j = nd2_vel[0]; j <= nd2_vel[5]; j++)
	//{
		//ib=0;
		//for (lb = lbx(1); lb <= lbx(2); lb++)
		//for (lb = lbx0; lb <= lbx1; lb++)
		//{
	kb=0;
	//for (i = nd2_vel(7+4*lb); i <= nd2_vel(8+4*lb); i++)
	for (i = nd2_vel[6+4*lb]; i <= nd2_vel[7+4*lb]; i++)
	{
		kb=kb+1;
		ib=ib+1;
		rth=drvh2(kb,lb);
		rti=drti2(kb,lb);

		for (k = 1; k <= nzbm1; k++)
		{
			damp0=damp2_x(k,j,lb);
			dmpx2=1./(1.+rth*damp0);
			dmpx1=dmpx2*2.-1.;
			dmpyz2=1./(1.+rti*damp0);
			dmpyz1=dmpyz2*2.-1.;
			ro1=rho(idmat2(k,i,j));
			rox=0.5*(ro1+rho(idmat2(k,i+1,j)));
			roy=0.5*(ro1+rho(idmat2(k,i,j+1)));
			roz=0.5*(ro1+rho(idmat2(k-1,i,j)));
			vtmpx=v2x(k,i,j)-v2x_px(k,ib,j);
			vtmpy=v2y(k,i,j)-v2y_px(k,ib,j);
			vtmpz=v2z(k,i,j)-v2z_px(k,ib,j);

			//if(j>nd2_vel(2) && j<nd2_vel(5))
			if(j>nd2_vel[1] && j<nd2_vel[4])
			{
				dtxy=dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+
					dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1);
				dtyy=dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+
					dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2);
				dtzy=dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+
					dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1);

				if(k==1)
				{
					dtxz=dzi2(2,k)/ca*(t2xz(k,i,j)-t2xz(k+1,i,j));
					dtyz=dzi2(2,k)/ca*(t2yz(k,i,j)-t2yz(k+1,i,j));
					dtzz=dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j));
				}
				//else if(k<nd2_vel(17))
				else if(k<nd2_vel[16])
				{
					dtxz=dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j)+
						dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j);
					dtyz=dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+
						dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j);
					dtzz=dzh2(1,k)*t2zz(k-2,i,j)+dzh2(2,k)*t2zz(k-1,i,j)+
						dzh2(3,k)*t2zz(k,  i,j)+dzh2(4,k)*t2zz(k+1,i,j);
				}
				else
				{
					dtxz=0.0;
					dtyz=0.0;
					dtzz=0.0;
				}
				vtmpx=vtmpx+(dtxy+dtxz)*rox;
				vtmpy=vtmpy+(dtyy+dtyz)*roy;
				vtmpz=vtmpz+(dtzy+dtzz)*roz;
			}

			v2x_px(k,ib,j)=v2x_px(k,ib,j)*dmpx1+dmpx2*
				rox*dxi2(2,i)/ca*(t2xx(k,i,j)-t2xx(k,i+1,j));
			v2x(k,i,j)=vtmpx+v2x_px(k,ib,j);

			v2y_px(k,ib,j)=v2y_px(k,ib,j)*dmpyz1+dmpyz2*
				roy*dxh2(2,i)/ca*(t2xy(k,i-1,j)-t2xy(k,i,j));
			v2y(k,i,j)=vtmpy+v2y_px(k,ib,j);

			v2z_px(k,ib,j)=v2z_px(k,ib,j)*dmpyz1+dmpyz2*
				roz*dxh2(2,i)/ca*(t2xz(k,i-1,j)-t2xz(k,i,j));
			v2z(k,i,j)=vtmpz+v2z_px(k,ib,j);
		}
	}
		//}
	//}

	return;
}


//-----------------------------------------------------------------------
__global__ void vel_PmlY_IIC(int   nzbm1,
				 float ca,
				 int   lby0,
				 int   lby1,
				 int   *nd2_vel,
				 float *drvh2M,
				 float *drti2M,
				 float *rhoM,
				 float *damp2_yM,
				 int   *idmat2M,
				 float *dxi2M,
				 float *dyi2M,
				 float *dzi2M,
				 float *dxh2M,
				 float *dyh2M,
				 float *dzh2M,
				 float *t2xxM,
				 float *t2xyM,
				 float *t2xzM,
				 float *t2yyM,
				 float *t2yzM,
				 float *t2zzM,
				 int   mw2_pml1,
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 float *v2xM,		//output
				 float *v2yM,
				 float *v2zM,
				 float *v2x_pyM,
				 float *v2y_pyM,
				 float *v2z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	float rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
		   vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz;

	//if( lby(1)>lby(2) ) return;
	if( lby0>lby1 ) 
	{
		return;
	}

	lb = blockIdx.x * blockDim.x + threadIdx.x + lby0;
	i  = blockIdx.y * blockDim.y + threadIdx.y + nd2_vel[6];

	if (lb > lby1 || i > nd2_vel[11])
	{
		return;
	}

	jbIni = 0;
	for (j = lby0; j < lb; j++)
	{
		for (k = nd2_vel[4*j]; k <= nd2_vel[1+4*j]; k++)
		{
			jbIni++;
		}
	}

	jb = jbIni;
	kb = 0;

	//for (lb = lby(1); lb <= lby(2); lb++)
	//for (lb = lby0; lb <= lby1; lb++)
	//{
		//kb=0;

		//for (i = nd2_vel(7); i <= nd2_vel(12); i++)
		//for (i = nd2_vel[6]; i <= nd2_vel[11]; i++)
		//{
			//for (j = nd2_vel(1+4*lb); j <= nd2_vel(2+4*lb); j++)
	for (j = nd2_vel[4*lb]; j <= nd2_vel[1+4*lb]; j++)
	{
		kb=kb+1;
		jb=jb+1;
		rth=drvh2(kb,lb);
		rti=drti2(kb,lb);

		for (k = 1; k <= nzbm1; k++)
		{
			damp0=damp2_y(k,i,lb);
			dmpy2=1./(1.+rth*damp0);
			dmpy1=dmpy2*2.-1.0;
			dmpxz2=1./(1.+rti*damp0);
			dmpxz1=dmpxz2*2.-1.;
			ro1=rho(idmat2(k,i,j));
			rox=0.5*(ro1+rho(idmat2(k,i+1,j)));
			roy=0.5*(ro1+rho(idmat2(k,i,j+1)));
			roz=0.5*(ro1+rho(idmat2(k-1,i,j)));
			vtmpx=v2x(k,i,j)-v2x_py(k,i,jb);
			vtmpy=v2y(k,i,j)-v2y_py(k,i,jb);
			vtmpz=v2z(k,i,j)-v2z_py(k,i,jb);
			//if(k<nd2_vel(17))
			if(k<nd2_vel[16])
			{
				if(k>1)
				{
					dtxz=dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j)+
						dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j);
					dtyz=dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+
						dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j);
					dtzz=dzh2(1,k)*t2zz(k-2,i,j)+dzh2(2,k)*t2zz(k-1,i,j)+
						dzh2(3,k)*t2zz(k,  i,j)+dzh2(4,k)*t2zz(k+1,i,j);
				}
				else
				{
					dtxz=dzi2(2,k)/ca*(t2xz(k,i,j)-t2xz(k+1,i,j));
					dtyz=dzi2(2,k)/ca*(t2yz(k,i,j)-t2yz(k+1,i,j));
					dtzz=dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j));
				}

				//if(i>nd2_vel(8) && i<nd2_vel(11))
				if(i>nd2_vel[7] && i<nd2_vel[10])
				{
					vtmpx=vtmpx+rox*(dtxz+
						dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+ 
						dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j));

					vtmpy=vtmpy+roy*(dtyz+
						dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+
						dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j));
					
					vtmpz=vtmpz+roz*(dtzz+
						dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+
						dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j));
				}
				else
				{
					vtmpx=vtmpx+rox*dtxz;
					vtmpy=vtmpy+roy*dtyz;
					vtmpz=vtmpz+roz*dtzz;
				}
			}
			else
			{
				//if(i>nd2_vel(8) && i<nd2_vel(11))
				if(i>nd2_vel[7] && i<nd2_vel[10])
				{
					vtmpx=vtmpx+rox*
						(dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+
						dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j));

					vtmpy=vtmpy+ roy*
						(dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+
						dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j));
					
					vtmpz=vtmpz+ roz*
						(dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+
						dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j));
				}
			}

			v2x_py(k,i,jb)=v2x_py(k,i,jb)*dmpxz1+dmpxz2*rox*
				dyh2(2,j)/ca*(t2xy(k,i,j-1)-t2xy(k,i,j));
			v2x(k,i,j)=vtmpx+v2x_py(k,i,jb);

			v2y_py(k,i,jb)=v2y_py(k,i,jb)*dmpy1+dmpy2*roy*
				dyi2(2,j)/ca*(t2yy(k,i,j)-t2yy(k,i,j+1));
			v2y(k,i,j)=vtmpy+v2y_py(k,i,jb);

			v2z_py(k,i,jb)=v2z_py(k,i,jb)*dmpxz1+dmpxz2*roz*
				dyh2(2,j)/ca*(t2yz(k,i,j-1)-t2yz(k,i,j));
			v2z(k,i,j)=vtmpz+v2z_py(k,i,jb);
		}
	}
		//}
	//}
 
	return;
}

//-----------------------------------------------------------------------
__global__ void vel_PmlZ_IIC(int   nzbm1,
				 float ca,
				 int   *nd2_vel,
				 float *drvh2M,
				 float *drti2M,
				 float *rhoM,
				 float *damp2_zM,
				 int   *idmat2M,
				 float *dxi2M,
				 float *dyi2M,
				 float *dzi2M,
				 float *dxh2M,
				 float *dyh2M,
				 float *dzh2M,
				 float *t2xxM,
				 float *t2xyM,
				 float *t2xzM,
				 float *t2yyM,
				 float *t2yzM,
				 float *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 float *v2xM,		//output
				 float *v2yM,
				 float *v2zM,
				 float *v2x_pzM,
				 float *v2y_pzM,
				 float *v2z_pzM)
{
	int i,j,k,kb;
	float damp0,dmpz2,dmpz1,dmpxy2,dmpxy1,ro1,rox,roy,roz,vtmpx,vtmpy,vtmpz;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_vel[0];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_vel[6];

	if (j > nd2_vel[5] || i > nd2_vel[11])
	{
		return;
	}

	//for (j = nd2_vel(1); j <= nd2_vel(6); j++)
	//for (j = nd2_vel[0]; j <= nd2_vel[5]; j++)
	//{
		//for (i = nd2_vel(7); i <= nd2_vel(12); i++)
		//for (i = nd2_vel[6]; i <= nd2_vel[11]; i++)
		//{
	kb=0;
	damp0=damp2_z(i,j);

	//for (k = nd2_vel(17); k <= nzbm1; k++)
	for (k = nd2_vel[16]; k <= nzbm1; k++)
	{
		kb=kb+1;
		dmpz2=1./(1.+damp0*drti2(kb,1));
		dmpz1=dmpz2*2.-1.;
		dmpxy2=1./(1.+damp0*drvh2(kb,1));
		dmpxy1=dmpxy2*2.-1.;
		ro1=rho(idmat2(k,i,j));
		rox=0.5*(ro1+rho(idmat2(k,i+1,j)));
		roy=0.5*(ro1+rho(idmat2(k,i,j+1)));
		roz=0.5*(ro1+rho(idmat2(k-1,i,j)));
		vtmpx=v2x(k,i,j)-v2x_pz(kb,i,j);
		vtmpy=v2y(k,i,j)-v2y_pz(kb,i,j);
		vtmpz=v2z(k,i,j)-v2z_pz(kb,i,j);

		//if(j>nd2_vel(2) && j<nd2_vel(5) &&
		//   i>nd2_vel(8) && i<nd2_vel(11))
		if(j>nd2_vel[1] && j<nd2_vel[4] &&
		   i>nd2_vel[7] && i<nd2_vel[10])
		{
			vtmpx=vtmpx+rox*
				(dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+
				dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j)+
				dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+
				dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1));

			vtmpy=vtmpy+roy*
			(dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+
				dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j)+
				dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+
				dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2));

			vtmpz=vtmpz+roz*
				(dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+
				dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j)+
				dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+
				dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1));
		}

		v2x_pz(kb,i,j)=v2x_pz(kb,i,j)*dmpxy1+dmpxy2*rox*
			dzi2(2,k)/ca*(t2xz(k,i,j)-t2xz(k+1,i,j));
		v2x(k,i,j)=vtmpx+v2x_pz(kb,i,j);

		v2y_pz(kb,i,j)=v2y_pz(kb,i,j)*dmpxy1+dmpxy2*roy*
			dzi2(2,k)/ca*(t2yz(k,i,j)-t2yz(k+1,i,j));
		v2y(k,i,j)=vtmpy+v2y_pz(kb,i,j);

		v2z_pz(kb,i,j)=v2z_pz(kb,i,j)*dmpz1+dmpz2*roz*
			dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j));
		v2z(k,i,j)=vtmpz+v2z_pz(kb,i,j);
	}
		//}
	//}

	return;
}


//stress computation----------------------------------------------

__global__ void stress_norm_xy_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   int *nd1_tyy,
					   int *idmat1M,
					   float ca,
					   float *clamdaM,
					   float *cmuM,
					   float *epdtM,
					   float *qwpM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dxh1M,
					   float *dyh1M,
					   float *dxi1M,
					   float *dyi1M,
					   float *dzi1M,
					   float *t1xxM,
					   float *t1xyM,
					   float *t1yyM,
					   float *t1zzM,
					   float *qt1xxM,
					   float *qt1xyM,
					   float *qt1yyM,
					   float *qt1zzM,
					   float *v1xM,
					   float *v1yM,
					   float *v1zM)
{
	int i,j,k,jkq,kodd,inod,irw;
	float sxx,syy,szz,sxy,qxx,qyy,qzz,qxy,cusxy,sss,cl,sm2,pm,et,et1,wtp,wts;

    j = blockIdx.x * blockDim.x + threadIdx.x + nd1_tyy[2];
    i = blockIdx.y * blockDim.y + threadIdx.y + nd1_tyy[8];

    if (j > nd1_tyy[3] || i > nd1_tyy[9])
    {
        return;
    }
	
//	for (j = nd1_tyy[2]; j <= nd1_tyy[3]; j++)
//	{
	kodd = 2 * ((j + nyb1) & 1) + 1;
//		for (i = nd1_tyy[8]; i <= nd1_tyy[9]; i++)
//		{
	jkq=((i+nxb1) & 1) + kodd;
	for (k = nd1_tyy[12]; k <= nd1_tyy[17]; k++)
	{
		sxx=dxh1(1,i)*v1x(k,i-2,j)+dxh1(2,i)*v1x(k,i-1,j)+
			dxh1(3,i)*v1x(k,i  ,j)+dxh1(4,i)*v1x(k,i+1,j);
		syy=dyh1(1,j)*v1y(k,i,j-2)+dyh1(2,j)*v1y(k,i,j-1)+
			dyh1(3,j)*v1y(k,i  ,j)+dyh1(4,j)*v1y(k,i,j+1);
		sxy=dxi1(1,i)*v1y(k,i-1,j)+dxi1(2,i)*v1y(k,i,  j)+
			dxi1(3,i)*v1y(k,i+1,j)+dxi1(4,i)*v1y(k,i+2,j)+
			dyi1(1,j)*v1x(k,i,j-1)+dyi1(2,j)*v1x(k,i,j  )+
			dyi1(3,j)*v1x(k,i,j+1)+dyi1(4,j)*v1x(k,i,j+2);
		if(k==1) {
		  	szz=dzi1(2,k)/ca*(22.*v1z(k,i,j)-17.*v1z(k+1,i,j)-
				9.*v1z(k+2,i,j)+5.*v1z(k+3,i,j)-v1z(k+4,i,j))/24.0;
		}
		else if(k==nztop) {
		  	szz=dzi1(2,k)/ca*(v1z(k,i,j)-v1z(k+1,i,j));
		}
		else
		{
		  	szz=dzi1(1,k)*v1z(k-1,i,j)+dzi1(2,k)*v1z(k,  i,j)+
				dzi1(3,k)*v1z(k+1,i,j)+dzi1(4,k)*v1z(k+2,i,j);
		}

		inod=idmat1(k,i,j);
		cl=clamda(inod);
		sm2=2.*cmu(inod);
		pm=cl+sm2;
		cusxy=sxy/(1./sm2+.5/cmu(idmat1(k,i+1,j+1)));
		sss=sxx+syy+szz;
		irw=jkq+4*(k&1);
		et=epdt(irw);
		et1=1.0-et;
		wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw));
		wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		qxx=qt1xx(k,i,j);
		qt1xx(k,i,j)=qxx*et+(wtp*sss-wts*(syy+szz))*et1;
		t1xx(k,i,j)=t1xx(k,i,j)+sm2*sxx+cl*sss-qxx-qt1xx(k,i,j);
		qyy=qt1yy(k,i,j);
		qt1yy(k,i,j)=qyy*et+(wtp*sss-wts*(sxx+szz))*et1;
		t1yy(k,i,j)=t1yy(k,i,j)+sm2*syy+cl*sss-qyy-qt1yy(k,i,j);
		qzz=qt1zz(k,i,j);
		qt1zz(k,i,j)=qzz*et+(wtp*sss-wts*(sxx+syy))*et1;
		t1zz(k,i,j)=t1zz(k,i,j)+sm2*szz+cl*sss-qzz-qt1zz(k,i,j);
		qxy=qt1xy(k,i,j);
		qt1xy(k,i,j)=qxy*et+wts/sm2*cusxy*et1;
		t1xy(k,i,j)=t1xy(k,i,j)+cusxy-qxy-qt1xy(k,i,j);
	}
//		}
//	}
	return;
}

//-----------------------------------------------------------------------------
__global__ void  stress_xz_yz_IC(int nxb1,
					  int nyb1,
					  int nxtop,
					  int nytop,
					  int nztop,
					  int *nd1_tyz,
					  int *idmat1M,
					  float ca,
					  float *cmuM,
					  float *epdtM,
					  float *qwsM,
					  float *qwt1M,
					  float *qwt2M,
					  float *dxi1M,
					  float *dyi1M,
					  float *dzh1M,
					  float *v1xM,
					  float *v1yM,
					  float *v1zM,
					  float *t1xzM,
					  float *t1yzM,
					  float *qt1xzM,
					  float *qt1yzM)
// Compute stress-XZand YZ component in Region I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
//	real, parameter:: tfr1=-577./528./ca,tfr2=201./176./ca, &
//                   tfr3=-9./176./ca,  tfr4=1./528./ca
{
//	float tfr1 = -577./528./ca;
//	float tfr2 = 201./176./ca;
//	float tfr3 = -9./176./ca;
//	float tfr4=1./528./ca;

	int i,j,k,kodd,inod,jkq,irw;
	float dvzx,dvzy,dvxz,dvyz,sm,cusxz,cusyz,et,et1,dmws,qxz,qyz;

    j = blockIdx.x * blockDim.x + threadIdx.x + nd1_tyz[2];
    i = blockIdx.y * blockDim.y + threadIdx.y + nd1_tyz[8];

    if (j > nd1_tyz[3] || i > nd1_tyz[9])
    {
        return;
    }

//	for (j=nd1_tyz[2]; j <=nd1_tyz[3]; j++)
//	//do j=nd1_tyz(3),nd1_tyz(4)
//	{
		//kodd=2*mod(j+nyb1,2)+1
	kodd=2*((j+nyb1)&1)+1;
//		for (i=nd1_tyz[8]; i<=nd1_tyz[9]; i++)
//		//do i=nd1_tyz(9),nd1_tyz(10)
//		{
			//jkq=mod(i+nxb1,2)+kodd
	jkq=((i+nxb1)&1)+kodd;
	for (k=nd1_tyz[12]; k<=nd1_tyz[17]; k++)
	//do k=nd1_tyz(13),nd1_tyz(18)
	{
		dvzx=dxi1(1,i)*v1z(k,i-1,j)+dxi1(2,i)*v1z(k,i,  j)+
			 dxi1(3,i)*v1z(k,i+1,j)+dxi1(4,i)*v1z(k,i+2,j);
		dvzy=dyi1(1,j)*v1z(k,i,j-1)+dyi1(2,j)*v1z(k,i,j  )+
			 dyi1(3,j)*v1z(k,i,j+1)+dyi1(4,j)*v1z(k,i,j+2);
		if(k<nztop) {
			dvxz=dzh1(1,k)*v1x(k-2,i,j)+dzh1(2,k)*v1x(k-1,i,j)+
				 dzh1(3,k)*v1x(k,  i,j)+dzh1(4,k)*v1x(k+1,i,j);
			dvyz=dzh1(1,k)*v1y(k-2,i,j)+dzh1(2,k)*v1y(k-1,i,j)+
				 dzh1(3,k)*v1y(k,  i,j)+dzh1(4,k)*v1y(k+1,i,j);
		}
		else {
			dvxz=dzh1(2,k)/ca*(v1x(k-1,i,j)-v1x(k,i,j));
			dvyz=dzh1(2,k)/ca*(v1y(k-1,i,j)-v1y(k,i,j));
		}

		inod=idmat1(k,i,j);
		sm=cmu(inod);
		cusxz=(dvzx+dvxz)/(.5/sm+.5/cmu(idmat1(k-1,i+1,j)));
		cusyz=(dvzy+dvyz)/(.5/sm+.5/cmu(idmat1(k-1,i,j+1)));
		//irw=jkq+4*mod(k,2);
		irw=jkq+4*(k&1);
		et=epdt(irw);
		et1=1.0-et;
		dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		qxz=qt1xz(k,i,j);
		qt1xz(k,i,j)=qxz*et+dmws*cusxz*et1;
		t1xz(k,i,j)=t1xz(k,i,j)+cusxz-qxz-qt1xz(k,i,j);
		qyz=qt1yz(k,i,j);
		qt1yz(k,i,j)=qyz*et+dmws*cusyz*et1;
		t1yz(k,i,j)=t1yz(k,i,j)+cusyz-qyz-qt1yz(k,i,j);
	}
//		}
//	}

	return;
}

__global__ void stress_resetVars(int ny1p1,
					  int nx1p1,
					  int nxtop,
					  int nytop,
					  int nztop,
					  float *t1xzM,
					  float *t1yzM)
{
	int i, j;

	j = blockIdx.x * blockDim.x + threadIdx.x - 1;
	i = blockIdx.y * blockDim.y + threadIdx.y + 1;

	if (j <= ny1p1 && i <= nxtop)
	{
		t1yz(1, i, j) = 0.0f;
	}

//	for (j=-1; j<=ny1p1; j++)
//	{
//		for (i = 1; i <= nxtop; i++)
//		{
//			t1yz(1,i,j)=0.0;
//		}
//	}

	j = j + 2;
	i = i - 2;

	if (j <= nytop && i <= nx1p1)
	{
		t1xz(1, i, j) = 0.0;
	}

//	for (j=1; j <= nytop; j++)
//	{
//		for (i=-1; i <=nx1p1; i++)
//		{
//			t1xz(1,i,j)=0.0;
//		}
//	}

	return;
}

//------------------------------------------------------------------------------------
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
						 float ca,
						 float *drti1M,
						 float *damp1_xM,
						 float *clamdaM,
						 float *cmuM,
						 float *epdtM,
						 float *qwpM,
						 float *qwsM,
						 float *qwt1M,
						 float *qwt2M,
						 float *dzi1M,
						 float *dxh1M,
						 float *dyh1M,
						 float *v1xM,
						 float *v1yM,
						 float *v1zM,
						 float *t1xxM,
						 float *t1yyM,
						 float *t1zzM,
						 float *t1xx_pxM,
						 float *t1yy_pxM,
						 float *qt1xxM,
						 float *qt1yyM,
						 float *qt1zzM,
						 float *qt1xx_pxM,
						 float *qt1yy_pxM)
// Compute the velocity of PML-x-I region
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
// real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	float taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;
	int nti;
	//if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return;

    j = blockIdx.x * blockDim.x + threadIdx.x + nd1_tyy[0];
    lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

    if (j > nd1_tyy[5] || lb > lbx1)
    {
        return;
    }
	nti = (lbx1 - lbx0 + 1) * mw1_pml + lbx1;
	
// 	for (j=nd1_tyy[0]; j <= nd1_tyy[5]; j++)
// 	//do j=nd1_tyy(1),nd1_tyy(6)
//	{
	kodd=2*((j+nyb1)&1)+1;
	ib=0;
	for (k = lbx0; k < lb; k++)
	{
		for (i = nd1_tyy[6+4*k]; i <= nd1_tyy[7+4*k]; i++)
			ib++;
	}
//		for (lb=lbx[0]; lb <=lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	kb=0;
	for (i = nd1_tyy[6+4*lb]; i <= nd1_tyy[7+4*lb]; i++)
	//do i=nd1_tyy(7+4*lb),nd1_tyy(8+4*lb)
	{
		kb=kb+1;
		ib=ib+1;
		rti=drti1(kb,lb);
		jkq=((i+nxb1)&1)+kodd;
		//jkq=mod(i+nxb1,2)+kodd
		for (k=nd1_tyy[12]; k <=nd1_tyy[17]; k++)
		//do k=nd1_tyy(13),nd1_tyy(18)
		{
			damp2=1./(1.+damp1_x(k,j,lb)*rti);
			damp1=damp2*2.0-1.;
			inod=idmat1(k,i,j);
			cl=clamda(inod);
			sm2=2.*cmu(inod);
			pm=cl+sm2;
			irw=jkq+4*(k&1);
			//irw=jkq+4*mod(k,2);
			et=epdt(irw);
			et1=1.0-et;
			wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw));
			wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxx=t1xx(k,i,j)-t1xx_px(k,ib,j);
			//debug
			//t1xx(k,i,j)=t1xx_px(k,ib,j);
			
			taoyy=t1yy(k,i,j)-t1yy_px(k,ib,j);
			taozz=t1zz(k,i,j)-t1yy_px(k,ib,j);

			if(j>nd1_tyy[1] && j<nd1_tyy[4]) {
			//if(j>nd1_tyy(2) .and. j<nd1_tyy(5)) {
				syy=dyh1(1,j)*v1y(k,i,j-2)+dyh1(2,j)*v1y(k,i,j-1)+
					dyh1(3,j)*v1y(k,i  ,j)+dyh1(4,j)*v1y(k,i,j+1);
				if(k==1) {
					szz=dzi1(2,k)/ca*(22.*v1z(k,i,j)-17.*v1z(k+1,i,j)-
						9.*v1z(k+2,i,j)+5.*v1z(k+3,i,j)-v1z(k+4,i,j))/24.;
				}
				else if(k==nztop) {
					szz=dzi1(2,k)/ca*(v1z(k,i,j)-v1z(k+1,i,j));
				}
				else {
					szz=dzi1(1,k)*v1z(k-1,i,j)+dzi1(2,k)*v1z(k,  i,j)+
						dzi1(3,k)*v1z(k+1,i,j)+dzi1(4,k)*v1z(k+2,i,j);
				}
				sss=syy+szz;
				qxx=qt1xx(k,i,j);
				qt1xx(k,i,j)=qxx*et+(wtp-wts)*sss*et1;
				taoxx=taoxx+cl*sss-qxx-qt1xx(k,i,j);
				qyy=qt1yy(k,i,j);
				qt1yy(k,i,j)=qyy*et+(wtp*sss-wts*szz)*et1;
				taoyy=taoyy+sm2*syy+cl*sss-qyy-qt1yy(k,i,j);
				qzz=qt1zz(k,i,j);
				qt1zz(k,i,j)=qzz*et+(wtp*sss-wts*syy)*et1;
				taozz=taozz+sm2*szz+cl*sss-qzz-qt1zz(k,i,j);
			}
			sxx=dxh1(2,i)/ca*(v1x(k,i-1,j)-v1x(k,i,j));
			qxx=qt1xx_px(k,ib,j);
			qt1xx_px(k,ib,j)=qxx*et+wtp*sxx*et1;
			t1xx_px(k,ib,j)=damp1*t1xx_px(k,ib,j)+
							damp2*(pm*sxx-qxx-qt1xx_px(k,ib,j));
			t1xx(k,i,j)=taoxx+t1xx_px(k,ib,j);
			qyy=qt1yy_px(k,ib,j);
			qt1yy_px(k,ib,j)=qyy*et+(wtp-wts)*sxx*et1;
			t1yy_px(k,ib,j)=damp1*t1yy_px(k,ib,j)+
							damp2*(cl*sxx-qyy-qt1yy_px(k,ib,j));
			t1yy(k,i,j)=taoyy+t1yy_px(k,ib,j);
			t1zz(k,i,j)=taozz+t1yy_px(k,ib,j);
		}
	}
//		}
//	}
	return;
}

__global__ void stress_norm_PmlY_IC(int nxb1,
						 int nyb1,
						 int mw1_pml1,
						 int nxtop,
						 int nztop,
						 int lby0,
						 int lby1,
						 int *nd1_tyy,
						 int *idmat1M,
						 float ca,
						 float *drti1M,
						 float *damp1_yM,
						 float *clamdaM,
						 float *cmuM,
						 float *epdtM,
						 float *qwpM,
						 float *qwsM,
						 float *qwt1M,
						 float *qwt2M,
						 float *dxh1M,
						 float *dyh1M,
						 float *dzi1M,
						 float *t1xxM,
						 float *t1yyM,
						 float *t1zzM,
						 float *qt1xxM,
						 float *qt1yyM,
						 float *qt1zzM,
						 float *t1xx_pyM,
						 float *t1yy_pyM,
						 float *qt1xx_pyM,
						 float *qt1yy_pyM,
						 float *v1xM,
						 float *v1yM,
						 float *v1zM)
// Compute the velocity of PML-x-I region
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
// real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	float taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

	//if(lby[0]>lby[1]) return;
	//if(lby(1)>lby(2) ) return

	i = blockIdx.x * blockDim.x + threadIdx.x + nd1_tyy[6];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;
	if (i > nd1_tyy[11] || lb > lby1)
	{
		return;
	}

//	for (i = nd1_tyy[6]; i <= nd1_tyy[11]; i++)
//	//do i=nd1_tyy(7),nd1_tyy(12)
//	{
	jb = 0;
	for (k = lby0; k < lb; k++)
	{
		for (j = nd1_tyy[4*k]; j <= nd1_tyy[1+4*k]; j++)
		{
			jb++;
		}
	}

//		for (lb=lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1),lby(2)
//		{
	kb=0;
	for (j = nd1_tyy[4*lb]; j <= nd1_tyy[1+4*lb]; j++)
	//do j=nd1_tyy(1+4*lb),nd1_tyy(2+4*lb)
	{
		kb=kb+1;
		jb=jb+1;
		rti=drti1(kb,lb);
		kodd=2 * ((j + nyb1) & 1) + 1;
		//kodd=2*mod(j+nyb1,2)+1
		jkq = ((i + nxb1) & 1) + kodd;
		//jkq=mod(i+nxb1,2)+kodd
		for (k=nd1_tyy[12]; k <=nd1_tyy[17]; k++)
		//do k=nd1_tyy(13),nd1_tyy(18)
		{
			damp2=1./(1.+damp1_y(k,i,lb)*rti);
			damp1=damp2*2.-1.;
			inod=idmat1(k,i,j);
			cl=clamda(inod);
			sm2=2.*cmu(inod);
			pm=cl+sm2;
			//irw=jkq+4*mod(k,2)
			irw=jkq + 4 * (k & 1);
			et=epdt(irw);
			et1=1.0-et;
			wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw));
			wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			if (i>nd1_tyy[7] && i<nd1_tyy[10]) {
			//if(i>nd1_tyy(8) .and. i<nd1_tyy(11)) then
				sxx=dxh1(1,i)*v1x(k,i-2,j)+dxh1(2,i)*v1x(k,i-1,j)+
					dxh1(3,i)*v1x(k,i  ,j)+dxh1(4,i)*v1x(k,i+1,j);
			}
			else {
				sxx=0.0;
			}

			if(k==1) {
				szz=dzi1(2,k)/ca*(22.*v1z(k,i,j)-17.*v1z(k+1,i,j)-
					9.*v1z(k+2,i,j)+5.*v1z(k+3,i,j)-v1z(k+4,i,j))/24.;
			}
			else if(k==nztop) {
				szz=dzi1(2,k)/ca*(v1z(k,i,j)-v1z(k+1,i,j));
			}
			else {
				szz=dzi1(1,k)*v1z(k-1,i,j)+dzi1(2,k)*v1z(k,  i,j)+
					dzi1(3,k)*v1z(k+1,i,j)+dzi1(4,k)*v1z(k+2,i,j);
			}
			sss=sxx+szz;
			qxx=qt1xx(k,i,j);
			qt1xx(k,i,j)=qxx*et+(wtp*sss-wts*szz)*et1;
			taoxx=t1xx(k,i,j)-t1xx_py(k,i,jb)+sm2*sxx+cl*sss-qxx-qt1xx(k,i,j);
			qyy=qt1yy(k,i,j);
			qt1yy(k,i,j)=qyy*et+(wtp-wts)*sss*et1;
			taoyy=t1yy(k,i,j)-t1yy_py(k,i,jb)+cl*sss-qyy-qt1yy(k,i,j);
			qzz=qt1zz(k,i,j);
			qt1zz(k,i,j)=qzz*et+(wtp*sss-wts*sxx)*et1;
			taozz=t1zz(k,i,j)-t1xx_py(k,i,jb)+sm2*szz+cl*sss-qzz-qt1zz(k,i,j);
			syy=dyh1(2,j)/ca*(v1y(k,i,j-1)-v1y(k,i,j));
			qxx=qt1xx_py(k,i,jb);
			qt1xx_py(k,i,jb)=qxx*et+(wtp-wts)*syy*et1;
			t1xx_py(k,i,jb)=damp1*t1xx_py(k,i,jb)+ 
							damp2*(cl*syy-qxx-qt1xx_py(k,i,jb));
			t1xx(k,i,j)=taoxx+t1xx_py(k,i,jb);
			t1zz(k,i,j)=taozz+t1xx_py(k,i,jb);
			qyy=qt1yy_py(k,i,jb);
			qt1yy_py(k,i,jb)=qyy*et+wtp*syy*et1;
			t1yy_py(k,i,jb)=damp1*t1yy_py(k,i,jb)+ 
							damp2*(pm*syy-qyy-qt1yy_py(k,i,jb));
			t1yy(k,i,j)=taoyy+t1yy_py(k,i,jb);
		}
	}
//		}
//	}
	return;
}

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
					   float ca,
					   float *drth1M,
					   float *damp1_xM,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dxi1M,
					   float *dyi1M,
					   float *t1xyM,
					   float *qt1xyM,
					   float *t1xy_pxM,
					   float *qt1xy_pxM,
					   float *v1xM,
					   float *v1yM)
// Compute the Stress-xy at region of PML-x-I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
// real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	float taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	nth = (lbx1 - lbx0 + 1) * mw1_pml + 1 - lbx0;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd1_txy[0];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;
	if (j > nd1_txy[5] || lb > lbx1)
	{
		return;
	}

	ib = 0;
	for (k = lbx0; k < lb; k++)
	{
		for (i = nd1_txy[6+4*k]; i <= nd1_txy[7+4*k]; i++)
		{
			ib++;
		}
	}
	//if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return
//	for (j = nd1_txy[0]; j <= nd1_txy[5]; j++)
//	//do j=nd1_txy(1),nd1_txy(6)
//	{
	kodd = 2 * ((j + nyb1) & 1) + 1;
		//kodd=2*mod(j+nyb1,2)+1
//		ib=0;
//		for (lb = lbx[0]; lb <= lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	kb=0;
	for (i = nd1_txy[6+4*lb]; i <= nd1_txy[7+4*lb]; i++)
	//do i=nd1_txy(7+4*lb),nd1_txy(8+4*lb)
	{
		kb=kb+1;
		ib=ib+1;
		rth=drth1(kb,lb);
		jkq=((i + nxb1) & 1) + kodd;
		//jkq=mod(i+nxb1,2)+kodd;
		for (k = nd1_txy[12]; k <= nd1_txy[17]; k++)
		//do k=nd1_txy(13),nd1_txy(18)
		{
			damp2=1./(1.+damp1_x(k,j,lb)*rth);
			damp1=damp2*2.-1.;
			inod=idmat1(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat1(k,i+1,j+1)));
			irw=jkq + 4 * (k & 1);
			//irw=jkq+4*mod(k,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxy=t1xy(k,i,j)-t1xy_px(k,ib,j);
			if(j > nd1_txy[1] && j<nd1_txy[4]) {
			//if(j>nd1_txy(2) .and. j<nd1_txy(5)) then
				cusxy=(dyi1(1,j)*v1x(k,i,j-1)+dyi1(2,j)*v1x(k,i,j)+
					 dyi1(3,j)*v1x(k,i,j+1)+dyi1(4,j)*v1x(k,i,j+2))*sm;
				qxy=qt1xy(k,i,j);
				qt1xy(k,i,j)=qxy*et+dmws*cusxy*et1;
				taoxy=taoxy+cusxy-qxy-qt1xy(k,i,j);
			}
			cusxy=sm*dxi1(2,i)/ca*(v1y(k,i,j)-v1y(k,i+1,j));
			qxy=qt1xy_px(k,ib,j);
			qt1xy_px(k,ib,j)=qxy*et+dmws*cusxy*et1;
			t1xy_px(k,ib,j)=damp1*t1xy_px(k,ib,j)+
							damp2*(cusxy-qxy-qt1xy_px(k,ib,j));
			t1xy(k,i,j)=taoxy+t1xy_px(k,ib,j);
		}
	}
//		}
//	}

	return;
}

__global__ void stress_xy_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd1_txy,
					   int *idmat1M,
					   float ca,
					   float *drth1M,
					   float *damp1_yM,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dxi1M,
					   float *dyi1M,
					   float *t1xyM,
					   float *qt1xyM,
					   float *t1xy_pyM,
					   float *qt1xy_pyM,
					   float *v1xM,
					   float *v1yM)
//Compute the Stress-xy at region of PML-y-I
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxy,cusyx,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	float taoxy,cusyx,qxy,rth,damp2,damp1,sm,dmws,et,et1;

	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	i = blockIdx.x * blockDim.x + threadIdx.x + nd1_txy[6];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;

	if (i > nd1_txy[11] || lb > lby1)
	{
		return;
	}

//	for (i = nd1_txy[6]; i <= nd1_txy[11]; i++)
//	//do i=nd1_txy(7),nd1_txy(12)
//	{
	jb=0;
	for (k = lby0; k < lb; k++)
	{
		for (j = nd1_txy[4*k]; j <= nd1_txy[1 + 4 * k]; j++)
		{
			jb++;
		}
	}

//		for (lb = lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1), lby(2)
//		{
	kb=0;
	for (j = nd1_txy[4*lb]; j <= nd1_txy[1 + 4 * lb]; j++)
	//do j=nd1_txy(1+4*lb),nd1_txy(2+4*lb)
	{
		kb=kb+1;
		jb=jb+1;
		rth=drth1(kb,lb);
		kodd=2 * ((j + nyb1) & 1) + 1;
		//kodd=2*mod(j+nyb1,2)+1;
		jkq=((i + nxb1) & 1) + kodd;
		//jkq=mod(i+nxb1,2)+kodd
		for (k = nd1_txy[12]; k <= nd1_txy[17]; k++)
		//do k=nd1_txy(13),nd1_txy(18)
		{
			damp2=1./(1.+damp1_y(k,i,lb)*rth);
			damp1=damp2*2.-1.;
			inod=idmat1(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat1(k,i+1,j+1)));
			irw=jkq+4*(k&1);
			//irw=jkq+4*mod(k,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxy=t1xy(k,i,j)-t1xy_py(k,i,jb);
			if(i > nd1_txy[7] && i<nd1_txy[10]) {
			//if(i>nd1_txy(8) .and. i<nd1_txy(11)) then
				cusyx=(dxi1(1,i)*v1y(k,i-1,j)+dxi1(2,i)*v1y(k,i,j)+ 
					 dxi1(3,i)*v1y(k,i+1,j)+dxi1(4,i)*v1y(k,i+2,j))*sm;
				qxy=qt1xy(k,i,j);
				qt1xy(k,i,j)=qxy*et+dmws*cusyx*et1;
				taoxy=taoxy+cusyx-qxy-qt1xy(k,i,j);
			}
			cusyx=sm*dyi1(2,j)/ca*(v1x(k,i,j)-v1x(k,i,j+1));
			qxy=qt1xy_py(k,i,jb);
			qt1xy_py(k,i,jb)=qxy*et+dmws*cusyx*et1;
			t1xy_py(k,i,jb)=damp1*t1xy_py(k,i,jb)+
							damp2*(cusyx-qxy-qt1xy_py(k,i,jb));
			t1xy(k,i,j)=taoxy+t1xy_py(k,i,jb);
		}
	}
//		}
//	}
	return;
}

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
					   float ca,
					   float *drth1M,
					   float *damp1_xM,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dxi1M,
					   float *dzh1M,
					   float *t1xzM,
					   float *qt1xzM,
					   float *t1xz_pxM,
					   float *qt1xz_pxM,
					   float *v1xM,
					   float *v1zM)
//Compute the stress-xz at PML-x-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,dvxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	float taoxz,cusxz,dvxz,qxz,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	//if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return
	nth = (lbx1 - lbx0 + 1) * mw1_pml + 1 - lbx0;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd1_txz[0];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

	if (j > nd1_txz[5] || lb > lbx1)
	{
		return;
	}

//	for (j = nd1_txz[0]; j <= nd1_txz[5]; j++)
//	//do j=nd1_txz(1),nd1_txz(6)
//	{
	kodd=2 * ((j+nyb1)&1)+1;
	//kodd=2*mod(j+nyb1,2)+1
	ib=0;
	for (k = lbx0; k < lb; k++)
	{
		for (i = nd1_txz[6+4*k]; i <= nd1_txz[7+4*k]; i++)
		{
			ib++;
		}
	}

//		for (lb = lbx[0]; lb <= lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	kb=0;
	for (i = nd1_txz[6+4*lb]; i <= nd1_txz[7+4*lb]; i++)
	//do i=nd1_txz(7+4*lb),nd1_txz(8+4*lb)
	{
		kb=kb+1;
		ib=ib+1;
		rth=drth1(kb,lb);
		jkq=((i+nxb1)&1)+kodd;
		//jkq=mod(i+nxb1,2)+kodd
		for (k = nd1_txz[12]; k <= nd1_txz[17]; k++)
		//do k=nd1_txz(13),nd1_txz(18)
		{
			damp2=1./(1.+damp1_x(k,j,lb)*rth);
			damp1=damp2*2.-1.;
			inod=idmat1(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i+1,j)));
			irw=jkq+4*(k&1);
			//irw=jkq+4*mod(k,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			if(k<nztop) {
				dvxz=dzh1(1,k)*v1x(k-2,i,j)+dzh1(2,k)*v1x(k-1,i,j)+
					dzh1(3,k)*v1x(k,  i,j)+dzh1(4,k)*v1x(k+1,i,j);
			}
			else {
				dvxz=dzh1(2,k)/ca*(v1x(k-1,i,j)-v1x(k,i,j));
			}
			cusxz=dvxz*sm;
			qxz=qt1xz(k,i,j);
			qt1xz(k,i,j)=qxz*et+dmws*cusxz*et1;
			taoxz=t1xz(k,i,j)-t1xz_px(k,ib,j)+cusxz-qxz-qt1xz(k,i,j);
			cusxz=sm*dxi1(2,i)/ca*(v1z(k,i,j)-v1z(k,i+1,j));
			qxz=qt1xz_px(k,ib,j);
			qt1xz_px(k,ib,j)=qxz*et+dmws*cusxz*et1;
			t1xz_px(k,ib,j)=damp1*t1xz_px(k,ib,j)+
							damp2*(cusxz-qxz-qt1xz_px(k,ib,j));
			t1xz(k,i,j)=taoxz+t1xz_px(k,ib,j);
		}
	}
//		}
//	}
 	return;
}

__global__ void stress_xz_PmlY_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd1_txz,
					   int *idmat1M,
					   float ca,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dxi1M,
					   float *dzh1M,
					   float *t1xzM,
					   float *qt1xzM,
					   float *v1xM,
					   float *v1zM)
//Compute the stress-xz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	float cusxz,dvxz,dvzx,qxz,sm,dmws,et,et1;

	//if (lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	i = blockIdx.x * blockDim.x + threadIdx.x + nd1_txz[8];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;

	if (i > nd1_txz[9] || lb > lby1)
	{
		return;
	}

//	for (i = nd1_txz[8]; i <= nd1_txz[9]; i++)
//	//do i=nd1_txz(9),nd1_txz(10)
//	{

//		for (lb=lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1),lby(2)
//		{
	for (j = nd1_txz[4*lb]; j <= nd1_txz[1+4*lb]; j++)
	//do j=nd1_txz(1+4*lb),nd1_txz(2+4*lb)
	{
		kodd=2 * ((j+nyb1)&1)+1;
		//kodd=2*mod(j+nyb1,2)+1
		jkq=((i+nxb1)&1)+kodd;
		//jkq=mod(i+nxb1,2)+kodd
		for (k = nd1_txz[12]; k <= nd1_txz[17]; k++)
		//do k=nd1_txz(13),nd1_txz(18)
		{
			inod=idmat1(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i+1,j)));
			irw=jkq+4*(k&1);
			//irw=jkq+4*mod(k,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			dvzx=dxi1(1,i)*v1z(k,i-1,j)+dxi1(2,i)*v1z(k,i,  j)+
				 dxi1(3,i)*v1z(k,i+1,j)+dxi1(4,i)*v1z(k,i+2,j);
			if(k<nztop) {
				dvxz=dzh1(1,k)*v1x(k-2,i,j)+dzh1(2,k)*v1x(k-1,i,j)+
					dzh1(3,k)*v1x(k,  i,j)+dzh1(4,k)*v1x(k+1,i,j);
			}
			else {
				dvxz=dzh1(2,k)/ca*(v1x(k-1,i,j)-v1x(k,i,j));
			}
			cusxz=(dvzx+dvxz)*sm;
			qxz=qt1xz(k,i,j);
			qt1xz(k,i,j)=qxz*et+dmws*cusxz*et1;
			t1xz(k,i,j)=t1xz(k,i,j)+cusxz-qxz-qt1xz(k,i,j);
		}
	}
//		}
//	}
	return;
}

__global__ void stress_yz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nztop,
					   int nxtop,
					   int lbx0,
					   int lbx1,
					   int *nd1_tyz,
					   int *idmat1M,
					   float ca,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dyi1M,
					   float *dzh1M,
					   float *t1yzM,
					   float *qt1yzM,
					   float *v1yM,
					   float *v1zM)
//Compute the stress-yz at PML-x-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusyz,dvyz,dvzy,qyz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	float cusyz,dvyz,dvzy,qyz,sm,dmws,et,et1;

	//if(lbx[0] > lbx[1]) return;
	//if(lbx(1)>lbx(2) ) return

	j = blockIdx.x * blockDim.x + threadIdx.x + nd1_tyz[2];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

	if (j > nd1_tyz[3] || lb > lbx1)
	{
		return;
	}

//	for (j = nd1_tyz[2]; j <= nd1_tyz[3]; j++)
//	//do j=nd1_tyz(3),nd1_tyz(4)
//	{
	kodd=2 * ((j+nyb1)&1)+1;
	//kodd=2*mod(j+nyb1,2)+1
//		for (lb = lbx[0]; lb <= lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	for (i = nd1_tyz[6+4*lb]; i <= nd1_tyz[7+4*lb]; i++)
	//do i=nd1_tyz(7+4*lb),nd1_tyz(8+4*lb)
	{
		jkq = ((i+nxb1)&1)+kodd;
		//jkq=mod(i+nxb1,2)+kodd
		for (k = nd1_tyz[12]; k <= nd1_tyz[17]; k++)
		//do k=nd1_tyz(13),nd1_tyz(18)
		{
			inod=idmat1(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i,j+1)));
			irw=jkq+4*(k&1);
			//irw=jkq+4*mod(k,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			dvzy=dyi1(1,j)*v1z(k,i,j-1)+dyi1(2,j)*v1z(k,i,j  )+
				 dyi1(3,j)*v1z(k,i,j+1)+dyi1(4,j)*v1z(k,i,j+2);
			if(k<nztop) {
				dvyz=dzh1(1,k)*v1y(k-2,i,j)+dzh1(2,k)*v1y(k-1,i,j)+
					dzh1(3,k)*v1y(k,  i,j)+dzh1(4,k)*v1y(k+1,i,j);
			}
			else {
				dvyz=dzh1(2,k)/ca*(v1y(k-1,i,j)-v1y(k,i,j));
			}
			cusyz=(dvzy+dvyz)*sm;
			qyz=qt1yz(k,i,j);
			qt1yz(k,i,j)=qyz*et+dmws*cusyz*et1;
			t1yz(k,i,j)=t1yz(k,i,j)+cusyz-qyz-qt1yz(k,i,j);
		}
	}
//		}
//	}
	return;
}

__global__ void stress_yz_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd1_tyz,
					   int *idmat1M,
					   float ca,
					   float *drth1M,
					   float *damp1_yM,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dyi1M,
					   float *dzh1M,
					   float *t1yzM,
					   float *qt1yzM,
					   float *t1yz_pyM,
					   float *qt1yz_pyM,
					   float *v1yM,
					   float *v1zM)
//Compute the stress-yz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	float taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1;

	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	i = blockIdx.x * blockDim.x + threadIdx.x + nd1_tyz[6];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;

	if (i > nd1_tyz[11] || lb > lby1)
	{
		return;
	}

//	for (i = nd1_tyz[6]; i <= nd1_tyz[11]; i++)
//	//do i=nd1_tyz(7),nd1_tyz(12)
//	{
	jb=0;
	for (k = lby0; k < lb; k++)
	{
		for (j = nd1_tyz[4*k]; j <= nd1_tyz[1+4*k]; j++)
		{
			jb++;
		}
	}
//		for (lb=lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1),lby(2)
//		{
	kb=0;
	for (j = nd1_tyz[4*lb]; j <= nd1_tyz[1+4*lb]; j++)
	//do j=nd1_tyz(1+4*lb),nd1_tyz(2+4*lb)
	{
		kb=kb+1;
		jb=jb+1;
		rth=drth1(kb,lb);
		kodd=2*((j+nyb1)&1)+1;
		//kodd=2*mod(j+nyb1,2)+1;
		jkq=((i+nxb1)&1)+kodd;
		//jkq=mod(i+nxb1,2)+kodd
		for (k=nd1_tyz[12]; k <= nd1_tyz[17]; k++)
		//do k=nd1_tyz(13),nd1_tyz(18)
		{
			damp2=1./(1.+damp1_y(k,i,lb)*rth);
			damp1=damp2*2.-1.;
			inod=idmat1(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i,j+1)));
			irw=jkq+4*(k&1);
			//irw=jkq+4*mod(k,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			if(k<nztop) {
				dvyz=dzh1(1,k)*v1y(k-2,i,j)+dzh1(2,k)*v1y(k-1,i,j)+
					dzh1(3,k)*v1y(k,  i,j)+dzh1(4,k)*v1y(k+1,i,j);
			}
			else {
				dvyz=dzh1(2,k)/ca*(v1y(k-1,i,j)-v1y(k,i,j));
			}
			cusyz=dvyz*sm;
			qyz=qt1yz(k,i,j);
			qt1yz(k,i,j)=qyz*et+dmws*cusyz*et1;
			taoyz=t1yz(k,i,j)-t1yz_py(k,i,jb)+cusyz-qyz-qt1yz(k,i,j);
			cusyz=sm*dyi1(2,j)/ca*(v1z(k,i,j)-v1z(k,i,j+1));
			qyz=qt1yz_py(k,i,jb);
			qt1yz_py(k,i,jb)=qyz*et+dmws*cusyz*et1;
			t1yz_py(k,i,jb)=damp1*t1yz_py(k,i,jb)+
							damp2*(cusyz-qyz-qt1yz_py(k,i,jb));
			t1yz(k,i,j)=taoyz+t1yz_py(k,i,jb);
		}
	}
//		}
//	}

	return;
}

__global__ void stress_norm_xy_II(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int *nd2_tyy,
					   int *idmat2M,
					   float *clamdaM,
					   float *cmuM,
					   float *epdtM,
					   float *qwpM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *t2xxM,
					   float *t2xyM,
					   float *t2yyM,
					   float *t2zzM,
					   float *qt2xxM,
					   float *qt2xyM,
					   float *qt2yyM,
					   float *qt2zzM,
					   float *dxh2M,
					   float *dyh2M,
					   float *dxi2M,
					   float *dyi2M,
					   float *dzi2M,
					   float *v2xM,
					   float *v2yM,
					   float *v2zM)
// Compute stress-Norm and XY component in Region II
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,kodd,inod,jkq,irw
// real:: sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy, &
//        cl,sm2,et,et1,dmws,pm,wtp,wts
{
	int i,j,k,kodd,inod,jkq,irw;
	float sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy,cl,sm2,et,et1,dmws,pm,wtp,wts;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyy[2];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_tyy[8];

	if (j > nd2_tyy[3] || i > nd2_tyy[9])
	{
		return;
	}

//	for (j=nd2_tyy[2]; j <= nd2_tyy[3]; j++)
//	//do j=nd2_tyy(3),nd2_tyy(4)
//	{
	kodd=2*((j+nyb2)&1)+1;
	//kodd=2*mod(j+nyb2,2)+1
//		for (i = nd2_tyy[8]; i <= nd2_tyy[9]; i++)
//		//do i=nd2_tyy(9),nd2_tyy(10)
//		{
	jkq=((i+nxb2)&1)+kodd;
	//jkq=mod(i+nxb2,2)+kodd
	for (k = nd2_tyy[12]; k <= nd2_tyy[15]; k++)
	//do k=nd2_tyy(13),nd2_tyy(16)
	{
		sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+
			dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j);
		syy=dyh2(1,j)*v2y(k,i,j-2)+dyh2(2,j)*v2y(k,i,j-1)+
			dyh2(3,j)*v2y(k,i  ,j)+dyh2(4,j)*v2y(k,i,j+1);
		sxy=dxi2(1,i)*v2y(k,i-1,j)+dxi2(2,i)*v2y(k,i,  j)+
			dxi2(3,i)*v2y(k,i+1,j)+dxi2(4,i)*v2y(k,i+2,j)+
			dyi2(1,j)*v2x(k,i,j-1)+dyi2(2,j)*v2x(k,i,j  )+
			dyi2(3,j)*v2x(k,i,j+1)+dyi2(4,j)*v2x(k,i,j+2);
		szz=dzi2(1,k)*v2z(k-1,i,j)+dzi2(2,k)*v2z(k,  i,j)+
			dzi2(3,k)*v2z(k+1,i,j)+dzi2(4,k)*v2z(k+2,i,j);
		sss=sxx+syy+szz;
		inod=idmat2(k,i,j);
		cl=clamda(inod);
		sm2=2.*cmu(inod);
		pm=cl+sm2;
		cusxy=sxy/(1./sm2+.5/cmu(idmat2(k,i+1,j+1)));
		irw=jkq+4*((k+nztop)&1);
		//irw=jkq+4*mod(k+nztop,2);
		et=epdt(irw);
		et1=1.0-et;
		wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw));
		wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		qxx=qt2xx(k,i,j);
		qt2xx(k,i,j)=qxx*et+(wtp*sss-wts*(syy+szz))*et1;
		t2xx(k,i,j)=t2xx(k,i,j)+sm2*sxx+cl*sss-qxx-qt2xx(k,i,j);
		qyy=qt2yy(k,i,j);
		qt2yy(k,i,j)=qyy*et+(wtp*sss-wts*(sxx+szz))*et1;
		t2yy(k,i,j)=t2yy(k,i,j)+sm2*syy+cl*sss-qyy-qt2yy(k,i,j);
		qzz=qt2zz(k,i,j);
		qt2zz(k,i,j)=qzz*et+(wtp*sss-wts*(sxx+syy))*et1;
		t2zz(k,i,j)=t2zz(k,i,j)+sm2*szz+cl*sss-qzz-qt2zz(k,i,j);
		qxy=qt2xy(k,i,j);
		qt2xy(k,i,j)=qxy*et+wts/sm2*cusxy*et1;
		t2xy(k,i,j)=t2xy(k,i,j)+cusxy-qxy-qt2xy(k,i,j);
	}
//		}
//	}
	return;
}

//call stress_xz_yz_II
__global__ void stress_xz_yz_IIC(int nxb2,
					  int nyb2,
					  int nztop,
					  int nxbtm,
					  int nzbtm,
					  int *nd2_tyz,
					  int *idmat2M,
					  float *cmuM,
					  float *epdtM,
					  float *qwsM,
					  float *qwt1M,
					  float *qwt2M,
					  float *dxi2M,
					  float *dyi2M,
					  float *dzh2M,
					  float *t2xzM,
					  float *t2yzM,
					  float *qt2xzM,
					  float *qt2yzM,
					  float *v2xM,
					  float *v2yM,
					  float *v2zM)
//Compute stress-XZ and YZ component in the Region II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,kodd,inod,jkq,irw
//real:: qxz,qyz,cusxz,cusyz,sm,et,et1,dmws
{
	int i,j,k,kodd,inod,jkq,irw;
	float qxz,qyz,cusxz,cusyz,sm,et,et1,dmws;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyz[2];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_tyz[8];

	if (j > nd2_tyz[3] || i > nd2_tyz[9])
	{
		return;
	}

//	for (j = nd2_tyz[2]; j <= nd2_tyz[3]; j++)
//	//do j=nd2_tyz(3),nd2_tyz(4)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		for (i = nd2_tyz[8]; i <= nd2_tyz[9]; i++)
//		//do i=nd2_tyz(9),nd2_tyz(10)
//		{
	jkq=((i+nxb2)&1)+kodd;
	//jkq=mod(i+nxb2,2)+kodd
	for (k=nd2_tyz[12]; k <= nd2_tyz[15]; k++)
	//do k=nd2_tyz(13),nd2_tyz(16)
	{
		inod=idmat2(k,i,j);
		sm=cmu(inod);
		cusxz=(dxi2(1,i)*v2z(k,i-1,j)+dxi2(2,i)*v2z(k,i,  j)+ 
			   dxi2(3,i)*v2z(k,i+1,j)+dxi2(4,i)*v2z(k,i+2,j)+ 
			   dzh2(1,k)*v2x(k-2,i,j)+dzh2(2,k)*v2x(k-1,i,j)+ 
			   dzh2(3,k)*v2x(k,  i,j)+dzh2(4,k)*v2x(k+1,i,j))/ 
			   (.5/sm+.5/cmu(idmat2(k-1,i+1,j)));
		cusyz=(dyi2(1,j)*v2z(k,i,j-1)+dyi2(2,j)*v2z(k,i,j  )+ 
			   dyi2(3,j)*v2z(k,i,j+1)+dyi2(4,j)*v2z(k,i,j+2)+ 
			   dzh2(1,k)*v2y(k-2,i,j)+dzh2(2,k)*v2y(k-1,i,j)+ 
			   dzh2(3,k)*v2y(k,  i,j)+dzh2(4,k)*v2y(k+1,i,j))/
			   (.5/sm+.5/cmu(idmat2(k-1,i,j+1)));
		irw=jkq+4*((k+nztop)&1);
		//irw=jkq+4*mod(k+nztop,2)
		et=epdt(irw);
		et1=1.0-et;
		dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		qxz=qt2xz(k,i,j);
		qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1;
		t2xz(k,i,j)=t2xz(k,i,j)+cusxz-qxz-qt2xz(k,i,j);
		qyz=qt2yz(k,i,j);
		qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1;
		t2yz(k,i,j)=t2yz(k,i,j)+cusyz-qyz-qt2yz(k,i,j);
	}
//		}
//	}
	return;
}

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
						  float ca,
						  float *drti2M,
						  float *damp2_xM,
						  float *clamdaM,
						  float *cmuM,
						  float *epdtM,
						  float *qwpM,
						  float *qwsM,
						  float *qwt1M,
						  float *qwt2M,
						  float *dxh2M,
						  float *dyh2M,
						  float *dzi2M,
						  float *t2xxM,
						  float *t2yyM,
						  float *t2zzM,
						  float *qt2xxM,
						  float *qt2yyM,
						  float *qt2zzM,
						  float *t2xx_pxM,
						  float *t2yy_pxM,
						  float *qt2xx_pxM,
						  float *qt2yy_pxM,
						  float *v2xM,
						  float *v2yM,
						  float *v2zM)
//Compute the Stress-norm at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	float taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;
	int nti;

	//if(lbx[0] > lbx[1]) return;
	//if( lbx(1)>lbx(2) ) return
	nti = (lbx1 - lbx0 + 1) * mw2_pml + lbx1;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyy[0];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

	if (j > nd2_tyy[5] || lb > lbx1)
	{
		return;
	}

	ib = 0;
	for (k = lbx0; k < lb; k++)
	{
		for (i=nd2_tyy[6+4*k]; i <= nd2_tyy[7+4*k]; i++)
		{
			ib++;
		}
	}

//	for (j=nd2_tyy[0]; j <= nd2_tyy[5]; j++)
//	//do j=nd2_tyy(1),nd2_tyy(6)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		ib=0;
//		for (lb=lbx[0]; lb <= lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	kb=0;
	for (i=nd2_tyy[6+4*lb]; i <= nd2_tyy[7+4*lb]; i++)
	//do i=nd2_tyy(7+4*lb),nd2_tyy(8+4*lb)
	{
		kb=kb+1;
		ib=ib+1;
		rti=drti2(kb,lb);
		jkq=((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd;
		for (k=nd2_tyy[12]; k <= nd2_tyy[17]; k++)
		//do k=nd2_tyy(13),nd2_tyy(18)
		{
			damp2=1./(1.+damp2_x(k,j,lb)*rti);
			damp1=damp2*2.0-1.0;
			inod=idmat2(k,i,j);
			cl=clamda(inod);
			sm2=2.*cmu(inod);
			pm=cl+sm2;
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw));
			wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxx=t2xx(k,i,j)-t2xx_px(k,ib,j);
			taoyy=t2yy(k,i,j)-t2yy_px(k,ib,j);
			taozz=t2zz(k,i,j)-t2yy_px(k,ib,j);
			if(j>nd2_tyy[1] && j<nd2_tyy[4]) {
			//if(j>nd2_tyy(2) .and. j<nd2_tyy(5)) {
				syy=dyh2(1,j)*v2y(k,i,j-2)+dyh2(2,j)*v2y(k,i,j-1)+
					dyh2(3,j)*v2y(k,i  ,j)+dyh2(4,j)*v2y(k,i,j+1);
				if(k<nd2_tyy[16]) {
				//if(k<nd2_tyy(17)) {
					szz=dzi2(1,k)*v2z(k-1,i,j)+dzi2(2,k)*v2z(k,  i,j)+ 
						dzi2(3,k)*v2z(k+1,i,j)+dzi2(4,k)*v2z(k+2,i,j);
				}
				else {
					szz=0.0;
				}
				sss=syy+szz;
				qxx=qt2xx(k,i,j);
				qt2xx(k,i,j)=qxx*et+(wtp-wts)*sss*et1;
				taoxx=taoxx+cl*sss-qxx-qt2xx(k,i,j);
				qyy=qt2yy(k,i,j);
				qt2yy(k,i,j)=qyy*et+(wtp*sss-wts*szz)*et1;
				taoyy=taoyy+sm2*syy+cl*sss-qyy-qt2yy(k,i,j);
				qzz=qt2zz(k,i,j);
				qt2zz(k,i,j)=qzz*et+(wtp*sss-wts*syy)*et1;
				taozz=taozz+sm2*szz+cl*sss-qzz-qt2zz(k,i,j);
			}
			sxx=dxh2(2,i)/ca*(v2x(k,i-1,j)-v2x(k,i,j));
			qxx=qt2xx_px(k,ib,j);
			qt2xx_px(k,ib,j)=qxx*et+wtp*sxx*et1;
			t2xx_px(k,ib,j)=damp1*t2xx_px(k,ib,j)+
							damp2*(pm*sxx-qxx-qt2xx_px(k,ib,j));
			t2xx(k,i,j)=taoxx+t2xx_px(k,ib,j);
			qyy=qt2yy_px(k,ib,j);
			qt2yy_px(k,ib,j)=qyy*et+(wtp-wts)*sxx*et1;
			t2yy_px(k,ib,j)=damp1*t2yy_px(k,ib,j)+
							damp2*(cl*sxx-qyy-qt2yy_px(k,ib,j));
			t2yy(k,i,j)=taoyy+t2yy_px(k,ib,j);
			t2zz(k,i,j)=taozz+t2yy_px(k,ib,j);
		}
	}
//		}
//	}
	return;
}

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
						 float ca,
						 float *drti2M,
						 float *damp2_yM,
						 float *clamdaM,
						 float *cmuM,
						 float *epdtM,
						 float *qwpM,
						 float *qwsM,
						 float *qwt1M,
						 float *qwt2M,
						 float *dxh2M,
						 float *dyh2M,
						 float *dzi2M,
						 float *t2xxM,
						 float *t2yyM,
						 float *t2zzM,
						 float *qt2xxM,
						 float *qt2yyM,
						 float *qt2zzM,
						 float *t2xx_pyM,
						 float *t2yy_pyM,
						 float *qt2xx_pyM,
						 float *qt2yy_pyM,
						 float *v2xM,
						 float *v2yM,
						 float *v2zM)
//Compute the stress-norm at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	float taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

	//if( lby[0] > lby[1] ) return;
	//if( lby(1)>lby(2) ) return;

	i = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyy[6];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;

	if (i > nd2_tyy[11] || lb > lby1)
	{
		return;
	}

	jb = 0;
	for (k = lby0; k < lb; k++)
	{
		for (j=nd2_tyy[4*k]; j <= nd2_tyy[1+4*k]; j++)
		{
			jb++;
		}
	}

//	for (i = nd2_tyy[6]; i <= nd2_tyy[11]; i++)
//	//do i=nd2_tyy(7),nd2_tyy(12)
//	{
//		jb=0;
//		for (lb = lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1),lby(2)
//		{
	kb=0;
	for (j=nd2_tyy[4*lb]; j <= nd2_tyy[1+4*lb]; j++)
	//do j=nd2_tyy(1+4*lb),nd2_tyy(2+4*lb)
	{
		kb=kb+1;
		jb=jb+1;
		rti=drti2(kb,lb);
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1;
		jkq=((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd
		for (k=nd2_tyy[12]; k <= nd2_tyy[17]; k++)
		//do k=nd2_tyy(13),nd2_tyy(18)
		{
			damp2=1./(1.+damp2_y(k,i,lb)*rti);
			damp1=damp2*2.0-1.;
			inod=idmat2(k,i,j);
			cl=clamda(inod);
			sm2=2.*cmu(inod);
			pm=cl+sm2;
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw));
			wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxx=t2xx(k,i,j)-t2xx_py(k,i,jb);
			taoyy=t2yy(k,i,j)-t2yy_py(k,i,jb);
			taozz=t2zz(k,i,j)-t2xx_py(k,i,jb);
			if(k<nd2_tyy[16]) {
			//if(k<nd2_tyy(17)) then
			  szz=dzi2(1,k)*v2z(k-1,i,j)+dzi2(2,k)*v2z(k,  i,j)+ 
				  dzi2(3,k)*v2z(k+1,i,j)+dzi2(4,k)*v2z(k+2,i,j);
			  if(i>nd2_tyy[7] && i<nd2_tyy[10]) {
			  //if(i>nd2_tyy(8) .and. i<nd2_tyy(11)) {
				sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+
					dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j);
			  }
			  else {
				sxx=0.0;
			  }
			  sss=sxx+szz;
			  qxx=qt2xx(k,i,j);
			  qt2xx(k,i,j)=qxx*et+(wtp*sss-wts*szz)*et1;
			  taoxx=taoxx+sm2*sxx+cl*sss-qxx-qt2xx(k,i,j);
			  qyy=qt2yy(k,i,j);
			  qt2yy(k,i,j)=qyy*et+(wtp-wts)*sss*et1;
			  taoyy=taoyy+cl*sss-qyy-qt2yy(k,i,j);
			  qzz=qt2zz(k,i,j);
			  qt2zz(k,i,j)=qzz*et+(wtp*sss-wts*sxx)*et1;
			  taozz=taozz+sm2*szz+cl*sss-qzz-qt2zz(k,i,j);
			}
			else {
			  if(i>nd2_tyy[7] && i<nd2_tyy[10]) {
			  //if(i>nd2_tyy(8) .and. i<nd2_tyy(11)) then
				sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+
					dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j);
				qxx=qt2xx(k,i,j);
				qt2xx(k,i,j)=qxx*et+wtp*sxx*et1;
				taoxx=taoxx+pm*sxx-qxx-qt2xx(k,i,j);
				qyy=qt2yy(k,i,j);
				qt2yy(k,i,j)=qyy*et+(wtp-wts)*sxx*et1;
				taoyy=taoyy+cl*sxx-qyy-qt2yy(k,i,j);
				qzz=qt2zz(k,i,j);
				qt2zz(k,i,j)=qzz*et+(wtp-wts)*sxx*et1;
				taozz=taozz+cl*sxx-qzz-qt2zz(k,i,j);
			  }
			}
			syy=dyh2(2,j)/ca*(v2y(k,i,j-1)-v2y(k,i,j));
			qxx=qt2xx_py(k,i,jb);
			qt2xx_py(k,i,jb)=qxx*et+(wtp-wts)*syy*et1;
			t2xx_py(k,i,jb)=damp1*t2xx_py(k,i,jb)+damp2*(cl*syy-qxx-qt2xx_py(k,i,jb));
			t2xx(k,i,j)=taoxx+t2xx_py(k,i,jb);
			t2zz(k,i,j)=taozz+t2xx_py(k,i,jb);
			qyy=qt2yy_py(k,i,jb);
			qt2yy_py(k,i,jb)=qyy*et+wtp*syy*et1;
			t2yy_py(k,i,jb)=damp1*t2yy_py(k,i,jb)+damp2*(pm*syy-qyy-qt2yy_py(k,i,jb));
			t2yy(k,i,j)=taoyy+t2yy_py(k,i,jb);
		}
	}
//		}
//	}
	return;
}

__global__ void stress_norm_PmlZ_IIC(int nxb2,
						  int nyb2,
						  int mw2_pml,
						  int mw2_pml1,
						  int nztop,
						  int nxbtm,
						  int nzbtm,
						  int *nd2_tyy,
						  int *idmat2M,
						  float ca,
						  float *damp2_zM,
						  float *drth2M,
						  float *clamdaM,
						  float *cmuM,
						  float *epdtM,
						  float *qwpM,
						  float *qwsM,
						  float *qwt1M,
						  float *qwt2M,
						  float *dxh2M,
						  float *dyh2M,
						  float *dzi2M,
						  float *t2xxM,
						  float *t2yyM,
						  float *t2zzM,
						  float *qt2xxM,
						  float *qt2yyM,
						  float *qt2zzM,
						  float *t2xx_pzM,
						  float *t2zz_pzM,
						  float *qt2xx_pzM,
						  float *qt2zz_pzM,
						  float *v2xM,
						  float *v2yM,
						  float *v2zM)
//Compute the stress-norm at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	float taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyy[0];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_tyy[6];

	if (j > nd2_tyy[5] || i > nd2_tyy[11])
	{
		return;
	}

//	for (j = nd2_tyy[0]; j <= nd2_tyy[5]; j++)
//	//do j=nd2_tyy(1),nd2_tyy(6)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		for (i=nd2_tyy[6]; i <= nd2_tyy[11]; i++)
//		//do i=nd2_tyy(7),nd2_tyy(12)
//		{
	jkq=((i+nxb2)&1)+kodd;
	//jkq=mod(i+nxb2,2)+kodd
	kb=0;
	for (k = nd2_tyy[16]; k <= nd2_tyy[17]; k++)
	//do k=nd2_tyy(17),nd2_tyy(18)
	{
		kb=kb+1;
		damp2=1./(1.+damp2_z(i,j)*drth2(kb,1));
		damp1=damp2*2.-1.;
		inod=idmat2(k,i,j);
		cl=clamda(inod);
		sm2=2.*cmu(inod);
		pm=cl+sm2;
		irw=jkq+4*((k+nztop)&1);
		//irw=jkq+4*mod(k+nztop,2)
		et=epdt(irw);
		et1=1.0-et;
		wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw));
		wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		taoxx=t2xx(k,i,j)-t2xx_pz(kb,i,j);
		taoyy=t2yy(k,i,j)-t2xx_pz(kb,i,j);
		taozz=t2zz(k,i,j)-t2zz_pz(kb,i,j);
		if(i>nd2_tyy[7] && i<nd2_tyy[10] && j>nd2_tyy[1] && j<nd2_tyy[4]) {
		//if(i>nd2_tyy(8) .and. i<nd2_tyy(11) .and. &
		//   j>nd2_tyy(2) .and. j<nd2_tyy(5)) then
		  sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+ 
			  dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j);
		  syy=dyh2(1,j)*v2y(k,i,j-2)+dyh2(2,j)*v2y(k,i,j-1)+ 
			  dyh2(3,j)*v2y(k,i  ,j)+dyh2(4,j)*v2y(k,i,j+1);
		  sss=sxx+syy;
		  qxx=qt2xx(k,i,j);
		  qt2xx(k,i,j)=qxx*et+(wtp*sss-wts*syy)*et1;
		  taoxx=taoxx+sm2*sxx+cl*sss-qxx-qt2xx(k,i,j);
		  qyy=qt2yy(k,i,j);
		  qt2yy(k,i,j)=qyy*et+(wtp*sss-wts*sxx)*et1;
		  taoyy=taoyy+sm2*syy+cl*sss-qyy-qt2yy(k,i,j);
		  qzz=qt2zz(k,i,j);
		  qt2zz(k,i,j)=qzz*et+(wtp-wts)*sss*et1;
		  taozz=taozz+cl*sss-qzz-qt2zz(k,i,j);
		}
		szz=dzi2(2,k)/ca*(v2z(k,i,j)-v2z(k+1,i,j));
		qxx=qt2xx_pz(kb,i,j);
		qt2xx_pz(kb,i,j)=qxx*et+(wtp-wts)*szz*et1;
		t2xx_pz(kb,i,j)=damp1*t2xx_pz(kb,i,j)+
						damp2*(cl*szz-qxx-qt2xx_pz(kb,i,j));
		t2xx(k,i,j)=taoxx+t2xx_pz(kb,i,j);
		t2yy(k,i,j)=taoyy+t2xx_pz(kb,i,j);
		qzz=qt2zz_pz(kb,i,j);
		qt2zz_pz(kb,i,j)=qzz*et+wtp*szz*et1;
		t2zz_pz(kb,i,j)=damp1*t2zz_pz(kb,i,j)+
						damp2*(pm*szz-qzz-qt2zz_pz(kb,i,j));
		t2zz(k,i,j)=taozz+t2zz_pz(kb,i,j);
	}

//		}
//	}
	return;
}

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
						float ca,
						float *drth2M,
						float *damp2_xM,
						float *cmuM,
						float *epdtM,
						float *qwsM,
						float *qwt1M,
						float *qwt2M,
						float *dxi2M,
						float *dyi2M,
						float *t2xyM,
						float *qt2xyM,
						float *t2xy_pxM,
						float *qt2xy_pxM,
						float *v2xM,
						float *v2yM)
//Compute the Stress-xy at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	float taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	//if(lbx[0] > lbx[1]) return;
	nth = (lbx1 - lbx0 + 1) * mw2_pml + 1 - lbx0;
	//nth = (lbx(2) - lbx(1) + 1) * mw2_pml + 1 - lbx(1)

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_txy[0];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

	if (j > nd2_txy[5] || lb > lbx1)
	{
		return;
	}

	ib = 0;
	for (k = lbx0; k < lb; k++)
	{
		for (i=nd2_txy[6+4*k]; i <= nd2_txy[7+4*k]; i++)
		{
			ib++;
		}
	}

//	for (j = nd2_txy[0]; j <= nd2_txy[5]; j++)
//	//do j=nd2_txy(1),nd2_txy(6)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		ib=0;
//		for (lb = lbx[0]; lb <= lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	kb=0;
	for (i=nd2_txy[6+4*lb]; i <= nd2_txy[7+4*lb]; i++)
	//do i=nd2_txy(7+4*lb),nd2_txy(8+4*lb)
	{
		kb=kb+1;
		ib=ib+1;
		rth=drth2(kb,lb);
		jkq=((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd
		for (k = nd2_txy[12]; k <= nd2_txy[17]; k++)
		//do k=nd2_txy(13),nd2_txy(18)
		{
			damp2=1./(1.+damp2_x(k,j,lb)*rth);
			damp1=damp2*2.0-1.;
			inod=idmat2(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat2(k,i+1,j+1)));
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxy=t2xy(k,i,j)-t2xy_px(k,ib,j);
			if(j > nd2_txy[1] && j<nd2_txy[4]) {
			//if(j>nd2_txy(2) .and. j<nd2_txy(5)) then
				cusxy=(dyi2(1,j)*v2x(k,i,j-1)+dyi2(2,j)*v2x(k,i,j)+ 
					 dyi2(3,j)*v2x(k,i,j+1)+dyi2(4,j)*v2x(k,i,j+2))*sm;
				qxy=qt2xy(k,i,j);
				qt2xy(k,i,j)=qxy*et+dmws*cusxy*et1;
				taoxy=taoxy+cusxy-qxy-qt2xy(k,i,j);
			}
			cusxy=sm*dxi2(2,i)/ca*(v2y(k,i,j)-v2y(k,i+1,j));
			qxy=qt2xy_px(k,ib,j);
			qt2xy_px(k,ib,j)=qxy*et+dmws*cusxy*et1;
			t2xy_px(k,ib,j)=damp1*t2xy_px(k,ib,j)+
							damp2*(cusxy-qxy-qt2xy_px(k,ib,j));
			t2xy(k,i,j)=taoxy+t2xy_px(k,ib,j);
		}
	}
//		}
//	}
	return;
}

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
						float ca,
						float *drth2M,
						float *damp2_yM,
						float *cmuM,
						float *epdtM,
						float *qwsM,
						float *qwt1M,
						float *qwt2M,
						float *dxi2M,
						float *dyi2M,
						float *t2xyM,
						float *qt2xyM,
						float *t2xy_pyM,
						float *qt2xy_pyM,
						float *v2xM,
						float *v2yM)
//Compute the Stress-xy at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	float taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;

	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return

	i = blockIdx.x * blockDim.x + threadIdx.x + nd2_txy[6];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;

	if (i > nd2_txy[11] || lb > lby1)
	{
		return;
	}

	jb = 0;
	for (k = lby0; k < lb; k++)
	{
		for (j=nd2_txy[4*k]; j <= nd2_txy[1+4*k]; j++)
		{
			jb++;
		}
	}

//	for (i = nd2_txy[6]; i <= nd2_txy[11]; i++)
//	//do i=nd2_txy(7),nd2_txy(12)
//	{
//		jb=0;
//		for (lb=lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1),lby(2)
//		{
	kb=0;
	for (j=nd2_txy[4*lb]; j <= nd2_txy[1+4*lb]; j++)
	//do j=nd2_txy(1+4*lb),nd2_txy(2+4*lb)
	{
		kb=kb+1;
		jb=jb+1;
		rth=drth2(kb,lb);
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		jkq=((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd
		for (k = nd2_txy[12]; k <= nd2_txy[17]; k++)
		//do k=nd2_txy(13),nd2_txy(18)
		{
			damp2=1./(1.+damp2_y(k,i,lb)*rth);
			damp1=damp2*2.-1.;
			inod=idmat2(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat2(k,i+1,j+1)));
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxy=t2xy(k,i,j)-t2xy_py(k,i,jb);
			if(i>nd2_txy[7] && i<nd2_txy[10]) {
			//if(i>nd2_txy(8) .and. i<nd2_txy(11)) then
				cusxy=(dxi2(1,i)*v2y(k,i-1,j)+dxi2(2,i)*v2y(k,i,j)+ 
					 dxi2(3,i)*v2y(k,i+1,j)+dxi2(4,i)*v2y(k,i+2,j))*sm;
				qxy=qt2xy(k,i,j);
				qt2xy(k,i,j)=qxy*et+dmws*cusxy*et1;
				taoxy=taoxy+cusxy-qxy-qt2xy(k,i,j);
			}
			cusxy=sm*dyi2(2,j)/ca*(v2x(k,i,j)-v2x(k,i,j+1));
			qxy=qt2xy_py(k,i,jb);
			qt2xy_py(k,i,jb)=qxy*et+dmws*cusxy*et1;
			t2xy_py(k,i,jb)=damp1*t2xy_py(k,i,jb)+ 
							damp2*(cusxy-qxy-qt2xy_py(k,i,jb));
			t2xy(k,i,j)=taoxy+t2xy_py(k,i,jb);
		}
	}
//		}
//	}
	return;
}

__global__ void stress_xy_PmlZ_II(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int *nd2_txy,
					   int *idmat2M,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dxi2M,
					   float *dyi2M,
					   float *t2xyM,
					   float *qt2xyM,
					   float *v2xM,
					   float *v2yM)
//Compute the Stress-xy at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusxy,qxy,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	float cusxy,qxy,sm,dmws,et,et1;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_txy[2];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_txy[8];

	if (j > nd2_txy[3] || i > nd2_txy[9])
	{
		return;
	}

//	for (j = nd2_txy[2]; j <= nd2_txy[3]; j++)
//	//do j=nd2_txy(3),nd2_txy(4)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		for (i = nd2_txy[8]; i <= nd2_txy[9]; i++)
//		//do i=nd2_txy(9),nd2_txy(10)
//		{
	jkq=((i+nxb2)&1)+kodd;
	//jkq=mod(i+nxb2,2)+kodd
	for (k=nd2_txy[16]; k <= nd2_txy[17]; k++)
	//do k=nd2_txy(17),nd2_txy(18)
	{
		inod=idmat2(k,i,j);
		sm=2./(1./cmu(inod)+1./cmu(idmat2(k,i+1,j+1)));
		irw=jkq+4*((k+nztop)&1);
		//irw=jkq+4*mod(k+nztop,2);
		et=epdt(irw);
		et1=1.0-et;
		dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		cusxy=(dxi2(1,i)*v2y(k,i-1,j)+dxi2(2,i)*v2y(k,i,  j)+
			   dxi2(3,i)*v2y(k,i+1,j)+dxi2(4,i)*v2y(k,i+2,j)+
			   dyi2(1,j)*v2x(k,i,j-1)+dyi2(2,j)*v2x(k,i,j  )+
			   dyi2(3,j)*v2x(k,i,j+1)+dyi2(4,j)*v2x(k,i,j+2))*sm;
		qxy=qt2xy(k,i,j);
		qt2xy(k,i,j)=qxy*et+dmws*cusxy*et1;
		t2xy(k,i,j)=t2xy(k,i,j)+cusxy-qxy-qt2xy(k,i,j);
	}
//		}
//	}

	return;
}

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
						float ca,
						float *drth2M,
						float *damp2_xM,
						float *cmuM,
						float *epdtM,
						float *qwsM,
						float *qwt1M,
						float *qwt2M,
						float *dxi2M,
						float *dzh2M,
						float *t2xzM,
						float *qt2xzM,
						float *t2xz_pxM,
						float *qt2xz_pxM,
						float *v2xM,
						float *v2zM)
//Compute the stress-xz at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	float taoxz,cusxz,qxz,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	//if(lbx[0] > lbx[1]) return;
	nth = (lbx1 - lbx0 + 1) * mw2_pml + 1 - lbx0;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_txz[0];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

	if (j > nd2_txz[5] || lb > lbx1)
	{
		return;
	}

	ib=0;
	for (k = lbx0; k < lb; k++)
	{
		for (i=nd2_txz[6+4*k]; i <= nd2_txz[7+4*k]; i++)
		{
			ib++;
		}
	}

//	for (j = nd2_txz[0]; j <= nd2_txz[5]; j++)
//	//do j=nd2_txz(1),nd2_txz(6)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		ib=0;
//		for (lb=lbx[0]; lb <= lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	kb=0;
	for (i=nd2_txz[6+4*lb]; i <= nd2_txz[7+4*lb]; i++)
	//do i=nd2_txz(7+4*lb),nd2_txz(8+4*lb)
	{
		kb=kb+1;
		ib=ib+1;
		rth=drth2(kb,lb);
		jkq=((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd
		for (k = nd2_txz[12]; k <= nd2_txz[17]; k++)
		//do k=nd2_txz(13),nd2_txz(18)
		{
			damp2=1./(1.+damp2_x(k,j,lb)*rth);
			damp1=damp2*2.-1.;
			inod=idmat2(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i+1,j)));
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoxz=t2xz(k,i,j)-t2xz_px(k,ib,j);
			if(k < nd2_txz[16]) {
			//if(k<nd2_txz(17)) then
			  cusxz=(dzh2(1,k)*v2x(k-2,i,j)+dzh2(2,k)*v2x(k-1,i,j)+
					 dzh2(3,k)*v2x(k,i,j)+dzh2(4,k)*v2x(k+1,i,j))*sm;
			  qxz=qt2xz(k,i,j);
			  qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1;
			  taoxz=taoxz+cusxz-qxz-qt2xz(k,i,j);
			}
			cusxz=sm*dxi2(2,i)/ca*(v2z(k,i,j)-v2z(k,i+1,j));
			qxz=qt2xz_px(k,ib,j);
			qt2xz_px(k,ib,j)=qxz*et+dmws*cusxz*et1;
			t2xz_px(k,ib,j)=damp1*t2xz_px(k,ib,j)+
							damp2*(cusxz-qxz-qt2xz_px(k,ib,j));
			t2xz(k,i,j)=taoxz+t2xz_px(k,ib,j);
		}
	}
//		}
//	}
	return;
}

__global__ void stress_xz_PmlY_IIC(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int lby0,
					   int lby1,
					   int *nd2_txz,
					   int *idmat2M,
					   float *cmuM,
					   float *epdtM,
					   float *qwsM,
					   float *qwt1M,
					   float *qwt2M,
					   float *dxi2M,
					   float *dzh2M,
					   float *v2xM,
					   float *v2zM,
					   float *t2xzM,
					   float *qt2xzM)
//Compute the stress-xz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: dvxz,dvzx,cusxz,qxz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	float dvxz,dvzx,cusxz,qxz,sm,dmws,et,et1;
	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return

	i = blockIdx.x * blockDim.x + threadIdx.x + nd2_txz[8];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;

	if (i > nd2_txz[9] || lb > lby1)
	{
		return;
	}

//	for (i = nd2_txz[8]; i <= nd2_txz[9]; i++)
//	//do i=nd2_txz(9),nd2_txz(10)
//	{
//		for (lb = lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1),lby(2)
//		{
	for (j=nd2_txz[4*lb]; j <= nd2_txz[1+4*lb]; j++)
	//do j=nd2_txz(1+4*lb),nd2_txz(2+4*lb)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		jkq=((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd
		for (k = nd2_txz[12]; k <= nd2_txz[15]; k++)
		//do k=nd2_txz(13),nd2_txz(16)
		{
			inod=idmat2(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i+1,j)));
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			dvzx=dxi2(1,i)*v2z(k,i-1,j)+dxi2(2,i)*v2z(k,i,  j)+
				 dxi2(3,i)*v2z(k,i+1,j)+dxi2(4,i)*v2z(k,i+2,j);
			dvxz=dzh2(1,k)*v2x(k-2,i,j)+dzh2(2,k)*v2x(k-1,i,j)+
				 dzh2(3,k)*v2x(k,  i,j)+dzh2(4,k)*v2x(k+1,i,j);
			cusxz=(dvzx+dvxz)*sm;
			qxz=qt2xz(k,i,j);
			qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1;
			t2xz(k,i,j)=t2xz(k,i,j)+cusxz-qxz-qt2xz(k,i,j);
		}
	}

//		}
//	}
	return;
}

__global__ void stress_xz_PmlZ_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *nd2_txz,
						int *idmat2M,
						float ca,
						float *drti2M,
						float *damp2_zM,
						float *cmuM,
						float *epdtM,
						float *qwsM,
						float *qwt1M,
						float *qwt2M,
						float *dxi2M,
						float *dzh2M,
						float *t2xzM,
						float *qt2xzM,
						float *t2xz_pzM,
						float *qt2xz_pzM,
						float *v2xM,
						float *v2zM)
//Compute the stress-xz at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	float taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1;
		
	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_txz[0];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_txz[6];

	if (j > nd2_txz[5] || i > nd2_txz[11])
	{
		return;
	}

//	for (j = nd2_txz[0]; j <= nd2_txz[5]; j++)
//	//do j=nd2_txz(1),nd2_txz(6)
//	{
	kodd = 2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		for (i = nd2_txz[6]; i <= nd2_txz[11]; i++)
//		//do i=nd2_txz(7),nd2_txz(12)
//		{
	jkq=((i+nxb2)&1)+kodd;
	//jkq=mod(i+nxb2,2)+kodd
	kb=0;
	for (k = nd2_txz[16]; k <= nd2_txz[17]; k++)
	//do k=nd2_txz(17),nd2_txz(18)
	{
		kb=kb+1;
		damp2=1./(1.+damp2_z(i,j)*drti2(kb,1));
		damp1=damp2*2.-1.;
		inod=idmat2(k,i,j);
		sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i+1,j)));
		irw=jkq+4*((k+nztop)&1);
		//irw=jkq+4*mod(k+nztop,2)
		et=epdt(irw);
		et1=1.0-et;
		dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		taoxz=t2xz(k,i,j)-t2xz_pz(kb,i,j);
		if(i > nd2_txz[7] && i<nd2_txz[10]) {
		//if(i>nd2_txz(8) .and. i<nd2_txz(11)) then
			cusxz=(dxi2(1,i)*v2z(k,i-1,j)+dxi2(2,i)*v2z(k,i,  j)+ 
				 dxi2(3,i)*v2z(k,i+1,j)+dxi2(4,i)*v2z(k,i+2,j))*sm;
			qxz=qt2xz(k,i,j);
			qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1;
			taoxz=taoxz+cusxz-qxz-qt2xz(k,i,j);
		}
		cusxz=sm*dzh2(2,k)/ca*(v2x(k-1,i,j)-v2x(k,i,j));
		qxz=qt2xz_pz(kb,i,j);
		qt2xz_pz(kb,i,j)=qxz*et+dmws*cusxz*et1;
		t2xz_pz(kb,i,j)=damp1*t2xz_pz(kb,i,j)+
						damp2*(cusxz-qxz-qt2xz_pz(kb,i,j));
		t2xz(k,i,j)=taoxz+t2xz_pz(kb,i,j);
	}
//		}
//	}
	return;
}

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
						float *cmuM,
						float *epdtM,
						float *qwsM,
						float *qwt1M,
						float *qwt2M,
						float *dyi2M,
						float *dzh2M,
						float *t2yzM,
						float *qt2yzM,
						float *v2yM,
						float *v2zM)
//Compute the stress-yz at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusyz,qyz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	float cusyz,qyz,sm,dmws,et,et1;

	//if(lbx[0] > lbx[1]) return;
	//if( lbx(1)>lbx(2) ) return

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyz[2];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

	if (j > nd2_tyz[3] || lb > lbx1)
	{
		return;
	}

//	for (j=nd2_tyz[2]; j <= nd2_tyz[3]; j++)
//	//do j=nd2_tyz(3),nd2_tyz(4)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		for (lb = lbx[0]; lb <= lbx[1]; lb++)
//		//do lb=lbx(1),lbx(2)
//		{
	for (i = nd2_tyz[6+4*lb]; i <= nd2_tyz[7+4*lb]; i++)
	//do i=nd2_tyz(7+4*lb),nd2_tyz(8+4*lb)
	{
		jkq=((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd
		for (k = nd2_tyz[12]; k <= nd2_tyz[15]; k++)
		//do k=nd2_tyz(13),nd2_tyz(16)
		{
			inod=idmat2(k,i,j);
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			cusyz=(dyi2(1,j)*v2z(k,i,j-1)+dyi2(2,j)*v2z(k,i,j  )+
				   dyi2(3,j)*v2z(k,i,j+1)+dyi2(4,j)*v2z(k,i,j+2)+
				   dzh2(1,k)*v2y(k-2,i,j)+dzh2(2,k)*v2y(k-1,i,j)+
				   dzh2(3,k)*v2y(k,  i,j)+dzh2(4,k)*v2y(k+1,i,j))/
				   (.5/cmu(inod)+.5/cmu(idmat2(k-1,i,j+1)));
			qyz=qt2yz(k,i,j);
			qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1;
			t2yz(k,i,j)=t2yz(k,i,j)+cusyz-qyz-qt2yz(k,i,j);
		}
	}
//		}
//	}
	return;
}

//call stress_yz_PmlY_II
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
						float ca,
						float *drth2M,
						float *damp2_yM,
						float *cmuM,
						float *epdtM,
						float *qwsM,
						float *qwt1M,
						float *qwt2M,
						float *dyi2M,
						float *dzh2M,
						float *t2yzM,
						float *qt2yzM,
						float *t2yz_pyM,
						float *qt2yz_pyM,
						float *v2yM,
						float *v2zM)
//Compute the stress-yz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	float taoyz,cusyz,qyz,rth,damp2,damp1,sm,dmws,et,et1;

	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	i = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyz[6];
	lb = blockIdx.y * blockDim.y + threadIdx.y + lby0;

	if (i > nd2_tyz[11] || lb > lby1)
	{
		return;
	}

	jb = 0;
	for (k = lby0; k < lb; k++)
	{
		for (j = nd2_tyz[4*k]; j <= nd2_tyz[1+4*k]; j++)
		{
			jb++;
		}
	}

//	for (i = nd2_tyz[6]; i <= nd2_tyz[11]; i++)
//	//do i=nd2_tyz(7),nd2_tyz(12)
//	{
//		jb=0;
//		for (lb = lby[0]; lb <= lby[1]; lb++)
//		//do lb=lby(1),lby(2)
//		{
	kb=0;
	for (j = nd2_tyz[4*lb]; j <= nd2_tyz[1+4*lb]; j++)
	//do j=nd2_tyz(1+4*lb),nd2_tyz(2+4*lb)
	{
		kb=kb+1;
		jb=jb+1;
		rth=drth2(kb,lb);
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		jkq = ((i+nxb2)&1)+kodd;
		//jkq=mod(i+nxb2,2)+kodd
		for (k = nd2_tyz[12]; k <= nd2_tyz[17]; k++)
		//do k=nd2_tyz(13),nd2_tyz(18)
		{
			damp2=1./(1.+damp2_y(k,i,lb)*rth);
			damp1=damp2*2.-1.;
			inod=idmat2(k,i,j);
			sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i,j+1)));
			irw=jkq+4*((k+nztop)&1);
			//irw=jkq+4*mod(k+nztop,2)
			et=epdt(irw);
			et1=1.0-et;
			dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
			taoyz=t2yz(k,i,j)-t2yz_py(k,i,jb);
			if(k<nd2_tyz[16]) {
			//if(k<nd2_tyz(17)) {
			  cusyz=(dzh2(1,k)*v2y(k-2,i,j)+dzh2(2,k)*v2y(k-1,i,j)+
					 dzh2(3,k)*v2y(k,  i,j)+dzh2(4,k)*v2y(k+1,i,j))*sm;
			  qyz=qt2yz(k,i,j);
			  qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1;
			  taoyz=taoyz+cusyz-qyz-qt2yz(k,i,j);
			}
			cusyz=sm*dyi2(2,j)/ca*(v2z(k,i,j)-v2z(k,i,j+1));
			qyz=qt2yz_py(k,i,jb);
			qt2yz_py(k,i,jb)=qyz*et+dmws*cusyz*et1;
			t2yz_py(k,i,jb)=damp1*t2yz_py(k,i,jb)+
							damp2*(cusyz-qyz-qt2yz_py(k,i,jb));
			t2yz(k,i,j)=taoyz+t2yz_py(k,i,jb);
		}
	}
//		}
//	}
	return;
}

//call stress_yz_PmlZ_II
__global__ void stress_yz_PmlZ_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *nd2_tyz,
						int *idmat2M,
						float ca,
						float *drti2M,
						float *damp2_zM,
						float *cmuM,
						float *epdtM,
						float *qwsM,
						float *qwt1M,
						float *qwt2M,
						float *dyi2M,
						float *dzh2M,
						float *t2yzM,
						float *qt2yzM,
						float *t2yz_pzM,
						float *qt2yz_pzM,
						float *v2yM,
						float *v2zM)
//Compute the stress-yz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	float taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_tyz[0];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_tyz[6];

	if (j > nd2_tyz[5] || i > nd2_tyz[11])
	{
		return;
	}

//	for (j = nd2_tyz[0]; j <= nd2_tyz[5]; j++)
//	//do j=nd2_tyz(1),nd2_tyz(6)
//	{
	kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
//		for (i = nd2_tyz[6]; i <= nd2_tyz[11]; i++)
//		//do i=nd2_tyz(7),nd2_tyz(12)
//		{
	jkq = ((i+nxb2)&1)+kodd;
	//jkq=mod(i+nxb2,2)+kodd
	kb=0;
	for (k = nd2_tyz[16]; k <= nd2_tyz[17]; k++)
	//do k=nd2_tyz(17),nd2_tyz(18)
	{
		kb=kb+1;
		damp2=1./(1.+damp2_z(i,j)*drti2(kb,1));
		damp1=damp2*2.-1.;
		inod=idmat2(k,i,j);
		sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i,j+1)));
		irw=jkq+4*((k+nztop)&1);
		//irw=jkq+4*mod(k+nztop,2);
		et=epdt(irw);
		et1=1.0-et;
		dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw));
		taoyz=t2yz(k,i,j)-t2yz_pz(kb,i,j);
		if (j > nd2_tyz[1] && j<nd2_tyz[4]) {
		//if(j>nd2_tyz(2) .and. j<nd2_tyz(5)) then
			cusyz=(dyi2(1,j)*v2z(k,i,j-1)+dyi2(2,j)*v2z(k,i,j)+
				 dyi2(3,j)*v2z(k,i,j+1)+dyi2(4,j)*v2z(k,i,j+2))*sm;
			qyz=qt2yz(k,i,j);
			qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1;
			taoyz=taoyz+cusyz-qyz-qt2yz(k,i,j);
		}
		cusyz=sm*dzh2(2,k)/ca*(v2y(k-1,i,j)-v2y(k,i,j));
		qyz=qt2yz_pz(kb,i,j);
		qt2yz_pz(kb,i,j)=qyz*et+dmws*cusyz*et1;
		t2yz_pz(kb,i,j)=damp1*t2yz_pz(kb,i,j)+
						damp2*(cusyz-qyz-qt2yz_pz(kb,i,j));
		t2yz(k,i,j)=taoyz+t2yz_pz(kb,i,j);
	}
//		}
//	}
	return;
}

#ifdef __cplusplus
extern "C" {
#endif

void compute_stressC(int *nxb1, int *nyb1, int *nx1p1, int *ny1p1, int *nxtop, int *nytop, int *nztop, int *mw1_pml,
		int *mw1_pml1, int *nmat, int *nll, int *lbx, int *lby, int *nd1_txy, int *nd1_txz,
		int *nd1_tyy, int *nd1_tyz, int *idmat1M, float *ca, float *drti1M, float *drth1M, float *damp1_xM, float *damp1_yM,
		float *clamdaM, float *cmuM, float *epdtM, float *qwpM, float *qwsM, float *qwt1M, float *qwt2M, float *dxh1M,
		float *dyh1M, float *dzh1M, float *dxi1M, float *dyi1M, float *dzi1M, float *t1xxM, float *t1xyM, float *t1xzM, 
		float *t1yyM, float *t1yzM, float *t1zzM, float *qt1xxM, float *qt1xyM, float *qt1xzM, float *qt1yyM, float *qt1yzM, 
		float *qt1zzM, float *t1xx_pxM, float *t1xy_pxM, float *t1xz_pxM, float *t1yy_pxM, float *qt1xx_pxM, float *qt1xy_pxM,
		float *qt1xz_pxM, float *qt1yy_pxM, float *t1xx_pyM, float *t1xy_pyM, float *t1yy_pyM, float *t1yz_pyM, float *qt1xx_pyM,
		float *qt1xy_pyM, float *qt1yy_pyM, float *qt1yz_pyM, void **v1xMp, void **v1yMp, void **v1zMp,
		int *nxb2, int *nyb2, int *nxbtm, int *nybtm, int *nzbtm, int *mw2_pml, int *mw2_pml1, int *nd2_txy, int *nd2_txz, 
		int *nd2_tyy, int *nd2_tyz, int *idmat2M, 
		float *drti2M, float *drth2M, float *damp2_xM, float *damp2_yM, float *damp2_zM, 
		float *t2xxM, float *t2xyM, float *t2xzM, float *t2yyM, float *t2yzM, float *t2zzM, 
		float *qt2xxM, float *qt2xyM, float *qt2xzM, float *qt2yyM, float *qt2yzM, float *qt2zzM, 
		float *dxh2M, float *dyh2M, float *dzh2M, float *dxi2M, float *dyi2M, float *dzi2M, 
		float *t2xx_pxM, float *t2xy_pxM, float *t2xz_pxM, float *t2yy_pxM, float *t2xx_pyM, float *t2xy_pyM,
		float *t2yy_pyM, float *t2yz_pyM, float *t2xx_pzM, float *t2xz_pzM, float *t2yz_pzM, float *t2zz_pzM,
		float *qt2xx_pxM, float *qt2xy_pxM, float *qt2xz_pxM, float *qt2yy_pxM, float *qt2xx_pyM, float *qt2xy_pyM, 
		float *qt2yy_pyM, float *qt2yz_pyM, float *qt2xx_pzM, float *qt2xz_pzM, float *qt2yz_pzM, float *qt2zz_pzM,
		void **v2xMp, void **v2yMp, void **v2zMp, int *myid)
{
	
    //printf("[CUDA] stress computation:\n");
	float *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;
	int blockSizeX = 8;
	int blockSizeY = 8;
	dim3 dimBlock(blockSizeX, blockSizeY);

	v1xM = (float *) *v1xMp;
	v1yM = (float *) *v1yMp;
	v1zM = (float *) *v1zMp;
	v2xM = (float *) *v2xMp;
	v2yM = (float *) *v2yMp;
	v2zM = (float *) *v2zMp;

  gettimeofday(&t1, NULL);
	cpy_h2d_stressInputsC(v1xM, v1yM, v1zM, v2xM, v2yM, v2zM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);

	cpy_h2d_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, 
			t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
  gettimeofday(&t2, NULL);
  tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
  totalTimeH2DS += tmpTime;


  gettimeofday(&t1, NULL);
    int gridSizeX1 = (nd1_tyy[3] - nd1_tyy[2])/blockSizeX + 1;
    int gridSizeY1 = (nd1_tyy[9] - nd1_tyy[8])/blockSizeY + 1;
	dim3 dimGrid1(gridSizeX1, gridSizeY1);

	//int size = (*nztop) * (*nxtop + 3) * (*nytop);
	//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
    //write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM0.txt");
	
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

    int gridSizeX2 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeX + 1;
    int gridSizeY2 = (nd1_tyz[9] - nd1_tyz[8])/blockSizeY + 1;
	dim3 dimGrid2(gridSizeX2, gridSizeY2);
	
	// for debug
	//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
    //write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM1.txt");

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
	// for debug
	//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
    //write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM2.txt");

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
		int gridSizeX4 = (nd1_tyy[5] - nd1_tyy[0])/blockSizeX + 1;
		int gridSizeY4 = (lbx[1] - lbx[0])/blockSizeY + 1;
		dim3 dimGrid4(gridSizeX4, gridSizeY4);
		
		//debug
		/*float *t1xx_px=(float*)malloc(sizeof(float) * (*nztop) * ((lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1]) * (*nytop));
		cudaMemcpy(t1xx_px, t1xx_pxD, sizeof(float) * (*nztop) * ((lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1]) * (*nytop), cudaMemcpyDeviceToHost);
		write_output(t1xx_px, (*nztop) * ((lbx[1] - lbx[0] + 1) * (*mw1_pml) + lbx[1]) * (*nytop), "OUTPUT_ARRAYS/t1xx_px_cuda.txt");*/
		
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM3.txt");
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX5 = (nd1_tyy[11] - nd1_tyy[6])/blockSizeX + 1;
		int gridSizeY5 = (lby[1] - lby[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM4.txt");
	}

	if (lbx[1] >= lbx[0]) 
	{
		int gridSizeX6 = (nd1_txy[5] - nd1_txy[0])/blockSizeX + 1;
		int gridSizeY6 = (lbx[1] - lbx[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM5.txt");
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX7 = (nd1_txy[11] - nd1_txy[6])/blockSizeX + 1;
		int gridSizeY7 = (lby[1] - lby[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM6.txt");
	}

	if (lbx[1] >= lbx[0])
	{
		int gridSizeX8 = (nd1_txz[5] - nd1_txz[0])/blockSizeX + 1;
		int gridSizeY8 = (lbx[1] - lbx[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM7.txt");
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX9 = (nd1_txz[9] - nd1_txz[8])/blockSizeX + 1;
		int gridSizeY9 = (lby[1] - lby[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM8.txt");
	}

	if (lbx[1] >= lbx[0])
	{
		int gridSizeX10 = (nd1_tyz[3] - nd1_tyz[2])/blockSizeX + 1;
		int gridSizeY10 = (lbx[1] - lbx[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM9.txt");
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX11 = (nd1_tyz[11] - nd1_tyz[6])/blockSizeX + 1;
		int gridSizeY11 = (lby[1] - lby[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM10.txt");
	}

	int gridSizeX12 = (nd2_tyy[3] - nd2_tyy[2])/blockSizeX + 1;
	int gridSizeY12 = (nd2_tyy[9] - nd2_tyy[8])/blockSizeY + 1;
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
	// for debug
	//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM11.txt");

	int gridSizeX13 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeX + 1;
	int gridSizeY13 = (nd2_tyz[9] - nd2_tyz[8])/blockSizeY + 1;
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
	// for debug
	//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM12.txt");

	if (lbx[1] >= lbx[0])
	{
		int gridSizeX14 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeX + 1;
		int gridSizeY14 = (lbx[1] - lbx[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM13.txt");
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX15 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeX + 1;
		int gridSizeY15 = (lby[1] - lby[0])/blockSizeY + 1;
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
		// for debug
		//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
		//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM14.txt");
	}

	int gridSizeX16 = (nd2_tyy[5] - nd2_tyy[0])/blockSizeX + 1;
	int gridSizeY16 = (nd2_tyy[11] - nd2_tyy[6])/blockSizeY + 1;
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
	// for debug
	//cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
	//write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM15.txt");

	if (lbx[1] >= lbx[0])
	{
		int gridSizeX17 = (nd2_txy[5] - nd2_txy[0])/blockSizeX + 1;
		int gridSizeY17 = (lbx[1] - lbx[0])/blockSizeY + 1;
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
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX18 = (nd2_txy[11] - nd2_txy[6])/blockSizeX + 1;
		int gridSizeY18 = (lby[1] - lby[0])/blockSizeY + 1;
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
	}

	int gridSizeX19 = (nd2_txy[3] - nd2_txy[2])/blockSizeX + 1;
	int gridSizeY19 = (nd2_txy[9] - nd2_txy[8])/blockSizeY + 1;
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

	if (lbx[1] >= lbx[0])
	{
		int gridSizeX20 = (nd2_txz[5] - nd2_txz[0])/blockSizeX + 1;
		int gridSizeY20 = (lbx[1] - lbx[0])/blockSizeY + 1;
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
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX21 = (nd2_txz[9] - nd2_txz[8])/blockSizeX + 1;
		int gridSizeY21 = (lby[1] - lby[0])/blockSizeY + 1;
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
	}

	int gridSizeX22 = (nd2_txz[5] - nd2_txz[0])/blockSizeX + 1;
	int gridSizeY22 = (nd2_txz[11] - nd2_txz[6])/blockSizeY + 1;
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

	if (lbx[1] >= lbx[0])
	{
		int gridSizeX23 = (nd2_tyz[3] - nd2_tyz[2])/blockSizeX + 1;
		int gridSizeY23 = (lbx[1] - lbx[0])/blockSizeY + 1;
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
	}

	if (lby[1] >= lby[0])
	{
		int gridSizeX24 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeX + 1;
		int gridSizeY24 = (lby[1] - lby[0])/blockSizeY + 1;
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
	}

	int gridSizeX25 = (nd2_tyz[5] - nd2_tyz[0])/blockSizeX + 1;
	int gridSizeY25 = (nd2_tyz[11] - nd2_tyz[6])/blockSizeY + 1;
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
  cudaThreadSynchronize();

  gettimeofday(&t2, NULL);
  tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
  totalTimeCompS += tmpTime;

  gettimeofday(&t1, NULL);
	cpy_d2h_stressOutputsC(t1xxM, t1xyM, t1xzM, t1yyM, t1yzM, t1zzM, t2xxM, t2xyM, t2xzM, t2yyM, 
			t2yzM, t2zzM, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm);
  gettimeofday(&t2, NULL);
  tmpTime = 1000.0 * (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000.0;
  totalTimeD2HS += tmpTime;
    
	// for debug
	// int size = (*nztop) * (*nxtop + 3) * (*nytop);
    // write_output(t1xxM, size, "OUTPUT_ARRAYS/t1xxM.txt");
    
	// size = (*nztop) * (*nxtop + 3) * (*nytop + 3);
    // write_output(t1xyM, size, "OUTPUT_ARRAYS/t1xyM.txt");
    // size = (*nztop + 1) * (*nxtop + 3) * (*nytop);
    // write_output(t1xzM, size, "OUTPUT_ARRAYS/t1xzM.txt");
    // size = (*nztop) * (*nxtop) * (*nytop + 3);
    // write_output(t1yyM, size, "OUTPUT_ARRAYS/t1yyM.txt");
    // size = (*nztop + 1) * (*nxtop) * (*nytop + 3);
    // write_output(t1yzM, size, "OUTPUT_ARRAYS/t1yzM.txt");
    // size = (*nztop) * (*nxtop) * (*nytop);
    // write_output(t1zzM, size, "OUTPUT_ARRAYS/t1zzM.txt");
    
	
	// size = (*nzbtm) * (*nxbtm + 3) * (*nybtm);
    // write_output(t2xxM, size, "OUTPUT_ARRAYS/t2xxM.txt");
    // size = (*nzbtm) * (*nxbtm + 3) * (*nybtm + 3);
    // write_output(t2xyM, size, "OUTPUT_ARRAYS/t2xyM.txt");
    // size = (*nzbtm + 1) * (*nxbtm + 3) * (*nybtm);
    // write_output(t2xzM, size, "OUTPUT_ARRAYS/t2xzM.txt");
    // size = (*nzbtm) * (*nxbtm) * (*nybtm + 3);
    // write_output(t2yyM, size, "OUTPUT_ARRAYS/t2yyM.txt");
    // size = (*nzbtm + 1) * (*nxbtm) * (*nybtm + 3);
    // write_output(t2yzM, size, "OUTPUT_ARRAYS/t2yzM.txt");
    // size = (*nzbtm + 1) * (*nxbtm) * (*nybtm);
    // write_output(t2zzM, size, "OUTPUT_ARRAYS/t2zzM.txt");

	/*************** correctness *******************/
  /*
  FILE *fp;
	// cudaRes = cudaMalloc((void **)&v1xD,  sizeof(float) * y(*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1x");
	// cudaRes = cudaMalloc((void **)&v1yD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1y");
	// cudaRes = cudaMalloc((void **)&v1zD,  sizeof(float) * (*nztop + 2) * (*nxtop + 3) * (*nytop + 3));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, v1z");

  const char* filename = "v1x.txt";
  const char* filename1 = "v1y.txt";
  const char* filename2 = "v1z.txt";
  int i;

  if((fp = fopen(filename, "w+")) == NULL)
    fprintf(stderr, "File write error!\n");
  
  for(i = 0; i< (*nztop + 2) * (*nxtop + 3) * (*nytop + 3); i++ )
  {
    fprintf(fp, "%f ", v1xM[i]);
  }
   fprintf(fp, "\n");
  fclose(fp);

  if((fp = fopen(filename1, "w+")) == NULL)
    fprintf(stderr, "File write error!\n");
  
  for(i = 0; i< (*nztop + 2) * (*nxtop + 3) * (*nytop + 3); i++ )
  {
    fprintf(fp, "%f ", v1yM[i]);
  }
   fprintf(fp, "\n");
  fclose(fp);

  if((fp = fopen(filename2, "w+")) == NULL)
    fprintf(stderr, "File write error!\n");
  
  for(i = 0; i< (*nztop + 2) * (*nxtop + 3) * (*nytop + 3); i++ )
  {
    fprintf(fp, "%f ", v1zM[i]);
  }
   fprintf(fp, "\n");
  fclose(fp);

	// cudaRes = cudaMalloc((void **)&t1xxD, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xx");
	// cudaRes = cudaMalloc((void **)&t1xyD, sizeof(float) * (*nztop) * (*nxtop + 3) * (*nytop + 3));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xy");
	// cudaRes = cudaMalloc((void **)&t1xzD, sizeof(float) * (*nztop + 1) * (*nxtop + 3) * (*nytop));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1xz");
	// cudaRes = cudaMalloc((void **)&t1yyD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop + 3));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1yy");
	// cudaRes = cudaMalloc((void **)&t1yzD, sizeof(float) * (*nztop + 1) * (*nxtop) * (*nytop + 3));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1yz");
	// cudaRes = cudaMalloc((void **)&t1zzD, sizeof(float) * (*nztop) * (*nxtop) * (*nytop));
	// CHECK_ERROR(cudaRes, "Allocate Device Memory1, t1zz");
  const char* filename3 = "x_t1xx.txt";
  const char* filename4 = "x_t1xy.txt";
  const char* filename5 = "x_t1xz.txt";
  if((fp = fopen(filename3, "w+")) == NULL)
    fprintf(stderr, "File write error!\n");
  
  for(i = 0; i< (*nztop) * (*nxtop + 3) * (*nytop); i++ )
  {
    fprintf(fp, "%f ", t1xxM[i]);
  }
   fprintf(fp, "\n");
  fclose(fp);
  if((fp = fopen(filename4, "w+")) == NULL)
    fprintf(stderr, "File write error!\n");
  
  for(i = 0; i< (*nztop) * (*nxtop + 3) * (*nytop+3); i++ )
  {
    fprintf(fp, "%f ", t1xyM[i]);
  }
   fprintf(fp, "\n");
  fclose(fp);
  if((fp = fopen(filename5, "w+")) == NULL)
    fprintf(stderr, "File write error!\n");
  
  for(i = 0; i< (*nztop+1) * (*nxtop + 3) * (*nytop); i++ )
  {
    fprintf(fp, "%f ", t1xzM[i]);
  }
   fprintf(fp, "\n");
  fclose(fp);
*/
	return;
}

#ifdef __cplusplus
}
#endif

