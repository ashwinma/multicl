#include <stdio.h>
#include "mpi.h"
//!XSC--------------------------------------------------------------------
#define drvh1(i, j) drvh1M[(i) - 1 + (j) * mw1_pml1]
#define drti1(i, j) drti1M[(i) - 1 + (j) * mw1_pml1]
#define drth1(i, j) drth1M[(i) - 1 + (j) * mw1_pml1]

#define damp1_x(i, j, k) damp1_xM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) - lbx[0]) * nytop)]
#define damp1_y(i, j, k) damp1_yM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) - lby[0]) * nxtop)]

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

#define damp2_x(i, j, k) damp2_xM[(i) - 1 + nzbtm * ((j) - 1 + ((k) - lbx[0]) * nybtm)]
#define damp2_y(i, j, k) damp2_yM[(i) - 1 + nzbtm * ((j) - 1 + ((k) - lby[0]) * nxbtm)]
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

//debug-------------------
int procID;
//---------------------------

void MPI_Init_C (MPI_Fint *ierr) 
{
    int argcc = 0;
    char **argvc = {""};
    *ierr = (MPI_Fint) MPI_Init(&argcc, NULL);
    return;
}


void velocity_inner_IC(int	nztop,
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

void velocity_inner_IIC(float ca,
					   int	 *nd2_vel,
					   float *rhoM,
					   float *dxi2M,
					   float *dyi2M,
					   float *dzi2M,
					   float *dxh2M,
					   float *dyh2,
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
					   float *v2zM);

void vel_PmlX_IC(float ca,
				int   *lbx,
				int	  *nd1_vel,
				float *rhoM,
				float *drvh1M,
				float *drti1M,
				float *damp1_xM,
				int	  *idmatM1,
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
				float *v1z_pxM);

void vel_PmlY_IC(int	  nztop,
				float ca,
				int	  *lby,
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
				float *v1z_pyM);

void vel_PmlX_IIC(int   nzbm1,
				 float ca,
				 int   *lbx,
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
				 float *v2z_pxM);

void vel_PmlY_IIC(int   nzbm1,
				 float ca,
				 int   *lby,
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
				 float *v2z_pyM);

void vel_PmlZ_IIC(int   nzbm1,
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
				 float *v2z_pzM);

//for stress computation---------------------------------
void stress_norm_xy_IC(int nxb1,
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
					   float *v1zM);

void  stress_xz_yz_IC(int nxb1,
					  int nyb1,
					  int nx1p1,
					  int ny1p1,
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
					  float *qt1yzM);

void stress_norm_PmlY_IC(int nxb1,
						 int nyb1,
						 int mw1_pml1,
						 int nxtop,
						 int nztop,
						 int *lby,
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
						 float *v1zM);

void stress_norm_PmlX_IC(int nxb1,
						 int nyb1,
						 int nxtop,
						 int nytop,
						 int nztop,
						 int mw1_pml,
						 int mw1_pml1,
						 int *lbx,
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
						 float *qt1yy_pxM);

void stress_xy_PmlX_IC(int nxb1,
					   int nyb1,
					   int mw1_pml,
					   int mw1_pml1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int *lbx,
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
					   float *v1yM);

void stress_xy_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int *lby,
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
					   float *v1yM);

void stress_xz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int mw1_pml,
					   int mw1_pml1,
					   int *lbx,
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
					   float *v1zM);

void stress_xz_PmlY_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   int *lby,
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
					   float *v1zM);

void stress_yz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nztop,
					   int nxtop,
					   int *lbx,
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
					   float *v1zM);

void stress_yz_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int *lby,
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
					   float *v1zM);

void stress_norm_xy_II(int nxb2,
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
					   float *v2zM);

void stress_xz_yz_IIC(int nxb2,
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
					  float *v2zM);

void stress_norm_PmlX_IIC(int nxb2,
						  int nyb2,
						  int mw2_pml,
						  int mw2_pml1,
						  int nztop,
						  int nxbtm,
						  int nybtm,
						  int nzbtm,
						  int *lbx,
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
						  float *v2zM);

void stress_norm_PmlY_II(int nxb2,
						 int nyb2,
						 int nztop,
						 int nxbtm,
						 int nzbtm,
						 int mw2_pml1,
						 int *lby,
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
						 float *v2zM);

void stress_norm_PmlZ_IIC(int nxb2,
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
						  float *v2zM);

void stress_xy_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int *lbx,
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
						float *v2yM);

void stress_xy_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nztop,
						int nxbtm,
						int nzbtm,
						int *lby,
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
						float *v2yM);

void stress_xy_PmlZ_II(int nxb2,
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
					   float *v2yM);

void stress_xz_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int *lbx,
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
						float *v2zM);

void stress_xz_PmlY_IIC(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int *lby,
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
					   float *qt2xzM);

void stress_xz_PmlZ_IIC(int nxb2,
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
						float *v2zM);

void stress_yz_PmlX_IIC(int nxb2,
						int nyb2,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *lbx,
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
						float *v2zM);

void stress_yz_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *lby,
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
						float *v2zM);

void stress_yz_PmlZ_IIC(int nxb2,
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
						float *v2zM);

//void compute_velocityCDebug( int *nztop,  int *nztm1,  float *ca, int *lbx,
void compute_velocityCDebug( int *nztop,  int *nztm1,  float *ca, int *lbx,
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
			  int *nxbtm,  int *nybtm,  int *nzbtm)

{
  float *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;

  // extract specific input/output pointers
  v1xM=(float *) *v1xMp;
  v1yM=(float *) *v1yMp;
  v1zM=(float *) *v1zMp;
  v2xM=(float *) *v2xMp;
  v2yM=(float *) *v2yMp;
  v2zM=(float *) *v2zMp;

	velocity_inner_IC(*nztop,
					 *nztm1,
					 *ca,
					 nd1_vel,
					 rhoM,
					 idmat1M,
					 dxi1M,
					 dyi1M,
					 dzi1M,
					 dxh1M,
					 dyh1M,
					 dzh1M,
					 t1xxM,
					 t1xyM,
					 t1xzM,
					 t1yyM,
					 t1yzM,
					 t1zzM,
					 *nxtop,	//dimension #
					 *nytop,
					 v1xM,		//output
					 v1yM,
					 v1zM);

	vel_PmlX_IC(*ca,
			   lbx,
			   nd1_vel,
			   rhoM,
			   drvh1M,
			   drti1M,
			   damp1_xM,
			   idmat1M,
			   dxi1M,
			   dyi1M,
			   dzi1M,
			   dxh1M,
			   dyh1M,
			   dzh1M,
			   t1xxM,
			   t1xyM,
			   t1xzM,
			   t1yyM,
			   t1yzM,
			   t1zzM,
			   *mw1_pml1,	//dimension #
			   *mw1_pml,
			   *nxtop,
			   *nytop,
			   *nztop,
			   v1xM,		//output
			   v1yM,
			   v1zM,
			   v1x_pxM,
			   v1y_pxM,
			   v1z_pxM);

	vel_PmlY_IC(*nztop,
			   *ca,
			   lby,
			   nd1_vel,
			   rhoM,
			   drvh1M,
			   drti1M,
			   idmat1M,
			   damp1_yM,
			   dxi1M,
			   dyi1M,
			   dzi1M,
			   dxh1M,
			   dyh1M,
			   dzh1M,
			   t1xxM,
			   t1xyM,
			   t1xzM,
			   t1yyM,
			   t1yzM,
			   t1zzM,
			   *mw1_pml1,	//dimension #s
			   *mw1_pml,
			   *nxtop,
			   *nytop,
			   v1xM,		//output
			   v1yM,
			   v1zM,
			   v1x_pyM,
			   v1y_pyM,
			   v1z_pyM);

	velocity_inner_IIC(*ca,
					  nd2_vel,
					  rhoM,
					  dxi2M,
					  dyi2M,
					  dzi2M,
					  dxh2M,
					  dyh2M,
					  dzh2M,
					  idmat2M,
					  t2xxM,
					  t2xyM,
					  t2xzM,
					  t2yyM,
					  t2yzM,
					  t2zzM,
					  *nxbtm,
					  *nybtm,
					  *nzbtm,
					  v2xM,		//output
					  v2yM,
					  v2zM);
  
	vel_PmlX_IIC(*nzbm1,
				*ca,
				lbx,
				nd2_vel,
				drvh2M,
				drti2M,
				rhoM,
				damp2_xM,
				idmat2M,
				dxi2M,
				dyi2M,
				dzi2M,
				dxh2M,
				dyh2M,
				dzh2M,
				t2xxM,
				t2xyM,
				t2xzM,
				t2yyM,
				t2yzM,
				t2zzM,
				*mw2_pml1,	//dimension #s
				*mw2_pml,
				*nxbtm,
				*nybtm,
				*nzbtm,
				v2xM,	//output
				v2yM,
				v2zM,
				v2x_pxM,
				v2y_pxM,
				v2z_pxM);

	vel_PmlY_IIC(*nzbm1,
				*ca,
				lby,
				nd2_vel,
				drvh2M,
				drti2M,
				rhoM,
				damp2_yM,
				idmat2M,
				dxi2M,
				dyi2M,
				dzi2M,
				dxh2M,
				dyh2M,
				dzh2M,
				t2xxM,
				t2xyM,
				t2xzM,
				t2yyM,
				t2yzM,
				t2zzM,
				*mw2_pml1,	//dimension #s
				*mw2_pml,
				*nxbtm,
				*nybtm,
				*nzbtm,
				v2xM,		//output
				v2yM,
				v2zM,
				v2x_pyM,
				v2y_pyM,
				v2z_pyM);
	
	vel_PmlZ_IIC(*nzbm1,
				*ca,
				nd2_vel,
				drvh2M,
				drti2M,
				rhoM,
				damp2_zM,
				idmat2M,
				dxi2M,
				dyi2M,
				dzi2M,
				dxh2M,
				dyh2M,
				dzh2M,
				t2xxM,
				t2xyM,
				t2xzM,
				t2yyM,
				t2yzM,
				t2zzM,
				*mw2_pml1,	//dimension #s
				*mw2_pml,
				*nxbtm,
				*nybtm,
				*nzbtm,
				v2xM,		//output
				v2yM,
				v2zM,
				v2x_pzM,
				v2y_pzM,
				v2z_pzM);

	return;
}

void velocity_inner_IC(int	nztop,
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

	//for (j = nd1_vel(3); j <= nd1_vel(4); j++)
	for (j = nd1_vel[2]; j <= nd1_vel[3]; j++)
	{
		//for (i = nd1_vel(9); i <= nd1_vel(10); i++)
		for (i = nd1_vel[8]; i <= nd1_vel[9]; i++)
		{
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
		}
	}

 	return;
}


//-----------------------------------------------------------------------
void velocity_inner_IIC(float ca,
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
	//Compute the velocity at Region II
	//use grid_node_comm
	//use wave_field_comm
	//implicit NONE
	//integer:: i,j,k
	int i, j, k;
	
	//for (j = nd2_vel(3); j <= nd2_vel(4); j++)
	for (j = nd2_vel[2]; j <= nd2_vel[3]; j++)
	{
		for (i = nd2_vel[8]; i <= nd2_vel[9]; i++)
		{
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
		}
	}
	
	return;
}


//-----------------------------------------------------------------------
void vel_PmlX_IC(float ca,
				int   *lbx,
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

	//int nv2x=(lbx(2) - lbx(1) + 1) * mw1_pml;
	int nv2x=(lbx[1] - lbx[0] + 1) * mw1_pml;
	
	//if ( lbx(1)>lbx(2) ) return;
	if ( lbx[0]>lbx[1] ) return;

	//for (j = nd1_vel(1); j <= nd1_vel(6); j++)
	for (j = nd1_vel[0]; j <= nd1_vel[5]; j++)
	{
		ib=0;
		//for (lb = lbx(1); lb <= lbx(2); lb++)
		for (lb = lbx[0]; lb <= lbx[1]; lb++)
		{	
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
		}
	}
	
	return;
}


//-----------------------------------------------------------------------
void vel_PmlY_IC(int	  nztop,
				float ca,
				int	  *lby,
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
	//!Compute the velocities in region of PML-y-I
	//use grid_node_comm
	//use wave_field_comm
	//implicit NONE
	int i,j,k,lb,jb,kb;
	float rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
        dtxz,dtyz,dtzz,vtmpx,vtmpy,vtmpz;

	//if( lby(1)>lby(2) ) 
	if( lby[0]>lby[1] ) 
 		return;

	jb=0;
	//for (lb = lby(1); lb <= lby(2); lb++)
	for (lb = lby[0]; lb <= lby[1]; lb++)
	{
		kb=0;
		//for (j = nd1_vel(1+4*lb); j <= nd1_vel(2+4*lb); j++)
		for (j = nd1_vel[4*lb]; j <= nd1_vel[1+4*lb]; j++)
		{
			kb=kb+1;
			jb=jb+1;
			rth=drvh1(kb,lb);
			rti=drti1(kb,lb);

			//for (i = nd1_vel(7); i <= nd1_vel(12); i++)
			for (i = nd1_vel[6]; i <= nd1_vel[11]; i++)
			{
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
		}
	}

 	return;
}


//-----------------------------------------------------------------------
void vel_PmlX_IIC(int   nzbm1,
				 float ca,
				 int   *lbx,
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
	int nv2y = (lbx[1] - lbx[0] + 1) * mw2_pml;

	//if ( lbx(1)>lbx(2) ) return;
	if ( lbx[0]>lbx[1] ) return;

	//for (j = nd2_vel(1); j <= nd2_vel(6); j++)
	for (j = nd2_vel[0]; j <= nd2_vel[5]; j++)
	{
		ib=0;
		//for (lb = lbx(1); lb <= lbx(2); lb++)
		for (lb = lbx[0]; lb <= lbx[1]; lb++)
		{
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
		}
	}

	return;
}


//-----------------------------------------------------------------------
void vel_PmlY_IIC(int   nzbm1,
				 float ca,
				 int   *lby,
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
	int i,j,k,lb,jb,kb;
	float rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
		   vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz;

	//if( lby(1)>lby(2) ) return;
	if( lby[0]>lby[1] ) return;
	jb=0;

	//for (lb = lby(1); lb <= lby(2); lb++)
	for (lb = lby[0]; lb <= lby[1]; lb++)
	{
		kb=0;

		//for (j = nd2_vel(1+4*lb); j <= nd2_vel(2+4*lb); j++)
		for (j = nd2_vel[4*lb]; j <= nd2_vel[1+4*lb]; j++)
		{
			kb=kb+1;
			jb=jb+1;
			rth=drvh2(kb,lb);
			rti=drti2(kb,lb);

			//for (i = nd2_vel(7); i <= nd2_vel(12); i++)
			for (i = nd2_vel[6]; i <= nd2_vel[11]; i++)
			{
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
		}
	}
 
	return;
}

//-----------------------------------------------------------------------
void vel_PmlZ_IIC(int   nzbm1,
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

	//for (j = nd2_vel(1); j <= nd2_vel(6); j++)
	for (j = nd2_vel[0]; j <= nd2_vel[5]; j++)
	{
		//for (i = nd2_vel(7); i <= nd2_vel(12); i++)
		for (i = nd2_vel[6]; i <= nd2_vel[11]; i++)
		{
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
		}
	}

	return;
}

//stress computation----------------------------------------------
void compute_stressCDebug(int *nxb1, int *nyb1, int *nx1p1, int *ny1p1, int *nxtop, int *nytop, int *nztop, int *mw1_pml,
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
		void **v2xMp, void **v2yMp, void **v2zMp, int *myid)
{
	
	float *v1xM, *v1yM, *v1zM, *v2xM, *v2yM, *v2zM;
	v1xM = (float *) *v1xMp;
	v1yM = (float *) *v1yMp;
	v1zM = (float *) *v1zMp;
	v2xM = (float *) *v2xMp;
	v2yM = (float *) *v2yMp;
	v2zM = (float *) *v2zMp;

	procID = *myid;

	stress_norm_xy_IC(*nxb1,
					  *nyb1,
					  *nxtop,
					  *nztop,
					  nd1_tyy,
					  idmat1M,
					  *ca,
					  clamdaM,
					  cmuM,
					  epdtM,
					  qwpM,
					  qwsM,
					  qwt1M,
					  qwt2M,
					  dxh1M,
					  dyh1M,
					  dxi1M,
					  dyi1M,
					  dzi1M,
					  t1xxM,
					  t1xyM,
					  t1yyM,
					  t1zzM,
					  qt1xxM,
					  qt1xyM,
					  qt1yyM,
					  qt1zzM,
					  v1xM,
					  v1yM,
					  v1zM);

	  stress_xz_yz_IC(*nxb1,
					  *nyb1,
					  *nx1p1,
					  *ny1p1,
					  *nxtop,
					  *nytop,
					  *nztop,
					  nd1_tyz,
					  idmat1M,
					  *ca,
					  cmuM,
					  epdtM,
					  qwsM,
					  qwt1M,
					  qwt2M,
					  dxi1M,
					  dyi1M,
					  dzh1M,
					  v1xM,
					  v1yM,
					  v1zM,
					  t1xzM,
					  t1yzM,
					  qt1xzM,
					  qt1yzM);

	 stress_norm_PmlX_IC(*nxb1,
						 *nyb1,
						 *nxtop,
						 *nytop,
						 *nztop,
						 *mw1_pml,
						 *mw1_pml1,
						 lbx,
						 nd1_tyy,
						 idmat1M,
						 *ca,
						 drti1M,
						 damp1_xM,
						 clamdaM,
						 cmuM,
						 epdtM,
						 qwpM,
						 qwsM,
						 qwt1M,
						 qwt2M,
						 dzi1M,
						 dxh1M,
						 dyh1M,
						 v1xM,
						 v1yM,
						 v1zM,
						 t1xxM,
						 t1yyM,
						 t1zzM,
						 t1xx_pxM,
						 t1yy_pxM,
						 qt1xxM,
						 qt1yyM,
						 qt1zzM,
						 qt1xx_pxM,
						 qt1yy_pxM);

	 stress_norm_PmlY_IC(*nxb1,
						 *nyb1,
						 *mw1_pml1,
						 *nxtop,
						 *nztop,
						 lby,
						 nd1_tyy,
						 idmat1M,
						 *ca,
						 drti1M,
						 damp1_yM,
						 clamdaM,
						 cmuM,
						 epdtM,
						 qwpM,
						 qwsM,
						 qwt1M,
						 qwt2M,
						 dxh1M,
						 dyh1M,
						 dzi1M,
						 t1xxM,
						 t1yyM,
						 t1zzM,
						 qt1xxM,
						 qt1yyM,
						 qt1zzM,
						 t1xx_pyM,
						 t1yy_pyM,
						 qt1xx_pyM,
						 qt1yy_pyM,
						 v1xM,
						 v1yM,
						 v1zM);

	 stress_xy_PmlX_IC(*nxb1,
					   *nyb1,
					   *mw1_pml,
					   *mw1_pml1,
					   *nxtop,
					   *nytop,
					   *nztop,
					   lbx,
					   nd1_txy,
					   idmat1M,
					   *ca,
					   drth1M,
					   damp1_xM,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dxi1M,
					   dyi1M,
					   t1xyM,
					   qt1xyM,
					   t1xy_pxM,
					   qt1xy_pxM,
					   v1xM,
					   v1yM);

	 stress_xy_PmlY_IC(*nxb1,
					   *nyb1,
					   *mw1_pml1,
					   *nxtop,
					   *nztop,
					   lby,
					   nd1_txy,
					   idmat1M,
					   *ca,
					   drth1M,
					   damp1_yM,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dxi1M,
					   dyi1M,
					   t1xyM,
					   qt1xyM,
					   t1xy_pyM,
					   qt1xy_pyM,
					   v1xM,
					   v1yM);

	 stress_xz_PmlX_IC(*nxb1,
					   *nyb1,
					   *nxtop,
					   *nytop,
					   *nztop,
					   *mw1_pml,
					   *mw1_pml1,
					   lbx,
					   nd1_txz,
					   idmat1M,
					   *ca,
					   drth1M,
					   damp1_xM,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dxi1M,
					   dzh1M,
					   t1xzM,
					   qt1xzM,
					   t1xz_pxM,
					   qt1xz_pxM,
					   v1xM,
					   v1zM);

	 stress_xz_PmlY_IC(*nxb1,
					   *nyb1,
					   *nxtop,
					   *nztop,
					   lby,
					   nd1_txz,
					   idmat1M,
					   *ca,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dxi1M,
					   dzh1M,
					   t1xzM,
					   qt1xzM,
					   v1xM,
					   v1zM);

	 stress_yz_PmlX_IC(*nxb1,
					   *nyb1,
					   *nztop,
					   *nxtop,
					   lbx,
					   nd1_tyz,
					   idmat1M,
					   *ca,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dyi1M,
					   dzh1M,
					   t1yzM,
					   qt1yzM,
					   v1yM,
					   v1zM);

	 stress_yz_PmlY_IC(*nxb1,
					   *nyb1,
					   *mw1_pml1,
					   *nxtop,
					   *nztop,
					   lby,
					   nd1_tyz,
					   idmat1M,
					   *ca,
					   drth1M,
					   damp1_yM,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dyi1M,
					   dzh1M,
					   t1yzM,
					   qt1yzM,
					   t1yz_pyM,
					   qt1yz_pyM,
					   v1yM,
					   v1zM);


	 stress_norm_xy_II(*nxb2,
					   *nyb2,
					   *nxbtm,
					   *nzbtm,
					   *nztop,
					   nd2_tyy,
					   idmat2M,
					   clamdaM,
					   cmuM,
					   epdtM,
					   qwpM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   t2xxM,
					   t2xyM,
					   t2yyM,
					   t2zzM,
					   qt2xxM,
					   qt2xyM,
					   qt2yyM,
					   qt2zzM,
					   dxh2M,
					   dyh2M,
					   dxi2M,
					   dyi2M,
					   dzi2M,
					   v2xM,
					   v2yM,
					   v2zM);

	 stress_xz_yz_IIC(*nxb2,
					  *nyb2,
					  *nztop,
					  *nxbtm,
					  *nzbtm,
					  nd2_tyz,
					  idmat2M,
					  cmuM,
					  epdtM,
					  qwsM,
					  qwt1M,
					  qwt2M,
					  dxi2M,
					  dyi2M,
					  dzh2M,
					  t2xzM,
					  t2yzM,
					  qt2xzM,
					  qt2yzM,
					  v2xM,
					  v2yM,
					  v2zM);

	 stress_norm_PmlX_IIC(*nxb2,
						  *nyb2,
						  *mw2_pml,
						  *mw2_pml1,
						  *nztop,
						  *nxbtm,
						  *nybtm,
						  *nzbtm,
						  lbx,
						  nd2_tyy,
						  idmat2M,
						  *ca,
						  drti2M,
						  damp2_xM,
						  clamdaM,
						  cmuM,
						  epdtM,
						  qwpM,
						  qwsM,
						  qwt1M,
						  qwt2M,
						  dxh2M,
						  dyh2M,
						  dzi2M,
						  t2xxM,
						  t2yyM,
						  t2zzM,
						  qt2xxM,
						  qt2yyM,
						  qt2zzM,
						  t2xx_pxM,
						  t2yy_pxM,
						  qt2xx_pxM,
						  qt2yy_pxM,
						  v2xM,
						  v2yM,
						  v2zM);

	 stress_norm_PmlY_II(*nxb2,
						 *nyb2,
						 *nztop,
						 *nxbtm,
						 *nzbtm,
						 *mw2_pml1,
						 lby,
						 nd2_tyy,
						 idmat2M,
						 *ca,
						 drti2M,
						 damp2_yM,
						 clamdaM,
						 cmuM,
						 epdtM,
						 qwpM,
						 qwsM,
						 qwt1M,
						 qwt2M,
						 dxh2M,
						 dyh2M,
						 dzi2M,
						 t2xxM,
						 t2yyM,
						 t2zzM,
						 qt2xxM,
						 qt2yyM,
						 qt2zzM,
						 t2xx_pyM,
						 t2yy_pyM,
						 qt2xx_pyM,
						 qt2yy_pyM,
						 v2xM,
						 v2yM,
						 v2zM);

	 stress_norm_PmlZ_IIC(*nxb2,
						  *nyb2,
						  *mw2_pml,
						  *mw2_pml1,
						  *nztop,
						  *nxbtm,
						  *nzbtm,
						  nd2_tyy,
						  idmat2M,
						  *ca,
						  damp2_zM,
						  drth2M,
						  clamdaM,
						  cmuM,
						  epdtM,
						  qwpM,
						  qwsM,
						  qwt1M,
						  qwt2M,
						  dxh2M,
						  dyh2M,
						  dzi2M,
						  t2xxM,
						  t2yyM,
						  t2zzM,
						  qt2xxM,
						  qt2yyM,
						  qt2zzM,
						  t2xx_pzM,
						  t2zz_pzM,
						  qt2xx_pzM,
						  qt2zz_pzM,
						  v2xM,
						  v2yM,
						  v2zM);

	 stress_xy_PmlX_IIC(*nxb2,
						*nyb2,
						*mw2_pml,
						*mw2_pml1,
						*nxbtm,
						*nybtm,
						*nzbtm,
						*nztop,
						lbx,
						nd2_txy,
						idmat2M,
						*ca,
						drth2M,
						damp2_xM,
						cmuM,
						epdtM,
						qwsM,
						qwt1M,
						qwt2M,
						dxi2M,
						dyi2M,
						t2xyM,
						qt2xyM,
						t2xy_pxM,
						qt2xy_pxM,
						v2xM,
						v2yM);

	 stress_xy_PmlY_IIC(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nztop,
						*nxbtm,
						*nzbtm,
						lby,
						nd2_txy,
						idmat2M,
						*ca,
						drth2M,
						damp2_yM,
						cmuM,
						epdtM,
						qwsM,
						qwt1M,
						qwt2M,
						dxi2M,
						dyi2M,
						t2xyM,
						qt2xyM,
						t2xy_pyM,
						qt2xy_pyM,
						v2xM,
						v2yM);

	 stress_xy_PmlZ_II(*nxb2,
					   *nyb2,
					   *nxbtm,
					   *nzbtm,
					   *nztop,
					   nd2_txy,
					   idmat2M,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dxi2M,
					   dyi2M,
					   t2xyM,
					   qt2xyM,
					   v2xM,
					   v2yM);

	 stress_xz_PmlX_IIC(*nxb2,
						*nyb2,
						*mw2_pml,
						*mw2_pml1,
						*nxbtm,
						*nybtm,
						*nzbtm,
						*nztop,
						lbx,
						nd2_txz,
						idmat2M,
						*ca,
						drth2M,
						damp2_xM,
						cmuM,
						epdtM,
						qwsM,
						qwt1M,
						qwt2M,
						dxi2M,
						dzh2M,
						t2xzM,
						qt2xzM,
						t2xz_pxM,
						qt2xz_pxM,
						v2xM,
						v2zM);

	stress_xz_PmlY_IIC(*nxb2,
					   *nyb2,
					   *nxbtm,
					   *nzbtm,
					   *nztop,
					   lby,
					   nd2_txz,
					   idmat2M,
					   cmuM,
					   epdtM,
					   qwsM,
					   qwt1M,
					   qwt2M,
					   dxi2M,
					   dzh2M,
					   v2xM,
					   v2zM,
					   t2xzM,
					   qt2xzM);

	 stress_xz_PmlZ_IIC(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nxbtm,
						*nzbtm,
						*nztop,
						nd2_txz,
						idmat2M,
						*ca,
						drti2M,
						damp2_zM,
						cmuM,
						epdtM,
						qwsM,
						qwt1M,
						qwt2M,
						dxi2M,
						dzh2M,
						t2xzM,
						qt2xzM,
						t2xz_pzM,
						qt2xz_pzM,
						v2xM,
						v2zM);

	 stress_yz_PmlX_IIC(*nxb2,
						*nyb2,
						*nxbtm,
						*nzbtm,
						*nztop,
						lbx,
						nd2_tyz,
						idmat2M,
						cmuM,
						epdtM,
						qwsM,
						qwt1M,
						qwt2M,
						dyi2M,
						dzh2M,
						t2yzM,
						qt2yzM,
						v2yM,
						v2zM);

	 stress_yz_PmlY_IIC(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nxbtm,
						*nzbtm,
						*nztop,
						lby,
						nd2_tyz,
						idmat2M,
						*ca,
						drth2M,
						damp2_yM,
						cmuM,
						epdtM,
						qwsM,
						qwt1M,
						qwt2M,
						dyi2M,
						dzh2M,
						t2yzM,
						qt2yzM,
						t2yz_pyM,
						qt2yz_pyM,
						v2yM,
						v2zM);

	 stress_yz_PmlZ_IIC(*nxb2,
						*nyb2,
						*mw2_pml1,
						*nxbtm,
						*nzbtm,
						*nztop,
						nd2_tyz,
						idmat2M,
						*ca,
						drti2M,
						damp2_zM,
						cmuM,
						epdtM,
						qwsM,
						qwt1M,
						qwt2M,
						dyi2M,
						dzh2M,
						t2yzM,
						qt2yzM,
						t2yz_pzM,
						qt2yz_pzM,
						v2yM,
						v2zM);
}

void stress_norm_xy_IC(int nxb1,
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
//! Compute stress-Norm and XY component in Region I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,jkq,kodd,inod,irw
// real:: sxx,syy,szz,sxy,qxx,qyy,qzz,qxy,cusxy,sss, &
//        cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,jkq,kodd,inod,irw;
	float sxx,syy,szz,sxy,qxx,qyy,qzz,qxy,cusxy,sss,cl,sm2,pm,et,et1,wtp,wts;

	for (j = nd1_tyy[2]; j <= nd1_tyy[3]; j++)
	//do j=nd1_tyy(3),nd1_tyy(4)
	{
		//kodd=2*mod(j+nyb1,2)+1
		kodd = 2 * ((j + nyb1) & 1) + 1;
		for (i = nd1_tyy[8]; i <= nd1_tyy[9]; i++)
		//do i=nd1_tyy(9),nd1_tyy(10)
		{
			//jkq=mod(i+nxb1,2)+kodd
			jkq=((i+nxb1) & 1) + kodd;
			for (k = nd1_tyy[12]; k <= nd1_tyy[17]; k++)
     		//do k=nd1_tyy(13),nd1_tyy(18)
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
				//irw=jkq+4*mod(k,2)
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
		}
	}
	return;
}

//-----------------------------------------------------------------------------
void  stress_xz_yz_IC(int nxb1,
					  int nyb1,
					  int nx1p1,
					  int ny1p1,
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
	float tfr1 = -577./528./ca;
	float tfr2 = 201./176./ca;
	float tfr3 = -9./176./ca;
	float tfr4=1./528./ca;

	int i,j,k,kodd,inod,jkq,irw;
	float dvzx,dvzy,dvxz,dvyz,sm,cusxz,cusyz,et,et1,dmws,qxz,qyz;

	for (j=nd1_tyz[2]; j <=nd1_tyz[3]; j++)
	//do j=nd1_tyz(3),nd1_tyz(4)
	{
		//kodd=2*mod(j+nyb1,2)+1
		kodd=2*((j+nyb1)&1)+1;
		for (i=nd1_tyz[8]; i<=nd1_tyz[9]; i++)
		//do i=nd1_tyz(9),nd1_tyz(10)
		{
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
		}
	}

	for (j=-1; j<=ny1p1; j++)
	//do j=-1,ny1p1
	{
		for (i = 1; i <= nxtop; i++)
		//do i=1,nxtop
		{
			t1yz(1,i,j)=0.0;
		}
	}

	for (j=1; j <= nytop; j++)
	//do j=1,nytop
	{
		for (i=-1; i <=nx1p1; i++)
		//do i=-1,nx1p1
		{
			t1xz(1,i,j)=0.0;
		}
	}
	return;
}

//------------------------------------------------------------------------------------

void stress_norm_PmlX_IC(int nxb1,
						 int nyb1,
						 int nxtop,
						 int nytop,
						 int nztop,
						 int mw1_pml,
						 int mw1_pml1,
						 int *lbx,
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
	if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return;

	nti = (lbx[1] - lbx[0] + 1) * mw1_pml + lbx[0];
	
 	for (j=nd1_tyy[0]; j <= nd1_tyy[5]; j++)
 	//do j=nd1_tyy(1),nd1_tyy(6)
	{
		kodd=2*((j+nyb1)&1)+1;
		ib=0;
		for (lb=lbx[0]; lb <=lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
	return;
}

void stress_norm_PmlY_IC(int nxb1,
						 int nyb1,
						 int mw1_pml1,
						 int nxtop,
						 int nztop,
						 int *lby,
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

	if(lby[0]>lby[1]) return;
	//if(lby(1)>lby(2) ) return
	jb=0;

	for (lb=lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1),lby(2)
	{
		kb=0;
		for (j = nd1_tyy[4*lb]; j <= nd1_tyy[1+4*lb]; j++)
		//do j=nd1_tyy(1+4*lb),nd1_tyy(2+4*lb)
		{
			kb=kb+1;
			jb=jb+1;
			rti=drti1(kb,lb);
			kodd=2 * ((j + nyb1) & 1) + 1;
			//kodd=2*mod(j+nyb1,2)+1
			for (i = nd1_tyy[6]; i <= nd1_tyy[11]; i++)
			//do i=nd1_tyy(7),nd1_tyy(12)
			{
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
		}
	}
	return;
}

void stress_xy_PmlX_IC(int nxb1,
					   int nyb1,
					   int mw1_pml,
					   int mw1_pml1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int *lbx,
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

	nth = (lbx[1] - lbx[0] + 1) * mw1_pml + 1 - lbx[0];

	if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return
	for (j = nd1_txy[0]; j <= nd1_txy[5]; j++)
	//do j=nd1_txy(1),nd1_txy(6)
	{
		kodd = 2 * ((j + nyb1) & 1) + 1;
		//kodd=2*mod(j+nyb1,2)+1
		ib=0;
		for (lb = lbx[0]; lb <= lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
	return;
}

void stress_xy_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int *lby,
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

	if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	jb=0;
	for (lb = lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1), lby(2)
	{
		kb=0;
		for (j = nd1_txy[4*lb]; j <= nd1_txy[1 + 4 * lb]; j++)
		//do j=nd1_txy(1+4*lb),nd1_txy(2+4*lb)
		{
			kb=kb+1;
			jb=jb+1;
			rth=drth1(kb,lb);
			kodd=2 * ((j + nyb1) & 1) + 1;
			//kodd=2*mod(j+nyb1,2)+1;
			for (i = nd1_txy[6]; i <= nd1_txy[11]; i++)
			//do i=nd1_txy(7),nd1_txy(12)
			{
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
		}
	}
	return;
}

void stress_xz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int mw1_pml,
					   int mw1_pml1,
					   int *lbx,
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

	if (lbx[0] > lbx[1]) return;

	nth = (lbx[1] - lbx[0] + 1) * mw1_pml + 1 - lbx[0];
	//if ( lbx(1)>lbx(2) ) return
	for (j = nd1_txz[0]; j <= nd1_txz[5]; j++)
	//do j=nd1_txz(1),nd1_txz(6)
	{
		kodd=2 * ((j+nyb1)&1)+1;
		//kodd=2*mod(j+nyb1,2)+1
		ib=0;
		for (lb = lbx[0]; lb <= lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
 	return;
}

void stress_xz_PmlY_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   int *lby,
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

	if (lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	for (lb=lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1),lby(2)
	{
		for (j = nd1_txz[4*lb]; j <= nd1_txz[1+4*lb]; j++)
		//do j=nd1_txz(1+4*lb),nd1_txz(2+4*lb)
		{
			kodd=2 * ((j+nyb1)&1)+1;
			//kodd=2*mod(j+nyb1,2)+1
			for (i = nd1_txz[8]; i <= nd1_txz[9]; i++)
			//do i=nd1_txz(9),nd1_txz(10)
			{
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
		}
	}
	return;
}

void stress_yz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nztop,
					   int nxtop,
					   int *lbx,
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

	if(lbx[0] > lbx[1]) return;
	//if( lbx(1)>lbx(2) ) return

	for (j = nd1_tyz[2]; j <= nd1_tyz[3]; j++)
	//do j=nd1_tyz(3),nd1_tyz(4)
	{
		kodd=2 * ((j+nyb1)&1)+1;
		//kodd=2*mod(j+nyb1,2)+1
		for (lb = lbx[0]; lb <= lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
	return;
}

void stress_yz_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int *lby,
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

	if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	jb=0;
	for (lb=lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1),lby(2)
	{
		kb=0;
		for (j = nd1_tyz[4*lb]; j <= nd1_tyz[1+4*lb]; j++)
		//do j=nd1_tyz(1+4*lb),nd1_tyz(2+4*lb)
		{
			kb=kb+1;
			jb=jb+1;
			rth=drth1(kb,lb);
			kodd=2*((j+nyb1)&1)+1;
			//kodd=2*mod(j+nyb1,2)+1;
			for (i = nd1_tyz[6]; i <= nd1_tyz[11]; i++)
			//do i=nd1_tyz(7),nd1_tyz(12)
			{
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
		}
	}

	return;
}

void stress_norm_xy_II(int nxb2,
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
	float sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy,
		cl,sm2,et,et1,dmws,pm,wtp,wts;

	for (j=nd2_tyy[2]; j <= nd2_tyy[3]; j++)
	//do j=nd2_tyy(3),nd2_tyy(4)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		for (i = nd2_tyy[8]; i <= nd2_tyy[9]; i++)
		//do i=nd2_tyy(9),nd2_tyy(10)
		{
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
		}
	}
	return;
}

//call stress_xz_yz_II
void stress_xz_yz_IIC(int nxb2,
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

	for (j = nd2_tyz[2]; j <= nd2_tyz[3]; j++)
	//do j=nd2_tyz(3),nd2_tyz(4)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		for (i = nd2_tyz[8]; i <= nd2_tyz[9]; i++)
		//do i=nd2_tyz(9),nd2_tyz(10)
		{
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
		}
	}
	return;
}

//call stress_norm_PmlX_II
void stress_norm_PmlX_IIC(int nxb2,
						  int nyb2,
						  int mw2_pml,
						  int mw2_pml1,
						  int nztop,
						  int nxbtm,
						  int nybtm,
						  int nzbtm,
						  int *lbx,
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

	if(lbx[0] > lbx[1]) return;
	//if( lbx(1)>lbx(2) ) return
	nti = (lbx[1] - lbx[0] + 1) * mw2_pml + lbx[1];
	for (j=nd2_tyy[0]; j <= nd2_tyy[5]; j++)
	//do j=nd2_tyy(1),nd2_tyy(6)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		ib=0;
		for (lb=lbx[0]; lb <= lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
	return;
}

//call stress_norm_PmlY_II
void stress_norm_PmlY_II(int nxb2,
						 int nyb2,
						 int nztop,
						 int nxbtm,
						 int nzbtm,
						 int mw2_pml1,
						 int *lby,
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

	if( lby[0] > lby[1] ) return;
	//if( lby(1)>lby(2) ) return;
	jb=0;
	for (lb = lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1),lby(2)
	{
		kb=0;
		for (j=nd2_tyy[4*lb]; j <= nd2_tyy[1+4*lb]; j++)
		//do j=nd2_tyy(1+4*lb),nd2_tyy(2+4*lb)
		{
			kb=kb+1;
			jb=jb+1;
			rti=drti2(kb,lb);
			kodd=2*((j+nyb2)&1)+1;
			//kodd=2*mod(j+nyb2,2)+1;
			for (i = nd2_tyy[6]; i <= nd2_tyy[11]; i++)
			//do i=nd2_tyy(7),nd2_tyy(12)
			{
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
		}
	}
	return;
}

void stress_norm_PmlZ_IIC(int nxb2,
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

	for (j = nd2_tyy[0]; j <= nd2_tyy[5]; j++)
	//do j=nd2_tyy(1),nd2_tyy(6)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		for (i=nd2_tyy[6]; i <= nd2_tyy[11]; i++)
		//do i=nd2_tyy(7),nd2_tyy(12)
		{
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
		}
	}
	return;
}

void stress_xy_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int *lbx,
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

	if(lbx[0] > lbx[1]) return;
	nth = (lbx[1] - lbx[0] + 1) * mw2_pml + 1 - lbx[0];
	//nth = (lbx(2) - lbx(1) + 1) * mw2_pml + 1 - lbx(1)
	for (j = nd2_txy[0]; j <= nd2_txy[5]; j++)
	//do j=nd2_txy(1),nd2_txy(6)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		ib=0;
		for (lb = lbx[0]; lb <= lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
	return;
}

void stress_xy_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nztop,
						int nxbtm,
						int nzbtm,
						int *lby,
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

	if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	jb=0;
	for (lb=lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1),lby(2)
	{
		kb=0;
		for (j=nd2_txy[4*lb]; j <= nd2_txy[1+4*lb]; j++)
		//do j=nd2_txy(1+4*lb),nd2_txy(2+4*lb)
		{
			kb=kb+1;
			jb=jb+1;
			rth=drth2(kb,lb);
			kodd=2*((j+nyb2)&1)+1;
			//kodd=2*mod(j+nyb2,2)+1
			for (i = nd2_txy[6]; i <= nd2_txy[11]; i++)
			//do i=nd2_txy(7),nd2_txy(12)
			{
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
		}
	}
	return;
}

void stress_xy_PmlZ_II(int nxb2,
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

	for (j = nd2_txy[2]; j <= nd2_txy[3]; j++)
	//do j=nd2_txy(3),nd2_txy(4)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		for (i = nd2_txy[8]; i <= nd2_txy[9]; i++)
		//do i=nd2_txy(9),nd2_txy(10)
		{
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
		}
	}
	return;
}

//call stress_xz_PmlX_II
void stress_xz_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int *lbx,
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

	if(lbx[0] > lbx[1]) return;

	nth = (lbx[1] - lbx[0] + 1) * mw2_pml + 1 - lbx[0];
	for (j = nd2_txz[0]; j <= nd2_txz[5]; j++)
	//do j=nd2_txz(1),nd2_txz(6)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		ib=0;
		for (lb=lbx[0]; lb <= lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
	return;
}

void stress_xz_PmlY_IIC(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int *lby,
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
	if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	for (lb = lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1),lby(2)
	{
		for (j=nd2_txz[4*lb]; j <= nd2_txz[1+4*lb]; j++)
		//do j=nd2_txz(1+4*lb),nd2_txz(2+4*lb)
		{
			kodd=2*((j+nyb2)&1)+1;
			//kodd=2*mod(j+nyb2,2)+1
			for (i = nd2_txz[8]; i <= nd2_txz[9]; i++)
			//do i=nd2_txz(9),nd2_txz(10)
			{
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
		}
	}
	return;
}

void stress_xz_PmlZ_IIC(int nxb2,
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
	
	for (j = nd2_txz[0]; j <= nd2_txz[5]; j++)
	//do j=nd2_txz(1),nd2_txz(6)
	{
		kodd = 2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		for (i = nd2_txz[6]; i <= nd2_txz[11]; i++)
		//do i=nd2_txz(7),nd2_txz(12)
		{
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
		}
	}
	return;
}

//call stress_yz_PmlX_II
void stress_yz_PmlX_IIC(int nxb2,
						int nyb2,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *lbx,
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

	if(lbx[0] > lbx[1]) return;
	//if( lbx(1)>lbx(2) ) return
	for (j=nd2_tyz[2]; j <= nd2_tyz[3]; j++)
	//do j=nd2_tyz(3),nd2_tyz(4)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		for (lb = lbx[0]; lb <= lbx[1]; lb++)
		//do lb=lbx(1),lbx(2)
		{
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
		}
	}
	return;
}

//call stress_yz_PmlY_II
void stress_yz_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int *lby,
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

	if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
	jb=0;
	for (lb = lby[0]; lb <= lby[1]; lb++)
	//do lb=lby(1),lby(2)
	{
		kb=0;
		for (j = nd2_tyz[4*lb]; j <= nd2_tyz[1+4*lb]; j++)
		//do j=nd2_tyz(1+4*lb),nd2_tyz(2+4*lb)
		{
			kb=kb+1;
			jb=jb+1;
			rth=drth2(kb,lb);
			kodd=2*((j+nyb2)&1)+1;
			//kodd=2*mod(j+nyb2,2)+1
			for (i = nd2_tyz[6]; i <= nd2_tyz[11]; i++)
			//do i=nd2_tyz(7),nd2_tyz(12)
			{
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
		}
	}
	return;
}

//call stress_yz_PmlZ_II
void stress_yz_PmlZ_IIC(int nxb2,
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

	for (j = nd2_tyz[0]; j <= nd2_tyz[5]; j++)
	//do j=nd2_tyz(1),nd2_tyz(6)
	{
		kodd=2*((j+nyb2)&1)+1;
		//kodd=2*mod(j+nyb2,2)+1
		for (i = nd2_tyz[6]; i <= nd2_tyz[11]; i++)
		//do i=nd2_tyz(7),nd2_tyz(12)
		{
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
		}
	}
	return;
}

