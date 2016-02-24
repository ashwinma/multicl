//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
//
//!XSC--------------------------------------------------------------------
/*
 * Switch bw optimised and originial(unoptimised) kernels
 */
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

//MPI-ACC
#define sdx51(i, j, k) sdx51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define sdx52(i, j, k) sdx52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]
#define sdx41(i, j, k) sdx41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define sdx42(i, j, k) sdx42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]

#define sdy51(i, j, k) sdy51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define sdy52(i, j, k) sdy52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]
#define sdy41(i, j, k) sdy41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define sdy42(i, j, k) sdy42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]

#define rcx51(i, j, k) rcx51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define rcx52(i, j, k) rcx52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]
#define rcx41(i, j, k) rcx41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define rcx42(i, j, k) rcx42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]

#define rcy51(i, j, k) rcy51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define rcy52(i, j, k) rcy52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]
#define rcy41(i, j, k) rcy41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define rcy42(i, j, k) rcy42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]

#define cix(i,j) cixM[i-1 + j*8 ]
#define ciy(i,j) ciyM[i-1 + j*8 ]
#define chx(i,j) chxM[i-1 + j*8 ]
#define chy(i,j) chyM[i-1 + j*8 ]

__kernel void velocity_inner_IC(int	nztop,
					  int 	nztm1,
					  float ca,
					  __global int	*nd1_vel,
					  __global float *rhoM,
					  __global int   *idmat1M,
					  __global float *dxi1M,
					  __global float *dyi1M,
					  __global float *dzi1M,
					  __global float *dxh1M,
					  __global float *dyh1M,
					  __global float *dzh1M,
					  __global float *t1xxM,
					  __global float *t1xyM,
					  __global float *t1xzM,
					  __global float *t1yyM,
					  __global float *t1yzM,
					  __global float *t1zzM,
					  int	nxtop,		//dimension #
					  int	nytop,
					  __global float *v1xM,		//output
					  __global float *v1yM,
					  __global float *v1zM)
{
	int i, j, k, k3;
	float dtxz, dtyz, dtzz;
	int offset_j = nd1_vel[2];
	int offset_k = 1;
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) ; // skipping the offset_k to take care of k=nztop case
	int k_bak = k; // backing up k
	int offset_i = nd1_vel[8];
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

        if (k > nztm1 || i> nd1_vel[9]){
		return;
	}

	for (j = nd1_vel[2]; j <= nd1_vel[3]; j++) 
	{

		// no need to  apply much optimization in following k==1 ir k==2 case
		if( k==1 || k==2 || k==0) 
		{

                        if(k ==0) k = nztop;
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
			//k ==0 to take care of k=nztop case
			else 
			{					
				dtxz=dzi1(2,k)/ca*(t1xz(k,i,j)-t1xz(k+1,i,j));
				dtyz=dzi1(2,k)/ca*(t1yz(k,i,j)-t1yz(k+1,i,j));
				dtzz=dzh1(2,k)/ca*(t1zz(k-1,i,j)-t1zz(k,i,j));
			}

/*                //if ((k + (nztop + 2) * ((i)  + (j) * (nxtop + 3))) == 235145)
                if ( k==1 && j==offset_j && i==38)
                {
                            printf("Test :\n");

                            printf("k: %d, i: %d, j:%d \n, \
                                    v1z(k,i,j) = %f, \n \
                                    rho(idmat1(k,i,j)) = %f, \n \ 
                                    rho(idmat1(k-1,i,j)) = %f, \n \
                                    dxh1(1,i) = %f, \n \
                                    t1xz(k,i-2,j) = %f, \n \
                                    dxh1(2,i) = %f, \n \
                                    t1xz(k,i-1,j) = %f, \n \
                                    dxh1(3,i) = %f, \n \
                                    t1xz(k,i,  j) = %f, \n \
                                    dxh1(4,i) = %f, \n \
                                    t1xz(k,i+1,j) = %f, \n \
                                    dyh1(1,j) = %f, \n \
                                    t1yz(k,i,j-2) = %f,  \n \
                                    dyh1(2,j) =%f, \n \
                                    t1yz(k,i,j-1) = %f,  \n \
                                    dyh1(3,j) = %f, \n \
                                    t1yz(k,i,j  ) = %f, \n \
                                    dyh1(4,j) = %f, \n \
                                    t1yz(k,i,j+1) = %f, \n \
                                    dtzz = %f, \n",
                                    k,i,j,
                                    v1z(k,i,j),
                                    rho(idmat1(k,i,j)),
                                    rho(idmat1(k-1,i,j)),
                                    dxh1(1,i),
                                    t1xz(k,i-2,j),
                                    dxh1(2,i),
                                    t1xz(k,i-1,j),
                                    dxh1(3,i),
                                    t1xz(k,i,  j),
                                    dxh1(4,i),
                                    t1xz(k,i+1,j),
                                    dyh1(1,j),
                                    t1yz(k,i,j-2),
                                    dyh1(2,j),
                                    t1yz(k,i,j-1),
                                    dyh1(3,j),
                                    t1yz(k,i,j  ),
                                    dyh1(4,j),
                                    t1yz(k,i,j+1),                                    
                                    dtzz);

                }
*/ 

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

//                       if (((k) + (nztop + 2) * ((i)  + (j) * (nxtop + 3))) == 235145)
//                            printf("v1z(k,i,j): %f\n", v1z(k,i,j));

                        k = k_bak;

		} 
		else 
		{
                    k = k_bak;

/*
#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1

#define dxi1(m, n) dxi1_S[(n-offset_i) % BLOCK_SIZE][m-1] 
#define dyi1(m, n) dyi1_S[(n-offset_j) % 72][m-1] 
#define dzi1(m, n) dzi1_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define dxh1(m, n) dxh1_S[(n-offset_i) % BLOCK_SIZE][m-1]  
#define dyh1(m, n) dyh1_S[(n-offset_j) % 72][m-1] 
#define dzh1(m, n) dzh1_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define BLOCK_SIZE 8
*/

 
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



#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1
#define dxi1(i, j) dxi1M[((j) - 1) * 4 + (i) - 1]
#define dyi1(i, j) dyi1M[((j) - 1) * 4 + (i) - 1]
#define dzi1(i, j) dzi1M[((j) - 1) * 4 + (i) - 1]
#define dxh1(i, j) dxh1M[((j) - 1) * 4 + (i) - 1]
#define dyh1(i, j) dyh1M[((j) - 1) * 4 + (i) - 1]
#define dzh1(i, j) dzh1M[((j) - 1) * 4 + (i) - 1]

                }
	}
 	return;
}

//-----------------------------------------------------------------------
__kernel void velocity_inner_IIC(float ca,
					   __global int	 *nd2_vel,
					   __global float *rhoM,
					   __global float *dxi2M,
					   __global float *dyi2M,
					   __global float *dzi2M,
					   __global float *dxh2M,
					   __global float *dyh2M,
					   __global float *dzh2M,
					   __global int 	 *idmat2M,
					   __global float *t2xxM,
					   __global float *t2xyM,
					   __global float *t2xzM,
					   __global float *t2yyM,
					   __global float *t2yzM,
					   __global float *t2zzM,
					   int   nxbtm,	//dimension #s
					   int   nybtm,
					   int   nzbtm,
					   __global float *v2xM,		//output
					   __global float *v2yM,
					   __global float *v2zM)
{
	int i, j, k, k_bak;

	//j = get_group_id(0) * get_local_size(0) + get_local_id(0) + nd2_vel[2];
        int offset_k = 2, offset_i = nd2_vel[8];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
        k_bak = k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (k > nd2_vel[15] || i > nd2_vel[9])
	{
		return;
	}
	
	//for (j = nd2_vel(3); j <= nd2_vel(4); j++)
	//for (j = nd2_vel[2]; j <= nd2_vel[3]; j++)
	//{
		//for (i = nd2_vel[8]; i <= nd2_vel[9]; i++)
		//{
	//k=1;
/*        
#define BLOCK_SIZE 8 
        __shared__ float dxi2_S[4][BLOCK_SIZE], dxh2_S[4][BLOCK_SIZE];
        __shared__ float dyi2_S[4][15],dyh2_S[4][15];
        __shared__ float dzi2_S[4][BLOCK_SIZE], dzh2_S[4][BLOCK_SIZE];

        int offset_j = nd2_vel[2];
        if (get_local_id(0) == 0) {
            //printf("BLOCK:(%d,%d): dxi2_S[0][%d ] = dxi2(%d,%d)\n",get_group_id(0), get_group_id(1), 0,(i-offset_i)%8,  1, i );
            dxi2_S[0][(i-offset_i)%BLOCK_SIZE] = dxi2(1,i);
            dxh2_S[0][(i-offset_i)%BLOCK_SIZE] = dxh2(1,i);
            dxi2_S[1][(i-offset_i)%BLOCK_SIZE] = dxi2(2,i);
            dxh2_S[1][(i-offset_i)%BLOCK_SIZE] = dxh2(2,i);
            dxi2_S[2][(i-offset_i)%BLOCK_SIZE] = dxi2(3,i);
            dxh2_S[2][(i-offset_i)%BLOCK_SIZE] = dxh2(3,i);
            dxi2_S[3][(i-offset_i)%BLOCK_SIZE] = dxi2(4,i);
            dxh2_S[3][(i-offset_i)%BLOCK_SIZE] = dxh2(4,i);
        }
        if(get_local_id(1) == 0) {
            for (int l = 0; l<4; l++){
                    //printf("BLOCK:(%d,%d): dzi2_S[%d][(%d-%d)\% 8 ] = dzi2(%d+1,%d)\n",get_group_id(0), get_group_id(1), l,k,offset_k,  l, k );
                    //printf("BLOCK:(%d,%d): dzi2_S[%d][(%d ] = dzi2(%d,%d)\n",get_group_id(0), get_group_id(1), l,(k-offset_k)%8,  l+1, k );
                    dzi2_S[l][(k -offset_k)%BLOCK_SIZE] = dzi2(l+1, k);
                    dzh2_S[l][(k -offset_k)%BLOCK_SIZE] = dzh2(l+1, k);
            }

            if (get_local_id(0) == 0) {
                for (j = nd2_vel[2] ; j <= nd2_vel[3]; j++) 
                {
                   for (int l=0; l<4 ; l++) { 
                    //printf("Block:(%d,%d) dyi2_S[%d][(%d-%d)%15 ] = dyi2(%d+1,%d)\n",get_group_id(0), get_group_id(1),l,j,offset_j, l, j );
                    //if (! dyi2_S[l][(j-offset_j)%BLOCK_SIZE])
                        dyi2_S[l][(j-offset_j)%15] = dyi2(l+1,j);
                    //if (! dyh2_S[l][(j-offset_j)%BLOCK_SIZE])
                        dyh2_S[l][(j-offset_j)%15] = dyh2(l+1,j);
                   }
                }
            }
        }
        __syncthreads();
*/
      
	for (j = nd2_vel[2] ; j <= nd2_vel[3]; j++){ 
            if( k == nd2_vel[15]) {
            //if( k == offset_k) {
                k = 1;
                v2x(k,i,j)= v2x(k,i,j)+
                        0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i+1,j)))*
                        (dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,j)+
                        dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j)+
                        dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+
                        dyh2(3,j)*t2xy(k,i,j)+dyh2(4,j)*t2xy(k,i,j+1)+
                        dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j)+
                        dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j));

                v2y(k,i,j)= v2y(k,i,j)+ 
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
                k = k_bak;
            }        
/*
#undef dxi2
#undef dyi2
#undef dzi2
#undef dxh2
#undef dyh2
#undef dzh2
#define dxi2(m, n) dxi2_S[m-1][(n-offset_i)%8] 
#define dyi2(m, n) dyi2_S[m-1][(n-offset_j)%15] 
#define dzi2(m, n) dzi2_S[m-1][(n-offset_k)%8] 
#define dxh2(m, n) dxh2_S[m-1][(n-offset_i)%8] 
#define dyh2(m, n) dyh2_S[m-1][(n-offset_j)%15] 
#define dzh2(m, n) dzh2_S[m-1][(n-offset_k)%8] 
  */        
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
#undef dxi2
#undef dyi2
#undef dzi2
#undef dxh2
#undef dyh2
#undef dzh2
#define dxi2(i, j) dxi2M[(i) - 1 + 4 * ((j) - 1)]
#define dyi2(i, j) dyi2M[(i) - 1 + 4 * ((j) - 1)]
#define dzi2(i, j) dzi2M[(i) - 1 + 4 * ((j) - 1)]
#define dxh2(i, j) dxh2M[(i) - 1 + 4 * ((j) - 1)]
#define dyh2(i, j) dyh2M[(i) - 1 + 4 * ((j) - 1)]
#define dzh2(i, j) dzh2M[(i) - 1 + 4 * ((j) - 1)]
            }
#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1
#define dxi1(i, j) dxi1M[((j) - 1) * 4 + (i) - 1]
#define dyi1(i, j) dyi1M[((j) - 1) * 4 + (i) - 1]
#define dzi1(i, j) dzi1M[((j) - 1) * 4 + (i) - 1]
#define dxh1(i, j) dxh1M[((j) - 1) * 4 + (i) - 1]
#define dyh1(i, j) dyh1M[((j) - 1) * 4 + (i) - 1]
#define dzh1(i, j) dzh1M[((j) - 1) * 4 + (i) - 1]

           
		//}
	//}
	
	return;
}



//-----------------------------------------------------------------------
__kernel void vel_PmlX_IC(float ca,
				int   lbx0,
				int   lbx1,
				__global int	  *nd1_vel,
				__global float *rhoM,
				__global float *drvh1M,
				__global float *drti1M,
				__global float *damp1_xM,
				__global int	  *idmat1M,
				__global float *dxi1M,
				__global float *dyi1M,
				__global float *dzi1M,
				__global float *dxh1M,
				__global float *dyh1M,
				__global float *dzh1M,
				__global float *t1xxM,
				__global float *t1xyM,
				__global float *t1xzM,
				__global float *t1yyM,
				__global float *t1yzM,
				__global float *t1zzM,
				int   mw1_pml1,	//dimension #
			    int   mw1_pml,
			    int   nxtop,
			    int   nytop,
			    int   nztop,
				__global float *v1xM,		//output
				__global float *v1yM,
				__global float *v1zM,
				__global float *v1x_pxM,
				__global float *v1y_pxM,
				__global float *v1z_pxM)
{
// !Compute the velocities in region of PML-x-I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
	int i,j,k,lb,ib,kb;
	float rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
        vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz,dtxy,dtyy,dtzy;

        int offset_i = nd1_vel[6+4*lb];
	int offset_k = 1;
	k  = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	int offset_j = nd1_vel[0];
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;


	//int nv2x=(lbx(2) - lbx(1) + 1) * mw1_pml;
	int nv2x=(lbx1 - lbx0 + 1) * mw1_pml;
	
	//if ( lbx(1)>lbx(2) ) return;
	if (lbx0 > lbx1) 
	{
		return;
	}

	if (j > nd1_vel[5] || k > nztop)
	{
		return;
	}

// KERNEL LAUNCH ERROR WITH SHARED MEMORY
/*
#define BLOCK_SIZE 8
	__shared__ float dxi1_S[BLOCK_SIZE][4], dxh1_S[BLOCK_SIZE][4];
	__shared__ float dyi1_S[110][4], dyh1_S[110][4];
	__shared__ float dzi1_S[BLOCK_SIZE][4], dzh1_S[BLOCK_SIZE][4];

        if(get_local_id(0) == 0) {
            for (int l = 0; l<4; l++){
               dxi1_S[(i-offset_i)%BLOCK_SIZE][l] = dxi1(l+1,i);
               dxh1_S[(i-offset_i)%BLOCK_SIZE][l] = dxh1(l+1,i);
            }
        }
        if (get_local_id(1) == 0) {
            for(int l =0; l<4; l++) {
                dzi1_S[(k -offset_k)%BLOCK_SIZE][l] = dzi1(l+1, k);
                dzh1_S[(k -offset_k)%BLOCK_SIZE][l] = dzh1(l+1, k);               
            }

            if (get_local_id(0) == 0 ) {
                for ( int j = nd1_vel[2]; j <= nd1_vel[3]; j++) {
                    for (int l =0; l<4; l++) {
                        dyi1_S[(j-offset_j)%72][l] = dyi1(l+1,j);
                        dyh1_S[(j-offset_j)%72][l] = dyh1(l+1,j);
                    }
               }
            }

        }
        __syncthreads();
*/
	//calculate the value of ib
	ib = 0;
	//for (i = nd1_vel(7+4*lb); i <= nd1_vel(8+4*lb); i++)
	for (lb = lbx0; lb <= lbx1; lb++) 
	{
		kb=0;
		for (i = nd1_vel[6+4*lb]; i <= nd1_vel[7+4*lb]; i++)
		{	
			kb=kb+1;
			ib=ib+1;
			rth=drvh1(kb,lb);
			rti=drti1(kb,lb);


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

/*
#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1

#define dxi1(m, n) dxi1_S[(n-offset_i) % BLOCK_SIZE][m-1] 
#define dyi1(m, n) dyi1_S[(n-offset_j) % 110][m-1] 
#define dzi1(m, n) dzi1_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define dxh1(m, n) dxh1_S[(n-offset_i) % BLOCK_SIZE][m-1]  
#define dyh1(m, n) dyh1_S[(n-offset_j) % 110][m-1] 
#define dzh1(m, n) dzh1_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define BLOCK_SIZE 8
*/

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
        
#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1
#define dxi1(i, j) dxi1M[((j) - 1) * 4 + (i) - 1]
#define dyi1(i, j) dyi1M[((j) - 1) * 4 + (i) - 1]
#define dzi1(i, j) dzi1M[((j) - 1) * 4 + (i) - 1]
#define dxh1(i, j) dxh1M[((j) - 1) * 4 + (i) - 1]
#define dyh1(i, j) dyh1M[((j) - 1) * 4 + (i) - 1]
#define dzh1(i, j) dzh1M[((j) - 1) * 4 + (i) - 1]

	return;
}

//-----------------------------------------------------------------------
__kernel void vel_PmlY_IC(int  nztop,
				float ca,
				int	  lby0,
				int   lby1,
				__global int   *nd1_vel,
				__global float *rhoM,
				__global float *drvh1M,
				__global float *drti1M,
				__global int   *idmat1M,
				__global float *damp1_yM,
				__global float *dxi1M,
				__global float *dyi1M,
				__global float *dzi1M,
				__global float *dxh1M,
				__global float *dyh1M,
				__global float *dzh1M,
				__global float *t1xxM,
				__global float *t1xyM,
				__global float *t1xzM,
				__global float *t1yyM,
				__global float *t1yzM,
				__global float *t1zzM,
				int   mw1_pml1, //dimension #s
				int   mw1_pml,
				int   nxtop,
				int   nytop,
				__global float *v1xM,		//output
				__global float *v1yM,
				__global float *v1zM,
				__global float *v1x_pyM,
				__global float *v1y_pyM,
				__global float *v1z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	float rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
        dtxz,dtyz,dtzz,vtmpx,vtmpy,vtmpz;

	//if( lby(1)>lby(2) ) 
	if( lby0>lby1 ) 
 		return;

	int offset_k = 1;
	k = get_local_size(0) * get_group_id(0) + get_local_id(0) + offset_k;
	int offset_i = nd1_vel[6];
	i  = get_local_size(1) * get_group_id(1) + get_local_id(1) + offset_i;

	if (k > nztop || i > nd1_vel[11])
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

	//for (lb = lby(1); lb <= lby(2); lb++)
	//for (lb = lby0; lb <= lby1; lb++)
	//{
	//	kb=0;

	//	//for (i = nd1_vel(7); i <= nd1_vel(12); i++)
	//	for (i = nd1_vel[6]; i <= nd1_vel[11]; i++)
	//	{
			//for (j = nd1_vel(1+4*lb); j <= nd1_vel(2+4*lb); j++)

/*
#define BLOCK_SIZE 8

	__shared__ float dxi1_S[BLOCK_SIZE][4], dxh1_S[BLOCK_SIZE][4];
	__shared__ float dyi1_S[BLOCK_SIZE][4], dyh1_S[BLOCK_SIZE][4];
	__shared__ float dzi1_S[BLOCK_SIZE][4], dzh1_S[BLOCK_SIZE][4];

	for (int l = 0; l<4; l++){
	    //if (! dxi2_S[l][(i-offset_i)%BLOCK_SIZE])
		dxi1_S[(i-offset_i)%BLOCK_SIZE][l] = dxi1(l+1,i);
	    //if (! dxh2_S[l][(i-offset_i)%BLOCK_SIZE])
		dxh1_S[(i-offset_i)%BLOCK_SIZE][l] = dxh1(l+1,i);
	    //if (! dzi2_S[l][(k -offset_k)%BLOCK_SIZE])
		dzi1_S[(k -offset_k)%BLOCK_SIZE][l] = dzi1(l+1, k);
	    //if (! dzh2_S[l][(k -offset_k)%BLOCK_SIZE])
		dzh1_S[(k -offset_k)%BLOCK_SIZE][l] = dzh1(l+1, k);
	}
*/
        jb =0;
	for(lb = lby0; lb <= lby1; lb++)
	{
		kb = 0;
		int offset_j = nd1_vel[4*lb];
		for (j = nd1_vel[4*lb]; j <= nd1_vel[1+4*lb]; j++)
		{
/*			for (int l=0; l<4 ; l++) { 
			    //if (! dyi2_S[l][(j-offset_j)%BLOCK_SIZE])
				dyi1_S[(j-offset_j)%BLOCK_SIZE][l] = dyi1(l+1,j);
			    //if ! dyh2_S[l][(j-offset_j)%BLOCK_SIZE])
				dyh1_S[(j-offset_j)%BLOCK_SIZE][l] = dyh1(l+1,j);
		   	}
*/
                        
			kb=kb+1;
			jb=jb+1;
			rth=drvh1(kb,lb);
			rti=drti1(kb,lb);

			damp0=damp1_y(k,i,lb);
			dmpy2=1./(1.+rth*damp0);
			dmpy1=dmpy2*2.-1.;
			dmpxz2=1./(1.+rti*damp0);
			dmpxz1=dmpxz2*2.-1.;
			ro1=rho(idmat1(k,i,j));
			rox=0.5*(ro1+rho(idmat1(k,i+1,j)));
			roy=0.5*(ro1+rho(idmat1(k,i,j+1)));
			roz=0.5*(ro1+rho(idmat1(k-1,i,j)));
/*
#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1

#define dxi1(m, n) dxi1_S[(n-offset_i) % BLOCK_SIZE][m-1] 
#define dyi1(m, n) dyi1_S[(n-offset_j) % BLOCK_SIZE][m-1] 
#define dzi1(m, n) dzi1_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define dxh1(m, n) dxh1_S[(n-offset_i) % BLOCK_SIZE][m-1]  
#define dyh1(m, n) dyh1_S[(n-offset_j) % BLOCK_SIZE][m-1] 
#define dzh1(m, n) dzh1_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define BLOCK_SIZE 8
*/
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
                        //if(k==1)// && i == 2 && j ==2) 
                        //if(((k) + (nztop + 2) * ((i) + (j) * (nxtop + 3)))== 12377) 
                        //{
                        //       printf("k :%d, i: %d, j: %d, vtpmz: %f, v1_py(k,i,jb): %f, jb: %d, \
                        //        v1z(k,i,j): %f\n", k,i,j,vtmpz, v1y_py(k,i,jb), jb,
                        //       v1z(k,i,j) ); 
                        //    }
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
/*        
#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1
#define dxi1(i, j) dxi1M[((j) - 1) * 4 + (i) - 1]
#define dyi1(i, j) dyi1M[((j) - 1) * 4 + (i) - 1]
#define dzi1(i, j) dzi1M[((j) - 1) * 4 + (i) - 1]
#define dxh1(i, j) dxh1M[((j) - 1) * 4 + (i) - 1]
#define dyh1(i, j) dyh1M[((j) - 1) * 4 + (i) - 1]
#define dzh1(i, j) dzh1M[((j) - 1) * 4 + (i) - 1]
*/
	//}
	//}

 	return;
}


//-----------------------------------------------------------------------

__kernel void vel_PmlX_IIC(int   nzbm1,
				 float ca,
				 int   lbx0,
				 int   lbx1,
				 __global int   *nd2_vel,
				 __global float *drvh2M,
				 __global float *drti2M,
				 __global float *rhoM,
				 __global float *damp2_xM,
				 __global int   *idmat2M,
				 __global float *dxi2M,
				 __global float *dyi2M,
				 __global float *dzi2M,
				 __global float *dxh2M,
				 __global float *dyh2M,
				 __global float *dzh2M,
				 __global float *t2xxM,
				 __global float *t2xyM,
				 __global float *t2xzM,
				 __global float *t2yyM,
				 __global float *t2yzM,
				 __global float *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 __global float *v2xM,	//output
				 __global float *v2yM,
				 __global float *v2zM,
				 __global float *v2x_pxM,
				 __global float *v2y_pxM,
				 __global float *v2z_pxM)
{
	int i,j,k,lb,ib,kb;
	float rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
		vtmpx,vtmpy,vtmpz,dtxy,dtyy,dtzy,dtxz,dtyz,dtzz;

	//int nv2y = (lbx(2) - lbx(1) + 1) * mw2_pml;
	int nv2y = (lbx1 - lbx0 + 1) * mw2_pml;

	//if ( lbx(1)>lbx(2) ) return;
	if ( lbx0>lbx1 ) return;


	int offset_k = 1;
	k  = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	int offset_j = nd2_vel[0];
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	if (j > nd2_vel[5] || k > nzbm1)
	{
		return;
	}
/*
#define BLOCK_SIZE 8

	__shared__ float dxi2_S[BLOCK_SIZE][4], dxh2_S[BLOCK_SIZE][4];
	__shared__ float dyi2_S[BLOCK_SIZE][4], dyh2_S[BLOCK_SIZE][4];
	__shared__ float dzi2_S[BLOCK_SIZE][4], dzh2_S[BLOCK_SIZE][4];

	for (int l = 0; l<4; l++){
	    //if (! dxi2_S[l][(i-offset_i)%BLOCK_SIZE])
		dyi2_S[(j-offset_j)%BLOCK_SIZE][l] = dyi2(l+1,j);
	    //if (! dxh2_S[l][(i-offset_i)%BLOCK_SIZE])
		dyh2_S[(j-offset_j)%BLOCK_SIZE][l] = dyh2(l+1,j);
	    //if (! dzi2_S[l][(k -offset_k)%BLOCK_SIZE])
		dzi2_S[(k -offset_k)%BLOCK_SIZE][l] = dzi2(l+1, k);
	    //if (! dzh2_S[l][(k -offset_k)%BLOCK_SIZE])
		dzh2_S[(k -offset_k)%BLOCK_SIZE][l] = dzh2(l+1, k);
	}
*/
	ib = 0;
	for (lb = lbx0; lb <= lbx1; lb++)
	{
		kb=0;
		//for (i = nd2_vel(7+4*lb); i <= nd2_vel(8+4*lb); i++)
		int offset_i = nd2_vel[6+4*lb];
		for (i = nd2_vel[6+4*lb]; i <= nd2_vel[7+4*lb]; i++)
		{
			kb=kb+1;
			ib=ib+1;
			rth=drvh2(kb,lb);
			rti=drti2(kb,lb);

/*			for (int l=0; l<4 ; l++) { 
			    //if (! dyi2_S[l][(j-offset_j)%BLOCK_SIZE])
				dxi2_S[(i-offset_i)%BLOCK_SIZE][l] = dxi2(l+1,i);
			    //if ! dyh2_S[l][(j-offset_j)%BLOCK_SIZE])
				dxh2_S[(i-offset_i)%BLOCK_SIZE][l] = dxh2(l+1,i);
			}
*/                        
/*
#undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1

#define dxi2(m, n) dxi2_S[(n-offset_i) % BLOCK_SIZE][m-1] 
#define dyi2(m, n) dyi2_S[(n-offset_j) % BLOCK_SIZE][m-1] 
#define dzi2(m, n) dzi2_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define dxh2(m, n) dxh2_S[(n-offset_i) % BLOCK_SIZE][m-1]  
#define dyh2(m, n) dyh2_S[(n-offset_j) % BLOCK_SIZE][m-1] 
#define dzh2(m, n) dzh2_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define BLOCK_SIZE 8
*/
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
/*        
#undef dxi2
#undef dyi2
#undef dzi2
#undef dxh2
#undef dyh2
#undef dzh2
#define dxi2(i, j) dxi2M[(i) - 1 + 4 * ((j) - 1)]
#define dyi2(i, j) dyi2M[(i) - 1 + 4 * ((j) - 1)]
#define dzi2(i, j) dzi2M[(i) - 1 + 4 * ((j) - 1)]
#define dxh2(i, j) dxh2M[(i) - 1 + 4 * ((j) - 1)]
#define dyh2(i, j) dyh2M[(i) - 1 + 4 * ((j) - 1)]
*/
	//}
	//}

	return;
}


//-----------------------------------------------------------------------
__kernel void vel_PmlY_IIC(int   nzbm1,
				 float ca,
				 int   lby0,
				 int   lby1,
				 __global int   *nd2_vel,
				 __global float *drvh2M,
				 __global float *drti2M,
				 __global float *rhoM,
				 __global float *damp2_yM,
				 __global int   *idmat2M,
				 __global float *dxi2M,
				 __global float *dyi2M,
				 __global float *dzi2M,
				 __global float *dxh2M,
				 __global float *dyh2M,
				 __global float *dzh2M,
				 __global float *t2xxM,
				 __global float *t2xyM,
				 __global float *t2xzM,
				 __global float *t2yyM,
				 __global float *t2yzM,
				 __global float *t2zzM,
				 int   mw2_pml1,
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 __global float *v2xM,		//output
				 __global float *v2yM,
				 __global float *v2zM,
				 __global float *v2x_pyM,
				 __global float *v2y_pyM,
				 __global float *v2z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	float rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
		   vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz;

	//if( lby(1)>lby(2) ) return;
	if( lby0>lby1 ) 
	{
		return;
	}

	int offset_k = 1;
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	int offset_i = nd2_vel[6];
	i  = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (k > nzbm1 || i > nd2_vel[11])
	{
		return;
	}

/*
#define BLOCK_SIZE 8

	__shared__ float dxi2_S[BLOCK_SIZE][4], dxh2_S[BLOCK_SIZE][4];
	__shared__ float dyi2_S[BLOCK_SIZE][4], dyh2_S[BLOCK_SIZE][4];
	__shared__ float dzi2_S[BLOCK_SIZE][4], dzh2_S[BLOCK_SIZE][4];

	for (int l = 0; l<4; l++){
	    //if (! dxi2_S[l][(i-offset_i)%BLOCK_SIZE])
		dxi2_S[(i-offset_i)%BLOCK_SIZE][l] = dxi2(l+1,i);
	    //if (! dxh2_S[l][(i-offset_i)%BLOCK_SIZE])
		dxh2_S[(i-offset_i)%BLOCK_SIZE][l] = dxh2(l+1,i);
	    //if (! dzi2_S[l][(k -offset_k)%BLOCK_SIZE])
		dzi2_S[(k -offset_k)%BLOCK_SIZE][l] = dzi2(l+1, k);
	    //if (! dzh2_S[l][(k -offset_k)%BLOCK_SIZE])
		dzh2_S[(k -offset_k)%BLOCK_SIZE][l] = dzh2(l+1, k);
	}
*/
	jb = 0;
	for (lb = lby0; lb <= lby1; lb++) 
	{
		kb = 0;
		int offset_j = nd2_vel[4*lb];
		for (j = nd2_vel[4*lb]; j <= nd2_vel[1+4*lb]; j++)
		{
/*                    
			for (int l=0; l<4 ; l++) { 
			    //if (! dyi2_S[l][(j-offset_j)%BLOCK_SIZE])
				dyi2_S[(j-offset_j)%BLOCK_SIZE][l] = dyi2(l+1,j);
			    //if ! dyh2_S[l][(j-offset_j)%BLOCK_SIZE])
				dyh2_S[(j-offset_j)%BLOCK_SIZE][l] = dyh2(l+1,j);
			}
*/
			kb=kb+1;
			jb=jb+1;
			rth=drvh2(kb,lb);
			rti=drti2(kb,lb);

/*
#undef dxi2
#undef dyi2
#undef dzi2
#undef dxh2
#undef dyh2
#undef dzh2

#define dxi2(m, n) dxi2_S[(n-offset_i) % BLOCK_SIZE][m-1] 
#define dyi2(m, n) dyi2_S[(n-offset_j) % BLOCK_SIZE][m-1] 
#define dzi2(m, n) dzi2_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define dxh2(m, n) dxh2_S[(n-offset_i) % BLOCK_SIZE][m-1]  
#define dyh2(m, n) dyh2_S[(n-offset_j) % BLOCK_SIZE][m-1] 
#define dzh2(m, n) dzh2_S[(n-offset_k) % BLOCK_SIZE][m-1] 
#define BLOCK_SIZE 8	
*/
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
/*
#undef dxi2
#undef dyi2
#undef dzi2
#undef dxh2
#undef dyh2
#undef dzh2
#define dxi2(i, j) dxi2M[(i) - 1 + 4 * ((j) - 1)]
#define dyi2(i, j) dyi2M[(i) - 1 + 4 * ((j) - 1)]
#define dzi2(i, j) dzi2M[(i) - 1 + 4 * ((j) - 1)]
#define dxh2(i, j) dxh2M[(i) - 1 + 4 * ((j) - 1)]
#define dyh2(i, j) dyh2M[(i) - 1 + 4 * ((j) - 1)]
*/
		//}
	//}
	return;
}


//-----------------------------------------------------------------------
__kernel void vel_PmlZ_IIC(int   nzbm1,
				 float ca,
				 __global int   *nd2_vel,
				 __global float *drvh2M,
				 __global float *drti2M,
				 __global float *rhoM,
				 __global float *damp2_zM,
				 __global int   *idmat2M,
				 __global float *dxi2M,
				 __global float *dyi2M,
				 __global float *dzi2M,
				 __global float *dxh2M,
				 __global float *dyh2M,
				 __global float *dzh2M,
				 __global float *t2xxM,
				 __global float *t2xyM,
				 __global float *t2xzM,
				 __global float *t2yyM,
				 __global float *t2yzM,
				 __global float *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 __global float *v2xM,		//output
				 __global float *v2yM,
				 __global float *v2zM,
				 __global float *v2x_pzM,
				 __global float *v2y_pzM,
				 __global float *v2z_pzM)
{
	int i,j,k,kb;
	float damp0,dmpz2,dmpz1,dmpxy2,dmpxy1,ro1,rox,roy,roz,vtmpx,vtmpy,vtmpz;

        int offset_i = nd2_vel[6];
        int offset_j = nd2_vel[0];
        int offset_k = nd2_vel[16];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

/*	if (j == nd2_vel[0] && i == nd2_vel[6])
	{
		for (k = 0; k < 18; k++)
		{
			printf("nd2_vel[%d] = %d\n", k, nd2_vel[k]);
		}
	}
*/        

	if (k > nzbm1 || i > nd2_vel[11])
	{
		return;
	}

// NO MUCH IMPROVEMENT FROM SHARED MEMORY         
/*#define BLOCK_SIZE 8 
        __shared__ float dxi2_S[4][BLOCK_SIZE], dxh2_S[4][BLOCK_SIZE];
        __shared__ float dyi2_S[4][36],dyh2_S[4][36];
        __shared__ float dzi2_S[4][BLOCK_SIZE], dzh2_S[4][BLOCK_SIZE];

        if (get_local_id(0) == 0) {
            dxi2_S[0][(i-offset_i)%BLOCK_SIZE] = dxi2(1,i);
            dxh2_S[0][(i-offset_i)%BLOCK_SIZE] = dxh2(1,i);
            dxi2_S[1][(i-offset_i)%BLOCK_SIZE] = dxi2(2,i);
            dxh2_S[1][(i-offset_i)%BLOCK_SIZE] = dxh2(2,i);
            dxi2_S[2][(i-offset_i)%BLOCK_SIZE] = dxi2(3,i);
            dxh2_S[2][(i-offset_i)%BLOCK_SIZE] = dxh2(3,i);
            dxi2_S[3][(i-offset_i)%BLOCK_SIZE] = dxi2(4,i);
            dxh2_S[3][(i-offset_i)%BLOCK_SIZE] = dxh2(4,i);
        }
        if(get_local_id(1) == 0) {
            for (int l = 0; l<4; l++){
                    dzi2_S[l][(k -offset_k)%BLOCK_SIZE] = dzi2(l+1, k);
                    dzh2_S[l][(k -offset_k)%BLOCK_SIZE] = dzh2(l+1, k);
            }
            if (get_local_id(0) == 0) {
                for (j = nd2_vel[0] ; j <= nd2_vel[5]; j++) 
                {
                   for (int l=0; l<4 ; l++) { 
                    //printf("Block:(%d,%d) dyi2_S[%d][(%d-%d)%15 ] = dyi2(%d+1,%d)\n",get_group_id(0), get_group_id(1),l,j,offset_j, l, j );
                    //if (! dyi2_S[l][(j-offset_j)%BLOCK_SIZE])
                        dyi2_S[l][(j-offset_j)%36] = dyi2(l+1,j);
                    //if (! dyh2_S[l][(j-offset_j)%BLOCK_SIZE])
                        dyh2_S[l][(j-offset_j)%36] = dyh2(l+1,j);
                   }
                }
            }
        }
        __syncthreads();

#undef dxi2
#undef dyi2
#undef dzi2
#undef dxh2
#undef dyh2
#undef dzh2
#define dxi2(m, n) dxi2_S[m-1][(n-offset_i)%8] 
#define dyi2(m, n) dyi2_S[m-1][(n-offset_j)%36] 
#define dzi2(m, n) dzi2_S[m-1][(n-offset_k)%8] 
#define dxh2(m, n) dxh2_S[m-1][(n-offset_i)%8] 
#define dyh2(m, n) dyh2_S[m-1][(n-offset_j)%36] 
#define dzh2(m, n) dzh2_S[m-1][(n-offset_k)%8] 
          
*/

       
	//for (j = nd2_vel(1); j <= nd2_vel(6); j++)
	for (j = nd2_vel[0]; j <= nd2_vel[5]; j++)
	{
			kb=0;
			damp0=damp2_z(i,j);
                                
                                kb = k - offset_k;
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
/*
#undef dxi2
#undef dyi2
#undef dzi2
#undef dxh2
#undef dyh2
#undef dzh2
#define dxi2(i, j) dxi2M[(i) - 1 + 4 * ((j) - 1)]
#define dyi2(i, j) dyi2M[(i) - 1 + 4 * ((j) - 1)]
#define dzi2(i, j) dzi2M[(i) - 1 + 4 * ((j) - 1)]
#define dxh2(i, j) dxh2M[(i) - 1 + 4 * ((j) - 1)]
#define dyh2(i, j) dyh2M[(i) - 1 + 4 * ((j) - 1)]
#define dzh2(i, j) dzh2M[(i) - 1 + 4 * ((j) - 1)]
 */
	//for (j = nd2_vel(1); j <= nd2_vel(6); j++)
	//for (j = nd2_vel[0]; j <= nd2_vel[5]; j++)
	//{
		//for (i = nd2_vel(7); i <= nd2_vel(12); i++)
		//for (i = nd2_vel[6]; i <= nd2_vel[11]; i++)
		//{
		//if ((k) + (nzbtm + 1) * ((i) + (nxbtm + 3) * (j)) == 16817)
		//{
		//	printf("dmpz1 = %f, dmpz2 = %f, roz = %f\n", dmpz1, dmpz2, roz);
		//	printf("dzh2=%f, t2zz1=%f, t2zz2=%f\n", dzh2(2,k), t2zz(k-1, i, j), t2zz(k, i, j));
		//	//printf("v2z = %f, v2z_pz = %f, vtmpz = %f, k = %d, i = %d, j = %d\n", v2z(k,i,j), v2z_pz(kb,i,j), vtmpz, k, i, j);
		//	printf("v2z = %f, v2z_pz = %f, k = %d, i = %d, j = %d\n", v2z(k,i,j), v2z_pz(kb,i,j), k, i, j);
		//}

	//}
		//}
	//}

	return;
}



//stress computation----------------------------------------------

__kernel void stress_norm_xy_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   __global int *nd1_tyy,
					   __global int *idmat1M,
					   float ca,
					   __global float *clamdaM,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwpM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dxh1M,
					   __global float *dyh1M,
					   __global float *dxi1M,
					   __global float *dyi1M,
					   __global float *dzi1M,
					   __global float *t1xxM,
					   __global float *t1xyM,
					   __global float *t1yyM,
					   __global float *t1zzM,
					   __global float *qt1xxM,
					   __global float *qt1xyM,
					   __global float *qt1yyM,
					   __global float *qt1zzM,
					   __global float *v1xM,
					   __global float *v1yM,
					   __global float *v1zM)
{
	int i,j,k,jkq,kodd,inod,irw;
	float sxx,syy,szz,sxy,qxx,qyy,qzz,qxy,cusxy,sss,cl,sm2,pm,et,et1,wtp,wts;

    int offset_k = nd1_tyy[12];
    int offset_i = nd1_tyy[8];
    k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
    i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

    if (k > nd1_tyy[17] || i > nd1_tyy[9])
    {
        return;
    }
	
	for (j = nd1_tyy[2]; j <= nd1_tyy[3]; j++)
	{
                kodd = 2 * ((j + nyb1) & 1) + 1;
                jkq=((i+nxb1) & 1) + kodd;
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
__kernel void  stress_xz_yz_IC(int nxb1,
					  int nyb1,
					  int nxtop,
					  int nytop,
					  int nztop,
					  __global int *nd1_tyz,
					  __global int *idmat1M,
					  float ca,
					  __global float *cmuM,
					  __global float *epdtM,
					  __global float *qwsM,
					  __global float *qwt1M,
					  __global float *qwt2M,
					  __global float *dxi1M,
					  __global float *dyi1M,
					  __global float *dzh1M,
					  __global float *v1xM,
					  __global float *v1yM,
					  __global float *v1zM,
					  __global float *t1xzM,
					  __global float *t1yzM,
					  __global float *qt1xzM,
					  __global float *qt1yzM)
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

    int offset_k = nd1_tyz[12];
    int offset_i = nd1_tyz[8];
    int offset_j = nd1_tyz[2];
    k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
    i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

    if (k > nd1_tyz[17] || i > nd1_tyz[9])
    {
        return;
    }

/*
#define BLOCK_SIZE 8
 	__shared__ float dxi1_S[BLOCK_SIZE][4];
	__shared__ float dyi1_S[72][4];
	__shared__ float dzh1_S[BLOCK_SIZE][4];

        if(get_local_id(0) == 0 && get_local_id(1) ==0 ) {
            int c_k = get_group_id(0) * get_local_size(0) + offset_k;
            int c_i = get_group_id(1) * get_local_size(1) + offset_i;
            for (int count=0; count < BLOCK_SIZE; count++) {
                for (int l = 0; l<4; l++){
                   dxi1_S[(c_i-offset_i)%BLOCK_SIZE][l] = dxi1(l+1,c_i);
                   dzh1_S[(c_k-offset_k)%BLOCK_SIZE][l] = dzh1(l+1, c_k);               
                   //dxh1_S[(i-offset_i)%BLOCK_SIZE][l] = dxh1(l+1,i);
                }
                c_i++;
                c_k++;
            }
            for ( int j = offset_j; j <= nd1_tyz[3]; j++) {
                  for (int l =0; l<4; l++) {
                      dyi1_S[(j-offset_j)%72][l] = dyi1(l+1,j);
                      //dyh1_S[(j-offset_j)%72][l] = dyh1(l+1,j);
                  }
             }

        }
    */
/*        if (get_local_id(1) == 0) {
            for(int l =0; l<4; l++) {
                //dzi1_S[(k -offset_k)%BLOCK_SIZE][l] = dzi1(l+1, k);
                dzh1_S[(k -offset_k)%BLOCK_SIZE][l] = dzh1(l+1, k);               
            }

            if (get_local_id(0) == 0 ) {
                for ( int j = offset_j; j <= nd1_tyz[3]; j++) {
                    for (int l =0; l<4; l++) {
                        dyi1_S[(j-offset_j)%72][l] = dyi1(l+1,j);
                        //dyh1_S[(j-offset_j)%72][l] = dyh1(l+1,j);
                    }
               }
            }
        }
*/
/*
        __syncthreads();

#undef dxi1
#undef dyi1
#undef dzh1

#define dxi1(m, n) dxi1_S[(n-offset_i) % BLOCK_SIZE][m-1] 
#define dyi1(m, n) dyi1_S[(n-offset_j) % 72][m-1] 
#define dzh1(m, n) dzh1_S[(n-offset_k) % BLOCK_SIZE][m-1] 
*/

	for (j=nd1_tyz[2]; j <=nd1_tyz[3]; j++)
	{
                kodd=2*((j+nyb1)&1)+1;
                jkq=((i+nxb1)&1)+kodd;
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
 #undef dxi1
#undef dyi1
#undef dzi1
#undef dxh1
#undef dyh1
#undef dzh1
#define dxi1(i, j) dxi1M[((j) - 1) * 4 + (i) - 1]
#define dyi1(i, j) dyi1M[((j) - 1) * 4 + (i) - 1]
#define dzi1(i, j) dzi1M[((j) - 1) * 4 + (i) - 1]
#define dxh1(i, j) dxh1M[((j) - 1) * 4 + (i) - 1]
#define dyh1(i, j) dyh1M[((j) - 1) * 4 + (i) - 1]
#define dzh1(i, j) dzh1M[((j) - 1) * 4 + (i) - 1]


	return;
}

__kernel void stress_resetVars(int ny1p1,
					  int nx1p1,
					  int nxtop,
					  int nytop,
					  int nztop,
					  __global float *t1xzM,
					  __global float *t1yzM)
{
	int i, j;

	j = get_group_id(0) * get_local_size(0) + get_local_id(0) - 1;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + 1;

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
__kernel void stress_norm_PmlX_IC(int nxb1,
						 int nyb1,
						 int nxtop,
						 int nytop,
						 int nztop,
						 int mw1_pml,
						 int mw1_pml1,
						 int lbx0,
						 int lbx1,
						 __global int *nd1_tyy,
						 __global int *idmat1M,
						 float ca,
						 __global float *drti1M,
						 __global float *damp1_xM,
						 __global float *clamdaM,
						 __global float *cmuM,
						 __global float *epdtM,
						 __global float *qwpM,
						 __global float *qwsM,
						 __global float *qwt1M,
						 __global float *qwt2M,
						 __global float *dzi1M,
						 __global float *dxh1M,
						 __global float *dyh1M,
						 __global float *v1xM,
						 __global float *v1yM,
						 __global float *v1zM,
						 __global float *t1xxM,
						 __global float *t1yyM,
						 __global float *t1zzM,
						 __global float *t1xx_pxM,
						 __global float *t1yy_pxM,
						 __global float *qt1xxM,
						 __global float *qt1yyM,
						 __global float *qt1zzM,
						 __global float *qt1xx_pxM,
						 __global float *qt1yy_pxM)
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

    int offset_k = nd1_tyy[12];
    int offset_j = nd1_tyy[0];
    k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
    j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

    if (j > nd1_tyy[5] || k> nd1_tyy[17])
    {
        return;
    }
    nti = (lbx1 - lbx0 + 1) * mw1_pml + lbx1;
	
    kodd=2*((j+nyb1)&1)+1;
    ib =0;
    for (lb=lbx0; lb <=lbx1; lb++ )
    {
	kb=0;
	for (i = nd1_tyy[6+4*lb]; i <= nd1_tyy[7+4*lb]; i++)
	{
		kb=kb+1;
		ib=ib+1;
		rti=drti1(kb,lb);
		jkq=((i+nxb1)&1)+kodd;
			damp2=1./(1.+damp1_x(k,j,lb)*rti);
			damp1=damp2*2.0-1.;
			inod=idmat1(k,i,j);
			cl=clamda(inod);
			sm2=2.*cmu(inod);
			pm=cl+sm2;
			irw=jkq+4*(k&1);
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
//		}
//	}
	return;
}

__kernel void stress_norm_PmlY_IC(int nxb1,
						 int nyb1,
						 int mw1_pml1,
						 int nxtop,
						 int nztop,
						 int lby0,
						 int lby1,
						 __global int *nd1_tyy,
						 __global int *idmat1M,
						 float ca,
						 __global float *drti1M,
						 __global float *damp1_yM,
						 __global float *clamdaM,
						 __global float *cmuM,
						 __global float *epdtM,
						 __global float *qwpM,
						 __global float *qwsM,
						 __global float *qwt1M,
						 __global float *qwt2M,
						 __global float *dxh1M,
						 __global float *dyh1M,
						 __global float *dzi1M,
						 __global float *t1xxM,
						 __global float *t1yyM,
						 __global float *t1zzM,
						 __global float *qt1xxM,
						 __global float *qt1yyM,
						 __global float *qt1zzM,
						 __global float *t1xx_pyM,
						 __global float *t1yy_pyM,
						 __global float *qt1xx_pyM,
						 __global float *qt1yy_pyM,
						 __global float *v1xM,
						 __global float *v1yM,
						 __global float *v1zM)
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

	//if(lby0>lby1) return;

        int offset_k = nd1_tyy[12];
        int offset_i = nd1_tyy[6];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;
//	lb = get_group_id(1) * get_local_size(1) + get_local_id(1) + lby0;
	if (i > nd1_tyy[11] || k> nd1_tyy[17])
	{
		return;
	}
        
        jb = 0;
        for (lb = lby0; lb<=lby1; lb++) 
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
                        jkq = ((i + nxb1) & 1) + kodd;
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

__kernel void stress_xy_PmlX_IC(int nxb1,
					   int nyb1,
					   int mw1_pml,
					   int mw1_pml1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int lbx0,
					   int lbx1,
					   __global int *nd1_txy,
					   __global int *idmat1M,
					   float ca,
					   __global float *drth1M,
					   __global float *damp1_xM,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dxi1M,
					   __global float *dyi1M,
					   __global float *t1xyM,
					   __global float *qt1xyM,
					   __global float *t1xy_pxM,
					   __global float *qt1xy_pxM,
					   __global float *v1xM,
					   __global float *v1yM)
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
    
        int offset_k = nd1_txy[12];
        int offset_j = nd1_txy[0];

	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	if (j > nd1_txy[5] || k > nd1_txy[17])
	{
		return;
	}

	kodd = 2 * ((j + nyb1) & 1) + 1;
        ib=0;
	for (lb = lbx0; lb <= lbx1; lb++)
	{
            kb=0;
            for (i = nd1_txy[6+4*lb]; i <= nd1_txy[7+4*lb]; i++)
            {
                        kb=kb+1;
                        ib=ib+1;
                        rth=drth1(kb,lb);
                        jkq=((i + nxb1) & 1) + kodd;
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

__kernel void stress_xy_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   __global int *nd1_txy,
					   __global int *idmat1M,
					   float ca,
					   __global float *drth1M,
					   __global float *damp1_yM,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dxi1M,
					   __global float *dyi1M,
					   __global float *t1xyM,
					   __global float *qt1xyM,
					   __global float *t1xy_pyM,
					   __global float *qt1xy_pyM,
					   __global float *v1xM,
					   __global float *v1yM)
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
        int offset_k = nd1_txy[12];
        int limit_k = nd1_txy[17];
        int offset_i = nd1_txy[6];
        int limit_i = nd1_txy[11];

	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;
	if (i > limit_i || k > limit_k)
	{
		return;
	}

	jb=0;
	for (lb = lby0; lb <= lby1; lb++)
	{
            kb=0;
            for (j = nd1_txy[4*lb]; j <= nd1_txy[1 + 4 * lb]; j++)
            {
                        kb=kb+1;
                        jb=jb+1;
                        rth=drth1(kb,lb);
                        kodd=2 * ((j + nyb1) & 1) + 1;
                        jkq=((i + nxb1) & 1) + kodd;
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

__kernel void stress_xz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nytop,
					   int nztop,
					   int mw1_pml,
					   int mw1_pml1,
					   int lbx0,
					   int lbx1,
					   __global int *nd1_txz,
					   __global int *idmat1M,
					   float ca,
					   __global float *drth1M,
					   __global float *damp1_xM,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dxi1M,
					   __global float *dzh1M,
					   __global float *t1xzM,
					   __global float *qt1xzM,
					   __global float *t1xz_pxM,
					   __global float *qt1xz_pxM,
					   __global float *v1xM,
					   __global float *v1zM)
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

        int offset_k = nd1_txz[12];
        int offset_j = nd1_txz[0];
        int limit_k = nd1_txz[17];
        int limit_j = nd1_txz[5];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	//if (j > limit_j || k> limit_k)
	if (j > nd1_txz[5] || k> nd1_txz[17])
	{
		return;
	}

	kodd=2 * ((j+nyb1)&1)+1;
	ib=0;

        for (lb = lbx0; lb <= lbx1; lb++)
	{
            kb=0;
            for (i = nd1_txz[6+4*lb]; i <= nd1_txz[7+4*lb]; i++)
            {
                        kb=kb+1;
                        ib=ib+1;
                        rth=drth1(kb,lb);
                        jkq=((i+nxb1)&1)+kodd;
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

__kernel void stress_xz_PmlY_IC(int nxb1,
					   int nyb1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   __global int *nd1_txz,
					   __global int *idmat1M,
					   float ca,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dxi1M,
					   __global float *dzh1M,
					   __global float *t1xzM,
					   __global float *qt1xzM,
					   __global float *v1xM,
					   __global float *v1zM)
//Compute the stress-xz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	float cusxz,dvxz,dvzx,qxz,sm,dmws,et,et1;

	//if (lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
        
        int offset_k = nd1_txz[12];
        int offset_i = nd1_txz[8];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (i > nd1_txz[9] || k> nd1_txz[17])
	{
		return;
	}

	for (lb=lby0; lb <= lby1; lb++)
	{
            for (j = nd1_txz[4*lb]; j <= nd1_txz[1+4*lb]; j++)
            {
                        kodd=2 * ((j+nyb1)&1)+1;
                        jkq=((i+nxb1)&1)+kodd;
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

__kernel void stress_yz_PmlX_IC(int nxb1,
					   int nyb1,
					   int nztop,
					   int nxtop,
					   int lbx0,
					   int lbx1,
					   __global int *nd1_tyz,
					   __global int *idmat1M,
					   float ca,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dyi1M,
					   __global float *dzh1M,
					   __global float *t1yzM,
					   __global float *qt1yzM,
					   __global float *v1yM,
					   __global float *v1zM)
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

        int offset_k = nd1_tyz[12];
        int offset_j = nd1_tyz[2];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	if (j > nd1_tyz[3] || k > nd1_tyz[17])
	{
		return;
	}

	kodd=2 * ((j+nyb1)&1)+1;
	for (lb = lbx0; lb <= lbx1; lb++)
	{
            for (i = nd1_tyz[6+4*lb]; i <= nd1_tyz[7+4*lb]; i++)
            {
                        jkq = ((i+nxb1)&1)+kodd;
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

__kernel void stress_yz_PmlY_IC(int nxb1,
					   int nyb1,
					   int mw1_pml1,
					   int nxtop,
					   int nztop,
					   int lby0,
					   int lby1,
					   __global int *nd1_tyz,
					   __global int *idmat1M,
					   float ca,
					   __global float *drth1M,
					   __global float *damp1_yM,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dyi1M,
					   __global float *dzh1M,
					   __global float *t1yzM,
					   __global float *qt1yzM,
					   __global float *t1yz_pyM,
					   __global float *qt1yz_pyM,
					   __global float *v1yM,
					   __global float *v1zM)
//Compute the stress-yz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	float taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1;

        int offset_k = nd1_tyz[12];
        int offset_i  = nd1_tyz[6];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (i > nd1_tyz[11] || k > nd1_tyz[17])
	{
		return;
	}

//	for (i = nd1_tyz[6]; i <= nd1_tyz[11]; i++)
//	//do i=nd1_tyz(7),nd1_tyz(12)
//	{
	jb=0;
	for (lb=lby0; lb <= lby1; lb++)
	{
            kb=0;
            for (j = nd1_tyz[4*lb]; j <= nd1_tyz[1+4*lb]; j++)
            {
                        kb=kb+1;
                        jb=jb+1;
                        rth=drth1(kb,lb);
                        kodd=2*((j+nyb1)&1)+1;
                        //kodd=2*mod(j+nyb1,2)+1;
                        jkq=((i+nxb1)&1)+kodd;
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

__kernel void stress_norm_xy_II(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   __global int *nd2_tyy,
					   __global int *idmat2M,
					   __global float *clamdaM,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwpM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *t2xxM,
					   __global float *t2xyM,
					   __global float *t2yyM,
					   __global float *t2zzM,
					   __global float *qt2xxM,
					   __global float *qt2xyM,
					   __global float *qt2yyM,
					   __global float *qt2zzM,
					   __global float *dxh2M,
					   __global float *dyh2M,
					   __global float *dxi2M,
					   __global float *dyi2M,
					   __global float *dzi2M,
					   __global float *v2xM,
					   __global float *v2yM,
					   __global float *v2zM)
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

        int offset_k = nd2_tyy[12];
        int offset_i = nd2_tyy[8];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (k > nd2_tyy[15] || i > nd2_tyy[9])
	{
		return;
	}

	for (j=nd2_tyy[2]; j <= nd2_tyy[3]; j++)
	{
                kodd=2*((j+nyb2)&1)+1;
                jkq=((i+nxb2)&1)+kodd;
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
__kernel void stress_xz_yz_IIC(int nxb2,
					  int nyb2,
					  int nztop,
					  int nxbtm,
					  int nzbtm,
					  __global int *nd2_tyz,
					  __global int *idmat2M,
					  __global float *cmuM,
					  __global float *epdtM,
					  __global float *qwsM,
					  __global float *qwt1M,
					  __global float *qwt2M,
					  __global float *dxi2M,
					  __global float *dyi2M,
					  __global float *dzh2M,
					  __global float *t2xzM,
					  __global float *t2yzM,
					  __global float *qt2xzM,
					  __global float *qt2yzM,
					  __global float *v2xM,
					  __global float *v2yM,
					  __global float *v2zM)
//Compute stress-XZ and YZ component in the Region II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,kodd,inod,jkq,irw
//real:: qxz,qyz,cusxz,cusyz,sm,et,et1,dmws
{
	int i,j,k,kodd,inod,jkq,irw;
	float qxz,qyz,cusxz,cusyz,sm,et,et1,dmws;

        int offset_k = nd2_tyz[12];
        int offset_i = nd2_tyz[8];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (k > nd2_tyz[15] || i > nd2_tyz[9])
	{
		return;
	}

	for (j = nd2_tyz[2]; j <= nd2_tyz[3]; j++)
	{
                kodd=2*((j+nyb2)&1)+1;
                jkq=((i+nxb2)&1)+kodd;
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
__kernel void stress_norm_PmlX_IIC(int nxb2,
						  int nyb2,
						  int mw2_pml,
						  int mw2_pml1,
						  int nztop,
						  int nxbtm,
						  int nybtm,
						  int nzbtm,
						  int lbx0,
						  int lbx1,
						  __global int *nd2_tyy,
						  __global int *idmat2M,
						  float ca,
						  __global float *drti2M,
						  __global float *damp2_xM,
						  __global float *clamdaM,
						  __global float *cmuM,
						  __global float *epdtM,
						  __global float *qwpM,
						  __global float *qwsM,
						  __global float *qwt1M,
						  __global float *qwt2M,
						  __global float *dxh2M,
						  __global float *dyh2M,
						  __global float *dzi2M,
						  __global float *t2xxM,
						  __global float *t2yyM,
						  __global float *t2zzM,
						  __global float *qt2xxM,
						  __global float *qt2yyM,
						  __global float *qt2zzM,
						  __global float *t2xx_pxM,
						  __global float *t2yy_pxM,
						  __global float *qt2xx_pxM,
						  __global float *qt2yy_pxM,
						  __global float *v2xM,
						  __global float *v2yM,
						  __global float *v2zM)
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
        
        int offset_k = nd2_tyy[12];
        int offset_j = nd2_tyy[0];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	if (j > nd2_tyy[5] || k > nd2_tyy[17])
	{
		return;
	}

	kodd=2*((j+nyb2)&1)+1;
	ib=0;
	for (lb=lbx0; lb <= lbx1; lb++)
	{
        	kb=0;
	        for (i=nd2_tyy[6+4*lb]; i <= nd2_tyy[7+4*lb]; i++)
	        {
                        kb=kb+1;
                        ib=ib+1;
                        rti=drti2(kb,lb);
                        jkq=((i+nxb2)&1)+kodd;
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

__kernel void stress_norm_PmlY_II(int nxb2,
						 int nyb2,
						 int nztop,
						 int nxbtm,
						 int nzbtm,
						 int mw2_pml1,
						 int lby0,
						 int lby1,
						 __global int *nd2_tyy,
						 __global int *idmat2M,
						 float ca,
						 __global float *drti2M,
						 __global float *damp2_yM,
						 __global float *clamdaM,
						 __global float *cmuM,
						 __global float *epdtM,
						 __global float *qwpM,
						 __global float *qwsM,
						 __global float *qwt1M,
						 __global float *qwt2M,
						 __global float *dxh2M,
						 __global float *dyh2M,
						 __global float *dzi2M,
						 __global float *t2xxM,
						 __global float *t2yyM,
						 __global float *t2zzM,
						 __global float *qt2xxM,
						 __global float *qt2yyM,
						 __global float *qt2zzM,
						 __global float *t2xx_pyM,
						 __global float *t2yy_pyM,
						 __global float *qt2xx_pyM,
						 __global float *qt2yy_pyM,
						 __global float *v2xM,
						 __global float *v2yM,
						 __global float *v2zM)
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

        int offset_i = nd2_tyy[6];
        int offset_k = nd2_tyy[12];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (i > nd2_tyy[11] || k > nd2_tyy[17])
	{
		return;
	}

	jb = 0;
        for (lb = lby0; lb <= lby1; lb++)
        {            
        	kb=0;

	        for (j=nd2_tyy[4*lb]; j <= nd2_tyy[1+4*lb]; j++)
	        {
                        kb=kb+1;
                        jb=jb+1;
                        rti=drti2(kb,lb);
                        kodd=2*((j+nyb2)&1)+1;
                        //kodd=2*mod(j+nyb2,2)+1;
                        jkq=((i+nxb2)&1)+kodd;
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

__kernel void stress_norm_PmlZ_IIC(int nxb2,
						  int nyb2,
						  int mw2_pml,
						  int mw2_pml1,
						  int nztop,
						  int nxbtm,
						  int nzbtm,
						  __global int *nd2_tyy,
						  __global int *idmat2M,
						  float ca,
						  __global float *damp2_zM,
						  __global float *drth2M,
						  __global float *clamdaM,
						  __global float *cmuM,
						  __global float *epdtM,
						  __global float *qwpM,
						  __global float *qwsM,
						  __global float *qwt1M,
						  __global float *qwt2M,
						  __global float *dxh2M,
						  __global float *dyh2M,
						  __global float *dzi2M,
						  __global float *t2xxM,
						  __global float *t2yyM,
						  __global float *t2zzM,
						  __global float *qt2xxM,
						  __global float *qt2yyM,
						  __global float *qt2zzM,
						  __global float *t2xx_pzM,
						  __global float *t2zz_pzM,
						  __global float *qt2xx_pzM,
						  __global float *qt2zz_pzM,
						  __global float *v2xM,
						  __global float *v2yM,
						  __global float *v2zM)
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

        int offset_k = nd2_tyy[16];
        int offset_i = nd2_tyy[6];
        
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (k > nd2_tyy[17] || i > nd2_tyy[11])
	{
		return;
	}

	for (j = nd2_tyy[0]; j <= nd2_tyy[5]; j++)
	{
                kodd=2*((j+nyb2)&1)+1;
                jkq=((i+nxb2)&1)+kodd;
                kb=0;
		kb=kb+1+k-offset_k;
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

__kernel void stress_xy_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int lbx0,
						int lbx1,
						__global int *nd2_txy,
						__global int *idmat2M,
						float ca,
						__global float *drth2M,
						__global float *damp2_xM,
						__global float *cmuM,
						__global float *epdtM,
						__global float *qwsM,
						__global float *qwt1M,
						__global float *qwt2M,
						__global float *dxi2M,
						__global float *dyi2M,
						__global float *t2xyM,
						__global float *qt2xyM,
						__global float *t2xy_pxM,
						__global float *qt2xy_pxM,
						__global float *v2xM,
						__global float *v2yM)
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

        int offset_k = nd2_txy[12];
        int offset_j = nd2_txy[0];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	if (j > nd2_txy[5] || k > nd2_txy[17])
	{
		return;
	}

	kodd=2*((j+nyb2)&1)+1;
	ib=0;
	for (lb = lbx0; lb <= lbx1; lb++)
        {            
            kb=0;
            for (i=nd2_txy[6+4*lb]; i <= nd2_txy[7+4*lb]; i++)
            {
                        kb=kb+1;
                        ib=ib+1;
                        rth=drth2(kb,lb);
                        jkq=((i+nxb2)&1)+kodd;
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

__kernel void stress_xy_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nztop,
						int nxbtm,
						int nzbtm,
						int lby0,
						int lby1,
						__global int *nd2_txy,
						__global int *idmat2M,
						float ca,
						__global float *drth2M,
						__global float *damp2_yM,
						__global float *cmuM,
						__global float *epdtM,
						__global float *qwsM,
						__global float *qwt1M,
						__global float *qwt2M,
						__global float *dxi2M,
						__global float *dyi2M,
						__global float *t2xyM,
						__global float *qt2xyM,
						__global float *t2xy_pyM,
						__global float *qt2xy_pyM,
						__global float *v2xM,
						__global float *v2yM)
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

        int offset_k = nd2_txy[12];
        int offset_i = nd2_txy[6];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (i > nd2_txy[11] || k > nd2_txy[17])
	{
		return;
	}

	jb = 0;
	for (lb=lby0; lb <= lby1; lb++)
	{
            kb=0;
            for (j=nd2_txy[4*lb]; j <= nd2_txy[1+4*lb]; j++)
            {
                        kb=kb+1;
                        jb=jb+1;
                        rth=drth2(kb,lb);
                        kodd=2*((j+nyb2)&1)+1;
                        jkq=((i+nxb2)&1)+kodd;
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

__kernel void stress_xy_PmlZ_II(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   __global int *nd2_txy,
					   __global int *idmat2M,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dxi2M,
					   __global float *dyi2M,
					   __global float *t2xyM,
					   __global float *qt2xyM,
					   __global float *v2xM,
					   __global float *v2yM)
//Compute the Stress-xy at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusxy,qxy,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	float cusxy,qxy,sm,dmws,et,et1;

        int offset_k = nd2_txy[16];
        int offset_i = nd2_txy[8];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (k > nd2_txy[17] || i > nd2_txy[9])
	{
		return;
	}

	for (j = nd2_txy[2]; j <= nd2_txy[3]; j++)
	{
                kodd=2*((j+nyb2)&1)+1;
                jkq=((i+nxb2)&1)+kodd;
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

__kernel void stress_xz_PmlX_IIC(int nxb2,
						int nyb2,
						int mw2_pml,
						int mw2_pml1,
						int nxbtm,
						int nybtm,
						int nzbtm,
						int nztop,
						int lbx0,
						int lbx1,
						__global int *nd2_txz,
						__global int *idmat2M,
						float ca,
						__global float *drth2M,
						__global float *damp2_xM,
						__global float *cmuM,
						__global float *epdtM,
						__global float *qwsM,
						__global float *qwt1M,
						__global float *qwt2M,
						__global float *dxi2M,
						__global float *dzh2M,
						__global float *t2xzM,
						__global float *qt2xzM,
						__global float *t2xz_pxM,
						__global float *qt2xz_pxM,
						__global float *v2xM,
						__global float *v2zM)
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

        int offset_k = nd2_txz[12];
        int offset_j = nd2_txz[0];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	if (j > nd2_txz[5] || k>nd2_txz[17])
	{
		return;
	}

	kodd=2*((j+nyb2)&1)+1;
	ib=0;
	for (lb=lbx0; lb <= lbx1; lb++)
	{
            kb=0;
            for (i=nd2_txz[6+4*lb]; i <= nd2_txz[7+4*lb]; i++)
            {
                        kb=kb+1;
                        ib=ib+1;
                        rth=drth2(kb,lb);
                        jkq=((i+nxb2)&1)+kodd;
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

__kernel void stress_xz_PmlY_IIC(int nxb2,
					   int nyb2,
					   int nxbtm,
					   int nzbtm,
					   int nztop,
					   int lby0,
					   int lby1,
					   __global int *nd2_txz,
					   __global int *idmat2M,
					   __global float *cmuM,
					   __global float *epdtM,
					   __global float *qwsM,
					   __global float *qwt1M,
					   __global float *qwt2M,
					   __global float *dxi2M,
					   __global float *dzh2M,
					   __global float *v2xM,
					   __global float *v2zM,
					   __global float *t2xzM,
					   __global float *qt2xzM)
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

        int offset_i = nd2_txz[8];
        int offset_k = nd2_txz[12];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (i > nd2_txz[9] || k > nd2_txz[15])
	{
		return;
	}

	for (lb = lby0; lb <= lby1; lb++)
	{
            for (j=nd2_txz[4*lb]; j <= nd2_txz[1+4*lb]; j++)
            {
                        kodd=2*((j+nyb2)&1)+1;
                        jkq=((i+nxb2)&1)+kodd;
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

__kernel void stress_xz_PmlZ_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						__global int *nd2_txz,
						__global int *idmat2M,
						float ca,
						__global float *drti2M,
						__global float *damp2_zM,
						__global float *cmuM,
						__global float *epdtM,
						__global float *qwsM,
						__global float *qwt1M,
						__global float *qwt2M,
						__global float *dxi2M,
						__global float *dzh2M,
						__global float *t2xzM,
						__global float *qt2xzM,
						__global float *t2xz_pzM,
						__global float *qt2xz_pzM,
						__global float *v2xM,
						__global float *v2zM)
//Compute the stress-xz at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	float taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1;

        int offset_k = nd2_txz[16];
        int offset_i = nd2_txz[6];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (k > nd2_txz[17] || i > nd2_txz[11])
	{
		return;
	}

	for (j = nd2_txz[0]; j <= nd2_txz[5]; j++)
	{
                kodd = 2*((j+nyb2)&1)+1;
                jkq=((i+nxb2)&1)+kodd;
                kb=0;
		kb=kb+1 + k -offset_k;
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
__kernel void stress_yz_PmlX_IIC(int nxb2,
						int nyb2,
						int nxbtm,
						int nzbtm,
						int nztop,
						int lbx0,
						int lbx1,
						__global int *nd2_tyz,
						__global int *idmat2M,
						__global float *cmuM,
						__global float *epdtM,
						__global float *qwsM,
						__global float *qwt1M,
						__global float *qwt2M,
						__global float *dyi2M,
						__global float *dzh2M,
						__global float *t2yzM,
						__global float *qt2yzM,
						__global float *v2yM,
						__global float *v2zM)
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

        int offset_k = nd2_tyz[12];
        int offset_j = nd2_tyz[2];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	j = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_j;

	if (j > nd2_tyz[3] || k > nd2_tyz[15])
	{
		return;
	}

	kodd=2*((j+nyb2)&1)+1;
	for (lb = lbx0; lb <= lbx1; lb++)
	{
            for (i = nd2_tyz[6+4*lb]; i <= nd2_tyz[7+4*lb]; i++)
            {
                        jkq=((i+nxb2)&1)+kodd;
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
__kernel void stress_yz_PmlY_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						int lby0,
						int lby1,
						__global int *nd2_tyz,
						__global int *idmat2M,
						float ca,
						__global float *drth2M,
						__global float *damp2_yM,
						__global float *cmuM,
						__global float *epdtM,
						__global float *qwsM,
						__global float *qwt1M,
						__global float *qwt2M,
						__global float *dyi2M,
						__global float *dzh2M,
						__global float *t2yzM,
						__global float *qt2yzM,
						__global float *t2yz_pyM,
						__global float *qt2yz_pyM,
						__global float *v2yM,
						__global float *v2zM)
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
        int offset_k = nd2_tyz[12];
        int offset_i = nd2_tyz[6];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + offset_i;

	if (i > nd2_tyz[11] || k > nd2_tyz[17])
	{
		return;
	}

	jb = 0;
	for (lb = lby0; lb <= lby1; lb++)
	{
            kb=0;
            for (j = nd2_tyz[4*lb]; j <= nd2_tyz[1+4*lb]; j++)
            {
                        kb=kb+1;
                        jb=jb+1;
                        rth=drth2(kb,lb);
                        kodd=2*((j+nyb2)&1)+1;
                        jkq = ((i+nxb2)&1)+kodd;
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
__kernel void stress_yz_PmlZ_IIC(int nxb2,
						int nyb2,
						int mw2_pml1,
						int nxbtm,
						int nzbtm,
						int nztop,
						__global int *nd2_tyz,
						__global int *idmat2M,
						float ca,
						__global float *drti2M,
						__global float *damp2_zM,
						__global float *cmuM,
						__global float *epdtM,
						__global float *qwsM,
						__global float *qwt1M,
						__global float *qwt2M,
						__global float *dyi2M,
						__global float *dzh2M,
						__global float *t2yzM,
						__global float *qt2yzM,
						__global float *t2yz_pzM,
						__global float *qt2yz_pzM,
						__global float *v2yM,
						__global float *v2zM)
//Compute the stress-yz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	float taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1;

        int offset_k = nd2_tyz[16];
	k = get_group_id(0) * get_local_size(0) + get_local_id(0) + offset_k;
	i = get_group_id(1) * get_local_size(1) + get_local_id(1) + nd2_tyz[6];

	if (k > nd2_tyz[17] || i > nd2_tyz[11])
	{
		return;
	}

	for (j = nd2_tyz[0]; j <= nd2_tyz[5]; j++)
	{
                kodd=2*((j+nyb2)&1)+1;
                jkq = ((i+nxb2)&1)+kodd;
                kb=0;
		kb=kb+1 + k-offset_k;
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


__kernel void vel_sdx51 (__global float* sdx51M, __global float* v1xM, __global float* v1yM, __global float* v1zM, int nytop, int nztop, int nxtop) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nytop) return;
sdx51(k,j,1) = v1x(k,1,j);
sdx51(k,j,2) = v1y(k,1,j);
sdx51(k,j,3) = v1y(k,2,j);
sdx51(k,j,4) = v1z(k,1,j);
sdx51(k,j,5) = v1z(k,2,j);
}

__kernel void vel_sdx52 (__global float* sdx52M, __global float* v2xM,  __global float* v2yM,  __global float* v2zM,  int nybtm,  int nzbtm,  int nxbtm) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nybtm) return;
sdx52(k,j,1) = v2x(k,1,j);
sdx52(k,j,2) = v2y(k,1,j);
sdx52(k,j,3) = v2y(k,2,j);
sdx52(k,j,4) = v2z(k,1,j);
sdx52(k,j,5) = v2z(k,2,j);
}

__kernel void vel_sdx41 ( __global float* sdx41M,  __global float* v1xM,  __global float* v1yM,  __global float* v1zM,  int nytop, int nztop, int nxtop, int nxtm1) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nytop) return;
sdx41(k,j,1) = v1x(k,nxtm1,j);
sdx41(k,j,2) = v1x(k,nxtop,j);
sdx41(k,j,3) = v1y(k,nxtop,j);
sdx41(k,j,4) = v1z(k,nxtop,j);
}


__kernel void vel_sdx42 ( __global float* sdx42M,  __global float* v2xM,  __global float* v2yM,  __global float* v2zM,  int nybtm, int nzbtm, int nxbtm, int nxbm1) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nybtm) return;
sdx42(k,j,1) = v2x(k,nxbm1,j);
sdx42(k,j,2) = v2x(k,nxbtm,j);
sdx42(k,j,3) = v2y(k,nxbtm,j);
sdx42(k,j,4) = v2z(k,nxbtm,j);
}

__kernel void vel_sdy51 ( __global float* sdy51M,  __global float* v1xM,  __global float* v1yM,  __global float* v1zM,  int nytop, int nztop, int nxtop) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nxtop) return;
sdy51(k,j,1) = v1x(k,j,1);
sdy51(k,j,2) = v1x(k,j,2);
sdy51(k,j,3) = v1y(k,j,1);
sdy51(k,j,4) = v1z(k,j,1);
sdy51(k,j,5) = v1z(k,j,2);
}

__kernel void vel_sdy52 ( __global float* sdy52M,  __global float* v2xM,  __global float* v2yM,  __global float* v2zM,  int nybtm, int nzbtm, int nxbtm) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nxbtm) return;
sdy52(k,j,1) = v2x(k,j,1);
sdy52(k,j,2) = v2x(k,j,2);
sdy52(k,j,3) = v2y(k,j,1);
sdy52(k,j,4) = v2z(k,j,1);
sdy52(k,j,5) = v2z(k,j,2);
}

__kernel void vel_sdy41 ( __global float* sdy41M,  __global float* v1xM,  __global float* v1yM,  __global float* v1zM,  int nytop, int nztop, int nxtop, int nytm1) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nxtop) return;
sdy41(k,j,1) = v1x(k,j, nytop);
sdy41(k,j,2) = v1y(k,j, nytm1);
sdy41(k,j,3) = v1y(k,j, nytop);
sdy41(k,j,4) = v1z(k,j, nytop);
}


__kernel void vel_sdy42 ( __global float* sdy42M,  __global float* v2xM,  __global float* v2yM,  __global float* v2zM,  int nybtm, int nzbtm, int nxbtm, int nybm1) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nxbtm) return;
sdy42(k,j,1) = v2x(k,j,nybtm);
sdy42(k,j,2) = v2y(k,j,nybm1);
sdy42(k,j,3) = v2y(k,j,nybtm);
sdy42(k,j,4) = v2z(k,j,nybtm);
}

__kernel void vel_rcx51 ( __global float* rcx51M,  __global float* v1xM,  __global float* v1yM,  __global float* v1zM,  int nytop, int nztop, int nxtop, int nx1p1, int nx1p2) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nytop) return;
v1x(k,nx1p1,j) = rcx51(k,j,1);
v1y(k,nx1p1,j) = rcx51(k,j,2);
v1y(k,nx1p2,j) = rcx51(k,j,3);
v1z(k,nx1p1,j) = rcx51(k,j,4);
v1z(k,nx1p2,j) = rcx51(k,j,5);
}

__kernel void vel_rcx52 (__global float* rcx52M, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nybtm, int nzbtm, int nxbtm, int nx2p1, int nx2p2) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nybtm) return;
v2x(k,nx2p1,j) = rcx52(k,j,1);
v2y(k,nx2p1,j) = rcx52(k,j,2);
v2y(k,nx2p2,j) = rcx52(k,j,3);
v2z(k,nx2p1,j) = rcx52(k,j,4);
v2z(k,nx2p2,j) = rcx52(k,j,5);
}

__kernel void vel_rcx41 (__global float* rcx41M, __global float* v1xM, __global float* v1yM, __global float* v1zM, int nytop, int nztop, int nxtop) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nytop) return;
v1x(k,-1,j) = rcx41(k,j,1) ;
v1x(k,0,j) = rcx41(k,j,2);
v1y(k,0,j) = rcx41(k,j,3);
v1z(k,0,j) = rcx41(k,j,4);
}


__kernel void vel_rcx42 (__global float* rcx42M, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nybtm, int nzbtm, int nxbtm) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nybtm) return;
v2x(k,-1,j) = rcx42(k,j,1);
v2x(k,0,j) = rcx42(k,j,2);
v2y(k,0,j) = rcx42(k,j,3);
v2z(k,0,j) = rcx42(k,j,4);
}

//--- rcy's
__kernel void vel_rcy51 (__global float* rcy51M, __global float* v1xM, __global float* v1yM, __global float* v1zM, int nytop, int nztop, int nxtop, int ny1p1, int ny1p2) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nxtop) return;
v1x(k,j,ny1p1) = rcy51(k,j,1);
v1x(k,j,ny1p2) = rcy51(k,j,2);
v1y(k,j,ny1p1) = rcy51(k,j,3);
v1z(k,j,ny1p1) = rcy51(k,j,4);
v1z(k,j,ny1p2) = rcy51(k,j,5);
}

__kernel void vel_rcy52 (__global float* rcy52M, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nybtm, int nzbtm, int nxbtm, int ny2p1, int ny2p2) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nxbtm) return;
v2x(k,j,ny2p1) = rcy52(k,j,1);
v2x(k,j,ny2p2) = rcy52(k,j,2);
v2y(k,j,ny2p1) = rcy52(k,j,3);
v2z(k,j,ny2p1) = rcy52(k,j,4);
v2z(k,j,ny2p2) = rcy52(k,j,5);
}

__kernel void vel_rcy41 (__global float* rcy41M, __global float* v1xM, __global float* v1yM, __global float* v1zM, int nytop, int nztop, int nxtop) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nztop || j >nxtop) return;
v1x(k,j, 0) = rcy41(k,j,1);
v1y(k,j,-1) = rcy41(k,j,2);
v1y(k,j, 0) = rcy41(k,j,3);
v1z(k,j, 0) = rcy41(k,j,4);
}


__kernel void vel_rcy42 (__global float* rcy42M, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nybtm, int nzbtm, int nxbtm) 
{
int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
if(k>nzbtm || j >nxbtm) return;
v2x(k,j,0) = rcy42(k,j,1) ;
v2y(k,j,-1) = rcy42(k,j,2) ;
v2y(k,j,0) = rcy42(k,j,3) ;
v2z(k,j,0) = rcy42(k,j,4) ;
}

__kernel void vel_sdx1(__global float* sdx1D, __global float* v2xM, __global float* v2yM,  __global float* v2zM, int nxbtm, int nzbtm, int ny2p1, int ny2p2) 
{
int j0 = ny2p2 +2;
for( int j=0; j<=ny2p2; j++) {
 sdx1D[j] = v2z(1,3,j);
 sdx1D[j+j0-1] = v2x(1,2,j);
 }
 j0 = 2*(ny2p2+1); 
 sdx1D[j0] = v2z(1,1,ny2p1);
 sdx1D[j0 + 1] = v2z(1,2,ny2p1);
 sdx1D[j0 + 2] = v2z(1,1,ny2p2);
 sdx1D[j0 + 3] = v2z(1,2,ny2p2);
 }

 __kernel void vel_sdy1(__global float* sdy1D, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nxbtm, int nzbtm, int nx2p1, int nx2p2) 
 {
 int i0 = nx2p2 +2;
 for( int i=0; i<=nx2p2; i++) {
  sdy1D[i] = v2z(1,i,3);
  sdy1D[i+i0-1] = v2y(1,i,2);
  }
  i0 = 2*(nx2p2+1); 
  sdy1D[i0] = v2x(1,nx2p1,1);
  sdy1D[i0 + 1] = v2x(1,nx2p1,2);
  sdy1D[i0 + 2] = v2y(1,nx2p1,1);
  sdy1D[i0 + 3] = v2y(1,nx2p2,1);
  }

  __kernel void vel_sdx2(__global float* sdx2D, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nxbm1, int nxbtm, int nzbtm,  int ny2p1, int ny2p2) 
  {
  for( int j=0; j<=ny2p2; j++) {
   sdx2D[j] = v2z(1,nxbm1,j);
   }
   int j0 = (ny2p2+1); 
   sdx2D[j0    ] = v2x(1,nxbm1, ny2p1);
   sdx2D[j0 + 1] = v2x(1,nxbtm, ny2p1);
   sdx2D[j0 + 2] = v2x(1,nxbm1, ny2p2);
   sdx2D[j0 + 3] = v2x(1,nxbtm, ny2p2);
   sdx2D[j0 + 4] = v2y(1,nxbtm, ny2p1);
   sdx2D[j0 + 5] = v2z(1,nxbtm, ny2p1);
   sdx2D[j0 + 6] = v2z(1,nxbtm, ny2p2);
   }

   __kernel void vel_sdy2(__global float* sdy2D, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nybm1, int nybtm, int nxbtm, int nzbtm, int nx2p1, int nx2p2) 
   {
   for( int i=0; i<=nx2p2; i++) {
sdy2D[i] = v2z(1,i, nybm1);
}
int i0 = (nx2p2+1); 
sdy2D[i0    ] = v2x(1,    -1, nybtm);
sdy2D[i0 + 1] = v2x(1,     0, nybtm);
sdy2D[i0 + 2] = v2x(1, nx2p1, nybtm);
sdy2D[i0 + 3] = v2y(1,     0, nybm1);
sdy2D[i0 + 4] = v2y(1, nx2p1, nybm1);
sdy2D[i0 + 5] = v2y(1, nx2p2, nybm1);
sdy2D[i0 + 6] = v2y(1,     0, nybtm);
sdy2D[i0 + 7] = v2y(1, nx2p1, nybtm);
sdy2D[i0 + 8] = v2y(1, nx2p2, nybtm);
sdy2D[i0 + 9] = v2z(1,     0, nybtm);
sdy2D[i0 + 10] = v2z(1,nx2p1, nybtm);
sdy2D[i0 + 11] = v2z(1,nx2p2, nybtm);
}

__kernel void vel_rcx1(__global float* rcx1D, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nxbtm, int nzbtm, int ny2p1, int ny2p2) 
{
int j0 = (ny2p2+1); 
v2x(1,-1,ny2p1) = rcx1D[j0];
v2x(1, 0,ny2p1) = rcx1D[j0 + 1];
v2x(1,-1,ny2p2) = rcx1D[j0 + 2];
v2x(1, 0,ny2p2) = rcx1D[j0 + 3];
v2y(1, 0,ny2p1) = rcx1D[j0 + 4];
v2z(1, 0,ny2p1) = rcx1D[j0 + 5];
v2z(1, 0,ny2p2) = rcx1D[j0 + 6];
}

__kernel void  vel_rcy1(__global float* rcy1D, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nxbtm, int nzbtm,  int nx2p1, int nx2p2) 
{
int i0 = nx2p2 +1;
v2x(1,    -1,0) = rcy1D[i0];
v2x(1,     0,0) = rcy1D[i0 + 1] ;
v2x(1, nx2p1,0) = rcy1D[i0 + 2] ;
v2y(1,    0,-1) = rcy1D[i0 + 3] ;
v2y(1,nx2p1,-1) = rcy1D[i0 + 4] ;
v2y(1,nx2p2,-1) = rcy1D[i0 + 5] ;
v2y(1,     0,0) = rcy1D[i0 + 6] ;
v2y(1,nx2p1, 0) = rcy1D[i0 + 7] ;
v2y(1,nx2p2, 0) = rcy1D[i0 + 8] ;
v2z(1,   0,  0) = rcy1D[i0 + 9] ;
v2z(1,nx2p1, 0) = rcy1D[i0 + 10] ;
v2z(1,nx2p2, 0) = rcy1D[i0 + 11] ;
}

__kernel void vel_rcx2(__global float* rcx2D, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nxbtm, int nzbtm,int nx2p1, int nx2p2, int ny2p1, int ny2p2) 
{
int j0 = 2*(ny2p2+1); 
v2z(1,nx2p1, ny2p1) = rcx2D[j0    ] ;
v2z(1,nx2p2, ny2p1) = rcx2D[j0 + 1] ;
v2z(1,nx2p1, ny2p2) = rcx2D[j0 + 2] ;
v2z(1,nx2p2, ny2p2) = rcx2D[j0 + 3] ;
}

__kernel void vel_rcy2(__global float* rcy2D, __global float* v2xM, __global float* v2yM, __global float* v2zM, int nxbtm, int nzbtm, int nx2p1, int nx2p2, int ny2p1, int ny2p2) 
{
int i0 = 2*(nx2p2+1); 
v2x(1,nx2p1, ny2p1) = rcy2D[i0    ];
v2x(1,nx2p1, ny2p2) = rcy2D[i0 + 1];
v2y(1,nx2p1, ny2p1) = rcy2D[i0 + 2];
v2y(1,nx2p2, ny2p1) = rcy2D[i0 + 3];
}

__kernel void vel1_interpl_3vbtm( int ny1p2, int ny2p2, int nz1p1, int nyvx, int nxbm1, int nxbtm, int nzbtm, int nxtop, int nztop, int neighb2,                __global float* chxM, __global float* v1xM, __global float * v2xM, __global float* rcx2D)
{

int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);

if (tx > nyvx || ty >= nxbm1) return;

int ii = ty + 1;
int jj = tx;

int i = -2 + 3*ii;
int j = ( jj==0 ? 0 : ((jj-1)*3+1 < ny1p2 ? (jj-1)*3+1 : ny1p2)) ;

float vv1 = v2x(1,-1 + ii-1, jj);
float vv2 = v2x(1, 0 + ii-1, jj);
float vv3 = v2x(1, 1 + ii-1, jj);
float vv4 = v2x(1, 2 + ii-1, jj);

/* 
if ( get_group_id(0) <= (nyvx/get_local_size(0) -1) && get_group_id(1) <= (nxbm1/get_local_size(1)-1)){
__shared__ float chxS[8][32];
if (get_local_id(1) < 8) {
int r1 = get_local_id(0)%8;
int c1 = (get_local_id(1)*32+get_local_id(0))/8;
int r2 = get_local_id(0)%8 +1;
int c2 = get_group_id(1)*get_local_size(1)+ (int)((get_local_id(1)*32+get_local_id(0))/8)+1 ;
chxS[r1][c1] =  chx(r2,c2);

//if (get_group_id(0) == 3 && get_group_id(1) ==3 ) {
//    printf("chxS[%d][%d] = chx[%d][%d] = %f %f\n", 
 //    r1, c1, r2, c2, chxS[r1][c1], chx(r2,c2));
//}
//barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}
barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

int ii1 = get_local_id(1);
//if (get_group_id(0) == 3 && get_group_id(1) ==3 ) {
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 1,ii, chx(1,ii), 0,ii1, chxS[0][ii1]);
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 2,ii, chx(2,ii), 1,ii1, chxS[1][ii1]);
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 3,ii, chx(3,ii), 2,ii1, chxS[2][ii1]);
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 4,ii, chx(4,ii), 3,ii1, chxS[3][ii1]);
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 5,ii, chx(5,ii), 4,ii1, chxS[4][ii1]);
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 6,ii, chx(6,ii), 5,ii1, chxS[5][ii1]);
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 7,ii, chx(7,ii), 6,ii1, chxS[6][ii1]);
//     printf("chx(%d, %d) = %f, chxS[%d][%d]= %f \n", 8,ii, chx(8,ii), 7,ii1, chxS[7][ii1]);
// }


v1x(nz1p1,i,j)=vv1*chxS[0][ii1]+vv2*chxS[1][ii1]+ 
vv3*chxS[2][ii1]+vv4*chxS[3][ii1];
vv1=vv2;
vv2=vv3;
vv3=vv4;
vv4=v2x(1,ii+2,jj);
v1x(nz1p1,i+1,j)=vv2;
v1x(nz1p1,i+2,j)=vv1*chxS[4][ii1]+vv2*chxS[5][ii1]+ 
vv3*chxS[6][ii1]+vv4*chxS[7][ii1];


} else {
*/
v1x(nz1p1,i,j)=vv1*chx(1,ii)+vv2*chx(2,ii)+ 
vv3*chx(3,ii)+vv4*chx(4,ii);
vv1=vv2;
vv2=vv3;
vv3=vv4;
vv4=v2x(1,ii+2,jj);
v1x(nz1p1,i+1,j)=vv2;
v1x(nz1p1,i+2,j)=vv1*chx(5,ii)+vv2*chx(6,ii)+ 
vv3*chx(7,ii)+vv4*chx(8,ii);

if(ii==nxbm1 && neighb2>-1) {
i=i+3;
int j0=ny2p2+2;
v1x(nz1p1,i,j  )=vv1*chx(1,nxbtm)+vv2*chx(2,nxbtm)+ 
vv3*chx(3,nxbtm)+vv4*chx(4,nxbtm);
v1x(nz1p1,i+1,j)=vv3;
v1x(nz1p1,i+2,j)=vv2*chx(5,nxbtm)+vv3*chx(6,nxbtm)+ 
rcx2D[jj+j0-1]*chx(8,nxbtm)+vv4*chx(7,nxbtm);             
}
// }
}

__kernel void vel3_interpl_3vbtm( int ny1p2, int nz1p1, int nyvx1,  int nxbm1, int nxbtm, int nzbtm, int nxtop, int nztop,   __global float* ciyM, __global float* v1xM)
{

int tx = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1; 
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;

if (tx > nxtop || ty > nyvx1 ) return;

int i = tx;
int jj = ty;

int j = 2 + 3*(jj-1);

float vv1 = v1x(nz1p1, i, ((jj==1) ? 0 : ((1+ (jj-2)*3 <ny1p2)? 1+(jj-2)*3 :ny1p2)));
float vv2 = v1x(nz1p1, i, ((1+(jj-1)*3 < ny1p2)?(1+(jj-1)*3):ny1p2));
float vv3 = v1x(nz1p1, i, ((4+(jj-1)*3 < ny1p2)?(4+(jj-1)*3):ny1p2));
float vv4 = v1x(nz1p1, i, ((7+(jj-1)*3 < ny1p2)?(7+(jj-1)*3):ny1p2));
v1x(nz1p1,i,j  )=vv1*ciy(1,jj)+vv2*ciy(2,jj)+ 
vv3*ciy(3,jj)+vv4*ciy(4,jj);
v1x(nz1p1,i,j+1)=vv1*ciy(5,jj)+vv2*ciy(6,jj)+ 
vv3*ciy(7,jj)+vv4*ciy(8,jj);
}


__kernel void vel4_interpl_3vbtm( int nx1p2, int ny2p2, int nz1p1, int nxvy,  int nybm1, int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop,   __global float* chyM, __global float* v1yM, __global float * v2yM)
{
int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);

int ii = tx;
int jj = ty+1;
if (ii>nxvy || jj>nybm1) return;

int j = -2 + 3*jj;
int i = ( ii==0 ? 0 : ((ii-1)*3+1 < nx1p2 ? (ii-1)*3+1 : nx1p2)) ;

float vv1 = v2y(1, ii,-1 + jj-1);
float vv2 = v2y(1, ii, 0 + jj-1);
float vv3 = v2y(1, ii, 1 + jj-1);
float vv4 = v2y(1, ii, 2 + jj-1);

v1y(nz1p1,i,j)=vv1*chy(1,jj)+vv2*chy(2,jj)+ 
vv3*chy(3,jj)+vv4*chy(4,jj);
vv1=vv2;
vv2=vv3;
vv3=vv4;
vv4=v2y(1,ii, 2+jj);
v1y(nz1p1,i,j+1)=vv2;
v1y(nz1p1,i,j+2)=vv1*chy(5,jj)+vv2*chy(6,jj)+ 
vv3*chy(7,jj)+vv4*chy(8,jj);
}


__kernel void vel5_interpl_3vbtm( int nx1p2, int nx2p2, int nz1p1, int nxvy,  int nybm1, int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop,   __global float* chyM, __global float* v1yM, __global float* v2yM, __global float* rcy2D)
{
int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
if (tx > nxvy) return;
int ii = tx;

int j = -2 + 3*nybm1;
int i = ( ii==0 ? 0 : ((ii-1)*3+1 < nx1p2 ? (ii-1)*3+1 : nx1p2)) ;

float vv1 = v2y(1, ii, nybm1-1);
float vv2 = v2y(1, ii, nybm1+0);
float vv3 = v2y(1, ii, nybm1+1);
float vv4 = v2y(1, ii, nybm1+2 );

j=j+3;
int i0 = nx2p2+2;
v1y(nz1p1,i,j)=vv1*chy(1,nybtm)+vv2*chy(2,nybtm)+ 
vv3*chy(3,nybtm)+vv4*chy(4,nybtm);
v1y(nz1p1,i,j+1)=vv3;
v1y(nz1p1,i,j+2)=vv2*chy(5,nybtm)+vv3*chy(6,nybtm)+ 
rcy2D[ii+i0-1]*chy(8,nybtm)+vv4*chy(7,nybtm);

}

__kernel void vel6_interpl_3vbtm( int nx1p2, int nz1p1, int nxvy1,  int nybm1, int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop,   __global float* cixM, __global float* v1yM)
{

int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);

int j = tx+1;
int ii = ty+1;
if (j>nytop || ii>nxvy1) return;

int i = 2 + 3*(ii-1);

float vv1 = v1y(nz1p1, (ii==1 ? 0 : (1+ (ii-2)*3 < nx1p2 ? 1+(ii-2)*3 : nx1p2)) , j);
float vv2 = v1y(nz1p1, (1 + (ii-1)*3 < nx1p2 ? 1+(ii-1)*3 : nx1p2), j );
float vv3 = v1y(nz1p1, (4 + (ii-1)*3 < nx1p2 ? 4+(ii-1)*3 : nx1p2), j);
float vv4 = v1y(nz1p1, (7 + (ii-1)*3 < nx1p2 ? 7+(ii-1)*3 : nx1p2), j);

v1y(nz1p1,i  ,j)=vv1*cix(1,ii)+vv2*cix(2,ii)+vv3*cix(3,ii)+vv4*cix(4,ii);
v1y(nz1p1,i+1,j)=vv1*cix(5,ii)+vv2*cix(6,ii)+vv3*cix(7,ii)+vv4*cix(8,ii);
}

__kernel void vel7_interpl_3vbtm( int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop,   __global float* ciyM, __global float* sdx1D,  __global float* rcx1D)
{

int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int jj = tx+1;

if (jj>nybtm) return;

int j = 2 + 3*(jj-1);

float vv1 = rcx1D[0 + (jj-1)];
float vv2 = rcx1D[1 + (jj-1)];
float vv3 = rcx1D[2 + (jj-1)];
float vv4 = rcx1D[3 + (jj-1)];
sdx1D[j-1]=vv2  ;
sdx1D[j]=vv1*ciy(1,jj)+vv2*ciy(2,jj)+vv3*ciy(3,jj)+vv4*ciy(4,jj);
sdx1D[j+1]=vv1*ciy(5,jj)+vv2*ciy(6,jj)+vv3*ciy(7,jj)+vv4*ciy(8,jj);

if (jj==nybtm) {
sdx1D[0] = (rcx1D[0] + rcx1D[1]*2.0)/3.0;
j=j+3;
sdx1D[j-1] = vv3; //vv3 and vv4 are put intentionally as assignments aint done
sdx1D[j] = (vv3*2 + vv4)/3.0;
}
}


__kernel void vel8_interpl_3vbtm( int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop,  __global float* ciyM, __global float* sdx2D,  __global float* rcx2D)
{
int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int jj = tx+1;

if (jj>nybtm) return;

int j = 2 + 3*(jj-1);

float vv1 = rcx2D[0 + (jj-1)];
float vv2 = rcx2D[1 + (jj-1)];
float vv3 = rcx2D[2 + (jj-1)];
float vv4 = rcx2D[3 + (jj-1)];
sdx2D[j-1]=vv2  ;
sdx2D[j]=vv1*ciy(1,jj)+vv2*ciy(2,jj)+vv3*ciy(3,jj)+vv4*ciy(4,jj);
sdx2D[j+1]=vv1*ciy(5,jj)+vv2*ciy(6,jj)+vv3*ciy(7,jj)+vv4*ciy(8,jj);

if (jj==nybtm) {
sdx2D[0] = (rcx2D[0] + rcx2D[1]*2.0)/3.0;
j=j+3;
sdx2D[j-1] = vv3; //vv3 and vv4 are put intentionally as assignments aint done
sdx2D[j] = (vv3*2 + vv4)/3.0;
}
}

__kernel void vel9_interpl_3vbtm( int nz1p1, int nx1p2, int ny2p1, int nxvy,  int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop,   int neighb4, __global float* ciyM, __global float* rcy1D, __global float* rcy2D, __global float* v1zM, __global float* v2zM)
{

int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);

int ii = tx;
int jj = ty+1;
if (ii>nxvy || jj>nybtm) return;

int j = 1 + 3*(jj-1);
int i = ((ii==0 )? 0 : (((ii-1)*3+1 < nx1p2)?(ii-1)*3+1:nx1p2 ));

float vv1 = ((jj==1) ? rcy1D[ii] : v2z(1,ii, jj-2));
float vv2 = v2z(1,ii, 0 + jj-1);
float vv3 = v2z(1,ii, 1 + jj-1);
float vv4 = v2z(1,ii, 2 + jj-1);

if (ty==0){
v1z(nz1p1,i,0)=vv1*ciy(5,0)+vv2*ciy(6,0)+vv3*ciy(7,0)+vv4*ciy(8,0);
}

vv1=vv2;
vv2=vv3;
vv3=vv4;
vv4=v2z(1,ii,3+jj-1);
v1z(nz1p1,i,j  )=vv2;
v1z(nz1p1,i,j+1)=vv1*ciy(1,jj)+vv2*ciy(2,jj)+ 
vv3*ciy(3,jj)+vv4*ciy(4,jj);
v1z(nz1p1,i,j+2)=vv1*ciy(5,jj)+vv2*ciy(6,jj)+ 
vv3*ciy(7,jj)+vv4*ciy(8,jj);

if (neighb4 > -1 && jj==nybtm) {
j = j+3;
v1z(nz1p1,i,j)=vv3;
v1z(nz1p1,i,j+1)=vv2*ciy(1,ny2p1)+vv3*ciy(2,ny2p1)+ 
rcy2D[ii]*ciy(4,ny2p1)+vv4*ciy(3,ny2p1);
}
}


__kernel void vel11_interpl_3vbtm( int nx1p2, int nx2p1, int ny1p1, int nz1p1, int nxvy1,   int nxbtm, int nybtm, int nzbtm, int nxtop, int nytop, int nztop,    __global float* cixM, __global float* sdx1D, __global float* sdx2D, __global float* v1zM)
{
int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);
int j = tx;
int ii = ty+1;

if (j>ny1p1 || ii>nxvy1) return;

int i = 2 + 3*(ii-1);
float vv1 = ((ii==1) ? sdx1D[j] :  v1z(nz1p1, ((ii==2) ? 0 : (((1+(ii-3)*3) <nx1p2) ? (1+(ii-3)*3) : nx1p2)) , j));
float vv2 = v1z(nz1p1, ((ii==1)?0:((1+(ii-2)*3<nx1p2)?1+(ii-2)*3:nx1p2)) , j);
float vv3 = v1z(nz1p1, ((1+(ii-1)*3<nx1p2)?1+(ii-1)*3:nx1p2), j);
float vv4 = v1z(nz1p1, ((4+(ii-1)*3<nx1p2)?4+(ii-1)*3:nx1p2), j);

if (ty==0){
v1z(nz1p1,0,j)=vv1*cix(5,0)+vv2*cix(6,0)+vv3*cix(7,0)+vv4*cix(8,0);
}

vv1=vv2;
vv2=vv3;
vv3=vv4;
vv4=v1z(nz1p1, ((7+(ii-1)*3<nx1p2)?7+(ii-1)*3:nx1p2), j);
vv4=v1z(nz1p1,(i+5<nx1p2?i+5:nx1p2),j);
v1z(nz1p1,i  ,j)=vv1*cix(1,ii)+vv2*cix(2,ii)+ 
vv3*cix(3,ii)+vv4*cix(4,ii);
v1z(nz1p1,i+1,j)=vv1*cix(5,ii)+vv2*cix(6,ii)+ 
vv3*cix(7,ii)+vv4*cix(8,ii);

if (ii==nxvy1){
i+=3;
v1z(nz1p1,2+3*nxvy1,j)=vv2*cix(1,nx2p1)+vv3*cix(2,nx2p1)+ 
sdx2D[j]*cix(4,nx2p1)+vv4*cix(3,nx2p1);
}
}

__kernel void vel13_interpl_3vbtm(__global float* v1xM, __global float* v2xM,   int nxbtm, int nybtm, int nzbtm,   int nxtop, int nytop, int nztop)
{
int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);
int jj = tx +1;
int ii= ty+1;
if(jj>nybtm || ii >nxbtm) return;
int j = jj*3 -2;
v2x(0,ii,jj)=v1x(nztop,ii*3-1,j);
}


__kernel void vel14_interpl_3vbtm(__global float* v1yM, __global float* v2yM,    int nxbtm, int nybtm, int nzbtm,   int nxtop, int nytop, int nztop)
{
int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);
int jj = tx +1;
int ii= ty+1;
if(jj>nybtm || ii >nxbtm) return;
int j = jj*3 -1;
v2y(0,ii,jj)=v1y(nztop,ii*3-2,j);
}

__kernel void vel15_interpl_3vbtm(__global float* v1zM, __global float* v2zM,    int nxbtm, int nybtm, int nzbtm,   int nxtop, int nytop, int nztop)
{
int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);
int jj = tx +1;
int ii= ty+1;
if(jj>nybtm || ii >nxbtm) return;
int j = jj*3 -2;
v2z(0,ii,jj)=v1z(nztop,ii*3-2,j);
}

__kernel void vel1_vxy_image_layer (__global float* v1xM, __global float* v1zM, __global int* nd1_velD, __global float* dxi1M,  __global float* dzh1M, int i, float dzdx, int nxbtm, int nybtm, int nzbtm,  int nxtop, int nytop, int nztop) 
{
int tx = get_group_id(0)* get_local_size(0) + get_local_id(0);
int j = tx + nd1_velD[0];
if (j > nd1_velD[5]) return;
v1x(0,i,j)=v1x(1,i,j)+dzdx*(v1z(1,i+1,j)-v1z(1,i,j));
} 

__kernel void vel2_vxy_image_layer (__global float* v1xM, __global float* v1zM, __global int* nd1_velD, __global float* dxi1M,  __global float* dzh1M, int iix, float dzdt, int nxbtm, int nybtm, int nzbtm,  int nxtop, int nytop, int nztop) 
{

int j = get_group_id(0)* get_local_size(0) + get_local_id(0) + nd1_velD[0];
int i = get_group_id(1)* get_local_size(1) + get_local_id(1) + nd1_velD[6];

if ( j>nd1_velD[5] || i>iix) return;
v1x(0,i,j)=v1x(1,i,j)+dzdt* (dxi1(1,i)*v1z(1,i-1,j)+dxi1(2,i)*v1z(1,i,  j)+ 
  dxi1(3,i)*v1z(1,i+1,j)+dxi1(4,i)*v1z(1,i+2,j));
}  


__kernel void vel3_vxy_image_layer (__global float* v1yM, __global float* v1zM, __global int* nd1_velD, __global float* dyi1M,  __global float* dzh1M, int j, float dzdy, int nxbtm, int nybtm, int nzbtm,  int nxtop, int nytop, int nztop) 
{

int tx = get_group_id(0)* get_local_size(0) + get_local_id(0);
int i = tx + nd1_velD[6];
if (i > nd1_velD[11]) return;

v1y(0,i,j)=v1y(1,i,j)+dzdy*(v1z(1,i,j+1)-v1z(1,i,j));
} 

__kernel void vel4_vxy_image_layer (__global float* v1yM, __global float* v1zM, __global int* nd1_velD, __global float* dyi1M,  __global float* dzh1M, int jjy, float dzdt, int nxbtm, int nybtm, int nzbtm,  int nxtop, int nytop, int nztop) 
{

int j = get_group_id(0)* get_local_size(0) + get_local_id(0) + nd1_velD[0];
int i = get_group_id(1)* get_local_size(1) + get_local_id(1) + nd1_velD[6];

if ( j>jjy || i>nd1_velD[11]) return;

v1y(0,i,j)=v1y(1,i,j)+dzdt* 
(dyi1(1,j)*v1z(1,i,j-1)+dyi1(2,j)*v1z(1,i,j)+ 
  dyi1(3,j)*v1z(1,i,j+1)+dyi1(4,j)*v1z(1,i,j+2));
}  


/*
__kernel void vel_vxy_image_layer1 ( __global float* v1xM, __global float* v1yM, __global float* v1zM, __global int* nd1_velD,  __global float* dxi1M, __global float* dyi1M, __global float* dzh1M,  int nxtm1, int nytm1, int nxtop, int nytop, int nztop,  int neighb1, int neighb2, int neighb3, int neighb4) 
{
int i,j,k, iix, jjy, nnum;
float dzdt, dzdx, dzdy;
float ca=9.0/8;
dzdt=ca/dzh1(3,1);
//--Vx
iix=nd1_velD[12-1];
if(neighb1 < 0 || neighb2 < 0) {
i=1;
if(neighb2 < 0) {
iix=nxtop-2;
i=nxtm1;
}
dzdx=dzdt*dxi1(3,i)/ca;
for( j=nd1_velD[0]; j<=nd1_velD[5] ; j++) {
v1x(0,i,j)=v1x(1,i,j)+dzdx*(v1z(1,i+1,j)-v1z(1,i,j));
}
}
for( j=nd1_velD[0]; j<=nd1_velD[5]; j++) {
for( i=nd1_velD[6]; i <= iix; i++) {
v1x(0,i,j)=v1x(1,i,j)+dzdt* (dxi1(1,i)*v1z(1,i-1,j)+dxi1(2,i)*v1z(1,i,  j)+ 
  dxi1(3,i)*v1z(1,i+1,j)+dxi1(4,i)*v1z(1,i+2,j));
}
}
//--Vy
jjy=nd1_velD[5];
if(neighb3 < 0 || neighb4 < 0) {
j=1;
if(neighb4 < 0) {
jjy=nytop-2;
j=nytm1;
}
dzdy=dzdt*dyi1(3,j)/ca;
for( i=nd1_velD[6]; i<= nd1_velD[11]; i++) {
v1y(0,i,j)=v1y(1,i,j)+dzdy*(v1z(1,i,j+1)-v1z(1,i,j));
}
}
for(j=nd1_velD[0]; j<=jjy; j++) {
   for(i=nd1_velD[6]; i<=nd1_velD[11]; i++) {
  v1y(0,i,j)=v1y(1,i,j)+dzdt* 
  (dyi1(1,j)*v1z(1,i,j-1)+dyi1(2,j)*v1z(1,i,j)+ 
dyi1(3,j)*v1z(1,i,j+1)+dyi1(4,j)*v1z(1,i,j+2));
  }
  }
  }
  */
  __kernel void vel_vxy_image_layer_sdx( __global float* sdx1D, __global float* sdx2D, __global float* v1xM, __global float* v1yM,  int nxtop, int nytop, int nztop)
  { 
  int tid = get_group_id(0)*get_local_size(0) + get_local_id(0)+1; 
  if(tid > nytop) return;
  sdx1D[tid-1] = v1x(0,1,tid);
  sdx2D[tid-1] = v1y(0,1,tid);
  }

  __kernel void vel_vxy_image_layer_sdy( __global float* sdy1D, __global float* sdy2D, __global float* v1xM, __global float* v1yM, int nxtop, int nytop, int nztop)
  {
  int tid = get_group_id(0)*get_local_size(0) + get_local_id(0)+1; 
  if(tid > nxtop) return;
  sdy1D[tid-1] = v1x(0,tid,1);
  sdy2D[tid-1] = v1y(0,tid,1);
  }

  __kernel void vel_vxy_image_layer_rcx( __global float* rcx1D, __global float* rcx2D, __global float* v1xM, __global float* v1yM, int nxtop, int nytop, int nztop, int nx1p1)
  {
  int tid = get_group_id(0)*get_local_size(0) + get_local_id(0)+1; 
  if(tid > nytop) return;
  v1x(0,nx1p1,tid) = rcx1D[tid-1];
  v1y(0,nx1p1,tid) = rcx2D[tid-1]; 
  }

  __kernel void vel_vxy_image_layer_rcy( __global float* rcx1D, __global float* rcx2D, __global float* v1xM, __global float* v1yM, int nxtop, int nytop, int nztop, int ny1p1)
  {
  int tid = get_group_id(0)*get_local_size(0) + get_local_id(0)+1; 
  if(tid > nxtop) return;
  //printf("DEBUG::  rcy1D[%d]=%f  rcy2D[%d]=%f\n", tid-1, rcy1D[tid-1], tid-1,rcy2D[tid-1]);
  v1x(0,tid,ny1p1) = rcx1D[tid-1]; //rcy name for rcx
  v1y(0,tid,ny1p1) = rcx2D[tid-1];
  }

  #define index_xyz_source(i,j,k) index_xyz_sourceD[i-1 + ixsX*(j-1) + ixsX*ixsY*(k-1)]
  #define famp(i,j) fampD[i-1 + fampX*(j-1)]

  __kernel void vel_add_dcs (__global float* t1xxM, __global float* t1xyM, __global float* t1xzM, __global float* t1yyM,  __global float* t1yzM, __global float* t1zzM, __global float* t2xxM, __global float* t2yyM, __global float* t2xyM,  __global float* t2xzM, __global float* t2yzM, __global float * t2zzM,  int nfadd, __global int* index_xyz_sourceD, int ixsX, int ixsY, int ixsZ,   __global float* fampD, int fampX, int fampY,  __global float* ruptmD, int ruptmX, __global float* risetD, int risetX,   __global float* sparam2D, int sparam2X,  __global float* sutmArrD, int nzrg11, int nzrg12, int nzrg13, int nzrg14,   int nxtop, int nytop, int nztop,  int nxbtm, int nybtm, int nzbtm)
  {
  int i,j,k;
  float sutm;

  int kap = get_group_id(0)*get_local_size(0) + get_local_id(0)+1; 
  if(kap > nfadd) return;

  sutm = sutmArrD[kap -1];

  i=index_xyz_source(1,kap,1);
  j=index_xyz_source(2,kap,1);
  k=index_xyz_source(3,kap,1);
  if(k>nzrg11) {
  k=k-nztop;
  t2xx(k,i,j)=t2xx(k,i,j)+famp(kap,1)*sutm;
  t2yy(k,i,j)=t2yy(k,i,j)+famp(kap,2)*sutm;
  t2zz(k,i,j)=t2zz(k,i,j)+famp(kap,3)*sutm;
  }
  else {
  t1xx(k,i,j)=t1xx(k,i,j)+famp(kap,1)*sutm;
  t1yy(k,i,j)=t1yy(k,i,j)+famp(kap,2)*sutm;
  t1zz(k,i,j)=t1zz(k,i,j)+famp(kap,3)*sutm;
  }

  i=index_xyz_source(1,kap,2);
  j=index_xyz_source(2,kap,2);
  k=index_xyz_source(3,kap,2);
  if(k>nzrg12) {
  k=k-nztop;
  t2xy(k,i,j)=t2xy(k,i,j)+famp(kap,4)*sutm;
  } else {
  t1xy(k,i,j)=t1xy(k,i,j)+famp(kap,4)*sutm;
  }
  // Mxz
  i=index_xyz_source(1,kap,3);
  j=index_xyz_source(2,kap,3);
  k=index_xyz_source(3,kap,3);
  if(k>nzrg13){
  k=k-nztop;
  t2xz(k,i,j)=t2xz(k,i,j)+famp(kap,5)*sutm;
  }
  else {
  t1xz(k,i,j)=t1xz(k,i,j)+famp(kap,5)*sutm;
  }
  // Myz
  i=index_xyz_source(1,kap,4);
  j=index_xyz_source(2,kap,4);
  k=index_xyz_source(3,kap,4);
  if(k>nzrg14) {
  k=k-nztop;
  t2yz(k,i,j)=t2yz(k,i,j)+famp(kap,6)*sutm;
  } else {
  t1yz(k,i,j)=t1yz(k,i,j)+famp(kap,6)*sutm;
  } 
  }

  __kernel void stress_sdx41 (__global float* sdx41M, __global float* t1xxM, __global float* t1xyM, __global float* t1xzM, int nytop, int nztop, int nxtop)
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nytop) return;
  sdx41(k,j,1) = t1xx(k,1,j);
  sdx41(k,j,2) = t1xx(k,2,j);
  sdx41(k,j,3) = t1xy(k,1,j);
  sdx41(k,j,4) = t1xz(k,1,j);
  }

  __kernel void stress_sdx42 (__global float* sdx42M, __global float* t2xxM, __global float* t2xyM, __global float* t2xzM, int nybtm, int nzbtm, int nxbtm) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nybtm) return;
  sdx42(k,j,1) = t2xx(k,1,j);
  sdx42(k,j,2) = t2xx(k,2,j);
  sdx42(k,j,3) = t2xy(k,1,j);
  sdx42(k,j,4) = t2xz(k,1,j);
  }

  __kernel void stress_sdx51 (__global float* sdx51M, __global float* t1xxM, __global float* t1xyM, __global float* t1xzM, int nytop, int nztop, int nxtop, int nxtm1) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nytop) return;
  sdx51(k,j,1) = t1xx(k,nxtop,j);
  sdx51(k,j,2) = t1xy(k,nxtm1,j);
  sdx51(k,j,3) = t1xy(k,nxtop,j);
  sdx51(k,j,4) = t1xz(k,nxtm1,j);
  sdx51(k,j,5) = t1xz(k,nxtop,j);
  }

  __kernel void stress_sdx52 (__global float* sdx52M, __global float* t2xxM, __global float* t2xyM, __global float* t2xzM, int nybtm, int nzbtm, int nxbtm, int nxbm1) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nybtm) return;
  sdx52(k,j,1) = t2xx(k,nxbtm,j);
  sdx52(k,j,2) = t2xy(k,nxbm1,j);
  sdx52(k,j,3) = t2xy(k,nxbtm,j);
  sdx52(k,j,4) = t2xz(k,nxbm1,j);
  sdx52(k,j,5) = t2xz(k,nxbtm,j);
  }

  __kernel void stress_sdy41 (__global float* sdy41M, __global float* t1yyM, __global float* t1xyM, __global float* t1yzM, int nytop, int nztop, int nxtop) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nxtop) return;
  sdy41(k,j,1) = t1yy(k, j, 1);
  sdy41(k,j,2) = t1yy(k, j, 2);
  sdy41(k,j,3) = t1xy(k, j, 1);
  sdy41(k,j,4) = t1yz(k, j, 1);
  }

  __kernel void stress_sdy42 (__global float* sdy42M, __global float* t2yyM, __global float* t2xyM, __global float* t2yzM, int nybtm, int nzbtm, int nxbtm) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nxbtm) return;
  sdy42(k,j,1) = t2yy(k, j, 1);
  sdy42(k,j,2) = t2yy(k, j, 2);
  sdy42(k,j,3) = t2xy(k, j, 1);
  sdy42(k,j,4) = t2yz(k, j, 1);
  }

  __kernel void stress_sdy51 (__global float* sdy51M, __global float* t1yyM, __global float* t1xyM, __global float* t1yzM, int nytop, int nztop, int nxtop, int nytm1) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nxtop) return;
  sdy51(k,j,1) = t1yy(k,j, nytop);
  sdy51(k,j,2) = t1xy(k,j, nytm1);
  sdy51(k,j,3) = t1xy(k,j, nytop);
  sdy51(k,j,4) = t1yz(k,j, nytm1);
  sdy51(k,j,5) = t1yz(k,j, nytop);
  }

  __kernel void stress_sdy52 (__global float* sdy52M, __global float* t2yyM, __global float* t2xyM, __global float* t2yzM, int nybtm, int nzbtm, int nxbtm, int nybm1) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nxbtm) return;
  sdy52(k,j,1) = t2yy(k,j,nybtm);
  sdy52(k,j,2) = t2xy(k,j,nybm1);
  sdy52(k,j,3) = t2xy(k,j,nybtm);
  sdy52(k,j,4) = t2yz(k,j,nybm1);
  sdy52(k,j,5) = t2yz(k,j,nybtm);
  }


  __kernel void stress_rcx41 (__global float* rcx41M, __global float* t1xxM, __global float* t1xyM, __global float* t1xzM, int nytop, int nztop, int nxtop, int nx1p1, int nx1p2) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nytop) return;
  t1xx(k,nx1p1,j) = rcx41(k,j,1);
  t1xx(k,nx1p2,j) = rcx41(k,j,2);
  t1xy(k,nx1p1,j) = rcx41(k,j,3);
  t1xz(k,nx1p1,j) = rcx41(k,j,4);
  }

  __kernel void stress_rcx42 (__global float* rcx42M, __global float* t2xxM, __global float* t2xyM, __global float* t2xzM, int nybtm, int nzbtm, int nxbtm, int nx2p1, int nx2p2) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nybtm) return;
  t2xx(k,nx2p1,j) = rcx42(k,j,1);
  t2xx(k,nx2p2,j) = rcx42(k,j,2);
  t2xy(k,nx2p1,j) = rcx42(k,j,3);
  t2xz(k,nx2p1,j) = rcx42(k,j,4);
  }

  __kernel void stress_rcx51 (__global float* rcx51M, __global float* t1xxM, __global float* t1xyM, __global float* t1xzM, int nytop, int nztop, int nxtop) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nytop) return;
  t1xx(k,0,j) = rcx51(k,j,1) ;
  t1xy(k,-1,j) = rcx51(k,j,2) ;
  t1xy(k, 0,j) = rcx51(k,j,3) ;
  t1xz(k,-1,j) = rcx51(k,j,4) ;
  t1xz(k,0,j) = rcx51(k,j,5) ;
  }
  __kernel void stress_rcx52 (__global float* rcx52M, __global float* t2xxM, __global float* t2xyM, __global float* t2xzM, int nybtm, int nzbtm, int nxbtm) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nybtm) return;
  t2xx(k, 0 ,j) = rcx52(k,j,1) ;
  t2xy(k,-1 ,j) = rcx52(k,j,2) ;
  t2xy(k, 0 ,j) = rcx52(k,j,3) ;
  t2xz(k,-1 ,j) = rcx52(k,j,4) ;
  t2xz(k, 0 ,j) = rcx52(k,j,5) ;
  }

  __kernel void stress_rcy41 (__global float* rcy41M, __global float* t1yyM, __global float* t1xyM, __global float* t1yzM, int nytop, int nztop, int nxtop, int ny1p1, int ny1p2) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nxtop) return;
  t1yy(k, j, ny1p1) = rcy41(k,j,1) ;
  t1yy(k, j, ny1p2) = rcy41(k,j,2) ;
  t1xy(k, j, ny1p1) = rcy41(k,j,3) ;
  t1yz(k, j, ny1p1) = rcy41(k,j,4) ;
  }

  __kernel void stress_rcy42 (__global float* rcy42M, __global float* t2yyM, __global float* t2xyM, __global float* t2yzM, int nybtm, int nzbtm, int nxbtm, int ny2p1, int ny2p2) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nxbtm) return;
  t2yy(k, j, ny2p1) = rcy42(k,j,1);
  t2yy(k, j, ny2p2) = rcy42(k,j,2);
  t2xy(k, j, ny2p1) = rcy42(k,j,3);
  t2yz(k, j, ny2p1) = rcy42(k,j,4);
  }

  __kernel void stress_rcy51 (__global float* rcy51M, __global float* t1yyM, __global float* t1xyM, __global float* t1yzM, int nytop, int nztop, int nxtop) 
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nztop || j >nxtop) return;
  t1yy(k,j, 0) = rcy51(k,j,1);
  t1xy(k,j, -1) = rcy51(k,j,2);
  t1xy(k,j, 0) = rcy51(k,j,3);
  t1yz(k,j, -1) = rcy51(k,j,4);
  t1yz(k,j, 0) = rcy51(k,j,5);
  }

  __kernel void stress_rcy52 (__global float* rcy52M, __global float* t2yyM, __global float* t2xyM, __global float* t2yzM, int nybtm, int nzbtm, int nxbtm)
  {
  int k = get_group_id(0)*get_local_size(0) + get_local_id(0) + 1;
  int j = get_group_id(1)*get_local_size(1) + get_local_id(1) + 1;
  if(k>nzbtm || j >nxbtm) return;
  t2yy(k,j,0) = rcy52(k,j,1);
  t2xy(k,j,-1) = rcy52(k,j,2);
  t2xy(k,j,0) = rcy52(k,j,3);
  t2yz(k,j,-1) = rcy52(k,j,4);
  t2yz(k,j,0) = rcy52(k,j,5);
  }


  __kernel void stress_interp1 (int ntx1, int nz1p1,int nxbtm, int nybtm, int nzbtm,int  nxtop, int nytop, int nztop,__global float* t1xzM, __global float* t2xzM )
  {
  int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
  int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);
  int jj = tx+1;
  int ii = ty+1;
  if (jj>nybtm || ii > ntx1) return;

  int j = -2 + 3*jj;
  int i = -1 + 3*ii;
  t2xz(0,ii,jj)=t1xz(nztop,i,j);
  t2xz(1,ii,jj)=t1xz(nz1p1,i,j);
  }

  __kernel void stress_interp2 (int nty1, int nz1p1,int nxbtm, int nybtm, int nzbtm,int  nxtop, int nytop, int nztop,__global float* t1yzM, __global float* t2yzM )
  {
  int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
  int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);
  int jj = tx+1;
  int ii = ty+1;
  if(jj>nty1 || ii>nxbtm) return;
  int j = -1 + 3*jj;
  int i = -2 + 3*ii;
  t2yz(0,ii,jj)=t1yz(nztop,i,j);
  t2yz(1,ii,jj)=t1yz(nz1p1,i,j);
  } 

  __kernel void stress_interp3 ( int nxbtm, int nybtm, int nzbtm, int  nxtop, int nytop, int nztop, __global float* t1zzM, __global float* t2zzM )
  {
  int tx = get_group_id(0)*get_local_size(0) + get_local_id(0);
  int ty = get_group_id(1)*get_local_size(1) + get_local_id(1);
  int jj = tx+1;
  int ii = ty+1;
  if(jj>nybtm || ii>nxbtm) return;
  int j = -2 + 3*jj;
  int i = -2 + 3*ii;
  t2zz(0,ii,jj)=t1zz(nztop,i,j);
  } 

  __kernel void stress_interp_stress (__global float* t1xzM, __global float* t1yzM, __global float* t1zzM,   __global float* t2xzM, __global float* t2yzM, __global float* t2zzM,   int neighb1, int neighb2, int neighb3, int neighb4,   int nxbm1, int nybm1,    int nxbtm, int nybtm, int nzbtm,   int nxtop, int nytop, int nztop, int nz1p1) 
{ 
  int i,j,ii,jj,ntx1,nty1;

  ntx1=nxbtm;
  if(neighb2 < 0) ntx1=nxbm1;
  nty1=nybtm;
  if(neighb4 < 0) nty1=nybm1;

  j=-2;
  for( jj=1; jj<=nybtm;jj++) {
 j=j+3;
 i=-1;
 for( ii=1; ii<=ntx1; ii++) {
i=i+3;
t2xz(0,ii,jj)=t1xz(nztop,i,j);
t2xz(1,ii,jj)=t1xz(nz1p1,i,j);
}
}

j=-1;
for( jj=1; jj<=nty1; jj++) {
   j=j+3;
   i=-2;
   for( ii=1; ii<=nxbtm; ii++) {
  i=i+3;
  t2yz(0,ii,jj)=t1yz(nztop,i,j);
  t2yz(1,ii,jj)=t1yz(nz1p1,i,j);
  }
  }

  j=-2;
  for( jj=1; jj<=nybtm; jj++) {
 j=j+3;
 i=-2;
 for( ii=1; ii<= nxbtm; ii++) {
i=i+3;
t2zz(0,ii,jj)=t1zz(nztop,i,j);
}
}
}

__kernel void vel1_dummy (int nxtop, int nztop, int nytop,  __global float* v1xM, __global float* v1yM, __global float* v1zM) 
{
int i = get_group_id(0)*get_local_size(0) + get_local_id(0) ;    
int j = get_group_id(1)*get_local_size(1) + get_local_id(1) - 1 ;

if( i>nztop+2-1 || j>nxtop+3-2 ) return;

for (int k=0 ; k<nytop+3; k++) {
 v1x(i,j,k) += (i+j+k);             
 v1y(i,j+1,k-1) += (i+j+k);             
 v1z(i,j+1,k) += (i+j+k);             
 }
 }

__kernel void vel2_dummy (int nxbtm, int nzbtm, int nybtm, __global float* v2xM, __global float* v2yM, __global float* v2zM) 
{
 int i = get_group_id(0)*get_local_size(0) + get_local_id(0) ;    
 int j = get_group_id(1)*get_local_size(1) + get_local_id(1) - 1 ;

 if( i>nzbtm+1-1 || j>nxbtm+3-2 ) return;

 for (int k=0 ; k<nybtm+3; k++) {
  v2x(i,j,k) += (i+j+k);             
  v2y(i,j+1,k-1) += (i+j+k);             
  v2z(i,j+1,k) += (i+j+k);             
  }
}


#ifdef __cplusplus
extern "C" {
#endif



#ifdef __cplusplus
}
#endif

