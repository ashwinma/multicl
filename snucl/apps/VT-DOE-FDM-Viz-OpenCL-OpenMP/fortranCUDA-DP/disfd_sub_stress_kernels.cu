//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu
// of USBR.
//
//-----------------------------------------------------------------------------

#include <stdio.h>
#include "switches.h"
#include "disfd_macros.h"

#if USE_Optimized_stress_norm_xy_IC ==1 
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
					   double *v1zM)
{
	int i,j,k,jkq,kodd,inod,irw;
	double sxx,syy,szz,sxy,qxx,qyy,qzz,qxy,cusxy,sss,cl,sm2,pm,et,et1,wtp,wts;

    int offset_k = nd1_tyy[12];
    int offset_i = nd1_tyy[8];
    k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
    i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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


#elif USE_Optimized_stress_norm_xy_IC == 0 
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
					   double *v1zM)
{
	int i,j,k,jkq,kodd,inod,irw;
	double sxx,syy,szz,sxy,qxx,qyy,qzz,qxy,cusxy,sss,cl,sm2,pm,et,et1,wtp,wts;

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
#endif
//-----------------------------------------------------------------------------
#if USE_Optimized_stress_xz_yz_IC == 1
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
					  double *qt1yzM)
// Compute stress-XZand YZ component in Region I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
//	real, parameter:: tfr1=-577./528./ca,tfr2=201./176./ca, &
//                   tfr3=-9./176./ca,  tfr4=1./528./ca
{
//	double tfr1 = -577./528./ca;
//	double tfr2 = 201./176./ca;
//	double tfr3 = -9./176./ca;
//	double tfr4=1./528./ca;

	int i,j,k,kodd,inod,jkq,irw;
	double dvzx,dvzy,dvxz,dvyz,sm,cusxz,cusyz,et,et1,dmws,qxz,qyz;

    int offset_k = nd1_tyz[12];
    int offset_i = nd1_tyz[8];
    int offset_j = nd1_tyz[2];
    k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
    i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

    if (k > nd1_tyz[17] || i > nd1_tyz[9])
    {
        return;
    }

/*
#define BLOCK_SIZE 8
 	__shared__ double dxi1_S[BLOCK_SIZE][4];
	__shared__ double dyi1_S[72][4];
	__shared__ double dzh1_S[BLOCK_SIZE][4];

        if(threadIdx.x == 0 && threadIdx.y ==0 ) {
            int c_k = blockIdx.x * blockDim.x + offset_k;
            int c_i = blockIdx.y * blockDim.y + offset_i;
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
/*        if (threadIdx.y == 0) {
            for(int l =0; l<4; l++) {
                //dzi1_S[(k -offset_k)%BLOCK_SIZE][l] = dzi1(l+1, k);
                dzh1_S[(k -offset_k)%BLOCK_SIZE][l] = dzh1(l+1, k);               
            }

            if (threadIdx.x == 0 ) {
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
#elif USE_Optimized_stress_xz_yz_IC == 0
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
					  double *qt1yzM)
// Compute stress-XZand YZ component in Region I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
//	real, parameter:: tfr1=-577./528./ca,tfr2=201./176./ca, &
//                   tfr3=-9./176./ca,  tfr4=1./528./ca
{
//	double tfr1 = -577./528./ca;
//	double tfr2 = 201./176./ca;
//	double tfr3 = -9./176./ca;
//	double tfr4=1./528./ca;

	int i,j,k,kodd,inod,jkq,irw;
	double dvzx,dvzy,dvxz,dvyz,sm,cusxz,cusyz,et,et1,dmws,qxz,qyz;

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
#endif

__global__ void stress_resetVars(int ny1p1,
					  int nx1p1,
					  int nxtop,
					  int nytop,
					  int nztop,
					  double *t1xzM,
					  double *t1yzM)
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

//----------------------------------------------------------------------------------
#if USE_Optimized_stress_norm_PmlX_IC == 1
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
						 double *qt1yy_pxM)
// Compute the velocity of PML-x-I region
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
// real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;
	int nti;
	//if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return;

    int offset_k = nd1_tyy[12];
    int offset_j = nd1_tyy[0];
    k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
    j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

    if (j > nd1_tyy[5] || k> nd1_tyy[17])
    {
        return;
    }
    nti = (lbx1 - lbx0 + 1) * mw1_pml + lbx0;
	
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


#elif USE_Optimized_stress_norm_PmlX_IC == 0
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
						 double *qt1yy_pxM)
// Compute the velocity of PML-x-I region
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
// real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;
	int nti;
	//if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return;

    j = blockIdx.x * blockDim.x + threadIdx.x + nd1_tyy[0];
    lb = blockIdx.y * blockDim.y + threadIdx.y + lbx0;

    if (j > nd1_tyy[5] || lb > lbx1)
    {
        return;
    }
	nti = (lbx1 - lbx0 + 1) * mw1_pml + lbx0;
	
// 	for (j=nd1_tyy[0]; j <= nd1_tyy[5]; j++)
// 	//do j=nd1_tyy(1),nd1_tyy(6)
//	{
	kodd=2*((j+nyb1)&1)+1;
	ib=0;
	for (k = lbx0; k < lb; k++)
	{
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
#endif 

#if USE_Optimized_stress_norm_PmlY_IC == 1
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
						 double *v1zM)
// Compute the velocity of PML-x-I region
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
// real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

	//if(lby0>lby1) return;

        int offset_k = nd1_tyy[12];
        int offset_i = nd1_tyy[6];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;
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

#elif USE_Optimized_stress_norm_PmlY_IC == 0
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
						 double *v1zM)
// Compute the velocity of PML-x-I region
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
// real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

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
	for (k = 0; k < lb; k++)
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
#endif

#if USE_Optimized_stress_xy_PmlX_IC == 1
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
					   double *v1yM)
// Compute the Stress-xy at region of PML-x-I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
// real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	nth = (lbx1 - lbx0 + 1) * mw1_pml + 1 - lbx0;
    
        int offset_k = nd1_txy[12];
        int offset_j = nd1_txy[0];

	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

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

#elif USE_Optimized_stress_xy_PmlX_IC == 0
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
					   double *v1yM)
// Compute the Stress-xy at region of PML-x-I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
// real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;
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

#endif

#if USE_Optimized_stress_xy_PmlY_IC == 1
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
					   double *v1yM)
//Compute the Stress-xy at region of PML-y-I
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxy,cusyx,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxy,cusyx,qxy,rth,damp2,damp1,sm,dmws,et,et1;

	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
        int offset_k = nd1_txy[12];
        int limit_k = nd1_txy[17];
        int offset_i = nd1_txy[6];
        int limit_i = nd1_txy[11];

	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;
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
#elif USE_Optimized_stress_xy_PmlY_IC == 0
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
					   double *v1yM)
//Compute the Stress-xy at region of PML-y-I
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxy,cusyx,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxy,cusyx,qxy,rth,damp2,damp1,sm,dmws,et,et1;

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
			kb++;
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
#endif

#if USE_Optimized_stress_xz_PmlX_IC == 1
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
					   double *v1zM)
//Compute the stress-xz at PML-x-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,dvxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxz,cusxz,dvxz,qxz,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	//if (lbx[0] > lbx[1]) return;
	//if ( lbx(1)>lbx(2) ) return
	nth = (lbx1 - lbx0 + 1) * mw1_pml + 1 - lbx0;

        int offset_k = nd1_txz[12];
        int offset_j = nd1_txz[0];
        int limit_k = nd1_txz[17];
        int limit_j = nd1_txz[5];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

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


#elif USE_Optimized_stress_xz_PmlX_IC == 0
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
					   double *v1zM)
//Compute the stress-xz at PML-x-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,dvxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxz,cusxz,dvxz,qxz,rth,damp2,damp1,sm,dmws,et,et1;
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
#endif

#if USE_Optimized_stress_xz_PmlY_IC == 1
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
					   double *v1zM)
//Compute the stress-xz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusxz,dvxz,dvzx,qxz,sm,dmws,et,et1;

	//if (lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
        
        int offset_k = nd1_txz[12];
        int offset_i = nd1_txz[8];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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

#elif USE_Optimized_stress_xz_PmlY_IC == 0
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
					   double *v1zM)
//Compute the stress-xz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusxz,dvxz,dvzx,qxz,sm,dmws,et,et1;

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
#endif

#if USE_Optimized_stress_yz_PmlX_IC == 1
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
					   double *v1zM)
//Compute the stress-yz at PML-x-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusyz,dvyz,dvzy,qyz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusyz,dvyz,dvzy,qyz,sm,dmws,et,et1;

	//if(lbx[0] > lbx[1]) return;
	//if(lbx(1)>lbx(2) ) return

        int offset_k = nd1_tyz[12];
        int offset_j = nd1_tyz[2];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

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


#elif USE_Optimized_stress_yz_PmlX_IC == 0
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
					   double *v1zM)
//Compute the stress-yz at PML-x-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusyz,dvyz,dvzy,qyz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusyz,dvyz,dvzy,qyz,sm,dmws,et,et1;

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
#endif

#if USE_Optimized_stress_yz_PmlY_IC == 1
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
					   double *v1zM)
//Compute the stress-yz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1;

        int offset_k = nd1_tyz[12];
        int offset_i  = nd1_tyz[6];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_yz_PmlY_IC == 0
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
					   double *v1zM)
//Compute the stress-yz at PML-y-I region
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1;

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
#endif

#if USE_Optimized_stress_norm_xy_II == 1
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
					   double *v2zM)
// Compute stress-Norm and XY component in Region II
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,kodd,inod,jkq,irw
// real:: sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy, &
//        cl,sm2,et,et1,dmws,pm,wtp,wts
{
	int i,j,k,kodd,inod,jkq,irw;
	double sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy,cl,sm2,et,et1,dmws,pm,wtp,wts;

        int offset_k = nd2_tyy[12];
        int offset_i = nd2_tyy[8];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_norm_xy_II == 0
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
					   double *v2zM)
// Compute stress-Norm and XY component in Region II
// use grid_node_comm
// use wave_field_comm
// implicit NONE
// integer:: i,j,k,kodd,inod,jkq,irw
// real:: sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy, &
//        cl,sm2,et,et1,dmws,pm,wtp,wts
{
	int i,j,k,kodd,inod,jkq,irw;
	double sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy,cl,sm2,et,et1,dmws,pm,wtp,wts;

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
#endif
//call stress_xz_yz_II
#if USE_Optimized_stress_xz_yz_IIC == 1
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
					  double *v2zM)
//Compute stress-XZ and YZ component in the Region II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,kodd,inod,jkq,irw
//real:: qxz,qyz,cusxz,cusyz,sm,et,et1,dmws
{
	int i,j,k,kodd,inod,jkq,irw;
	double qxz,qyz,cusxz,cusyz,sm,et,et1,dmws;

        int offset_k = nd2_tyz[12];
        int offset_i = nd2_tyz[8];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_xz_yz_IIC == 0
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
					  double *v2zM)
//Compute stress-XZ and YZ component in the Region II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,kodd,inod,jkq,irw
//real:: qxz,qyz,cusxz,cusyz,sm,et,et1,dmws
{
	int i,j,k,kodd,inod,jkq,irw;
	double qxz,qyz,cusxz,cusyz,sm,et,et1,dmws;

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
#endif
//call stress_norm_PmlX_II

#if USE_Optimized_stress_norm_PmlX_IIC == 1
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
						  double *v2zM)
//Compute the Stress-norm at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;
	int nti;

	//if(lbx[0] > lbx[1]) return;
	//if( lbx(1)>lbx(2) ) return
	nti = (lbx1 - lbx0 + 1) * mw2_pml + lbx1;
        
        int offset_k = nd2_tyy[12];
        int offset_j = nd2_tyy[0];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

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
#elif USE_Optimized_stress_norm_PmlX_IIC == 0
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
						  double *v2zM)
//Compute the Stress-norm at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;
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
#endif

#if USE_Optimized_stress_norm_PmlY_II == 1
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
						 double *v2zM)
//Compute the stress-norm at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

        int offset_i = nd2_tyy[6];
        int offset_k = nd2_tyy[12];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_norm_PmlY_II == 0
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
                                     double *v2zM)
//Compute the stress-norm at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

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
#endif

#if USE_Optimized_stress_norm_PmlZ_IIC == 1
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
						  double *v2zM)
//Compute the stress-norm at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

        int offset_k = nd2_tyy[16];
        int offset_i = nd2_tyy[6];
        
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_norm_PmlZ_IIC == 0
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
						  double *v2zM)
//Compute the stress-norm at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
//       damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	double taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts;

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
#endif

#if USE_Optimized_stress_xy_PmlX_IIC == 1
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
						double *v2yM)
//Compute the Stress-xy at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	//if(lbx[0] > lbx[1]) return;
	nth = (lbx1 - lbx0 + 1) * mw2_pml + 1 - lbx0;
	//nth = (lbx(2) - lbx(1) + 1) * mw2_pml + 1 - lbx(1)

        int offset_k = nd2_txy[12];
        int offset_j = nd2_txy[0];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

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
#elif USE_Optimized_stress_xy_PmlX_IIC == 0
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
						double *v2yM)
//Compute the Stress-xy at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;
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
#endif

#if USE_Optimized_stress_xy_PmlY_IIC == 1
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
						double *v2yM)
//Compute the Stress-xy at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;

	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return

        int offset_k = nd2_txy[12];
        int offset_i = nd2_txy[6];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_xy_PmlY_IIC == 0
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
						double *v2yM)
//Compute the Stress-xy at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1;

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
#endif

#if USE_Optimized_stress_xy_PmlZ_II == 1
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
					   double *v2yM)
//Compute the Stress-xy at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusxy,qxy,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusxy,qxy,sm,dmws,et,et1;

        int offset_k = nd2_txy[16];
        int offset_i = nd2_txy[8];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_xy_PmlZ_II == 0
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
					   double *v2yM)
//Compute the Stress-xy at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusxy,qxy,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusxy,qxy,sm,dmws,et,et1;

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
#endif

#if USE_Optimized_stress_xz_PmlX_IIC == 1
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
						double *v2zM)
//Compute the stress-xz at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxz,cusxz,qxz,rth,damp2,damp1,sm,dmws,et,et1;
	int nth;

	//if(lbx[0] > lbx[1]) return;
	nth = (lbx1 - lbx0 + 1) * mw2_pml + 1 - lbx0;

        int offset_k = nd2_txz[12];
        int offset_j = nd2_txz[0];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

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
#elif USE_Optimized_stress_xz_PmlX_IIC == 0
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
						double *v2zM)
//Compute the stress-xz at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,ib,kb,kodd,jkq,inod,irw;
	double taoxz,cusxz,qxz,rth,damp2,damp1,sm,dmws,et,et1;
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
#endif

#if USE_Optimized_stress_xz_PmlY_IIC == 1
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
					   double *qt2xzM)
//Compute the stress-xz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: dvxz,dvzx,cusxz,qxz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double dvxz,dvzx,cusxz,qxz,sm,dmws,et,et1;
	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return

        int offset_i = nd2_txz[8];
        int offset_k = nd2_txz[12];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_xz_PmlY_IIC == 0
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
					   double *qt2xzM)
//Compute the stress-xz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: dvxz,dvzx,cusxz,qxz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double dvxz,dvzx,cusxz,qxz,sm,dmws,et,et1;
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
#endif

#if USE_Optimized_stress_xz_PmlZ_IIC == 1
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
						double *v2zM)
//Compute the stress-xz at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	double taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1;

        int offset_k = nd2_txz[16];
        int offset_i = nd2_txz[6];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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

#elif USE_Optimized_stress_xz_PmlZ_IIC == 0
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
						double *v2zM)
//Compute the stress-xz at region of PML-z-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	double taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1;
		
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
#endif

//call stress_yz_PmlX_II
#if USE_Optimized_stress_yz_PmlX_IIC == 1
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
						double *v2zM)
//Compute the stress-yz at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusyz,qyz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusyz,qyz,sm,dmws,et,et1;

	//if(lbx[0] > lbx[1]) return;
	//if( lbx(1)>lbx(2) ) return

        int offset_k = nd2_tyz[12];
        int offset_j = nd2_tyz[2];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

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
#elif USE_Optimized_stress_yz_PmlX_IIC == 0
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
						double *v2zM)
//Compute the stress-yz at region of PML-x-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kodd,jkq,inod,irw
//real:: cusyz,qyz,sm,dmws,et,et1
{
	int i,j,k,lb,kodd,jkq,inod,irw;
	double cusyz,qyz,sm,dmws,et,et1;

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
#endif

//call stress_yz_PmlY_II
#if USE_Optimized_stress_yz_PmlY_IIC == 1
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
						double *v2zM)
//Compute the stress-yz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoyz,cusyz,qyz,rth,damp2,damp1,sm,dmws,et,et1;

	//if(lby[0] > lby[1]) return;
	//if( lby(1)>lby(2) ) return
        int offset_k = nd2_tyz[12];
        int offset_i = nd2_tyz[6];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#elif USE_Optimized_stress_yz_PmlY_IIC == 0
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
						double *v2zM)
//Compute the stress-yz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,jb,kb,kodd,jkq,inod,irw;
	double taoyz,cusyz,qyz,rth,damp2,damp1,sm,dmws,et,et1;

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
#endif
//call stress_yz_PmlZ_II
#if USE_Optimized_stress_yz_PmlZ_IIC == 1
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
						double *v2zM)
//Compute the stress-yz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	double taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1;

        int offset_k = nd2_tyz[16];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_tyz[6];

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
#elif USE_Optimized_stress_yz_PmlZ_IIC == 0
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
						double *v2zM)
//Compute the stress-yz at region of PML-y-II
//use grid_node_comm
//use wave_field_comm
//implicit NONE
//integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
//real:: taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1
{
	int i,j,k,lb,kb,kodd,jkq,inod,irw;
	double taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1;

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
#endif

