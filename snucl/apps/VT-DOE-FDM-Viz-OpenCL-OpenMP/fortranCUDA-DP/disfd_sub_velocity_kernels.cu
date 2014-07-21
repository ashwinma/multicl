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

__device__ void print_DebugInt(int *a, int size)
{
	int i;
	for (i = 0; i < size; i++)
	printf("a[%d] = %d\n", i, a[i]);
}

__device__ void print_Debugdouble(double *a, int size)
{
	int i;
	for (i = 0; i < size; i++)
	printf("a[%d] = %f\n", i, a[i]);
}


#if USE_Optimized_velocity_inner_IC == 1
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
					  double *v1zM)


{
	int i, j, k, k3;
	double dtxz, dtyz, dtzz;
	int offset_j = nd1_vel[2];
	int offset_k = 1;
	k = blockIdx.x * blockDim.x + threadIdx.x ; // skipping the offset_k to take care of k=nztop case
	int k_bak = k; // backing up k
	int offset_i = nd1_vel[8];
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

        if (k > nztm1 || i> nd1_vel[9]){
		return;
	}
/*
#define BLOCK_SIZE 8
	__shared__ double dxi1_S[BLOCK_SIZE][4], dxh1_S[BLOCK_SIZE][4];
	__shared__ double dyi1_S[72][4], dyh1_S[72][4];
	__shared__ double dzi1_S[BLOCK_SIZE][4], dzh1_S[BLOCK_SIZE][4];

        if(threadIdx.x == 0) {
            for (int l = 0; l<4; l++){
               dxi1_S[(i-offset_i)%BLOCK_SIZE][l] = dxi1(l+1,i);
               dxh1_S[(i-offset_i)%BLOCK_SIZE][l] = dxh1(l+1,i);
            }
        }
        if (threadIdx.y == 0) {
            for(int l =0; l<4; l++) {
                dzi1_S[(k -offset_k)%BLOCK_SIZE][l] = dzi1(l+1, k);
                dzh1_S[(k -offset_k)%BLOCK_SIZE][l] = dzh1(l+1, k);               
            }

            if (threadIdx.x == 0 ) {
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
		//}
	//}

 	return;
}

#elif USE_Optimized_velocity_inner_IC == 0
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
					  double *v1zM)


{
	int i, j, k, k3;
	double dtxz, dtyz, dtzz;
	j = blockIdx.x * blockDim.x + threadIdx.x + nd1_vel[2];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd1_vel[8];
/*
	if (j == nd1_vel[2] && i == nd1_vel[8])
	{
		printf("nxtop = %d, nytop = %d, nztop = %d\n", nxtop, nytop, nztop);
		printf("ca = %f\n", ca);
		printf("nztm1 = %d\n", nztm1);
		for (k = 0; k < 13; k++)
			printf("rho[%d] = %f\n", k, rhoM[k]);
		print_Debugdouble(rhoM, 13);
	}
*/

	if (j > nd1_vel[3] || i > nd1_vel[9])
	{
		return;
	}

	//for (j = nd1_vel(3); j <= nd1_vel(4); j++)
	//for (j = nd1_vel[2]; j <= nd1_vel[3]; j++)
	//{
		//for (i = nd1_vel(9); i <= nd1_vel(10); i++)
		//for (i = nd1_vel[8]; i <= nd1_vel[9]; i++)
		//{
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

//		if (j == 38 && i == 38)
//		printf("j = %d, i = %d, dtxz = %f, dtyz = %f, dtzz = %f\n", j, i, dtxz, dtyz, dtzz);
/*
if (((k) + (nztop + 2) * ((i)  + (j) * (nxtop + 3))) == 235145)
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

//		v1x(k, i, j) = 1.0;
//		v1y(k, i, j) = 1.0;
//		v1z(k, i, j) = 1.0;
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


//		v1x(k, i, j) = 1.0;
//		v1y(k, i, j) = 1.0;
//		v1z(k, i, j) = 1.0;
//		if (j == 38 && i%10 ==0)
//		{
//	
//		printf("k:%d, i:%d, j:%d, rho = %f, rho = %f,id1 = %d, id2 = %d, d1 = %f, d2=%f,d3=%f,d4=%f,d5=%f,d6=%f\n", 
//				k, i, j, rho(idmat1(k,i,j)),rho(idmat1(k-1,i,j)), idmat1(k,i,j), idmat1(k-1,i,j),
//			dxh1(1,i)*t1xz(k,i-2,j)+dxh1(2,i)*t1xz(k,i-1,j),
//				dxh1(3,i)*t1xz(k,i,  j)+dxh1(4,i)*t1xz(k,i+1,j),
//				dyh1(1,j)*t1yz(k,i,j-2)+dyh1(2,j)*t1yz(k,i,j-1),
//				dyh1(3,j)*t1yz(k,i,j  )+dyh1(4,j)*t1yz(k,i,j+1),
//				dzh1(1,k)*t1zz(k-2,i,j)+dzh1(2,k)*t1zz(k-1,i,j),
//				dzh1(3,k)*t1zz(k  ,i,j)+dzh1(4,k)*t1zz(k+1,i,j));
//                      	printf("k = %d, i = %d, j = %d, v1x = %f, v1y = %f, v1z = %f\n", 
//				k, i, j, v1x(k, i, j), v1y(k, i, j), v1z(k, i, j));
//		}
	}

		//}
	//}

 	return;
}
#endif

//-----------------------------------------------------------------------
#if USE_Optimized_velocity_inner_IIC == 1
__global__ void velocity_inner_IIC(double ca,
					   int	 *nd2_vel,
					   double *rhoM,
					   double *dxi2M,
					   double *dyi2M,
					   double *dzi2M,
					   double *dxh2M,
					   double *dyh2M,
					   double *dzh2M,
					   int 	 *idmat2M,
					   double *t2xxM,
					   double *t2xyM,
					   double *t2xzM,
					   double *t2yyM,
					   double *t2yzM,
					   double *t2zzM,
					   int   nxbtm,	//dimension #s
					   int   nybtm,
					   int   nzbtm,
					   double *v2xM,		//output
					   double *v2yM,
					   double *v2zM)
{
	int i, j, k, k_bak;

	//j = blockIdx.x * blockDim.x + threadIdx.x + nd2_vel[2];
        int offset_k = 2, offset_i = nd2_vel[8];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
        k_bak = k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
        __shared__ double dxi2_S[4][BLOCK_SIZE], dxh2_S[4][BLOCK_SIZE];
        __shared__ double dyi2_S[4][15],dyh2_S[4][15];
        __shared__ double dzi2_S[4][BLOCK_SIZE], dzh2_S[4][BLOCK_SIZE];

        int offset_j = nd2_vel[2];
        if (threadIdx.x == 0) {
            //printf("BLOCK:(%d,%d): dxi2_S[0][%d ] = dxi2(%d,%d)\n",blockIdx.x, blockIdx.y, 0,(i-offset_i)%8,  1, i );
            dxi2_S[0][(i-offset_i)%BLOCK_SIZE] = dxi2(1,i);
            dxh2_S[0][(i-offset_i)%BLOCK_SIZE] = dxh2(1,i);
            dxi2_S[1][(i-offset_i)%BLOCK_SIZE] = dxi2(2,i);
            dxh2_S[1][(i-offset_i)%BLOCK_SIZE] = dxh2(2,i);
            dxi2_S[2][(i-offset_i)%BLOCK_SIZE] = dxi2(3,i);
            dxh2_S[2][(i-offset_i)%BLOCK_SIZE] = dxh2(3,i);
            dxi2_S[3][(i-offset_i)%BLOCK_SIZE] = dxi2(4,i);
            dxh2_S[3][(i-offset_i)%BLOCK_SIZE] = dxh2(4,i);
        }
        if(threadIdx.y == 0) {
            for (int l = 0; l<4; l++){
                    //printf("BLOCK:(%d,%d): dzi2_S[%d][(%d-%d)\% 8 ] = dzi2(%d+1,%d)\n",blockIdx.x, blockIdx.y, l,k,offset_k,  l, k );
                    //printf("BLOCK:(%d,%d): dzi2_S[%d][(%d ] = dzi2(%d,%d)\n",blockIdx.x, blockIdx.y, l,(k-offset_k)%8,  l+1, k );
                    dzi2_S[l][(k -offset_k)%BLOCK_SIZE] = dzi2(l+1, k);
                    dzh2_S[l][(k -offset_k)%BLOCK_SIZE] = dzh2(l+1, k);
            }

            if (threadIdx.x == 0) {
                for (j = nd2_vel[2] ; j <= nd2_vel[3]; j++) 
                {
                   for (int l=0; l<4 ; l++) { 
                    //printf("Block:(%d,%d) dyi2_S[%d][(%d-%d)%15 ] = dyi2(%d+1,%d)\n",blockIdx.x, blockIdx.y,l,j,offset_j, l, j );
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

#elif USE_Optimized_velocity_inner_IIC == 0
__global__ void velocity_inner_IIC(double ca,
					   int	 *nd2_vel,
					   double *rhoM,
					   double *dxi2M,
					   double *dyi2M,
					   double *dzi2M,
					   double *dxh2M,
					   double *dyh2M,
					   double *dzh2M,
					   int 	 *idmat2M,
					   double *t2xxM,
					   double *t2xyM,
					   double *t2xzM,
					   double *t2yyM,
					   double *t2yzM,
					   double *t2zzM,
					   int   nxbtm,	//dimension #s
					   int   nybtm,
					   int   nzbtm,
					   double *v2xM,		//output
					   double *v2yM,
					   double *v2zM)
{
	int i, j, k;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_vel[2];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_vel[8];
/*
	if (j == nd2_vel[2] && i == nd2_vel[8])
	{
		for (k = 0; k < 18; k++)
		{
			printf("nd2_vel[%d] = %d\n", k, nd2_vel[k]);
		}
	}
*/
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
	
//		if (j == nd2_vel[2] && i == nd2_vel[8])
//			printf("k:%d\n", k);
//		if ((k) + (nzbtm + 1) + ((i) + (nxbtm + 3) * (j)) == 1287)
//			printf("v2z = %f, %d, k:%d, i:%d, j:%d, rho = %f, rho = %f, id1 = %d, id2 = %d, d1 = %f, d2=%f,d3=%f,d4=%f,d5=%f,d6=%f\n", 
//				v2z(k,i,j), (k) + (nzbtm + 1) + ((i) + (nxbtm + 3) * (j)), k, i, j, rho(idmat2(k,i,j)), rho(idmat2(k-1,i,j)), idmat2(k,i,j), idmat2(k-1,i,j), 
//				dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j),
//				dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j),
//				dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1),
//				dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1),
//				dzh2(1,k)*t2zz(k-2,i,j)+dzh2(2,k)*t2zz(k-1,i,j),
//				dzh2(3,k)*t2zz(k  ,i,j)+dzh2(4,k)*t2zz(k+1,i,j));
	}
		//}
	//}
	return;
}
#endif

//-----------------------------------------------------------------------

#if USE_Optimized_vel_PmlX_IC == 1
__global__ void vel_PmlX_IC(double ca,
				int   lbx0,
				int   lbx1,
				int	  *nd1_vel,
				double *rhoM,
				double *drvh1M,
				double *drti1M,
				double *damp1_xM,
				int	  *idmat1M,
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
				int   mw1_pml1,	//dimension #
			    int   mw1_pml,
			    int   nxtop,
			    int   nytop,
			    int   nztop,
				double *v1xM,		//output
				double *v1yM,
				double *v1zM,
				double *v1x_pxM,
				double *v1y_pxM,
				double *v1z_pxM)
{
// !Compute the velocities in region of PML-x-I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
	int i,j,k,lb,ib,kb;
	double rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
        vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz,dtxy,dtyy,dtzy;

        int offset_i = nd1_vel[6+4*lb];
	int offset_k = 1;
	k  = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	int offset_j = nd1_vel[0];
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;


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
	__shared__ double dxi1_S[BLOCK_SIZE][4], dxh1_S[BLOCK_SIZE][4];
	__shared__ double dyi1_S[110][4], dyh1_S[110][4];
	__shared__ double dzi1_S[BLOCK_SIZE][4], dzh1_S[BLOCK_SIZE][4];

        if(threadIdx.x == 0) {
            for (int l = 0; l<4; l++){
               dxi1_S[(i-offset_i)%BLOCK_SIZE][l] = dxi1(l+1,i);
               dxh1_S[(i-offset_i)%BLOCK_SIZE][l] = dxh1(l+1,i);
            }
        }
        if (threadIdx.y == 0) {
            for(int l =0; l<4; l++) {
                dzi1_S[(k -offset_k)%BLOCK_SIZE][l] = dzi1(l+1, k);
                dzh1_S[(k -offset_k)%BLOCK_SIZE][l] = dzh1(l+1, k);               
            }

            if (threadIdx.x == 0 ) {
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

#elif USE_Optimized_vel_PmlX_IC == 0
__global__ void vel_PmlX_IC(double ca,
				int   lbx0,
				int   lbx1,
				int	  *nd1_vel,
				double *rhoM,
				double *drvh1M,
				double *drti1M,
				double *damp1_xM,
				int	  *idmat1M,
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
				int   mw1_pml1,	//dimension #
			    int   mw1_pml,
			    int   nxtop,
			    int   nytop,
			    int   nztop,
				double *v1xM,		//output
				double *v1yM,
				double *v1zM,
				double *v1x_pxM,
				double *v1y_pxM,
				double *v1z_pxM)
{
// !Compute the velocities in region of PML-x-I
// use grid_node_comm
// use wave_field_comm
// implicit NONE
	int i,j,k,lb,ib,kb;
	double rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
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
#endif

//-----------------------------------------------------------------------
#if USE_Optimized_vel_PmlY_IC == 1
__global__ void vel_PmlY_IC(int  nztop,
				double ca,
				int	  lby0,
				int   lby1,
				int   *nd1_vel,
				double *rhoM,
				double *drvh1M,
				double *drti1M,
				int   *idmat1M,
				double *damp1_yM,
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
				int   mw1_pml1, //dimension #s
				int   mw1_pml,
				int   nxtop,
				int   nytop,
				double *v1xM,		//output
				double *v1yM,
				double *v1zM,
				double *v1x_pyM,
				double *v1y_pyM,
				double *v1z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	double rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
        dtxz,dtyz,dtzz,vtmpx,vtmpy,vtmpz;

	//if( lby(1)>lby(2) ) 
	if( lby0>lby1 ) 
 		return;

	int offset_k = 1;
	k = blockDim.x * blockIdx.x + threadIdx.x + offset_k;
	int offset_i = nd1_vel[6];
	i  = blockDim.y * blockIdx.y + threadIdx.y + offset_i;

	if (k > nztop || i > nd1_vel[11])
	{
		return;
	}

	jbIni = 0;
	/*for (k = lby0; k < lb; i++)
	{
		for (j = nd1_vel[4*k]; j <= nd1_vel[1+4*lb]; j++)
		{
			jbIni++;
		}
	}

	jb = jbIni;
	*/

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

	__shared__ double dxi1_S[BLOCK_SIZE][4], dxh1_S[BLOCK_SIZE][4];
	__shared__ double dyi1_S[BLOCK_SIZE][4], dyh1_S[BLOCK_SIZE][4];
	__shared__ double dzi1_S[BLOCK_SIZE][4], dzh1_S[BLOCK_SIZE][4];

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
#elif USE_Optimized_vel_PmlY_IC == 0
__global__ void vel_PmlY_IC(int  nztop,
				double ca,
				int	  lby0,
				int   lby1,
				int   *nd1_vel,
				double *rhoM,
				double *drvh1M,
				double *drti1M,
				int   *idmat1M,
				double *damp1_yM,
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
				int   mw1_pml1, //dimension #s
				int   mw1_pml,
				int   nxtop,
				int   nytop,
				double *v1xM,		//output
				double *v1yM,
				double *v1zM,
				double *v1x_pyM,
				double *v1y_pyM,
				double *v1z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	double rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
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
	for (k = lby0; k < lb; i++)
	{
		for (j = nd1_vel[4*k]; j <= nd1_vel[1+4*lb]; j++)
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

                        //if(((k) + (nztop + 2) * ((i) + (j) * (nxtop + 3)))== 12377) 
                        //{
                        //        printf("k :%d, i: %d, j: %d, vtpmz: %f, v1_py(k,i,jb): %f, jb: %d, \
                        //        v1z(k,i,j): %f\n", k,i,j,vtmpz, v1y_py(k,i,jb), jb,
                        //        v1z(k,i,j) ); 
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
	//}
	//}

 	return;
}
#endif

//-----------------------------------------------------------------------
#if USE_Optimized_vel_PmlX_IIC == 1
__global__ void vel_PmlX_IIC(int   nzbm1,
				 double ca,
				 int   lbx0,
				 int   lbx1,
				 int   *nd2_vel,
				 double *drvh2M,
				 double *drti2M,
				 double *rhoM,
				 double *damp2_xM,
				 int   *idmat2M,
				 double *dxi2M,
				 double *dyi2M,
				 double *dzi2M,
				 double *dxh2M,
				 double *dyh2M,
				 double *dzh2M,
				 double *t2xxM,
				 double *t2xyM,
				 double *t2xzM,
				 double *t2yyM,
				 double *t2yzM,
				 double *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2xM,	//output
				 double *v2yM,
				 double *v2zM,
				 double *v2x_pxM,
				 double *v2y_pxM,
				 double *v2z_pxM)
{
	int i,j,k,lb,ib,kb;
	double rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
		vtmpx,vtmpy,vtmpz,dtxy,dtyy,dtzy,dtxz,dtyz,dtzz;

	//int nv2y = (lbx(2) - lbx(1) + 1) * mw2_pml;
	int nv2y = (lbx1 - lbx0 + 1) * mw2_pml;

	//if ( lbx(1)>lbx(2) ) return;
	if ( lbx0>lbx1 ) return;


	int offset_k = 1;
	k  = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	int offset_j = nd2_vel[0];
	j = blockIdx.y * blockDim.y + threadIdx.y + offset_j;

	if (j > nd2_vel[5] || k > nzbm1)
	{
		return;
	}
/*
#define BLOCK_SIZE 8

	__shared__ double dxi2_S[BLOCK_SIZE][4], dxh2_S[BLOCK_SIZE][4];
	__shared__ double dyi2_S[BLOCK_SIZE][4], dyh2_S[BLOCK_SIZE][4];
	__shared__ double dzi2_S[BLOCK_SIZE][4], dzh2_S[BLOCK_SIZE][4];

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

#elif USE_Optimized_vel_PmlX_IIC == 0
__global__ void vel_PmlX_IIC(int   nzbm1,
				 double ca,
				 int   lbx0,
				 int   lbx1,
				 int   *nd2_vel,
				 double *drvh2M,
				 double *drti2M,
				 double *rhoM,
				 double *damp2_xM,
				 int   *idmat2M,
				 double *dxi2M,
				 double *dyi2M,
				 double *dzi2M,
				 double *dxh2M,
				 double *dyh2M,
				 double *dzh2M,
				 double *t2xxM,
				 double *t2xyM,
				 double *t2xzM,
				 double *t2yyM,
				 double *t2yzM,
				 double *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2xM,	//output
				 double *v2yM,
				 double *v2zM,
				 double *v2x_pxM,
				 double *v2y_pxM,
				 double *v2z_pxM)
{
	int i,j,k,lb,ib,kb;
	double rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz,
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
		for (i = nd2_vel[6+4*k]; i < nd2_vel[7+4*k]; i++)
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
#endif


//-----------------------------------------------------------------------
#if USE_Optimized_vel_PmlY_IIC == 1
__global__ void vel_PmlY_IIC(int   nzbm1,
				 double ca,
				 int   lby0,
				 int   lby1,
				 int   *nd2_vel,
				 double *drvh2M,
				 double *drti2M,
				 double *rhoM,
				 double *damp2_yM,
				 int   *idmat2M,
				 double *dxi2M,
				 double *dyi2M,
				 double *dzi2M,
				 double *dxh2M,
				 double *dyh2M,
				 double *dzh2M,
				 double *t2xxM,
				 double *t2xyM,
				 double *t2xzM,
				 double *t2yyM,
				 double *t2yzM,
				 double *t2zzM,
				 int   mw2_pml1,
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2xM,		//output
				 double *v2yM,
				 double *v2zM,
				 double *v2x_pyM,
				 double *v2y_pyM,
				 double *v2z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	double rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
		   vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz;

	//if( lby(1)>lby(2) ) return;
	if( lby0>lby1 ) 
	{
		return;
	}

	int offset_k = 1;
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	int offset_i = nd2_vel[6];
	i  = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

	if (k > nzbm1 || i > nd2_vel[11])
	{
		return;
	}

/*
#define BLOCK_SIZE 8

	__shared__ double dxi2_S[BLOCK_SIZE][4], dxh2_S[BLOCK_SIZE][4];
	__shared__ double dyi2_S[BLOCK_SIZE][4], dyh2_S[BLOCK_SIZE][4];
	__shared__ double dzi2_S[BLOCK_SIZE][4], dzh2_S[BLOCK_SIZE][4];

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


#elif USE_Optimized_vel_PmlY_IIC == 0
__global__ void vel_PmlY_IIC(int   nzbm1,
				 double ca,
				 int   lby0,
				 int   lby1,
				 int   *nd2_vel,
				 double *drvh2M,
				 double *drti2M,
				 double *rhoM,
				 double *damp2_yM,
				 int   *idmat2M,
				 double *dxi2M,
				 double *dyi2M,
				 double *dzi2M,
				 double *dxh2M,
				 double *dyh2M,
				 double *dzh2M,
				 double *t2xxM,
				 double *t2xyM,
				 double *t2xzM,
				 double *t2yyM,
				 double *t2yzM,
				 double *t2zzM,
				 int   mw2_pml1,
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2xM,		//output
				 double *v2yM,
				 double *v2zM,
				 double *v2x_pyM,
				 double *v2y_pyM,
				 double *v2z_pyM)
{
	int i,j,k,lb,jb,kb, jbIni;
	double rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz,
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
#endif
//-----------------------------------------------------------------------
#if USE_Optimized_vel_PmlZ_IIC == 1
__global__ void vel_PmlZ_IIC(int   nzbm1,
				 double ca,
				 int   *nd2_vel,
				 double *drvh2M,
				 double *drti2M,
				 double *rhoM,
				 double *damp2_zM,
				 int   *idmat2M,
				 double *dxi2M,
				 double *dyi2M,
				 double *dzi2M,
				 double *dxh2M,
				 double *dyh2M,
				 double *dzh2M,
				 double *t2xxM,
				 double *t2xyM,
				 double *t2xzM,
				 double *t2yyM,
				 double *t2yzM,
				 double *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2xM,		//output
				 double *v2yM,
				 double *v2zM,
				 double *v2x_pzM,
				 double *v2y_pzM,
				 double *v2z_pzM)
{
	int i,j,k,kb;
	double damp0,dmpz2,dmpz1,dmpxy2,dmpxy1,ro1,rox,roy,roz,vtmpx,vtmpy,vtmpz;

        int offset_i = nd2_vel[6];
        int offset_j = nd2_vel[0];
        int offset_k = nd2_vel[16];
	k = blockIdx.x * blockDim.x + threadIdx.x + offset_k;
	i = blockIdx.y * blockDim.y + threadIdx.y + offset_i;

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
#define BLOCK_SIZE 8 
        __shared__ double dxi2_S[4][BLOCK_SIZE], dxh2_S[4][BLOCK_SIZE];
        __shared__ double dyi2_S[4][36],dyh2_S[4][36];
        __shared__ double dzi2_S[4][BLOCK_SIZE], dzh2_S[4][BLOCK_SIZE];

        if (threadIdx.x == 0) {
            //printf("BLOCK:(%d,%d): dxi2_S[0][%d ] = dxi2(%d,%d)\n",blockIdx.x, blockIdx.y, 0,(i-offset_i)%8,  1, i );
            dxi2_S[0][(i-offset_i)%BLOCK_SIZE] = dxi2(1,i);
            dxh2_S[0][(i-offset_i)%BLOCK_SIZE] = dxh2(1,i);
            dxi2_S[1][(i-offset_i)%BLOCK_SIZE] = dxi2(2,i);
            dxh2_S[1][(i-offset_i)%BLOCK_SIZE] = dxh2(2,i);
            dxi2_S[2][(i-offset_i)%BLOCK_SIZE] = dxi2(3,i);
            dxh2_S[2][(i-offset_i)%BLOCK_SIZE] = dxh2(3,i);
            dxi2_S[3][(i-offset_i)%BLOCK_SIZE] = dxi2(4,i);
            dxh2_S[3][(i-offset_i)%BLOCK_SIZE] = dxh2(4,i);
        }
        if(threadIdx.y == 0) {
            for (int l = 0; l<4; l++){
                    //printf("BLOCK:(%d,%d): dzi2_S[%d][(%d-%d)\% 8 ] = dzi2(%d+1,%d)\n",blockIdx.x, blockIdx.y, l,k,offset_k,  l, k );
                    //printf("BLOCK:(%d,%d): dzi2_S[%d][(%d ] = dzi2(%d,%d)\n",blockIdx.x, blockIdx.y, l,(k-offset_k)%8,  l+1, k );
                    dzi2_S[l][(k -offset_k)%BLOCK_SIZE] = dzi2(l+1, k);
                    dzh2_S[l][(k -offset_k)%BLOCK_SIZE] = dzh2(l+1, k);
            }

            if (threadIdx.x == 0) {
                for (j = nd2_vel[0] ; j <= nd2_vel[5]; j++) 
                {
                   for (int l=0; l<4 ; l++) { 
                    //printf("Block:(%d,%d) dyi2_S[%d][(%d-%d)%15 ] = dyi2(%d+1,%d)\n",blockIdx.x, blockIdx.y,l,j,offset_j, l, j );
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


#elif USE_Optimized_vel_PmlZ_IIC == 0
__global__ void vel_PmlZ_IIC(int   nzbm1,
				 double ca,
				 int   *nd2_vel,
				 double *drvh2M,
				 double *drti2M,
				 double *rhoM,
				 double *damp2_zM,
				 int   *idmat2M,
				 double *dxi2M,
				 double *dyi2M,
				 double *dzi2M,
				 double *dxh2M,
				 double *dyh2M,
				 double *dzh2M,
				 double *t2xxM,
				 double *t2xyM,
				 double *t2xzM,
				 double *t2yyM,
				 double *t2yzM,
				 double *t2zzM,
				 int   mw2_pml1,	//dimension #s
				 int   mw2_pml,
				 int   nxbtm,
				 int   nybtm,
				 int   nzbtm,
				 double *v2xM,		//output
				 double *v2yM,
				 double *v2zM,
				 double *v2x_pzM,
				 double *v2y_pzM,
				 double *v2z_pzM)
{
	int i,j,k,kb;
	double damp0,dmpz2,dmpz1,dmpxy2,dmpxy1,ro1,rox,roy,roz,vtmpx,vtmpy,vtmpz;

	j = blockIdx.x * blockDim.x + threadIdx.x + nd2_vel[0];
	i = blockIdx.y * blockDim.y + threadIdx.y + nd2_vel[6];
/*
	if (j == nd2_vel[0] && i == nd2_vel[6])
	{
		for (k = 0; k < 18; k++)
		{
			printf("nd2_vel[%d] = %d\n", k, nd2_vel[k]);
		}
	}
*/
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
/*        
	if (j == nd2_vel[0] && i == nd2_vel[6])
		printf("damp0 = %f, damp2 = %f, i = %d, j = %d\n", damp0, damp2_z(i, j), i, j);
*/                

	//for (k = nd2_vel(17); k <= nzbm1; k++)
	for (k = nd2_vel[16]; k <= nzbm1; k++)
	{
		kb=kb+1;
		dmpz2=1./(1.+damp0*drti2(kb,1));
		dmpz1=dmpz2*2.-1.;
//		if (j == nd2_vel[0] && i == nd2_vel[6])
//			printf("dmpz1 = %f, dmpz2 = %f\n", dmpz1, dmpz2);
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
#endif

