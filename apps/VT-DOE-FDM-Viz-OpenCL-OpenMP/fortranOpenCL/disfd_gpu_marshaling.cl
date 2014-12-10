//#include "disfd_macros.h"

// < MPI-ACC  >
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


