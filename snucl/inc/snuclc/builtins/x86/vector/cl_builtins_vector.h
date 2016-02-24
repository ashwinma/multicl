/*****************************************************************************/
/* Copyright (C) 2010, 2011 Seoul National University                        */
/* and Samsung Electronics Co., Ltd.                                         */
/*                                                                           */
/* Contributed by Sangmin Seo <sangmin@aces.snu.ac.kr>, Jungwon Kim          */
/* <jungwon@aces.snu.ac.kr>, Jaejin Lee <jlee@cse.snu.ac.kr>, Seungkyun Kim  */
/* <seungkyun@aces.snu.ac.kr>, Jungho Park <jungho@aces.snu.ac.kr>,          */
/* Honggyu Kim <honggyu@aces.snu.ac.kr>, Jeongho Nah                         */
/* <jeongho@aces.snu.ac.kr>, Sung Jong Seo <sj1557.seo@samsung.com>,         */
/* Seung Hak Lee <s.hak.lee@samsung.com>, Seung Mo Cho                       */
/* <seungm.cho@samsung.com>, Hyo Jung Song <hjsong@samsung.com>,             */
/* Sang-Bum Suh <sbuk.suh@samsung.com>, and Jong-Deok Choi                   */
/* <jd11.choi@samsung.com>                                                   */
/*                                                                           */
/* All rights reserved.                                                      */
/*                                                                           */
/* This file is part of the SNU-SAMSUNG OpenCL runtime.                      */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is free software: you can redistribute it  */
/* and/or modify it under the terms of the GNU Lesser General Public License */
/* as published by the Free Software Foundation, either version 3 of the     */
/* License, or (at your option) any later version.                           */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is distributed in the hope that it will be */
/* useful, but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  */
/* Public License for more details.                                          */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with the SNU-SAMSUNG OpenCL runtime. If not, see                    */
/* <http://www.gnu.org/licenses/>.                                           */
/*****************************************************************************/

#ifndef __CL_BUILTINS_VECTOR_H
#define __CL_BUILTINS_VECTOR_H

#include <cl_cpu_types.h>

/* 6.11.7 Vector Data Load and Store Functions */
/* Table 6.14 Vector Data Load and Store Functions */

char2    vload2(size_t, const schar *);
char3    vload3(size_t, const schar *);
char4    vload4(size_t, const schar *);
char8    vload8(size_t, const schar *);
char16   vload16(size_t, const schar *);
uchar2   vload2(size_t, const uchar *);
uchar3   vload3(size_t, const uchar *);
uchar4   vload4(size_t, const uchar *);
uchar8   vload8(size_t, const uchar *);
uchar16  vload16(size_t, const uchar *);
short2   vload2(size_t, const short *);
short3   vload3(size_t, const short *);
short4   vload4(size_t, const short *);
short8   vload8(size_t, const short *);
short16  vload16(size_t, const short *);
ushort2  vload2(size_t, const ushort *);
ushort3  vload3(size_t, const ushort *);
ushort4  vload4(size_t, const ushort *);
ushort8  vload8(size_t, const ushort *);
ushort16 vload16(size_t, const ushort *);
int2     vload2(size_t, const int *);
int3     vload3(size_t, const int *);
int4     vload4(size_t, const int *);
int8     vload8(size_t, const int *);
int16    vload16(size_t, const int *);
uint2    vload2(size_t, const uint *);
uint3    vload3(size_t, const uint *);
uint4    vload4(size_t, const uint *);
uint8    vload8(size_t, const uint *);
uint16   vload16(size_t, const uint *);
long2    vload2(size_t, const llong *);
long3    vload3(size_t, const llong *);
long4    vload4(size_t, const llong *);
long8    vload8(size_t, const llong *);
long16   vload16(size_t, const llong *);
ulong2   vload2(size_t, const ullong *);
ulong3   vload3(size_t, const ullong *);
ulong4   vload4(size_t, const ullong *);
ulong8   vload8(size_t, const ullong *);
ulong16  vload16(size_t, const ullong *);
float2   vload2(size_t, const float *);
float3   vload3(size_t, const float *);
float4   vload4(size_t, const float *);
float8   vload8(size_t, const float *);
float16  vload16(size_t, const float *);
double2   vload2(size_t, const double *);
double3   vload3(size_t, const double *);
double4   vload4(size_t, const double *);
double8   vload8(size_t, const double *);
double16  vload16(size_t, const double *);

/* vstore functions */
void vstore2(char2, size_t, schar *);
void vstore3(char3, size_t, schar *);
void vstore4(char4, size_t, schar *);
void vstore8(char8, size_t, schar *);
void vstore16(char16, size_t, schar *);
void vstore2(uchar2, size_t, uchar *);
void vstore3(uchar3, size_t, uchar *);
void vstore4(uchar4, size_t, uchar *);
void vstore8(uchar8, size_t, uchar *);
void vstore16(uchar16, size_t, uchar *);
void vstore2(short2, size_t, short *);
void vstore3(short3, size_t, short *);
void vstore4(short4, size_t, short *);
void vstore8(short8, size_t, short *);
void vstore16(short16, size_t, short *);
void vstore2(ushort2, size_t, ushort *);
void vstore3(ushort3, size_t, ushort *);
void vstore4(ushort4, size_t, ushort *);
void vstore8(ushort8, size_t, ushort *);
void vstore16(ushort16, size_t, ushort *);
void vstore2(int2, size_t, int *);
void vstore3(int3, size_t, int *);
void vstore4(int4, size_t, int *);
void vstore8(int8, size_t, int *);
void vstore16(int16, size_t, int *);
void vstore2(uint2, size_t, uint *);
void vstore3(uint3, size_t, uint *);
void vstore4(uint4, size_t, uint *);
void vstore8(uint8, size_t, uint *);
void vstore16(uint16, size_t, uint *);
void vstore2(long2, size_t, llong *);
void vstore3(long3, size_t, llong *);
void vstore4(long4, size_t, llong *);
void vstore8(long8, size_t, llong *);
void vstore16(long16, size_t, llong *);
void vstore2(ulong2, size_t, ullong *);
void vstore3(ulong3, size_t, ullong *);
void vstore4(ulong4, size_t, ullong *);
void vstore8(ulong8, size_t, ullong *);
void vstore16(ulong16, size_t, ullong *);
void vstore2(float2, size_t, float *);
void vstore3(float3, size_t, float *);
void vstore4(float4, size_t, float *);
void vstore8(float8, size_t, float *);
void vstore16(float16, size_t, float *);
void vstore2(double2, size_t, double *);
void vstore3(double3, size_t, double *);
void vstore4(double4, size_t, double *);
void vstore8(double8, size_t, double *);
void vstore16(double16, size_t, double *);

/* vload_half */
float    vload_half(size_t, const half *);
float2   vload_half2(size_t, const half *);
float3   vload_half3(size_t, const half *);
float4   vload_half4(size_t, const half *);
float8   vload_half8(size_t, const half *);
float16  vload_half16(size_t, const half *);

/* vloada_half */
#define vloada_half		  vload_half
#define vloada_half2		vload_half2
float3  vloada_half3(size_t, const half *);
#define vloada_half4		vload_half4
#define vloada_half8		vload_half8
#define vloada_half16	  vload_half16

/* vstore_half */
void vstore_half(float, size_t, half *);
void vstore_half2(float2, size_t, half *);
void vstore_half3(float3, size_t, half *);
void vstore_half4(float4, size_t, half *);
void vstore_half8(float8, size_t, half *);
void vstore_half16(float16, size_t, half *);

void vstore_half_rte(float, size_t, half *);
void vstore_half2_rte(float2, size_t, half *);
void vstore_half3_rte(float3, size_t, half *);
void vstore_half4_rte(float4, size_t, half *);
void vstore_half8_rte(float8, size_t, half *);
void vstore_half16_rte(float16, size_t, half *);

void vstore_half_rtz(float, size_t, half *);
void vstore_half2_rtz(float2, size_t, half *);
void vstore_half3_rtz(float3, size_t, half *);
void vstore_half4_rtz(float4, size_t, half *);
void vstore_half8_rtz(float8, size_t, half *);
void vstore_half16_rtz(float16, size_t, half *);

void vstore_half_rtp(float, size_t, half *);
void vstore_half2_rtp(float2, size_t, half *);
void vstore_half3_rtp(float3, size_t, half *);
void vstore_half4_rtp(float4, size_t, half *);
void vstore_half8_rtp(float8, size_t, half *);
void vstore_half16_rtp(float16, size_t, half *);

void vstore_half_rtn(float, size_t, half *);
void vstore_half2_rtn(float2, size_t, half *);
void vstore_half3_rtn(float3, size_t, half *);
void vstore_half4_rtn(float4, size_t, half *);
void vstore_half8_rtn(float8, size_t, half *);
void vstore_half16_rtn(float16, size_t, half *);

void vstore_half(double, size_t, half *);
void vstore_half2(double2, size_t, half *);
void vstore_half3(double3, size_t, half *);
void vstore_half4(double4, size_t, half *);
void vstore_half8(double8, size_t, half *);
void vstore_half16(double16, size_t, half *);

void vstore_half_rte(double, size_t, half *);
void vstore_half2_rte(double2, size_t, half *);
void vstore_half3_rte(double3, size_t, half *);
void vstore_half4_rte(double4, size_t, half *);
void vstore_half8_rte(double8, size_t, half *);
void vstore_half16_rte(double16, size_t, half *);

void vstore_half_rtz(double, size_t, half *);
void vstore_half2_rtz(double2, size_t, half *);
void vstore_half3_rtz(double3, size_t, half *);
void vstore_half4_rtz(double4, size_t, half *);
void vstore_half8_rtz(double8, size_t, half *);
void vstore_half16_rtz(double16, size_t, half *);

void vstore_half_rtp(double, size_t, half *);
void vstore_half2_rtp(double2, size_t, half *);
void vstore_half3_rtp(double3, size_t, half *);
void vstore_half4_rtp(double4, size_t, half *);
void vstore_half8_rtp(double8, size_t, half *);
void vstore_half16_rtp(double16, size_t, half *);

void vstore_half_rtn(double, size_t, half *);
void vstore_half2_rtn(double2, size_t, half *);
void vstore_half3_rtn(double3, size_t, half *);
void vstore_half4_rtn(double4, size_t, half *);
void vstore_half8_rtn(double8, size_t, half *);
void vstore_half16_rtn(double16, size_t, half *);

/* vstorea_half */
#define vstorea_half		    vstore_half
#define vstorea_half2	      vstore_half2
void    vstorea_half3(float3, size_t, half *);
void    vstorea_half3(double3, size_t, half *);
#define vstorea_half4	      vstore_half4
#define vstorea_half8	      vstore_half8
#define vstorea_half16	    vstore_half16

#define vstorea_half_rte		vstore_half_rte
#define vstorea_half2_rte	  vstore_half2_rte
void    vstorea_half3_rte(float3, size_t, half *);
void    vstorea_half3_rte(double3, size_t, half *);
#define vstorea_half4_rte	  vstore_half4_rte
#define vstorea_half8_rte	  vstore_half8_rte
#define vstorea_half16_rte	vstore_half16_rte

#define vstorea_half_rtz		vstore_half_rtz
#define vstorea_half2_rtz	  vstore_half2_rtz
void    vstorea_half3_rtz(float3, size_t, half *);
void    vstorea_half3_rtz(double3, size_t, half *);
#define vstorea_half4_rtz	  vstore_half4_rtz
#define vstorea_half8_rtz	  vstore_half8_rtz
#define vstorea_half16_rtz	vstore_half16_rtz

#define vstorea_half_rtp		vstore_half_rtp
#define vstorea_half2_rtp	  vstore_half2_rtp
void    vstorea_half3_rtp(float3, size_t, half *);
void    vstorea_half3_rtp(double3, size_t, half *);
#define vstorea_half4_rtp	  vstore_half4_rtp
#define vstorea_half8_rtp	  vstore_half8_rtp
#define vstorea_half16_rtp	vstore_half16_rtp

#define vstorea_half_rtn		vstore_half_rtn
#define vstorea_half2_rtn	  vstore_half2_rtn
void    vstorea_half3_rtn(float3, size_t, half *);
void    vstorea_half3_rtn(double3, size_t, half *);
#define vstorea_half4_rtn	  vstore_half4_rtn
#define vstorea_half8_rtn	  vstore_half8_rtn
#define vstorea_half16_rtn	vstore_half16_rtn

#endif //__CL_BUILTINS_VECTOR_H

