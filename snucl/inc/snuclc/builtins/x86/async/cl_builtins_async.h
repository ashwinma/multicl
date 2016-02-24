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

#ifndef __CL_BUILTINS_ASYNC_H
#define __CL_BUILTINS_ASYNC_H

#include <cl_cpu_types.h>

/* 6.11.11 Async Copies from Global to Local Memory, Local to Global Memory, */
/*         and Prefetch                                                      */
/* Table 6.19 Built-in Async Copy and Prefetch Functions */
event_t async_work_group_copy_g2l(schar *, const schar *, size_t, event_t);
event_t async_work_group_copy_g2l(char2 *, const char2 *, size_t, event_t);
event_t async_work_group_copy_g2l(char3 *, const char3 *, size_t, event_t);
event_t async_work_group_copy_g2l(char4 *, const char4 *, size_t, event_t);
event_t async_work_group_copy_g2l(char8 *, const char8 *, size_t, event_t);
event_t async_work_group_copy_g2l(char16 *, const char16 *, size_t, event_t);
event_t async_work_group_copy_g2l(uchar *, const uchar *, size_t, event_t);
event_t async_work_group_copy_g2l(uchar2 *, const uchar2 *, size_t, event_t);
event_t async_work_group_copy_g2l(uchar3 *, const uchar3 *, size_t, event_t);
event_t async_work_group_copy_g2l(uchar4 *, const uchar4 *, size_t, event_t);
event_t async_work_group_copy_g2l(uchar8 *, const uchar8 *, size_t, event_t);
event_t async_work_group_copy_g2l(uchar16 *, const uchar16 *, size_t, event_t);
event_t async_work_group_copy_g2l(short *, const short *, size_t, event_t);
event_t async_work_group_copy_g2l(short2 *, const short2 *, size_t, event_t);
event_t async_work_group_copy_g2l(short3 *, const short3 *, size_t, event_t);
event_t async_work_group_copy_g2l(short4 *, const short4 *, size_t, event_t);
event_t async_work_group_copy_g2l(short8 *, const short8 *, size_t, event_t);
event_t async_work_group_copy_g2l(short16 *, const short16 *, size_t, event_t);
event_t async_work_group_copy_g2l(ushort *, const ushort *, size_t, event_t);
event_t async_work_group_copy_g2l(ushort2 *, const ushort2 *, size_t, event_t);
event_t async_work_group_copy_g2l(ushort3 *, const ushort3 *, size_t, event_t);
event_t async_work_group_copy_g2l(ushort4 *, const ushort4 *, size_t, event_t);
event_t async_work_group_copy_g2l(ushort8 *, const ushort8 *, size_t, event_t);
event_t async_work_group_copy_g2l(ushort16 *, const ushort16 *, size_t, event_t);
event_t async_work_group_copy_g2l(int *, const int *, size_t, event_t);
event_t async_work_group_copy_g2l(int2 *, const int2 *, size_t, event_t);
event_t async_work_group_copy_g2l(int3 *, const int3 *, size_t, event_t);
event_t async_work_group_copy_g2l(int4 *, const int4 *, size_t, event_t);
event_t async_work_group_copy_g2l(int8 *, const int8 *, size_t, event_t);
event_t async_work_group_copy_g2l(int16 *, const int16 *, size_t, event_t);
event_t async_work_group_copy_g2l(uint *, const uint *, size_t, event_t);
event_t async_work_group_copy_g2l(uint2 *, const uint2 *, size_t, event_t);
event_t async_work_group_copy_g2l(uint3 *, const uint3 *, size_t, event_t);
event_t async_work_group_copy_g2l(uint4 *, const uint4 *, size_t, event_t);
event_t async_work_group_copy_g2l(uint8 *, const uint8 *, size_t, event_t);
event_t async_work_group_copy_g2l(uint16 *, const uint16 *, size_t, event_t);
event_t async_work_group_copy_g2l(llong *, const llong *, size_t, event_t);
event_t async_work_group_copy_g2l(long2 *, const long2 *, size_t, event_t);
event_t async_work_group_copy_g2l(long3 *, const long3 *, size_t, event_t);
event_t async_work_group_copy_g2l(long4 *, const long4 *, size_t, event_t);
event_t async_work_group_copy_g2l(long8 *, const long8 *, size_t, event_t);
event_t async_work_group_copy_g2l(long16 *, const long16 *, size_t, event_t);
event_t async_work_group_copy_g2l(ullong *, const ullong *, size_t, event_t);
event_t async_work_group_copy_g2l(ulong2 *, const ulong2 *, size_t, event_t);
event_t async_work_group_copy_g2l(ulong3 *, const ulong3 *, size_t, event_t);
event_t async_work_group_copy_g2l(ulong4 *, const ulong4 *, size_t, event_t);
event_t async_work_group_copy_g2l(ulong8 *, const ulong8 *, size_t, event_t);
event_t async_work_group_copy_g2l(ulong16 *, const ulong16 *, size_t, event_t);
event_t async_work_group_copy_g2l(float *, const float *, size_t, event_t);
event_t async_work_group_copy_g2l(float2 *, const float2 *, size_t, event_t);
event_t async_work_group_copy_g2l(float3 *, const float3 *, size_t, event_t);
event_t async_work_group_copy_g2l(float4 *, const float4 *, size_t, event_t);
event_t async_work_group_copy_g2l(float8 *, const float8 *, size_t, event_t);
event_t async_work_group_copy_g2l(float16 *, const float16 *, size_t, event_t);
event_t async_work_group_copy_g2l(double *, const double *, size_t, event_t);
event_t async_work_group_copy_g2l(double2 *, const double2 *, size_t, event_t);
event_t async_work_group_copy_g2l(double3 *, const double3 *, size_t, event_t);
event_t async_work_group_copy_g2l(double4 *, const double4 *, size_t, event_t);
event_t async_work_group_copy_g2l(double8 *, const double8 *, size_t, event_t);
event_t async_work_group_copy_g2l(double16 *, const double16 *, size_t, event_t);

event_t async_work_group_copy_l2g(schar *, const schar *, size_t, event_t);
event_t async_work_group_copy_l2g(char2 *, const char2 *, size_t, event_t);
event_t async_work_group_copy_l2g(char3 *, const char3 *, size_t, event_t);
event_t async_work_group_copy_l2g(char4 *, const char4 *, size_t, event_t);
event_t async_work_group_copy_l2g(char8 *, const char8 *, size_t, event_t);
event_t async_work_group_copy_l2g(char16 *, const char16 *, size_t, event_t);
event_t async_work_group_copy_l2g(uchar *, const uchar *, size_t, event_t);
event_t async_work_group_copy_l2g(uchar2 *, const uchar2 *, size_t, event_t);
event_t async_work_group_copy_l2g(uchar3 *, const uchar3 *, size_t, event_t);
event_t async_work_group_copy_l2g(uchar4 *, const uchar4 *, size_t, event_t);
event_t async_work_group_copy_l2g(uchar8 *, const uchar8 *, size_t, event_t);
event_t async_work_group_copy_l2g(uchar16 *, const uchar16 *, size_t, event_t);
event_t async_work_group_copy_l2g(short *, const short *, size_t, event_t);
event_t async_work_group_copy_l2g(short2 *, const short2 *, size_t, event_t);
event_t async_work_group_copy_l2g(short3 *, const short3 *, size_t, event_t);
event_t async_work_group_copy_l2g(short4 *, const short4 *, size_t, event_t);
event_t async_work_group_copy_l2g(short8 *, const short8 *, size_t, event_t);
event_t async_work_group_copy_l2g(short16 *, const short16 *, size_t, event_t);
event_t async_work_group_copy_l2g(ushort *, const ushort *, size_t, event_t);
event_t async_work_group_copy_l2g(ushort2 *, const ushort2 *, size_t, event_t);
event_t async_work_group_copy_l2g(ushort3 *, const ushort3 *, size_t, event_t);
event_t async_work_group_copy_l2g(ushort4 *, const ushort4 *, size_t, event_t);
event_t async_work_group_copy_l2g(ushort8 *, const ushort8 *, size_t, event_t);
event_t async_work_group_copy_l2g(ushort16 *, const ushort16 *, size_t, event_t);
event_t async_work_group_copy_l2g(int *, const int *, size_t, event_t);
event_t async_work_group_copy_l2g(int2 *, const int2 *, size_t, event_t);
event_t async_work_group_copy_l2g(int3 *, const int3 *, size_t, event_t);
event_t async_work_group_copy_l2g(int4 *, const int4 *, size_t, event_t);
event_t async_work_group_copy_l2g(int8 *, const int8 *, size_t, event_t);
event_t async_work_group_copy_l2g(int16 *, const int16 *, size_t, event_t);
event_t async_work_group_copy_l2g(uint *, const uint *, size_t, event_t);
event_t async_work_group_copy_l2g(uint2 *, const uint2 *, size_t, event_t);
event_t async_work_group_copy_l2g(uint3 *, const uint3 *, size_t, event_t);
event_t async_work_group_copy_l2g(uint4 *, const uint4 *, size_t, event_t);
event_t async_work_group_copy_l2g(uint8 *, const uint8 *, size_t, event_t);
event_t async_work_group_copy_l2g(uint16 *, const uint16 *, size_t, event_t);
event_t async_work_group_copy_l2g(llong *, const llong *, size_t, event_t);
event_t async_work_group_copy_l2g(long2 *, const long2 *, size_t, event_t);
event_t async_work_group_copy_l2g(long3 *, const long3 *, size_t, event_t);
event_t async_work_group_copy_l2g(long4 *, const long4 *, size_t, event_t);
event_t async_work_group_copy_l2g(long8 *, const long8 *, size_t, event_t);
event_t async_work_group_copy_l2g(long16 *, const long16 *, size_t, event_t);
event_t async_work_group_copy_l2g(ullong *, const ullong *, size_t, event_t);
event_t async_work_group_copy_l2g(ulong2 *, const ulong2 *, size_t, event_t);
event_t async_work_group_copy_l2g(ulong3 *, const ulong3 *, size_t, event_t);
event_t async_work_group_copy_l2g(ulong4 *, const ulong4 *, size_t, event_t);
event_t async_work_group_copy_l2g(ulong8 *, const ulong8 *, size_t, event_t);
event_t async_work_group_copy_l2g(ulong16 *, const ulong16 *, size_t, event_t);
event_t async_work_group_copy_l2g(float *, const float *, size_t, event_t);
event_t async_work_group_copy_l2g(float2 *, const float2 *, size_t, event_t);
event_t async_work_group_copy_l2g(float3 *, const float3 *, size_t, event_t);
event_t async_work_group_copy_l2g(float4 *, const float4 *, size_t, event_t);
event_t async_work_group_copy_l2g(float8 *, const float8 *, size_t, event_t);
event_t async_work_group_copy_l2g(float16 *, const float16 *, size_t, event_t);
event_t async_work_group_copy_l2g(double *, const double *, size_t, event_t);
event_t async_work_group_copy_l2g(double2 *, const double2 *, size_t, event_t);
event_t async_work_group_copy_l2g(double3 *, const double3 *, size_t, event_t);
event_t async_work_group_copy_l2g(double4 *, const double4 *, size_t, event_t);
event_t async_work_group_copy_l2g(double8 *, const double8 *, size_t, event_t);
event_t async_work_group_copy_l2g(double16 *, const double16 *, size_t, event_t);


void wait_group_events(int, event_t *);


void prefetch(const schar *, size_t);
void prefetch(const char2 *, size_t);
void prefetch(const char3 *, size_t);
void prefetch(const char4 *, size_t);
void prefetch(const char8 *, size_t);
void prefetch(const char16 *, size_t);
void prefetch(const uchar *, size_t);
void prefetch(const uchar2 *, size_t);
void prefetch(const uchar3 *, size_t);
void prefetch(const uchar4 *, size_t);
void prefetch(const uchar8 *, size_t);
void prefetch(const uchar16 *, size_t);
void prefetch(const short *, size_t);
void prefetch(const short2 *, size_t);
void prefetch(const short3 *, size_t);
void prefetch(const short4 *, size_t);
void prefetch(const short8 *, size_t);
void prefetch(const short16 *, size_t);
void prefetch(const ushort *, size_t);
void prefetch(const ushort2 *, size_t);
void prefetch(const ushort3 *, size_t);
void prefetch(const ushort4 *, size_t);
void prefetch(const ushort8 *, size_t);
void prefetch(const ushort16 *, size_t);
void prefetch(const int *, size_t);
void prefetch(const int2 *, size_t);
void prefetch(const int3 *, size_t);
void prefetch(const int4 *, size_t);
void prefetch(const int8 *, size_t);
void prefetch(const int16 *, size_t);
void prefetch(const uint *, size_t);
void prefetch(const uint2 *, size_t);
void prefetch(const uint3 *, size_t);
void prefetch(const uint4 *, size_t);
void prefetch(const uint8 *, size_t);
void prefetch(const uint16 *, size_t);
void prefetch(const llong *, size_t);
void prefetch(const long2 *, size_t);
void prefetch(const long3 *, size_t);
void prefetch(const long4 *, size_t);
void prefetch(const long8 *, size_t);
void prefetch(const long16 *, size_t);
void prefetch(const ullong *, size_t);
void prefetch(const ulong2 *, size_t);
void prefetch(const ulong3 *, size_t);
void prefetch(const ulong4 *, size_t);
void prefetch(const ulong8 *, size_t);
void prefetch(const ulong16 *, size_t);
void prefetch(const float *, size_t);
void prefetch(const float2 *, size_t);
void prefetch(const float3 *, size_t);
void prefetch(const float4 *, size_t);
void prefetch(const float8 *, size_t);
void prefetch(const float16 *, size_t);
void prefetch(const double *, size_t);
void prefetch(const double2 *, size_t);
void prefetch(const double3 *, size_t);
void prefetch(const double4 *, size_t);
void prefetch(const double8 *, size_t);
void prefetch(const double16 *, size_t);

#define SNUCL_ASYNC_STRIDE_G2L_H(type)  \
event_t async_work_group_strided_copy_g2l(type *dst, const type *src, size_t num_elements, size_t src_stride, event_t event); \

#define SNUCL_ASYNC_STRIDE_L2G_H(type)  \
event_t async_work_group_strided_copy_l2g(type *dst, const type *src, size_t num_elements, size_t dst_stride, event_t event); \

#include "async_strided_copy_h.inc"

#endif //__CL_BUILTINS_ASYNC_H

