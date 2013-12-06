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

#ifndef __CL_BUILTINS_CONVERSION_H
#define __CL_BUILTINS_CONVERSION_H

#include <cl_cpu_types.h>

/* For vector type casting from scalar type */
inline char2    convert_char2(schar x)      { return __CL_CHAR2(x, x); }
inline char3    convert_char3(schar x)      { return __CL_CHAR3(x, x, x); }
inline char4    convert_char4(schar x)      { return __CL_CHAR4(x, x, x, x); }
inline char8    convert_char8(schar x)      { return __CL_CHAR8(x, x, x, x, x, x, x, x); }
inline char16   convert_char16(schar x)     { return __CL_CHAR16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline float2   convert_float2(float x)     { return __CL_FLOAT2(x, x); }
inline float3   convert_float3(float x)     { return __CL_FLOAT3(x, x, x); }
inline float4   convert_float4(float x)     { return __CL_FLOAT4(x, x, x, x); }
inline float8   convert_float8(float x)     { return __CL_FLOAT8(x, x, x, x, x, x, x, x); }
inline float16  convert_float16(float x)    { return __CL_FLOAT16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline double2   convert_double2(double x)  { return __CL_DOUBLE2(x, x); }
inline double3   convert_double3(double x)  { return __CL_DOUBLE3(x, x, x); }
inline double4   convert_double4(double x)  { return __CL_DOUBLE4(x, x, x, x); }
inline double8   convert_double8(double x)  { return __CL_DOUBLE8(x, x, x, x, x, x, x, x); }
inline double16  convert_double16(double x) { return __CL_DOUBLE16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline int2     convert_int2(int x)         { return __CL_INT2(x, x); }
inline int3     convert_int3(int x)         { return __CL_INT3(x, x, x); }
inline int4     convert_int4(int x)         { return __CL_INT4(x, x, x, x); }
inline int8     convert_int8(int x)         { return __CL_INT8(x, x, x, x, x, x, x, x); }
inline int16    convert_int16(int x)        { return __CL_INT16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline long2    convert_long2(llong x)      { return __CL_LONG2(x, x); }
inline long3    convert_long3(llong x)      { return __CL_LONG3(x, x, x); }
inline long4    convert_long4(llong x)      { return __CL_LONG4(x, x, x, x); }
inline long8    convert_long8(llong x)      { return __CL_LONG8(x, x, x, x, x, x, x, x); }
inline long16   convert_long16(llong x)     { return __CL_LONG16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline short2   convert_short2(short x)     { return __CL_SHORT2(x, x); }
inline short3   convert_short3(short x)     { return __CL_SHORT3(x, x, x); }
inline short4   convert_short4(short x)     { return __CL_SHORT4(x, x, x, x); }
inline short8   convert_short8(short x)     { return __CL_SHORT8(x, x, x, x, x, x, x, x); }
inline short16  convert_short16(short x)    { return __CL_SHORT16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline uchar2   convert_uchar2(uchar x)     { return __CL_UCHAR2(x, x); }
inline uchar3   convert_uchar3(uchar x)     { return __CL_UCHAR3(x, x, x); }
inline uchar4   convert_uchar4(uchar x)     { return __CL_UCHAR4(x, x, x, x); }
inline uchar8   convert_uchar8(uchar x)     { return __CL_UCHAR8(x, x, x, x, x, x, x, x); }
inline uchar16  convert_uchar16(uchar x)    { return __CL_UCHAR16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline uint2    convert_uint2(uint x)       { return __CL_UINT2(x, x); }
inline uint3    convert_uint3(uint x)       { return __CL_UINT3(x, x, x); }
inline uint4    convert_uint4(uint x)       { return __CL_UINT4(x, x, x, x); }
inline uint8    convert_uint8(uint x)       { return __CL_UINT8(x, x, x, x, x, x, x, x); }
inline uint16   convert_uint16(uint x)      { return __CL_UINT16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline ulong2   convert_ulong2(ullong x)    { return __CL_ULONG2(x, x); }
inline ulong3   convert_ulong3(ullong x)    { return __CL_ULONG3(x, x, x); }
inline ulong4   convert_ulong4(ullong x)    { return __CL_ULONG4(x, x, x, x); }
inline ulong8   convert_ulong8(ullong x)    { return __CL_ULONG8(x, x, x, x, x, x, x, x); }
inline ulong16  convert_ulong16(ullong x)   { return __CL_ULONG16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }
inline ushort2  convert_ushort2(ushort x)   { return __CL_USHORT2(x, x); }
inline ushort3  convert_ushort3(ushort x)   { return __CL_USHORT3(x, x, x); }
inline ushort4  convert_ushort4(ushort x)   { return __CL_USHORT4(x, x, x, x); }
inline ushort8  convert_ushort8(ushort x)   { return __CL_USHORT8(x, x, x, x, x, x, x, x); }
inline ushort16 convert_ushort16(ushort x)  { return __CL_USHORT16(x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x); }

/* 6.2.3 Explicit Conversions */

uchar convert_uchar(uchar x);
uchar2 convert_uchar2(uchar2 x);
uchar3 convert_uchar3(uchar3 x);
uchar4 convert_uchar4(uchar4 x);
uchar8 convert_uchar8(uchar8 x);
uchar16 convert_uchar16(uchar16 x);

uchar convert_uchar_rte(uchar x);
uchar2 convert_uchar2_rte(uchar2 x);
uchar3 convert_uchar3_rte(uchar3 x);
uchar4 convert_uchar4_rte(uchar4 x);
uchar8 convert_uchar8_rte(uchar8 x);
uchar16 convert_uchar16_rte(uchar16 x);

uchar convert_uchar_rtp(uchar x);
uchar2 convert_uchar2_rtp(uchar2 x);
uchar3 convert_uchar3_rtp(uchar3 x);
uchar4 convert_uchar4_rtp(uchar4 x);
uchar8 convert_uchar8_rtp(uchar8 x);
uchar16 convert_uchar16_rtp(uchar16 x);

uchar convert_uchar_rtn(uchar x);
uchar2 convert_uchar2_rtn(uchar2 x);
uchar3 convert_uchar3_rtn(uchar3 x);
uchar4 convert_uchar4_rtn(uchar4 x);
uchar8 convert_uchar8_rtn(uchar8 x);
uchar16 convert_uchar16_rtn(uchar16 x);

uchar convert_uchar_rtz(uchar x);
uchar2 convert_uchar2_rtz(uchar2 x);
uchar3 convert_uchar3_rtz(uchar3 x);
uchar4 convert_uchar4_rtz(uchar4 x);
uchar8 convert_uchar8_rtz(uchar8 x);
uchar16 convert_uchar16_rtz(uchar16 x);

uchar convert_uchar_sat(uchar x);
uchar2 convert_uchar2_sat(uchar2 x);
uchar3 convert_uchar3_sat(uchar3 x);
uchar4 convert_uchar4_sat(uchar4 x);
uchar8 convert_uchar8_sat(uchar8 x);
uchar16 convert_uchar16_sat(uchar16 x);

uchar convert_uchar_sat_rte(uchar x);
uchar2 convert_uchar2_sat_rte(uchar2 x);
uchar3 convert_uchar3_sat_rte(uchar3 x);
uchar4 convert_uchar4_sat_rte(uchar4 x);
uchar8 convert_uchar8_sat_rte(uchar8 x);
uchar16 convert_uchar16_sat_rte(uchar16 x);

uchar convert_uchar_sat_rtp(uchar x);
uchar2 convert_uchar2_sat_rtp(uchar2 x);
uchar3 convert_uchar3_sat_rtp(uchar3 x);
uchar4 convert_uchar4_sat_rtp(uchar4 x);
uchar8 convert_uchar8_sat_rtp(uchar8 x);
uchar16 convert_uchar16_sat_rtp(uchar16 x);

uchar convert_uchar_sat_rtn(uchar x);
uchar2 convert_uchar2_sat_rtn(uchar2 x);
uchar3 convert_uchar3_sat_rtn(uchar3 x);
uchar4 convert_uchar4_sat_rtn(uchar4 x);
uchar8 convert_uchar8_sat_rtn(uchar8 x);
uchar16 convert_uchar16_sat_rtn(uchar16 x);

uchar convert_uchar_sat_rtz(uchar x);
uchar2 convert_uchar2_sat_rtz(uchar2 x);
uchar3 convert_uchar3_sat_rtz(uchar3 x);
uchar4 convert_uchar4_sat_rtz(uchar4 x);
uchar8 convert_uchar8_sat_rtz(uchar8 x);
uchar16 convert_uchar16_sat_rtz(uchar16 x);

uchar convert_uchar(schar x);
uchar2 convert_uchar2(char2 x);
uchar3 convert_uchar3(char3 x);
uchar4 convert_uchar4(char4 x);
uchar8 convert_uchar8(char8 x);
uchar16 convert_uchar16(char16 x);

uchar convert_uchar_rte(schar x);
uchar2 convert_uchar2_rte(char2 x);
uchar3 convert_uchar3_rte(char3 x);
uchar4 convert_uchar4_rte(char4 x);
uchar8 convert_uchar8_rte(char8 x);
uchar16 convert_uchar16_rte(char16 x);

uchar convert_uchar_rtp(schar x);
uchar2 convert_uchar2_rtp(char2 x);
uchar3 convert_uchar3_rtp(char3 x);
uchar4 convert_uchar4_rtp(char4 x);
uchar8 convert_uchar8_rtp(char8 x);
uchar16 convert_uchar16_rtp(char16 x);

uchar convert_uchar_rtn(schar x);
uchar2 convert_uchar2_rtn(char2 x);
uchar3 convert_uchar3_rtn(char3 x);
uchar4 convert_uchar4_rtn(char4 x);
uchar8 convert_uchar8_rtn(char8 x);
uchar16 convert_uchar16_rtn(char16 x);

uchar convert_uchar_rtz(schar x);
uchar2 convert_uchar2_rtz(char2 x);
uchar3 convert_uchar3_rtz(char3 x);
uchar4 convert_uchar4_rtz(char4 x);
uchar8 convert_uchar8_rtz(char8 x);
uchar16 convert_uchar16_rtz(char16 x);

uchar convert_uchar_sat(schar x);
uchar2 convert_uchar2_sat(char2 x);
uchar3 convert_uchar3_sat(char3 x);
uchar4 convert_uchar4_sat(char4 x);
uchar8 convert_uchar8_sat(char8 x);
uchar16 convert_uchar16_sat(char16 x);

uchar convert_uchar_sat_rte(schar x);
uchar2 convert_uchar2_sat_rte(char2 x);
uchar3 convert_uchar3_sat_rte(char3 x);
uchar4 convert_uchar4_sat_rte(char4 x);
uchar8 convert_uchar8_sat_rte(char8 x);
uchar16 convert_uchar16_sat_rte(char16 x);

uchar convert_uchar_sat_rtp(schar x);
uchar2 convert_uchar2_sat_rtp(char2 x);
uchar3 convert_uchar3_sat_rtp(char3 x);
uchar4 convert_uchar4_sat_rtp(char4 x);
uchar8 convert_uchar8_sat_rtp(char8 x);
uchar16 convert_uchar16_sat_rtp(char16 x);

uchar convert_uchar_sat_rtn(schar x);
uchar2 convert_uchar2_sat_rtn(char2 x);
uchar3 convert_uchar3_sat_rtn(char3 x);
uchar4 convert_uchar4_sat_rtn(char4 x);
uchar8 convert_uchar8_sat_rtn(char8 x);
uchar16 convert_uchar16_sat_rtn(char16 x);

uchar convert_uchar_sat_rtz(schar x);
uchar2 convert_uchar2_sat_rtz(char2 x);
uchar3 convert_uchar3_sat_rtz(char3 x);
uchar4 convert_uchar4_sat_rtz(char4 x);
uchar8 convert_uchar8_sat_rtz(char8 x);
uchar16 convert_uchar16_sat_rtz(char16 x);

uchar convert_uchar(ushort x);
uchar2 convert_uchar2(ushort2 x);
uchar3 convert_uchar3(ushort3 x);
uchar4 convert_uchar4(ushort4 x);
uchar8 convert_uchar8(ushort8 x);
uchar16 convert_uchar16(ushort16 x);

uchar convert_uchar_rte(ushort x);
uchar2 convert_uchar2_rte(ushort2 x);
uchar3 convert_uchar3_rte(ushort3 x);
uchar4 convert_uchar4_rte(ushort4 x);
uchar8 convert_uchar8_rte(ushort8 x);
uchar16 convert_uchar16_rte(ushort16 x);

uchar convert_uchar_rtp(ushort x);
uchar2 convert_uchar2_rtp(ushort2 x);
uchar3 convert_uchar3_rtp(ushort3 x);
uchar4 convert_uchar4_rtp(ushort4 x);
uchar8 convert_uchar8_rtp(ushort8 x);
uchar16 convert_uchar16_rtp(ushort16 x);

uchar convert_uchar_rtn(ushort x);
uchar2 convert_uchar2_rtn(ushort2 x);
uchar3 convert_uchar3_rtn(ushort3 x);
uchar4 convert_uchar4_rtn(ushort4 x);
uchar8 convert_uchar8_rtn(ushort8 x);
uchar16 convert_uchar16_rtn(ushort16 x);

uchar convert_uchar_rtz(ushort x);
uchar2 convert_uchar2_rtz(ushort2 x);
uchar3 convert_uchar3_rtz(ushort3 x);
uchar4 convert_uchar4_rtz(ushort4 x);
uchar8 convert_uchar8_rtz(ushort8 x);
uchar16 convert_uchar16_rtz(ushort16 x);

uchar convert_uchar_sat(ushort x);
uchar2 convert_uchar2_sat(ushort2 x);
uchar3 convert_uchar3_sat(ushort3 x);
uchar4 convert_uchar4_sat(ushort4 x);
uchar8 convert_uchar8_sat(ushort8 x);
uchar16 convert_uchar16_sat(ushort16 x);

uchar convert_uchar_sat_rte(ushort x);
uchar2 convert_uchar2_sat_rte(ushort2 x);
uchar3 convert_uchar3_sat_rte(ushort3 x);
uchar4 convert_uchar4_sat_rte(ushort4 x);
uchar8 convert_uchar8_sat_rte(ushort8 x);
uchar16 convert_uchar16_sat_rte(ushort16 x);

uchar convert_uchar_sat_rtp(ushort x);
uchar2 convert_uchar2_sat_rtp(ushort2 x);
uchar3 convert_uchar3_sat_rtp(ushort3 x);
uchar4 convert_uchar4_sat_rtp(ushort4 x);
uchar8 convert_uchar8_sat_rtp(ushort8 x);
uchar16 convert_uchar16_sat_rtp(ushort16 x);

uchar convert_uchar_sat_rtn(ushort x);
uchar2 convert_uchar2_sat_rtn(ushort2 x);
uchar3 convert_uchar3_sat_rtn(ushort3 x);
uchar4 convert_uchar4_sat_rtn(ushort4 x);
uchar8 convert_uchar8_sat_rtn(ushort8 x);
uchar16 convert_uchar16_sat_rtn(ushort16 x);

uchar convert_uchar_sat_rtz(ushort x);
uchar2 convert_uchar2_sat_rtz(ushort2 x);
uchar3 convert_uchar3_sat_rtz(ushort3 x);
uchar4 convert_uchar4_sat_rtz(ushort4 x);
uchar8 convert_uchar8_sat_rtz(ushort8 x);
uchar16 convert_uchar16_sat_rtz(ushort16 x);

uchar convert_uchar(short x);
uchar2 convert_uchar2(short2 x);
uchar3 convert_uchar3(short3 x);
uchar4 convert_uchar4(short4 x);
uchar8 convert_uchar8(short8 x);
uchar16 convert_uchar16(short16 x);

uchar convert_uchar_rte(short x);
uchar2 convert_uchar2_rte(short2 x);
uchar3 convert_uchar3_rte(short3 x);
uchar4 convert_uchar4_rte(short4 x);
uchar8 convert_uchar8_rte(short8 x);
uchar16 convert_uchar16_rte(short16 x);

uchar convert_uchar_rtp(short x);
uchar2 convert_uchar2_rtp(short2 x);
uchar3 convert_uchar3_rtp(short3 x);
uchar4 convert_uchar4_rtp(short4 x);
uchar8 convert_uchar8_rtp(short8 x);
uchar16 convert_uchar16_rtp(short16 x);

uchar convert_uchar_rtn(short x);
uchar2 convert_uchar2_rtn(short2 x);
uchar3 convert_uchar3_rtn(short3 x);
uchar4 convert_uchar4_rtn(short4 x);
uchar8 convert_uchar8_rtn(short8 x);
uchar16 convert_uchar16_rtn(short16 x);

uchar convert_uchar_rtz(short x);
uchar2 convert_uchar2_rtz(short2 x);
uchar3 convert_uchar3_rtz(short3 x);
uchar4 convert_uchar4_rtz(short4 x);
uchar8 convert_uchar8_rtz(short8 x);
uchar16 convert_uchar16_rtz(short16 x);

uchar convert_uchar_sat(short x);
uchar2 convert_uchar2_sat(short2 x);
uchar3 convert_uchar3_sat(short3 x);
uchar4 convert_uchar4_sat(short4 x);
uchar8 convert_uchar8_sat(short8 x);
uchar16 convert_uchar16_sat(short16 x);

uchar convert_uchar_sat_rte(short x);
uchar2 convert_uchar2_sat_rte(short2 x);
uchar3 convert_uchar3_sat_rte(short3 x);
uchar4 convert_uchar4_sat_rte(short4 x);
uchar8 convert_uchar8_sat_rte(short8 x);
uchar16 convert_uchar16_sat_rte(short16 x);

uchar convert_uchar_sat_rtp(short x);
uchar2 convert_uchar2_sat_rtp(short2 x);
uchar3 convert_uchar3_sat_rtp(short3 x);
uchar4 convert_uchar4_sat_rtp(short4 x);
uchar8 convert_uchar8_sat_rtp(short8 x);
uchar16 convert_uchar16_sat_rtp(short16 x);

uchar convert_uchar_sat_rtn(short x);
uchar2 convert_uchar2_sat_rtn(short2 x);
uchar3 convert_uchar3_sat_rtn(short3 x);
uchar4 convert_uchar4_sat_rtn(short4 x);
uchar8 convert_uchar8_sat_rtn(short8 x);
uchar16 convert_uchar16_sat_rtn(short16 x);

uchar convert_uchar_sat_rtz(short x);
uchar2 convert_uchar2_sat_rtz(short2 x);
uchar3 convert_uchar3_sat_rtz(short3 x);
uchar4 convert_uchar4_sat_rtz(short4 x);
uchar8 convert_uchar8_sat_rtz(short8 x);
uchar16 convert_uchar16_sat_rtz(short16 x);

uchar convert_uchar(uint x);
uchar2 convert_uchar2(uint2 x);
uchar3 convert_uchar3(uint3 x);
uchar4 convert_uchar4(uint4 x);
uchar8 convert_uchar8(uint8 x);
uchar16 convert_uchar16(uint16 x);

uchar convert_uchar_rte(uint x);
uchar2 convert_uchar2_rte(uint2 x);
uchar3 convert_uchar3_rte(uint3 x);
uchar4 convert_uchar4_rte(uint4 x);
uchar8 convert_uchar8_rte(uint8 x);
uchar16 convert_uchar16_rte(uint16 x);

uchar convert_uchar_rtp(uint x);
uchar2 convert_uchar2_rtp(uint2 x);
uchar3 convert_uchar3_rtp(uint3 x);
uchar4 convert_uchar4_rtp(uint4 x);
uchar8 convert_uchar8_rtp(uint8 x);
uchar16 convert_uchar16_rtp(uint16 x);

uchar convert_uchar_rtn(uint x);
uchar2 convert_uchar2_rtn(uint2 x);
uchar3 convert_uchar3_rtn(uint3 x);
uchar4 convert_uchar4_rtn(uint4 x);
uchar8 convert_uchar8_rtn(uint8 x);
uchar16 convert_uchar16_rtn(uint16 x);

uchar convert_uchar_rtz(uint x);
uchar2 convert_uchar2_rtz(uint2 x);
uchar3 convert_uchar3_rtz(uint3 x);
uchar4 convert_uchar4_rtz(uint4 x);
uchar8 convert_uchar8_rtz(uint8 x);
uchar16 convert_uchar16_rtz(uint16 x);

uchar convert_uchar_sat(uint x);
uchar2 convert_uchar2_sat(uint2 x);
uchar3 convert_uchar3_sat(uint3 x);
uchar4 convert_uchar4_sat(uint4 x);
uchar8 convert_uchar8_sat(uint8 x);
uchar16 convert_uchar16_sat(uint16 x);

uchar convert_uchar_sat_rte(uint x);
uchar2 convert_uchar2_sat_rte(uint2 x);
uchar3 convert_uchar3_sat_rte(uint3 x);
uchar4 convert_uchar4_sat_rte(uint4 x);
uchar8 convert_uchar8_sat_rte(uint8 x);
uchar16 convert_uchar16_sat_rte(uint16 x);

uchar convert_uchar_sat_rtp(uint x);
uchar2 convert_uchar2_sat_rtp(uint2 x);
uchar3 convert_uchar3_sat_rtp(uint3 x);
uchar4 convert_uchar4_sat_rtp(uint4 x);
uchar8 convert_uchar8_sat_rtp(uint8 x);
uchar16 convert_uchar16_sat_rtp(uint16 x);

uchar convert_uchar_sat_rtn(uint x);
uchar2 convert_uchar2_sat_rtn(uint2 x);
uchar3 convert_uchar3_sat_rtn(uint3 x);
uchar4 convert_uchar4_sat_rtn(uint4 x);
uchar8 convert_uchar8_sat_rtn(uint8 x);
uchar16 convert_uchar16_sat_rtn(uint16 x);

uchar convert_uchar_sat_rtz(uint x);
uchar2 convert_uchar2_sat_rtz(uint2 x);
uchar3 convert_uchar3_sat_rtz(uint3 x);
uchar4 convert_uchar4_sat_rtz(uint4 x);
uchar8 convert_uchar8_sat_rtz(uint8 x);
uchar16 convert_uchar16_sat_rtz(uint16 x);

uchar convert_uchar(int x);
uchar2 convert_uchar2(int2 x);
uchar3 convert_uchar3(int3 x);
uchar4 convert_uchar4(int4 x);
uchar8 convert_uchar8(int8 x);
uchar16 convert_uchar16(int16 x);

uchar convert_uchar_rte(int x);
uchar2 convert_uchar2_rte(int2 x);
uchar3 convert_uchar3_rte(int3 x);
uchar4 convert_uchar4_rte(int4 x);
uchar8 convert_uchar8_rte(int8 x);
uchar16 convert_uchar16_rte(int16 x);

uchar convert_uchar_rtp(int x);
uchar2 convert_uchar2_rtp(int2 x);
uchar3 convert_uchar3_rtp(int3 x);
uchar4 convert_uchar4_rtp(int4 x);
uchar8 convert_uchar8_rtp(int8 x);
uchar16 convert_uchar16_rtp(int16 x);

uchar convert_uchar_rtn(int x);
uchar2 convert_uchar2_rtn(int2 x);
uchar3 convert_uchar3_rtn(int3 x);
uchar4 convert_uchar4_rtn(int4 x);
uchar8 convert_uchar8_rtn(int8 x);
uchar16 convert_uchar16_rtn(int16 x);

uchar convert_uchar_rtz(int x);
uchar2 convert_uchar2_rtz(int2 x);
uchar3 convert_uchar3_rtz(int3 x);
uchar4 convert_uchar4_rtz(int4 x);
uchar8 convert_uchar8_rtz(int8 x);
uchar16 convert_uchar16_rtz(int16 x);

uchar convert_uchar_sat(int x);
uchar2 convert_uchar2_sat(int2 x);
uchar3 convert_uchar3_sat(int3 x);
uchar4 convert_uchar4_sat(int4 x);
uchar8 convert_uchar8_sat(int8 x);
uchar16 convert_uchar16_sat(int16 x);

uchar convert_uchar_sat_rte(int x);
uchar2 convert_uchar2_sat_rte(int2 x);
uchar3 convert_uchar3_sat_rte(int3 x);
uchar4 convert_uchar4_sat_rte(int4 x);
uchar8 convert_uchar8_sat_rte(int8 x);
uchar16 convert_uchar16_sat_rte(int16 x);

uchar convert_uchar_sat_rtp(int x);
uchar2 convert_uchar2_sat_rtp(int2 x);
uchar3 convert_uchar3_sat_rtp(int3 x);
uchar4 convert_uchar4_sat_rtp(int4 x);
uchar8 convert_uchar8_sat_rtp(int8 x);
uchar16 convert_uchar16_sat_rtp(int16 x);

uchar convert_uchar_sat_rtn(int x);
uchar2 convert_uchar2_sat_rtn(int2 x);
uchar3 convert_uchar3_sat_rtn(int3 x);
uchar4 convert_uchar4_sat_rtn(int4 x);
uchar8 convert_uchar8_sat_rtn(int8 x);
uchar16 convert_uchar16_sat_rtn(int16 x);

uchar convert_uchar_sat_rtz(int x);
uchar2 convert_uchar2_sat_rtz(int2 x);
uchar3 convert_uchar3_sat_rtz(int3 x);
uchar4 convert_uchar4_sat_rtz(int4 x);
uchar8 convert_uchar8_sat_rtz(int8 x);
uchar16 convert_uchar16_sat_rtz(int16 x);

uchar convert_uchar(float x);
uchar2 convert_uchar2(float2 x);
uchar3 convert_uchar3(float3 x);
uchar4 convert_uchar4(float4 x);
uchar8 convert_uchar8(float8 x);
uchar16 convert_uchar16(float16 x);

uchar convert_uchar_rte(float x);
uchar2 convert_uchar2_rte(float2 x);
uchar3 convert_uchar3_rte(float3 x);
uchar4 convert_uchar4_rte(float4 x);
uchar8 convert_uchar8_rte(float8 x);
uchar16 convert_uchar16_rte(float16 x);

uchar convert_uchar_rtp(float x);
uchar2 convert_uchar2_rtp(float2 x);
uchar3 convert_uchar3_rtp(float3 x);
uchar4 convert_uchar4_rtp(float4 x);
uchar8 convert_uchar8_rtp(float8 x);
uchar16 convert_uchar16_rtp(float16 x);

uchar convert_uchar_rtn(float x);
uchar2 convert_uchar2_rtn(float2 x);
uchar3 convert_uchar3_rtn(float3 x);
uchar4 convert_uchar4_rtn(float4 x);
uchar8 convert_uchar8_rtn(float8 x);
uchar16 convert_uchar16_rtn(float16 x);

uchar convert_uchar_rtz(float x);
uchar2 convert_uchar2_rtz(float2 x);
uchar3 convert_uchar3_rtz(float3 x);
uchar4 convert_uchar4_rtz(float4 x);
uchar8 convert_uchar8_rtz(float8 x);
uchar16 convert_uchar16_rtz(float16 x);

uchar convert_uchar_sat(float x);
uchar2 convert_uchar2_sat(float2 x);
uchar3 convert_uchar3_sat(float3 x);
uchar4 convert_uchar4_sat(float4 x);
uchar8 convert_uchar8_sat(float8 x);
uchar16 convert_uchar16_sat(float16 x);

uchar convert_uchar_sat_rte(float x);
uchar2 convert_uchar2_sat_rte(float2 x);
uchar3 convert_uchar3_sat_rte(float3 x);
uchar4 convert_uchar4_sat_rte(float4 x);
uchar8 convert_uchar8_sat_rte(float8 x);
uchar16 convert_uchar16_sat_rte(float16 x);

uchar convert_uchar_sat_rtp(float x);
uchar2 convert_uchar2_sat_rtp(float2 x);
uchar3 convert_uchar3_sat_rtp(float3 x);
uchar4 convert_uchar4_sat_rtp(float4 x);
uchar8 convert_uchar8_sat_rtp(float8 x);
uchar16 convert_uchar16_sat_rtp(float16 x);

uchar convert_uchar_sat_rtn(float x);
uchar2 convert_uchar2_sat_rtn(float2 x);
uchar3 convert_uchar3_sat_rtn(float3 x);
uchar4 convert_uchar4_sat_rtn(float4 x);
uchar8 convert_uchar8_sat_rtn(float8 x);
uchar16 convert_uchar16_sat_rtn(float16 x);

uchar convert_uchar_sat_rtz(float x);
uchar2 convert_uchar2_sat_rtz(float2 x);
uchar3 convert_uchar3_sat_rtz(float3 x);
uchar4 convert_uchar4_sat_rtz(float4 x);
uchar8 convert_uchar8_sat_rtz(float8 x);
uchar16 convert_uchar16_sat_rtz(float16 x);

uchar convert_uchar(double x);
uchar2 convert_uchar2(double2 x);
uchar3 convert_uchar3(double3 x);
uchar4 convert_uchar4(double4 x);
uchar8 convert_uchar8(double8 x);
uchar16 convert_uchar16(double16 x);

uchar convert_uchar_rte(double x);
uchar2 convert_uchar2_rte(double2 x);
uchar3 convert_uchar3_rte(double3 x);
uchar4 convert_uchar4_rte(double4 x);
uchar8 convert_uchar8_rte(double8 x);
uchar16 convert_uchar16_rte(double16 x);

uchar convert_uchar_rtp(double x);
uchar2 convert_uchar2_rtp(double2 x);
uchar3 convert_uchar3_rtp(double3 x);
uchar4 convert_uchar4_rtp(double4 x);
uchar8 convert_uchar8_rtp(double8 x);
uchar16 convert_uchar16_rtp(double16 x);

uchar convert_uchar_rtn(double x);
uchar2 convert_uchar2_rtn(double2 x);
uchar3 convert_uchar3_rtn(double3 x);
uchar4 convert_uchar4_rtn(double4 x);
uchar8 convert_uchar8_rtn(double8 x);
uchar16 convert_uchar16_rtn(double16 x);

uchar convert_uchar_rtz(double x);
uchar2 convert_uchar2_rtz(double2 x);
uchar3 convert_uchar3_rtz(double3 x);
uchar4 convert_uchar4_rtz(double4 x);
uchar8 convert_uchar8_rtz(double8 x);
uchar16 convert_uchar16_rtz(double16 x);

uchar convert_uchar_sat(double x);
uchar2 convert_uchar2_sat(double2 x);
uchar3 convert_uchar3_sat(double3 x);
uchar4 convert_uchar4_sat(double4 x);
uchar8 convert_uchar8_sat(double8 x);
uchar16 convert_uchar16_sat(double16 x);

uchar convert_uchar_sat_rte(double x);
uchar2 convert_uchar2_sat_rte(double2 x);
uchar3 convert_uchar3_sat_rte(double3 x);
uchar4 convert_uchar4_sat_rte(double4 x);
uchar8 convert_uchar8_sat_rte(double8 x);
uchar16 convert_uchar16_sat_rte(double16 x);

uchar convert_uchar_sat_rtp(double x);
uchar2 convert_uchar2_sat_rtp(double2 x);
uchar3 convert_uchar3_sat_rtp(double3 x);
uchar4 convert_uchar4_sat_rtp(double4 x);
uchar8 convert_uchar8_sat_rtp(double8 x);
uchar16 convert_uchar16_sat_rtp(double16 x);

uchar convert_uchar_sat_rtn(double x);
uchar2 convert_uchar2_sat_rtn(double2 x);
uchar3 convert_uchar3_sat_rtn(double3 x);
uchar4 convert_uchar4_sat_rtn(double4 x);
uchar8 convert_uchar8_sat_rtn(double8 x);
uchar16 convert_uchar16_sat_rtn(double16 x);

uchar convert_uchar_sat_rtz(double x);
uchar2 convert_uchar2_sat_rtz(double2 x);
uchar3 convert_uchar3_sat_rtz(double3 x);
uchar4 convert_uchar4_sat_rtz(double4 x);
uchar8 convert_uchar8_sat_rtz(double8 x);
uchar16 convert_uchar16_sat_rtz(double16 x);

uchar convert_uchar(ullong x);
uchar2 convert_uchar2(ulong2 x);
uchar3 convert_uchar3(ulong3 x);
uchar4 convert_uchar4(ulong4 x);
uchar8 convert_uchar8(ulong8 x);
uchar16 convert_uchar16(ulong16 x);

uchar convert_uchar_rte(ullong x);
uchar2 convert_uchar2_rte(ulong2 x);
uchar3 convert_uchar3_rte(ulong3 x);
uchar4 convert_uchar4_rte(ulong4 x);
uchar8 convert_uchar8_rte(ulong8 x);
uchar16 convert_uchar16_rte(ulong16 x);

uchar convert_uchar_rtp(ullong x);
uchar2 convert_uchar2_rtp(ulong2 x);
uchar3 convert_uchar3_rtp(ulong3 x);
uchar4 convert_uchar4_rtp(ulong4 x);
uchar8 convert_uchar8_rtp(ulong8 x);
uchar16 convert_uchar16_rtp(ulong16 x);

uchar convert_uchar_rtn(ullong x);
uchar2 convert_uchar2_rtn(ulong2 x);
uchar3 convert_uchar3_rtn(ulong3 x);
uchar4 convert_uchar4_rtn(ulong4 x);
uchar8 convert_uchar8_rtn(ulong8 x);
uchar16 convert_uchar16_rtn(ulong16 x);

uchar convert_uchar_rtz(ullong x);
uchar2 convert_uchar2_rtz(ulong2 x);
uchar3 convert_uchar3_rtz(ulong3 x);
uchar4 convert_uchar4_rtz(ulong4 x);
uchar8 convert_uchar8_rtz(ulong8 x);
uchar16 convert_uchar16_rtz(ulong16 x);

uchar convert_uchar_sat(ullong x);
uchar2 convert_uchar2_sat(ulong2 x);
uchar3 convert_uchar3_sat(ulong3 x);
uchar4 convert_uchar4_sat(ulong4 x);
uchar8 convert_uchar8_sat(ulong8 x);
uchar16 convert_uchar16_sat(ulong16 x);

uchar convert_uchar_sat_rte(ullong x);
uchar2 convert_uchar2_sat_rte(ulong2 x);
uchar3 convert_uchar3_sat_rte(ulong3 x);
uchar4 convert_uchar4_sat_rte(ulong4 x);
uchar8 convert_uchar8_sat_rte(ulong8 x);
uchar16 convert_uchar16_sat_rte(ulong16 x);

uchar convert_uchar_sat_rtp(ullong x);
uchar2 convert_uchar2_sat_rtp(ulong2 x);
uchar3 convert_uchar3_sat_rtp(ulong3 x);
uchar4 convert_uchar4_sat_rtp(ulong4 x);
uchar8 convert_uchar8_sat_rtp(ulong8 x);
uchar16 convert_uchar16_sat_rtp(ulong16 x);

uchar convert_uchar_sat_rtn(ullong x);
uchar2 convert_uchar2_sat_rtn(ulong2 x);
uchar3 convert_uchar3_sat_rtn(ulong3 x);
uchar4 convert_uchar4_sat_rtn(ulong4 x);
uchar8 convert_uchar8_sat_rtn(ulong8 x);
uchar16 convert_uchar16_sat_rtn(ulong16 x);

uchar convert_uchar_sat_rtz(ullong x);
uchar2 convert_uchar2_sat_rtz(ulong2 x);
uchar3 convert_uchar3_sat_rtz(ulong3 x);
uchar4 convert_uchar4_sat_rtz(ulong4 x);
uchar8 convert_uchar8_sat_rtz(ulong8 x);
uchar16 convert_uchar16_sat_rtz(ulong16 x);

uchar convert_uchar(llong x);
uchar2 convert_uchar2(long2 x);
uchar3 convert_uchar3(long3 x);
uchar4 convert_uchar4(long4 x);
uchar8 convert_uchar8(long8 x);
uchar16 convert_uchar16(long16 x);

uchar convert_uchar_rte(llong x);
uchar2 convert_uchar2_rte(long2 x);
uchar3 convert_uchar3_rte(long3 x);
uchar4 convert_uchar4_rte(long4 x);
uchar8 convert_uchar8_rte(long8 x);
uchar16 convert_uchar16_rte(long16 x);

uchar convert_uchar_rtp(llong x);
uchar2 convert_uchar2_rtp(long2 x);
uchar3 convert_uchar3_rtp(long3 x);
uchar4 convert_uchar4_rtp(long4 x);
uchar8 convert_uchar8_rtp(long8 x);
uchar16 convert_uchar16_rtp(long16 x);

uchar convert_uchar_rtn(llong x);
uchar2 convert_uchar2_rtn(long2 x);
uchar3 convert_uchar3_rtn(long3 x);
uchar4 convert_uchar4_rtn(long4 x);
uchar8 convert_uchar8_rtn(long8 x);
uchar16 convert_uchar16_rtn(long16 x);

uchar convert_uchar_rtz(llong x);
uchar2 convert_uchar2_rtz(long2 x);
uchar3 convert_uchar3_rtz(long3 x);
uchar4 convert_uchar4_rtz(long4 x);
uchar8 convert_uchar8_rtz(long8 x);
uchar16 convert_uchar16_rtz(long16 x);

uchar convert_uchar_sat(llong x);
uchar2 convert_uchar2_sat(long2 x);
uchar3 convert_uchar3_sat(long3 x);
uchar4 convert_uchar4_sat(long4 x);
uchar8 convert_uchar8_sat(long8 x);
uchar16 convert_uchar16_sat(long16 x);

uchar convert_uchar_sat_rte(llong x);
uchar2 convert_uchar2_sat_rte(long2 x);
uchar3 convert_uchar3_sat_rte(long3 x);
uchar4 convert_uchar4_sat_rte(long4 x);
uchar8 convert_uchar8_sat_rte(long8 x);
uchar16 convert_uchar16_sat_rte(long16 x);

uchar convert_uchar_sat_rtp(llong x);
uchar2 convert_uchar2_sat_rtp(long2 x);
uchar3 convert_uchar3_sat_rtp(long3 x);
uchar4 convert_uchar4_sat_rtp(long4 x);
uchar8 convert_uchar8_sat_rtp(long8 x);
uchar16 convert_uchar16_sat_rtp(long16 x);

uchar convert_uchar_sat_rtn(llong x);
uchar2 convert_uchar2_sat_rtn(long2 x);
uchar3 convert_uchar3_sat_rtn(long3 x);
uchar4 convert_uchar4_sat_rtn(long4 x);
uchar8 convert_uchar8_sat_rtn(long8 x);
uchar16 convert_uchar16_sat_rtn(long16 x);

uchar convert_uchar_sat_rtz(llong x);
uchar2 convert_uchar2_sat_rtz(long2 x);
uchar3 convert_uchar3_sat_rtz(long3 x);
uchar4 convert_uchar4_sat_rtz(long4 x);
uchar8 convert_uchar8_sat_rtz(long8 x);
uchar16 convert_uchar16_sat_rtz(long16 x);

schar convert_char(uchar x);
char2 convert_char2(uchar2 x);
char3 convert_char3(uchar3 x);
char4 convert_char4(uchar4 x);
char8 convert_char8(uchar8 x);
char16 convert_char16(uchar16 x);

schar convert_char_rte(uchar x);
char2 convert_char2_rte(uchar2 x);
char3 convert_char3_rte(uchar3 x);
char4 convert_char4_rte(uchar4 x);
char8 convert_char8_rte(uchar8 x);
char16 convert_char16_rte(uchar16 x);

schar convert_char_rtp(uchar x);
char2 convert_char2_rtp(uchar2 x);
char3 convert_char3_rtp(uchar3 x);
char4 convert_char4_rtp(uchar4 x);
char8 convert_char8_rtp(uchar8 x);
char16 convert_char16_rtp(uchar16 x);

schar convert_char_rtn(uchar x);
char2 convert_char2_rtn(uchar2 x);
char3 convert_char3_rtn(uchar3 x);
char4 convert_char4_rtn(uchar4 x);
char8 convert_char8_rtn(uchar8 x);
char16 convert_char16_rtn(uchar16 x);

schar convert_char_rtz(uchar x);
char2 convert_char2_rtz(uchar2 x);
char3 convert_char3_rtz(uchar3 x);
char4 convert_char4_rtz(uchar4 x);
char8 convert_char8_rtz(uchar8 x);
char16 convert_char16_rtz(uchar16 x);

schar convert_char_sat(uchar x);
char2 convert_char2_sat(uchar2 x);
char3 convert_char3_sat(uchar3 x);
char4 convert_char4_sat(uchar4 x);
char8 convert_char8_sat(uchar8 x);
char16 convert_char16_sat(uchar16 x);

schar convert_char_sat_rte(uchar x);
char2 convert_char2_sat_rte(uchar2 x);
char3 convert_char3_sat_rte(uchar3 x);
char4 convert_char4_sat_rte(uchar4 x);
char8 convert_char8_sat_rte(uchar8 x);
char16 convert_char16_sat_rte(uchar16 x);

schar convert_char_sat_rtp(uchar x);
char2 convert_char2_sat_rtp(uchar2 x);
char3 convert_char3_sat_rtp(uchar3 x);
char4 convert_char4_sat_rtp(uchar4 x);
char8 convert_char8_sat_rtp(uchar8 x);
char16 convert_char16_sat_rtp(uchar16 x);

schar convert_char_sat_rtn(uchar x);
char2 convert_char2_sat_rtn(uchar2 x);
char3 convert_char3_sat_rtn(uchar3 x);
char4 convert_char4_sat_rtn(uchar4 x);
char8 convert_char8_sat_rtn(uchar8 x);
char16 convert_char16_sat_rtn(uchar16 x);

schar convert_char_sat_rtz(uchar x);
char2 convert_char2_sat_rtz(uchar2 x);
char3 convert_char3_sat_rtz(uchar3 x);
char4 convert_char4_sat_rtz(uchar4 x);
char8 convert_char8_sat_rtz(uchar8 x);
char16 convert_char16_sat_rtz(uchar16 x);

schar convert_char(schar x);
char2 convert_char2(char2 x);
char3 convert_char3(char3 x);
char4 convert_char4(char4 x);
char8 convert_char8(char8 x);
char16 convert_char16(char16 x);

schar convert_char_rte(schar x);
char2 convert_char2_rte(char2 x);
char3 convert_char3_rte(char3 x);
char4 convert_char4_rte(char4 x);
char8 convert_char8_rte(char8 x);
char16 convert_char16_rte(char16 x);

schar convert_char_rtp(schar x);
char2 convert_char2_rtp(char2 x);
char3 convert_char3_rtp(char3 x);
char4 convert_char4_rtp(char4 x);
char8 convert_char8_rtp(char8 x);
char16 convert_char16_rtp(char16 x);

schar convert_char_rtn(schar x);
char2 convert_char2_rtn(char2 x);
char3 convert_char3_rtn(char3 x);
char4 convert_char4_rtn(char4 x);
char8 convert_char8_rtn(char8 x);
char16 convert_char16_rtn(char16 x);

schar convert_char_rtz(schar x);
char2 convert_char2_rtz(char2 x);
char3 convert_char3_rtz(char3 x);
char4 convert_char4_rtz(char4 x);
char8 convert_char8_rtz(char8 x);
char16 convert_char16_rtz(char16 x);

schar convert_char_sat(schar x);
char2 convert_char2_sat(char2 x);
char3 convert_char3_sat(char3 x);
char4 convert_char4_sat(char4 x);
char8 convert_char8_sat(char8 x);
char16 convert_char16_sat(char16 x);

schar convert_char_sat_rte(schar x);
char2 convert_char2_sat_rte(char2 x);
char3 convert_char3_sat_rte(char3 x);
char4 convert_char4_sat_rte(char4 x);
char8 convert_char8_sat_rte(char8 x);
char16 convert_char16_sat_rte(char16 x);

schar convert_char_sat_rtp(schar x);
char2 convert_char2_sat_rtp(char2 x);
char3 convert_char3_sat_rtp(char3 x);
char4 convert_char4_sat_rtp(char4 x);
char8 convert_char8_sat_rtp(char8 x);
char16 convert_char16_sat_rtp(char16 x);

schar convert_char_sat_rtn(schar x);
char2 convert_char2_sat_rtn(char2 x);
char3 convert_char3_sat_rtn(char3 x);
char4 convert_char4_sat_rtn(char4 x);
char8 convert_char8_sat_rtn(char8 x);
char16 convert_char16_sat_rtn(char16 x);

schar convert_char_sat_rtz(schar x);
char2 convert_char2_sat_rtz(char2 x);
char3 convert_char3_sat_rtz(char3 x);
char4 convert_char4_sat_rtz(char4 x);
char8 convert_char8_sat_rtz(char8 x);
char16 convert_char16_sat_rtz(char16 x);

schar convert_char(ushort x);
char2 convert_char2(ushort2 x);
char3 convert_char3(ushort3 x);
char4 convert_char4(ushort4 x);
char8 convert_char8(ushort8 x);
char16 convert_char16(ushort16 x);

schar convert_char_rte(ushort x);
char2 convert_char2_rte(ushort2 x);
char3 convert_char3_rte(ushort3 x);
char4 convert_char4_rte(ushort4 x);
char8 convert_char8_rte(ushort8 x);
char16 convert_char16_rte(ushort16 x);

schar convert_char_rtp(ushort x);
char2 convert_char2_rtp(ushort2 x);
char3 convert_char3_rtp(ushort3 x);
char4 convert_char4_rtp(ushort4 x);
char8 convert_char8_rtp(ushort8 x);
char16 convert_char16_rtp(ushort16 x);

schar convert_char_rtn(ushort x);
char2 convert_char2_rtn(ushort2 x);
char3 convert_char3_rtn(ushort3 x);
char4 convert_char4_rtn(ushort4 x);
char8 convert_char8_rtn(ushort8 x);
char16 convert_char16_rtn(ushort16 x);

schar convert_char_rtz(ushort x);
char2 convert_char2_rtz(ushort2 x);
char3 convert_char3_rtz(ushort3 x);
char4 convert_char4_rtz(ushort4 x);
char8 convert_char8_rtz(ushort8 x);
char16 convert_char16_rtz(ushort16 x);

schar convert_char_sat(ushort x);
char2 convert_char2_sat(ushort2 x);
char3 convert_char3_sat(ushort3 x);
char4 convert_char4_sat(ushort4 x);
char8 convert_char8_sat(ushort8 x);
char16 convert_char16_sat(ushort16 x);

schar convert_char_sat_rte(ushort x);
char2 convert_char2_sat_rte(ushort2 x);
char3 convert_char3_sat_rte(ushort3 x);
char4 convert_char4_sat_rte(ushort4 x);
char8 convert_char8_sat_rte(ushort8 x);
char16 convert_char16_sat_rte(ushort16 x);

schar convert_char_sat_rtp(ushort x);
char2 convert_char2_sat_rtp(ushort2 x);
char3 convert_char3_sat_rtp(ushort3 x);
char4 convert_char4_sat_rtp(ushort4 x);
char8 convert_char8_sat_rtp(ushort8 x);
char16 convert_char16_sat_rtp(ushort16 x);

schar convert_char_sat_rtn(ushort x);
char2 convert_char2_sat_rtn(ushort2 x);
char3 convert_char3_sat_rtn(ushort3 x);
char4 convert_char4_sat_rtn(ushort4 x);
char8 convert_char8_sat_rtn(ushort8 x);
char16 convert_char16_sat_rtn(ushort16 x);

schar convert_char_sat_rtz(ushort x);
char2 convert_char2_sat_rtz(ushort2 x);
char3 convert_char3_sat_rtz(ushort3 x);
char4 convert_char4_sat_rtz(ushort4 x);
char8 convert_char8_sat_rtz(ushort8 x);
char16 convert_char16_sat_rtz(ushort16 x);

schar convert_char(short x);
char2 convert_char2(short2 x);
char3 convert_char3(short3 x);
char4 convert_char4(short4 x);
char8 convert_char8(short8 x);
char16 convert_char16(short16 x);

schar convert_char_rte(short x);
char2 convert_char2_rte(short2 x);
char3 convert_char3_rte(short3 x);
char4 convert_char4_rte(short4 x);
char8 convert_char8_rte(short8 x);
char16 convert_char16_rte(short16 x);

schar convert_char_rtp(short x);
char2 convert_char2_rtp(short2 x);
char3 convert_char3_rtp(short3 x);
char4 convert_char4_rtp(short4 x);
char8 convert_char8_rtp(short8 x);
char16 convert_char16_rtp(short16 x);

schar convert_char_rtn(short x);
char2 convert_char2_rtn(short2 x);
char3 convert_char3_rtn(short3 x);
char4 convert_char4_rtn(short4 x);
char8 convert_char8_rtn(short8 x);
char16 convert_char16_rtn(short16 x);

schar convert_char_rtz(short x);
char2 convert_char2_rtz(short2 x);
char3 convert_char3_rtz(short3 x);
char4 convert_char4_rtz(short4 x);
char8 convert_char8_rtz(short8 x);
char16 convert_char16_rtz(short16 x);

schar convert_char_sat(short x);
char2 convert_char2_sat(short2 x);
char3 convert_char3_sat(short3 x);
char4 convert_char4_sat(short4 x);
char8 convert_char8_sat(short8 x);
char16 convert_char16_sat(short16 x);

schar convert_char_sat_rte(short x);
char2 convert_char2_sat_rte(short2 x);
char3 convert_char3_sat_rte(short3 x);
char4 convert_char4_sat_rte(short4 x);
char8 convert_char8_sat_rte(short8 x);
char16 convert_char16_sat_rte(short16 x);

schar convert_char_sat_rtp(short x);
char2 convert_char2_sat_rtp(short2 x);
char3 convert_char3_sat_rtp(short3 x);
char4 convert_char4_sat_rtp(short4 x);
char8 convert_char8_sat_rtp(short8 x);
char16 convert_char16_sat_rtp(short16 x);

schar convert_char_sat_rtn(short x);
char2 convert_char2_sat_rtn(short2 x);
char3 convert_char3_sat_rtn(short3 x);
char4 convert_char4_sat_rtn(short4 x);
char8 convert_char8_sat_rtn(short8 x);
char16 convert_char16_sat_rtn(short16 x);

schar convert_char_sat_rtz(short x);
char2 convert_char2_sat_rtz(short2 x);
char3 convert_char3_sat_rtz(short3 x);
char4 convert_char4_sat_rtz(short4 x);
char8 convert_char8_sat_rtz(short8 x);
char16 convert_char16_sat_rtz(short16 x);

schar convert_char(uint x);
char2 convert_char2(uint2 x);
char3 convert_char3(uint3 x);
char4 convert_char4(uint4 x);
char8 convert_char8(uint8 x);
char16 convert_char16(uint16 x);

schar convert_char_rte(uint x);
char2 convert_char2_rte(uint2 x);
char3 convert_char3_rte(uint3 x);
char4 convert_char4_rte(uint4 x);
char8 convert_char8_rte(uint8 x);
char16 convert_char16_rte(uint16 x);

schar convert_char_rtp(uint x);
char2 convert_char2_rtp(uint2 x);
char3 convert_char3_rtp(uint3 x);
char4 convert_char4_rtp(uint4 x);
char8 convert_char8_rtp(uint8 x);
char16 convert_char16_rtp(uint16 x);

schar convert_char_rtn(uint x);
char2 convert_char2_rtn(uint2 x);
char3 convert_char3_rtn(uint3 x);
char4 convert_char4_rtn(uint4 x);
char8 convert_char8_rtn(uint8 x);
char16 convert_char16_rtn(uint16 x);

schar convert_char_rtz(uint x);
char2 convert_char2_rtz(uint2 x);
char3 convert_char3_rtz(uint3 x);
char4 convert_char4_rtz(uint4 x);
char8 convert_char8_rtz(uint8 x);
char16 convert_char16_rtz(uint16 x);

schar convert_char_sat(uint x);
char2 convert_char2_sat(uint2 x);
char3 convert_char3_sat(uint3 x);
char4 convert_char4_sat(uint4 x);
char8 convert_char8_sat(uint8 x);
char16 convert_char16_sat(uint16 x);

schar convert_char_sat_rte(uint x);
char2 convert_char2_sat_rte(uint2 x);
char3 convert_char3_sat_rte(uint3 x);
char4 convert_char4_sat_rte(uint4 x);
char8 convert_char8_sat_rte(uint8 x);
char16 convert_char16_sat_rte(uint16 x);

schar convert_char_sat_rtp(uint x);
char2 convert_char2_sat_rtp(uint2 x);
char3 convert_char3_sat_rtp(uint3 x);
char4 convert_char4_sat_rtp(uint4 x);
char8 convert_char8_sat_rtp(uint8 x);
char16 convert_char16_sat_rtp(uint16 x);

schar convert_char_sat_rtn(uint x);
char2 convert_char2_sat_rtn(uint2 x);
char3 convert_char3_sat_rtn(uint3 x);
char4 convert_char4_sat_rtn(uint4 x);
char8 convert_char8_sat_rtn(uint8 x);
char16 convert_char16_sat_rtn(uint16 x);

schar convert_char_sat_rtz(uint x);
char2 convert_char2_sat_rtz(uint2 x);
char3 convert_char3_sat_rtz(uint3 x);
char4 convert_char4_sat_rtz(uint4 x);
char8 convert_char8_sat_rtz(uint8 x);
char16 convert_char16_sat_rtz(uint16 x);

schar convert_char(int x);
char2 convert_char2(int2 x);
char3 convert_char3(int3 x);
char4 convert_char4(int4 x);
char8 convert_char8(int8 x);
char16 convert_char16(int16 x);

schar convert_char_rte(int x);
char2 convert_char2_rte(int2 x);
char3 convert_char3_rte(int3 x);
char4 convert_char4_rte(int4 x);
char8 convert_char8_rte(int8 x);
char16 convert_char16_rte(int16 x);

schar convert_char_rtp(int x);
char2 convert_char2_rtp(int2 x);
char3 convert_char3_rtp(int3 x);
char4 convert_char4_rtp(int4 x);
char8 convert_char8_rtp(int8 x);
char16 convert_char16_rtp(int16 x);

schar convert_char_rtn(int x);
char2 convert_char2_rtn(int2 x);
char3 convert_char3_rtn(int3 x);
char4 convert_char4_rtn(int4 x);
char8 convert_char8_rtn(int8 x);
char16 convert_char16_rtn(int16 x);

schar convert_char_rtz(int x);
char2 convert_char2_rtz(int2 x);
char3 convert_char3_rtz(int3 x);
char4 convert_char4_rtz(int4 x);
char8 convert_char8_rtz(int8 x);
char16 convert_char16_rtz(int16 x);

schar convert_char_sat(int x);
char2 convert_char2_sat(int2 x);
char3 convert_char3_sat(int3 x);
char4 convert_char4_sat(int4 x);
char8 convert_char8_sat(int8 x);
char16 convert_char16_sat(int16 x);

schar convert_char_sat_rte(int x);
char2 convert_char2_sat_rte(int2 x);
char3 convert_char3_sat_rte(int3 x);
char4 convert_char4_sat_rte(int4 x);
char8 convert_char8_sat_rte(int8 x);
char16 convert_char16_sat_rte(int16 x);

schar convert_char_sat_rtp(int x);
char2 convert_char2_sat_rtp(int2 x);
char3 convert_char3_sat_rtp(int3 x);
char4 convert_char4_sat_rtp(int4 x);
char8 convert_char8_sat_rtp(int8 x);
char16 convert_char16_sat_rtp(int16 x);

schar convert_char_sat_rtn(int x);
char2 convert_char2_sat_rtn(int2 x);
char3 convert_char3_sat_rtn(int3 x);
char4 convert_char4_sat_rtn(int4 x);
char8 convert_char8_sat_rtn(int8 x);
char16 convert_char16_sat_rtn(int16 x);

schar convert_char_sat_rtz(int x);
char2 convert_char2_sat_rtz(int2 x);
char3 convert_char3_sat_rtz(int3 x);
char4 convert_char4_sat_rtz(int4 x);
char8 convert_char8_sat_rtz(int8 x);
char16 convert_char16_sat_rtz(int16 x);

schar convert_char(float x);
char2 convert_char2(float2 x);
char3 convert_char3(float3 x);
char4 convert_char4(float4 x);
char8 convert_char8(float8 x);
char16 convert_char16(float16 x);

schar convert_char_rte(float x);
char2 convert_char2_rte(float2 x);
char3 convert_char3_rte(float3 x);
char4 convert_char4_rte(float4 x);
char8 convert_char8_rte(float8 x);
char16 convert_char16_rte(float16 x);

schar convert_char_rtp(float x);
char2 convert_char2_rtp(float2 x);
char3 convert_char3_rtp(float3 x);
char4 convert_char4_rtp(float4 x);
char8 convert_char8_rtp(float8 x);
char16 convert_char16_rtp(float16 x);

schar convert_char_rtn(float x);
char2 convert_char2_rtn(float2 x);
char3 convert_char3_rtn(float3 x);
char4 convert_char4_rtn(float4 x);
char8 convert_char8_rtn(float8 x);
char16 convert_char16_rtn(float16 x);

schar convert_char_rtz(float x);
char2 convert_char2_rtz(float2 x);
char3 convert_char3_rtz(float3 x);
char4 convert_char4_rtz(float4 x);
char8 convert_char8_rtz(float8 x);
char16 convert_char16_rtz(float16 x);

schar convert_char_sat(float x);
char2 convert_char2_sat(float2 x);
char3 convert_char3_sat(float3 x);
char4 convert_char4_sat(float4 x);
char8 convert_char8_sat(float8 x);
char16 convert_char16_sat(float16 x);

schar convert_char_sat_rte(float x);
char2 convert_char2_sat_rte(float2 x);
char3 convert_char3_sat_rte(float3 x);
char4 convert_char4_sat_rte(float4 x);
char8 convert_char8_sat_rte(float8 x);
char16 convert_char16_sat_rte(float16 x);

schar convert_char_sat_rtp(float x);
char2 convert_char2_sat_rtp(float2 x);
char3 convert_char3_sat_rtp(float3 x);
char4 convert_char4_sat_rtp(float4 x);
char8 convert_char8_sat_rtp(float8 x);
char16 convert_char16_sat_rtp(float16 x);

schar convert_char_sat_rtn(float x);
char2 convert_char2_sat_rtn(float2 x);
char3 convert_char3_sat_rtn(float3 x);
char4 convert_char4_sat_rtn(float4 x);
char8 convert_char8_sat_rtn(float8 x);
char16 convert_char16_sat_rtn(float16 x);

schar convert_char_sat_rtz(float x);
char2 convert_char2_sat_rtz(float2 x);
char3 convert_char3_sat_rtz(float3 x);
char4 convert_char4_sat_rtz(float4 x);
char8 convert_char8_sat_rtz(float8 x);
char16 convert_char16_sat_rtz(float16 x);

schar convert_char(double x);
char2 convert_char2(double2 x);
char3 convert_char3(double3 x);
char4 convert_char4(double4 x);
char8 convert_char8(double8 x);
char16 convert_char16(double16 x);

schar convert_char_rte(double x);
char2 convert_char2_rte(double2 x);
char3 convert_char3_rte(double3 x);
char4 convert_char4_rte(double4 x);
char8 convert_char8_rte(double8 x);
char16 convert_char16_rte(double16 x);

schar convert_char_rtp(double x);
char2 convert_char2_rtp(double2 x);
char3 convert_char3_rtp(double3 x);
char4 convert_char4_rtp(double4 x);
char8 convert_char8_rtp(double8 x);
char16 convert_char16_rtp(double16 x);

schar convert_char_rtn(double x);
char2 convert_char2_rtn(double2 x);
char3 convert_char3_rtn(double3 x);
char4 convert_char4_rtn(double4 x);
char8 convert_char8_rtn(double8 x);
char16 convert_char16_rtn(double16 x);

schar convert_char_rtz(double x);
char2 convert_char2_rtz(double2 x);
char3 convert_char3_rtz(double3 x);
char4 convert_char4_rtz(double4 x);
char8 convert_char8_rtz(double8 x);
char16 convert_char16_rtz(double16 x);

schar convert_char_sat(double x);
char2 convert_char2_sat(double2 x);
char3 convert_char3_sat(double3 x);
char4 convert_char4_sat(double4 x);
char8 convert_char8_sat(double8 x);
char16 convert_char16_sat(double16 x);

schar convert_char_sat_rte(double x);
char2 convert_char2_sat_rte(double2 x);
char3 convert_char3_sat_rte(double3 x);
char4 convert_char4_sat_rte(double4 x);
char8 convert_char8_sat_rte(double8 x);
char16 convert_char16_sat_rte(double16 x);

schar convert_char_sat_rtp(double x);
char2 convert_char2_sat_rtp(double2 x);
char3 convert_char3_sat_rtp(double3 x);
char4 convert_char4_sat_rtp(double4 x);
char8 convert_char8_sat_rtp(double8 x);
char16 convert_char16_sat_rtp(double16 x);

schar convert_char_sat_rtn(double x);
char2 convert_char2_sat_rtn(double2 x);
char3 convert_char3_sat_rtn(double3 x);
char4 convert_char4_sat_rtn(double4 x);
char8 convert_char8_sat_rtn(double8 x);
char16 convert_char16_sat_rtn(double16 x);

schar convert_char_sat_rtz(double x);
char2 convert_char2_sat_rtz(double2 x);
char3 convert_char3_sat_rtz(double3 x);
char4 convert_char4_sat_rtz(double4 x);
char8 convert_char8_sat_rtz(double8 x);
char16 convert_char16_sat_rtz(double16 x);

schar convert_char(ullong x);
char2 convert_char2(ulong2 x);
char3 convert_char3(ulong3 x);
char4 convert_char4(ulong4 x);
char8 convert_char8(ulong8 x);
char16 convert_char16(ulong16 x);

schar convert_char_rte(ullong x);
char2 convert_char2_rte(ulong2 x);
char3 convert_char3_rte(ulong3 x);
char4 convert_char4_rte(ulong4 x);
char8 convert_char8_rte(ulong8 x);
char16 convert_char16_rte(ulong16 x);

schar convert_char_rtp(ullong x);
char2 convert_char2_rtp(ulong2 x);
char3 convert_char3_rtp(ulong3 x);
char4 convert_char4_rtp(ulong4 x);
char8 convert_char8_rtp(ulong8 x);
char16 convert_char16_rtp(ulong16 x);

schar convert_char_rtn(ullong x);
char2 convert_char2_rtn(ulong2 x);
char3 convert_char3_rtn(ulong3 x);
char4 convert_char4_rtn(ulong4 x);
char8 convert_char8_rtn(ulong8 x);
char16 convert_char16_rtn(ulong16 x);

schar convert_char_rtz(ullong x);
char2 convert_char2_rtz(ulong2 x);
char3 convert_char3_rtz(ulong3 x);
char4 convert_char4_rtz(ulong4 x);
char8 convert_char8_rtz(ulong8 x);
char16 convert_char16_rtz(ulong16 x);

schar convert_char_sat(ullong x);
char2 convert_char2_sat(ulong2 x);
char3 convert_char3_sat(ulong3 x);
char4 convert_char4_sat(ulong4 x);
char8 convert_char8_sat(ulong8 x);
char16 convert_char16_sat(ulong16 x);

schar convert_char_sat_rte(ullong x);
char2 convert_char2_sat_rte(ulong2 x);
char3 convert_char3_sat_rte(ulong3 x);
char4 convert_char4_sat_rte(ulong4 x);
char8 convert_char8_sat_rte(ulong8 x);
char16 convert_char16_sat_rte(ulong16 x);

schar convert_char_sat_rtp(ullong x);
char2 convert_char2_sat_rtp(ulong2 x);
char3 convert_char3_sat_rtp(ulong3 x);
char4 convert_char4_sat_rtp(ulong4 x);
char8 convert_char8_sat_rtp(ulong8 x);
char16 convert_char16_sat_rtp(ulong16 x);

schar convert_char_sat_rtn(ullong x);
char2 convert_char2_sat_rtn(ulong2 x);
char3 convert_char3_sat_rtn(ulong3 x);
char4 convert_char4_sat_rtn(ulong4 x);
char8 convert_char8_sat_rtn(ulong8 x);
char16 convert_char16_sat_rtn(ulong16 x);

schar convert_char_sat_rtz(ullong x);
char2 convert_char2_sat_rtz(ulong2 x);
char3 convert_char3_sat_rtz(ulong3 x);
char4 convert_char4_sat_rtz(ulong4 x);
char8 convert_char8_sat_rtz(ulong8 x);
char16 convert_char16_sat_rtz(ulong16 x);

schar convert_char(llong x);
char2 convert_char2(long2 x);
char3 convert_char3(long3 x);
char4 convert_char4(long4 x);
char8 convert_char8(long8 x);
char16 convert_char16(long16 x);

schar convert_char_rte(llong x);
char2 convert_char2_rte(long2 x);
char3 convert_char3_rte(long3 x);
char4 convert_char4_rte(long4 x);
char8 convert_char8_rte(long8 x);
char16 convert_char16_rte(long16 x);

schar convert_char_rtp(llong x);
char2 convert_char2_rtp(long2 x);
char3 convert_char3_rtp(long3 x);
char4 convert_char4_rtp(long4 x);
char8 convert_char8_rtp(long8 x);
char16 convert_char16_rtp(long16 x);

schar convert_char_rtn(llong x);
char2 convert_char2_rtn(long2 x);
char3 convert_char3_rtn(long3 x);
char4 convert_char4_rtn(long4 x);
char8 convert_char8_rtn(long8 x);
char16 convert_char16_rtn(long16 x);

schar convert_char_rtz(llong x);
char2 convert_char2_rtz(long2 x);
char3 convert_char3_rtz(long3 x);
char4 convert_char4_rtz(long4 x);
char8 convert_char8_rtz(long8 x);
char16 convert_char16_rtz(long16 x);

schar convert_char_sat(llong x);
char2 convert_char2_sat(long2 x);
char3 convert_char3_sat(long3 x);
char4 convert_char4_sat(long4 x);
char8 convert_char8_sat(long8 x);
char16 convert_char16_sat(long16 x);

schar convert_char_sat_rte(llong x);
char2 convert_char2_sat_rte(long2 x);
char3 convert_char3_sat_rte(long3 x);
char4 convert_char4_sat_rte(long4 x);
char8 convert_char8_sat_rte(long8 x);
char16 convert_char16_sat_rte(long16 x);

schar convert_char_sat_rtp(llong x);
char2 convert_char2_sat_rtp(long2 x);
char3 convert_char3_sat_rtp(long3 x);
char4 convert_char4_sat_rtp(long4 x);
char8 convert_char8_sat_rtp(long8 x);
char16 convert_char16_sat_rtp(long16 x);

schar convert_char_sat_rtn(llong x);
char2 convert_char2_sat_rtn(long2 x);
char3 convert_char3_sat_rtn(long3 x);
char4 convert_char4_sat_rtn(long4 x);
char8 convert_char8_sat_rtn(long8 x);
char16 convert_char16_sat_rtn(long16 x);

schar convert_char_sat_rtz(llong x);
char2 convert_char2_sat_rtz(long2 x);
char3 convert_char3_sat_rtz(long3 x);
char4 convert_char4_sat_rtz(long4 x);
char8 convert_char8_sat_rtz(long8 x);
char16 convert_char16_sat_rtz(long16 x);

ushort convert_ushort(uchar x);
ushort2 convert_ushort2(uchar2 x);
ushort3 convert_ushort3(uchar3 x);
ushort4 convert_ushort4(uchar4 x);
ushort8 convert_ushort8(uchar8 x);
ushort16 convert_ushort16(uchar16 x);

ushort convert_ushort_rte(uchar x);
ushort2 convert_ushort2_rte(uchar2 x);
ushort3 convert_ushort3_rte(uchar3 x);
ushort4 convert_ushort4_rte(uchar4 x);
ushort8 convert_ushort8_rte(uchar8 x);
ushort16 convert_ushort16_rte(uchar16 x);

ushort convert_ushort_rtp(uchar x);
ushort2 convert_ushort2_rtp(uchar2 x);
ushort3 convert_ushort3_rtp(uchar3 x);
ushort4 convert_ushort4_rtp(uchar4 x);
ushort8 convert_ushort8_rtp(uchar8 x);
ushort16 convert_ushort16_rtp(uchar16 x);

ushort convert_ushort_rtn(uchar x);
ushort2 convert_ushort2_rtn(uchar2 x);
ushort3 convert_ushort3_rtn(uchar3 x);
ushort4 convert_ushort4_rtn(uchar4 x);
ushort8 convert_ushort8_rtn(uchar8 x);
ushort16 convert_ushort16_rtn(uchar16 x);

ushort convert_ushort_rtz(uchar x);
ushort2 convert_ushort2_rtz(uchar2 x);
ushort3 convert_ushort3_rtz(uchar3 x);
ushort4 convert_ushort4_rtz(uchar4 x);
ushort8 convert_ushort8_rtz(uchar8 x);
ushort16 convert_ushort16_rtz(uchar16 x);

ushort convert_ushort_sat(uchar x);
ushort2 convert_ushort2_sat(uchar2 x);
ushort3 convert_ushort3_sat(uchar3 x);
ushort4 convert_ushort4_sat(uchar4 x);
ushort8 convert_ushort8_sat(uchar8 x);
ushort16 convert_ushort16_sat(uchar16 x);

ushort convert_ushort_sat_rte(uchar x);
ushort2 convert_ushort2_sat_rte(uchar2 x);
ushort3 convert_ushort3_sat_rte(uchar3 x);
ushort4 convert_ushort4_sat_rte(uchar4 x);
ushort8 convert_ushort8_sat_rte(uchar8 x);
ushort16 convert_ushort16_sat_rte(uchar16 x);

ushort convert_ushort_sat_rtp(uchar x);
ushort2 convert_ushort2_sat_rtp(uchar2 x);
ushort3 convert_ushort3_sat_rtp(uchar3 x);
ushort4 convert_ushort4_sat_rtp(uchar4 x);
ushort8 convert_ushort8_sat_rtp(uchar8 x);
ushort16 convert_ushort16_sat_rtp(uchar16 x);

ushort convert_ushort_sat_rtn(uchar x);
ushort2 convert_ushort2_sat_rtn(uchar2 x);
ushort3 convert_ushort3_sat_rtn(uchar3 x);
ushort4 convert_ushort4_sat_rtn(uchar4 x);
ushort8 convert_ushort8_sat_rtn(uchar8 x);
ushort16 convert_ushort16_sat_rtn(uchar16 x);

ushort convert_ushort_sat_rtz(uchar x);
ushort2 convert_ushort2_sat_rtz(uchar2 x);
ushort3 convert_ushort3_sat_rtz(uchar3 x);
ushort4 convert_ushort4_sat_rtz(uchar4 x);
ushort8 convert_ushort8_sat_rtz(uchar8 x);
ushort16 convert_ushort16_sat_rtz(uchar16 x);

ushort convert_ushort(schar x);
ushort2 convert_ushort2(char2 x);
ushort3 convert_ushort3(char3 x);
ushort4 convert_ushort4(char4 x);
ushort8 convert_ushort8(char8 x);
ushort16 convert_ushort16(char16 x);

ushort convert_ushort_rte(schar x);
ushort2 convert_ushort2_rte(char2 x);
ushort3 convert_ushort3_rte(char3 x);
ushort4 convert_ushort4_rte(char4 x);
ushort8 convert_ushort8_rte(char8 x);
ushort16 convert_ushort16_rte(char16 x);

ushort convert_ushort_rtp(schar x);
ushort2 convert_ushort2_rtp(char2 x);
ushort3 convert_ushort3_rtp(char3 x);
ushort4 convert_ushort4_rtp(char4 x);
ushort8 convert_ushort8_rtp(char8 x);
ushort16 convert_ushort16_rtp(char16 x);

ushort convert_ushort_rtn(schar x);
ushort2 convert_ushort2_rtn(char2 x);
ushort3 convert_ushort3_rtn(char3 x);
ushort4 convert_ushort4_rtn(char4 x);
ushort8 convert_ushort8_rtn(char8 x);
ushort16 convert_ushort16_rtn(char16 x);

ushort convert_ushort_rtz(schar x);
ushort2 convert_ushort2_rtz(char2 x);
ushort3 convert_ushort3_rtz(char3 x);
ushort4 convert_ushort4_rtz(char4 x);
ushort8 convert_ushort8_rtz(char8 x);
ushort16 convert_ushort16_rtz(char16 x);

ushort convert_ushort_sat(schar x);
ushort2 convert_ushort2_sat(char2 x);
ushort3 convert_ushort3_sat(char3 x);
ushort4 convert_ushort4_sat(char4 x);
ushort8 convert_ushort8_sat(char8 x);
ushort16 convert_ushort16_sat(char16 x);

ushort convert_ushort_sat_rte(schar x);
ushort2 convert_ushort2_sat_rte(char2 x);
ushort3 convert_ushort3_sat_rte(char3 x);
ushort4 convert_ushort4_sat_rte(char4 x);
ushort8 convert_ushort8_sat_rte(char8 x);
ushort16 convert_ushort16_sat_rte(char16 x);

ushort convert_ushort_sat_rtp(schar x);
ushort2 convert_ushort2_sat_rtp(char2 x);
ushort3 convert_ushort3_sat_rtp(char3 x);
ushort4 convert_ushort4_sat_rtp(char4 x);
ushort8 convert_ushort8_sat_rtp(char8 x);
ushort16 convert_ushort16_sat_rtp(char16 x);

ushort convert_ushort_sat_rtn(schar x);
ushort2 convert_ushort2_sat_rtn(char2 x);
ushort3 convert_ushort3_sat_rtn(char3 x);
ushort4 convert_ushort4_sat_rtn(char4 x);
ushort8 convert_ushort8_sat_rtn(char8 x);
ushort16 convert_ushort16_sat_rtn(char16 x);

ushort convert_ushort_sat_rtz(schar x);
ushort2 convert_ushort2_sat_rtz(char2 x);
ushort3 convert_ushort3_sat_rtz(char3 x);
ushort4 convert_ushort4_sat_rtz(char4 x);
ushort8 convert_ushort8_sat_rtz(char8 x);
ushort16 convert_ushort16_sat_rtz(char16 x);

ushort convert_ushort(ushort x);
ushort2 convert_ushort2(ushort2 x);
ushort3 convert_ushort3(ushort3 x);
ushort4 convert_ushort4(ushort4 x);
ushort8 convert_ushort8(ushort8 x);
ushort16 convert_ushort16(ushort16 x);

ushort convert_ushort_rte(ushort x);
ushort2 convert_ushort2_rte(ushort2 x);
ushort3 convert_ushort3_rte(ushort3 x);
ushort4 convert_ushort4_rte(ushort4 x);
ushort8 convert_ushort8_rte(ushort8 x);
ushort16 convert_ushort16_rte(ushort16 x);

ushort convert_ushort_rtp(ushort x);
ushort2 convert_ushort2_rtp(ushort2 x);
ushort3 convert_ushort3_rtp(ushort3 x);
ushort4 convert_ushort4_rtp(ushort4 x);
ushort8 convert_ushort8_rtp(ushort8 x);
ushort16 convert_ushort16_rtp(ushort16 x);

ushort convert_ushort_rtn(ushort x);
ushort2 convert_ushort2_rtn(ushort2 x);
ushort3 convert_ushort3_rtn(ushort3 x);
ushort4 convert_ushort4_rtn(ushort4 x);
ushort8 convert_ushort8_rtn(ushort8 x);
ushort16 convert_ushort16_rtn(ushort16 x);

ushort convert_ushort_rtz(ushort x);
ushort2 convert_ushort2_rtz(ushort2 x);
ushort3 convert_ushort3_rtz(ushort3 x);
ushort4 convert_ushort4_rtz(ushort4 x);
ushort8 convert_ushort8_rtz(ushort8 x);
ushort16 convert_ushort16_rtz(ushort16 x);

ushort convert_ushort_sat(ushort x);
ushort2 convert_ushort2_sat(ushort2 x);
ushort3 convert_ushort3_sat(ushort3 x);
ushort4 convert_ushort4_sat(ushort4 x);
ushort8 convert_ushort8_sat(ushort8 x);
ushort16 convert_ushort16_sat(ushort16 x);

ushort convert_ushort_sat_rte(ushort x);
ushort2 convert_ushort2_sat_rte(ushort2 x);
ushort3 convert_ushort3_sat_rte(ushort3 x);
ushort4 convert_ushort4_sat_rte(ushort4 x);
ushort8 convert_ushort8_sat_rte(ushort8 x);
ushort16 convert_ushort16_sat_rte(ushort16 x);

ushort convert_ushort_sat_rtp(ushort x);
ushort2 convert_ushort2_sat_rtp(ushort2 x);
ushort3 convert_ushort3_sat_rtp(ushort3 x);
ushort4 convert_ushort4_sat_rtp(ushort4 x);
ushort8 convert_ushort8_sat_rtp(ushort8 x);
ushort16 convert_ushort16_sat_rtp(ushort16 x);

ushort convert_ushort_sat_rtn(ushort x);
ushort2 convert_ushort2_sat_rtn(ushort2 x);
ushort3 convert_ushort3_sat_rtn(ushort3 x);
ushort4 convert_ushort4_sat_rtn(ushort4 x);
ushort8 convert_ushort8_sat_rtn(ushort8 x);
ushort16 convert_ushort16_sat_rtn(ushort16 x);

ushort convert_ushort_sat_rtz(ushort x);
ushort2 convert_ushort2_sat_rtz(ushort2 x);
ushort3 convert_ushort3_sat_rtz(ushort3 x);
ushort4 convert_ushort4_sat_rtz(ushort4 x);
ushort8 convert_ushort8_sat_rtz(ushort8 x);
ushort16 convert_ushort16_sat_rtz(ushort16 x);

ushort convert_ushort(short x);
ushort2 convert_ushort2(short2 x);
ushort3 convert_ushort3(short3 x);
ushort4 convert_ushort4(short4 x);
ushort8 convert_ushort8(short8 x);
ushort16 convert_ushort16(short16 x);

ushort convert_ushort_rte(short x);
ushort2 convert_ushort2_rte(short2 x);
ushort3 convert_ushort3_rte(short3 x);
ushort4 convert_ushort4_rte(short4 x);
ushort8 convert_ushort8_rte(short8 x);
ushort16 convert_ushort16_rte(short16 x);

ushort convert_ushort_rtp(short x);
ushort2 convert_ushort2_rtp(short2 x);
ushort3 convert_ushort3_rtp(short3 x);
ushort4 convert_ushort4_rtp(short4 x);
ushort8 convert_ushort8_rtp(short8 x);
ushort16 convert_ushort16_rtp(short16 x);

ushort convert_ushort_rtn(short x);
ushort2 convert_ushort2_rtn(short2 x);
ushort3 convert_ushort3_rtn(short3 x);
ushort4 convert_ushort4_rtn(short4 x);
ushort8 convert_ushort8_rtn(short8 x);
ushort16 convert_ushort16_rtn(short16 x);

ushort convert_ushort_rtz(short x);
ushort2 convert_ushort2_rtz(short2 x);
ushort3 convert_ushort3_rtz(short3 x);
ushort4 convert_ushort4_rtz(short4 x);
ushort8 convert_ushort8_rtz(short8 x);
ushort16 convert_ushort16_rtz(short16 x);

ushort convert_ushort_sat(short x);
ushort2 convert_ushort2_sat(short2 x);
ushort3 convert_ushort3_sat(short3 x);
ushort4 convert_ushort4_sat(short4 x);
ushort8 convert_ushort8_sat(short8 x);
ushort16 convert_ushort16_sat(short16 x);

ushort convert_ushort_sat_rte(short x);
ushort2 convert_ushort2_sat_rte(short2 x);
ushort3 convert_ushort3_sat_rte(short3 x);
ushort4 convert_ushort4_sat_rte(short4 x);
ushort8 convert_ushort8_sat_rte(short8 x);
ushort16 convert_ushort16_sat_rte(short16 x);

ushort convert_ushort_sat_rtp(short x);
ushort2 convert_ushort2_sat_rtp(short2 x);
ushort3 convert_ushort3_sat_rtp(short3 x);
ushort4 convert_ushort4_sat_rtp(short4 x);
ushort8 convert_ushort8_sat_rtp(short8 x);
ushort16 convert_ushort16_sat_rtp(short16 x);

ushort convert_ushort_sat_rtn(short x);
ushort2 convert_ushort2_sat_rtn(short2 x);
ushort3 convert_ushort3_sat_rtn(short3 x);
ushort4 convert_ushort4_sat_rtn(short4 x);
ushort8 convert_ushort8_sat_rtn(short8 x);
ushort16 convert_ushort16_sat_rtn(short16 x);

ushort convert_ushort_sat_rtz(short x);
ushort2 convert_ushort2_sat_rtz(short2 x);
ushort3 convert_ushort3_sat_rtz(short3 x);
ushort4 convert_ushort4_sat_rtz(short4 x);
ushort8 convert_ushort8_sat_rtz(short8 x);
ushort16 convert_ushort16_sat_rtz(short16 x);

ushort convert_ushort(uint x);
ushort2 convert_ushort2(uint2 x);
ushort3 convert_ushort3(uint3 x);
ushort4 convert_ushort4(uint4 x);
ushort8 convert_ushort8(uint8 x);
ushort16 convert_ushort16(uint16 x);

ushort convert_ushort_rte(uint x);
ushort2 convert_ushort2_rte(uint2 x);
ushort3 convert_ushort3_rte(uint3 x);
ushort4 convert_ushort4_rte(uint4 x);
ushort8 convert_ushort8_rte(uint8 x);
ushort16 convert_ushort16_rte(uint16 x);

ushort convert_ushort_rtp(uint x);
ushort2 convert_ushort2_rtp(uint2 x);
ushort3 convert_ushort3_rtp(uint3 x);
ushort4 convert_ushort4_rtp(uint4 x);
ushort8 convert_ushort8_rtp(uint8 x);
ushort16 convert_ushort16_rtp(uint16 x);

ushort convert_ushort_rtn(uint x);
ushort2 convert_ushort2_rtn(uint2 x);
ushort3 convert_ushort3_rtn(uint3 x);
ushort4 convert_ushort4_rtn(uint4 x);
ushort8 convert_ushort8_rtn(uint8 x);
ushort16 convert_ushort16_rtn(uint16 x);

ushort convert_ushort_rtz(uint x);
ushort2 convert_ushort2_rtz(uint2 x);
ushort3 convert_ushort3_rtz(uint3 x);
ushort4 convert_ushort4_rtz(uint4 x);
ushort8 convert_ushort8_rtz(uint8 x);
ushort16 convert_ushort16_rtz(uint16 x);

ushort convert_ushort_sat(uint x);
ushort2 convert_ushort2_sat(uint2 x);
ushort3 convert_ushort3_sat(uint3 x);
ushort4 convert_ushort4_sat(uint4 x);
ushort8 convert_ushort8_sat(uint8 x);
ushort16 convert_ushort16_sat(uint16 x);

ushort convert_ushort_sat_rte(uint x);
ushort2 convert_ushort2_sat_rte(uint2 x);
ushort3 convert_ushort3_sat_rte(uint3 x);
ushort4 convert_ushort4_sat_rte(uint4 x);
ushort8 convert_ushort8_sat_rte(uint8 x);
ushort16 convert_ushort16_sat_rte(uint16 x);

ushort convert_ushort_sat_rtp(uint x);
ushort2 convert_ushort2_sat_rtp(uint2 x);
ushort3 convert_ushort3_sat_rtp(uint3 x);
ushort4 convert_ushort4_sat_rtp(uint4 x);
ushort8 convert_ushort8_sat_rtp(uint8 x);
ushort16 convert_ushort16_sat_rtp(uint16 x);

ushort convert_ushort_sat_rtn(uint x);
ushort2 convert_ushort2_sat_rtn(uint2 x);
ushort3 convert_ushort3_sat_rtn(uint3 x);
ushort4 convert_ushort4_sat_rtn(uint4 x);
ushort8 convert_ushort8_sat_rtn(uint8 x);
ushort16 convert_ushort16_sat_rtn(uint16 x);

ushort convert_ushort_sat_rtz(uint x);
ushort2 convert_ushort2_sat_rtz(uint2 x);
ushort3 convert_ushort3_sat_rtz(uint3 x);
ushort4 convert_ushort4_sat_rtz(uint4 x);
ushort8 convert_ushort8_sat_rtz(uint8 x);
ushort16 convert_ushort16_sat_rtz(uint16 x);

ushort convert_ushort(int x);
ushort2 convert_ushort2(int2 x);
ushort3 convert_ushort3(int3 x);
ushort4 convert_ushort4(int4 x);
ushort8 convert_ushort8(int8 x);
ushort16 convert_ushort16(int16 x);

ushort convert_ushort_rte(int x);
ushort2 convert_ushort2_rte(int2 x);
ushort3 convert_ushort3_rte(int3 x);
ushort4 convert_ushort4_rte(int4 x);
ushort8 convert_ushort8_rte(int8 x);
ushort16 convert_ushort16_rte(int16 x);

ushort convert_ushort_rtp(int x);
ushort2 convert_ushort2_rtp(int2 x);
ushort3 convert_ushort3_rtp(int3 x);
ushort4 convert_ushort4_rtp(int4 x);
ushort8 convert_ushort8_rtp(int8 x);
ushort16 convert_ushort16_rtp(int16 x);

ushort convert_ushort_rtn(int x);
ushort2 convert_ushort2_rtn(int2 x);
ushort3 convert_ushort3_rtn(int3 x);
ushort4 convert_ushort4_rtn(int4 x);
ushort8 convert_ushort8_rtn(int8 x);
ushort16 convert_ushort16_rtn(int16 x);

ushort convert_ushort_rtz(int x);
ushort2 convert_ushort2_rtz(int2 x);
ushort3 convert_ushort3_rtz(int3 x);
ushort4 convert_ushort4_rtz(int4 x);
ushort8 convert_ushort8_rtz(int8 x);
ushort16 convert_ushort16_rtz(int16 x);

ushort convert_ushort_sat(int x);
ushort2 convert_ushort2_sat(int2 x);
ushort3 convert_ushort3_sat(int3 x);
ushort4 convert_ushort4_sat(int4 x);
ushort8 convert_ushort8_sat(int8 x);
ushort16 convert_ushort16_sat(int16 x);

ushort convert_ushort_sat_rte(int x);
ushort2 convert_ushort2_sat_rte(int2 x);
ushort3 convert_ushort3_sat_rte(int3 x);
ushort4 convert_ushort4_sat_rte(int4 x);
ushort8 convert_ushort8_sat_rte(int8 x);
ushort16 convert_ushort16_sat_rte(int16 x);

ushort convert_ushort_sat_rtp(int x);
ushort2 convert_ushort2_sat_rtp(int2 x);
ushort3 convert_ushort3_sat_rtp(int3 x);
ushort4 convert_ushort4_sat_rtp(int4 x);
ushort8 convert_ushort8_sat_rtp(int8 x);
ushort16 convert_ushort16_sat_rtp(int16 x);

ushort convert_ushort_sat_rtn(int x);
ushort2 convert_ushort2_sat_rtn(int2 x);
ushort3 convert_ushort3_sat_rtn(int3 x);
ushort4 convert_ushort4_sat_rtn(int4 x);
ushort8 convert_ushort8_sat_rtn(int8 x);
ushort16 convert_ushort16_sat_rtn(int16 x);

ushort convert_ushort_sat_rtz(int x);
ushort2 convert_ushort2_sat_rtz(int2 x);
ushort3 convert_ushort3_sat_rtz(int3 x);
ushort4 convert_ushort4_sat_rtz(int4 x);
ushort8 convert_ushort8_sat_rtz(int8 x);
ushort16 convert_ushort16_sat_rtz(int16 x);

ushort convert_ushort(float x);
ushort2 convert_ushort2(float2 x);
ushort3 convert_ushort3(float3 x);
ushort4 convert_ushort4(float4 x);
ushort8 convert_ushort8(float8 x);
ushort16 convert_ushort16(float16 x);

ushort convert_ushort_rte(float x);
ushort2 convert_ushort2_rte(float2 x);
ushort3 convert_ushort3_rte(float3 x);
ushort4 convert_ushort4_rte(float4 x);
ushort8 convert_ushort8_rte(float8 x);
ushort16 convert_ushort16_rte(float16 x);

ushort convert_ushort_rtp(float x);
ushort2 convert_ushort2_rtp(float2 x);
ushort3 convert_ushort3_rtp(float3 x);
ushort4 convert_ushort4_rtp(float4 x);
ushort8 convert_ushort8_rtp(float8 x);
ushort16 convert_ushort16_rtp(float16 x);

ushort convert_ushort_rtn(float x);
ushort2 convert_ushort2_rtn(float2 x);
ushort3 convert_ushort3_rtn(float3 x);
ushort4 convert_ushort4_rtn(float4 x);
ushort8 convert_ushort8_rtn(float8 x);
ushort16 convert_ushort16_rtn(float16 x);

ushort convert_ushort_rtz(float x);
ushort2 convert_ushort2_rtz(float2 x);
ushort3 convert_ushort3_rtz(float3 x);
ushort4 convert_ushort4_rtz(float4 x);
ushort8 convert_ushort8_rtz(float8 x);
ushort16 convert_ushort16_rtz(float16 x);

ushort convert_ushort_sat(float x);
ushort2 convert_ushort2_sat(float2 x);
ushort3 convert_ushort3_sat(float3 x);
ushort4 convert_ushort4_sat(float4 x);
ushort8 convert_ushort8_sat(float8 x);
ushort16 convert_ushort16_sat(float16 x);

ushort convert_ushort_sat_rte(float x);
ushort2 convert_ushort2_sat_rte(float2 x);
ushort3 convert_ushort3_sat_rte(float3 x);
ushort4 convert_ushort4_sat_rte(float4 x);
ushort8 convert_ushort8_sat_rte(float8 x);
ushort16 convert_ushort16_sat_rte(float16 x);

ushort convert_ushort_sat_rtp(float x);
ushort2 convert_ushort2_sat_rtp(float2 x);
ushort3 convert_ushort3_sat_rtp(float3 x);
ushort4 convert_ushort4_sat_rtp(float4 x);
ushort8 convert_ushort8_sat_rtp(float8 x);
ushort16 convert_ushort16_sat_rtp(float16 x);

ushort convert_ushort_sat_rtn(float x);
ushort2 convert_ushort2_sat_rtn(float2 x);
ushort3 convert_ushort3_sat_rtn(float3 x);
ushort4 convert_ushort4_sat_rtn(float4 x);
ushort8 convert_ushort8_sat_rtn(float8 x);
ushort16 convert_ushort16_sat_rtn(float16 x);

ushort convert_ushort_sat_rtz(float x);
ushort2 convert_ushort2_sat_rtz(float2 x);
ushort3 convert_ushort3_sat_rtz(float3 x);
ushort4 convert_ushort4_sat_rtz(float4 x);
ushort8 convert_ushort8_sat_rtz(float8 x);
ushort16 convert_ushort16_sat_rtz(float16 x);

ushort convert_ushort(double x);
ushort2 convert_ushort2(double2 x);
ushort3 convert_ushort3(double3 x);
ushort4 convert_ushort4(double4 x);
ushort8 convert_ushort8(double8 x);
ushort16 convert_ushort16(double16 x);

ushort convert_ushort_rte(double x);
ushort2 convert_ushort2_rte(double2 x);
ushort3 convert_ushort3_rte(double3 x);
ushort4 convert_ushort4_rte(double4 x);
ushort8 convert_ushort8_rte(double8 x);
ushort16 convert_ushort16_rte(double16 x);

ushort convert_ushort_rtp(double x);
ushort2 convert_ushort2_rtp(double2 x);
ushort3 convert_ushort3_rtp(double3 x);
ushort4 convert_ushort4_rtp(double4 x);
ushort8 convert_ushort8_rtp(double8 x);
ushort16 convert_ushort16_rtp(double16 x);

ushort convert_ushort_rtn(double x);
ushort2 convert_ushort2_rtn(double2 x);
ushort3 convert_ushort3_rtn(double3 x);
ushort4 convert_ushort4_rtn(double4 x);
ushort8 convert_ushort8_rtn(double8 x);
ushort16 convert_ushort16_rtn(double16 x);

ushort convert_ushort_rtz(double x);
ushort2 convert_ushort2_rtz(double2 x);
ushort3 convert_ushort3_rtz(double3 x);
ushort4 convert_ushort4_rtz(double4 x);
ushort8 convert_ushort8_rtz(double8 x);
ushort16 convert_ushort16_rtz(double16 x);

ushort convert_ushort_sat(double x);
ushort2 convert_ushort2_sat(double2 x);
ushort3 convert_ushort3_sat(double3 x);
ushort4 convert_ushort4_sat(double4 x);
ushort8 convert_ushort8_sat(double8 x);
ushort16 convert_ushort16_sat(double16 x);

ushort convert_ushort_sat_rte(double x);
ushort2 convert_ushort2_sat_rte(double2 x);
ushort3 convert_ushort3_sat_rte(double3 x);
ushort4 convert_ushort4_sat_rte(double4 x);
ushort8 convert_ushort8_sat_rte(double8 x);
ushort16 convert_ushort16_sat_rte(double16 x);

ushort convert_ushort_sat_rtp(double x);
ushort2 convert_ushort2_sat_rtp(double2 x);
ushort3 convert_ushort3_sat_rtp(double3 x);
ushort4 convert_ushort4_sat_rtp(double4 x);
ushort8 convert_ushort8_sat_rtp(double8 x);
ushort16 convert_ushort16_sat_rtp(double16 x);

ushort convert_ushort_sat_rtn(double x);
ushort2 convert_ushort2_sat_rtn(double2 x);
ushort3 convert_ushort3_sat_rtn(double3 x);
ushort4 convert_ushort4_sat_rtn(double4 x);
ushort8 convert_ushort8_sat_rtn(double8 x);
ushort16 convert_ushort16_sat_rtn(double16 x);

ushort convert_ushort_sat_rtz(double x);
ushort2 convert_ushort2_sat_rtz(double2 x);
ushort3 convert_ushort3_sat_rtz(double3 x);
ushort4 convert_ushort4_sat_rtz(double4 x);
ushort8 convert_ushort8_sat_rtz(double8 x);
ushort16 convert_ushort16_sat_rtz(double16 x);

ushort convert_ushort(ullong x);
ushort2 convert_ushort2(ulong2 x);
ushort3 convert_ushort3(ulong3 x);
ushort4 convert_ushort4(ulong4 x);
ushort8 convert_ushort8(ulong8 x);
ushort16 convert_ushort16(ulong16 x);

ushort convert_ushort_rte(ullong x);
ushort2 convert_ushort2_rte(ulong2 x);
ushort3 convert_ushort3_rte(ulong3 x);
ushort4 convert_ushort4_rte(ulong4 x);
ushort8 convert_ushort8_rte(ulong8 x);
ushort16 convert_ushort16_rte(ulong16 x);

ushort convert_ushort_rtp(ullong x);
ushort2 convert_ushort2_rtp(ulong2 x);
ushort3 convert_ushort3_rtp(ulong3 x);
ushort4 convert_ushort4_rtp(ulong4 x);
ushort8 convert_ushort8_rtp(ulong8 x);
ushort16 convert_ushort16_rtp(ulong16 x);

ushort convert_ushort_rtn(ullong x);
ushort2 convert_ushort2_rtn(ulong2 x);
ushort3 convert_ushort3_rtn(ulong3 x);
ushort4 convert_ushort4_rtn(ulong4 x);
ushort8 convert_ushort8_rtn(ulong8 x);
ushort16 convert_ushort16_rtn(ulong16 x);

ushort convert_ushort_rtz(ullong x);
ushort2 convert_ushort2_rtz(ulong2 x);
ushort3 convert_ushort3_rtz(ulong3 x);
ushort4 convert_ushort4_rtz(ulong4 x);
ushort8 convert_ushort8_rtz(ulong8 x);
ushort16 convert_ushort16_rtz(ulong16 x);

ushort convert_ushort_sat(ullong x);
ushort2 convert_ushort2_sat(ulong2 x);
ushort3 convert_ushort3_sat(ulong3 x);
ushort4 convert_ushort4_sat(ulong4 x);
ushort8 convert_ushort8_sat(ulong8 x);
ushort16 convert_ushort16_sat(ulong16 x);

ushort convert_ushort_sat_rte(ullong x);
ushort2 convert_ushort2_sat_rte(ulong2 x);
ushort3 convert_ushort3_sat_rte(ulong3 x);
ushort4 convert_ushort4_sat_rte(ulong4 x);
ushort8 convert_ushort8_sat_rte(ulong8 x);
ushort16 convert_ushort16_sat_rte(ulong16 x);

ushort convert_ushort_sat_rtp(ullong x);
ushort2 convert_ushort2_sat_rtp(ulong2 x);
ushort3 convert_ushort3_sat_rtp(ulong3 x);
ushort4 convert_ushort4_sat_rtp(ulong4 x);
ushort8 convert_ushort8_sat_rtp(ulong8 x);
ushort16 convert_ushort16_sat_rtp(ulong16 x);

ushort convert_ushort_sat_rtn(ullong x);
ushort2 convert_ushort2_sat_rtn(ulong2 x);
ushort3 convert_ushort3_sat_rtn(ulong3 x);
ushort4 convert_ushort4_sat_rtn(ulong4 x);
ushort8 convert_ushort8_sat_rtn(ulong8 x);
ushort16 convert_ushort16_sat_rtn(ulong16 x);

ushort convert_ushort_sat_rtz(ullong x);
ushort2 convert_ushort2_sat_rtz(ulong2 x);
ushort3 convert_ushort3_sat_rtz(ulong3 x);
ushort4 convert_ushort4_sat_rtz(ulong4 x);
ushort8 convert_ushort8_sat_rtz(ulong8 x);
ushort16 convert_ushort16_sat_rtz(ulong16 x);

ushort convert_ushort(llong x);
ushort2 convert_ushort2(long2 x);
ushort3 convert_ushort3(long3 x);
ushort4 convert_ushort4(long4 x);
ushort8 convert_ushort8(long8 x);
ushort16 convert_ushort16(long16 x);

ushort convert_ushort_rte(llong x);
ushort2 convert_ushort2_rte(long2 x);
ushort3 convert_ushort3_rte(long3 x);
ushort4 convert_ushort4_rte(long4 x);
ushort8 convert_ushort8_rte(long8 x);
ushort16 convert_ushort16_rte(long16 x);

ushort convert_ushort_rtp(llong x);
ushort2 convert_ushort2_rtp(long2 x);
ushort3 convert_ushort3_rtp(long3 x);
ushort4 convert_ushort4_rtp(long4 x);
ushort8 convert_ushort8_rtp(long8 x);
ushort16 convert_ushort16_rtp(long16 x);

ushort convert_ushort_rtn(llong x);
ushort2 convert_ushort2_rtn(long2 x);
ushort3 convert_ushort3_rtn(long3 x);
ushort4 convert_ushort4_rtn(long4 x);
ushort8 convert_ushort8_rtn(long8 x);
ushort16 convert_ushort16_rtn(long16 x);

ushort convert_ushort_rtz(llong x);
ushort2 convert_ushort2_rtz(long2 x);
ushort3 convert_ushort3_rtz(long3 x);
ushort4 convert_ushort4_rtz(long4 x);
ushort8 convert_ushort8_rtz(long8 x);
ushort16 convert_ushort16_rtz(long16 x);

ushort convert_ushort_sat(llong x);
ushort2 convert_ushort2_sat(long2 x);
ushort3 convert_ushort3_sat(long3 x);
ushort4 convert_ushort4_sat(long4 x);
ushort8 convert_ushort8_sat(long8 x);
ushort16 convert_ushort16_sat(long16 x);

ushort convert_ushort_sat_rte(llong x);
ushort2 convert_ushort2_sat_rte(long2 x);
ushort3 convert_ushort3_sat_rte(long3 x);
ushort4 convert_ushort4_sat_rte(long4 x);
ushort8 convert_ushort8_sat_rte(long8 x);
ushort16 convert_ushort16_sat_rte(long16 x);

ushort convert_ushort_sat_rtp(llong x);
ushort2 convert_ushort2_sat_rtp(long2 x);
ushort3 convert_ushort3_sat_rtp(long3 x);
ushort4 convert_ushort4_sat_rtp(long4 x);
ushort8 convert_ushort8_sat_rtp(long8 x);
ushort16 convert_ushort16_sat_rtp(long16 x);

ushort convert_ushort_sat_rtn(llong x);
ushort2 convert_ushort2_sat_rtn(long2 x);
ushort3 convert_ushort3_sat_rtn(long3 x);
ushort4 convert_ushort4_sat_rtn(long4 x);
ushort8 convert_ushort8_sat_rtn(long8 x);
ushort16 convert_ushort16_sat_rtn(long16 x);

ushort convert_ushort_sat_rtz(llong x);
ushort2 convert_ushort2_sat_rtz(long2 x);
ushort3 convert_ushort3_sat_rtz(long3 x);
ushort4 convert_ushort4_sat_rtz(long4 x);
ushort8 convert_ushort8_sat_rtz(long8 x);
ushort16 convert_ushort16_sat_rtz(long16 x);

short convert_short(uchar x);
short2 convert_short2(uchar2 x);
short3 convert_short3(uchar3 x);
short4 convert_short4(uchar4 x);
short8 convert_short8(uchar8 x);
short16 convert_short16(uchar16 x);

short convert_short_rte(uchar x);
short2 convert_short2_rte(uchar2 x);
short3 convert_short3_rte(uchar3 x);
short4 convert_short4_rte(uchar4 x);
short8 convert_short8_rte(uchar8 x);
short16 convert_short16_rte(uchar16 x);

short convert_short_rtp(uchar x);
short2 convert_short2_rtp(uchar2 x);
short3 convert_short3_rtp(uchar3 x);
short4 convert_short4_rtp(uchar4 x);
short8 convert_short8_rtp(uchar8 x);
short16 convert_short16_rtp(uchar16 x);

short convert_short_rtn(uchar x);
short2 convert_short2_rtn(uchar2 x);
short3 convert_short3_rtn(uchar3 x);
short4 convert_short4_rtn(uchar4 x);
short8 convert_short8_rtn(uchar8 x);
short16 convert_short16_rtn(uchar16 x);

short convert_short_rtz(uchar x);
short2 convert_short2_rtz(uchar2 x);
short3 convert_short3_rtz(uchar3 x);
short4 convert_short4_rtz(uchar4 x);
short8 convert_short8_rtz(uchar8 x);
short16 convert_short16_rtz(uchar16 x);

short convert_short_sat(uchar x);
short2 convert_short2_sat(uchar2 x);
short3 convert_short3_sat(uchar3 x);
short4 convert_short4_sat(uchar4 x);
short8 convert_short8_sat(uchar8 x);
short16 convert_short16_sat(uchar16 x);

short convert_short_sat_rte(uchar x);
short2 convert_short2_sat_rte(uchar2 x);
short3 convert_short3_sat_rte(uchar3 x);
short4 convert_short4_sat_rte(uchar4 x);
short8 convert_short8_sat_rte(uchar8 x);
short16 convert_short16_sat_rte(uchar16 x);

short convert_short_sat_rtp(uchar x);
short2 convert_short2_sat_rtp(uchar2 x);
short3 convert_short3_sat_rtp(uchar3 x);
short4 convert_short4_sat_rtp(uchar4 x);
short8 convert_short8_sat_rtp(uchar8 x);
short16 convert_short16_sat_rtp(uchar16 x);

short convert_short_sat_rtn(uchar x);
short2 convert_short2_sat_rtn(uchar2 x);
short3 convert_short3_sat_rtn(uchar3 x);
short4 convert_short4_sat_rtn(uchar4 x);
short8 convert_short8_sat_rtn(uchar8 x);
short16 convert_short16_sat_rtn(uchar16 x);

short convert_short_sat_rtz(uchar x);
short2 convert_short2_sat_rtz(uchar2 x);
short3 convert_short3_sat_rtz(uchar3 x);
short4 convert_short4_sat_rtz(uchar4 x);
short8 convert_short8_sat_rtz(uchar8 x);
short16 convert_short16_sat_rtz(uchar16 x);

short convert_short(schar x);
short2 convert_short2(char2 x);
short3 convert_short3(char3 x);
short4 convert_short4(char4 x);
short8 convert_short8(char8 x);
short16 convert_short16(char16 x);

short convert_short_rte(schar x);
short2 convert_short2_rte(char2 x);
short3 convert_short3_rte(char3 x);
short4 convert_short4_rte(char4 x);
short8 convert_short8_rte(char8 x);
short16 convert_short16_rte(char16 x);

short convert_short_rtp(schar x);
short2 convert_short2_rtp(char2 x);
short3 convert_short3_rtp(char3 x);
short4 convert_short4_rtp(char4 x);
short8 convert_short8_rtp(char8 x);
short16 convert_short16_rtp(char16 x);

short convert_short_rtn(schar x);
short2 convert_short2_rtn(char2 x);
short3 convert_short3_rtn(char3 x);
short4 convert_short4_rtn(char4 x);
short8 convert_short8_rtn(char8 x);
short16 convert_short16_rtn(char16 x);

short convert_short_rtz(schar x);
short2 convert_short2_rtz(char2 x);
short3 convert_short3_rtz(char3 x);
short4 convert_short4_rtz(char4 x);
short8 convert_short8_rtz(char8 x);
short16 convert_short16_rtz(char16 x);

short convert_short_sat(schar x);
short2 convert_short2_sat(char2 x);
short3 convert_short3_sat(char3 x);
short4 convert_short4_sat(char4 x);
short8 convert_short8_sat(char8 x);
short16 convert_short16_sat(char16 x);

short convert_short_sat_rte(schar x);
short2 convert_short2_sat_rte(char2 x);
short3 convert_short3_sat_rte(char3 x);
short4 convert_short4_sat_rte(char4 x);
short8 convert_short8_sat_rte(char8 x);
short16 convert_short16_sat_rte(char16 x);

short convert_short_sat_rtp(schar x);
short2 convert_short2_sat_rtp(char2 x);
short3 convert_short3_sat_rtp(char3 x);
short4 convert_short4_sat_rtp(char4 x);
short8 convert_short8_sat_rtp(char8 x);
short16 convert_short16_sat_rtp(char16 x);

short convert_short_sat_rtn(schar x);
short2 convert_short2_sat_rtn(char2 x);
short3 convert_short3_sat_rtn(char3 x);
short4 convert_short4_sat_rtn(char4 x);
short8 convert_short8_sat_rtn(char8 x);
short16 convert_short16_sat_rtn(char16 x);

short convert_short_sat_rtz(schar x);
short2 convert_short2_sat_rtz(char2 x);
short3 convert_short3_sat_rtz(char3 x);
short4 convert_short4_sat_rtz(char4 x);
short8 convert_short8_sat_rtz(char8 x);
short16 convert_short16_sat_rtz(char16 x);

short convert_short(ushort x);
short2 convert_short2(ushort2 x);
short3 convert_short3(ushort3 x);
short4 convert_short4(ushort4 x);
short8 convert_short8(ushort8 x);
short16 convert_short16(ushort16 x);

short convert_short_rte(ushort x);
short2 convert_short2_rte(ushort2 x);
short3 convert_short3_rte(ushort3 x);
short4 convert_short4_rte(ushort4 x);
short8 convert_short8_rte(ushort8 x);
short16 convert_short16_rte(ushort16 x);

short convert_short_rtp(ushort x);
short2 convert_short2_rtp(ushort2 x);
short3 convert_short3_rtp(ushort3 x);
short4 convert_short4_rtp(ushort4 x);
short8 convert_short8_rtp(ushort8 x);
short16 convert_short16_rtp(ushort16 x);

short convert_short_rtn(ushort x);
short2 convert_short2_rtn(ushort2 x);
short3 convert_short3_rtn(ushort3 x);
short4 convert_short4_rtn(ushort4 x);
short8 convert_short8_rtn(ushort8 x);
short16 convert_short16_rtn(ushort16 x);

short convert_short_rtz(ushort x);
short2 convert_short2_rtz(ushort2 x);
short3 convert_short3_rtz(ushort3 x);
short4 convert_short4_rtz(ushort4 x);
short8 convert_short8_rtz(ushort8 x);
short16 convert_short16_rtz(ushort16 x);

short convert_short_sat(ushort x);
short2 convert_short2_sat(ushort2 x);
short3 convert_short3_sat(ushort3 x);
short4 convert_short4_sat(ushort4 x);
short8 convert_short8_sat(ushort8 x);
short16 convert_short16_sat(ushort16 x);

short convert_short_sat_rte(ushort x);
short2 convert_short2_sat_rte(ushort2 x);
short3 convert_short3_sat_rte(ushort3 x);
short4 convert_short4_sat_rte(ushort4 x);
short8 convert_short8_sat_rte(ushort8 x);
short16 convert_short16_sat_rte(ushort16 x);

short convert_short_sat_rtp(ushort x);
short2 convert_short2_sat_rtp(ushort2 x);
short3 convert_short3_sat_rtp(ushort3 x);
short4 convert_short4_sat_rtp(ushort4 x);
short8 convert_short8_sat_rtp(ushort8 x);
short16 convert_short16_sat_rtp(ushort16 x);

short convert_short_sat_rtn(ushort x);
short2 convert_short2_sat_rtn(ushort2 x);
short3 convert_short3_sat_rtn(ushort3 x);
short4 convert_short4_sat_rtn(ushort4 x);
short8 convert_short8_sat_rtn(ushort8 x);
short16 convert_short16_sat_rtn(ushort16 x);

short convert_short_sat_rtz(ushort x);
short2 convert_short2_sat_rtz(ushort2 x);
short3 convert_short3_sat_rtz(ushort3 x);
short4 convert_short4_sat_rtz(ushort4 x);
short8 convert_short8_sat_rtz(ushort8 x);
short16 convert_short16_sat_rtz(ushort16 x);

short convert_short(short x);
short2 convert_short2(short2 x);
short3 convert_short3(short3 x);
short4 convert_short4(short4 x);
short8 convert_short8(short8 x);
short16 convert_short16(short16 x);

short convert_short_rte(short x);
short2 convert_short2_rte(short2 x);
short3 convert_short3_rte(short3 x);
short4 convert_short4_rte(short4 x);
short8 convert_short8_rte(short8 x);
short16 convert_short16_rte(short16 x);

short convert_short_rtp(short x);
short2 convert_short2_rtp(short2 x);
short3 convert_short3_rtp(short3 x);
short4 convert_short4_rtp(short4 x);
short8 convert_short8_rtp(short8 x);
short16 convert_short16_rtp(short16 x);

short convert_short_rtn(short x);
short2 convert_short2_rtn(short2 x);
short3 convert_short3_rtn(short3 x);
short4 convert_short4_rtn(short4 x);
short8 convert_short8_rtn(short8 x);
short16 convert_short16_rtn(short16 x);

short convert_short_rtz(short x);
short2 convert_short2_rtz(short2 x);
short3 convert_short3_rtz(short3 x);
short4 convert_short4_rtz(short4 x);
short8 convert_short8_rtz(short8 x);
short16 convert_short16_rtz(short16 x);

short convert_short_sat(short x);
short2 convert_short2_sat(short2 x);
short3 convert_short3_sat(short3 x);
short4 convert_short4_sat(short4 x);
short8 convert_short8_sat(short8 x);
short16 convert_short16_sat(short16 x);

short convert_short_sat_rte(short x);
short2 convert_short2_sat_rte(short2 x);
short3 convert_short3_sat_rte(short3 x);
short4 convert_short4_sat_rte(short4 x);
short8 convert_short8_sat_rte(short8 x);
short16 convert_short16_sat_rte(short16 x);

short convert_short_sat_rtp(short x);
short2 convert_short2_sat_rtp(short2 x);
short3 convert_short3_sat_rtp(short3 x);
short4 convert_short4_sat_rtp(short4 x);
short8 convert_short8_sat_rtp(short8 x);
short16 convert_short16_sat_rtp(short16 x);

short convert_short_sat_rtn(short x);
short2 convert_short2_sat_rtn(short2 x);
short3 convert_short3_sat_rtn(short3 x);
short4 convert_short4_sat_rtn(short4 x);
short8 convert_short8_sat_rtn(short8 x);
short16 convert_short16_sat_rtn(short16 x);

short convert_short_sat_rtz(short x);
short2 convert_short2_sat_rtz(short2 x);
short3 convert_short3_sat_rtz(short3 x);
short4 convert_short4_sat_rtz(short4 x);
short8 convert_short8_sat_rtz(short8 x);
short16 convert_short16_sat_rtz(short16 x);

short convert_short(uint x);
short2 convert_short2(uint2 x);
short3 convert_short3(uint3 x);
short4 convert_short4(uint4 x);
short8 convert_short8(uint8 x);
short16 convert_short16(uint16 x);

short convert_short_rte(uint x);
short2 convert_short2_rte(uint2 x);
short3 convert_short3_rte(uint3 x);
short4 convert_short4_rte(uint4 x);
short8 convert_short8_rte(uint8 x);
short16 convert_short16_rte(uint16 x);

short convert_short_rtp(uint x);
short2 convert_short2_rtp(uint2 x);
short3 convert_short3_rtp(uint3 x);
short4 convert_short4_rtp(uint4 x);
short8 convert_short8_rtp(uint8 x);
short16 convert_short16_rtp(uint16 x);

short convert_short_rtn(uint x);
short2 convert_short2_rtn(uint2 x);
short3 convert_short3_rtn(uint3 x);
short4 convert_short4_rtn(uint4 x);
short8 convert_short8_rtn(uint8 x);
short16 convert_short16_rtn(uint16 x);

short convert_short_rtz(uint x);
short2 convert_short2_rtz(uint2 x);
short3 convert_short3_rtz(uint3 x);
short4 convert_short4_rtz(uint4 x);
short8 convert_short8_rtz(uint8 x);
short16 convert_short16_rtz(uint16 x);

short convert_short_sat(uint x);
short2 convert_short2_sat(uint2 x);
short3 convert_short3_sat(uint3 x);
short4 convert_short4_sat(uint4 x);
short8 convert_short8_sat(uint8 x);
short16 convert_short16_sat(uint16 x);

short convert_short_sat_rte(uint x);
short2 convert_short2_sat_rte(uint2 x);
short3 convert_short3_sat_rte(uint3 x);
short4 convert_short4_sat_rte(uint4 x);
short8 convert_short8_sat_rte(uint8 x);
short16 convert_short16_sat_rte(uint16 x);

short convert_short_sat_rtp(uint x);
short2 convert_short2_sat_rtp(uint2 x);
short3 convert_short3_sat_rtp(uint3 x);
short4 convert_short4_sat_rtp(uint4 x);
short8 convert_short8_sat_rtp(uint8 x);
short16 convert_short16_sat_rtp(uint16 x);

short convert_short_sat_rtn(uint x);
short2 convert_short2_sat_rtn(uint2 x);
short3 convert_short3_sat_rtn(uint3 x);
short4 convert_short4_sat_rtn(uint4 x);
short8 convert_short8_sat_rtn(uint8 x);
short16 convert_short16_sat_rtn(uint16 x);

short convert_short_sat_rtz(uint x);
short2 convert_short2_sat_rtz(uint2 x);
short3 convert_short3_sat_rtz(uint3 x);
short4 convert_short4_sat_rtz(uint4 x);
short8 convert_short8_sat_rtz(uint8 x);
short16 convert_short16_sat_rtz(uint16 x);

short convert_short(int x);
short2 convert_short2(int2 x);
short3 convert_short3(int3 x);
short4 convert_short4(int4 x);
short8 convert_short8(int8 x);
short16 convert_short16(int16 x);

short convert_short_rte(int x);
short2 convert_short2_rte(int2 x);
short3 convert_short3_rte(int3 x);
short4 convert_short4_rte(int4 x);
short8 convert_short8_rte(int8 x);
short16 convert_short16_rte(int16 x);

short convert_short_rtp(int x);
short2 convert_short2_rtp(int2 x);
short3 convert_short3_rtp(int3 x);
short4 convert_short4_rtp(int4 x);
short8 convert_short8_rtp(int8 x);
short16 convert_short16_rtp(int16 x);

short convert_short_rtn(int x);
short2 convert_short2_rtn(int2 x);
short3 convert_short3_rtn(int3 x);
short4 convert_short4_rtn(int4 x);
short8 convert_short8_rtn(int8 x);
short16 convert_short16_rtn(int16 x);

short convert_short_rtz(int x);
short2 convert_short2_rtz(int2 x);
short3 convert_short3_rtz(int3 x);
short4 convert_short4_rtz(int4 x);
short8 convert_short8_rtz(int8 x);
short16 convert_short16_rtz(int16 x);

short convert_short_sat(int x);
short2 convert_short2_sat(int2 x);
short3 convert_short3_sat(int3 x);
short4 convert_short4_sat(int4 x);
short8 convert_short8_sat(int8 x);
short16 convert_short16_sat(int16 x);

short convert_short_sat_rte(int x);
short2 convert_short2_sat_rte(int2 x);
short3 convert_short3_sat_rte(int3 x);
short4 convert_short4_sat_rte(int4 x);
short8 convert_short8_sat_rte(int8 x);
short16 convert_short16_sat_rte(int16 x);

short convert_short_sat_rtp(int x);
short2 convert_short2_sat_rtp(int2 x);
short3 convert_short3_sat_rtp(int3 x);
short4 convert_short4_sat_rtp(int4 x);
short8 convert_short8_sat_rtp(int8 x);
short16 convert_short16_sat_rtp(int16 x);

short convert_short_sat_rtn(int x);
short2 convert_short2_sat_rtn(int2 x);
short3 convert_short3_sat_rtn(int3 x);
short4 convert_short4_sat_rtn(int4 x);
short8 convert_short8_sat_rtn(int8 x);
short16 convert_short16_sat_rtn(int16 x);

short convert_short_sat_rtz(int x);
short2 convert_short2_sat_rtz(int2 x);
short3 convert_short3_sat_rtz(int3 x);
short4 convert_short4_sat_rtz(int4 x);
short8 convert_short8_sat_rtz(int8 x);
short16 convert_short16_sat_rtz(int16 x);

short convert_short(float x);
short2 convert_short2(float2 x);
short3 convert_short3(float3 x);
short4 convert_short4(float4 x);
short8 convert_short8(float8 x);
short16 convert_short16(float16 x);

short convert_short_rte(float x);
short2 convert_short2_rte(float2 x);
short3 convert_short3_rte(float3 x);
short4 convert_short4_rte(float4 x);
short8 convert_short8_rte(float8 x);
short16 convert_short16_rte(float16 x);

short convert_short_rtp(float x);
short2 convert_short2_rtp(float2 x);
short3 convert_short3_rtp(float3 x);
short4 convert_short4_rtp(float4 x);
short8 convert_short8_rtp(float8 x);
short16 convert_short16_rtp(float16 x);

short convert_short_rtn(float x);
short2 convert_short2_rtn(float2 x);
short3 convert_short3_rtn(float3 x);
short4 convert_short4_rtn(float4 x);
short8 convert_short8_rtn(float8 x);
short16 convert_short16_rtn(float16 x);

short convert_short_rtz(float x);
short2 convert_short2_rtz(float2 x);
short3 convert_short3_rtz(float3 x);
short4 convert_short4_rtz(float4 x);
short8 convert_short8_rtz(float8 x);
short16 convert_short16_rtz(float16 x);

short convert_short_sat(float x);
short2 convert_short2_sat(float2 x);
short3 convert_short3_sat(float3 x);
short4 convert_short4_sat(float4 x);
short8 convert_short8_sat(float8 x);
short16 convert_short16_sat(float16 x);

short convert_short_sat_rte(float x);
short2 convert_short2_sat_rte(float2 x);
short3 convert_short3_sat_rte(float3 x);
short4 convert_short4_sat_rte(float4 x);
short8 convert_short8_sat_rte(float8 x);
short16 convert_short16_sat_rte(float16 x);

short convert_short_sat_rtp(float x);
short2 convert_short2_sat_rtp(float2 x);
short3 convert_short3_sat_rtp(float3 x);
short4 convert_short4_sat_rtp(float4 x);
short8 convert_short8_sat_rtp(float8 x);
short16 convert_short16_sat_rtp(float16 x);

short convert_short_sat_rtn(float x);
short2 convert_short2_sat_rtn(float2 x);
short3 convert_short3_sat_rtn(float3 x);
short4 convert_short4_sat_rtn(float4 x);
short8 convert_short8_sat_rtn(float8 x);
short16 convert_short16_sat_rtn(float16 x);

short convert_short_sat_rtz(float x);
short2 convert_short2_sat_rtz(float2 x);
short3 convert_short3_sat_rtz(float3 x);
short4 convert_short4_sat_rtz(float4 x);
short8 convert_short8_sat_rtz(float8 x);
short16 convert_short16_sat_rtz(float16 x);

short convert_short(double x);
short2 convert_short2(double2 x);
short3 convert_short3(double3 x);
short4 convert_short4(double4 x);
short8 convert_short8(double8 x);
short16 convert_short16(double16 x);

short convert_short_rte(double x);
short2 convert_short2_rte(double2 x);
short3 convert_short3_rte(double3 x);
short4 convert_short4_rte(double4 x);
short8 convert_short8_rte(double8 x);
short16 convert_short16_rte(double16 x);

short convert_short_rtp(double x);
short2 convert_short2_rtp(double2 x);
short3 convert_short3_rtp(double3 x);
short4 convert_short4_rtp(double4 x);
short8 convert_short8_rtp(double8 x);
short16 convert_short16_rtp(double16 x);

short convert_short_rtn(double x);
short2 convert_short2_rtn(double2 x);
short3 convert_short3_rtn(double3 x);
short4 convert_short4_rtn(double4 x);
short8 convert_short8_rtn(double8 x);
short16 convert_short16_rtn(double16 x);

short convert_short_rtz(double x);
short2 convert_short2_rtz(double2 x);
short3 convert_short3_rtz(double3 x);
short4 convert_short4_rtz(double4 x);
short8 convert_short8_rtz(double8 x);
short16 convert_short16_rtz(double16 x);

short convert_short_sat(double x);
short2 convert_short2_sat(double2 x);
short3 convert_short3_sat(double3 x);
short4 convert_short4_sat(double4 x);
short8 convert_short8_sat(double8 x);
short16 convert_short16_sat(double16 x);

short convert_short_sat_rte(double x);
short2 convert_short2_sat_rte(double2 x);
short3 convert_short3_sat_rte(double3 x);
short4 convert_short4_sat_rte(double4 x);
short8 convert_short8_sat_rte(double8 x);
short16 convert_short16_sat_rte(double16 x);

short convert_short_sat_rtp(double x);
short2 convert_short2_sat_rtp(double2 x);
short3 convert_short3_sat_rtp(double3 x);
short4 convert_short4_sat_rtp(double4 x);
short8 convert_short8_sat_rtp(double8 x);
short16 convert_short16_sat_rtp(double16 x);

short convert_short_sat_rtn(double x);
short2 convert_short2_sat_rtn(double2 x);
short3 convert_short3_sat_rtn(double3 x);
short4 convert_short4_sat_rtn(double4 x);
short8 convert_short8_sat_rtn(double8 x);
short16 convert_short16_sat_rtn(double16 x);

short convert_short_sat_rtz(double x);
short2 convert_short2_sat_rtz(double2 x);
short3 convert_short3_sat_rtz(double3 x);
short4 convert_short4_sat_rtz(double4 x);
short8 convert_short8_sat_rtz(double8 x);
short16 convert_short16_sat_rtz(double16 x);

short convert_short(ullong x);
short2 convert_short2(ulong2 x);
short3 convert_short3(ulong3 x);
short4 convert_short4(ulong4 x);
short8 convert_short8(ulong8 x);
short16 convert_short16(ulong16 x);

short convert_short_rte(ullong x);
short2 convert_short2_rte(ulong2 x);
short3 convert_short3_rte(ulong3 x);
short4 convert_short4_rte(ulong4 x);
short8 convert_short8_rte(ulong8 x);
short16 convert_short16_rte(ulong16 x);

short convert_short_rtp(ullong x);
short2 convert_short2_rtp(ulong2 x);
short3 convert_short3_rtp(ulong3 x);
short4 convert_short4_rtp(ulong4 x);
short8 convert_short8_rtp(ulong8 x);
short16 convert_short16_rtp(ulong16 x);

short convert_short_rtn(ullong x);
short2 convert_short2_rtn(ulong2 x);
short3 convert_short3_rtn(ulong3 x);
short4 convert_short4_rtn(ulong4 x);
short8 convert_short8_rtn(ulong8 x);
short16 convert_short16_rtn(ulong16 x);

short convert_short_rtz(ullong x);
short2 convert_short2_rtz(ulong2 x);
short3 convert_short3_rtz(ulong3 x);
short4 convert_short4_rtz(ulong4 x);
short8 convert_short8_rtz(ulong8 x);
short16 convert_short16_rtz(ulong16 x);

short convert_short_sat(ullong x);
short2 convert_short2_sat(ulong2 x);
short3 convert_short3_sat(ulong3 x);
short4 convert_short4_sat(ulong4 x);
short8 convert_short8_sat(ulong8 x);
short16 convert_short16_sat(ulong16 x);

short convert_short_sat_rte(ullong x);
short2 convert_short2_sat_rte(ulong2 x);
short3 convert_short3_sat_rte(ulong3 x);
short4 convert_short4_sat_rte(ulong4 x);
short8 convert_short8_sat_rte(ulong8 x);
short16 convert_short16_sat_rte(ulong16 x);

short convert_short_sat_rtp(ullong x);
short2 convert_short2_sat_rtp(ulong2 x);
short3 convert_short3_sat_rtp(ulong3 x);
short4 convert_short4_sat_rtp(ulong4 x);
short8 convert_short8_sat_rtp(ulong8 x);
short16 convert_short16_sat_rtp(ulong16 x);

short convert_short_sat_rtn(ullong x);
short2 convert_short2_sat_rtn(ulong2 x);
short3 convert_short3_sat_rtn(ulong3 x);
short4 convert_short4_sat_rtn(ulong4 x);
short8 convert_short8_sat_rtn(ulong8 x);
short16 convert_short16_sat_rtn(ulong16 x);

short convert_short_sat_rtz(ullong x);
short2 convert_short2_sat_rtz(ulong2 x);
short3 convert_short3_sat_rtz(ulong3 x);
short4 convert_short4_sat_rtz(ulong4 x);
short8 convert_short8_sat_rtz(ulong8 x);
short16 convert_short16_sat_rtz(ulong16 x);

short convert_short(llong x);
short2 convert_short2(long2 x);
short3 convert_short3(long3 x);
short4 convert_short4(long4 x);
short8 convert_short8(long8 x);
short16 convert_short16(long16 x);

short convert_short_rte(llong x);
short2 convert_short2_rte(long2 x);
short3 convert_short3_rte(long3 x);
short4 convert_short4_rte(long4 x);
short8 convert_short8_rte(long8 x);
short16 convert_short16_rte(long16 x);

short convert_short_rtp(llong x);
short2 convert_short2_rtp(long2 x);
short3 convert_short3_rtp(long3 x);
short4 convert_short4_rtp(long4 x);
short8 convert_short8_rtp(long8 x);
short16 convert_short16_rtp(long16 x);

short convert_short_rtn(llong x);
short2 convert_short2_rtn(long2 x);
short3 convert_short3_rtn(long3 x);
short4 convert_short4_rtn(long4 x);
short8 convert_short8_rtn(long8 x);
short16 convert_short16_rtn(long16 x);

short convert_short_rtz(llong x);
short2 convert_short2_rtz(long2 x);
short3 convert_short3_rtz(long3 x);
short4 convert_short4_rtz(long4 x);
short8 convert_short8_rtz(long8 x);
short16 convert_short16_rtz(long16 x);

short convert_short_sat(llong x);
short2 convert_short2_sat(long2 x);
short3 convert_short3_sat(long3 x);
short4 convert_short4_sat(long4 x);
short8 convert_short8_sat(long8 x);
short16 convert_short16_sat(long16 x);

short convert_short_sat_rte(llong x);
short2 convert_short2_sat_rte(long2 x);
short3 convert_short3_sat_rte(long3 x);
short4 convert_short4_sat_rte(long4 x);
short8 convert_short8_sat_rte(long8 x);
short16 convert_short16_sat_rte(long16 x);

short convert_short_sat_rtp(llong x);
short2 convert_short2_sat_rtp(long2 x);
short3 convert_short3_sat_rtp(long3 x);
short4 convert_short4_sat_rtp(long4 x);
short8 convert_short8_sat_rtp(long8 x);
short16 convert_short16_sat_rtp(long16 x);

short convert_short_sat_rtn(llong x);
short2 convert_short2_sat_rtn(long2 x);
short3 convert_short3_sat_rtn(long3 x);
short4 convert_short4_sat_rtn(long4 x);
short8 convert_short8_sat_rtn(long8 x);
short16 convert_short16_sat_rtn(long16 x);

short convert_short_sat_rtz(llong x);
short2 convert_short2_sat_rtz(long2 x);
short3 convert_short3_sat_rtz(long3 x);
short4 convert_short4_sat_rtz(long4 x);
short8 convert_short8_sat_rtz(long8 x);
short16 convert_short16_sat_rtz(long16 x);

uint convert_uint(uchar x);
uint2 convert_uint2(uchar2 x);
uint3 convert_uint3(uchar3 x);
uint4 convert_uint4(uchar4 x);
uint8 convert_uint8(uchar8 x);
uint16 convert_uint16(uchar16 x);

uint convert_uint_rte(uchar x);
uint2 convert_uint2_rte(uchar2 x);
uint3 convert_uint3_rte(uchar3 x);
uint4 convert_uint4_rte(uchar4 x);
uint8 convert_uint8_rte(uchar8 x);
uint16 convert_uint16_rte(uchar16 x);

uint convert_uint_rtp(uchar x);
uint2 convert_uint2_rtp(uchar2 x);
uint3 convert_uint3_rtp(uchar3 x);
uint4 convert_uint4_rtp(uchar4 x);
uint8 convert_uint8_rtp(uchar8 x);
uint16 convert_uint16_rtp(uchar16 x);

uint convert_uint_rtn(uchar x);
uint2 convert_uint2_rtn(uchar2 x);
uint3 convert_uint3_rtn(uchar3 x);
uint4 convert_uint4_rtn(uchar4 x);
uint8 convert_uint8_rtn(uchar8 x);
uint16 convert_uint16_rtn(uchar16 x);

uint convert_uint_rtz(uchar x);
uint2 convert_uint2_rtz(uchar2 x);
uint3 convert_uint3_rtz(uchar3 x);
uint4 convert_uint4_rtz(uchar4 x);
uint8 convert_uint8_rtz(uchar8 x);
uint16 convert_uint16_rtz(uchar16 x);

uint convert_uint_sat(uchar x);
uint2 convert_uint2_sat(uchar2 x);
uint3 convert_uint3_sat(uchar3 x);
uint4 convert_uint4_sat(uchar4 x);
uint8 convert_uint8_sat(uchar8 x);
uint16 convert_uint16_sat(uchar16 x);

uint convert_uint_sat_rte(uchar x);
uint2 convert_uint2_sat_rte(uchar2 x);
uint3 convert_uint3_sat_rte(uchar3 x);
uint4 convert_uint4_sat_rte(uchar4 x);
uint8 convert_uint8_sat_rte(uchar8 x);
uint16 convert_uint16_sat_rte(uchar16 x);

uint convert_uint_sat_rtp(uchar x);
uint2 convert_uint2_sat_rtp(uchar2 x);
uint3 convert_uint3_sat_rtp(uchar3 x);
uint4 convert_uint4_sat_rtp(uchar4 x);
uint8 convert_uint8_sat_rtp(uchar8 x);
uint16 convert_uint16_sat_rtp(uchar16 x);

uint convert_uint_sat_rtn(uchar x);
uint2 convert_uint2_sat_rtn(uchar2 x);
uint3 convert_uint3_sat_rtn(uchar3 x);
uint4 convert_uint4_sat_rtn(uchar4 x);
uint8 convert_uint8_sat_rtn(uchar8 x);
uint16 convert_uint16_sat_rtn(uchar16 x);

uint convert_uint_sat_rtz(uchar x);
uint2 convert_uint2_sat_rtz(uchar2 x);
uint3 convert_uint3_sat_rtz(uchar3 x);
uint4 convert_uint4_sat_rtz(uchar4 x);
uint8 convert_uint8_sat_rtz(uchar8 x);
uint16 convert_uint16_sat_rtz(uchar16 x);

uint convert_uint(schar x);
uint2 convert_uint2(char2 x);
uint3 convert_uint3(char3 x);
uint4 convert_uint4(char4 x);
uint8 convert_uint8(char8 x);
uint16 convert_uint16(char16 x);

uint convert_uint_rte(schar x);
uint2 convert_uint2_rte(char2 x);
uint3 convert_uint3_rte(char3 x);
uint4 convert_uint4_rte(char4 x);
uint8 convert_uint8_rte(char8 x);
uint16 convert_uint16_rte(char16 x);

uint convert_uint_rtp(schar x);
uint2 convert_uint2_rtp(char2 x);
uint3 convert_uint3_rtp(char3 x);
uint4 convert_uint4_rtp(char4 x);
uint8 convert_uint8_rtp(char8 x);
uint16 convert_uint16_rtp(char16 x);

uint convert_uint_rtn(schar x);
uint2 convert_uint2_rtn(char2 x);
uint3 convert_uint3_rtn(char3 x);
uint4 convert_uint4_rtn(char4 x);
uint8 convert_uint8_rtn(char8 x);
uint16 convert_uint16_rtn(char16 x);

uint convert_uint_rtz(schar x);
uint2 convert_uint2_rtz(char2 x);
uint3 convert_uint3_rtz(char3 x);
uint4 convert_uint4_rtz(char4 x);
uint8 convert_uint8_rtz(char8 x);
uint16 convert_uint16_rtz(char16 x);

uint convert_uint_sat(schar x);
uint2 convert_uint2_sat(char2 x);
uint3 convert_uint3_sat(char3 x);
uint4 convert_uint4_sat(char4 x);
uint8 convert_uint8_sat(char8 x);
uint16 convert_uint16_sat(char16 x);

uint convert_uint_sat_rte(schar x);
uint2 convert_uint2_sat_rte(char2 x);
uint3 convert_uint3_sat_rte(char3 x);
uint4 convert_uint4_sat_rte(char4 x);
uint8 convert_uint8_sat_rte(char8 x);
uint16 convert_uint16_sat_rte(char16 x);

uint convert_uint_sat_rtp(schar x);
uint2 convert_uint2_sat_rtp(char2 x);
uint3 convert_uint3_sat_rtp(char3 x);
uint4 convert_uint4_sat_rtp(char4 x);
uint8 convert_uint8_sat_rtp(char8 x);
uint16 convert_uint16_sat_rtp(char16 x);

uint convert_uint_sat_rtn(schar x);
uint2 convert_uint2_sat_rtn(char2 x);
uint3 convert_uint3_sat_rtn(char3 x);
uint4 convert_uint4_sat_rtn(char4 x);
uint8 convert_uint8_sat_rtn(char8 x);
uint16 convert_uint16_sat_rtn(char16 x);

uint convert_uint_sat_rtz(schar x);
uint2 convert_uint2_sat_rtz(char2 x);
uint3 convert_uint3_sat_rtz(char3 x);
uint4 convert_uint4_sat_rtz(char4 x);
uint8 convert_uint8_sat_rtz(char8 x);
uint16 convert_uint16_sat_rtz(char16 x);

uint convert_uint(ushort x);
uint2 convert_uint2(ushort2 x);
uint3 convert_uint3(ushort3 x);
uint4 convert_uint4(ushort4 x);
uint8 convert_uint8(ushort8 x);
uint16 convert_uint16(ushort16 x);

uint convert_uint_rte(ushort x);
uint2 convert_uint2_rte(ushort2 x);
uint3 convert_uint3_rte(ushort3 x);
uint4 convert_uint4_rte(ushort4 x);
uint8 convert_uint8_rte(ushort8 x);
uint16 convert_uint16_rte(ushort16 x);

uint convert_uint_rtp(ushort x);
uint2 convert_uint2_rtp(ushort2 x);
uint3 convert_uint3_rtp(ushort3 x);
uint4 convert_uint4_rtp(ushort4 x);
uint8 convert_uint8_rtp(ushort8 x);
uint16 convert_uint16_rtp(ushort16 x);

uint convert_uint_rtn(ushort x);
uint2 convert_uint2_rtn(ushort2 x);
uint3 convert_uint3_rtn(ushort3 x);
uint4 convert_uint4_rtn(ushort4 x);
uint8 convert_uint8_rtn(ushort8 x);
uint16 convert_uint16_rtn(ushort16 x);

uint convert_uint_rtz(ushort x);
uint2 convert_uint2_rtz(ushort2 x);
uint3 convert_uint3_rtz(ushort3 x);
uint4 convert_uint4_rtz(ushort4 x);
uint8 convert_uint8_rtz(ushort8 x);
uint16 convert_uint16_rtz(ushort16 x);

uint convert_uint_sat(ushort x);
uint2 convert_uint2_sat(ushort2 x);
uint3 convert_uint3_sat(ushort3 x);
uint4 convert_uint4_sat(ushort4 x);
uint8 convert_uint8_sat(ushort8 x);
uint16 convert_uint16_sat(ushort16 x);

uint convert_uint_sat_rte(ushort x);
uint2 convert_uint2_sat_rte(ushort2 x);
uint3 convert_uint3_sat_rte(ushort3 x);
uint4 convert_uint4_sat_rte(ushort4 x);
uint8 convert_uint8_sat_rte(ushort8 x);
uint16 convert_uint16_sat_rte(ushort16 x);

uint convert_uint_sat_rtp(ushort x);
uint2 convert_uint2_sat_rtp(ushort2 x);
uint3 convert_uint3_sat_rtp(ushort3 x);
uint4 convert_uint4_sat_rtp(ushort4 x);
uint8 convert_uint8_sat_rtp(ushort8 x);
uint16 convert_uint16_sat_rtp(ushort16 x);

uint convert_uint_sat_rtn(ushort x);
uint2 convert_uint2_sat_rtn(ushort2 x);
uint3 convert_uint3_sat_rtn(ushort3 x);
uint4 convert_uint4_sat_rtn(ushort4 x);
uint8 convert_uint8_sat_rtn(ushort8 x);
uint16 convert_uint16_sat_rtn(ushort16 x);

uint convert_uint_sat_rtz(ushort x);
uint2 convert_uint2_sat_rtz(ushort2 x);
uint3 convert_uint3_sat_rtz(ushort3 x);
uint4 convert_uint4_sat_rtz(ushort4 x);
uint8 convert_uint8_sat_rtz(ushort8 x);
uint16 convert_uint16_sat_rtz(ushort16 x);

uint convert_uint(short x);
uint2 convert_uint2(short2 x);
uint3 convert_uint3(short3 x);
uint4 convert_uint4(short4 x);
uint8 convert_uint8(short8 x);
uint16 convert_uint16(short16 x);

uint convert_uint_rte(short x);
uint2 convert_uint2_rte(short2 x);
uint3 convert_uint3_rte(short3 x);
uint4 convert_uint4_rte(short4 x);
uint8 convert_uint8_rte(short8 x);
uint16 convert_uint16_rte(short16 x);

uint convert_uint_rtp(short x);
uint2 convert_uint2_rtp(short2 x);
uint3 convert_uint3_rtp(short3 x);
uint4 convert_uint4_rtp(short4 x);
uint8 convert_uint8_rtp(short8 x);
uint16 convert_uint16_rtp(short16 x);

uint convert_uint_rtn(short x);
uint2 convert_uint2_rtn(short2 x);
uint3 convert_uint3_rtn(short3 x);
uint4 convert_uint4_rtn(short4 x);
uint8 convert_uint8_rtn(short8 x);
uint16 convert_uint16_rtn(short16 x);

uint convert_uint_rtz(short x);
uint2 convert_uint2_rtz(short2 x);
uint3 convert_uint3_rtz(short3 x);
uint4 convert_uint4_rtz(short4 x);
uint8 convert_uint8_rtz(short8 x);
uint16 convert_uint16_rtz(short16 x);

uint convert_uint_sat(short x);
uint2 convert_uint2_sat(short2 x);
uint3 convert_uint3_sat(short3 x);
uint4 convert_uint4_sat(short4 x);
uint8 convert_uint8_sat(short8 x);
uint16 convert_uint16_sat(short16 x);

uint convert_uint_sat_rte(short x);
uint2 convert_uint2_sat_rte(short2 x);
uint3 convert_uint3_sat_rte(short3 x);
uint4 convert_uint4_sat_rte(short4 x);
uint8 convert_uint8_sat_rte(short8 x);
uint16 convert_uint16_sat_rte(short16 x);

uint convert_uint_sat_rtp(short x);
uint2 convert_uint2_sat_rtp(short2 x);
uint3 convert_uint3_sat_rtp(short3 x);
uint4 convert_uint4_sat_rtp(short4 x);
uint8 convert_uint8_sat_rtp(short8 x);
uint16 convert_uint16_sat_rtp(short16 x);

uint convert_uint_sat_rtn(short x);
uint2 convert_uint2_sat_rtn(short2 x);
uint3 convert_uint3_sat_rtn(short3 x);
uint4 convert_uint4_sat_rtn(short4 x);
uint8 convert_uint8_sat_rtn(short8 x);
uint16 convert_uint16_sat_rtn(short16 x);

uint convert_uint_sat_rtz(short x);
uint2 convert_uint2_sat_rtz(short2 x);
uint3 convert_uint3_sat_rtz(short3 x);
uint4 convert_uint4_sat_rtz(short4 x);
uint8 convert_uint8_sat_rtz(short8 x);
uint16 convert_uint16_sat_rtz(short16 x);

uint convert_uint(uint x);
uint2 convert_uint2(uint2 x);
uint3 convert_uint3(uint3 x);
uint4 convert_uint4(uint4 x);
uint8 convert_uint8(uint8 x);
uint16 convert_uint16(uint16 x);

uint convert_uint_rte(uint x);
uint2 convert_uint2_rte(uint2 x);
uint3 convert_uint3_rte(uint3 x);
uint4 convert_uint4_rte(uint4 x);
uint8 convert_uint8_rte(uint8 x);
uint16 convert_uint16_rte(uint16 x);

uint convert_uint_rtp(uint x);
uint2 convert_uint2_rtp(uint2 x);
uint3 convert_uint3_rtp(uint3 x);
uint4 convert_uint4_rtp(uint4 x);
uint8 convert_uint8_rtp(uint8 x);
uint16 convert_uint16_rtp(uint16 x);

uint convert_uint_rtn(uint x);
uint2 convert_uint2_rtn(uint2 x);
uint3 convert_uint3_rtn(uint3 x);
uint4 convert_uint4_rtn(uint4 x);
uint8 convert_uint8_rtn(uint8 x);
uint16 convert_uint16_rtn(uint16 x);

uint convert_uint_rtz(uint x);
uint2 convert_uint2_rtz(uint2 x);
uint3 convert_uint3_rtz(uint3 x);
uint4 convert_uint4_rtz(uint4 x);
uint8 convert_uint8_rtz(uint8 x);
uint16 convert_uint16_rtz(uint16 x);

uint convert_uint_sat(uint x);
uint2 convert_uint2_sat(uint2 x);
uint3 convert_uint3_sat(uint3 x);
uint4 convert_uint4_sat(uint4 x);
uint8 convert_uint8_sat(uint8 x);
uint16 convert_uint16_sat(uint16 x);

uint convert_uint_sat_rte(uint x);
uint2 convert_uint2_sat_rte(uint2 x);
uint3 convert_uint3_sat_rte(uint3 x);
uint4 convert_uint4_sat_rte(uint4 x);
uint8 convert_uint8_sat_rte(uint8 x);
uint16 convert_uint16_sat_rte(uint16 x);

uint convert_uint_sat_rtp(uint x);
uint2 convert_uint2_sat_rtp(uint2 x);
uint3 convert_uint3_sat_rtp(uint3 x);
uint4 convert_uint4_sat_rtp(uint4 x);
uint8 convert_uint8_sat_rtp(uint8 x);
uint16 convert_uint16_sat_rtp(uint16 x);

uint convert_uint_sat_rtn(uint x);
uint2 convert_uint2_sat_rtn(uint2 x);
uint3 convert_uint3_sat_rtn(uint3 x);
uint4 convert_uint4_sat_rtn(uint4 x);
uint8 convert_uint8_sat_rtn(uint8 x);
uint16 convert_uint16_sat_rtn(uint16 x);

uint convert_uint_sat_rtz(uint x);
uint2 convert_uint2_sat_rtz(uint2 x);
uint3 convert_uint3_sat_rtz(uint3 x);
uint4 convert_uint4_sat_rtz(uint4 x);
uint8 convert_uint8_sat_rtz(uint8 x);
uint16 convert_uint16_sat_rtz(uint16 x);

uint convert_uint(int x);
uint2 convert_uint2(int2 x);
uint3 convert_uint3(int3 x);
uint4 convert_uint4(int4 x);
uint8 convert_uint8(int8 x);
uint16 convert_uint16(int16 x);

uint convert_uint_rte(int x);
uint2 convert_uint2_rte(int2 x);
uint3 convert_uint3_rte(int3 x);
uint4 convert_uint4_rte(int4 x);
uint8 convert_uint8_rte(int8 x);
uint16 convert_uint16_rte(int16 x);

uint convert_uint_rtp(int x);
uint2 convert_uint2_rtp(int2 x);
uint3 convert_uint3_rtp(int3 x);
uint4 convert_uint4_rtp(int4 x);
uint8 convert_uint8_rtp(int8 x);
uint16 convert_uint16_rtp(int16 x);

uint convert_uint_rtn(int x);
uint2 convert_uint2_rtn(int2 x);
uint3 convert_uint3_rtn(int3 x);
uint4 convert_uint4_rtn(int4 x);
uint8 convert_uint8_rtn(int8 x);
uint16 convert_uint16_rtn(int16 x);

uint convert_uint_rtz(int x);
uint2 convert_uint2_rtz(int2 x);
uint3 convert_uint3_rtz(int3 x);
uint4 convert_uint4_rtz(int4 x);
uint8 convert_uint8_rtz(int8 x);
uint16 convert_uint16_rtz(int16 x);

uint convert_uint_sat(int x);
uint2 convert_uint2_sat(int2 x);
uint3 convert_uint3_sat(int3 x);
uint4 convert_uint4_sat(int4 x);
uint8 convert_uint8_sat(int8 x);
uint16 convert_uint16_sat(int16 x);

uint convert_uint_sat_rte(int x);
uint2 convert_uint2_sat_rte(int2 x);
uint3 convert_uint3_sat_rte(int3 x);
uint4 convert_uint4_sat_rte(int4 x);
uint8 convert_uint8_sat_rte(int8 x);
uint16 convert_uint16_sat_rte(int16 x);

uint convert_uint_sat_rtp(int x);
uint2 convert_uint2_sat_rtp(int2 x);
uint3 convert_uint3_sat_rtp(int3 x);
uint4 convert_uint4_sat_rtp(int4 x);
uint8 convert_uint8_sat_rtp(int8 x);
uint16 convert_uint16_sat_rtp(int16 x);

uint convert_uint_sat_rtn(int x);
uint2 convert_uint2_sat_rtn(int2 x);
uint3 convert_uint3_sat_rtn(int3 x);
uint4 convert_uint4_sat_rtn(int4 x);
uint8 convert_uint8_sat_rtn(int8 x);
uint16 convert_uint16_sat_rtn(int16 x);

uint convert_uint_sat_rtz(int x);
uint2 convert_uint2_sat_rtz(int2 x);
uint3 convert_uint3_sat_rtz(int3 x);
uint4 convert_uint4_sat_rtz(int4 x);
uint8 convert_uint8_sat_rtz(int8 x);
uint16 convert_uint16_sat_rtz(int16 x);

uint convert_uint(float x);
uint2 convert_uint2(float2 x);
uint3 convert_uint3(float3 x);
uint4 convert_uint4(float4 x);
uint8 convert_uint8(float8 x);
uint16 convert_uint16(float16 x);

uint convert_uint_rte(float x);
uint2 convert_uint2_rte(float2 x);
uint3 convert_uint3_rte(float3 x);
uint4 convert_uint4_rte(float4 x);
uint8 convert_uint8_rte(float8 x);
uint16 convert_uint16_rte(float16 x);

uint convert_uint_rtp(float x);
uint2 convert_uint2_rtp(float2 x);
uint3 convert_uint3_rtp(float3 x);
uint4 convert_uint4_rtp(float4 x);
uint8 convert_uint8_rtp(float8 x);
uint16 convert_uint16_rtp(float16 x);

uint convert_uint_rtn(float x);
uint2 convert_uint2_rtn(float2 x);
uint3 convert_uint3_rtn(float3 x);
uint4 convert_uint4_rtn(float4 x);
uint8 convert_uint8_rtn(float8 x);
uint16 convert_uint16_rtn(float16 x);

uint convert_uint_rtz(float x);
uint2 convert_uint2_rtz(float2 x);
uint3 convert_uint3_rtz(float3 x);
uint4 convert_uint4_rtz(float4 x);
uint8 convert_uint8_rtz(float8 x);
uint16 convert_uint16_rtz(float16 x);

uint convert_uint_sat(float x);
uint2 convert_uint2_sat(float2 x);
uint3 convert_uint3_sat(float3 x);
uint4 convert_uint4_sat(float4 x);
uint8 convert_uint8_sat(float8 x);
uint16 convert_uint16_sat(float16 x);

uint convert_uint_sat_rte(float x);
uint2 convert_uint2_sat_rte(float2 x);
uint3 convert_uint3_sat_rte(float3 x);
uint4 convert_uint4_sat_rte(float4 x);
uint8 convert_uint8_sat_rte(float8 x);
uint16 convert_uint16_sat_rte(float16 x);

uint convert_uint_sat_rtp(float x);
uint2 convert_uint2_sat_rtp(float2 x);
uint3 convert_uint3_sat_rtp(float3 x);
uint4 convert_uint4_sat_rtp(float4 x);
uint8 convert_uint8_sat_rtp(float8 x);
uint16 convert_uint16_sat_rtp(float16 x);

uint convert_uint_sat_rtn(float x);
uint2 convert_uint2_sat_rtn(float2 x);
uint3 convert_uint3_sat_rtn(float3 x);
uint4 convert_uint4_sat_rtn(float4 x);
uint8 convert_uint8_sat_rtn(float8 x);
uint16 convert_uint16_sat_rtn(float16 x);

uint convert_uint_sat_rtz(float x);
uint2 convert_uint2_sat_rtz(float2 x);
uint3 convert_uint3_sat_rtz(float3 x);
uint4 convert_uint4_sat_rtz(float4 x);
uint8 convert_uint8_sat_rtz(float8 x);
uint16 convert_uint16_sat_rtz(float16 x);

uint convert_uint(double x);
uint2 convert_uint2(double2 x);
uint3 convert_uint3(double3 x);
uint4 convert_uint4(double4 x);
uint8 convert_uint8(double8 x);
uint16 convert_uint16(double16 x);

uint convert_uint_rte(double x);
uint2 convert_uint2_rte(double2 x);
uint3 convert_uint3_rte(double3 x);
uint4 convert_uint4_rte(double4 x);
uint8 convert_uint8_rte(double8 x);
uint16 convert_uint16_rte(double16 x);

uint convert_uint_rtp(double x);
uint2 convert_uint2_rtp(double2 x);
uint3 convert_uint3_rtp(double3 x);
uint4 convert_uint4_rtp(double4 x);
uint8 convert_uint8_rtp(double8 x);
uint16 convert_uint16_rtp(double16 x);

uint convert_uint_rtn(double x);
uint2 convert_uint2_rtn(double2 x);
uint3 convert_uint3_rtn(double3 x);
uint4 convert_uint4_rtn(double4 x);
uint8 convert_uint8_rtn(double8 x);
uint16 convert_uint16_rtn(double16 x);

uint convert_uint_rtz(double x);
uint2 convert_uint2_rtz(double2 x);
uint3 convert_uint3_rtz(double3 x);
uint4 convert_uint4_rtz(double4 x);
uint8 convert_uint8_rtz(double8 x);
uint16 convert_uint16_rtz(double16 x);

uint convert_uint_sat(double x);
uint2 convert_uint2_sat(double2 x);
uint3 convert_uint3_sat(double3 x);
uint4 convert_uint4_sat(double4 x);
uint8 convert_uint8_sat(double8 x);
uint16 convert_uint16_sat(double16 x);

uint convert_uint_sat_rte(double x);
uint2 convert_uint2_sat_rte(double2 x);
uint3 convert_uint3_sat_rte(double3 x);
uint4 convert_uint4_sat_rte(double4 x);
uint8 convert_uint8_sat_rte(double8 x);
uint16 convert_uint16_sat_rte(double16 x);

uint convert_uint_sat_rtp(double x);
uint2 convert_uint2_sat_rtp(double2 x);
uint3 convert_uint3_sat_rtp(double3 x);
uint4 convert_uint4_sat_rtp(double4 x);
uint8 convert_uint8_sat_rtp(double8 x);
uint16 convert_uint16_sat_rtp(double16 x);

uint convert_uint_sat_rtn(double x);
uint2 convert_uint2_sat_rtn(double2 x);
uint3 convert_uint3_sat_rtn(double3 x);
uint4 convert_uint4_sat_rtn(double4 x);
uint8 convert_uint8_sat_rtn(double8 x);
uint16 convert_uint16_sat_rtn(double16 x);

uint convert_uint_sat_rtz(double x);
uint2 convert_uint2_sat_rtz(double2 x);
uint3 convert_uint3_sat_rtz(double3 x);
uint4 convert_uint4_sat_rtz(double4 x);
uint8 convert_uint8_sat_rtz(double8 x);
uint16 convert_uint16_sat_rtz(double16 x);

uint convert_uint(ullong x);
uint2 convert_uint2(ulong2 x);
uint3 convert_uint3(ulong3 x);
uint4 convert_uint4(ulong4 x);
uint8 convert_uint8(ulong8 x);
uint16 convert_uint16(ulong16 x);

uint convert_uint_rte(ullong x);
uint2 convert_uint2_rte(ulong2 x);
uint3 convert_uint3_rte(ulong3 x);
uint4 convert_uint4_rte(ulong4 x);
uint8 convert_uint8_rte(ulong8 x);
uint16 convert_uint16_rte(ulong16 x);

uint convert_uint_rtp(ullong x);
uint2 convert_uint2_rtp(ulong2 x);
uint3 convert_uint3_rtp(ulong3 x);
uint4 convert_uint4_rtp(ulong4 x);
uint8 convert_uint8_rtp(ulong8 x);
uint16 convert_uint16_rtp(ulong16 x);

uint convert_uint_rtn(ullong x);
uint2 convert_uint2_rtn(ulong2 x);
uint3 convert_uint3_rtn(ulong3 x);
uint4 convert_uint4_rtn(ulong4 x);
uint8 convert_uint8_rtn(ulong8 x);
uint16 convert_uint16_rtn(ulong16 x);

uint convert_uint_rtz(ullong x);
uint2 convert_uint2_rtz(ulong2 x);
uint3 convert_uint3_rtz(ulong3 x);
uint4 convert_uint4_rtz(ulong4 x);
uint8 convert_uint8_rtz(ulong8 x);
uint16 convert_uint16_rtz(ulong16 x);

uint convert_uint_sat(ullong x);
uint2 convert_uint2_sat(ulong2 x);
uint3 convert_uint3_sat(ulong3 x);
uint4 convert_uint4_sat(ulong4 x);
uint8 convert_uint8_sat(ulong8 x);
uint16 convert_uint16_sat(ulong16 x);

uint convert_uint_sat_rte(ullong x);
uint2 convert_uint2_sat_rte(ulong2 x);
uint3 convert_uint3_sat_rte(ulong3 x);
uint4 convert_uint4_sat_rte(ulong4 x);
uint8 convert_uint8_sat_rte(ulong8 x);
uint16 convert_uint16_sat_rte(ulong16 x);

uint convert_uint_sat_rtp(ullong x);
uint2 convert_uint2_sat_rtp(ulong2 x);
uint3 convert_uint3_sat_rtp(ulong3 x);
uint4 convert_uint4_sat_rtp(ulong4 x);
uint8 convert_uint8_sat_rtp(ulong8 x);
uint16 convert_uint16_sat_rtp(ulong16 x);

uint convert_uint_sat_rtn(ullong x);
uint2 convert_uint2_sat_rtn(ulong2 x);
uint3 convert_uint3_sat_rtn(ulong3 x);
uint4 convert_uint4_sat_rtn(ulong4 x);
uint8 convert_uint8_sat_rtn(ulong8 x);
uint16 convert_uint16_sat_rtn(ulong16 x);

uint convert_uint_sat_rtz(ullong x);
uint2 convert_uint2_sat_rtz(ulong2 x);
uint3 convert_uint3_sat_rtz(ulong3 x);
uint4 convert_uint4_sat_rtz(ulong4 x);
uint8 convert_uint8_sat_rtz(ulong8 x);
uint16 convert_uint16_sat_rtz(ulong16 x);

uint convert_uint(llong x);
uint2 convert_uint2(long2 x);
uint3 convert_uint3(long3 x);
uint4 convert_uint4(long4 x);
uint8 convert_uint8(long8 x);
uint16 convert_uint16(long16 x);

uint convert_uint_rte(llong x);
uint2 convert_uint2_rte(long2 x);
uint3 convert_uint3_rte(long3 x);
uint4 convert_uint4_rte(long4 x);
uint8 convert_uint8_rte(long8 x);
uint16 convert_uint16_rte(long16 x);

uint convert_uint_rtp(llong x);
uint2 convert_uint2_rtp(long2 x);
uint3 convert_uint3_rtp(long3 x);
uint4 convert_uint4_rtp(long4 x);
uint8 convert_uint8_rtp(long8 x);
uint16 convert_uint16_rtp(long16 x);

uint convert_uint_rtn(llong x);
uint2 convert_uint2_rtn(long2 x);
uint3 convert_uint3_rtn(long3 x);
uint4 convert_uint4_rtn(long4 x);
uint8 convert_uint8_rtn(long8 x);
uint16 convert_uint16_rtn(long16 x);

uint convert_uint_rtz(llong x);
uint2 convert_uint2_rtz(long2 x);
uint3 convert_uint3_rtz(long3 x);
uint4 convert_uint4_rtz(long4 x);
uint8 convert_uint8_rtz(long8 x);
uint16 convert_uint16_rtz(long16 x);

uint convert_uint_sat(llong x);
uint2 convert_uint2_sat(long2 x);
uint3 convert_uint3_sat(long3 x);
uint4 convert_uint4_sat(long4 x);
uint8 convert_uint8_sat(long8 x);
uint16 convert_uint16_sat(long16 x);

uint convert_uint_sat_rte(llong x);
uint2 convert_uint2_sat_rte(long2 x);
uint3 convert_uint3_sat_rte(long3 x);
uint4 convert_uint4_sat_rte(long4 x);
uint8 convert_uint8_sat_rte(long8 x);
uint16 convert_uint16_sat_rte(long16 x);

uint convert_uint_sat_rtp(llong x);
uint2 convert_uint2_sat_rtp(long2 x);
uint3 convert_uint3_sat_rtp(long3 x);
uint4 convert_uint4_sat_rtp(long4 x);
uint8 convert_uint8_sat_rtp(long8 x);
uint16 convert_uint16_sat_rtp(long16 x);

uint convert_uint_sat_rtn(llong x);
uint2 convert_uint2_sat_rtn(long2 x);
uint3 convert_uint3_sat_rtn(long3 x);
uint4 convert_uint4_sat_rtn(long4 x);
uint8 convert_uint8_sat_rtn(long8 x);
uint16 convert_uint16_sat_rtn(long16 x);

uint convert_uint_sat_rtz(llong x);
uint2 convert_uint2_sat_rtz(long2 x);
uint3 convert_uint3_sat_rtz(long3 x);
uint4 convert_uint4_sat_rtz(long4 x);
uint8 convert_uint8_sat_rtz(long8 x);
uint16 convert_uint16_sat_rtz(long16 x);

int convert_int(uchar x);
int2 convert_int2(uchar2 x);
int3 convert_int3(uchar3 x);
int4 convert_int4(uchar4 x);
int8 convert_int8(uchar8 x);
int16 convert_int16(uchar16 x);

int convert_int_rte(uchar x);
int2 convert_int2_rte(uchar2 x);
int3 convert_int3_rte(uchar3 x);
int4 convert_int4_rte(uchar4 x);
int8 convert_int8_rte(uchar8 x);
int16 convert_int16_rte(uchar16 x);

int convert_int_rtp(uchar x);
int2 convert_int2_rtp(uchar2 x);
int3 convert_int3_rtp(uchar3 x);
int4 convert_int4_rtp(uchar4 x);
int8 convert_int8_rtp(uchar8 x);
int16 convert_int16_rtp(uchar16 x);

int convert_int_rtn(uchar x);
int2 convert_int2_rtn(uchar2 x);
int3 convert_int3_rtn(uchar3 x);
int4 convert_int4_rtn(uchar4 x);
int8 convert_int8_rtn(uchar8 x);
int16 convert_int16_rtn(uchar16 x);

int convert_int_rtz(uchar x);
int2 convert_int2_rtz(uchar2 x);
int3 convert_int3_rtz(uchar3 x);
int4 convert_int4_rtz(uchar4 x);
int8 convert_int8_rtz(uchar8 x);
int16 convert_int16_rtz(uchar16 x);

int convert_int_sat(uchar x);
int2 convert_int2_sat(uchar2 x);
int3 convert_int3_sat(uchar3 x);
int4 convert_int4_sat(uchar4 x);
int8 convert_int8_sat(uchar8 x);
int16 convert_int16_sat(uchar16 x);

int convert_int_sat_rte(uchar x);
int2 convert_int2_sat_rte(uchar2 x);
int3 convert_int3_sat_rte(uchar3 x);
int4 convert_int4_sat_rte(uchar4 x);
int8 convert_int8_sat_rte(uchar8 x);
int16 convert_int16_sat_rte(uchar16 x);

int convert_int_sat_rtp(uchar x);
int2 convert_int2_sat_rtp(uchar2 x);
int3 convert_int3_sat_rtp(uchar3 x);
int4 convert_int4_sat_rtp(uchar4 x);
int8 convert_int8_sat_rtp(uchar8 x);
int16 convert_int16_sat_rtp(uchar16 x);

int convert_int_sat_rtn(uchar x);
int2 convert_int2_sat_rtn(uchar2 x);
int3 convert_int3_sat_rtn(uchar3 x);
int4 convert_int4_sat_rtn(uchar4 x);
int8 convert_int8_sat_rtn(uchar8 x);
int16 convert_int16_sat_rtn(uchar16 x);

int convert_int_sat_rtz(uchar x);
int2 convert_int2_sat_rtz(uchar2 x);
int3 convert_int3_sat_rtz(uchar3 x);
int4 convert_int4_sat_rtz(uchar4 x);
int8 convert_int8_sat_rtz(uchar8 x);
int16 convert_int16_sat_rtz(uchar16 x);

int convert_int(schar x);
int2 convert_int2(char2 x);
int3 convert_int3(char3 x);
int4 convert_int4(char4 x);
int8 convert_int8(char8 x);
int16 convert_int16(char16 x);

int convert_int_rte(schar x);
int2 convert_int2_rte(char2 x);
int3 convert_int3_rte(char3 x);
int4 convert_int4_rte(char4 x);
int8 convert_int8_rte(char8 x);
int16 convert_int16_rte(char16 x);

int convert_int_rtp(schar x);
int2 convert_int2_rtp(char2 x);
int3 convert_int3_rtp(char3 x);
int4 convert_int4_rtp(char4 x);
int8 convert_int8_rtp(char8 x);
int16 convert_int16_rtp(char16 x);

int convert_int_rtn(schar x);
int2 convert_int2_rtn(char2 x);
int3 convert_int3_rtn(char3 x);
int4 convert_int4_rtn(char4 x);
int8 convert_int8_rtn(char8 x);
int16 convert_int16_rtn(char16 x);

int convert_int_rtz(schar x);
int2 convert_int2_rtz(char2 x);
int3 convert_int3_rtz(char3 x);
int4 convert_int4_rtz(char4 x);
int8 convert_int8_rtz(char8 x);
int16 convert_int16_rtz(char16 x);

int convert_int_sat(schar x);
int2 convert_int2_sat(char2 x);
int3 convert_int3_sat(char3 x);
int4 convert_int4_sat(char4 x);
int8 convert_int8_sat(char8 x);
int16 convert_int16_sat(char16 x);

int convert_int_sat_rte(schar x);
int2 convert_int2_sat_rte(char2 x);
int3 convert_int3_sat_rte(char3 x);
int4 convert_int4_sat_rte(char4 x);
int8 convert_int8_sat_rte(char8 x);
int16 convert_int16_sat_rte(char16 x);

int convert_int_sat_rtp(schar x);
int2 convert_int2_sat_rtp(char2 x);
int3 convert_int3_sat_rtp(char3 x);
int4 convert_int4_sat_rtp(char4 x);
int8 convert_int8_sat_rtp(char8 x);
int16 convert_int16_sat_rtp(char16 x);

int convert_int_sat_rtn(schar x);
int2 convert_int2_sat_rtn(char2 x);
int3 convert_int3_sat_rtn(char3 x);
int4 convert_int4_sat_rtn(char4 x);
int8 convert_int8_sat_rtn(char8 x);
int16 convert_int16_sat_rtn(char16 x);

int convert_int_sat_rtz(schar x);
int2 convert_int2_sat_rtz(char2 x);
int3 convert_int3_sat_rtz(char3 x);
int4 convert_int4_sat_rtz(char4 x);
int8 convert_int8_sat_rtz(char8 x);
int16 convert_int16_sat_rtz(char16 x);

int convert_int(ushort x);
int2 convert_int2(ushort2 x);
int3 convert_int3(ushort3 x);
int4 convert_int4(ushort4 x);
int8 convert_int8(ushort8 x);
int16 convert_int16(ushort16 x);

int convert_int_rte(ushort x);
int2 convert_int2_rte(ushort2 x);
int3 convert_int3_rte(ushort3 x);
int4 convert_int4_rte(ushort4 x);
int8 convert_int8_rte(ushort8 x);
int16 convert_int16_rte(ushort16 x);

int convert_int_rtp(ushort x);
int2 convert_int2_rtp(ushort2 x);
int3 convert_int3_rtp(ushort3 x);
int4 convert_int4_rtp(ushort4 x);
int8 convert_int8_rtp(ushort8 x);
int16 convert_int16_rtp(ushort16 x);

int convert_int_rtn(ushort x);
int2 convert_int2_rtn(ushort2 x);
int3 convert_int3_rtn(ushort3 x);
int4 convert_int4_rtn(ushort4 x);
int8 convert_int8_rtn(ushort8 x);
int16 convert_int16_rtn(ushort16 x);

int convert_int_rtz(ushort x);
int2 convert_int2_rtz(ushort2 x);
int3 convert_int3_rtz(ushort3 x);
int4 convert_int4_rtz(ushort4 x);
int8 convert_int8_rtz(ushort8 x);
int16 convert_int16_rtz(ushort16 x);

int convert_int_sat(ushort x);
int2 convert_int2_sat(ushort2 x);
int3 convert_int3_sat(ushort3 x);
int4 convert_int4_sat(ushort4 x);
int8 convert_int8_sat(ushort8 x);
int16 convert_int16_sat(ushort16 x);

int convert_int_sat_rte(ushort x);
int2 convert_int2_sat_rte(ushort2 x);
int3 convert_int3_sat_rte(ushort3 x);
int4 convert_int4_sat_rte(ushort4 x);
int8 convert_int8_sat_rte(ushort8 x);
int16 convert_int16_sat_rte(ushort16 x);

int convert_int_sat_rtp(ushort x);
int2 convert_int2_sat_rtp(ushort2 x);
int3 convert_int3_sat_rtp(ushort3 x);
int4 convert_int4_sat_rtp(ushort4 x);
int8 convert_int8_sat_rtp(ushort8 x);
int16 convert_int16_sat_rtp(ushort16 x);

int convert_int_sat_rtn(ushort x);
int2 convert_int2_sat_rtn(ushort2 x);
int3 convert_int3_sat_rtn(ushort3 x);
int4 convert_int4_sat_rtn(ushort4 x);
int8 convert_int8_sat_rtn(ushort8 x);
int16 convert_int16_sat_rtn(ushort16 x);

int convert_int_sat_rtz(ushort x);
int2 convert_int2_sat_rtz(ushort2 x);
int3 convert_int3_sat_rtz(ushort3 x);
int4 convert_int4_sat_rtz(ushort4 x);
int8 convert_int8_sat_rtz(ushort8 x);
int16 convert_int16_sat_rtz(ushort16 x);

int convert_int(short x);
int2 convert_int2(short2 x);
int3 convert_int3(short3 x);
int4 convert_int4(short4 x);
int8 convert_int8(short8 x);
int16 convert_int16(short16 x);

int convert_int_rte(short x);
int2 convert_int2_rte(short2 x);
int3 convert_int3_rte(short3 x);
int4 convert_int4_rte(short4 x);
int8 convert_int8_rte(short8 x);
int16 convert_int16_rte(short16 x);

int convert_int_rtp(short x);
int2 convert_int2_rtp(short2 x);
int3 convert_int3_rtp(short3 x);
int4 convert_int4_rtp(short4 x);
int8 convert_int8_rtp(short8 x);
int16 convert_int16_rtp(short16 x);

int convert_int_rtn(short x);
int2 convert_int2_rtn(short2 x);
int3 convert_int3_rtn(short3 x);
int4 convert_int4_rtn(short4 x);
int8 convert_int8_rtn(short8 x);
int16 convert_int16_rtn(short16 x);

int convert_int_rtz(short x);
int2 convert_int2_rtz(short2 x);
int3 convert_int3_rtz(short3 x);
int4 convert_int4_rtz(short4 x);
int8 convert_int8_rtz(short8 x);
int16 convert_int16_rtz(short16 x);

int convert_int_sat(short x);
int2 convert_int2_sat(short2 x);
int3 convert_int3_sat(short3 x);
int4 convert_int4_sat(short4 x);
int8 convert_int8_sat(short8 x);
int16 convert_int16_sat(short16 x);

int convert_int_sat_rte(short x);
int2 convert_int2_sat_rte(short2 x);
int3 convert_int3_sat_rte(short3 x);
int4 convert_int4_sat_rte(short4 x);
int8 convert_int8_sat_rte(short8 x);
int16 convert_int16_sat_rte(short16 x);

int convert_int_sat_rtp(short x);
int2 convert_int2_sat_rtp(short2 x);
int3 convert_int3_sat_rtp(short3 x);
int4 convert_int4_sat_rtp(short4 x);
int8 convert_int8_sat_rtp(short8 x);
int16 convert_int16_sat_rtp(short16 x);

int convert_int_sat_rtn(short x);
int2 convert_int2_sat_rtn(short2 x);
int3 convert_int3_sat_rtn(short3 x);
int4 convert_int4_sat_rtn(short4 x);
int8 convert_int8_sat_rtn(short8 x);
int16 convert_int16_sat_rtn(short16 x);

int convert_int_sat_rtz(short x);
int2 convert_int2_sat_rtz(short2 x);
int3 convert_int3_sat_rtz(short3 x);
int4 convert_int4_sat_rtz(short4 x);
int8 convert_int8_sat_rtz(short8 x);
int16 convert_int16_sat_rtz(short16 x);

int convert_int(uint x);
int2 convert_int2(uint2 x);
int3 convert_int3(uint3 x);
int4 convert_int4(uint4 x);
int8 convert_int8(uint8 x);
int16 convert_int16(uint16 x);

int convert_int_rte(uint x);
int2 convert_int2_rte(uint2 x);
int3 convert_int3_rte(uint3 x);
int4 convert_int4_rte(uint4 x);
int8 convert_int8_rte(uint8 x);
int16 convert_int16_rte(uint16 x);

int convert_int_rtp(uint x);
int2 convert_int2_rtp(uint2 x);
int3 convert_int3_rtp(uint3 x);
int4 convert_int4_rtp(uint4 x);
int8 convert_int8_rtp(uint8 x);
int16 convert_int16_rtp(uint16 x);

int convert_int_rtn(uint x);
int2 convert_int2_rtn(uint2 x);
int3 convert_int3_rtn(uint3 x);
int4 convert_int4_rtn(uint4 x);
int8 convert_int8_rtn(uint8 x);
int16 convert_int16_rtn(uint16 x);

int convert_int_rtz(uint x);
int2 convert_int2_rtz(uint2 x);
int3 convert_int3_rtz(uint3 x);
int4 convert_int4_rtz(uint4 x);
int8 convert_int8_rtz(uint8 x);
int16 convert_int16_rtz(uint16 x);

int convert_int_sat(uint x);
int2 convert_int2_sat(uint2 x);
int3 convert_int3_sat(uint3 x);
int4 convert_int4_sat(uint4 x);
int8 convert_int8_sat(uint8 x);
int16 convert_int16_sat(uint16 x);

int convert_int_sat_rte(uint x);
int2 convert_int2_sat_rte(uint2 x);
int3 convert_int3_sat_rte(uint3 x);
int4 convert_int4_sat_rte(uint4 x);
int8 convert_int8_sat_rte(uint8 x);
int16 convert_int16_sat_rte(uint16 x);

int convert_int_sat_rtp(uint x);
int2 convert_int2_sat_rtp(uint2 x);
int3 convert_int3_sat_rtp(uint3 x);
int4 convert_int4_sat_rtp(uint4 x);
int8 convert_int8_sat_rtp(uint8 x);
int16 convert_int16_sat_rtp(uint16 x);

int convert_int_sat_rtn(uint x);
int2 convert_int2_sat_rtn(uint2 x);
int3 convert_int3_sat_rtn(uint3 x);
int4 convert_int4_sat_rtn(uint4 x);
int8 convert_int8_sat_rtn(uint8 x);
int16 convert_int16_sat_rtn(uint16 x);

int convert_int_sat_rtz(uint x);
int2 convert_int2_sat_rtz(uint2 x);
int3 convert_int3_sat_rtz(uint3 x);
int4 convert_int4_sat_rtz(uint4 x);
int8 convert_int8_sat_rtz(uint8 x);
int16 convert_int16_sat_rtz(uint16 x);

int convert_int(int x);
int2 convert_int2(int2 x);
int3 convert_int3(int3 x);
int4 convert_int4(int4 x);
int8 convert_int8(int8 x);
int16 convert_int16(int16 x);

int convert_int_rte(int x);
int2 convert_int2_rte(int2 x);
int3 convert_int3_rte(int3 x);
int4 convert_int4_rte(int4 x);
int8 convert_int8_rte(int8 x);
int16 convert_int16_rte(int16 x);

int convert_int_rtp(int x);
int2 convert_int2_rtp(int2 x);
int3 convert_int3_rtp(int3 x);
int4 convert_int4_rtp(int4 x);
int8 convert_int8_rtp(int8 x);
int16 convert_int16_rtp(int16 x);

int convert_int_rtn(int x);
int2 convert_int2_rtn(int2 x);
int3 convert_int3_rtn(int3 x);
int4 convert_int4_rtn(int4 x);
int8 convert_int8_rtn(int8 x);
int16 convert_int16_rtn(int16 x);

int convert_int_rtz(int x);
int2 convert_int2_rtz(int2 x);
int3 convert_int3_rtz(int3 x);
int4 convert_int4_rtz(int4 x);
int8 convert_int8_rtz(int8 x);
int16 convert_int16_rtz(int16 x);

int convert_int_sat(int x);
int2 convert_int2_sat(int2 x);
int3 convert_int3_sat(int3 x);
int4 convert_int4_sat(int4 x);
int8 convert_int8_sat(int8 x);
int16 convert_int16_sat(int16 x);

int convert_int_sat_rte(int x);
int2 convert_int2_sat_rte(int2 x);
int3 convert_int3_sat_rte(int3 x);
int4 convert_int4_sat_rte(int4 x);
int8 convert_int8_sat_rte(int8 x);
int16 convert_int16_sat_rte(int16 x);

int convert_int_sat_rtp(int x);
int2 convert_int2_sat_rtp(int2 x);
int3 convert_int3_sat_rtp(int3 x);
int4 convert_int4_sat_rtp(int4 x);
int8 convert_int8_sat_rtp(int8 x);
int16 convert_int16_sat_rtp(int16 x);

int convert_int_sat_rtn(int x);
int2 convert_int2_sat_rtn(int2 x);
int3 convert_int3_sat_rtn(int3 x);
int4 convert_int4_sat_rtn(int4 x);
int8 convert_int8_sat_rtn(int8 x);
int16 convert_int16_sat_rtn(int16 x);

int convert_int_sat_rtz(int x);
int2 convert_int2_sat_rtz(int2 x);
int3 convert_int3_sat_rtz(int3 x);
int4 convert_int4_sat_rtz(int4 x);
int8 convert_int8_sat_rtz(int8 x);
int16 convert_int16_sat_rtz(int16 x);

int convert_int(float x);
int2 convert_int2(float2 x);
int3 convert_int3(float3 x);
int4 convert_int4(float4 x);
int8 convert_int8(float8 x);
int16 convert_int16(float16 x);

int convert_int_rte(float x);
int2 convert_int2_rte(float2 x);
int3 convert_int3_rte(float3 x);
int4 convert_int4_rte(float4 x);
int8 convert_int8_rte(float8 x);
int16 convert_int16_rte(float16 x);

int convert_int_rtp(float x);
int2 convert_int2_rtp(float2 x);
int3 convert_int3_rtp(float3 x);
int4 convert_int4_rtp(float4 x);
int8 convert_int8_rtp(float8 x);
int16 convert_int16_rtp(float16 x);

int convert_int_rtn(float x);
int2 convert_int2_rtn(float2 x);
int3 convert_int3_rtn(float3 x);
int4 convert_int4_rtn(float4 x);
int8 convert_int8_rtn(float8 x);
int16 convert_int16_rtn(float16 x);

int convert_int_rtz(float x);
int2 convert_int2_rtz(float2 x);
int3 convert_int3_rtz(float3 x);
int4 convert_int4_rtz(float4 x);
int8 convert_int8_rtz(float8 x);
int16 convert_int16_rtz(float16 x);

int convert_int_sat(float x);
int2 convert_int2_sat(float2 x);
int3 convert_int3_sat(float3 x);
int4 convert_int4_sat(float4 x);
int8 convert_int8_sat(float8 x);
int16 convert_int16_sat(float16 x);

int convert_int_sat_rte(float x);
int2 convert_int2_sat_rte(float2 x);
int3 convert_int3_sat_rte(float3 x);
int4 convert_int4_sat_rte(float4 x);
int8 convert_int8_sat_rte(float8 x);
int16 convert_int16_sat_rte(float16 x);

int convert_int_sat_rtp(float x);
int2 convert_int2_sat_rtp(float2 x);
int3 convert_int3_sat_rtp(float3 x);
int4 convert_int4_sat_rtp(float4 x);
int8 convert_int8_sat_rtp(float8 x);
int16 convert_int16_sat_rtp(float16 x);

int convert_int_sat_rtn(float x);
int2 convert_int2_sat_rtn(float2 x);
int3 convert_int3_sat_rtn(float3 x);
int4 convert_int4_sat_rtn(float4 x);
int8 convert_int8_sat_rtn(float8 x);
int16 convert_int16_sat_rtn(float16 x);

int convert_int_sat_rtz(float x);
int2 convert_int2_sat_rtz(float2 x);
int3 convert_int3_sat_rtz(float3 x);
int4 convert_int4_sat_rtz(float4 x);
int8 convert_int8_sat_rtz(float8 x);
int16 convert_int16_sat_rtz(float16 x);

int convert_int(double x);
int2 convert_int2(double2 x);
int3 convert_int3(double3 x);
int4 convert_int4(double4 x);
int8 convert_int8(double8 x);
int16 convert_int16(double16 x);

int convert_int_rte(double x);
int2 convert_int2_rte(double2 x);
int3 convert_int3_rte(double3 x);
int4 convert_int4_rte(double4 x);
int8 convert_int8_rte(double8 x);
int16 convert_int16_rte(double16 x);

int convert_int_rtp(double x);
int2 convert_int2_rtp(double2 x);
int3 convert_int3_rtp(double3 x);
int4 convert_int4_rtp(double4 x);
int8 convert_int8_rtp(double8 x);
int16 convert_int16_rtp(double16 x);

int convert_int_rtn(double x);
int2 convert_int2_rtn(double2 x);
int3 convert_int3_rtn(double3 x);
int4 convert_int4_rtn(double4 x);
int8 convert_int8_rtn(double8 x);
int16 convert_int16_rtn(double16 x);

int convert_int_rtz(double x);
int2 convert_int2_rtz(double2 x);
int3 convert_int3_rtz(double3 x);
int4 convert_int4_rtz(double4 x);
int8 convert_int8_rtz(double8 x);
int16 convert_int16_rtz(double16 x);

int convert_int_sat(double x);
int2 convert_int2_sat(double2 x);
int3 convert_int3_sat(double3 x);
int4 convert_int4_sat(double4 x);
int8 convert_int8_sat(double8 x);
int16 convert_int16_sat(double16 x);

int convert_int_sat_rte(double x);
int2 convert_int2_sat_rte(double2 x);
int3 convert_int3_sat_rte(double3 x);
int4 convert_int4_sat_rte(double4 x);
int8 convert_int8_sat_rte(double8 x);
int16 convert_int16_sat_rte(double16 x);

int convert_int_sat_rtp(double x);
int2 convert_int2_sat_rtp(double2 x);
int3 convert_int3_sat_rtp(double3 x);
int4 convert_int4_sat_rtp(double4 x);
int8 convert_int8_sat_rtp(double8 x);
int16 convert_int16_sat_rtp(double16 x);

int convert_int_sat_rtn(double x);
int2 convert_int2_sat_rtn(double2 x);
int3 convert_int3_sat_rtn(double3 x);
int4 convert_int4_sat_rtn(double4 x);
int8 convert_int8_sat_rtn(double8 x);
int16 convert_int16_sat_rtn(double16 x);

int convert_int_sat_rtz(double x);
int2 convert_int2_sat_rtz(double2 x);
int3 convert_int3_sat_rtz(double3 x);
int4 convert_int4_sat_rtz(double4 x);
int8 convert_int8_sat_rtz(double8 x);
int16 convert_int16_sat_rtz(double16 x);

int convert_int(ullong x);
int2 convert_int2(ulong2 x);
int3 convert_int3(ulong3 x);
int4 convert_int4(ulong4 x);
int8 convert_int8(ulong8 x);
int16 convert_int16(ulong16 x);

int convert_int_rte(ullong x);
int2 convert_int2_rte(ulong2 x);
int3 convert_int3_rte(ulong3 x);
int4 convert_int4_rte(ulong4 x);
int8 convert_int8_rte(ulong8 x);
int16 convert_int16_rte(ulong16 x);

int convert_int_rtp(ullong x);
int2 convert_int2_rtp(ulong2 x);
int3 convert_int3_rtp(ulong3 x);
int4 convert_int4_rtp(ulong4 x);
int8 convert_int8_rtp(ulong8 x);
int16 convert_int16_rtp(ulong16 x);

int convert_int_rtn(ullong x);
int2 convert_int2_rtn(ulong2 x);
int3 convert_int3_rtn(ulong3 x);
int4 convert_int4_rtn(ulong4 x);
int8 convert_int8_rtn(ulong8 x);
int16 convert_int16_rtn(ulong16 x);

int convert_int_rtz(ullong x);
int2 convert_int2_rtz(ulong2 x);
int3 convert_int3_rtz(ulong3 x);
int4 convert_int4_rtz(ulong4 x);
int8 convert_int8_rtz(ulong8 x);
int16 convert_int16_rtz(ulong16 x);

int convert_int_sat(ullong x);
int2 convert_int2_sat(ulong2 x);
int3 convert_int3_sat(ulong3 x);
int4 convert_int4_sat(ulong4 x);
int8 convert_int8_sat(ulong8 x);
int16 convert_int16_sat(ulong16 x);

int convert_int_sat_rte(ullong x);
int2 convert_int2_sat_rte(ulong2 x);
int3 convert_int3_sat_rte(ulong3 x);
int4 convert_int4_sat_rte(ulong4 x);
int8 convert_int8_sat_rte(ulong8 x);
int16 convert_int16_sat_rte(ulong16 x);

int convert_int_sat_rtp(ullong x);
int2 convert_int2_sat_rtp(ulong2 x);
int3 convert_int3_sat_rtp(ulong3 x);
int4 convert_int4_sat_rtp(ulong4 x);
int8 convert_int8_sat_rtp(ulong8 x);
int16 convert_int16_sat_rtp(ulong16 x);

int convert_int_sat_rtn(ullong x);
int2 convert_int2_sat_rtn(ulong2 x);
int3 convert_int3_sat_rtn(ulong3 x);
int4 convert_int4_sat_rtn(ulong4 x);
int8 convert_int8_sat_rtn(ulong8 x);
int16 convert_int16_sat_rtn(ulong16 x);

int convert_int_sat_rtz(ullong x);
int2 convert_int2_sat_rtz(ulong2 x);
int3 convert_int3_sat_rtz(ulong3 x);
int4 convert_int4_sat_rtz(ulong4 x);
int8 convert_int8_sat_rtz(ulong8 x);
int16 convert_int16_sat_rtz(ulong16 x);

int convert_int(llong x);
int2 convert_int2(long2 x);
int3 convert_int3(long3 x);
int4 convert_int4(long4 x);
int8 convert_int8(long8 x);
int16 convert_int16(long16 x);

int convert_int_rte(llong x);
int2 convert_int2_rte(long2 x);
int3 convert_int3_rte(long3 x);
int4 convert_int4_rte(long4 x);
int8 convert_int8_rte(long8 x);
int16 convert_int16_rte(long16 x);

int convert_int_rtp(llong x);
int2 convert_int2_rtp(long2 x);
int3 convert_int3_rtp(long3 x);
int4 convert_int4_rtp(long4 x);
int8 convert_int8_rtp(long8 x);
int16 convert_int16_rtp(long16 x);

int convert_int_rtn(llong x);
int2 convert_int2_rtn(long2 x);
int3 convert_int3_rtn(long3 x);
int4 convert_int4_rtn(long4 x);
int8 convert_int8_rtn(long8 x);
int16 convert_int16_rtn(long16 x);

int convert_int_rtz(llong x);
int2 convert_int2_rtz(long2 x);
int3 convert_int3_rtz(long3 x);
int4 convert_int4_rtz(long4 x);
int8 convert_int8_rtz(long8 x);
int16 convert_int16_rtz(long16 x);

int convert_int_sat(llong x);
int2 convert_int2_sat(long2 x);
int3 convert_int3_sat(long3 x);
int4 convert_int4_sat(long4 x);
int8 convert_int8_sat(long8 x);
int16 convert_int16_sat(long16 x);

int convert_int_sat_rte(llong x);
int2 convert_int2_sat_rte(long2 x);
int3 convert_int3_sat_rte(long3 x);
int4 convert_int4_sat_rte(long4 x);
int8 convert_int8_sat_rte(long8 x);
int16 convert_int16_sat_rte(long16 x);

int convert_int_sat_rtp(llong x);
int2 convert_int2_sat_rtp(long2 x);
int3 convert_int3_sat_rtp(long3 x);
int4 convert_int4_sat_rtp(long4 x);
int8 convert_int8_sat_rtp(long8 x);
int16 convert_int16_sat_rtp(long16 x);

int convert_int_sat_rtn(llong x);
int2 convert_int2_sat_rtn(long2 x);
int3 convert_int3_sat_rtn(long3 x);
int4 convert_int4_sat_rtn(long4 x);
int8 convert_int8_sat_rtn(long8 x);
int16 convert_int16_sat_rtn(long16 x);

int convert_int_sat_rtz(llong x);
int2 convert_int2_sat_rtz(long2 x);
int3 convert_int3_sat_rtz(long3 x);
int4 convert_int4_sat_rtz(long4 x);
int8 convert_int8_sat_rtz(long8 x);
int16 convert_int16_sat_rtz(long16 x);

float convert_float(uchar x);
float2 convert_float2(uchar2 x);
float3 convert_float3(uchar3 x);
float4 convert_float4(uchar4 x);
float8 convert_float8(uchar8 x);
float16 convert_float16(uchar16 x);

float convert_float_rte(uchar x);
float2 convert_float2_rte(uchar2 x);
float3 convert_float3_rte(uchar3 x);
float4 convert_float4_rte(uchar4 x);
float8 convert_float8_rte(uchar8 x);
float16 convert_float16_rte(uchar16 x);

float convert_float_rtp(uchar x);
float2 convert_float2_rtp(uchar2 x);
float3 convert_float3_rtp(uchar3 x);
float4 convert_float4_rtp(uchar4 x);
float8 convert_float8_rtp(uchar8 x);
float16 convert_float16_rtp(uchar16 x);

float convert_float_rtn(uchar x);
float2 convert_float2_rtn(uchar2 x);
float3 convert_float3_rtn(uchar3 x);
float4 convert_float4_rtn(uchar4 x);
float8 convert_float8_rtn(uchar8 x);
float16 convert_float16_rtn(uchar16 x);

float convert_float_rtz(uchar x);
float2 convert_float2_rtz(uchar2 x);
float3 convert_float3_rtz(uchar3 x);
float4 convert_float4_rtz(uchar4 x);
float8 convert_float8_rtz(uchar8 x);
float16 convert_float16_rtz(uchar16 x);

float convert_float(schar x);
float2 convert_float2(char2 x);
float3 convert_float3(char3 x);
float4 convert_float4(char4 x);
float8 convert_float8(char8 x);
float16 convert_float16(char16 x);

float convert_float_rte(schar x);
float2 convert_float2_rte(char2 x);
float3 convert_float3_rte(char3 x);
float4 convert_float4_rte(char4 x);
float8 convert_float8_rte(char8 x);
float16 convert_float16_rte(char16 x);

float convert_float_rtp(schar x);
float2 convert_float2_rtp(char2 x);
float3 convert_float3_rtp(char3 x);
float4 convert_float4_rtp(char4 x);
float8 convert_float8_rtp(char8 x);
float16 convert_float16_rtp(char16 x);

float convert_float_rtn(schar x);
float2 convert_float2_rtn(char2 x);
float3 convert_float3_rtn(char3 x);
float4 convert_float4_rtn(char4 x);
float8 convert_float8_rtn(char8 x);
float16 convert_float16_rtn(char16 x);

float convert_float_rtz(schar x);
float2 convert_float2_rtz(char2 x);
float3 convert_float3_rtz(char3 x);
float4 convert_float4_rtz(char4 x);
float8 convert_float8_rtz(char8 x);
float16 convert_float16_rtz(char16 x);

float convert_float(ushort x);
float2 convert_float2(ushort2 x);
float3 convert_float3(ushort3 x);
float4 convert_float4(ushort4 x);
float8 convert_float8(ushort8 x);
float16 convert_float16(ushort16 x);

float convert_float_rte(ushort x);
float2 convert_float2_rte(ushort2 x);
float3 convert_float3_rte(ushort3 x);
float4 convert_float4_rte(ushort4 x);
float8 convert_float8_rte(ushort8 x);
float16 convert_float16_rte(ushort16 x);

float convert_float_rtp(ushort x);
float2 convert_float2_rtp(ushort2 x);
float3 convert_float3_rtp(ushort3 x);
float4 convert_float4_rtp(ushort4 x);
float8 convert_float8_rtp(ushort8 x);
float16 convert_float16_rtp(ushort16 x);

float convert_float_rtn(ushort x);
float2 convert_float2_rtn(ushort2 x);
float3 convert_float3_rtn(ushort3 x);
float4 convert_float4_rtn(ushort4 x);
float8 convert_float8_rtn(ushort8 x);
float16 convert_float16_rtn(ushort16 x);

float convert_float_rtz(ushort x);
float2 convert_float2_rtz(ushort2 x);
float3 convert_float3_rtz(ushort3 x);
float4 convert_float4_rtz(ushort4 x);
float8 convert_float8_rtz(ushort8 x);
float16 convert_float16_rtz(ushort16 x);

float convert_float(short x);
float2 convert_float2(short2 x);
float3 convert_float3(short3 x);
float4 convert_float4(short4 x);
float8 convert_float8(short8 x);
float16 convert_float16(short16 x);

float convert_float_rte(short x);
float2 convert_float2_rte(short2 x);
float3 convert_float3_rte(short3 x);
float4 convert_float4_rte(short4 x);
float8 convert_float8_rte(short8 x);
float16 convert_float16_rte(short16 x);

float convert_float_rtp(short x);
float2 convert_float2_rtp(short2 x);
float3 convert_float3_rtp(short3 x);
float4 convert_float4_rtp(short4 x);
float8 convert_float8_rtp(short8 x);
float16 convert_float16_rtp(short16 x);

float convert_float_rtn(short x);
float2 convert_float2_rtn(short2 x);
float3 convert_float3_rtn(short3 x);
float4 convert_float4_rtn(short4 x);
float8 convert_float8_rtn(short8 x);
float16 convert_float16_rtn(short16 x);

float convert_float_rtz(short x);
float2 convert_float2_rtz(short2 x);
float3 convert_float3_rtz(short3 x);
float4 convert_float4_rtz(short4 x);
float8 convert_float8_rtz(short8 x);
float16 convert_float16_rtz(short16 x);

float convert_float(uint x);
float2 convert_float2(uint2 x);
float3 convert_float3(uint3 x);
float4 convert_float4(uint4 x);
float8 convert_float8(uint8 x);
float16 convert_float16(uint16 x);

float convert_float_rte(uint x);
float2 convert_float2_rte(uint2 x);
float3 convert_float3_rte(uint3 x);
float4 convert_float4_rte(uint4 x);
float8 convert_float8_rte(uint8 x);
float16 convert_float16_rte(uint16 x);

float convert_float_rtp(uint x);
float2 convert_float2_rtp(uint2 x);
float3 convert_float3_rtp(uint3 x);
float4 convert_float4_rtp(uint4 x);
float8 convert_float8_rtp(uint8 x);
float16 convert_float16_rtp(uint16 x);

float convert_float_rtn(uint x);
float2 convert_float2_rtn(uint2 x);
float3 convert_float3_rtn(uint3 x);
float4 convert_float4_rtn(uint4 x);
float8 convert_float8_rtn(uint8 x);
float16 convert_float16_rtn(uint16 x);

float convert_float_rtz(uint x);
float2 convert_float2_rtz(uint2 x);
float3 convert_float3_rtz(uint3 x);
float4 convert_float4_rtz(uint4 x);
float8 convert_float8_rtz(uint8 x);
float16 convert_float16_rtz(uint16 x);

float convert_float(int x);
float2 convert_float2(int2 x);
float3 convert_float3(int3 x);
float4 convert_float4(int4 x);
float8 convert_float8(int8 x);
float16 convert_float16(int16 x);

float convert_float_rte(int x);
float2 convert_float2_rte(int2 x);
float3 convert_float3_rte(int3 x);
float4 convert_float4_rte(int4 x);
float8 convert_float8_rte(int8 x);
float16 convert_float16_rte(int16 x);

float convert_float_rtp(int x);
float2 convert_float2_rtp(int2 x);
float3 convert_float3_rtp(int3 x);
float4 convert_float4_rtp(int4 x);
float8 convert_float8_rtp(int8 x);
float16 convert_float16_rtp(int16 x);

float convert_float_rtn(int x);
float2 convert_float2_rtn(int2 x);
float3 convert_float3_rtn(int3 x);
float4 convert_float4_rtn(int4 x);
float8 convert_float8_rtn(int8 x);
float16 convert_float16_rtn(int16 x);

float convert_float_rtz(int x);
float2 convert_float2_rtz(int2 x);
float3 convert_float3_rtz(int3 x);
float4 convert_float4_rtz(int4 x);
float8 convert_float8_rtz(int8 x);
float16 convert_float16_rtz(int16 x);

float convert_float(float x);
float2 convert_float2(float2 x);
float3 convert_float3(float3 x);
float4 convert_float4(float4 x);
float8 convert_float8(float8 x);
float16 convert_float16(float16 x);

float convert_float_rte(float x);
float2 convert_float2_rte(float2 x);
float3 convert_float3_rte(float3 x);
float4 convert_float4_rte(float4 x);
float8 convert_float8_rte(float8 x);
float16 convert_float16_rte(float16 x);

float convert_float_rtp(float x);
float2 convert_float2_rtp(float2 x);
float3 convert_float3_rtp(float3 x);
float4 convert_float4_rtp(float4 x);
float8 convert_float8_rtp(float8 x);
float16 convert_float16_rtp(float16 x);

float convert_float_rtn(float x);
float2 convert_float2_rtn(float2 x);
float3 convert_float3_rtn(float3 x);
float4 convert_float4_rtn(float4 x);
float8 convert_float8_rtn(float8 x);
float16 convert_float16_rtn(float16 x);

float convert_float_rtz(float x);
float2 convert_float2_rtz(float2 x);
float3 convert_float3_rtz(float3 x);
float4 convert_float4_rtz(float4 x);
float8 convert_float8_rtz(float8 x);
float16 convert_float16_rtz(float16 x);

float convert_float(double x);
float2 convert_float2(double2 x);
float3 convert_float3(double3 x);
float4 convert_float4(double4 x);
float8 convert_float8(double8 x);
float16 convert_float16(double16 x);

float convert_float_rte(double x);
float2 convert_float2_rte(double2 x);
float3 convert_float3_rte(double3 x);
float4 convert_float4_rte(double4 x);
float8 convert_float8_rte(double8 x);
float16 convert_float16_rte(double16 x);

float convert_float_rtp(double x);
float2 convert_float2_rtp(double2 x);
float3 convert_float3_rtp(double3 x);
float4 convert_float4_rtp(double4 x);
float8 convert_float8_rtp(double8 x);
float16 convert_float16_rtp(double16 x);

float convert_float_rtn(double x);
float2 convert_float2_rtn(double2 x);
float3 convert_float3_rtn(double3 x);
float4 convert_float4_rtn(double4 x);
float8 convert_float8_rtn(double8 x);
float16 convert_float16_rtn(double16 x);

float convert_float_rtz(double x);
float2 convert_float2_rtz(double2 x);
float3 convert_float3_rtz(double3 x);
float4 convert_float4_rtz(double4 x);
float8 convert_float8_rtz(double8 x);
float16 convert_float16_rtz(double16 x);

float convert_float(ullong x);
float2 convert_float2(ulong2 x);
float3 convert_float3(ulong3 x);
float4 convert_float4(ulong4 x);
float8 convert_float8(ulong8 x);
float16 convert_float16(ulong16 x);

float convert_float_rte(ullong x);
float2 convert_float2_rte(ulong2 x);
float3 convert_float3_rte(ulong3 x);
float4 convert_float4_rte(ulong4 x);
float8 convert_float8_rte(ulong8 x);
float16 convert_float16_rte(ulong16 x);

float convert_float_rtp(ullong x);
float2 convert_float2_rtp(ulong2 x);
float3 convert_float3_rtp(ulong3 x);
float4 convert_float4_rtp(ulong4 x);
float8 convert_float8_rtp(ulong8 x);
float16 convert_float16_rtp(ulong16 x);

float convert_float_rtn(ullong x);
float2 convert_float2_rtn(ulong2 x);
float3 convert_float3_rtn(ulong3 x);
float4 convert_float4_rtn(ulong4 x);
float8 convert_float8_rtn(ulong8 x);
float16 convert_float16_rtn(ulong16 x);

float convert_float_rtz(ullong x);
float2 convert_float2_rtz(ulong2 x);
float3 convert_float3_rtz(ulong3 x);
float4 convert_float4_rtz(ulong4 x);
float8 convert_float8_rtz(ulong8 x);
float16 convert_float16_rtz(ulong16 x);

float convert_float(llong x);
float2 convert_float2(long2 x);
float3 convert_float3(long3 x);
float4 convert_float4(long4 x);
float8 convert_float8(long8 x);
float16 convert_float16(long16 x);

float convert_float_rte(llong x);
float2 convert_float2_rte(long2 x);
float3 convert_float3_rte(long3 x);
float4 convert_float4_rte(long4 x);
float8 convert_float8_rte(long8 x);
float16 convert_float16_rte(long16 x);

float convert_float_rtp(llong x);
float2 convert_float2_rtp(long2 x);
float3 convert_float3_rtp(long3 x);
float4 convert_float4_rtp(long4 x);
float8 convert_float8_rtp(long8 x);
float16 convert_float16_rtp(long16 x);

float convert_float_rtn(llong x);
float2 convert_float2_rtn(long2 x);
float3 convert_float3_rtn(long3 x);
float4 convert_float4_rtn(long4 x);
float8 convert_float8_rtn(long8 x);
float16 convert_float16_rtn(long16 x);

float convert_float_rtz(llong x);
float2 convert_float2_rtz(long2 x);
float3 convert_float3_rtz(long3 x);
float4 convert_float4_rtz(long4 x);
float8 convert_float8_rtz(long8 x);
float16 convert_float16_rtz(long16 x);

double convert_double(uchar x);
double2 convert_double2(uchar2 x);
double3 convert_double3(uchar3 x);
double4 convert_double4(uchar4 x);
double8 convert_double8(uchar8 x);
double16 convert_double16(uchar16 x);

double convert_double_rte(uchar x);
double2 convert_double2_rte(uchar2 x);
double3 convert_double3_rte(uchar3 x);
double4 convert_double4_rte(uchar4 x);
double8 convert_double8_rte(uchar8 x);
double16 convert_double16_rte(uchar16 x);

double convert_double_rtp(uchar x);
double2 convert_double2_rtp(uchar2 x);
double3 convert_double3_rtp(uchar3 x);
double4 convert_double4_rtp(uchar4 x);
double8 convert_double8_rtp(uchar8 x);
double16 convert_double16_rtp(uchar16 x);

double convert_double_rtn(uchar x);
double2 convert_double2_rtn(uchar2 x);
double3 convert_double3_rtn(uchar3 x);
double4 convert_double4_rtn(uchar4 x);
double8 convert_double8_rtn(uchar8 x);
double16 convert_double16_rtn(uchar16 x);

double convert_double_rtz(uchar x);
double2 convert_double2_rtz(uchar2 x);
double3 convert_double3_rtz(uchar3 x);
double4 convert_double4_rtz(uchar4 x);
double8 convert_double8_rtz(uchar8 x);
double16 convert_double16_rtz(uchar16 x);

double convert_double(schar x);
double2 convert_double2(char2 x);
double3 convert_double3(char3 x);
double4 convert_double4(char4 x);
double8 convert_double8(char8 x);
double16 convert_double16(char16 x);

double convert_double_rte(schar x);
double2 convert_double2_rte(char2 x);
double3 convert_double3_rte(char3 x);
double4 convert_double4_rte(char4 x);
double8 convert_double8_rte(char8 x);
double16 convert_double16_rte(char16 x);

double convert_double_rtp(schar x);
double2 convert_double2_rtp(char2 x);
double3 convert_double3_rtp(char3 x);
double4 convert_double4_rtp(char4 x);
double8 convert_double8_rtp(char8 x);
double16 convert_double16_rtp(char16 x);

double convert_double_rtn(schar x);
double2 convert_double2_rtn(char2 x);
double3 convert_double3_rtn(char3 x);
double4 convert_double4_rtn(char4 x);
double8 convert_double8_rtn(char8 x);
double16 convert_double16_rtn(char16 x);

double convert_double_rtz(schar x);
double2 convert_double2_rtz(char2 x);
double3 convert_double3_rtz(char3 x);
double4 convert_double4_rtz(char4 x);
double8 convert_double8_rtz(char8 x);
double16 convert_double16_rtz(char16 x);

double convert_double(ushort x);
double2 convert_double2(ushort2 x);
double3 convert_double3(ushort3 x);
double4 convert_double4(ushort4 x);
double8 convert_double8(ushort8 x);
double16 convert_double16(ushort16 x);

double convert_double_rte(ushort x);
double2 convert_double2_rte(ushort2 x);
double3 convert_double3_rte(ushort3 x);
double4 convert_double4_rte(ushort4 x);
double8 convert_double8_rte(ushort8 x);
double16 convert_double16_rte(ushort16 x);

double convert_double_rtp(ushort x);
double2 convert_double2_rtp(ushort2 x);
double3 convert_double3_rtp(ushort3 x);
double4 convert_double4_rtp(ushort4 x);
double8 convert_double8_rtp(ushort8 x);
double16 convert_double16_rtp(ushort16 x);

double convert_double_rtn(ushort x);
double2 convert_double2_rtn(ushort2 x);
double3 convert_double3_rtn(ushort3 x);
double4 convert_double4_rtn(ushort4 x);
double8 convert_double8_rtn(ushort8 x);
double16 convert_double16_rtn(ushort16 x);

double convert_double_rtz(ushort x);
double2 convert_double2_rtz(ushort2 x);
double3 convert_double3_rtz(ushort3 x);
double4 convert_double4_rtz(ushort4 x);
double8 convert_double8_rtz(ushort8 x);
double16 convert_double16_rtz(ushort16 x);

double convert_double(short x);
double2 convert_double2(short2 x);
double3 convert_double3(short3 x);
double4 convert_double4(short4 x);
double8 convert_double8(short8 x);
double16 convert_double16(short16 x);

double convert_double_rte(short x);
double2 convert_double2_rte(short2 x);
double3 convert_double3_rte(short3 x);
double4 convert_double4_rte(short4 x);
double8 convert_double8_rte(short8 x);
double16 convert_double16_rte(short16 x);

double convert_double_rtp(short x);
double2 convert_double2_rtp(short2 x);
double3 convert_double3_rtp(short3 x);
double4 convert_double4_rtp(short4 x);
double8 convert_double8_rtp(short8 x);
double16 convert_double16_rtp(short16 x);

double convert_double_rtn(short x);
double2 convert_double2_rtn(short2 x);
double3 convert_double3_rtn(short3 x);
double4 convert_double4_rtn(short4 x);
double8 convert_double8_rtn(short8 x);
double16 convert_double16_rtn(short16 x);

double convert_double_rtz(short x);
double2 convert_double2_rtz(short2 x);
double3 convert_double3_rtz(short3 x);
double4 convert_double4_rtz(short4 x);
double8 convert_double8_rtz(short8 x);
double16 convert_double16_rtz(short16 x);

double convert_double(uint x);
double2 convert_double2(uint2 x);
double3 convert_double3(uint3 x);
double4 convert_double4(uint4 x);
double8 convert_double8(uint8 x);
double16 convert_double16(uint16 x);

double convert_double_rte(uint x);
double2 convert_double2_rte(uint2 x);
double3 convert_double3_rte(uint3 x);
double4 convert_double4_rte(uint4 x);
double8 convert_double8_rte(uint8 x);
double16 convert_double16_rte(uint16 x);

double convert_double_rtp(uint x);
double2 convert_double2_rtp(uint2 x);
double3 convert_double3_rtp(uint3 x);
double4 convert_double4_rtp(uint4 x);
double8 convert_double8_rtp(uint8 x);
double16 convert_double16_rtp(uint16 x);

double convert_double_rtn(uint x);
double2 convert_double2_rtn(uint2 x);
double3 convert_double3_rtn(uint3 x);
double4 convert_double4_rtn(uint4 x);
double8 convert_double8_rtn(uint8 x);
double16 convert_double16_rtn(uint16 x);

double convert_double_rtz(uint x);
double2 convert_double2_rtz(uint2 x);
double3 convert_double3_rtz(uint3 x);
double4 convert_double4_rtz(uint4 x);
double8 convert_double8_rtz(uint8 x);
double16 convert_double16_rtz(uint16 x);

double convert_double(int x);
double2 convert_double2(int2 x);
double3 convert_double3(int3 x);
double4 convert_double4(int4 x);
double8 convert_double8(int8 x);
double16 convert_double16(int16 x);

double convert_double_rte(int x);
double2 convert_double2_rte(int2 x);
double3 convert_double3_rte(int3 x);
double4 convert_double4_rte(int4 x);
double8 convert_double8_rte(int8 x);
double16 convert_double16_rte(int16 x);

double convert_double_rtp(int x);
double2 convert_double2_rtp(int2 x);
double3 convert_double3_rtp(int3 x);
double4 convert_double4_rtp(int4 x);
double8 convert_double8_rtp(int8 x);
double16 convert_double16_rtp(int16 x);

double convert_double_rtn(int x);
double2 convert_double2_rtn(int2 x);
double3 convert_double3_rtn(int3 x);
double4 convert_double4_rtn(int4 x);
double8 convert_double8_rtn(int8 x);
double16 convert_double16_rtn(int16 x);

double convert_double_rtz(int x);
double2 convert_double2_rtz(int2 x);
double3 convert_double3_rtz(int3 x);
double4 convert_double4_rtz(int4 x);
double8 convert_double8_rtz(int8 x);
double16 convert_double16_rtz(int16 x);

double convert_double(float x);
double2 convert_double2(float2 x);
double3 convert_double3(float3 x);
double4 convert_double4(float4 x);
double8 convert_double8(float8 x);
double16 convert_double16(float16 x);

double convert_double_rte(float x);
double2 convert_double2_rte(float2 x);
double3 convert_double3_rte(float3 x);
double4 convert_double4_rte(float4 x);
double8 convert_double8_rte(float8 x);
double16 convert_double16_rte(float16 x);

double convert_double_rtp(float x);
double2 convert_double2_rtp(float2 x);
double3 convert_double3_rtp(float3 x);
double4 convert_double4_rtp(float4 x);
double8 convert_double8_rtp(float8 x);
double16 convert_double16_rtp(float16 x);

double convert_double_rtn(float x);
double2 convert_double2_rtn(float2 x);
double3 convert_double3_rtn(float3 x);
double4 convert_double4_rtn(float4 x);
double8 convert_double8_rtn(float8 x);
double16 convert_double16_rtn(float16 x);

double convert_double_rtz(float x);
double2 convert_double2_rtz(float2 x);
double3 convert_double3_rtz(float3 x);
double4 convert_double4_rtz(float4 x);
double8 convert_double8_rtz(float8 x);
double16 convert_double16_rtz(float16 x);

double convert_double(double x);
double2 convert_double2(double2 x);
double3 convert_double3(double3 x);
double4 convert_double4(double4 x);
double8 convert_double8(double8 x);
double16 convert_double16(double16 x);

double convert_double_rte(double x);
double2 convert_double2_rte(double2 x);
double3 convert_double3_rte(double3 x);
double4 convert_double4_rte(double4 x);
double8 convert_double8_rte(double8 x);
double16 convert_double16_rte(double16 x);

double convert_double_rtp(double x);
double2 convert_double2_rtp(double2 x);
double3 convert_double3_rtp(double3 x);
double4 convert_double4_rtp(double4 x);
double8 convert_double8_rtp(double8 x);
double16 convert_double16_rtp(double16 x);

double convert_double_rtn(double x);
double2 convert_double2_rtn(double2 x);
double3 convert_double3_rtn(double3 x);
double4 convert_double4_rtn(double4 x);
double8 convert_double8_rtn(double8 x);
double16 convert_double16_rtn(double16 x);

double convert_double_rtz(double x);
double2 convert_double2_rtz(double2 x);
double3 convert_double3_rtz(double3 x);
double4 convert_double4_rtz(double4 x);
double8 convert_double8_rtz(double8 x);
double16 convert_double16_rtz(double16 x);

double convert_double(ullong x);
double2 convert_double2(ulong2 x);
double3 convert_double3(ulong3 x);
double4 convert_double4(ulong4 x);
double8 convert_double8(ulong8 x);
double16 convert_double16(ulong16 x);

double convert_double_rte(ullong x);
double2 convert_double2_rte(ulong2 x);
double3 convert_double3_rte(ulong3 x);
double4 convert_double4_rte(ulong4 x);
double8 convert_double8_rte(ulong8 x);
double16 convert_double16_rte(ulong16 x);

double convert_double_rtp(ullong x);
double2 convert_double2_rtp(ulong2 x);
double3 convert_double3_rtp(ulong3 x);
double4 convert_double4_rtp(ulong4 x);
double8 convert_double8_rtp(ulong8 x);
double16 convert_double16_rtp(ulong16 x);

double convert_double_rtn(ullong x);
double2 convert_double2_rtn(ulong2 x);
double3 convert_double3_rtn(ulong3 x);
double4 convert_double4_rtn(ulong4 x);
double8 convert_double8_rtn(ulong8 x);
double16 convert_double16_rtn(ulong16 x);

double convert_double_rtz(ullong x);
double2 convert_double2_rtz(ulong2 x);
double3 convert_double3_rtz(ulong3 x);
double4 convert_double4_rtz(ulong4 x);
double8 convert_double8_rtz(ulong8 x);
double16 convert_double16_rtz(ulong16 x);

double convert_double(llong x);
double2 convert_double2(long2 x);
double3 convert_double3(long3 x);
double4 convert_double4(long4 x);
double8 convert_double8(long8 x);
double16 convert_double16(long16 x);

double convert_double_rte(llong x);
double2 convert_double2_rte(long2 x);
double3 convert_double3_rte(long3 x);
double4 convert_double4_rte(long4 x);
double8 convert_double8_rte(long8 x);
double16 convert_double16_rte(long16 x);

double convert_double_rtp(llong x);
double2 convert_double2_rtp(long2 x);
double3 convert_double3_rtp(long3 x);
double4 convert_double4_rtp(long4 x);
double8 convert_double8_rtp(long8 x);
double16 convert_double16_rtp(long16 x);

double convert_double_rtn(llong x);
double2 convert_double2_rtn(long2 x);
double3 convert_double3_rtn(long3 x);
double4 convert_double4_rtn(long4 x);
double8 convert_double8_rtn(long8 x);
double16 convert_double16_rtn(long16 x);

double convert_double_rtz(llong x);
double2 convert_double2_rtz(long2 x);
double3 convert_double3_rtz(long3 x);
double4 convert_double4_rtz(long4 x);
double8 convert_double8_rtz(long8 x);
double16 convert_double16_rtz(long16 x);

ullong convert_ulong(uchar x);
ulong2 convert_ulong2(uchar2 x);
ulong3 convert_ulong3(uchar3 x);
ulong4 convert_ulong4(uchar4 x);
ulong8 convert_ulong8(uchar8 x);
ulong16 convert_ulong16(uchar16 x);

ullong convert_ulong_rte(uchar x);
ulong2 convert_ulong2_rte(uchar2 x);
ulong3 convert_ulong3_rte(uchar3 x);
ulong4 convert_ulong4_rte(uchar4 x);
ulong8 convert_ulong8_rte(uchar8 x);
ulong16 convert_ulong16_rte(uchar16 x);

ullong convert_ulong_rtp(uchar x);
ulong2 convert_ulong2_rtp(uchar2 x);
ulong3 convert_ulong3_rtp(uchar3 x);
ulong4 convert_ulong4_rtp(uchar4 x);
ulong8 convert_ulong8_rtp(uchar8 x);
ulong16 convert_ulong16_rtp(uchar16 x);

ullong convert_ulong_rtn(uchar x);
ulong2 convert_ulong2_rtn(uchar2 x);
ulong3 convert_ulong3_rtn(uchar3 x);
ulong4 convert_ulong4_rtn(uchar4 x);
ulong8 convert_ulong8_rtn(uchar8 x);
ulong16 convert_ulong16_rtn(uchar16 x);

ullong convert_ulong_rtz(uchar x);
ulong2 convert_ulong2_rtz(uchar2 x);
ulong3 convert_ulong3_rtz(uchar3 x);
ulong4 convert_ulong4_rtz(uchar4 x);
ulong8 convert_ulong8_rtz(uchar8 x);
ulong16 convert_ulong16_rtz(uchar16 x);

ullong convert_ulong_sat(uchar x);
ulong2 convert_ulong2_sat(uchar2 x);
ulong3 convert_ulong3_sat(uchar3 x);
ulong4 convert_ulong4_sat(uchar4 x);
ulong8 convert_ulong8_sat(uchar8 x);
ulong16 convert_ulong16_sat(uchar16 x);

ullong convert_ulong_sat_rte(uchar x);
ulong2 convert_ulong2_sat_rte(uchar2 x);
ulong3 convert_ulong3_sat_rte(uchar3 x);
ulong4 convert_ulong4_sat_rte(uchar4 x);
ulong8 convert_ulong8_sat_rte(uchar8 x);
ulong16 convert_ulong16_sat_rte(uchar16 x);

ullong convert_ulong_sat_rtp(uchar x);
ulong2 convert_ulong2_sat_rtp(uchar2 x);
ulong3 convert_ulong3_sat_rtp(uchar3 x);
ulong4 convert_ulong4_sat_rtp(uchar4 x);
ulong8 convert_ulong8_sat_rtp(uchar8 x);
ulong16 convert_ulong16_sat_rtp(uchar16 x);

ullong convert_ulong_sat_rtn(uchar x);
ulong2 convert_ulong2_sat_rtn(uchar2 x);
ulong3 convert_ulong3_sat_rtn(uchar3 x);
ulong4 convert_ulong4_sat_rtn(uchar4 x);
ulong8 convert_ulong8_sat_rtn(uchar8 x);
ulong16 convert_ulong16_sat_rtn(uchar16 x);

ullong convert_ulong_sat_rtz(uchar x);
ulong2 convert_ulong2_sat_rtz(uchar2 x);
ulong3 convert_ulong3_sat_rtz(uchar3 x);
ulong4 convert_ulong4_sat_rtz(uchar4 x);
ulong8 convert_ulong8_sat_rtz(uchar8 x);
ulong16 convert_ulong16_sat_rtz(uchar16 x);

ullong convert_ulong(schar x);
ulong2 convert_ulong2(char2 x);
ulong3 convert_ulong3(char3 x);
ulong4 convert_ulong4(char4 x);
ulong8 convert_ulong8(char8 x);
ulong16 convert_ulong16(char16 x);

ullong convert_ulong_rte(schar x);
ulong2 convert_ulong2_rte(char2 x);
ulong3 convert_ulong3_rte(char3 x);
ulong4 convert_ulong4_rte(char4 x);
ulong8 convert_ulong8_rte(char8 x);
ulong16 convert_ulong16_rte(char16 x);

ullong convert_ulong_rtp(schar x);
ulong2 convert_ulong2_rtp(char2 x);
ulong3 convert_ulong3_rtp(char3 x);
ulong4 convert_ulong4_rtp(char4 x);
ulong8 convert_ulong8_rtp(char8 x);
ulong16 convert_ulong16_rtp(char16 x);

ullong convert_ulong_rtn(schar x);
ulong2 convert_ulong2_rtn(char2 x);
ulong3 convert_ulong3_rtn(char3 x);
ulong4 convert_ulong4_rtn(char4 x);
ulong8 convert_ulong8_rtn(char8 x);
ulong16 convert_ulong16_rtn(char16 x);

ullong convert_ulong_rtz(schar x);
ulong2 convert_ulong2_rtz(char2 x);
ulong3 convert_ulong3_rtz(char3 x);
ulong4 convert_ulong4_rtz(char4 x);
ulong8 convert_ulong8_rtz(char8 x);
ulong16 convert_ulong16_rtz(char16 x);

ullong convert_ulong_sat(schar x);
ulong2 convert_ulong2_sat(char2 x);
ulong3 convert_ulong3_sat(char3 x);
ulong4 convert_ulong4_sat(char4 x);
ulong8 convert_ulong8_sat(char8 x);
ulong16 convert_ulong16_sat(char16 x);

ullong convert_ulong_sat_rte(schar x);
ulong2 convert_ulong2_sat_rte(char2 x);
ulong3 convert_ulong3_sat_rte(char3 x);
ulong4 convert_ulong4_sat_rte(char4 x);
ulong8 convert_ulong8_sat_rte(char8 x);
ulong16 convert_ulong16_sat_rte(char16 x);

ullong convert_ulong_sat_rtp(schar x);
ulong2 convert_ulong2_sat_rtp(char2 x);
ulong3 convert_ulong3_sat_rtp(char3 x);
ulong4 convert_ulong4_sat_rtp(char4 x);
ulong8 convert_ulong8_sat_rtp(char8 x);
ulong16 convert_ulong16_sat_rtp(char16 x);

ullong convert_ulong_sat_rtn(schar x);
ulong2 convert_ulong2_sat_rtn(char2 x);
ulong3 convert_ulong3_sat_rtn(char3 x);
ulong4 convert_ulong4_sat_rtn(char4 x);
ulong8 convert_ulong8_sat_rtn(char8 x);
ulong16 convert_ulong16_sat_rtn(char16 x);

ullong convert_ulong_sat_rtz(schar x);
ulong2 convert_ulong2_sat_rtz(char2 x);
ulong3 convert_ulong3_sat_rtz(char3 x);
ulong4 convert_ulong4_sat_rtz(char4 x);
ulong8 convert_ulong8_sat_rtz(char8 x);
ulong16 convert_ulong16_sat_rtz(char16 x);

ullong convert_ulong(ushort x);
ulong2 convert_ulong2(ushort2 x);
ulong3 convert_ulong3(ushort3 x);
ulong4 convert_ulong4(ushort4 x);
ulong8 convert_ulong8(ushort8 x);
ulong16 convert_ulong16(ushort16 x);

ullong convert_ulong_rte(ushort x);
ulong2 convert_ulong2_rte(ushort2 x);
ulong3 convert_ulong3_rte(ushort3 x);
ulong4 convert_ulong4_rte(ushort4 x);
ulong8 convert_ulong8_rte(ushort8 x);
ulong16 convert_ulong16_rte(ushort16 x);

ullong convert_ulong_rtp(ushort x);
ulong2 convert_ulong2_rtp(ushort2 x);
ulong3 convert_ulong3_rtp(ushort3 x);
ulong4 convert_ulong4_rtp(ushort4 x);
ulong8 convert_ulong8_rtp(ushort8 x);
ulong16 convert_ulong16_rtp(ushort16 x);

ullong convert_ulong_rtn(ushort x);
ulong2 convert_ulong2_rtn(ushort2 x);
ulong3 convert_ulong3_rtn(ushort3 x);
ulong4 convert_ulong4_rtn(ushort4 x);
ulong8 convert_ulong8_rtn(ushort8 x);
ulong16 convert_ulong16_rtn(ushort16 x);

ullong convert_ulong_rtz(ushort x);
ulong2 convert_ulong2_rtz(ushort2 x);
ulong3 convert_ulong3_rtz(ushort3 x);
ulong4 convert_ulong4_rtz(ushort4 x);
ulong8 convert_ulong8_rtz(ushort8 x);
ulong16 convert_ulong16_rtz(ushort16 x);

ullong convert_ulong_sat(ushort x);
ulong2 convert_ulong2_sat(ushort2 x);
ulong3 convert_ulong3_sat(ushort3 x);
ulong4 convert_ulong4_sat(ushort4 x);
ulong8 convert_ulong8_sat(ushort8 x);
ulong16 convert_ulong16_sat(ushort16 x);

ullong convert_ulong_sat_rte(ushort x);
ulong2 convert_ulong2_sat_rte(ushort2 x);
ulong3 convert_ulong3_sat_rte(ushort3 x);
ulong4 convert_ulong4_sat_rte(ushort4 x);
ulong8 convert_ulong8_sat_rte(ushort8 x);
ulong16 convert_ulong16_sat_rte(ushort16 x);

ullong convert_ulong_sat_rtp(ushort x);
ulong2 convert_ulong2_sat_rtp(ushort2 x);
ulong3 convert_ulong3_sat_rtp(ushort3 x);
ulong4 convert_ulong4_sat_rtp(ushort4 x);
ulong8 convert_ulong8_sat_rtp(ushort8 x);
ulong16 convert_ulong16_sat_rtp(ushort16 x);

ullong convert_ulong_sat_rtn(ushort x);
ulong2 convert_ulong2_sat_rtn(ushort2 x);
ulong3 convert_ulong3_sat_rtn(ushort3 x);
ulong4 convert_ulong4_sat_rtn(ushort4 x);
ulong8 convert_ulong8_sat_rtn(ushort8 x);
ulong16 convert_ulong16_sat_rtn(ushort16 x);

ullong convert_ulong_sat_rtz(ushort x);
ulong2 convert_ulong2_sat_rtz(ushort2 x);
ulong3 convert_ulong3_sat_rtz(ushort3 x);
ulong4 convert_ulong4_sat_rtz(ushort4 x);
ulong8 convert_ulong8_sat_rtz(ushort8 x);
ulong16 convert_ulong16_sat_rtz(ushort16 x);

ullong convert_ulong(short x);
ulong2 convert_ulong2(short2 x);
ulong3 convert_ulong3(short3 x);
ulong4 convert_ulong4(short4 x);
ulong8 convert_ulong8(short8 x);
ulong16 convert_ulong16(short16 x);

ullong convert_ulong_rte(short x);
ulong2 convert_ulong2_rte(short2 x);
ulong3 convert_ulong3_rte(short3 x);
ulong4 convert_ulong4_rte(short4 x);
ulong8 convert_ulong8_rte(short8 x);
ulong16 convert_ulong16_rte(short16 x);

ullong convert_ulong_rtp(short x);
ulong2 convert_ulong2_rtp(short2 x);
ulong3 convert_ulong3_rtp(short3 x);
ulong4 convert_ulong4_rtp(short4 x);
ulong8 convert_ulong8_rtp(short8 x);
ulong16 convert_ulong16_rtp(short16 x);

ullong convert_ulong_rtn(short x);
ulong2 convert_ulong2_rtn(short2 x);
ulong3 convert_ulong3_rtn(short3 x);
ulong4 convert_ulong4_rtn(short4 x);
ulong8 convert_ulong8_rtn(short8 x);
ulong16 convert_ulong16_rtn(short16 x);

ullong convert_ulong_rtz(short x);
ulong2 convert_ulong2_rtz(short2 x);
ulong3 convert_ulong3_rtz(short3 x);
ulong4 convert_ulong4_rtz(short4 x);
ulong8 convert_ulong8_rtz(short8 x);
ulong16 convert_ulong16_rtz(short16 x);

ullong convert_ulong_sat(short x);
ulong2 convert_ulong2_sat(short2 x);
ulong3 convert_ulong3_sat(short3 x);
ulong4 convert_ulong4_sat(short4 x);
ulong8 convert_ulong8_sat(short8 x);
ulong16 convert_ulong16_sat(short16 x);

ullong convert_ulong_sat_rte(short x);
ulong2 convert_ulong2_sat_rte(short2 x);
ulong3 convert_ulong3_sat_rte(short3 x);
ulong4 convert_ulong4_sat_rte(short4 x);
ulong8 convert_ulong8_sat_rte(short8 x);
ulong16 convert_ulong16_sat_rte(short16 x);

ullong convert_ulong_sat_rtp(short x);
ulong2 convert_ulong2_sat_rtp(short2 x);
ulong3 convert_ulong3_sat_rtp(short3 x);
ulong4 convert_ulong4_sat_rtp(short4 x);
ulong8 convert_ulong8_sat_rtp(short8 x);
ulong16 convert_ulong16_sat_rtp(short16 x);

ullong convert_ulong_sat_rtn(short x);
ulong2 convert_ulong2_sat_rtn(short2 x);
ulong3 convert_ulong3_sat_rtn(short3 x);
ulong4 convert_ulong4_sat_rtn(short4 x);
ulong8 convert_ulong8_sat_rtn(short8 x);
ulong16 convert_ulong16_sat_rtn(short16 x);

ullong convert_ulong_sat_rtz(short x);
ulong2 convert_ulong2_sat_rtz(short2 x);
ulong3 convert_ulong3_sat_rtz(short3 x);
ulong4 convert_ulong4_sat_rtz(short4 x);
ulong8 convert_ulong8_sat_rtz(short8 x);
ulong16 convert_ulong16_sat_rtz(short16 x);

ullong convert_ulong(uint x);
ulong2 convert_ulong2(uint2 x);
ulong3 convert_ulong3(uint3 x);
ulong4 convert_ulong4(uint4 x);
ulong8 convert_ulong8(uint8 x);
ulong16 convert_ulong16(uint16 x);

ullong convert_ulong_rte(uint x);
ulong2 convert_ulong2_rte(uint2 x);
ulong3 convert_ulong3_rte(uint3 x);
ulong4 convert_ulong4_rte(uint4 x);
ulong8 convert_ulong8_rte(uint8 x);
ulong16 convert_ulong16_rte(uint16 x);

ullong convert_ulong_rtp(uint x);
ulong2 convert_ulong2_rtp(uint2 x);
ulong3 convert_ulong3_rtp(uint3 x);
ulong4 convert_ulong4_rtp(uint4 x);
ulong8 convert_ulong8_rtp(uint8 x);
ulong16 convert_ulong16_rtp(uint16 x);

ullong convert_ulong_rtn(uint x);
ulong2 convert_ulong2_rtn(uint2 x);
ulong3 convert_ulong3_rtn(uint3 x);
ulong4 convert_ulong4_rtn(uint4 x);
ulong8 convert_ulong8_rtn(uint8 x);
ulong16 convert_ulong16_rtn(uint16 x);

ullong convert_ulong_rtz(uint x);
ulong2 convert_ulong2_rtz(uint2 x);
ulong3 convert_ulong3_rtz(uint3 x);
ulong4 convert_ulong4_rtz(uint4 x);
ulong8 convert_ulong8_rtz(uint8 x);
ulong16 convert_ulong16_rtz(uint16 x);

ullong convert_ulong_sat(uint x);
ulong2 convert_ulong2_sat(uint2 x);
ulong3 convert_ulong3_sat(uint3 x);
ulong4 convert_ulong4_sat(uint4 x);
ulong8 convert_ulong8_sat(uint8 x);
ulong16 convert_ulong16_sat(uint16 x);

ullong convert_ulong_sat_rte(uint x);
ulong2 convert_ulong2_sat_rte(uint2 x);
ulong3 convert_ulong3_sat_rte(uint3 x);
ulong4 convert_ulong4_sat_rte(uint4 x);
ulong8 convert_ulong8_sat_rte(uint8 x);
ulong16 convert_ulong16_sat_rte(uint16 x);

ullong convert_ulong_sat_rtp(uint x);
ulong2 convert_ulong2_sat_rtp(uint2 x);
ulong3 convert_ulong3_sat_rtp(uint3 x);
ulong4 convert_ulong4_sat_rtp(uint4 x);
ulong8 convert_ulong8_sat_rtp(uint8 x);
ulong16 convert_ulong16_sat_rtp(uint16 x);

ullong convert_ulong_sat_rtn(uint x);
ulong2 convert_ulong2_sat_rtn(uint2 x);
ulong3 convert_ulong3_sat_rtn(uint3 x);
ulong4 convert_ulong4_sat_rtn(uint4 x);
ulong8 convert_ulong8_sat_rtn(uint8 x);
ulong16 convert_ulong16_sat_rtn(uint16 x);

ullong convert_ulong_sat_rtz(uint x);
ulong2 convert_ulong2_sat_rtz(uint2 x);
ulong3 convert_ulong3_sat_rtz(uint3 x);
ulong4 convert_ulong4_sat_rtz(uint4 x);
ulong8 convert_ulong8_sat_rtz(uint8 x);
ulong16 convert_ulong16_sat_rtz(uint16 x);

ullong convert_ulong(int x);
ulong2 convert_ulong2(int2 x);
ulong3 convert_ulong3(int3 x);
ulong4 convert_ulong4(int4 x);
ulong8 convert_ulong8(int8 x);
ulong16 convert_ulong16(int16 x);

ullong convert_ulong_rte(int x);
ulong2 convert_ulong2_rte(int2 x);
ulong3 convert_ulong3_rte(int3 x);
ulong4 convert_ulong4_rte(int4 x);
ulong8 convert_ulong8_rte(int8 x);
ulong16 convert_ulong16_rte(int16 x);

ullong convert_ulong_rtp(int x);
ulong2 convert_ulong2_rtp(int2 x);
ulong3 convert_ulong3_rtp(int3 x);
ulong4 convert_ulong4_rtp(int4 x);
ulong8 convert_ulong8_rtp(int8 x);
ulong16 convert_ulong16_rtp(int16 x);

ullong convert_ulong_rtn(int x);
ulong2 convert_ulong2_rtn(int2 x);
ulong3 convert_ulong3_rtn(int3 x);
ulong4 convert_ulong4_rtn(int4 x);
ulong8 convert_ulong8_rtn(int8 x);
ulong16 convert_ulong16_rtn(int16 x);

ullong convert_ulong_rtz(int x);
ulong2 convert_ulong2_rtz(int2 x);
ulong3 convert_ulong3_rtz(int3 x);
ulong4 convert_ulong4_rtz(int4 x);
ulong8 convert_ulong8_rtz(int8 x);
ulong16 convert_ulong16_rtz(int16 x);

ullong convert_ulong_sat(int x);
ulong2 convert_ulong2_sat(int2 x);
ulong3 convert_ulong3_sat(int3 x);
ulong4 convert_ulong4_sat(int4 x);
ulong8 convert_ulong8_sat(int8 x);
ulong16 convert_ulong16_sat(int16 x);

ullong convert_ulong_sat_rte(int x);
ulong2 convert_ulong2_sat_rte(int2 x);
ulong3 convert_ulong3_sat_rte(int3 x);
ulong4 convert_ulong4_sat_rte(int4 x);
ulong8 convert_ulong8_sat_rte(int8 x);
ulong16 convert_ulong16_sat_rte(int16 x);

ullong convert_ulong_sat_rtp(int x);
ulong2 convert_ulong2_sat_rtp(int2 x);
ulong3 convert_ulong3_sat_rtp(int3 x);
ulong4 convert_ulong4_sat_rtp(int4 x);
ulong8 convert_ulong8_sat_rtp(int8 x);
ulong16 convert_ulong16_sat_rtp(int16 x);

ullong convert_ulong_sat_rtn(int x);
ulong2 convert_ulong2_sat_rtn(int2 x);
ulong3 convert_ulong3_sat_rtn(int3 x);
ulong4 convert_ulong4_sat_rtn(int4 x);
ulong8 convert_ulong8_sat_rtn(int8 x);
ulong16 convert_ulong16_sat_rtn(int16 x);

ullong convert_ulong_sat_rtz(int x);
ulong2 convert_ulong2_sat_rtz(int2 x);
ulong3 convert_ulong3_sat_rtz(int3 x);
ulong4 convert_ulong4_sat_rtz(int4 x);
ulong8 convert_ulong8_sat_rtz(int8 x);
ulong16 convert_ulong16_sat_rtz(int16 x);

ullong convert_ulong(float x);
ulong2 convert_ulong2(float2 x);
ulong3 convert_ulong3(float3 x);
ulong4 convert_ulong4(float4 x);
ulong8 convert_ulong8(float8 x);
ulong16 convert_ulong16(float16 x);

ullong convert_ulong_rte(float x);
ulong2 convert_ulong2_rte(float2 x);
ulong3 convert_ulong3_rte(float3 x);
ulong4 convert_ulong4_rte(float4 x);
ulong8 convert_ulong8_rte(float8 x);
ulong16 convert_ulong16_rte(float16 x);

ullong convert_ulong_rtp(float x);
ulong2 convert_ulong2_rtp(float2 x);
ulong3 convert_ulong3_rtp(float3 x);
ulong4 convert_ulong4_rtp(float4 x);
ulong8 convert_ulong8_rtp(float8 x);
ulong16 convert_ulong16_rtp(float16 x);

ullong convert_ulong_rtn(float x);
ulong2 convert_ulong2_rtn(float2 x);
ulong3 convert_ulong3_rtn(float3 x);
ulong4 convert_ulong4_rtn(float4 x);
ulong8 convert_ulong8_rtn(float8 x);
ulong16 convert_ulong16_rtn(float16 x);

ullong convert_ulong_rtz(float x);
ulong2 convert_ulong2_rtz(float2 x);
ulong3 convert_ulong3_rtz(float3 x);
ulong4 convert_ulong4_rtz(float4 x);
ulong8 convert_ulong8_rtz(float8 x);
ulong16 convert_ulong16_rtz(float16 x);

ullong convert_ulong_sat(float x);
ulong2 convert_ulong2_sat(float2 x);
ulong3 convert_ulong3_sat(float3 x);
ulong4 convert_ulong4_sat(float4 x);
ulong8 convert_ulong8_sat(float8 x);
ulong16 convert_ulong16_sat(float16 x);

ullong convert_ulong_sat_rte(float x);
ulong2 convert_ulong2_sat_rte(float2 x);
ulong3 convert_ulong3_sat_rte(float3 x);
ulong4 convert_ulong4_sat_rte(float4 x);
ulong8 convert_ulong8_sat_rte(float8 x);
ulong16 convert_ulong16_sat_rte(float16 x);

ullong convert_ulong_sat_rtp(float x);
ulong2 convert_ulong2_sat_rtp(float2 x);
ulong3 convert_ulong3_sat_rtp(float3 x);
ulong4 convert_ulong4_sat_rtp(float4 x);
ulong8 convert_ulong8_sat_rtp(float8 x);
ulong16 convert_ulong16_sat_rtp(float16 x);

ullong convert_ulong_sat_rtn(float x);
ulong2 convert_ulong2_sat_rtn(float2 x);
ulong3 convert_ulong3_sat_rtn(float3 x);
ulong4 convert_ulong4_sat_rtn(float4 x);
ulong8 convert_ulong8_sat_rtn(float8 x);
ulong16 convert_ulong16_sat_rtn(float16 x);

ullong convert_ulong_sat_rtz(float x);
ulong2 convert_ulong2_sat_rtz(float2 x);
ulong3 convert_ulong3_sat_rtz(float3 x);
ulong4 convert_ulong4_sat_rtz(float4 x);
ulong8 convert_ulong8_sat_rtz(float8 x);
ulong16 convert_ulong16_sat_rtz(float16 x);

ullong convert_ulong(double x);
ulong2 convert_ulong2(double2 x);
ulong3 convert_ulong3(double3 x);
ulong4 convert_ulong4(double4 x);
ulong8 convert_ulong8(double8 x);
ulong16 convert_ulong16(double16 x);

ullong convert_ulong_rte(double x);
ulong2 convert_ulong2_rte(double2 x);
ulong3 convert_ulong3_rte(double3 x);
ulong4 convert_ulong4_rte(double4 x);
ulong8 convert_ulong8_rte(double8 x);
ulong16 convert_ulong16_rte(double16 x);

ullong convert_ulong_rtp(double x);
ulong2 convert_ulong2_rtp(double2 x);
ulong3 convert_ulong3_rtp(double3 x);
ulong4 convert_ulong4_rtp(double4 x);
ulong8 convert_ulong8_rtp(double8 x);
ulong16 convert_ulong16_rtp(double16 x);

ullong convert_ulong_rtn(double x);
ulong2 convert_ulong2_rtn(double2 x);
ulong3 convert_ulong3_rtn(double3 x);
ulong4 convert_ulong4_rtn(double4 x);
ulong8 convert_ulong8_rtn(double8 x);
ulong16 convert_ulong16_rtn(double16 x);

ullong convert_ulong_rtz(double x);
ulong2 convert_ulong2_rtz(double2 x);
ulong3 convert_ulong3_rtz(double3 x);
ulong4 convert_ulong4_rtz(double4 x);
ulong8 convert_ulong8_rtz(double8 x);
ulong16 convert_ulong16_rtz(double16 x);

ullong convert_ulong_sat(double x);
ulong2 convert_ulong2_sat(double2 x);
ulong3 convert_ulong3_sat(double3 x);
ulong4 convert_ulong4_sat(double4 x);
ulong8 convert_ulong8_sat(double8 x);
ulong16 convert_ulong16_sat(double16 x);

ullong convert_ulong_sat_rte(double x);
ulong2 convert_ulong2_sat_rte(double2 x);
ulong3 convert_ulong3_sat_rte(double3 x);
ulong4 convert_ulong4_sat_rte(double4 x);
ulong8 convert_ulong8_sat_rte(double8 x);
ulong16 convert_ulong16_sat_rte(double16 x);

ullong convert_ulong_sat_rtp(double x);
ulong2 convert_ulong2_sat_rtp(double2 x);
ulong3 convert_ulong3_sat_rtp(double3 x);
ulong4 convert_ulong4_sat_rtp(double4 x);
ulong8 convert_ulong8_sat_rtp(double8 x);
ulong16 convert_ulong16_sat_rtp(double16 x);

ullong convert_ulong_sat_rtn(double x);
ulong2 convert_ulong2_sat_rtn(double2 x);
ulong3 convert_ulong3_sat_rtn(double3 x);
ulong4 convert_ulong4_sat_rtn(double4 x);
ulong8 convert_ulong8_sat_rtn(double8 x);
ulong16 convert_ulong16_sat_rtn(double16 x);

ullong convert_ulong_sat_rtz(double x);
ulong2 convert_ulong2_sat_rtz(double2 x);
ulong3 convert_ulong3_sat_rtz(double3 x);
ulong4 convert_ulong4_sat_rtz(double4 x);
ulong8 convert_ulong8_sat_rtz(double8 x);
ulong16 convert_ulong16_sat_rtz(double16 x);

ullong convert_ulong(ullong x);
ulong2 convert_ulong2(ulong2 x);
ulong3 convert_ulong3(ulong3 x);
ulong4 convert_ulong4(ulong4 x);
ulong8 convert_ulong8(ulong8 x);
ulong16 convert_ulong16(ulong16 x);

ullong convert_ulong_rte(ullong x);
ulong2 convert_ulong2_rte(ulong2 x);
ulong3 convert_ulong3_rte(ulong3 x);
ulong4 convert_ulong4_rte(ulong4 x);
ulong8 convert_ulong8_rte(ulong8 x);
ulong16 convert_ulong16_rte(ulong16 x);

ullong convert_ulong_rtp(ullong x);
ulong2 convert_ulong2_rtp(ulong2 x);
ulong3 convert_ulong3_rtp(ulong3 x);
ulong4 convert_ulong4_rtp(ulong4 x);
ulong8 convert_ulong8_rtp(ulong8 x);
ulong16 convert_ulong16_rtp(ulong16 x);

ullong convert_ulong_rtn(ullong x);
ulong2 convert_ulong2_rtn(ulong2 x);
ulong3 convert_ulong3_rtn(ulong3 x);
ulong4 convert_ulong4_rtn(ulong4 x);
ulong8 convert_ulong8_rtn(ulong8 x);
ulong16 convert_ulong16_rtn(ulong16 x);

ullong convert_ulong_rtz(ullong x);
ulong2 convert_ulong2_rtz(ulong2 x);
ulong3 convert_ulong3_rtz(ulong3 x);
ulong4 convert_ulong4_rtz(ulong4 x);
ulong8 convert_ulong8_rtz(ulong8 x);
ulong16 convert_ulong16_rtz(ulong16 x);

ullong convert_ulong_sat(ullong x);
ulong2 convert_ulong2_sat(ulong2 x);
ulong3 convert_ulong3_sat(ulong3 x);
ulong4 convert_ulong4_sat(ulong4 x);
ulong8 convert_ulong8_sat(ulong8 x);
ulong16 convert_ulong16_sat(ulong16 x);

ullong convert_ulong_sat_rte(ullong x);
ulong2 convert_ulong2_sat_rte(ulong2 x);
ulong3 convert_ulong3_sat_rte(ulong3 x);
ulong4 convert_ulong4_sat_rte(ulong4 x);
ulong8 convert_ulong8_sat_rte(ulong8 x);
ulong16 convert_ulong16_sat_rte(ulong16 x);

ullong convert_ulong_sat_rtp(ullong x);
ulong2 convert_ulong2_sat_rtp(ulong2 x);
ulong3 convert_ulong3_sat_rtp(ulong3 x);
ulong4 convert_ulong4_sat_rtp(ulong4 x);
ulong8 convert_ulong8_sat_rtp(ulong8 x);
ulong16 convert_ulong16_sat_rtp(ulong16 x);

ullong convert_ulong_sat_rtn(ullong x);
ulong2 convert_ulong2_sat_rtn(ulong2 x);
ulong3 convert_ulong3_sat_rtn(ulong3 x);
ulong4 convert_ulong4_sat_rtn(ulong4 x);
ulong8 convert_ulong8_sat_rtn(ulong8 x);
ulong16 convert_ulong16_sat_rtn(ulong16 x);

ullong convert_ulong_sat_rtz(ullong x);
ulong2 convert_ulong2_sat_rtz(ulong2 x);
ulong3 convert_ulong3_sat_rtz(ulong3 x);
ulong4 convert_ulong4_sat_rtz(ulong4 x);
ulong8 convert_ulong8_sat_rtz(ulong8 x);
ulong16 convert_ulong16_sat_rtz(ulong16 x);

ullong convert_ulong(llong x);
ulong2 convert_ulong2(long2 x);
ulong3 convert_ulong3(long3 x);
ulong4 convert_ulong4(long4 x);
ulong8 convert_ulong8(long8 x);
ulong16 convert_ulong16(long16 x);

ullong convert_ulong_rte(llong x);
ulong2 convert_ulong2_rte(long2 x);
ulong3 convert_ulong3_rte(long3 x);
ulong4 convert_ulong4_rte(long4 x);
ulong8 convert_ulong8_rte(long8 x);
ulong16 convert_ulong16_rte(long16 x);

ullong convert_ulong_rtp(llong x);
ulong2 convert_ulong2_rtp(long2 x);
ulong3 convert_ulong3_rtp(long3 x);
ulong4 convert_ulong4_rtp(long4 x);
ulong8 convert_ulong8_rtp(long8 x);
ulong16 convert_ulong16_rtp(long16 x);

ullong convert_ulong_rtn(llong x);
ulong2 convert_ulong2_rtn(long2 x);
ulong3 convert_ulong3_rtn(long3 x);
ulong4 convert_ulong4_rtn(long4 x);
ulong8 convert_ulong8_rtn(long8 x);
ulong16 convert_ulong16_rtn(long16 x);

ullong convert_ulong_rtz(llong x);
ulong2 convert_ulong2_rtz(long2 x);
ulong3 convert_ulong3_rtz(long3 x);
ulong4 convert_ulong4_rtz(long4 x);
ulong8 convert_ulong8_rtz(long8 x);
ulong16 convert_ulong16_rtz(long16 x);

ullong convert_ulong_sat(llong x);
ulong2 convert_ulong2_sat(long2 x);
ulong3 convert_ulong3_sat(long3 x);
ulong4 convert_ulong4_sat(long4 x);
ulong8 convert_ulong8_sat(long8 x);
ulong16 convert_ulong16_sat(long16 x);

ullong convert_ulong_sat_rte(llong x);
ulong2 convert_ulong2_sat_rte(long2 x);
ulong3 convert_ulong3_sat_rte(long3 x);
ulong4 convert_ulong4_sat_rte(long4 x);
ulong8 convert_ulong8_sat_rte(long8 x);
ulong16 convert_ulong16_sat_rte(long16 x);

ullong convert_ulong_sat_rtp(llong x);
ulong2 convert_ulong2_sat_rtp(long2 x);
ulong3 convert_ulong3_sat_rtp(long3 x);
ulong4 convert_ulong4_sat_rtp(long4 x);
ulong8 convert_ulong8_sat_rtp(long8 x);
ulong16 convert_ulong16_sat_rtp(long16 x);

ullong convert_ulong_sat_rtn(llong x);
ulong2 convert_ulong2_sat_rtn(long2 x);
ulong3 convert_ulong3_sat_rtn(long3 x);
ulong4 convert_ulong4_sat_rtn(long4 x);
ulong8 convert_ulong8_sat_rtn(long8 x);
ulong16 convert_ulong16_sat_rtn(long16 x);

ullong convert_ulong_sat_rtz(llong x);
ulong2 convert_ulong2_sat_rtz(long2 x);
ulong3 convert_ulong3_sat_rtz(long3 x);
ulong4 convert_ulong4_sat_rtz(long4 x);
ulong8 convert_ulong8_sat_rtz(long8 x);
ulong16 convert_ulong16_sat_rtz(long16 x);

llong convert_long(uchar x);
long2 convert_long2(uchar2 x);
long3 convert_long3(uchar3 x);
long4 convert_long4(uchar4 x);
long8 convert_long8(uchar8 x);
long16 convert_long16(uchar16 x);

llong convert_long_rte(uchar x);
long2 convert_long2_rte(uchar2 x);
long3 convert_long3_rte(uchar3 x);
long4 convert_long4_rte(uchar4 x);
long8 convert_long8_rte(uchar8 x);
long16 convert_long16_rte(uchar16 x);

llong convert_long_rtp(uchar x);
long2 convert_long2_rtp(uchar2 x);
long3 convert_long3_rtp(uchar3 x);
long4 convert_long4_rtp(uchar4 x);
long8 convert_long8_rtp(uchar8 x);
long16 convert_long16_rtp(uchar16 x);

llong convert_long_rtn(uchar x);
long2 convert_long2_rtn(uchar2 x);
long3 convert_long3_rtn(uchar3 x);
long4 convert_long4_rtn(uchar4 x);
long8 convert_long8_rtn(uchar8 x);
long16 convert_long16_rtn(uchar16 x);

llong convert_long_rtz(uchar x);
long2 convert_long2_rtz(uchar2 x);
long3 convert_long3_rtz(uchar3 x);
long4 convert_long4_rtz(uchar4 x);
long8 convert_long8_rtz(uchar8 x);
long16 convert_long16_rtz(uchar16 x);

llong convert_long_sat(uchar x);
long2 convert_long2_sat(uchar2 x);
long3 convert_long3_sat(uchar3 x);
long4 convert_long4_sat(uchar4 x);
long8 convert_long8_sat(uchar8 x);
long16 convert_long16_sat(uchar16 x);

llong convert_long_sat_rte(uchar x);
long2 convert_long2_sat_rte(uchar2 x);
long3 convert_long3_sat_rte(uchar3 x);
long4 convert_long4_sat_rte(uchar4 x);
long8 convert_long8_sat_rte(uchar8 x);
long16 convert_long16_sat_rte(uchar16 x);

llong convert_long_sat_rtp(uchar x);
long2 convert_long2_sat_rtp(uchar2 x);
long3 convert_long3_sat_rtp(uchar3 x);
long4 convert_long4_sat_rtp(uchar4 x);
long8 convert_long8_sat_rtp(uchar8 x);
long16 convert_long16_sat_rtp(uchar16 x);

llong convert_long_sat_rtn(uchar x);
long2 convert_long2_sat_rtn(uchar2 x);
long3 convert_long3_sat_rtn(uchar3 x);
long4 convert_long4_sat_rtn(uchar4 x);
long8 convert_long8_sat_rtn(uchar8 x);
long16 convert_long16_sat_rtn(uchar16 x);

llong convert_long_sat_rtz(uchar x);
long2 convert_long2_sat_rtz(uchar2 x);
long3 convert_long3_sat_rtz(uchar3 x);
long4 convert_long4_sat_rtz(uchar4 x);
long8 convert_long8_sat_rtz(uchar8 x);
long16 convert_long16_sat_rtz(uchar16 x);

llong convert_long(schar x);
long2 convert_long2(char2 x);
long3 convert_long3(char3 x);
long4 convert_long4(char4 x);
long8 convert_long8(char8 x);
long16 convert_long16(char16 x);

llong convert_long_rte(schar x);
long2 convert_long2_rte(char2 x);
long3 convert_long3_rte(char3 x);
long4 convert_long4_rte(char4 x);
long8 convert_long8_rte(char8 x);
long16 convert_long16_rte(char16 x);

llong convert_long_rtp(schar x);
long2 convert_long2_rtp(char2 x);
long3 convert_long3_rtp(char3 x);
long4 convert_long4_rtp(char4 x);
long8 convert_long8_rtp(char8 x);
long16 convert_long16_rtp(char16 x);

llong convert_long_rtn(schar x);
long2 convert_long2_rtn(char2 x);
long3 convert_long3_rtn(char3 x);
long4 convert_long4_rtn(char4 x);
long8 convert_long8_rtn(char8 x);
long16 convert_long16_rtn(char16 x);

llong convert_long_rtz(schar x);
long2 convert_long2_rtz(char2 x);
long3 convert_long3_rtz(char3 x);
long4 convert_long4_rtz(char4 x);
long8 convert_long8_rtz(char8 x);
long16 convert_long16_rtz(char16 x);

llong convert_long_sat(schar x);
long2 convert_long2_sat(char2 x);
long3 convert_long3_sat(char3 x);
long4 convert_long4_sat(char4 x);
long8 convert_long8_sat(char8 x);
long16 convert_long16_sat(char16 x);

llong convert_long_sat_rte(schar x);
long2 convert_long2_sat_rte(char2 x);
long3 convert_long3_sat_rte(char3 x);
long4 convert_long4_sat_rte(char4 x);
long8 convert_long8_sat_rte(char8 x);
long16 convert_long16_sat_rte(char16 x);

llong convert_long_sat_rtp(schar x);
long2 convert_long2_sat_rtp(char2 x);
long3 convert_long3_sat_rtp(char3 x);
long4 convert_long4_sat_rtp(char4 x);
long8 convert_long8_sat_rtp(char8 x);
long16 convert_long16_sat_rtp(char16 x);

llong convert_long_sat_rtn(schar x);
long2 convert_long2_sat_rtn(char2 x);
long3 convert_long3_sat_rtn(char3 x);
long4 convert_long4_sat_rtn(char4 x);
long8 convert_long8_sat_rtn(char8 x);
long16 convert_long16_sat_rtn(char16 x);

llong convert_long_sat_rtz(schar x);
long2 convert_long2_sat_rtz(char2 x);
long3 convert_long3_sat_rtz(char3 x);
long4 convert_long4_sat_rtz(char4 x);
long8 convert_long8_sat_rtz(char8 x);
long16 convert_long16_sat_rtz(char16 x);

llong convert_long(ushort x);
long2 convert_long2(ushort2 x);
long3 convert_long3(ushort3 x);
long4 convert_long4(ushort4 x);
long8 convert_long8(ushort8 x);
long16 convert_long16(ushort16 x);

llong convert_long_rte(ushort x);
long2 convert_long2_rte(ushort2 x);
long3 convert_long3_rte(ushort3 x);
long4 convert_long4_rte(ushort4 x);
long8 convert_long8_rte(ushort8 x);
long16 convert_long16_rte(ushort16 x);

llong convert_long_rtp(ushort x);
long2 convert_long2_rtp(ushort2 x);
long3 convert_long3_rtp(ushort3 x);
long4 convert_long4_rtp(ushort4 x);
long8 convert_long8_rtp(ushort8 x);
long16 convert_long16_rtp(ushort16 x);

llong convert_long_rtn(ushort x);
long2 convert_long2_rtn(ushort2 x);
long3 convert_long3_rtn(ushort3 x);
long4 convert_long4_rtn(ushort4 x);
long8 convert_long8_rtn(ushort8 x);
long16 convert_long16_rtn(ushort16 x);

llong convert_long_rtz(ushort x);
long2 convert_long2_rtz(ushort2 x);
long3 convert_long3_rtz(ushort3 x);
long4 convert_long4_rtz(ushort4 x);
long8 convert_long8_rtz(ushort8 x);
long16 convert_long16_rtz(ushort16 x);

llong convert_long_sat(ushort x);
long2 convert_long2_sat(ushort2 x);
long3 convert_long3_sat(ushort3 x);
long4 convert_long4_sat(ushort4 x);
long8 convert_long8_sat(ushort8 x);
long16 convert_long16_sat(ushort16 x);

llong convert_long_sat_rte(ushort x);
long2 convert_long2_sat_rte(ushort2 x);
long3 convert_long3_sat_rte(ushort3 x);
long4 convert_long4_sat_rte(ushort4 x);
long8 convert_long8_sat_rte(ushort8 x);
long16 convert_long16_sat_rte(ushort16 x);

llong convert_long_sat_rtp(ushort x);
long2 convert_long2_sat_rtp(ushort2 x);
long3 convert_long3_sat_rtp(ushort3 x);
long4 convert_long4_sat_rtp(ushort4 x);
long8 convert_long8_sat_rtp(ushort8 x);
long16 convert_long16_sat_rtp(ushort16 x);

llong convert_long_sat_rtn(ushort x);
long2 convert_long2_sat_rtn(ushort2 x);
long3 convert_long3_sat_rtn(ushort3 x);
long4 convert_long4_sat_rtn(ushort4 x);
long8 convert_long8_sat_rtn(ushort8 x);
long16 convert_long16_sat_rtn(ushort16 x);

llong convert_long_sat_rtz(ushort x);
long2 convert_long2_sat_rtz(ushort2 x);
long3 convert_long3_sat_rtz(ushort3 x);
long4 convert_long4_sat_rtz(ushort4 x);
long8 convert_long8_sat_rtz(ushort8 x);
long16 convert_long16_sat_rtz(ushort16 x);

llong convert_long(short x);
long2 convert_long2(short2 x);
long3 convert_long3(short3 x);
long4 convert_long4(short4 x);
long8 convert_long8(short8 x);
long16 convert_long16(short16 x);

llong convert_long_rte(short x);
long2 convert_long2_rte(short2 x);
long3 convert_long3_rte(short3 x);
long4 convert_long4_rte(short4 x);
long8 convert_long8_rte(short8 x);
long16 convert_long16_rte(short16 x);

llong convert_long_rtp(short x);
long2 convert_long2_rtp(short2 x);
long3 convert_long3_rtp(short3 x);
long4 convert_long4_rtp(short4 x);
long8 convert_long8_rtp(short8 x);
long16 convert_long16_rtp(short16 x);

llong convert_long_rtn(short x);
long2 convert_long2_rtn(short2 x);
long3 convert_long3_rtn(short3 x);
long4 convert_long4_rtn(short4 x);
long8 convert_long8_rtn(short8 x);
long16 convert_long16_rtn(short16 x);

llong convert_long_rtz(short x);
long2 convert_long2_rtz(short2 x);
long3 convert_long3_rtz(short3 x);
long4 convert_long4_rtz(short4 x);
long8 convert_long8_rtz(short8 x);
long16 convert_long16_rtz(short16 x);

llong convert_long_sat(short x);
long2 convert_long2_sat(short2 x);
long3 convert_long3_sat(short3 x);
long4 convert_long4_sat(short4 x);
long8 convert_long8_sat(short8 x);
long16 convert_long16_sat(short16 x);

llong convert_long_sat_rte(short x);
long2 convert_long2_sat_rte(short2 x);
long3 convert_long3_sat_rte(short3 x);
long4 convert_long4_sat_rte(short4 x);
long8 convert_long8_sat_rte(short8 x);
long16 convert_long16_sat_rte(short16 x);

llong convert_long_sat_rtp(short x);
long2 convert_long2_sat_rtp(short2 x);
long3 convert_long3_sat_rtp(short3 x);
long4 convert_long4_sat_rtp(short4 x);
long8 convert_long8_sat_rtp(short8 x);
long16 convert_long16_sat_rtp(short16 x);

llong convert_long_sat_rtn(short x);
long2 convert_long2_sat_rtn(short2 x);
long3 convert_long3_sat_rtn(short3 x);
long4 convert_long4_sat_rtn(short4 x);
long8 convert_long8_sat_rtn(short8 x);
long16 convert_long16_sat_rtn(short16 x);

llong convert_long_sat_rtz(short x);
long2 convert_long2_sat_rtz(short2 x);
long3 convert_long3_sat_rtz(short3 x);
long4 convert_long4_sat_rtz(short4 x);
long8 convert_long8_sat_rtz(short8 x);
long16 convert_long16_sat_rtz(short16 x);

llong convert_long(uint x);
long2 convert_long2(uint2 x);
long3 convert_long3(uint3 x);
long4 convert_long4(uint4 x);
long8 convert_long8(uint8 x);
long16 convert_long16(uint16 x);

llong convert_long_rte(uint x);
long2 convert_long2_rte(uint2 x);
long3 convert_long3_rte(uint3 x);
long4 convert_long4_rte(uint4 x);
long8 convert_long8_rte(uint8 x);
long16 convert_long16_rte(uint16 x);

llong convert_long_rtp(uint x);
long2 convert_long2_rtp(uint2 x);
long3 convert_long3_rtp(uint3 x);
long4 convert_long4_rtp(uint4 x);
long8 convert_long8_rtp(uint8 x);
long16 convert_long16_rtp(uint16 x);

llong convert_long_rtn(uint x);
long2 convert_long2_rtn(uint2 x);
long3 convert_long3_rtn(uint3 x);
long4 convert_long4_rtn(uint4 x);
long8 convert_long8_rtn(uint8 x);
long16 convert_long16_rtn(uint16 x);

llong convert_long_rtz(uint x);
long2 convert_long2_rtz(uint2 x);
long3 convert_long3_rtz(uint3 x);
long4 convert_long4_rtz(uint4 x);
long8 convert_long8_rtz(uint8 x);
long16 convert_long16_rtz(uint16 x);

llong convert_long_sat(uint x);
long2 convert_long2_sat(uint2 x);
long3 convert_long3_sat(uint3 x);
long4 convert_long4_sat(uint4 x);
long8 convert_long8_sat(uint8 x);
long16 convert_long16_sat(uint16 x);

llong convert_long_sat_rte(uint x);
long2 convert_long2_sat_rte(uint2 x);
long3 convert_long3_sat_rte(uint3 x);
long4 convert_long4_sat_rte(uint4 x);
long8 convert_long8_sat_rte(uint8 x);
long16 convert_long16_sat_rte(uint16 x);

llong convert_long_sat_rtp(uint x);
long2 convert_long2_sat_rtp(uint2 x);
long3 convert_long3_sat_rtp(uint3 x);
long4 convert_long4_sat_rtp(uint4 x);
long8 convert_long8_sat_rtp(uint8 x);
long16 convert_long16_sat_rtp(uint16 x);

llong convert_long_sat_rtn(uint x);
long2 convert_long2_sat_rtn(uint2 x);
long3 convert_long3_sat_rtn(uint3 x);
long4 convert_long4_sat_rtn(uint4 x);
long8 convert_long8_sat_rtn(uint8 x);
long16 convert_long16_sat_rtn(uint16 x);

llong convert_long_sat_rtz(uint x);
long2 convert_long2_sat_rtz(uint2 x);
long3 convert_long3_sat_rtz(uint3 x);
long4 convert_long4_sat_rtz(uint4 x);
long8 convert_long8_sat_rtz(uint8 x);
long16 convert_long16_sat_rtz(uint16 x);

llong convert_long(int x);
long2 convert_long2(int2 x);
long3 convert_long3(int3 x);
long4 convert_long4(int4 x);
long8 convert_long8(int8 x);
long16 convert_long16(int16 x);

llong convert_long_rte(int x);
long2 convert_long2_rte(int2 x);
long3 convert_long3_rte(int3 x);
long4 convert_long4_rte(int4 x);
long8 convert_long8_rte(int8 x);
long16 convert_long16_rte(int16 x);

llong convert_long_rtp(int x);
long2 convert_long2_rtp(int2 x);
long3 convert_long3_rtp(int3 x);
long4 convert_long4_rtp(int4 x);
long8 convert_long8_rtp(int8 x);
long16 convert_long16_rtp(int16 x);

llong convert_long_rtn(int x);
long2 convert_long2_rtn(int2 x);
long3 convert_long3_rtn(int3 x);
long4 convert_long4_rtn(int4 x);
long8 convert_long8_rtn(int8 x);
long16 convert_long16_rtn(int16 x);

llong convert_long_rtz(int x);
long2 convert_long2_rtz(int2 x);
long3 convert_long3_rtz(int3 x);
long4 convert_long4_rtz(int4 x);
long8 convert_long8_rtz(int8 x);
long16 convert_long16_rtz(int16 x);

llong convert_long_sat(int x);
long2 convert_long2_sat(int2 x);
long3 convert_long3_sat(int3 x);
long4 convert_long4_sat(int4 x);
long8 convert_long8_sat(int8 x);
long16 convert_long16_sat(int16 x);

llong convert_long_sat_rte(int x);
long2 convert_long2_sat_rte(int2 x);
long3 convert_long3_sat_rte(int3 x);
long4 convert_long4_sat_rte(int4 x);
long8 convert_long8_sat_rte(int8 x);
long16 convert_long16_sat_rte(int16 x);

llong convert_long_sat_rtp(int x);
long2 convert_long2_sat_rtp(int2 x);
long3 convert_long3_sat_rtp(int3 x);
long4 convert_long4_sat_rtp(int4 x);
long8 convert_long8_sat_rtp(int8 x);
long16 convert_long16_sat_rtp(int16 x);

llong convert_long_sat_rtn(int x);
long2 convert_long2_sat_rtn(int2 x);
long3 convert_long3_sat_rtn(int3 x);
long4 convert_long4_sat_rtn(int4 x);
long8 convert_long8_sat_rtn(int8 x);
long16 convert_long16_sat_rtn(int16 x);

llong convert_long_sat_rtz(int x);
long2 convert_long2_sat_rtz(int2 x);
long3 convert_long3_sat_rtz(int3 x);
long4 convert_long4_sat_rtz(int4 x);
long8 convert_long8_sat_rtz(int8 x);
long16 convert_long16_sat_rtz(int16 x);

llong convert_long(float x);
long2 convert_long2(float2 x);
long3 convert_long3(float3 x);
long4 convert_long4(float4 x);
long8 convert_long8(float8 x);
long16 convert_long16(float16 x);

llong convert_long_rte(float x);
long2 convert_long2_rte(float2 x);
long3 convert_long3_rte(float3 x);
long4 convert_long4_rte(float4 x);
long8 convert_long8_rte(float8 x);
long16 convert_long16_rte(float16 x);

llong convert_long_rtp(float x);
long2 convert_long2_rtp(float2 x);
long3 convert_long3_rtp(float3 x);
long4 convert_long4_rtp(float4 x);
long8 convert_long8_rtp(float8 x);
long16 convert_long16_rtp(float16 x);

llong convert_long_rtn(float x);
long2 convert_long2_rtn(float2 x);
long3 convert_long3_rtn(float3 x);
long4 convert_long4_rtn(float4 x);
long8 convert_long8_rtn(float8 x);
long16 convert_long16_rtn(float16 x);

llong convert_long_rtz(float x);
long2 convert_long2_rtz(float2 x);
long3 convert_long3_rtz(float3 x);
long4 convert_long4_rtz(float4 x);
long8 convert_long8_rtz(float8 x);
long16 convert_long16_rtz(float16 x);

llong convert_long_sat(float x);
long2 convert_long2_sat(float2 x);
long3 convert_long3_sat(float3 x);
long4 convert_long4_sat(float4 x);
long8 convert_long8_sat(float8 x);
long16 convert_long16_sat(float16 x);

llong convert_long_sat_rte(float x);
long2 convert_long2_sat_rte(float2 x);
long3 convert_long3_sat_rte(float3 x);
long4 convert_long4_sat_rte(float4 x);
long8 convert_long8_sat_rte(float8 x);
long16 convert_long16_sat_rte(float16 x);

llong convert_long_sat_rtp(float x);
long2 convert_long2_sat_rtp(float2 x);
long3 convert_long3_sat_rtp(float3 x);
long4 convert_long4_sat_rtp(float4 x);
long8 convert_long8_sat_rtp(float8 x);
long16 convert_long16_sat_rtp(float16 x);

llong convert_long_sat_rtn(float x);
long2 convert_long2_sat_rtn(float2 x);
long3 convert_long3_sat_rtn(float3 x);
long4 convert_long4_sat_rtn(float4 x);
long8 convert_long8_sat_rtn(float8 x);
long16 convert_long16_sat_rtn(float16 x);

llong convert_long_sat_rtz(float x);
long2 convert_long2_sat_rtz(float2 x);
long3 convert_long3_sat_rtz(float3 x);
long4 convert_long4_sat_rtz(float4 x);
long8 convert_long8_sat_rtz(float8 x);
long16 convert_long16_sat_rtz(float16 x);

llong convert_long(double x);
long2 convert_long2(double2 x);
long3 convert_long3(double3 x);
long4 convert_long4(double4 x);
long8 convert_long8(double8 x);
long16 convert_long16(double16 x);

llong convert_long_rte(double x);
long2 convert_long2_rte(double2 x);
long3 convert_long3_rte(double3 x);
long4 convert_long4_rte(double4 x);
long8 convert_long8_rte(double8 x);
long16 convert_long16_rte(double16 x);

llong convert_long_rtp(double x);
long2 convert_long2_rtp(double2 x);
long3 convert_long3_rtp(double3 x);
long4 convert_long4_rtp(double4 x);
long8 convert_long8_rtp(double8 x);
long16 convert_long16_rtp(double16 x);

llong convert_long_rtn(double x);
long2 convert_long2_rtn(double2 x);
long3 convert_long3_rtn(double3 x);
long4 convert_long4_rtn(double4 x);
long8 convert_long8_rtn(double8 x);
long16 convert_long16_rtn(double16 x);

llong convert_long_rtz(double x);
long2 convert_long2_rtz(double2 x);
long3 convert_long3_rtz(double3 x);
long4 convert_long4_rtz(double4 x);
long8 convert_long8_rtz(double8 x);
long16 convert_long16_rtz(double16 x);

llong convert_long_sat(double x);
long2 convert_long2_sat(double2 x);
long3 convert_long3_sat(double3 x);
long4 convert_long4_sat(double4 x);
long8 convert_long8_sat(double8 x);
long16 convert_long16_sat(double16 x);

llong convert_long_sat_rte(double x);
long2 convert_long2_sat_rte(double2 x);
long3 convert_long3_sat_rte(double3 x);
long4 convert_long4_sat_rte(double4 x);
long8 convert_long8_sat_rte(double8 x);
long16 convert_long16_sat_rte(double16 x);

llong convert_long_sat_rtp(double x);
long2 convert_long2_sat_rtp(double2 x);
long3 convert_long3_sat_rtp(double3 x);
long4 convert_long4_sat_rtp(double4 x);
long8 convert_long8_sat_rtp(double8 x);
long16 convert_long16_sat_rtp(double16 x);

llong convert_long_sat_rtn(double x);
long2 convert_long2_sat_rtn(double2 x);
long3 convert_long3_sat_rtn(double3 x);
long4 convert_long4_sat_rtn(double4 x);
long8 convert_long8_sat_rtn(double8 x);
long16 convert_long16_sat_rtn(double16 x);

llong convert_long_sat_rtz(double x);
long2 convert_long2_sat_rtz(double2 x);
long3 convert_long3_sat_rtz(double3 x);
long4 convert_long4_sat_rtz(double4 x);
long8 convert_long8_sat_rtz(double8 x);
long16 convert_long16_sat_rtz(double16 x);

llong convert_long(ullong x);
long2 convert_long2(ulong2 x);
long3 convert_long3(ulong3 x);
long4 convert_long4(ulong4 x);
long8 convert_long8(ulong8 x);
long16 convert_long16(ulong16 x);

llong convert_long_rte(ullong x);
long2 convert_long2_rte(ulong2 x);
long3 convert_long3_rte(ulong3 x);
long4 convert_long4_rte(ulong4 x);
long8 convert_long8_rte(ulong8 x);
long16 convert_long16_rte(ulong16 x);

llong convert_long_rtp(ullong x);
long2 convert_long2_rtp(ulong2 x);
long3 convert_long3_rtp(ulong3 x);
long4 convert_long4_rtp(ulong4 x);
long8 convert_long8_rtp(ulong8 x);
long16 convert_long16_rtp(ulong16 x);

llong convert_long_rtn(ullong x);
long2 convert_long2_rtn(ulong2 x);
long3 convert_long3_rtn(ulong3 x);
long4 convert_long4_rtn(ulong4 x);
long8 convert_long8_rtn(ulong8 x);
long16 convert_long16_rtn(ulong16 x);

llong convert_long_rtz(ullong x);
long2 convert_long2_rtz(ulong2 x);
long3 convert_long3_rtz(ulong3 x);
long4 convert_long4_rtz(ulong4 x);
long8 convert_long8_rtz(ulong8 x);
long16 convert_long16_rtz(ulong16 x);

llong convert_long_sat(ullong x);
long2 convert_long2_sat(ulong2 x);
long3 convert_long3_sat(ulong3 x);
long4 convert_long4_sat(ulong4 x);
long8 convert_long8_sat(ulong8 x);
long16 convert_long16_sat(ulong16 x);

llong convert_long_sat_rte(ullong x);
long2 convert_long2_sat_rte(ulong2 x);
long3 convert_long3_sat_rte(ulong3 x);
long4 convert_long4_sat_rte(ulong4 x);
long8 convert_long8_sat_rte(ulong8 x);
long16 convert_long16_sat_rte(ulong16 x);

llong convert_long_sat_rtp(ullong x);
long2 convert_long2_sat_rtp(ulong2 x);
long3 convert_long3_sat_rtp(ulong3 x);
long4 convert_long4_sat_rtp(ulong4 x);
long8 convert_long8_sat_rtp(ulong8 x);
long16 convert_long16_sat_rtp(ulong16 x);

llong convert_long_sat_rtn(ullong x);
long2 convert_long2_sat_rtn(ulong2 x);
long3 convert_long3_sat_rtn(ulong3 x);
long4 convert_long4_sat_rtn(ulong4 x);
long8 convert_long8_sat_rtn(ulong8 x);
long16 convert_long16_sat_rtn(ulong16 x);

llong convert_long_sat_rtz(ullong x);
long2 convert_long2_sat_rtz(ulong2 x);
long3 convert_long3_sat_rtz(ulong3 x);
long4 convert_long4_sat_rtz(ulong4 x);
long8 convert_long8_sat_rtz(ulong8 x);
long16 convert_long16_sat_rtz(ulong16 x);

llong convert_long(llong x);
long2 convert_long2(long2 x);
long3 convert_long3(long3 x);
long4 convert_long4(long4 x);
long8 convert_long8(long8 x);
long16 convert_long16(long16 x);

llong convert_long_rte(llong x);
long2 convert_long2_rte(long2 x);
long3 convert_long3_rte(long3 x);
long4 convert_long4_rte(long4 x);
long8 convert_long8_rte(long8 x);
long16 convert_long16_rte(long16 x);

llong convert_long_rtp(llong x);
long2 convert_long2_rtp(long2 x);
long3 convert_long3_rtp(long3 x);
long4 convert_long4_rtp(long4 x);
long8 convert_long8_rtp(long8 x);
long16 convert_long16_rtp(long16 x);

llong convert_long_rtn(llong x);
long2 convert_long2_rtn(long2 x);
long3 convert_long3_rtn(long3 x);
long4 convert_long4_rtn(long4 x);
long8 convert_long8_rtn(long8 x);
long16 convert_long16_rtn(long16 x);

llong convert_long_rtz(llong x);
long2 convert_long2_rtz(long2 x);
long3 convert_long3_rtz(long3 x);
long4 convert_long4_rtz(long4 x);
long8 convert_long8_rtz(long8 x);
long16 convert_long16_rtz(long16 x);

llong convert_long_sat(llong x);
long2 convert_long2_sat(long2 x);
long3 convert_long3_sat(long3 x);
long4 convert_long4_sat(long4 x);
long8 convert_long8_sat(long8 x);
long16 convert_long16_sat(long16 x);

llong convert_long_sat_rte(llong x);
long2 convert_long2_sat_rte(long2 x);
long3 convert_long3_sat_rte(long3 x);
long4 convert_long4_sat_rte(long4 x);
long8 convert_long8_sat_rte(long8 x);
long16 convert_long16_sat_rte(long16 x);

llong convert_long_sat_rtp(llong x);
long2 convert_long2_sat_rtp(long2 x);
long3 convert_long3_sat_rtp(long3 x);
long4 convert_long4_sat_rtp(long4 x);
long8 convert_long8_sat_rtp(long8 x);
long16 convert_long16_sat_rtp(long16 x);

llong convert_long_sat_rtn(llong x);
long2 convert_long2_sat_rtn(long2 x);
long3 convert_long3_sat_rtn(long3 x);
long4 convert_long4_sat_rtn(long4 x);
long8 convert_long8_sat_rtn(long8 x);
long16 convert_long16_sat_rtn(long16 x);

llong convert_long_sat_rtz(llong x);
long2 convert_long2_sat_rtz(long2 x);
long3 convert_long3_sat_rtz(long3 x);
long4 convert_long4_sat_rtz(long4 x);
long8 convert_long8_sat_rtz(long8 x);
long16 convert_long16_sat_rtz(long16 x);

#endif //__CL_BUILTINS_CONVERSION_H

