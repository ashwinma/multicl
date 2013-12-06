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
/* Al rights reserved.                                                      */
/*                                                                           */
/* This file is part of the SNU-SAMSUNG OpenCL runtime.                      */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is free software: you can redistribute it  */
/* and/or modify it under the terms of the GNU Lesser General Public License */
/* as published by the Free Software Foundation, either version 3 of the     */
/* License, or (at your option) any later version.                           */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is distributed in the hope that it wil be */
/* useful, but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  */
/* Public License for more details.                                          */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with the SNU-SAMSUNG OpenCL runtime. If not, see                    */
/* <http://www.gnu.org/licenses/>.                                           */
/*****************************************************************************/

#ifndef SELECT_H_
#define SELECT_H_

#include <cl_cpu_types.h>

// 1. select_uchar_uchar
uchar    select(uchar, uchar, uchar);
uchar2   select(uchar2, uchar2, uchar2);
uchar3   select(uchar3, uchar3, uchar3);
uchar4   select(uchar4, uchar4, uchar4);
uchar8   select(uchar8, uchar8, uchar8);
uchar16  select(uchar16, uchar16, uchar16);

// 2. select_uchar_char
uchar    select(uchar, uchar, schar);
uchar2   select(uchar2, uchar2, char2);
uchar3   select(uchar3, uchar3, char3);
uchar4   select(uchar4, uchar4, char4);
uchar8   select(uchar8, uchar8, char8);
uchar16  select(uchar16, uchar16, char16);

// 3. select_char_uchar
schar    select(schar, schar, uchar);
char2    select(char2, char2, uchar2);
char3   select(char3, char3, uchar3);
char4    select(char4, char4, uchar4);
char8    select(char8, char8, uchar8);
char16   select(char16, char16, uchar16);

// 4. select_char_char
schar    select(schar, schar, schar);
char2    select(char2, char2, char2);
char3   select(char3, char3, char3);
char4    select(char4, char4, char4);
char8    select(char8, char8, char8);
char16   select(char16, char16, char16);

// 5. select_ushort_ushort
ushort   select(ushort, ushort, ushort);
ushort2  select(ushort2, ushort2, ushort2);
ushort3   select(ushort3, ushort3, ushort3);
ushort4  select(ushort4, ushort4, ushort4);
ushort8  select(ushort8, ushort8, ushort8);
ushort16 select(ushort16, ushort16, ushort16);

// 6. select_ushort_short
ushort   select(ushort, ushort, short);
ushort2  select(ushort2, ushort2, short2);
ushort3   select(ushort3, ushort3, short3);
ushort4  select(ushort4, ushort4, short4);
ushort8  select(ushort8, ushort8, short8);
ushort16 select(ushort16, ushort16, short16);

// 7. select_short_ushort
short   select(short, short, ushort);
short2  select(short2, short2, ushort2);
short3   select(short3, short3, ushort3);
short4  select(short4, short4, ushort4);
short8  select(short8, short8, ushort8);
short16 select(short16, short16, ushort16);

// 8. select_short_short
short   select(short, short, short);
short2  select(short2, short2, short2);
short3   select(short3, short3, short3);
short4  select(short4, short4, short4);
short8  select(short8, short8, short8);
short16 select(short16, short16, short16);

// 9. select_uint_uint
uint   select(uint, uint, uint);
uint2  select(uint2, uint2, uint2);
uint3   select(uint3, uint3, uint3);
uint4  select(uint4, uint4, uint4);
uint8  select(uint8, uint8, uint8);
uint16 select(uint16, uint16, uint16);

// 10. select_uint_int
uint   select(uint, uint, int);
uint2  select(uint2, uint2, int2);
uint3   select(uint3, uint3, int3);
uint4  select(uint4, uint4, int4);
uint8  select(uint8, uint8, int8);
uint16 select(uint16, uint16, int16);

// 11. select_int_uint
int   select(int, int, uint);
int2  select(int2, int2, uint2);
int3   select(int3, int3, uint3);
int4  select(int4, int4, uint4);
int8  select(int8, int8, uint8);
int16 select(int16, int16, uint16);

// 12. select_int_int
int   select(int, int, int);
int2  select(int2, int2, int2);
int3   select(int3, int3, int3);
int4  select(int4, int4, int4);
int8  select(int8, int8, int8);
int16 select(int16, int16, int16);

// 13. select_ulong_ulong
ullong  select(ullong, ullong, ullong);
ulong2  select(ulong2, ulong2, ulong2);
ulong3   select(ulong3, ulong3, ulong3);
ulong4  select(ulong4, ulong4, ulong4);
ulong8  select(ulong8, ulong8, ulong8);
ulong16 select(ulong16, ulong16, ulong16);

// 14. select_ulong_long
ullong  select(ullong, ullong, llong);
ulong2  select(ulong2, ulong2, long2);
ulong3   select(ulong3, ulong3, long3);
ulong4  select(ulong4, ulong4, long4);
ulong8  select(ulong8, ulong8, long8);
ulong16 select(ulong16, ulong16, long16);

// 15. select_long_ulong
llong  select(llong, llong, ullong);
long2  select(long2, long2, ulong2);
long3   select(long3, long3, ulong3);
long4  select(long4, long4, ulong4);
long8  select(long8, long8, ulong8);
long16 select(long16, long16, ulong16);

// 16. select_long_long
llong  select(llong, llong, llong);
long2  select(long2, long2, long2);
long3   select(long3, long3, long3);
long4  select(long4, long4, long4);
long8  select(long8, long8, long8);
long16 select(long16, long16, long16);

// 17. select_float_uint
float   select(float, float, uint);
float2  select(float2, float2, uint2);
float3   select(float3, float3, uint3);
float4  select(float4, float4, uint4);
float8  select(float8, float8, uint8);
float16 select(float16, float16, uint16);

// 18. select_float_int
float   select(float, float, int);
float2  select(float2, float2, int2);
float3   select(float3, float3, int3);
float4  select(float4, float4, int4);
float8  select(float8, float8, int8);
float16 select(float16, float16, int16);

// 19. select_double_ulong
double   select(double, double, ullong);
double2  select(double2, double2, ulong2);
double3   select(double3, double3, ulong3);
double4  select(double4, double4, ulong4);
double8  select(double8, double8, ulong8);
double16 select(double16, double16, ulong16);

// 20. select_double_long
double   select(double, double, llong);
double2  select(double2, double2, long2);
double3   select(double3, double3, long3);
double4  select(double4, double4, long4);
double8  select(double8, double8, long8);
double16 select(double16, double16, long16);


#if 0
// schar select(schar a, schar b, S c)
schar	   select(schar a, schar b, schar c);
char2    select(char2 a, char2 b, char2 c);
char3   select(char3, char3, char3);
char4    select(char4 a, char4 b, char4 c);
char8    select(char8 a, char8 b, char8 c);
char16   select(char16 a, char16 b, char16 c);

schar	   select(schar a, schar b, short c);
char2    select(char2 a, char2 b, short2 c);
char3   select(char3, char3, short3);
char4    select(char4 a, char4 b, short4 c);
char8    select(char8 a, char8 b, short8 c);
char16   select(char16 a, char16 b, short16 c);

schar	   select(schar a, schar b, int c);
char2    select(char2 a, char2 b, int2 c);
char3   select(char3, char3, int3);
char4    select(char4 a, char4 b, int4 c);
char8    select(char8 a, char8 b, int8 c);
char16   select(char16 a, char16 b, int16 c);

schar	   select(schar a, schar b, llong c);
char2    select(char2 a, char2 b, long2 c);
char3   select(char3, char3, long3);
char4    select(char4 a, char4 b, long4 c);
char8    select(char8 a, char8 b, long8 c);
char16   select(char16 a, char16 b, long16 c);

// uchar select(uchar a, uchar b, S c)
uchar    select(uchar a, uchar b, schar c);
uchar2   select(uchar2 a, uchar2 b, char2 c);
uchar3   select(uchar3 a, uchar3 b, char3 c);
uchar4   select(uchar4 a, uchar4 b, char4 c);
uchar8   select(uchar8 a, uchar8 b, char8 c);
uchar16  select(uchar16 a, uchar16 b, char16 c);

uchar    select(uchar a, uchar b, short c);
uchar2   select(uchar2 a, uchar2 b, short2 c);
uchar3   select(uchar3 a, uchar3 b, short3 c);
uchar4   select(uchar4 a, uchar4 b, short4 c);
uchar8   select(uchar8 a, uchar8 b, short8 c);
uchar16  select(uchar16 a, uchar16 b, short16 c);

uchar    select(uchar a, uchar b, int c);
uchar2   select(uchar2 a, uchar2 b, int2 c);
uchar3   select(uchar3 a, uchar3 b, int3 c);
uchar4   select(uchar4 a, uchar4 b, int4 c);
uchar8   select(uchar8 a, uchar8 b, int8 c);
uchar16  select(uchar16 a, uchar16 b, int16 c);

uchar    select(uchar a, uchar b, llong c);
uchar2   select(uchar2 a, uchar2 b, long2 c);
uchar3   select(uchar3 a, uchar3 b, long3 c);
uchar4   select(uchar4 a, uchar4 b, long4 c);
uchar8   select(uchar8 a, uchar8 b, long8 c);
uchar16  select(uchar16 a, uchar16 b, long16 c);

// short select(short a, short b, S c)
short    select(short a, short b, schar c);
short2   select(short2 a, short2 b, char2 c);
short3   select(short3 a, short3 b, char3 c);
short4   select(short4 a, short4 b, char4 c);
short8   select(short8 a, short8 b, char8 c);
short16  select(short16 a, short16 b, char16 c);

short    select(short a, short b, short c);
short2   select(short2 a, short2 b, short2 c);
short3   select(short3 a, short3 b, short3 c);
short4   select(short4 a, short4 b, short4 c);
short8   select(short8 a, short8 b, short8 c);
short16  select(short16 a, short16 b, short16 c);

short    select(short a, short b, int c);
short2   select(short2 a, short2 b, int2 c);
short3   select(short3 a, short3 b, int3 c);
short4   select(short4 a, short4 b, int4 c);
short8   select(short8 a, short8 b, int8 c);
short16  select(short16 a, short16 b, int16 c);

short    select(short a, short b, llong c);
short2   select(short2 a, short2 b, long2 c);
short3   select(short3 a, short3 b, long3 c);
short4   select(short4 a, short4 b, long4 c);
short8   select(short8 a, short8 b, long8 c);
short16  select(short16 a, short16 b, long16 c);

// ushort select(ushort a, ushort b, S c)
ushort   select(ushort a, ushort b, schar c);
ushort2  select(ushort2 a, ushort2 b, char2 c);
ushort3   select(ushort3 a, ushort3 b, char3 c);
ushort4  select(ushort4 a, ushort4 b, char4 c);
ushort8  select(ushort8 a, ushort8 b, char8 c);
ushort16 select(ushort16 a, ushort16 b, char16 c);

ushort   select(ushort a, ushort b, short c);
ushort2  select(ushort2 a, ushort2 b, short2 c);
ushort3   select(ushort3 a, ushort3 b, short3 c);
ushort4  select(ushort4 a, ushort4 b, short4 c);
ushort8  select(ushort8 a, ushort8 b, short8 c);
ushort16 select(ushort16 a, ushort16 b, short16 c);

ushort   select(ushort a, ushort b, int c);
ushort2  select(ushort2 a, ushort2 b, int2 c);
ushort3   select(ushort3 a, ushort3 b, int3 c);
ushort4  select(ushort4 a, ushort4 b, int4 c);
ushort8  select(ushort8 a, ushort8 b, int8 c);
ushort16 select(ushort16 a, ushort16 b, int16 c);

ushort   select(ushort a, ushort b, llong c);
ushort2  select(ushort2 a, ushort2 b, long2 c);
ushort3   select(ushort3 a, ushort3 b, long3 c);
ushort4  select(ushort4 a, ushort4 b, long4 c);
ushort8  select(ushort8 a, ushort8 b, long8 c);
ushort16 select(ushort16 a, ushort16 b, long16 c);

// int select(int a, int b, S c)
int      select(int a, int b, schar c);
int2     select(int2 a, int2 b, char2 c);
int3   select(int3 a, int3 b, char3 c);
int4     select(int4 a, int4 b, char4 c);
int8     select(int8 a, int8 b, char8 c);
int16    select(int16 a, int16 b, char16 c);

int      select(int a, int b, short c);
int2     select(int2 a, int2 b, short2 c);
int3   select(int3 a, int3 b, short3 c);
int4     select(int4 a, int4 b, short4 c);
int8     select(int8 a, int8 b, short8 c);
int16    select(int16 a, int16 b, short16 c);

int      select(int a, int b, int c);
int2     select(int2 a, int2 b, int2 c);
int3   select(int3 a, int3 b, int3 c);
int4     select(int4 a, int4 b, int4 c);
int8     select(int8 a, int8 b, int8 c);
int16    select(int16 a, int16 b, int16 c);

int      select(int a, int b, llong c);
int2     select(int2 a, int2 b, long2 c);
int3   select(int3 a, int3 b, long3 c);
int4     select(int4 a, int4 b, long4 c);
int8     select(int8 a, int8 b, long8 c);
int16    select(int16 a, int16 b, long16 c);

// uint select(uint a, uint b, S c)
uint     select(uint a, uint b, schar c);
uint2    select(uint2 a, uint2 b, char2 c);
uint3   select(uint3 a, uint3 b, char3 c);
uint4    select(uint4 a, uint4 b, char4 c);
uint8    select(uint8 a, uint8 b, char8 c);
uint16   select(uint16 a, uint16 b, char16 c);

uint     select(uint a, uint b, short c);
uint2    select(uint2 a, uint2 b, short2 c);
uint3   select(uint3 a, uint3 b, short3 c);
uint4    select(uint4 a, uint4 b, short4 c);
uint8    select(uint8 a, uint8 b, short8 c);
uint16   select(uint16 a, uint16 b, short16 c);

uint     select(uint a, uint b, int c);
uint2    select(uint2 a, uint2 b, int2 c);
uint3   select(uint3 a, uint3 b, int3 c);
uint4    select(uint4 a, uint4 b, int4 c);
uint8    select(uint8 a, uint8 b, int8 c);
uint16   select(uint16 a, uint16 b, int16 c);

uint     select(uint a, uint b, llong c);
uint2    select(uint2 a, uint2 b, long2 c);
uint3   select(uint3 a, uint3 b, long3 c);
uint4    select(uint4 a, uint4 b, long4 c);
uint8    select(uint8 a, uint8 b, long8 c);
uint16   select(uint16 a, uint16 b, long16 c);

// llong select(llong a, llong b, S c)
llong    select(llong a, llong b, schar c);
long2    select(long2 a, long2 b, char2 c);
long3   select(long3 a, long3 b, char3 c);
long4    select(long4 a, long4 b, char4 c);
long8    select(long8 a, long8 b, char8 c);
long16   select(long16 a, long16 b, char16 c);

llong    select(llong a, llong b, short c);
long2    select(long2 a, long2 b, short2 c);
long3   select(long3 a, long3 b, short3 c);
long4    select(long4 a, long4 b, short4 c);
long8    select(long8 a, long8 b, short8 c);
long16   select(long16 a, long16 b, short16 c);

llong    select(llong a, llong b, int c);
long2    select(long2 a, long2 b, int2 c);
long3   select(long3 a, long3 b, int3 c);
long4    select(long4 a, long4 b, int4 c);
long8    select(long8 a, long8 b, int8 c);
long16   select(long16 a, long16 b, int16 c);

llong    select(llong a, llong b, llong c);
long2    select(long2 a, long2 b, long2 c);
long3   select(long3 a, long3 b, long3 c);
long4    select(long4 a, long4 b, long4 c);
long8    select(long8 a, long8 b, long8 c);
long16   select(long16 a, long16 b, long16 c);

// ullong select(ullong a, ullong b, S c)
ullong   select(ullong a, ullong b, schar c);
ulong2   select(ulong2 a, ulong2 b, char2 c);
ulong3   select(ulong3 a, ulong3 b, char3 c);
ulong4   select(ulong4 a, ulong4 b, char4 c);
ulong8   select(ulong8 a, ulong8 b, char8 c);
ulong16  select(ulong16 a, ulong16 b, char16 c);

ullong   select(ullong a, ullong b, short c);
ulong2   select(ulong2 a, ulong2 b, short2 c);
ulong3   select(ulong3 a, ulong3 b, short3 c);
ulong4   select(ulong4 a, ulong4 b, short4 c);
ulong8   select(ulong8 a, ulong8 b, short8 c);
ulong16  select(ulong16 a, ulong16 b, short16 c);

ullong   select(ullong a, ullong b, int c);
ulong2   select(ulong2 a, ulong2 b, int2 c);
ulong3   select(ulong3 a, ulong3 b, int3 c);
ulong4   select(ulong4 a, ulong4 b, int4 c);
ulong8   select(ulong8 a, ulong8 b, int8 c);
ulong16  select(ulong16 a, ulong16 b, int16 c);

ullong   select(ullong a, ullong b, llong c);
ulong2   select(ulong2 a, ulong2 b, long2 c);
ulong3   select(ulong3 a, ulong3 b, long3 c);
ulong4   select(ulong4 a, ulong4 b, long4 c);
ulong8   select(ulong8 a, ulong8 b, long8 c);
ulong16  select(ulong16 a, ulong16 b, long16 c);

// float select(float a, float b, S c)
float   select(float a, float b, schar c);
float2   select(float2 a, float2 b, char2 c);
float3   select(float3 a, float3 b, char3 c);
float4   select(float4 a, float4 b, char4 c);
float8   select(float8 a, float8 b, char8 c);
float16  select(float16 a, float16 b, char16 c);

float   select(float a, float b, short c);
float2   select(float2 a, float2 b, short2 c);
float3   select(float3 a, float3 b, short3 c);
float4   select(float4 a, float4 b, short4 c);
float8   select(float8 a, float8 b, short8 c);
float16  select(float16 a, float16 b, short16 c);

float   select(float a, float b, int c);
float2   select(float2 a, float2 b, int2 c);
float3   select(float3 a, float3 b, int3 c);
float4   select(float4 a, float4 b, int4 c);
float8   select(float8 a, float8 b, int8 c);
float16  select(float16 a, float16 b, int16 c);

float   select(float a, float b, llong c);
float2   select(float2 a, float2 b, long2 c);
float3   select(float3 a, float3 b, long3 c);
float4   select(float4 a, float4 b, long4 c);
float8   select(float8 a, float8 b, long8 c);
float16  select(float16 a, float16 b, long16 c);

// double select(double a, double b, S c)
double   select(double a, double b, schar c);
double2   select(double2 a, double2 b, char2 c);
double3   select(double3 a, double3 b, char3 c);
double4   select(double4 a, double4 b, char4 c);
double8   select(double8 a, double8 b, char8 c);
double16  select(double16 a, double16 b, char16 c);

double   select(double a, double b, short c);
double2   select(double2 a, double2 b, short2 c);
double3   select(double3 a, double3 b, short3 c);
double4   select(double4 a, double4 b, short4 c);
double8   select(double8 a, double8 b, short8 c);
double16  select(double16 a, double16 b, short16 c);

double   select(double a, double b, int c);
double2   select(double2 a, double2 b, int2 c);
double3   select(double3 a, double3 b, int3 c);
double4   select(double4 a, double4 b, int4 c);
double8   select(double8 a, double8 b, int8 c);
double16  select(double16 a, double16 b, int16 c);

double   select(double a, double b, llong c);
double2   select(double2 a, double2 b, long2 c);
double3   select(double3 a, double3 b, long3 c);
double4   select(double4 a, double4 b, long4 c);
double8   select(double8 a, double8 b, long8 c);
double16  select(double16 a, double16 b, long16 c);

// schar select(schar a, schar b, U c)
schar	   select(schar a, schar b, schar c);
char2    select(char2 a, char2 b, char2 c);
char3   select(char3 a, char3 b, char3 c);
char4    select(char4 a, char4 b, char4 c);
char8    select(char8 a, char8 b, char8 c);
char16   select(char16 a, char16 b, char16 c);

schar	   select(schar a, schar b, ushort c);
char2    select(char2 a, char2 b, ushort2 c);
char3   select(char3 a, char3 b, ushort3 c);
char4    select(char4 a, char4 b, ushort4 c);
char8    select(char8 a, char8 b, ushort8 c);
char16   select(char16 a, char16 b, ushort16 c);

schar	   select(schar a, schar b, uint c);
char2    select(char2 a, char2 b, uint2 c);
char3   select(char3 a, char3 b, uint3 c);
char4    select(char4 a, char4 b, uint4 c);
char8    select(char8 a, char8 b, uint8 c);
char16   select(char16 a, char16 b, uint16 c);

schar	   select(schar a, schar b, ullong c);
char2    select(char2 a, char2 b, ulong2 c);
char3   select(char3 a, char3 b, ulong3 c);
char4    select(char4 a, char4 b, ulong4 c);
char8    select(char8 a, char8 b, ulong8 c);
char16   select(char16 a, char16 b, ulong16 c);

// uchar select(uchar a, uchar b, U c)
uchar	   select(uchar a, uchar b, uchar c);
uchar2   select(uchar2 a, uchar2 b, uchar2 c);
uchar3   select(uchar3 a, uchar3 b, uchar3 c);
uchar4   select(uchar4 a, uchar4 b, uchar4 c);
uchar8   select(uchar8 a, uchar8 b, uchar8 c);
uchar16  select(uchar16 a, uchar16 b, uchar16 c);

uchar	   select(uchar a, uchar b, ushort c);
uchar2   select(uchar2 a, uchar2 b, ushort2 c);
uchar3   select(uchar3 a, uchar3 b, ushort3 c);
uchar4   select(uchar4 a, uchar4 b, ushort4 c);
uchar8   select(uchar8 a, uchar8 b, ushort8 c);
uchar16  select(uchar16 a, uchar16 b, ushort16 c);

uchar	   select(uchar a, uchar b, uint c);
uchar2   select(uchar2 a, uchar2 b, uint2 c);
uchar3   select(uchar3 a, uchar3 b, uint3 c);
uchar4   select(uchar4 a, uchar4 b, uint4 c);
uchar8   select(uchar8 a, uchar8 b, uint8 c);
uchar16  select(uchar16 a, uchar16 b, uint16 c);

uchar	   select(uchar a, uchar b, ullong c);
uchar2   select(uchar2 a, uchar2 b, ulong2 c);
uchar3   select(uchar3 a, uchar3 b, ulong3 c);
uchar4   select(uchar4 a, uchar4 b, ulong4 c);
uchar8   select(uchar8 a, uchar8 b, ulong8 c);
uchar16  select(uchar16 a, uchar16 b, ulong16 c);


// short select(short a, short b, U c)
short	   select(short a, short b, short c);
short2   select(short2 a, short2 b, short2 c);
short3   select(short3 a, short3 b, short3 c);
short4   select(short4 a, short4 b, short4 c);
short8   select(short8 a, short8 b, short8 c);
short16  select(short16 a, short16 b, short16 c);

short	   select(short a, short b, ushort c);
short2   select(short2 a, short2 b, ushort2 c);
short3   select(short3 a, short3 b, ushort3 c);
short4   select(short4 a, short4 b, ushort4 c);
short8   select(short8 a, short8 b, ushort8 c);
short16  select(short16 a, short16 b, ushort16 c);

short	   select(short a, short b, uint c);
short2   select(short2 a, short2 b, uint2 c);
short3   select(short3 a, short3 b, uint3 c);
short4   select(short4 a, short4 b, uint4 c);
short8   select(short8 a, short8 b, uint8 c);
short16  select(short16 a, short16 b, uint16 c);

short	   select(short a, short b, ullong c);
short2   select(short2 a, short2 b, ulong2 c);
short3   select(short3 a, short3 b, ulong3 c);
short4   select(short4 a, short4 b, ulong4 c);
short8   select(short8 a, short8 b, ulong8 c);
short16  select(short16 a, short16 b, ulong16 c);


// ushort select(ushort a, ushort b, U c)
ushort   select(ushort a, ushort b, uchar c);
ushort2  select(ushort2, ushort2, uchar2);
ushort3  select(ushort3, ushort3, uchar3);
ushort4  select(ushort4, ushort4, uchar4);
ushort8  select(ushort8, ushort8, uchar8);
ushort16 select(ushort16, ushort16, uchar16);

ushort   select(ushort a, ushort b, ushort c);
ushort2  select(ushort2, ushort2, ushort2);
ushort3  select(ushort3, ushort3, ushort3);
ushort4  select(ushort4, ushort4, ushort4);
ushort8  select(ushort8, ushort8, ushort8);
ushort16 select(ushort16, ushort16, ushort16);

ushort   select(ushort a, ushort b, uint c);
ushort2  select(ushort2, ushort2, uint2);
ushort3  select(ushort3, ushort3, uint3);
ushort4  select(ushort4, ushort4, uint4);
ushort8  select(ushort8, ushort8, uint8);
ushort16 select(ushort16, ushort16, uint16);

ushort   select(ushort a, ushort b, ullong c);
ushort2  select(ushort2, ushort2, ulong2);
ushort3  select(ushort3, ushort3, ulong3);
ushort4  select(ushort4, ushort4, ulong4);
ushort8  select(ushort8, ushort8, ulong8);
ushort16 select(ushort16, ushort16, ulong16);


// int select(int a, int b, U c)
int		   select(int a, int b, uchar c);
int2     select(int2 a, int2 b, uchar2 c);
int3     select(int3 a, int3 b, uchar3 c);
int4     select(int4 a, int4 b, uchar4 c);
int8     select(int8 a, int8 b, uchar8 c);
int16    select(int16 a, int16 b, uchar16 c);

int		   select(int a, int b, ushort c);
int2     select(int2 a, int2 b, ushort2 c);
int3     select(int3 a, int3 b, ushort3 c);
int4     select(int4 a, int4 b, ushort4 c);
int8     select(int8 a, int8 b, ushort8 c);
int16    select(int16 a, int16 b, ushort16 c);

int		   select(int a, int b, uint c);
int2     select(int2 a, int2 b, uint2 c);
int3     select(int3 a, int3 b, uint3 c);
int4     select(int4 a, int4 b, uint4 c);
int8     select(int8 a, int8 b, uint8 c);
int16    select(int16 a, int16 b, uint16 c);

int		   select(int a, int b, ullong c);
int2     select(int2 a, int2 b, ulong2 c);
int3     select(int3 a, int3 b, ulong3 c);
int4     select(int4 a, int4 b, ulong4 c);
int8     select(int8 a, int8 b, ulong8 c);
int16    select(int16 a, int16 b, ulong16 c);


// uint select(uint a, uint b, U c)
uint	   select(uint a, uint b, uchar c);
uint2    select(uint2 a, uint2 b, uchar2 c);
uint3    select(uint3 a, uint3 b, uchar3 c);
uint4    select(uint4 a, uint4 b, uchar4 c);
uint8    select(uint8 a, uint8 b, uchar8 c);
uint16   select(uint16 a, uint16 b, uchar16 c);

uint	   select(uint a, uint b, ushort c);
uint2    select(uint2 a, uint2 b, ushort2 c);
uint3    select(uint3 a, uint3 b, ushort3 c);
uint4    select(uint4 a, uint4 b, ushort4 c);
uint8    select(uint8 a, uint8 b, ushort8 c);
uint16   select(uint16 a, uint16 b, ushort16 c);

uint	   select(uint a, uint b, uint c);
uint2    select(uint2 a, uint2 b, uint2 c);
uint3    select(uint3 a, uint3 b, uint3 c);
uint4    select(uint4 a, uint4 b, uint4 c);
uint8    select(uint8 a, uint8 b, uint8 c);
uint16   select(uint16 a, uint16 b, uint16 c);

uint	   select(uint a, uint b, ullong c);
uint2    select(uint2 a, uint2 b, ulong2 c);
uint3    select(uint3 a, uint3 b, ulong3 c);
uint4    select(uint4 a, uint4 b, ulong4 c);
uint8    select(uint8 a, uint8 b, ulong8 c);
uint16   select(uint16 a, uint16 b, ulong16 c);

// llong select(llong a, llong b, U c)
llong	   select(llong a, llong b, uchar c);
long2    select(long2 a, long2 b, uchar2 c);
long3    select(long3 a, long3 b, uchar3 c);
long4    select(long4 a, long4 b, uchar4 c);
long8    select(long8 a, long8 b, uchar8 c);
long16   select(long16 a, long16 b, uchar16 c);

llong	   select(llong a, llong b, ushort c);
long2    select(long2 a, long2 b, ushort2 c);
long3    select(long3 a, long3 b, ushort3 c);
long4    select(long4 a, long4 b, ushort4 c);
long8    select(long8 a, long8 b, ushort8 c);
long16   select(long16 a, long16 b, ushort16 c);

llong	   select(llong a, llong b, uint c);
long2    select(long2 a, long2 b, uint2 c);
long3    select(long3 a, long3 b, uint3 c);
long4    select(long4 a, long4 b, uint4 c);
long8    select(long8 a, long8 b, uint8 c);
long16   select(long16 a, long16 b, uint16 c);

llong	   select(llong a, llong b, ullong c);
long2    select(long2 a, long2 b, ulong2 c);
long3    select(long3 a, long3 b, ulong3 c);
long4    select(long4 a, long4 b, ulong4 c);
long8    select(long8 a, long8 b, ulong8 c);
long16   select(long16 a, long16 b, ulong16 c);


// ullong select(ullong a, ullong b, U c)
ullong	 select(ullong a, ullong b, uchar c);
ulong2   select(ulong2 a, ulong2 b, uchar2 c);
ulong3    select(ulong3 a, ulong3 b, uchar3 c);
ulong4   select(ulong4 a, ulong4 b, uchar4 c);
ulong8   select(ulong8 a, ulong8 b, uchar8 c);
ulong16  select(ulong16 a, ulong16 b, uchar16 c);

ullong	 select(ullong a, ullong b, ushort c);
ulong2   select(ulong2 a, ulong2 b, ushort2 c);
ulong3    select(ulong3 a, ulong3 b, ushort3 c);
ulong4   select(ulong4 a, ulong4 b, ushort4 c);
ulong8   select(ulong8 a, ulong8 b, ushort8 c);
ulong16  select(ulong16 a, ulong16 b, ushort16 c);

ullong	 select(ullong a, ullong b, ulnt c);
ulong2   select(ulong2 a, ulong2 b, uint2 c);
ulong3    select(ulong3 a, ulong3 b, uint3 c);
ulong4   select(ulong4 a, ulong4 b, uint4 c);
ulong8   select(ulong8 a, ulong8 b, uint8 c);
ulong16  select(ulong16 a, ulong16 b, uint16 c);

ullong	 select(ullong a, ullong b, ullong c);
ulong2   select(ulong2 a, ulong2 b, ulong2 c);
ulong3    select(ulong3 a, ulong3 b, ulong3 c);
ulong4   select(ulong4 a, ulong4 b, ulong4 c);
ulong8   select(ulong8 a, ulong8 b, ulong8 c);
ulong16  select(ulong16 a, ulong16 b, ulong16 c);

// float select(float a, float b, U c)
float	 select(float a, float b, uchar c);
float2   select(float2 a, float2 b, uchar2 c);
float3    select(ulong3 a, ulong3 b, uchar3 c);
float4   select(float4 a, float4 b, uchar4 c);
float8   select(float8 a, float8 b, uchar8 c);
float16  select(float16 a, float16 b, uchar16 c);

float	 select(float a, float b, ushort c);
float2   select(float2 a, float2 b, ushort2 c);
float3    select(float3 a, float3 b, uchar3 c);
float4   select(float4 a, float4 b, ushort4 c);
float8   select(float8 a, float8 b, ushort8 c);
float16  select(float16 a, float16 b, ushort16 c);

float	 select(float a, float b, uint c);
float2   select(float2 a, float2 b, uint2 c);
float3    select(float3 a, float3 b, uint3 c);
float4   select(float4 a, float4 b, uint4 c);
float8   select(float8 a, float8 b, uint8 c);
float16  select(float16 a, float16 b, uint16 c);

float	 select(float a, float b, ullong c);
float2   select(float2 a, float2 b, ulong2 c);
float3    select(float3 a, float3 b, ulong3 c);
float4   select(float4 a, float4 b, ulong4 c);
float8   select(float8 a, float8 b, ulong8 c);
float16  select(float16 a, float16 b, ulong16 c);

// double select(double a, double b, U c)
double	 select(double a, double b, uchar c);
double2   select(double2 a, double2 b, uchar2 c);
double3    select(double3 a, double3 b, uchar3 c);
double4   select(double4 a, double4 b, uchar4 c);
double8   select(double8 a, double8 b, uchar8 c);
double16  select(double16 a, double16 b, uchar16 c);

double	 select(double a, double b, ushort c);
double2   select(double2 a, double2 b, ushort2 c);
double3    select(double3 a, double3 b, ushort3 c);
double4   select(double4 a, double4 b, ushort4 c);
double8   select(double8 a, double8 b, ushort8 c);
double16  select(double16 a, double16 b, ushort16 c);

double	 select(double a, double b, uint c);
double2   select(double2 a, double2 b, uint2 c);
double3    select(double3 a, double3 b, uint3 c);
double4   select(double4 a, double4 b, uint4 c);
double8   select(double8 a, double8 b, uint8 c);
double16  select(double16 a, double16 b, uint16 c);

double	 select(double a, double b, ullong c);
double2   select(double2 a, double2 b, ulong2 c);
double3    select(double3 a, double3 b, ulong3 c);
double4   select(double4 a, double4 b, ulong4 c);
double8   select(double8 a, double8 b, ulong8 c);
double16  select(double16 a, double16 b, ulong16 c);

#endif

#endif
