/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2011-2013 Seoul National University.                        */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE          */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   Center for Manycore Programming                                         */
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 151-744, Korea                         */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jungwon Kim, Sangmin Seo, Gangwon Jo, Jun Lee, Jeongho Nah,             */
/*   Jungho Park, Junghyun Kim, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/* This file is based on the SNU-SAMSUNG OpenCL Compiler and is distributed  */
/* under GNU General Public License.                                         */
/* See LICENSE.SNU-SAMSUNG_OpenCL_C_Compiler.TXT for details.                */
/*****************************************************************************/

#ifndef __CL_BUILTINS_RELATIONAL_H__
#define __CL_BUILTINS_RELATIONAL_H__

#include "cl_types.h"

/* 6.12.6 Relational Functions */
/* Table 6.14 Scalar and Vector Relational Functions */

int      isequal(float, float) __attribute__((overloadable));
int2     isequal(float2, float2) __attribute__((overloadable));
int3     isequal(float3, float3) __attribute__((overloadable));
int4     isequal(float4, float4) __attribute__((overloadable));
int8     isequal(float8, float8) __attribute__((overloadable));
int16    isequal(float16, float16) __attribute__((overloadable));
int      isequal(double, double) __attribute__((overloadable));
long2    isequal(double2, double2) __attribute__((overloadable));
long3    isequal(double3, double3) __attribute__((overloadable));
long4    isequal(double4, double4) __attribute__((overloadable));
long8    isequal(double8, double8) __attribute__((overloadable));
long16   isequal(double16, double16) __attribute__((overloadable));

int      isnotequal(float, float) __attribute__((overloadable));
int2     isnotequal(float2, float2) __attribute__((overloadable));
int3     isnotequal(float3, float3) __attribute__((overloadable));
int4     isnotequal(float4, float4) __attribute__((overloadable));
int8     isnotequal(float8, float8) __attribute__((overloadable));
int16    isnotequal(float16, float16) __attribute__((overloadable));
int      isnotequal(double, double) __attribute__((overloadable));
long2    isnotequal(double2, double2) __attribute__((overloadable));
long3    isnotequal(double3, double3) __attribute__((overloadable));
long4    isnotequal(double4, double4) __attribute__((overloadable));
long8    isnotequal(double8, double8) __attribute__((overloadable));
long16   isnotequal(double16, double16) __attribute__((overloadable));

int      isgreater(float, float) __attribute__((overloadable));
int2     isgreater(float2, float2) __attribute__((overloadable));
int3     isgreater(float3, float3) __attribute__((overloadable));
int4     isgreater(float4, float4) __attribute__((overloadable));
int8     isgreater(float8, float8) __attribute__((overloadable));
int16    isgreater(float16, float16) __attribute__((overloadable));
int      isgreater(double, double) __attribute__((overloadable));
long2    isgreater(double2, double2) __attribute__((overloadable));
long3    isgreater(double3, double3) __attribute__((overloadable));
long4    isgreater(double4, double4) __attribute__((overloadable));
long8    isgreater(double8, double8) __attribute__((overloadable));
long16   isgreater(double16, double16) __attribute__((overloadable));

int      isgreaterequal(float, float) __attribute__((overloadable));
int2     isgreaterequal(float2, float2) __attribute__((overloadable));
int3     isgreaterequal(float3, float3) __attribute__((overloadable));
int4     isgreaterequal(float4, float4) __attribute__((overloadable));
int8     isgreaterequal(float8, float8) __attribute__((overloadable));
int16    isgreaterequal(float16, float16) __attribute__((overloadable));
int      isgreaterequal(double, double) __attribute__((overloadable));
long2    isgreaterequal(double2, double2) __attribute__((overloadable));
long3    isgreaterequal(double3, double3) __attribute__((overloadable));
long4    isgreaterequal(double4, double4) __attribute__((overloadable));
long8    isgreaterequal(double8, double8) __attribute__((overloadable));
long16   isgreaterequal(double16, double16) __attribute__((overloadable));

int      isless(float, float) __attribute__((overloadable));
int2     isless(float2, float2) __attribute__((overloadable));
int3     isless(float3, float3) __attribute__((overloadable));
int4     isless(float4, float4) __attribute__((overloadable));
int8     isless(float8, float8) __attribute__((overloadable));
int16    isless(float16, float16) __attribute__((overloadable));
int      isless(double, double) __attribute__((overloadable));
long2    isless(double2, double2) __attribute__((overloadable));
long3    isless(double3, double3) __attribute__((overloadable));
long4    isless(double4, double4) __attribute__((overloadable));
long8    isless(double8, double8) __attribute__((overloadable));
long16   isless(double16, double16) __attribute__((overloadable));

int      islessequal(float, float) __attribute__((overloadable));
int2     islessequal(float2, float2) __attribute__((overloadable));
int3     islessequal(float3, float3) __attribute__((overloadable));
int4     islessequal(float4, float4) __attribute__((overloadable));
int8     islessequal(float8, float8) __attribute__((overloadable));
int16    islessequal(float16, float16) __attribute__((overloadable));
int      islessequal(double, double) __attribute__((overloadable));
long2    islessequal(double2, double2) __attribute__((overloadable));
long3    islessequal(double3, double3) __attribute__((overloadable));
long4    islessequal(double4, double4) __attribute__((overloadable));
long8    islessequal(double8, double8) __attribute__((overloadable));
long16   islessequal(double16, double16) __attribute__((overloadable));

int      islessgreater(float, float) __attribute__((overloadable));
int2     islessgreater(float2, float2) __attribute__((overloadable));
int3     islessgreater(float3, float3) __attribute__((overloadable));
int4     islessgreater(float4, float4) __attribute__((overloadable));
int8     islessgreater(float8, float8) __attribute__((overloadable));
int16    islessgreater(float16, float16) __attribute__((overloadable));
int      islessgreater(double, double) __attribute__((overloadable));
long2    islessgreater(double2, double2) __attribute__((overloadable));
long3    islessgreater(double3, double3) __attribute__((overloadable));
long4    islessgreater(double4, double4) __attribute__((overloadable));
long8    islessgreater(double8, double8) __attribute__((overloadable));
long16   islessgreater(double16, double16) __attribute__((overloadable));

int      isfinite(float) __attribute__((overloadable));
int2     isfinite(float2) __attribute__((overloadable));
int3     isfinite(float3) __attribute__((overloadable));
int4     isfinite(float4) __attribute__((overloadable));
int8     isfinite(float8) __attribute__((overloadable));
int16    isfinite(float16) __attribute__((overloadable));
int      isfinite(double) __attribute__((overloadable));
long2    isfinite(double2) __attribute__((overloadable));
long3    isfinite(double3) __attribute__((overloadable));
long4    isfinite(double4) __attribute__((overloadable));
long8    isfinite(double8) __attribute__((overloadable));
long16   isfinite(double16) __attribute__((overloadable));

int      isinf(float) __attribute__((overloadable));
int2     isinf(float2) __attribute__((overloadable));
int3     isinf(float3) __attribute__((overloadable));
int4     isinf(float4) __attribute__((overloadable));
int8     isinf(float8) __attribute__((overloadable));
int16    isinf(float16) __attribute__((overloadable));
int      isinf(double) __attribute__((overloadable));
long2    isinf(double2) __attribute__((overloadable));
long3    isinf(double3) __attribute__((overloadable));
long4    isinf(double4) __attribute__((overloadable));
long8    isinf(double8) __attribute__((overloadable));
long16   isinf(double16) __attribute__((overloadable));

int      isnan(float) __attribute__((overloadable));
int2     isnan(float2) __attribute__((overloadable));
int3     isnan(float3) __attribute__((overloadable));
int4     isnan(float4) __attribute__((overloadable));
int8     isnan(float8) __attribute__((overloadable));
int16    isnan(float16) __attribute__((overloadable));
int      isnan(double) __attribute__((overloadable));
long2    isnan(double2) __attribute__((overloadable));
long3    isnan(double3) __attribute__((overloadable));
long4    isnan(double4) __attribute__((overloadable));
long8    isnan(double8) __attribute__((overloadable));
long16   isnan(double16) __attribute__((overloadable));

int      isnormal(float) __attribute__((overloadable));
int2     isnormal(float2) __attribute__((overloadable));
int3     isnormal(float3) __attribute__((overloadable));
int4     isnormal(float4) __attribute__((overloadable));
int8     isnormal(float8) __attribute__((overloadable));
int16    isnormal(float16) __attribute__((overloadable));
int      isnormal(double) __attribute__((overloadable));
long2    isnormal(double2) __attribute__((overloadable));
long3    isnormal(double3) __attribute__((overloadable));
long4    isnormal(double4) __attribute__((overloadable));
long8    isnormal(double8) __attribute__((overloadable));
long16   isnormal(double16) __attribute__((overloadable));

int      isordered(float, float) __attribute__((overloadable));
int2     isordered(float2, float2) __attribute__((overloadable));
int3     isordered(float3, float3) __attribute__((overloadable));
int4     isordered(float4, float4) __attribute__((overloadable));
int8     isordered(float8, float8) __attribute__((overloadable));
int16    isordered(float16, float16) __attribute__((overloadable));
int      isordered(double, double) __attribute__((overloadable));
long2    isordered(double2, double2) __attribute__((overloadable));
long3    isordered(double3, double3) __attribute__((overloadable));
long4    isordered(double4, double4) __attribute__((overloadable));
long8    isordered(double8, double8) __attribute__((overloadable));
long16   isordered(double16, double16) __attribute__((overloadable));

int      isunordered(float, float) __attribute__((overloadable));
int2     isunordered(float2, float2) __attribute__((overloadable));
int3     isunordered(float3, float3) __attribute__((overloadable));
int4     isunordered(float4, float4) __attribute__((overloadable));
int8     isunordered(float8, float8) __attribute__((overloadable));
int16    isunordered(float16, float16) __attribute__((overloadable));
int      isunordered(double, double) __attribute__((overloadable));
long2    isunordered(double2, double2) __attribute__((overloadable));
long3    isunordered(double3, double3) __attribute__((overloadable));
long4    isunordered(double4, double4) __attribute__((overloadable));
long8    isunordered(double8, double8) __attribute__((overloadable));
long16   isunordered(double16, double16) __attribute__((overloadable));

int      signbit(float) __attribute__((overloadable));
int2     signbit(float2) __attribute__((overloadable));
int3     signbit(float3) __attribute__((overloadable));
int4     signbit(float4) __attribute__((overloadable));
int8     signbit(float8) __attribute__((overloadable));
int16    signbit(float16) __attribute__((overloadable));
int      signbit(double) __attribute__((overloadable));
long2    signbit(double2) __attribute__((overloadable));
long3    signbit(double3) __attribute__((overloadable));
long4    signbit(double4) __attribute__((overloadable));
long8    signbit(double8) __attribute__((overloadable));
long16   signbit(double16) __attribute__((overloadable));

int      any(char) __attribute__((overloadable));
int      any(char2) __attribute__((overloadable));
int      any(char3) __attribute__((overloadable));
int      any(char4) __attribute__((overloadable));
int      any(char8) __attribute__((overloadable));
int      any(char16) __attribute__((overloadable));
int      any(short) __attribute__((overloadable));
int      any(short2) __attribute__((overloadable));
int      any(short3) __attribute__((overloadable));
int      any(short4) __attribute__((overloadable));
int      any(short8) __attribute__((overloadable));
int      any(short16) __attribute__((overloadable));
int      any(int) __attribute__((overloadable));
int      any(int2) __attribute__((overloadable));
int      any(int3) __attribute__((overloadable));
int      any(int4) __attribute__((overloadable));
int      any(int8) __attribute__((overloadable));
int      any(int16) __attribute__((overloadable));
int      any(long) __attribute__((overloadable));
int      any(long2) __attribute__((overloadable));
int      any(long3) __attribute__((overloadable));
int      any(long4) __attribute__((overloadable));
int      any(long8) __attribute__((overloadable));
int      any(long16) __attribute__((overloadable));

int      all(char) __attribute__((overloadable));
int      all(char2) __attribute__((overloadable));
int      all(char3) __attribute__((overloadable));
int      all(char4) __attribute__((overloadable));
int      all(char8) __attribute__((overloadable));
int      all(char16) __attribute__((overloadable));
int      all(short) __attribute__((overloadable));
int      all(short2) __attribute__((overloadable));
int      all(short3) __attribute__((overloadable));
int      all(short4) __attribute__((overloadable));
int      all(short8) __attribute__((overloadable));
int      all(short16) __attribute__((overloadable));
int      all(int) __attribute__((overloadable));
int      all(int2) __attribute__((overloadable));
int      all(int3) __attribute__((overloadable));
int      all(int4) __attribute__((overloadable));
int      all(int8) __attribute__((overloadable));
int      all(int16) __attribute__((overloadable));
int      all(long) __attribute__((overloadable));
int      all(long2) __attribute__((overloadable));
int      all(long3) __attribute__((overloadable));
int      all(long4) __attribute__((overloadable));
int      all(long8) __attribute__((overloadable));
int      all(long16) __attribute__((overloadable));

char     bitselect(char, char, char) __attribute__((overloadable));
char2    bitselect(char2, char2, char2) __attribute__((overloadable));
char3    bitselect(char3, char3, char3) __attribute__((overloadable));
char4    bitselect(char4, char4, char4) __attribute__((overloadable));
char8    bitselect(char8, char8, char8) __attribute__((overloadable));
char16   bitselect(char16, char16, char16) __attribute__((overloadable));
uchar    bitselect(uchar, uchar, uchar) __attribute__((overloadable));
uchar2   bitselect(uchar2, uchar2, uchar2) __attribute__((overloadable));
uchar3   bitselect(uchar3, uchar3, uchar3) __attribute__((overloadable));
uchar4   bitselect(uchar4, uchar4, uchar4) __attribute__((overloadable));
uchar8   bitselect(uchar8, uchar8, uchar8) __attribute__((overloadable));
uchar16  bitselect(uchar16, uchar16, uchar16) __attribute__((overloadable));
short    bitselect(short, short, short) __attribute__((overloadable));
short2   bitselect(short2, short2, short2) __attribute__((overloadable));
short3   bitselect(short3, short3, short3) __attribute__((overloadable));
short4   bitselect(short4, short4, short4) __attribute__((overloadable));
short8   bitselect(short8, short8, short8) __attribute__((overloadable));
short16  bitselect(short16, short16, short16) __attribute__((overloadable));
ushort   bitselect(ushort, ushort, ushort) __attribute__((overloadable));
ushort2  bitselect(ushort2, ushort2, ushort2) __attribute__((overloadable));
ushort3  bitselect(ushort3, ushort3, ushort3) __attribute__((overloadable));
ushort4  bitselect(ushort4, ushort4, ushort4) __attribute__((overloadable));
ushort8  bitselect(ushort8, ushort8, ushort8) __attribute__((overloadable));
ushort16 bitselect(ushort16, ushort16, ushort16) __attribute__((overloadable));
int      bitselect(int, int, int) __attribute__((overloadable));
int2     bitselect(int2, int2, int2) __attribute__((overloadable));
int3     bitselect(int3, int3, int3) __attribute__((overloadable));
int4     bitselect(int4, int4, int4) __attribute__((overloadable));
int8     bitselect(int8, int8, int8) __attribute__((overloadable));
int16    bitselect(int16, int16, int16) __attribute__((overloadable));
uint     bitselect(uint, uint, uint) __attribute__((overloadable));
uint2    bitselect(uint2, uint2, uint2) __attribute__((overloadable));
uint3    bitselect(uint3, uint3, uint3) __attribute__((overloadable));
uint4    bitselect(uint4, uint4, uint4) __attribute__((overloadable));
uint8    bitselect(uint8, uint8, uint8) __attribute__((overloadable));
uint16   bitselect(uint16, uint16, uint16) __attribute__((overloadable));
long     bitselect(long, long, long) __attribute__((overloadable));
long2    bitselect(long2, long2, long2) __attribute__((overloadable));
long3    bitselect(long3, long3, long3) __attribute__((overloadable));
long4    bitselect(long4, long4, long4) __attribute__((overloadable));
long8    bitselect(long8, long8, long8) __attribute__((overloadable));
long16   bitselect(long16, long16, long16) __attribute__((overloadable));
ulong    bitselect(ulong, ulong, ulong) __attribute__((overloadable));
ulong2   bitselect(ulong2, ulong2, ulong2) __attribute__((overloadable));
ulong3   bitselect(ulong3, ulong3, ulong3) __attribute__((overloadable));
ulong4   bitselect(ulong4, ulong4, ulong4) __attribute__((overloadable));
ulong8   bitselect(ulong8, ulong8, ulong8) __attribute__((overloadable));
ulong16  bitselect(ulong16, ulong16, ulong16) __attribute__((overloadable));
float    bitselect(float, float, float) __attribute__((overloadable));
float2   bitselect(float2, float2, float2) __attribute__((overloadable));
float3   bitselect(float3, float3, float3) __attribute__((overloadable));
float4   bitselect(float4, float4, float4) __attribute__((overloadable));
float8   bitselect(float8, float8, float8) __attribute__((overloadable));
float16  bitselect(float16, float16, float16) __attribute__((overloadable));
double   bitselect(double, double, double) __attribute__((overloadable));
double2  bitselect(double2, double2, double2) __attribute__((overloadable));
double3  bitselect(double3, double3, double3) __attribute__((overloadable));
double4  bitselect(double4, double4, double4) __attribute__((overloadable));
double8  bitselect(double8, double8, double8) __attribute__((overloadable));
double16 bitselect(double16, double16, double16) __attribute__((overloadable));


// 1. select_uchar_uchar
uchar    select(uchar, uchar, uchar) __attribute__((overloadable));
uchar2   select(uchar2, uchar2, uchar2) __attribute__((overloadable));
uchar3   select(uchar3, uchar3, uchar3) __attribute__((overloadable));
uchar4   select(uchar4, uchar4, uchar4) __attribute__((overloadable));
uchar8   select(uchar8, uchar8, uchar8) __attribute__((overloadable));
uchar16  select(uchar16, uchar16, uchar16) __attribute__((overloadable));

// 2. select_uchar_char
uchar    select(uchar, uchar, char) __attribute__((overloadable));
uchar2   select(uchar2, uchar2, char2) __attribute__((overloadable));
uchar3   select(uchar3, uchar3, char3) __attribute__((overloadable));
uchar4   select(uchar4, uchar4, char4) __attribute__((overloadable));
uchar8   select(uchar8, uchar8, char8) __attribute__((overloadable));
uchar16  select(uchar16, uchar16, char16) __attribute__((overloadable));

// 3. select_char_uchar
char     select(char, char, uchar) __attribute__((overloadable));
char2    select(char2, char2, uchar2) __attribute__((overloadable));
char3    select(char3, char3, uchar3) __attribute__((overloadable));
char4    select(char4, char4, uchar4) __attribute__((overloadable));
char8    select(char8, char8, uchar8) __attribute__((overloadable));
char16   select(char16, char16, uchar16) __attribute__((overloadable));

// 4. select_char_char
char     select(char, char, char) __attribute__((overloadable));
char2    select(char2, char2, char2) __attribute__((overloadable));
char3    select(char3, char3, char3) __attribute__((overloadable));
char4    select(char4, char4, char4) __attribute__((overloadable));
char8    select(char8, char8, char8) __attribute__((overloadable));
char16   select(char16, char16, char16) __attribute__((overloadable));

// 5. select_ushort_ushort
ushort   select(ushort, ushort, ushort) __attribute__((overloadable));
ushort2  select(ushort2, ushort2, ushort2) __attribute__((overloadable));
ushort3  select(ushort3, ushort3, ushort3) __attribute__((overloadable));
ushort4  select(ushort4, ushort4, ushort4) __attribute__((overloadable));
ushort8  select(ushort8, ushort8, ushort8) __attribute__((overloadable));
ushort16 select(ushort16, ushort16, ushort16) __attribute__((overloadable));

// 6. select_ushort_short
ushort   select(ushort, ushort, short) __attribute__((overloadable));
ushort2  select(ushort2, ushort2, short2) __attribute__((overloadable));
ushort3  select(ushort3, ushort3, short3) __attribute__((overloadable));
ushort4  select(ushort4, ushort4, short4) __attribute__((overloadable));
ushort8  select(ushort8, ushort8, short8) __attribute__((overloadable));
ushort16 select(ushort16, ushort16, short16) __attribute__((overloadable));

// 7. select_short_ushort
short   select(short, short, ushort) __attribute__((overloadable));
short2  select(short2, short2, ushort2) __attribute__((overloadable));
short3  select(short3, short3, ushort3) __attribute__((overloadable));
short4  select(short4, short4, ushort4) __attribute__((overloadable));
short8  select(short8, short8, ushort8) __attribute__((overloadable));
short16 select(short16, short16, ushort16) __attribute__((overloadable));

// 8. select_short_short
short   select(short, short, short) __attribute__((overloadable));
short2  select(short2, short2, short2) __attribute__((overloadable));
short3  select(short3, short3, short3) __attribute__((overloadable));
short4  select(short4, short4, short4) __attribute__((overloadable));
short8  select(short8, short8, short8) __attribute__((overloadable));
short16 select(short16, short16, short16) __attribute__((overloadable));

// 9. select_uint_uint
uint   select(uint, uint, uint) __attribute__((overloadable));
uint2  select(uint2, uint2, uint2) __attribute__((overloadable));
uint3  select(uint3, uint3, uint3) __attribute__((overloadable));
uint4  select(uint4, uint4, uint4) __attribute__((overloadable));
uint8  select(uint8, uint8, uint8) __attribute__((overloadable));
uint16 select(uint16, uint16, uint16) __attribute__((overloadable));

// 10. select_uint_int
uint   select(uint, uint, int) __attribute__((overloadable));
uint2  select(uint2, uint2, int2) __attribute__((overloadable));
uint3  select(uint3, uint3, int3) __attribute__((overloadable));
uint4  select(uint4, uint4, int4) __attribute__((overloadable));
uint8  select(uint8, uint8, int8) __attribute__((overloadable));
uint16 select(uint16, uint16, int16) __attribute__((overloadable));

// 11. select_int_uint
int   select(int, int, uint) __attribute__((overloadable));
int2  select(int2, int2, uint2) __attribute__((overloadable));
int3  select(int3, int3, uint3) __attribute__((overloadable));
int4  select(int4, int4, uint4) __attribute__((overloadable));
int8  select(int8, int8, uint8) __attribute__((overloadable));
int16 select(int16, int16, uint16) __attribute__((overloadable));

// 12. select_int_int
int   select(int, int, int) __attribute__((overloadable));
int2  select(int2, int2, int2) __attribute__((overloadable));
int3  select(int3, int3, int3) __attribute__((overloadable));
int4  select(int4, int4, int4) __attribute__((overloadable));
int8  select(int8, int8, int8) __attribute__((overloadable));
int16 select(int16, int16, int16) __attribute__((overloadable));

// 13. select_float_uint
float   select(float, float, uint) __attribute__((overloadable));
float2  select(float2, float2, uint2) __attribute__((overloadable));
float3  select(float3, float3, uint3) __attribute__((overloadable));
float4  select(float4, float4, uint4) __attribute__((overloadable));
float8  select(float8, float8, uint8) __attribute__((overloadable));
float16 select(float16, float16, uint16) __attribute__((overloadable));

// 14. select_float_int
float   select(float, float, int) __attribute__((overloadable));
float2  select(float2, float2, int2) __attribute__((overloadable));
float3  select(float3, float3, int3) __attribute__((overloadable));
float4  select(float4, float4, int4) __attribute__((overloadable));
float8  select(float8, float8, int8) __attribute__((overloadable));
float16 select(float16, float16, int16) __attribute__((overloadable));

// 15. select_ulong_ulong
ulong   select(ulong, ulong, ulong) __attribute__((overloadable));
ulong2  select(ulong2, ulong2, ulong2) __attribute__((overloadable));
ulong3  select(ulong3, ulong3, ulong3) __attribute__((overloadable));
ulong4  select(ulong4, ulong4, ulong4) __attribute__((overloadable));
ulong8  select(ulong8, ulong8, ulong8) __attribute__((overloadable));
ulong16 select(ulong16, ulong16, ulong16) __attribute__((overloadable));

// 16. select_ulong_long
ulong   select(ulong, ulong, long) __attribute__((overloadable));
ulong2  select(ulong2, ulong2, long2) __attribute__((overloadable));
ulong3  select(ulong3, ulong3, long3) __attribute__((overloadable));
ulong4  select(ulong4, ulong4, long4) __attribute__((overloadable));
ulong8  select(ulong8, ulong8, long8) __attribute__((overloadable));
ulong16 select(ulong16, ulong16, long16) __attribute__((overloadable));

// 17. select_long_ulong
long   select(long, long, ulong) __attribute__((overloadable));
long2  select(long2, long2, ulong2) __attribute__((overloadable));
long3  select(long3, long3, ulong3) __attribute__((overloadable));
long4  select(long4, long4, ulong4) __attribute__((overloadable));
long8  select(long8, long8, ulong8) __attribute__((overloadable));
long16 select(long16, long16, ulong16) __attribute__((overloadable));

// 18. select_long_long
long   select(long, long, long) __attribute__((overloadable));
long2  select(long2, long2, long2) __attribute__((overloadable));
long3  select(long3, long3, long3) __attribute__((overloadable));
long4  select(long4, long4, long4) __attribute__((overloadable));
long8  select(long8, long8, long8) __attribute__((overloadable));
long16 select(long16, long16, long16) __attribute__((overloadable));

// 19. select_double_long
double   select(double, double, long) __attribute__((overloadable));
double2  select(double2, double2, long2) __attribute__((overloadable));
double3  select(double3, double3, long3) __attribute__((overloadable));
double4  select(double4, double4, long4) __attribute__((overloadable));
double8  select(double8, double8, long8) __attribute__((overloadable));
double16 select(double16, double16, long16) __attribute__((overloadable));

// 20. select_double_ulong
double   select(double, double, ulong) __attribute__((overloadable));
double2  select(double2, double2, ulong2) __attribute__((overloadable));
double3  select(double3, double3, ulong3) __attribute__((overloadable));
double4  select(double4, double4, ulong4) __attribute__((overloadable));
double8  select(double8, double8, ulong8) __attribute__((overloadable));
double16 select(double16, double16, ulong16) __attribute__((overloadable));

#endif //__CL_BUILTINS_RELATIONAL_H__
