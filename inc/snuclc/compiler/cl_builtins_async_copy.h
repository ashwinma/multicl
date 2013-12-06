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

#ifndef __CL_BUILTINS_ASYNC_H__
#define __CL_BUILTINS_ASYNC_H__

#include "cl_types.h"

/* 6.12.10 Async Copies from Global to Local Memory, Local to Global Memory, */
/*         and Prefetch                                                      */
/* Table 6.18 Built-in Async Copy and Prefetch Functions */
event_t async_work_group_copy(__local char *, const __global char *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local char2 *, const __global char2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local char3 *, const __global char3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local char4 *, const __global char4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local char8 *, const __global char8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local char16 *, const __global char16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uchar *, const __global uchar *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uchar2 *, const __global uchar2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uchar3 *, const __global uchar3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uchar4 *, const __global uchar4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uchar8 *, const __global uchar8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uchar16 *, const __global uchar16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local short *, const __global short *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local short2 *, const __global short2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local short3 *, const __global short3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local short4 *, const __global short4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local short8 *, const __global short8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local short16 *, const __global short16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ushort *, const __global ushort *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ushort2 *, const __global ushort2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ushort3 *, const __global ushort3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ushort4 *, const __global ushort4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ushort8 *, const __global ushort8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ushort16 *, const __global ushort16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local int *, const __global int *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local int2 *, const __global int2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local int3 *, const __global int3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local int4 *, const __global int4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local int8 *, const __global int8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local int16 *, const __global int16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uint *, const __global uint *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uint2 *, const __global uint2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uint3 *, const __global uint3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uint4 *, const __global uint4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uint8 *, const __global uint8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local uint16 *, const __global uint16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local long *, const __global long *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local long2 *, const __global long2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local long3 *, const __global long3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local long4 *, const __global long4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local long8 *, const __global long8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local long16 *, const __global long16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ulong *, const __global ulong *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ulong2 *, const __global ulong2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ulong3 *, const __global ulong3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ulong4 *, const __global ulong4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ulong8 *, const __global ulong8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local ulong16 *, const __global ulong16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local float *, const __global float *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local float2 *, const __global float2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local float3 *, const __global float3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local float4 *, const __global float4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local float8 *, const __global float8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local float16 *, const __global float16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local double *, const __global double *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local double2 *, const __global double2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local double3 *, const __global double3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local double4 *, const __global double4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local double8 *, const __global double8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__local double16 *, const __global double16 *, size_t, event_t) __attribute__((overloadable));

event_t async_work_group_copy(__global char *, const __local char *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global char2 *, const __local char2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global char3 *, const __local char3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global char4 *, const __local char4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global char8 *, const __local char8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global char16 *, const __local char16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uchar *, const __local uchar *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uchar2 *, const __local uchar2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uchar3 *, const __local uchar3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uchar4 *, const __local uchar4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uchar8 *, const __local uchar8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uchar16 *, const __local uchar16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global short *, const __local short *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global short2 *, const __local short2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global short3 *, const __local short3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global short4 *, const __local short4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global short8 *, const __local short8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global short16 *, const __local short16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ushort *, const __local ushort *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ushort2 *, const __local ushort2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ushort3 *, const __local ushort3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ushort4 *, const __local ushort4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ushort8 *, const __local ushort8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ushort16 *, const __local ushort16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global int *, const __local int *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global int2 *, const __local int2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global int3 *, const __local int3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global int4 *, const __local int4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global int8 *, const __local int8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global int16 *, const __local int16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uint *, const __local uint *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uint2 *, const __local uint2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uint3 *, const __local uint3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uint4 *, const __local uint4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uint8 *, const __local uint8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global uint16 *, const __local uint16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global long *, const __local long *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global long2 *, const __local long2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global long3 *, const __local long3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global long4 *, const __local long4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global long8 *, const __local long8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global long16 *, const __local long16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ulong *, const __local ulong *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ulong2 *, const __local ulong2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ulong3 *, const __local ulong3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ulong4 *, const __local ulong4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ulong8 *, const __local ulong8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global ulong16 *, const __local ulong16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global float *, const __local float *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global float2 *, const __local float2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global float3 *, const __local float3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global float4 *, const __local float4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global float8 *, const __local float8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global float16 *, const __local float16 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global double *, const __local double *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global double2 *, const __local double2 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global double3 *, const __local double3 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global double4 *, const __local double4 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global double8 *, const __local double8 *, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_copy(__global double16 *, const __local double16 *, size_t, event_t) __attribute__((overloadable));

event_t async_work_group_strided_copy(__local char *, const __global char *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local char2 *, const __global char2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local char3 *, const __global char3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local char4 *, const __global char4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local char8 *, const __global char8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local char16 *, const __global char16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uchar *, const __global uchar *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uchar2 *, const __global uchar2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uchar3 *, const __global uchar3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uchar4 *, const __global uchar4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uchar8 *, const __global uchar8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uchar16 *, const __global uchar16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local short *, const __global short *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local short2 *, const __global short2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local short3 *, const __global short3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local short4 *, const __global short4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local short8 *, const __global short8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local short16 *, const __global short16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ushort *, const __global ushort *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ushort2 *, const __global ushort2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ushort3 *, const __global ushort3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ushort4 *, const __global ushort4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ushort8 *, const __global ushort8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ushort16 *, const __global ushort16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local int *, const __global int *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local int2 *, const __global int2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local int3 *, const __global int3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local int4 *, const __global int4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local int8 *, const __global int8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local int16 *, const __global int16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uint *, const __global uint *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uint2 *, const __global uint2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uint3 *, const __global uint3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uint4 *, const __global uint4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uint8 *, const __global uint8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local uint16 *, const __global uint16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local long *, const __global long *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local long2 *, const __global long2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local long3 *, const __global long3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local long4 *, const __global long4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local long8 *, const __global long8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local long16 *, const __global long16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ulong *, const __global ulong *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ulong2 *, const __global ulong2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ulong3 *, const __global ulong3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ulong4 *, const __global ulong4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ulong8 *, const __global ulong8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local ulong16 *, const __global ulong16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local float *, const __global float *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local float2 *, const __global float2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local float3 *, const __global float3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local float4 *, const __global float4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local float8 *, const __global float8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local float16 *, const __global float16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local double *, const __global double *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local double2 *, const __global double2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local double3 *, const __global double3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local double4 *, const __global double4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local double8 *, const __global double8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__local double16 *, const __global double16 *, size_t, size_t, event_t) __attribute__((overloadable));

event_t async_work_group_strided_copy(__global char *, const __local char *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global char2 *, const __local char2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global char3 *, const __local char3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global char4 *, const __local char4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global char8 *, const __local char8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global char16 *, const __local char16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uchar *, const __local uchar *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uchar2 *, const __local uchar2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uchar3 *, const __local uchar3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uchar4 *, const __local uchar4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uchar8 *, const __local uchar8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uchar16 *, const __local uchar16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global short *, const __local short *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global short2 *, const __local short2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global short3 *, const __local short3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global short4 *, const __local short4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global short8 *, const __local short8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global short16 *, const __local short16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ushort *, const __local ushort *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ushort2 *, const __local ushort2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ushort3 *, const __local ushort3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ushort4 *, const __local ushort4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ushort8 *, const __local ushort8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ushort16 *, const __local ushort16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global int *, const __local int *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global int2 *, const __local int2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global int3 *, const __local int3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global int4 *, const __local int4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global int8 *, const __local int8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global int16 *, const __local int16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uint *, const __local uint *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uint2 *, const __local uint2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uint3 *, const __local uint3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uint4 *, const __local uint4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uint8 *, const __local uint8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global uint16 *, const __local uint16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global long *, const __local long *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global long2 *, const __local long2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global long3 *, const __local long3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global long4 *, const __local long4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global long8 *, const __local long8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global long16 *, const __local long16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ulong *, const __local ulong *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ulong2 *, const __local ulong2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ulong3 *, const __local ulong3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ulong4 *, const __local ulong4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ulong8 *, const __local ulong8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global ulong16 *, const __local ulong16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global float *, const __local float *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global float2 *, const __local float2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global float3 *, const __local float3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global float4 *, const __local float4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global float8 *, const __local float8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global float16 *, const __local float16 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global double *, const __local double *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global double2 *, const __local double2 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global double3 *, const __local double3 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global double4 *, const __local double4 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global double8 *, const __local double8 *, size_t, size_t, event_t) __attribute__((overloadable));
event_t async_work_group_strided_copy(__global double16 *, const __local double16 *, size_t, size_t, event_t) __attribute__((overloadable));

void wait_group_events(int, event_t *) __attribute__((overloadable));

void prefetch(const __global char *, size_t) __attribute__((overloadable));
void prefetch(const __global char2 *, size_t) __attribute__((overloadable));
void prefetch(const __global char3 *, size_t) __attribute__((overloadable));
void prefetch(const __global char4 *, size_t) __attribute__((overloadable));
void prefetch(const __global char8 *, size_t) __attribute__((overloadable));
void prefetch(const __global char16 *, size_t) __attribute__((overloadable));
void prefetch(const __global uchar *, size_t) __attribute__((overloadable));
void prefetch(const __global uchar2 *, size_t) __attribute__((overloadable));
void prefetch(const __global uchar3 *, size_t) __attribute__((overloadable));
void prefetch(const __global uchar4 *, size_t) __attribute__((overloadable));
void prefetch(const __global uchar8 *, size_t) __attribute__((overloadable));
void prefetch(const __global uchar16 *, size_t) __attribute__((overloadable));
void prefetch(const __global short *, size_t) __attribute__((overloadable));
void prefetch(const __global short2 *, size_t) __attribute__((overloadable));
void prefetch(const __global short3 *, size_t) __attribute__((overloadable));
void prefetch(const __global short4 *, size_t) __attribute__((overloadable));
void prefetch(const __global short8 *, size_t) __attribute__((overloadable));
void prefetch(const __global short16 *, size_t) __attribute__((overloadable));
void prefetch(const __global ushort *, size_t) __attribute__((overloadable));
void prefetch(const __global ushort2 *, size_t) __attribute__((overloadable));
void prefetch(const __global ushort3 *, size_t) __attribute__((overloadable));
void prefetch(const __global ushort4 *, size_t) __attribute__((overloadable));
void prefetch(const __global ushort8 *, size_t) __attribute__((overloadable));
void prefetch(const __global ushort16 *, size_t) __attribute__((overloadable));
void prefetch(const __global int *, size_t) __attribute__((overloadable));
void prefetch(const __global int2 *, size_t) __attribute__((overloadable));
void prefetch(const __global int3 *, size_t) __attribute__((overloadable));
void prefetch(const __global int4 *, size_t) __attribute__((overloadable));
void prefetch(const __global int8 *, size_t) __attribute__((overloadable));
void prefetch(const __global int16 *, size_t) __attribute__((overloadable));
void prefetch(const __global uint *, size_t) __attribute__((overloadable));
void prefetch(const __global uint2 *, size_t) __attribute__((overloadable));
void prefetch(const __global uint3 *, size_t) __attribute__((overloadable));
void prefetch(const __global uint4 *, size_t) __attribute__((overloadable));
void prefetch(const __global uint8 *, size_t) __attribute__((overloadable));
void prefetch(const __global uint16 *, size_t) __attribute__((overloadable));
void prefetch(const __global long *, size_t) __attribute__((overloadable));
void prefetch(const __global long2 *, size_t) __attribute__((overloadable));
void prefetch(const __global long3 *, size_t) __attribute__((overloadable));
void prefetch(const __global long4 *, size_t) __attribute__((overloadable));
void prefetch(const __global long8 *, size_t) __attribute__((overloadable));
void prefetch(const __global long16 *, size_t) __attribute__((overloadable));
void prefetch(const __global ulong *, size_t) __attribute__((overloadable));
void prefetch(const __global ulong2 *, size_t) __attribute__((overloadable));
void prefetch(const __global ulong3 *, size_t) __attribute__((overloadable));
void prefetch(const __global ulong4 *, size_t) __attribute__((overloadable));
void prefetch(const __global ulong8 *, size_t) __attribute__((overloadable));
void prefetch(const __global ulong16 *, size_t) __attribute__((overloadable));
void prefetch(const __global float *, size_t) __attribute__((overloadable));
void prefetch(const __global float2 *, size_t) __attribute__((overloadable));
void prefetch(const __global float3 *, size_t) __attribute__((overloadable));
void prefetch(const __global float4 *, size_t) __attribute__((overloadable));
void prefetch(const __global float8 *, size_t) __attribute__((overloadable));
void prefetch(const __global float16 *, size_t) __attribute__((overloadable));
void prefetch(const __global double *, size_t) __attribute__((overloadable));
void prefetch(const __global double2 *, size_t) __attribute__((overloadable));
void prefetch(const __global double3 *, size_t) __attribute__((overloadable));
void prefetch(const __global double4 *, size_t) __attribute__((overloadable));
void prefetch(const __global double8 *, size_t) __attribute__((overloadable));
void prefetch(const __global double16 *, size_t) __attribute__((overloadable));

#endif //__CL_BUILTINS_ASYNC_H__
