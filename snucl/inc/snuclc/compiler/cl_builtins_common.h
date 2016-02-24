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

#ifndef __CL_BUILTINS_COMMON_H__
#define __CL_BUILTINS_COMMON_H__

#include "cl_types.h"

/* 6.12.4 Common Functions */
/* Table 6.12 Scalar and Vector Argument Built-in Common Function Table */
float    clamp(float, float, float) __attribute__((overloadable));
float2   clamp(float2, float2, float2) __attribute__((overloadable));
float3   clamp(float3, float3, float3) __attribute__((overloadable));
float4   clamp(float4, float4, float4) __attribute__((overloadable));
float8   clamp(float8, float8, float8) __attribute__((overloadable));
float16  clamp(float16, float16, float16) __attribute__((overloadable));
double   clamp(double, double, double) __attribute__((overloadable));
double2  clamp(double2, double2, double2) __attribute__((overloadable));
double3  clamp(double3, double3, double3) __attribute__((overloadable));
double4  clamp(double4, double4, double4) __attribute__((overloadable));
double8  clamp(double8, double8, double8) __attribute__((overloadable));
double16 clamp(double16, double16, double16) __attribute__((overloadable));

float2   clamp(float2, float, float) __attribute__((overloadable));
float3   clamp(float3, float, float) __attribute__((overloadable));
float4   clamp(float4, float, float) __attribute__((overloadable));
float8   clamp(float8, float, float) __attribute__((overloadable));
float16  clamp(float16, float, float) __attribute__((overloadable));

double2  clamp(double2, double, double) __attribute__((overloadable));
double3  clamp(double3, double, double) __attribute__((overloadable));
double4  clamp(double4, double, double) __attribute__((overloadable));
double8  clamp(double8, double, double) __attribute__((overloadable));
double16 clamp(double16, double, double) __attribute__((overloadable));

float    degrees(float) __attribute__((overloadable));
float2   degrees(float2) __attribute__((overloadable));
float3   degrees(float3) __attribute__((overloadable));
float4   degrees(float4) __attribute__((overloadable));
float8   degrees(float8) __attribute__((overloadable));
float16  degrees(float16) __attribute__((overloadable));
double   degrees(double) __attribute__((overloadable));
double2  degrees(double2) __attribute__((overloadable));
double3  degrees(double3) __attribute__((overloadable));
double4  degrees(double4) __attribute__((overloadable));
double8  degrees(double8) __attribute__((overloadable));
double16 degrees(double16) __attribute__((overloadable));

float    max(float, float) __attribute__((overloadable));
float2   max(float2, float2) __attribute__((overloadable));
float3   max(float3, float3) __attribute__((overloadable));
float4   max(float4, float4) __attribute__((overloadable));
float8   max(float8, float8) __attribute__((overloadable));
float16  max(float16, float16) __attribute__((overloadable));
double   max(double, double) __attribute__((overloadable));
double2  max(double2, double2) __attribute__((overloadable));
double3  max(double3, double3) __attribute__((overloadable));
double4  max(double4, double4) __attribute__((overloadable));
double8  max(double8, double8) __attribute__((overloadable));
double16 max(double16, double16) __attribute__((overloadable));

float2   max(float2, float) __attribute__((overloadable));
float3   max(float3, float) __attribute__((overloadable));
float4   max(float4, float) __attribute__((overloadable));
float8   max(float8, float) __attribute__((overloadable));
float16  max(float16, float) __attribute__((overloadable));

double2  max(double2, double) __attribute__((overloadable));
double3  max(double3, double) __attribute__((overloadable));
double4  max(double4, double) __attribute__((overloadable));
double8  max(double8, double) __attribute__((overloadable));
double16 max(double16, double) __attribute__((overloadable));

float    min(float, float) __attribute__((overloadable));
float2   min(float2, float2) __attribute__((overloadable));
float3   min(float3, float3) __attribute__((overloadable));
float4   min(float4, float4) __attribute__((overloadable));
float8   min(float8, float8) __attribute__((overloadable));
float16  min(float16, float16) __attribute__((overloadable));
double   min(double, double) __attribute__((overloadable));
double2  min(double2, double2) __attribute__((overloadable));
double3  min(double3, double3) __attribute__((overloadable));
double4  min(double4, double4) __attribute__((overloadable));
double8  min(double8, double8) __attribute__((overloadable));
double16 min(double16, double16) __attribute__((overloadable));

float2   min(float2, float) __attribute__((overloadable));
float3   min(float3, float) __attribute__((overloadable));
float4   min(float4, float) __attribute__((overloadable));
float8   min(float8, float) __attribute__((overloadable));
float16  min(float16, float) __attribute__((overloadable));

double2  min(double2, double) __attribute__((overloadable));
double3  min(double3, double) __attribute__((overloadable));
double4  min(double4, double) __attribute__((overloadable));
double8  min(double8, double) __attribute__((overloadable));
double16 min(double16, double) __attribute__((overloadable));

float    mix(float, float, float) __attribute__((overloadable));
float2   mix(float2, float2, float2) __attribute__((overloadable));
float3   mix(float3, float3, float3) __attribute__((overloadable));
float4   mix(float4, float4, float4) __attribute__((overloadable));
float8   mix(float8, float8, float8) __attribute__((overloadable));
float16  mix(float16, float16, float16) __attribute__((overloadable));
double   mix(double, double, double) __attribute__((overloadable));
double2  mix(double2, double2, double2) __attribute__((overloadable));
double3  mix(double3, double3, double3) __attribute__((overloadable));
double4  mix(double4, double4, double4) __attribute__((overloadable));
double8  mix(double8, double8, double8) __attribute__((overloadable));
double16 mix(double16, double16, double16) __attribute__((overloadable));

float2   mix(float2, float2, float) __attribute__((overloadable));
float3   mix(float3, float3, float) __attribute__((overloadable));
float4   mix(float4, float4, float) __attribute__((overloadable));
float8   mix(float8, float8, float) __attribute__((overloadable));
float16  mix(float16, float16, float) __attribute__((overloadable));
double2  mix(double2, double2, double) __attribute__((overloadable));
double3  mix(double3, double3, double) __attribute__((overloadable));
double4  mix(double4, double4, double) __attribute__((overloadable));
double8  mix(double8, double8, double) __attribute__((overloadable));
double16 mix(double16, double16, double) __attribute__((overloadable));

float    radians(float) __attribute__((overloadable));
float2   radians(float2) __attribute__((overloadable));
float3   radians(float3) __attribute__((overloadable));
float4   radians(float4) __attribute__((overloadable));
float8   radians(float8) __attribute__((overloadable));
float16  radians(float16) __attribute__((overloadable));
double   radians(double) __attribute__((overloadable));
double2  radians(double2) __attribute__((overloadable));
double3  radians(double3) __attribute__((overloadable));
double4  radians(double4) __attribute__((overloadable));
double8  radians(double8) __attribute__((overloadable));
double16 radians(double16) __attribute__((overloadable));

float    step(float, float) __attribute__((overloadable));
float2   step(float2, float2) __attribute__((overloadable));
float3   step(float3, float3) __attribute__((overloadable));
float4   step(float4, float4) __attribute__((overloadable));
float8   step(float8, float8) __attribute__((overloadable));
float16  step(float16, float16) __attribute__((overloadable));
double   step(double, double) __attribute__((overloadable));
double2  step(double2, double2) __attribute__((overloadable));
double3  step(double3, double3) __attribute__((overloadable));
double4  step(double4, double4) __attribute__((overloadable));
double8  step(double8, double8) __attribute__((overloadable));
double16 step(double16, double16) __attribute__((overloadable));

float2   step(float, float2) __attribute__((overloadable));
float3   step(float, float3) __attribute__((overloadable));
float4   step(float, float4) __attribute__((overloadable));
float8   step(float, float8) __attribute__((overloadable));
float16  step(float, float16) __attribute__((overloadable));

double2  step(double, double2) __attribute__((overloadable));
double3  step(double, double3) __attribute__((overloadable));
double4  step(double, double4) __attribute__((overloadable));
double8  step(double, double8) __attribute__((overloadable));
double16 step(double, double16) __attribute__((overloadable));

float    smoothstep(float, float, float) __attribute__((overloadable));
float2   smoothstep(float2, float2, float2) __attribute__((overloadable));
float3   smoothstep(float3, float3, float3) __attribute__((overloadable));
float4   smoothstep(float4, float4, float4) __attribute__((overloadable));
float8   smoothstep(float8, float8, float8) __attribute__((overloadable));
float16  smoothstep(float16, float16, float16) __attribute__((overloadable));
double   smoothstep(double, double, double) __attribute__((overloadable));
double2  smoothstep(double2, double2, double2) __attribute__((overloadable));
double3  smoothstep(double3, double3, double3) __attribute__((overloadable));
double4  smoothstep(double4, double4, double4) __attribute__((overloadable));
double8  smoothstep(double8, double8, double8) __attribute__((overloadable));
double16 smoothstep(double16, double16, double16) __attribute__((overloadable));

float2   smoothstep(float, float, float2) __attribute__((overloadable));
float3   smoothstep(float, float, float3) __attribute__((overloadable));
float4   smoothstep(float, float, float4) __attribute__((overloadable));
float8   smoothstep(float, float, float8) __attribute__((overloadable));
float16  smoothstep(float, float, float16) __attribute__((overloadable));

double2  smoothstep(double, double, double2) __attribute__((overloadable));
double3  smoothstep(double, double, double3) __attribute__((overloadable));
double4  smoothstep(double, double, double4) __attribute__((overloadable));
double8  smoothstep(double, double, double8) __attribute__((overloadable));
double16 smoothstep(double, double, double16) __attribute__((overloadable));

float    sign(float) __attribute__((overloadable));
float2   sign(float2) __attribute__((overloadable));
float3   sign(float3) __attribute__((overloadable));
float4   sign(float4) __attribute__((overloadable));
float8   sign(float8) __attribute__((overloadable));
float16  sign(float16) __attribute__((overloadable));
double   sign(double) __attribute__((overloadable));
double2  sign(double2) __attribute__((overloadable));
double3  sign(double3) __attribute__((overloadable));
double4  sign(double4) __attribute__((overloadable));
double8  sign(double8) __attribute__((overloadable));
double16 sign(double16) __attribute__((overloadable));

#endif //__CL_BUILTINS_COMMON_H__
