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

#ifndef __CL_BUILTINS_GEOMETRIC_H__
#define __CL_BUILTINS_GEOMETRIC_H__

#include "cl_types.h"

/* 6.12.5 Geometric Functions */
/* Table 6.13 Scalar and Vector Argument Built-in Geometic Function Table */
float4  cross(float4, float4) __attribute__((overloadable));
float3  cross(float3, float3) __attribute__((overloadable));
double4 cross(double4, double4) __attribute__((overloadable));
double3 cross(double3, double3) __attribute__((overloadable));

float   dot(float, float) __attribute__((overloadable));
float   dot(float2, float2) __attribute__((overloadable));
float   dot(float3, float3) __attribute__((overloadable));
float   dot(float4, float4) __attribute__((overloadable));
double  dot(double, double) __attribute__((overloadable));
double  dot(double2, double2) __attribute__((overloadable));
double  dot(double3, double3) __attribute__((overloadable));
double  dot(double4, double4) __attribute__((overloadable));

float   distance(float, float) __attribute__((overloadable));
float   distance(float2, float2) __attribute__((overloadable));
float   distance(float3, float3) __attribute__((overloadable));
float   distance(float4, float4) __attribute__((overloadable));
double  distance(double, double) __attribute__((overloadable));
double  distance(double2, double2) __attribute__((overloadable));
double  distance(double3, double3) __attribute__((overloadable));
double  distance(double4, double4) __attribute__((overloadable));

float   length(float) __attribute__((overloadable));
float   length(float2) __attribute__((overloadable));
float   length(float3) __attribute__((overloadable));
float   length(float4) __attribute__((overloadable));
double  length(double) __attribute__((overloadable));
double  length(double2) __attribute__((overloadable));
double  length(double3) __attribute__((overloadable));
double  length(double4) __attribute__((overloadable));

float   normalize(float) __attribute__((overloadable));
float2  normalize(float2) __attribute__((overloadable));
float3  normalize(float3) __attribute__((overloadable));
float4  normalize(float4) __attribute__((overloadable));
double  normalize(double) __attribute__((overloadable));
double2 normalize(double2) __attribute__((overloadable));
double3 normalize(double3) __attribute__((overloadable));
double4 normalize(double4) __attribute__((overloadable));

float   fast_distance(float, float) __attribute__((overloadable));
float   fast_distance(float2, float2) __attribute__((overloadable));
float   fast_distance(float3, float3) __attribute__((overloadable));
float   fast_distance(float4, float4) __attribute__((overloadable));

float   fast_length(float) __attribute__((overloadable));
float   fast_length(float2) __attribute__((overloadable));
float   fast_length(float3) __attribute__((overloadable));
float   fast_length(float4) __attribute__((overloadable));

float   fast_normalize(float) __attribute__((overloadable));
float2  fast_normalize(float2) __attribute__((overloadable));
float3  fast_normalize(float3) __attribute__((overloadable));
float4  fast_normalize(float4) __attribute__((overloadable));

#endif //__CL_BUILTINS_GEOMETRIC_H__
