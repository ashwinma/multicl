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

#ifndef __CL_BUILTINS_CONVERT_FLOAT3_H__
#define __CL_BUILTINS_CONVERT_FLOAT3_H__

#include "cl_types.h"

float3 convert_float3(char3) __attribute__((overloadable));
float3 convert_float3(uchar3) __attribute__((overloadable));
float3 convert_float3(short3) __attribute__((overloadable));
float3 convert_float3(ushort3) __attribute__((overloadable));
float3 convert_float3(int3) __attribute__((overloadable));
float3 convert_float3(uint3) __attribute__((overloadable));
float3 convert_float3(long3) __attribute__((overloadable));
float3 convert_float3(ulong3) __attribute__((overloadable));
float3 convert_float3(float3) __attribute__((overloadable));
float3 convert_float3(double3) __attribute__((overloadable));
float3 convert_float3_rte(char3) __attribute__((overloadable));
float3 convert_float3_rte(uchar3) __attribute__((overloadable));
float3 convert_float3_rte(short3) __attribute__((overloadable));
float3 convert_float3_rte(ushort3) __attribute__((overloadable));
float3 convert_float3_rte(int3) __attribute__((overloadable));
float3 convert_float3_rte(uint3) __attribute__((overloadable));
float3 convert_float3_rte(long3) __attribute__((overloadable));
float3 convert_float3_rte(ulong3) __attribute__((overloadable));
float3 convert_float3_rte(float3) __attribute__((overloadable));
float3 convert_float3_rte(double3) __attribute__((overloadable));
float3 convert_float3_rtz(char3) __attribute__((overloadable));
float3 convert_float3_rtz(uchar3) __attribute__((overloadable));
float3 convert_float3_rtz(short3) __attribute__((overloadable));
float3 convert_float3_rtz(ushort3) __attribute__((overloadable));
float3 convert_float3_rtz(int3) __attribute__((overloadable));
float3 convert_float3_rtz(uint3) __attribute__((overloadable));
float3 convert_float3_rtz(long3) __attribute__((overloadable));
float3 convert_float3_rtz(ulong3) __attribute__((overloadable));
float3 convert_float3_rtz(float3) __attribute__((overloadable));
float3 convert_float3_rtz(double3) __attribute__((overloadable));
float3 convert_float3_rtp(char3) __attribute__((overloadable));
float3 convert_float3_rtp(uchar3) __attribute__((overloadable));
float3 convert_float3_rtp(short3) __attribute__((overloadable));
float3 convert_float3_rtp(ushort3) __attribute__((overloadable));
float3 convert_float3_rtp(int3) __attribute__((overloadable));
float3 convert_float3_rtp(uint3) __attribute__((overloadable));
float3 convert_float3_rtp(long3) __attribute__((overloadable));
float3 convert_float3_rtp(ulong3) __attribute__((overloadable));
float3 convert_float3_rtp(float3) __attribute__((overloadable));
float3 convert_float3_rtp(double3) __attribute__((overloadable));
float3 convert_float3_rtn(char3) __attribute__((overloadable));
float3 convert_float3_rtn(uchar3) __attribute__((overloadable));
float3 convert_float3_rtn(short3) __attribute__((overloadable));
float3 convert_float3_rtn(ushort3) __attribute__((overloadable));
float3 convert_float3_rtn(int3) __attribute__((overloadable));
float3 convert_float3_rtn(uint3) __attribute__((overloadable));
float3 convert_float3_rtn(long3) __attribute__((overloadable));
float3 convert_float3_rtn(ulong3) __attribute__((overloadable));
float3 convert_float3_rtn(float3) __attribute__((overloadable));
float3 convert_float3_rtn(double3) __attribute__((overloadable));
float3 convert_float3_sat(char3) __attribute__((overloadable));
float3 convert_float3_sat(uchar3) __attribute__((overloadable));
float3 convert_float3_sat(short3) __attribute__((overloadable));
float3 convert_float3_sat(ushort3) __attribute__((overloadable));
float3 convert_float3_sat(int3) __attribute__((overloadable));
float3 convert_float3_sat(uint3) __attribute__((overloadable));
float3 convert_float3_sat(long3) __attribute__((overloadable));
float3 convert_float3_sat(ulong3) __attribute__((overloadable));
float3 convert_float3_sat(float3) __attribute__((overloadable));
float3 convert_float3_sat(double3) __attribute__((overloadable));
float3 convert_float3_sat_rte(char3) __attribute__((overloadable));
float3 convert_float3_sat_rte(uchar3) __attribute__((overloadable));
float3 convert_float3_sat_rte(short3) __attribute__((overloadable));
float3 convert_float3_sat_rte(ushort3) __attribute__((overloadable));
float3 convert_float3_sat_rte(int3) __attribute__((overloadable));
float3 convert_float3_sat_rte(uint3) __attribute__((overloadable));
float3 convert_float3_sat_rte(long3) __attribute__((overloadable));
float3 convert_float3_sat_rte(ulong3) __attribute__((overloadable));
float3 convert_float3_sat_rte(float3) __attribute__((overloadable));
float3 convert_float3_sat_rte(double3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(char3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(uchar3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(short3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(ushort3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(int3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(uint3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(long3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(ulong3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(float3) __attribute__((overloadable));
float3 convert_float3_sat_rtz(double3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(char3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(uchar3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(short3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(ushort3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(int3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(uint3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(long3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(ulong3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(float3) __attribute__((overloadable));
float3 convert_float3_sat_rtp(double3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(char3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(uchar3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(short3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(ushort3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(int3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(uint3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(long3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(ulong3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(float3) __attribute__((overloadable));
float3 convert_float3_sat_rtn(double3) __attribute__((overloadable));

#endif //__CL_BUILTINS_CONVERT_FLOAT3_H__
