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

#ifndef __CL_BUILTINS_CONVERT_UCHAR2_H__
#define __CL_BUILTINS_CONVERT_UCHAR2_H__

#include "cl_types.h"

uchar2 convert_uchar2(char2) __attribute__((overloadable));
uchar2 convert_uchar2(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2(short2) __attribute__((overloadable));
uchar2 convert_uchar2(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2(int2) __attribute__((overloadable));
uchar2 convert_uchar2(uint2) __attribute__((overloadable));
uchar2 convert_uchar2(long2) __attribute__((overloadable));
uchar2 convert_uchar2(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2(float2) __attribute__((overloadable));
uchar2 convert_uchar2(double2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(char2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(short2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(int2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(long2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(float2) __attribute__((overloadable));
uchar2 convert_uchar2_rte(double2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(char2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(short2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(int2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(long2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(float2) __attribute__((overloadable));
uchar2 convert_uchar2_rtz(double2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(char2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(short2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(int2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(long2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(float2) __attribute__((overloadable));
uchar2 convert_uchar2_rtp(double2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(char2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(short2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(int2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(long2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(float2) __attribute__((overloadable));
uchar2 convert_uchar2_rtn(double2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(char2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(short2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(int2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(long2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(float2) __attribute__((overloadable));
uchar2 convert_uchar2_sat(double2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(char2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(short2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(int2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(long2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(float2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rte(double2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(char2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(short2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(int2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(long2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(float2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtz(double2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(char2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(short2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(int2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(long2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(float2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtp(double2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(char2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(uchar2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(short2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(ushort2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(int2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(uint2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(long2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(ulong2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(float2) __attribute__((overloadable));
uchar2 convert_uchar2_sat_rtn(double2) __attribute__((overloadable));

#endif //__CL_BUILTINS_CONVERT_UCHAR2_H__
