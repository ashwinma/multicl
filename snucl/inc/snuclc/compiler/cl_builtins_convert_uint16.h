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

#ifndef __CL_BUILTINS_CONVERT_UINT16_H__
#define __CL_BUILTINS_CONVERT_UINT16_H__

#include "cl_types.h"

uint16 convert_uint16(char16) __attribute__((overloadable));
uint16 convert_uint16(uchar16) __attribute__((overloadable));
uint16 convert_uint16(short16) __attribute__((overloadable));
uint16 convert_uint16(ushort16) __attribute__((overloadable));
uint16 convert_uint16(int16) __attribute__((overloadable));
uint16 convert_uint16(uint16) __attribute__((overloadable));
uint16 convert_uint16(long16) __attribute__((overloadable));
uint16 convert_uint16(ulong16) __attribute__((overloadable));
uint16 convert_uint16(float16) __attribute__((overloadable));
uint16 convert_uint16(double16) __attribute__((overloadable));
uint16 convert_uint16_rte(char16) __attribute__((overloadable));
uint16 convert_uint16_rte(uchar16) __attribute__((overloadable));
uint16 convert_uint16_rte(short16) __attribute__((overloadable));
uint16 convert_uint16_rte(ushort16) __attribute__((overloadable));
uint16 convert_uint16_rte(int16) __attribute__((overloadable));
uint16 convert_uint16_rte(uint16) __attribute__((overloadable));
uint16 convert_uint16_rte(long16) __attribute__((overloadable));
uint16 convert_uint16_rte(ulong16) __attribute__((overloadable));
uint16 convert_uint16_rte(float16) __attribute__((overloadable));
uint16 convert_uint16_rte(double16) __attribute__((overloadable));
uint16 convert_uint16_rtz(char16) __attribute__((overloadable));
uint16 convert_uint16_rtz(uchar16) __attribute__((overloadable));
uint16 convert_uint16_rtz(short16) __attribute__((overloadable));
uint16 convert_uint16_rtz(ushort16) __attribute__((overloadable));
uint16 convert_uint16_rtz(int16) __attribute__((overloadable));
uint16 convert_uint16_rtz(uint16) __attribute__((overloadable));
uint16 convert_uint16_rtz(long16) __attribute__((overloadable));
uint16 convert_uint16_rtz(ulong16) __attribute__((overloadable));
uint16 convert_uint16_rtz(float16) __attribute__((overloadable));
uint16 convert_uint16_rtz(double16) __attribute__((overloadable));
uint16 convert_uint16_rtp(char16) __attribute__((overloadable));
uint16 convert_uint16_rtp(uchar16) __attribute__((overloadable));
uint16 convert_uint16_rtp(short16) __attribute__((overloadable));
uint16 convert_uint16_rtp(ushort16) __attribute__((overloadable));
uint16 convert_uint16_rtp(int16) __attribute__((overloadable));
uint16 convert_uint16_rtp(uint16) __attribute__((overloadable));
uint16 convert_uint16_rtp(long16) __attribute__((overloadable));
uint16 convert_uint16_rtp(ulong16) __attribute__((overloadable));
uint16 convert_uint16_rtp(float16) __attribute__((overloadable));
uint16 convert_uint16_rtp(double16) __attribute__((overloadable));
uint16 convert_uint16_rtn(char16) __attribute__((overloadable));
uint16 convert_uint16_rtn(uchar16) __attribute__((overloadable));
uint16 convert_uint16_rtn(short16) __attribute__((overloadable));
uint16 convert_uint16_rtn(ushort16) __attribute__((overloadable));
uint16 convert_uint16_rtn(int16) __attribute__((overloadable));
uint16 convert_uint16_rtn(uint16) __attribute__((overloadable));
uint16 convert_uint16_rtn(long16) __attribute__((overloadable));
uint16 convert_uint16_rtn(ulong16) __attribute__((overloadable));
uint16 convert_uint16_rtn(float16) __attribute__((overloadable));
uint16 convert_uint16_rtn(double16) __attribute__((overloadable));
uint16 convert_uint16_sat(char16) __attribute__((overloadable));
uint16 convert_uint16_sat(uchar16) __attribute__((overloadable));
uint16 convert_uint16_sat(short16) __attribute__((overloadable));
uint16 convert_uint16_sat(ushort16) __attribute__((overloadable));
uint16 convert_uint16_sat(int16) __attribute__((overloadable));
uint16 convert_uint16_sat(uint16) __attribute__((overloadable));
uint16 convert_uint16_sat(long16) __attribute__((overloadable));
uint16 convert_uint16_sat(ulong16) __attribute__((overloadable));
uint16 convert_uint16_sat(float16) __attribute__((overloadable));
uint16 convert_uint16_sat(double16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(char16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(uchar16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(short16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(ushort16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(int16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(uint16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(long16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(ulong16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(float16) __attribute__((overloadable));
uint16 convert_uint16_sat_rte(double16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(char16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(uchar16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(short16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(ushort16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(int16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(uint16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(long16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(ulong16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(float16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtz(double16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(char16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(uchar16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(short16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(ushort16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(int16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(uint16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(long16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(ulong16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(float16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtp(double16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(char16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(uchar16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(short16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(ushort16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(int16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(uint16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(long16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(ulong16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(float16) __attribute__((overloadable));
uint16 convert_uint16_sat_rtn(double16) __attribute__((overloadable));

#endif //__CL_BUILTINS_CONVERT_UINT16_H__
