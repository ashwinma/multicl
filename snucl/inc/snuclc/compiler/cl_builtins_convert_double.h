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

#ifndef __CL_BUILTINS_CONVERT_DOUBLE_H__
#define __CL_BUILTINS_CONVERT_DOUBLE_H__

#include "cl_types.h"

double convert_double(bool) __attribute__((overloadable));
double convert_double(char) __attribute__((overloadable));
double convert_double(uchar) __attribute__((overloadable));
double convert_double(short) __attribute__((overloadable));
double convert_double(ushort) __attribute__((overloadable));
double convert_double(int) __attribute__((overloadable));
double convert_double(uint) __attribute__((overloadable));
double convert_double(long) __attribute__((overloadable));
double convert_double(ulong) __attribute__((overloadable));
double convert_double(float) __attribute__((overloadable));
double convert_double(double) __attribute__((overloadable));
double convert_double_rte(bool) __attribute__((overloadable));
double convert_double_rte(char) __attribute__((overloadable));
double convert_double_rte(uchar) __attribute__((overloadable));
double convert_double_rte(short) __attribute__((overloadable));
double convert_double_rte(ushort) __attribute__((overloadable));
double convert_double_rte(int) __attribute__((overloadable));
double convert_double_rte(uint) __attribute__((overloadable));
double convert_double_rte(long) __attribute__((overloadable));
double convert_double_rte(ulong) __attribute__((overloadable));
double convert_double_rte(float) __attribute__((overloadable));
double convert_double_rte(double) __attribute__((overloadable));
double convert_double_rtz(bool) __attribute__((overloadable));
double convert_double_rtz(char) __attribute__((overloadable));
double convert_double_rtz(uchar) __attribute__((overloadable));
double convert_double_rtz(short) __attribute__((overloadable));
double convert_double_rtz(ushort) __attribute__((overloadable));
double convert_double_rtz(int) __attribute__((overloadable));
double convert_double_rtz(uint) __attribute__((overloadable));
double convert_double_rtz(long) __attribute__((overloadable));
double convert_double_rtz(ulong) __attribute__((overloadable));
double convert_double_rtz(float) __attribute__((overloadable));
double convert_double_rtz(double) __attribute__((overloadable));
double convert_double_rtp(bool) __attribute__((overloadable));
double convert_double_rtp(char) __attribute__((overloadable));
double convert_double_rtp(uchar) __attribute__((overloadable));
double convert_double_rtp(short) __attribute__((overloadable));
double convert_double_rtp(ushort) __attribute__((overloadable));
double convert_double_rtp(int) __attribute__((overloadable));
double convert_double_rtp(uint) __attribute__((overloadable));
double convert_double_rtp(long) __attribute__((overloadable));
double convert_double_rtp(ulong) __attribute__((overloadable));
double convert_double_rtp(float) __attribute__((overloadable));
double convert_double_rtp(double) __attribute__((overloadable));
double convert_double_rtn(bool) __attribute__((overloadable));
double convert_double_rtn(char) __attribute__((overloadable));
double convert_double_rtn(uchar) __attribute__((overloadable));
double convert_double_rtn(short) __attribute__((overloadable));
double convert_double_rtn(ushort) __attribute__((overloadable));
double convert_double_rtn(int) __attribute__((overloadable));
double convert_double_rtn(uint) __attribute__((overloadable));
double convert_double_rtn(long) __attribute__((overloadable));
double convert_double_rtn(ulong) __attribute__((overloadable));
double convert_double_rtn(float) __attribute__((overloadable));
double convert_double_rtn(double) __attribute__((overloadable));
double convert_double_sat(bool) __attribute__((overloadable));
double convert_double_sat(char) __attribute__((overloadable));
double convert_double_sat(uchar) __attribute__((overloadable));
double convert_double_sat(short) __attribute__((overloadable));
double convert_double_sat(ushort) __attribute__((overloadable));
double convert_double_sat(int) __attribute__((overloadable));
double convert_double_sat(uint) __attribute__((overloadable));
double convert_double_sat(long) __attribute__((overloadable));
double convert_double_sat(ulong) __attribute__((overloadable));
double convert_double_sat(float) __attribute__((overloadable));
double convert_double_sat(double) __attribute__((overloadable));
double convert_double_sat_rte(bool) __attribute__((overloadable));
double convert_double_sat_rte(char) __attribute__((overloadable));
double convert_double_sat_rte(uchar) __attribute__((overloadable));
double convert_double_sat_rte(short) __attribute__((overloadable));
double convert_double_sat_rte(ushort) __attribute__((overloadable));
double convert_double_sat_rte(int) __attribute__((overloadable));
double convert_double_sat_rte(uint) __attribute__((overloadable));
double convert_double_sat_rte(long) __attribute__((overloadable));
double convert_double_sat_rte(ulong) __attribute__((overloadable));
double convert_double_sat_rte(float) __attribute__((overloadable));
double convert_double_sat_rte(double) __attribute__((overloadable));
double convert_double_sat_rtz(bool) __attribute__((overloadable));
double convert_double_sat_rtz(char) __attribute__((overloadable));
double convert_double_sat_rtz(uchar) __attribute__((overloadable));
double convert_double_sat_rtz(short) __attribute__((overloadable));
double convert_double_sat_rtz(ushort) __attribute__((overloadable));
double convert_double_sat_rtz(int) __attribute__((overloadable));
double convert_double_sat_rtz(uint) __attribute__((overloadable));
double convert_double_sat_rtz(long) __attribute__((overloadable));
double convert_double_sat_rtz(ulong) __attribute__((overloadable));
double convert_double_sat_rtz(float) __attribute__((overloadable));
double convert_double_sat_rtz(double) __attribute__((overloadable));
double convert_double_sat_rtp(bool) __attribute__((overloadable));
double convert_double_sat_rtp(char) __attribute__((overloadable));
double convert_double_sat_rtp(uchar) __attribute__((overloadable));
double convert_double_sat_rtp(short) __attribute__((overloadable));
double convert_double_sat_rtp(ushort) __attribute__((overloadable));
double convert_double_sat_rtp(int) __attribute__((overloadable));
double convert_double_sat_rtp(uint) __attribute__((overloadable));
double convert_double_sat_rtp(long) __attribute__((overloadable));
double convert_double_sat_rtp(ulong) __attribute__((overloadable));
double convert_double_sat_rtp(float) __attribute__((overloadable));
double convert_double_sat_rtp(double) __attribute__((overloadable));
double convert_double_sat_rtn(bool) __attribute__((overloadable));
double convert_double_sat_rtn(char) __attribute__((overloadable));
double convert_double_sat_rtn(uchar) __attribute__((overloadable));
double convert_double_sat_rtn(short) __attribute__((overloadable));
double convert_double_sat_rtn(ushort) __attribute__((overloadable));
double convert_double_sat_rtn(int) __attribute__((overloadable));
double convert_double_sat_rtn(uint) __attribute__((overloadable));
double convert_double_sat_rtn(long) __attribute__((overloadable));
double convert_double_sat_rtn(ulong) __attribute__((overloadable));
double convert_double_sat_rtn(float) __attribute__((overloadable));
double convert_double_sat_rtn(double) __attribute__((overloadable));

#endif //__CL_BUILTINS_CONVERT_DOUBLE_H__
