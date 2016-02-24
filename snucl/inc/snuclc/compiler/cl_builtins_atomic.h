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

#ifndef __CL_BUILTINS_ATOMIC_H__
#define __CL_BUILTINS_ATOMIC_H__

/* 6.12.11 Atomic Functions */
/* Table 6.19 Built-in Atomic Functions */
int   atomic_add(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_add(volatile __global uint *, uint) __attribute__((overloadable));
int   atomic_add(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_add(volatile __local uint *, uint) __attribute__((overloadable));

int   atomic_sub(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_sub(volatile __global uint *, uint) __attribute__((overloadable));
int   atomic_sub(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_sub(volatile __local uint *, uint) __attribute__((overloadable));

int   atomic_xchg(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_xchg(volatile __global uint *, uint) __attribute__((overloadable));
float atomic_xchg(volatile __global float *, float) __attribute__((overloadable));
int   atomic_xchg(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_xchg(volatile __local uint *, uint) __attribute__((overloadable));
float atomic_xchg(volatile __local float *, float) __attribute__((overloadable));

int   atomic_inc(volatile __global int *) __attribute__((overloadable));
uint  atomic_inc(volatile __global uint *) __attribute__((overloadable));
int   atomic_inc(volatile __local int *) __attribute__((overloadable));
uint  atomic_inc(volatile __local uint *) __attribute__((overloadable));

int   atomic_dec(volatile __global int *) __attribute__((overloadable));
uint  atomic_dec(volatile __global uint *) __attribute__((overloadable));
int   atomic_dec(volatile __local int *) __attribute__((overloadable));
uint  atomic_dec(volatile __local uint *) __attribute__((overloadable));

int   atomic_cmpxchg(volatile __global int *, int, int) __attribute__((overloadable));
uint  atomic_cmpxchg(volatile __global uint *, uint, uint) __attribute__((overloadable));
int   atomic_cmpxchg(volatile __local int *, int, int) __attribute__((overloadable));
uint  atomic_cmpxchg(volatile __local uint *, uint, uint) __attribute__((overloadable));

int   atomic_min(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_min(volatile __global uint *, uint) __attribute__((overloadable));
int   atomic_min(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_min(volatile __local uint *, uint) __attribute__((overloadable));

int   atomic_max(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_max(volatile __global uint *, uint) __attribute__((overloadable));
int   atomic_max(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_max(volatile __local uint *, uint) __attribute__((overloadable));

int   atomic_and(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_and(volatile __global uint *, uint) __attribute__((overloadable));
int   atomic_and(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_and(volatile __local uint *, uint) __attribute__((overloadable));

int   atomic_or(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_or(volatile __global uint *, uint) __attribute__((overloadable));
int   atomic_or(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_or(volatile __local uint *, uint) __attribute__((overloadable));

int   atomic_xor(volatile __global int *, int) __attribute__((overloadable));
uint  atomic_xor(volatile __global uint *, uint) __attribute__((overloadable));
int   atomic_xor(volatile __local int *, int) __attribute__((overloadable));
uint  atomic_xor(volatile __local uint *, uint) __attribute__((overloadable));

#endif //__CL_BUILTINS_ATOMIC_H__
