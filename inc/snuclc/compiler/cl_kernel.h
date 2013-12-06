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

#ifndef __CL_KERNEL_H__
#define __CL_KERNEL_H__

// Address qualifiers
#define __global      __attribute__((annotate("cl_global")))
#define global        __attribute__((annotate("cl_global")))
#define __constant    __attribute__((annotate("cl_constant")))
#define constant      __attribute__((annotate("cl_constant")))
#define __local       __attribute__((annotate("cl_local")))
#define local         __attribute__((annotate("cl_local")))
#define __private     __attribute__((annotate("cl_private")))
#define private       __attribute__((annotate("cl_private")))

// Access qualifiers
#define __read_only   __attribute__((annotate("cl_read_only")))
#define read_only     __attribute__((annotate("cl_read_only")))
#define __write_only  __attribute__((annotate("cl_write_only")))
#define write_only    __attribute__((annotate("cl_write_only")))
#define __read_write  __attribute__((annotate("cl_read_write")))
#define read_write    __attribute__((annotate("cl_read_write")))

// Macros
#define __OPENCL_VERSION__      120
#define CL_VERSION_1_0          100
#define CL_VERSION_1_1          110
#define CL_VERSION_1_2          120
#define __OPENCL_C_VERSION__    120
#define __ENDIAN_LITTLE__       1
#define __IMAGE_SUPPORT__       1
#define __kernel_exec(X, typen) __kernel __attribute__((work_group_size_hint(X, 1, 1))) \
												       	__attribute__((vec_type_hint(typen)))
#define kernel_exec(X, typen)   __kernel __attribute__((work_group_size_hint(X, 1, 1))) \
												       	__attribute__((vec_type_hint(typen)))

#include "cl_extensions.h"
#include "cl_types.h"
#include "cl_builtins.h"
#include "cl_kernel_constants.h"

#endif //__CL_KERNEL_H__
