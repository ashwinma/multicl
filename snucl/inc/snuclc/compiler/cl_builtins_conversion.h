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

#ifndef __CL_BUILTINS_CONVERSION_H__
#define __CL_BUILTINS_CONVERSION_H__

/* 6.2.3 Explicit Conversions */
#include "cl_builtins_convert_bool.h"
#include "cl_builtins_convert_char.h"
#include "cl_builtins_convert_uchar.h"
#include "cl_builtins_convert_short.h"
#include "cl_builtins_convert_ushort.h"
#include "cl_builtins_convert_int.h"
#include "cl_builtins_convert_uint.h"
#include "cl_builtins_convert_long.h"
#include "cl_builtins_convert_ulong.h"
#include "cl_builtins_convert_double.h"

/* vector conversions */
#include "cl_builtins_convert_char2.h"
#include "cl_builtins_convert_char3.h"
#include "cl_builtins_convert_char4.h"
#include "cl_builtins_convert_char8.h"
#include "cl_builtins_convert_char16.h"
#include "cl_builtins_convert_uchar2.h"
#include "cl_builtins_convert_uchar3.h"
#include "cl_builtins_convert_uchar4.h"
#include "cl_builtins_convert_uchar8.h"
#include "cl_builtins_convert_uchar16.h"
#include "cl_builtins_convert_short2.h"
#include "cl_builtins_convert_short3.h"
#include "cl_builtins_convert_short4.h"
#include "cl_builtins_convert_short8.h"
#include "cl_builtins_convert_short16.h"
#include "cl_builtins_convert_ushort2.h"
#include "cl_builtins_convert_ushort3.h"
#include "cl_builtins_convert_ushort4.h"
#include "cl_builtins_convert_ushort8.h"
#include "cl_builtins_convert_ushort16.h"
#include "cl_builtins_convert_int2.h"
#include "cl_builtins_convert_int3.h"
#include "cl_builtins_convert_int4.h"
#include "cl_builtins_convert_int8.h"
#include "cl_builtins_convert_int16.h"
#include "cl_builtins_convert_uint2.h"
#include "cl_builtins_convert_uint3.h"
#include "cl_builtins_convert_uint4.h"
#include "cl_builtins_convert_uint8.h"
#include "cl_builtins_convert_uint16.h"
#include "cl_builtins_convert_long2.h"
#include "cl_builtins_convert_long3.h"
#include "cl_builtins_convert_long4.h"
#include "cl_builtins_convert_long8.h"
#include "cl_builtins_convert_long16.h"
#include "cl_builtins_convert_ulong2.h"
#include "cl_builtins_convert_ulong3.h"
#include "cl_builtins_convert_ulong4.h"
#include "cl_builtins_convert_ulong8.h"
#include "cl_builtins_convert_ulong16.h"
#include "cl_builtins_convert_float2.h"
#include "cl_builtins_convert_float3.h"
#include "cl_builtins_convert_float4.h"
#include "cl_builtins_convert_float8.h"
#include "cl_builtins_convert_float16.h"
#include "cl_builtins_convert_double2.h"
#include "cl_builtins_convert_double3.h"
#include "cl_builtins_convert_double4.h"
#include "cl_builtins_convert_double8.h"
#include "cl_builtins_convert_double16.h"

#endif //__CL_BUILTINS_CONVERSION_H__
