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

#ifndef __CL_TYPES_H__
#define __CL_TYPES_H__

/* bool, true, false */
/* Don't define bool, true, and false in C++, except as a GNU extension. */
#ifndef __cplusplus
#define bool  _Bool
#define true  1
#define false 0
#elif defined(__GNUC__) && !defined(__STRICT_ANSI__)
/* Define _Bool, bool, false, true as a GNU extension. */
#define _Bool bool
#define bool  bool
#define false false
#define true  true
#endif

#define __bool_true_false_are_defined 1

/* size_t, ptrdiff_t */
typedef __typeof__(((int*)0)-((int*)0)) ptrdiff_t;
#ifndef _SIZE_T
#define _SIZE_T
typedef __typeof__(sizeof(int)) size_t;
#endif

/* intptr_t, uintptr_t */
typedef int intptr_t;
typedef unsigned int uintptr_t;

/* vector types */
typedef char            char2    __attribute__((ext_vector_type(2)));
typedef char            char3    __attribute__((ext_vector_type(3)));
typedef char            char4    __attribute__((ext_vector_type(4)));
typedef char            char8    __attribute__((ext_vector_type(8)));
typedef char            char16   __attribute__((ext_vector_type(16)));
typedef unsigned char   uchar2   __attribute__((ext_vector_type(2)));
typedef unsigned char   uchar3   __attribute__((ext_vector_type(3)));
typedef unsigned char   uchar4   __attribute__((ext_vector_type(4)));
typedef unsigned char   uchar8   __attribute__((ext_vector_type(8)));
typedef unsigned char   uchar16  __attribute__((ext_vector_type(16)));

typedef short           short2   __attribute__((ext_vector_type(2)));
typedef short           short3   __attribute__((ext_vector_type(3)));
typedef short           short4   __attribute__((ext_vector_type(4)));
typedef short           short8   __attribute__((ext_vector_type(8)));
typedef short           short16  __attribute__((ext_vector_type(16)));
typedef unsigned short  ushort2  __attribute__((ext_vector_type(2)));
typedef unsigned short  ushort3  __attribute__((ext_vector_type(3)));
typedef unsigned short  ushort4  __attribute__((ext_vector_type(4)));
typedef unsigned short  ushort8  __attribute__((ext_vector_type(8)));
typedef unsigned short  ushort16 __attribute__((ext_vector_type(16)));

typedef int             int2     __attribute__((ext_vector_type(2)));
typedef int             int3     __attribute__((ext_vector_type(3)));
typedef int             int4     __attribute__((ext_vector_type(4)));
typedef int             int8     __attribute__((ext_vector_type(8)));
typedef int             int16    __attribute__((ext_vector_type(16)));
typedef unsigned int    uint2    __attribute__((ext_vector_type(2)));
typedef unsigned int    uint3    __attribute__((ext_vector_type(3)));
typedef unsigned int    uint4    __attribute__((ext_vector_type(4)));
typedef unsigned int    uint8    __attribute__((ext_vector_type(8)));
typedef unsigned int    uint16   __attribute__((ext_vector_type(16)));

typedef long            long2    __attribute__((ext_vector_type(2)));
typedef long            long3    __attribute__((ext_vector_type(3)));
typedef long            long4    __attribute__((ext_vector_type(4)));
typedef long            long8    __attribute__((ext_vector_type(8)));
typedef long            long16   __attribute__((ext_vector_type(16)));
typedef unsigned long   ulong2   __attribute__((ext_vector_type(2)));
typedef unsigned long   ulong3   __attribute__((ext_vector_type(3)));
typedef unsigned long   ulong4   __attribute__((ext_vector_type(4)));
typedef unsigned long   ulong8   __attribute__((ext_vector_type(8)));
typedef unsigned long   ulong16  __attribute__((ext_vector_type(16)));
                                                                  
typedef half            half2   __attribute__((ext_vector_type(2)));
typedef half            half3   __attribute__((ext_vector_type(3)));
typedef half            half4   __attribute__((ext_vector_type(4)));
typedef half            half8   __attribute__((ext_vector_type(8)));
typedef half            half16  __attribute__((ext_vector_type(16)));

typedef float           float2   __attribute__((ext_vector_type(2)));
typedef float           float3   __attribute__((ext_vector_type(3)));
typedef float           float4   __attribute__((ext_vector_type(4)));
typedef float           float8   __attribute__((ext_vector_type(8)));
typedef float           float16  __attribute__((ext_vector_type(16)));

typedef double          double2  __attribute__((ext_vector_type(2)));
typedef double          double3  __attribute__((ext_vector_type(3)));
typedef double          double4  __attribute__((ext_vector_type(4)));
typedef double          double8  __attribute__((ext_vector_type(8)));
typedef double          double16 __attribute__((ext_vector_type(16)));

/* cl_mem_fence_flags */
enum _cl_mem_fence_flags {
  CLK_LOCAL_MEM_FENCE,
  CLK_GLOBAL_MEM_FENCE
};
typedef enum _cl_mem_fence_flags cl_mem_fence_flags;

/* other built-in data types */
typedef char**     image2d_t;
typedef void***    image3d_t;
typedef char**     image2d_array_t;
typedef char**     image1d_t;
typedef char**     image1d_buffer_t;
typedef char**     image1d_array_t;
typedef int        sampler_t;
typedef int        event_t;

/* sampler: normalized coords */
enum _sampler_normalized_coords {
  CLK_NORMALIZED_COORDS_TRUE = 0x0100,
  CLK_NORMALIZED_COORDS_FALSE = 0x0080
};

/* sampler: addressing mode */
enum _sampler_addressing_mode {
  CLK_ADDRESS_CLAMP_TO_EDGE = 0x0010, 
  CLK_ADDRESS_CLAMP = 0x0008, 
  CLK_ADDRESS_NONE = 0x0004, 
  CLK_ADDRESS_REPEAT = 0x0002, 
  CLK_ADDRESS_MIRRORED_REPEAT = 0x0001
};

/* sampler: filter mode */
enum _sampler_filter_mode {
  CLK_FILTER_NEAREST = 0x0040, 
  CLK_FILTER_LINEAR = 0x0020
};

#endif  //__CL_TYPES_H__
