/*****************************************************************************/
/* Copyright (C) 2010, 2011 Seoul National University                        */
/* and Samsung Electronics Co., Ltd.                                         */
/*                                                                           */
/* Contributed by Sangmin Seo <sangmin@aces.snu.ac.kr>, Jungwon Kim          */
/* <jungwon@aces.snu.ac.kr>, Jaejin Lee <jlee@cse.snu.ac.kr>, Seungkyun Kim  */
/* <seungkyun@aces.snu.ac.kr>, Jungho Park <jungho@aces.snu.ac.kr>,          */
/* Honggyu Kim <honggyu@aces.snu.ac.kr>, Jeongho Nah                         */
/* <jeongho@aces.snu.ac.kr>, Sung Jong Seo <sj1557.seo@samsung.com>,         */
/* Seung Hak Lee <s.hak.lee@samsung.com>, Seung Mo Cho                       */
/* <seungm.cho@samsung.com>, Hyo Jung Song <hjsong@samsung.com>,             */
/* Sang-Bum Suh <sbuk.suh@samsung.com>, and Jong-Deok Choi                   */
/* <jd11.choi@samsung.com>                                                   */
/*                                                                           */
/* All rights reserved.                                                      */
/*                                                                           */
/* This file is part of the SNU-SAMSUNG OpenCL runtime.                      */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is free software: you can redistribute it  */
/* and/or modify it under the terms of the GNU Lesser General Public License */
/* as published by the Free Software Foundation, either version 3 of the     */
/* License, or (at your option) any later version.                           */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is distributed in the hope that it will be */
/* useful, but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  */
/* Public License for more details.                                          */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with the SNU-SAMSUNG OpenCL runtime. If not, see                    */
/* <http://www.gnu.org/licenses/>.                                           */
/*****************************************************************************/

#ifndef __CL_VECTOR_H
#define __CL_VECTOR_H

/* vector type definitions using struct */
#include "cl_vector_charn.h"
#include "cl_vector_ucharn.h"
#include "cl_vector_shortn.h"
#include "cl_vector_ushortn.h"
#include "cl_vector_intn.h"
#include "cl_vector_uintn.h"
#include "cl_vector_longn.h"
#include "cl_vector_ulongn.h"
#include "cl_vector_floatn.h"
#include "cl_vector_doublen.h"
#include "cl_vector_ldoublen.h"

/* vector types */
typedef struct __cl_char2       char2;
typedef struct __cl_char3       char3;
typedef struct __cl_char4       char4;
typedef struct __cl_char8       char8;
typedef struct __cl_char16      char16;
typedef struct __cl_uchar2      uchar2;
typedef struct __cl_uchar3      uchar3;
typedef struct __cl_uchar4      uchar4;
typedef struct __cl_uchar8      uchar8;
typedef struct __cl_uchar16     uchar16;
typedef struct __cl_short2      short2;
typedef struct __cl_short3      short3;
typedef struct __cl_short4      short4;
typedef struct __cl_short8      short8;
typedef struct __cl_short16     short16;
typedef struct __cl_ushort2     ushort2;
typedef struct __cl_ushort3     ushort3;
typedef struct __cl_ushort4     ushort4;
typedef struct __cl_ushort8     ushort8;
typedef struct __cl_ushort16    ushort16;
typedef struct __cl_int2        int2;
typedef struct __cl_int3        int3;
typedef struct __cl_int4        int4;
typedef struct __cl_int8        int8;
typedef struct __cl_int16       int16;
typedef struct __cl_uint2       uint2;
typedef struct __cl_uint3       uint3;
typedef struct __cl_uint4       uint4;
typedef struct __cl_uint8       uint8;
typedef struct __cl_uint16      uint16;
typedef struct __cl_long2       long2;
typedef struct __cl_long3       long3;
typedef struct __cl_long4       long4;
typedef struct __cl_long8       long8;
typedef struct __cl_long16      long16;
typedef struct __cl_ulong2      ulong2;
typedef struct __cl_ulong3      ulong3;
typedef struct __cl_ulong4      ulong4;
typedef struct __cl_ulong8      ulong8;
typedef struct __cl_ulong16     ulong16;
typedef struct __cl_float2      float2;
typedef struct __cl_float3      float3;
typedef struct __cl_float4      float4;
typedef struct __cl_float8      float8;
typedef struct __cl_float16     float16;
typedef struct __cl_double2     double2;
typedef struct __cl_double3     double3;
typedef struct __cl_double4     double4;
typedef struct __cl_double8     double8;
typedef struct __cl_double16    double16;
typedef struct __cl_ldouble2    ldouble2;
typedef struct __cl_ldouble3    ldouble3;
typedef struct __cl_ldouble4    ldouble4;
typedef struct __cl_ldouble8    ldouble8;
typedef struct __cl_ldouble16   ldouble16;

#endif //__CL_VECTOR_H

