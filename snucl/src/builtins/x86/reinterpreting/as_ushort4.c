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

#include <cl_cpu_ops.h>
#include "as_type_util.h"

SNUCL_AS_TYPE(ushort4, ushort4, schar);
SNUCL_AS_TYPE(ushort4, ushort4, char2);
SNUCL_AS_TYPE(ushort4, ushort4, char3);
SNUCL_AS_TYPE(ushort4, ushort4, char4);
SNUCL_AS_TYPE(ushort4, ushort4, char8);
SNUCL_AS_TYPE(ushort4, ushort4, char16);
SNUCL_AS_TYPE(ushort4, ushort4, uchar);
SNUCL_AS_TYPE(ushort4, ushort4, uchar2);
SNUCL_AS_TYPE(ushort4, ushort4, uchar3);
SNUCL_AS_TYPE(ushort4, ushort4, uchar4);
SNUCL_AS_TYPE(ushort4, ushort4, uchar8);
SNUCL_AS_TYPE(ushort4, ushort4, uchar16);
SNUCL_AS_TYPE(ushort4, ushort4, short);
SNUCL_AS_TYPE(ushort4, ushort4, short2);
SNUCL_AS_TYPE(ushort4, ushort4, short3);
SNUCL_AS_TYPE(ushort4, ushort4, short4);
SNUCL_AS_TYPE(ushort4, ushort4, short8);
SNUCL_AS_TYPE(ushort4, ushort4, short16);
SNUCL_AS_TYPE(ushort4, ushort4, ushort);
SNUCL_AS_TYPE(ushort4, ushort4, ushort2);
SNUCL_AS_TYPE(ushort4, ushort4, ushort3);
SNUCL_AS_TYPE(ushort4, ushort4, ushort4);
SNUCL_AS_TYPE(ushort4, ushort4, ushort8);
SNUCL_AS_TYPE(ushort4, ushort4, ushort16);
SNUCL_AS_TYPE(ushort4, ushort4, int);
SNUCL_AS_TYPE(ushort4, ushort4, int2);
SNUCL_AS_TYPE(ushort4, ushort4, int3);
SNUCL_AS_TYPE(ushort4, ushort4, int4);
SNUCL_AS_TYPE(ushort4, ushort4, int8);
SNUCL_AS_TYPE(ushort4, ushort4, int16);
SNUCL_AS_TYPE(ushort4, ushort4, uint);
SNUCL_AS_TYPE(ushort4, ushort4, uint2);
SNUCL_AS_TYPE(ushort4, ushort4, uint3);
SNUCL_AS_TYPE(ushort4, ushort4, uint4);
SNUCL_AS_TYPE(ushort4, ushort4, uint8);
SNUCL_AS_TYPE(ushort4, ushort4, uint16);
SNUCL_AS_TYPE(ushort4, ushort4, llong);
SNUCL_AS_TYPE(ushort4, ushort4, long2);
SNUCL_AS_TYPE(ushort4, ushort4, long3);
SNUCL_AS_TYPE(ushort4, ushort4, long4);
SNUCL_AS_TYPE(ushort4, ushort4, long8);
SNUCL_AS_TYPE(ushort4, ushort4, long16);
SNUCL_AS_TYPE(ushort4, ushort4, ullong);
SNUCL_AS_TYPE(ushort4, ushort4, ulong2);
SNUCL_AS_TYPE(ushort4, ushort4, ulong3);
SNUCL_AS_TYPE(ushort4, ushort4, ulong4);
SNUCL_AS_TYPE(ushort4, ushort4, ulong8);
SNUCL_AS_TYPE(ushort4, ushort4, ulong16);
SNUCL_AS_TYPE(ushort4, ushort4, float);
SNUCL_AS_TYPE(ushort4, ushort4, float2);
SNUCL_AS_TYPE(ushort4, ushort4, float3);
SNUCL_AS_TYPE(ushort4, ushort4, float4);
SNUCL_AS_TYPE(ushort4, ushort4, float8);
SNUCL_AS_TYPE(ushort4, ushort4, float16);
SNUCL_AS_TYPE(ushort4, ushort4, double);
SNUCL_AS_TYPE(ushort4, ushort4, double2);
SNUCL_AS_TYPE(ushort4, ushort4, double3);
SNUCL_AS_TYPE(ushort4, ushort4, double4);
SNUCL_AS_TYPE(ushort4, ushort4, double8);
SNUCL_AS_TYPE(ushort4, ushort4, double16);

