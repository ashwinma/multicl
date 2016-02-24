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
#include "convert_util.h"

uint convert_uint(ullong x) {
	return (uint)x;
}

uint2 convert_uint2(ulong2 x) {
	return (uint2){(uint)x[0], (uint)x[1]};
}

uint3 convert_uint3(ulong3 x) {
	return (uint3){(uint)x[0], (uint)x[1], (uint)x[2]};
}

uint4 convert_uint4(ulong4 x) {
	return (uint4){(uint)x[0], (uint)x[1], (uint)x[2], (uint)x[3]};
}

uint8 convert_uint8(ulong8 x) {
	return (uint8){(uint)x[0], (uint)x[1], (uint)x[2], (uint)x[3], 
	              (uint)x[4], (uint)x[5], (uint)x[6], (uint)x[7]};
}

uint16 convert_uint16(ulong16 x) {
	return (uint16){(uint)x[0], (uint)x[1], (uint)x[2], (uint)x[3],     
	              (uint)x[4], (uint)x[5], (uint)x[6], (uint)x[7],     
	              (uint)x[8], (uint)x[9], (uint)x[10], (uint)x[11],   
	              (uint)x[12], (uint)x[13], (uint)x[14], (uint)x[15]};
}

