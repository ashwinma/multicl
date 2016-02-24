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

int convert_int_rtz(ullong x) {
	return (int)x;
}

int2 convert_int2_rtz(ulong2 x) {
	return (int2){(int)x[0], (int)x[1]};
}

int3 convert_int3_rtz(ulong3 x) {
	return (int3){(int)x[0], (int)x[1], (int)x[2]};
}

int4 convert_int4_rtz(ulong4 x) {
	return (int4){(int)x[0], (int)x[1], (int)x[2], (int)x[3]};
}

int8 convert_int8_rtz(ulong8 x) {
	return (int8){(int)x[0], (int)x[1], (int)x[2], (int)x[3], 
	              (int)x[4], (int)x[5], (int)x[6], (int)x[7]};
}

int16 convert_int16_rtz(ulong16 x) {
	return (int16){(int)x[0], (int)x[1], (int)x[2], (int)x[3],     
	              (int)x[4], (int)x[5], (int)x[6], (int)x[7],     
	              (int)x[8], (int)x[9], (int)x[10], (int)x[11],   
	              (int)x[12], (int)x[13], (int)x[14], (int)x[15]};
}

