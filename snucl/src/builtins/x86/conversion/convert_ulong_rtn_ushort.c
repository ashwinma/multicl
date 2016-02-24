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

ullong convert_ulong_rtn(ushort x) {
	return (ullong)x;
}

ulong2 convert_ulong2_rtn(ushort2 x) {
	return (ulong2){(ullong)x[0], (ullong)x[1]};
}

ulong3 convert_ulong3_rtn(ushort3 x) {
	return (ulong3){(ullong)x[0], (ullong)x[1], (ullong)x[2]};
}

ulong4 convert_ulong4_rtn(ushort4 x) {
	return (ulong4){(ullong)x[0], (ullong)x[1], (ullong)x[2], (ullong)x[3]};
}

ulong8 convert_ulong8_rtn(ushort8 x) {
	return (ulong8){(ullong)x[0], (ullong)x[1], (ullong)x[2], (ullong)x[3], 
	              (ullong)x[4], (ullong)x[5], (ullong)x[6], (ullong)x[7]};
}

ulong16 convert_ulong16_rtn(ushort16 x) {
	return (ulong16){(ullong)x[0], (ullong)x[1], (ullong)x[2], (ullong)x[3],     
	              (ullong)x[4], (ullong)x[5], (ullong)x[6], (ullong)x[7],     
	              (ullong)x[8], (ullong)x[9], (ullong)x[10], (ullong)x[11],   
	              (ullong)x[12], (ullong)x[13], (ullong)x[14], (ullong)x[15]};
}

