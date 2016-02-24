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

ushort convert_ushort_rte(schar x) {
	return (ushort)x;
}

ushort2 convert_ushort2_rte(char2 x) {
	return (ushort2){(ushort)x[0], (ushort)x[1]};
}

ushort3 convert_ushort3_rte(char3 x) {
	return (ushort3){(ushort)x[0], (ushort)x[1], (ushort)x[2]};
}

ushort4 convert_ushort4_rte(char4 x) {
	return (ushort4){(ushort)x[0], (ushort)x[1], (ushort)x[2], (ushort)x[3]};
}

ushort8 convert_ushort8_rte(char8 x) {
	return (ushort8){(ushort)x[0], (ushort)x[1], (ushort)x[2], (ushort)x[3], 
	              (ushort)x[4], (ushort)x[5], (ushort)x[6], (ushort)x[7]};
}

ushort16 convert_ushort16_rte(char16 x) {
	return (ushort16){(ushort)x[0], (ushort)x[1], (ushort)x[2], (ushort)x[3],     
	              (ushort)x[4], (ushort)x[5], (ushort)x[6], (ushort)x[7],     
	              (ushort)x[8], (ushort)x[9], (ushort)x[10], (ushort)x[11],   
	              (ushort)x[12], (ushort)x[13], (ushort)x[14], (ushort)x[15]};
}

