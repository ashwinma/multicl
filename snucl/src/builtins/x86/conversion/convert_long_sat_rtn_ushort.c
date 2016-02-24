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

llong convert_long_sat_rtn(ushort x) {
	return (llong)x;
}

long2 convert_long2_sat_rtn(ushort2 x) {
	return (long2){(llong)x[0], (llong)x[1]};
}

long3 convert_long3_sat_rtn(ushort3 x) {
	return (long3){(llong)x[0], (llong)x[1], (llong)x[2]};
}

long4 convert_long4_sat_rtn(ushort4 x) {
	return (long4){(llong)x[0], (llong)x[1], (llong)x[2], (llong)x[3]};
}

long8 convert_long8_sat_rtn(ushort8 x) {
	return (long8){(llong)x[0], (llong)x[1], (llong)x[2], (llong)x[3], 
	              (llong)x[4], (llong)x[5], (llong)x[6], (llong)x[7]};
}

long16 convert_long16_sat_rtn(ushort16 x) {
	return (long16){(llong)x[0], (llong)x[1], (llong)x[2], (llong)x[3],     
	              (llong)x[4], (llong)x[5], (llong)x[6], (llong)x[7],     
	              (llong)x[8], (llong)x[9], (llong)x[10], (llong)x[11],   
	              (llong)x[12], (llong)x[13], (llong)x[14], (llong)x[15]};
}

