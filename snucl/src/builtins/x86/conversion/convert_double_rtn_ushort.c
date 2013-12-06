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

double convert_double_rtn(ushort x) {
	return (double)x;
}

double2 convert_double2_rtn(ushort2 x) {
	return (double2){(double)x[0], (double)x[1]};
}

double3 convert_double3_rtn(ushort3 x) {
	return (double3){(double)x[0], (double)x[1], (double)x[2]};
}

double4 convert_double4_rtn(ushort4 x) {
	return (double4){(double)x[0], (double)x[1], (double)x[2], (double)x[3]};
}

double8 convert_double8_rtn(ushort8 x) {
	return (double8){(double)x[0], (double)x[1], (double)x[2], (double)x[3], 
	              (double)x[4], (double)x[5], (double)x[6], (double)x[7]};
}

double16 convert_double16_rtn(ushort16 x) {
	return (double16){(double)x[0], (double)x[1], (double)x[2], (double)x[3],     
	              (double)x[4], (double)x[5], (double)x[6], (double)x[7],     
	              (double)x[8], (double)x[9], (double)x[10], (double)x[11],   
	              (double)x[12], (double)x[13], (double)x[14], (double)x[15]};
}

