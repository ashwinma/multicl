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

short convert_short(uchar x) {
	return (short)x;
}

short2 convert_short2(uchar2 x) {
	return (short2){(short)x[0], (short)x[1]};
}

short3 convert_short3(uchar3 x) {
	return (short3){(short)x[0], (short)x[1], (short)x[2]};
}

short4 convert_short4(uchar4 x) {
	return (short4){(short)x[0], (short)x[1], (short)x[2], (short)x[3]};
}

short8 convert_short8(uchar8 x) {
	return (short8){(short)x[0], (short)x[1], (short)x[2], (short)x[3], 
	              (short)x[4], (short)x[5], (short)x[6], (short)x[7]};
}

short16 convert_short16(uchar16 x) {
	return (short16){(short)x[0], (short)x[1], (short)x[2], (short)x[3],     
	              (short)x[4], (short)x[5], (short)x[6], (short)x[7],     
	              (short)x[8], (short)x[9], (short)x[10], (short)x[11],   
	              (short)x[12], (short)x[13], (short)x[14], (short)x[15]};
}

