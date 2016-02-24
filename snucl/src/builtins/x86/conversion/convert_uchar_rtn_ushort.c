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

uchar convert_uchar_rtn(ushort x) {
	return (uchar)x;
}

uchar2 convert_uchar2_rtn(ushort2 x) {
	return (uchar2){(uchar)x[0], (uchar)x[1]};
}

uchar3 convert_uchar3_rtn(ushort3 x) {
	return (uchar3){(uchar)x[0], (uchar)x[1], (uchar)x[2]};
}

uchar4 convert_uchar4_rtn(ushort4 x) {
	return (uchar4){(uchar)x[0], (uchar)x[1], (uchar)x[2], (uchar)x[3]};
}

uchar8 convert_uchar8_rtn(ushort8 x) {
	return (uchar8){(uchar)x[0], (uchar)x[1], (uchar)x[2], (uchar)x[3], 
	              (uchar)x[4], (uchar)x[5], (uchar)x[6], (uchar)x[7]};
}

uchar16 convert_uchar16_rtn(ushort16 x) {
	return (uchar16){(uchar)x[0], (uchar)x[1], (uchar)x[2], (uchar)x[3],     
	              (uchar)x[4], (uchar)x[5], (uchar)x[6], (uchar)x[7],     
	              (uchar)x[8], (uchar)x[9], (uchar)x[10], (uchar)x[11],   
	              (uchar)x[12], (uchar)x[13], (uchar)x[14], (uchar)x[15]};
}

