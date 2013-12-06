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

schar convert_char_rtn(uint x) {
	return (schar)x;
}

char2 convert_char2_rtn(uint2 x) {
	return (char2){(schar)x[0], (schar)x[1]};
}

char3 convert_char3_rtn(uint3 x) {
	return (char3){(schar)x[0], (schar)x[1], (schar)x[2]};
}

char4 convert_char4_rtn(uint4 x) {
	return (char4){(schar)x[0], (schar)x[1], (schar)x[2], (schar)x[3]};
}

char8 convert_char8_rtn(uint8 x) {
	return (char8){(schar)x[0], (schar)x[1], (schar)x[2], (schar)x[3], 
	              (schar)x[4], (schar)x[5], (schar)x[6], (schar)x[7]};
}

char16 convert_char16_rtn(uint16 x) {
	return (char16){(schar)x[0], (schar)x[1], (schar)x[2], (schar)x[3],     
	              (schar)x[4], (schar)x[5], (schar)x[6], (schar)x[7],     
	              (schar)x[8], (schar)x[9], (schar)x[10], (schar)x[11],   
	              (schar)x[12], (schar)x[13], (schar)x[14], (schar)x[15]};
}

