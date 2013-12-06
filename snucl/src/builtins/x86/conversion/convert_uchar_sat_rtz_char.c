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

uchar convert_uchar_sat_rtz(schar x) {
	return (uchar)(x < 0 ? 0 : x);
}

uchar2 convert_uchar2_sat_rtz(char2 x) {
	uchar2 ret;                                              
	for (int i = 0; i < 2; i++)                            
		ret[i] = (uchar)(x[i] < 0 ? 0 : x[i]);
	return ret;                                            
}

uchar3 convert_uchar3_sat_rtz(char3 x) {
	uchar3 ret;                                              
	for (int i = 0; i < 3; i++)                            
		ret[i] = (uchar)(x[i] < 0 ? 0 : x[i]);
	return ret;                                            
}

uchar4 convert_uchar4_sat_rtz(char4 x) {
	uchar4 ret;                                              
	for (int i = 0; i < 4; i++)                            
		ret[i] = (uchar)(x[i] < 0 ? 0 : x[i]);
	return ret;                                            
}

uchar8 convert_uchar8_sat_rtz(char8 x) {
	uchar8 ret;                                              
	for (int i = 0; i < 8; i++)                            
		ret[i] = (uchar)(x[i] < 0 ? 0 : x[i]);
	return ret;                                            
}

uchar16 convert_uchar16_sat_rtz(char16 x) {
	uchar16 ret;                                              
	for (int i = 0; i < 16; i++)                           
		ret[i] = (uchar)(x[i] < 0 ? 0 : x[i]);
	return ret;                                            
}

