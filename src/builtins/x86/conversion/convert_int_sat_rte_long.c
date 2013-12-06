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

int convert_int_sat_rte(llong x) {
	return (int)(x < INT_MIN ? INT_MIN : (x > INT_MAX ? INT_MAX : x));
}

int2 convert_int2_sat_rte(long2 x) {
	int2 ret;                                                                         
	for (int i = 0; i < 2; i++)                                                       
		ret[i] = (int)(x[i] < INT_MIN ? INT_MIN : (x[i] > INT_MAX ? INT_MAX : x[i]));
	return ret;                                                                       
}

int3 convert_int3_sat_rte(long3 x) {
	int3 ret;                                                                         
	for (int i = 0; i < 3; i++)                                                       
		ret[i] = (int)(x[i] < INT_MIN ? INT_MIN : (x[i] > INT_MAX ? INT_MAX : x[i]));
	return ret;                                                                       
}

int4 convert_int4_sat_rte(long4 x) {
	int4 ret;                                                                         
	for (int i = 0; i < 4; i++)                                                       
		ret[i] = (int)(x[i] < INT_MIN ? INT_MIN : (x[i] > INT_MAX ? INT_MAX : x[i]));
	return ret;                                                                       
}

int8 convert_int8_sat_rte(long8 x) {
	int8 ret;                                                                         
	for (int i = 0; i < 8; i++)                                                       
		ret[i] = (int)(x[i] < INT_MIN ? INT_MIN : (x[i] > INT_MAX ? INT_MAX : x[i]));
	return ret;                                                                       
}

int16 convert_int16_sat_rte(long16 x) {
	int16 ret;                                                                         
	for (int i = 0; i < 16; i++)                                                      
		ret[i] = (int)(x[i] < INT_MIN ? INT_MIN : (x[i] > INT_MAX ? INT_MAX : x[i]));
	return ret;                                                                       
}

