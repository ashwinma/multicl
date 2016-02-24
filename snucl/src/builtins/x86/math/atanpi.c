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
#include "math_util.h"

float atanpi(float x) {
	return (atanf(x)/M_PI);
}

float2 atanpi(float2 x) {
	float2 ret;
	ret[0] = (atanf(x[0])/M_PI);
	ret[1] = (atanf(x[1])/M_PI);
	return ret;
}

float3 atanpi(float3 x) {
	float3 ret;
	for(int i=0;i<3;i++) {
		ret[i] =  (atanf(x[i])/M_PI);
  }
	return ret;
}
float4 atanpi(float4 x) {
	float4 ret;
	for(int i=0;i<4;i++) {
		ret[i] =  (atanf(x[i])/M_PI);
  }
	return ret;
}

float8 atanpi(float8 x) {
	float8 ret;
	for(int i=0;i<8;i++) {
		ret[i] =  (atanf(x[i])/M_PI);
  }
	return ret;
}

float16 atanpi(float16 x) {
	float16 ret;
	for(int i=0;i<16;i++) {
		ret[i] =  (atanf(x[i])/M_PI);
  }
	return ret;
}

double atanpi(double x) {
	return (atan(x)/M_PI);
}

double2 atanpi(double2 x) {
	double2 ret;
	ret[0] = (atan(x[0])/M_PI);
	ret[1] = (atan(x[1])/M_PI);
	return ret;
}

double3 atanpi(double3 x) {
	double3 ret;
	for(int i=0;i<3;i++) {
		ret[i] =  (atan(x[i])/M_PI);
  }
	return ret;
}


double4 atanpi(double4 x) {
	double4 ret;
	for(int i=0;i<4;i++) {
		ret[i] =  (atan(x[i])/M_PI);
  }
	return ret;
}

double8 atanpi(double8 x) {
	double8 ret;
	for(int i=0;i<8;i++) {
		ret[i] =  (atan(x[i])/M_PI);
  }
	return ret;
}

double16 atanpi(double16 x) {
	double16 ret;
	for(int i=0;i<16;i++) {
		ret[i] =  (atan(x[i])/M_PI);
  }
	return ret;
}
