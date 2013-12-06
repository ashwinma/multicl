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

double snu_lgamma(double x);

float lgamma(float x) {
	return snu_lgamma(x);
}

float2 lgamma(float2 x) {
	float2 ret;
	ret[0] = snu_lgamma(x[0]);
	ret[1] = snu_lgamma(x[1]);
	return ret;
}

float3 lgamma(float3 x) {
	float3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}

float4 lgamma(float4 x) {
	float4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}

float8 lgamma(float8 x) {
	float8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}

float16 lgamma(float16 x) {
	float16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}

double lgamma(double x) {
	return snu_lgamma(x);
}

double2 lgamma(double2 x) {
	double2 ret;
	ret[0] = snu_lgamma(x[0]);
	ret[1] = snu_lgamma(x[1]);
	return ret;
}

double3 lgamma(double3 x) {
	double3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}

double4 lgamma(double4 x) {
	double4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}

double8 lgamma(double8 x) {
	double8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}

double16 lgamma(double16 x) {
	double16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_lgamma(x[i]);
  }
	return ret;
}
