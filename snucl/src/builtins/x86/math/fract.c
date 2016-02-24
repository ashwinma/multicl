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

float snu_fractf(float x, float *ip);
double snu_fract(double x, double *ip);

float fract(float x, float *iptr) {
	return snu_fractf(x, iptr);
}

float2 fract(float2 x, float2 *iptr) {
	float2 ret;
	ret[0] = snu_fractf(x[0], &(iptr->v[0]));
	ret[1] = snu_fractf(x[1], &(iptr->v[1]));
	return ret;
}

float3 fract(float3 x, float3 *iptr) {
	float3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_fractf(x[i], &(iptr->v[i]));
  }
	return ret;
}


float4 fract(float4 x, float4 *iptr) {
	float4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_fractf(x[i], &(iptr->v[i]));
  }
	return ret;
}

float8 fract(float8 x, float8 *iptr) {
	float8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_fractf(x[i], &(iptr->v[i]));
  }
	return ret;
}

float16 fract(float16 x, float16 *iptr) {
	float16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_fractf(x[i], &(iptr->v[i]));
  }
	return ret;
}

double fract(double x, double *iptr) {
	return snu_fract(x, iptr);
}

double2 fract(double2 x, double2 *iptr) {
	double2 ret;
	ret[0] = snu_fract(x[0], &(iptr->v[0]));
	ret[1] = snu_fract(x[1], &(iptr->v[1]));
	return ret;
}

double3 fract(double3 x, double3 *iptr) {
	double3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_fract(x[i], &(iptr->v[i]));
  }
	return ret;
}

double4 fract(double4 x, double4 *iptr) {
	double4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_fract(x[i], &(iptr->v[i]));
  }
	return ret;
}

double8 fract(double8 x, double8 *iptr) {
	double8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_fract(x[i], &(iptr->v[i]));
  }
	return ret;
}

double16 fract(double16 x, double16 *iptr) {
	double16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_fract(x[i], &(iptr->v[i]));
  }
	return ret;
}
