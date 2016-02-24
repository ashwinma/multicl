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

double snu_lgamma_r(double x, int *signp);

float lgamma_r(float x, int *signp) {
	return snu_lgamma_r(x, signp);
}

float2 lgamma_r(float2 x, int2 *signp) {
	float2 ret;
	ret[0] = snu_lgamma_r(x[0], &(signp->v[0]));
	ret[1] = snu_lgamma_r(x[1], &(signp->v[1]));
	return ret;
}

float3 lgamma_r(float3 x, int3 *signp) {
	float3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}
float4 lgamma_r(float4 x, int4 *signp) {
	float4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}

float8 lgamma_r(float8 x, int8 *signp) {
	float8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}

float16 lgamma_r(float16 x, int16 *signp) {
	float16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}

double lgamma_r(double x, int *signp) {
	return snu_lgamma_r(x, signp);
}

double2 lgamma_r(double2 x, int2 *signp) {
	double2 ret;
	ret[0] = snu_lgamma_r(x[0], &(signp->v[0]));
	ret[1] = snu_lgamma_r(x[1], &(signp->v[1]));
	return ret;
}

double3 lgamma_r(double3 x, int3 *signp) {
	double3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}

double4 lgamma_r(double4 x, int4 *signp) {
	double4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}

double8 lgamma_r(double8 x, int8 *signp) {
	double8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}

double16 lgamma_r(double16 x, int16 *signp) {
	double16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_lgamma_r(x[i], &(signp->v[i]));
  }
	return ret;
}
