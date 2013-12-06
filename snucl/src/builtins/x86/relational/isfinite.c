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

int snu_isfinite(double x);
int snu_isfinitev(double x);

int isfinite(float x) {
	return snu_isfinite(x);
}

int2 isfinite(float2 x) {
	int2 ret;
	ret[0] = snu_isfinitev(x[0]);
	ret[1] = snu_isfinitev(x[1]);
	return ret;
}

int3 isfinite(float3 x) {
	int3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}

int4 isfinite(float4 x) {
	int4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}

int8 isfinite(float8 x) {
	int8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}

int16 isfinite(float16 x) {
	int16 ret;
	for(int i=0;i<16;i++) {
    ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}

int isfinite(double x) {
	return snu_isfinite(x);
}

long2 isfinite(double2 x) {
	long2 ret;
	ret[0] = snu_isfinitev(x[0]);
	ret[1] = snu_isfinitev(x[1]);
	return ret;
}

long3 isfinite(double3 x) {
	long3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}

long4 isfinite(double4 x) {
	long4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}

long8 isfinite(double8 x) {
	long8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}

long16 isfinite(double16 x) {
	long16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_isfinitev(x[i]);
  }
	return ret;
}
