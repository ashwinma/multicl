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

int snu_isgreaterequal(double x, double y) {
  return x>=y;
} 
int snu_isgreaterequalv(double x, double y) {
  return x>=y? -1 : 0; 
} 

int isgreaterequal(float x, float y) {
	return snu_isgreaterequal(x, y);
}

int2 isgreaterequal(float2 x, float2 y) {
	int2 ret;
	ret[0] = snu_isgreaterequalv(x[0], y[0]);
	ret[1] = snu_isgreaterequalv(x[1], y[1]);
	return ret;
}

int3 isgreaterequal(float3 x, float3 y) {
	int3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}

int4 isgreaterequal(float4 x, float4 y) {
	int4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}

int8 isgreaterequal(float8 x, float8 y) {
	int8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}

int16 isgreaterequal(float16 x, float16 y) {
	int16 ret;
	for(int i=0;i<16;i++) {
    ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}

int isgreaterequal(double x, double y) {
	return snu_isgreaterequal(x, y);
}

long2 isgreaterequal(double2 x, double2 y) {
	long2 ret;
	ret[0] = snu_isgreaterequalv(x[0], y[0]);
	ret[1] = snu_isgreaterequalv(x[1], y[1]);
	return ret;
}

long3 isgreaterequal(double3 x, double3 y) {
	long3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}


long4 isgreaterequal(double4 x, double4 y) {
	long4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}

long8 isgreaterequal(double8 x, double8 y) {
	long8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}

long16 isgreaterequal(double16 x, double16 y) {
	long16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_isgreaterequalv(x[i], y[i]);
  }
	return ret;
}
