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

double snu_nan(ullong x);

float nan(uint x) {
	return snu_nan(x);
}

float2 nan(uint2 x) {
	float2 ret;
	ret[0] = snu_nan(x[0]);
	ret[1] = snu_nan(x[1]);
	return ret;
}

float3 nan(uint3 x) {
	float3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;

}
float4 nan(uint4 x) {
	float4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;
}

float8 nan(uint8 x) {
	float8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;
}

float16 nan(uint16 x) {
	float16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;
}

//double nan(uint x, int dummy) {
//  dummy += 0;
//	return snu_nan(x);
//}
// //double2 nan(uint2 x, int dummy) {
//	double2 ret;
//	ret[0] = snu_nan(x[0]);
//	ret[1] = snu_nan(x[1]);
//  dummy += 0;
//	return ret;
//}
//
//double4 nan(uint4 x, int dummy) {
//	double4 ret;
//	for(int i=0;i<4;i++) {
//		ret[i] = snu_nan(x[i]);
//  }
//  dummy += 0;
//	return ret;
//}
//
//double8 nan(uint8 x, int dummy) {
//	double8 ret;
//	for(int i=0;i<8;i++) {
//		ret[i] = snu_nan(x[i]);
//  }
//  dummy += 0;
//	return ret;
//}
//
//double16 nan(uint16 x, int dummy) {
//	double16 ret;
//	for(int i=0;i<16;i++) {
//		ret[i] = snu_nan(x[i]);
//  }
//  dummy += 0;
//	return ret;
//}

double nan(ullong x) {
	return snu_nan(x);
}

double2 nan(ulong2 x) {
	double2 ret;
	ret[0] = snu_nan(x[0]);
	ret[1] = snu_nan(x[1]);
	return ret;
}

double3 nan(ulong3 x) {
	double3 ret;
	for(int i=0;i<3;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;
}

double4 nan(ulong4 x) {
	double4 ret;
	for(int i=0;i<4;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;
}

double8 nan(ulong8 x) {
	double8 ret;
	for(int i=0;i<8;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;
}

double16 nan(ulong16 x) {
	double16 ret;
	for(int i=0;i<16;i++) {
		ret[i] = snu_nan(x[i]);
  }
	return ret;
}
