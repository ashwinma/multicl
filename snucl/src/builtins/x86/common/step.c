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
#include "step.h"

float step(float edge, float x){
	return x<edge? 0.0:1.0;
}

float2 step(float2 edge, float2 x){
	float2 ret;
	ret.v[0] = x.v[0]<edge.v[0]? 0.0:1.0;
	ret.v[1] = x.v[1]<edge.v[1]? 0.0:1.0;
	return ret;
}

float3 step(float3 edge, float3 x){
	float3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

float4 step(float4 edge, float4 x){
	float4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

float8 step(float8 edge, float8 x){
	float8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

float16 step(float16 edge, float16 x){
	float16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

float2 step(float edge, float2 x){
	float2 ret;
	ret.v[0] = x.v[0]<edge? 0.0:1.0;
	ret.v[1] = x.v[1]<edge? 0.0:1.0;
	return ret;
}

float3 step(float edge, float3 x){
	float3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}

float4 step(float edge, float4 x){
	float4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}

float8 step(float edge, float8 x){
	float8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}

float16 step(float edge, float16 x){
	float16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}

double step(double edge, double x){
	return x<edge? 0.0:1.0;
}

double2 step(double2 edge, double2 x){
	double2 ret;
	ret.v[0] = x.v[0]<edge.v[0]? 0.0:1.0;
	ret.v[1] = x.v[1]<edge.v[1]? 0.0:1.0;
	return ret;
}

double3 step(double3 edge, double3 x){
	double3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

double4 step(double4 edge, double4 x){
	double4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

double8 step(double8 edge, double8 x){
	double8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

double16 step(double16 edge, double16 x){
	double16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = x.v[i]<edge.v[i]? 0.0:1.0;
  }
	return ret;
}

double2 step(double edge, double2 x){
	double2 ret;
	ret.v[0] = x.v[0]<edge? 0.0:1.0;
	ret.v[1] = x.v[1]<edge? 0.0:1.0;
	return ret;
}

double3 step(double edge, double3 x){
	double3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}

double4 step(double edge, double4 x){
	double4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}

double8 step(double edge, double8 x){
	double8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}

double16 step(double edge, double16 x){
	double16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = x.v[i]<edge? 0.0:1.0;
  }
	return ret;
}
