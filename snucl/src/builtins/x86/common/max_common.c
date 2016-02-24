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
#include "max_common.h"

inline static float _maxf(float x, float y){
  return x<y ? y:x;
}

float max(float x, float y){
	return _maxf(x, y);
}
float2 max(float2 x, float2 y){
	float2 ret;
	ret.v[0] = _maxf(x.v[0], y.v[0]); 
	ret.v[1] = _maxf(x.v[1], y.v[1]);
	return ret;
}
float3 max(float3 x, float3 y){
	float3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;
}
float4 max(float4 x, float4 y){
	float4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;
}
float8 max(float8 x, float8 y){
	float8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;
}
float16 max(float16 x, float16 y){
	float16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;
}

float2 max(float2 x, float y){
	float2 ret;
	ret.v[0] = _maxf(x.v[0], y); 
	ret.v[1] = _maxf(x.v[1], y);
	return ret;
}

float3 max(float3 x, float y){
	float3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}

float4 max(float4 x, float y){
	float4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}

float8 max(float8 x, float y){
	float8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}

float16 max(float16 x, float y){
	float16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}

inline static double _maxf(double x, double y){
  return x<y ? y:x;
}

double max(double x, double y){
	return _maxf(x, y);
}

double2 max(double2 x, double2 y){
	double2 ret;
	ret.v[0] = _maxf(x.v[0], y.v[0]); 
	ret.v[1] = _maxf(x.v[1], y.v[1]);
	return ret;
}

double3 max(double3 x, double3 y){
	double3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;

}
double4 max(double4 x, double4 y){
	double4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;
}

double8 max(double8 x, double8 y){
	double8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;
}

double16 max(double16 x, double16 y){
	double16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = _maxf(x.v[i], y.v[i]); 
  }
	return ret;
}

double2 max(double2 x, double y){
	double2 ret;
	ret.v[0] = _maxf(x.v[0], y); 
	ret.v[1] = _maxf(x.v[1], y);
	return ret;
}

double3 max(double3 x, double y){
	double3 ret;
  for (int i=0; i<3; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}

double4 max(double4 x, double y){
	double4 ret;
  for (int i=0; i<4; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}

double8 max(double8 x, double y){
	double8 ret;
  for (int i=0; i<8; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}

double16 max(double16 x, double y){
	double16 ret;
  for (int i=0; i<16; ++i) {
    ret.v[i] = _maxf(x.v[i], y); 
  }
	return ret;
}
