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
#include "clamp.h"

inline static float _minf(float x, float y){
  return y<x ? y:x;
}
inline static float _maxf(float x, float y){
  return x<y ? y:x;
}
inline static double _min(double x, double y){
  return y<x ? y:x;
}
inline static double _max(double x, double y){
  return x<y ? y:x;
}

float clamp(float x, float minval, float maxval){
	return _minf(_maxf(x,minval),maxval);
}

float2 clamp(float2 x, float2 minval, float2 maxval){
	float2 ret;
	ret.v[0] = _minf(_maxf(x.v[0],minval.v[0]),maxval.v[0]);
	ret.v[1] = _minf(_maxf(x.v[1],minval.v[1]),maxval.v[1]);
	return ret;
}

float3 clamp(float3 x, float3 minval, float3 maxval){
	float3 ret;
  for(int i=0; i<3; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;
}

float4 clamp(float4 x, float4 minval, float4 maxval){
	float4 ret;
  for(int i=0; i<4; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;
}

float8 clamp(float8 x, float8 minval, float8 maxval){
	float8 ret;
  for(int i=0; i<8; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;

}

float16 clamp(float16 x, float16 minval, float16 maxval){
	float16 ret;
  for(int i=0; i<16; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;
}

float2 clamp(float2 x, float minval, float maxval){
	float2 ret;
	ret.v[0] = _minf(_maxf(x.v[0],minval),maxval);
	ret.v[1] = _minf(_maxf(x.v[1],minval),maxval);
	return ret;
}

float3 clamp(float3 x, float minval, float maxval){
	float3 ret;
  for(int i=0; i<3; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval),maxval);
  }
	return ret;
}

float4 clamp(float4 x, float minval, float maxval){
	float4 ret;
  for(int i=0; i<4; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval),maxval);
  }
	return ret;
}

float8 clamp(float8 x, float minval, float maxval){
	float8 ret; 
  for(int i=0; i<8; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval),maxval);
  }
	return ret;
}

float16 clamp(float16 x, float minval, float maxval){
	float16 ret; 
  for(int i=0; i<16; ++i) {
    ret.v[i] = _minf(_maxf(x.v[i],minval),maxval);
  }
	return ret;
}
 
double clamp(double x, double minval, double maxval){
	return _min(_max(x,minval),maxval);
}

double2 clamp(double2 x, double2 minval, double2 maxval){
	double2 ret;
	ret.v[0] = _min(_max(x.v[0],minval.v[0]),maxval.v[0]);
	ret.v[1] = _min(_max(x.v[1],minval.v[1]),maxval.v[1]);
	return ret;
}

double3 clamp(double3 x, double3 minval, double3 maxval){
	double3 ret;
  for(int i=0; i<3; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;
}

double4 clamp(double4 x, double4 minval, double4 maxval){
	double4 ret;
  for(int i=0; i<4; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;
}

double8 clamp(double8 x, double8 minval, double8 maxval){
	double8 ret;
  for(int i=0; i<8; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;

}

double16 clamp(double16 x, double16 minval, double16 maxval){
	double16 ret;
  for(int i=0; i<16; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval.v[i]),maxval.v[i]);
  }
	return ret;
}

double2 clamp(double2 x, double minval, double maxval){
	double2 ret;
	ret.v[0] = _min(_max(x.v[0],minval),maxval);
	ret.v[1] = _min(_max(x.v[1],minval),maxval);
	return ret;
}

double3 clamp(double3 x, double minval, double maxval){
	double3 ret;
  for(int i=0; i<3; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval),maxval);
  }
	return ret;
}

double4 clamp(double4 x, double minval, double maxval){
	double4 ret;
  for(int i=0; i<4; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval),maxval);
  }
	return ret;
}

double8 clamp(double8 x, double minval, double maxval){
	double8 ret; 
  for(int i=0; i<8; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval),maxval);
  }
	return ret;
}

double16 clamp(double16 x, double minval, double maxval){
	double16 ret; 
  for(int i=0; i<16; ++i) {
    ret.v[i] = _min(_max(x.v[i],minval),maxval);
  }
	return ret;
}
