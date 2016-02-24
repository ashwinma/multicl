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
#include "smoothstep.h"
#include "clamp.h"
#include <stdio.h>

float smoothstep(float edge0, float edge1, float x){
	float t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float2 smoothstep(float2 edge0, float2 edge1, float2 x){
	float2 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float3 smoothstep(float3 edge0, float3 edge1, float3 x){
	float3 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float4 smoothstep(float4 edge0, float4 edge1, float4 x){
	float4 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float8 smoothstep(float8 edge0, float8 edge1, float8 x){
	float8 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float16 smoothstep(float16 edge0, float16 edge1, float16 x){
	float16 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float2 smoothstep(float edge0, float edge1, float2 x){
	float2 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float3 smoothstep(float edge0, float edge1, float3 x){
	float3 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float4 smoothstep(float edge0, float edge1, float4 x){
	float4 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float8 smoothstep(float edge0, float edge1, float8 x){
	float8 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

float16 smoothstep(float edge0, float edge1, float16 x){
	float16 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0f, 1.0f);
	return t*t*(3-2*t);
}

double smoothstep(double edge0, double edge1, double x){
	double t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double2 smoothstep(double2 edge0, double2 edge1, double2 x){
	double2 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double3 smoothstep(double3 edge0, double3 edge1, double3 x){
	double3 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double4 smoothstep(double4 edge0, double4 edge1, double4 x){
	double4 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double8 smoothstep(double8 edge0, double8 edge1, double8 x){
	double8 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double16 smoothstep(double16 edge0, double16 edge1, double16 x){
	double16 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double2 smoothstep(double edge0, double edge1, double2 x){
	double2 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double3 smoothstep(double edge0, double edge1, double3 x){
	double3 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double4 smoothstep(double edge0, double edge1, double4 x){
	double4 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double8 smoothstep(double edge0, double edge1, double8 x){
	double8 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}

double16 smoothstep(double edge0, double edge1, double16 x){
	double16 t;
	t = clamp((x-edge0)/(edge1-edge0), 0.0, 1.0);
	return t*t*(3-2*t);
}
