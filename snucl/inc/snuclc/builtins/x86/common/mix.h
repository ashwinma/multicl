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

#ifndef MIX_H_
#define MIX_H_

#include <cl_cpu_types.h>

float mix(float x, float y, float a);
float2 mix(float2 x, float2 y, float2 a);
float3 mix(float3 x, float3 y, float3 a);
float4 mix(float4 x, float4 y, float4 a);
float8 mix(float8 x, float8 y, float8 a);
float16 mix(float16 x, float16 y, float16 a);

float2 mix(float2 x, float2 y, float a);
float3 mix(float3 x, float3 y, float a);
float4 mix(float4 x, float4 y, float a);
float8 mix(float8 x, float8 y, float a);
float16 mix(float16 x, float16 y, float a);

double mix(double x, double y, double a);
double2 mix(double2 x, double2 y, double2 a);
double3 mix(double3 x, double3 y, double3 a);
double4 mix(double4 x, double4 y, double4 a);
double8 mix(double8 x, double8 y, double8 a);
double16 mix(double16 x, double16 y, double16 a);

double2 mix(double2 x, double2 y, double a);
double3 mix(double3 x, double3 y, double a);
double4 mix(double4 x, double4 y, double a);
double8 mix(double8 x, double8 y, double a);
double16 mix(double16 x, double16 y, double a);

#endif
