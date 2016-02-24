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

#ifndef CLAMP_H_
#define CLAMP_H_

#include <cl_cpu_types.h>

float clamp(float x, float minval, float maxval);
float2 clamp(float2 x, float2 minval, float2 maxval);
float3 clamp(float3 x, float3 minval, float3 maxval);
float4 clamp(float4 x, float4 minval, float4 maxval);
float8 clamp(float8 x, float8 minval, float8 maxval);
float16 clamp(float16 x, float16 minval, float16 maxval);

float2 clamp(float2 x, float minval, float maxval);
float3 clamp(float3 x, float minval, float maxval);
float4 clamp(float4 x, float minval, float maxval);
float8 clamp(float8 x, float minval, float maxval);
float16 clamp(float16 x, float minval, float maxval);

double clamp(double x, double minval, double maxval);
double2 clamp(double2 x, double2 minval, double2 maxval);
double3 clamp(double3 x, double3 minval, double3 maxval);
double4 clamp(double4 x, double4 minval, double4 maxval);
double8 clamp(double8 x, double8 minval, double8 maxval);
double16 clamp(double16 x, double16 minval, double16 maxval);

double2 clamp(double2 x, double minval, double maxval);
double3 clamp(double3 x, double minval, double maxval);
double4 clamp(double4 x, double minval, double maxval);
double8 clamp(double8 x, double minval, double maxval);
double16 clamp(double16 x, double minval, double maxval);

#endif
