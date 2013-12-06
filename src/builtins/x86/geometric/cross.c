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

float3 cross(float3 p0, float3 p1){
	float3 rst;
	rst.v[0]=(p0.v[1]*p1.v[2])-(p0.v[2]*p1.v[1]);
	rst.v[1]=(p0.v[2]*p1.v[0])-(p0.v[0]*p1.v[2]);
	rst.v[2]=(p0.v[0]*p1.v[1])-(p0.v[1]*p1.v[0]);
	return rst;
}

float4 cross(float4 p0, float4 p1){
	float4 rst;
	rst.v[0]=(p0.v[1]*p1.v[2])-(p0.v[2]*p1.v[1]);
	rst.v[1]=(p0.v[2]*p1.v[0])-(p0.v[0]*p1.v[2]);
	rst.v[2]=(p0.v[0]*p1.v[1])-(p0.v[1]*p1.v[0]);
	rst.v[3]=0.0f;
	return rst;
}


double3 cross(double3 p0, double3 p1){
	double3 rst;
	rst.v[0]=(p0.v[1]*p1.v[2])-(p0.v[2]*p1.v[1]);
	rst.v[1]=(p0.v[2]*p1.v[0])-(p0.v[0]*p1.v[2]);
	rst.v[2]=(p0.v[0]*p1.v[1])-(p0.v[1]*p1.v[0]);
	return rst;
}

double4 cross(double4 p0, double4 p1){
	double4 rst;
	rst.v[0]=(p0.v[1]*p1.v[2])-(p0.v[2]*p1.v[1]);
	rst.v[1]=(p0.v[2]*p1.v[0])-(p0.v[0]*p1.v[2]);
	rst.v[2]=(p0.v[0]*p1.v[1])-(p0.v[1]*p1.v[0]);
	rst.v[3]=0.0f;
	return rst;
}

