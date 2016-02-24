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
#include <math.h>
#include <math/half_sqrt.h>
#include <math/pow.h>


float fast_length(float p){
	return half_sqrt((float)pow((double)p, 2.0));
}

float fast_length(float2 p){
	/*
  float2 c;
  c.v[0] = c.v[1] = 2.0f;
  float2 rst = pow(p, c);
  return half_sqrt(rst.v[0] + rst.v[1]);
  */

  return half_sqrt( (float)(pow((double)p[0], 2.0) + pow((double)p[1], 2.0)) );
}

float fast_length(float3 p){
	/*
  float3 c;
  c.v[0] = c.v[1] = c.v[2] = c.v[3] = 2.0f;
  float3 rst = pow(p, c);
  return half_sqrt(rst.v[0] + rst.v[1] + rst.v[2]);
  */

  return half_sqrt( (float)(pow((double)p[0], 2.0) + pow((double)p[1], 2.0) + pow((double)p[2], 2.0)) );
}

float fast_length(float4 p){
	/*
  float4 c;
  c.v[0] = c.v[1] = c.v[2] = c.v[3] = 2.0f;
  float4 rst = pow(p, c);
  return half_sqrt(rst.v[0] + rst.v[1] + rst.v[2] + rst.v[3]);
  */

  return half_sqrt( (float)(pow((double)p[0], 2.0) + pow((double)p[1], 2.0) + pow((double)p[2], 2.0) + pow((double)p[3], 2.0)) );
}
