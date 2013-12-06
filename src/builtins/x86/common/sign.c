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
#include "sign.h"

float sign(float x){
	if      (x>0.0)   return 1.0;
	else if (x==-0.0) return -0.0;
	else if (x==+0.0) return +0.0;
  else if (x<0.0)   return -1.0;
	else              return 0.0;
}

float2 sign(float2 x){
	if      (x.v[0]>0.0)   x.v[0] = 1.0;
	else if (x.v[0]==-0.0) x.v[0] = -0.0;
	else if (x.v[0]==+0.0) x.v[0] = +0.0;
  else if (x.v[0]<0.0)   x.v[0] = -1.0;
	else                   x.v[0] = 0.0;

	if      (x.v[1]>0.0)   x.v[1] = 1.0;
	else if (x.v[1]==-0.0) x.v[1] = -0.0;
	else if (x.v[1]==+0.0) x.v[1] = +0.0;
  else if (x.v[1]<0.0)   x.v[1] = -1.0;
	else                   x.v[1] = 0.0;

	return x;
}

float3 sign(float3 x) {
  for (int i=0; i<3; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}

float4 sign(float4 x) {
  for (int i=0; i<4; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}

float8 sign(float8 x) {
  for (int i=0; i<8; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}

float16 sign(float16 x){
  for (int i=0; i<16; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}

double sign(double x){
	if      (x>0.0)   return 1.0;
	else if (x==-0.0) return -0.0;
	else if (x==+0.0) return +0.0;
  else if (x<0.0)   return -1.0;
	else              return 0.0;
}

double2 sign(double2 x){
	if      (x.v[0]>0.0)   x.v[0] = 1.0;
	else if (x.v[0]==-0.0) x.v[0] = -0.0;
	else if (x.v[0]==+0.0) x.v[0] = +0.0;
  else if (x.v[0]<0.0)   x.v[0] = -1.0;
	else                   x.v[0] = 0.0;

	if      (x.v[1]>0.0)   x.v[1] = 1.0;
	else if (x.v[1]==-0.0) x.v[1] = -0.0;
	else if (x.v[1]==+0.0) x.v[1] = +0.0;
  else if (x.v[1]<0.0)   x.v[1] = -1.0;
	else                   x.v[1] = 0.0;

	return x;
}

double3 sign(double3 x) {
  for (int i=0; i<3; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}

double4 sign(double4 x) {
  for (int i=0; i<4; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}

double8 sign(double8 x) {
  for (int i=0; i<8; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}

double16 sign(double16 x){
  for (int i=0; i<16; ++i) {
    if      (x.v[i]>0.0)   x.v[i] = 1.0;
    else if (x.v[i]==-0.0) x.v[i] = -0.0;
    else if (x.v[i]==+0.0) x.v[i] = +0.0;
    else if (x.v[i]<0.0)   x.v[i] = -1.0;
    else                   x.v[i] = 0.0;
  }
	return x;
}
