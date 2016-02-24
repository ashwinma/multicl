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

//llong, ullong
llong upsample(int hi, uint lo){
  return ((llong)hi << 32) | lo;
}
long2 upsample(int2 hi, uint2 lo){
	long2 rst;
	rst.v[0]=((llong)hi.v[0]<<(llong)32)|(llong)lo.v[0];
	rst.v[1]=((llong)hi.v[1]<<(llong)32)|(llong)lo.v[1];
  return rst;
}
long3 upsample(int3 hi, uint3 lo){
	long3 rst;
	rst.v[0]=((llong)hi.v[0]<<(llong)32)|(llong)lo.v[0];
	rst.v[1]=((llong)hi.v[1]<<(llong)32)|(llong)lo.v[1];
	rst.v[2]=((llong)hi.v[2]<<(llong)32)|(llong)lo.v[2];
	
	return rst;
}
long4 upsample(int4 hi, uint4 lo){
	long4 rst;
	rst.v[0]=((llong)hi.v[0]<<(llong)32)|(llong)lo.v[0];
	rst.v[1]=((llong)hi.v[1]<<(llong)32)|(llong)lo.v[1];
	rst.v[2]=((llong)hi.v[2]<<(llong)32)|(llong)lo.v[2];
	rst.v[3]=((llong)hi.v[3]<<(llong)32)|(llong)lo.v[3];

	return rst;
}
long8 upsample(int8 hi, uint8 lo){
	long8 rst;
	rst.v[0]=((llong)hi.v[0]<<(llong)32)|(llong)lo.v[0];
	rst.v[1]=((llong)hi.v[1]<<(llong)32)|(llong)lo.v[1];
	rst.v[2]=((llong)hi.v[2]<<(llong)32)|(llong)lo.v[2];
	rst.v[3]=((llong)hi.v[3]<<(llong)32)|(llong)lo.v[3];
	rst.v[4]=((llong)hi.v[4]<<(llong)32)|(llong)lo.v[4];
	rst.v[5]=((llong)hi.v[5]<<(llong)32)|(llong)lo.v[5];
	rst.v[6]=((llong)hi.v[6]<<(llong)32)|(llong)lo.v[6];
	rst.v[7]=((llong)hi.v[7]<<(llong)32)|(llong)lo.v[7];
	return rst;
}
long16 upsample(int16 hi, uint16 lo){
	long16 rst;
	rst.v[0]=((llong)hi.v[0]<<(llong)32)|(llong)lo.v[0];
	rst.v[1]=((llong)hi.v[1]<<(llong)32)|(llong)lo.v[1];
	rst.v[2]=((llong)hi.v[2]<<(llong)32)|(llong)lo.v[2];
	rst.v[3]=((llong)hi.v[3]<<(llong)32)|(llong)lo.v[3];
	rst.v[4]=((llong)hi.v[4]<<(llong)32)|(llong)lo.v[4];
	rst.v[5]=((llong)hi.v[5]<<(llong)32)|(llong)lo.v[5];
	rst.v[6]=((llong)hi.v[6]<<(llong)32)|(llong)lo.v[6];
	rst.v[7]=((llong)hi.v[7]<<(llong)32)|(llong)lo.v[7];
	rst.v[8]=((llong)hi.v[8]<<(llong)32)|(llong)lo.v[8];
	rst.v[9]=((llong)hi.v[9]<<(llong)32)|(llong)lo.v[9];
	rst.v[10]=((llong)hi.v[10]<<(llong)32)|(llong)lo.v[10];
	rst.v[11]=((llong)hi.v[11]<<(llong)32)|(llong)lo.v[11];
	rst.v[12]=((llong)hi.v[12]<<(llong)32)|(llong)lo.v[12];
	rst.v[13]=((llong)hi.v[13]<<(llong)32)|(llong)lo.v[13];
	rst.v[14]=((llong)hi.v[14]<<(llong)32)|(llong)lo.v[14];
	rst.v[15]=((llong)hi.v[15]<<(llong)32)|(llong)lo.v[15];
  return rst;
}

