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

// uint
uint upsample(ushort hi, ushort lo){
  return ((uint)hi << 16) | lo;
}
uint2 upsample(ushort2 hi, ushort2 lo){
	uint2 rst;
	rst.v[0]=((uint)hi.v[0]<<(uint)16)|(uint)lo.v[0];
	rst.v[1]=((uint)hi.v[1]<<(uint)16)|(uint)lo.v[1];
  return rst;
}
uint3 upsample(ushort3 hi, ushort3 lo){
	uint3 rst;
	rst.v[0]=((uint)hi.v[0]<<(uint)16)|(uint)lo.v[0];
	rst.v[1]=((uint)hi.v[1]<<(uint)16)|(uint)lo.v[1];
	rst.v[2]=((uint)hi.v[2]<<(uint)16)|(uint)lo.v[2];
	rst.v[3]=((uint)hi.v[3]<<(uint)16)|(uint)lo.v[3];
  return rst;
}
uint4 upsample(ushort4 hi, ushort4 lo){
	uint4 rst;
	rst.v[0]=((uint)hi.v[0]<<(uint)16)|(uint)lo.v[0];
	rst.v[1]=((uint)hi.v[1]<<(uint)16)|(uint)lo.v[1];
	rst.v[2]=((uint)hi.v[2]<<(uint)16)|(uint)lo.v[2];
	rst.v[3]=((uint)hi.v[3]<<(uint)16)|(uint)lo.v[3];
  return rst;
}
uint8 upsample(ushort8 hi, ushort8 lo){
	uint8 rst;
	rst.v[0]=((uint)hi.v[0]<<(uint)16)|(uint)lo.v[0];
	rst.v[1]=((uint)hi.v[1]<<(uint)16)|(uint)lo.v[1];
	rst.v[2]=((uint)hi.v[2]<<(uint)16)|(uint)lo.v[2];
	rst.v[3]=((uint)hi.v[3]<<(uint)16)|(uint)lo.v[3];
	rst.v[4]=((uint)hi.v[4]<<(uint)16)|(uint)lo.v[4];
	rst.v[5]=((uint)hi.v[5]<<(uint)16)|(uint)lo.v[5];
	rst.v[6]=((uint)hi.v[6]<<(uint)16)|(uint)lo.v[6];
	rst.v[7]=((uint)hi.v[7]<<(uint)16)|(uint)lo.v[7];

	return rst;
}
uint16 upsample(ushort16 hi, ushort16 lo){
	uint16 rst;
	rst.v[0]=((uint)hi.v[0]<<(uint)16)|(uint)lo.v[0];
	rst.v[1]=((uint)hi.v[1]<<(uint)16)|(uint)lo.v[1];
	rst.v[2]=((uint)hi.v[2]<<(uint)16)|(uint)lo.v[2];
	rst.v[3]=((uint)hi.v[3]<<(uint)16)|(uint)lo.v[3];
	rst.v[4]=((uint)hi.v[4]<<(uint)16)|(uint)lo.v[4];
	rst.v[5]=((uint)hi.v[5]<<(uint)16)|(uint)lo.v[5];
	rst.v[6]=((uint)hi.v[6]<<(uint)16)|(uint)lo.v[6];
	rst.v[7]=((uint)hi.v[7]<<(uint)16)|(uint)lo.v[7];
	rst.v[8]=((uint)hi.v[8]<<(uint)16)|(uint)lo.v[8];
	rst.v[9]=((uint)hi.v[9]<<(uint)16)|(uint)lo.v[9];
	rst.v[10]=((uint)hi.v[10]<<(uint)16)|(uint)lo.v[10];
	rst.v[11]=((uint)hi.v[11]<<(uint)16)|(uint)lo.v[11];
	rst.v[12]=((uint)hi.v[12]<<(uint)16)|(uint)lo.v[12];
	rst.v[13]=((uint)hi.v[13]<<(uint)16)|(uint)lo.v[13];
	rst.v[14]=((uint)hi.v[14]<<(uint)16)|(uint)lo.v[14];
	rst.v[15]=((uint)hi.v[15]<<(uint)16)|(uint)lo.v[15];
	return rst;
}

