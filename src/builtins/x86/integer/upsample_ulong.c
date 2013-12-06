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

ullong upsample(uint hi, uint lo){
  return ((ullong)hi << 32) | lo;
}
ulong2 upsample(uint2 hi, uint2 lo){
	ulong2 rst;
	rst.v[0]=((ullong)hi.v[0]<<(ullong)32)|(ullong)lo.v[0];
	rst.v[1]=((ullong)hi.v[1]<<(ullong)32)|(ullong)lo.v[1];
  return rst;
}
ulong3 upsample(uint3 hi, uint3 lo){
	ulong3 rst;
	rst.v[0]=((ullong)hi.v[0]<<(ullong)32)|(ullong)lo.v[0];
	rst.v[1]=((ullong)hi.v[1]<<(ullong)32)|(ullong)lo.v[1];
	rst.v[2]=((ullong)hi.v[2]<<(ullong)32)|(ullong)lo.v[2];

	return rst;
}
ulong4 upsample(uint4 hi, uint4 lo){
	ulong4 rst;
	rst.v[0]=((ullong)hi.v[0]<<(ullong)32)|(ullong)lo.v[0];
	rst.v[1]=((ullong)hi.v[1]<<(ullong)32)|(ullong)lo.v[1];
	rst.v[2]=((ullong)hi.v[2]<<(ullong)32)|(ullong)lo.v[2];
	rst.v[3]=((ullong)hi.v[3]<<(ullong)32)|(ullong)lo.v[3];

	return rst;
}
ulong8 upsample(uint8 hi, uint8 lo){
	ulong8 rst;
	rst.v[0]=((ullong)hi.v[0]<<(ullong)32)|(ullong)lo.v[0];
	rst.v[1]=((ullong)hi.v[1]<<(ullong)32)|(ullong)lo.v[1];
	rst.v[2]=((ullong)hi.v[2]<<(ullong)32)|(ullong)lo.v[2];
	rst.v[3]=((ullong)hi.v[3]<<(ullong)32)|(ullong)lo.v[3];
	rst.v[4]=((ullong)hi.v[4]<<(ullong)32)|(ullong)lo.v[4];
	rst.v[5]=((ullong)hi.v[5]<<(ullong)32)|(ullong)lo.v[5];
	rst.v[6]=((ullong)hi.v[6]<<(ullong)32)|(ullong)lo.v[6];
	rst.v[7]=((ullong)hi.v[7]<<(ullong)32)|(ullong)lo.v[7];
  return rst;
}
ulong16 upsample(uint16 hi, uint16 lo){
	ulong16 rst;
	rst.v[0]=((ullong)hi.v[0]<<(ullong)32)|(ullong)lo.v[0];
	rst.v[1]=((ullong)hi.v[1]<<(ullong)32)|(ullong)lo.v[1];
	rst.v[2]=((ullong)hi.v[2]<<(ullong)32)|(ullong)lo.v[2];
	rst.v[3]=((ullong)hi.v[3]<<(ullong)32)|(ullong)lo.v[3];
	rst.v[4]=((ullong)hi.v[4]<<(ullong)32)|(ullong)lo.v[4];
	rst.v[5]=((ullong)hi.v[5]<<(ullong)32)|(ullong)lo.v[5];
	rst.v[6]=((ullong)hi.v[6]<<(ullong)32)|(ullong)lo.v[6];
	rst.v[7]=((ullong)hi.v[7]<<(ullong)32)|(ullong)lo.v[7];
	rst.v[8]=((ullong)hi.v[8]<<(ullong)32)|(ullong)lo.v[8];
	rst.v[9]=((ullong)hi.v[9]<<(ullong)32)|(ullong)lo.v[9];
	rst.v[10]=((ullong)hi.v[10]<<(ullong)32)|(ullong)lo.v[10];
	rst.v[11]=((ullong)hi.v[11]<<(ullong)32)|(ullong)lo.v[11];
	rst.v[12]=((ullong)hi.v[12]<<(ullong)32)|(ullong)lo.v[12];
	rst.v[13]=((ullong)hi.v[13]<<(ullong)32)|(ullong)lo.v[13];
	rst.v[14]=((ullong)hi.v[14]<<(ullong)32)|(ullong)lo.v[14];
	rst.v[15]=((ullong)hi.v[15]<<(ullong)32)|(ullong)lo.v[15];
  return rst;	
}				

