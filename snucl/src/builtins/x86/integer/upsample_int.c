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
// int
int upsample(short hi, ushort lo){
  return ((int)hi << 16) | lo;
}
int2 upsample(short2 hi, ushort2 lo){
	int2 rst;
	rst.v[0]=((int)hi.v[0]<<(int)16)|(int)lo.v[0];
	rst.v[1]=((int)hi.v[1]<<(int)16)|(int)lo.v[1];
  return rst;
}
int3 upsample(short3 hi, ushort3 lo){
	int3 rst;
	rst.v[0]=((int)hi.v[0]<<(int)16)|(int)lo.v[0];
	rst.v[1]=((int)hi.v[1]<<(int)16)|(int)lo.v[1];
	rst.v[2]=((int)hi.v[2]<<(int)16)|(int)lo.v[2];
	return rst;
}
int4 upsample(short4 hi, ushort4 lo){
	int4 rst;
	rst.v[0]=((int)hi.v[0]<<(int)16)|(int)lo.v[0];
	rst.v[1]=((int)hi.v[1]<<(int)16)|(int)lo.v[1];
	rst.v[2]=((int)hi.v[2]<<(int)16)|(int)lo.v[2];
	rst.v[3]=((int)hi.v[3]<<(int)16)|(int)lo.v[3];
	return rst;
}
int8 upsample(short8 hi, ushort8 lo){
	int8 rst;
	rst.v[0]=((int)hi.v[0]<<(int)16)|(int)lo.v[0];
	rst.v[1]=((int)hi.v[1]<<(int)16)|(int)lo.v[1];
	rst.v[2]=((int)hi.v[2]<<(int)16)|(int)lo.v[2];
	rst.v[3]=((int)hi.v[3]<<(int)16)|(int)lo.v[3];
	rst.v[4]=((int)hi.v[4]<<(int)16)|(int)lo.v[4];
	rst.v[5]=((int)hi.v[5]<<(int)16)|(int)lo.v[5];
	rst.v[6]=((int)hi.v[6]<<(int)16)|(int)lo.v[6];
	rst.v[7]=((int)hi.v[7]<<(int)16)|(int)lo.v[7];
	return rst;
}
int16 upsample(short16 hi, ushort16 lo){
	int16 rst;
	rst.v[0]=((int)hi.v[0]<<(int)16)|(int)lo.v[0];
	rst.v[1]=((int)hi.v[1]<<(int)16)|(int)lo.v[1];
	rst.v[2]=((int)hi.v[2]<<(int)16)|(int)lo.v[2];
	rst.v[3]=((int)hi.v[3]<<(int)16)|(int)lo.v[3];
	rst.v[4]=((int)hi.v[4]<<(int)16)|(int)lo.v[4];
	rst.v[5]=((int)hi.v[5]<<(int)16)|(int)lo.v[5];
	rst.v[6]=((int)hi.v[6]<<(int)16)|(int)lo.v[6];
	rst.v[7]=((int)hi.v[7]<<(int)16)|(int)lo.v[7];
	rst.v[8]=((int)hi.v[8]<<(int)16)|(int)lo.v[8];
	rst.v[9]=((int)hi.v[9]<<(int)16)|(int)lo.v[9];
	rst.v[10]=((int)hi.v[10]<<(int)16)|(int)lo.v[10];
	rst.v[11]=((int)hi.v[11]<<(int)16)|(int)lo.v[11];
	rst.v[12]=((int)hi.v[12]<<(int)16)|(int)lo.v[12];
	rst.v[13]=((int)hi.v[13]<<(int)16)|(int)lo.v[13];
	rst.v[14]=((int)hi.v[14]<<(int)16)|(int)lo.v[14];
	rst.v[15]=((int)hi.v[15]<<(int)16)|(int)lo.v[15];
	return rst;
}

