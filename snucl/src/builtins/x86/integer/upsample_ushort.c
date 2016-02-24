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

// ushort
ushort upsample(uchar hi, uchar lo){
	return ((ushort)hi << 8) | lo;;
}
ushort2 upsample(uchar2 hi, uchar2 lo){
	ushort2 rst;
	rst.v[0]=((ushort)hi.v[0]<<(ushort)8)|(ushort)lo.v[0];
	rst.v[1]=((ushort)hi.v[1]<<(ushort)8)|(ushort)lo.v[1];
  return rst;
}
ushort3 upsample(uchar3 hi, uchar3 lo){
	ushort3 rst;
	rst.v[0]=((ushort)hi.v[0]<<(ushort)8)|(ushort)lo.v[0];
	rst.v[1]=((ushort)hi.v[1]<<(ushort)8)|(ushort)lo.v[1];
	rst.v[2]=((ushort)hi.v[2]<<(ushort)8)|(ushort)lo.v[2];
  return rst;
}
ushort4 upsample(uchar4 hi, uchar4 lo){
	ushort4 rst;
	rst.v[0]=((ushort)hi.v[0]<<(ushort)8)|(ushort)lo.v[0];
	rst.v[1]=((ushort)hi.v[1]<<(ushort)8)|(ushort)lo.v[1];
	rst.v[2]=((ushort)hi.v[2]<<(ushort)8)|(ushort)lo.v[2];
	rst.v[3]=((ushort)hi.v[3]<<(ushort)8)|(ushort)lo.v[3];
  return rst;
}
ushort8 upsample(uchar8 hi, uchar8 lo){
	ushort8 rst;
	rst.v[0]=((ushort)hi.v[0]<<(ushort)8)|(ushort)lo.v[0];
	rst.v[1]=((ushort)hi.v[1]<<(ushort)8)|(ushort)lo.v[1];
	rst.v[2]=((ushort)hi.v[2]<<(ushort)8)|(ushort)lo.v[2];
	rst.v[3]=((ushort)hi.v[3]<<(ushort)8)|(ushort)lo.v[3];
	rst.v[4]=((ushort)hi.v[4]<<(ushort)8)|(ushort)lo.v[4];
	rst.v[5]=((ushort)hi.v[5]<<(ushort)8)|(ushort)lo.v[5];
	rst.v[6]=((ushort)hi.v[6]<<(ushort)8)|(ushort)lo.v[6];
	rst.v[7]=((ushort)hi.v[7]<<(ushort)8)|(ushort)lo.v[7];
  return rst;
}
ushort16 upsample(uchar16 hi, uchar16 lo){
	ushort16 rst;
	rst.v[0]=((ushort)hi.v[0]<<(ushort)8)|(ushort)lo.v[0];
	rst.v[1]=((ushort)hi.v[1]<<(ushort)8)|(ushort)lo.v[1];
	rst.v[2]=((ushort)hi.v[2]<<(ushort)8)|(ushort)lo.v[2];
	rst.v[3]=((ushort)hi.v[3]<<(ushort)8)|(ushort)lo.v[3];
	rst.v[4]=((ushort)hi.v[4]<<(ushort)8)|(ushort)lo.v[4];
	rst.v[5]=((ushort)hi.v[5]<<(ushort)8)|(ushort)lo.v[5];
	rst.v[6]=((ushort)hi.v[6]<<(ushort)8)|(ushort)lo.v[6];
	rst.v[7]=((ushort)hi.v[7]<<(ushort)8)|(ushort)lo.v[7];
	rst.v[8]=((ushort)hi.v[8]<<(ushort)8)|(ushort)lo.v[8];
	rst.v[9]=((ushort)hi.v[9]<<(ushort)8)|(ushort)lo.v[9];
	rst.v[10]=((ushort)hi.v[10]<<(ushort)8)|(ushort)lo.v[10];
	rst.v[11]=((ushort)hi.v[11]<<(ushort)8)|(ushort)lo.v[11];
	rst.v[12]=((ushort)hi.v[12]<<(ushort)8)|(ushort)lo.v[12];
	rst.v[13]=((ushort)hi.v[13]<<(ushort)8)|(ushort)lo.v[13];
	rst.v[14]=((ushort)hi.v[14]<<(ushort)8)|(ushort)lo.v[14];
	rst.v[15]=((ushort)hi.v[15]<<(ushort)8)|(ushort)lo.v[15];
  return rst;
}
