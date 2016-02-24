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

//short. shortn
short upsample(schar hi, uchar lo){
  return (((short)hi << 8) | lo);
}
short2 upsample(char2 hi, uchar2 lo){
	short2 rst;
	rst.v[0]=((short)hi.v[0]<<(short)8)|(short)lo.v[0];
	rst.v[1]=((short)hi.v[1]<<(short)8)|(short)lo.v[1];
  return rst;
}
short3 upsample(char3 hi, uchar3 lo){
	short3 rst;
	rst.v[0]=((short)hi.v[0]<<(short)8)|(short)lo.v[0];
	rst.v[1]=((short)hi.v[1]<<(short)8)|(short)lo.v[1];
	rst.v[2]=((short)hi.v[2]<<(short)8)|(short)lo.v[2];
  return rst;
}
short4 upsample(char4 hi, uchar4 lo){
	short4 rst;
	rst.v[0]=((short)hi.v[0]<<(short)8)|(short)lo.v[0];
	rst.v[1]=((short)hi.v[1]<<(short)8)|(short)lo.v[1];
	rst.v[2]=((short)hi.v[2]<<(short)8)|(short)lo.v[2];
	rst.v[3]=((short)hi.v[3]<<(short)8)|(short)lo.v[3];
  return rst;
}
short8 upsample(char8 hi, uchar8 lo){
	short8 rst;
	rst.v[0]=((short)hi.v[0]<<(short)8)|(short)lo.v[0];
	rst.v[1]=((short)hi.v[1]<<(short)8)|(short)lo.v[1];
	rst.v[2]=((short)hi.v[2]<<(short)8)|(short)lo.v[2];
	rst.v[3]=((short)hi.v[3]<<(short)8)|(short)lo.v[3];
	rst.v[4]=((short)hi.v[4]<<(short)8)|(short)lo.v[4];
	rst.v[5]=((short)hi.v[5]<<(short)8)|(short)lo.v[5];
	rst.v[6]=((short)hi.v[6]<<(short)8)|(short)lo.v[6];
	rst.v[7]=((short)hi.v[7]<<(short)8)|(short)lo.v[7];
  return rst;
}
short16 upsample(char16 hi, uchar16 lo){
	short16 rst;
	rst.v[0]=((short)hi.v[0]<<(short)8)|(short)lo.v[0];
	rst.v[1]=((short)hi.v[1]<<(short)8)|(short)lo.v[1];
	rst.v[2]=((short)hi.v[2]<<(short)8)|(short)lo.v[2];
	rst.v[3]=((short)hi.v[3]<<(short)8)|(short)lo.v[3];
	rst.v[4]=((short)hi.v[4]<<(short)8)|(short)lo.v[4];
	rst.v[5]=((short)hi.v[5]<<(short)8)|(short)lo.v[5];
	rst.v[6]=((short)hi.v[6]<<(short)8)|(short)lo.v[6];
	rst.v[7]=((short)hi.v[7]<<(short)8)|(short)lo.v[7];
	rst.v[8]=((short)hi.v[8]<<(short)8)|(short)lo.v[8];
	rst.v[9]=((short)hi.v[9]<<(short)8)|(short)lo.v[9];
	rst.v[10]=((short)hi.v[10]<<(short)8)|(short)lo.v[10];
	rst.v[11]=((short)hi.v[11]<<(short)8)|(short)lo.v[11];
	rst.v[12]=((short)hi.v[12]<<(short)8)|(short)lo.v[12];
	rst.v[13]=((short)hi.v[13]<<(short)8)|(short)lo.v[13];
	rst.v[14]=((short)hi.v[14]<<(short)8)|(short)lo.v[14];
	rst.v[15]=((short)hi.v[15]<<(short)8)|(short)lo.v[15];
	return rst;
}

