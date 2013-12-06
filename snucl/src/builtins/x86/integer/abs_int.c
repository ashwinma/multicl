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
#include <cl_cpu_types.h>


//uint
/*
uint abs(int x){
	return x<0? -x:x;
}
*/

uint2 abs(int2 x){
	uint2 rst;
	rst.v[0]= x.v[0]<0? -x.v[0]:x.v[0];
	rst.v[1]= x.v[1]<0? -x.v[1]:x.v[1];
	return rst;
}

uint3 abs(int3 x){
	uint3 rst;
	rst.v[0]= x.v[0]<0? -x.v[0]:x.v[0];
	rst.v[1]= x.v[1]<0? -x.v[1]:x.v[1];
	rst.v[2]= x.v[2]<0? -x.v[2]:x.v[2];
	return rst;

}

uint4 abs(int4 x){
	uint4 rst;
	rst.v[0]= x.v[0]<0? -x.v[0]:x.v[0];
	rst.v[1]= x.v[1]<0? -x.v[1]:x.v[1];
	rst.v[2]= x.v[2]<0? -x.v[2]:x.v[2];
	rst.v[3]= x.v[3]<0? -x.v[3]:x.v[3];
	return rst;

}

uint8 abs(int8 x){

	uint8 rst;
	rst.v[0]= x.v[0]<0? -x.v[0]:x.v[0];
	rst.v[1]= x.v[1]<0? -x.v[1]:x.v[1];
	rst.v[2]= x.v[2]<0? -x.v[2]:x.v[2];
	rst.v[3]= x.v[3]<0? -x.v[3]:x.v[3];
	rst.v[4]= x.v[4]<0? -x.v[4]:x.v[4];
	rst.v[5]= x.v[5]<0? -x.v[5]:x.v[5];
	rst.v[6]= x.v[6]<0? -x.v[6]:x.v[6];
	rst.v[7]= x.v[7]<0? -x.v[7]:x.v[7];
	return rst;
}

uint16 abs(int16 x){
	uint16 rst;
	rst.v[0]= x.v[0]<0? -x.v[0]:x.v[0];
	rst.v[1]= x.v[1]<0? -x.v[1]:x.v[1];
	rst.v[2]= x.v[2]<0? -x.v[2]:x.v[2];
	rst.v[3]= x.v[3]<0? -x.v[3]:x.v[3];
	rst.v[4]= x.v[4]<0? -x.v[4]:x.v[4];
	rst.v[5]= x.v[5]<0? -x.v[5]:x.v[5];
	rst.v[6]= x.v[6]<0? -x.v[6]:x.v[6];
	rst.v[7]= x.v[7]<0? -x.v[7]:x.v[7];
	rst.v[8]= x.v[8]<0? -x.v[8]:x.v[8];
	rst.v[9]= x.v[9]<0? -x.v[9]:x.v[9];
	rst.v[10]= x.v[10]<0? -x.v[10]:x.v[10];
	rst.v[11]= x.v[11]<0? -x.v[11]:x.v[11];
	rst.v[12]= x.v[12]<0? -x.v[12]:x.v[12];
	rst.v[13]= x.v[13]<0? -x.v[13]:x.v[13];
	rst.v[14]= x.v[14]<0? -x.v[14]:x.v[14];
	rst.v[15]= x.v[15]<0? -x.v[15]:x.v[15];
	return rst;
}

