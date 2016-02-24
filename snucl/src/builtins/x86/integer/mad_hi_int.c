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
int mad_hi(int a, int b, int c){
	return (int)(((llong)a * (llong)b) >> 32) + c;
}

int2 mad_hi(int2 a, int2 b, int2 c){
	int2 rst;
	for (int i = 0; i < 2; i++)
    rst[i] = (int)(((llong)a[i] * (llong)b[i]) >> 32);
    rst=rst + c;
	return rst;
}
int3 mad_hi(int3 a, int3 b, int3 c){
	int3 rst;
  for (int i = 0; i < 3; i++)
    rst[i] = (int)(((llong)a[i] * (llong)b[i]) >> 32);
    rst=rst + c;
	return rst;
}
int4 mad_hi(int4 a, int4 b, int4 c){
	int4 rst;
  for (int i = 0; i < 4; i++)
    rst[i] = (int)(((llong)a[i] * (llong)b[i]) >> 32);
    rst=rst + c;
	return rst;
}
int8 mad_hi(int8 a, int8 b, int8 c){
	int8 rst;
  for (int i = 0; i < 8; i++)
    rst[i] = (int)(((llong)a[i] * (llong)b[i]) >> 32);
    rst=rst + c;
	return rst;
}
int16 mad_hi(int16 a, int16 b, int16 c){
	int16 rst;
  for (int i = 0; i < 16; i++)
    rst[i] = (int)(((llong)a[i] * (llong)b[i]) >> 32);
    rst=rst + c;
	return rst;
}


