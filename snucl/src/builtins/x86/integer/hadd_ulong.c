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


// ullong
ullong hadd(ullong x, ullong y){
  ullong overflow = (x & 0x1) + (y & 0x1);
	return (x >> 1) + (y >> 1) + (overflow >> 1);
}
ulong2 hadd(ulong2 x, ulong2 y){
  ullong overflow;
	ulong2 rst;
  for (int i = 0; i < 2; i++) {
    overflow = (x[i] & 0x1) + (y[i] & 0x1);
    rst[i] = (x[i] >> 1) + (y[i] >> 1) + (overflow >> 1);
  }
  return rst;
}
ulong3 hadd(ulong3 x, ulong3 y){
  ullong overflow;
	ulong3 rst;
  for (int i = 0; i < 3; i++) {
    overflow = (x[i] & 0x1) + (y[i] & 0x1);
    rst[i] = (x[i] >> 1) + (y[i] >> 1) + (overflow >> 1);
  }
  return rst;
}
ulong4 hadd(ulong4 x, ulong4 y){
  ullong overflow;
	ulong4 rst;
  for (int i = 0; i < 4; i++) {
    overflow = (x[i] & 0x1) + (y[i] & 0x1);
    rst[i] = (x[i] >> 1) + (y[i] >> 1) + (overflow >> 1);
  }
  return rst;
}
ulong8 hadd(ulong8 x, ulong8 y){
  ullong overflow;
	ulong8 rst;
  for (int i = 0; i < 8; i++) {
    overflow = (x[i] & 0x1) + (y[i] & 0x1);
    rst[i] = (x[i] >> 1) + (y[i] >> 1) + (overflow >> 1);
  }
  return rst;
}
ulong16 hadd(ulong16 x, ulong16 y){
  ullong overflow;
	ulong16 rst;
  for (int i = 0; i < 16; i++) {
    overflow = (x[i] & 0x1) + (y[i] & 0x1);
    rst[i] = (x[i] >> 1) + (y[i] >> 1) + (overflow >> 1);
  }
  return rst;
}

