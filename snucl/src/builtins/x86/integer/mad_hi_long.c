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
#include "mul64.h"

// llong
llong mad_hi(llong a, llong b, llong c){
  llong s_hi;
  llong s_lo;
  mul_long(a, b, &s_hi, &s_lo);
  return s_hi + c;
}
long2 mad_hi(long2 a, long2 b, long2 c){
	long2 rst;
  llong s_hi;
  llong s_lo;
  for (int i = 0; i < 2; i++) {
    mul_long(a[i], b[i], &s_hi, &s_lo);
    rst[i] = s_hi;
  }
  return rst + c;
}
long3 mad_hi(long3 a, long3 b, long3 c){
	long3 rst;
  llong s_hi;
  llong s_lo;
  for (int i = 0; i < 3; i++) {
    mul_long(a[i], b[i], &s_hi, &s_lo);
    rst[i] = s_hi;
  }
  return rst + c;
}
long4 mad_hi(long4 a, long4 b, long4 c){
	long4 rst;
  llong s_hi;
  llong s_lo;
  for (int i = 0; i < 4; i++) {
    mul_long(a[i], b[i], &s_hi, &s_lo);
    rst[i] = s_hi;
  }
  return rst + c;
}
long8 mad_hi(long8 a, long8 b, long8 c){
	long8 rst;
  llong s_hi;
  llong s_lo;
  for (int i = 0; i < 8; i++) {
    mul_long(a[i], b[i], &s_hi, &s_lo);
    rst[i] = s_hi;
  }
  return rst + c;
}
long16 mad_hi(long16 a, long16 b, long16 c){
	long16 rst;
  llong s_hi;
  llong s_lo;
  for (int i = 0; i < 16; i++) {
    mul_long(a[i], b[i], &s_hi, &s_lo);
    rst[i] = s_hi;
  }
  return rst + c;
}


