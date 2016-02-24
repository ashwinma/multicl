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

// ullong
ullong mad_hi(ullong a, ullong b, ullong c){
  ullong u_hi, u_lo;
  mul_ulong(a, b, &u_hi, &u_lo);
  return u_hi + c;
}
ulong2 mad_hi(ulong2 a, ulong2 b, ulong2 c){
	ulong2 rst;
  ullong u_hi, u_lo;
  for (int i = 0; i < 2; i++) {
    mul_ulong(a[i], b[i], &u_hi, &u_lo);
    rst[i] = u_hi;
  }
  return rst + c;
}
ulong3 mad_hi(ulong3 a, ulong3 b, ulong3 c){
	ulong3 rst;
  ullong u_hi, u_lo;
  for (int i = 0; i < 3; i++) {
    mul_ulong(a[i], b[i], &u_hi, &u_lo);
    rst[i] = u_hi;
  }
  return rst + c;
}
ulong4 mad_hi(ulong4 a, ulong4 b, ulong4 c){
	ulong4 rst;
  ullong u_hi, u_lo;
  for (int i = 0; i < 4; i++) {
    mul_ulong(a[i], b[i], &u_hi, &u_lo);
    rst[i] = u_hi;
  }
  return rst + c;
}
ulong8 mad_hi(ulong8 a, ulong8 b, ulong8 c){
	ulong8 rst;
  ullong u_hi, u_lo;
  for (int i = 0; i < 8; i++) {
    mul_ulong(a[i], b[i], &u_hi, &u_lo);
    rst[i] = u_hi;
  }
  return rst + c;
}
ulong16 mad_hi(ulong16 a, ulong16 b, ulong16 c){
	ulong16 rst;
  ullong u_hi, u_lo;
  for (int i = 0; i < 16; i++) {
    mul_ulong(a[i], b[i], &u_hi, &u_lo);
    rst[i] = u_hi;
  }
  return rst + c;
}
