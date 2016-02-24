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
#include "mad_sat_internal.h"

ullong mad_sat_signed(llong a, llong b, llong c) {
  llong mul_hi;
  llong mul_lo, sum;

  mul_long(a, b, &mul_hi, &mul_lo);

  sum = mul_lo + c;

  if (c >= 0) {
    if (mul_lo > sum) {
      mul_hi++;
      if (LONG_MIN == mul_hi) {
        mul_hi = LONG_MAX;
        sum = ULONG_MAX;
      }
    }
  }
  else {
    if (mul_lo < sum) {
      mul_hi--;
      if (LONG_MAX == mul_hi) {
        mul_hi = LONG_MIN;
        sum = 0;
      }
    }
  }

  if (mul_hi > 0)
    sum = LONG_MAX;
  else if (mul_hi < -1)
    sum = LONG_MIN;

  return sum;
}

ullong mad_sat_unsigned(ullong a, ullong b, ullong c) {
  ullong mul_hi, mul_lo;

  mul_ulong(a, b, &mul_hi, &mul_lo);

  mul_lo += c;
  mul_hi += mul_lo < c;
  if (mul_hi) mul_lo = ULONG_MAX;

  return mul_lo;
}


