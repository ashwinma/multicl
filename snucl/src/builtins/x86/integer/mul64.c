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

void mul_ulong(ullong x, ullong y, ullong *high, ullong *low) {
  ullong x_hi = x >> 32;
  ullong x_lo = x & 0xFFFFFFFF;
  ullong y_hi = y >> 32;
  ullong y_lo = y & 0xFFFFFFFF;

  ullong a = x_hi * y_hi;

  ullong b = x_hi * y_lo;
  ullong b_hi = b >> 32;
  ullong b_lo = b & 0xFFFFFFFF;

  ullong c = x_lo * y_hi;
  ullong c_hi = c >> 32;
  ullong c_lo = c & 0xFFFFFFFF;

  ullong d = x_lo * y_lo;
  ullong d_hi = d >> 32;
  ullong d_lo = d & 0xFFFFFFFF;

  ullong r_hi = b_lo + c_lo + d_hi;

  *low  = (r_hi << 32) + d_lo;
  *high = a + b_hi + c_hi + (r_hi >> 32);   // the last term is a carry.
}


void mul_long(llong x, llong y, llong *high, llong *low) {
  unsigned x_sign = x < 0;
  unsigned y_sign = y < 0;

  ullong ux = x_sign ? -x : x;
  ullong uy = y_sign ? -y : y;

  ullong uhigh, ulow;
  mul_ulong(ux, uy, &uhigh, &ulow);

  if (x_sign ^ y_sign) {
    // different sign bits
    ulow  = -ulow;
    uhigh = uhigh ^ (ullong)-1;
  }

  *low  = (llong)ulow;
  *high = (llong)uhigh;
}

