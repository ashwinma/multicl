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
#include "mad_sat_internal.h"
// short
short mad_sat(short a, short b, short c) {
  llong rst = (llong)mad_sat_signed(a, b, c);
  rst = MIN(rst, (llong)SHRT_MAX);
  rst = MAX(rst, (llong)SHRT_MIN);
  return (short)rst;
}
short2 mad_sat(short2 a, short2 b, short2 c) {
  short2 rst;
  for (int i = 0; i < 2; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
short3 mad_sat(short3 a, short3 b, short3 c) {
  short3 rst;
  for (int i = 0; i < 3; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
short4 mad_sat(short4 a, short4 b, short4 c) {
  short4 rst;
  for (int i = 0; i < 4; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
short8 mad_sat(short8 a, short8 b, short8 c) {
  short8 rst;
  for (int i = 0; i < 8; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
short16 mad_sat(short16 a, short16 b, short16 c) {
  short16 rst;
  for (int i = 0; i < 16; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}


