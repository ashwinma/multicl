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
// uchar
uchar mad_sat(uchar a, uchar b, uchar c) {
  ullong rst = mad_sat_unsigned(a, b, c);
  rst = MIN(rst, (ullong)UCHAR_MAX);
  return (uchar)rst;
}
uchar2 mad_sat(uchar2 a, uchar2 b, uchar2 c) {
  uchar2 rst;
  for (int i = 0; i < 2; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
uchar3 mad_sat(uchar3 a, uchar3 b, uchar3 c) {
  uchar3 rst;
  for (int i = 0; i < 3; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
uchar4 mad_sat(uchar4 a, uchar4 b, uchar4 c) {
  uchar4 rst;
  for (int i = 0; i < 4; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
uchar8 mad_sat(uchar8 a, uchar8 b, uchar8 c) {
  uchar8 rst;
  for (int i = 0; i < 8; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
uchar16 mad_sat(uchar16 a, uchar16 b, uchar16 c) {
  uchar16 rst;
  for (int i = 0; i < 16; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}

