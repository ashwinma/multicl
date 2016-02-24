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
// ushort
ushort mad_sat(ushort a, ushort b, ushort c) {
  ullong rst = mad_sat_unsigned(a, b, c);
  rst = MIN(rst, (ullong)USHRT_MAX);
  return (ushort)rst;
}
ushort2 mad_sat(ushort2 a, ushort2 b, ushort2 c) {
  ushort2 rst;
  for (int i = 0; i < 2; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
ushort3 mad_sat(ushort3 a, ushort3 b, ushort3 c) {
  ushort3 rst;
  for (int i = 0; i < 3; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
ushort4 mad_sat(ushort4 a, ushort4 b, ushort4 c) {
  ushort4 rst;
  for (int i = 0; i < 4; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
ushort8 mad_sat(ushort8 a, ushort8 b, ushort8 c) {
  ushort8 rst;
  for (int i = 0; i < 8; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}
ushort16 mad_sat(ushort16 a, ushort16 b, ushort16 c) {
  ushort16 rst;
  for (int i = 0; i < 16; i++)
    rst[i] = mad_sat(a[i], b[i], c[i]);
  return rst;
}


