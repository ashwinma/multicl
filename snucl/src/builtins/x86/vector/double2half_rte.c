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

#include "half_util.h"

half double2half_rte(double f)
{
  union { uint64_t u; double f; } fv, ft;
  fv.f = f;

  uint16_t sign     = (fv.u >> 48) & 0x8000;
  uint32_t exponent = (fv.u >> 52) & 0x7FF;
  uint64_t mantissa = fv.u & 0x000FFFFFFFFFFFFFULL;

  fv.u = fv.u & 0x7FFFFFFFFFFFFFFFULL;

  // infinity, NaN
  if (exponent == 0x7FF) {
    if (mantissa == 0) {
      // infinity
      return (half)(sign | 0x7C00);
    } else {
      // NaN
      return (half)(sign | 0x7E00 | (mantissa>>13));
    }
  }

  // overflow
  if (fv.f >= 0x1.FFEp15)
    return (half)(sign | 0x7C00);

  // underflow
  if (fv.f <= 0x1.0p-25)
    return (half)sign;

  // very small
  if (fv.f < 0x1.8p-24)
    return (half)(sign | 1);

  // subnormal
  if (fv.f < 0x1.0p-14) {
    fv.f = fv.f * 0x1.0p-1050;
    return (half)(sign | fv.u);
  }

  // normal
  ft.f = f * 0x1.0p42;
  ft.u = ft.u & 0x7FF0000000000000ULL;
  ft.f = ((fv.f + ft.f) - ft.f) * 0x1.0p-1008;
  return (half)(sign | (uint16_t)(ft.u >> 42));
}


