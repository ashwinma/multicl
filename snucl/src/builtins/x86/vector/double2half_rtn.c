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

half double2half_rtn(double f)
{
  union { uint64_t u; double f; } fv;
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
  if (f >= 0x1.0p16)
    return (half)((f == (DBL_MAX + DBL_MAX)) ? 0x7C00 : 0x7BFF);
  if (f < -0x1.FFCp15)
    return (half)0xFC00;

  // underflow
  if (fv.f < 0x1.0p-24)
    return (half)(f < 0 ? 0x8001 : sign);

  // subnormal
  if (fv.f < 0x1.0p-14) {
    fv.f = fv.f * 0x1.0p24;
    uint16_t u = (uint16_t)fv.f;
    return (half)(sign | (u + ((double)u != fv.f && f < 0.0f)));
  }

  // normal
  fv.f = f;
  fv.u = fv.u & 0xFFFFFC0000000000ULL;
  if (fv.f > f && sign) fv.u += 0x0000040000000000ULL;
  fv.u -= 0x3F00000000000000ULL;
  return (half)(sign | (uint16_t)(fv.u >> 42));
}

