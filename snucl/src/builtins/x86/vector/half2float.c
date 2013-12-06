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

#include <cl_cpu_types.h>

// This algorithm is originated from 
// http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
float half2float(uint16_t h) {
  union { uint32_t u; float f; } rst;
  uint32_t exponent, mantissa;
  uint32_t offset = 1024;
  uint32_t i = h >> 10;

  // sign + exponent
  if (i == 0) {
    exponent = 0;
    offset = 0;
  } else if (i <= 30) {
    exponent = i << 23;
  } else if (i == 31) {
    exponent = 0x47800000;
  } else if (i == 32) {
    exponent = 0x80000000;
    offset = 0;
  } else if (i <= 62) {
    exponent = 0x80000000 + ((i - 32) << 23);
  } else { //i == 63
    exponent = 0xC7800000;
  }

  // mantissa
  offset += (h & 0x3FF);
  if (offset == 0) {
    mantissa = 0;
  } else if (offset <= 1023) {
    uint32_t m = offset << 13;
    uint32_t e = 0;
    while (!(m & 0x00800000)) {
      e -= 0x00800000;
      m <<= 1;
    }
    m &= ~0x00800000;
    e += 0x38800000;
    mantissa = m | e;
  } else { //offset <= 2047
    mantissa = 0x38000000 + ((offset - 1024) << 13);
  }

  rst.u = exponent + mantissa;
  return rst.f;
} 

