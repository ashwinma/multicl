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

#ifndef __CL_VECTOR_USHORTN_H
#define __CL_VECTOR_USHORTN_H

#include "cl_scalar.h"

struct __cl_ushort2 {
  ushort v[2] __attribute__((aligned(4)));

  inline __cl_ushort2& operator=(ushort x) {
    v[0] = x; v[1] = x;
    return *this;
  }

  inline const ushort& operator[](const int i) const {
    return v[i];
  }

  inline ushort& operator[](const int i) {
    return v[i];
  }
};


struct __cl_ushort3 {
  ushort v[4] __attribute__((aligned(8)));

  inline __cl_ushort3& operator=(ushort x) {
    v[0] = x; v[1] = x; v[2] = x;
    return *this;
  }

  inline const ushort& operator[](const int i) const {
    return v[i];
  }

  inline ushort& operator[](const int i) {
    return v[i];
  }
};

struct __cl_ushort4 {
  ushort v[4] __attribute__((aligned(8)));

  inline __cl_ushort4& operator=(ushort x) {
    v[0] = x; v[1] = x; v[2] = x; v[3] = x;
    return *this;
  }

  inline const ushort& operator[](const int i) const {
    return v[i];
  }

  inline ushort& operator[](const int i) {
    return v[i];
  }
};

struct __cl_ushort8 {
  ushort v[8] __attribute__((aligned(16)));

  inline __cl_ushort8& operator=(ushort x) {
    v[0] = x; v[1] = x; v[2] = x; v[3] = x;
    v[4] = x; v[5] = x; v[6] = x; v[7] = x;
    return *this;
  }

  inline const ushort& operator[](const int i) const {
    return v[i];
  }

  inline ushort& operator[](const int i) {
    return v[i];
  }
};

struct __cl_ushort16 {
  ushort v[16] __attribute__((aligned(32)));

  inline __cl_ushort16& operator=(ushort x) {
    v[0] = x; v[1] = x; v[2] = x; v[3] = x;
    v[4] = x; v[5] = x; v[6] = x; v[7] = x;
    v[8] = x; v[9] = x; v[10] = x; v[11] = x;
    v[12] = x; v[13] = x; v[14] = x; v[15] = x;
    return *this;
  }

  inline const ushort& operator[](const int i) const {
    return v[i];
  }

  inline ushort& operator[](const int i) {
    return v[i];
  }
};


/* vector literal */
#define __CL_USHORT2(s0,s1)          (ushort2){s0,s1}
#define __CL_USHORT3(s0,s1,s2)       (ushort3){s0,s1,s2,0}
#define __CL_USHORT4(s0,s1,s2,s3)    (ushort4){s0,s1,s2,s3}
#define __CL_USHORT8(s0,s1,s2,s3,s4,s5,s6,s7)   \
                                     (ushort8){s0,s1,s2,s3,s4,s5,s6,s7}
#define __CL_USHORT16(s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf)   \
             (ushort16){s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,sa,sb,sc,sd,se,sf}



#endif //__CL_VECTOR_USHORTN_H

