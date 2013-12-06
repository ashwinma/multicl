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
#include <miscvec/cl_builtins_miscvec.h>

/* shuffle() and shuffle2() */
#define SNUCL_SHUFFLE(rtype, itype, mtype)      \
rtype shuffle(itype x, mtype mask) {            \
  rtype ret;                                    \
  if (sizeof(ret[0]) != sizeof(mask[0]))        \
    return ret;                                 \
  size_t elems = vec_step(mask);                \
  for (int i=0; i<elems; ++i) {                 \
    uint index = mask[i];                       \
    ret[i] = x[index];                          \
  }                                             \
  return ret;                                   \
}

#define SNUCL_SHUFFLE2(rtype, itype, mtype)     \
rtype shuffle2(itype x, itype y, mtype mask) {  \
  rtype ret;                                    \
  if (sizeof(ret[0]) != sizeof(mask[0]))        \
    return ret;                                 \
  size_t elems = vec_step(mask);                \
  size_t src_elems = vec_step(x);               \
  for (int i=0; i<elems; ++i) {                 \
    uint index = mask[i];                       \
		if (index/src_elems)                        \
		  ret[i] = y[index%src_elems];              \
		else                                        \
		  ret[i] = x[index%src_elems];              \
  }                                             \
  return ret;                                   \
}

#include "shuffle_funcs.inc"
