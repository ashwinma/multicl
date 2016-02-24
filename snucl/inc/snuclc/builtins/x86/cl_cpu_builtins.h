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

#ifndef __CL_SPU_BUILTINS_H
#define __CL_SPU_BUILTINS_H

/* 6.2.3 Explicit Conversions */
#include "conversion/cl_builtins_conversion.h"

/* 6.2.4.2 Reinterpreting Types Using as_typen() */
#include "reinterpreting/cl_builtins_reinterpreting.h"

/* 9.5 - 9.7 Atomic Functions */
#include "atomic/cl_builtins_atomic.h"

/* 6.11.2 Math Functions */
#include "math/cl_builtins_math.h"

/* 6.11.3 Integer Functions */
#include "integer/cl_builtins_integer.h"

/* 6.11.4 Common Functions */
#include "common/cl_builtins_common.h"

/* 6.11.5 Geogetric Functions */
#include "geometric/cl_builtins_geometric.h"

/* 6.11.6 Relational Functions */
#include "relational/cl_builtins_relational.h"

/* 6.11.7 Vector Load and Store Functions */
#include "vector/cl_builtins_vector.h"

/* 6.11.11 Async Copies from Global to Local Memory, Local to Global Memory,
 *         and Prefetch */
#include "async/cl_builtins_async.h"

/* 6.12.12 Miscellaneous Vector Functions */
#include "miscvec/cl_builtins_miscvec.h"

/* 6.12.13 Printf Functions */
#include "printf/cl_builtins_printf.h"

/* 6.12.14 Image Read and Write Functions */
#include "image/cl_builtins_image.h"

#endif //__CL_SPU_BUILTINS_H

