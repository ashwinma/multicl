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

#ifndef __CL_OPS_UTIL_H
#define __CL_OPS_UTIL_H

#include <cl_cpu_ops.h>

#define __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x,op,y)     {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y[1])}}

#define __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y)}}

#define __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[1])}}


#define __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x,op,y)     {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y[1]),	\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y[2])}}
#define __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y),	\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y)}}
#define __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[1]),	\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[2])}}

#define __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x,op,y)     {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y[1]),	\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y[2]) , __CL_SAFE_INT_DIV_ZERO(x[3], op, y[3])}}
#define __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y),	\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y) , __CL_SAFE_INT_DIV_ZERO(x[3], op, y)}}
#define __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[1]),	\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[2]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[3])}}

#define __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x,op,y)     {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y[1]),\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y[2]) , __CL_SAFE_INT_DIV_ZERO(x[3], op, y[3]),\
													 __CL_SAFE_INT_DIV_ZERO(x[4], op, y[4]) , __CL_SAFE_INT_DIV_ZERO(x[5], op, y[5]),\
													 __CL_SAFE_INT_DIV_ZERO(x[6], op, y[6]) , __CL_SAFE_INT_DIV_ZERO(x[7], op, y[7])}}

#define __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y),	\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y) , __CL_SAFE_INT_DIV_ZERO(x[3], op, y),	\
													 __CL_SAFE_INT_DIV_ZERO(x[4], op, y) , __CL_SAFE_INT_DIV_ZERO(x[5], op, y),	\
													 __CL_SAFE_INT_DIV_ZERO(x[6], op, y) , __CL_SAFE_INT_DIV_ZERO(x[7], op, y)}}

#define __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x,op,y)      {{__CL_SAFE_INT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[1]),	\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[2]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[3]),	\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[4]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[5]),	\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[6]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[7])}}

#define __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x,op,y)   {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y[1]),		\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y[2]) , __CL_SAFE_INT_DIV_ZERO(x[3], op, y[3]),		\
													 __CL_SAFE_INT_DIV_ZERO(x[4], op, y[4]) , __CL_SAFE_INT_DIV_ZERO(x[5], op, y[5]),		\
													 __CL_SAFE_INT_DIV_ZERO(x[6], op, y[6]) , __CL_SAFE_INT_DIV_ZERO(x[7], op, y[7]),		\
													 __CL_SAFE_INT_DIV_ZERO(x[8], op, y[8]) , __CL_SAFE_INT_DIV_ZERO(x[9], op, y[9]),		\
													 __CL_SAFE_INT_DIV_ZERO(x[10], op, y[10]) , __CL_SAFE_INT_DIV_ZERO(x[11], op, y[11]),	\
													 __CL_SAFE_INT_DIV_ZERO(x[12], op, y[12]) , __CL_SAFE_INT_DIV_ZERO(x[13], op, y[13]),	\
													 __CL_SAFE_INT_DIV_ZERO(x[14], op, y[14]) , __CL_SAFE_INT_DIV_ZERO(x[15], op, y[15])}}

#define __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x,op,y)     {{__CL_SAFE_INT_DIV_ZERO(x[0], op, y) , __CL_SAFE_INT_DIV_ZERO(x[1], op, y),		\
													 __CL_SAFE_INT_DIV_ZERO(x[2], op, y) , __CL_SAFE_INT_DIV_ZERO(x[3], op, y),		\
													 __CL_SAFE_INT_DIV_ZERO(x[4], op, y) , __CL_SAFE_INT_DIV_ZERO(x[5], op, y),		\
													 __CL_SAFE_INT_DIV_ZERO(x[6], op, y) , __CL_SAFE_INT_DIV_ZERO(x[7], op, y),		\
													 __CL_SAFE_INT_DIV_ZERO(x[8], op, y) , __CL_SAFE_INT_DIV_ZERO(x[9], op, y),		\
													 __CL_SAFE_INT_DIV_ZERO(x[10], op, y) , __CL_SAFE_INT_DIV_ZERO(x[11], op, y),	\
													 __CL_SAFE_INT_DIV_ZERO(x[12], op, y) , __CL_SAFE_INT_DIV_ZERO(x[13], op, y),	\
													 __CL_SAFE_INT_DIV_ZERO(x[14], op, y) , __CL_SAFE_INT_DIV_ZERO(x[15], op, y)}}

#define __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x,op,y)     {{__CL_SAFE_INT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[1]),		\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[2]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[3]),		\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[4]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[5]),		\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[6]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[7]),		\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[8]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[9]),		\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[10]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[11]),	\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[12]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[13]),	\
													 __CL_SAFE_INT_DIV_ZERO(x, op, y[14]) , __CL_SAFE_INT_DIV_ZERO(x, op, y[15])}}

#define __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x,op,y)     {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y[1])}}

#define __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y)}}

#define __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[1])}}


#define __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x,op,y)     {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y[1]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y[2])}}
#define __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y)}}
#define __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[1]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[2])}}

#define __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x,op,y)     {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y[1]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y[2]) , __CL_SAFE_UINT_DIV_ZERO(x[3], op, y[3])}}
#define __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[3], op, y)}}
#define __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[1]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[2]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[3])}}

#define __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x,op,y)     {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y[1]),\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y[2]) , __CL_SAFE_UINT_DIV_ZERO(x[3], op, y[3]),\
													 __CL_SAFE_UINT_DIV_ZERO(x[4], op, y[4]) , __CL_SAFE_UINT_DIV_ZERO(x[5], op, y[5]),\
													 __CL_SAFE_UINT_DIV_ZERO(x[6], op, y[6]) , __CL_SAFE_UINT_DIV_ZERO(x[7], op, y[7])}}

#define __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[3], op, y),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[4], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[5], op, y),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[6], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[7], op, y)}}

#define __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x,op,y)      {{__CL_SAFE_UINT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[1]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[2]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[3]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[4]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[5]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[6]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[7])}}

#define __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x,op,y)   {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y[1]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y[2]) , __CL_SAFE_UINT_DIV_ZERO(x[3], op, y[3]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[4], op, y[4]) , __CL_SAFE_UINT_DIV_ZERO(x[5], op, y[5]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[6], op, y[6]) , __CL_SAFE_UINT_DIV_ZERO(x[7], op, y[7]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[8], op, y[8]) , __CL_SAFE_UINT_DIV_ZERO(x[9], op, y[9]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[10], op, y[10]) , __CL_SAFE_UINT_DIV_ZERO(x[11], op, y[11]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[12], op, y[12]) , __CL_SAFE_UINT_DIV_ZERO(x[13], op, y[13]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[14], op, y[14]) , __CL_SAFE_UINT_DIV_ZERO(x[15], op, y[15])}}

#define __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x,op,y)     {{__CL_SAFE_UINT_DIV_ZERO(x[0], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[1], op, y),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[2], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[3], op, y),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[4], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[5], op, y),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[6], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[7], op, y),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[8], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[9], op, y),		\
													 __CL_SAFE_UINT_DIV_ZERO(x[10], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[11], op, y),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[12], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[13], op, y),	\
													 __CL_SAFE_UINT_DIV_ZERO(x[14], op, y) , __CL_SAFE_UINT_DIV_ZERO(x[15], op, y)}}

#define __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x,op,y)     {{__CL_SAFE_UINT_DIV_ZERO(x, op, y[0]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[1]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[2]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[3]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[4]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[5]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[6]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[7]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[8]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[9]),		\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[10]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[11]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[12]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[13]),	\
													 __CL_SAFE_UINT_DIV_ZERO(x, op, y[14]) , __CL_SAFE_UINT_DIV_ZERO(x, op, y[15])}}


#define __CL_V2_OP_V2(x,op,y)     {{x[0] op y[0], x[1] op y[1]}}
#define __CL_V2_OP_S(x,op,y)      {{x[0] op y,    x[1] op y}}
#define __CL_S_OP_V2(x,op,y)      {{x    op y[0], x    op y[1]}}
#define __CL_V3_OP_V3(x,op,y)     {{x[0] op y[0], x[1] op y[1], \
                                   x[2] op y[2]}}
#define __CL_V3_OP_S(x,op,y)      {{x[0] op y,    x[1] op y, \
                                   x[2] op y}}
#define __CL_S_OP_V3(x,op,y)      {{x    op y[0], x    op y[1], \
                                   x    op y[2]}}

#define __CL_V4_OP_V4(x,op,y)     {{x[0] op y[0], x[1] op y[1], \
                                   x[2] op y[2], x[3] op y[3]}}
#define __CL_V4_OP_S(x,op,y)      {{x[0] op y,    x[1] op y, \
                                   x[2] op y,    x[3] op y}}
#define __CL_S_OP_V4(x,op,y)      {{x    op y[0], x    op y[1], \
                                   x    op y[2], x    op y[3]}}
#define __CL_V8_OP_V8(x,op,y)     {{x[0] op y[0], x[1] op y[1], \
                                   x[2] op y[2], x[3] op y[3], \
                                   x[4] op y[4], x[5] op y[5], \
                                   x[6] op y[6], x[7] op y[7]}}
#define __CL_V8_OP_S(x,op,y)      {{x[0] op y,    x[1] op y, \
                                   x[2] op y,    x[3] op y, \
                                   x[4] op y,    x[5] op y, \
                                   x[6] op y,    x[7] op y}}
#define __CL_S_OP_V8(x,op,y)      {{x    op y[0], x    op y[1], \
                                   x    op y[2], x    op y[3], \
                                   x    op y[4], x    op y[5], \
                                   x    op y[6], x    op y[7]}}
#define __CL_V16_OP_V16(x,op,y)   {{x[0] op y[0], x[1] op y[1], \
                                   x[2] op y[2], x[3] op y[3], \
                                   x[4] op y[4], x[5] op y[5], \
                                   x[6] op y[6], x[7] op y[7], \
                                   x[8] op y[8], x[9] op y[9], \
                                   x[10] op y[10], x[11] op y[11], \
                                   x[12] op y[12], x[13] op y[13], \
                                   x[14] op y[14], x[15] op y[15]}}
#define __CL_V16_OP_S(x,op,y)     {{x[0] op y,    x[1] op y, \
                                   x[2] op y,    x[3] op y, \
                                   x[4] op y,    x[5] op y, \
                                   x[6] op y,    x[7] op y, \
                                   x[8] op y,    x[9] op y, \
                                   x[10] op y,   x[11] op y, \
                                   x[12] op y,   x[13] op y, \
                                   x[14] op y,   x[15] op y}}
#define __CL_S_OP_V16(x,op,y)     {{x op y[0],    x op y[1], \
                                   x op y[2],    x op y[3], \
                                   x op y[4],    x op y[5], \
                                   x op y[6],    x op y[7], \
                                   x op y[8],    x op y[9], \
                                   x op y[10],   x op y[11], \
                                   x op y[12],   x op y[13], \
                                   x op y[14],   x op y[15]}}



#define __CL_OP_V2(op,x)          {{op x[0], op x[1]}}
#define __CL_OP_V3(op,x)          {{op x[0], op x[1], op x[2]}}
#define __CL_OP_V4(op,x)          {{op x[0], op x[1], op x[2], op x[3]}}
#define __CL_OP_V8(op,x)          {{op x[0], op x[1], op x[2], op x[3], \
                                   op x[4], op x[5], op x[6], op x[7]}}
#define __CL_OP_V16(op,x)         {{op x[0], op x[1], op x[2], op x[3], \
                                   op x[4], op x[5], op x[6], op x[7], \
                                   op x[8], op x[9], op x[10],op x[11], \
                                   op x[12],op x[13],op x[14],op x[15]}}

#define __CL_V2_POST_OP(x,op)     x[0]op; x[1]op
#define __CL_V3_POST_OP(x,op)     x[0]op; x[1]op; x[2]op
#define __CL_V4_POST_OP(x,op)     x[0]op; x[1]op; x[2]op; x[3]op
#define __CL_V8_POST_OP(x,op)     x[0]op; x[1]op; x[2]op; x[3]op; \
                                  x[4]op; x[5]op; x[6]op; x[7]op
#define __CL_V16_POST_OP(x,op)    x[0]op; x[1]op; x[2]op; x[3]op; \
                                  x[4]op; x[5]op; x[6]op; x[7]op; \
                                  x[8]op; x[9]op; x[10]op; x[11]op; \
                                  x[12]op; x[13]op; x[14]op; x[15]op

#define __CL_PRE_OP_V2(op,x)      op x[0]; op x[1]
#define __CL_PRE_OP_V3(op,x)      op x[0]; op x[1]; op x[2]
#define __CL_PRE_OP_V4(op,x)      op x[0]; op x[1]; op x[2]; op x[3]
#define __CL_PRE_OP_V8(op,x)      op x[0]; op x[1]; op x[2]; op x[3]; \
                                  op x[4]; op x[5]; op x[6]; op x[7]
#define __CL_PRE_OP_V16(op,x)     op x[0]; op x[1]; op x[2]; op x[3]; \
                                  op x[4]; op x[5]; op x[6]; op x[7]; \
                                  op x[8]; op x[9]; op x[10]; op x[11]; \
                                  op x[12]; op x[13]; op x[14]; op x[15]

#define __CL_V2_LOP_V2(x,op,y)    {{-(x[0] op y[0]), -(x[1] op y[1])}}
#define __CL_V2_LOP_S(x,op,y)     {{-(x[0] op y),    -(x[1] op y)}}
#define __CL_S_LOP_V2(x,op,y)     {{-(x op y[0]),    -(x op y[1])}}

#define __CL_V3_LOP_V3(x,op,y)    {{-(x[0] op y[0]), -(x[1] op y[1]), \
                                   -(x[2] op y[2])}}
#define __CL_V3_LOP_S(x,op,y)     {{-(x[0] op y),    -(x[1] op y), \
                                   -(x[2] op y)}}
#define __CL_S_LOP_V3(x,op,y)     {{-(x op y[0]),    -(x op y[1]), \
                                   -(x op y[2])}}

#define __CL_V4_LOP_V4(x,op,y)    {{-(x[0] op y[0]), -(x[1] op y[1]), \
                                   -(x[2] op y[2]), -(x[3] op y[3])}}
#define __CL_V4_LOP_S(x,op,y)     {{-(x[0] op y),    -(x[1] op y), \
                                   -(x[2] op y),    -(x[3] op y)}}
#define __CL_S_LOP_V4(x,op,y)     {{-(x op y[0]),    -(x op y[1]), \
                                   -(x op y[2]),    -(x op y[3])}}
#define __CL_V8_LOP_V8(x,op,y)    {{-(x[0] op y[0]), -(x[1] op y[1]), \
                                   -(x[2] op y[2]), -(x[3] op y[3]), \
                                   -(x[4] op y[4]), -(x[5] op y[5]), \
                                   -(x[6] op y[6]), -(x[7] op y[7])}}
#define __CL_V8_LOP_S(x,op,y)     {{-(x[0] op y),    -(x[1] op y), \
                                   -(x[2] op y),    -(x[3] op y), \
                                   -(x[4] op y),    -(x[5] op y), \
                                   -(x[6] op y),    -(x[7] op y)}}
#define __CL_S_LOP_V8(x,op,y)     {{-(x op y[0]),    -(x op y[1]), \
                                   -(x op y[2]),    -(x op y[3]), \
                                   -(x op y[4]),    -(x op y[5]), \
                                   -(x op y[6]),    -(x op y[7])}}
#define __CL_V16_LOP_V16(x,op,y)  {{-(x[0] op y[0]), -(x[1] op y[1]), \
                                   -(x[2] op y[2]), -(x[3] op y[3]), \
                                   -(x[4] op y[4]), -(x[5] op y[5]), \
                                   -(x[6] op y[6]), -(x[7] op y[7]), \
                                   -(x[8] op y[8]), -(x[9] op y[9]), \
                                   -(x[10] op y[10]), -(x[11] op y[11]), \
                                   -(x[12] op y[12]), -(x[13] op y[13]), \
                                   -(x[14] op y[14]), -(x[15] op y[15])}}
#define __CL_V16_LOP_S(x,op,y)    {{-(x[0] op y),    -(x[1] op y), \
                                   -(x[2] op y),    -(x[3] op y), \
                                   -(x[4] op y),    -(x[5] op y), \
                                   -(x[6] op y),    -(x[7] op y), \
                                   -(x[8] op y),    -(x[9] op y), \
                                   -(x[10] op y),   -(x[11] op y), \
                                   -(x[12] op y),   -(x[13] op y), \
                                   -(x[14] op y),   -(x[15] op y)}}
#define __CL_S_LOP_V16(x,op,y)    {{-(x op y[0]),    -(x op y[1]), \
                                   -(x op y[2]),    -(x op y[3]), \
                                   -(x op y[4]),    -(x op y[5]), \
                                   -(x op y[6]),    -(x op y[7]), \
                                   -(x op y[8]),    -(x op y[9]), \
                                   -(x op y[10]),   -(x op y[11]), \
                                   -(x op y[12]),   -(x op y[13]), \
                                   -(x op y[14]),   -(x op y[14])}}

#define __CL_LOP_V2(op,x)         {{-(op x[0]), -(op x[1])}}
#define __CL_LOP_V3(op,x)         {{-(op x[0]), -(op x[1]), \
                                   -(op x[2])}}
#define __CL_LOP_V4(op,x)         {{-(op x[0]), -(op x[1]), \
                                   -(op x[2]), -(op x[3])}}
#define __CL_LOP_V8(op,x)         {{-(op x[0]), -(op x[1]), \
                                   -(op x[2]), -(op x[3]), \
                                   -(op x[4]), -(op x[5]), \
                                   -(op x[6]), -(op x[7])}}
#define __CL_LOP_V16(op,x)        {{-(op x[0]), -(op x[1]), \
                                   -(op x[2]), -(op x[3]), \
                                   -(op x[4]), -(op x[5]), \
                                   -(op x[6]), -(op x[7]), \
                                   -(op x[8]), -(op x[9]), \
                                   -(op x[10]),-(op x[11]), \
                                   -(op x[12]),-(op x[13]), \
                                   -(op x[14]),-(op x[15])}}

#define __CL_V_OP_ASSIGN_V(t,x,op,y)    t z = x op y; x = z; return z

#endif //__CL_OPS_UTIL_H

