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

#ifndef _cl_kernel_constants_h_
#define _cl_kernel_constants_h_

#define CHAR_BIT         8
#define SCHAR_MAX        127
#define SCHAR_MIN        (-127-1)
#define CHAR_MAX         SCHAR_MAX
#define CHAR_MIN         SCHAR_MIN
#define UCHAR_MAX        255
#define SHRT_MAX         32767
#define SHRT_MIN         (-32767-1)
#define USHRT_MAX        65535
#define INT_MAX          2147483647
#define INT_MIN          (-2147483647-1)
#define UINT_MAX         0xffffffffU
#define LONG_MAX         0x7FFFFFFFFFFFFFFFLL
#define LONG_MIN         (-0x7FFFFFFFFFFFFFFFLL - 1LL)
#define ULONG_MAX        0xFFFFFFFFFFFFFFFFULL

#define FLT_DIG          6
#define FLT_MANT_DIG     24
#define FLT_MAX_10_EXP   +38
#define FLT_MAX_EXP      +128
#define FLT_MIN_10_EXP   -37
#define FLT_MIN_EXP      -125
#define FLT_RADIX        2

#ifdef TARGET_MACH_C64XP
#define FLT_MAX          340282346638528859811704183484516925440.0f
#define FLT_MIN          0.000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517f
#define FLT_EPSILON      0.000000119209289550781250f
#else
#define FLT_MAX          0x1.fffffep127f
#define FLT_MIN          0x1.0p-126f
#define FLT_EPSILON      0x1.0p-23f
#endif

#undef FP_ILOGB0
#define FP_ILOGB0        INT_MIN
#undef FP_ILOGBNAN
#define FP_ILOGBNAN      INT_MAX

#endif /* _cl_kernel_constants_h_ */
