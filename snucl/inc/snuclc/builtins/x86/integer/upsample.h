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

#ifndef UPSAMPLE_H_
#define UPSAMPLE_H_

#include <cl_cpu_types.h>

//short. shortn
short   upsample(schar hi, uchar lo);
short2  upsample(char2 hi, uchar2 lo);
short3  upsample(char3 hi, uchar3 lo);
short4  upsample(char4 hi, uchar4 lo);
short8  upsample(char8 hi, uchar8 lo);
short16 upsample(char16 hi, uchar16 lo);

ushort   upsample(uchar hi, uchar lo);
ushort2  upsample(uchar2 hi, uchar2 lo);
ushort3  upsample(uchar3 hi, uchar3 lo);
ushort4  upsample(uchar4 hi, uchar4 lo);
ushort8  upsample(uchar8 hi, uchar8 lo);
ushort16 upsample(uchar16 hi, uchar16 lo);

//int, uint
int upsample(short hi, ushort lo);
int2 upsample(short2 hi, ushort2 lo);
int3 upsample(short3 hi, ushort3 lo);
int4 upsample(short4 hi, ushort4 lo);
int8 upsample(short8 hi, ushort8 lo);
int16 upsample(short16 hi, ushort16 lo);

uint   upsample(ushort hi, ushort lo);
uint2  upsample(ushort2 hi, ushort2 lo);
uint3  upsample(ushort3 hi, ushort3 lo);
uint4  upsample(ushort4 hi, ushort4 lo);
uint8  upsample(ushort8 hi, ushort8 lo);
uint16 upsample(ushort16 hi, ushort16 lo);

//llong, ullong
llong  upsample(int hi, uint lo);
long2  upsample(int2 hi, uint2 lo);
long3  upsample(int3 hi, uint3 lo);
long4  upsample(int4 hi, uint4 lo);
long8  upsample(int8 hi, uint8 lo);
long16 upsample(int16 hi, uint16 lo);

ullong  upsample(uint hi, uint lo);
ulong2  upsample(uint2 hi, uint2 lo);
ulong3  upsample(uint3 hi, uint3 lo);
ulong4  upsample(uint4 hi, uint4 lo);
ulong8  upsample(uint8 hi, uint8 lo);
ulong16 upsample(uint16 hi, uint16 lo);

#endif
