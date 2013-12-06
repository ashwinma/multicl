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

#ifndef POPCOUNT_H_
#define POPCOUNT_H_

#include <cl_cpu_types.h>

schar  popcount(schar x);
char2  popcount(char2 x);
char3  popcount(char3 x);
char4  popcount(char4 x);
char8  popcount(char8 x);
char16 popcount(char16 x);

uchar   popcount(uchar x);
uchar2  popcount(uchar2 x);
uchar3  popcount(uchar3 x);
uchar4  popcount(uchar4 x);
uchar8  popcount(uchar8 x);
uchar16 popcount(uchar16 x);

short   popcount(short x);
short2  popcount(short2 x);
short3  popcount(short3 x);
short4  popcount(short4 x);
short8  popcount(short8 x);
short16 popcount(short16 x);

ushort   popcount(ushort x);
ushort2  popcount(ushort2 x);
ushort3  popcount(ushort3 x);
ushort4  popcount(ushort4 x);
ushort8  popcount(ushort8 x);
ushort16 popcount(ushort16 x);

int   popcount(int x);
int2  popcount(int2 x);
int3  popcount(int3 x);
int4  popcount(int4 x);
int8  popcount(int8 x);
int16 popcount(int16 x);

uint   popcount(uint x);
uint2  popcount(uint2 x);
uint3  popcount(uint3 x);
uint4  popcount(uint4 x);
uint8  popcount(uint8 x);
uint16 popcount(uint16 x);

llong  popcount(llong x);
long2  popcount(long2 x);
long3  popcount(long3 x);
long4  popcount(long4 x);
long8  popcount(long8 x);
long16 popcount(long16 x);

ullong  popcount(ullong x);
ulong2  popcount(ulong2 x);
ulong3  popcount(ulong3 x);
ulong4  popcount(ulong4 x);
ulong8  popcount(ulong8 x);
ulong16 popcount(ulong16 x);

#endif
