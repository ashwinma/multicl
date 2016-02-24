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

#ifndef CLZ_H_
#define CLZ_H_

#include <cl_cpu_types.h>

schar  clz(schar x);
char2  clz(char2 x);
char3  clz(char3 x);
char4  clz(char4 x);
char8  clz(char8 x);
char16 clz(char16 x);

uchar   clz(uchar x);
uchar2  clz(uchar2 x);
uchar3  clz(uchar3 x);
uchar4  clz(uchar4 x);
uchar8  clz(uchar8 x);
uchar16 clz(uchar16 x);

short   clz(short x);
short2  clz(short2 x);
short3  clz(short3 x);
short4  clz(short4 x);
short8  clz(short8 x);
short16 clz(short16 x);

ushort   clz(ushort x);
ushort2  clz(ushort2 x);
ushort3  clz(ushort3 x);
ushort4  clz(ushort4 x);
ushort8  clz(ushort8 x);
ushort16 clz(ushort16 x);

int   clz(int x);
int2  clz(int2 x);
int3  clz(int3 x);
int4  clz(int4 x);
int8  clz(int8 x);
int16 clz(int16 x);

uint   clz(uint x);
uint2  clz(uint2 x);
uint3  clz(uint3 x);
uint4  clz(uint4 x);
uint8  clz(uint8 x);
uint16 clz(uint16 x);

llong  clz(llong x);
long2  clz(long2 x);
long3  clz(long3 x);
long4  clz(long4 x);
long8  clz(long8 x);
long16 clz(long16 x);

ullong  clz(ullong x);
ulong2  clz(ulong2 x);
ulong3  clz(ulong3 x);
ulong4  clz(ulong4 x);
ulong8  clz(ulong8 x);
ulong16 clz(ulong16 x);

#endif
