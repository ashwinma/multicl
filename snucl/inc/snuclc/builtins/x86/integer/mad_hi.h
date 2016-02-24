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

#ifndef MAD_HI_H_
#define MAD_HI_H_

#include <cl_cpu_types.h>

//schar, uchar
schar  mad_hi(schar a, schar b, schar c);
char2  mad_hi(char2 a, char2 b, char2 c);
char3  mad_hi(char3 a, char3 b, char3 c);
char4  mad_hi(char4 a, char4 b, char4 c);
char8  mad_hi(char8 a, char8 b, char8 c);
char16 mad_hi(char16 a, char16 b, char16 c);

uchar   mad_hi(uchar a, uchar b, uchar c);
uchar2  mad_hi(uchar2 a, uchar2 b, uchar2 c);
uchar3  mad_hi(uchar3 a, uchar3 b, uchar3 c);
uchar4  mad_hi(uchar4 a, uchar4 b, uchar4 c);
uchar8  mad_hi(uchar8 a, uchar8 b, uchar8 c);
uchar16 mad_hi(uchar16 a, uchar16 b, uchar16 c);

//short. shortn
short mad_hi(short a, short b, short c);
short2 mad_hi(short2 a, short2 b, short2 c);
short3 mad_hi(short3 a, short3 b, short3 c);
short4 mad_hi(short4 a, short4 b, short4 c);
short8 mad_hi(short8 a, short8 b, short8 c);
short16 mad_hi(short16 a, short16 b, short16 c);

ushort mad_hi(ushort a, ushort b, ushort c);
ushort2 mad_hi(ushort2 a, ushort2 b, ushort2 c);
ushort3 mad_hi(ushort3 a, ushort3 b, ushort3 c);
ushort4 mad_hi(ushort4 a, ushort4 b, ushort4 c);
ushort8 mad_hi(ushort8 a, ushort8 b, ushort8 c);
ushort16 mad_hi(ushort16 a, ushort16 b, ushort16 c);

//int, uint
int   mad_hi(int a, int b, int c);
int2  mad_hi(int2 a, int2 b, int2 c);
int3  mad_hi(int3 a, int3 b, int3 c);
int4  mad_hi(int4 a, int4 b, int4 c);
int8  mad_hi(int8 a, int8 b, int8 c);
int16 mad_hi(int16 a, int16 b, int16 c);

uint   mad_hi(uint a, uint b, uint c);
uint2  mad_hi(uint2 a, uint2 b, uint2 c);
uint3  mad_hi(uint3 a, uint3 b, uint3 c);
uint4  mad_hi(uint4 a, uint4 b, uint4 c);
uint8  mad_hi(uint8 a, uint8 b, uint8 c);
uint16 mad_hi(uint16 a, uint16 b, uint16 c);

//llong, ullong
llong  mad_hi(llong a, llong b, llong c);
long2  mad_hi(long2 a, long2 b, long2 c);
long3  mad_hi(long3 a, long3 b, long3 c);
long4  mad_hi(long4 a, long4 b, long4 c);
long8  mad_hi(long8 a, long8 b, long8 c);
long16 mad_hi(long16 a, long16 b, long16 c);

ullong  mad_hi(ullong a, ullong b, ullong c);
ulong2  mad_hi(ulong2 a, ulong2 b, ulong2 c);
ulong3  mad_hi(ulong3 a, ulong3 b, ulong3 c);
ulong4  mad_hi(ulong4 a, ulong4 b, ulong4 c);
ulong8  mad_hi(ulong8 a, ulong8 b, ulong8 c);
ulong16 mad_hi(ulong16 a, ulong16 b, ulong16 c);

#endif
