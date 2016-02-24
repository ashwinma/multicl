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

#ifndef MUL_HI_H_
#define MUL_HI_H_

#include <cl_cpu_types.h>

//schar, uchar
schar  mul_hi(schar x, schar y);
char2  mul_hi(char2 x, char2 y);
char3  mul_hi(char3 x, char3 y);
char4  mul_hi(char4 x, char4 y);
char8  mul_hi(char8 x, char8 y);
char16 mul_hi(char16 x, char16 y);

uchar   mul_hi(uchar x, uchar y);
uchar2  mul_hi(uchar2 x, uchar2 y);
uchar3  mul_hi(uchar3 x, uchar3 y);
uchar4  mul_hi(uchar4 x, uchar4 y);
uchar8  mul_hi(uchar8 x, uchar8 y);
uchar16 mul_hi(uchar16 x, uchar16 y);

//short. shortn
short   mul_hi(short x, short y);
short2  mul_hi(short2 x, short2 y);
short3  mul_hi(short3 x, short3 y);
short4  mul_hi(short4 x, short4 y);
short8  mul_hi(short8 x, short8 y);
short16 mul_hi(short16 x, short16 y);

ushort   mul_hi(ushort x, ushort y);
ushort2  mul_hi(ushort2 x, ushort2 y);
ushort3  mul_hi(ushort3 x, ushort3 y);
ushort4  mul_hi(ushort4 x, ushort4 y);
ushort8  mul_hi(ushort8 x, ushort8 y);
ushort16 mul_hi(ushort16 x, ushort16 y);

//int, uint
int   mul_hi(int x, int y);
int2  mul_hi(int2 x, int2 y);
int3  mul_hi(int3 x, int3 y);
int4  mul_hi(int4 x, int4 y);
int8  mul_hi(int8 x, int8 y);
int16 mul_hi(int16 x, int16 y);

uint   mul_hi(uint x, uint y);
uint2  mul_hi(uint2 x, uint2 y);
uint3  mul_hi(uint3 x, uint3 y);
uint4  mul_hi(uint4 x, uint4 y);
uint8  mul_hi(uint8 x, uint8 y);
uint16 mul_hi(uint16 x, uint16 y);

//llong, ullong
llong  mul_hi(llong x, llong y);
long2  mul_hi(long2 x, long2 y);
long3  mul_hi(long3 x, long3 y);
long4  mul_hi(long4 x, long4 y);
long8  mul_hi(long8 x, long8 y);
long16 mul_hi(long16 x, long16 y);

ullong  mul_hi(ullong x, ullong y);
ulong2  mul_hi(ulong2 x, ulong2 y);
ulong3  mul_hi(ulong3 x, ulong3 y);
ulong4  mul_hi(ulong4 x, ulong4 y);
ulong8  mul_hi(ulong8 x, ulong8 y);
ulong16 mul_hi(ulong16 x, ulong16 y);

#endif
