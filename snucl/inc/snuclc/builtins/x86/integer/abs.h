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

#ifndef ABS_H_
#define ABS_H_

#include <cl_cpu_types.h>

//unsigned schar(T x)
uchar   abs(schar x);
uchar2  abs(char2 x);
uchar3  abs(char3 x);
uchar4  abs(char4 x);
uchar8  abs(char8 x);
uchar16 abs(char16 x);

uchar   abs(uchar x);
uchar2  abs(uchar2 x);
uchar3  abs(uchar3 x);
uchar4  abs(uchar4 x);
uchar8  abs(uchar8 x);
uchar16 abs(uchar16 x);

//unsigned short(T x)
ushort   abs(short x);
ushort2  abs(short2 x);
ushort3  abs(short3 x);
ushort4  abs(short4 x);
ushort8  abs(short8 x);
ushort16 abs(short16 x);

ushort   abs(ushort x);
ushort2  abs(ushort2 x);
ushort3  abs(ushort3 x);
ushort4  abs(ushort4 x);
ushort8  abs(ushort8 x);
ushort16 abs(ushort16 x);

//unsigned int(T x)
//uint   abs(int x);
uint2  abs(int2 x);
uint3  abs(int3 x);
uint4  abs(int4 x);
uint8  abs(int8 x);
uint16 abs(int16 x);

uint   abs(uint x);
uint2  abs(uint2 x);
uint3  abs(uint3 x);
uint4  abs(uint4 x);
uint8  abs(uint8 x);
uint16 abs(uint16 x);

//unsigned llong(T x)
ullong  abs(llong x);
ulong2  abs(long2 x);
ulong3  abs(long3 x);
ulong4  abs(long4 x);
ulong8  abs(long8 x);
ulong16 abs(long16 x);

ullong  abs(ullong x);
ulong2  abs(ulong2 x);
ulong3  abs(ulong3 x);
ulong4  abs(ulong4 x);
ulong8  abs(ulong8 x);
ulong16 abs(ulong16 x);

#endif
