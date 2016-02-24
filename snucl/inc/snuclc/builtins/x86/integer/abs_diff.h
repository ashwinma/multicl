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

#ifndef ABS_DIFF_H_
#define ABS_DIFF_H_

#include <cl_cpu_types.h>

//unsigned schar(T x)
uchar   abs_diff(schar x, schar y);
uchar2  abs_diff(char2 x, char2 y);
uchar3  abs_diff(char3 x, char3 y);
uchar4  abs_diff(char4 x, char4 y);
uchar8  abs_diff(char8 x, char8 y);
uchar16 abs_diff(char16 x, char16 y);

uchar   abs_diff(uchar x, uchar y);
uchar2  abs_diff(uchar2 x, uchar2 y);
uchar3  abs_diff(uchar3 x, uchar3 y);
uchar4  abs_diff(uchar4 x, uchar4 y);
uchar8  abs_diff(uchar8 x, uchar8 y);
uchar16 abs_diff(uchar16 x, uchar16 y);

//unsigned short(T x)
ushort   abs_diff(short x, short y);
ushort2  abs_diff(short2 x, short2 y);
ushort3  abs_diff(short3 x, short3 y);
ushort4  abs_diff(short4 x, short4 y);
ushort8  abs_diff(short8 x, short8 y);
ushort16 abs_diff(short16 x, short16 y);

ushort   abs_diff(ushort x, ushort y);
ushort2  abs_diff(ushort2 x, ushort2 y);
ushort3  abs_diff(ushort3 x, ushort3 y);
ushort4  abs_diff(ushort4 x, ushort4 y);
ushort8  abs_diff(ushort8 x, ushort8 y);
ushort16 abs_diff(ushort16 x, ushort16 y);

//unsigned int(T x)
uint   abs_diff(int x, int y);
uint2  abs_diff(int2 x, int2 y);
uint3  abs_diff(int3 x, int3 y);
uint4  abs_diff(int4 x, int4 y);
uint8  abs_diff(int8 x, int8 y);
uint16 abs_diff(int16 x, int16 y);

uint   abs_diff(uint x, uint y);
uint2  abs_diff(uint2 x, uint2 y);
uint3  abs_diff(uint3 x, uint3 y);
uint4  abs_diff(uint4 x, uint4 y);
uint8  abs_diff(uint8 x, uint8 y);
uint16 abs_diff(uint16 x, uint16 y);

//unsigned llong(T x)
ullong  abs_diff(llong x, llong y);
ulong2  abs_diff(long2 x, long2 y);
ulong3  abs_diff(long3 x, long3 y);
ulong4  abs_diff(long4 x, long4 y);
ulong8  abs_diff(long8 x, long8 y);
ulong16 abs_diff(long16 x, long16 y);

ullong  abs_diff(ullong x, ullong y);
ulong2  abs_diff(ulong2 x, ulong2 y);
ulong3  abs_diff(ulong3 x, ulong3 y);
ulong4  abs_diff(ulong4 x, ulong4 y);
ulong8  abs_diff(ulong8 x, ulong8 y);
ulong16 abs_diff(ulong16 x, ulong16 y);

#endif
