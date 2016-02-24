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

#ifndef ADD_SAT_H_
#define ADD_SAT_H_

#include <cl_cpu_types.h>

//schar, uchar
schar  add_sat(schar x, schar y);
char2  add_sat(char2 x, char2 y);
char3  add_sat(char3 x, char3 y);
char4  add_sat(char4 x, char4 y);
char8  add_sat(char8 x, char8 y);
char16 add_sat(char16 x, char16 y);

uchar   add_sat(uchar x, uchar y);
uchar2  add_sat(uchar2 x, uchar2 y);
uchar3  add_sat(uchar3 x, uchar3 y);
uchar4  add_sat(uchar4 x, uchar4 y);
uchar8  add_sat(uchar8 x, uchar8 y);
uchar16 add_sat(uchar16 x, uchar16 y);

//short. shortn
short   add_sat(short x, short y);
short2  add_sat(short2 x, short2 y);
short3  add_sat(short3 x, short3 y);
short4  add_sat(short4 x, short4 y);
short8  add_sat(short8 x, short8 y);
short16 add_sat(short16 x, short16 y);

ushort   add_sat(ushort x, ushort y);
ushort2  add_sat(ushort2 x, ushort2 y);
ushort3  add_sat(ushort3 x, ushort3 y);
ushort4  add_sat(ushort4 x, ushort4 y);
ushort8  add_sat(ushort8 x, ushort8 y);
ushort16 add_sat(ushort16 x, ushort16 y);

//int, uint
int   add_sat(int x, int y);
int2  add_sat(int2 x, int2 y);
int3  add_sat(int3 x, int3 y);
int4  add_sat(int4 x, int4 y);
int8  add_sat(int8 x, int8 y);
int16 add_sat(int16 x, int16 y);

uint   add_sat(uint x, uint y);
uint2  add_sat(uint2 x, uint2 y);
uint3  add_sat(uint3 x, uint3 y);
uint4  add_sat(uint4 x, uint4 y);
uint8  add_sat(uint8 x, uint8 y);
uint16 add_sat(uint16 x, uint16 y);

//llong, ullong
llong  add_sat(llong x, llong y);
long2  add_sat(long2 x, long2 y);
long3  add_sat(long3 x, long3 y);
long4  add_sat(long4 x, long4 y);
long8  add_sat(long8 x, long8 y);
long16 add_sat(long16 x, long16 y);

ullong  add_sat(ullong x, ullong y);
ulong2  add_sat(ulong2 x, ulong2 y);
ulong3  add_sat(ulong3 x, ulong3 y);
ulong4  add_sat(ulong4 x, ulong4 y);
ulong8  add_sat(ulong8 x, ulong8 y);
ulong16 add_sat(ulong16 x, ulong16 y);

#endif
