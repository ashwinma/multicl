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

#ifndef ROTATE_H_
#define ROTATE_H_

#include <cl_cpu_types.h>

//schar, uchar
schar  rotate(schar v, schar i);
char2  rotate(char2 v, char2 i);
char3  rotate(char3 v, char3 i);
char4  rotate(char4 v, char4 i);
char8  rotate(char8 v, char8 i);
char16 rotate(char16 v, char16 i);

uchar   rotate(uchar v, uchar i);
uchar2  rotate(uchar2 v, uchar2 i);
uchar3  rotate(uchar3 v, uchar3 i);
uchar4  rotate(uchar4 v, uchar4 i);
uchar8  rotate(uchar8 v, uchar8 i);
uchar16 rotate(uchar16 v, uchar16 i);

//short. shortn
short   rotate(short v, short i);
short2  rotate(short2 v, short2 i);
short3  rotate(short3 v, short3 i);
short4  rotate(short4 v, short4 i);
short8  rotate(short8 v, short8 i);
short16 rotate(short16 v, short16 i);

ushort   rotate(ushort v, ushort i);
ushort2  rotate(ushort2 v, ushort2 i);
ushort3  rotate(ushort3 v, ushort3 i);
ushort4  rotate(ushort4 v, ushort4 i);
ushort8  rotate(ushort8 v, ushort8 i);
ushort16 rotate(ushort16 v, ushort16 i);

//int, uint
int   rotate(int v, int i);
int2  rotate(int2 v, int2 i);
int3  rotate(int3 v, int3 i);
int4  rotate(int4 v, int4 i);
int8  rotate(int8 v, int8 i);
int16 rotate(int16 v, int16 i);

uint   rotate(uint v, uint i);
uint2  rotate(uint2 v, uint2 i);
uint3  rotate(uint3 v, uint3 i);
uint4  rotate(uint4 v, uint4 i);
uint8  rotate(uint8 v, uint8 i);
uint16 rotate(uint16 v, uint16 i);

//llong, ullong
llong  rotate(llong v, llong i);
long2  rotate(long2 v, long2 i);
long3  rotate(long3 v, long3 i);
long4  rotate(long4 v, long4 i);
long8  rotate(long8 v, long8 i);
long16 rotate(long16 v, long16 i);

ullong  rotate(ullong v, ullong i);
ulong2  rotate(ulong2 v, ulong2 i);
ulong3  rotate(ulong3 v, ulong3 i);
ulong4  rotate(ulong4 v, ulong4 i);
ulong8  rotate(ulong8 v, ulong8 i);
ulong16 rotate(ulong16 v, ulong16 i);

#endif
