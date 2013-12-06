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

#ifndef MAX_INTEGER_H_
#define MAX_INTEGER_H_

#include <cl_cpu_types.h>


//schar, uchar
schar  max(schar x, schar y);
char2  max(char2 x, char2 y);
char3  max(char3 x, char3 y);
char4  max(char4 x, char4 y);
char8  max(char8 x, char8 y);
char16 max(char16 x, char16 y);

uchar   max(uchar x, uchar y);
uchar2  max(uchar2 x, uchar2 y);
uchar3  max(uchar3 x, uchar3 y);
uchar4  max(uchar4 x, uchar4 y);
uchar8  max(uchar8 x, uchar8 y);
uchar16 max(uchar16 x, uchar16 y);


//short. ushort
short   max(short x, short y);
short2  max(short2 x, short2 y);
short3  max(short3 x, short3 y);
short4  max(short4 x, short4 y);
short8  max(short8 x, short8 y);
short16 max(short16 x, short16 y);

ushort   max(ushort x, ushort y);
ushort2  max(ushort2 x, ushort2 y);
ushort3  max(ushort3 x, ushort3 y);
ushort4  max(ushort4 x, ushort4 y);
ushort8  max(ushort8 x, ushort8 y);
ushort16 max(ushort16 x, ushort16 y);

//int, uint
int   max(int x, int y);
int2  max(int2 x, int2 y);
int3  max(int3 x, int3 y);
int4  max(int4 x, int4 y);
int8  max(int8 x, int8 y);
int16 max(int16 x, int16 y);

uint   max(uint x, uint y);
uint2  max(uint2 x, uint2 y);
uint3  max(uint3 x, uint3 y);
uint4  max(uint4 x, uint4 y);
uint8  max(uint8 x, uint8 y);
uint16 max(uint16 x, uint16 y);

//llong, ullong
llong  max(llong x, llong y);
long2  max(long2 x, long2 y);
long3  max(long3 x, long3 y);
long4  max(long4 x, long4 y);
long8  max(long8 x, long8 y);
long16 max(long16 x, long16 y);

ullong  max(ullong x, ullong y);
ulong2  max(ulong2 x, ulong2 y);
ulong3  max(ulong3 x, ulong3 y);
ulong4  max(ulong4 x, ulong4 y);
ulong8  max(ulong8 x, ulong8 y);
ulong16 max(ulong16 x, ulong16 y);

//// sgentype

//schar, schar
char2  max(char2 x, schar y);
char3  max(char3 x, schar y);
char4  max(char4 x, schar y);
char8  max(char8 x, schar y);
char16 max(char16 x, schar y);

uchar2  max(uchar2 x, uchar y);
uchar3  max(uchar3 x, uchar y);
uchar4  max(uchar4 x, uchar y);
uchar8  max(uchar8 x, uchar y);
uchar16 max(uchar16 x, uchar y);


//short. ushort
short2  max(short2 x, short y);
short3  max(short3 x, short y);
short4  max(short4 x, short y);
short8  max(short8 x, short y);
short16 max(short16 x, short y);

ushort2  max(ushort2 x, ushort y);
ushort3  max(ushort3 x, ushort y);
ushort4  max(ushort4 x, ushort y);
ushort8  max(ushort8 x, ushort y);
ushort16 max(ushort16 x, ushort y);

//int, uint
int2  max(int2 x, int y);
int3  max(int3 x, int y);
int4  max(int4 x, int y);
int8  max(int8 x, int y);
int16 max(int16 x, int y);

uint2  max(uint2 x, uint y);
uint3  max(uint3 x, uint y);
uint4  max(uint4 x, uint y);
uint8  max(uint8 x, uint y);
uint16 max(uint16 x, uint y);

//llong, ullong
long2  max(long2 x, llong y);
long3  max(long3 x, llong y);
long4  max(long4 x, llong y);
long8  max(long8 x, llong y);
long16 max(long16 x, llong y);

ulong2  max(ulong2 x, ullong y);
ulong3  max(ulong3 x, ullong y);
ulong4  max(ulong4 x, ullong y);
ulong8  max(ulong8 x, ullong y);
ulong16 max(ulong16 x, ullong y);


#endif
