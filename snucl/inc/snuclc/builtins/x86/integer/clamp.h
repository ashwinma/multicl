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
/* allong with the SNU-SAMSUNG OpenCL runtime. If not, see                    */
/* <http://www.gnu.org/licenses/>.                                           */
/*****************************************************************************/

#ifndef INTEGER_CLAMP_H_
#define INTEGER_CLAMP_H_

#include <cl_cpu_types.h>

schar clamp(schar x, schar minval, schar maxval);
char2 clamp(char2 x, char2 minval, char2 maxval);
char3 clamp(char3 x, char3 minval, char3 maxval);
char4 clamp(char4 x, char4 minval, char4 maxval);
char8 clamp(char8 x, char8 minval, char8 maxval);
char16 clamp(char16 x, char16 minval, char16 maxval);

char2 clamp(char2 x, schar minval, schar maxval);
char3 clamp(char3 x, schar minval, schar maxval);
char4 clamp(char4 x, schar minval, schar maxval);
char8 clamp(char8 x, schar minval, schar maxval);
char16 clamp(char16 x, schar minval, schar maxval);


uchar clamp(uchar x, uchar minval, uchar maxval);
uchar2 clamp(uchar2 x, uchar2 minval, uchar2 maxval);
uchar3 clamp(uchar3 x, uchar3 minval, uchar3 maxval);
uchar4 clamp(uchar4 x, uchar4 minval, uchar4 maxval);
uchar8 clamp(uchar8 x, uchar8 minval, uchar8 maxval);
uchar16 clamp(uchar16 x, uchar16 minval, uchar16 maxval);

uchar2 clamp(uchar2 x, uchar minval, uchar maxval);
uchar3 clamp(uchar3 x, uchar minval, uchar maxval);
uchar4 clamp(uchar4 x, uchar minval, uchar maxval);
uchar8 clamp(uchar8 x, uchar minval, uchar maxval);
uchar16 clamp(uchar16 x, uchar minval, uchar maxval);


short clamp(short x, short minval, short maxval);
short2 clamp(short2 x, short2 minval, short2 maxval);
short3 clamp(short3 x, short3 minval, short3 maxval);
short4 clamp(short4 x, short4 minval, short4 maxval);
short8 clamp(short8 x, short8 minval, short8 maxval);
short16 clamp(short16 x, short16 minval, short16 maxval);

short2 clamp(short2 x, short minval, short maxval);
short3 clamp(short3 x, short minval, short maxval);
short4 clamp(short4 x, short minval, short maxval);
short8 clamp(short8 x, short minval, short maxval);
short16 clamp(short16 x, short minval, short maxval);


ushort clamp(ushort x, ushort minval, ushort maxval);
ushort2 clamp(ushort2 x, ushort2 minval, ushort2 maxval);
ushort3 clamp(ushort3 x, ushort3 minval, ushort3 maxval);
ushort4 clamp(ushort4 x, ushort4 minval, ushort4 maxval);
ushort8 clamp(ushort8 x, ushort8 minval, ushort8 maxval);
ushort16 clamp(ushort16 x, ushort16 minval, ushort16 maxval);

ushort2 clamp(ushort2 x, ushort minval, ushort maxval);
ushort3 clamp(ushort3 x, ushort minval, ushort maxval);
ushort4 clamp(ushort4 x, ushort minval, ushort maxval);
ushort8 clamp(ushort8 x, ushort minval, ushort maxval);
ushort16 clamp(ushort16 x, ushort minval, ushort maxval);


int clamp(int x, int minval, int maxval);
int2 clamp(int2 x, int2 minval, int2 maxval);
int3 clamp(int3 x, int3 minval, int3 maxval);
int4 clamp(int4 x, int4 minval, int4 maxval);
int8 clamp(int8 x, int8 minval, int8 maxval);
int16 clamp(int16 x, int16 minval, int16 maxval);

int2 clamp(int2 x, int minval, int maxval);
int3 clamp(int3 x, int minval, int maxval);
int4 clamp(int4 x, int minval, int maxval);
int8 clamp(int8 x, int minval, int maxval);
int16 clamp(int16 x, int minval, int maxval);


uint clamp(uint x, uint minval, uint maxval);
uint2 clamp(uint2 x, uint2 minval, uint2 maxval);
uint3 clamp(uint3 x, uint3 minval, uint3 maxval);
uint4 clamp(uint4 x, uint4 minval, uint4 maxval);
uint8 clamp(uint8 x, uint8 minval, uint8 maxval);
uint16 clamp(uint16 x, uint16 minval, uint16 maxval);

uint2 clamp(uint2 x, uint minval, uint maxval);
uint3 clamp(uint3 x, uint minval, uint maxval);
uint4 clamp(uint4 x, uint minval, uint maxval);
uint8 clamp(uint8 x, uint minval, uint maxval);
uint16 clamp(uint16 x, uint minval, uint maxval);


llong clamp(llong x, llong minval, llong maxval);
long2 clamp(long2 x, long2 minval, long2 maxval);
long3 clamp(long3 x, long3 minval, long3 maxval);
long4 clamp(long4 x, long4 minval, long4 maxval);
long8 clamp(long8 x, long8 minval, long8 maxval);
long16 clamp(long16 x, long16 minval, long16 maxval);

long2 clamp(long2 x, llong minval, llong maxval);
long3 clamp(long3 x, llong minval, llong maxval);
long4 clamp(long4 x, llong minval, llong maxval);
long8 clamp(long8 x, llong minval, llong maxval);
long16 clamp(long16 x, llong minval, llong maxval);


ullong clamp(ullong x, ullong minval, ullong maxval);
ulong2 clamp(ulong2 x, ulong2 minval, ulong2 maxval);
ulong3 clamp(ulong3 x, ulong3 minval, ulong3 maxval);
ulong4 clamp(ulong4 x, ulong4 minval, ulong4 maxval);
ulong8 clamp(ulong8 x, ulong8 minval, ulong8 maxval);
ulong16 clamp(ulong16 x, ulong16 minval, ulong16 maxval);

ulong2 clamp(ulong2 x, ullong minval, ullong maxval);
ulong3 clamp(ulong3 x, ullong minval, ullong maxval);
ulong4 clamp(ulong4 x, ullong minval, ullong maxval);
ulong8 clamp(ulong8 x, ullong minval, ullong maxval);
ulong16 clamp(ulong16 x, ullong minval, ullong maxval);


#endif
