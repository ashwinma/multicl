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

#ifndef MAD_SAT_H_
#define MAD_SAT_H_

#include <cl_cpu_types.h>

schar    mad_sat(schar, schar, schar);
char2    mad_sat(char2, char2, char2);
char3    mad_sat(char3, char3, char3);
char4    mad_sat(char4, char4, char4);
char8    mad_sat(char8, char8, char8);
char16   mad_sat(char16, char16, char16);

short    mad_sat(short, short, short);
short2   mad_sat(short2, short2, short2);
short3   mad_sat(short3, short3, short3);
short4   mad_sat(short4, short4, short4);
short8   mad_sat(short8, short8, short8);
short16  mad_sat(short16, short16, short16);

int      mad_sat(int, int, int);
int2     mad_sat(int2, int2, int2);
int3     mad_sat(int3, int3, int3);
int4     mad_sat(int4, int4, int4);
int8     mad_sat(int8, int8, int8);
int16    mad_sat(int16, int16, int16);

llong    mad_sat(llong, llong, llong);
long2    mad_sat(long2, long2, long2);
long3    mad_sat(long3, long3, long3);
long4    mad_sat(long4, long4, long4);
long8    mad_sat(long8, long8, long8);
long16   mad_sat(long16, long16, long16);

uchar    mad_sat(uchar, uchar, uchar);
uchar2   mad_sat(uchar2, uchar2, uchar2);
uchar3   mad_sat(uchar3, uchar3, uchar3);
uchar4   mad_sat(uchar4, uchar4, uchar4);
uchar8   mad_sat(uchar8, uchar8, uchar8);
uchar16  mad_sat(uchar16, uchar16, uchar16);

ushort   mad_sat(ushort, ushort, ushort);
ushort2  mad_sat(ushort2, ushort2, ushort2);
ushort3  mad_sat(ushort3, ushort3, ushort3);
ushort4  mad_sat(ushort4, ushort4, ushort4);
ushort8  mad_sat(ushort8, ushort8, ushort8);
ushort16 mad_sat(ushort16, ushort16, ushort16);

uint     mad_sat(uint, uint, uint);
uint2    mad_sat(uint2, uint2, uint2);
uint3    mad_sat(uint3, uint3, uint3);
uint4    mad_sat(uint4, uint4, uint4);
uint8    mad_sat(uint8, uint8, uint8);
uint16   mad_sat(uint16, uint16, uint16);

ullong   mad_sat(ullong, ullong, ullong);
ulong2   mad_sat(ulong2, ulong2, ulong2);
ulong3   mad_sat(ulong3, ulong3, ulong3);
ulong4   mad_sat(ulong4, ulong4, ulong4);
ulong8   mad_sat(ulong8, ulong8, ulong8);
ulong16  mad_sat(ulong16, ulong16, ulong16);

#endif
