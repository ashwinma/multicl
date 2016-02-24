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

#include <cl_cpu_ops.h>
#include <integer/min.h>
#include <integer/max.h>

int clamp(int x, int minval, int maxval)
{
	return min(max(x, minval), maxval);
}

int2 clamp(int2 x, int2 minval, int2 maxval)
{
	int2 rst;
	rst[0] = min(max(x[0], minval[0]), maxval[0]);
	rst[1] = min(max(x[1], minval[1]), maxval[1]);
	return rst;
}

int3 clamp(int3 x, int3 minval, int3 maxval)
{
	int3 rst;
	for (int i = 0; i < 3; i++)
		rst[i] = min(max(x[i], minval[i]), maxval[i]);
	return rst;
}

int4 clamp(int4 x, int4 minval, int4 maxval)
{
	int4 rst;
	for (int i = 0; i < 4; i++)
		rst[i] = min(max(x[i], minval[i]), maxval[i]);
	return rst;
}

int8 clamp(int8 x, int8 minval, int8 maxval)
{
	int8 rst;
	for (int i = 0; i < 8; i++)
		rst[i] = min(max(x[i], minval[i]), maxval[i]);
	return rst;
}

int16 clamp(int16 x, int16 minval, int16 maxval)
{
	int16 rst;
	for (int i = 0; i < 16; i++)
		rst[i] = min(max(x[i], minval[i]), maxval[i]);
	return rst;
}


int2 clamp(int2 x, int minval, int maxval)
{
	int2 rst;
	rst[0] = min(max(x[0], minval), maxval);
	rst[1] = min(max(x[1], minval), maxval);
	return rst;
}

int3 clamp(int3 x, int minval, int maxval)
{
	int3 rst;
	for (int i = 0; i < 3; i++)
		rst[i] = min(max(x[i], minval), maxval);
	return rst;
}

int4 clamp(int4 x, int minval, int maxval)
{
	int4 rst;
	for (int i = 0; i < 4; i++)
		rst[i] = min(max(x[i], minval), maxval);
	return rst;
}

int8 clamp(int8 x, int minval, int maxval)
{
	int8 rst;
	for (int i = 0; i < 8; i++)
		rst[i] = min(max(x[i], minval), maxval);
	return rst;
}

int16 clamp(int16 x, int minval, int maxval)
{
	int16 rst;
	for (int i = 0; i < 16; i++)
		rst[i] = min(max(x[i], minval), maxval);
	return rst;
}


