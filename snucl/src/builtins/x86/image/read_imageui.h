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


#ifndef READ_IMAGEUI_H_
#define READ_IMAGEUI_H_

#include <cl_cpu_types.h>
#include "image_util.h"

///////////////////// read imageui
// no sampler
uint4 read_imageui(image_t image, int4 coord);
uint4 read_imageui(image_t image, int2 coord);
uint4 read_imageui(image_t image, int coord);
uint4 read_imageui(image3d_t image, int4 coord);

// with sampler
uint4 read_imageui(image_t image, sampler_t sampler, float coord);
uint4 read_imageui(image_t image, sampler_t sampler, float2 coord);
uint4 read_imageui(image_t image, sampler_t sampler, float4 coord);
uint4 read_imageui(image3d_t image, sampler_t sampler, float coord);

uint4 read_imageui(image_t image, sampler_t sampler, int coord);
uint4 read_imageui(image_t image, sampler_t sampler, int2 coord);
uint4 read_imageui(image_t image, sampler_t sampler, int4 coord);
uint4 read_imageui(image3d_t image, sampler_t sampler, int coord);

uint4 _read_imageui(CLMem* image, sampler_t sampler, int4 coord, int dim_size);
uint4 _read_imageui(CLMem* image, sampler_t sampler, float4 coord, int dim_size);

///////////////////// get pixel
uint4 get_pixelui(CLMem* image, int4 ijk);

uint4 _get_pixelui_UNSIGNED_INT8(CLMem* image, int4 ijk);
uint4 _get_pixelui_UNSIGNED_INT16(CLMem* image, int4 ijk);
uint4 _get_pixelui_UNSIGNED_INT32(CLMem* image, int4 ijk);



#endif


