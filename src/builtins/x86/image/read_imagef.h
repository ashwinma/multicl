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


#ifndef READ_IMAGEF_H_
#define READ_IMAGEF_H_

#include <cl_cpu_types.h>
#include "image_util.h"

///////////////////// read imagef
// no sampler
float4 read_imagef(image_t _image, int4 coord);
float4 read_imagef(image_t _image, int2 coord);
float4 read_imagef(image_t _image, int coord);
float4 read_imagef(image3d_t _image, int4 coord);

// with sampler
float4 read_imagef(image_t _image, sampler_t sampler, float coord);
float4 read_imagef(image_t _image, sampler_t sampler, float2 coord);
float4 read_imagef(image_t _image, sampler_t sampler, float4 coord);
float4 read_imagef(image3d_t _image, sampler_t sampler, float4 coord);

float4 read_imagef(image_t _image, sampler_t sampler, int coord);
float4 read_imagef(image_t _image, sampler_t sampler, int2 coord);
float4 read_imagef(image_t _image, sampler_t sampler, int4 coord);
float4 read_imagef(image3d_t _image, sampler_t sampler, int4 coord);

float4 _read_imagef(CLMem *image, sampler_t sampler, int4 coord, int dim_size);
float4 _read_imagef(CLMem *image, sampler_t sampler, float4 coord, int dim_size);



///////////////////// get pixelf
float4 get_pixelf(CLMem* image, int4 ijk);
float4 get_pixelf(CLMem* image, int4 ijk0, int4 ijk1, float4 abc);

float4 _get_pixelf_UNORM_INT8(CLMem* image, int4 ijk);
float4 _get_pixelf_UNORM_INT16(CLMem* image, int4 ijk);
float4 _get_pixelf_HALF_FLOAT(CLMem* image, int4 ijk);
float4 _get_pixelf_FLOAT(CLMem* image, int4 ijk);

float4 _get_pixelf_UNORM_INT8(CLMem* image, int4 ijk0, int4 ijk1, float4 abc);
float4 _get_pixelf_UNORM_INT16(CLMem* image, int4 ijk0, int4 ijk1, float4 abc);
float4 _get_pixelf_HALF_FLOAT(CLMem* image, int4 ijk0, int4 ijk1, float4 abc);
float4 _get_pixelf_FLOAT(CLMem* image, int4 ijk0, int4 ijk1, float4 abc);


///////////////////// conversion
float normalize_UNORM_INT8(double dval);
float normalize_UNORM_INT16(double dval);

#endif


