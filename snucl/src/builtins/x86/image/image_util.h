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

#ifndef IMAGE_UTIL_H_
#define IMAGE_UTIL_H_

#include <cl_cpu_types.h>

typedef enum _sampler_normalized_coords cl_sampler_normalized_coords;
typedef enum _sampler_filter_modes cl_sampler_filter_modes;
typedef enum _sampler_addressing_modes cl_sampler_addressing_modes;

typedef uint             cl_mem_object_type;
typedef uint             cl_channel_order;
typedef uint             cl_channel_type;

/* cl_channel_type */
#define CL_SNORM_INT8                               0x10D0
#define CL_SNORM_INT16                              0x10D1
#define CL_UNORM_INT8                               0x10D2
#define CL_UNORM_INT16                              0x10D3
#define CL_UNORM_SHORT_565                          0x10D4
#define CL_UNORM_SHORT_555                          0x10D5
#define CL_UNORM_INT_101010                         0x10D6
#define CL_SIGNED_INT8                              0x10D7
#define CL_SIGNED_INT16                             0x10D8
#define CL_SIGNED_INT32                             0x10D9
#define CL_UNSIGNED_INT8                            0x10DA
#define CL_UNSIGNED_INT16                           0x10DB
#define CL_UNSIGNED_INT32                           0x10DC
#define CL_HALF_FLOAT                               0x10DD
#define CL_FLOAT                                    0x10DE

/* cl_mem_object_type */
#define CL_MEM_OBJECT_BUFFER                        0x10F0
#define CL_MEM_OBJECT_IMAGE2D                       0x10F1
#define CL_MEM_OBJECT_IMAGE3D                       0x10F2
#define CL_MEM_OBJECT_IMAGE2D_ARRAY                 0x10F3
#define CL_MEM_OBJECT_IMAGE1D                       0x10F4
#define CL_MEM_OBJECT_IMAGE1D_ARRAY                 0x10F5
#define CL_MEM_OBJECT_IMAGE1D_BUFFER                0x10F6

/* cl_channel_order */
#define CL_R                                        0x10B0
#define CL_A                                        0x10B1
#define CL_RG                                       0x10B2
#define CL_RA                                       0x10B3
#define CL_RGB                                      0x10B4
#define CL_RGBA                                     0x10B5
#define CL_BGRA                                     0x10B6
#define CL_ARGB                                     0x10B7
#define CL_INTENSITY                                0x10B8
#define CL_LUMINANCE                                0x10B9
#define CL_Rx                                       0x10BA
#define CL_RGx                                      0x10BB
#define CL_RGBx                                     0x10BC


typedef struct _cl_image_format {
  cl_channel_order        image_channel_order;
  cl_channel_type         image_channel_data_type;
} cl_image_format;


typedef struct _cl_image_desc {
  cl_mem_object_type      image_type;
  size_t                  image_width;
  size_t                  image_height;
  size_t                  image_depth;
  size_t                  image_array_size;
  size_t                  image_row_pitch;
  size_t                  image_slice_pitch;
  uint                 num_mip_levels;
  uint                 num_samples;
  void*                   buffer;
} cl_image_desc;

class CLMem {
public:
  void *                ptr;
  size_t                size;
  cl_image_format       image_format;
  cl_image_desc         image_desc;
  size_t                row_pitch;
  size_t                slice_pitch;
  size_t                elem_size;
  size_t                channels;
};

class CLMem3D {
public:
  void *                ptr;
  size_t                size;
  cl_image_format       image_format;
  cl_image_desc         image_desc;
  size_t                row_pitch;
  size_t                slice_pitch;
  size_t                elem_size;
  size_t                channels;
};

//typedef CLMem* image_t;
//typedef CLMem* image1d_t;
//typedef CLMem* image2d_t;
//typedef CLMem3D* image3d_t;
//typedef CLMem* image1d_array_t;
//typedef CLMem* image2d_array_t;
//typedef CLMem* image1d_buffer_t;

typedef schar**     image_t;
typedef schar**     image2d_t;
typedef void***    image3d_t;
typedef schar**     image2d_array_t;
typedef schar**     image1d_t;
typedef schar**     image1d_buffer_t;
typedef schar**     image1d_array_t;
typedef int        sampler_t;
typedef int        event_t;

double frac(double x);

bool isOutside(int x, int size);
bool isOutside(int4 ijk, int4 dim);

void addr_clamp(int a, int b, int c, int *i);
int addr_clamp(int a, int b, int c);

int4 getDim(CLMem *image);

int4 getImageArrayCoord(CLMem *image, int4 coord);

bool isImageArray(CLMem* image);
float4 getImageArrayCoord(CLMem *image, float4 coord);

///////////////////// addressing mode (get i, j, k)
void addr_repeat_nearest(float s, int width, int *i);
void addr_repeat_linear(float s, int width, int *i0, int *i1, float *u);

void addr_mirror_nearest(float s, int width, int *i);
void addr_mirror_linear(float s, int width, int *i0, int *i1, float *u);

#endif
