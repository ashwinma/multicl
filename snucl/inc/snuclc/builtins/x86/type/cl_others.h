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

#ifndef __CL_CPU_OTHERS_H
#define __CL_CPU_OTHERS_H

/* other built-in data types */
typedef schar**     image2d_t;
typedef void***    image3d_t;
typedef schar**     image2d_array_t;
typedef schar**     image1d_t;
typedef schar**     image1d_buffer_t;
typedef schar**     image1d_array_t;
typedef int        sampler_t;
typedef int        event_t;
//typedef schar**     image2d_t;
//typedef void***    image3d_t;
//typedef image2d_t* image2d_array_t;
//typedef schar*      image1d_t;
//typedef void*      image1d_buffer_t;
//typedef image1d_t* image1d_array_t;

/* sampler: normalized coords */
enum _sampler_normalized_coords {
  CLK_NORMALIZED_COORDS_TRUE = 0x0100,
  CLK_NORMALIZED_COORDS_FALSE = 0x0080
};

/* sampler: addressing mode */
enum _sampler_addressing_modes {
  CLK_ADDRESS_CLAMP_TO_EDGE = 0x0010, 
  CLK_ADDRESS_CLAMP = 0x0008, 
  CLK_ADDRESS_NONE = 0x0004, 
  CLK_ADDRESS_REPEAT = 0x0002, 
  CLK_ADDRESS_MIRRORED_REPEAT = 0x0001
};

/* sampler: filter mode */
enum _sampler_filter_modes {
  CLK_FILTER_NEAREST = 0x0040, 
  CLK_FILTER_LINEAR = 0x0020
};

#endif //__CL_CPU_OTHERS_H

