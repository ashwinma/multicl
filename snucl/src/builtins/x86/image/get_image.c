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


#include "get_image.h"


int get_image_width(image_t image)
{
  return ((CLMem*)image)->image_desc.image_width;
}

int get_image_width(image3d_t image)
{
  return ((CLMem*)image)->image_desc.image_width;
}

int get_image_height(image_t image) {
  return ((CLMem*)image)->image_desc.image_height;
}

int get_image_height(image3d_t image) {
  return ((CLMem*)image)->image_desc.image_height;
}

int get_image_depth(image_t image) {
  return ((CLMem*)image)->image_desc.image_depth;
}

int get_image_depth(image3d_t image) {
  return ((CLMem*)image)->image_desc.image_depth;
}

int get_image_channel_data_type(image_t image) {
  return ((CLMem*)image)->image_format.image_channel_data_type;
}

int get_image_channel_order(image_t image) {
  return ((CLMem*)image)->image_format.image_channel_order;
}

int2 get_image_dim(image_t image) {
  int2 dim;
  dim[0] = ((CLMem*)image)->image_desc.image_width;
  dim[1] = ((CLMem*)image)->image_desc.image_height;
  return dim; 
}

int4 get_image_dim(image3d_t image) {
  int4 dim;
  dim[0] = ((CLMem*)image)->image_desc.image_width;
  dim[1] = ((CLMem*)image)->image_desc.image_height;
  dim[2] = ((CLMem*)image)->image_desc.image_depth;
  dim[3] = 0;
  return dim; 
}

size_t get_image_array_size(image_t image) {
  return ((CLMem*)image)->image_desc.image_array_size;
}


