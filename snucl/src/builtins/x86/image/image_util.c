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
#include "image_util.h"
#include <integer/cl_builtins_integer.h>
#include <math/cl_builtins_math.h>
#include <vector/cl_builtins_vector.h>
#include <stdio.h>

double frac(double x)
{
     return x - floor(x);
}

bool isOutside(int x, int size) {
  
  if( size != 0 && (x < 0 || x > size-1) ) 
    return true;
  else 
    return false;
}

bool isOutside(int4 ijk, int4 dim) {
  for( int d = 0; d < 3; d++ ) {
    if( dim[d] != 0 && (ijk[d] < 0 || ijk[d] > dim[d]-1) )
      return true;
  }
  return false; 
}

void addr_clamp(int a, int b, int c, int *i) {
  *i = (a < b) ? b : ((a > c) ? c : a);
  return;
}

int addr_clamp(int a, int b, int c) {
  return (a < b) ? b : ((a > c) ? c : a);
}


int4 getDim(CLMem* image) {
  int4 dim = {0, 0, 0, 0};
  cl_mem_object_type type = image->image_desc.image_type;
  
  dim[0] = image->image_desc.image_width;
  if( type == CL_MEM_OBJECT_IMAGE1D_ARRAY )
    dim[1] = image->image_desc.image_array_size;
  else
    dim[1] = image->image_desc.image_height;
  if( type == CL_MEM_OBJECT_IMAGE2D_ARRAY )
    dim[2] = image->image_desc.image_array_size;
  else
    dim[2] = image->image_desc.image_depth;
  dim[3] = 0;
  return dim;
}

bool isImageArray(CLMem* image) {
  cl_mem_object_type type = image->image_desc.image_type;

  if( type == CL_MEM_OBJECT_IMAGE1D_ARRAY || type == CL_MEM_OBJECT_IMAGE2D_ARRAY ) {
    return true;
  } else {
    return false;
  }
}

int4 getImageArrayCoord(CLMem* image, int4 coord) {
  int4 dim; 
  dim = getDim(image);
  cl_mem_object_type type = image->image_desc.image_type;

  if( type == CL_MEM_OBJECT_IMAGE2D_ARRAY ) {
    coord[2] = addr_clamp((int)floor(coord[2] + 0.5f), 0, dim[2]-1);
  }
  if( type == CL_MEM_OBJECT_IMAGE1D_ARRAY ) {
    coord[1] = addr_clamp((int)floor(coord[1] + 0.5f), 0, dim[1]-1);
  }
  return coord;
}

float4 getImageArrayCoord(CLMem* image, float4 coord) {
  int4 dim;
  cl_mem_object_type type = image->image_desc.image_type;
  dim = getDim(image);

//  coord[0] = 0.022059*dim[0];
//  coord[1] = -0.001733*dim[1];

  if( type == CL_MEM_OBJECT_IMAGE2D_ARRAY ) {
    coord[2] = (int)addr_clamp(floor(coord[2] + 0.5f), 0, dim[2]-1);
  }
  if( type == CL_MEM_OBJECT_IMAGE1D_ARRAY ) {
    coord[1] = (int)addr_clamp(floor(coord[1] + 0.5f), 0, dim[1]-1);
  }
//  printf("corrd : %f, %f\n", coord[0], coord[1]);

  return coord;
}

///////////////////// addressing mode (get i, j, k)
void addr_repeat_nearest(float s, int width, int *i) {
  float u;
  u = (s - floor(s)) * width;
  *i =(int)floor(u);
  if( *i > width -1 )
    *i = *i - width;
  return;
}
void addr_repeat_linear(float s, int width, int *i0, int *i1, float *u) {
  *u = (s - floor(s)) * width;
  *i0 =(int)floor(*u - 0.5);
  *i1 = *i0 + 1;
  if( *i0 < 0 )
      *i0 = width + *i0;
  if( *i1 > width -1 )
    *i1 = *i1 - width;
  return;
}

void addr_mirror_nearest(float s, int width, int *i) {
  double s_;
  float u;
  s_ = 2.0f * rint(0.5f * s);
  s_ = fabs(s - s_);
  u = s_ * width;
  *i = (int)floor(u);
  *i = min(*i, width - 1);
  return;
}
void addr_mirror_linear(float s, int width, int *i0, int *i1, float *u) {
  double s_;
  s_ = 2.0f * rint(0.5f * s);
  s_ = fabs(s - s_);
  *u = s_ * width;
  *i0 = (int)floor(*u - 0.5f);
  *i1 = *i0 + 1;
  *i0 = max(*i0, 0);
  *i1 = min(*i1, width -1);
  return;
}



