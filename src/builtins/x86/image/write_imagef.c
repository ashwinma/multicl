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
#include "write_imagef.h"
#include <conversion/cl_builtins_conversion.h>
#include <vector/cl_builtins_vector.h>
#include <stdio.h>

///////////////////// write_image
void write_imagef(image_t _image, int4 coord, float4 color) {
  CLMem* image = (CLMem*) _image;
  coord = getImageArrayCoord(image, coord);
  set_pixelf(image, coord, color, 3);
}

void write_imagef(image_t _image, int2 coord, float4 color) {
  
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  CLMem* image = (CLMem*) _image;
  _coord = getImageArrayCoord(image, _coord);

  set_pixelf(image, _coord, color, 2);
}

void write_imagef(image_t _image, int coord, float4 color) {

  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord;
  
  CLMem* image = (CLMem*) _image;
  set_pixelf(image, _coord, color, 1);
}

///////////////////// set pixel
void set_pixelf(CLMem* image, int4 ijk, float4 color, int dim_size) {

	uint channel_type = image->image_format.image_channel_data_type;
  cl_channel_order channel_order;
  int4 dim;
  int d;
  
  channel_order = image->image_format.image_channel_order;
  dim = getDim(image);
  ijk = getImageArrayCoord(image, ijk);

  for( d = 0; d < dim_size; d++ ) {
    if( isOutside(ijk[d], dim[d]) ) {
      return;
    }
  }
	
  if( channel_order == CL_BGRA ) {
    float tmp;
    tmp = color[0]; 
    color[0] = color[2];
    color[2] = tmp;   
  }

  if( channel_type == CL_UNORM_INT8 ) {
    _set_pixelf_UNORM_INT8(image, ijk, color);
	} else if(  channel_type == CL_UNORM_INT16 ) {
		_set_pixelf_UNORM_INT16(image, ijk, color);
	} else if( channel_type == CL_HALF_FLOAT ) {
		_set_pixelf_HALF_FLOAT(image, ijk, color);
	} else if( channel_type == CL_FLOAT ) {
		_set_pixelf_FLOAT(image, ijk, color);
	} 
  return;
}

void _set_pixelf_UNORM_INT8(CLMem* image, int4 ijk, float4 color) {
	uchar uchar_vals[4];
  void * buffer = image->ptr; 
	int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
    uchar_vals[ch] = conversion_UNORM_INT8(color[ch]);
		*((uchar *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch) = uchar_vals[ch];
  }
  return;
}

void _set_pixelf_UNORM_INT16(CLMem* image, int4 ijk, float4 color) {
	ushort ushort_vals[4];
  void * buffer = image->ptr; 
	int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
    ushort_vals[ch] = conversion_UNORM_INT16(color[ch]);
		*((ushort *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch) = ushort_vals[ch];
  }
  return;
}

void _set_pixelf_HALF_FLOAT(CLMem* image, int4 ijk, float4 color) {
	half half_vals[4];
	int4 dim = getDim(image);
  void * buffer = image->ptr; 
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
    half_vals[ch] = conversion_HALF_FLOAT(color[ch]);
		*((half *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0] *channels + ch) = half_vals[ch];
  }
  return;
}

void _set_pixelf_FLOAT(CLMem* image, int4 ijk, float4 color) {
  void * buffer = image->ptr; 
	int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
		*((float *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch) = color[ch];
  }
  return;
}

////////////////////////// conversion
uchar conversion_UNORM_INT8(float f) {
  return convert_uchar_sat_rte(f * 255.0f);
}
ushort conversion_UNORM_INT16(float f) {
  return convert_ushort_sat_rte(f * 65535.0f);
}
half conversion_HALF_FLOAT(float f) {
  half h;
  vstore_half(f, 0, &h);
  return h;
}
