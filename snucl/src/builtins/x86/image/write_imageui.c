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
#include "write_imageui.h"
#include <conversion/cl_builtins_conversion.h>

///////////////////// write_image
void write_imageui(image_t _image, int4 coord, uint4 color) {
  CLMem* image = (CLMem*) _image;
  coord = getImageArrayCoord(image, coord);
  set_pixelui(image, coord, color, 3);
}

void write_imageui(image_t _image, int2 coord, uint4 color) {
  
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  _coord = getImageArrayCoord(image, _coord);
  set_pixelui(image, _coord, color, 2);
}

void write_imageui(image_t _image, int coord, uint4 color) {
  CLMem* image = (CLMem*) _image;

  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord;
  
  set_pixelui(image, _coord, color, 1);
}

///////////////////// set pixel
void set_pixelui(CLMem* image, int4 ijk, uint4 color, int dim_size) {
	uint channel_type = image->image_format.image_channel_data_type;
  cl_channel_order channel_order;
  int4 dim;
  int d;
  
  channel_order = image->image_format.image_channel_order;
  dim = getDim(image);
  ijk = getImageArrayCoord(image, ijk);
  
  for( d = 0; d < dim_size; d++ ) {
    if( isOutside(ijk[d], dim[d]) )
      return;
  }
  
  if( channel_order == CL_BGRA ) {
    int tmp;
    tmp = color[0]; 
    color[0] = color[2];
    color[2] = tmp;   
  }
	
  if( channel_type == CL_UNSIGNED_INT8 ) {
		_set_pixelui_UNSIGNED_INT8(image, ijk, color);
	} else if(  channel_type == CL_UNSIGNED_INT16 ) {
		_set_pixelui_UNSIGNED_INT16(image, ijk, color);
	} else if( channel_type == CL_UNSIGNED_INT32 ) {
		_set_pixelui_UNSIGNED_INT32(image, ijk, color);
	} 
	return;
}

void _set_pixelui_UNSIGNED_INT8(CLMem* image, int4 ijk, uint4 color) {
	uchar uchar_vals[4];
  void * buffer = image->ptr; 
	int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
    uchar_vals[ch] = conversion_UNSIGNED_INT8(color[ch]);
		*((schar *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch) = uchar_vals[ch];
  }
  return;
}

void _set_pixelui_UNSIGNED_INT16(CLMem* image, int4 ijk, uint4 color) {
	ushort ushort_vals[4];
  void * buffer = image->ptr; 
	int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
    ushort_vals[ch] = conversion_UNSIGNED_INT16(color[ch]);
		*((ushort *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch) = ushort_vals[ch];
  }
  return;
}

void _set_pixelui_UNSIGNED_INT32(CLMem* image, int4 ijk, uint4 color) {
	uint uint_vals[4];
  void * buffer = image->ptr; 
	int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
		*((uint *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch) = color[ch];
  }
  return;
}

////////////////////////// conversion
uchar conversion_UNSIGNED_INT8(uint ui) {
  return convert_uchar_sat(ui);
}
ushort conversion_UNSIGNED_INT16(uint us) {
  return convert_ushort_sat(us);
}

