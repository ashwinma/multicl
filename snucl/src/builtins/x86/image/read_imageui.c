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
#include "read_imageui.h"
#include <relational/cl_builtins_relational.h>
#include <math/cl_builtins_math.h>
#include <integer/cl_builtins_integer.h>
#include <vector/cl_builtins_vector.h>
#include <stdio.h>

///////////////////// read imageui
// no sampler
uint4 read_imageui(image_t _image, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  coord = getImageArrayCoord(image, coord);
  return get_pixelui(image, coord);
}
uint4 read_imageui(image_t _image, int2 coord) 
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  _coord = getImageArrayCoord(image, _coord);
  return get_pixelui(image, _coord);
}
uint4 read_imageui(image_t _image, int coord)
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord;
  return get_pixelui(image, _coord);
}
uint4 read_imageui(image3d_t _image, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  return get_pixelui(image, coord);
}
// with sampler
uint4 read_imageui(image_t _image, sampler_t sampler, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imageui(image, sampler, coord, 3);
}
uint4 read_imageui(image_t _image, sampler_t sampler, int2 coord) 
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  return _read_imageui(image, sampler, _coord, 2);
}
uint4 read_imageui(image_t _image, sampler_t sampler, int coord)
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord;
  return _read_imageui(image, sampler, _coord, 1);
}
uint4 read_imageui(image3d_t  _image, sampler_t sampler, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imageui(image, sampler, coord, 3);
}
uint4 read_imageui(image_t _image, sampler_t sampler, float4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imageui(image, sampler, coord, 3);
}
uint4 read_imageui(image_t _image, sampler_t sampler, float2 coord)
{
  CLMem* image = (CLMem*) _image;
  float4 _coord = {0.0f, 0.0f, 0.0f, 0.0f};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  return _read_imageui(image, sampler, _coord, 2);
}
uint4 read_imageui(image_t _image, sampler_t sampler, float coord)
{
  CLMem* image = (CLMem*) _image;
  float4 _coord = {0.0f, 0.0f, 0.0f, 0.0f};
  _coord[0] = coord;
  return _read_imageui(image, sampler, _coord, 1);
}
uint4 read_imageui(image3d_t _image, sampler_t sampler, float4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imageui(image, sampler, coord, 3);
}
uint4 _read_imageui(CLMem* image, sampler_t sampler, int4 coord, int dim_size) {
  uint4 ret_val = {0, 0, 0, 0};   // return value
  // variables for sampling
  int4 str = {0, 0, 0, 0}, uvw = {0, 0, 0, 0};
  int4 ijk = {0, 0, 0, 0}, ijk0 = {0, 0, 0, 0}, ijk1 = {0, 0, 0, 0}, abc = {0, 0, 0, 0};
  int4 dim = {0, 0, 0, 0};
  int d;

  dim = getDim(image);
  for( d = 0; d < dim_size; d++ )
    str[d] = coord[d];

  uvw = str;

    // normalized coord
  if( sampler & CLK_NORMALIZED_COORDS_FALSE ) {
    uvw = str;
  }
  
  if( isImageArray(image) ) {
    uvw = getImageArrayCoord(image, uvw);
    dim_size--;
  }

  // addressing mode
  if( sampler & CLK_FILTER_NEAREST ) {
    if( sampler & CLK_ADDRESS_CLAMP_TO_EDGE ) {
      for( d = 0; d < dim_size; d++ )
        addr_clamp((int)floor((double)uvw[d]), 0, dim[d]-1, &ijk[d]); 
    } else if( sampler & CLK_ADDRESS_CLAMP ) {
      for( d = 0; d < dim_size; d++ )
        addr_clamp((int)floor((double)uvw[d]), -1, dim[d], &ijk[d]); 
    } else if( sampler & CLK_ADDRESS_NONE ) {
      for( d = 0; d < dim_size; d++ )
        ijk[d] = (int)floor((double)uvw[d]);
    } 
    
    if( isImageArray(image) ) {
      int arrayDim = dim_size;
      ijk[arrayDim] = uvw[arrayDim];
    }

    // get pixel according to channel type
    ret_val = get_pixelui(image, ijk);
  } 
  return ret_val;
}
uint4 _read_imageui(CLMem* image, sampler_t sampler, float4 coord, int dim_size) {
  uint4 ret_val = {0, 0, 0, 0};   // return value
  // variables for sampling
  float4 str = {0.0f, 0.0f, 0.0f, 0.0f}, uvw = {0.0f, 0.0f, 0.0f, 0.0f};
  int4 ijk = {0, 0, 0, 0}, ijk0 = {0, 0, 0, 0}, ijk1 = {0, 0, 0, 0}, abc = {0, 0, 0, 0};
  int4 dim = {0, 0, 0, 0};
  int d;

  // initialize
  dim = getDim(image); 
  for( d = 0; d < dim_size; d++ ) {
//    if( isnan(coord[d]) || isfinite(coord[d]) )
//      return ret_val;
  }

  for( d = 0; d < dim_size; d++ )
    str[d] = coord[d];
    
  uvw = str;
  
  // normalized coord
  if( sampler & CLK_NORMALIZED_COORDS_FALSE ) {
    uvw = str;
  } else if( sampler & CLK_NORMALIZED_COORDS_TRUE ) {
    for( d = 0; d < dim_size; d++ )
      uvw[d] = str[d] * dim[d];
  }

  if( isImageArray(image) ) {
    uvw = getImageArrayCoord(image, uvw);
    dim_size--;
  }
    // addressing mode
  if( sampler & CLK_FILTER_NEAREST ) {
    if( sampler & CLK_ADDRESS_CLAMP_TO_EDGE ) {
      for( d = 0; d < dim_size; d++ )
        addr_clamp((int)floor(uvw[d]), 0, dim[d]-1, &ijk[d]); 
    } else if( sampler & CLK_ADDRESS_CLAMP ) {
      for( d = 0; d < dim_size; d++ )
        addr_clamp((int)floor(uvw[d]), -1, dim[d], &ijk[d]); 
    } else if( sampler & CLK_ADDRESS_NONE ) {
      for( d = 0; d < dim_size; d++ )
        ijk[d] = (int)floor(uvw[d]);
    } else if( sampler & CLK_ADDRESS_REPEAT ) {
      for( d = 0; d < dim_size; d++ )
        addr_repeat_nearest(str[d], dim[d], &ijk[d]);
    } else if( sampler & CLK_ADDRESS_MIRRORED_REPEAT ) {
      for( d = 0; d < dim_size; d++ )
        addr_mirror_nearest(str[d], dim[d], &ijk[d]);
    }
    
    if( isImageArray(image) ) {
      int arrayDim = dim_size;
      ijk[arrayDim] = uvw[arrayDim];
    }
    
    // get pixel according to channel type
    ret_val = get_pixelui(image, ijk);
  } 
  return ret_val;
}


///////////////////// get pixelui
uint4 get_pixelui(CLMem* image, int4 ijk) {
	uint4 ret_val = {0, 0, 0, 0};   // return value
  cl_channel_order channel_order = image->image_format.image_channel_order;
	uint channel_type = image->image_format.image_channel_data_type;
  int4 dim = getDim(image);

//  printf("dim : %d, %d, %d, %d\n", dim[0], dim[1], dim[2], dim[3]);
  if( isOutside(ijk, dim) )
      return ret_val;

	if( channel_type == CL_UNSIGNED_INT8 ) {
		ret_val = _get_pixelui_UNSIGNED_INT8(image, ijk);
	} else if(  channel_type == CL_UNSIGNED_INT16 ) {
		ret_val = _get_pixelui_UNSIGNED_INT16(image, ijk);
	} else if( channel_type == CL_UNSIGNED_INT32 ) {
		ret_val = _get_pixelui_UNSIGNED_INT32(image, ijk);
	} 
  if( channel_order == CL_BGRA ) {
    uint tmp;
    tmp = ret_val[0]; 
    ret_val[0] = ret_val[2];
    ret_val[2] = tmp;   
  }

	return ret_val;
}

uint4 _get_pixelui_UNSIGNED_INT8(CLMem* image, int4 ijk) {
	uint4 ret_val = {0, 0, 0, 0};   // return value
	uchar uchar_vals[4];
  void * buffer = image->ptr; 
  int4 dim = getDim(image);
  int channels = image->channels;
  int ch = 0;

  for( ch = 0; ch < 4; ch++ ) {
    uchar_vals[ch] = *((uchar *)buffer + ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch);
    ret_val[ch] = (uint)(uchar_vals[ch]);
  }
  return ret_val;
}

uint4 _get_pixelui_UNSIGNED_INT16(CLMem* image, int4 ijk) {
	uint4 ret_val = {0, 0, 0, 0};   // return value
 	ushort ushort_vals[4];
	void * buffer = image->ptr; 
  int4 dim = getDim(image);
  int channels = image->channels;
  int ch = 0;

  for( ch = 0; ch < 4; ch++ ) {
    ushort_vals[ch] = *((ushort *)buffer + ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch);
    ret_val[ch] = (uint)(ushort_vals[ch]);
  }
  return ret_val;
}

uint4 _get_pixelui_UNSIGNED_INT32(CLMem* image, int4 ijk) {
	uint4 ret_val = {0, 0, 0, 0};   // return value
  uint uint_vals[4];
  void * buffer = image->ptr; 
  int4 dim = getDim(image);
  int channels = image->channels;
  int ch = 0;

  // get pixel
  for( ch = 0; ch < 4; ch++ ) {
    uint_vals[ch] = *((uint *)buffer + ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch);
    ret_val[ch] = (uint)(uint_vals[ch]);
  }
  return ret_val;
}
