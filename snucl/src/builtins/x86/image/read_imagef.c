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
#include "read_imagef.h"
#include <relational/cl_builtins_relational.h>
#include <math/cl_builtins_math.h>
#include <integer/cl_builtins_integer.h>
#include <vector/cl_builtins_vector.h>
#include <stdio.h>

///////////////////// read imagef
// no sampler
float4 read_imagef(image_t _image, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  coord = getImageArrayCoord(image, coord);
  return get_pixelf(image, coord);
}
float4 read_imagef(image_t _image, int2 coord) 
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  _coord = getImageArrayCoord(image, _coord);
  return get_pixelf(image, _coord);
}
float4 read_imagef(image_t _image, int coord)
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord;
  return get_pixelf(image, _coord);
}
float4 read_imagef(image3d_t _image, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  return get_pixelf(image, coord);
}
// with sampler
float4 read_imagef(image_t _image, sampler_t sampler, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imagef(image, sampler, coord, 3);
}
float4 read_imagef(image_t _image, sampler_t sampler, int2 coord) 
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  return _read_imagef(image, sampler, _coord, 2);
}
float4 read_imagef(image_t _image, sampler_t sampler, int coord)
{
  CLMem* image = (CLMem*) _image;
  int4 _coord = {0, 0, 0, 0};
  _coord[0] = coord;
  return _read_imagef(image, sampler, _coord, 1);
}
float4 read_imagef(image3d_t _image, sampler_t sampler, int4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imagef(image, sampler, coord, 3);
}
float4 read_imagef(image_t _image, sampler_t sampler, float4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imagef(image, sampler, coord, 3);
}
float4 read_imagef(image_t _image, sampler_t sampler, float2 coord)
{
  CLMem* image = (CLMem*) _image;
  float4 _coord = {0.0f, 0.0f, 0.0f, 0.0f};
  _coord[0] = coord[0];
  _coord[1] = coord[1];
  return _read_imagef(image, sampler, _coord, 2);
}
float4 read_imagef(image_t _image, sampler_t sampler, float coord)
{
  CLMem* image = (CLMem*) _image;
  float4 _coord = {0.0f, 0.0f, 0.0f, 0.0f};
  _coord[0] = coord;
  return _read_imagef(image, sampler, _coord, 1);
}
float4 read_imagef(image3d_t _image, sampler_t sampler, float4 coord)
{
  CLMem* image = (CLMem*) _image;
  return _read_imagef(image, sampler, coord, 3);
}

float4 _read_imagef(CLMem* image, sampler_t sampler, int4 coord, int dim_size)
{
  float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
  // variables for sampling
  int4 str = {0, 0, 0, 0}, uvw = {0, 0, 0, 0};
  int4 ijk = {0, 0, 0, 0}, ijk0 = {0, 0, 0, 0}, ijk1 = {0, 0, 0, 0}, abc = {0, 0, 0, 0};
  int4 dim = {0, 0, 0, 0};
  int d;

  // initialize
 
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

	  ret_val = get_pixelf(image, ijk);
  }

  return ret_val;
}
float4 _read_imagef(CLMem* image, sampler_t sampler, float4 coord, int dim_size)
{
  float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
  // variables for sampling
  float4 str = {0, 0, 0, 0}, uvw = {0, 0, 0, 0};
  int4 ijk = {0, 0, 0, 0}, ijk0 = {0, 0, 0, 0}, ijk1 = {0, 0, 0, 0};
  float4 abc = {0, 0, 0, 0};
  int4 dim = {0, 0, 0, 0};
  int d;

  // initialize
  dim = getDim(image);
  for( d = 0; d < dim_size; d++ ) {
    if( isnan(coord[d]) || isfinite(coord[d]) ) {
      //return ret_val;
    }
  }

  // normalized coord
  for( d = 0; d < dim_size; d++ )
    str[d] = coord[d];

  uvw = str;
  
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
    ret_val = get_pixelf(image, ijk);

  } else if( sampler & CLK_FILTER_LINEAR ) {
    if( sampler & CLK_ADDRESS_CLAMP_TO_EDGE ) {
      for( d = 0; d < dim_size; d++ ) {
        addr_clamp((int)floor(uvw[d] - 0.5), 0, dim[d]-1, &(ijk0[d])); 
        addr_clamp((int)floor(uvw[d] - 0.5)+1, 0, dim[d]-1, &(ijk1[d]));
      }
    } else if( sampler & CLK_ADDRESS_CLAMP ) {
      for( d = 0; d < dim_size; d++ ) {
        addr_clamp((int)floor(uvw[d] - 0.5), -1, dim[d], &ijk0[d]); 
        addr_clamp((int)floor(uvw[d] - 0.5)+1, -1, dim[d], &ijk1[d]);
      }
    } else if( sampler & CLK_ADDRESS_NONE ) {
      for( d = 0; d < dim_size; d++ ) {
        ijk0[d] = (int)floor(uvw[d] - 0.5);
        ijk1[d] = (int)floor(uvw[d] - 0.5) + 1;
      } 
    } else if( sampler & CLK_ADDRESS_REPEAT ) {
      for( d = 0; d < dim_size; d++ )
        addr_repeat_linear(str[d], dim[d], &ijk0[d], &ijk1[d], &uvw[d]);
    } else if( sampler & CLK_ADDRESS_MIRRORED_REPEAT ) {
      for( d = 0; d < dim_size; d++ )
        addr_mirror_linear(str[d], dim[d], &ijk0[d], &ijk1[d], &uvw[d]);
    }
  
    for( d = 0; d < dim_size; d++ ) {
      abc[d] = frac(uvw[d] - 0.5);
    }
    
    if( isImageArray(image) ) {
      int arrayDim = dim_size;
      ijk0[arrayDim] = uvw[arrayDim];
      ijk1[arrayDim] = uvw[arrayDim];
    }

	// get pixel according to channel type
	ret_val = get_pixelf(image, ijk0, ijk1, abc);
  }
 

  return ret_val;
}


///////////////////// get pixel
float4 get_pixelf(CLMem* image, int4 ijk) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
  cl_channel_order channel_order = image->image_format.image_channel_order;
	uint channel_type = image->image_format.image_channel_data_type;
  int4 dim = getDim(image);
  if( isOutside(ijk, dim) )
      return ret_val;

	if( channel_type == CL_UNORM_INT8 ) {
		ret_val = _get_pixelf_UNORM_INT8(image, ijk);
	} else if(  channel_type == CL_UNORM_INT16 ) {
		ret_val = _get_pixelf_UNORM_INT16(image, ijk);
	} else if( channel_type == CL_HALF_FLOAT ) {
		ret_val = _get_pixelf_HALF_FLOAT(image, ijk);
	} else if( channel_type == CL_FLOAT ) {
		ret_val = _get_pixelf_FLOAT(image, ijk);
	} 
  if( channel_order == CL_BGRA ) {
    float tmp;
    tmp = ret_val[0]; 
    ret_val[0] = ret_val[2];
    ret_val[2] = tmp;   
  } 
	return ret_val;
}
float4 get_pixelf(CLMem* image, int4 ijk0, int4 ijk1, float4 abc) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
  cl_channel_order channel_order = image->image_format.image_channel_order;
	uint channel_type = image->image_format.image_channel_data_type;

	if( channel_type == CL_UNORM_INT8 ) {
		ret_val = _get_pixelf_UNORM_INT8(image, ijk0, ijk1, abc);
	} else if(  channel_type == CL_UNORM_INT16 ) {
		ret_val = _get_pixelf_UNORM_INT16(image, ijk0, ijk1, abc);
	} else if( channel_type == CL_HALF_FLOAT ) {
		ret_val = _get_pixelf_HALF_FLOAT(image, ijk0, ijk1, abc);
	} else if( channel_type == CL_FLOAT ) {
		ret_val = _get_pixelf_FLOAT(image, ijk0, ijk1, abc);
	}
  if( channel_order == CL_BGRA ) {
    float tmp;
    tmp = ret_val[0]; 
    ret_val[0] = ret_val[2];
    ret_val[2] = tmp;   
  } 
	return ret_val;
}

float4 _get_pixelf_UNORM_INT8(CLMem* image, int4 ijk) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
	void * buffer = image->ptr; 
  int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;
 
  for( ch = 0; ch < 4; ch++ ) {
		double dval = *((uchar *)buffer+ ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch);
		ret_val[ch] = normalize_UNORM_INT8(dval);
	}
	return ret_val;
}
float4 _get_pixelf_UNORM_INT16(CLMem* image, int4 ijk) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
	void * buffer = image->ptr; 
  int4 dim = getDim(image);
	int channels = image->channels;
	int ch = 0;

	for( ch = 0; ch < 4; ch++ ) {
		double dval = *((ushort *)buffer + ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch);
		ret_val[ch] = normalize_UNORM_INT16(dval);
	}
	return ret_val;
}
float4 _get_pixelf_HALF_FLOAT(CLMem* image, int4 ijk) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
	half half_vals[4];
  void * buffer = image->ptr; 
  int4 dim = getDim(image);
  int channels = image->channels;
  int ch = 0;
  
  for( ch = 0; ch < 4; ch++ ) {
    half_vals[ch] = *((half *)buffer + ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch);
	  ret_val[ch] = vload_half(0, &half_vals[ch]);
  }

  return ret_val;
}
float4 _get_pixelf_FLOAT(CLMem* image, int4 ijk) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
	void * buffer = image->ptr; 
  int4 dim = getDim(image);
  int channels = image->channels;
  int ch = 0;
  
  for( ch = 0; ch < 4; ch++ ) {
    double dval = *((float *)buffer + ijk[2]*dim[0]*channels*dim[1] + ijk[1]*dim[0]*channels + ijk[0]*channels + ch);
	  ret_val[ch] = dval;
  }
	return ret_val;
}

float4 _get_pixelf_UNORM_INT8(CLMem* image, int4 ijk0, int4 ijk1, float4 abc) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
  void * buffer = image->ptr; 
  int channels = image->channels;
  cl_mem_object_type type = image->image_desc.image_type;
  uchar pixel[8]; 
  int ch = 0;
  int4 dim = getDim(image);
  
  if( type == CL_MEM_OBJECT_IMAGE1D || type == CL_MEM_OBJECT_IMAGE1D_BUFFER ) {
    abc[1] = 1;
    abc[2] = 1;
  } else if( type == CL_MEM_OBJECT_IMAGE2D || type == CL_MEM_OBJECT_IMAGE1D_ARRAY ) {
    abc[2] = 1;
  }
 
  // get pixel
  for( ch = 0; ch < 4; ch++ ) {

    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[0] = 0;
    else
      pixel[0] = *((uchar *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[1] = 0;
    else
      pixel[1] = *((uchar *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch); 
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[2] = 0;
    else
      pixel[2] = *((uchar *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[3] = 0;
    else
      pixel[3] = *((uchar *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[4] = 0;
    else 
      pixel[4] = *((uchar *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch); 
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[5] = 0;
    else 
      pixel[5] = *((uchar *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[6] = 0;
    else 
      pixel[6] = *((uchar *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[7] = 0;
    else 
      pixel[7] = *((uchar *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);    

    double dval = (1 - abc[0]) * (1 - abc[1]) * (1 - abc[2]) * pixel[0] 
     + abc[0] * (1 - abc[1]) * (1 - abc[2]) * pixel[1] 
     + (1 - abc[0]) * abc[1] * (1 - abc[2]) * pixel[2] 
     + abc[0] * abc[1] * (1 - abc[2]) * pixel[3] 
     + (1 - abc[0]) * (1 - abc[1]) * abc[2] * pixel[4] 
     + abc[0] * (1 - abc[1]) * abc[2] * pixel[5] 
     + (1 - abc[0]) * abc[1] * abc[2] * pixel[6] 
     + abc[0] * abc[1] * abc[2] * pixel[7];
	  
    ret_val[ch] = normalize_UNORM_INT8(dval);
  }
	return ret_val;
}
float4 _get_pixelf_UNORM_INT16(CLMem* image, int4 ijk0, int4 ijk1, float4 abc) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
  void * buffer = image->ptr; 
  int channels = image->channels;
  cl_mem_object_type type = image->image_desc.image_type;
  ushort pixel[8]; 
  int ch = 0;
  int4 dim = getDim(image);
  
  if( type == CL_MEM_OBJECT_IMAGE1D || type == CL_MEM_OBJECT_IMAGE1D_BUFFER ) {
    abc[1] = 1;
    abc[2] = 1;
  } else if( type == CL_MEM_OBJECT_IMAGE2D || type == CL_MEM_OBJECT_IMAGE1D_ARRAY ) {
    abc[2] = 1;
  }
 
  // get pixel
  for( ch = 0; ch < 4; ch++ ) {

    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[0] = 0;
    else
      pixel[0] = *((ushort *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[1] = 0;
    else
      pixel[1] = *((ushort *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch); 
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[2] = 0;
    else
      pixel[2] = *((ushort *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[3] = 0;
    else
      pixel[3] = *((ushort *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[4] = 0;
    else 
      pixel[4] = *((ushort *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch); 
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[5] = 0;
    else 
      pixel[5] = *((ushort *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[6] = 0;
    else 
      pixel[6] = *((ushort *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[7] = 0;
    else 
      pixel[7] = *((ushort *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);    
  

   double dval = (1 - abc[0]) * (1 - abc[1]) * (1 - abc[2]) * pixel[0] 
     + abc[0] * (1 - abc[1]) * (1 - abc[2]) * pixel[1] 
     + (1 - abc[0]) * abc[1] * (1 - abc[2]) * pixel[2] 
     + abc[0] * abc[1] * (1 - abc[2]) * pixel[3] 
     + (1 - abc[0]) * (1 - abc[1]) * abc[2] * pixel[4] 
     + abc[0] * (1 - abc[1]) * abc[2] * pixel[5] 
     + (1 - abc[0]) * abc[1] * abc[2] * pixel[6] 
     + abc[0] * abc[1] * abc[2] * pixel[7];
	  
    ret_val[ch] = normalize_UNORM_INT16(dval);
  }
	return ret_val;
}
float4 _get_pixelf_HALF_FLOAT(CLMem* image, int4 ijk0, int4 ijk1, float4 abc) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
	half half_vals[4];  
  void * buffer = image->ptr; 
  int channels = image->channels;
  cl_mem_object_type type = image->image_desc.image_type;
  half pixel[8]; 
  int ch = 0;
  int4 dim = getDim(image);
  
  if( type == CL_MEM_OBJECT_IMAGE1D || type == CL_MEM_OBJECT_IMAGE1D_BUFFER ) {
    abc[1] = 1;
    abc[2] = 1;
  } else if( type == CL_MEM_OBJECT_IMAGE2D || type == CL_MEM_OBJECT_IMAGE1D_ARRAY ) {
    abc[2] = 1;
  }
 
  // get pixel
  for( ch = 0; ch < 4; ch++ ) {

    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[0] = 0;
    else
      pixel[0] = *((half *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[1] = 0;
    else
      pixel[1] = *((half *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch); 
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[2] = 0;
    else
      pixel[2] = *((half *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[3] = 0;
    else
      pixel[3] = *((half *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[4] = 0;
    else 
      pixel[4] = *((half *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch); 
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[5] = 0;
    else 
      pixel[5] = *((half *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[6] = 0;
    else 
      pixel[6] = *((half *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[7] = 0;
    else 
      pixel[7] = *((half *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);    
  
   ret_val[ch] = (1 - abc[0]) * (1 - abc[1]) * (1 - abc[2]) * vload_half(0, &pixel[0]) 
     + abc[0] * (1 - abc[1]) * (1 - abc[2]) * vload_half(0, &pixel[1]) 
     + (1 - abc[0]) * abc[1] * (1 - abc[2]) * vload_half(0, &pixel[2]) 
     + abc[0] * abc[1] * (1 - abc[2]) * vload_half(0, &pixel[3]) 
     + (1 - abc[0]) * (1 - abc[1]) * abc[2] * vload_half(0, &pixel[4])  
     + abc[0] * (1 - abc[1]) * abc[2] * vload_half(0, &pixel[5]) 
     + (1 - abc[0]) * abc[1] * abc[2] * vload_half(0, &pixel[6]) 
     + abc[0] * abc[1] * abc[2] * vload_half(0, &pixel[7]);
  }
	return ret_val;
}
float4 _get_pixelf_FLOAT(CLMem* image, int4 ijk0, int4 ijk1, float4 abc) {
	float4 ret_val = {0.0f, 0.0f, 0.0f, 0.0f};   // return value
  void * buffer = image->ptr; 
  int channels = image->channels;
  cl_mem_object_type type = image->image_desc.image_type;
  float pixel[8]; 
  int ch = 0;
  int4 dim = getDim(image);
  
  if( type == CL_MEM_OBJECT_IMAGE1D || type == CL_MEM_OBJECT_IMAGE1D_BUFFER ) {
    abc[1] = 1;
    abc[2] = 1;
  } else if( type == CL_MEM_OBJECT_IMAGE2D || type == CL_MEM_OBJECT_IMAGE1D_ARRAY ) {
    abc[2] = 1;
  }
 
  // get pixel
  for( ch = 0; ch < 4; ch++ ) {

    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[0] = 0;
    else
      pixel[0] = *((float *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[1] = 0;
    else
      pixel[1] = *((float *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch); 
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[2] = 0;
    else
      pixel[2] = *((float *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk0[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[3] = 0;
    else
      pixel[3] = *((float *)buffer + ijk0[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[4] = 0;
    else 
      pixel[4] = *((float *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk0[0]*channels + ch); 
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk0[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[5] = 0;
    else 
      pixel[5] = *((float *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk0[1]*dim[0]*channels + ijk1[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk0[0], dim[0]) )
      pixel[6] = 0;
    else 
      pixel[6] = *((float *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk0[0]*channels + ch);
    if( isOutside(ijk1[2], dim[2]) || isOutside(ijk1[1], dim[1]) || isOutside(ijk1[0], dim[0]) )
      pixel[7] = 0;
    else 
      pixel[7] = *((float *)buffer + ijk1[2]*dim[0]*channels*dim[1] + ijk1[1]*dim[0]*channels + ijk1[0]*channels + ch);    
  

   double dval  = (1 - abc[0]) * (1 - abc[1]) * (1 - abc[2]) * pixel[0] 
     + abc[0] * (1 - abc[1]) * (1 - abc[2]) * pixel[1] 
     + (1 - abc[0]) * abc[1] * (1 - abc[2]) * pixel[2] 
     + abc[0] * abc[1] * (1 - abc[2]) * pixel[3] 
     + (1 - abc[0]) * (1 - abc[1]) * abc[2] * pixel[4] 
     + abc[0] * (1 - abc[1]) * abc[2] * pixel[5] 
     + (1 - abc[0]) * abc[1] * abc[2] * pixel[6] 
     + abc[0] * abc[1] * abc[2] * pixel[7];
	  
    ret_val[ch] = dval;
  }
	return ret_val;
}


float normalize_UNORM_INT8(double dval) {
	return (float)(dval / 255.0f);
}
float normalize_UNORM_INT16(double dval) {
	return (float)(dval / 65535.0f);
}
