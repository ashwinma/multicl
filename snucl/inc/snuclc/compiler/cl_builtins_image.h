/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2011-2013 Seoul National University.                        */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE          */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   Center for Manycore Programming                                         */
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 151-744, Korea                         */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jungwon Kim, Sangmin Seo, Gangwon Jo, Jun Lee, Jeongho Nah,             */
/*   Jungho Park, Junghyun Kim, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/* This file is based on the SNU-SAMSUNG OpenCL Compiler and is distributed  */
/* under GNU General Public License.                                         */
/* See LICENSE.SNU-SAMSUNG_OpenCL_C_Compiler.TXT for details.                */
/*****************************************************************************/

#ifndef __CL_BUILTINS_IMAGE_H__
#define __CL_BUILTINS_IMAGE_H__

#include "cl_types.h"

#define __IMAGE_SUPPORT__ 1
/* 6.12.14 Image Read and Write Functions */

/* 6.12.14.2 Built-in Image Read Functions */
float4 read_imagef(image2d_t, sampler_t, int2) __attribute__((overloadable));
float4 read_imagef(image2d_t, sampler_t, float2) __attribute__((overloadable));

int4   read_imagei(image2d_t, sampler_t, int2) __attribute__((overloadable));
int4   read_imagei(image2d_t, sampler_t, float2) __attribute__((overloadable));
uint4  read_imageui(image2d_t, sampler_t, int2) __attribute__((overloadable));
uint4  read_imageui(image2d_t, sampler_t, float2) __attribute__((overloadable));

float4 read_imagef(image3d_t, sampler_t, int4) __attribute__((overloadable));
float4 read_imagef(image3d_t, sampler_t, float4) __attribute__((overloadable));

int4   read_imagei(image3d_t, sampler_t, int4) __attribute__((overloadable));
int4   read_imagei(image3d_t, sampler_t, float4) __attribute__((overloadable));
uint4  read_imageui(image3d_t, sampler_t, int4) __attribute__((overloadable));
uint4  read_imageui(image3d_t, sampler_t, float4) __attribute__((overloadable));

float4 read_imagef(image2d_array_t, sampler_t, int4) __attribute__((overloadable));
float4 read_imagef(image2d_array_t, sampler_t, float4) __attribute__((overloadable));

int4   read_imagei(image2d_array_t, sampler_t, int4) __attribute__((overloadable));
int4   read_imagei(image2d_array_t, sampler_t, float4) __attribute__((overloadable));
uint4  read_imageui(image2d_array_t, sampler_t, int4) __attribute__((overloadable));
uint4  read_imageui(image2d_array_t, sampler_t, float4) __attribute__((overloadable));

float4 read_imagef(image1d_t, sampler_t, int) __attribute__((overloadable));
float4 read_imagef(image1d_t, sampler_t, float) __attribute__((overloadable));

int4   read_imagei(image1d_t, sampler_t, int) __attribute__((overloadable));
int4   read_imagei(image1d_t, sampler_t, float) __attribute__((overloadable));
uint4  read_imageui(image1d_t, sampler_t, int) __attribute__((overloadable));
uint4  read_imageui(image1d_t, sampler_t, float) __attribute__((overloadable));

float4 read_imagef(image1d_array_t, sampler_t, int2) __attribute__((overloadable));
float4 read_imagef(image1d_array_t, sampler_t, float4) __attribute__((overloadable));

int4   read_imagei(image1d_array_t, sampler_t, int2) __attribute__((overloadable));
int4   read_imagei(image1d_array_t, sampler_t, float2) __attribute__((overloadable));
uint4  read_imageui(image1d_array_t, sampler_t, int2) __attribute__((overloadable));
uint4  read_imageui(image1d_array_t, sampler_t, float2) __attribute__((overloadable));


/* 6.12.14.3 Built-in Image Sampler-less Read Functions */
float4 read_imagef(image2d_t, int2) __attribute__((overloadable));

int4   read_imagei(image2d_t, int2) __attribute__((overloadable));
uint4  read_imageui(image2d_t, int2) __attribute__((overloadable));

float4 read_imagef(image3d_t, int4) __attribute__((overloadable));

int4   read_imagei(image3d_t, int4) __attribute__((overloadable));
uint4  read_imageui(image3d_t, int4) __attribute__((overloadable));

float4 read_imagef(image2d_array_t, int4) __attribute__((overloadable));

int4   read_imagei(image2d_array_t, int4) __attribute__((overloadable));
uint4  read_imageui(image2d_array_t, int4) __attribute__((overloadable));

float4 read_imagef(image1d_t, int) __attribute__((overloadable));
float4 read_imagef(image1d_buffer_t, int) __attribute__((overloadable));

int4   read_imagei(image1d_t, int) __attribute__((overloadable));
uint4  read_imageui(image1d_t, int) __attribute__((overloadable));
int4   read_imagei(image1d_buffer_t, int) __attribute__((overloadable));
uint4  read_imageui(image1d_buffer_t, int) __attribute__((overloadable));

float4 read_imagef(image1d_array_t, int2) __attribute__((overloadable));

int4   read_imagei(image1d_array_t, int2) __attribute__((overloadable));
uint4  read_imageui(image1d_array_t, int2) __attribute__((overloadable));


/* 6.12.14.4 Built-in Image Write Functions */
void   write_imagef(image2d_t, int2, float4) __attribute__((overloadable));
void   write_imagei(image2d_t, int2, int4) __attribute__((overloadable));
void   write_imageui(image2d_t, int2, uint4) __attribute__((overloadable));

void   write_imagef(image2d_array_t, int4, float4) __attribute__((overloadable));
void   write_imagei(image2d_array_t, int4, int4) __attribute__((overloadable));
void   write_imageui(image2d_array_t, int4, uint4) __attribute__((overloadable));

void   write_imagef(image1d_t, int, float4) __attribute__((overloadable));
void   write_imagei(image1d_t, int, int4) __attribute__((overloadable));
void   write_imageui(image1d_t, int, uint4) __attribute__((overloadable));
void   write_imagef(image1d_buffer_t, int, float4) __attribute__((overloadable));
void   write_imagei(image1d_buffer_t, int, int4) __attribute__((overloadable));
void   write_imageui(image1d_buffer_t, int, uint4) __attribute__((overloadable));

void   write_imagef(image1d_array_t, int2, float4) __attribute__((overloadable));
void   write_imagei(image1d_array_t, int2, int4) __attribute__((overloadable));
void   write_imageui(image1d_array_t, int2, uint4) __attribute__((overloadable));


/* 6.12.14.5 Built-in Image Query Functions */
int    get_image_width(image1d_t) __attribute__((overloadable));
int    get_image_width(image1d_buffer_t) __attribute__((overloadable));
int    get_image_width(image2d_t) __attribute__((overloadable));
int    get_image_width(image3d_t) __attribute__((overloadable));
int    get_image_width(image1d_array_t) __attribute__((overloadable));
int    get_image_width(image2d_array_t) __attribute__((overloadable));

int    get_image_height(image2d_t) __attribute__((overloadable));
int    get_image_height(image3d_t) __attribute__((overloadable));
int    get_image_height(image2d_array_t) __attribute__((overloadable));

int    get_image_depth(image3d_t) __attribute__((overloadable));

int    get_image_channel_data_type(image1d_t) __attribute__((overloadable));
int    get_image_channel_data_type(image1d_buffer_t) __attribute__((overloadable));
int    get_image_channel_data_type(image2d_t) __attribute__((overloadable));
int    get_image_channel_data_type(image3d_t) __attribute__((overloadable));
int    get_image_channel_data_type(image1d_array_t) __attribute__((overloadable));
int    get_image_channel_data_type(image2d_array_t) __attribute__((overloadable));

int    get_image_channel_order(image1d_t) __attribute__((overloadable));
int    get_image_channel_order(image1d_buffer_t) __attribute__((overloadable));
int    get_image_channel_order(image2d_t) __attribute__((overloadable));
int    get_image_channel_order(image3d_t) __attribute__((overloadable));
int    get_image_channel_order(image1d_array_t) __attribute__((overloadable));
int    get_image_channel_order(image2d_array_t) __attribute__((overloadable));

//int2   get_image_dim(image2d_t) __attribute__((overloadable));
int2   get_image_dim(image2d_array_t) __attribute__((overloadable));
//int4   get_image_dim(image3d_t) __attribute__((overloadable));

size_t get_image_array_size(image2d_array_t) __attribute__((overloadable));
size_t get_image_array_size(image1d_array_t) __attribute__((overloadable));

#endif //__CL_BUILTINS_IMAGE_H__
