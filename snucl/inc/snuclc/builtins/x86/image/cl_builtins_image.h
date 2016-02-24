
#ifndef CL_BUILTINS_IMAGE_H 
#define CL_BUILTINS_IMAGE_H 

#include <cl_cpu_types.h>

#define __IMAGE_SUPPORT__ 1
/* 6.12.14 Image Read and Write Functions */

/* 6.12.14.2 Built-in Image Read Functions */
float4 read_imagef(image2d_t, sampler_t, int2);
float4 read_imagef(image2d_t, sampler_t, float2);

int4   read_imagei(image2d_t, sampler_t, int2);
int4   read_imagei(image2d_t, sampler_t, float2);
uint4  read_imageui(image2d_t, sampler_t, int2);
uint4  read_imageui(image2d_t, sampler_t, float2);

float4 read_imagef(image3d_t, sampler_t, int4);
float4 read_imagef(image3d_t, sampler_t, float4);

int4   read_imagei(image3d_t, sampler_t, int4);
int4   read_imagei(image3d_t, sampler_t, float4);
uint4  read_imageui(image3d_t, sampler_t, int4);
uint4  read_imageui(image3d_t, sampler_t, float4);

float4 read_imagef(image2d_array_t, sampler_t, int4);
float4 read_imagef(image2d_array_t, sampler_t, float4);

int4   read_imagei(image2d_array_t, sampler_t, int4);
int4   read_imagei(image2d_array_t, sampler_t, float4);
uint4  read_imageui(image2d_array_t, sampler_t, int4);
uint4  read_imageui(image2d_array_t, sampler_t, float4);

float4 read_imagef(image1d_t, sampler_t, int);
float4 read_imagef(image1d_t, sampler_t, float);

int4   read_imagei(image1d_t, sampler_t, int);
int4   read_imagei(image1d_t, sampler_t, float);
uint4  read_imageui(image1d_t, sampler_t, int);
uint4  read_imageui(image1d_t, sampler_t, float);

float4 read_imagef(image1d_array_t, sampler_t, int2);
float4 read_imagef(image1d_array_t, sampler_t, float4);

int4   read_imagei(image1d_array_t, sampler_t, int2);
int4   read_imagei(image1d_array_t, sampler_t, float2);
uint4  read_imageui(image1d_array_t, sampler_t, int2);
uint4  read_imageui(image1d_array_t, sampler_t, float2);


/* 6.12.14.3 Built-in Image Sampler-less Read Functions */
float4 read_imagef(image2d_t, int2);

int4   read_imagei(image2d_t, int2);
uint4  read_imageui(image2d_t, int2);

float4 read_imagef(image3d_t, int4);

int4   read_imagei(image3d_t, int4);
uint4  read_imageui(image3d_t, int4);

float4 read_imagef(image2d_array_t, int4);

int4   read_imagei(image2d_array_t, int4);
uint4  read_imageui(image2d_array_t, int4);

float4 read_imagef(image1d_t, int);
float4 read_imagef(image1d_buffer_t, int);

int4   read_imagei(image1d_t, int);
uint4  read_imageui(image1d_t, int);
int4   read_imagei(image1d_buffer_t, int);
uint4  read_imageui(image1d_buffer_t, int);

float4 read_imagef(image1d_array_t, int2);

int4   read_imagei(image1d_array_t, int2);
uint4  read_imageui(image1d_array_t, int2);


/* 6.12.14.4 Built-in Image Write Functions */
void   write_imagef(image2d_t, int2, float4);
void   write_imagei(image2d_t, int2, int4);
void   write_imageui(image2d_t, int2, uint4);

void   write_imagef(image2d_array_t, int4, float4);
void   write_imagei(image2d_array_t, int4, int4);
void   write_imageui(image2d_array_t, int4, uint4);

void   write_imagef(image1d_t, int, float4);
void   write_imagei(image1d_t, int, int4);
void   write_imageui(image1d_t, int, uint4);
void   write_imagef(image1d_buffer_t, int, float4);
void   write_imagei(image1d_buffer_t, int, int4);
void   write_imageui(image1d_buffer_t, int, uint4);

void   write_imagef(image1d_array_t, int2, float4);
void   write_imagei(image1d_array_t, int2, int4);
void   write_imageui(image1d_array_t, int2, uint4);


/* 6.12.14.5 Built-in Image Query Functions */
int    get_image_width(image1d_t);
int    get_image_width(image1d_buffer_t);
int    get_image_width(image2d_t);
int    get_image_width(image3d_t);
int    get_image_width(image1d_array_t);
int    get_image_width(image2d_array_t);

int    get_image_height(image2d_t);
int    get_image_height(image3d_t);
int    get_image_height(image2d_array_t);

int    get_image_depth(image3d_t);

int    get_image_channel_data_type(image1d_t);
int    get_image_channel_data_type(image1d_buffer_t);
int    get_image_channel_data_type(image2d_t);
int    get_image_channel_data_type(image3d_t);
int    get_image_channel_data_type(image1d_array_t);
int    get_image_channel_data_type(image2d_array_t);

int    get_image_channel_order(image1d_t);
int    get_image_channel_order(image1d_buffer_t);
int    get_image_channel_order(image2d_t);
int    get_image_channel_order(image3d_t);
int    get_image_channel_order(image1d_array_t);
int    get_image_channel_order(image2d_array_t);

//int2   get_image_dim(image2d_t);
int2   get_image_dim(image2d_array_t);
//int4   get_image_dim(image3d_t);

size_t get_image_array_size(image2d_array_t);
size_t get_image_array_size(image1d_array_t);
//float4 read_imagef(image_t image, int4 coord);
//float4 read_imagef(image_t image, int2 coord);
//float4 read_imagef(image_t image, int coord);
//
//float4 read_imagef(image_t image, sampler_t sampler, float coord);
//float4 read_imagef(image_t image, sampler_t sampler, float2 coord);
//float4 read_imagef(image_t image, sampler_t sampler, float4 coord);
//
//float4 read_imagef(image_t image, sampler_t sampler, int coord);
//float4 read_imagef(image_t image, sampler_t sampler, int2 coord);
//float4 read_imagef(image_t image, sampler_t sampler, int4 coord);
//
//int4 read_imagei(image_t image, int4 coord);
//int4 read_imagei(image_t image, int2 coord);
//int4 read_imagei(image_t image, int coord);
//
//int4 read_imagei(image_t image, sampler_t sampler, int4 coord);
//int4 read_imagei(image_t image, sampler_t sampler, int2 coord);
//int4 read_imagei(image_t image, sampler_t sampler, int coord);
//
//int4 read_imagei(image_t image, sampler_t sampler, float4 coord);
//int4 read_imagei(image_t image, sampler_t sampler, float2 coord);
//int4 read_imagei(image_t image, sampler_t sampler, float coord);
//
//uint4 read_imagei(image_t image, int4 coord);
//uint4 read_imagei(image_t image, int2 coord);
//uint4 read_imagei(image_t image, int coord);
//
//uint4 read_imageui(image_t image, sampler_t sampler, int2 coord);
//uint4 read_imageui(image_t image, sampler_t sampler, float2 coord);
//
//uint4 read_imageui(image_t image, sampler_t sampler, int4 coord);
//uint4 read_imageui(image_t image, sampler_t sampler, float4 coord);
//
//uint4 read_imageui(image_t image, sampler_t sampler, int coord);
//uint4 read_imageui(image_t image, sampler_t sampler, float coord);
//
//void write_imageui(image_t image, int2 coord, uint4 color);
//void write_imageui(image_t image, int4 coord, uint4 color);
//void write_imageui(image_t image, int coord, uint4 color);
//
//void write_imagef(image_t image, int2 coord, float4 color);
//void write_imagef(image_t, int4 coord, float4 color);
//void write_imagef(image2d_t image, int coord, float4 color);
//
//void write_imagei(image_t image, int2 coord, int4 color);
//void write_imagei(image_t image, int4 coord, int4 color);
//void write_imagei(image_t image, int coord, int4 color);
#endif


