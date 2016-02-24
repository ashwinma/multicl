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

#include<stdio.h>
#include <cl_cpu_ops.h>
#include<math.h>


 typedef union
 {
   float value;

   unsigned int word;
 } ieee_float_shape_type;

#define GET_FLOAT_WORD(i,d)  do { ieee_float_shape_type gf_u;  gf_u.value = (d);   (*i) = gf_u.word;  } while (0)

#define SET_FLOAT_WORD(d,i) do {   ieee_float_shape_type sf_u;  sf_u.word = (i);   (*d) = sf_u.value; } while (0)

 typedef union
 {
   double value;

   ullong word;
 } ieee_double_shape_type;
 

#define GET_DOUBLE_WORD(i,d)  do { ieee_double_shape_type gf_u;  gf_u.value = (d);   (*i) = gf_u.word;  } while (0)

#define SET_DOUBLE_WORD(d,i) do {   ieee_double_shape_type sf_u;  sf_u.word = (i);   (*d) = sf_u.value; } while (0)



//schar, uchar
schar bitselect(schar a, schar b, schar c){
	
	schar rst;

	rst= ((~c&a ) | (c&b));

	return rst;


}


char2 bitselect(char2 a, char2 b, char2 c){

	char2 rst;
	int count =8;
	int i=0;
	for (i=0;i<2;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	
	}

	return rst;
}

char3 bitselect(char3 a, char3 b, char3 c){
	char3 rst;
	int count =8;
	int i=0;
	for (i=0;i<3;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	
	}
	return rst;
}

char4 bitselect(char4 a, char4 b, char4 c){
	char4 rst;
	int count =8;
	int i=0;
	for (i=0;i<4;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	
	}
	return rst;
}
char8 bitselect(char8 a, char8 b, char8 c){
	char8 rst;

	int count =8;
	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
char16 bitselect(char16 a, char16 b, char16 c){
	char16 rst;

	int count =8;
	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
		
	}
	return rst;
}
uchar bitselect(uchar a, uchar b, uchar c){
	schar rst;
	int count =8;

	
	rst= ((~c&a ) | (c&b));
	

	return rst;
}
uchar2 bitselect(uchar2 a, uchar2 b, uchar2 c){
	uchar2 rst;
	int count =8;
	int i=0;
	for (i=0;i<2;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
		
	}
	return rst;
}
uchar3 bitselect(uchar3 a, uchar3 b, uchar3 c){
	uchar3 rst;
	int count =8;
	int i=0;
	for (i=0;i<3;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	
	}
	return rst;
}

uchar4 bitselect(uchar4 a, uchar4 b, uchar4 c){
	uchar4 rst;
	int count =8;
	int i=0;
	for (i=0;i<4;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	
	}
	return rst;
}
uchar8 bitselect(uchar8 a, uchar8 b, uchar8 c){
	uchar8 rst;

	int count =8;
	int i=0;
	for (i=0;i<8;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
		
	}
	return rst;
}
uchar16 bitselect(uchar16 a, uchar16 b, uchar16 c){
	uchar16 rst;

	int count =8;
	int i=0;
	for (i=0;i<16;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	
	}
	return rst;
}

//short. shortn
short bitselect(short a, short b, short c){
	short rst;
	int count =16;

	rst= ((~c&a ) | (c&b));

	return rst;
}
short2 bitselect(short2 a, short2 b, short2 c){
	short2 rst;

	int count =16;
	int i=0;
	for (i=0;i<2;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
short3 bitselect(short3 a, short3 b, short3 c){
	short3 rst;

	int count = 16;
	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}

short4 bitselect(short4 a, short4 b, short4 c){
	short4 rst;

	int count = 16;
	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
short8 bitselect(short8 a, short8 b, short8 c){
	short8 rst;
	int count = 16;
	int i=0;
	for (i=0;i<8;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
short16 bitselect(short16 a, short16 b, short16 c){
	short16 rst;
	int count = 16 ;
	int i=0;
	for (i=0;i<16;i++)
	{
	rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}

ushort bitselect(ushort a, ushort b, ushort c){
	ushort rst;
	int count = 16 ;

	rst= ((~c&a ) | (c&b));

	return rst;
}
ushort2 bitselect(ushort2 a, ushort2 b, ushort2 c){
	ushort2 rst;

	int count =16 ;
	int i=0;
	for (i=0;i<2;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ushort3 bitselect(ushort3 a, ushort3 b, ushort3 c){
	ushort3 rst;

	int count = 16 ;
	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}

ushort4 bitselect(ushort4 a, ushort4 b, ushort4 c){
	ushort4 rst;

	int count = 16 ;
	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ushort8 bitselect(ushort8 a, ushort8 b, ushort8 c){
	ushort8 rst;
	int count = 16 ;
	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ushort16 bitselect(ushort16 a, ushort16 b, ushort16 c){
	ushort16 rst;
	int count =16 ;
	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}

//int, uint
int bitselect(int a, int b, int c){
	int rst;
	int count =32;

	rst= ((~c&a ) | (c&b));

	return rst;
}
int2 bitselect(int2 a, int2 b, int2 c){
	
	int2 rst;
	int count =32;
	int i=0;
	for (i=0;i<2;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
int3 bitselect(int3 a, int3 b, int3 c){
	int3 rst;

	int count =32;
	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}

int4 bitselect(int4 a, int4 b, int4 c){
	int4 rst;

	int count =32;
	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
int8 bitselect(int8 a, int8 b, int8 c){
	int8 rst;
	int count =32;
	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
int16 bitselect(int16 a, int16 b, int16 c){
	int16 rst;
	int count =32 ;
	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}

uint bitselect(uint a, uint b, uint c){
	uint rst;
	int count =32;

	rst= ((~c&a ) | (c&b));

	return rst;
}
uint2 bitselect(uint2 a, uint2 b, uint2 c){
	
	uint2 rst;
	int count =32;
	int i=0;
	for (i=0;i<2;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
uint3 bitselect(uint3 a, uint3 b, uint3 c){
	uint3 rst;

	int count =32;
	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
uint4 bitselect(uint4 a, uint4 b, uint4 c){
	uint4 rst;

	int count =32;
	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
uint8 bitselect(uint8 a, uint8 b, uint8 c){
	uint8 rst;
	int count =32;
	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
uint16 bitselect(uint16 a, uint16 b, uint16 c){
	uint16 rst;
	int count =32 ;
	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}

//llong, ullong
llong bitselect(llong a, llong b, llong c){
	llong rst;
	int count =32;

	rst= ((~c&a ) | (c&b));

	return rst;
}
long2 bitselect(long2 a, long2 b, long2 c){
	
	long2 rst;
	int count =32;
	int i=0;
	for (i=0;i<2;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
long3 bitselect(long3 a, long3 b, long3 c){
	
	long3 rst;
	int count =32;
	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
long4 bitselect(long4 a, long4 b, long4 c){
	
	long4 rst;
	int count =32;
	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
long8 bitselect(long8 a, long8 b, long8 c){
	long8 rst;
	int count =32;
	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
long16 bitselect(long16 a, long16 b, long16 c){
	long16 rst;
	int count =32;
	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ullong bitselect(ullong a, ullong b, ullong c){
	ullong rst;
	
	rst= ((~c&a ) | (c&b));
	return rst;
}
ulong2 bitselect(ulong2 a, ulong2 b, ulong2 c){
	ulong2 rst;

	int i=0;
	for (i=0;i<2;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ulong3 bitselect(ulong3 a, ulong3 b, ulong3 c){
	ulong3 rst;
	
	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ulong4 bitselect(ulong4 a, ulong4 b, ulong4 c){
	ulong4 rst;
	
	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ulong8 bitselect(ulong8 a, ulong8 b, ulong8 c){
	ulong8 rst;

	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}
ulong16 bitselect(ulong16 a, ulong16 b, ulong16 c){
	ulong16 rst;

	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= ((~c[i]&a[i] ) | (c[i]&b[i]));
	}
	return rst;
}


float check_bitselect(float a, float b , float c)
{
	float rst;

	unsigned int ax;
	unsigned int bx;
	unsigned int cx;
	unsigned int rstx;

	GET_FLOAT_WORD (&ax, a);
	GET_FLOAT_WORD (&bx, b);
	GET_FLOAT_WORD (&cx, c);

	rstx= ((~cx&ax ) | (cx&bx));

	SET_FLOAT_WORD(&rst,rstx);

	return rst;


}
//float
float bitselect(float x, float y, float z){


	float rst;
    
  	rst=check_bitselect(x,y,z);	

	return rst;

}

float2 bitselect(float2 a, float2 b, float2 c){
	float2 rst;

	int i=0;
	for (i=0;i<2;i++)
	{
		rst[i]= check_bitselect(a[i],b[i],c[i]);
	}
	return rst;
}

float3 bitselect(float3 a, float3 b, float3 c){
	float3 rst;

	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= check_bitselect(a[i],b[i],c[i]);	
	}
	return rst;
}

float4 bitselect(float4 a, float4 b, float4 c){
	float4 rst;

	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= check_bitselect(a[i],b[i],c[i]);	
	}
	return rst;
}

float8 bitselect(float8 a, float8 b, float8 c){
	float8 rst;
	
	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= check_bitselect(a[i],b[i],c[i]);	
	}
	return rst;
}

float16 bitselect(float16 a, float16 b, float16 c){
	float16 rst;
	
	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= check_bitselect(a[i],b[i],c[i]);	
	}
	return rst;
}


//double
double check_bitselect_double(double a, double b , double c)
{
	double rst;

	ullong ax;
	ullong bx;
	ullong cx;
	ullong rstx;

	GET_DOUBLE_WORD (&ax, a);
	GET_DOUBLE_WORD (&bx, b);
	GET_DOUBLE_WORD (&cx, c);

	rstx= ((~cx&ax ) | (cx&bx));

	SET_DOUBLE_WORD(&rst,rstx);

	return rst;


}
double bitselect(double x, double y, double z){


	double rst;
    
  	rst=check_bitselect_double(x,y,z);	

	return rst;

}

double2 bitselect(double2 a, double2 b, double2 c){
	double2 rst;

	int i=0;
	for (i=0;i<2;i++)
	{
		rst[i]= check_bitselect_double(a[i],b[i],c[i]);
	}
	return rst;
}

double3 bitselect(double3 a, double3 b, double3 c){
	double3 rst;

	int i=0;
	for (i=0;i<3;i++)
	{
		rst[i]= check_bitselect_double(a[i],b[i],c[i]);	
	}
	return rst;
}


double4 bitselect(double4 a, double4 b, double4 c){
	double4 rst;

	int i=0;
	for (i=0;i<4;i++)
	{
		rst[i]= check_bitselect_double(a[i],b[i],c[i]);	
	}
	return rst;
}

double8 bitselect(double8 a, double8 b, double8 c){
	double8 rst;
	
	int i=0;
	for (i=0;i<8;i++)
	{
		rst[i]= check_bitselect_double(a[i],b[i],c[i]);	
	}
	return rst;
}

double16 bitselect(double16 a, double16 b, double16 c){
	double16 rst;
	
	int i=0;
	for (i=0;i<16;i++)
	{
		rst[i]= check_bitselect_double(a[i],b[i],c[i]);	
	}
	return rst;
}
