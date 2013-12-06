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
#include <math.h>
#include <math/sqrt.h>
#include <math/pow.h>
#include "../math/math_util.h"
#include <stdio.h>
 
double snu_hypot(double x, double y);

// float, float
float length(float p){
  return sqrt( (double)pow((double)p, 2.0));
}

float length(float2 p) {
  return sqrt( (double)(pow((double)p[0], 2.0) + pow((double)p[1], 2.0)) );
}

float length(float3 p) {

  return sqrt( (double)(pow((double)p[0], 2.0) + pow((double)p[1], 2.0) + pow((double)p[2], 2.0)) );
}

float length(float4 p) {
  return sqrt( (double)(pow((double)p[0], 2.0) + pow((double)p[1], 2.0) + pow((double)p[2], 2.0) + pow((double)p[3], 2.0)) );
}


// double, double
double length(double p){
  
  if( isinf(p) )
    return INFINITY;
  if( isnan(p) )
    return NAN;
  if( p == 0.0 )
    return 0.0;

  return fabs(p);
}

double length(double2 p) {
  return snu_hypot(p[0], p[1]);
}

double length(double3 p) {
  if ( isinf(p[0]) || isinf(p[1]) || isinf(p[2]) ) 
    return INFINITY; 
  if ( isnan(p[0]) || isnan(p[1]) || isnan(p[2]) )
    return NAN;          
  if ( p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0 )
    return 0.0;           

  p[0] = fabs(p[0]);             
  p[1] = fabs(p[1]);
  p[2] = fabs(p[2]);
  double t = fmin(p[0], p[1]);
  p[0] = fmax(p[0], p[1]);
  p[1] = t;

  t = fmin(p[0], p[2]);
  p[0] = fmax(p[0], p[2]);
  p[2] = t;                       
  return p[0] * sqrt(1 + pow(p[1]/p[0], 2.0) + pow(p[2]/p[0], 2.0));
}

double length(double4 p) {
  if ( isinf(p[0]) || isinf(p[1]) || isinf(p[2]) || isinf(p[3])  ) 
    return INFINITY; 
  if ( isnan(p[0]) || isnan(p[1]) || isnan(p[2]) || isnan(p[3]) )
    return NAN;          
  if ( p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0 && p[3] == 0.0 )
    return 0.0;           

  p[0] = fabs(p[0]);             
  p[1] = fabs(p[1]);
  p[2] = fabs(p[2]);
  p[3] = fabs(p[3]);
  double t = fmin(p[0], p[1]);
  p[0] = fmax(p[0], p[1]);
  p[1] = t;

  t = fmin(p[0], p[2]);
  p[0] = fmax(p[0], p[2]);
  p[2] = t;                       
  
  t = fmin(p[0], p[3]);
  p[0] = fmax(p[0], p[3]);
  p[3] = t;                       
  
  return p[0] * sqrt(1 + pow(p[1]/p[0], 2.0) + pow(p[2]/p[0], 2.0) + pow(p[3]/p[0], 2.0));

}

