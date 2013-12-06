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
#include <geometric/length.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <math/sqrt.h>
#include <math/pow.h>
#include "../math/math_util.h"

float normalize(float p){
  return p / length(p);
}

float2 normalize(float2 p){
  double2 dval;
  for( int i = 0; i < 2; i++ ) {
    dval[i] = (double)p[i];
  }
  for( int i = 0; i < 2; i++ ) {
    p[i] = (float)(p[i]/ length(dval));
  }
  return p;
}

float3 normalize(float3 p){
  double3 dval;
  for( int i = 0; i < 3; i++ ) {
    dval[i] = (double)p[i];
  }
  for( int i = 0; i < 3; i++ ) {
    p[i] = (float)(p[i]/ length(dval));
  }
  return p;
}

float4 normalize(float4 p){
  double4 dval;
  for( int i = 0; i < 4; i++ ) {
    dval[i] = (double)p[i];
  }
  for( int i = 0; i < 4; i++ ) {
    p[i] = (float)(p[i]/ length(dval));
  }
  return p;
}

double normalize(double p){
  return p / length(p);
}

double2 normalize(double2 p){
  long double _p[2];
  long double length;
  int i = 0;

  for( i = 0; i < 2; i++ ) {
    _p[i] = p[i];
    
    if ( isinf(_p[i]) ) { 
      length = INFINITY; 
      break;
    }
    if ( isnan(_p[i]) ) {
      length = NAN;    
      break;
    }
    if ( _p[i] == 0.0 ) {
      length = 0.0;
      break;
    }
    _p[i] = fabs(_p[i]);
  }

  long double t = fmin(_p[0], _p[1]);
  _p[0] = fmax(_p[0], _p[1]);
  _p[1] = t;

  length = _p[0] * sqrt(1.0 + (double)(_p[1]/_p[0]*_p[1]/_p[0]));

  for( int i = 0; i < 2; i++ )
    p[i] = p[i]/length;
  
  return p;

}

double3 normalize(double3 p){
  long double _p[3];
  long double length;
  int i = 0;

  for( i = 0; i < 3; i++ ) {
    _p[i] = p[i];
    
    if ( isinf(_p[i]) ) { 
      length = INFINITY; 
      break;
    }
    if ( isnan(_p[i]) ) {
      length = NAN;    
      break;
    }
    if ( _p[i] == 0.0 ) {
      length = 0.0;
      break;
    }
    _p[i] = fabs(_p[i]);
  }

  long double t = fmin(_p[0], _p[1]);
  _p[0] = fmax(_p[0], _p[1]);
  _p[1] = t;

  t = fmin(_p[0], _p[2]);
  _p[0] = fmax(_p[0], _p[2]);
  _p[2] = t;

  length = _p[0] * sqrt(1.0 + (double)(_p[1]/_p[0]*_p[1]/_p[0]) + (double)(_p[2]/_p[0]*_p[2]/_p[0]));

  for( int i = 0; i < 3; i++ )
    p[i] = p[i]/length;
  
  return p;
}

double4 normalize(double4 p){
  long double _p[4];
  long double length;
  int i = 0;

  for( i = 0; i < 4; i++ ) {
    _p[i] = p[i];
    
    if ( isinf(_p[i]) ) { 
      length = INFINITY; 
      break;
    }
    if ( isnan(_p[i]) ) {
      length = NAN;    
      break;
    }
    if ( _p[i] == 0.0 ) {
      length = 0.0;
      break;
    }
    _p[i] = fabs(_p[i]);
  }

  long double t = fmin(_p[0], _p[1]);
  _p[0] = fmax(_p[0], _p[1]);
  _p[1] = t;

  t = fmin(_p[0], _p[2]);
  _p[0] = fmax(_p[0], _p[2]);
  _p[2] = t;

  t = fmin(_p[0], _p[3]);
  _p[0] = fmax(_p[0], _p[3]);
  _p[3] = t;
  
  length = _p[0] * sqrt(1.0 + (double)(_p[1]/_p[0]*_p[1]/_p[0]) + (double)(_p[2]/_p[0]*_p[2]/_p[0]) + (double)(_p[3]/_p[0]*_p[3]/_p[0]));

  for( int i = 0; i < 4; i++ )
    p[i] = p[i]/length;
  
  return p;
}
