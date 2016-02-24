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

#ifndef __CL_BUILTINS_MATH_H__
#define __CL_BUILTINS_MATH_H__

#include "cl_types.h"

/* 6.12.2 Math Functions */
/* Table 6.8 Scalar and Vector Argument Bulit-in Math Function Table */
float    acos(float) __attribute__((overloadable));
float2   acos(float2) __attribute__((overloadable));
float3   acos(float3) __attribute__((overloadable));
float4   acos(float4) __attribute__((overloadable));
float8   acos(float8) __attribute__((overloadable));
float16  acos(float16) __attribute__((overloadable));
double   acos(double) __attribute__((overloadable));
double2  acos(double2) __attribute__((overloadable));
double3  acos(double3) __attribute__((overloadable));
double4  acos(double4) __attribute__((overloadable));
double8  acos(double8) __attribute__((overloadable));
double16 acos(double16) __attribute__((overloadable));

float    acosh(float) __attribute__((overloadable));
float2   acosh(float2) __attribute__((overloadable));
float3   acosh(float3) __attribute__((overloadable));
float4   acosh(float4) __attribute__((overloadable));
float8   acosh(float8) __attribute__((overloadable));
float16  acosh(float16) __attribute__((overloadable));
double   acosh(double) __attribute__((overloadable));
double2  acosh(double2) __attribute__((overloadable));
double3  acosh(double3) __attribute__((overloadable));
double4  acosh(double4) __attribute__((overloadable));
double8  acosh(double8) __attribute__((overloadable));
double16 acosh(double16) __attribute__((overloadable));

float    acospi(float) __attribute__((overloadable));
float2   acospi(float2) __attribute__((overloadable));
float3   acospi(float3) __attribute__((overloadable));
float4   acospi(float4) __attribute__((overloadable));
float8   acospi(float8) __attribute__((overloadable));
float16  acospi(float16) __attribute__((overloadable));
double   acospi(double) __attribute__((overloadable));
double2  acospi(double2) __attribute__((overloadable));
double3  acospi(double3) __attribute__((overloadable));
double4  acospi(double4) __attribute__((overloadable));
double8  acospi(double8) __attribute__((overloadable));
double16 acospi(double16) __attribute__((overloadable));

float    asin(float) __attribute__((overloadable));
float2   asin(float2) __attribute__((overloadable));
float3   asin(float3) __attribute__((overloadable));
float4   asin(float4) __attribute__((overloadable));
float8   asin(float8) __attribute__((overloadable));
float16  asin(float16) __attribute__((overloadable));
double   asin(double) __attribute__((overloadable));
double2  asin(double2) __attribute__((overloadable));
double3  asin(double3) __attribute__((overloadable));
double4  asin(double4) __attribute__((overloadable));
double8  asin(double8) __attribute__((overloadable));
double16 asin(double16) __attribute__((overloadable));

float    asinh(float) __attribute__((overloadable));
float2   asinh(float2) __attribute__((overloadable));
float3   asinh(float3) __attribute__((overloadable));
float4   asinh(float4) __attribute__((overloadable));
float8   asinh(float8) __attribute__((overloadable));
float16  asinh(float16) __attribute__((overloadable));
double   asinh(double) __attribute__((overloadable));
double2  asinh(double2) __attribute__((overloadable));
double3  asinh(double3) __attribute__((overloadable));
double4  asinh(double4) __attribute__((overloadable));
double8  asinh(double8) __attribute__((overloadable));
double16 asinh(double16) __attribute__((overloadable));

float    asinpi(float) __attribute__((overloadable));
float2   asinpi(float2) __attribute__((overloadable));
float3   asinpi(float3) __attribute__((overloadable));
float4   asinpi(float4) __attribute__((overloadable));
float8   asinpi(float8) __attribute__((overloadable));
float16  asinpi(float16) __attribute__((overloadable));
double   asinpi(double) __attribute__((overloadable));
double2  asinpi(double2) __attribute__((overloadable));
double3  asinpi(double3) __attribute__((overloadable));
double4  asinpi(double4) __attribute__((overloadable));
double8  asinpi(double8) __attribute__((overloadable));
double16 asinpi(double16) __attribute__((overloadable));

float    atan(float) __attribute__((overloadable));
float2   atan(float2) __attribute__((overloadable));
float3   atan(float3) __attribute__((overloadable));
float4   atan(float4) __attribute__((overloadable));
float8   atan(float8) __attribute__((overloadable));
float16  atan(float16) __attribute__((overloadable));
double   atan(double) __attribute__((overloadable));
double2  atan(double2) __attribute__((overloadable));
double3  atan(double3) __attribute__((overloadable));
double4  atan(double4) __attribute__((overloadable));
double8  atan(double8) __attribute__((overloadable));
double16 atan(double16) __attribute__((overloadable));

float    atan2(float, float) __attribute__((overloadable));
float2   atan2(float2, float2) __attribute__((overloadable));
float3   atan2(float3, float3) __attribute__((overloadable));
float4   atan2(float4, float4) __attribute__((overloadable));
float8   atan2(float8, float8) __attribute__((overloadable));
float16  atan2(float16, float16) __attribute__((overloadable));
double   atan2(double, double) __attribute__((overloadable));
double2  atan2(double2, double2) __attribute__((overloadable));
double3  atan2(double3, double3) __attribute__((overloadable));
double4  atan2(double4, double4) __attribute__((overloadable));
double8  atan2(double8, double8) __attribute__((overloadable));
double16 atan2(double16, double16) __attribute__((overloadable));

float    atanh(float) __attribute__((overloadable));
float2   atanh(float2) __attribute__((overloadable));
float3   atanh(float3) __attribute__((overloadable));
float4   atanh(float4) __attribute__((overloadable));
float8   atanh(float8) __attribute__((overloadable));
float16  atanh(float16) __attribute__((overloadable));
double   atanh(double) __attribute__((overloadable));
double2  atanh(double2) __attribute__((overloadable));
double3  atanh(double3) __attribute__((overloadable));
double4  atanh(double4) __attribute__((overloadable));
double8  atanh(double8) __attribute__((overloadable));
double16 atanh(double16) __attribute__((overloadable));

float    atanpi(float) __attribute__((overloadable));
float2   atanpi(float2) __attribute__((overloadable));
float3   atanpi(float3) __attribute__((overloadable));
float4   atanpi(float4) __attribute__((overloadable));
float8   atanpi(float8) __attribute__((overloadable));
float16  atanpi(float16) __attribute__((overloadable));
double   atanpi(double) __attribute__((overloadable));
double2  atanpi(double2) __attribute__((overloadable));
double3  atanpi(double3) __attribute__((overloadable));
double4  atanpi(double4) __attribute__((overloadable));
double8  atanpi(double8) __attribute__((overloadable));
double16 atanpi(double16) __attribute__((overloadable));

float    atan2pi(float, float) __attribute__((overloadable));
float2   atan2pi(float2, float2) __attribute__((overloadable));
float3   atan2pi(float3, float3) __attribute__((overloadable));
float4   atan2pi(float4, float4) __attribute__((overloadable));
float8   atan2pi(float8, float8) __attribute__((overloadable));
float16  atan2pi(float16, float16) __attribute__((overloadable));
double   atan2pi(double, double) __attribute__((overloadable));
double2  atan2pi(double2, double2) __attribute__((overloadable));
double3  atan2pi(double3, double3) __attribute__((overloadable));
double4  atan2pi(double4, double4) __attribute__((overloadable));
double8  atan2pi(double8, double8) __attribute__((overloadable));
double16 atan2pi(double16, double16) __attribute__((overloadable));

float    cbrt(float) __attribute__((overloadable));
float2   cbrt(float2) __attribute__((overloadable));
float3   cbrt(float3) __attribute__((overloadable));
float4   cbrt(float4) __attribute__((overloadable));
float8   cbrt(float8) __attribute__((overloadable));
float16  cbrt(float16) __attribute__((overloadable));
double   cbrt(double) __attribute__((overloadable));
double2  cbrt(double2) __attribute__((overloadable));
double3  cbrt(double3) __attribute__((overloadable));
double4  cbrt(double4) __attribute__((overloadable));
double8  cbrt(double8) __attribute__((overloadable));
double16 cbrt(double16) __attribute__((overloadable));

float    ceil(float) __attribute__((overloadable));
float2   ceil(float2) __attribute__((overloadable));
float3   ceil(float3) __attribute__((overloadable));
float4   ceil(float4) __attribute__((overloadable));
float8   ceil(float8) __attribute__((overloadable));
float16  ceil(float16) __attribute__((overloadable));
double   ceil(double) __attribute__((overloadable));
double2  ceil(double2) __attribute__((overloadable));
double3  ceil(double3) __attribute__((overloadable));
double4  ceil(double4) __attribute__((overloadable));
double8  ceil(double8) __attribute__((overloadable));
double16 ceil(double16) __attribute__((overloadable));

float    copysign(float, float) __attribute__((overloadable));
float2   copysign(float2, float2) __attribute__((overloadable));
float3   copysign(float3, float3) __attribute__((overloadable));
float4   copysign(float4, float4) __attribute__((overloadable));
float8   copysign(float8, float8) __attribute__((overloadable));
float16  copysign(float16, float16) __attribute__((overloadable));
double   copysign(double, double) __attribute__((overloadable));
double2  copysign(double2, double2) __attribute__((overloadable));
double3  copysign(double3, double3) __attribute__((overloadable));
double4  copysign(double4, double4) __attribute__((overloadable));
double8  copysign(double8, double8) __attribute__((overloadable));
double16 copysign(double16, double16) __attribute__((overloadable));

float    cos(float) __attribute__((overloadable));
float2   cos(float2) __attribute__((overloadable));
float3   cos(float3) __attribute__((overloadable));
float4   cos(float4) __attribute__((overloadable));
float8   cos(float8) __attribute__((overloadable));
float16  cos(float16) __attribute__((overloadable));
double   cos(double) __attribute__((overloadable));
double2  cos(double2) __attribute__((overloadable));
double3  cos(double3) __attribute__((overloadable));
double4  cos(double4) __attribute__((overloadable));
double8  cos(double8) __attribute__((overloadable));
double16 cos(double16) __attribute__((overloadable));

float    cosh(float) __attribute__((overloadable));
float2   cosh(float2) __attribute__((overloadable));
float3   cosh(float3) __attribute__((overloadable));
float4   cosh(float4) __attribute__((overloadable));
float8   cosh(float8) __attribute__((overloadable));
float16  cosh(float16) __attribute__((overloadable));
double   cosh(double) __attribute__((overloadable));
double2  cosh(double2) __attribute__((overloadable));
double3  cosh(double3) __attribute__((overloadable));
double4  cosh(double4) __attribute__((overloadable));
double8  cosh(double8) __attribute__((overloadable));
double16 cosh(double16) __attribute__((overloadable));

float    cospi(float) __attribute__((overloadable));
float2   cospi(float2) __attribute__((overloadable));
float3   cospi(float3) __attribute__((overloadable));
float4   cospi(float4) __attribute__((overloadable));
float8   cospi(float8) __attribute__((overloadable));
float16  cospi(float16) __attribute__((overloadable));
double   cospi(double) __attribute__((overloadable));
double2  cospi(double2) __attribute__((overloadable));
double3  cospi(double3) __attribute__((overloadable));
double4  cospi(double4) __attribute__((overloadable));
double8  cospi(double8) __attribute__((overloadable));
double16 cospi(double16) __attribute__((overloadable));

float    erfc(float) __attribute__((overloadable));
float2   erfc(float2) __attribute__((overloadable));
float3   erfc(float3) __attribute__((overloadable));
float4   erfc(float4) __attribute__((overloadable));
float8   erfc(float8) __attribute__((overloadable));
float16  erfc(float16) __attribute__((overloadable));
double   erfc(double) __attribute__((overloadable));
double2  erfc(double2) __attribute__((overloadable));
double3  erfc(double3) __attribute__((overloadable));
double4  erfc(double4) __attribute__((overloadable));
double8  erfc(double8) __attribute__((overloadable));
double16 erfc(double16) __attribute__((overloadable));

float    erf(float) __attribute__((overloadable));
float2   erf(float2) __attribute__((overloadable));
float3   erf(float3) __attribute__((overloadable));
float4   erf(float4) __attribute__((overloadable));
float8   erf(float8) __attribute__((overloadable));
float16  erf(float16) __attribute__((overloadable));
double   erf(double) __attribute__((overloadable));
double2  erf(double2) __attribute__((overloadable));
double3  erf(double3) __attribute__((overloadable));
double4  erf(double4) __attribute__((overloadable));
double8  erf(double8) __attribute__((overloadable));
double16 erf(double16) __attribute__((overloadable));

float    exp(float) __attribute__((overloadable));
float2   exp(float2) __attribute__((overloadable));
float3   exp(float3) __attribute__((overloadable));
float4   exp(float4) __attribute__((overloadable));
float8   exp(float8) __attribute__((overloadable));
float16  exp(float16) __attribute__((overloadable));
double   exp(double) __attribute__((overloadable));
double2  exp(double2) __attribute__((overloadable));
double3  exp(double3) __attribute__((overloadable));
double4  exp(double4) __attribute__((overloadable));
double8  exp(double8) __attribute__((overloadable));
double16 exp(double16) __attribute__((overloadable));

float    exp2(float) __attribute__((overloadable));
float2   exp2(float2) __attribute__((overloadable));
float3   exp2(float3) __attribute__((overloadable));
float4   exp2(float4) __attribute__((overloadable));
float8   exp2(float8) __attribute__((overloadable));
float16  exp2(float16) __attribute__((overloadable));
double   exp2(double) __attribute__((overloadable));
double2  exp2(double2) __attribute__((overloadable));
double3  exp2(double3) __attribute__((overloadable));
double4  exp2(double4) __attribute__((overloadable));
double8  exp2(double8) __attribute__((overloadable));
double16 exp2(double16) __attribute__((overloadable));

float    exp10(float) __attribute__((overloadable));
float2   exp10(float2) __attribute__((overloadable));
float3   exp10(float3) __attribute__((overloadable));
float4   exp10(float4) __attribute__((overloadable));
float8   exp10(float8) __attribute__((overloadable));
float16  exp10(float16) __attribute__((overloadable));
double   exp10(double) __attribute__((overloadable));
double2  exp10(double2) __attribute__((overloadable));
double3  exp10(double3) __attribute__((overloadable));
double4  exp10(double4) __attribute__((overloadable));
double8  exp10(double8) __attribute__((overloadable));
double16 exp10(double16) __attribute__((overloadable));

float    expm1(float) __attribute__((overloadable));
float2   expm1(float2) __attribute__((overloadable));
float3   expm1(float3) __attribute__((overloadable));
float4   expm1(float4) __attribute__((overloadable));
float8   expm1(float8) __attribute__((overloadable));
float16  expm1(float16) __attribute__((overloadable));
double   expm1(double) __attribute__((overloadable));
double2  expm1(double2) __attribute__((overloadable));
double3  expm1(double3) __attribute__((overloadable));
double4  expm1(double4) __attribute__((overloadable));
double8  expm1(double8) __attribute__((overloadable));
double16 expm1(double16) __attribute__((overloadable));

float    fabs(float) __attribute__((overloadable));
float2   fabs(float2) __attribute__((overloadable));
float3   fabs(float3) __attribute__((overloadable));
float4   fabs(float4) __attribute__((overloadable));
float8   fabs(float8) __attribute__((overloadable));
float16  fabs(float16) __attribute__((overloadable));
double   fabs(double) __attribute__((overloadable));
double2  fabs(double2) __attribute__((overloadable));
double3  fabs(double3) __attribute__((overloadable));
double4  fabs(double4) __attribute__((overloadable));
double8  fabs(double8) __attribute__((overloadable));
double16 fabs(double16) __attribute__((overloadable));

float    fdim(float, float) __attribute__((overloadable));
float2   fdim(float2, float2) __attribute__((overloadable));
float3   fdim(float3, float3) __attribute__((overloadable));
float4   fdim(float4, float4) __attribute__((overloadable));
float8   fdim(float8, float8) __attribute__((overloadable));
float16  fdim(float16, float16) __attribute__((overloadable));
double   fdim(double, double) __attribute__((overloadable));
double2  fdim(double2, double2) __attribute__((overloadable));
double3  fdim(double3, double3) __attribute__((overloadable));
double4  fdim(double4, double4) __attribute__((overloadable));
double8  fdim(double8, double8) __attribute__((overloadable));
double16 fdim(double16, double16) __attribute__((overloadable));

float    floor(float) __attribute__((overloadable));
float2   floor(float2) __attribute__((overloadable));
float3   floor(float3) __attribute__((overloadable));
float4   floor(float4) __attribute__((overloadable));
float8   floor(float8) __attribute__((overloadable));
float16  floor(float16) __attribute__((overloadable));
double   floor(double) __attribute__((overloadable));
double2  floor(double2) __attribute__((overloadable));
double3  floor(double3) __attribute__((overloadable));
double4  floor(double4) __attribute__((overloadable));
double8  floor(double8) __attribute__((overloadable));
double16 floor(double16) __attribute__((overloadable));

float    fma(float, float, float) __attribute__((overloadable));
float2   fma(float2, float2, float2) __attribute__((overloadable));
float3   fma(float3, float3, float3) __attribute__((overloadable));
float4   fma(float4, float4, float4) __attribute__((overloadable));
float8   fma(float8, float8, float8) __attribute__((overloadable));
float16  fma(float16, float16, float16) __attribute__((overloadable));
double   fma(double, double, double) __attribute__((overloadable));
double2  fma(double2, double2, double2) __attribute__((overloadable));
double3  fma(double3, double3, double3) __attribute__((overloadable));
double4  fma(double4, double4, double4) __attribute__((overloadable));
double8  fma(double8, double8, double8) __attribute__((overloadable));
double16 fma(double16, double16, double16) __attribute__((overloadable));

float    fmax(float, float) __attribute__((overloadable));
float2   fmax(float2, float2) __attribute__((overloadable));
float3   fmax(float3, float3) __attribute__((overloadable));
float4   fmax(float4, float4) __attribute__((overloadable));
float8   fmax(float8, float8) __attribute__((overloadable));
float16  fmax(float16, float16) __attribute__((overloadable));
double   fmax(double, double) __attribute__((overloadable));
double2  fmax(double2, double2) __attribute__((overloadable));
double3  fmax(double3, double3) __attribute__((overloadable));
double4  fmax(double4, double4) __attribute__((overloadable));
double8  fmax(double8, double8) __attribute__((overloadable));
double16 fmax(double16, double16) __attribute__((overloadable));

float2   fmax(float2, float) __attribute__((overloadable));
float3   fmax(float3, float) __attribute__((overloadable));
float4   fmax(float4, float) __attribute__((overloadable));
float8   fmax(float8, float) __attribute__((overloadable));
float16  fmax(float16, float) __attribute__((overloadable));

double2  fmax(double2, double) __attribute__((overloadable));
double3  fmax(double3, double) __attribute__((overloadable));
double4  fmax(double4, double) __attribute__((overloadable));
double8  fmax(double8, double) __attribute__((overloadable));
double16 fmax(double16, double) __attribute__((overloadable));

float    fmin(float, float) __attribute__((overloadable));
float2   fmin(float2, float2) __attribute__((overloadable));
float3   fmin(float3, float3) __attribute__((overloadable));
float4   fmin(float4, float4) __attribute__((overloadable));
float8   fmin(float8, float8) __attribute__((overloadable));
float16  fmin(float16, float16) __attribute__((overloadable));
double   fmin(double, double) __attribute__((overloadable));
double2  fmin(double2, double2) __attribute__((overloadable));
double3  fmin(double3, double3) __attribute__((overloadable));
double4  fmin(double4, double4) __attribute__((overloadable));
double8  fmin(double8, double8) __attribute__((overloadable));
double16 fmin(double16, double16) __attribute__((overloadable));

float2   fmin(float2, float) __attribute__((overloadable));
float3   fmin(float3, float) __attribute__((overloadable));
float4   fmin(float4, float) __attribute__((overloadable));
float8   fmin(float8, float) __attribute__((overloadable));
float16  fmin(float16, float) __attribute__((overloadable));

double2  fmin(double2, double) __attribute__((overloadable));
double3  fmin(double3, double) __attribute__((overloadable));
double4  fmin(double4, double) __attribute__((overloadable));
double8  fmin(double8, double) __attribute__((overloadable));
double16 fmin(double16, double) __attribute__((overloadable));

float    fmod(float, float) __attribute__((overloadable));
float2   fmod(float2, float2) __attribute__((overloadable));
float3   fmod(float3, float3) __attribute__((overloadable));
float4   fmod(float4, float4) __attribute__((overloadable));
float8   fmod(float8, float8) __attribute__((overloadable));
float16  fmod(float16, float16) __attribute__((overloadable));
double   fmod(double, double) __attribute__((overloadable));
double2  fmod(double2, double2) __attribute__((overloadable));
double3  fmod(double3, double3) __attribute__((overloadable));
double4  fmod(double4, double4) __attribute__((overloadable));
double8  fmod(double8, double8) __attribute__((overloadable));
double16 fmod(double16, double16) __attribute__((overloadable));

float    fract(float, float *) __attribute__((overloadable));
float2   fract(float2, float2 *) __attribute__((overloadable));
float3   fract(float3, float3 *) __attribute__((overloadable));
float4   fract(float4, float4 *) __attribute__((overloadable));
float8   fract(float8, float8 *) __attribute__((overloadable));
float16  fract(float16, float16 *) __attribute__((overloadable));
double   fract(double, double *) __attribute__((overloadable));
double2  fract(double2, double2 *) __attribute__((overloadable));
double3  fract(double3, double3 *) __attribute__((overloadable));
double4  fract(double4, double4 *) __attribute__((overloadable));
double8  fract(double8, double8 *) __attribute__((overloadable));
double16 fract(double16, double16 *) __attribute__((overloadable));

float    frexp(float, int *) __attribute__((overloadable));
float2   frexp(float2, int2 *) __attribute__((overloadable));
float3   frexp(float3, int3 *) __attribute__((overloadable));
float4   frexp(float4, int4 *) __attribute__((overloadable));
float8   frexp(float8, int8 *) __attribute__((overloadable));
float16  frexp(float16, int16 *) __attribute__((overloadable));
double   frexp(double, int *) __attribute__((overloadable));
double2  frexp(double2, int2 *) __attribute__((overloadable));
double3  frexp(double3, int3 *) __attribute__((overloadable));
double4  frexp(double4, int4 *) __attribute__((overloadable));
double8  frexp(double8, int8 *) __attribute__((overloadable));
double16 frexp(double16, int16 *) __attribute__((overloadable));

float    hypot(float, float) __attribute__((overloadable));
float2   hypot(float2, float2) __attribute__((overloadable));
float3   hypot(float3, float3) __attribute__((overloadable));
float4   hypot(float4, float4) __attribute__((overloadable));
float8   hypot(float8, float8) __attribute__((overloadable));
float16  hypot(float16, float16) __attribute__((overloadable));
double   hypot(double, double) __attribute__((overloadable));
double2  hypot(double2, double2) __attribute__((overloadable));
double3  hypot(double3, double3) __attribute__((overloadable));
double4  hypot(double4, double4) __attribute__((overloadable));
double8  hypot(double8, double8) __attribute__((overloadable));
double16 hypot(double16, double16) __attribute__((overloadable));

int      ilogb(float) __attribute__((overloadable));
int2     ilogb(float2) __attribute__((overloadable));
int3     ilogb(float3) __attribute__((overloadable));
int4     ilogb(float4) __attribute__((overloadable));
int8     ilogb(float8) __attribute__((overloadable));
int16    ilogb(float16) __attribute__((overloadable));
int      ilogb(double) __attribute__((overloadable));
int2     ilogb(double2) __attribute__((overloadable));
int3     ilogb(double3) __attribute__((overloadable));
int4     ilogb(double4) __attribute__((overloadable));
int8     ilogb(double8) __attribute__((overloadable));
int16    ilogb(double16) __attribute__((overloadable));

float    ldexp(float, int) __attribute__((overloadable));
float2   ldexp(float2, int2) __attribute__((overloadable));
float3   ldexp(float3, int3) __attribute__((overloadable));
float4   ldexp(float4, int4) __attribute__((overloadable));
float8   ldexp(float8, int8) __attribute__((overloadable));
float16  ldexp(float16, int16) __attribute__((overloadable));
double   ldexp(double, int) __attribute__((overloadable));
double2  ldexp(double2, int2) __attribute__((overloadable));
double3  ldexp(double3, int3) __attribute__((overloadable));
double4  ldexp(double4, int4) __attribute__((overloadable));
double8  ldexp(double8, int8) __attribute__((overloadable));
double16 ldexp(double16, int16) __attribute__((overloadable));

float    lgamma(float) __attribute__((overloadable));
float2   lgamma(float2) __attribute__((overloadable));
float3   lgamma(float3) __attribute__((overloadable));
float4   lgamma(float4) __attribute__((overloadable));
float8   lgamma(float8) __attribute__((overloadable));
float16  lgamma(float16) __attribute__((overloadable));
double   lgamma(double) __attribute__((overloadable));
double2  lgamma(double2) __attribute__((overloadable));
double3  lgamma(double3) __attribute__((overloadable));
double4  lgamma(double4) __attribute__((overloadable));
double8  lgamma(double8) __attribute__((overloadable));
double16 lgamma(double16) __attribute__((overloadable));

float    lgamma_r(float, int *) __attribute__((overloadable));
float2   lgamma_r(float2, int2 *) __attribute__((overloadable));
float3   lgamma_r(float3, int3 *) __attribute__((overloadable));
float4   lgamma_r(float4, int4 *) __attribute__((overloadable));
float8   lgamma_r(float8, int8 *) __attribute__((overloadable));
float16  lgamma_r(float16, int16 *) __attribute__((overloadable));
double   lgamma_r(double, int *) __attribute__((overloadable));
double2  lgamma_r(double2, int2 *) __attribute__((overloadable));
double3  lgamma_r(double3, int3 *) __attribute__((overloadable));
double4  lgamma_r(double4, int4 *) __attribute__((overloadable));
double8  lgamma_r(double8, int8 *) __attribute__((overloadable));
double16 lgamma_r(double16, int16 *) __attribute__((overloadable));

float    log(float) __attribute__((overloadable));
float2   log(float2) __attribute__((overloadable));
float3   log(float3) __attribute__((overloadable));
float4   log(float4) __attribute__((overloadable));
float8   log(float8) __attribute__((overloadable));
float16  log(float16) __attribute__((overloadable));
double   log(double) __attribute__((overloadable));
double2  log(double2) __attribute__((overloadable));
double3  log(double3) __attribute__((overloadable));
double4  log(double4) __attribute__((overloadable));
double8  log(double8) __attribute__((overloadable));
double16 log(double16) __attribute__((overloadable));

float    log2(float) __attribute__((overloadable));
float2   log2(float2) __attribute__((overloadable));
float3   log2(float3) __attribute__((overloadable));
float4   log2(float4) __attribute__((overloadable));
float8   log2(float8) __attribute__((overloadable));
float16  log2(float16) __attribute__((overloadable));
double   log2(double) __attribute__((overloadable));
double2  log2(double2) __attribute__((overloadable));
double3  log2(double3) __attribute__((overloadable));
double4  log2(double4) __attribute__((overloadable));
double8  log2(double8) __attribute__((overloadable));
double16 log2(double16) __attribute__((overloadable));

float    log10(float) __attribute__((overloadable));
float2   log10(float2) __attribute__((overloadable));
float3   log10(float3) __attribute__((overloadable));
float4   log10(float4) __attribute__((overloadable));
float8   log10(float8) __attribute__((overloadable));
float16  log10(float16) __attribute__((overloadable));
double   log10(double) __attribute__((overloadable));
double2  log10(double2) __attribute__((overloadable));
double3  log10(double3) __attribute__((overloadable));
double4  log10(double4) __attribute__((overloadable));
double8  log10(double8) __attribute__((overloadable));
double16 log10(double16) __attribute__((overloadable));

float    log1p(float) __attribute__((overloadable));
float2   log1p(float2) __attribute__((overloadable));
float3   log1p(float3) __attribute__((overloadable));
float4   log1p(float4) __attribute__((overloadable));
float8   log1p(float8) __attribute__((overloadable));
float16  log1p(float16) __attribute__((overloadable));
double   log1p(double) __attribute__((overloadable));
double2  log1p(double2) __attribute__((overloadable));
double3  log1p(double3) __attribute__((overloadable));
double4  log1p(double4) __attribute__((overloadable));
double8  log1p(double8) __attribute__((overloadable));
double16 log1p(double16) __attribute__((overloadable));

float    logb(float) __attribute__((overloadable));
float2   logb(float2) __attribute__((overloadable));
float3   logb(float3) __attribute__((overloadable));
float4   logb(float4) __attribute__((overloadable));
float8   logb(float8) __attribute__((overloadable));
float16  logb(float16) __attribute__((overloadable));
double   logb(double) __attribute__((overloadable));
double2  logb(double2) __attribute__((overloadable));
double3  logb(double3) __attribute__((overloadable));
double4  logb(double4) __attribute__((overloadable));
double8  logb(double8) __attribute__((overloadable));
double16 logb(double16) __attribute__((overloadable));

float    mad(float, float, float) __attribute__((overloadable));
float2   mad(float2, float2, float2) __attribute__((overloadable));
float3   mad(float3, float3, float3) __attribute__((overloadable));
float4   mad(float4, float4, float4) __attribute__((overloadable));
float8   mad(float8, float8, float8) __attribute__((overloadable));
float16  mad(float16, float16, float16) __attribute__((overloadable));
double   mad(double, double, double) __attribute__((overloadable));
double2  mad(double2, double2, double2) __attribute__((overloadable));
double3  mad(double3, double3, double3) __attribute__((overloadable));
double4  mad(double4, double4, double4) __attribute__((overloadable));
double8  mad(double8, double8, double8) __attribute__((overloadable));
double16 mad(double16, double16, double16) __attribute__((overloadable));

float    maxmag(float, float) __attribute__((overloadable));
float2   maxmag(float2, float2) __attribute__((overloadable));
float3   maxmag(float3, float3) __attribute__((overloadable));
float4   maxmag(float4, float4) __attribute__((overloadable));
float8   maxmag(float8, float8) __attribute__((overloadable));
float16  maxmag(float16, float16) __attribute__((overloadable));
double   maxmag(double, double) __attribute__((overloadable));
double2  maxmag(double2, double2) __attribute__((overloadable));
double3  maxmag(double3, double3) __attribute__((overloadable));
double4  maxmag(double4, double4) __attribute__((overloadable));
double8  maxmag(double8, double8) __attribute__((overloadable));
double16 maxmag(double16, double16) __attribute__((overloadable));

float    minmag(float, float) __attribute__((overloadable));
float2   minmag(float2, float2) __attribute__((overloadable));
float3   minmag(float3, float3) __attribute__((overloadable));
float4   minmag(float4, float4) __attribute__((overloadable));
float8   minmag(float8, float8) __attribute__((overloadable));
float16  minmag(float16, float16) __attribute__((overloadable));
double   minmag(double, double) __attribute__((overloadable));
double2  minmag(double2, double2) __attribute__((overloadable));
double3  minmag(double3, double3) __attribute__((overloadable));
double4  minmag(double4, double4) __attribute__((overloadable));
double8  minmag(double8, double8) __attribute__((overloadable));
double16 minmag(double16, double16) __attribute__((overloadable));

float    modf(float, float *) __attribute__((overloadable));
float2   modf(float2, float2 *) __attribute__((overloadable));
float3   modf(float3, float3 *) __attribute__((overloadable));
float4   modf(float4, float4 *) __attribute__((overloadable));
float8   modf(float8, float8 *) __attribute__((overloadable));
float16  modf(float16, float16 *) __attribute__((overloadable));
double   modf(double, double *) __attribute__((overloadable));
double2  modf(double2, double2 *) __attribute__((overloadable));
double3  modf(double3, double3 *) __attribute__((overloadable));
double4  modf(double4, double4 *) __attribute__((overloadable));
double8  modf(double8, double8 *) __attribute__((overloadable));
double16 modf(double16, double16 *) __attribute__((overloadable));

float    nan(uint) __attribute__((overloadable));
float2   nan(uint2) __attribute__((overloadable));
float3   nan(uint3) __attribute__((overloadable));
float4   nan(uint4) __attribute__((overloadable));
float8   nan(uint8) __attribute__((overloadable));
float16  nan(uint16) __attribute__((overloadable));
double   nan(ulong) __attribute__((overloadable));
double2  nan(ulong2) __attribute__((overloadable));
double3  nan(ulong3) __attribute__((overloadable));
double4  nan(ulong4) __attribute__((overloadable));
double8  nan(ulong8) __attribute__((overloadable));
double16 nan(ulong16) __attribute__((overloadable));

float    nextafter(float, float) __attribute__((overloadable));
float2   nextafter(float2, float2) __attribute__((overloadable));
float3   nextafter(float3, float3) __attribute__((overloadable));
float4   nextafter(float4, float4) __attribute__((overloadable));
float8   nextafter(float8, float8) __attribute__((overloadable));
float16  nextafter(float16, float16) __attribute__((overloadable));
double   nextafter(double, double) __attribute__((overloadable));
double2  nextafter(double2, double2) __attribute__((overloadable));
double3  nextafter(double3, double3) __attribute__((overloadable));
double4  nextafter(double4, double4) __attribute__((overloadable));
double8  nextafter(double8, double8) __attribute__((overloadable));
double16 nextafter(double16, double16) __attribute__((overloadable));

float    pow(float, float) __attribute__((overloadable));
float2   pow(float2, float2) __attribute__((overloadable));
float3   pow(float3, float3) __attribute__((overloadable));
float4   pow(float4, float4) __attribute__((overloadable));
float8   pow(float8, float8) __attribute__((overloadable));
float16  pow(float16, float16) __attribute__((overloadable));
double   pow(double, double) __attribute__((overloadable));
double2  pow(double2, double2) __attribute__((overloadable));
double3  pow(double3, double3) __attribute__((overloadable));
double4  pow(double4, double4) __attribute__((overloadable));
double8  pow(double8, double8) __attribute__((overloadable));
double16 pow(double16, double16) __attribute__((overloadable));

float    pown(float, int) __attribute__((overloadable));
float2   pown(float2, int2) __attribute__((overloadable));
float3   pown(float3, int3) __attribute__((overloadable));
float4   pown(float4, int4) __attribute__((overloadable));
float8   pown(float8, int8) __attribute__((overloadable));
float16  pown(float16, int16) __attribute__((overloadable));
double   pown(double, int) __attribute__((overloadable));
double2  pown(double2, int2) __attribute__((overloadable));
double3  pown(double3, int3) __attribute__((overloadable));
double4  pown(double4, int4) __attribute__((overloadable));
double8  pown(double8, int8) __attribute__((overloadable));
double16 pown(double16, int16) __attribute__((overloadable));

float    powr(float, float) __attribute__((overloadable));
float2   powr(float2, float2) __attribute__((overloadable));
float3   powr(float3, float3) __attribute__((overloadable));
float4   powr(float4, float4) __attribute__((overloadable));
float8   powr(float8, float8) __attribute__((overloadable));
float16  powr(float16, float16) __attribute__((overloadable));
double   powr(double, double) __attribute__((overloadable));
double2  powr(double2, double2) __attribute__((overloadable));
double3  powr(double3, double3) __attribute__((overloadable));
double4  powr(double4, double4) __attribute__((overloadable));
double8  powr(double8, double8) __attribute__((overloadable));
double16 powr(double16, double16) __attribute__((overloadable));

float    remainder(float, float) __attribute__((overloadable));
float2   remainder(float2, float2) __attribute__((overloadable));
float3   remainder(float3, float3) __attribute__((overloadable));
float4   remainder(float4, float4) __attribute__((overloadable));
float8   remainder(float8, float8) __attribute__((overloadable));
float16  remainder(float16, float16) __attribute__((overloadable));
double   remainder(double, double) __attribute__((overloadable));
double2  remainder(double2, double2) __attribute__((overloadable));
double3  remainder(double3, double3) __attribute__((overloadable));
double4  remainder(double4, double4) __attribute__((overloadable));
double8  remainder(double8, double8) __attribute__((overloadable));
double16 remainder(double16, double16) __attribute__((overloadable));

float    remquo(float, float, int *) __attribute__((overloadable));
float2   remquo(float2, float2, int2 *) __attribute__((overloadable));
float3   remquo(float3, float3, int3 *) __attribute__((overloadable));
float4   remquo(float4, float4, int4 *) __attribute__((overloadable));
float8   remquo(float8, float8, int8 *) __attribute__((overloadable));
float16  remquo(float16, float16, int16 *) __attribute__((overloadable));
double   remquo(double, double, int *) __attribute__((overloadable));
double2  remquo(double2, double2, int2 *) __attribute__((overloadable));
double3  remquo(double3, double3, int3 *) __attribute__((overloadable));
double4  remquo(double4, double4, int4 *) __attribute__((overloadable));
double8  remquo(double8, double8, int8 *) __attribute__((overloadable));
double16 remquo(double16, double16, int16 *) __attribute__((overloadable));

float    rint(float) __attribute__((overloadable));
float2   rint(float2) __attribute__((overloadable));
float3   rint(float3) __attribute__((overloadable));
float4   rint(float4) __attribute__((overloadable));
float8   rint(float8) __attribute__((overloadable));
float16  rint(float16) __attribute__((overloadable));
double   rint(double) __attribute__((overloadable));
double2  rint(double2) __attribute__((overloadable));
double3  rint(double3) __attribute__((overloadable));
double4  rint(double4) __attribute__((overloadable));
double8  rint(double8) __attribute__((overloadable));
double16 rint(double16) __attribute__((overloadable));

float    rootn(float, int) __attribute__((overloadable));
float2   rootn(float2, int2) __attribute__((overloadable));
float3   rootn(float3, int3) __attribute__((overloadable));
float4   rootn(float4, int4) __attribute__((overloadable));
float8   rootn(float8, int8) __attribute__((overloadable));
float16  rootn(float16, int16) __attribute__((overloadable));
double   rootn(double, int) __attribute__((overloadable));
double2  rootn(double2, int2) __attribute__((overloadable));
double3  rootn(double3, int3) __attribute__((overloadable));
double4  rootn(double4, int4) __attribute__((overloadable));
double8  rootn(double8, int8) __attribute__((overloadable));
double16 rootn(double16, int16) __attribute__((overloadable));

float    round(float) __attribute__((overloadable));
float2   round(float2) __attribute__((overloadable));
float3   round(float3) __attribute__((overloadable));
float4   round(float4) __attribute__((overloadable));
float8   round(float8) __attribute__((overloadable));
float16  round(float16) __attribute__((overloadable));
double   round(double) __attribute__((overloadable));
double2  round(double2) __attribute__((overloadable));
double3  round(double3) __attribute__((overloadable));
double4  round(double4) __attribute__((overloadable));
double8  round(double8) __attribute__((overloadable));
double16 round(double16) __attribute__((overloadable));

float    rsqrt(float) __attribute__((overloadable));
float2   rsqrt(float2) __attribute__((overloadable));
float3   rsqrt(float3) __attribute__((overloadable));
float4   rsqrt(float4) __attribute__((overloadable));
float8   rsqrt(float8) __attribute__((overloadable));
float16  rsqrt(float16) __attribute__((overloadable));
double   rsqrt(double) __attribute__((overloadable));
double2  rsqrt(double2) __attribute__((overloadable));
double3  rsqrt(double3) __attribute__((overloadable));
double4  rsqrt(double4) __attribute__((overloadable));
double8  rsqrt(double8) __attribute__((overloadable));
double16 rsqrt(double16) __attribute__((overloadable));

float    sin(float) __attribute__((overloadable));
float2   sin(float2) __attribute__((overloadable));
float3   sin(float3) __attribute__((overloadable));
float4   sin(float4) __attribute__((overloadable));
float8   sin(float8) __attribute__((overloadable));
float16  sin(float16) __attribute__((overloadable));
double   sin(double) __attribute__((overloadable));
double2  sin(double2) __attribute__((overloadable));
double3  sin(double3) __attribute__((overloadable));
double4  sin(double4) __attribute__((overloadable));
double8  sin(double8) __attribute__((overloadable));
double16 sin(double16) __attribute__((overloadable));

float    sincos(float, float *) __attribute__((overloadable));
float2   sincos(float2, float2 *) __attribute__((overloadable));
float3   sincos(float3, float3 *) __attribute__((overloadable));
float4   sincos(float4, float4 *) __attribute__((overloadable));
float8   sincos(float8, float8 *) __attribute__((overloadable));
float16  sincos(float16, float16 *) __attribute__((overloadable));
double   sincos(double, double *) __attribute__((overloadable));
double2  sincos(double2, double2 *) __attribute__((overloadable));
double3  sincos(double3, double3 *) __attribute__((overloadable));
double4  sincos(double4, double4 *) __attribute__((overloadable));
double8  sincos(double8, double8 *) __attribute__((overloadable));
double16 sincos(double16, double16 *) __attribute__((overloadable));

float    sinh(float) __attribute__((overloadable));
float2   sinh(float2) __attribute__((overloadable));
float3   sinh(float3) __attribute__((overloadable));
float4   sinh(float4) __attribute__((overloadable));
float8   sinh(float8) __attribute__((overloadable));
float16  sinh(float16) __attribute__((overloadable));
double   sinh(double) __attribute__((overloadable));
double2  sinh(double2) __attribute__((overloadable));
double3  sinh(double3) __attribute__((overloadable));
double4  sinh(double4) __attribute__((overloadable));
double8  sinh(double8) __attribute__((overloadable));
double16 sinh(double16) __attribute__((overloadable));

float    sinpi(float) __attribute__((overloadable));
float2   sinpi(float2) __attribute__((overloadable));
float3   sinpi(float3) __attribute__((overloadable));
float4   sinpi(float4) __attribute__((overloadable));
float8   sinpi(float8) __attribute__((overloadable));
float16  sinpi(float16) __attribute__((overloadable));
double   sinpi(double) __attribute__((overloadable));
double2  sinpi(double2) __attribute__((overloadable));
double3  sinpi(double3) __attribute__((overloadable));
double4  sinpi(double4) __attribute__((overloadable));
double8  sinpi(double8) __attribute__((overloadable));
double16 sinpi(double16) __attribute__((overloadable));

float    sqrt(float) __attribute__((overloadable));
float2   sqrt(float2) __attribute__((overloadable));
float3   sqrt(float3) __attribute__((overloadable));
float4   sqrt(float4) __attribute__((overloadable));
float8   sqrt(float8) __attribute__((overloadable));
float16  sqrt(float16) __attribute__((overloadable));
double   sqrt(double) __attribute__((overloadable));
double2  sqrt(double2) __attribute__((overloadable));
double3  sqrt(double3) __attribute__((overloadable));
double4  sqrt(double4) __attribute__((overloadable));
double8  sqrt(double8) __attribute__((overloadable));
double16 sqrt(double16) __attribute__((overloadable));

float    tan(float) __attribute__((overloadable));
float2   tan(float2) __attribute__((overloadable));
float3   tan(float3) __attribute__((overloadable));
float4   tan(float4) __attribute__((overloadable));
float8   tan(float8) __attribute__((overloadable));
float16  tan(float16) __attribute__((overloadable));
double   tan(double) __attribute__((overloadable));
double2  tan(double2) __attribute__((overloadable));
double3  tan(double3) __attribute__((overloadable));
double4  tan(double4) __attribute__((overloadable));
double8  tan(double8) __attribute__((overloadable));
double16 tan(double16) __attribute__((overloadable));

float    tanh(float) __attribute__((overloadable));
float2   tanh(float2) __attribute__((overloadable));
float3   tanh(float3) __attribute__((overloadable));
float4   tanh(float4) __attribute__((overloadable));
float8   tanh(float8) __attribute__((overloadable));
float16  tanh(float16) __attribute__((overloadable));
double   tanh(double) __attribute__((overloadable));
double2  tanh(double2) __attribute__((overloadable));
double3  tanh(double3) __attribute__((overloadable));
double4  tanh(double4) __attribute__((overloadable));
double8  tanh(double8) __attribute__((overloadable));
double16 tanh(double16) __attribute__((overloadable));

float    tanpi(float) __attribute__((overloadable));
float2   tanpi(float2) __attribute__((overloadable));
float3   tanpi(float3) __attribute__((overloadable));
float4   tanpi(float4) __attribute__((overloadable));
float8   tanpi(float8) __attribute__((overloadable));
float16  tanpi(float16) __attribute__((overloadable));
double   tanpi(double) __attribute__((overloadable));
double2  tanpi(double2) __attribute__((overloadable));
double3  tanpi(double3) __attribute__((overloadable));
double4  tanpi(double4) __attribute__((overloadable));
double8  tanpi(double8) __attribute__((overloadable));
double16 tanpi(double16) __attribute__((overloadable));

float    tgamma(float) __attribute__((overloadable));
float2   tgamma(float2) __attribute__((overloadable));
float3   tgamma(float3) __attribute__((overloadable));
float4   tgamma(float4) __attribute__((overloadable));
float8   tgamma(float8) __attribute__((overloadable));
float16  tgamma(float16) __attribute__((overloadable));
double   tgamma(double) __attribute__((overloadable));
double2  tgamma(double2) __attribute__((overloadable));
double3  tgamma(double3) __attribute__((overloadable));
double4  tgamma(double4) __attribute__((overloadable));
double8  tgamma(double8) __attribute__((overloadable));
double16 tgamma(double16) __attribute__((overloadable));

float    trunc(float) __attribute__((overloadable));
float2   trunc(float2) __attribute__((overloadable));
float3   trunc(float3) __attribute__((overloadable));
float4   trunc(float4) __attribute__((overloadable));
float8   trunc(float8) __attribute__((overloadable));
float16  trunc(float16) __attribute__((overloadable));
double   trunc(double) __attribute__((overloadable));
double2  trunc(double2) __attribute__((overloadable));
double3  trunc(double3) __attribute__((overloadable));
double4  trunc(double4) __attribute__((overloadable));
double8  trunc(double8) __attribute__((overloadable));
double16 trunc(double16) __attribute__((overloadable));


/* Table 6.9 Scalar and Vector Argument Built-in half__ and native__ Math Functions */
float    half_cos(float) __attribute__((overloadable));
float2   half_cos(float2) __attribute__((overloadable));
float3   half_cos(float3) __attribute__((overloadable));
float4   half_cos(float4) __attribute__((overloadable));
float8   half_cos(float8) __attribute__((overloadable));
float16  half_cos(float16) __attribute__((overloadable));

float    half_divide(float, float) __attribute__((overloadable));
float2   half_divide(float2, float2) __attribute__((overloadable));
float3   half_divide(float3, float3) __attribute__((overloadable));
float4   half_divide(float4, float4) __attribute__((overloadable));
float8   half_divide(float8, float8) __attribute__((overloadable));
float16  half_divide(float16, float16) __attribute__((overloadable));

float    half_exp(float) __attribute__((overloadable));
float2   half_exp(float2) __attribute__((overloadable));
float3   half_exp(float3) __attribute__((overloadable));
float4   half_exp(float4) __attribute__((overloadable));
float8   half_exp(float8) __attribute__((overloadable));
float16  half_exp(float16) __attribute__((overloadable));

float    half_exp2(float) __attribute__((overloadable));
float2   half_exp2(float2) __attribute__((overloadable));
float3   half_exp2(float3) __attribute__((overloadable));
float4   half_exp2(float4) __attribute__((overloadable));
float8   half_exp2(float8) __attribute__((overloadable));
float16  half_exp2(float16) __attribute__((overloadable));

float    half_exp10(float) __attribute__((overloadable));
float2   half_exp10(float2) __attribute__((overloadable));
float3   half_exp10(float3) __attribute__((overloadable));
float4   half_exp10(float4) __attribute__((overloadable));
float8   half_exp10(float8) __attribute__((overloadable));
float16  half_exp10(float16) __attribute__((overloadable));

float    half_log(float) __attribute__((overloadable));
float2   half_log(float2) __attribute__((overloadable));
float3   half_log(float3) __attribute__((overloadable));
float4   half_log(float4) __attribute__((overloadable));
float8   half_log(float8) __attribute__((overloadable));
float16  half_log(float16) __attribute__((overloadable));

float    half_log2(float) __attribute__((overloadable));
float2   half_log2(float2) __attribute__((overloadable));
float3   half_log2(float3) __attribute__((overloadable));
float4   half_log2(float4) __attribute__((overloadable));
float8   half_log2(float8) __attribute__((overloadable));
float16  half_log2(float16) __attribute__((overloadable));

float    half_log10(float) __attribute__((overloadable));
float2   half_log10(float2) __attribute__((overloadable));
float3   half_log10(float3) __attribute__((overloadable));
float4   half_log10(float4) __attribute__((overloadable));
float8   half_log10(float8) __attribute__((overloadable));
float16  half_log10(float16) __attribute__((overloadable));

float    half_powr(float, float) __attribute__((overloadable));
float2   half_powr(float2, float2) __attribute__((overloadable));
float3   half_powr(float3, float3) __attribute__((overloadable));
float4   half_powr(float4, float4) __attribute__((overloadable));
float8   half_powr(float8, float8) __attribute__((overloadable));
float16  half_powr(float16, float16) __attribute__((overloadable));

float    half_recip(float) __attribute__((overloadable));
float2   half_recip(float2) __attribute__((overloadable));
float3   half_recip(float3) __attribute__((overloadable));
float4   half_recip(float4) __attribute__((overloadable));
float8   half_recip(float8) __attribute__((overloadable));
float16  half_recip(float16) __attribute__((overloadable));

float    half_rsqrt(float) __attribute__((overloadable));
float2   half_rsqrt(float2) __attribute__((overloadable));
float3   half_rsqrt(float3) __attribute__((overloadable));
float4   half_rsqrt(float4) __attribute__((overloadable));
float8   half_rsqrt(float8) __attribute__((overloadable));
float16  half_rsqrt(float16) __attribute__((overloadable));

float    half_sin(float) __attribute__((overloadable));
float2   half_sin(float2) __attribute__((overloadable));
float3   half_sin(float3) __attribute__((overloadable));
float4   half_sin(float4) __attribute__((overloadable));
float8   half_sin(float8) __attribute__((overloadable));
float16  half_sin(float16) __attribute__((overloadable));

float    half_sqrt(float) __attribute__((overloadable));
float2   half_sqrt(float2) __attribute__((overloadable));
float3   half_sqrt(float3) __attribute__((overloadable));
float4   half_sqrt(float4) __attribute__((overloadable));
float8   half_sqrt(float8) __attribute__((overloadable));
float16  half_sqrt(float16) __attribute__((overloadable));

float    half_tan(float) __attribute__((overloadable));
float2   half_tan(float2) __attribute__((overloadable));
float3   half_tan(float3) __attribute__((overloadable));
float4   half_tan(float4) __attribute__((overloadable));
float8   half_tan(float8) __attribute__((overloadable));
float16  half_tan(float16) __attribute__((overloadable));

float    native_cos(float) __attribute__((overloadable));
float2   native_cos(float2) __attribute__((overloadable));
float3   native_cos(float3) __attribute__((overloadable));
float4   native_cos(float4) __attribute__((overloadable));
float8   native_cos(float8) __attribute__((overloadable));
float16  native_cos(float16) __attribute__((overloadable));

float    native_divide(float, float) __attribute__((overloadable));
float2   native_divide(float2, float2) __attribute__((overloadable));
float3   native_divide(float3, float3) __attribute__((overloadable));
float4   native_divide(float4, float4) __attribute__((overloadable));
float8   native_divide(float8, float8) __attribute__((overloadable));
float16  native_divide(float16, float16) __attribute__((overloadable));

float    native_exp(float) __attribute__((overloadable));
float2   native_exp(float2) __attribute__((overloadable));
float3   native_exp(float3) __attribute__((overloadable));
float4   native_exp(float4) __attribute__((overloadable));
float8   native_exp(float8) __attribute__((overloadable));
float16  native_exp(float16) __attribute__((overloadable));

float    native_exp2(float) __attribute__((overloadable));
float2   native_exp2(float2) __attribute__((overloadable));
float3   native_exp2(float3) __attribute__((overloadable));
float4   native_exp2(float4) __attribute__((overloadable));
float8   native_exp2(float8) __attribute__((overloadable));
float16  native_exp2(float16) __attribute__((overloadable));

float    native_exp10(float) __attribute__((overloadable));
float2   native_exp10(float2) __attribute__((overloadable));
float3   native_exp10(float3) __attribute__((overloadable));
float4   native_exp10(float4) __attribute__((overloadable));
float8   native_exp10(float8) __attribute__((overloadable));
float16  native_exp10(float16) __attribute__((overloadable));

float    native_log(float) __attribute__((overloadable));
float2   native_log(float2) __attribute__((overloadable));
float3   native_log(float3) __attribute__((overloadable));
float4   native_log(float4) __attribute__((overloadable));
float8   native_log(float8) __attribute__((overloadable));
float16  native_log(float16) __attribute__((overloadable));

float    native_log2(float) __attribute__((overloadable));
float2   native_log2(float2) __attribute__((overloadable));
float3   native_log2(float3) __attribute__((overloadable));
float4   native_log2(float4) __attribute__((overloadable));
float8   native_log2(float8) __attribute__((overloadable));
float16  native_log2(float16) __attribute__((overloadable));

float    native_log10(float) __attribute__((overloadable));
float2   native_log10(float2) __attribute__((overloadable));
float3   native_log10(float3) __attribute__((overloadable));
float4   native_log10(float4) __attribute__((overloadable));
float8   native_log10(float8) __attribute__((overloadable));
float16  native_log10(float16) __attribute__((overloadable));

float    native_powr(float, float) __attribute__((overloadable));
float2   native_powr(float2, float2) __attribute__((overloadable));
float3   native_powr(float3, float3) __attribute__((overloadable));
float4   native_powr(float4, float4) __attribute__((overloadable));
float8   native_powr(float8, float8) __attribute__((overloadable));
float16  native_powr(float16, float16) __attribute__((overloadable));

float    native_recip(float) __attribute__((overloadable));
float2   native_recip(float2) __attribute__((overloadable));
float3   native_recip(float3) __attribute__((overloadable));
float4   native_recip(float4) __attribute__((overloadable));
float8   native_recip(float8) __attribute__((overloadable));
float16  native_recip(float16) __attribute__((overloadable));

float    native_rsqrt(float) __attribute__((overloadable));
float2   native_rsqrt(float2) __attribute__((overloadable));
float3   native_rsqrt(float3) __attribute__((overloadable));
float4   native_rsqrt(float4) __attribute__((overloadable));
float8   native_rsqrt(float8) __attribute__((overloadable));
float16  native_rsqrt(float16) __attribute__((overloadable));

float    native_sin(float) __attribute__((overloadable));
float2   native_sin(float2) __attribute__((overloadable));
float3   native_sin(float3) __attribute__((overloadable));
float4   native_sin(float4) __attribute__((overloadable));
float8   native_sin(float8) __attribute__((overloadable));
float16  native_sin(float16) __attribute__((overloadable));

float    native_sqrt(float) __attribute__((overloadable));
float2   native_sqrt(float2) __attribute__((overloadable));
float3   native_sqrt(float3) __attribute__((overloadable));
float4   native_sqrt(float4) __attribute__((overloadable));
float8   native_sqrt(float8) __attribute__((overloadable));
float16  native_sqrt(float16) __attribute__((overloadable));

float    native_tan(float) __attribute__((overloadable));
float2   native_tan(float2) __attribute__((overloadable));
float3   native_tan(float3) __attribute__((overloadable));
float4   native_tan(float4) __attribute__((overloadable));
float8   native_tan(float8) __attribute__((overloadable));
float16  native_tan(float16) __attribute__((overloadable));

#endif //__CL_BUILTINS_MATH_H__
