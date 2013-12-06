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

#ifndef MATH_UTIL_H_
#define MATH_UTIL_H_
#include <cl_cpu_ops.h>
#include <cl_cpu_types.h>
#include <string.h>
#include <math.h>

#define RADIX (30)
#define DIGITS 6
#define GET_EXP(x) ((int)(x>>52)&(0x7ff))

#ifndef M_PI
#define M_PI    3.14159265358979323846264338327950288
#endif

#ifndef M_PI_4
#define M_PI_4 (M_PI/4)
#endif

typedef union
{
	uint64_t i;
	double d;
}uint64d_t;

static const uint64d_t _CL_NAN = { 0x7ff8000000000000ULL };
#define cl_make_nan() _CL_NAN.d

/*===---             Correctly Rounded mathematical library             ---===*/
/*                                                                            */
/** version: crlibm-0.10beta                                                  */
/** website: http://lipforge.ens-lyon.fr/www/crlibm                           */
/*                                                                            */
/* Copyright (C) 2002  David Defour and Florent de Dinechin

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA  */

/* Size of an SCS digit */
#define SCS_NB_BITS 30
/* Number of digits in the SCS structure */
#define SCS_NB_WORDS 8
/** @internal An union to cast floats into doubles or the other way round. For
  internal purpose only */
#define SCS_RADIX   ((unsigned int)(1<<SCS_NB_BITS))

typedef union {
  int32_t i[2]; /* Signed (may be useful) */
  int64_t l;    /* Signed (may be useful) */
  double d;
} db_number;

struct scs {
  /** the digits, as 32 bits words */
  uint32_t h_word[SCS_NB_WORDS];
  /** Used to store Nan,+/-0, Inf, etc and then let the hardware handle them */
  db_number exception;
  /** This corresponds to the exponent in an FP format, but here we are
    in base 2^32  */
  int index;
  /** The sign equals 1 or -1*/
  int sign;
};
typedef struct scs scs;
/** scs_ptr is a pointer on a SCS structure */
typedef struct scs * scs_ptr;
/** scs_t is an array of one SCS struct to lighten syntax : you may
   declare a scs_t object, and pass it to the scs functions (which
    expect pointers) without using ampersands.
    */
typedef struct scs scs_t[1];

#define DB_ONE    {{0x00000000 ,0x3ff00000}}
static const int two_over_pi[]=
{0x28be60db, 0x24e44152, 0x27f09d5f, 0x11f534dd,
  0x3036d8a5, 0x1993c439, 0x0107f945, 0x23abdebb,
  0x31586dc9, 0x06e3a424, 0x374b8019, 0x092eea09,
  0x3464873f, 0x21deb1cb, 0x04a69cfb, 0x288235f5,
  0x0baed121, 0x0e99c702, 0x1ad17df9, 0x013991d6,
  0x0e60d4ce, 0x1f49c845, 0x3e2ef7e4, 0x283b1ff8,
  0x25fff781, 0x1980fef2, 0x3c462d68, 0x0a6d1f6d,
  0x0d9fb3c9, 0x3cb09b74, 0x3d18fd9a, 0x1e5fea2d,
  0x1d49eeb1, 0x3ebe5f17, 0x2cf41ce7, 0x378a5292,
  0x3a9afed7, 0x3b11f8d5, 0x3421580c, 0x3046fc7b,
  0x1aeafc33, 0x3bc209af, 0x10d876a7, 0x2391615e,
  0x3986c219, 0x199855f1, 0x1281a102, 0x0dffd880,
  0x135cc9cc, 0x10606155};

/*
 * This scs number store 211 bits of pi/2
 */
static const scs Pio2=
{{0x00000001, 0x2487ed51, 0x042d1846, 0x26263314,
  0x1701b839, 0x28948127, 0x01114cf9, 0x23a0105d},
DB_ONE,  0,   1 };
#define Pio2_ptr  (scs_ptr)(& Pio2)

#define R_HW  result->h_word
#define R_SGN result->sign
#define R_IND result->index
#define R_EXP result->exception.d

#define X_HW  x->h_word
#define X_SGN x->sign
#define X_IND x->index
#define X_EXP x->exception.d

#define Y_HW  y->h_word
#define Y_SGN y->sign
#define Y_IND y->index
#define Y_EXP y->exception.d

#define Z_HW  z->h_word
#define Z_SGN z->sign
#define Z_IND z->index
#define Z_EXP z->exception.d

#define W_HW  w->h_word
#define W_SGN w->sign
#define W_IND w->index
#define W_EXP w->exception.d

/*  Compute the carry of r1, remove it from r1, and add it to r0 */
#define SCS_CARRY_PROPAGATE(r1,r0,tmp) \
        {tmp = r1>>SCS_NB_BITS; r0 += tmp; r1 -= (tmp<<SCS_NB_BITS);}
#define HI 1
#define LO 0
/* An int such that SCS_MAX_RANGE * SCS_NB_BITS < 1024,
   where 1024 is the max of the exponent of a double number.
   Used in scs2double.c along with radix_rng_double et al.
   The value of 32 is OK for all practical values of SCS_NB_BITS */
#define SCS_MAX_RANGE  32
const db_number radix_two_double  = {{0x00000000 , ((1023+2*SCS_NB_BITS)<<20)             }};
const db_number radix_mtwo_double = {{0x00000000 , ((1023-2*SCS_NB_BITS)<<20)             }};
const db_number radix_rng_double  = {{0x00000000 , ((1023+SCS_NB_BITS*SCS_MAX_RANGE)<<20) }};
#define SCS_RADIX_TWO_DOUBLE     radix_two_double.d   /* 2^(2.SCS_NB_BITS)         */
#define SCS_RADIX_RNG_DOUBLE     radix_rng_double.d   /* 2^(SCS_NB_BITS.SCS_MAX_RANGE) */
#define SCS_RADIX_MTWO_DOUBLE    radix_mtwo_double.d  /* 2^-(2.SCS_NB_BITS)        */
#define ULL(bits) 0x##bits##uLL
#define SCS_MASK_RADIX ((unsigned int)(SCS_RADIX-1))

void scs_set_d(scs_ptr result, double x);
void scs_get_d(double *result, scs_ptr x);
void scs_set(scs_ptr result, scs_ptr x);
void scs_mul(scs_ptr result, scs_ptr x, scs_ptr y);
void scs_add(scs_ptr result, scs_ptr x, scs_ptr y);
void do_add(scs_ptr result, scs_ptr x, scs_ptr y);
void do_sub(scs_ptr result, scs_ptr x, scs_ptr y);
void inline scs_zero(scs_ptr result);
/* End of CRlibm */

//double snu_cos(double x);
//double snu_sin(double x);
//double snu_tan(double x);
//double snu_sincos(double x, double *y);
////float snu_sincosf(double x, float *y);
//double snu_fma(double xx,  double yy,  double zz);
double reduce1(double x);
long double reduce1l(long double x);
int snu_payne_hanek(scs_ptr result, const scs_ptr x);

#endif 
