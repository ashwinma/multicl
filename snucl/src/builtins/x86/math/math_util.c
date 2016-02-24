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
#include <limits.h>
#include <stdint.h>
#include <float.h>
#include <stdio.h>
#include "math_util.h"
#pragma STDC FP_CONTRACT OFF

double snu_copysign(double y, double x) {
  return copysign(y, x);
}
double snu_ldexp(double x, int k) {
  return ldexpl(x, k);
}
int snu_isequal(double x, double y) {
  return x==y; 
} 
int snu_isequalv(double x, double y) {
  return x==y? -1 : 0; 
} 
int snu_isfinite(double x) {
  return isfinite(x)?  1 : 0; 
} 
int snu_isfinitev(double x) {
  return isfinite(x)? -1 : 0; 
} 
int snu_islessgreater(double x, double y) {
  return islessgreater(x, y)?  1 : 0; 
} 
int snu_islessgreaterv(double x, double y) {
  return islessgreater(x, y)? -1 : 0; 
} 
int snu_isinf(double x) {
  return isinf(x)?  1 : 0; 
} 
int snu_isinfv(double x) {
  return isinf(x)? -1 : 0; 
} 
int snu_isnan(double x) {
  return isnan(x)?  1 : 0; 
} 
int snu_isnanv(double x) {
  return isnan(x)? -1 : 0; 
} 
int snu_isnormalf(float x) {
  return 0 != isnormal(x); 
} 
int snu_isnormalfv(float x) {
  return isnormal(x)? -1 : 0; 
} 
int snu_isnormal(double x) {
  return 0 != isnormal(x); 
} 
int snu_isnormalv(double x) {
  return isnormal(x)? -1 : 0; 
} 
int snu_isunordered(double x, double y) {
  return isnan(x)||isnan(y)?  1 : 0; 
} 
int snu_isunorderedv(double x, double y) {
  return isnan(x)||isnan(y)? -1 : 0; 
} 
int snu_signbit(double x) {
  return signbit(x)?  1 : 0; 
} 
int snu_signbitv(double x) {
  return signbit(x)? -1 : 0; 
} 

double snu_logb(double x){
  double result = logbl((long double)x);
  if (result <= -1022)
    return logbl(x * ldexp(1.0, 64)) - 64;
  else
    return result;
}

double snu_powr(double x, double y){
  if(x<0.0) return NAN;
  if(isnan(x)||isnan(y)) return NAN;
  if(x==1.0) {
    if(fabs(y)==INFINITY) return NAN;
    return 1.0;
  }
  if(y==0.0) {
    if(x==0.0||x==INFINITY) return NAN; 
    return 1.0;
  }
  if(x==0.0) {
    if(y<0.0) return INFINITY;
    return 0.0;
  }
  if(isinf(x)) {		
    if(y<0) return 0;
    return INFINITY;
  }
  return pow(x, y);
}

double snu_pow(double x, double y){
  return pow(x, y);
}

double snu_sqrt(double x) {
  return sqrt(x);
}

double snu_rint(double x) {
  if(fabs(x)<ldexp(0x1LL, 52)) {
    double m = copysign(ldexp(0x1LL, 52),x);
    double r = (x + m) - m;
    x = copysign(r,x); 
  }
  return x;
}

double snu_rootn(double x, int y) {
  if(0==y) return NAN;
  if(x<0&&0==(y&1)) return NAN;
  if(x==0.0) {
    switch(y&0x80000001) {
      case 0:
        return 0.0f;
      case 1:
        return x;
      case 0x80000000:
        return INFINITY;
      case 0x80000001:
        return copysign(INFINITY, x);
    }    
  }
#if 0
  double sign = x;
  x = fabs(x);
  x = exp2(log2(x)/(double)y); 
  return copysign(x, sign);
#else
  long double xx = (long double)x;
  long double sign = xx;
  xx = fabsl(xx);
  xx = exp2l(log2l(xx) / (long double)y);
  return (double)copysignl(xx, sign);
#endif
}

double snu_cos(double x) {
  scs_t sc1, sc2;
  scs_set_d(sc1, x);
  int N = snu_payne_hanek( sc2, sc1);

  scs_get_d(&x, sc2);
  unsigned int c = N & 3;
  switch ( c ) {
    case 0:
      return  cosl(x);
    case 1:
      return -sinl(x);
    case 2:
      return -cosl(x);
    case 3:
      return  sinl(x);			
  }
  return 0.0;
}

double snu_sin(double x) {
  scs_t sc1, sc2;
  scs_set_d(sc1, x);
  int N = snu_payne_hanek( sc2, sc1);

  scs_get_d(&x, sc2);
  unsigned int c = N & 3;
  switch ( c ) {
    case 0:
      return  sinl(x);
    case 1:
      return  cosl(x);
    case 2:
      return -sinl(x);
    case 3:
      return -cosl(x);			
  }
  return 0.0;
}

double snu_tan(double x) {
  scs_t sc1, sc2;
  scs_set_d(sc1, x);
  int N = snu_payne_hanek( sc2, sc1);

  scs_get_d(&x, sc2);
  unsigned int c = N & 3;
  switch ( c ) {
    case 0:
      return  tanl(x);
    case 1:
      return  -1.0 / tanl(x);
    case 2:
      return  tanl(x);
    case 3:
      return  -1.0 / tanl(x);
  }
  return 0.0;
}

double snu_sinpi(double x) {   
  long double r = reduce1l(x); 
  if( r<-0.5L ) r = -1.0L - r;
  else if ( r>0.5L ) r = 1.0L - r;
  if( r==0.0L ) return copysignl(0.0L, x);
  return snu_sin(r*M_PI);   
}

double snu_cospi(double xx) {
  long double x = xx;
  if(fabsl(x)>=ldexpl(0x1LL, 54)) {
    if( fabsl(x)==INFINITY)
      return NAN;
    return 1.0L; 
  }
  x = reduce1l(x+0.5L);
  if( x<-0.5L ) x = -1.0L-x;
  else if ( x>0.5L ) x = 1.0L-x;
  if( x==0.0L ) return 0.0L;
  return snu_sin(x*M_PI);
}

double snu_tanpi(double xx) {
  long double x = xx;
  long double sign = copysignl(1.0L, x);
  long double z = fabsl(x);
  if( z>=ldexpl(0x1LL, 53) ) {
    if( z==INFINITY ) return x-x;
    return copysignl(0.0L, x);
  }
  double nearest = rintl(z);
  int i = (int) nearest;
  z -= nearest;                   
  if( (i&1) && z == 0.0L ) sign = -sign;
  sign *= copysignl(1.0L, z);
  z = fabsl(z);
  if( z>0.25L ) {
    z = 0.5L-z;
    return sign/snu_tan(z*M_PI);
  }
  return sign*snu_tan(z*M_PI);
}

float snu_sincosf(double x, float *y) {
  scs_t sc1, sc2;
  scs_set_d(sc1, x);
  int N = snu_payne_hanek( sc2, sc1);

  scs_get_d(&x, sc2);
  int c = N & 3;
  switch (c) {
    case 0:
      *y = cos(x);
      return  sin(x);
    case 1:
      *y = -sin(x);
      return  cos(x);
    case 2:
      *y = -cos(x);
      return -sin(x);
    case 3:
      *y = sin(x);
      return -cos(x);			
  }
  return 0.0;	
}

double snu_sincos(double x, double *y) {
  scs_t sc1, sc2;
  scs_set_d(sc1, x);
  int N = snu_payne_hanek( sc2, sc1);

  scs_get_d(&x, sc2);
  int c = N & 3;
  switch (c) {
    case 0:
      *y = cosl(x);
      return  sinl(x);
    case 1:
      *y = -sinl(x);
      return  cosl(x);
    case 2:
      *y = -cosl(x);
      return -sinl(x);
    case 3:
      *y = sinl(x);
      return -cosl(x);			
  }
  return 0.0;	
}

double snu_acos(double x) {
  return acos(x);
}

double snu_acosh(double x) {
  return acosh(x);
}

double snu_asin(double x) {
  return asin(x);
}

double snu_asinh(double x) {
  return asinh(x);
}

double snu_atan(double x) {
  return atan(x);
}

double snu_atan2(double y, double x) {
  return atan2(y, x);
}

double snu_atanh(double x) {
/*
 * ====================================================
 * This function is from fdlibm: http://www.netlib.org
 *   It is Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice 
 * is preserved.
 * ====================================================
 */
  if( isnan(x)  )
    return x + x;

  long double signed_half = copysignl( 0.5L, x );
  x = fabsl(x);
  if( x > 1.0L )
    return NAN;

  if( x < 0.5L )
    return signed_half * log1pl( 2.0L * ( x + x*x / (1-x) ) );

  return signed_half * log1pl(2.0L * x / (1-x));
}

// Compute cube-root
double snu_cbrt(double x) {
  return copysignl(powl(fabsl(x), 1.0L/3.0L), x);
}

double snu_ceil(double x) {
  return ceil(x);
}

double snu_cosh(double x) {
  return cosh(x);
}

double snu_sinh(double x) {
  return sinh(x);
}

double snu_tanh(double x) {
  return tanh(x);
}

double snu_erf(double x) {
  return erf(x);
}

double snu_erfc(double x) {
  return erfc(x);
}

double snu_exp(double x) {
  return exp(x);
}

double snu_exp10(double x ){
  return exp2l(x*ldexpl(0xd49a784bcd1b8afeLL, -62));
}

double snu_exp2(double x) {
  return exp2(x);
}

double snu_expm1(double x) {
  return expm1(x);
}

double snu_fabs(double x) {
  return fabs(x);
}

double snu_fdim(double xx, double yy) {
  if( xx!=xx ) return xx;
  if( yy!=yy ) return yy;
  return  (xx>yy) ? xx -= yy : 0.0f;
}

double snu_floor(double x) {
  return floor(x);
}

double snu_fmax(double x, double y) {
  return fmax(x, y);
}

double snu_fmin(double x, double y) {
  return fmin(x, y);
}

double snu_fmod(double xx, double yy)
{
  double x = (double)xx;
  double y = (double)yy;
	if( x == 0.0L && fabsl(y) > 0.0L )
		return x;

	if( fabsl(x) == INFINITY || y == 0.0L )
		return cl_make_nan();

	if( fabsl(y) == INFINITY )	// we know x is finite from above
		return x;

	return fmod( (double) x, (double) y );
}

double snu_frexp(double x, int *exp) {
  return frexp(x, exp);
}

double snu_lgamma_r(double x, int *signp) {
  return lgamma_r(x, signp);
}

float snu_fractf(float x, float *ip) {
  float i;
  float f = modff((float)x, &i);
  if ( f<0.0 ) {
    f = 1.0f + f;
    i -= 1.0f;
    if( f==1.0f ) {
      f = ldexp(0x1fffffeL, -25);
    }
  }
  *ip = i;
  return f;
}

double snu_fract(double x, double *ip) {
  double i;
  double f = modf((double)x, &i);
  if ( f<0.0 ) {
    f = 1.0 + f;
    i -= 1.0;
    if( f==1.0 ) {
      f = ldexp(0x1fffffffffffffLL, -53);
    }
  }
  *ip = i;
  return f;
}

double snu_log(double x) {
  return log(x);
}

double snu_log10(double x) {
  return log10(x);
}

double snu_log2(double x) {
  return log2(x);
}

float snu_hypotf(float x, float y) {
  if (isinf(x) || isinf(y))
    return INFINITY;
  double a = (double)x;
  double b = (double)y;
  return (float)sqrt(a * a + b * b);
}

double snu_hypot(double x, double y) {
  if (isinf(x) || isinf(y))
    return INFINITY;
  if (isnan(x) || isnan(y))
    return NAN;
  if (x == 0.0 && y == 0.0)
    return 0.0;
  x = fabs(x);
  y = fabs(y);
  double t = fmin(x, y);
  x = fmax(x, y);
  y = t;
  return x * sqrt(1 + (y/x)*(y/x));
}

int snu_ilogb(double x) {
  union { 
    double d; 
    ullong u;
  } val_union;

  val_union.d = x;
  int exp = GET_EXP(val_union.u);
  if(exp==0x7ff) {
    if(val_union.u&0x000fffffffffffffULL) return INT_MAX;
    else return INT_MAX;
  }
  else if(exp==0) {
    val_union.d = x*ldexp((double)0x1LL, 64);
    exp = GET_EXP(val_union.u);
    if(exp==0) return INT_MIN;
    return exp-(1023+64);
  }
  return exp-1023;
}

double snu_lgamma(double x) {
  return lgamma(x);
}

double snu_log1p(double x) {
  return log1p(x);
}

double snu_maxmag(double x, double y) {
  double ax = fabs(x);
  double ay = fabs(y);
  if (ax > ay) return x;
  else if (ax < ay) return y;
  else return snu_fmax(x, y);
}

double snu_minmag(double x, double y) {
  double ax = fabs(x);
  double ay = fabs(y);
  if (ax < ay) return x;
  else if (ax > ay) return y;
  else return snu_fmin(x, y);
}

float snu_modff(float x, float *iptr) {
  return modff(x, iptr);
}

double snu_modf(double x, double *iptr) {
  return modf(x, iptr);
}

double snu_nan(ullong x) {
    union{ ullong u; float f; } ret;
    ret.u = x|0x7fc00000u; // NaN
    return (double)ret.f;
}

float snu_nextafterf(float x, float y) {
  if( x != x ) return x;
  if( y != y ) return y;
  if( x == y ) return y;

  typedef union {
    int i;
    float f;
  } int32f_t;
  int32f_t a, b;
  a.f  = x;
  b.f  = y;

  if(a.i&0x80000000) a.i = 0x80000000 - a.i;
  if(b.i&0x80000000) b.i = 0x80000000 - b.i;

  a.i += (a.i<b.i) ? 1 : -1;
  a.i = (a.i<0) ? 0x80000000-a.i : a.i;

  return a.f;
}

double snu_nextafter(double x, double y) {
  if( x != x ) return x;
  if( y != y ) return y;
  if( x == y ) return y;

  typedef union {
    llong i;
    double d;
  } int64f_t;
  int64f_t a, b;
  a.d  = x;
  b.d  = y;

  if(a.i&0x8000000000000000ULL) a.i = 0x8000000000000000ULL - a.i;
  if(b.i&0x8000000000000000ULL) b.i = 0x8000000000000000ULL - b.i;

  a.i += (a.i<b.i) ? 1 : -1;
  a.i = (a.i<0) ? 0x8000000000000000ULL-a.i : a.i;

  return a.d;
}

double snu_tgamma(double x) {
  return tgamma(x);
}

double snu_trunc(double x) {
  return trunc(x);
}

double snu_remainder(double x, double y) {
  return remainder(x, y);
}

double snu_round(double x) {
  return round(x);
}

double snu_remquo(double x, double y, int *n) {
//  return remquol(xd, yd, n);
  // remquo in glibc only computes the lower three bits of x/y
  const ullong MSB_64 = (1ULL << 63);

  union {
    ullong u;
    double d;
  } ux, uy;

  ux.d = x;
  uy.d = y;

  if (isnan(x) || isinf(x) || (ux.u << 1) == 0 ||
      isnan(y) || isinf(y) || (uy.u << 1) == 0) {
    return remquo(x, y, n);
  }

  ullong signX, signY;
  ullong mantX, mantY;
  int expX, expY;

  // X = signX * mantX(0 0 0 0 2^-1 2^-2 2^-3 ... 2^-60) * 2^expX
  // Y = signY * mantY(0 0 0 0 2^-1 2^-2 2^-3 ... 2^-60) * 2^expY
  signX = ux.u & MSB_64;
  signY = uy.u & MSB_64;
  mantX = (ullong)(fabs(frexp(x, &expX)) * ldexp(1.0, 60));
  mantY = (ullong)(fabs(frexp(y, &expY)) * ldexp(1.0, 60));

  ullong div, rem;
  div = mantX / mantY;
  rem = mantX % mantY;
  while (expX > expY) {
    rem <<= 1;
    div = (div << 1) | (rem >= mantY);
    rem %= mantY;
    expX--;
  }

  ullong sdiv;

  if (expY - expX >= 64)
    sdiv = 0;
  else
    sdiv = (div >> (expY - expX));

  if (expY - expX == 0) {
    if (rem > mantY / 2 || (rem >= (mantY + 1) / 2 && (sdiv & 0x1)))
      sdiv++;
  } else if (expY - expX <= 64) {
    int shift = 64 - (expY - expX);
    if ((div << shift) > MSB_64 ||
        ((div << shift) == MSB_64 && (rem > 0 || (sdiv & 0x1))))
      sdiv++;
  }

  *n = (sdiv & 0x7F);
  if (signX ^ signY)
    *n *= -1;

  int dummy_quo;
  return remquo(x, y, &dummy_quo);
}

/// Used in sinpi/cospi
long double reduce1l(long double x) {
  static long double unit_exp = 0; 
  if(0.0L==unit_exp)
    unit_exp = scalbnl( 1.0L, LDBL_MANT_DIG);

  if(fabsl(x)>=unit_exp) {
    if(fabsl(x)==INFINITY)
      return NAN;
    return 0.0L;
  }
  const long double r = copysignl(unit_exp, x);
  long double z = x + r;
  z -= r;
  return x - z;    
}

static inline ullong shift_right_ulong_with_round(ullong x, int n) {
  if (n >= 64)
    return (x != 0);
  else
    return (x >> n) | ((x << (64 - n)) != 0);
}

float snu_fmaf(float a, float b, float c) {
  const uint kMSB32 = (1U << 31);
  const ullong MSB_64 = (1ULL << 63);
  union {
    uint u;
    float f;
  } ua, ub, uc, ur;

  ua.f = a;
  ub.f = b;
  uc.f = c;

  int ea = isnan(a) || isinf(a) || (ua.u << 1) == 0;
  int eb = isnan(b) || isinf(b) || (ub.u << 1) == 0;
  int ec = isnan(c) || isinf(c) || (uc.u << 1) == 0;
  if (ea || eb || ec) {
    if (isinf(c) && !ea && !eb)
      return c;
    else
      return (a * b) + c;
  }

  uint signA, signB, signC;
  uint mantA, mantB, mantC;
  int expA, expB, expC;

  // A = signA * mantA(2^-1 2^-2 2^-3 ... 2^-32) * 2^expA
  // B = signB * mantB(2^-1 2^-2 2^-3 ... 2^-32) * 2^expB
  // C = signC * mantC(2^-1 2^-2 2^-3 ... 2^-32) * 2^expC
  signA = ua.u & kMSB32;
  signB = ub.u & kMSB32;
  signC = uc.u & kMSB32;
  mantA = (uint)(fabsf(frexp(a, &expA)) * ldexp(1.0, 32));
  mantB = (uint)(fabsf(frexp(b, &expB)) * ldexp(1.0, 32));
  mantC = (uint)(fabsf(frexp(c, &expC)) * ldexp(1.0, 32));

  // AB = signAB * lmantAB(2^-1 2^-2 2^-3 ... 2^-64) * 2^expAB
  uint signAB;
  ullong lmantAB;
  int expAB;

  signAB = signA ^ signB;
  lmantAB = (ullong)mantA * (ullong)mantB;
  expAB = expA + expB;
  if ((lmantAB & MSB_64) == 0) {
    lmantAB <<= 1;
    expAB--;
  }

  // AB = signAB * lmantAB(2^-1 2^-2 2^-3 ... 2^-64) * 2^exp
  // C  = signC  * lmantC (2^-1 2^-2 2^-3 ... 2^-64) * 2^exp
  ullong lmantC;
  lmantC = (ullong)mantC << 32;
  if (expAB > expC) {
    lmantC = shift_right_ulong_with_round(lmantC, expAB - expC);
    expC = expAB;
  } else if (expAB < expC) {
    lmantAB = shift_right_ulong_with_round(lmantAB, expC - expAB);
    expAB = expC;
  }

  // R = signR * lmantR(2^-1 2^-2 2^-3 ... 2^-64) * 2^expR
  uint signR;
  ullong lmantR;
  int expR;

  expR = expAB;
  if (signAB == signC) {
    // R = AB + C
    signR = signAB;
    lmantR = lmantAB + lmantC;
    if (lmantR < lmantAB) { // overflow
      lmantR = (lmantR >> 1) | ((lmantR & 1) != 0);
      lmantR |= MSB_64;
      expR++;
    }
  } else {
    // R = AB - C
    signR = signAB;
    lmantR = lmantAB - lmantC;
    if (lmantR > lmantAB) { // overflow
      signR ^= kMSB32;
      lmantR = ~lmantR + 1;
    }
    if (lmantR > 0) {
      while ((lmantR & MSB_64) == 0) {
        lmantR <<= 1;
        expR--;
      }
    } else {
      signR = 0;
    }
  }

  // Restore to a float number
  expR--;
  if (lmantR == 0) {
    ur.f = 0.0;
  } else if (expR >= 128 || expR <= -150) {
    ur.f = ldexp((float)lmantR, expR - 63);
  } else {
    int sig_digits = 24;
    if (expR <= -127)
      sig_digits += expR + 126;

    if ((lmantR << sig_digits) < MSB_64 ||
        ((lmantR << sig_digits) == MSB_64 &&
         (lmantR << (sig_digits - 1)) < MSB_64)) {
      lmantR = lmantR >> (64 - sig_digits);
    } else {
      lmantR = (lmantR >> (64 - sig_digits)) + 1;
    }
    ur.f = ldexp((float)lmantR, expR - (sig_digits - 1));
  }
  ur.u |= signR;
  return ur.f;
}

typedef struct s_uint128 {
  ullong high;
  ullong low;
  s_uint128() {}
  s_uint128(ullong v): high(0), low(v) {}
} uint128;

inline uint128 operator+(const uint128& x, const uint128& y) {
  uint128 r;
  r.low = x.low + y.low;
  r.high = x.high + y.high;
  if (r.low < x.low) // overflow
    r.high++;
  return r;
}

inline uint128 operator-(const uint128& x, const uint128& y) {
  uint128 r;
  r.low = x.low - y.low;
  r.high = x.high - y.high;
  if (r.low > x.low) // overflow
    r.high--;
  return r;
}

inline uint128 operator&(const uint128& x, const uint128& y) {
  uint128 r;
  r.low = (x.low & y.low);
  r.high = (x.high & y.high);
  return r;
}

inline uint128 operator|(const uint128& x, const uint128& y) {
  uint128 r;
  r.low = (x.low | y.low);
  r.high = (x.high | y.high);
  return r;
}

inline uint128& operator|=(uint128& lhs, const uint128& rhs) {
  lhs = lhs | rhs;
  return lhs;
}

inline uint128 operator~(const uint128& x) {
  uint128 r;
  r.low = ~x.low;
  r.high = ~x.high;
  return r;
}

inline bool operator==(const uint128& x, const uint128& y) {
  return (x.low == y.low && x.high == y.high);
}

inline bool operator!=(const uint128& x, const uint128& y) {
  return (x.low != y.low || x.high != y.high);
}

inline bool operator<(const uint128& x, const uint128& y) {
  return (x.high < y.high || (x.high == y.high && x.low < y.low));
}

inline bool operator>(const uint128& x, const uint128& y) {
  return (x.high > y.high || (x.high == y.high && x.low > y.low));
}

inline uint128 operator<<(const uint128& x, const uint n) {
  uint128 r;
  if (n < 64) {
    r.high = (x.high << n) | (x.low >> (64 - n));
    r.low = (x.low << n);
  } else if (n < 128) {
    r.high = (x.low << (n - 64));
    r.low = 0;
  } else {
    r.high = 0;
    r.low = 0;
  }
  return r;
}

inline uint128& operator<<=(uint128& lhs, const uint n) {
  lhs = lhs << n;
  return lhs;
}

inline uint128 operator>>(const uint128& x, const uint n) {
  uint128 r;
  if (n < 64) {
    r.low = (x.low >> n) | (x.high << (64 - n));
    r.high = (x.high >> n);
  } else if (n < 128) {
    r.low = (x.high >> (n - 64));
    r.high = 0;
  } else {
    r.low = 0;
    r.high = 0;
  }
  return r;
}

static inline uint128 shift_right_uint128_with_round(const uint128& x, int n) {
  if (n >= 128)
    return (x != 0);
  else
    return (x >> n) | ((x << (128 - n)) != 0);
}

// From integer/mul64.c
inline uint128 mul_ulong(ullong x, ullong y) {
  ullong x_hi = x >> 32;
  ullong x_lo = x & 0xFFFFFFFF;
  ullong y_hi = y >> 32;
  ullong y_lo = y & 0xFFFFFFFF;

  ullong a = x_hi * y_hi;

  ullong b = x_hi * y_lo;
  ullong b_hi = b >> 32;
  ullong b_lo = b & 0xFFFFFFFF;

  ullong c = x_lo * y_hi;
  ullong c_hi = c >> 32;
  ullong c_lo = c & 0xFFFFFFFF;

  ullong d = x_lo * y_lo;
  ullong d_hi = d >> 32;
  ullong d_lo = d & 0xFFFFFFFF;

  ullong r_hi = b_lo + c_lo + d_hi;

  uint128 r;
  r.low = (r_hi << 32) + d_lo;
  r.high = a + b_hi + c_hi + (r_hi >> 32);
  return r;
}

double snu_fma(double a, double b, double c) {
  const ullong MSB_64 = (1ULL << 63);
  const uint128 MSB_128 = (uint128)1ULL << 127;
  union {
    ullong u;
    double d;
  } ua, ub, uc, ur;

  ua.d = a;
  ub.d = b;
  uc.d = c;

  int ea = isnan(a) || isinf(a) || (ua.u << 1) == 0;
  int eb = isnan(b) || isinf(b) || (ub.u << 1) == 0;
  int ec = isnan(c) || isinf(c) || (uc.u << 1) == 0;
  if (ea || eb || ec) {
    if (isinf(c) && !ea && !eb)
      return c;
    else
      return (a * b) + c;
  }

  ullong signA, signB, signC;
  ullong mantA, mantB, mantC;
  int expA, expB, expC;

  // A = signA * mantA(2^-1 2^-2 2^-3 ... 2^-64) * 2^expA
  // B = signB * mantB(2^-1 2^-2 2^-3 ... 2^-64) * 2^expB
  // C = signC * mantC(2^-1 2^-2 2^-3 ... 2^-64) * 2^expC
  signA = ua.u & MSB_64;
  signB = ub.u & MSB_64;
  signC = uc.u & MSB_64;
  mantA = (ullong)(fabs(frexp(a, &expA)) * ldexp(1.0, 64));
  mantB = (ullong)(fabs(frexp(b, &expB)) * ldexp(1.0, 64));
  mantC = (ullong)(fabs(frexp(c, &expC)) * ldexp(1.0, 64));

  // AB = signAB * lmantAB(2^-1 2^-2 2^-3 ... 2^-128) * 2^expAB
  ullong signAB;
  uint128 lmantAB;
  int expAB;

  signAB = signA ^ signB;
  lmantAB = mul_ulong(mantA, mantB);
  expAB = expA + expB;
  if ((lmantAB & MSB_128) == 0) {
    lmantAB <<= 1;
    expAB--;
  }

  // AB = signAB * lmantAB(2^-1 2^-2 2^-3 ... 2^-128) * 2^exp
  // C  = signC  * lmantC (2^-1 2^-2 2^-3 ... 2^-128) * 2^exp
  uint128 lmantC;
  lmantC = (uint128)mantC << 64;
  if (expAB > expC) {
    lmantC = shift_right_uint128_with_round(lmantC, expAB - expC);
    expC = expAB;
  } else if (expAB < expC) {
    lmantAB = shift_right_uint128_with_round(lmantAB, expC - expAB);
    expAB = expC;
  }

  // R = signR * lmantR(2^-1 2^-2 2^-3 ... 2^-128) * 2^expR
  ullong signR;
  uint128 lmantR;
  int expR;

  expR = expAB;
  if (signAB == signC) {
    // R = AB + C
    signR = signAB;
    lmantR = lmantAB + lmantC;
    if (lmantR < lmantAB) { // overflow
      lmantR = (lmantR >> 1) | ((lmantR & 1) != 0);
      lmantR |= MSB_128;
      expR++;
    }
  } else {
    // R = AB - C
    signR = signAB;
    lmantR = lmantAB - lmantC;
    if (lmantR > lmantAB) { // overflow
      signR ^= MSB_64;
      lmantR = ~lmantR + 1;
    }
    if (lmantR > 0) {
      while ((lmantR & MSB_128) == 0) {
        lmantR <<= 1;
        expR--;
      }
    } else {
      signR = 0;
    }
  }

  // Restore to a double number
  expR--;
  if (lmantR == 0) {
    ur.d = 0.0;
  } else if (expR >= 1024 || expR <= -1075) {
    ur.d = ldexp((double)lmantR.high, expR - 63);
  } else {
    int sig_digits = 53;
    if (expR <= -1023)
      sig_digits += expR + 1022;

    if ((lmantR << sig_digits) < MSB_128 ||
        ((lmantR << sig_digits) == MSB_128 &&
         (lmantR << (sig_digits - 1)) < MSB_128)) {
      lmantR = lmantR >> (128 - sig_digits);
    } else {
      lmantR = (lmantR >> (128 - sig_digits)) + 1;
    }

    ur.d = ldexp((double)lmantR.low, expR - (sig_digits - 1));
  }
  ur.u |= signR;
  return ur.d;
}
