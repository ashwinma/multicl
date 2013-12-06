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

#ifndef __CONVERT_UTIL_H__
#define __CONVERT_UTIL_H__

#include <math.h>
#include <fenv.h>

//--------------------------
// TYPE MACRO
//--------------------------
#define	DB_LLONG_MAX	ldexp((double)0x1LL, 63)
#define	DB_LLONG_MIN	ldexp((double)(-0x1LL), 63)
#define	DB_ULLONG_MAX	ldexp((double)0x1LL, 64)

#define CL_LONG_MAX         ((llong) 0x7FFFFFFFFFFFFFFFLL)
#define CL_LONG_MIN         ((llong) -0x7FFFFFFFFFFFFFFFLL - 1LL)
#define CL_ULONG_MAX        ((ullong) 0xFFFFFFFFFFFFFFFFULL)

#ifndef LLONG_MAX
	#define	LLONG_MAX	CL_LONG_MAX	//64bit
#endif

#ifndef LLONG_MIN
	#define	LLONG_MIN	CL_LONG_MIN	//64bit
#endif

#ifndef ULLONG_MAX
	#define	ULLONG_MAX	CL_ULONG_MAX//64bit
#endif

//--------------------------
// ROUND MODE FUNC
//--------------------------

#define	__CHANGE_FP_MODE(fpmode)	int old_fpmode = fegetround(); if((fpmode) != old_fpmode) {	fesetround((fpmode)); }	

#define	__RESTORE_FP_MODE(fpmode)	if((fpmode) != old_fpmode) {	fesetround((old_fpmode)); }

#define	__SAFE_INT_ZERO_TO_FP_ZERO(x, type)		((0 == (x)) ? 0.0f : (type)(x))

#define	_CL_FPMODE_RTE		(FE_TONEAREST)
#define	_CL_FPMODE_RTP		(FE_UPWARD)	
#define	_CL_FPMODE_RTN		(FE_DOWNWARD)
#define	_CL_FPMODE_RTZ		(FE_TOWARDZERO)
#define	_CL_FPMODE_DEF_FLT	(_CL_FPMODE_RTE)
#define	_CL_FPMODE_DEF_INT	(_CL_FPMODE_RTZ)

//--------------------------
// INTEGER CLAMP FUNC
//--------------------------
#define SCLAMP( _lo, _x, _hi )   ( (_x) < (_lo) ? (_lo) : ((_x) > (_hi) ? (_hi) : (_x)))

//--------------------------
// FLOAT CLAMP FUNC
//--------------------------
inline uchar clamp_float_uchar(float f) {
  return (uchar)rintf(f);
}

inline schar clamp_float_schar(float f) {
  return (schar)rintf(f);
}

inline ushort clamp_float_ushort(float f) {
  return (ushort)rintf(f);
}

inline short clamp_float_short(float f) {
  return (short)rintf(f);
}

inline uint clamp_float_uint(float f) {
  return (uint)rintf(f);
}

inline int clamp_float_int(float f) {
  return (int)rintf(f);
}

inline ullong clamp_float_ullong(float f) {
  return (ullong)rintf(f);
}

inline llong clamp_float_llong(float f) {
  return (llong)llrintf(f);
}

inline uchar clamp_float_uchar_sat(float f) {
  if (f < 0.0f) return 0;
  if (f > UCHAR_MAX) return UCHAR_MAX;
  long rx = lrintf(f);
  return (uchar)SCLAMP(0, rx, UCHAR_MAX);
}

inline schar clamp_float_schar_sat(float f) {
  if (f < SCHAR_MIN) return SCHAR_MIN;
  if (f > SCHAR_MAX) return SCHAR_MAX;
  long rx = lrintf(f);
  return (schar)SCLAMP(SCHAR_MIN, rx, SCHAR_MAX);
}

inline ushort clamp_float_ushort_sat(float f) {
  if (f < 0.0f) return 0;
  if (f > USHRT_MAX) return USHRT_MAX;
  long rx = lrintf(f);
  return (ushort)SCLAMP(0, rx, USHRT_MAX);
}

inline short clamp_float_short_sat(float f) {
  if (f < SHRT_MIN) return SHRT_MIN;
  if (f > SHRT_MAX) return SHRT_MAX;
  long rx = lrintf(f);
  return (short)SCLAMP(SHRT_MIN, rx, SHRT_MAX);
}

inline uint clamp_float_uint_sat(float f) {
  if ((double)f < (double)0.0) return 0;
  if ((double)f > (double)UINT_MAX) return UINT_MAX;
  long rx = lrintf(f);
  return (uint)SCLAMP(0, rx, UINT_MAX);
}

inline int clamp_float_int_sat(float f) {
  if ((double)f < (double)INT_MIN) return INT_MIN;
  if ((double)f > (double)INT_MAX) return INT_MAX;
  long rx = lrintf(f);
  return (int)SCLAMP(INT_MIN, rx, INT_MAX);
}

inline ullong clamp_float_ullong_sat(float f) {
  float rx = rintf(f);
  return (ullong)(rx >= DB_ULLONG_MAX ? ULLONG_MAX : rx < 0.0f ? 0 : rx );
}

inline llong clamp_float_llong_sat(float f) {
  float rx = rintf(f);
  ullong ret = (ullong)rx;
  return (rx >= DB_LLONG_MAX ? LLONG_MAX : rx < DB_LLONG_MIN ? LLONG_MIN : ret );
}

//--------------------------
// DOUBLE CLAMP FUNC
//--------------------------
inline uchar clamp_double_uchar(double d) {
  return (uchar)rint(d);
}

inline schar clamp_double_schar(double d) {
  return (schar)rint(d);
}

inline ushort clamp_double_ushort(double d) {
  return (ushort)rint(d);
}

inline short clamp_double_short(double d) {
  return (short)rint(d);
}

inline uint clamp_double_uint(double d) {
  return (uint)rint(d);
}

inline int clamp_double_int(double d) {
  return (int)rint(d);
}

inline ullong clamp_double_ullong(double d) {
  return (ullong)rint(d);
}

inline llong clamp_double_llong(double d) {
  return (llong)llrint(d);
}

inline uchar clamp_double_uchar_sat(double d) {
  if (d < 0.0) return 0;
  if (d > UCHAR_MAX) return UCHAR_MAX;
  llong rx = llrint(d);
  return (uchar)SCLAMP(0, rx, UCHAR_MAX);
}

inline schar clamp_double_schar_sat(double d) {
  if (d < SCHAR_MIN) return SCHAR_MIN;
  if (d > SCHAR_MAX) return SCHAR_MAX;
  llong rx = llrint(d);
  return (schar)SCLAMP(CHAR_MIN, rx, CHAR_MAX);
}

inline ushort clamp_double_ushort_sat(double d) {
  if (d < 0.0) return 0;
  if (d > USHRT_MAX) return USHRT_MAX;
  llong rx = llrint(d);
  return (ushort)SCLAMP(0, rx, USHRT_MAX);
}

inline short clamp_double_short_sat(double d) {
  if (d < SHRT_MIN) return SHRT_MIN;
  if (d > SHRT_MAX) return SHRT_MAX;
  llong rx = llrint(d);
  return (short)SCLAMP(SHRT_MIN, rx, SHRT_MAX);
}

inline uint clamp_double_uint_sat(double d) {
  if (d < 0.0) return 0;
  if (d > UINT_MAX) return UINT_MAX;
  llong rx = llrint(d);
  return (uint)SCLAMP(0, rx, UINT_MAX);
}

inline int clamp_double_int_sat(double d) {
  if (d < INT_MIN) return INT_MIN;
  if (d > INT_MAX) return INT_MAX;
  llong rx = llrint(d);
  return (int)SCLAMP(INT_MIN, rx, INT_MAX);
}

inline ullong clamp_double_ullong_sat(double d) {
  double rx = rint(d);
  return (ullong)(rx >= DB_ULLONG_MAX ? ULLONG_MAX : rx < 0.0f ? 0 : rx );
}

inline llong clamp_double_llong_sat(double d) {
  double rx = rint(d);
  ullong ret = (ullong)rx;
  return (rx >= DB_LLONG_MAX ? LLONG_MAX : rx < DB_LLONG_MIN ? LLONG_MIN : ret );
}

#endif
