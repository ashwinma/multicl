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

#ifndef __CL_OPS_ULONGN_H
#define __CL_OPS_ULONGN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>

///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* ulong2 addition (+) */
inline ulong2 operator+(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline ulong2 operator+(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline ulong2 operator+(ullong x, const ulong2& y) {
  return y + x;
}

/* ulong3 addition (+) */
inline ulong3 operator+(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline ulong3 operator+(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline ulong3 operator+(ullong x, const ulong3& y) {
  return y + x;
}

/* ulong4 addition (+) */
inline ulong4 operator+(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline ulong4 operator+(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline ulong4 operator+(ullong x, const ulong4& y) {
  return y + x;
}

/* ulong8 addition (+) */
inline ulong8 operator+(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline ulong8 operator+(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline ulong8 operator+(ullong x, const ulong8& y) {
  return y + x;
}

/* ulong16 addition (+) */
inline ulong16 operator+(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline ulong16 operator+(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline ulong16 operator+(ullong x, const ulong16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* ulong2 subtraction (-) */
inline ulong2 operator-(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline ulong2 operator-(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline ulong2 operator-(ullong x, const ulong2& y) {
  ulong2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* ulong3 subtraction (-) */
inline ulong3 operator-(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline ulong3 operator-(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline ulong3 operator-(ullong x, const ulong3& y) {
  ulong3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* ulong4 subtraction (-) */
inline ulong4 operator-(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline ulong4 operator-(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline ulong4 operator-(ullong x, const ulong4& y) {
  ulong4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* ulong8 subtraction (-) */
inline ulong8 operator-(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline ulong8 operator-(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline ulong8 operator-(ullong x, const ulong8& y) {
  ulong8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* ulong16 subtraction (-) */
inline ulong16 operator-(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline ulong16 operator-(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline ulong16 operator-(ullong x, const ulong16& y) {
  ulong16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* ulong2 multiplication (*) */
inline ulong2 operator*(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline ulong2 operator*(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline ulong2 operator*(ullong x, const ulong2& y) {
  return y * x;
}

/* ulong3 multiplication (*) */
inline ulong3 operator*(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline ulong3 operator*(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline ulong3 operator*(ullong x, const ulong3& y) {
  return y + x;
}


/* ulong4 multiplication (*) */
inline ulong4 operator*(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline ulong4 operator*(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline ulong4 operator*(ullong x, const ulong4& y) {
  return y + x;
}

/* ulong8 multiplication (*) */
inline ulong8 operator*(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline ulong8 operator*(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline ulong8 operator*(ullong x, const ulong8& y) {
  return y * x;
}

/* ulong16 multiplication (*) */
inline ulong16 operator*(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline ulong16 operator*(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline ulong16 operator*(ullong x, const ulong16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* ulong2 division (/) */
inline ulong2 operator/(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline ulong2 operator/(const ulong2& x, ullong y) {
  ulong2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline ulong2 operator/(ullong x, const ulong2& y) {
  ulong2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* ulong3 division (/) */
inline ulong3 operator/(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline ulong3 operator/(const ulong3& x, ullong y) {
  ulong3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline ulong3 operator/(ullong x, const ulong3& y) {
  ulong3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* ulong4 division (/) */
inline ulong4 operator/(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline ulong4 operator/(const ulong4& x, ullong y) {
  ulong4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline ulong4 operator/(ullong x, const ulong4& y) {
  ulong4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* ulong8 division (/) */
inline ulong8 operator/(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline ulong8 operator/(const ulong8& x, ullong y) {
  ulong8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline ulong8 operator/(ullong x, const ulong8& y) {
  ulong8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* ulong16 division (/) */
inline ulong16 operator/(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline ulong16 operator/(const ulong16& x, ullong y) {
  ulong16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline ulong16 operator/(ullong x, const ulong16& y) {
  ulong16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* ulong2 remainder (%) */
inline ulong2 operator%(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline ulong2 operator%(const ulong2& x, ullong y) {
  ulong2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline ulong2 operator%(ullong x, const ulong2& y) {
  ulong2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* ulong3 remainder (%) */
inline ulong3 operator%(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline ulong3 operator%(const ulong3& x, ullong y) {
  ulong3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline ulong3 operator%(ullong x, const ulong3& y) {
  ulong3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* ulong4 remainder (%) */
inline ulong4 operator%(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline ulong4 operator%(const ulong4& x, ullong y) {
  ulong4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline ulong4 operator%(ullong x, const ulong4& y) {
  ulong4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* ulong8 remainder (%) */
inline ulong8 operator%(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline ulong8 operator%(const ulong8& x, ullong y) {
  ulong8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline ulong8 operator%(ullong x, const ulong8& y) {
  ulong8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* ulong16 remainder (%) */
inline ulong16 operator%(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline ulong16 operator%(const ulong16& x, ullong y) {
  ulong16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline ulong16 operator%(ullong x, const ulong16& y) {
  ulong16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* ulong2 unary positive (+) */
inline ulong2 operator+(const ulong2& x) {
  return x;
}
/* ulong3 unary positive (+) */
inline ulong3 operator+(const ulong3& x) {
  return x;
}

/* ulong4 unary positive (+) */
inline ulong4 operator+(const ulong4& x) {
  return x;
}

/* ulong8 unary positive (+) */
inline ulong8 operator+(const ulong8& x) {
  return x;
}

/* ulong16 unary positive (+) */
inline ulong16 operator+(const ulong16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* ulong2 unary negative (-) */
inline ulong2 operator-(const ulong2& x) {
  ulong2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* ulong3 unary negative (-) */
inline ulong3 operator-(const ulong3& x) {
  ulong3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* ulong4 unary negative (-) */
inline ulong4 operator-(const ulong4& x) {
  ulong4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* ulong8 unary negative (-) */
inline ulong8 operator-(const ulong8& x) {
  ulong8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* ulong16 unary negative (-) */
inline ulong16 operator-(const ulong16& x) {
  ulong16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* ulong2 unary post-increment (++) */
inline ulong2 operator++(ulong2 &x, int n) {
  ulong2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ulong3 unary post-increment (++) */
inline ulong3 operator++(ulong3 &x, int n) {
  ulong3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ulong4 unary post-increment (++) */
inline ulong4 operator++(ulong4 &x, int n) {
  ulong4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ulong8 unary post-increment (++) */
inline ulong8 operator++(ulong8 &x, int n) {
  ulong8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ulong16 unary post-increment (++) */
inline ulong16 operator++(ulong16 &x, int n) {
  ulong16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* ulong2 unary pre-increment (++) */
inline ulong2 operator++(ulong2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* ulong3 unary pre-increment (++) */
inline ulong3 operator++(ulong3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* ulong4 unary pre-increment (++) */
inline ulong4 operator++(ulong4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* ulong8 unary pre-increment (++) */
inline ulong8 operator++(ulong8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* ulong16 unary pre-increment (++) */
inline ulong16 operator++(ulong16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* ulong2 unary post-decrement (--) */
inline ulong2 operator--(ulong2 &x, int n) {
  ulong2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* ulong3 unary post-decrement (--) */
inline ulong3 operator--(ulong3 &x, int n) {
  ulong3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* ulong4 unary post-decrement (--) */
inline ulong4 operator--(ulong4 &x, int n) {
  ulong4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* ulong8 unary post-decrement (--) */
inline ulong8 operator--(ulong8 &x, int n) {
  ulong8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* ulong16 unary post-decrement (--) */
inline ulong16 operator--(ulong16 &x, int n) {
  ulong16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* ulong2 unary pre-decrement (--) */
inline ulong2 operator--(ulong2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* ulong3 unary pre-decrement (--) */
inline ulong3 operator--(ulong3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* ulong4 unary pre-decrement (--) */
inline ulong4 operator--(ulong4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* ulong8 unary pre-decrement (--) */
inline ulong8 operator--(ulong8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* ulong16 unary pre-decrement (--) */
inline ulong16 operator--(ulong16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* ulong2 relational greater than (>) */
inline long2 operator>(const ulong2& x, const ulong2& y) {
  long2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline long2 operator>(const ulong2& x, ullong y) {
  long2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline long2 operator>(ullong x, const ulong2& y) {
  long2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* ulong3 relational greater than (>) */
inline long3 operator>(const ulong3& x, const ulong3& y) {
  long3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline long3 operator>(const ulong3& x, ullong y) {
  long3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline long3 operator>(ullong x, const ulong3& y) {
  long3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* ulong4 relational greater than (>) */
inline long4 operator>(const ulong4& x, const ulong4& y) {
  long4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline long4 operator>(const ulong4& x, ullong y) {
  long4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline long4 operator>(ullong x, const ulong4& y) {
  long4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* ulong8 relational greater than (>) */
inline long8 operator>(const ulong8& x, const ulong8& y) {
  long8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline long8 operator>(const ulong8& x, ullong y) {
  long8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline long8 operator>(ullong x, const ulong8& y) {
  long8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* ulong16 relational greater than (>) */
inline long16 operator>(const ulong16& x, const ulong16& y) {
  long16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline long16 operator>(const ulong16& x, ullong y) {
  long16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline long16 operator>(ullong x, const ulong16& y) {
  long16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* ulong2 relational less than (<) */
inline long2 operator<(const ulong2& x, const ulong2& y) {
  return y > x;
}
inline long2 operator<(const ulong2& x, ullong y) {
  return y > x;
}
inline long2 operator<(ullong x, const ulong2& y) {
  return y > x;
}

/* ulong3 relational less than (<) */
inline long3 operator<(const ulong3& x, const ulong3& y) {
  return y > x;
}
inline long3 operator<(const ulong3& x, ullong y) {
  return y > x;
}
inline long3 operator<(ullong x, const ulong3& y) {
  return y > x;
}

/* ulong4 relational less than (<) */
inline long4 operator<(const ulong4& x, const ulong4& y) {
  return y > x;
}
inline long4 operator<(const ulong4& x, ullong y) {
  return y > x;
}
inline long4 operator<(ullong x, const ulong4& y) {
  return y > x;
}

/* ulong8 relational less than (<) */
inline long8 operator<(const ulong8& x, const ulong8& y) {
  return y > x;
}
inline long8 operator<(const ulong8& x, ullong y) {
  return y > x;
}
inline long8 operator<(ullong x, const ulong8& y) {
  return y > x;
}

/* ulong16 relational less than (<) */
inline long16 operator<(const ulong16& x, const ulong16& y) {
  return y > x;
}
inline long16 operator<(const ulong16& x, ullong y) {
  return y > x;
}
inline long16 operator<(ullong x, const ulong16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* ulong2 relational greater than or equal (>=) */
inline long2 operator>=(const ulong2& x, const ulong2& y) {
  long2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline long2 operator>=(const ulong2& x, ullong y) {
  long2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline long2 operator>=(ullong x, const ulong2& y) {
  long2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* ulong3 relational greater than or equal (>=) */
inline long3 operator>=(const ulong3& x, const ulong3& y) {
  long3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline long3 operator>=(const ulong3& x, ullong y) {
  long3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline long3 operator>=(ullong x, const ulong3& y) {
  long3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* ulong4 relational greater than or equal (>=) */
inline long4 operator>=(const ulong4& x, const ulong4& y) {
  long4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline long4 operator>=(const ulong4& x, ullong y) {
  long4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline long4 operator>=(ullong x, const ulong4& y) {
  long4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* ulong8 relational greater than or equal (>=) */
inline long8 operator>=(const ulong8& x, const ulong8& y) {
  long8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline long8 operator>=(const ulong8& x, ullong y) {
  long8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline long8 operator>=(ullong x, const ulong8& y) {
  long8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* ulong16 relational greater than or equal (>=) */
inline long16 operator>=(const ulong16& x, const ulong16& y) {
  long16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline long16 operator>=(const ulong16& x, ullong y) {
  long16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline long16 operator>=(ullong x, const ulong16& y) {
  long16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* ulong2 relational less than or equal (<=) */
inline long2 operator<=(const ulong2& x, const ulong2& y) {
  return y >= x;
}
inline long2 operator<=(const ulong2& x, ullong y) {
  return y >= x;
}
inline long2 operator<=(ullong x, const ulong2& y) {
  return y >= x;
}

/* ulong3 relational less than or equal (<=) */
inline long3 operator<=(const ulong3& x, const ulong3& y) {
  return y >= x;
}
inline long3 operator<=(const ulong3& x, ullong y) {
  return y >= x;
}
inline long3 operator<=(ullong x, const ulong3& y) {
  return y >= x;
}

/* ulong4 relational less than or equal (<=) */
inline long4 operator<=(const ulong4& x, const ulong4& y) {
  return y >= x;
}
inline long4 operator<=(const ulong4& x, ullong y) {
  return y >= x;
}
inline long4 operator<=(ullong x, const ulong4& y) {
  return y >= x;
}

/* ulong8 relational less than or equal (<=) */
inline long8 operator<=(const ulong8& x, const ulong8& y) {
  return y >= x;
}
inline long8 operator<=(const ulong8& x, ullong y) {
  return y >= x;
}
inline long8 operator<=(ullong x, const ulong8& y) {
  return y >= x;
}

/* ulong16 relational less than or equal (<=) */
inline long16 operator<=(const ulong16& x, const ulong16& y) {
  return y >= x;
}
inline long16 operator<=(const ulong16& x, ullong y) {
  return y >= x;
}
inline long16 operator<=(ullong x, const ulong16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* ulong2 equal (==) */
inline long2 operator==(const ulong2& x, const ulong2& y) {
  long2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline long2 operator==(const ulong2& x, ullong y) {
  long2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline long2 operator==(ullong x, const ulong2& y) {
  long2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* ulong3 equal (==) */
inline long3 operator==(const ulong3& x, const ulong3& y) {
  long3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline long3 operator==(const ulong3& x, ullong y) {
  long3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline long3 operator==(ullong x, const ulong3& y) {
  long3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* ulong4 equal (==) */
inline long4 operator==(const ulong4& x, const ulong4& y) {
  long4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline long4 operator==(const ulong4& x, ullong y) {
  long4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline long4 operator==(ullong x, const ulong4& y) {
  long4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* ulong8 equal (==) */
inline long8 operator==(const ulong8& x, const ulong8& y) {
  long8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline long8 operator==(const ulong8& x, ullong y) {
  long8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline long8 operator==(ullong x, const ulong8& y) {
  long8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* ulong16 equal (==) */
inline long16 operator==(const ulong16& x, const ulong16& y) {
  long16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline long16 operator==(const ulong16& x, ullong y) {
  long16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline long16 operator==(ullong x, const ulong16& y) {
  long16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* ulong2 not equal (!=) */
inline long2 operator!=(const ulong2& x, const ulong2& y) {
  long2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline long2 operator!=(const ulong2& x, ullong y) {
  long2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline long2 operator!=(ullong x, const ulong2& y) {
  long2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* ulong3 not equal (!=) */
inline long3 operator!=(const ulong3& x, const ulong3& y) {
  long3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline long3 operator!=(const ulong3& x, ullong y) {
  long3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline long3 operator!=(ullong x, const ulong3& y) {
  long3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* ulong4 not equal (!=) */
inline long4 operator!=(const ulong4& x, const ulong4& y) {
  long4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline long4 operator!=(const ulong4& x, ullong y) {
  long4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline long4 operator!=(ullong x, const ulong4& y) {
  long4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* ulong8 not equal (!=) */
inline long8 operator!=(const ulong8& x, const ulong8& y) {
  long8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline long8 operator!=(const ulong8& x, ullong y) {
  long8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline long8 operator!=(ullong x, const ulong8& y) {
  long8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* ulong16 not equal (!=) */
inline long16 operator!=(const ulong16& x, const ulong16& y) {
  long16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline long16 operator!=(const ulong16& x, ullong y) {
  long16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline long16 operator!=(ullong x, const ulong16& y) {
  long16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* ulong2 bitwise and (&) */
inline ulong2 operator&(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline ulong2 operator&(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline ulong2 operator&(ullong x, const ulong2& y) {
  return y & x;
}

/* ulong3 bitwise and (&) */
inline ulong3 operator&(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline ulong3 operator&(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline ulong3 operator&(ullong x, const ulong3& y) {
  return y & x;
}


/* ulong4 bitwise and (&) */
inline ulong4 operator&(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline ulong4 operator&(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline ulong4 operator&(ullong x, const ulong4& y) {
  return y & x;
}

/* ulong8 bitwise and (&) */
inline ulong8 operator&(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline ulong8 operator&(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline ulong8 operator&(ullong x, const ulong8& y) {
  return y & x;
}

/* ulong16 bitwise and (&) */
inline ulong16 operator&(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline ulong16 operator&(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline ulong16 operator&(ullong x, const ulong16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* ulong2 bitwise or (|) */
inline ulong2 operator|(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline ulong2 operator|(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline ulong2 operator|(ullong x, const ulong2& y) {
  return y | x;
}

/* ulong3 bitwise or (|) */
inline ulong3 operator|(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline ulong3 operator|(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline ulong3 operator|(ullong x, const ulong3& y) {
  return y | x;
}

/* ulong4 bitwise or (|) */
inline ulong4 operator|(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline ulong4 operator|(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline ulong4 operator|(ullong x, const ulong4& y) {
  return y | x;
}

/* ulong8 bitwise or (|) */
inline ulong8 operator|(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline ulong8 operator|(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline ulong8 operator|(ullong x, const ulong8& y) {
  return y | x;
}

/* ulong16 bitwise or (|) */
inline ulong16 operator|(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline ulong16 operator|(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline ulong16 operator|(ullong x, const ulong16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* ulong2 bitwise exclusive or (^) */
inline ulong2 operator^(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline ulong2 operator^(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline ulong2 operator^(ullong x, const ulong2& y) {
  return y ^ x;
}

/* ulong3 bitwise exclusive or (^) */
inline ulong3 operator^(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline ulong3 operator^(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline ulong3 operator^(ullong x, const ulong3& y) {
  return y ^ x;
}

/* ulong4 bitwise exclusive or (^) */
inline ulong4 operator^(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline ulong4 operator^(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline ulong4 operator^(ullong x, const ulong4& y) {
  return y ^ x;
}

/* ulong8 bitwise exclusive or (^) */
inline ulong8 operator^(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline ulong8 operator^(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline ulong8 operator^(ullong x, const ulong8& y) {
  return y ^ x;
}

/* ulong16 bitwise exclusive or (^) */
inline ulong16 operator^(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline ulong16 operator^(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline ulong16 operator^(ullong x, const ulong16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* ulong2 bitwise not (~) */
inline ulong2 operator~(const ulong2& x) {
  ulong2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* ulong3 bitwise not (~) */
inline ulong3 operator~(const ulong3& x) {
  ulong3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* ulong4 bitwise not (~) */
inline ulong4 operator~(const ulong4& x) {
  ulong4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* ulong8 bitwise not (~) */
inline ulong8 operator~(const ulong8& x) {
  ulong8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* ulong16 bitwise not (~) */
inline ulong16 operator~(const ulong16& x) {
  ulong16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* ulong2 logical and (&&) */
inline long2 operator&&(const ulong2& x, const ulong2& y) {
  long2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline long2 operator&&(const ulong2& x, ullong y) {
  long2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline long2 operator&&(ullong x, const ulong2& y) {
  long2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* ulong3 logical and (&&) */
inline long3 operator&&(const ulong3& x, const ulong3& y) {
  long3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline long3 operator&&(const ulong3& x, ullong y) {
  long3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline long3 operator&&(ullong x, const ulong3& y) {
  long3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* ulong4 logical and (&&) */
inline long4 operator&&(const ulong4& x, const ulong4& y) {
  long4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline long4 operator&&(const ulong4& x, ullong y) {
  long4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline long4 operator&&(ullong x, const ulong4& y) {
  long4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* ulong8 logical and (&&) */
inline long8 operator&&(const ulong8& x, const ulong8& y) {
  long8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline long8 operator&&(const ulong8& x, ullong y) {
  long8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline long8 operator&&(ullong x, const ulong8& y) {
  long8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* ulong16 logical and (&&) */
inline long16 operator&&(const ulong16& x, const ulong16& y) {
  long16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline long16 operator&&(const ulong16& x, ullong y) {
  long16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline long16 operator&&(ullong x, const ulong16& y) {
  long16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* ulong2 logical or (||) */
inline long2 operator||(const ulong2& x, const ulong2& y) {
  long2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline long2 operator||(const ulong2& x, ullong y) {
  long2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline long2 operator||(ullong x, const ulong2& y) {
  long2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* ulong3 logical or (||) */
inline long3 operator||(const ulong3& x, const ulong3& y) {
  long3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline long3 operator||(const ulong3& x, ullong y) {
  long3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline long3 operator||(ullong x, const ulong3& y) {
  long3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* ulong4 logical or (||) */
inline long4 operator||(const ulong4& x, const ulong4& y) {
  long4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline long4 operator||(const ulong4& x, ullong y) {
  long4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline long4 operator||(ullong x, const ulong4& y) {
  long4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* ulong8 logical or (||) */
inline long8 operator||(const ulong8& x, const ulong8& y) {
  long8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline long8 operator||(const ulong8& x, ullong y) {
  long8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline long8 operator||(ullong x, const ulong8& y) {
  long8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* ulong16 logical or (||) */
inline long16 operator||(const ulong16& x, const ulong16& y) {
  long16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline long16 operator||(const ulong16& x, ullong y) {
  long16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline long16 operator||(ullong x, const ulong16& y) {
  long16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* ulong2 logical not (!) */
inline long2 operator!(const ulong2& x) {
  long2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* ulong3 logical not (!) */
inline long3 operator!(const ulong3& x) {
  long3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* ulong4 logical not (!) */
inline long4 operator!(const ulong4& x) {
  long4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* ulong8 logical not (!) */
inline long8 operator!(const ulong8& x) {
  long8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* ulong16 logical not (!) */
inline long16 operator!(const ulong16& x) {
  long16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* ulong2 right-shift (>>) */
inline ulong2 operator>>(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, >>, y);
  return rst;
}
inline ulong2 operator>>(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, >>, y);
  return rst;
}
/* ulong3 right-shift (>>) */
inline ulong3 operator>>(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, >>, y);
  return rst;
}
inline ulong3 operator>>(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, >>, y);
  return rst;
}

/* ulong4 right-shift (>>) */
inline ulong4 operator>>(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, >>, y);
  return rst;
}
inline ulong4 operator>>(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, >>, y);
  return rst;
}

/* ulong8 right-shift (>>) */
inline ulong8 operator>>(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, >>, y);
  return rst;
}
inline ulong8 operator>>(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, >>, y);
  return rst;
}

/* ulong16 right-shift (>>) */
inline ulong16 operator>>(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, >>, y);
  return rst;
}
inline ulong16 operator>>(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, >>, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* ulong2 left-shift (<<) */
inline ulong2 operator<<(const ulong2& x, const ulong2& y) {
  ulong2 rst = __CL_V2_OP_V2(x, <<, y);
  return rst;
}
inline ulong2 operator<<(const ulong2& x, ullong y) {
  ulong2 rst = __CL_V2_OP_S(x, <<, y);
  return rst;
}

/* ulong3 left-shift (<<) */
inline ulong3 operator<<(const ulong3& x, const ulong3& y) {
  ulong3 rst = __CL_V3_OP_V3(x, <<, y);
  return rst;
}
inline ulong3 operator<<(const ulong3& x, ullong y) {
  ulong3 rst = __CL_V3_OP_S(x, <<, y);
  return rst;
}

/* ulong4 left-shift (<<) */
inline ulong4 operator<<(const ulong4& x, const ulong4& y) {
  ulong4 rst = __CL_V4_OP_V4(x, <<, y);
  return rst;
}
inline ulong4 operator<<(const ulong4& x, ullong y) {
  ulong4 rst = __CL_V4_OP_S(x, <<, y);
  return rst;
}

/* ulong8 left-shift (<<) */
inline ulong8 operator<<(const ulong8& x, const ulong8& y) {
  ulong8 rst = __CL_V8_OP_V8(x, <<, y);
  return rst;
}
inline ulong8 operator<<(const ulong8& x, ullong y) {
  ulong8 rst = __CL_V8_OP_S(x, <<, y);
  return rst;
}

/* ulong16 left-shift (<<) */
inline ulong16 operator<<(const ulong16& x, const ulong16& y) {
  ulong16 rst = __CL_V16_OP_V16(x, <<, y);
  return rst;
}
inline ulong16 operator<<(const ulong16& x, ullong y) {
  ulong16 rst = __CL_V16_OP_S(x, <<, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* ulong2 assignment add into (+=) */
inline ulong2 operator+=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, +, y);
}
inline ulong2 operator+=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, +, y);
}

/* ulong3 assignment add into (+=) */
inline ulong3 operator+=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, +, y);
}
inline ulong3 operator+=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, +, y);
}

/* ulong4 assignment add into (+=) */
inline ulong4 operator+=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, +, y);
}
inline ulong4 operator+=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, +, y);
}

/* ulong8 assignment add into (+=) */
inline ulong8 operator+=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, +, y);
}
inline ulong8 operator+=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, +, y);
}

/* ulong16 assignment add into (+=) */
inline ulong16 operator+=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, +, y);
}
inline ulong16 operator+=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* ulong2 assignment subtract from (-=) */
inline ulong2 operator-=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, -, y);
}
inline ulong2 operator-=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, -, y);
}

/* ulong3 assignment subtract from (-=) */
inline ulong3 operator-=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, -, y);
}
inline ulong3 operator-=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, -, y);
}

/* ulong4 assignment subtract from (-=) */
inline ulong4 operator-=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, -, y);
}
inline ulong4 operator-=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, -, y);
}

/* ulong8 assignment subtract from (-=) */
inline ulong8 operator-=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, -, y);
}
inline ulong8 operator-=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, -, y);
}

/* ulong16 assignment subtract from (-=) */
inline ulong16 operator-=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, -, y);
}
inline ulong16 operator-=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* ulong2 assignment multiply into (*=) */
inline ulong2 operator*=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, *, y);
}
inline ulong2 operator*=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, *, y);
}

/* ulong3 assignment multiply into (*=) */
inline ulong3 operator*=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, *, y);
}
inline ulong3 operator*=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, *, y);
}

/* ulong4 assignment multiply into (*=) */
inline ulong4 operator*=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, *, y);
}
inline ulong4 operator*=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, *, y);
}

/* ulong8 assignment multiply into (*=) */
inline ulong8 operator*=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, *, y);
}
inline ulong8 operator*=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, *, y);
}

/* ulong16 assignment multiply into (*=) */
inline ulong16 operator*=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, *, y);
}
inline ulong16 operator*=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* ulong2 assignment divide into (/=) */
inline ulong2 operator/=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, /, y);
}
inline ulong2 operator/=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, /, y);
}

/* ulong3 assignment divide into (/=) */
inline ulong3 operator/=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, /, y);
}
inline ulong3 operator/=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, /, y);
}

/* ulong4 assignment divide into (/=) */
inline ulong4 operator/=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, /, y);
}
inline ulong4 operator/=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, /, y);
}

/* ulong8 assignment divide into (/=) */
inline ulong8 operator/=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, /, y);
}
inline ulong8 operator/=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, /, y);
}

/* ulong16 assignment divide into (/=) */
inline ulong16 operator/=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, /, y);
}
inline ulong16 operator/=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* ulong2 assignment modulus into (%=) */
inline ulong2 operator%=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, %, y);
}
inline ulong2 operator%=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, %, y);
}

/* ulong3 assignment modulus into (%=) */
inline ulong3 operator%=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, %, y);
}
inline ulong3 operator%=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, %, y);
}

/* ulong4 assignment modulus into (%=) */
inline ulong4 operator%=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, %, y);
}
inline ulong4 operator%=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, %, y);
}

/* ulong8 assignment modulus into (%=) */
inline ulong8 operator%=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, %, y);
}
inline ulong8 operator%=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, %, y);
}

/* ulong16 assignment modulus into (%=) */
inline ulong16 operator%=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, %, y);
}
inline ulong16 operator%=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* ulong2 assignment left shift by (<<=) */
inline ulong2 operator<<=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, <<, y);
}
inline ulong2 operator<<=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, <<, y);
}

/* ulong3 assignment left shift by (<<=) */
inline ulong3 operator<<=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, <<, y);
}
inline ulong3 operator<<=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, <<, y);
}


/* ulong4 assignment left shift by (<<=) */
inline ulong4 operator<<=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, <<, y);
}
inline ulong4 operator<<=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, <<, y);
}

/* ulong8 assignment left shift by (<<=) */
inline ulong8 operator<<=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, <<, y);
}
inline ulong8 operator<<=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, <<, y);
}

/* ulong16 assignment left shift by (<<=) */
inline ulong16 operator<<=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, <<, y);
}
inline ulong16 operator<<=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* ulong2 assignment right shift by (>>=) */
inline ulong2 operator>>=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, >>, y);
}
inline ulong2 operator>>=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, >>, y);
}

/* ulong3 assignment right shift by (>>=) */
inline ulong3 operator>>=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, >>, y);
}
inline ulong3 operator>>=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, >>, y);
}
/* ulong4 assignment right shift by (>>=) */
inline ulong4 operator>>=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, >>, y);
}
inline ulong4 operator>>=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, >>, y);
}

/* ulong8 assignment right shift by (>>=) */
inline ulong8 operator>>=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, >>, y);
}
inline ulong8 operator>>=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, >>, y);
}

/* ulong16 assignment right shift by (>>=) */
inline ulong16 operator>>=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, >>, y);
}
inline ulong16 operator>>=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* ulong2 assignment and into (&=) */
inline ulong2 operator&=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, &, y);
}
inline ulong2 operator&=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, &, y);
}

/* ulong3 assignment and into (&=) */
inline ulong3 operator&=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, &, y);
}
inline ulong3 operator&=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, &, y);
}

/* ulong4 assignment and into (&=) */
inline ulong4 operator&=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, &, y);
}
inline ulong4 operator&=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, &, y);
}

/* ulong8 assignment and into (&=) */
inline ulong8 operator&=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, &, y);
}
inline ulong8 operator&=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, &, y);
}

/* ulong16 assignment and into (&=) */
inline ulong16 operator&=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, &, y);
}
inline ulong16 operator&=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* ulong2 assignment inclusive or into (|=) */
inline ulong2 operator|=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, |, y);
}
inline ulong2 operator|=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, |, y);
}

/* ulong3 assignment inclusive or into (|=) */
inline ulong3 operator|=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, |, y);
}
inline ulong3 operator|=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, |, y);

}
/* ulong4 assignment inclusive or into (|=) */
inline ulong4 operator|=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, |, y);
}
inline ulong4 operator|=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, |, y);
}

/* ulong8 assignment inclusive or into (|=) */
inline ulong8 operator|=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, |, y);
}
inline ulong8 operator|=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, |, y);
}

/* ulong16 assignment inclusive or into (|=) */
inline ulong16 operator|=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, |, y);
}
inline ulong16 operator|=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* ulong2 assignment exclusive or into (^=) */
inline ulong2 operator^=(ulong2 &x, const ulong2& y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, ^, y);
}
inline ulong2 operator^=(ulong2 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong2, x, ^, y);
}

/* ulong3 assignment exclusive or into (^=) */
inline ulong3 operator^=(ulong3 &x, const ulong3& y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, ^, y);
}
inline ulong3 operator^=(ulong3 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong3, x, ^, y);
}

/* ulong4 assignment exclusive or into (^=) */
inline ulong4 operator^=(ulong4 &x, const ulong4& y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, ^, y);
}
inline ulong4 operator^=(ulong4 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong4, x, ^, y);
}

/* ulong8 assignment exclusive or into (^=) */
inline ulong8 operator^=(ulong8 &x, const ulong8& y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, ^, y);
}
inline ulong8 operator^=(ulong8 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong8, x, ^, y);
}

/* ulong16 assignment exclusive or into (^=) */
inline ulong16 operator^=(ulong16 &x, const ulong16& y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, ^, y);
}
inline ulong16 operator^=(ulong16 &x, ullong y) {
  __CL_V_OP_ASSIGN_V(ulong16, x, ^, y);
}

#endif //__CL_OPS_ULONGN_H

