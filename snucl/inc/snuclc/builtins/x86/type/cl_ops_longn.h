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

#ifndef __CL_OPS_LONGN_H
#define __CL_OPS_LONGN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>

///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* long2 addition (+) */
inline long2 operator+(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline long2 operator+(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline long2 operator+(llong x, const long2& y) {
  return y + x;
}

/* long3 addition (+) */
inline long3 operator+(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline long3 operator+(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline long3 operator+(llong x, const long3& y) {
  return y + x;
}

/* long4 addition (+) */
inline long4 operator+(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline long4 operator+(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline long4 operator+(llong x, const long4& y) {
  return y + x;
}

/* long8 addition (+) */
inline long8 operator+(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline long8 operator+(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline long8 operator+(llong x, const long8& y) {
  return y + x;
}

/* long16 addition (+) */
inline long16 operator+(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline long16 operator+(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline long16 operator+(llong x, const long16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* long2 subtraction (-) */
inline long2 operator-(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline long2 operator-(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline long2 operator-(llong x, const long2& y) {
  long2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* long3 subtraction (-) */
inline long3 operator-(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline long3 operator-(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline long3 operator-(llong x, const long3& y) {
  long3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* long4 subtraction (-) */
inline long4 operator-(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline long4 operator-(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline long4 operator-(llong x, const long4& y) {
  long4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* long8 subtraction (-) */
inline long8 operator-(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline long8 operator-(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline long8 operator-(llong x, const long8& y) {
  long8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* long16 subtraction (-) */
inline long16 operator-(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline long16 operator-(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline long16 operator-(llong x, const long16& y) {
  long16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* long2 multiplication (*) */
inline long2 operator*(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline long2 operator*(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline long2 operator*(llong x, const long2& y) {
  return y * x;
}

/* long3 multiplication (*) */
inline long3 operator*(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline long3 operator*(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline long3 operator*(llong x, const long3& y) {
  return y + x;
}


/* long4 multiplication (*) */
inline long4 operator*(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline long4 operator*(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline long4 operator*(llong x, const long4& y) {
  return y + x;
}

/* long8 multiplication (*) */
inline long8 operator*(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline long8 operator*(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline long8 operator*(llong x, const long8& y) {
  return y * x;
}

/* long16 multiplication (*) */
inline long16 operator*(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline long16 operator*(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline long16 operator*(llong x, const long16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* long2 division (/) */
inline long2 operator/(const long2& x, const long2& y) {
  long2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline long2 operator/(const long2& x, llong y) {
  long2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline long2 operator/(llong x, const long2& y) {
  long2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* long3 division (/) */
inline long3 operator/(const long3& x, const long3& y) {
  long3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline long3 operator/(const long3& x, llong y) {
  long3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline long3 operator/(llong x, const long3& y) {
  long3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* long4 division (/) */
inline long4 operator/(const long4& x, const long4& y) {
  long4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline long4 operator/(const long4& x, llong y) {
  long4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline long4 operator/(llong x, const long4& y) {
  long4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* long8 division (/) */
inline long8 operator/(const long8& x, const long8& y) {
  long8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline long8 operator/(const long8& x, llong y) {
  long8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline long8 operator/(llong x, const long8& y) {
  long8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* long16 division (/) */
inline long16 operator/(const long16& x, const long16& y) {
  long16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline long16 operator/(const long16& x, llong y) {
  long16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline long16 operator/(llong x, const long16& y) {
  long16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* long2 remainder (%) */
inline long2 operator%(const long2& x, const long2& y) {
  long2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline long2 operator%(const long2& x, llong y) {
  long2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline long2 operator%(llong x, const long2& y) {
  long2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* long3 remainder (%) */
inline long3 operator%(const long3& x, const long3& y) {
  long3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline long3 operator%(const long3& x, llong y) {
  long3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline long3 operator%(llong x, const long3& y) {
  long3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* long4 remainder (%) */
inline long4 operator%(const long4& x, const long4& y) {
  long4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline long4 operator%(const long4& x, llong y) {
  long4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline long4 operator%(llong x, const long4& y) {
  long4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* long8 remainder (%) */
inline long8 operator%(const long8& x, const long8& y) {
  long8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline long8 operator%(const long8& x, llong y) {
  long8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline long8 operator%(llong x, const long8& y) {
  long8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* long16 remainder (%) */
inline long16 operator%(const long16& x, const long16& y) {
  long16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline long16 operator%(const long16& x, llong y) {
  long16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline long16 operator%(llong x, const long16& y) {
  long16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* long2 unary positive (+) */
inline long2 operator+(const long2& x) {
  return x;
}
/* long3 unary positive (+) */
inline long3 operator+(const long3& x) {
  return x;
}

/* long4 unary positive (+) */
inline long4 operator+(const long4& x) {
  return x;
}

/* long8 unary positive (+) */
inline long8 operator+(const long8& x) {
  return x;
}

/* long16 unary positive (+) */
inline long16 operator+(const long16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* long2 unary negative (-) */
inline long2 operator-(const long2& x) {
  long2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* long3 unary negative (-) */
inline long3 operator-(const long3& x) {
  long3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* long4 unary negative (-) */
inline long4 operator-(const long4& x) {
  long4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* long8 unary negative (-) */
inline long8 operator-(const long8& x) {
  long8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* long16 unary negative (-) */
inline long16 operator-(const long16& x) {
  long16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* long2 unary post-increment (++) */
inline long2 operator++(long2 &x, int n) {
  long2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* long3 unary post-increment (++) */
inline long3 operator++(long3 &x, int n) {
  long3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* long4 unary post-increment (++) */
inline long4 operator++(long4 &x, int n) {
  long4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* long8 unary post-increment (++) */
inline long8 operator++(long8 &x, int n) {
  long8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* long16 unary post-increment (++) */
inline long16 operator++(long16 &x, int n) {
  long16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* long2 unary pre-increment (++) */
inline long2 operator++(long2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* long3 unary pre-increment (++) */
inline long3 operator++(long3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* long4 unary pre-increment (++) */
inline long4 operator++(long4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* long8 unary pre-increment (++) */
inline long8 operator++(long8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* long16 unary pre-increment (++) */
inline long16 operator++(long16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* long2 unary post-decrement (--) */
inline long2 operator--(long2 &x, int n) {
  long2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* long3 unary post-decrement (--) */
inline long3 operator--(long3 &x, int n) {
  long3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* long4 unary post-decrement (--) */
inline long4 operator--(long4 &x, int n) {
  long4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* long8 unary post-decrement (--) */
inline long8 operator--(long8 &x, int n) {
  long8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* long16 unary post-decrement (--) */
inline long16 operator--(long16 &x, int n) {
  long16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* long2 unary pre-decrement (--) */
inline long2 operator--(long2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* long3 unary pre-decrement (--) */
inline long3 operator--(long3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* long4 unary pre-decrement (--) */
inline long4 operator--(long4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* long8 unary pre-decrement (--) */
inline long8 operator--(long8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* long16 unary pre-decrement (--) */
inline long16 operator--(long16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* long2 relational greater than (>) */
inline long2 operator>(const long2& x, const long2& y) {
  long2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline long2 operator>(const long2& x, llong y) {
  long2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline long2 operator>(llong x, const long2& y) {
  long2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* long3 relational greater than (>) */
inline long3 operator>(const long3& x, const long3& y) {
  long3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline long3 operator>(const long3& x, llong y) {
  long3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline long3 operator>(llong x, const long3& y) {
  long3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* long4 relational greater than (>) */
inline long4 operator>(const long4& x, const long4& y) {
  long4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline long4 operator>(const long4& x, llong y) {
  long4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline long4 operator>(llong x, const long4& y) {
  long4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* long8 relational greater than (>) */
inline long8 operator>(const long8& x, const long8& y) {
  long8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline long8 operator>(const long8& x, llong y) {
  long8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline long8 operator>(llong x, const long8& y) {
  long8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* long16 relational greater than (>) */
inline long16 operator>(const long16& x, const long16& y) {
  long16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline long16 operator>(const long16& x, llong y) {
  long16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline long16 operator>(llong x, const long16& y) {
  long16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* long2 relational less than (<) */
inline long2 operator<(const long2& x, const long2& y) {
  return y > x;
}
inline long2 operator<(const long2& x, llong y) {
  return y > x;
}
inline long2 operator<(llong x, const long2& y) {
  return y > x;
}

/* long3 relational less than (<) */
inline long3 operator<(const long3& x, const long3& y) {
  return y > x;
}
inline long3 operator<(const long3& x, llong y) {
  return y > x;
}
inline long3 operator<(llong x, const long3& y) {
  return y > x;
}

/* long4 relational less than (<) */
inline long4 operator<(const long4& x, const long4& y) {
  return y > x;
}
inline long4 operator<(const long4& x, llong y) {
  return y > x;
}
inline long4 operator<(llong x, const long4& y) {
  return y > x;
}

/* long8 relational less than (<) */
inline long8 operator<(const long8& x, const long8& y) {
  return y > x;
}
inline long8 operator<(const long8& x, llong y) {
  return y > x;
}
inline long8 operator<(llong x, const long8& y) {
  return y > x;
}

/* long16 relational less than (<) */
inline long16 operator<(const long16& x, const long16& y) {
  return y > x;
}
inline long16 operator<(const long16& x, llong y) {
  return y > x;
}
inline long16 operator<(llong x, const long16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* long2 relational greater than or equal (>=) */
inline long2 operator>=(const long2& x, const long2& y) {
  long2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline long2 operator>=(const long2& x, llong y) {
  long2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline long2 operator>=(llong x, const long2& y) {
  long2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* long3 relational greater than or equal (>=) */
inline long3 operator>=(const long3& x, const long3& y) {
  long3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline long3 operator>=(const long3& x, llong y) {
  long3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline long3 operator>=(llong x, const long3& y) {
  long3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* long4 relational greater than or equal (>=) */
inline long4 operator>=(const long4& x, const long4& y) {
  long4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline long4 operator>=(const long4& x, llong y) {
  long4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline long4 operator>=(llong x, const long4& y) {
  long4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* long8 relational greater than or equal (>=) */
inline long8 operator>=(const long8& x, const long8& y) {
  long8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline long8 operator>=(const long8& x, llong y) {
  long8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline long8 operator>=(llong x, const long8& y) {
  long8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* long16 relational greater than or equal (>=) */
inline long16 operator>=(const long16& x, const long16& y) {
  long16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline long16 operator>=(const long16& x, llong y) {
  long16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline long16 operator>=(llong x, const long16& y) {
  long16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* long2 relational less than or equal (<=) */
inline long2 operator<=(const long2& x, const long2& y) {
  return y >= x;
}
inline long2 operator<=(const long2& x, llong y) {
  return y >= x;
}
inline long2 operator<=(llong x, const long2& y) {
  return y >= x;
}

/* long3 relational less than or equal (<=) */
inline long3 operator<=(const long3& x, const long3& y) {
  return y >= x;
}
inline long3 operator<=(const long3& x, llong y) {
  return y >= x;
}
inline long3 operator<=(llong x, const long3& y) {
  return y >= x;
}

/* long4 relational less than or equal (<=) */
inline long4 operator<=(const long4& x, const long4& y) {
  return y >= x;
}
inline long4 operator<=(const long4& x, llong y) {
  return y >= x;
}
inline long4 operator<=(llong x, const long4& y) {
  return y >= x;
}

/* long8 relational less than or equal (<=) */
inline long8 operator<=(const long8& x, const long8& y) {
  return y >= x;
}
inline long8 operator<=(const long8& x, llong y) {
  return y >= x;
}
inline long8 operator<=(llong x, const long8& y) {
  return y >= x;
}

/* long16 relational less than or equal (<=) */
inline long16 operator<=(const long16& x, const long16& y) {
  return y >= x;
}
inline long16 operator<=(const long16& x, llong y) {
  return y >= x;
}
inline long16 operator<=(llong x, const long16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* long2 equal (==) */
inline long2 operator==(const long2& x, const long2& y) {
  long2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline long2 operator==(const long2& x, llong y) {
  long2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline long2 operator==(llong x, const long2& y) {
  long2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* long3 equal (==) */
inline long3 operator==(const long3& x, const long3& y) {
  long3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline long3 operator==(const long3& x, llong y) {
  long3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline long3 operator==(llong x, const long3& y) {
  long3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* long4 equal (==) */
inline long4 operator==(const long4& x, const long4& y) {
  long4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline long4 operator==(const long4& x, llong y) {
  long4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline long4 operator==(llong x, const long4& y) {
  long4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* long8 equal (==) */
inline long8 operator==(const long8& x, const long8& y) {
  long8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline long8 operator==(const long8& x, llong y) {
  long8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline long8 operator==(llong x, const long8& y) {
  long8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* long16 equal (==) */
inline long16 operator==(const long16& x, const long16& y) {
  long16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline long16 operator==(const long16& x, llong y) {
  long16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline long16 operator==(llong x, const long16& y) {
  long16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* long2 not equal (!=) */
inline long2 operator!=(const long2& x, const long2& y) {
  long2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline long2 operator!=(const long2& x, llong y) {
  long2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline long2 operator!=(llong x, const long2& y) {
  long2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* long3 not equal (!=) */
inline long3 operator!=(const long3& x, const long3& y) {
  long3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline long3 operator!=(const long3& x, llong y) {
  long3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline long3 operator!=(llong x, const long3& y) {
  long3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* long4 not equal (!=) */
inline long4 operator!=(const long4& x, const long4& y) {
  long4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline long4 operator!=(const long4& x, llong y) {
  long4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline long4 operator!=(llong x, const long4& y) {
  long4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* long8 not equal (!=) */
inline long8 operator!=(const long8& x, const long8& y) {
  long8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline long8 operator!=(const long8& x, llong y) {
  long8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline long8 operator!=(llong x, const long8& y) {
  long8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* long16 not equal (!=) */
inline long16 operator!=(const long16& x, const long16& y) {
  long16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline long16 operator!=(const long16& x, llong y) {
  long16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline long16 operator!=(llong x, const long16& y) {
  long16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* long2 bitwise and (&) */
inline long2 operator&(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline long2 operator&(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline long2 operator&(llong x, const long2& y) {
  return y & x;
}

/* long3 bitwise and (&) */
inline long3 operator&(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline long3 operator&(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline long3 operator&(llong x, const long3& y) {
  return y & x;
}


/* long4 bitwise and (&) */
inline long4 operator&(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline long4 operator&(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline long4 operator&(llong x, const long4& y) {
  return y & x;
}

/* long8 bitwise and (&) */
inline long8 operator&(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline long8 operator&(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline long8 operator&(llong x, const long8& y) {
  return y & x;
}

/* long16 bitwise and (&) */
inline long16 operator&(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline long16 operator&(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline long16 operator&(llong x, const long16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* long2 bitwise or (|) */
inline long2 operator|(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline long2 operator|(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline long2 operator|(llong x, const long2& y) {
  return y | x;
}

/* long3 bitwise or (|) */
inline long3 operator|(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline long3 operator|(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline long3 operator|(llong x, const long3& y) {
  return y | x;
}

/* long4 bitwise or (|) */
inline long4 operator|(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline long4 operator|(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline long4 operator|(llong x, const long4& y) {
  return y | x;
}

/* long8 bitwise or (|) */
inline long8 operator|(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline long8 operator|(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline long8 operator|(llong x, const long8& y) {
  return y | x;
}

/* long16 bitwise or (|) */
inline long16 operator|(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline long16 operator|(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline long16 operator|(llong x, const long16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* long2 bitwise exclusive or (^) */
inline long2 operator^(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline long2 operator^(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline long2 operator^(llong x, const long2& y) {
  return y ^ x;
}

/* long3 bitwise exclusive or (^) */
inline long3 operator^(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline long3 operator^(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline long3 operator^(llong x, const long3& y) {
  return y ^ x;
}

/* long4 bitwise exclusive or (^) */
inline long4 operator^(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline long4 operator^(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline long4 operator^(llong x, const long4& y) {
  return y ^ x;
}

/* long8 bitwise exclusive or (^) */
inline long8 operator^(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline long8 operator^(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline long8 operator^(llong x, const long8& y) {
  return y ^ x;
}

/* long16 bitwise exclusive or (^) */
inline long16 operator^(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline long16 operator^(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline long16 operator^(llong x, const long16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* long2 bitwise not (~) */
inline long2 operator~(const long2& x) {
  long2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* long3 bitwise not (~) */
inline long3 operator~(const long3& x) {
  long3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* long4 bitwise not (~) */
inline long4 operator~(const long4& x) {
  long4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* long8 bitwise not (~) */
inline long8 operator~(const long8& x) {
  long8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* long16 bitwise not (~) */
inline long16 operator~(const long16& x) {
  long16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* long2 logical and (&&) */
inline long2 operator&&(const long2& x, const long2& y) {
  long2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline long2 operator&&(const long2& x, llong y) {
  long2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline long2 operator&&(llong x, const long2& y) {
  long2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* long3 logical and (&&) */
inline long3 operator&&(const long3& x, const long3& y) {
  long3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline long3 operator&&(const long3& x, llong y) {
  long3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline long3 operator&&(llong x, const long3& y) {
  long3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* long4 logical and (&&) */
inline long4 operator&&(const long4& x, const long4& y) {
  long4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline long4 operator&&(const long4& x, llong y) {
  long4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline long4 operator&&(llong x, const long4& y) {
  long4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* long8 logical and (&&) */
inline long8 operator&&(const long8& x, const long8& y) {
  long8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline long8 operator&&(const long8& x, llong y) {
  long8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline long8 operator&&(llong x, const long8& y) {
  long8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* long16 logical and (&&) */
inline long16 operator&&(const long16& x, const long16& y) {
  long16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline long16 operator&&(const long16& x, llong y) {
  long16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline long16 operator&&(llong x, const long16& y) {
  long16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* long2 logical or (||) */
inline long2 operator||(const long2& x, const long2& y) {
  long2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline long2 operator||(const long2& x, llong y) {
  long2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline long2 operator||(llong x, const long2& y) {
  long2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* long3 logical or (||) */
inline long3 operator||(const long3& x, const long3& y) {
  long3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline long3 operator||(const long3& x, llong y) {
  long3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline long3 operator||(llong x, const long3& y) {
  long3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* long4 logical or (||) */
inline long4 operator||(const long4& x, const long4& y) {
  long4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline long4 operator||(const long4& x, llong y) {
  long4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline long4 operator||(llong x, const long4& y) {
  long4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* long8 logical or (||) */
inline long8 operator||(const long8& x, const long8& y) {
  long8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline long8 operator||(const long8& x, llong y) {
  long8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline long8 operator||(llong x, const long8& y) {
  long8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* long16 logical or (||) */
inline long16 operator||(const long16& x, const long16& y) {
  long16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline long16 operator||(const long16& x, llong y) {
  long16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline long16 operator||(llong x, const long16& y) {
  long16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* long2 logical not (!) */
inline long2 operator!(const long2& x) {
  long2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* long3 logical not (!) */
inline long3 operator!(const long3& x) {
  long3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* long4 logical not (!) */
inline long4 operator!(const long4& x) {
  long4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* long8 logical not (!) */
inline long8 operator!(const long8& x) {
  long8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* long16 logical not (!) */
inline long16 operator!(const long16& x) {
  long16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* long2 right-shift (>>) */
inline long2 operator>>(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, >>, y);
  return rst;
}
inline long2 operator>>(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, >>, y);
  return rst;
}
/* long3 right-shift (>>) */
inline long3 operator>>(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, >>, y);
  return rst;
}
inline long3 operator>>(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, >>, y);
  return rst;
}

/* long4 right-shift (>>) */
inline long4 operator>>(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, >>, y);
  return rst;
}
inline long4 operator>>(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, >>, y);
  return rst;
}

/* long8 right-shift (>>) */
inline long8 operator>>(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, >>, y);
  return rst;
}
inline long8 operator>>(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, >>, y);
  return rst;
}

/* long16 right-shift (>>) */
inline long16 operator>>(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, >>, y);
  return rst;
}
inline long16 operator>>(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, >>, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* long2 left-shift (<<) */
inline long2 operator<<(const long2& x, const long2& y) {
  long2 rst = __CL_V2_OP_V2(x, <<, y);
  return rst;
}
inline long2 operator<<(const long2& x, llong y) {
  long2 rst = __CL_V2_OP_S(x, <<, y);
  return rst;
}

/* long3 left-shift (<<) */
inline long3 operator<<(const long3& x, const long3& y) {
  long3 rst = __CL_V3_OP_V3(x, <<, y);
  return rst;
}
inline long3 operator<<(const long3& x, llong y) {
  long3 rst = __CL_V3_OP_S(x, <<, y);
  return rst;
}

/* long4 left-shift (<<) */
inline long4 operator<<(const long4& x, const long4& y) {
  long4 rst = __CL_V4_OP_V4(x, <<, y);
  return rst;
}
inline long4 operator<<(const long4& x, llong y) {
  long4 rst = __CL_V4_OP_S(x, <<, y);
  return rst;
}

/* long8 left-shift (<<) */
inline long8 operator<<(const long8& x, const long8& y) {
  long8 rst = __CL_V8_OP_V8(x, <<, y);
  return rst;
}
inline long8 operator<<(const long8& x, llong y) {
  long8 rst = __CL_V8_OP_S(x, <<, y);
  return rst;
}

/* long16 left-shift (<<) */
inline long16 operator<<(const long16& x, const long16& y) {
  long16 rst = __CL_V16_OP_V16(x, <<, y);
  return rst;
}
inline long16 operator<<(const long16& x, llong y) {
  long16 rst = __CL_V16_OP_S(x, <<, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* long2 assignment add into (+=) */
inline long2 operator+=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, +, y);
}
inline long2 operator+=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, +, y);
}

/* long3 assignment add into (+=) */
inline long3 operator+=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, +, y);
}
inline long3 operator+=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, +, y);
}

/* long4 assignment add into (+=) */
inline long4 operator+=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, +, y);
}
inline long4 operator+=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, +, y);
}

/* long8 assignment add into (+=) */
inline long8 operator+=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, +, y);
}
inline long8 operator+=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, +, y);
}

/* long16 assignment add into (+=) */
inline long16 operator+=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, +, y);
}
inline long16 operator+=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* long2 assignment subtract from (-=) */
inline long2 operator-=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, -, y);
}
inline long2 operator-=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, -, y);
}

/* long3 assignment subtract from (-=) */
inline long3 operator-=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, -, y);
}
inline long3 operator-=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, -, y);
}

/* long4 assignment subtract from (-=) */
inline long4 operator-=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, -, y);
}
inline long4 operator-=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, -, y);
}

/* long8 assignment subtract from (-=) */
inline long8 operator-=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, -, y);
}
inline long8 operator-=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, -, y);
}

/* long16 assignment subtract from (-=) */
inline long16 operator-=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, -, y);
}
inline long16 operator-=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* long2 assignment multiply into (*=) */
inline long2 operator*=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, *, y);
}
inline long2 operator*=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, *, y);
}

/* long3 assignment multiply into (*=) */
inline long3 operator*=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, *, y);
}
inline long3 operator*=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, *, y);
}

/* long4 assignment multiply into (*=) */
inline long4 operator*=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, *, y);
}
inline long4 operator*=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, *, y);
}

/* long8 assignment multiply into (*=) */
inline long8 operator*=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, *, y);
}
inline long8 operator*=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, *, y);
}

/* long16 assignment multiply into (*=) */
inline long16 operator*=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, *, y);
}
inline long16 operator*=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* long2 assignment divide into (/=) */
inline long2 operator/=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, /, y);
}
inline long2 operator/=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, /, y);
}

/* long3 assignment divide into (/=) */
inline long3 operator/=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, /, y);
}
inline long3 operator/=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, /, y);
}

/* long4 assignment divide into (/=) */
inline long4 operator/=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, /, y);
}
inline long4 operator/=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, /, y);
}

/* long8 assignment divide into (/=) */
inline long8 operator/=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, /, y);
}
inline long8 operator/=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, /, y);
}

/* long16 assignment divide into (/=) */
inline long16 operator/=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, /, y);
}
inline long16 operator/=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* long2 assignment modulus into (%=) */
inline long2 operator%=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, %, y);
}
inline long2 operator%=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, %, y);
}

/* long3 assignment modulus into (%=) */
inline long3 operator%=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, %, y);
}
inline long3 operator%=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, %, y);
}

/* long4 assignment modulus into (%=) */
inline long4 operator%=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, %, y);
}
inline long4 operator%=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, %, y);
}

/* long8 assignment modulus into (%=) */
inline long8 operator%=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, %, y);
}
inline long8 operator%=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, %, y);
}

/* long16 assignment modulus into (%=) */
inline long16 operator%=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, %, y);
}
inline long16 operator%=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* long2 assignment left shift by (<<=) */
inline long2 operator<<=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, <<, y);
}
inline long2 operator<<=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, <<, y);
}

/* long3 assignment left shift by (<<=) */
inline long3 operator<<=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, <<, y);
}
inline long3 operator<<=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, <<, y);
}


/* long4 assignment left shift by (<<=) */
inline long4 operator<<=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, <<, y);
}
inline long4 operator<<=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, <<, y);
}

/* long8 assignment left shift by (<<=) */
inline long8 operator<<=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, <<, y);
}
inline long8 operator<<=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, <<, y);
}

/* long16 assignment left shift by (<<=) */
inline long16 operator<<=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, <<, y);
}
inline long16 operator<<=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* long2 assignment right shift by (>>=) */
inline long2 operator>>=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, >>, y);
}
inline long2 operator>>=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, >>, y);
}

/* long3 assignment right shift by (>>=) */
inline long3 operator>>=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, >>, y);
}
inline long3 operator>>=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, >>, y);
}
/* long4 assignment right shift by (>>=) */
inline long4 operator>>=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, >>, y);
}
inline long4 operator>>=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, >>, y);
}

/* long8 assignment right shift by (>>=) */
inline long8 operator>>=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, >>, y);
}
inline long8 operator>>=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, >>, y);
}

/* long16 assignment right shift by (>>=) */
inline long16 operator>>=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, >>, y);
}
inline long16 operator>>=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* long2 assignment and into (&=) */
inline long2 operator&=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, &, y);
}
inline long2 operator&=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, &, y);
}

/* long3 assignment and into (&=) */
inline long3 operator&=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, &, y);
}
inline long3 operator&=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, &, y);
}

/* long4 assignment and into (&=) */
inline long4 operator&=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, &, y);
}
inline long4 operator&=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, &, y);
}

/* long8 assignment and into (&=) */
inline long8 operator&=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, &, y);
}
inline long8 operator&=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, &, y);
}

/* long16 assignment and into (&=) */
inline long16 operator&=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, &, y);
}
inline long16 operator&=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* long2 assignment inclusive or into (|=) */
inline long2 operator|=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, |, y);
}
inline long2 operator|=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, |, y);
}

/* long3 assignment inclusive or into (|=) */
inline long3 operator|=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, |, y);
}
inline long3 operator|=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, |, y);

}
/* long4 assignment inclusive or into (|=) */
inline long4 operator|=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, |, y);
}
inline long4 operator|=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, |, y);
}

/* long8 assignment inclusive or into (|=) */
inline long8 operator|=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, |, y);
}
inline long8 operator|=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, |, y);
}

/* long16 assignment inclusive or into (|=) */
inline long16 operator|=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, |, y);
}
inline long16 operator|=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* long2 assignment exclusive or into (^=) */
inline long2 operator^=(long2 &x, const long2& y) {
  __CL_V_OP_ASSIGN_V(long2, x, ^, y);
}
inline long2 operator^=(long2 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long2, x, ^, y);
}

/* long3 assignment exclusive or into (^=) */
inline long3 operator^=(long3 &x, const long3& y) {
  __CL_V_OP_ASSIGN_V(long3, x, ^, y);
}
inline long3 operator^=(long3 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long3, x, ^, y);
}

/* long4 assignment exclusive or into (^=) */
inline long4 operator^=(long4 &x, const long4& y) {
  __CL_V_OP_ASSIGN_V(long4, x, ^, y);
}
inline long4 operator^=(long4 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long4, x, ^, y);
}

/* long8 assignment exclusive or into (^=) */
inline long8 operator^=(long8 &x, const long8& y) {
  __CL_V_OP_ASSIGN_V(long8, x, ^, y);
}
inline long8 operator^=(long8 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long8, x, ^, y);
}

/* long16 assignment exclusive or into (^=) */
inline long16 operator^=(long16 &x, const long16& y) {
  __CL_V_OP_ASSIGN_V(long16, x, ^, y);
}
inline long16 operator^=(long16 &x, llong y) {
  __CL_V_OP_ASSIGN_V(long16, x, ^, y);
}


#endif //__CL_OPS_LONGN_H

