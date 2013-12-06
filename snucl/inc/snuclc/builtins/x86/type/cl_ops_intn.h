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

#ifndef __CL_OPS_INTN_H
#define __CL_OPS_INTN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>

///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* int2 addition (+) */
inline int2 operator+(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline int2 operator+(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline int2 operator+(int x, const int2& y) {
  return y + x;
}

/* int3 addition (+) */
inline int3 operator+(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline int3 operator+(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline int3 operator+(int x, const int3& y) {
  return y + x;
}

/* int4 addition (+) */
inline int4 operator+(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline int4 operator+(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline int4 operator+(int x, const int4& y) {
  return y + x;
}

/* int8 addition (+) */
inline int8 operator+(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline int8 operator+(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline int8 operator+(int x, const int8& y) {
  return y + x;
}

/* int16 addition (+) */
inline int16 operator+(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline int16 operator+(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline int16 operator+(int x, const int16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* int2 subtraction (-) */
inline int2 operator-(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline int2 operator-(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline int2 operator-(int x, const int2& y) {
  int2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* int3 subtraction (-) */
inline int3 operator-(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline int3 operator-(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline int3 operator-(int x, const int3& y) {
  int3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* int4 subtraction (-) */
inline int4 operator-(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline int4 operator-(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline int4 operator-(int x, const int4& y) {
  int4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* int8 subtraction (-) */
inline int8 operator-(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline int8 operator-(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline int8 operator-(int x, const int8& y) {
  int8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* int16 subtraction (-) */
inline int16 operator-(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline int16 operator-(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline int16 operator-(int x, const int16& y) {
  int16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* int2 multiplication (*) */
inline int2 operator*(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline int2 operator*(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline int2 operator*(int x, const int2& y) {
  return y * x;
}

/* int3 multiplication (*) */
inline int3 operator*(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline int3 operator*(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline int3 operator*(int x, const int3& y) {
  return y + x;
}


/* int4 multiplication (*) */
inline int4 operator*(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline int4 operator*(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline int4 operator*(int x, const int4& y) {
  return y + x;
}

/* int8 multiplication (*) */
inline int8 operator*(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline int8 operator*(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline int8 operator*(int x, const int8& y) {
  return y * x;
}

/* int16 multiplication (*) */
inline int16 operator*(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline int16 operator*(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline int16 operator*(int x, const int16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* int2 division (/) */
inline int2 operator/(const int2& x, const int2& y) {
  int2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline int2 operator/(const int2& x, int y) {
  int2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline int2 operator/(int x, const int2& y) {
  int2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* int3 division (/) */
inline int3 operator/(const int3& x, const int3& y) {
  int3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline int3 operator/(const int3& x, int y) {
  int3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline int3 operator/(int x, const int3& y) {
  int3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* int4 division (/) */
inline int4 operator/(const int4& x, const int4& y) {
  int4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline int4 operator/(const int4& x, int y) {
  int4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline int4 operator/(int x, const int4& y) {
  int4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* int8 division (/) */
inline int8 operator/(const int8& x, const int8& y) {
  int8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline int8 operator/(const int8& x, int y) {
  int8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline int8 operator/(int x, const int8& y) {
  int8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* int16 division (/) */
inline int16 operator/(const int16& x, const int16& y) {
  int16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline int16 operator/(const int16& x, int y) {
  int16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline int16 operator/(int x, const int16& y) {
  int16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* int2 remainder (%) */
inline int2 operator%(const int2& x, const int2& y) {
  int2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline int2 operator%(const int2& x, int y) {
  int2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline int2 operator%(int x, const int2& y) {
  int2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* int3 remainder (%) */
inline int3 operator%(const int3& x, const int3& y) {
  int3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline int3 operator%(const int3& x, int y) {
  int3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline int3 operator%(int x, const int3& y) {
  int3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* int4 remainder (%) */
inline int4 operator%(const int4& x, const int4& y) {
  int4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline int4 operator%(const int4& x, int y) {
  int4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline int4 operator%(int x, const int4& y) {
  int4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* int8 remainder (%) */
inline int8 operator%(const int8& x, const int8& y) {
  int8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline int8 operator%(const int8& x, int y) {
  int8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline int8 operator%(int x, const int8& y) {
  int8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* int16 remainder (%) */
inline int16 operator%(const int16& x, const int16& y) {
  int16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline int16 operator%(const int16& x, int y) {
  int16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline int16 operator%(int x, const int16& y) {
  int16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* int2 unary positive (+) */
inline int2 operator+(const int2& x) {
  return x;
}
/* int3 unary positive (+) */
inline int3 operator+(const int3& x) {
  return x;
}

/* int4 unary positive (+) */
inline int4 operator+(const int4& x) {
  return x;
}

/* int8 unary positive (+) */
inline int8 operator+(const int8& x) {
  return x;
}

/* int16 unary positive (+) */
inline int16 operator+(const int16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* int2 unary negative (-) */
inline int2 operator-(const int2& x) {
  int2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* int3 unary negative (-) */
inline int3 operator-(const int3& x) {
  int3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* int4 unary negative (-) */
inline int4 operator-(const int4& x) {
  int4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* int8 unary negative (-) */
inline int8 operator-(const int8& x) {
  int8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* int16 unary negative (-) */
inline int16 operator-(const int16& x) {
  int16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* int2 unary post-increment (++) */
inline int2 operator++(int2 &x, int n) {
  int2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* int3 unary post-increment (++) */
inline int3 operator++(int3 &x, int n) {
  int3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* int4 unary post-increment (++) */
inline int4 operator++(int4 &x, int n) {
  int4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* int8 unary post-increment (++) */
inline int8 operator++(int8 &x, int n) {
  int8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* int16 unary post-increment (++) */
inline int16 operator++(int16 &x, int n) {
  int16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* int2 unary pre-increment (++) */
inline int2 operator++(int2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* int3 unary pre-increment (++) */
inline int3 operator++(int3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* int4 unary pre-increment (++) */
inline int4 operator++(int4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* int8 unary pre-increment (++) */
inline int8 operator++(int8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* int16 unary pre-increment (++) */
inline int16 operator++(int16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* int2 unary post-decrement (--) */
inline int2 operator--(int2 &x, int n) {
  int2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* int3 unary post-decrement (--) */
inline int3 operator--(int3 &x, int n) {
  int3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* int4 unary post-decrement (--) */
inline int4 operator--(int4 &x, int n) {
  int4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* int8 unary post-decrement (--) */
inline int8 operator--(int8 &x, int n) {
  int8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* int16 unary post-decrement (--) */
inline int16 operator--(int16 &x, int n) {
  int16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* int2 unary pre-decrement (--) */
inline int2 operator--(int2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* int3 unary pre-decrement (--) */
inline int3 operator--(int3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* int4 unary pre-decrement (--) */
inline int4 operator--(int4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* int8 unary pre-decrement (--) */
inline int8 operator--(int8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* int16 unary pre-decrement (--) */
inline int16 operator--(int16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* int2 relational greater than (>) */
inline int2 operator>(const int2& x, const int2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline int2 operator>(const int2& x, int y) {
  int2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline int2 operator>(int x, const int2& y) {
  int2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* int3 relational greater than (>) */
inline int3 operator>(const int3& x, const int3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline int3 operator>(const int3& x, int y) {
  int3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline int3 operator>(int x, const int3& y) {
  int3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* int4 relational greater than (>) */
inline int4 operator>(const int4& x, const int4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline int4 operator>(const int4& x, int y) {
  int4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline int4 operator>(int x, const int4& y) {
  int4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* int8 relational greater than (>) */
inline int8 operator>(const int8& x, const int8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline int8 operator>(const int8& x, int y) {
  int8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline int8 operator>(int x, const int8& y) {
  int8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* int16 relational greater than (>) */
inline int16 operator>(const int16& x, const int16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline int16 operator>(const int16& x, int y) {
  int16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline int16 operator>(int x, const int16& y) {
  int16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* int2 relational less than (<) */
inline int2 operator<(const int2& x, const int2& y) {
  return y > x;
}
inline int2 operator<(const int2& x, int y) {
  return y > x;
}
inline int2 operator<(int x, const int2& y) {
  return y > x;
}

/* int3 relational less than (<) */
inline int3 operator<(const int3& x, const int3& y) {
  return y > x;
}
inline int3 operator<(const int3& x, int y) {
  return y > x;
}
inline int3 operator<(int x, const int3& y) {
  return y > x;
}

/* int4 relational less than (<) */
inline int4 operator<(const int4& x, const int4& y) {
  return y > x;
}
inline int4 operator<(const int4& x, int y) {
  return y > x;
}
inline int4 operator<(int x, const int4& y) {
  return y > x;
}

/* int8 relational less than (<) */
inline int8 operator<(const int8& x, const int8& y) {
  return y > x;
}
inline int8 operator<(const int8& x, int y) {
  return y > x;
}
inline int8 operator<(int x, const int8& y) {
  return y > x;
}

/* int16 relational less than (<) */
inline int16 operator<(const int16& x, const int16& y) {
  return y > x;
}
inline int16 operator<(const int16& x, int y) {
  return y > x;
}
inline int16 operator<(int x, const int16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* int2 relational greater than or equal (>=) */
inline int2 operator>=(const int2& x, const int2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline int2 operator>=(const int2& x, int y) {
  int2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline int2 operator>=(int x, const int2& y) {
  int2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* int3 relational greater than or equal (>=) */
inline int3 operator>=(const int3& x, const int3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline int3 operator>=(const int3& x, int y) {
  int3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline int3 operator>=(int x, const int3& y) {
  int3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* int4 relational greater than or equal (>=) */
inline int4 operator>=(const int4& x, const int4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline int4 operator>=(const int4& x, int y) {
  int4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline int4 operator>=(int x, const int4& y) {
  int4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* int8 relational greater than or equal (>=) */
inline int8 operator>=(const int8& x, const int8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline int8 operator>=(const int8& x, int y) {
  int8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline int8 operator>=(int x, const int8& y) {
  int8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* int16 relational greater than or equal (>=) */
inline int16 operator>=(const int16& x, const int16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline int16 operator>=(const int16& x, int y) {
  int16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline int16 operator>=(int x, const int16& y) {
  int16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* int2 relational less than or equal (<=) */
inline int2 operator<=(const int2& x, const int2& y) {
  return y >= x;
}
inline int2 operator<=(const int2& x, int y) {
  return y >= x;
}
inline int2 operator<=(int x, const int2& y) {
  return y >= x;
}

/* int3 relational less than or equal (<=) */
inline int3 operator<=(const int3& x, const int3& y) {
  return y >= x;
}
inline int3 operator<=(const int3& x, int y) {
  return y >= x;
}
inline int3 operator<=(int x, const int3& y) {
  return y >= x;
}

/* int4 relational less than or equal (<=) */
inline int4 operator<=(const int4& x, const int4& y) {
  return y >= x;
}
inline int4 operator<=(const int4& x, int y) {
  return y >= x;
}
inline int4 operator<=(int x, const int4& y) {
  return y >= x;
}

/* int8 relational less than or equal (<=) */
inline int8 operator<=(const int8& x, const int8& y) {
  return y >= x;
}
inline int8 operator<=(const int8& x, int y) {
  return y >= x;
}
inline int8 operator<=(int x, const int8& y) {
  return y >= x;
}

/* int16 relational less than or equal (<=) */
inline int16 operator<=(const int16& x, const int16& y) {
  return y >= x;
}
inline int16 operator<=(const int16& x, int y) {
  return y >= x;
}
inline int16 operator<=(int x, const int16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* int2 equal (==) */
inline int2 operator==(const int2& x, const int2& y) {
  int2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline int2 operator==(const int2& x, int y) {
  int2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline int2 operator==(int x, const int2& y) {
  int2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* int3 equal (==) */
inline int3 operator==(const int3& x, const int3& y) {
  int3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline int3 operator==(const int3& x, int y) {
  int3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline int3 operator==(int x, const int3& y) {
  int3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* int4 equal (==) */
inline int4 operator==(const int4& x, const int4& y) {
  int4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline int4 operator==(const int4& x, int y) {
  int4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline int4 operator==(int x, const int4& y) {
  int4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* int8 equal (==) */
inline int8 operator==(const int8& x, const int8& y) {
  int8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline int8 operator==(const int8& x, int y) {
  int8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline int8 operator==(int x, const int8& y) {
  int8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* int16 equal (==) */
inline int16 operator==(const int16& x, const int16& y) {
  int16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline int16 operator==(const int16& x, int y) {
  int16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline int16 operator==(int x, const int16& y) {
  int16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* int2 not equal (!=) */
inline int2 operator!=(const int2& x, const int2& y) {
  int2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline int2 operator!=(const int2& x, int y) {
  int2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline int2 operator!=(int x, const int2& y) {
  int2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* int3 not equal (!=) */
inline int3 operator!=(const int3& x, const int3& y) {
  int3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline int3 operator!=(const int3& x, int y) {
  int3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline int3 operator!=(int x, const int3& y) {
  int3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* int4 not equal (!=) */
inline int4 operator!=(const int4& x, const int4& y) {
  int4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline int4 operator!=(const int4& x, int y) {
  int4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline int4 operator!=(int x, const int4& y) {
  int4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* int8 not equal (!=) */
inline int8 operator!=(const int8& x, const int8& y) {
  int8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline int8 operator!=(const int8& x, int y) {
  int8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline int8 operator!=(int x, const int8& y) {
  int8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* int16 not equal (!=) */
inline int16 operator!=(const int16& x, const int16& y) {
  int16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline int16 operator!=(const int16& x, int y) {
  int16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline int16 operator!=(int x, const int16& y) {
  int16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* int2 bitwise and (&) */
inline int2 operator&(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline int2 operator&(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline int2 operator&(int x, const int2& y) {
  return y & x;
}

/* int3 bitwise and (&) */
inline int3 operator&(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline int3 operator&(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline int3 operator&(int x, const int3& y) {
  return y & x;
}


/* int4 bitwise and (&) */
inline int4 operator&(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline int4 operator&(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline int4 operator&(int x, const int4& y) {
  return y & x;
}

/* int8 bitwise and (&) */
inline int8 operator&(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline int8 operator&(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline int8 operator&(int x, const int8& y) {
  return y & x;
}

/* int16 bitwise and (&) */
inline int16 operator&(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline int16 operator&(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline int16 operator&(int x, const int16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* int2 bitwise or (|) */
inline int2 operator|(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline int2 operator|(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline int2 operator|(int x, const int2& y) {
  return y | x;
}

/* int3 bitwise or (|) */
inline int3 operator|(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline int3 operator|(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline int3 operator|(int x, const int3& y) {
  return y | x;
}

/* int4 bitwise or (|) */
inline int4 operator|(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline int4 operator|(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline int4 operator|(int x, const int4& y) {
  return y | x;
}

/* int8 bitwise or (|) */
inline int8 operator|(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline int8 operator|(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline int8 operator|(int x, const int8& y) {
  return y | x;
}

/* int16 bitwise or (|) */
inline int16 operator|(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline int16 operator|(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline int16 operator|(int x, const int16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* int2 bitwise exclusive or (^) */
inline int2 operator^(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline int2 operator^(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline int2 operator^(int x, const int2& y) {
  return y ^ x;
}

/* int3 bitwise exclusive or (^) */
inline int3 operator^(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline int3 operator^(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline int3 operator^(int x, const int3& y) {
  return y ^ x;
}

/* int4 bitwise exclusive or (^) */
inline int4 operator^(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline int4 operator^(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline int4 operator^(int x, const int4& y) {
  return y ^ x;
}

/* int8 bitwise exclusive or (^) */
inline int8 operator^(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline int8 operator^(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline int8 operator^(int x, const int8& y) {
  return y ^ x;
}

/* int16 bitwise exclusive or (^) */
inline int16 operator^(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline int16 operator^(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline int16 operator^(int x, const int16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* int2 bitwise not (~) */
inline int2 operator~(const int2& x) {
  int2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* int3 bitwise not (~) */
inline int3 operator~(const int3& x) {
  int3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* int4 bitwise not (~) */
inline int4 operator~(const int4& x) {
  int4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* int8 bitwise not (~) */
inline int8 operator~(const int8& x) {
  int8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* int16 bitwise not (~) */
inline int16 operator~(const int16& x) {
  int16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* int2 logical and (&&) */
inline int2 operator&&(const int2& x, const int2& y) {
  int2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline int2 operator&&(const int2& x, int y) {
  int2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline int2 operator&&(int x, const int2& y) {
  int2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* int3 logical and (&&) */
inline int3 operator&&(const int3& x, const int3& y) {
  int3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline int3 operator&&(const int3& x, int y) {
  int3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline int3 operator&&(int x, const int3& y) {
  int3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* int4 logical and (&&) */
inline int4 operator&&(const int4& x, const int4& y) {
  int4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline int4 operator&&(const int4& x, int y) {
  int4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline int4 operator&&(int x, const int4& y) {
  int4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* int8 logical and (&&) */
inline int8 operator&&(const int8& x, const int8& y) {
  int8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline int8 operator&&(const int8& x, int y) {
  int8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline int8 operator&&(int x, const int8& y) {
  int8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* int16 logical and (&&) */
inline int16 operator&&(const int16& x, const int16& y) {
  int16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline int16 operator&&(const int16& x, int y) {
  int16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline int16 operator&&(int x, const int16& y) {
  int16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* int2 logical or (||) */
inline int2 operator||(const int2& x, const int2& y) {
  int2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline int2 operator||(const int2& x, int y) {
  int2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline int2 operator||(int x, const int2& y) {
  int2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* int3 logical or (||) */
inline int3 operator||(const int3& x, const int3& y) {
  int3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline int3 operator||(const int3& x, int y) {
  int3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline int3 operator||(int x, const int3& y) {
  int3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* int4 logical or (||) */
inline int4 operator||(const int4& x, const int4& y) {
  int4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline int4 operator||(const int4& x, int y) {
  int4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline int4 operator||(int x, const int4& y) {
  int4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* int8 logical or (||) */
inline int8 operator||(const int8& x, const int8& y) {
  int8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline int8 operator||(const int8& x, int y) {
  int8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline int8 operator||(int x, const int8& y) {
  int8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* int16 logical or (||) */
inline int16 operator||(const int16& x, const int16& y) {
  int16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline int16 operator||(const int16& x, int y) {
  int16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline int16 operator||(int x, const int16& y) {
  int16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* int2 logical not (!) */
inline int2 operator!(const int2& x) {
  int2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* int3 logical not (!) */
inline int3 operator!(const int3& x) {
  int3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* int4 logical not (!) */
inline int4 operator!(const int4& x) {
  int4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* int8 logical not (!) */
inline int8 operator!(const int8& x) {
  int8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* int16 logical not (!) */
inline int16 operator!(const int16& x) {
  int16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* int2 right-shift (>>) */
inline int2 operator>>(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, >>, y);
  return rst;
}
inline int2 operator>>(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, >>, y);
  return rst;
}
/* int3 right-shift (>>) */
inline int3 operator>>(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, >>, y);
  return rst;
}
inline int3 operator>>(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, >>, y);
  return rst;
}

/* int4 right-shift (>>) */
inline int4 operator>>(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, >>, y);
  return rst;
}
inline int4 operator>>(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, >>, y);
  return rst;
}

/* int8 right-shift (>>) */
inline int8 operator>>(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, >>, y);
  return rst;
}
inline int8 operator>>(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, >>, y);
  return rst;
}

/* int16 right-shift (>>) */
inline int16 operator>>(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, >>, y);
  return rst;
}
inline int16 operator>>(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, >>, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* int2 left-shift (<<) */
inline int2 operator<<(const int2& x, const int2& y) {
  int2 rst = __CL_V2_OP_V2(x, <<, y);
  return rst;
}
inline int2 operator<<(const int2& x, int y) {
  int2 rst = __CL_V2_OP_S(x, <<, y);
  return rst;
}

/* int3 left-shift (<<) */
inline int3 operator<<(const int3& x, const int3& y) {
  int3 rst = __CL_V3_OP_V3(x, <<, y);
  return rst;
}
inline int3 operator<<(const int3& x, int y) {
  int3 rst = __CL_V3_OP_S(x, <<, y);
  return rst;
}

/* int4 left-shift (<<) */
inline int4 operator<<(const int4& x, const int4& y) {
  int4 rst = __CL_V4_OP_V4(x, <<, y);
  return rst;
}
inline int4 operator<<(const int4& x, int y) {
  int4 rst = __CL_V4_OP_S(x, <<, y);
  return rst;
}

/* int8 left-shift (<<) */
inline int8 operator<<(const int8& x, const int8& y) {
  int8 rst = __CL_V8_OP_V8(x, <<, y);
  return rst;
}
inline int8 operator<<(const int8& x, int y) {
  int8 rst = __CL_V8_OP_S(x, <<, y);
  return rst;
}

/* int16 left-shift (<<) */
inline int16 operator<<(const int16& x, const int16& y) {
  int16 rst = __CL_V16_OP_V16(x, <<, y);
  return rst;
}
inline int16 operator<<(const int16& x, int y) {
  int16 rst = __CL_V16_OP_S(x, <<, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* int2 assignment add into (+=) */
inline int2 operator+=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, +, y);
}
inline int2 operator+=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, +, y);
}

/* int3 assignment add into (+=) */
inline int3 operator+=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, +, y);
}
inline int3 operator+=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, +, y);
}

/* int4 assignment add into (+=) */
inline int4 operator+=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, +, y);
}
inline int4 operator+=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, +, y);
}

/* int8 assignment add into (+=) */
inline int8 operator+=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, +, y);
}
inline int8 operator+=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, +, y);
}

/* int16 assignment add into (+=) */
inline int16 operator+=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, +, y);
}
inline int16 operator+=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* int2 assignment subtract from (-=) */
inline int2 operator-=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, -, y);
}
inline int2 operator-=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, -, y);
}

/* int3 assignment subtract from (-=) */
inline int3 operator-=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, -, y);
}
inline int3 operator-=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, -, y);
}

/* int4 assignment subtract from (-=) */
inline int4 operator-=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, -, y);
}
inline int4 operator-=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, -, y);
}

/* int8 assignment subtract from (-=) */
inline int8 operator-=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, -, y);
}
inline int8 operator-=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, -, y);
}

/* int16 assignment subtract from (-=) */
inline int16 operator-=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, -, y);
}
inline int16 operator-=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* int2 assignment multiply into (*=) */
inline int2 operator*=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, *, y);
}
inline int2 operator*=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, *, y);
}

/* int3 assignment multiply into (*=) */
inline int3 operator*=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, *, y);
}
inline int3 operator*=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, *, y);
}

/* int4 assignment multiply into (*=) */
inline int4 operator*=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, *, y);
}
inline int4 operator*=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, *, y);
}

/* int8 assignment multiply into (*=) */
inline int8 operator*=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, *, y);
}
inline int8 operator*=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, *, y);
}

/* int16 assignment multiply into (*=) */
inline int16 operator*=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, *, y);
}
inline int16 operator*=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* int2 assignment divide into (/=) */
inline int2 operator/=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, /, y);
}
inline int2 operator/=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, /, y);
}

/* int3 assignment divide into (/=) */
inline int3 operator/=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, /, y);
}
inline int3 operator/=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, /, y);
}

/* int4 assignment divide into (/=) */
inline int4 operator/=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, /, y);
}
inline int4 operator/=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, /, y);
}

/* int8 assignment divide into (/=) */
inline int8 operator/=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, /, y);
}
inline int8 operator/=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, /, y);
}

/* int16 assignment divide into (/=) */
inline int16 operator/=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, /, y);
}
inline int16 operator/=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* int2 assignment modulus into (%=) */
inline int2 operator%=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, %, y);
}
inline int2 operator%=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, %, y);
}

/* int3 assignment modulus into (%=) */
inline int3 operator%=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, %, y);
}
inline int3 operator%=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, %, y);
}

/* int4 assignment modulus into (%=) */
inline int4 operator%=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, %, y);
}
inline int4 operator%=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, %, y);
}

/* int8 assignment modulus into (%=) */
inline int8 operator%=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, %, y);
}
inline int8 operator%=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, %, y);
}

/* int16 assignment modulus into (%=) */
inline int16 operator%=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, %, y);
}
inline int16 operator%=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* int2 assignment left shift by (<<=) */
inline int2 operator<<=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, <<, y);
}
inline int2 operator<<=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, <<, y);
}

/* int3 assignment left shift by (<<=) */
inline int3 operator<<=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, <<, y);
}
inline int3 operator<<=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, <<, y);
}


/* int4 assignment left shift by (<<=) */
inline int4 operator<<=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, <<, y);
}
inline int4 operator<<=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, <<, y);
}

/* int8 assignment left shift by (<<=) */
inline int8 operator<<=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, <<, y);
}
inline int8 operator<<=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, <<, y);
}

/* int16 assignment left shift by (<<=) */
inline int16 operator<<=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, <<, y);
}
inline int16 operator<<=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* int2 assignment right shift by (>>=) */
inline int2 operator>>=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, >>, y);
}
inline int2 operator>>=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, >>, y);
}

/* int3 assignment right shift by (>>=) */
inline int3 operator>>=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, >>, y);
}
inline int3 operator>>=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, >>, y);
}
/* int4 assignment right shift by (>>=) */
inline int4 operator>>=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, >>, y);
}
inline int4 operator>>=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, >>, y);
}

/* int8 assignment right shift by (>>=) */
inline int8 operator>>=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, >>, y);
}
inline int8 operator>>=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, >>, y);
}

/* int16 assignment right shift by (>>=) */
inline int16 operator>>=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, >>, y);
}
inline int16 operator>>=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* int2 assignment and into (&=) */
inline int2 operator&=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, &, y);
}
inline int2 operator&=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, &, y);
}

/* int3 assignment and into (&=) */
inline int3 operator&=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, &, y);
}
inline int3 operator&=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, &, y);
}

/* int4 assignment and into (&=) */
inline int4 operator&=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, &, y);
}
inline int4 operator&=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, &, y);
}

/* int8 assignment and into (&=) */
inline int8 operator&=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, &, y);
}
inline int8 operator&=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, &, y);
}

/* int16 assignment and into (&=) */
inline int16 operator&=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, &, y);
}
inline int16 operator&=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* int2 assignment inclusive or into (|=) */
inline int2 operator|=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, |, y);
}
inline int2 operator|=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, |, y);
}

/* int3 assignment inclusive or into (|=) */
inline int3 operator|=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, |, y);
}
inline int3 operator|=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, |, y);

}
/* int4 assignment inclusive or into (|=) */
inline int4 operator|=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, |, y);
}
inline int4 operator|=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, |, y);
}

/* int8 assignment inclusive or into (|=) */
inline int8 operator|=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, |, y);
}
inline int8 operator|=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, |, y);
}

/* int16 assignment inclusive or into (|=) */
inline int16 operator|=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, |, y);
}
inline int16 operator|=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* int2 assignment exclusive or into (^=) */
inline int2 operator^=(int2 &x, const int2& y) {
  __CL_V_OP_ASSIGN_V(int2, x, ^, y);
}
inline int2 operator^=(int2 &x, int y) {
  __CL_V_OP_ASSIGN_V(int2, x, ^, y);
}

/* int3 assignment exclusive or into (^=) */
inline int3 operator^=(int3 &x, const int3& y) {
  __CL_V_OP_ASSIGN_V(int3, x, ^, y);
}
inline int3 operator^=(int3 &x, int y) {
  __CL_V_OP_ASSIGN_V(int3, x, ^, y);
}

/* int4 assignment exclusive or into (^=) */
inline int4 operator^=(int4 &x, const int4& y) {
  __CL_V_OP_ASSIGN_V(int4, x, ^, y);
}
inline int4 operator^=(int4 &x, int y) {
  __CL_V_OP_ASSIGN_V(int4, x, ^, y);
}

/* int8 assignment exclusive or into (^=) */
inline int8 operator^=(int8 &x, const int8& y) {
  __CL_V_OP_ASSIGN_V(int8, x, ^, y);
}
inline int8 operator^=(int8 &x, int y) {
  __CL_V_OP_ASSIGN_V(int8, x, ^, y);
}

/* int16 assignment exclusive or into (^=) */
inline int16 operator^=(int16 &x, const int16& y) {
  __CL_V_OP_ASSIGN_V(int16, x, ^, y);
}
inline int16 operator^=(int16 &x, int y) {
  __CL_V_OP_ASSIGN_V(int16, x, ^, y);
}

#endif //__CL_OPS_INTN_H

