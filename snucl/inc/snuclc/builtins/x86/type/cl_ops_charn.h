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

#ifndef __CL_OPS_CHARN_H
#define __CL_OPS_CHARN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>


///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* char2 addition (+) */
inline char2 operator+(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline char2 operator+(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline char2 operator+(schar x, const char2& y) {
  return y + x;
}

/* char3 addition (+) */
inline char3 operator+(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline char3 operator+(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline char3 operator+(schar x, const char3& y) {
  return y + x;
}

/* char4 addition (+) */
inline char4 operator+(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline char4 operator+(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline char4 operator+(schar x, const char4& y) {
  return y + x;
}

/* char8 addition (+) */
inline char8 operator+(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline char8 operator+(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline char8 operator+(schar x, const char8& y) {
  return y + x;
}

/* char16 addition (+) */
inline char16 operator+(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline char16 operator+(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline char16 operator+(schar x, const char16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* char2 subtraction (-) */
inline char2 operator-(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline char2 operator-(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline char2 operator-(schar x, const char2& y) {
  char2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* char3 subtraction (-) */
inline char3 operator-(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline char3 operator-(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline char3 operator-(schar x, const char3& y) {
  char3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* char4 subtraction (-) */
inline char4 operator-(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline char4 operator-(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline char4 operator-(schar x, const char4& y) {
  char4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* char8 subtraction (-) */
inline char8 operator-(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline char8 operator-(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline char8 operator-(schar x, const char8& y) {
  char8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* char16 subtraction (-) */
inline char16 operator-(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline char16 operator-(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline char16 operator-(schar x, const char16& y) {
  char16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* char2 multiplication (*) */
inline char2 operator*(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline char2 operator*(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline char2 operator*(schar x, const char2& y) {
  return y * x;
}

/* char3 multiplication (*) */
inline char3 operator*(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline char3 operator*(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline char3 operator*(schar x, const char3& y) {
  return y + x;
}


/* char4 multiplication (*) */
inline char4 operator*(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline char4 operator*(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline char4 operator*(schar x, const char4& y) {
  return y + x;
}

/* char8 multiplication (*) */
inline char8 operator*(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline char8 operator*(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline char8 operator*(schar x, const char8& y) {
  return y * x;
}

/* char16 multiplication (*) */
inline char16 operator*(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline char16 operator*(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline char16 operator*(schar x, const char16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* char2 division (/) */
inline char2 operator/(const char2& x, const char2& y) {
  char2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline char2 operator/(const char2& x, schar y) {
  char2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline char2 operator/(schar x, const char2& y) {
  char2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* char3 division (/) */
inline char3 operator/(const char3& x, const char3& y) {
  char3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline char3 operator/(const char3& x, schar y) {
  char3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline char3 operator/(schar x, const char3& y) {
  char3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* char4 division (/) */
inline char4 operator/(const char4& x, const char4& y) {
  char4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline char4 operator/(const char4& x, schar y) {
  char4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline char4 operator/(schar x, const char4& y) {
  char4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* char8 division (/) */
inline char8 operator/(const char8& x, const char8& y) {
  char8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline char8 operator/(const char8& x, schar y) {
  char8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline char8 operator/(schar x, const char8& y) {
  char8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* char16 division (/) */
inline char16 operator/(const char16& x, const char16& y) {
  char16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline char16 operator/(const char16& x, schar y) {
  char16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline char16 operator/(schar x, const char16& y) {
  char16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* char2 remainder (%) */
inline char2 operator%(const char2& x, const char2& y) {
  char2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline char2 operator%(const char2& x, schar y) {
  char2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline char2 operator%(schar x, const char2& y) {
  char2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* char3 remainder (%) */
inline char3 operator%(const char3& x, const char3& y) {
  char3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline char3 operator%(const char3& x, schar y) {
  char3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline char3 operator%(schar x, const char3& y) {
  char3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* char4 remainder (%) */
inline char4 operator%(const char4& x, const char4& y) {
  char4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline char4 operator%(const char4& x, schar y) {
  char4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline char4 operator%(schar x, const char4& y) {
  char4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* char8 remainder (%) */
inline char8 operator%(const char8& x, const char8& y) {
  char8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline char8 operator%(const char8& x, schar y) {
  char8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline char8 operator%(schar x, const char8& y) {
  char8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* char16 remainder (%) */
inline char16 operator%(const char16& x, const char16& y) {
  char16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline char16 operator%(const char16& x, schar y) {
  char16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline char16 operator%(schar x, const char16& y) {
  char16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* char2 unary positive (+) */
inline char2 operator+(const char2& x) {
  return x;
}
/* char3 unary positive (+) */
inline char3 operator+(const char3& x) {
  return x;
}

/* char4 unary positive (+) */
inline char4 operator+(const char4& x) {
  return x;
}

/* char8 unary positive (+) */
inline char8 operator+(const char8& x) {
  return x;
}

/* char16 unary positive (+) */
inline char16 operator+(const char16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* char2 unary negative (-) */
inline char2 operator-(const char2& x) {
  char2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* char3 unary negative (-) */
inline char3 operator-(const char3& x) {
  char3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* char4 unary negative (-) */
inline char4 operator-(const char4& x) {
  char4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* char8 unary negative (-) */
inline char8 operator-(const char8& x) {
  char8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* char16 unary negative (-) */
inline char16 operator-(const char16& x) {
  char16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* char2 unary post-increment (++) */
inline char2 operator++(char2 &x, int n) {
  char2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* char3 unary post-increment (++) */
inline char3 operator++(char3 &x, int n) {
  char3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* char4 unary post-increment (++) */
inline char4 operator++(char4 &x, int n) {
  char4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* char8 unary post-increment (++) */
inline char8 operator++(char8 &x, int n) {
  char8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* char16 unary post-increment (++) */
inline char16 operator++(char16 &x, int n) {
  char16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* char2 unary pre-increment (++) */
inline char2 operator++(char2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* char3 unary pre-increment (++) */
inline char3 operator++(char3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* char4 unary pre-increment (++) */
inline char4 operator++(char4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* char8 unary pre-increment (++) */
inline char8 operator++(char8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* char16 unary pre-increment (++) */
inline char16 operator++(char16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* char2 unary post-decrement (--) */
inline char2 operator--(char2 &x, int n) {
  char2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* char3 unary post-decrement (--) */
inline char3 operator--(char3 &x, int n) {
  char3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* char4 unary post-decrement (--) */
inline char4 operator--(char4 &x, int n) {
  char4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* char8 unary post-decrement (--) */
inline char8 operator--(char8 &x, int n) {
  char8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* char16 unary post-decrement (--) */
inline char16 operator--(char16 &x, int n) {
  char16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* char2 unary pre-decrement (--) */
inline char2 operator--(char2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* char3 unary pre-decrement (--) */
inline char3 operator--(char3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* char4 unary pre-decrement (--) */
inline char4 operator--(char4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* char8 unary pre-decrement (--) */
inline char8 operator--(char8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* char16 unary pre-decrement (--) */
inline char16 operator--(char16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* char2 relational greater than (>) */
inline char2 operator>(const char2& x, const char2& y) {
  char2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline char2 operator>(const char2& x, schar y) {
  char2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline char2 operator>(schar x, const char2& y) {
  char2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* char3 relational greater than (>) */
inline char3 operator>(const char3& x, const char3& y) {
  char3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline char3 operator>(const char3& x, schar y) {
  char3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline char3 operator>(schar x, const char3& y) {
  char3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* char4 relational greater than (>) */
inline char4 operator>(const char4& x, const char4& y) {
  char4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline char4 operator>(const char4& x, schar y) {
  char4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline char4 operator>(schar x, const char4& y) {
  char4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* char8 relational greater than (>) */
inline char8 operator>(const char8& x, const char8& y) {
  char8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline char8 operator>(const char8& x, schar y) {
  char8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline char8 operator>(schar x, const char8& y) {
  char8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* char16 relational greater than (>) */
inline char16 operator>(const char16& x, const char16& y) {
  char16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline char16 operator>(const char16& x, schar y) {
  char16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline char16 operator>(schar x, const char16& y) {
  char16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* char2 relational less than (<) */
inline char2 operator<(const char2& x, const char2& y) {
  return y > x;
}
inline char2 operator<(const char2& x, schar y) {
  return y > x;
}
inline char2 operator<(schar x, const char2& y) {
  return y > x;
}

/* char3 relational less than (<) */
inline char3 operator<(const char3& x, const char3& y) {
  return y > x;
}
inline char3 operator<(const char3& x, schar y) {
  return y > x;
}
inline char3 operator<(schar x, const char3& y) {
  return y > x;
}

/* char4 relational less than (<) */
inline char4 operator<(const char4& x, const char4& y) {
  return y > x;
}
inline char4 operator<(const char4& x, schar y) {
  return y > x;
}
inline char4 operator<(schar x, const char4& y) {
  return y > x;
}

/* char8 relational less than (<) */
inline char8 operator<(const char8& x, const char8& y) {
  return y > x;
}
inline char8 operator<(const char8& x, schar y) {
  return y > x;
}
inline char8 operator<(schar x, const char8& y) {
  return y > x;
}

/* char16 relational less than (<) */
inline char16 operator<(const char16& x, const char16& y) {
  return y > x;
}
inline char16 operator<(const char16& x, schar y) {
  return y > x;
}
inline char16 operator<(schar x, const char16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* char2 relational greater than or equal (>=) */
inline char2 operator>=(const char2& x, const char2& y) {
  char2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline char2 operator>=(const char2& x, schar y) {
  char2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline char2 operator>=(schar x, const char2& y) {
  char2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* char3 relational greater than or equal (>=) */
inline char3 operator>=(const char3& x, const char3& y) {
  char3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline char3 operator>=(const char3& x, schar y) {
  char3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline char3 operator>=(schar x, const char3& y) {
  char3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* char4 relational greater than or equal (>=) */
inline char4 operator>=(const char4& x, const char4& y) {
  char4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline char4 operator>=(const char4& x, schar y) {
  char4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline char4 operator>=(schar x, const char4& y) {
  char4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* char8 relational greater than or equal (>=) */
inline char8 operator>=(const char8& x, const char8& y) {
  char8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline char8 operator>=(const char8& x, schar y) {
  char8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline char8 operator>=(schar x, const char8& y) {
  char8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* char16 relational greater than or equal (>=) */
inline char16 operator>=(const char16& x, const char16& y) {
  char16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline char16 operator>=(const char16& x, schar y) {
  char16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline char16 operator>=(schar x, const char16& y) {
  char16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* char2 relational less than or equal (<=) */
inline char2 operator<=(const char2& x, const char2& y) {
  return y >= x;
}
inline char2 operator<=(const char2& x, schar y) {
  return y >= x;
}
inline char2 operator<=(schar x, const char2& y) {
  return y >= x;
}

/* char3 relational less than or equal (<=) */
inline char3 operator<=(const char3& x, const char3& y) {
  return y >= x;
}
inline char3 operator<=(const char3& x, schar y) {
  return y >= x;
}
inline char3 operator<=(schar x, const char3& y) {
  return y >= x;
}

/* char4 relational less than or equal (<=) */
inline char4 operator<=(const char4& x, const char4& y) {
  return y >= x;
}
inline char4 operator<=(const char4& x, schar y) {
  return y >= x;
}
inline char4 operator<=(schar x, const char4& y) {
  return y >= x;
}

/* char8 relational less than or equal (<=) */
inline char8 operator<=(const char8& x, const char8& y) {
  return y >= x;
}
inline char8 operator<=(const char8& x, schar y) {
  return y >= x;
}
inline char8 operator<=(schar x, const char8& y) {
  return y >= x;
}

/* char16 relational less than or equal (<=) */
inline char16 operator<=(const char16& x, const char16& y) {
  return y >= x;
}
inline char16 operator<=(const char16& x, schar y) {
  return y >= x;
}
inline char16 operator<=(schar x, const char16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* char2 equal (==) */
inline char2 operator==(const char2& x, const char2& y) {
  char2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline char2 operator==(const char2& x, schar y) {
  char2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline char2 operator==(schar x, const char2& y) {
  char2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* char3 equal (==) */
inline char3 operator==(const char3& x, const char3& y) {
  char3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline char3 operator==(const char3& x, schar y) {
  char3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline char3 operator==(schar x, const char3& y) {
  char3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* char4 equal (==) */
inline char4 operator==(const char4& x, const char4& y) {
  char4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline char4 operator==(const char4& x, schar y) {
  char4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline char4 operator==(schar x, const char4& y) {
  char4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* char8 equal (==) */
inline char8 operator==(const char8& x, const char8& y) {
  char8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline char8 operator==(const char8& x, schar y) {
  char8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline char8 operator==(schar x, const char8& y) {
  char8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* char16 equal (==) */
inline char16 operator==(const char16& x, const char16& y) {
  char16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline char16 operator==(const char16& x, schar y) {
  char16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline char16 operator==(schar x, const char16& y) {
  char16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* char2 not equal (!=) */
inline char2 operator!=(const char2& x, const char2& y) {
  char2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline char2 operator!=(const char2& x, schar y) {
  char2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline char2 operator!=(schar x, const char2& y) {
  char2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* char3 not equal (!=) */
inline char3 operator!=(const char3& x, const char3& y) {
  char3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline char3 operator!=(const char3& x, schar y) {
  char3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline char3 operator!=(schar x, const char3& y) {
  char3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* char4 not equal (!=) */
inline char4 operator!=(const char4& x, const char4& y) {
  char4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline char4 operator!=(const char4& x, schar y) {
  char4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline char4 operator!=(schar x, const char4& y) {
  char4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* char8 not equal (!=) */
inline char8 operator!=(const char8& x, const char8& y) {
  char8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline char8 operator!=(const char8& x, schar y) {
  char8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline char8 operator!=(schar x, const char8& y) {
  char8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* char16 not equal (!=) */
inline char16 operator!=(const char16& x, const char16& y) {
  char16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline char16 operator!=(const char16& x, schar y) {
  char16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline char16 operator!=(schar x, const char16& y) {
  char16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* char2 bitwise and (&) */
inline char2 operator&(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline char2 operator&(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline char2 operator&(schar x, const char2& y) {
  return y & x;
}

/* char3 bitwise and (&) */
inline char3 operator&(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline char3 operator&(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline char3 operator&(schar x, const char3& y) {
  return y & x;
}


/* char4 bitwise and (&) */
inline char4 operator&(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline char4 operator&(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline char4 operator&(schar x, const char4& y) {
  return y & x;
}

/* char8 bitwise and (&) */
inline char8 operator&(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline char8 operator&(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline char8 operator&(schar x, const char8& y) {
  return y & x;
}

/* char16 bitwise and (&) */
inline char16 operator&(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline char16 operator&(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline char16 operator&(schar x, const char16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* char2 bitwise or (|) */
inline char2 operator|(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline char2 operator|(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline char2 operator|(schar x, const char2& y) {
  return y | x;
}

/* char3 bitwise or (|) */
inline char3 operator|(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline char3 operator|(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline char3 operator|(schar x, const char3& y) {
  return y | x;
}

/* char4 bitwise or (|) */
inline char4 operator|(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline char4 operator|(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline char4 operator|(schar x, const char4& y) {
  return y | x;
}

/* char8 bitwise or (|) */
inline char8 operator|(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline char8 operator|(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline char8 operator|(schar x, const char8& y) {
  return y | x;
}

/* char16 bitwise or (|) */
inline char16 operator|(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline char16 operator|(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline char16 operator|(schar x, const char16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* char2 bitwise exclusive or (^) */
inline char2 operator^(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline char2 operator^(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline char2 operator^(schar x, const char2& y) {
  return y ^ x;
}

/* char3 bitwise exclusive or (^) */
inline char3 operator^(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline char3 operator^(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline char3 operator^(schar x, const char3& y) {
  return y ^ x;
}

/* char4 bitwise exclusive or (^) */
inline char4 operator^(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline char4 operator^(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline char4 operator^(schar x, const char4& y) {
  return y ^ x;
}

/* char8 bitwise exclusive or (^) */
inline char8 operator^(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline char8 operator^(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline char8 operator^(schar x, const char8& y) {
  return y ^ x;
}

/* char16 bitwise exclusive or (^) */
inline char16 operator^(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline char16 operator^(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline char16 operator^(schar x, const char16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* char2 bitwise not (~) */
inline char2 operator~(const char2& x) {
  char2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* char3 bitwise not (~) */
inline char3 operator~(const char3& x) {
  char3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* char4 bitwise not (~) */
inline char4 operator~(const char4& x) {
  char4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* char8 bitwise not (~) */
inline char8 operator~(const char8& x) {
  char8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* char16 bitwise not (~) */
inline char16 operator~(const char16& x) {
  char16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* char2 logical and (&&) */
inline char2 operator&&(const char2& x, const char2& y) {
  char2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline char2 operator&&(const char2& x, schar y) {
  char2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline char2 operator&&(schar x, const char2& y) {
  char2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* char3 logical and (&&) */
inline char3 operator&&(const char3& x, const char3& y) {
  char3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline char3 operator&&(const char3& x, schar y) {
  char3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline char3 operator&&(schar x, const char3& y) {
  char3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* char4 logical and (&&) */
inline char4 operator&&(const char4& x, const char4& y) {
  char4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline char4 operator&&(const char4& x, schar y) {
  char4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline char4 operator&&(schar x, const char4& y) {
  char4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* char8 logical and (&&) */
inline char8 operator&&(const char8& x, const char8& y) {
  char8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline char8 operator&&(const char8& x, schar y) {
  char8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline char8 operator&&(schar x, const char8& y) {
  char8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* char16 logical and (&&) */
inline char16 operator&&(const char16& x, const char16& y) {
  char16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline char16 operator&&(const char16& x, schar y) {
  char16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline char16 operator&&(schar x, const char16& y) {
  char16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* char2 logical or (||) */
inline char2 operator||(const char2& x, const char2& y) {
  char2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline char2 operator||(const char2& x, schar y) {
  char2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline char2 operator||(schar x, const char2& y) {
  char2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* char3 logical or (||) */
inline char3 operator||(const char3& x, const char3& y) {
  char3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline char3 operator||(const char3& x, schar y) {
  char3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline char3 operator||(schar x, const char3& y) {
  char3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* char4 logical or (||) */
inline char4 operator||(const char4& x, const char4& y) {
  char4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline char4 operator||(const char4& x, schar y) {
  char4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline char4 operator||(schar x, const char4& y) {
  char4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* char8 logical or (||) */
inline char8 operator||(const char8& x, const char8& y) {
  char8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline char8 operator||(const char8& x, schar y) {
  char8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline char8 operator||(schar x, const char8& y) {
  char8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* char16 logical or (||) */
inline char16 operator||(const char16& x, const char16& y) {
  char16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline char16 operator||(const char16& x, schar y) {
  char16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline char16 operator||(schar x, const char16& y) {
  char16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* char2 logical not (!) */
inline char2 operator!(const char2& x) {
  char2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* char3 logical not (!) */
inline char3 operator!(const char3& x) {
  char3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* char4 logical not (!) */
inline char4 operator!(const char4& x) {
  char4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* char8 logical not (!) */
inline char8 operator!(const char8& x) {
  char8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* char16 logical not (!) */
inline char16 operator!(const char16& x) {
  char16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* char2 right-shift (>>) */
inline char2 operator>>(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, >>, (y & 0x7));
  return rst;
}
inline char2 operator>>(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, >>, (y & 0x7));
  return rst;
}
/* char3 right-shift (>>) */
inline char3 operator>>(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, >>, (y & 0x7));
  return rst;
}
inline char3 operator>>(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, >>, (y & 0x7));
  return rst;
}

/* char4 right-shift (>>) */
inline char4 operator>>(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, >>, (y & 0x7));
  return rst;
}
inline char4 operator>>(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, >>, (y & 0x7));
  return rst;
}

/* char8 right-shift (>>) */
inline char8 operator>>(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, >>, (y & 0x7));
  return rst;
}
inline char8 operator>>(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, >>, (y & 0x7));
  return rst;
}

/* char16 right-shift (>>) */
inline char16 operator>>(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, >>, (y & 0x7));
  return rst;
}
inline char16 operator>>(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, >>, (y & 0x7));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* char2 left-shift (<<) */
inline char2 operator<<(const char2& x, const char2& y) {
  char2 rst = __CL_V2_OP_V2(x, <<, (y & 0x7));
  return rst;
}
inline char2 operator<<(const char2& x, schar y) {
  char2 rst = __CL_V2_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* char3 left-shift (<<) */
inline char3 operator<<(const char3& x, const char3& y) {
  char3 rst = __CL_V3_OP_V3(x, <<, (y & 0x7));
  return rst;
}
inline char3 operator<<(const char3& x, schar y) {
  char3 rst = __CL_V3_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* char4 left-shift (<<) */
inline char4 operator<<(const char4& x, const char4& y) {
  char4 rst = __CL_V4_OP_V4(x, <<, (y & 0x7));
  return rst;
}
inline char4 operator<<(const char4& x, schar y) {
  char4 rst = __CL_V4_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* char8 left-shift (<<) */
inline char8 operator<<(const char8& x, const char8& y) {
  char8 rst = __CL_V8_OP_V8(x, <<, (y & 0x7));
  return rst;
}
inline char8 operator<<(const char8& x, schar y) {
  char8 rst = __CL_V8_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* char16 left-shift (<<) */
inline char16 operator<<(const char16& x, const char16& y) {
  char16 rst = __CL_V16_OP_V16(x, <<, (y & 0x7));
  return rst;
}
inline char16 operator<<(const char16& x, schar y) {
  char16 rst = __CL_V16_OP_S(x, <<, (y & 0x7));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* char2 assignment add into (+=) */
inline char2 operator+=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, +, y);
}
inline char2 operator+=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, +, y);
}

/* char3 assignment add into (+=) */
inline char3 operator+=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, +, y);
}
inline char3 operator+=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, +, y);
}

/* char4 assignment add into (+=) */
inline char4 operator+=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, +, y);
}
inline char4 operator+=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, +, y);
}

/* char8 assignment add into (+=) */
inline char8 operator+=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, +, y);
}
inline char8 operator+=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, +, y);
}

/* char16 assignment add into (+=) */
inline char16 operator+=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, +, y);
}
inline char16 operator+=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* char2 assignment subtract from (-=) */
inline char2 operator-=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, -, y);
}
inline char2 operator-=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, -, y);
}

/* char3 assignment subtract from (-=) */
inline char3 operator-=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, -, y);
}
inline char3 operator-=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, -, y);
}

/* char4 assignment subtract from (-=) */
inline char4 operator-=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, -, y);
}
inline char4 operator-=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, -, y);
}

/* char8 assignment subtract from (-=) */
inline char8 operator-=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, -, y);
}
inline char8 operator-=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, -, y);
}

/* char16 assignment subtract from (-=) */
inline char16 operator-=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, -, y);
}
inline char16 operator-=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* char2 assignment multiply into (*=) */
inline char2 operator*=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, *, y);
}
inline char2 operator*=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, *, y);
}

/* char3 assignment multiply into (*=) */
inline char3 operator*=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, *, y);
}
inline char3 operator*=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, *, y);
}

/* char4 assignment multiply into (*=) */
inline char4 operator*=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, *, y);
}
inline char4 operator*=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, *, y);
}

/* char8 assignment multiply into (*=) */
inline char8 operator*=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, *, y);
}
inline char8 operator*=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, *, y);
}

/* char16 assignment multiply into (*=) */
inline char16 operator*=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, *, y);
}
inline char16 operator*=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* char2 assignment divide into (/=) */
inline char2 operator/=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, /, y);
}
inline char2 operator/=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, /, y);
}

/* char3 assignment divide into (/=) */
inline char3 operator/=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, /, y);
}
inline char3 operator/=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, /, y);
}

/* char4 assignment divide into (/=) */
inline char4 operator/=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, /, y);
}
inline char4 operator/=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, /, y);
}

/* char8 assignment divide into (/=) */
inline char8 operator/=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, /, y);
}
inline char8 operator/=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, /, y);
}

/* char16 assignment divide into (/=) */
inline char16 operator/=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, /, y);
}
inline char16 operator/=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* char2 assignment modulus into (%=) */
inline char2 operator%=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, %, y);
}
inline char2 operator%=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, %, y);
}

/* char3 assignment modulus into (%=) */
inline char3 operator%=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, %, y);
}
inline char3 operator%=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, %, y);
}

/* char4 assignment modulus into (%=) */
inline char4 operator%=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, %, y);
}
inline char4 operator%=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, %, y);
}

/* char8 assignment modulus into (%=) */
inline char8 operator%=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, %, y);
}
inline char8 operator%=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, %, y);
}

/* char16 assignment modulus into (%=) */
inline char16 operator%=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, %, y);
}
inline char16 operator%=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* char2 assignment left shift by (<<=) */
inline char2 operator<<=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, <<, y);
}
inline char2 operator<<=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, <<, y);
}

/* char3 assignment left shift by (<<=) */
inline char3 operator<<=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, <<, y);
}
inline char3 operator<<=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, <<, y);
}


/* char4 assignment left shift by (<<=) */
inline char4 operator<<=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, <<, y);
}
inline char4 operator<<=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, <<, y);
}

/* char8 assignment left shift by (<<=) */
inline char8 operator<<=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, <<, y);
}
inline char8 operator<<=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, <<, y);
}

/* char16 assignment left shift by (<<=) */
inline char16 operator<<=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, <<, y);
}
inline char16 operator<<=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* char2 assignment right shift by (>>=) */
inline char2 operator>>=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, >>, y);
}
inline char2 operator>>=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, >>, y);
}

/* char3 assignment right shift by (>>=) */
inline char3 operator>>=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, >>, y);
}
inline char3 operator>>=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, >>, y);
}
/* char4 assignment right shift by (>>=) */
inline char4 operator>>=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, >>, y);
}
inline char4 operator>>=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, >>, y);
}

/* char8 assignment right shift by (>>=) */
inline char8 operator>>=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, >>, y);
}
inline char8 operator>>=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, >>, y);
}

/* char16 assignment right shift by (>>=) */
inline char16 operator>>=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, >>, y);
}
inline char16 operator>>=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* char2 assignment and into (&=) */
inline char2 operator&=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, &, y);
}
inline char2 operator&=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, &, y);
}

/* char3 assignment and into (&=) */
inline char3 operator&=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, &, y);
}
inline char3 operator&=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, &, y);
}

/* char4 assignment and into (&=) */
inline char4 operator&=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, &, y);
}
inline char4 operator&=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, &, y);
}

/* char8 assignment and into (&=) */
inline char8 operator&=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, &, y);
}
inline char8 operator&=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, &, y);
}

/* char16 assignment and into (&=) */
inline char16 operator&=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, &, y);
}
inline char16 operator&=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* char2 assignment inclusive or into (|=) */
inline char2 operator|=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, |, y);
}
inline char2 operator|=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, |, y);
}

/* char3 assignment inclusive or into (|=) */
inline char3 operator|=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, |, y);
}
inline char3 operator|=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, |, y);

}
/* char4 assignment inclusive or into (|=) */
inline char4 operator|=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, |, y);
}
inline char4 operator|=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, |, y);
}

/* char8 assignment inclusive or into (|=) */
inline char8 operator|=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, |, y);
}
inline char8 operator|=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, |, y);
}

/* char16 assignment inclusive or into (|=) */
inline char16 operator|=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, |, y);
}
inline char16 operator|=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* char2 assignment exclusive or into (^=) */
inline char2 operator^=(char2 &x, const char2& y) {
  __CL_V_OP_ASSIGN_V(char2, x, ^, y);
}
inline char2 operator^=(char2 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char2, x, ^, y);
}

/* char3 assignment exclusive or into (^=) */
inline char3 operator^=(char3 &x, const char3& y) {
  __CL_V_OP_ASSIGN_V(char3, x, ^, y);
}
inline char3 operator^=(char3 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char3, x, ^, y);
}

/* char4 assignment exclusive or into (^=) */
inline char4 operator^=(char4 &x, const char4& y) {
  __CL_V_OP_ASSIGN_V(char4, x, ^, y);
}
inline char4 operator^=(char4 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char4, x, ^, y);
}

/* char8 assignment exclusive or into (^=) */
inline char8 operator^=(char8 &x, const char8& y) {
  __CL_V_OP_ASSIGN_V(char8, x, ^, y);
}
inline char8 operator^=(char8 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char8, x, ^, y);
}

/* char16 assignment exclusive or into (^=) */
inline char16 operator^=(char16 &x, const char16& y) {
  __CL_V_OP_ASSIGN_V(char16, x, ^, y);
}
inline char16 operator^=(char16 &x, schar y) {
  __CL_V_OP_ASSIGN_V(char16, x, ^, y);
}

#endif //__CL_OPS_CHARN_H

