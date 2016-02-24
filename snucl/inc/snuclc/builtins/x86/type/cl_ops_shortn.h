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

#ifndef __CL_OPS_SHORTN_H
#define __CL_OPS_SHORTN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>


///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* short2 addition (+) */
inline short2 operator+(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline short2 operator+(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline short2 operator+(short x, const short2& y) {
  return y + x;
}

/* short3 addition (+) */
inline short3 operator+(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline short3 operator+(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline short3 operator+(short x, const short3& y) {
  return y + x;
}

/* short4 addition (+) */
inline short4 operator+(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline short4 operator+(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline short4 operator+(short x, const short4& y) {
  return y + x;
}

/* short8 addition (+) */
inline short8 operator+(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline short8 operator+(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline short8 operator+(short x, const short8& y) {
  return y + x;
}

/* short16 addition (+) */
inline short16 operator+(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline short16 operator+(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline short16 operator+(short x, const short16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* short2 subtraction (-) */
inline short2 operator-(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline short2 operator-(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline short2 operator-(short x, const short2& y) {
  short2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* short3 subtraction (-) */
inline short3 operator-(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline short3 operator-(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline short3 operator-(short x, const short3& y) {
  short3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* short4 subtraction (-) */
inline short4 operator-(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline short4 operator-(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline short4 operator-(short x, const short4& y) {
  short4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* short8 subtraction (-) */
inline short8 operator-(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline short8 operator-(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline short8 operator-(short x, const short8& y) {
  short8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* short16 subtraction (-) */
inline short16 operator-(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline short16 operator-(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline short16 operator-(short x, const short16& y) {
  short16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* short2 multiplication (*) */
inline short2 operator*(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline short2 operator*(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline short2 operator*(short x, const short2& y) {
  return y * x;
}

/* short3 multiplication (*) */
inline short3 operator*(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline short3 operator*(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline short3 operator*(short x, const short3& y) {
  return y + x;
}


/* short4 multiplication (*) */
inline short4 operator*(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline short4 operator*(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline short4 operator*(short x, const short4& y) {
  return y + x;
}

/* short8 multiplication (*) */
inline short8 operator*(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline short8 operator*(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline short8 operator*(short x, const short8& y) {
  return y * x;
}

/* short16 multiplication (*) */
inline short16 operator*(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline short16 operator*(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline short16 operator*(short x, const short16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* short2 division (/) */
inline short2 operator/(const short2& x, const short2& y) {
  short2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline short2 operator/(const short2& x, short y) {
  short2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline short2 operator/(short x, const short2& y) {
  short2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* short3 division (/) */
inline short3 operator/(const short3& x, const short3& y) {
  short3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline short3 operator/(const short3& x, short y) {
  short3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline short3 operator/(short x, const short3& y) {
  short3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* short4 division (/) */
inline short4 operator/(const short4& x, const short4& y) {
  short4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline short4 operator/(const short4& x, short y) {
  short4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline short4 operator/(short x, const short4& y) {
  short4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* short8 division (/) */
inline short8 operator/(const short8& x, const short8& y) {
  short8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline short8 operator/(const short8& x, short y) {
  short8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline short8 operator/(short x, const short8& y) {
  short8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* short16 division (/) */
inline short16 operator/(const short16& x, const short16& y) {
  short16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline short16 operator/(const short16& x, short y) {
  short16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline short16 operator/(short x, const short16& y) {
  short16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* short2 remainder (%) */
inline short2 operator%(const short2& x, const short2& y) {
  short2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline short2 operator%(const short2& x, short y) {
  short2 rst = __CL_SAFE_INT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline short2 operator%(short x, const short2& y) {
  short2 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* short3 remainder (%) */
inline short3 operator%(const short3& x, const short3& y) {
  short3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline short3 operator%(const short3& x, short y) {
  short3 rst = __CL_SAFE_INT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline short3 operator%(short x, const short3& y) {
  short3 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* short4 remainder (%) */
inline short4 operator%(const short4& x, const short4& y) {
  short4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline short4 operator%(const short4& x, short y) {
  short4 rst = __CL_SAFE_INT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline short4 operator%(short x, const short4& y) {
  short4 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* short8 remainder (%) */
inline short8 operator%(const short8& x, const short8& y) {
  short8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline short8 operator%(const short8& x, short y) {
  short8 rst = __CL_SAFE_INT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline short8 operator%(short x, const short8& y) {
  short8 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* short16 remainder (%) */
inline short16 operator%(const short16& x, const short16& y) {
  short16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline short16 operator%(const short16& x, short y) {
  short16 rst = __CL_SAFE_INT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline short16 operator%(short x, const short16& y) {
  short16 rst = __CL_SAFE_INT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* short2 unary positive (+) */
inline short2 operator+(const short2& x) {
  return x;
}
/* short3 unary positive (+) */
inline short3 operator+(const short3& x) {
  return x;
}

/* short4 unary positive (+) */
inline short4 operator+(const short4& x) {
  return x;
}

/* short8 unary positive (+) */
inline short8 operator+(const short8& x) {
  return x;
}

/* short16 unary positive (+) */
inline short16 operator+(const short16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* short2 unary negative (-) */
inline short2 operator-(const short2& x) {
  short2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* short3 unary negative (-) */
inline short3 operator-(const short3& x) {
  short3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* short4 unary negative (-) */
inline short4 operator-(const short4& x) {
  short4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* short8 unary negative (-) */
inline short8 operator-(const short8& x) {
  short8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* short16 unary negative (-) */
inline short16 operator-(const short16& x) {
  short16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* short2 unary post-increment (++) */
inline short2 operator++(short2 &x, int n) {
  short2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* short3 unary post-increment (++) */
inline short3 operator++(short3 &x, int n) {
  short3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* short4 unary post-increment (++) */
inline short4 operator++(short4 &x, int n) {
  short4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* short8 unary post-increment (++) */
inline short8 operator++(short8 &x, int n) {
  short8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* short16 unary post-increment (++) */
inline short16 operator++(short16 &x, int n) {
  short16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* short2 unary pre-increment (++) */
inline short2 operator++(short2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* short3 unary pre-increment (++) */
inline short3 operator++(short3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* short4 unary pre-increment (++) */
inline short4 operator++(short4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* short8 unary pre-increment (++) */
inline short8 operator++(short8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* short16 unary pre-increment (++) */
inline short16 operator++(short16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* short2 unary post-decrement (--) */
inline short2 operator--(short2 &x, int n) {
  short2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* short3 unary post-decrement (--) */
inline short3 operator--(short3 &x, int n) {
  short3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* short4 unary post-decrement (--) */
inline short4 operator--(short4 &x, int n) {
  short4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* short8 unary post-decrement (--) */
inline short8 operator--(short8 &x, int n) {
  short8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* short16 unary post-decrement (--) */
inline short16 operator--(short16 &x, int n) {
  short16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* short2 unary pre-decrement (--) */
inline short2 operator--(short2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* short3 unary pre-decrement (--) */
inline short3 operator--(short3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* short4 unary pre-decrement (--) */
inline short4 operator--(short4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* short8 unary pre-decrement (--) */
inline short8 operator--(short8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* short16 unary pre-decrement (--) */
inline short16 operator--(short16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* short2 relational greater than (>) */
inline short2 operator>(const short2& x, const short2& y) {
  short2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline short2 operator>(const short2& x, short y) {
  short2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline short2 operator>(short x, const short2& y) {
  short2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* short3 relational greater than (>) */
inline short3 operator>(const short3& x, const short3& y) {
  short3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline short3 operator>(const short3& x, short y) {
  short3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline short3 operator>(short x, const short3& y) {
  short3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* short4 relational greater than (>) */
inline short4 operator>(const short4& x, const short4& y) {
  short4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline short4 operator>(const short4& x, short y) {
  short4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline short4 operator>(short x, const short4& y) {
  short4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* short8 relational greater than (>) */
inline short8 operator>(const short8& x, const short8& y) {
  short8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline short8 operator>(const short8& x, short y) {
  short8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline short8 operator>(short x, const short8& y) {
  short8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* short16 relational greater than (>) */
inline short16 operator>(const short16& x, const short16& y) {
  short16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline short16 operator>(const short16& x, short y) {
  short16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline short16 operator>(short x, const short16& y) {
  short16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* short2 relational less than (<) */
inline short2 operator<(const short2& x, const short2& y) {
  return y > x;
}
inline short2 operator<(const short2& x, short y) {
  return y > x;
}
inline short2 operator<(short x, const short2& y) {
  return y > x;
}

/* short3 relational less than (<) */
inline short3 operator<(const short3& x, const short3& y) {
  return y > x;
}
inline short3 operator<(const short3& x, short y) {
  return y > x;
}
inline short3 operator<(short x, const short3& y) {
  return y > x;
}

/* short4 relational less than (<) */
inline short4 operator<(const short4& x, const short4& y) {
  return y > x;
}
inline short4 operator<(const short4& x, short y) {
  return y > x;
}
inline short4 operator<(short x, const short4& y) {
  return y > x;
}

/* short8 relational less than (<) */
inline short8 operator<(const short8& x, const short8& y) {
  return y > x;
}
inline short8 operator<(const short8& x, short y) {
  return y > x;
}
inline short8 operator<(short x, const short8& y) {
  return y > x;
}

/* short16 relational less than (<) */
inline short16 operator<(const short16& x, const short16& y) {
  return y > x;
}
inline short16 operator<(const short16& x, short y) {
  return y > x;
}
inline short16 operator<(short x, const short16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* short2 relational greater than or equal (>=) */
inline short2 operator>=(const short2& x, const short2& y) {
  short2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline short2 operator>=(const short2& x, short y) {
  short2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline short2 operator>=(short x, const short2& y) {
  short2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* short3 relational greater than or equal (>=) */
inline short3 operator>=(const short3& x, const short3& y) {
  short3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline short3 operator>=(const short3& x, short y) {
  short3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline short3 operator>=(short x, const short3& y) {
  short3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* short4 relational greater than or equal (>=) */
inline short4 operator>=(const short4& x, const short4& y) {
  short4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline short4 operator>=(const short4& x, short y) {
  short4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline short4 operator>=(short x, const short4& y) {
  short4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* short8 relational greater than or equal (>=) */
inline short8 operator>=(const short8& x, const short8& y) {
  short8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline short8 operator>=(const short8& x, short y) {
  short8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline short8 operator>=(short x, const short8& y) {
  short8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* short16 relational greater than or equal (>=) */
inline short16 operator>=(const short16& x, const short16& y) {
  short16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline short16 operator>=(const short16& x, short y) {
  short16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline short16 operator>=(short x, const short16& y) {
  short16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* short2 relational less than or equal (<=) */
inline short2 operator<=(const short2& x, const short2& y) {
  return y >= x;
}
inline short2 operator<=(const short2& x, short y) {
  return y >= x;
}
inline short2 operator<=(short x, const short2& y) {
  return y >= x;
}

/* short3 relational less than or equal (<=) */
inline short3 operator<=(const short3& x, const short3& y) {
  return y >= x;
}
inline short3 operator<=(const short3& x, short y) {
  return y >= x;
}
inline short3 operator<=(short x, const short3& y) {
  return y >= x;
}

/* short4 relational less than or equal (<=) */
inline short4 operator<=(const short4& x, const short4& y) {
  return y >= x;
}
inline short4 operator<=(const short4& x, short y) {
  return y >= x;
}
inline short4 operator<=(short x, const short4& y) {
  return y >= x;
}

/* short8 relational less than or equal (<=) */
inline short8 operator<=(const short8& x, const short8& y) {
  return y >= x;
}
inline short8 operator<=(const short8& x, short y) {
  return y >= x;
}
inline short8 operator<=(short x, const short8& y) {
  return y >= x;
}

/* short16 relational less than or equal (<=) */
inline short16 operator<=(const short16& x, const short16& y) {
  return y >= x;
}
inline short16 operator<=(const short16& x, short y) {
  return y >= x;
}
inline short16 operator<=(short x, const short16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* short2 equal (==) */
inline short2 operator==(const short2& x, const short2& y) {
  short2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline short2 operator==(const short2& x, short y) {
  short2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline short2 operator==(short x, const short2& y) {
  short2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* short3 equal (==) */
inline short3 operator==(const short3& x, const short3& y) {
  short3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline short3 operator==(const short3& x, short y) {
  short3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline short3 operator==(short x, const short3& y) {
  short3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* short4 equal (==) */
inline short4 operator==(const short4& x, const short4& y) {
  short4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline short4 operator==(const short4& x, short y) {
  short4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline short4 operator==(short x, const short4& y) {
  short4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* short8 equal (==) */
inline short8 operator==(const short8& x, const short8& y) {
  short8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline short8 operator==(const short8& x, short y) {
  short8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline short8 operator==(short x, const short8& y) {
  short8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* short16 equal (==) */
inline short16 operator==(const short16& x, const short16& y) {
  short16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline short16 operator==(const short16& x, short y) {
  short16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline short16 operator==(short x, const short16& y) {
  short16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* short2 not equal (!=) */
inline short2 operator!=(const short2& x, const short2& y) {
  short2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline short2 operator!=(const short2& x, short y) {
  short2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline short2 operator!=(short x, const short2& y) {
  short2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* short3 not equal (!=) */
inline short3 operator!=(const short3& x, const short3& y) {
  short3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline short3 operator!=(const short3& x, short y) {
  short3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline short3 operator!=(short x, const short3& y) {
  short3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* short4 not equal (!=) */
inline short4 operator!=(const short4& x, const short4& y) {
  short4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline short4 operator!=(const short4& x, short y) {
  short4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline short4 operator!=(short x, const short4& y) {
  short4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* short8 not equal (!=) */
inline short8 operator!=(const short8& x, const short8& y) {
  short8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline short8 operator!=(const short8& x, short y) {
  short8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline short8 operator!=(short x, const short8& y) {
  short8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* short16 not equal (!=) */
inline short16 operator!=(const short16& x, const short16& y) {
  short16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline short16 operator!=(const short16& x, short y) {
  short16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline short16 operator!=(short x, const short16& y) {
  short16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* short2 bitwise and (&) */
inline short2 operator&(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline short2 operator&(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline short2 operator&(short x, const short2& y) {
  return y & x;
}

/* short3 bitwise and (&) */
inline short3 operator&(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline short3 operator&(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline short3 operator&(short x, const short3& y) {
  return y & x;
}


/* short4 bitwise and (&) */
inline short4 operator&(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline short4 operator&(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline short4 operator&(short x, const short4& y) {
  return y & x;
}

/* short8 bitwise and (&) */
inline short8 operator&(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline short8 operator&(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline short8 operator&(short x, const short8& y) {
  return y & x;
}

/* short16 bitwise and (&) */
inline short16 operator&(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline short16 operator&(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline short16 operator&(short x, const short16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* short2 bitwise or (|) */
inline short2 operator|(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline short2 operator|(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline short2 operator|(short x, const short2& y) {
  return y | x;
}

/* short3 bitwise or (|) */
inline short3 operator|(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline short3 operator|(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline short3 operator|(short x, const short3& y) {
  return y | x;
}

/* short4 bitwise or (|) */
inline short4 operator|(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline short4 operator|(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline short4 operator|(short x, const short4& y) {
  return y | x;
}

/* short8 bitwise or (|) */
inline short8 operator|(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline short8 operator|(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline short8 operator|(short x, const short8& y) {
  return y | x;
}

/* short16 bitwise or (|) */
inline short16 operator|(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline short16 operator|(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline short16 operator|(short x, const short16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* short2 bitwise exclusive or (^) */
inline short2 operator^(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline short2 operator^(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline short2 operator^(short x, const short2& y) {
  return y ^ x;
}

/* short3 bitwise exclusive or (^) */
inline short3 operator^(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline short3 operator^(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline short3 operator^(short x, const short3& y) {
  return y ^ x;
}

/* short4 bitwise exclusive or (^) */
inline short4 operator^(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline short4 operator^(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline short4 operator^(short x, const short4& y) {
  return y ^ x;
}

/* short8 bitwise exclusive or (^) */
inline short8 operator^(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline short8 operator^(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline short8 operator^(short x, const short8& y) {
  return y ^ x;
}

/* short16 bitwise exclusive or (^) */
inline short16 operator^(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline short16 operator^(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline short16 operator^(short x, const short16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* short2 bitwise not (~) */
inline short2 operator~(const short2& x) {
  short2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* short3 bitwise not (~) */
inline short3 operator~(const short3& x) {
  short3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* short4 bitwise not (~) */
inline short4 operator~(const short4& x) {
  short4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* short8 bitwise not (~) */
inline short8 operator~(const short8& x) {
  short8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* short16 bitwise not (~) */
inline short16 operator~(const short16& x) {
  short16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* short2 logical and (&&) */
inline short2 operator&&(const short2& x, const short2& y) {
  short2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline short2 operator&&(const short2& x, short y) {
  short2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline short2 operator&&(short x, const short2& y) {
  short2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* short3 logical and (&&) */
inline short3 operator&&(const short3& x, const short3& y) {
  short3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline short3 operator&&(const short3& x, short y) {
  short3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline short3 operator&&(short x, const short3& y) {
  short3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* short4 logical and (&&) */
inline short4 operator&&(const short4& x, const short4& y) {
  short4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline short4 operator&&(const short4& x, short y) {
  short4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline short4 operator&&(short x, const short4& y) {
  short4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* short8 logical and (&&) */
inline short8 operator&&(const short8& x, const short8& y) {
  short8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline short8 operator&&(const short8& x, short y) {
  short8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline short8 operator&&(short x, const short8& y) {
  short8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* short16 logical and (&&) */
inline short16 operator&&(const short16& x, const short16& y) {
  short16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline short16 operator&&(const short16& x, short y) {
  short16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline short16 operator&&(short x, const short16& y) {
  short16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* short2 logical or (||) */
inline short2 operator||(const short2& x, const short2& y) {
  short2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline short2 operator||(const short2& x, short y) {
  short2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline short2 operator||(short x, const short2& y) {
  short2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* short3 logical or (||) */
inline short3 operator||(const short3& x, const short3& y) {
  short3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline short3 operator||(const short3& x, short y) {
  short3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline short3 operator||(short x, const short3& y) {
  short3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* short4 logical or (||) */
inline short4 operator||(const short4& x, const short4& y) {
  short4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline short4 operator||(const short4& x, short y) {
  short4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline short4 operator||(short x, const short4& y) {
  short4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* short8 logical or (||) */
inline short8 operator||(const short8& x, const short8& y) {
  short8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline short8 operator||(const short8& x, short y) {
  short8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline short8 operator||(short x, const short8& y) {
  short8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* short16 logical or (||) */
inline short16 operator||(const short16& x, const short16& y) {
  short16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline short16 operator||(const short16& x, short y) {
  short16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline short16 operator||(short x, const short16& y) {
  short16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* short2 logical not (!) */
inline short2 operator!(const short2& x) {
  short2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* short3 logical not (!) */
inline short3 operator!(const short3& x) {
  short3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* short4 logical not (!) */
inline short4 operator!(const short4& x) {
  short4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* short8 logical not (!) */
inline short8 operator!(const short8& x) {
  short8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* short16 logical not (!) */
inline short16 operator!(const short16& x) {
  short16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* short2 right-shift (>>) */
inline short2 operator>>(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, >>, (y & 0xF));
  return rst;
}
inline short2 operator>>(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, >>, (y & 0xF));
  return rst;
}
/* short3 right-shift (>>) */
inline short3 operator>>(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, >>, (y & 0xF));
  return rst;
}
inline short3 operator>>(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, >>, (y & 0xF));
  return rst;
}

/* short4 right-shift (>>) */
inline short4 operator>>(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, >>, (y & 0xF));
  return rst;
}
inline short4 operator>>(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, >>, (y & 0xF));
  return rst;
}

/* short8 right-shift (>>) */
inline short8 operator>>(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, >>, (y & 0xF));
  return rst;
}
inline short8 operator>>(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, >>, (y & 0xF));
  return rst;
}

/* short16 right-shift (>>) */
inline short16 operator>>(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, >>, (y & 0xF));
  return rst;
}
inline short16 operator>>(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, >>, (y & 0xF));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* short2 left-shift (<<) */
inline short2 operator<<(const short2& x, const short2& y) {
  short2 rst = __CL_V2_OP_V2(x, <<, (y & 0xF));
  return rst;
}
inline short2 operator<<(const short2& x, short y) {
  short2 rst = __CL_V2_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* short3 left-shift (<<) */
inline short3 operator<<(const short3& x, const short3& y) {
  short3 rst = __CL_V3_OP_V3(x, <<, (y & 0xF));
  return rst;
}
inline short3 operator<<(const short3& x, short y) {
  short3 rst = __CL_V3_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* short4 left-shift (<<) */
inline short4 operator<<(const short4& x, const short4& y) {
  short4 rst = __CL_V4_OP_V4(x, <<, (y & 0xF));
  return rst;
}
inline short4 operator<<(const short4& x, short y) {
  short4 rst = __CL_V4_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* short8 left-shift (<<) */
inline short8 operator<<(const short8& x, const short8& y) {
  short8 rst = __CL_V8_OP_V8(x, <<, (y & 0xF));
  return rst;
}
inline short8 operator<<(const short8& x, short y) {
  short8 rst = __CL_V8_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* short16 left-shift (<<) */
inline short16 operator<<(const short16& x, const short16& y) {
  short16 rst = __CL_V16_OP_V16(x, <<, (y & 0xF));
  return rst;
}
inline short16 operator<<(const short16& x, short y) {
  short16 rst = __CL_V16_OP_S(x, <<, (y & 0xF));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* short2 assignment add into (+=) */
inline short2 operator+=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, +, y);
}
inline short2 operator+=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, +, y);
}

/* short3 assignment add into (+=) */
inline short3 operator+=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, +, y);
}
inline short3 operator+=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, +, y);
}

/* short4 assignment add into (+=) */
inline short4 operator+=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, +, y);
}
inline short4 operator+=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, +, y);
}

/* short8 assignment add into (+=) */
inline short8 operator+=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, +, y);
}
inline short8 operator+=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, +, y);
}

/* short16 assignment add into (+=) */
inline short16 operator+=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, +, y);
}
inline short16 operator+=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* short2 assignment subtract from (-=) */
inline short2 operator-=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, -, y);
}
inline short2 operator-=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, -, y);
}

/* short3 assignment subtract from (-=) */
inline short3 operator-=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, -, y);
}
inline short3 operator-=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, -, y);
}

/* short4 assignment subtract from (-=) */
inline short4 operator-=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, -, y);
}
inline short4 operator-=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, -, y);
}

/* short8 assignment subtract from (-=) */
inline short8 operator-=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, -, y);
}
inline short8 operator-=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, -, y);
}

/* short16 assignment subtract from (-=) */
inline short16 operator-=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, -, y);
}
inline short16 operator-=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* short2 assignment multiply into (*=) */
inline short2 operator*=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, *, y);
}
inline short2 operator*=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, *, y);
}

/* short3 assignment multiply into (*=) */
inline short3 operator*=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, *, y);
}
inline short3 operator*=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, *, y);
}

/* short4 assignment multiply into (*=) */
inline short4 operator*=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, *, y);
}
inline short4 operator*=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, *, y);
}

/* short8 assignment multiply into (*=) */
inline short8 operator*=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, *, y);
}
inline short8 operator*=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, *, y);
}

/* short16 assignment multiply into (*=) */
inline short16 operator*=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, *, y);
}
inline short16 operator*=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* short2 assignment divide into (/=) */
inline short2 operator/=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, /, y);
}
inline short2 operator/=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, /, y);
}

/* short3 assignment divide into (/=) */
inline short3 operator/=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, /, y);
}
inline short3 operator/=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, /, y);
}

/* short4 assignment divide into (/=) */
inline short4 operator/=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, /, y);
}
inline short4 operator/=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, /, y);
}

/* short8 assignment divide into (/=) */
inline short8 operator/=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, /, y);
}
inline short8 operator/=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, /, y);
}

/* short16 assignment divide into (/=) */
inline short16 operator/=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, /, y);
}
inline short16 operator/=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* short2 assignment modulus into (%=) */
inline short2 operator%=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, %, y);
}
inline short2 operator%=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, %, y);
}

/* short3 assignment modulus into (%=) */
inline short3 operator%=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, %, y);
}
inline short3 operator%=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, %, y);
}

/* short4 assignment modulus into (%=) */
inline short4 operator%=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, %, y);
}
inline short4 operator%=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, %, y);
}

/* short8 assignment modulus into (%=) */
inline short8 operator%=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, %, y);
}
inline short8 operator%=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, %, y);
}

/* short16 assignment modulus into (%=) */
inline short16 operator%=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, %, y);
}
inline short16 operator%=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* short2 assignment left shift by (<<=) */
inline short2 operator<<=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, <<, y);
}
inline short2 operator<<=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, <<, y);
}

/* short3 assignment left shift by (<<=) */
inline short3 operator<<=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, <<, y);
}
inline short3 operator<<=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, <<, y);
}


/* short4 assignment left shift by (<<=) */
inline short4 operator<<=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, <<, y);
}
inline short4 operator<<=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, <<, y);
}

/* short8 assignment left shift by (<<=) */
inline short8 operator<<=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, <<, y);
}
inline short8 operator<<=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, <<, y);
}

/* short16 assignment left shift by (<<=) */
inline short16 operator<<=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, <<, y);
}
inline short16 operator<<=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* short2 assignment right shift by (>>=) */
inline short2 operator>>=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, >>, y);
}
inline short2 operator>>=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, >>, y);
}

/* short3 assignment right shift by (>>=) */
inline short3 operator>>=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, >>, y);
}
inline short3 operator>>=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, >>, y);
}
/* short4 assignment right shift by (>>=) */
inline short4 operator>>=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, >>, y);
}
inline short4 operator>>=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, >>, y);
}

/* short8 assignment right shift by (>>=) */
inline short8 operator>>=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, >>, y);
}
inline short8 operator>>=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, >>, y);
}

/* short16 assignment right shift by (>>=) */
inline short16 operator>>=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, >>, y);
}
inline short16 operator>>=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* short2 assignment and into (&=) */
inline short2 operator&=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, &, y);
}
inline short2 operator&=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, &, y);
}

/* short3 assignment and into (&=) */
inline short3 operator&=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, &, y);
}
inline short3 operator&=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, &, y);
}

/* short4 assignment and into (&=) */
inline short4 operator&=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, &, y);
}
inline short4 operator&=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, &, y);
}

/* short8 assignment and into (&=) */
inline short8 operator&=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, &, y);
}
inline short8 operator&=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, &, y);
}

/* short16 assignment and into (&=) */
inline short16 operator&=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, &, y);
}
inline short16 operator&=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* short2 assignment inclusive or into (|=) */
inline short2 operator|=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, |, y);
}
inline short2 operator|=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, |, y);
}

/* short3 assignment inclusive or into (|=) */
inline short3 operator|=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, |, y);
}
inline short3 operator|=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, |, y);

}
/* short4 assignment inclusive or into (|=) */
inline short4 operator|=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, |, y);
}
inline short4 operator|=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, |, y);
}

/* short8 assignment inclusive or into (|=) */
inline short8 operator|=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, |, y);
}
inline short8 operator|=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, |, y);
}

/* short16 assignment inclusive or into (|=) */
inline short16 operator|=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, |, y);
}
inline short16 operator|=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* short2 assignment exclusive or into (^=) */
inline short2 operator^=(short2 &x, const short2& y) {
  __CL_V_OP_ASSIGN_V(short2, x, ^, y);
}
inline short2 operator^=(short2 &x, short y) {
  __CL_V_OP_ASSIGN_V(short2, x, ^, y);
}

/* short3 assignment exclusive or into (^=) */
inline short3 operator^=(short3 &x, const short3& y) {
  __CL_V_OP_ASSIGN_V(short3, x, ^, y);
}
inline short3 operator^=(short3 &x, short y) {
  __CL_V_OP_ASSIGN_V(short3, x, ^, y);
}

/* short4 assignment exclusive or into (^=) */
inline short4 operator^=(short4 &x, const short4& y) {
  __CL_V_OP_ASSIGN_V(short4, x, ^, y);
}
inline short4 operator^=(short4 &x, short y) {
  __CL_V_OP_ASSIGN_V(short4, x, ^, y);
}

/* short8 assignment exclusive or into (^=) */
inline short8 operator^=(short8 &x, const short8& y) {
  __CL_V_OP_ASSIGN_V(short8, x, ^, y);
}
inline short8 operator^=(short8 &x, short y) {
  __CL_V_OP_ASSIGN_V(short8, x, ^, y);
}

/* short16 assignment exclusive or into (^=) */
inline short16 operator^=(short16 &x, const short16& y) {
  __CL_V_OP_ASSIGN_V(short16, x, ^, y);
}
inline short16 operator^=(short16 &x, short y) {
  __CL_V_OP_ASSIGN_V(short16, x, ^, y);
}
#endif //__CL_OPS_SHORTN_H

