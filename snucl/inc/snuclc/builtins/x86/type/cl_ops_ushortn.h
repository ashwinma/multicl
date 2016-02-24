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

#ifndef __CL_OPS_USHORTN_H
#define __CL_OPS_USHORTN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>

///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* ushort2 addition (+) */
inline ushort2 operator+(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline ushort2 operator+(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline ushort2 operator+(ushort x, const ushort2& y) {
  return y + x;
}

/* ushort3 addition (+) */
inline ushort3 operator+(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline ushort3 operator+(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline ushort3 operator+(ushort x, const ushort3& y) {
  return y + x;
}

/* ushort4 addition (+) */
inline ushort4 operator+(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline ushort4 operator+(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline ushort4 operator+(ushort x, const ushort4& y) {
  return y + x;
}

/* ushort8 addition (+) */
inline ushort8 operator+(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline ushort8 operator+(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline ushort8 operator+(ushort x, const ushort8& y) {
  return y + x;
}

/* ushort16 addition (+) */
inline ushort16 operator+(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline ushort16 operator+(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline ushort16 operator+(ushort x, const ushort16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* ushort2 subtraction (-) */
inline ushort2 operator-(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline ushort2 operator-(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline ushort2 operator-(ushort x, const ushort2& y) {
  ushort2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* ushort3 subtraction (-) */
inline ushort3 operator-(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline ushort3 operator-(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline ushort3 operator-(ushort x, const ushort3& y) {
  ushort3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* ushort4 subtraction (-) */
inline ushort4 operator-(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline ushort4 operator-(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline ushort4 operator-(ushort x, const ushort4& y) {
  ushort4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* ushort8 subtraction (-) */
inline ushort8 operator-(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline ushort8 operator-(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline ushort8 operator-(ushort x, const ushort8& y) {
  ushort8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* ushort16 subtraction (-) */
inline ushort16 operator-(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline ushort16 operator-(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline ushort16 operator-(ushort x, const ushort16& y) {
  ushort16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* ushort2 multiplication (*) */
inline ushort2 operator*(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline ushort2 operator*(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline ushort2 operator*(ushort x, const ushort2& y) {
  return y * x;
}

/* ushort3 multiplication (*) */
inline ushort3 operator*(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline ushort3 operator*(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline ushort3 operator*(ushort x, const ushort3& y) {
  return y + x;
}


/* ushort4 multiplication (*) */
inline ushort4 operator*(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline ushort4 operator*(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline ushort4 operator*(ushort x, const ushort4& y) {
  return y + x;
}

/* ushort8 multiplication (*) */
inline ushort8 operator*(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline ushort8 operator*(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline ushort8 operator*(ushort x, const ushort8& y) {
  return y * x;
}

/* ushort16 multiplication (*) */
inline ushort16 operator*(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline ushort16 operator*(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline ushort16 operator*(ushort x, const ushort16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* ushort2 division (/) */
inline ushort2 operator/(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline ushort2 operator/(const ushort2& x, ushort y) {
  ushort2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline ushort2 operator/(ushort x, const ushort2& y) {
  ushort2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* ushort3 division (/) */
inline ushort3 operator/(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline ushort3 operator/(const ushort3& x, ushort y) {
  ushort3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline ushort3 operator/(ushort x, const ushort3& y) {
  ushort3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* ushort4 division (/) */
inline ushort4 operator/(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline ushort4 operator/(const ushort4& x, ushort y) {
  ushort4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline ushort4 operator/(ushort x, const ushort4& y) {
  ushort4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* ushort8 division (/) */
inline ushort8 operator/(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline ushort8 operator/(const ushort8& x, ushort y) {
  ushort8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline ushort8 operator/(ushort x, const ushort8& y) {
  ushort8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* ushort16 division (/) */
inline ushort16 operator/(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline ushort16 operator/(const ushort16& x, ushort y) {
  ushort16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline ushort16 operator/(ushort x, const ushort16& y) {
  ushort16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* ushort2 remainder (%) */
inline ushort2 operator%(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline ushort2 operator%(const ushort2& x, ushort y) {
  ushort2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline ushort2 operator%(ushort x, const ushort2& y) {
  ushort2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* ushort3 remainder (%) */
inline ushort3 operator%(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline ushort3 operator%(const ushort3& x, ushort y) {
  ushort3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline ushort3 operator%(ushort x, const ushort3& y) {
  ushort3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* ushort4 remainder (%) */
inline ushort4 operator%(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline ushort4 operator%(const ushort4& x, ushort y) {
  ushort4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline ushort4 operator%(ushort x, const ushort4& y) {
  ushort4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* ushort8 remainder (%) */
inline ushort8 operator%(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline ushort8 operator%(const ushort8& x, ushort y) {
  ushort8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline ushort8 operator%(ushort x, const ushort8& y) {
  ushort8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* ushort16 remainder (%) */
inline ushort16 operator%(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline ushort16 operator%(const ushort16& x, ushort y) {
  ushort16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline ushort16 operator%(ushort x, const ushort16& y) {
  ushort16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* ushort2 unary positive (+) */
inline ushort2 operator+(const ushort2& x) {
  return x;
}
/* ushort3 unary positive (+) */
inline ushort3 operator+(const ushort3& x) {
  return x;
}

/* ushort4 unary positive (+) */
inline ushort4 operator+(const ushort4& x) {
  return x;
}

/* ushort8 unary positive (+) */
inline ushort8 operator+(const ushort8& x) {
  return x;
}

/* ushort16 unary positive (+) */
inline ushort16 operator+(const ushort16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* ushort2 unary negative (-) */
inline ushort2 operator-(const ushort2& x) {
  ushort2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* ushort3 unary negative (-) */
inline ushort3 operator-(const ushort3& x) {
  ushort3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* ushort4 unary negative (-) */
inline ushort4 operator-(const ushort4& x) {
  ushort4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* ushort8 unary negative (-) */
inline ushort8 operator-(const ushort8& x) {
  ushort8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* ushort16 unary negative (-) */
inline ushort16 operator-(const ushort16& x) {
  ushort16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* ushort2 unary post-increment (++) */
inline ushort2 operator++(ushort2 &x, int n) {
  ushort2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ushort3 unary post-increment (++) */
inline ushort3 operator++(ushort3 &x, int n) {
  ushort3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ushort4 unary post-increment (++) */
inline ushort4 operator++(ushort4 &x, int n) {
  ushort4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ushort8 unary post-increment (++) */
inline ushort8 operator++(ushort8 &x, int n) {
  ushort8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* ushort16 unary post-increment (++) */
inline ushort16 operator++(ushort16 &x, int n) {
  ushort16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* ushort2 unary pre-increment (++) */
inline ushort2 operator++(ushort2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* ushort3 unary pre-increment (++) */
inline ushort3 operator++(ushort3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* ushort4 unary pre-increment (++) */
inline ushort4 operator++(ushort4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* ushort8 unary pre-increment (++) */
inline ushort8 operator++(ushort8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* ushort16 unary pre-increment (++) */
inline ushort16 operator++(ushort16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* ushort2 unary post-decrement (--) */
inline ushort2 operator--(ushort2 &x, int n) {
  ushort2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* ushort3 unary post-decrement (--) */
inline ushort3 operator--(ushort3 &x, int n) {
  ushort3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* ushort4 unary post-decrement (--) */
inline ushort4 operator--(ushort4 &x, int n) {
  ushort4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* ushort8 unary post-decrement (--) */
inline ushort8 operator--(ushort8 &x, int n) {
  ushort8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* ushort16 unary post-decrement (--) */
inline ushort16 operator--(ushort16 &x, int n) {
  ushort16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* ushort2 unary pre-decrement (--) */
inline ushort2 operator--(ushort2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* ushort3 unary pre-decrement (--) */
inline ushort3 operator--(ushort3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* ushort4 unary pre-decrement (--) */
inline ushort4 operator--(ushort4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* ushort8 unary pre-decrement (--) */
inline ushort8 operator--(ushort8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* ushort16 unary pre-decrement (--) */
inline ushort16 operator--(ushort16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* ushort2 relational greater than (>) */
inline short2 operator>(const ushort2& x, const ushort2& y) {
  short2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline short2 operator>(const ushort2& x, ushort y) {
  short2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline short2 operator>(ushort x, const ushort2& y) {
  short2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* ushort3 relational greater than (>) */
inline short3 operator>(const ushort3& x, const ushort3& y) {
  short3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline short3 operator>(const ushort3& x, ushort y) {
  short3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline short3 operator>(ushort x, const ushort3& y) {
  short3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* ushort4 relational greater than (>) */
inline short4 operator>(const ushort4& x, const ushort4& y) {
  short4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline short4 operator>(const ushort4& x, ushort y) {
  short4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline short4 operator>(ushort x, const ushort4& y) {
  short4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* ushort8 relational greater than (>) */
inline short8 operator>(const ushort8& x, const ushort8& y) {
  short8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline short8 operator>(const ushort8& x, ushort y) {
  short8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline short8 operator>(ushort x, const ushort8& y) {
  short8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* ushort16 relational greater than (>) */
inline short16 operator>(const ushort16& x, const ushort16& y) {
  short16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline short16 operator>(const ushort16& x, ushort y) {
  short16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline short16 operator>(ushort x, const ushort16& y) {
  short16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* ushort2 relational less than (<) */
inline short2 operator<(const ushort2& x, const ushort2& y) {
  return y > x;
}
inline short2 operator<(const ushort2& x, ushort y) {
  return y > x;
}
inline short2 operator<(ushort x, const ushort2& y) {
  return y > x;
}

/* ushort3 relational less than (<) */
inline short3 operator<(const ushort3& x, const ushort3& y) {
  return y > x;
}
inline short3 operator<(const ushort3& x, ushort y) {
  return y > x;
}
inline short3 operator<(ushort x, const ushort3& y) {
  return y > x;
}

/* ushort4 relational less than (<) */
inline short4 operator<(const ushort4& x, const ushort4& y) {
  return y > x;
}
inline short4 operator<(const ushort4& x, ushort y) {
  return y > x;
}
inline short4 operator<(ushort x, const ushort4& y) {
  return y > x;
}

/* ushort8 relational less than (<) */
inline short8 operator<(const ushort8& x, const ushort8& y) {
  return y > x;
}
inline short8 operator<(const ushort8& x, ushort y) {
  return y > x;
}
inline short8 operator<(ushort x, const ushort8& y) {
  return y > x;
}

/* ushort16 relational less than (<) */
inline short16 operator<(const ushort16& x, const ushort16& y) {
  return y > x;
}
inline short16 operator<(const ushort16& x, ushort y) {
  return y > x;
}
inline short16 operator<(ushort x, const ushort16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* ushort2 relational greater than or equal (>=) */
inline short2 operator>=(const ushort2& x, const ushort2& y) {
  short2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline short2 operator>=(const ushort2& x, ushort y) {
  short2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline short2 operator>=(ushort x, const ushort2& y) {
  short2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* ushort3 relational greater than or equal (>=) */
inline short3 operator>=(const ushort3& x, const ushort3& y) {
  short3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline short3 operator>=(const ushort3& x, ushort y) {
  short3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline short3 operator>=(ushort x, const ushort3& y) {
  short3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* ushort4 relational greater than or equal (>=) */
inline short4 operator>=(const ushort4& x, const ushort4& y) {
  short4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline short4 operator>=(const ushort4& x, ushort y) {
  short4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline short4 operator>=(ushort x, const ushort4& y) {
  short4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* ushort8 relational greater than or equal (>=) */
inline short8 operator>=(const ushort8& x, const ushort8& y) {
  short8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline short8 operator>=(const ushort8& x, ushort y) {
  short8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline short8 operator>=(ushort x, const ushort8& y) {
  short8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* ushort16 relational greater than or equal (>=) */
inline short16 operator>=(const ushort16& x, const ushort16& y) {
  short16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline short16 operator>=(const ushort16& x, ushort y) {
  short16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline short16 operator>=(ushort x, const ushort16& y) {
  short16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* ushort2 relational less than or equal (<=) */
inline short2 operator<=(const ushort2& x, const ushort2& y) {
  return y >= x;
}
inline short2 operator<=(const ushort2& x, ushort y) {
  return y >= x;
}
inline short2 operator<=(ushort x, const ushort2& y) {
  return y >= x;
}

/* ushort3 relational less than or equal (<=) */
inline short3 operator<=(const ushort3& x, const ushort3& y) {
  return y >= x;
}
inline short3 operator<=(const ushort3& x, ushort y) {
  return y >= x;
}
inline short3 operator<=(ushort x, const ushort3& y) {
  return y >= x;
}

/* ushort4 relational less than or equal (<=) */
inline short4 operator<=(const ushort4& x, const ushort4& y) {
  return y >= x;
}
inline short4 operator<=(const ushort4& x, ushort y) {
  return y >= x;
}
inline short4 operator<=(ushort x, const ushort4& y) {
  return y >= x;
}

/* ushort8 relational less than or equal (<=) */
inline short8 operator<=(const ushort8& x, const ushort8& y) {
  return y >= x;
}
inline short8 operator<=(const ushort8& x, ushort y) {
  return y >= x;
}
inline short8 operator<=(ushort x, const ushort8& y) {
  return y >= x;
}

/* ushort16 relational less than or equal (<=) */
inline short16 operator<=(const ushort16& x, const ushort16& y) {
  return y >= x;
}
inline short16 operator<=(const ushort16& x, ushort y) {
  return y >= x;
}
inline short16 operator<=(ushort x, const ushort16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* ushort2 equal (==) */
inline short2 operator==(const ushort2& x, const ushort2& y) {
  short2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline short2 operator==(const ushort2& x, ushort y) {
  short2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline short2 operator==(ushort x, const ushort2& y) {
  short2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* ushort3 equal (==) */
inline short3 operator==(const ushort3& x, const ushort3& y) {
  short3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline short3 operator==(const ushort3& x, ushort y) {
  short3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline short3 operator==(ushort x, const ushort3& y) {
  short3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* ushort4 equal (==) */
inline short4 operator==(const ushort4& x, const ushort4& y) {
  short4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline short4 operator==(const ushort4& x, ushort y) {
  short4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline short4 operator==(ushort x, const ushort4& y) {
  short4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* ushort8 equal (==) */
inline short8 operator==(const ushort8& x, const ushort8& y) {
  short8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline short8 operator==(const ushort8& x, ushort y) {
  short8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline short8 operator==(ushort x, const ushort8& y) {
  short8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* ushort16 equal (==) */
inline short16 operator==(const ushort16& x, const ushort16& y) {
  short16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline short16 operator==(const ushort16& x, ushort y) {
  short16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline short16 operator==(ushort x, const ushort16& y) {
  short16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* ushort2 not equal (!=) */
inline short2 operator!=(const ushort2& x, const ushort2& y) {
  short2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline short2 operator!=(const ushort2& x, ushort y) {
  short2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline short2 operator!=(ushort x, const ushort2& y) {
  short2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* ushort3 not equal (!=) */
inline short3 operator!=(const ushort3& x, const ushort3& y) {
  short3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline short3 operator!=(const ushort3& x, ushort y) {
  short3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline short3 operator!=(ushort x, const ushort3& y) {
  short3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* ushort4 not equal (!=) */
inline short4 operator!=(const ushort4& x, const ushort4& y) {
  short4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline short4 operator!=(const ushort4& x, ushort y) {
  short4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline short4 operator!=(ushort x, const ushort4& y) {
  short4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* ushort8 not equal (!=) */
inline short8 operator!=(const ushort8& x, const ushort8& y) {
  short8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline short8 operator!=(const ushort8& x, ushort y) {
  short8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline short8 operator!=(ushort x, const ushort8& y) {
  short8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* ushort16 not equal (!=) */
inline short16 operator!=(const ushort16& x, const ushort16& y) {
  short16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline short16 operator!=(const ushort16& x, ushort y) {
  short16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline short16 operator!=(ushort x, const ushort16& y) {
  short16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* ushort2 bitwise and (&) */
inline ushort2 operator&(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline ushort2 operator&(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline ushort2 operator&(ushort x, const ushort2& y) {
  return y & x;
}

/* ushort3 bitwise and (&) */
inline ushort3 operator&(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline ushort3 operator&(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline ushort3 operator&(ushort x, const ushort3& y) {
  return y & x;
}


/* ushort4 bitwise and (&) */
inline ushort4 operator&(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline ushort4 operator&(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline ushort4 operator&(ushort x, const ushort4& y) {
  return y & x;
}

/* ushort8 bitwise and (&) */
inline ushort8 operator&(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline ushort8 operator&(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline ushort8 operator&(ushort x, const ushort8& y) {
  return y & x;
}

/* ushort16 bitwise and (&) */
inline ushort16 operator&(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline ushort16 operator&(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline ushort16 operator&(ushort x, const ushort16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* ushort2 bitwise or (|) */
inline ushort2 operator|(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline ushort2 operator|(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline ushort2 operator|(ushort x, const ushort2& y) {
  return y | x;
}

/* ushort3 bitwise or (|) */
inline ushort3 operator|(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline ushort3 operator|(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline ushort3 operator|(ushort x, const ushort3& y) {
  return y | x;
}

/* ushort4 bitwise or (|) */
inline ushort4 operator|(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline ushort4 operator|(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline ushort4 operator|(ushort x, const ushort4& y) {
  return y | x;
}

/* ushort8 bitwise or (|) */
inline ushort8 operator|(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline ushort8 operator|(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline ushort8 operator|(ushort x, const ushort8& y) {
  return y | x;
}

/* ushort16 bitwise or (|) */
inline ushort16 operator|(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline ushort16 operator|(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline ushort16 operator|(ushort x, const ushort16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* ushort2 bitwise exclusive or (^) */
inline ushort2 operator^(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline ushort2 operator^(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline ushort2 operator^(ushort x, const ushort2& y) {
  return y ^ x;
}

/* ushort3 bitwise exclusive or (^) */
inline ushort3 operator^(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline ushort3 operator^(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline ushort3 operator^(ushort x, const ushort3& y) {
  return y ^ x;
}

/* ushort4 bitwise exclusive or (^) */
inline ushort4 operator^(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline ushort4 operator^(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline ushort4 operator^(ushort x, const ushort4& y) {
  return y ^ x;
}

/* ushort8 bitwise exclusive or (^) */
inline ushort8 operator^(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline ushort8 operator^(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline ushort8 operator^(ushort x, const ushort8& y) {
  return y ^ x;
}

/* ushort16 bitwise exclusive or (^) */
inline ushort16 operator^(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline ushort16 operator^(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline ushort16 operator^(ushort x, const ushort16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* ushort2 bitwise not (~) */
inline ushort2 operator~(const ushort2& x) {
  ushort2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* ushort3 bitwise not (~) */
inline ushort3 operator~(const ushort3& x) {
  ushort3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* ushort4 bitwise not (~) */
inline ushort4 operator~(const ushort4& x) {
  ushort4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* ushort8 bitwise not (~) */
inline ushort8 operator~(const ushort8& x) {
  ushort8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* ushort16 bitwise not (~) */
inline ushort16 operator~(const ushort16& x) {
  ushort16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* ushort2 logical and (&&) */
inline short2 operator&&(const ushort2& x, const ushort2& y) {
  short2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline short2 operator&&(const ushort2& x, ushort y) {
  short2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline short2 operator&&(ushort x, const ushort2& y) {
  short2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* ushort3 logical and (&&) */
inline short3 operator&&(const ushort3& x, const ushort3& y) {
  short3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline short3 operator&&(const ushort3& x, ushort y) {
  short3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline short3 operator&&(ushort x, const ushort3& y) {
  short3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* ushort4 logical and (&&) */
inline short4 operator&&(const ushort4& x, const ushort4& y) {
  short4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline short4 operator&&(const ushort4& x, ushort y) {
  short4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline short4 operator&&(ushort x, const ushort4& y) {
  short4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* ushort8 logical and (&&) */
inline short8 operator&&(const ushort8& x, const ushort8& y) {
  short8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline short8 operator&&(const ushort8& x, ushort y) {
  short8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline short8 operator&&(ushort x, const ushort8& y) {
  short8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* ushort16 logical and (&&) */
inline short16 operator&&(const ushort16& x, const ushort16& y) {
  short16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline short16 operator&&(const ushort16& x, ushort y) {
  short16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline short16 operator&&(ushort x, const ushort16& y) {
  short16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* ushort2 logical or (||) */
inline short2 operator||(const ushort2& x, const ushort2& y) {
  short2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline short2 operator||(const ushort2& x, ushort y) {
  short2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline short2 operator||(ushort x, const ushort2& y) {
  short2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* ushort3 logical or (||) */
inline short3 operator||(const ushort3& x, const ushort3& y) {
  short3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline short3 operator||(const ushort3& x, ushort y) {
  short3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline short3 operator||(ushort x, const ushort3& y) {
  short3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* ushort4 logical or (||) */
inline short4 operator||(const ushort4& x, const ushort4& y) {
  short4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline short4 operator||(const ushort4& x, ushort y) {
  short4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline short4 operator||(ushort x, const ushort4& y) {
  short4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* ushort8 logical or (||) */
inline short8 operator||(const ushort8& x, const ushort8& y) {
  short8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline short8 operator||(const ushort8& x, ushort y) {
  short8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline short8 operator||(ushort x, const ushort8& y) {
  short8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* ushort16 logical or (||) */
inline short16 operator||(const ushort16& x, const ushort16& y) {
  short16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline short16 operator||(const ushort16& x, ushort y) {
  short16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline short16 operator||(ushort x, const ushort16& y) {
  short16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* ushort2 logical not (!) */
inline short2 operator!(const ushort2& x) {
  short2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* ushort3 logical not (!) */
inline short3 operator!(const ushort3& x) {
  short3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* ushort4 logical not (!) */
inline short4 operator!(const ushort4& x) {
  short4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* ushort8 logical not (!) */
inline short8 operator!(const ushort8& x) {
  short8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* ushort16 logical not (!) */
inline short16 operator!(const ushort16& x) {
  short16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* ushort2 right-shift (>>) */
inline ushort2 operator>>(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, >>, (y & 0xF));
  return rst;
}
inline ushort2 operator>>(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, >>, (y & 0xF));
  return rst;
}
/* ushort3 right-shift (>>) */
inline ushort3 operator>>(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, >>, (y & 0xF));
  return rst;
}
inline ushort3 operator>>(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, >>, (y & 0xF));
  return rst;
}

/* ushort4 right-shift (>>) */
inline ushort4 operator>>(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, >>, (y & 0xF));
  return rst;
}
inline ushort4 operator>>(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, >>, (y & 0xF));
  return rst;
}

/* ushort8 right-shift (>>) */
inline ushort8 operator>>(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, >>, (y & 0xF));
  return rst;
}
inline ushort8 operator>>(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, >>, (y & 0xF));
  return rst;
}

/* ushort16 right-shift (>>) */
inline ushort16 operator>>(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, >>, (y & 0xF));
  return rst;
}
inline ushort16 operator>>(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, >>, (y & 0xF));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* ushort2 left-shift (<<) */
inline ushort2 operator<<(const ushort2& x, const ushort2& y) {
  ushort2 rst = __CL_V2_OP_V2(x, <<, (y & 0xF));
  return rst;
}
inline ushort2 operator<<(const ushort2& x, ushort y) {
  ushort2 rst = __CL_V2_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* ushort3 left-shift (<<) */
inline ushort3 operator<<(const ushort3& x, const ushort3& y) {
  ushort3 rst = __CL_V3_OP_V3(x, <<, (y & 0xF));
  return rst;
}
inline ushort3 operator<<(const ushort3& x, ushort y) {
  ushort3 rst = __CL_V3_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* ushort4 left-shift (<<) */
inline ushort4 operator<<(const ushort4& x, const ushort4& y) {
  ushort4 rst = __CL_V4_OP_V4(x, <<, (y & 0xF));
  return rst;
}
inline ushort4 operator<<(const ushort4& x, ushort y) {
  ushort4 rst = __CL_V4_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* ushort8 left-shift (<<) */
inline ushort8 operator<<(const ushort8& x, const ushort8& y) {
  ushort8 rst = __CL_V8_OP_V8(x, <<, (y & 0xF));
  return rst;
}
inline ushort8 operator<<(const ushort8& x, ushort y) {
  ushort8 rst = __CL_V8_OP_S(x, <<, (y & 0xF));
  return rst;
}

/* ushort16 left-shift (<<) */
inline ushort16 operator<<(const ushort16& x, const ushort16& y) {
  ushort16 rst = __CL_V16_OP_V16(x, <<, (y & 0xF));
  return rst;
}
inline ushort16 operator<<(const ushort16& x, ushort y) {
  ushort16 rst = __CL_V16_OP_S(x, <<, (y & 0xF));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* ushort2 assignment add into (+=) */
inline ushort2 operator+=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, +, y);
}
inline ushort2 operator+=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, +, y);
}

/* ushort3 assignment add into (+=) */
inline ushort3 operator+=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, +, y);
}
inline ushort3 operator+=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, +, y);
}

/* ushort4 assignment add into (+=) */
inline ushort4 operator+=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, +, y);
}
inline ushort4 operator+=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, +, y);
}

/* ushort8 assignment add into (+=) */
inline ushort8 operator+=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, +, y);
}
inline ushort8 operator+=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, +, y);
}

/* ushort16 assignment add into (+=) */
inline ushort16 operator+=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, +, y);
}
inline ushort16 operator+=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* ushort2 assignment subtract from (-=) */
inline ushort2 operator-=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, -, y);
}
inline ushort2 operator-=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, -, y);
}

/* ushort3 assignment subtract from (-=) */
inline ushort3 operator-=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, -, y);
}
inline ushort3 operator-=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, -, y);
}

/* ushort4 assignment subtract from (-=) */
inline ushort4 operator-=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, -, y);
}
inline ushort4 operator-=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, -, y);
}

/* ushort8 assignment subtract from (-=) */
inline ushort8 operator-=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, -, y);
}
inline ushort8 operator-=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, -, y);
}

/* ushort16 assignment subtract from (-=) */
inline ushort16 operator-=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, -, y);
}
inline ushort16 operator-=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* ushort2 assignment multiply into (*=) */
inline ushort2 operator*=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, *, y);
}
inline ushort2 operator*=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, *, y);
}

/* ushort3 assignment multiply into (*=) */
inline ushort3 operator*=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, *, y);
}
inline ushort3 operator*=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, *, y);
}

/* ushort4 assignment multiply into (*=) */
inline ushort4 operator*=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, *, y);
}
inline ushort4 operator*=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, *, y);
}

/* ushort8 assignment multiply into (*=) */
inline ushort8 operator*=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, *, y);
}
inline ushort8 operator*=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, *, y);
}

/* ushort16 assignment multiply into (*=) */
inline ushort16 operator*=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, *, y);
}
inline ushort16 operator*=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* ushort2 assignment divide into (/=) */
inline ushort2 operator/=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, /, y);
}
inline ushort2 operator/=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, /, y);
}

/* ushort3 assignment divide into (/=) */
inline ushort3 operator/=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, /, y);
}
inline ushort3 operator/=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, /, y);
}

/* ushort4 assignment divide into (/=) */
inline ushort4 operator/=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, /, y);
}
inline ushort4 operator/=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, /, y);
}

/* ushort8 assignment divide into (/=) */
inline ushort8 operator/=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, /, y);
}
inline ushort8 operator/=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, /, y);
}

/* ushort16 assignment divide into (/=) */
inline ushort16 operator/=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, /, y);
}
inline ushort16 operator/=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* ushort2 assignment modulus into (%=) */
inline ushort2 operator%=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, %, y);
}
inline ushort2 operator%=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, %, y);
}

/* ushort3 assignment modulus into (%=) */
inline ushort3 operator%=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, %, y);
}
inline ushort3 operator%=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, %, y);
}

/* ushort4 assignment modulus into (%=) */
inline ushort4 operator%=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, %, y);
}
inline ushort4 operator%=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, %, y);
}

/* ushort8 assignment modulus into (%=) */
inline ushort8 operator%=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, %, y);
}
inline ushort8 operator%=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, %, y);
}

/* ushort16 assignment modulus into (%=) */
inline ushort16 operator%=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, %, y);
}
inline ushort16 operator%=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* ushort2 assignment left shift by (<<=) */
inline ushort2 operator<<=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, <<, y);
}
inline ushort2 operator<<=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, <<, y);
}

/* ushort3 assignment left shift by (<<=) */
inline ushort3 operator<<=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, <<, y);
}
inline ushort3 operator<<=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, <<, y);
}


/* ushort4 assignment left shift by (<<=) */
inline ushort4 operator<<=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, <<, y);
}
inline ushort4 operator<<=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, <<, y);
}

/* ushort8 assignment left shift by (<<=) */
inline ushort8 operator<<=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, <<, y);
}
inline ushort8 operator<<=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, <<, y);
}

/* ushort16 assignment left shift by (<<=) */
inline ushort16 operator<<=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, <<, y);
}
inline ushort16 operator<<=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* ushort2 assignment right shift by (>>=) */
inline ushort2 operator>>=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, >>, y);
}
inline ushort2 operator>>=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, >>, y);
}

/* ushort3 assignment right shift by (>>=) */
inline ushort3 operator>>=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, >>, y);
}
inline ushort3 operator>>=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, >>, y);
}
/* ushort4 assignment right shift by (>>=) */
inline ushort4 operator>>=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, >>, y);
}
inline ushort4 operator>>=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, >>, y);
}

/* ushort8 assignment right shift by (>>=) */
inline ushort8 operator>>=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, >>, y);
}
inline ushort8 operator>>=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, >>, y);
}

/* ushort16 assignment right shift by (>>=) */
inline ushort16 operator>>=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, >>, y);
}
inline ushort16 operator>>=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* ushort2 assignment and into (&=) */
inline ushort2 operator&=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, &, y);
}
inline ushort2 operator&=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, &, y);
}

/* ushort3 assignment and into (&=) */
inline ushort3 operator&=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, &, y);
}
inline ushort3 operator&=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, &, y);
}

/* ushort4 assignment and into (&=) */
inline ushort4 operator&=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, &, y);
}
inline ushort4 operator&=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, &, y);
}

/* ushort8 assignment and into (&=) */
inline ushort8 operator&=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, &, y);
}
inline ushort8 operator&=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, &, y);
}

/* ushort16 assignment and into (&=) */
inline ushort16 operator&=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, &, y);
}
inline ushort16 operator&=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* ushort2 assignment inclusive or into (|=) */
inline ushort2 operator|=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, |, y);
}
inline ushort2 operator|=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, |, y);
}

/* ushort3 assignment inclusive or into (|=) */
inline ushort3 operator|=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, |, y);
}
inline ushort3 operator|=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, |, y);

}
/* ushort4 assignment inclusive or into (|=) */
inline ushort4 operator|=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, |, y);
}
inline ushort4 operator|=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, |, y);
}

/* ushort8 assignment inclusive or into (|=) */
inline ushort8 operator|=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, |, y);
}
inline ushort8 operator|=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, |, y);
}

/* ushort16 assignment inclusive or into (|=) */
inline ushort16 operator|=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, |, y);
}
inline ushort16 operator|=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* ushort2 assignment exclusive or into (^=) */
inline ushort2 operator^=(ushort2 &x, const ushort2& y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, ^, y);
}
inline ushort2 operator^=(ushort2 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort2, x, ^, y);
}

/* ushort3 assignment exclusive or into (^=) */
inline ushort3 operator^=(ushort3 &x, const ushort3& y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, ^, y);
}
inline ushort3 operator^=(ushort3 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort3, x, ^, y);
}

/* ushort4 assignment exclusive or into (^=) */
inline ushort4 operator^=(ushort4 &x, const ushort4& y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, ^, y);
}
inline ushort4 operator^=(ushort4 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort4, x, ^, y);
}

/* ushort8 assignment exclusive or into (^=) */
inline ushort8 operator^=(ushort8 &x, const ushort8& y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, ^, y);
}
inline ushort8 operator^=(ushort8 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort8, x, ^, y);
}

/* ushort16 assignment exclusive or into (^=) */
inline ushort16 operator^=(ushort16 &x, const ushort16& y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, ^, y);
}
inline ushort16 operator^=(ushort16 &x, ushort y) {
  __CL_V_OP_ASSIGN_V(ushort16, x, ^, y);
}

#endif //__CL_OPS_USHORTN_H

