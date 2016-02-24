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

#ifndef __CL_OPS_UINTN_H
#define __CL_OPS_UINTN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>


///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* uint2 addition (+) */
inline uint2 operator+(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline uint2 operator+(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline uint2 operator+(uint x, const uint2& y) {
  return y + x;
}

/* uint3 addition (+) */
inline uint3 operator+(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline uint3 operator+(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline uint3 operator+(uint x, const uint3& y) {
  return y + x;
}

/* uint4 addition (+) */
inline uint4 operator+(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline uint4 operator+(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline uint4 operator+(uint x, const uint4& y) {
  return y + x;
}

/* uint8 addition (+) */
inline uint8 operator+(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline uint8 operator+(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline uint8 operator+(uint x, const uint8& y) {
  return y + x;
}

/* uint16 addition (+) */
inline uint16 operator+(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline uint16 operator+(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline uint16 operator+(uint x, const uint16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* uint2 subtraction (-) */
inline uint2 operator-(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline uint2 operator-(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline uint2 operator-(uint x, const uint2& y) {
  uint2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* uint3 subtraction (-) */
inline uint3 operator-(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline uint3 operator-(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline uint3 operator-(uint x, const uint3& y) {
  uint3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* uint4 subtraction (-) */
inline uint4 operator-(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline uint4 operator-(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline uint4 operator-(uint x, const uint4& y) {
  uint4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* uint8 subtraction (-) */
inline uint8 operator-(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline uint8 operator-(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline uint8 operator-(uint x, const uint8& y) {
  uint8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* uint16 subtraction (-) */
inline uint16 operator-(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline uint16 operator-(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline uint16 operator-(uint x, const uint16& y) {
  uint16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* uint2 multiplication (*) */
inline uint2 operator*(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline uint2 operator*(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline uint2 operator*(uint x, const uint2& y) {
  return y * x;
}

/* uint3 multiplication (*) */
inline uint3 operator*(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline uint3 operator*(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline uint3 operator*(uint x, const uint3& y) {
  return y + x;
}


/* uint4 multiplication (*) */
inline uint4 operator*(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline uint4 operator*(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline uint4 operator*(uint x, const uint4& y) {
  return y + x;
}

/* uint8 multiplication (*) */
inline uint8 operator*(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline uint8 operator*(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline uint8 operator*(uint x, const uint8& y) {
  return y * x;
}

/* uint16 multiplication (*) */
inline uint16 operator*(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline uint16 operator*(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline uint16 operator*(uint x, const uint16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* uint2 division (/) */
inline uint2 operator/(const uint2& x, const uint2& y) {
  uint2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline uint2 operator/(const uint2& x, uint y) {
  uint2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline uint2 operator/(uint x, const uint2& y) {
  uint2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* uint3 division (/) */
inline uint3 operator/(const uint3& x, const uint3& y) {
  uint3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline uint3 operator/(const uint3& x, uint y) {
  uint3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline uint3 operator/(uint x, const uint3& y) {
  uint3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* uint4 division (/) */
inline uint4 operator/(const uint4& x, const uint4& y) {
  uint4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline uint4 operator/(const uint4& x, uint y) {
  uint4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline uint4 operator/(uint x, const uint4& y) {
  uint4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* uint8 division (/) */
inline uint8 operator/(const uint8& x, const uint8& y) {
  uint8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline uint8 operator/(const uint8& x, uint y) {
  uint8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline uint8 operator/(uint x, const uint8& y) {
  uint8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* uint16 division (/) */
inline uint16 operator/(const uint16& x, const uint16& y) {
  uint16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline uint16 operator/(const uint16& x, uint y) {
  uint16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline uint16 operator/(uint x, const uint16& y) {
  uint16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* uint2 remainder (%) */
inline uint2 operator%(const uint2& x, const uint2& y) {
  uint2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline uint2 operator%(const uint2& x, uint y) {
  uint2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline uint2 operator%(uint x, const uint2& y) {
  uint2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* uint3 remainder (%) */
inline uint3 operator%(const uint3& x, const uint3& y) {
  uint3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline uint3 operator%(const uint3& x, uint y) {
  uint3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline uint3 operator%(uint x, const uint3& y) {
  uint3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* uint4 remainder (%) */
inline uint4 operator%(const uint4& x, const uint4& y) {
  uint4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline uint4 operator%(const uint4& x, uint y) {
  uint4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline uint4 operator%(uint x, const uint4& y) {
  uint4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* uint8 remainder (%) */
inline uint8 operator%(const uint8& x, const uint8& y) {
  uint8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline uint8 operator%(const uint8& x, uint y) {
  uint8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline uint8 operator%(uint x, const uint8& y) {
  uint8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* uint16 remainder (%) */
inline uint16 operator%(const uint16& x, const uint16& y) {
  uint16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline uint16 operator%(const uint16& x, uint y) {
  uint16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline uint16 operator%(uint x, const uint16& y) {
  uint16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* uint2 unary positive (+) */
inline uint2 operator+(const uint2& x) {
  return x;
}
/* uint3 unary positive (+) */
inline uint3 operator+(const uint3& x) {
  return x;
}

/* uint4 unary positive (+) */
inline uint4 operator+(const uint4& x) {
  return x;
}

/* uint8 unary positive (+) */
inline uint8 operator+(const uint8& x) {
  return x;
}

/* uint16 unary positive (+) */
inline uint16 operator+(const uint16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* uint2 unary negative (-) */
inline uint2 operator-(const uint2& x) {
  uint2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* uint3 unary negative (-) */
inline uint3 operator-(const uint3& x) {
  uint3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* uint4 unary negative (-) */
inline uint4 operator-(const uint4& x) {
  uint4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* uint8 unary negative (-) */
inline uint8 operator-(const uint8& x) {
  uint8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* uint16 unary negative (-) */
inline uint16 operator-(const uint16& x) {
  uint16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* uint2 unary post-increment (++) */
inline uint2 operator++(uint2 &x, int n) {
  uint2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uint3 unary post-increment (++) */
inline uint3 operator++(uint3 &x, int n) {
  uint3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uint4 unary post-increment (++) */
inline uint4 operator++(uint4 &x, int n) {
  uint4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uint8 unary post-increment (++) */
inline uint8 operator++(uint8 &x, int n) {
  uint8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uint16 unary post-increment (++) */
inline uint16 operator++(uint16 &x, int n) {
  uint16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* uint2 unary pre-increment (++) */
inline uint2 operator++(uint2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* uint3 unary pre-increment (++) */
inline uint3 operator++(uint3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* uint4 unary pre-increment (++) */
inline uint4 operator++(uint4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* uint8 unary pre-increment (++) */
inline uint8 operator++(uint8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* uint16 unary pre-increment (++) */
inline uint16 operator++(uint16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* uint2 unary post-decrement (--) */
inline uint2 operator--(uint2 &x, int n) {
  uint2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* uint3 unary post-decrement (--) */
inline uint3 operator--(uint3 &x, int n) {
  uint3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* uint4 unary post-decrement (--) */
inline uint4 operator--(uint4 &x, int n) {
  uint4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* uint8 unary post-decrement (--) */
inline uint8 operator--(uint8 &x, int n) {
  uint8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* uint16 unary post-decrement (--) */
inline uint16 operator--(uint16 &x, int n) {
  uint16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* uint2 unary pre-decrement (--) */
inline uint2 operator--(uint2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* uint3 unary pre-decrement (--) */
inline uint3 operator--(uint3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* uint4 unary pre-decrement (--) */
inline uint4 operator--(uint4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* uint8 unary pre-decrement (--) */
inline uint8 operator--(uint8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* uint16 unary pre-decrement (--) */
inline uint16 operator--(uint16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* uint2 relational greater than (>) */
inline int2 operator>(const uint2& x, const uint2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline int2 operator>(const uint2& x, uint y) {
  int2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline int2 operator>(uint x, const uint2& y) {
  int2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* uint3 relational greater than (>) */
inline int3 operator>(const uint3& x, const uint3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline int3 operator>(const uint3& x, uint y) {
  int3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline int3 operator>(uint x, const uint3& y) {
  int3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* uint4 relational greater than (>) */
inline int4 operator>(const uint4& x, const uint4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline int4 operator>(const uint4& x, uint y) {
  int4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline int4 operator>(uint x, const uint4& y) {
  int4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* uint8 relational greater than (>) */
inline int8 operator>(const uint8& x, const uint8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline int8 operator>(const uint8& x, uint y) {
  int8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline int8 operator>(uint x, const uint8& y) {
  int8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* uint16 relational greater than (>) */
inline int16 operator>(const uint16& x, const uint16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline int16 operator>(const uint16& x, uint y) {
  int16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline int16 operator>(uint x, const uint16& y) {
  int16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* uint2 relational less than (<) */
inline int2 operator<(const uint2& x, const uint2& y) {
  return y > x;
}
inline int2 operator<(const uint2& x, uint y) {
  return y > x;
}
inline int2 operator<(uint x, const uint2& y) {
  return y > x;
}

/* uint3 relational less than (<) */
inline int3 operator<(const uint3& x, const uint3& y) {
  return y > x;
}
inline int3 operator<(const uint3& x, uint y) {
  return y > x;
}
inline int3 operator<(uint x, const uint3& y) {
  return y > x;
}

/* uint4 relational less than (<) */
inline int4 operator<(const uint4& x, const uint4& y) {
  return y > x;
}
inline int4 operator<(const uint4& x, uint y) {
  return y > x;
}
inline int4 operator<(uint x, const uint4& y) {
  return y > x;
}

/* uint8 relational less than (<) */
inline int8 operator<(const uint8& x, const uint8& y) {
  return y > x;
}
inline int8 operator<(const uint8& x, uint y) {
  return y > x;
}
inline int8 operator<(uint x, const uint8& y) {
  return y > x;
}

/* uint16 relational less than (<) */
inline int16 operator<(const uint16& x, const uint16& y) {
  return y > x;
}
inline int16 operator<(const uint16& x, uint y) {
  return y > x;
}
inline int16 operator<(uint x, const uint16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* uint2 relational greater than or equal (>=) */
inline int2 operator>=(const uint2& x, const uint2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline int2 operator>=(const uint2& x, uint y) {
  int2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline int2 operator>=(uint x, const uint2& y) {
  int2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* uint3 relational greater than or equal (>=) */
inline int3 operator>=(const uint3& x, const uint3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline int3 operator>=(const uint3& x, uint y) {
  int3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline int3 operator>=(uint x, const uint3& y) {
  int3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* uint4 relational greater than or equal (>=) */
inline int4 operator>=(const uint4& x, const uint4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline int4 operator>=(const uint4& x, uint y) {
  int4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline int4 operator>=(uint x, const uint4& y) {
  int4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* uint8 relational greater than or equal (>=) */
inline int8 operator>=(const uint8& x, const uint8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline int8 operator>=(const uint8& x, uint y) {
  int8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline int8 operator>=(uint x, const uint8& y) {
  int8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* uint16 relational greater than or equal (>=) */
inline int16 operator>=(const uint16& x, const uint16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline int16 operator>=(const uint16& x, uint y) {
  int16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline int16 operator>=(uint x, const uint16& y) {
  int16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* uint2 relational less than or equal (<=) */
inline int2 operator<=(const uint2& x, const uint2& y) {
  return y >= x;
}
inline int2 operator<=(const uint2& x, uint y) {
  return y >= x;
}
inline int2 operator<=(uint x, const uint2& y) {
  return y >= x;
}

/* uint3 relational less than or equal (<=) */
inline int3 operator<=(const uint3& x, const uint3& y) {
  return y >= x;
}
inline int3 operator<=(const uint3& x, uint y) {
  return y >= x;
}
inline int3 operator<=(uint x, const uint3& y) {
  return y >= x;
}

/* uint4 relational less than or equal (<=) */
inline int4 operator<=(const uint4& x, const uint4& y) {
  return y >= x;
}
inline int4 operator<=(const uint4& x, uint y) {
  return y >= x;
}
inline int4 operator<=(uint x, const uint4& y) {
  return y >= x;
}

/* uint8 relational less than or equal (<=) */
inline int8 operator<=(const uint8& x, const uint8& y) {
  return y >= x;
}
inline int8 operator<=(const uint8& x, uint y) {
  return y >= x;
}
inline int8 operator<=(uint x, const uint8& y) {
  return y >= x;
}

/* uint16 relational less than or equal (<=) */
inline int16 operator<=(const uint16& x, const uint16& y) {
  return y >= x;
}
inline int16 operator<=(const uint16& x, uint y) {
  return y >= x;
}
inline int16 operator<=(uint x, const uint16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* uint2 equal (==) */
inline int2 operator==(const uint2& x, const uint2& y) {
  int2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline int2 operator==(const uint2& x, uint y) {
  int2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline int2 operator==(uint x, const uint2& y) {
  int2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* uint3 equal (==) */
inline int3 operator==(const uint3& x, const uint3& y) {
  int3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline int3 operator==(const uint3& x, uint y) {
  int3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline int3 operator==(uint x, const uint3& y) {
  int3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* uint4 equal (==) */
inline int4 operator==(const uint4& x, const uint4& y) {
  int4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline int4 operator==(const uint4& x, uint y) {
  int4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline int4 operator==(uint x, const uint4& y) {
  int4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* uint8 equal (==) */
inline int8 operator==(const uint8& x, const uint8& y) {
  int8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline int8 operator==(const uint8& x, uint y) {
  int8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline int8 operator==(uint x, const uint8& y) {
  int8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* uint16 equal (==) */
inline int16 operator==(const uint16& x, const uint16& y) {
  int16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline int16 operator==(const uint16& x, uint y) {
  int16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline int16 operator==(uint x, const uint16& y) {
  int16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* uint2 not equal (!=) */
inline int2 operator!=(const uint2& x, const uint2& y) {
  int2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline int2 operator!=(const uint2& x, uint y) {
  int2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline int2 operator!=(uint x, const uint2& y) {
  int2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* uint3 not equal (!=) */
inline int3 operator!=(const uint3& x, const uint3& y) {
  int3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline int3 operator!=(const uint3& x, uint y) {
  int3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline int3 operator!=(uint x, const uint3& y) {
  int3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* uint4 not equal (!=) */
inline int4 operator!=(const uint4& x, const uint4& y) {
  int4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline int4 operator!=(const uint4& x, uint y) {
  int4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline int4 operator!=(uint x, const uint4& y) {
  int4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* uint8 not equal (!=) */
inline int8 operator!=(const uint8& x, const uint8& y) {
  int8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline int8 operator!=(const uint8& x, uint y) {
  int8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline int8 operator!=(uint x, const uint8& y) {
  int8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* uint16 not equal (!=) */
inline int16 operator!=(const uint16& x, const uint16& y) {
  int16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline int16 operator!=(const uint16& x, uint y) {
  int16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline int16 operator!=(uint x, const uint16& y) {
  int16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* uint2 bitwise and (&) */
inline uint2 operator&(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline uint2 operator&(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline uint2 operator&(uint x, const uint2& y) {
  return y & x;
}

/* uint3 bitwise and (&) */
inline uint3 operator&(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline uint3 operator&(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline uint3 operator&(uint x, const uint3& y) {
  return y & x;
}


/* uint4 bitwise and (&) */
inline uint4 operator&(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline uint4 operator&(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline uint4 operator&(uint x, const uint4& y) {
  return y & x;
}

/* uint8 bitwise and (&) */
inline uint8 operator&(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline uint8 operator&(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline uint8 operator&(uint x, const uint8& y) {
  return y & x;
}

/* uint16 bitwise and (&) */
inline uint16 operator&(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline uint16 operator&(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline uint16 operator&(uint x, const uint16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* uint2 bitwise or (|) */
inline uint2 operator|(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline uint2 operator|(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline uint2 operator|(uint x, const uint2& y) {
  return y | x;
}

/* uint3 bitwise or (|) */
inline uint3 operator|(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline uint3 operator|(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline uint3 operator|(uint x, const uint3& y) {
  return y | x;
}

/* uint4 bitwise or (|) */
inline uint4 operator|(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline uint4 operator|(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline uint4 operator|(uint x, const uint4& y) {
  return y | x;
}

/* uint8 bitwise or (|) */
inline uint8 operator|(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline uint8 operator|(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline uint8 operator|(uint x, const uint8& y) {
  return y | x;
}

/* uint16 bitwise or (|) */
inline uint16 operator|(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline uint16 operator|(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline uint16 operator|(uint x, const uint16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* uint2 bitwise exclusive or (^) */
inline uint2 operator^(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline uint2 operator^(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline uint2 operator^(uint x, const uint2& y) {
  return y ^ x;
}

/* uint3 bitwise exclusive or (^) */
inline uint3 operator^(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline uint3 operator^(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline uint3 operator^(uint x, const uint3& y) {
  return y ^ x;
}

/* uint4 bitwise exclusive or (^) */
inline uint4 operator^(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline uint4 operator^(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline uint4 operator^(uint x, const uint4& y) {
  return y ^ x;
}

/* uint8 bitwise exclusive or (^) */
inline uint8 operator^(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline uint8 operator^(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline uint8 operator^(uint x, const uint8& y) {
  return y ^ x;
}

/* uint16 bitwise exclusive or (^) */
inline uint16 operator^(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline uint16 operator^(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline uint16 operator^(uint x, const uint16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* uint2 bitwise not (~) */
inline uint2 operator~(const uint2& x) {
  uint2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* uint3 bitwise not (~) */
inline uint3 operator~(const uint3& x) {
  uint3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* uint4 bitwise not (~) */
inline uint4 operator~(const uint4& x) {
  uint4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* uint8 bitwise not (~) */
inline uint8 operator~(const uint8& x) {
  uint8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* uint16 bitwise not (~) */
inline uint16 operator~(const uint16& x) {
  uint16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* uint2 logical and (&&) */
inline int2 operator&&(const uint2& x, const uint2& y) {
  int2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline int2 operator&&(const uint2& x, uint y) {
  int2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline int2 operator&&(uint x, const uint2& y) {
  int2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* uint3 logical and (&&) */
inline int3 operator&&(const uint3& x, const uint3& y) {
  int3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline int3 operator&&(const uint3& x, uint y) {
  int3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline int3 operator&&(uint x, const uint3& y) {
  int3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* uint4 logical and (&&) */
inline int4 operator&&(const uint4& x, const uint4& y) {
  int4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline int4 operator&&(const uint4& x, uint y) {
  int4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline int4 operator&&(uint x, const uint4& y) {
  int4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* uint8 logical and (&&) */
inline int8 operator&&(const uint8& x, const uint8& y) {
  int8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline int8 operator&&(const uint8& x, uint y) {
  int8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline int8 operator&&(uint x, const uint8& y) {
  int8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* uint16 logical and (&&) */
inline int16 operator&&(const uint16& x, const uint16& y) {
  int16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline int16 operator&&(const uint16& x, uint y) {
  int16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline int16 operator&&(uint x, const uint16& y) {
  int16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* uint2 logical or (||) */
inline int2 operator||(const uint2& x, const uint2& y) {
  int2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline int2 operator||(const uint2& x, uint y) {
  int2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline int2 operator||(uint x, const uint2& y) {
  int2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* uint3 logical or (||) */
inline int3 operator||(const uint3& x, const uint3& y) {
  int3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline int3 operator||(const uint3& x, uint y) {
  int3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline int3 operator||(uint x, const uint3& y) {
  int3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* uint4 logical or (||) */
inline int4 operator||(const uint4& x, const uint4& y) {
  int4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline int4 operator||(const uint4& x, uint y) {
  int4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline int4 operator||(uint x, const uint4& y) {
  int4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* uint8 logical or (||) */
inline int8 operator||(const uint8& x, const uint8& y) {
  int8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline int8 operator||(const uint8& x, uint y) {
  int8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline int8 operator||(uint x, const uint8& y) {
  int8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* uint16 logical or (||) */
inline int16 operator||(const uint16& x, const uint16& y) {
  int16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline int16 operator||(const uint16& x, uint y) {
  int16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline int16 operator||(uint x, const uint16& y) {
  int16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* uint2 logical not (!) */
inline int2 operator!(const uint2& x) {
  int2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* uint3 logical not (!) */
inline int3 operator!(const uint3& x) {
  int3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* uint4 logical not (!) */
inline int4 operator!(const uint4& x) {
  int4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* uint8 logical not (!) */
inline int8 operator!(const uint8& x) {
  int8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* uint16 logical not (!) */
inline int16 operator!(const uint16& x) {
  int16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* uint2 right-shift (>>) */
inline uint2 operator>>(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, >>, y);
  return rst;
}
inline uint2 operator>>(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, >>, y);
  return rst;
}
/* uint3 right-shift (>>) */
inline uint3 operator>>(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, >>, y);
  return rst;
}
inline uint3 operator>>(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, >>, y);
  return rst;
}

/* uint4 right-shift (>>) */
inline uint4 operator>>(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, >>, y);
  return rst;
}
inline uint4 operator>>(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, >>, y);
  return rst;
}

/* uint8 right-shift (>>) */
inline uint8 operator>>(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, >>, y);
  return rst;
}
inline uint8 operator>>(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, >>, y);
  return rst;
}

/* uint16 right-shift (>>) */
inline uint16 operator>>(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, >>, y);
  return rst;
}
inline uint16 operator>>(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, >>, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* uint2 left-shift (<<) */
inline uint2 operator<<(const uint2& x, const uint2& y) {
  uint2 rst = __CL_V2_OP_V2(x, <<, y);
  return rst;
}
inline uint2 operator<<(const uint2& x, uint y) {
  uint2 rst = __CL_V2_OP_S(x, <<, y);
  return rst;
}

/* uint3 left-shift (<<) */
inline uint3 operator<<(const uint3& x, const uint3& y) {
  uint3 rst = __CL_V3_OP_V3(x, <<, y);
  return rst;
}
inline uint3 operator<<(const uint3& x, uint y) {
  uint3 rst = __CL_V3_OP_S(x, <<, y);
  return rst;
}

/* uint4 left-shift (<<) */
inline uint4 operator<<(const uint4& x, const uint4& y) {
  uint4 rst = __CL_V4_OP_V4(x, <<, y);
  return rst;
}
inline uint4 operator<<(const uint4& x, uint y) {
  uint4 rst = __CL_V4_OP_S(x, <<, y);
  return rst;
}

/* uint8 left-shift (<<) */
inline uint8 operator<<(const uint8& x, const uint8& y) {
  uint8 rst = __CL_V8_OP_V8(x, <<, y);
  return rst;
}
inline uint8 operator<<(const uint8& x, uint y) {
  uint8 rst = __CL_V8_OP_S(x, <<, y);
  return rst;
}

/* uint16 left-shift (<<) */
inline uint16 operator<<(const uint16& x, const uint16& y) {
  uint16 rst = __CL_V16_OP_V16(x, <<, y);
  return rst;
}
inline uint16 operator<<(const uint16& x, uint y) {
  uint16 rst = __CL_V16_OP_S(x, <<, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* uint2 assignment add into (+=) */
inline uint2 operator+=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, +, y);
}
inline uint2 operator+=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, +, y);
}

/* uint3 assignment add into (+=) */
inline uint3 operator+=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, +, y);
}
inline uint3 operator+=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, +, y);
}

/* uint4 assignment add into (+=) */
inline uint4 operator+=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, +, y);
}
inline uint4 operator+=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, +, y);
}

/* uint8 assignment add into (+=) */
inline uint8 operator+=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, +, y);
}
inline uint8 operator+=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, +, y);
}

/* uint16 assignment add into (+=) */
inline uint16 operator+=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, +, y);
}
inline uint16 operator+=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* uint2 assignment subtract from (-=) */
inline uint2 operator-=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, -, y);
}
inline uint2 operator-=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, -, y);
}

/* uint3 assignment subtract from (-=) */
inline uint3 operator-=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, -, y);
}
inline uint3 operator-=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, -, y);
}

/* uint4 assignment subtract from (-=) */
inline uint4 operator-=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, -, y);
}
inline uint4 operator-=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, -, y);
}

/* uint8 assignment subtract from (-=) */
inline uint8 operator-=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, -, y);
}
inline uint8 operator-=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, -, y);
}

/* uint16 assignment subtract from (-=) */
inline uint16 operator-=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, -, y);
}
inline uint16 operator-=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* uint2 assignment multiply into (*=) */
inline uint2 operator*=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, *, y);
}
inline uint2 operator*=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, *, y);
}

/* uint3 assignment multiply into (*=) */
inline uint3 operator*=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, *, y);
}
inline uint3 operator*=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, *, y);
}

/* uint4 assignment multiply into (*=) */
inline uint4 operator*=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, *, y);
}
inline uint4 operator*=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, *, y);
}

/* uint8 assignment multiply into (*=) */
inline uint8 operator*=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, *, y);
}
inline uint8 operator*=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, *, y);
}

/* uint16 assignment multiply into (*=) */
inline uint16 operator*=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, *, y);
}
inline uint16 operator*=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* uint2 assignment divide into (/=) */
inline uint2 operator/=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, /, y);
}
inline uint2 operator/=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, /, y);
}

/* uint3 assignment divide into (/=) */
inline uint3 operator/=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, /, y);
}
inline uint3 operator/=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, /, y);
}

/* uint4 assignment divide into (/=) */
inline uint4 operator/=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, /, y);
}
inline uint4 operator/=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, /, y);
}

/* uint8 assignment divide into (/=) */
inline uint8 operator/=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, /, y);
}
inline uint8 operator/=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, /, y);
}

/* uint16 assignment divide into (/=) */
inline uint16 operator/=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, /, y);
}
inline uint16 operator/=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* uint2 assignment modulus into (%=) */
inline uint2 operator%=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, %, y);
}
inline uint2 operator%=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, %, y);
}

/* uint3 assignment modulus into (%=) */
inline uint3 operator%=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, %, y);
}
inline uint3 operator%=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, %, y);
}

/* uint4 assignment modulus into (%=) */
inline uint4 operator%=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, %, y);
}
inline uint4 operator%=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, %, y);
}

/* uint8 assignment modulus into (%=) */
inline uint8 operator%=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, %, y);
}
inline uint8 operator%=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, %, y);
}

/* uint16 assignment modulus into (%=) */
inline uint16 operator%=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, %, y);
}
inline uint16 operator%=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* uint2 assignment left shift by (<<=) */
inline uint2 operator<<=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, <<, y);
}
inline uint2 operator<<=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, <<, y);
}

/* uint3 assignment left shift by (<<=) */
inline uint3 operator<<=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, <<, y);
}
inline uint3 operator<<=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, <<, y);
}


/* uint4 assignment left shift by (<<=) */
inline uint4 operator<<=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, <<, y);
}
inline uint4 operator<<=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, <<, y);
}

/* uint8 assignment left shift by (<<=) */
inline uint8 operator<<=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, <<, y);
}
inline uint8 operator<<=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, <<, y);
}

/* uint16 assignment left shift by (<<=) */
inline uint16 operator<<=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, <<, y);
}
inline uint16 operator<<=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* uint2 assignment right shift by (>>=) */
inline uint2 operator>>=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, >>, y);
}
inline uint2 operator>>=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, >>, y);
}

/* uint3 assignment right shift by (>>=) */
inline uint3 operator>>=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, >>, y);
}
inline uint3 operator>>=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, >>, y);
}
/* uint4 assignment right shift by (>>=) */
inline uint4 operator>>=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, >>, y);
}
inline uint4 operator>>=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, >>, y);
}

/* uint8 assignment right shift by (>>=) */
inline uint8 operator>>=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, >>, y);
}
inline uint8 operator>>=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, >>, y);
}

/* uint16 assignment right shift by (>>=) */
inline uint16 operator>>=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, >>, y);
}
inline uint16 operator>>=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* uint2 assignment and into (&=) */
inline uint2 operator&=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, &, y);
}
inline uint2 operator&=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, &, y);
}

/* uint3 assignment and into (&=) */
inline uint3 operator&=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, &, y);
}
inline uint3 operator&=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, &, y);
}

/* uint4 assignment and into (&=) */
inline uint4 operator&=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, &, y);
}
inline uint4 operator&=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, &, y);
}

/* uint8 assignment and into (&=) */
inline uint8 operator&=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, &, y);
}
inline uint8 operator&=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, &, y);
}

/* uint16 assignment and into (&=) */
inline uint16 operator&=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, &, y);
}
inline uint16 operator&=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* uint2 assignment inclusive or into (|=) */
inline uint2 operator|=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, |, y);
}
inline uint2 operator|=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, |, y);
}

/* uint3 assignment inclusive or into (|=) */
inline uint3 operator|=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, |, y);
}
inline uint3 operator|=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, |, y);

}
/* uint4 assignment inclusive or into (|=) */
inline uint4 operator|=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, |, y);
}
inline uint4 operator|=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, |, y);
}

/* uint8 assignment inclusive or into (|=) */
inline uint8 operator|=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, |, y);
}
inline uint8 operator|=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, |, y);
}

/* uint16 assignment inclusive or into (|=) */
inline uint16 operator|=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, |, y);
}
inline uint16 operator|=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* uint2 assignment exclusive or into (^=) */
inline uint2 operator^=(uint2 &x, const uint2& y) {
  __CL_V_OP_ASSIGN_V(uint2, x, ^, y);
}
inline uint2 operator^=(uint2 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint2, x, ^, y);
}

/* uint3 assignment exclusive or into (^=) */
inline uint3 operator^=(uint3 &x, const uint3& y) {
  __CL_V_OP_ASSIGN_V(uint3, x, ^, y);
}
inline uint3 operator^=(uint3 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint3, x, ^, y);
}

/* uint4 assignment exclusive or into (^=) */
inline uint4 operator^=(uint4 &x, const uint4& y) {
  __CL_V_OP_ASSIGN_V(uint4, x, ^, y);
}
inline uint4 operator^=(uint4 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint4, x, ^, y);
}

/* uint8 assignment exclusive or into (^=) */
inline uint8 operator^=(uint8 &x, const uint8& y) {
  __CL_V_OP_ASSIGN_V(uint8, x, ^, y);
}
inline uint8 operator^=(uint8 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint8, x, ^, y);
}

/* uint16 assignment exclusive or into (^=) */
inline uint16 operator^=(uint16 &x, const uint16& y) {
  __CL_V_OP_ASSIGN_V(uint16, x, ^, y);
}
inline uint16 operator^=(uint16 &x, uint y) {
  __CL_V_OP_ASSIGN_V(uint16, x, ^, y);
}

#endif //__CL_OPS_UINTN_H

