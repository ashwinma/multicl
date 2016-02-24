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

#ifndef __CL_OPS_UCHARN_H
#define __CL_OPS_UCHARN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>



///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* uchar2 addition (+) */
inline uchar2 operator+(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline uchar2 operator+(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline uchar2 operator+(uchar x, const uchar2& y) {
  return y + x;
}

/* uchar3 addition (+) */
inline uchar3 operator+(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline uchar3 operator+(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline uchar3 operator+(uchar x, const uchar3& y) {
  return y + x;
}

/* uchar4 addition (+) */
inline uchar4 operator+(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline uchar4 operator+(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline uchar4 operator+(uchar x, const uchar4& y) {
  return y + x;
}

/* uchar8 addition (+) */
inline uchar8 operator+(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline uchar8 operator+(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline uchar8 operator+(uchar x, const uchar8& y) {
  return y + x;
}

/* uchar16 addition (+) */
inline uchar16 operator+(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline uchar16 operator+(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline uchar16 operator+(uchar x, const uchar16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* uchar2 subtraction (-) */
inline uchar2 operator-(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline uchar2 operator-(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline uchar2 operator-(uchar x, const uchar2& y) {
  uchar2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* uchar3 subtraction (-) */
inline uchar3 operator-(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline uchar3 operator-(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline uchar3 operator-(uchar x, const uchar3& y) {
  uchar3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* uchar4 subtraction (-) */
inline uchar4 operator-(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline uchar4 operator-(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline uchar4 operator-(uchar x, const uchar4& y) {
  uchar4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* uchar8 subtraction (-) */
inline uchar8 operator-(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline uchar8 operator-(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline uchar8 operator-(uchar x, const uchar8& y) {
  uchar8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* uchar16 subtraction (-) */
inline uchar16 operator-(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline uchar16 operator-(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline uchar16 operator-(uchar x, const uchar16& y) {
  uchar16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* uchar2 multiplication (*) */
inline uchar2 operator*(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline uchar2 operator*(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline uchar2 operator*(uchar x, const uchar2& y) {
  return y * x;
}

/* uchar3 multiplication (*) */
inline uchar3 operator*(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline uchar3 operator*(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline uchar3 operator*(uchar x, const uchar3& y) {
  return y + x;
}


/* uchar4 multiplication (*) */
inline uchar4 operator*(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline uchar4 operator*(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline uchar4 operator*(uchar x, const uchar4& y) {
  return y + x;
}

/* uchar8 multiplication (*) */
inline uchar8 operator*(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline uchar8 operator*(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline uchar8 operator*(uchar x, const uchar8& y) {
  return y * x;
}

/* uchar16 multiplication (*) */
inline uchar16 operator*(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline uchar16 operator*(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline uchar16 operator*(uchar x, const uchar16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* uchar2 division (/) */
inline uchar2 operator/(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, /, y);
  return rst;
}
inline uchar2 operator/(const uchar2& x, uchar y) {
  uchar2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, /, y);
  return rst;
}
inline uchar2 operator/(uchar x, const uchar2& y) {
  uchar2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, /, y);
  return rst;
}

/* uchar3 division (/) */
inline uchar3 operator/(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, /, y);
  return rst;
}
inline uchar3 operator/(const uchar3& x, uchar y) {
  uchar3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, /, y);
  return rst;
}
inline uchar3 operator/(uchar x, const uchar3& y) {
  uchar3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, /, y);
  return rst;
}

/* uchar4 division (/) */
inline uchar4 operator/(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, /, y);
  return rst;
}
inline uchar4 operator/(const uchar4& x, uchar y) {
  uchar4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, /, y);
  return rst;
}
inline uchar4 operator/(uchar x, const uchar4& y) {
  uchar4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, /, y);
  return rst;
}

/* uchar8 division (/) */
inline uchar8 operator/(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, /, y);
  return rst;
}
inline uchar8 operator/(const uchar8& x, uchar y) {
  uchar8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, /, y);
  return rst;
}
inline uchar8 operator/(uchar x, const uchar8& y) {
  uchar8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, /, y);
  return rst;
}

/* uchar16 division (/) */
inline uchar16 operator/(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, /, y);
  return rst;
}
inline uchar16 operator/(const uchar16& x, uchar y) {
  uchar16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, /, y);
  return rst;
}
inline uchar16 operator/(uchar x, const uchar16& y) {
  uchar16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////

/* uchar2 remainder (%) */
inline uchar2 operator%(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_V2(x, %, y);
  return rst;
}
inline uchar2 operator%(const uchar2& x, uchar y) {
  uchar2 rst = __CL_SAFE_UINT_DIV_ZERO_V2_OP_S(x, %, y);
  return rst;
}
inline uchar2 operator%(uchar x, const uchar2& y) {
  uchar2 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V2(x, %, y);
  return rst;
}

/* uchar3 remainder (%) */
inline uchar3 operator%(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_V3(x, %, y);
  return rst;
}
inline uchar3 operator%(const uchar3& x, uchar y) {
  uchar3 rst = __CL_SAFE_UINT_DIV_ZERO_V3_OP_S(x, %, y);
  return rst;
}
inline uchar3 operator%(uchar x, const uchar3& y) {
  uchar3 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V3(x, %, y);
  return rst;
}

/* uchar4 remainder (%) */
inline uchar4 operator%(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_V4(x, %, y);
  return rst;
}
inline uchar4 operator%(const uchar4& x, uchar y) {
  uchar4 rst = __CL_SAFE_UINT_DIV_ZERO_V4_OP_S(x, %, y);
  return rst;
}
inline uchar4 operator%(uchar x, const uchar4& y) {
  uchar4 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V4(x, %, y);
  return rst;
}

/* uchar8 remainder (%) */
inline uchar8 operator%(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_V8(x, %, y);
  return rst;
}
inline uchar8 operator%(const uchar8& x, uchar y) {
  uchar8 rst = __CL_SAFE_UINT_DIV_ZERO_V8_OP_S(x, %, y);
  return rst;
}
inline uchar8 operator%(uchar x, const uchar8& y) {
  uchar8 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V8(x, %, y);
  return rst;
}

/* uchar16 remainder (%) */
inline uchar16 operator%(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_V16(x, %, y);
  return rst;
}
inline uchar16 operator%(const uchar16& x, uchar y) {
  uchar16 rst = __CL_SAFE_UINT_DIV_ZERO_V16_OP_S(x, %, y);
  return rst;
}
inline uchar16 operator%(uchar x, const uchar16& y) {
  uchar16 rst = __CL_SAFE_UINT_DIV_ZERO_S_OP_V16(x, %, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* uchar2 unary positive (+) */
inline uchar2 operator+(const uchar2& x) {
  return x;
}
/* uchar3 unary positive (+) */
inline uchar3 operator+(const uchar3& x) {
  return x;
}

/* uchar4 unary positive (+) */
inline uchar4 operator+(const uchar4& x) {
  return x;
}

/* uchar8 unary positive (+) */
inline uchar8 operator+(const uchar8& x) {
  return x;
}

/* uchar16 unary positive (+) */
inline uchar16 operator+(const uchar16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* uchar2 unary negative (-) */
inline uchar2 operator-(const uchar2& x) {
  uchar2 rst = __CL_OP_V2(-, x);
  return rst;
}
/* uchar3 unary negative (-) */
inline uchar3 operator-(const uchar3& x) {
  uchar3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* uchar4 unary negative (-) */
inline uchar4 operator-(const uchar4& x) {
  uchar4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* uchar8 unary negative (-) */
inline uchar8 operator-(const uchar8& x) {
  uchar8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* uchar16 unary negative (-) */
inline uchar16 operator-(const uchar16& x) {
  uchar16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////

/* uchar2 unary post-increment (++) */
inline uchar2 operator++(uchar2 &x, int n) {
  uchar2 rst = x;
  __CL_V2_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uchar3 unary post-increment (++) */
inline uchar3 operator++(uchar3 &x, int n) {
  uchar3 rst = x;
  __CL_V3_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uchar4 unary post-increment (++) */
inline uchar4 operator++(uchar4 &x, int n) {
  uchar4 rst = x;
  __CL_V4_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uchar8 unary post-increment (++) */
inline uchar8 operator++(uchar8 &x, int n) {
  uchar8 rst = x;
  __CL_V8_POST_OP(x, ++);
  n = n + 0;
  return rst;
}

/* uchar16 unary post-increment (++) */
inline uchar16 operator++(uchar16 &x, int n) {
  uchar16 rst = x;
  __CL_V16_POST_OP(x, ++);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////

/* uchar2 unary pre-increment (++) */
inline uchar2 operator++(uchar2 &x) {
  __CL_PRE_OP_V2(++, x);
  return x;
}

/* uchar3 unary pre-increment (++) */
inline uchar3 operator++(uchar3 &x) {
  __CL_PRE_OP_V3(++, x);
  return x;
}

/* uchar4 unary pre-increment (++) */
inline uchar4 operator++(uchar4 &x) {
  __CL_PRE_OP_V4(++, x);
  return x;
}

/* uchar8 unary pre-increment (++) */
inline uchar8 operator++(uchar8 &x) {
  __CL_PRE_OP_V8(++, x);
  return x;
}

/* uchar16 unary pre-increment (++) */
inline uchar16 operator++(uchar16 &x) {
  __CL_PRE_OP_V16(++, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////

/* uchar2 unary post-decrement (--) */
inline uchar2 operator--(uchar2 &x, int n) {
  uchar2 rst = x;
  __CL_V2_POST_OP(x, --);
  n = n + 0;
  return rst;
}
/* uchar3 unary post-decrement (--) */
inline uchar3 operator--(uchar3 &x, int n) {
  uchar3 rst = x;
  __CL_V3_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* uchar4 unary post-decrement (--) */
inline uchar4 operator--(uchar4 &x, int n) {
  uchar4 rst = x;
  __CL_V4_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* uchar8 unary post-decrement (--) */
inline uchar8 operator--(uchar8 &x, int n) {
  uchar8 rst = x;
  __CL_V8_POST_OP(x, --);
  n = n + 0;
  return rst;
}

/* uchar16 unary post-decrement (--) */
inline uchar16 operator--(uchar16 &x, int n) {
  uchar16 rst = x;
  __CL_V16_POST_OP(x, --);
  n = n + 0;
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////

/* uchar2 unary pre-decrement (--) */
inline uchar2 operator--(uchar2 &x) {
  __CL_PRE_OP_V2(--, x);
  return x;
}

/* uchar3 unary pre-decrement (--) */
inline uchar3 operator--(uchar3 &x) {
  __CL_PRE_OP_V3(--, x);
  return x;
}

/* uchar4 unary pre-decrement (--) */
inline uchar4 operator--(uchar4 &x) {
  __CL_PRE_OP_V4(--, x);
  return x;
}

/* uchar8 unary pre-decrement (--) */
inline uchar8 operator--(uchar8 &x) {
  __CL_PRE_OP_V8(--, x);
  return x;
}

/* uchar16 unary pre-decrement (--) */
inline uchar16 operator--(uchar16 &x) {
  __CL_PRE_OP_V16(--, x);
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* uchar2 relational greater than (>) */
inline char2 operator>(const uchar2& x, const uchar2& y) {
  char2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline char2 operator>(const uchar2& x, uchar y) {
  char2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline char2 operator>(uchar x, const uchar2& y) {
  char2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* uchar3 relational greater than (>) */
inline char3 operator>(const uchar3& x, const uchar3& y) {
  char3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline char3 operator>(const uchar3& x, uchar y) {
  char3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline char3 operator>(uchar x, const uchar3& y) {
  char3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* uchar4 relational greater than (>) */
inline char4 operator>(const uchar4& x, const uchar4& y) {
  char4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline char4 operator>(const uchar4& x, uchar y) {
  char4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline char4 operator>(uchar x, const uchar4& y) {
  char4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* uchar8 relational greater than (>) */
inline char8 operator>(const uchar8& x, const uchar8& y) {
  char8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline char8 operator>(const uchar8& x, uchar y) {
  char8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline char8 operator>(uchar x, const uchar8& y) {
  char8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* uchar16 relational greater than (>) */
inline char16 operator>(const uchar16& x, const uchar16& y) {
  char16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline char16 operator>(const uchar16& x, uchar y) {
  char16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline char16 operator>(uchar x, const uchar16& y) {
  char16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* uchar2 relational less than (<) */
inline char2 operator<(const uchar2& x, const uchar2& y) {
  return y > x;
}
inline char2 operator<(const uchar2& x, uchar y) {
  return y > x;
}
inline char2 operator<(uchar x, const uchar2& y) {
  return y > x;
}

/* uchar3 relational less than (<) */
inline char3 operator<(const uchar3& x, const uchar3& y) {
  return y > x;
}
inline char3 operator<(const uchar3& x, uchar y) {
  return y > x;
}
inline char3 operator<(uchar x, const uchar3& y) {
  return y > x;
}

/* uchar4 relational less than (<) */
inline char4 operator<(const uchar4& x, const uchar4& y) {
  return y > x;
}
inline char4 operator<(const uchar4& x, uchar y) {
  return y > x;
}
inline char4 operator<(uchar x, const uchar4& y) {
  return y > x;
}

/* uchar8 relational less than (<) */
inline char8 operator<(const uchar8& x, const uchar8& y) {
  return y > x;
}
inline char8 operator<(const uchar8& x, uchar y) {
  return y > x;
}
inline char8 operator<(uchar x, const uchar8& y) {
  return y > x;
}

/* uchar16 relational less than (<) */
inline char16 operator<(const uchar16& x, const uchar16& y) {
  return y > x;
}
inline char16 operator<(const uchar16& x, uchar y) {
  return y > x;
}
inline char16 operator<(uchar x, const uchar16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* uchar2 relational greater than or equal (>=) */
inline char2 operator>=(const uchar2& x, const uchar2& y) {
  char2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline char2 operator>=(const uchar2& x, uchar y) {
  char2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline char2 operator>=(uchar x, const uchar2& y) {
  char2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* uchar3 relational greater than or equal (>=) */
inline char3 operator>=(const uchar3& x, const uchar3& y) {
  char3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline char3 operator>=(const uchar3& x, uchar y) {
  char3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline char3 operator>=(uchar x, const uchar3& y) {
  char3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* uchar4 relational greater than or equal (>=) */
inline char4 operator>=(const uchar4& x, const uchar4& y) {
  char4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline char4 operator>=(const uchar4& x, uchar y) {
  char4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline char4 operator>=(uchar x, const uchar4& y) {
  char4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* uchar8 relational greater than or equal (>=) */
inline char8 operator>=(const uchar8& x, const uchar8& y) {
  char8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline char8 operator>=(const uchar8& x, uchar y) {
  char8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline char8 operator>=(uchar x, const uchar8& y) {
  char8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* uchar16 relational greater than or equal (>=) */
inline char16 operator>=(const uchar16& x, const uchar16& y) {
  char16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline char16 operator>=(const uchar16& x, uchar y) {
  char16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline char16 operator>=(uchar x, const uchar16& y) {
  char16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* uchar2 relational less than or equal (<=) */
inline char2 operator<=(const uchar2& x, const uchar2& y) {
  return y >= x;
}
inline char2 operator<=(const uchar2& x, uchar y) {
  return y >= x;
}
inline char2 operator<=(uchar x, const uchar2& y) {
  return y >= x;
}

/* uchar3 relational less than or equal (<=) */
inline char3 operator<=(const uchar3& x, const uchar3& y) {
  return y >= x;
}
inline char3 operator<=(const uchar3& x, uchar y) {
  return y >= x;
}
inline char3 operator<=(uchar x, const uchar3& y) {
  return y >= x;
}

/* uchar4 relational less than or equal (<=) */
inline char4 operator<=(const uchar4& x, const uchar4& y) {
  return y >= x;
}
inline char4 operator<=(const uchar4& x, uchar y) {
  return y >= x;
}
inline char4 operator<=(uchar x, const uchar4& y) {
  return y >= x;
}

/* uchar8 relational less than or equal (<=) */
inline char8 operator<=(const uchar8& x, const uchar8& y) {
  return y >= x;
}
inline char8 operator<=(const uchar8& x, uchar y) {
  return y >= x;
}
inline char8 operator<=(uchar x, const uchar8& y) {
  return y >= x;
}

/* uchar16 relational less than or equal (<=) */
inline char16 operator<=(const uchar16& x, const uchar16& y) {
  return y >= x;
}
inline char16 operator<=(const uchar16& x, uchar y) {
  return y >= x;
}
inline char16 operator<=(uchar x, const uchar16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* uchar2 equal (==) */
inline char2 operator==(const uchar2& x, const uchar2& y) {
  char2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline char2 operator==(const uchar2& x, uchar y) {
  char2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline char2 operator==(uchar x, const uchar2& y) {
  char2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* uchar3 equal (==) */
inline char3 operator==(const uchar3& x, const uchar3& y) {
  char3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline char3 operator==(const uchar3& x, uchar y) {
  char3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline char3 operator==(uchar x, const uchar3& y) {
  char3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* uchar4 equal (==) */
inline char4 operator==(const uchar4& x, const uchar4& y) {
  char4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline char4 operator==(const uchar4& x, uchar y) {
  char4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline char4 operator==(uchar x, const uchar4& y) {
  char4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* uchar8 equal (==) */
inline char8 operator==(const uchar8& x, const uchar8& y) {
  char8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline char8 operator==(const uchar8& x, uchar y) {
  char8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline char8 operator==(uchar x, const uchar8& y) {
  char8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* uchar16 equal (==) */
inline char16 operator==(const uchar16& x, const uchar16& y) {
  char16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline char16 operator==(const uchar16& x, uchar y) {
  char16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline char16 operator==(uchar x, const uchar16& y) {
  char16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* uchar2 not equal (!=) */
inline char2 operator!=(const uchar2& x, const uchar2& y) {
  char2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline char2 operator!=(const uchar2& x, uchar y) {
  char2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline char2 operator!=(uchar x, const uchar2& y) {
  char2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* uchar3 not equal (!=) */
inline char3 operator!=(const uchar3& x, const uchar3& y) {
  char3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline char3 operator!=(const uchar3& x, uchar y) {
  char3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline char3 operator!=(uchar x, const uchar3& y) {
  char3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;

}
/* uchar4 not equal (!=) */
inline char4 operator!=(const uchar4& x, const uchar4& y) {
  char4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline char4 operator!=(const uchar4& x, uchar y) {
  char4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline char4 operator!=(uchar x, const uchar4& y) {
  char4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* uchar8 not equal (!=) */
inline char8 operator!=(const uchar8& x, const uchar8& y) {
  char8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline char8 operator!=(const uchar8& x, uchar y) {
  char8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline char8 operator!=(uchar x, const uchar8& y) {
  char8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* uchar16 not equal (!=) */
inline char16 operator!=(const uchar16& x, const uchar16& y) {
  char16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline char16 operator!=(const uchar16& x, uchar y) {
  char16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline char16 operator!=(uchar x, const uchar16& y) {
  char16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////

/* uchar2 bitwise and (&) */
inline uchar2 operator&(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, &, y);
  return rst;
}
inline uchar2 operator&(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, &, y);
  return rst;
}
inline uchar2 operator&(uchar x, const uchar2& y) {
  return y & x;
}

/* uchar3 bitwise and (&) */
inline uchar3 operator&(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, &, y);
  return rst;
}
inline uchar3 operator&(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, &, y);
  return rst;
}
inline uchar3 operator&(uchar x, const uchar3& y) {
  return y & x;
}


/* uchar4 bitwise and (&) */
inline uchar4 operator&(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, &, y);
  return rst;
}
inline uchar4 operator&(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, &, y);
  return rst;
}
inline uchar4 operator&(uchar x, const uchar4& y) {
  return y & x;
}

/* uchar8 bitwise and (&) */
inline uchar8 operator&(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, &, y);
  return rst;
}
inline uchar8 operator&(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, &, y);
  return rst;
}
inline uchar8 operator&(uchar x, const uchar8& y) {
  return y & x;
}

/* uchar16 bitwise and (&) */
inline uchar16 operator&(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, &, y);
  return rst;
}
inline uchar16 operator&(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, &, y);
  return rst;
}
inline uchar16 operator&(uchar x, const uchar16& y) {
  return y & x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////

/* uchar2 bitwise or (|) */
inline uchar2 operator|(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, |, y);
  return rst;
}
inline uchar2 operator|(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, |, y);
  return rst;
}
inline uchar2 operator|(uchar x, const uchar2& y) {
  return y | x;
}

/* uchar3 bitwise or (|) */
inline uchar3 operator|(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, |, y);
  return rst;
}
inline uchar3 operator|(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, |, y);
  return rst;
}
inline uchar3 operator|(uchar x, const uchar3& y) {
  return y | x;
}

/* uchar4 bitwise or (|) */
inline uchar4 operator|(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, |, y);
  return rst;
}
inline uchar4 operator|(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, |, y);
  return rst;
}
inline uchar4 operator|(uchar x, const uchar4& y) {
  return y | x;
}

/* uchar8 bitwise or (|) */
inline uchar8 operator|(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, |, y);
  return rst;
}
inline uchar8 operator|(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, |, y);
  return rst;
}
inline uchar8 operator|(uchar x, const uchar8& y) {
  return y | x;
}

/* uchar16 bitwise or (|) */
inline uchar16 operator|(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, |, y);
  return rst;
}
inline uchar16 operator|(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, |, y);
  return rst;
}
inline uchar16 operator|(uchar x, const uchar16& y) {
  return y | x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////

/* uchar2 bitwise exclusive or (^) */
inline uchar2 operator^(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, ^, y);
  return rst;
}
inline uchar2 operator^(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, ^, y);
  return rst;
}
inline uchar2 operator^(uchar x, const uchar2& y) {
  return y ^ x;
}

/* uchar3 bitwise exclusive or (^) */
inline uchar3 operator^(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, ^, y);
  return rst;
}
inline uchar3 operator^(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, ^, y);
  return rst;
}
inline uchar3 operator^(uchar x, const uchar3& y) {
  return y ^ x;
}

/* uchar4 bitwise exclusive or (^) */
inline uchar4 operator^(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, ^, y);
  return rst;
}
inline uchar4 operator^(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, ^, y);
  return rst;
}
inline uchar4 operator^(uchar x, const uchar4& y) {
  return y ^ x;
}

/* uchar8 bitwise exclusive or (^) */
inline uchar8 operator^(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, ^, y);
  return rst;
}
inline uchar8 operator^(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, ^, y);
  return rst;
}
inline uchar8 operator^(uchar x, const uchar8& y) {
  return y ^ x;
}

/* uchar16 bitwise exclusive or (^) */
inline uchar16 operator^(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, ^, y);
  return rst;
}
inline uchar16 operator^(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, ^, y);
  return rst;
}
inline uchar16 operator^(uchar x, const uchar16& y) {
  return y ^ x;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////

/* uchar2 bitwise not (~) */
inline uchar2 operator~(const uchar2& x) {
  uchar2 rst = __CL_OP_V2(~, x);
  return rst;
}

/* uchar3 bitwise not (~) */
inline uchar3 operator~(const uchar3& x) {
  uchar3 rst = __CL_OP_V3(~, x);
  return rst;
}

/* uchar4 bitwise not (~) */
inline uchar4 operator~(const uchar4& x) {
  uchar4 rst = __CL_OP_V4(~, x);
  return rst;
}

/* uchar8 bitwise not (~) */
inline uchar8 operator~(const uchar8& x) {
  uchar8 rst = __CL_OP_V8(~, x);
  return rst;
}

/* uchar16 bitwise not (~) */
inline uchar16 operator~(const uchar16& x) {
  uchar16 rst = __CL_OP_V16(~, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* uchar2 logical and (&&) */
inline char2 operator&&(const uchar2& x, const uchar2& y) {
  char2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline char2 operator&&(const uchar2& x, uchar y) {
  char2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline char2 operator&&(uchar x, const uchar2& y) {
  char2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* uchar3 logical and (&&) */
inline char3 operator&&(const uchar3& x, const uchar3& y) {
  char3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline char3 operator&&(const uchar3& x, uchar y) {
  char3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline char3 operator&&(uchar x, const uchar3& y) {
  char3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* uchar4 logical and (&&) */
inline char4 operator&&(const uchar4& x, const uchar4& y) {
  char4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline char4 operator&&(const uchar4& x, uchar y) {
  char4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline char4 operator&&(uchar x, const uchar4& y) {
  char4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* uchar8 logical and (&&) */
inline char8 operator&&(const uchar8& x, const uchar8& y) {
  char8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline char8 operator&&(const uchar8& x, uchar y) {
  char8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline char8 operator&&(uchar x, const uchar8& y) {
  char8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* uchar16 logical and (&&) */
inline char16 operator&&(const uchar16& x, const uchar16& y) {
  char16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline char16 operator&&(const uchar16& x, uchar y) {
  char16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline char16 operator&&(uchar x, const uchar16& y) {
  char16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////

/* uchar2 logical or (||) */
inline char2 operator||(const uchar2& x, const uchar2& y) {
  char2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline char2 operator||(const uchar2& x, uchar y) {
  char2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline char2 operator||(uchar x, const uchar2& y) {
  char2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* uchar3 logical or (||) */
inline char3 operator||(const uchar3& x, const uchar3& y) {
  char3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline char3 operator||(const uchar3& x, uchar y) {
  char3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline char3 operator||(uchar x, const uchar3& y) {
  char3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* uchar4 logical or (||) */
inline char4 operator||(const uchar4& x, const uchar4& y) {
  char4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline char4 operator||(const uchar4& x, uchar y) {
  char4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline char4 operator||(uchar x, const uchar4& y) {
  char4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* uchar8 logical or (||) */
inline char8 operator||(const uchar8& x, const uchar8& y) {
  char8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline char8 operator||(const uchar8& x, uchar y) {
  char8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline char8 operator||(uchar x, const uchar8& y) {
  char8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* uchar16 logical or (||) */
inline char16 operator||(const uchar16& x, const uchar16& y) {
  char16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline char16 operator||(const uchar16& x, uchar y) {
  char16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline char16 operator||(uchar x, const uchar16& y) {
  char16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* uchar2 logical not (!) */
inline char2 operator!(const uchar2& x) {
  char2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* uchar3 logical not (!) */
inline char3 operator!(const uchar3& x) {
  char3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* uchar4 logical not (!) */
inline char4 operator!(const uchar4& x) {
  char4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* uchar8 logical not (!) */
inline char8 operator!(const uchar8& x) {
  char8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* uchar16 logical not (!) */
inline char16 operator!(const uchar16& x) {
  char16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////

/* uchar2 right-shift (>>) */
inline uchar2 operator>>(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, >>, (y & 0x7));
  return rst;
}
inline uchar2 operator>>(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, >>, (y & 0x7));
  return rst;
}
/* uchar3 right-shift (>>) */
inline uchar3 operator>>(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, >>, (y & 0x7));
  return rst;
}
inline uchar3 operator>>(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, >>, (y & 0x7));
  return rst;
}

/* uchar4 right-shift (>>) */
inline uchar4 operator>>(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, >>, (y & 0x7));
  return rst;
}
inline uchar4 operator>>(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, >>, (y & 0x7));
  return rst;
}

/* uchar8 right-shift (>>) */
inline uchar8 operator>>(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, >>, (y & 0x7));
  return rst;
}
inline uchar8 operator>>(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, >>, (y & 0x7));
  return rst;
}

/* uchar16 right-shift (>>) */
inline uchar16 operator>>(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, >>, (y & 0x7));
  return rst;
}
inline uchar16 operator>>(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, >>, (y & 0x7));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////

/* uchar2 left-shift (<<) */
inline uchar2 operator<<(const uchar2& x, const uchar2& y) {
  uchar2 rst = __CL_V2_OP_V2(x, <<, (y & 0x7));
  return rst;
}
inline uchar2 operator<<(const uchar2& x, uchar y) {
  uchar2 rst = __CL_V2_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* uchar3 left-shift (<<) */
inline uchar3 operator<<(const uchar3& x, const uchar3& y) {
  uchar3 rst = __CL_V3_OP_V3(x, <<, (y & 0x7));
  return rst;
}
inline uchar3 operator<<(const uchar3& x, uchar y) {
  uchar3 rst = __CL_V3_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* uchar4 left-shift (<<) */
inline uchar4 operator<<(const uchar4& x, const uchar4& y) {
  uchar4 rst = __CL_V4_OP_V4(x, <<, (y & 0x7));
  return rst;
}
inline uchar4 operator<<(const uchar4& x, uchar y) {
  uchar4 rst = __CL_V4_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* uchar8 left-shift (<<) */
inline uchar8 operator<<(const uchar8& x, const uchar8& y) {
  uchar8 rst = __CL_V8_OP_V8(x, <<, (y & 0x7));
  return rst;
}
inline uchar8 operator<<(const uchar8& x, uchar y) {
  uchar8 rst = __CL_V8_OP_S(x, <<, (y & 0x7));
  return rst;
}

/* uchar16 left-shift (<<) */
inline uchar16 operator<<(const uchar16& x, const uchar16& y) {
  uchar16 rst = __CL_V16_OP_V16(x, <<, (y & 0x7));
  return rst;
}
inline uchar16 operator<<(const uchar16& x, uchar y) {
  uchar16 rst = __CL_V16_OP_S(x, <<, (y & 0x7));
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* uchar2 assignment add into (+=) */
inline uchar2 operator+=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, +, y);
}
inline uchar2 operator+=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, +, y);
}

/* uchar3 assignment add into (+=) */
inline uchar3 operator+=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, +, y);
}
inline uchar3 operator+=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, +, y);
}

/* uchar4 assignment add into (+=) */
inline uchar4 operator+=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, +, y);
}
inline uchar4 operator+=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, +, y);
}

/* uchar8 assignment add into (+=) */
inline uchar8 operator+=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, +, y);
}
inline uchar8 operator+=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, +, y);
}

/* uchar16 assignment add into (+=) */
inline uchar16 operator+=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, +, y);
}
inline uchar16 operator+=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* uchar2 assignment subtract from (-=) */
inline uchar2 operator-=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, -, y);
}
inline uchar2 operator-=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, -, y);
}

/* uchar3 assignment subtract from (-=) */
inline uchar3 operator-=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, -, y);
}
inline uchar3 operator-=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, -, y);
}

/* uchar4 assignment subtract from (-=) */
inline uchar4 operator-=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, -, y);
}
inline uchar4 operator-=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, -, y);
}

/* uchar8 assignment subtract from (-=) */
inline uchar8 operator-=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, -, y);
}
inline uchar8 operator-=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, -, y);
}

/* uchar16 assignment subtract from (-=) */
inline uchar16 operator-=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, -, y);
}
inline uchar16 operator-=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* uchar2 assignment multiply into (*=) */
inline uchar2 operator*=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, *, y);
}
inline uchar2 operator*=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, *, y);
}

/* uchar3 assignment multiply into (*=) */
inline uchar3 operator*=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, *, y);
}
inline uchar3 operator*=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, *, y);
}

/* uchar4 assignment multiply into (*=) */
inline uchar4 operator*=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, *, y);
}
inline uchar4 operator*=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, *, y);
}

/* uchar8 assignment multiply into (*=) */
inline uchar8 operator*=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, *, y);
}
inline uchar8 operator*=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, *, y);
}

/* uchar16 assignment multiply into (*=) */
inline uchar16 operator*=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, *, y);
}
inline uchar16 operator*=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* uchar2 assignment divide into (/=) */
inline uchar2 operator/=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, /, y);
}
inline uchar2 operator/=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, /, y);
}

/* uchar3 assignment divide into (/=) */
inline uchar3 operator/=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, /, y);
}
inline uchar3 operator/=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, /, y);
}

/* uchar4 assignment divide into (/=) */
inline uchar4 operator/=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, /, y);
}
inline uchar4 operator/=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, /, y);
}

/* uchar8 assignment divide into (/=) */
inline uchar8 operator/=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, /, y);
}
inline uchar8 operator/=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, /, y);
}

/* uchar16 assignment divide into (/=) */
inline uchar16 operator/=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, /, y);
}
inline uchar16 operator/=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////

/* uchar2 assignment modulus into (%=) */
inline uchar2 operator%=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, %, y);
}
inline uchar2 operator%=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, %, y);
}

/* uchar3 assignment modulus into (%=) */
inline uchar3 operator%=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, %, y);
}
inline uchar3 operator%=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, %, y);
}

/* uchar4 assignment modulus into (%=) */
inline uchar4 operator%=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, %, y);
}
inline uchar4 operator%=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, %, y);
}

/* uchar8 assignment modulus into (%=) */
inline uchar8 operator%=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, %, y);
}
inline uchar8 operator%=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, %, y);
}

/* uchar16 assignment modulus into (%=) */
inline uchar16 operator%=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, %, y);
}
inline uchar16 operator%=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, %, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////

/* uchar2 assignment left shift by (<<=) */
inline uchar2 operator<<=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, <<, y);
}
inline uchar2 operator<<=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, <<, y);
}

/* uchar3 assignment left shift by (<<=) */
inline uchar3 operator<<=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, <<, y);
}
inline uchar3 operator<<=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, <<, y);
}


/* uchar4 assignment left shift by (<<=) */
inline uchar4 operator<<=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, <<, y);
}
inline uchar4 operator<<=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, <<, y);
}

/* uchar8 assignment left shift by (<<=) */
inline uchar8 operator<<=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, <<, y);
}
inline uchar8 operator<<=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, <<, y);
}

/* uchar16 assignment left shift by (<<=) */
inline uchar16 operator<<=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, <<, y);
}
inline uchar16 operator<<=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, <<, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////

/* uchar2 assignment right shift by (>>=) */
inline uchar2 operator>>=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, >>, y);
}
inline uchar2 operator>>=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, >>, y);
}

/* uchar3 assignment right shift by (>>=) */
inline uchar3 operator>>=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, >>, y);
}
inline uchar3 operator>>=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, >>, y);
}
/* uchar4 assignment right shift by (>>=) */
inline uchar4 operator>>=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, >>, y);
}
inline uchar4 operator>>=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, >>, y);
}

/* uchar8 assignment right shift by (>>=) */
inline uchar8 operator>>=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, >>, y);
}
inline uchar8 operator>>=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, >>, y);
}

/* uchar16 assignment right shift by (>>=) */
inline uchar16 operator>>=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, >>, y);
}
inline uchar16 operator>>=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, >>, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////

/* uchar2 assignment and into (&=) */
inline uchar2 operator&=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, &, y);
}
inline uchar2 operator&=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, &, y);
}

/* uchar3 assignment and into (&=) */
inline uchar3 operator&=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, &, y);
}
inline uchar3 operator&=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, &, y);
}

/* uchar4 assignment and into (&=) */
inline uchar4 operator&=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, &, y);
}
inline uchar4 operator&=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, &, y);
}

/* uchar8 assignment and into (&=) */
inline uchar8 operator&=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, &, y);
}
inline uchar8 operator&=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, &, y);
}

/* uchar16 assignment and into (&=) */
inline uchar16 operator&=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, &, y);
}
inline uchar16 operator&=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, &, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////

/* uchar2 assignment inclusive or into (|=) */
inline uchar2 operator|=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, |, y);
}
inline uchar2 operator|=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, |, y);
}

/* uchar3 assignment inclusive or into (|=) */
inline uchar3 operator|=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, |, y);
}
inline uchar3 operator|=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, |, y);

}
/* uchar4 assignment inclusive or into (|=) */
inline uchar4 operator|=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, |, y);
}
inline uchar4 operator|=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, |, y);
}

/* uchar8 assignment inclusive or into (|=) */
inline uchar8 operator|=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, |, y);
}
inline uchar8 operator|=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, |, y);
}

/* uchar16 assignment inclusive or into (|=) */
inline uchar16 operator|=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, |, y);
}
inline uchar16 operator|=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, |, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////

/* uchar2 assignment exclusive or into (^=) */
inline uchar2 operator^=(uchar2 &x, const uchar2& y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, ^, y);
}
inline uchar2 operator^=(uchar2 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar2, x, ^, y);
}

/* uchar3 assignment exclusive or into (^=) */
inline uchar3 operator^=(uchar3 &x, const uchar3& y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, ^, y);
}
inline uchar3 operator^=(uchar3 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar3, x, ^, y);
}

/* uchar4 assignment exclusive or into (^=) */
inline uchar4 operator^=(uchar4 &x, const uchar4& y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, ^, y);
}
inline uchar4 operator^=(uchar4 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar4, x, ^, y);
}

/* uchar8 assignment exclusive or into (^=) */
inline uchar8 operator^=(uchar8 &x, const uchar8& y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, ^, y);
}
inline uchar8 operator^=(uchar8 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar8, x, ^, y);
}

/* uchar16 assignment exclusive or into (^=) */
inline uchar16 operator^=(uchar16 &x, const uchar16& y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, ^, y);
}
inline uchar16 operator^=(uchar16 &x, uchar y) {
  __CL_V_OP_ASSIGN_V(uchar16, x, ^, y);
}

#endif //__CL_OPS_UucharN_H

