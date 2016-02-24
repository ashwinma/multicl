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

#ifndef __CL_OPS_DOUBLEN_H
#define __CL_OPS_DOUBLEN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>


///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* ldouble2 addition (+) */
inline ldouble2 operator+(const ldouble2& x, const ldouble2& y) {
  ldouble2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline ldouble2 operator+(const ldouble2& x, ldouble y) {
  ldouble2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline ldouble2 operator+(ldouble x, const ldouble2& y) {
  return y + x;
}

/* ldouble3 addition (+) */
inline ldouble3 operator+(const ldouble3& x, const ldouble3& y) {
  ldouble3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline ldouble3 operator+(const ldouble3& x, ldouble y) {
  ldouble3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline ldouble3 operator+(ldouble x, const ldouble3& y) {
  return y + x;
}

/* ldouble4 addition (+) */
inline ldouble4 operator+(const ldouble4& x, const ldouble4& y) {
  ldouble4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline ldouble4 operator+(const ldouble4& x, ldouble y) {
  ldouble4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline ldouble4 operator+(ldouble x, const ldouble4& y) {
  return y + x;
}

/* ldouble8 addition (+) */
inline ldouble8 operator+(const ldouble8& x, const ldouble8& y) {
  ldouble8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline ldouble8 operator+(const ldouble8& x, ldouble y) {
  ldouble8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline ldouble8 operator+(ldouble x, const ldouble8& y) {
  return y + x;
}

/* ldouble16 addition (+) */
inline ldouble16 operator+(const ldouble16& x, const ldouble16& y) {
  ldouble16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline ldouble16 operator+(const ldouble16& x, ldouble y) {
  ldouble16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline ldouble16 operator+(ldouble x, const ldouble16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* ldouble2 subtraction (-) */
inline ldouble2 operator-(const ldouble2& x, const ldouble2& y) {
  ldouble2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline ldouble2 operator-(const ldouble2& x, ldouble y) {
  ldouble2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline ldouble2 operator-(ldouble x, const ldouble2& y) {
  ldouble2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* ldouble3 subtraction (-) */
inline ldouble3 operator-(const ldouble3& x, const ldouble3& y) {
  ldouble3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline ldouble3 operator-(const ldouble3& x, ldouble y) {
  ldouble3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline ldouble3 operator-(ldouble x, const ldouble3& y) {
  ldouble3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* ldouble4 subtraction (-) */
inline ldouble4 operator-(const ldouble4& x, const ldouble4& y) {
  ldouble4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline ldouble4 operator-(const ldouble4& x, ldouble y) {
  ldouble4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline ldouble4 operator-(ldouble x, const ldouble4& y) {
  ldouble4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* ldouble8 subtraction (-) */
inline ldouble8 operator-(const ldouble8& x, const ldouble8& y) {
  ldouble8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline ldouble8 operator-(const ldouble8& x, ldouble y) {
  ldouble8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline ldouble8 operator-(ldouble x, const ldouble8& y) {
  ldouble8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* ldouble16 subtraction (-) */
inline ldouble16 operator-(const ldouble16& x, const ldouble16& y) {
  ldouble16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline ldouble16 operator-(const ldouble16& x, ldouble y) {
  ldouble16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline ldouble16 operator-(ldouble x, const ldouble16& y) {
  ldouble16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* ldouble2 multiplication (*) */
inline ldouble2 operator*(const ldouble2& x, const ldouble2& y) {
  ldouble2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline ldouble2 operator*(const ldouble2& x, ldouble y) {
  ldouble2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline ldouble2 operator*(ldouble x, const ldouble2& y) {
  return y * x;
}

/* ldouble3 multiplication (*) */
inline ldouble3 operator*(const ldouble3& x, const ldouble3& y) {
  ldouble3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline ldouble3 operator*(const ldouble3& x, ldouble y) {
  ldouble3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline ldouble3 operator*(ldouble x, const ldouble3& y) {
  return y + x;
}

/* ldouble4 multiplication (*) */
inline ldouble4 operator*(const ldouble4& x, const ldouble4& y) {
  ldouble4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline ldouble4 operator*(const ldouble4& x, ldouble y) {
  ldouble4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline ldouble4 operator*(ldouble x, const ldouble4& y) {
  return y + x;
}

/* ldouble8 multiplication (*) */
inline ldouble8 operator*(const ldouble8& x, const ldouble8& y) {
  ldouble8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline ldouble8 operator*(const ldouble8& x, ldouble y) {
  ldouble8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline ldouble8 operator*(ldouble x, const ldouble8& y) {
  return y * x;
}

/* ldouble16 multiplication (*) */
inline ldouble16 operator*(const ldouble16& x, const ldouble16& y) {
  ldouble16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline ldouble16 operator*(const ldouble16& x, ldouble y) {
  ldouble16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline ldouble16 operator*(ldouble x, const ldouble16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* ldouble2 division (/) */
inline ldouble2 operator/(const ldouble2& x, const ldouble2& y) {
  ldouble2 rst = __CL_V2_OP_V2(x, /, y);
  return rst;
}
inline ldouble2 operator/(const ldouble2& x, ldouble y) {
  ldouble2 rst = __CL_V2_OP_S(x, /, y);
  return rst;
}
inline ldouble2 operator/(ldouble x, const ldouble2& y) {
  ldouble2 rst = __CL_S_OP_V2(x, /, y);
  return rst;
}

/* ldouble3 division (/) */
inline ldouble3 operator/(const ldouble3& x, const ldouble3& y) {
  ldouble3 rst = __CL_V3_OP_V3(x, /, y);
  return rst;
}
inline ldouble3 operator/(const ldouble3& x, ldouble y) {
  ldouble3 rst = __CL_V3_OP_S(x, /, y);
  return rst;
}
inline ldouble3 operator/(ldouble x, const ldouble3& y) {
  ldouble3 rst = __CL_S_OP_V3(x, /, y);
  return rst;
}

/* ldouble4 division (/) */
inline ldouble4 operator/(const ldouble4& x, const ldouble4& y) {
  ldouble4 rst = __CL_V4_OP_V4(x, /, y);
  return rst;
}
inline ldouble4 operator/(const ldouble4& x, ldouble y) {
  ldouble4 rst = __CL_V4_OP_S(x, /, y);
  return rst;
}
inline ldouble4 operator/(ldouble x, const ldouble4& y) {
  ldouble4 rst = __CL_S_OP_V4(x, /, y);
  return rst;
}

/* ldouble8 division (/) */
inline ldouble8 operator/(const ldouble8& x, const ldouble8& y) {
  ldouble8 rst = __CL_V8_OP_V8(x, /, y);
  return rst;
}
inline ldouble8 operator/(const ldouble8& x, ldouble y) {
  ldouble8 rst = __CL_V8_OP_S(x, /, y);
  return rst;
}
inline ldouble8 operator/(ldouble x, const ldouble8& y) {
  ldouble8 rst = __CL_S_OP_V8(x, /, y);
  return rst;
}

/* ldouble16 division (/) */
inline ldouble16 operator/(const ldouble16& x, const ldouble16& y) {
  ldouble16 rst = __CL_V16_OP_V16(x, /, y);
  return rst;
}
inline ldouble16 operator/(const ldouble16& x, ldouble y) {
  ldouble16 rst = __CL_V16_OP_S(x, /, y);
  return rst;
}
inline ldouble16 operator/(ldouble x, const ldouble16& y) {
  ldouble16 rst = __CL_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////
/* Remainder (%) does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* ldouble2 unary positive (+) */
inline ldouble2 operator+(const ldouble2& x) {
  return x;
}

/* ldouble3 unary positive (+) */
inline ldouble3 operator+(const ldouble3& x) {
  return x;
}

/* ldouble4 unary positive (+) */
inline ldouble4 operator+(const ldouble4& x) {
  return x;
}

/* ldouble8 unary positive (+) */
inline ldouble8 operator+(const ldouble8& x) {
  return x;
}

/* ldouble16 unary positive (+) */
inline ldouble16 operator+(const ldouble16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* ldouble2 unary negative (-) */
inline ldouble2 operator-(const ldouble2& x) {
  ldouble2 rst = __CL_OP_V2(-, x);
  return rst;
}

/* ldouble3 unary negative (-) */
inline ldouble3 operator-(const ldouble3& x) {
  ldouble3 rst = __CL_OP_V3(-, x);
  return rst;
}

/* ldouble4 unary negative (-) */
inline ldouble4 operator-(const ldouble4& x) {
  ldouble4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* ldouble8 unary negative (-) */
inline ldouble8 operator-(const ldouble8& x) {
  ldouble8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* ldouble16 unary negative (-) */
inline ldouble16 operator-(const ldouble16& x) {
  ldouble16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////
/* Unary post-increment does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////
/* Unary pre-increment does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////
/* Unary post-decrement does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////
/* Unary pre-decrement does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* ldouble2 relational greater than (>) */
inline int2 operator>(const ldouble2& x, const ldouble2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline int2 operator>(const ldouble2& x, ldouble y) {
  int2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline int2 operator>(ldouble x, const ldouble2& y) {
  int2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* ldouble3 relational greater than (>) */
inline int3 operator>(const ldouble3& x, const ldouble3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline int3 operator>(const ldouble3& x, ldouble y) {
  int3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline int3 operator>(ldouble x, const ldouble3& y) {
  int3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* ldouble4 relational greater than (>) */
inline int4 operator>(const ldouble4& x, const ldouble4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline int4 operator>(const ldouble4& x, ldouble y) {
  int4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline int4 operator>(ldouble x, const ldouble4& y) {
  int4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* ldouble8 relational greater than (>) */
inline int8 operator>(const ldouble8& x, const ldouble8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline int8 operator>(const ldouble8& x, ldouble y) {
  int8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline int8 operator>(ldouble x, const ldouble8& y) {
  int8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* ldouble16 relational greater than (>) */
inline int16 operator>(const ldouble16& x, const ldouble16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline int16 operator>(const ldouble16& x, ldouble y) {
  int16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline int16 operator>(ldouble x, const ldouble16& y) {
  int16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* ldouble2 relational less than (<) */
inline int2 operator<(const ldouble2& x, const ldouble2& y) {
  return y > x;
}
inline int2 operator<(const ldouble2& x, ldouble y) {
  return y > x;
}
inline int2 operator<(ldouble x, const ldouble2& y) {
  return y > x;
}

/* ldouble3 relational less than (<) */
inline int3 operator<(const ldouble3& x, const ldouble3& y) {
  return y > x;
}
inline int3 operator<(const ldouble3& x, ldouble y) {
  return y > x;
}
inline int3 operator<(ldouble x, const ldouble3& y) {
  return y > x;
}

/* ldouble4 relational less than (<) */
inline int4 operator<(const ldouble4& x, const ldouble4& y) {
  return y > x;
}
inline int4 operator<(const ldouble4& x, ldouble y) {
  return y > x;
}
inline int4 operator<(ldouble x, const ldouble4& y) {
  return y > x;
}

/* ldouble8 relational less than (<) */
inline int8 operator<(const ldouble8& x, const ldouble8& y) {
  return y > x;
}
inline int8 operator<(const ldouble8& x, ldouble y) {
  return y > x;
}
inline int8 operator<(ldouble x, const ldouble8& y) {
  return y > x;
}

/* ldouble16 relational less than (<) */
inline int16 operator<(const ldouble16& x, const ldouble16& y) {
  return y > x;
}
inline int16 operator<(const ldouble16& x, ldouble y) {
  return y > x;
}
inline int16 operator<(ldouble x, const ldouble16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* ldouble2 relational greater than or equal (>=) */
inline int2 operator>=(const ldouble2& x, const ldouble2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline int2 operator>=(const ldouble2& x, ldouble y) {
  int2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline int2 operator>=(ldouble x, const ldouble2& y) {
  int2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* ldouble3 relational greater than or equal (>=) */
inline int3 operator>=(const ldouble3& x, const ldouble3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline int3 operator>=(const ldouble3& x, ldouble y) {
  int3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline int3 operator>=(ldouble x, const ldouble3& y) {
  int3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* ldouble4 relational greater than or equal (>=) */
inline int4 operator>=(const ldouble4& x, const ldouble4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline int4 operator>=(const ldouble4& x, ldouble y) {
  int4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline int4 operator>=(ldouble x, const ldouble4& y) {
  int4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* ldouble8 relational greater than or equal (>=) */
inline int8 operator>=(const ldouble8& x, const ldouble8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline int8 operator>=(const ldouble8& x, ldouble y) {
  int8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline int8 operator>=(ldouble x, const ldouble8& y) {
  int8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* ldouble16 relational greater than or equal (>=) */
inline int16 operator>=(const ldouble16& x, const ldouble16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline int16 operator>=(const ldouble16& x, ldouble y) {
  int16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline int16 operator>=(ldouble x, const ldouble16& y) {
  int16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* ldouble2 relational less than or equal (<=) */
inline int2 operator<=(const ldouble2& x, const ldouble2& y) {
  return y >= x;
}
inline int2 operator<=(const ldouble2& x, ldouble y) {
  return y >= x;
}
inline int2 operator<=(ldouble x, const ldouble2& y) {
  return y >= x;
}

/* ldouble3 relational less than or equal (<=) */
inline int3 operator<=(const ldouble3& x, const ldouble3& y) {
  return y >= x;
}
inline int3 operator<=(const ldouble3& x, ldouble y) {
  return y >= x;
}
inline int3 operator<=(ldouble x, const ldouble3& y) {
  return y >= x;
}

/* ldouble4 relational less than or equal (<=) */
inline int4 operator<=(const ldouble4& x, const ldouble4& y) {
  return y >= x;
}
inline int4 operator<=(const ldouble4& x, ldouble y) {
  return y >= x;
}
inline int4 operator<=(ldouble x, const ldouble4& y) {
  return y >= x;
}

/* ldouble8 relational less than or equal (<=) */
inline int8 operator<=(const ldouble8& x, const ldouble8& y) {
  return y >= x;
}
inline int8 operator<=(const ldouble8& x, ldouble y) {
  return y >= x;
}
inline int8 operator<=(ldouble x, const ldouble8& y) {
  return y >= x;
}

/* ldouble16 relational less than or equal (<=) */
inline int16 operator<=(const ldouble16& x, const ldouble16& y) {
  return y >= x;
}
inline int16 operator<=(const ldouble16& x, ldouble y) {
  return y >= x;
}
inline int16 operator<=(ldouble x, const ldouble16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* ldouble2 equal (==) */
inline int2 operator==(const ldouble2& x, const ldouble2& y) {
  int2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline int2 operator==(const ldouble2& x, ldouble y) {
  int2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline int2 operator==(ldouble x, const ldouble2& y) {
  int2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* ldouble3 equal (==) */
inline int3 operator==(const ldouble3& x, const ldouble3& y) {
  int3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline int3 operator==(const ldouble3& x, ldouble y) {
  int3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline int3 operator==(ldouble x, const ldouble3& y) {
  int3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* ldouble4 equal (==) */
inline int4 operator==(const ldouble4& x, const ldouble4& y) {
  int4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline int4 operator==(const ldouble4& x, ldouble y) {
  int4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline int4 operator==(ldouble x, const ldouble4& y) {
  int4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* ldouble8 equal (==) */
inline int8 operator==(const ldouble8& x, const ldouble8& y) {
  int8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline int8 operator==(const ldouble8& x, ldouble y) {
  int8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline int8 operator==(ldouble x, const ldouble8& y) {
  int8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* ldouble16 equal (==) */
inline int16 operator==(const ldouble16& x, const ldouble16& y) {
  int16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline int16 operator==(const ldouble16& x, ldouble y) {
  int16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline int16 operator==(ldouble x, const ldouble16& y) {
  int16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* ldouble2 not equal (!=) */
inline int2 operator!=(const ldouble2& x, const ldouble2& y) {
  int2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline int2 operator!=(const ldouble2& x, ldouble y) {
  int2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline int2 operator!=(ldouble x, const ldouble2& y) {
  int2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* ldouble3 not equal (!=) */
inline int3 operator!=(const ldouble3& x, const ldouble3& y) {
  int3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline int3 operator!=(const ldouble3& x, ldouble y) {
  int3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline int3 operator!=(ldouble x, const ldouble3& y) {
  int3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;
}

/* ldouble4 not equal (!=) */
inline int4 operator!=(const ldouble4& x, const ldouble4& y) {
  int4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline int4 operator!=(const ldouble4& x, ldouble y) {
  int4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline int4 operator!=(ldouble x, const ldouble4& y) {
  int4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* ldouble8 not equal (!=) */
inline int8 operator!=(const ldouble8& x, const ldouble8& y) {
  int8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline int8 operator!=(const ldouble8& x, ldouble y) {
  int8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline int8 operator!=(ldouble x, const ldouble8& y) {
  int8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* ldouble16 not equal (!=) */
inline int16 operator!=(const ldouble16& x, const ldouble16& y) {
  int16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline int16 operator!=(const ldouble16& x, ldouble y) {
  int16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline int16 operator!=(ldouble x, const ldouble16& y) {
  int16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////
/* Bitwise and does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////
/* Bitwise or does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////
/* Bitwise exclusive or does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////
/* Bitwise not does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////
/* Logical and does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////
/* Logical or does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////
/* Logical not does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////
/* Right-shift does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////
/* Left-shift does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* ldouble2 assignment add into (+=) */
inline ldouble2 operator+=(ldouble2 &x, const ldouble2& y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, +, y);
}
inline ldouble2 operator+=(ldouble2 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, +, y);
}

/* ldouble3 assignment add into (+=) */
inline ldouble3 operator+=(ldouble3 &x, const ldouble3& y) {
  __CL_V_OP_ASSIGN_V(ldouble3, x, +, y);
}

/* ldouble4 assignment add into (+=) */
inline ldouble4 operator+=(ldouble4 &x, const ldouble4& y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, +, y);
}
inline ldouble4 operator+=(ldouble4 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, +, y);
}

/* ldouble8 assignment add into (+=) */
inline ldouble8 operator+=(ldouble8 &x, const ldouble8& y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, +, y);
}
inline ldouble8 operator+=(ldouble8 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, +, y);
}

/* ldouble16 assignment add into (+=) */
inline ldouble16 operator+=(ldouble16 &x, const ldouble16& y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, +, y);
}
inline ldouble16 operator+=(ldouble16 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* ldouble2 assignment subtract from (-=) */
inline ldouble2 operator-=(ldouble2 &x, const ldouble2& y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, -, y);
}
inline ldouble2 operator-=(ldouble2 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, -, y);
}

/* ldouble3 assignment subtract from (-=) */
inline ldouble3 operator-=(ldouble3 &x, const ldouble3& y) {
  __CL_V_OP_ASSIGN_V(ldouble3, x, -, y);
}
inline ldouble3 operator-=(ldouble3 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble3, x, -, y);
}

/* ldouble4 assignment subtract from (-=) */
inline ldouble4 operator-=(ldouble4 &x, const ldouble4& y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, -, y);
}
inline ldouble4 operator-=(ldouble4 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, -, y);
}

/* ldouble8 assignment subtract from (-=) */
inline ldouble8 operator-=(ldouble8 &x, const ldouble8& y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, -, y);
}
inline ldouble8 operator-=(ldouble8 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, -, y);
}

/* ldouble16 assignment subtract from (-=) */
inline ldouble16 operator-=(ldouble16 &x, const ldouble16& y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, -, y);
}
inline ldouble16 operator-=(ldouble16 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* ldouble2 assignment multiply into (*=) */
inline ldouble2 operator*=(ldouble2 &x, const ldouble2& y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, *, y);
}
inline ldouble2 operator*=(ldouble2 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, *, y);
}

/* ldouble3 assignment multiply into (*=) */
inline ldouble3 operator*=(ldouble3 &x, const ldouble3& y) {
  __CL_V_OP_ASSIGN_V(ldouble3, x, *, y);
}
inline ldouble3 operator*=(ldouble3 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble3, x, *, y);
}

/* ldouble4 assignment multiply into (*=) */
inline ldouble4 operator*=(ldouble4 &x, const ldouble4& y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, *, y);
}
inline ldouble4 operator*=(ldouble4 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, *, y);
}

/* ldouble8 assignment multiply into (*=) */
inline ldouble8 operator*=(ldouble8 &x, const ldouble8& y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, *, y);
}
inline ldouble8 operator*=(ldouble8 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, *, y);
}

/* ldouble16 assignment multiply into (*=) */
inline ldouble16 operator*=(ldouble16 &x, const ldouble16& y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, *, y);
}
inline ldouble16 operator*=(ldouble16 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* ldouble2 assignment divide into (/=) */
inline ldouble2 operator/=(ldouble2 &x, const ldouble2& y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, /, y);
}
inline ldouble2 operator/=(ldouble2 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble2, x, /, y);
}

/* ldouble3 assignment divide into (/=) */
inline ldouble3 operator/=(ldouble3 &x, const ldouble3& y) {
  __CL_V_OP_ASSIGN_V(ldouble3, x, /, y);
}
inline ldouble3 operator/=(ldouble3 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble3, x, /, y);
}

/* ldouble4 assignment divide into (/=) */
inline ldouble4 operator/=(ldouble4 &x, const ldouble4& y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, /, y);
}
inline ldouble4 operator/=(ldouble4 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble4, x, /, y);
}

/* ldouble8 assignment divide into (/=) */
inline ldouble8 operator/=(ldouble8 &x, const ldouble8& y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, /, y);
}
inline ldouble8 operator/=(ldouble8 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble8, x, /, y);
}

/* ldouble16 assignment divide into (/=) */
inline ldouble16 operator/=(ldouble16 &x, const ldouble16& y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, /, y);
}
inline ldouble16 operator/=(ldouble16 &x, ldouble y) {
  __CL_V_OP_ASSIGN_V(ldouble16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////
/* Does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////
/* Does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////
/* Does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////
/* Does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////
/* Does not operate on ldoublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////
/* Does not operate on ldoublen types. */


#endif //__CL_OPS_DOUBLEN_H


