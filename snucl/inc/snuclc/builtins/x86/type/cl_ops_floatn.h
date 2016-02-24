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

#ifndef __CL_OPS_FLOATN_H
#define __CL_OPS_FLOATN_H

#include <cl_cpu_types.h>
#include <type/cl_ops_util.h>


///////////////////////////////////////////////////////////
/// 6.3.a. ADDITION (+)
///////////////////////////////////////////////////////////

/* float2 addition (+) */
inline float2 operator+(const float2& x, const float2& y) {
  float2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline float2 operator+(const float2& x, float y) {
  float2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline float2 operator+(float x, const float2& y) {
  return y + x;
}

/* float3 addition (+) */
inline float3 operator+(const float3& x, const float3& y) {
  float3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline float3 operator+(const float3& x, float y) {
  float3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline float3 operator+(float x, const float3& y) {
  return y + x;
}

/* float4 addition (+) */
inline float4 operator+(const float4& x, const float4& y) {
  float4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline float4 operator+(const float4& x, float y) {
  float4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline float4 operator+(float x, const float4& y) {
  return y + x;
}

/* float8 addition (+) */
inline float8 operator+(const float8& x, const float8& y) {
  float8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline float8 operator+(const float8& x, float y) {
  float8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline float8 operator+(float x, const float8& y) {
  return y + x;
}

/* float16 addition (+) */
inline float16 operator+(const float16& x, const float16& y) {
  float16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline float16 operator+(const float16& x, float y) {
  float16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline float16 operator+(float x, const float16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* float2 subtraction (-) */
inline float2 operator-(const float2& x, const float2& y) {
  float2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline float2 operator-(const float2& x, float y) {
  float2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline float2 operator-(float x, const float2& y) {
  float2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* float3 subtraction (-) */
inline float3 operator-(const float3& x, const float3& y) {
  float3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline float3 operator-(const float3& x, float y) {
  float3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline float3 operator-(float x, const float3& y) {
  float3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* float4 subtraction (-) */
inline float4 operator-(const float4& x, const float4& y) {
  float4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline float4 operator-(const float4& x, float y) {
  float4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline float4 operator-(float x, const float4& y) {
  float4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* float8 subtraction (-) */
inline float8 operator-(const float8& x, const float8& y) {
  float8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline float8 operator-(const float8& x, float y) {
  float8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline float8 operator-(float x, const float8& y) {
  float8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* float16 subtraction (-) */
inline float16 operator-(const float16& x, const float16& y) {
  float16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline float16 operator-(const float16& x, float y) {
  float16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline float16 operator-(float x, const float16& y) {
  float16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* float2 multiplication (*) */
inline float2 operator*(const float2& x, const float2& y) {
  float2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline float2 operator*(const float2& x, float y) {
  float2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline float2 operator*(float x, const float2& y) {
  return y * x;
}

/* float3 multiplication (*) */
inline float3 operator*(const float3& x, const float3& y) {
  float3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline float3 operator*(const float3& x, float y) {
  float3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline float3 operator*(float x, const float3& y) {
  return y * x;

}
/* float4 multiplication (*) */
inline float4 operator*(const float4& x, const float4& y) {
  float4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline float4 operator*(const float4& x, float y) {
  float4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline float4 operator*(float x, const float4& y) {
  return y * x;
}

/* float8 multiplication (*) */
inline float8 operator*(const float8& x, const float8& y) {
  float8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline float8 operator*(const float8& x, float y) {
  float8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline float8 operator*(float x, const float8& y) {
  return y * x;
}

/* float16 multiplication (*) */
inline float16 operator*(const float16& x, const float16& y) {
  float16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline float16 operator*(const float16& x, float y) {
  float16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline float16 operator*(float x, const float16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* float2 division (/) */
inline float2 operator/(const float2& x, const float2& y) {
  float2 rst = __CL_V2_OP_V2(x, /, y);
  return rst;
}
inline float2 operator/(const float2& x, float y) {
  float2 rst = __CL_V2_OP_S(x, /, y);
  return rst;
}
inline float2 operator/(float x, const float2& y) {
  float2 rst = __CL_S_OP_V2(x, /, y);
  return rst;
}

/* float3 division (/) */
inline float3 operator/(const float3& x, const float3& y) {
  float3 rst = __CL_V3_OP_V3(x, /, y);
  return rst;
}
inline float3 operator/(const float3& x, float y) {
  float3 rst = __CL_V3_OP_S(x, /, y);
  return rst;
}
inline float3 operator/(float x, const float3& y) {
  float3 rst = __CL_S_OP_V3(x, /, y);
  return rst;

}
/* float4 division (/) */
inline float4 operator/(const float4& x, const float4& y) {
  float4 rst = __CL_V4_OP_V4(x, /, y);
  return rst;
}
inline float4 operator/(const float4& x, float y) {
  float4 rst = __CL_V4_OP_S(x, /, y);
  return rst;
}
inline float4 operator/(float x, const float4& y) {
  float4 rst = __CL_S_OP_V4(x, /, y);
  return rst;
}

/* float8 division (/) */
inline float8 operator/(const float8& x, const float8& y) {
  float8 rst = __CL_V8_OP_V8(x, /, y);
  return rst;
}
inline float8 operator/(const float8& x, float y) {
  float8 rst = __CL_V8_OP_S(x, /, y);
  return rst;
}
inline float8 operator/(float x, const float8& y) {
  float8 rst = __CL_S_OP_V8(x, /, y);
  return rst;
}

/* float16 division (/) */
inline float16 operator/(const float16& x, const float16& y) {
  float16 rst = __CL_V16_OP_V16(x, /, y);
  return rst;
}
inline float16 operator/(const float16& x, float y) {
  float16 rst = __CL_V16_OP_S(x, /, y);
  return rst;
}
inline float16 operator/(float x, const float16& y) {
  float16 rst = __CL_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////
/* Remainder (%) does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* float2 unary positive (+) */
inline float2 operator+(const float2& x) {
  return x;
}

/* float3 unary positive (+) */
inline float3 operator+(const float3& x) {
  return x;
}


/* float4 unary positive (+) */
inline float4 operator+(const float4& x) {
  return x;
}

/* float8 unary positive (+) */
inline float8 operator+(const float8& x) {
  return x;
}

/* float16 unary positive (+) */
inline float16 operator+(const float16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* float2 unary negative (-) */
inline float2 operator-(const float2& x) {
  float2 rst = __CL_OP_V2(-, x);
  return rst;
}

/* float3 unary negative (-) */
inline float3 operator-(const float3& x) {
  float3 rst = __CL_OP_V3(-, x);
  return rst;
}


/* float4 unary negative (-) */
inline float4 operator-(const float4& x) {
  float4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* float8 unary negative (-) */
inline float8 operator-(const float8& x) {
  float8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* float16 unary negative (-) */
inline float16 operator-(const float16& x) {
  float16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////
/* Unary post-increment does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////
/* Unary pre-increment does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////
/* Unary post-decrement does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////
/* Unary pre-decrement does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* float2 relational greater than (>) */
inline int2 operator>(const float2& x, const float2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline int2 operator>(const float2& x, float y) {
  int2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline int2 operator>(float x, const float2& y) {
  int2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* float3 relational greater than (>) */
inline int3 operator>(const float3& x, const float3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline int3 operator>(const float3& x, float y) {
  int3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline int3 operator>(float x, const float3& y) {
  int3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}

/* float4 relational greater than (>) */
inline int4 operator>(const float4& x, const float4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline int4 operator>(const float4& x, float y) {
  int4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline int4 operator>(float x, const float4& y) {
  int4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* float8 relational greater than (>) */
inline int8 operator>(const float8& x, const float8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline int8 operator>(const float8& x, float y) {
  int8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline int8 operator>(float x, const float8& y) {
  int8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* float16 relational greater than (>) */
inline int16 operator>(const float16& x, const float16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline int16 operator>(const float16& x, float y) {
  int16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline int16 operator>(float x, const float16& y) {
  int16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* float2 relational less than (<) */
inline int2 operator<(const float2& x, const float2& y) {
  return y > x;
}
inline int2 operator<(const float2& x, float y) {
  return y > x;
}
inline int2 operator<(float x, const float2& y) {
  return y > x;
}

/* float3 relational less than (<) */
inline int3 operator<(const float3& x, const float3& y) {
  return y > x;
}
inline int3 operator<(const float3& x, float y) {
  return y > x;
}
inline int3 operator<(float x, const float3& y) {
  return y > x;
}

/* float4 relational less than (<) */
inline int4 operator<(const float4& x, const float4& y) {
  return y > x;
}
inline int4 operator<(const float4& x, float y) {
  return y > x;
}
inline int4 operator<(float x, const float4& y) {
  return y > x;
}

/* float8 relational less than (<) */
inline int8 operator<(const float8& x, const float8& y) {
  return y > x;
}
inline int8 operator<(const float8& x, float y) {
  return y > x;
}
inline int8 operator<(float x, const float8& y) {
  return y > x;
}

/* float16 relational less than (<) */
inline int16 operator<(const float16& x, const float16& y) {
  return y > x;
}
inline int16 operator<(const float16& x, float y) {
  return y > x;
}
inline int16 operator<(float x, const float16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* float2 relational greater than or equal (>=) */
inline int2 operator>=(const float2& x, const float2& y) {
  int2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline int2 operator>=(const float2& x, float y) {
  int2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline int2 operator>=(float x, const float2& y) {
  int2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* float3 relational greater than or equal (>=) */
inline int3 operator>=(const float3& x, const float3& y) {
  int3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline int3 operator>=(const float3& x, float y) {
  int3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline int3 operator>=(float x, const float3& y) {
  int3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;

}
/* float4 relational greater than or equal (>=) */
inline int4 operator>=(const float4& x, const float4& y) {
  int4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline int4 operator>=(const float4& x, float y) {
  int4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline int4 operator>=(float x, const float4& y) {
  int4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* float8 relational greater than or equal (>=) */
inline int8 operator>=(const float8& x, const float8& y) {
  int8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline int8 operator>=(const float8& x, float y) {
  int8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline int8 operator>=(float x, const float8& y) {
  int8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* float16 relational greater than or equal (>=) */
inline int16 operator>=(const float16& x, const float16& y) {
  int16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline int16 operator>=(const float16& x, float y) {
  int16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline int16 operator>=(float x, const float16& y) {
  int16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* float2 relational less than or equal (<=) */
inline int2 operator<=(const float2& x, const float2& y) {
  return y >= x;
}
inline int2 operator<=(const float2& x, float y) {
  return y >= x;
}
inline int2 operator<=(float x, const float2& y) {
  return y >= x;
}

/* float3 relational less than or equal (<=) */
inline int3 operator<=(const float3& x, const float3& y) {
  return y >= x;
}
inline int3 operator<=(const float3& x, float y) {
  return y >= x;
}
inline int3 operator<=(float x, const float3& y) {
  return y >= x;
}
/* float4 relational less than or equal (<=) */
inline int4 operator<=(const float4& x, const float4& y) {
  return y >= x;
}
inline int4 operator<=(const float4& x, float y) {
  return y >= x;
}
inline int4 operator<=(float x, const float4& y) {
  return y >= x;
}

/* float8 relational less than or equal (<=) */
inline int8 operator<=(const float8& x, const float8& y) {
  return y >= x;
}
inline int8 operator<=(const float8& x, float y) {
  return y >= x;
}
inline int8 operator<=(float x, const float8& y) {
  return y >= x;
}

/* float16 relational less than or equal (<=) */
inline int16 operator<=(const float16& x, const float16& y) {
  return y >= x;
}
inline int16 operator<=(const float16& x, float y) {
  return y >= x;
}
inline int16 operator<=(float x, const float16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* float2 equal (==) */
inline int2 operator==(const float2& x, const float2& y) {
  int2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline int2 operator==(const float2& x, float y) {
  int2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline int2 operator==(float x, const float2& y) {
  int2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* float3 equal (==) */
inline int3 operator==(const float3& x, const float3& y) {
  int3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline int3 operator==(const float3& x, float y) {
  int3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline int3 operator==(float x, const float3& y) {
  int3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* float4 equal (==) */
inline int4 operator==(const float4& x, const float4& y) {
  int4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline int4 operator==(const float4& x, float y) {
  int4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline int4 operator==(float x, const float4& y) {
  int4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* float8 equal (==) */
inline int8 operator==(const float8& x, const float8& y) {
  int8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline int8 operator==(const float8& x, float y) {
  int8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline int8 operator==(float x, const float8& y) {
  int8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* float16 equal (==) */
inline int16 operator==(const float16& x, const float16& y) {
  int16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline int16 operator==(const float16& x, float y) {
  int16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline int16 operator==(float x, const float16& y) {
  int16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* float2 not equal (!=) */
inline int2 operator!=(const float2& x, const float2& y) {
  int2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline int2 operator!=(const float2& x, float y) {
  int2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline int2 operator!=(float x, const float2& y) {
  int2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* float3 not equal (!=) */
inline int3 operator!=(const float3& x, const float3& y) {
  int3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline int3 operator!=(const float3& x, float y) {
  int3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline int3 operator!=(float x, const float3& y) {
  int3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;
}

/* float4 not equal (!=) */
inline int4 operator!=(const float4& x, const float4& y) {
  int4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline int4 operator!=(const float4& x, float y) {
  int4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline int4 operator!=(float x, const float4& y) {
  int4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* float8 not equal (!=) */
inline int8 operator!=(const float8& x, const float8& y) {
  int8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline int8 operator!=(const float8& x, float y) {
  int8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline int8 operator!=(float x, const float8& y) {
  int8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* float16 not equal (!=) */
inline int16 operator!=(const float16& x, const float16& y) {
  int16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline int16 operator!=(const float16& x, float y) {
  int16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline int16 operator!=(float x, const float16& y) {
  int16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////
/* Bitwise and does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////
/* Bitwise or does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////
/* Bitwise exclusive or does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////
/* Bitwise not does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* float2 logical and (&&) */
inline int2 operator&&(const float2& x, const float2& y) {
  int2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline int2 operator&&(const float2& x, float y) {
  int2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline int2 operator&&(float x, const float2& y) {
  int2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* float3 logical and (&&) */
inline int3 operator&&(const float3& x, const float3& y) {
  int3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline int3 operator&&(const float3& x, float y) {
  int3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline int3 operator&&(float x, const float3& y) {
  int3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* float4 logical and (&&) */
inline int4 operator&&(const float4& x, const float4& y) {
  int4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline int4 operator&&(const float4& x, float y) {
  int4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline int4 operator&&(float x, const float4& y) {
  int4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* float8 logical and (&&) */
inline int8 operator&&(const float8& x, const float8& y) {
  int8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline int8 operator&&(const float8& x, float y) {
  int8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline int8 operator&&(float x, const float8& y) {
  int8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* float16 logical and (&&) */
inline int16 operator&&(const float16& x, const float16& y) {
  int16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline int16 operator&&(const float16& x, float y) {
  int16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline int16 operator&&(float x, const float16& y) {
  int16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////
/* Logical or does not operate on floatn types. */

/* float2 logical or (||) */
inline int2 operator||(const float2& x, const float2& y) {
  int2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline int2 operator||(const float2& x, float y) {
  int2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline int2 operator||(float x, const float2& y) {
  int2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* float3 logical or (||) */
inline int3 operator||(const float3& x, const float3& y) {
  int3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline int3 operator||(const float3& x, float y) {
  int3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline int3 operator||(float x, const float3& y) {
  int3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* float4 logical or (||) */
inline int4 operator||(const float4& x, const float4& y) {
  int4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline int4 operator||(const float4& x, float y) {
  int4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline int4 operator||(float x, const float4& y) {
  int4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* float8 logical or (||) */
inline int8 operator||(const float8& x, const float8& y) {
  int8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline int8 operator||(const float8& x, float y) {
  int8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline int8 operator||(float x, const float8& y) {
  int8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* float16 logical or (||) */
inline int16 operator||(const float16& x, const float16& y) {
  int16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline int16 operator||(const float16& x, float y) {
  int16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline int16 operator||(float x, const float16& y) {
  int16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* float2 logical not (!) */
inline int2 operator!(const float2& x) {
  int2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* float3 logical not (!) */
inline int3 operator!(const float3& x) {
  int3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* float4 logical not (!) */
inline int4 operator!(const float4& x) {
  int4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* float8 logical not (!) */
inline int8 operator!(const float8& x) {
  int8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* float16 logical not (!) */
inline int16 operator!(const float16& x) {
  int16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////
/* Right-shift does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////
/* Left-shift does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* float2 assignment add into (+=) */
inline float2 operator+=(float2 &x, const float2& y) {
  __CL_V_OP_ASSIGN_V(float2, x, +, y);
}
inline float2 operator+=(float2 &x, float y) {
  __CL_V_OP_ASSIGN_V(float2, x, +, y);
}

/* float3 assignment add into (+=) */
inline float3 operator+=(float3 &x, const float3& y) {
  __CL_V_OP_ASSIGN_V(float3, x, +, y);
}
inline float3 operator+=(float3 &x, float y) {
  __CL_V_OP_ASSIGN_V(float3, x, +, y);
}

/* float4 assignment add into (+=) */
inline float4 operator+=(float4 &x, const float4& y) {
  __CL_V_OP_ASSIGN_V(float4, x, +, y);
}
inline float4 operator+=(float4 &x, float y) {
  __CL_V_OP_ASSIGN_V(float4, x, +, y);
}

/* float8 assignment add into (+=) */
inline float8 operator+=(float8 &x, const float8& y) {
  __CL_V_OP_ASSIGN_V(float8, x, +, y);
}
inline float8 operator+=(float8 &x, float y) {
  __CL_V_OP_ASSIGN_V(float8, x, +, y);
}

/* float16 assignment add into (+=) */
inline float16 operator+=(float16 &x, const float16& y) {
  __CL_V_OP_ASSIGN_V(float16, x, +, y);
}
inline float16 operator+=(float16 &x, float y) {
  __CL_V_OP_ASSIGN_V(float16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* float2 assignment subtract from (-=) */
inline float2 operator-=(float2 &x, const float2& y) {
  __CL_V_OP_ASSIGN_V(float2, x, -, y);
}
inline float2 operator-=(float2 &x, float y) {
  __CL_V_OP_ASSIGN_V(float2, x, -, y);
}

/* float3 assignment subtract from (-=) */
inline float3 operator-=(float3 &x, const float3& y) {
  __CL_V_OP_ASSIGN_V(float3, x, -, y);
}
inline float3 operator-=(float3 &x, float y) {
  __CL_V_OP_ASSIGN_V(float3, x, -, y);
}

/* float4 assignment subtract from (-=) */
inline float4 operator-=(float4 &x, const float4& y) {
  __CL_V_OP_ASSIGN_V(float4, x, -, y);
}
inline float4 operator-=(float4 &x, float y) {
  __CL_V_OP_ASSIGN_V(float4, x, -, y);
}

/* float8 assignment subtract from (-=) */
inline float8 operator-=(float8 &x, const float8& y) {
  __CL_V_OP_ASSIGN_V(float8, x, -, y);
}
inline float8 operator-=(float8 &x, float y) {
  __CL_V_OP_ASSIGN_V(float8, x, -, y);
}

/* float16 assignment subtract from (-=) */
inline float16 operator-=(float16 &x, const float16& y) {
  __CL_V_OP_ASSIGN_V(float16, x, -, y);
}
inline float16 operator-=(float16 &x, float y) {
  __CL_V_OP_ASSIGN_V(float16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* float2 assignment multiply into (*=) */
inline float2 operator*=(float2 &x, const float2& y) {
  __CL_V_OP_ASSIGN_V(float2, x, *, y);
}
inline float2 operator*=(float2 &x, float y) {
  __CL_V_OP_ASSIGN_V(float2, x, *, y);
}

/* float3 assignment multiply into (*=) */
inline float3 operator*=(float3 &x, const float3& y) {
  __CL_V_OP_ASSIGN_V(float3, x, *, y);
}
inline float3 operator*=(float3 &x, float y) {
  __CL_V_OP_ASSIGN_V(float3, x, *, y);
}

/* float4 assignment multiply into (*=) */
inline float4 operator*=(float4 &x, const float4& y) {
  __CL_V_OP_ASSIGN_V(float4, x, *, y);
}
inline float4 operator*=(float4 &x, float y) {
  __CL_V_OP_ASSIGN_V(float4, x, *, y);
}

/* float8 assignment multiply into (*=) */
inline float8 operator*=(float8 &x, const float8& y) {
  __CL_V_OP_ASSIGN_V(float8, x, *, y);
}
inline float8 operator*=(float8 &x, float y) {
  __CL_V_OP_ASSIGN_V(float8, x, *, y);
}

/* float16 assignment multiply into (*=) */
inline float16 operator*=(float16 &x, const float16& y) {
  __CL_V_OP_ASSIGN_V(float16, x, *, y);
}
inline float16 operator*=(float16 &x, float y) {
  __CL_V_OP_ASSIGN_V(float16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* float2 assignment divide into (/=) */
inline float2 operator/=(float2 &x, const float2& y) {
  __CL_V_OP_ASSIGN_V(float2, x, /, y);
}
inline float2 operator/=(float2 &x, float y) {
  __CL_V_OP_ASSIGN_V(float2, x, /, y);
}

/* float3 assignment divide into (/=) */
inline float3 operator/=(float3 &x, const float3& y) {
  __CL_V_OP_ASSIGN_V(float3, x, /, y);
}
inline float3 operator/=(float3 &x, float y) {
  __CL_V_OP_ASSIGN_V(float3, x, /, y);
}

/* float4 assignment divide into (/=) */
inline float4 operator/=(float4 &x, const float4& y) {
  __CL_V_OP_ASSIGN_V(float4, x, /, y);
}
inline float4 operator/=(float4 &x, float y) {
  __CL_V_OP_ASSIGN_V(float4, x, /, y);
}

/* float8 assignment divide into (/=) */
inline float8 operator/=(float8 &x, const float8& y) {
  __CL_V_OP_ASSIGN_V(float8, x, /, y);
}
inline float8 operator/=(float8 &x, float y) {
  __CL_V_OP_ASSIGN_V(float8, x, /, y);
}

/* float16 assignment divide into (/=) */
inline float16 operator/=(float16 &x, const float16& y) {
  __CL_V_OP_ASSIGN_V(float16, x, /, y);
}
inline float16 operator/=(float16 &x, float y) {
  __CL_V_OP_ASSIGN_V(float16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////
/* Does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////
/* Does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////
/* Does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////
/* Does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////
/* Does not operate on floatn types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////
/* Does not operate on floatn types. */


#endif //__CL_OPS_FLOATN_H

