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

/* double2 addition (+) */
inline double2 operator+(const double2& x, const double2& y) {
  double2 rst = __CL_V2_OP_V2(x, +, y);
  return rst;
}
inline double2 operator+(const double2& x, double y) {
  double2 rst = __CL_V2_OP_S(x, +, y);
  return rst;
}
inline double2 operator+(double x, const double2& y) {
  return y + x;
}

/* double3 addition (+) */
inline double3 operator+(const double3& x, const double3& y) {
  double3 rst = __CL_V3_OP_V3(x, +, y);
  return rst;
}
inline double3 operator+(const double3& x, double y) {
  double3 rst = __CL_V3_OP_S(x, +, y);
  return rst;
}
inline double3 operator+(double x, const double3& y) {
  return y + x;
}

/* double4 addition (+) */
inline double4 operator+(const double4& x, const double4& y) {
  double4 rst = __CL_V4_OP_V4(x, +, y);
  return rst;
}
inline double4 operator+(const double4& x, double y) {
  double4 rst = __CL_V4_OP_S(x, +, y);
  return rst;
}
inline double4 operator+(double x, const double4& y) {
  return y + x;
}

/* double8 addition (+) */
inline double8 operator+(const double8& x, const double8& y) {
  double8 rst = __CL_V8_OP_V8(x, +, y);
  return rst;
}
inline double8 operator+(const double8& x, double y) {
  double8 rst = __CL_V8_OP_S(x, +, y);
  return rst;
}
inline double8 operator+(double x, const double8& y) {
  return y + x;
}

/* double16 addition (+) */
inline double16 operator+(const double16& x, const double16& y) {
  double16 rst = __CL_V16_OP_V16(x, +, y);
  return rst;
}
inline double16 operator+(const double16& x, double y) {
  double16 rst = __CL_V16_OP_S(x, +, y);
  return rst;
}
inline double16 operator+(double x, const double16& y) {
  return y + x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. SUBTRACTION (-)
///////////////////////////////////////////////////////////

/* double2 subtraction (-) */
inline double2 operator-(const double2& x, const double2& y) {
  double2 rst = __CL_V2_OP_V2(x, -, y);
  return rst;
}
inline double2 operator-(const double2& x, double y) {
  double2 rst = __CL_V2_OP_S(x, -, y);
  return rst;
}
inline double2 operator-(double x, const double2& y) {
  double2 rst = __CL_S_OP_V2(x, -, y);
  return rst;
}

/* double3 subtraction (-) */
inline double3 operator-(const double3& x, const double3& y) {
  double3 rst = __CL_V3_OP_V3(x, -, y);
  return rst;
}
inline double3 operator-(const double3& x, double y) {
  double3 rst = __CL_V3_OP_S(x, -, y);
  return rst;
}
inline double3 operator-(double x, const double3& y) {
  double3 rst = __CL_S_OP_V3(x, -, y);
  return rst;
}

/* double4 subtraction (-) */
inline double4 operator-(const double4& x, const double4& y) {
  double4 rst = __CL_V4_OP_V4(x, -, y);
  return rst;
}
inline double4 operator-(const double4& x, double y) {
  double4 rst = __CL_V4_OP_S(x, -, y);
  return rst;
}
inline double4 operator-(double x, const double4& y) {
  double4 rst = __CL_S_OP_V4(x, -, y);
  return rst;
}

/* double8 subtraction (-) */
inline double8 operator-(const double8& x, const double8& y) {
  double8 rst = __CL_V8_OP_V8(x, -, y);
  return rst;
}
inline double8 operator-(const double8& x, double y) {
  double8 rst = __CL_V8_OP_S(x, -, y);
  return rst;
}
inline double8 operator-(double x, const double8& y) {
  double8 rst = __CL_S_OP_V8(x, -, y);
  return rst;
}

/* double16 subtraction (-) */
inline double16 operator-(const double16& x, const double16& y) {
  double16 rst = __CL_V16_OP_V16(x, -, y);
  return rst;
}
inline double16 operator-(const double16& x, double y) {
  double16 rst = __CL_V16_OP_S(x, -, y);
  return rst;
}
inline double16 operator-(double x, const double16& y) {
  double16 rst = __CL_S_OP_V16(x, -, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. MULTIPLICATION (*)
///////////////////////////////////////////////////////////

/* double2 multiplication (*) */
inline double2 operator*(const double2& x, const double2& y) {
  double2 rst = __CL_V2_OP_V2(x, *, y);
  return rst;
}
inline double2 operator*(const double2& x, double y) {
  double2 rst = __CL_V2_OP_S(x, *, y);
  return rst;
}
inline double2 operator*(double x, const double2& y) {
  return y * x;
}

/* double3 multiplication (*) */
inline double3 operator*(const double3& x, const double3& y) {
  double3 rst = __CL_V3_OP_V3(x, *, y);
  return rst;
}
inline double3 operator*(const double3& x, double y) {
  double3 rst = __CL_V3_OP_S(x, *, y);
  return rst;
}
inline double3 operator*(double x, const double3& y) {
  return y + x;
}

/* double4 multiplication (*) */
inline double4 operator*(const double4& x, const double4& y) {
  double4 rst = __CL_V4_OP_V4(x, *, y);
  return rst;
}
inline double4 operator*(const double4& x, double y) {
  double4 rst = __CL_V4_OP_S(x, *, y);
  return rst;
}
inline double4 operator*(double x, const double4& y) {
  return y + x;
}

/* double8 multiplication (*) */
inline double8 operator*(const double8& x, const double8& y) {
  double8 rst = __CL_V8_OP_V8(x, *, y);
  return rst;
}
inline double8 operator*(const double8& x, double y) {
  double8 rst = __CL_V8_OP_S(x, *, y);
  return rst;
}
inline double8 operator*(double x, const double8& y) {
  return y * x;
}

/* double16 multiplication (*) */
inline double16 operator*(const double16& x, const double16& y) {
  double16 rst = __CL_V16_OP_V16(x, *, y);
  return rst;
}
inline double16 operator*(const double16& x, double y) {
  double16 rst = __CL_V16_OP_S(x, *, y);
  return rst;
}
inline double16 operator*(double x, const double16& y) {
  return y * x;
}


///////////////////////////////////////////////////////////
/// 6.3.a. DIVISION (/)
///////////////////////////////////////////////////////////

/* double2 division (/) */
inline double2 operator/(const double2& x, const double2& y) {
  double2 rst = __CL_V2_OP_V2(x, /, y);
  return rst;
}
inline double2 operator/(const double2& x, double y) {
  double2 rst = __CL_V2_OP_S(x, /, y);
  return rst;
}
inline double2 operator/(double x, const double2& y) {
  double2 rst = __CL_S_OP_V2(x, /, y);
  return rst;
}

/* double3 division (/) */
inline double3 operator/(const double3& x, const double3& y) {
  double3 rst = __CL_V3_OP_V3(x, /, y);
  return rst;
}
inline double3 operator/(const double3& x, double y) {
  double3 rst = __CL_V3_OP_S(x, /, y);
  return rst;
}
inline double3 operator/(double x, const double3& y) {
  double3 rst = __CL_S_OP_V3(x, /, y);
  return rst;
}

/* double4 division (/) */
inline double4 operator/(const double4& x, const double4& y) {
  double4 rst = __CL_V4_OP_V4(x, /, y);
  return rst;
}
inline double4 operator/(const double4& x, double y) {
  double4 rst = __CL_V4_OP_S(x, /, y);
  return rst;
}
inline double4 operator/(double x, const double4& y) {
  double4 rst = __CL_S_OP_V4(x, /, y);
  return rst;
}

/* double8 division (/) */
inline double8 operator/(const double8& x, const double8& y) {
  double8 rst = __CL_V8_OP_V8(x, /, y);
  return rst;
}
inline double8 operator/(const double8& x, double y) {
  double8 rst = __CL_V8_OP_S(x, /, y);
  return rst;
}
inline double8 operator/(double x, const double8& y) {
  double8 rst = __CL_S_OP_V8(x, /, y);
  return rst;
}

/* double16 division (/) */
inline double16 operator/(const double16& x, const double16& y) {
  double16 rst = __CL_V16_OP_V16(x, /, y);
  return rst;
}
inline double16 operator/(const double16& x, double y) {
  double16 rst = __CL_V16_OP_S(x, /, y);
  return rst;
}
inline double16 operator/(double x, const double16& y) {
  double16 rst = __CL_S_OP_V16(x, /, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.a. REMAINDER (%)
///////////////////////////////////////////////////////////
/* Remainder (%) does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY POSITIVE (+)
///////////////////////////////////////////////////////////

/* double2 unary positive (+) */
inline double2 operator+(const double2& x) {
  return x;
}

/* double3 unary positive (+) */
inline double3 operator+(const double3& x) {
  return x;
}

/* double4 unary positive (+) */
inline double4 operator+(const double4& x) {
  return x;
}

/* double8 unary positive (+) */
inline double8 operator+(const double8& x) {
  return x;
}

/* double16 unary positive (+) */
inline double16 operator+(const double16& x) {
  return x;
}


///////////////////////////////////////////////////////////
/// 6.3.b. ARITHMETIC UNARY NEGATIVE (-)
///////////////////////////////////////////////////////////

/* double2 unary negative (-) */
inline double2 operator-(const double2& x) {
  double2 rst = __CL_OP_V2(-, x);
  return rst;
}

/* double3 unary negative (-) */
inline double3 operator-(const double3& x) {
  double3 rst = __CL_OP_V3(-, x);
  return rst;
}


/* double4 unary negative (-) */
inline double4 operator-(const double4& x) {
  double4 rst = __CL_OP_V4(-, x);
  return rst;
}

/* double8 unary negative (-) */
inline double8 operator-(const double8& x) {
  double8 rst = __CL_OP_V8(-, x);
  return rst;
}

/* double16 unary negative (-) */
inline double16 operator-(const double16& x) {
  double16 rst = __CL_OP_V16(-, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-INCREMENT (++)
///////////////////////////////////////////////////////////
/* Unary post-increment does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-INCREMENT (++)
///////////////////////////////////////////////////////////
/* Unary pre-increment does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY POST-DECREMENT (--)
///////////////////////////////////////////////////////////
/* Unary post-decrement does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.c. ARITHMETIC UNARY PRE-DECREMENT (--)
///////////////////////////////////////////////////////////
/* Unary pre-decrement does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN (>)
///////////////////////////////////////////////////////////

/* double2 relational greater than (>) */
inline long2 operator>(const double2& x, const double2& y) {
  long2 rst = __CL_V2_LOP_V2(x, >, y);
  return rst;
}
inline long2 operator>(const double2& x, double y) {
  long2 rst = __CL_V2_LOP_S(x, >, y);
  return rst;
}
inline long2 operator>(double x, const double2& y) {
  long2 rst = __CL_S_LOP_V2(x, >, y);
  return rst;
}

/* double3 relational greater than (>) */
inline long3 operator>(const double3& x, const double3& y) {
  long3 rst = __CL_V3_LOP_V3(x, >, y);
  return rst;
}
inline long3 operator>(const double3& x, double y) {
  long3 rst = __CL_V3_LOP_S(x, >, y);
  return rst;
}
inline long3 operator>(double x, const double3& y) {
  long3 rst = __CL_S_LOP_V3(x, >, y);
  return rst;
}


/* double4 relational greater than (>) */
inline long4 operator>(const double4& x, const double4& y) {
  long4 rst = __CL_V4_LOP_V4(x, >, y);
  return rst;
}
inline long4 operator>(const double4& x, double y) {
  long4 rst = __CL_V4_LOP_S(x, >, y);
  return rst;
}
inline long4 operator>(double x, const double4& y) {
  long4 rst = __CL_S_LOP_V4(x, >, y);
  return rst;
}

/* double8 relational greater than (>) */
inline long8 operator>(const double8& x, const double8& y) {
  long8 rst = __CL_V8_LOP_V8(x, >, y);
  return rst;
}
inline long8 operator>(const double8& x, double y) {
  long8 rst = __CL_V8_LOP_S(x, >, y);
  return rst;
}
inline long8 operator>(double x, const double8& y) {
  long8 rst = __CL_S_LOP_V8(x, >, y);
  return rst;
}

/* double16 relational greater than (>) */
inline long16 operator>(const double16& x, const double16& y) {
  long16 rst = __CL_V16_LOP_V16(x, >, y);
  return rst;
}
inline long16 operator>(const double16& x, double y) {
  long16 rst = __CL_V16_LOP_S(x, >, y);
  return rst;
}
inline long16 operator>(double x, const double16& y) {
  long16 rst = __CL_S_LOP_V16(x, >, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN (<)
///////////////////////////////////////////////////////////

/* double2 relational less than (<) */
inline long2 operator<(const double2& x, const double2& y) {
  return y > x;
}
inline long2 operator<(const double2& x, double y) {
  return y > x;
}
inline long2 operator<(double x, const double2& y) {
  return y > x;
}

/* double3 relational less than (<) */
inline long3 operator<(const double3& x, const double3& y) {
  return y > x;
}
inline long3 operator<(const double3& x, double y) {
  return y > x;
}
inline long3 operator<(double x, const double3& y) {
  return y > x;
}

/* double4 relational less than (<) */
inline long4 operator<(const double4& x, const double4& y) {
  return y > x;
}
inline long4 operator<(const double4& x, double y) {
  return y > x;
}
inline long4 operator<(double x, const double4& y) {
  return y > x;
}

/* double8 relational less than (<) */
inline long8 operator<(const double8& x, const double8& y) {
  return y > x;
}
inline long8 operator<(const double8& x, double y) {
  return y > x;
}
inline long8 operator<(double x, const double8& y) {
  return y > x;
}

/* double16 relational less than (<) */
inline long16 operator<(const double16& x, const double16& y) {
  return y > x;
}
inline long16 operator<(const double16& x, double y) {
  return y > x;
}
inline long16 operator<(double x, const double16& y) {
  return y > x;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - GREATER THAN OR EQUAL (>=)
///////////////////////////////////////////////////////////

/* double2 relational greater than or equal (>=) */
inline long2 operator>=(const double2& x, const double2& y) {
  long2 rst = __CL_V2_LOP_V2(x, >=, y);
  return rst;
}
inline long2 operator>=(const double2& x, double y) {
  long2 rst = __CL_V2_LOP_S(x, >=, y);
  return rst;
}
inline long2 operator>=(double x, const double2& y) {
  long2 rst = __CL_S_LOP_V2(x, >=, y);
  return rst;
}

/* double3 relational greater than or equal (>=) */
inline long3 operator>=(const double3& x, const double3& y) {
  long3 rst = __CL_V3_LOP_V3(x, >=, y);
  return rst;
}
inline long3 operator>=(const double3& x, double y) {
  long3 rst = __CL_V3_LOP_S(x, >=, y);
  return rst;
}
inline long3 operator>=(double x, const double3& y) {
  long3 rst = __CL_S_LOP_V3(x, >=, y);
  return rst;
}

/* double4 relational greater than or equal (>=) */
inline long4 operator>=(const double4& x, const double4& y) {
  long4 rst = __CL_V4_LOP_V4(x, >=, y);
  return rst;
}
inline long4 operator>=(const double4& x, double y) {
  long4 rst = __CL_V4_LOP_S(x, >=, y);
  return rst;
}
inline long4 operator>=(double x, const double4& y) {
  long4 rst = __CL_S_LOP_V4(x, >=, y);
  return rst;
}

/* double8 relational greater than or equal (>=) */
inline long8 operator>=(const double8& x, const double8& y) {
  long8 rst = __CL_V8_LOP_V8(x, >=, y);
  return rst;
}
inline long8 operator>=(const double8& x, double y) {
  long8 rst = __CL_V8_LOP_S(x, >=, y);
  return rst;
}
inline long8 operator>=(double x, const double8& y) {
  long8 rst = __CL_S_LOP_V8(x, >=, y);
  return rst;
}

/* double16 relational greater than or equal (>=) */
inline long16 operator>=(const double16& x, const double16& y) {
  long16 rst = __CL_V16_LOP_V16(x, >=, y);
  return rst;
}
inline long16 operator>=(const double16& x, double y) {
  long16 rst = __CL_V16_LOP_S(x, >=, y);
  return rst;
}
inline long16 operator>=(double x, const double16& y) {
  long16 rst = __CL_S_LOP_V16(x, >=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.d. RELATIONAL - LESS THAN OR EQUAL (<=)
///////////////////////////////////////////////////////////

/* double2 relational less than or equal (<=) */
inline long2 operator<=(const double2& x, const double2& y) {
  return y >= x;
}
inline long2 operator<=(const double2& x, double y) {
  return y >= x;
}
inline long2 operator<=(double x, const double2& y) {
  return y >= x;
}

/* double3 relational less than or equal (<=) */
inline long3 operator<=(const double3& x, const double3& y) {
  return y >= x;
}
inline long3 operator<=(const double3& x, double y) {
  return y >= x;
}
inline long3 operator<=(double x, const double3& y) {
  return y >= x;
}

/* double4 relational less than or equal (<=) */
inline long4 operator<=(const double4& x, const double4& y) {
  return y >= x;
}
inline long4 operator<=(const double4& x, double y) {
  return y >= x;
}
inline long4 operator<=(double x, const double4& y) {
  return y >= x;
}

/* double8 relational less than or equal (<=) */
inline long8 operator<=(const double8& x, const double8& y) {
  return y >= x;
}
inline long8 operator<=(const double8& x, double y) {
  return y >= x;
}
inline long8 operator<=(double x, const double8& y) {
  return y >= x;
}

/* double16 relational less than or equal (<=) */
inline long16 operator<=(const double16& x, const double16& y) {
  return y >= x;
}
inline long16 operator<=(const double16& x, double y) {
  return y >= x;
}
inline long16 operator<=(double x, const double16& y) {
  return y >= x;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - EQUAL (==)
///////////////////////////////////////////////////////////

/* double2 equal (==) */
inline long2 operator==(const double2& x, const double2& y) {
  long2 rst = __CL_V2_LOP_V2(x, ==, y);
  return rst;
}
inline long2 operator==(const double2& x, double y) {
  long2 rst = __CL_V2_LOP_S(x, ==, y);
  return rst;
}
inline long2 operator==(double x, const double2& y) {
  long2 rst = __CL_S_LOP_V2(x, ==, y);
  return rst;
}

/* double3 equal (==) */
inline long3 operator==(const double3& x, const double3& y) {
  long3 rst = __CL_V3_LOP_V3(x, ==, y);
  return rst;
}
inline long3 operator==(const double3& x, double y) {
  long3 rst = __CL_V3_LOP_S(x, ==, y);
  return rst;
}
inline long3 operator==(double x, const double3& y) {
  long3 rst = __CL_S_LOP_V3(x, ==, y);
  return rst;
}

/* double4 equal (==) */
inline long4 operator==(const double4& x, const double4& y) {
  long4 rst = __CL_V4_LOP_V4(x, ==, y);
  return rst;
}
inline long4 operator==(const double4& x, double y) {
  long4 rst = __CL_V4_LOP_S(x, ==, y);
  return rst;
}
inline long4 operator==(double x, const double4& y) {
  long4 rst = __CL_S_LOP_V4(x, ==, y);
  return rst;
}

/* double8 equal (==) */
inline long8 operator==(const double8& x, const double8& y) {
  long8 rst = __CL_V8_LOP_V8(x, ==, y);
  return rst;
}
inline long8 operator==(const double8& x, double y) {
  long8 rst = __CL_V8_LOP_S(x, ==, y);
  return rst;
}
inline long8 operator==(double x, const double8& y) {
  long8 rst = __CL_S_LOP_V8(x, ==, y);
  return rst;
}

/* double16 equal (==) */
inline long16 operator==(const double16& x, const double16& y) {
  long16 rst = __CL_V16_LOP_V16(x, ==, y);
  return rst;
}
inline long16 operator==(const double16& x, double y) {
  long16 rst = __CL_V16_LOP_S(x, ==, y);
  return rst;
}
inline long16 operator==(double x, const double16& y) {
  long16 rst = __CL_S_LOP_V16(x, ==, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.e. EQUALITY - NOT EQUAL (!=)
///////////////////////////////////////////////////////////

/* double2 not equal (!=) */
inline long2 operator!=(const double2& x, const double2& y) {
  long2 rst = __CL_V2_LOP_V2(x, !=, y);
  return rst;
}
inline long2 operator!=(const double2& x, double y) {
  long2 rst = __CL_V2_LOP_S(x, !=, y);
  return rst;
}
inline long2 operator!=(double x, const double2& y) {
  long2 rst = __CL_S_LOP_V2(x, !=, y);
  return rst;
}

/* double3 not equal (!=) */
inline long3 operator!=(const double3& x, const double3& y) {
  long3 rst = __CL_V3_LOP_V3(x, !=, y);
  return rst;
}
inline long3 operator!=(const double3& x, double y) {
  long3 rst = __CL_V3_LOP_S(x, !=, y);
  return rst;
}
inline long3 operator!=(double x, const double3& y) {
  long3 rst = __CL_S_LOP_V3(x, !=, y);
  return rst;
}

/* double4 not equal (!=) */
inline long4 operator!=(const double4& x, const double4& y) {
  long4 rst = __CL_V4_LOP_V4(x, !=, y);
  return rst;
}
inline long4 operator!=(const double4& x, double y) {
  long4 rst = __CL_V4_LOP_S(x, !=, y);
  return rst;
}
inline long4 operator!=(double x, const double4& y) {
  long4 rst = __CL_S_LOP_V4(x, !=, y);
  return rst;
}

/* double8 not equal (!=) */
inline long8 operator!=(const double8& x, const double8& y) {
  long8 rst = __CL_V8_LOP_V8(x, !=, y);
  return rst;
}
inline long8 operator!=(const double8& x, double y) {
  long8 rst = __CL_V8_LOP_S(x, !=, y);
  return rst;
}
inline long8 operator!=(double x, const double8& y) {
  long8 rst = __CL_S_LOP_V8(x, !=, y);
  return rst;
}

/* double16 not equal (!=) */
inline long16 operator!=(const double16& x, const double16& y) {
  long16 rst = __CL_V16_LOP_V16(x, !=, y);
  return rst;
}
inline long16 operator!=(const double16& x, double y) {
  long16 rst = __CL_V16_LOP_S(x, !=, y);
  return rst;
}
inline long16 operator!=(double x, const double16& y) {
  long16 rst = __CL_S_LOP_V16(x, !=, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - AND (&)
///////////////////////////////////////////////////////////
/* Bitwise and does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - OR (|)
///////////////////////////////////////////////////////////
/* Bitwise or does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - EXCLUSIVE OR (^)
///////////////////////////////////////////////////////////
/* Bitwise exclusive or does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.f. BITWISE - NOT (~)
///////////////////////////////////////////////////////////
/* Bitwise not does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - AND (&&)
///////////////////////////////////////////////////////////

/* double2 logical and (&&) */
inline long2 operator&&(const double2& x, const double2& y) {
  long2 rst = __CL_V2_LOP_V2(x, &&, y);
  return rst;
}
inline long2 operator&&(const double2& x, double y) {
  long2 rst = __CL_V2_LOP_S(x, &&, y);
  return rst;
}
inline long2 operator&&(double x, const double2& y) {
  long2 rst = __CL_S_LOP_V2(x, &&, y);
  return rst;
}

/* double3 logical and (&&) */
inline long3 operator&&(const double3& x, const double3& y) {
  long3 rst = __CL_V3_LOP_V3(x, &&, y);
  return rst;
}
inline long3 operator&&(const double3& x, double y) {
  long3 rst = __CL_V3_LOP_S(x, &&, y);
  return rst;
}
inline long3 operator&&(double x, const double3& y) {
  long3 rst = __CL_S_LOP_V3(x, &&, y);
  return rst;
}

/* double4 logical and (&&) */
inline long4 operator&&(const double4& x, const double4& y) {
  long4 rst = __CL_V4_LOP_V4(x, &&, y);
  return rst;
}
inline long4 operator&&(const double4& x, double y) {
  long4 rst = __CL_V4_LOP_S(x, &&, y);
  return rst;
}
inline long4 operator&&(double x, const double4& y) {
  long4 rst = __CL_S_LOP_V4(x, &&, y);
  return rst;
}

/* double8 logical and (&&) */
inline long8 operator&&(const double8& x, const double8& y) {
  long8 rst = __CL_V8_LOP_V8(x, &&, y);
  return rst;
}
inline long8 operator&&(const double8& x, double y) {
  long8 rst = __CL_V8_LOP_S(x, &&, y);
  return rst;
}
inline long8 operator&&(double x, const double8& y) {
  long8 rst = __CL_S_LOP_V8(x, &&, y);
  return rst;
}

/* double16 logical and (&&) */
inline long16 operator&&(const double16& x, const double16& y) {
  long16 rst = __CL_V16_LOP_V16(x, &&, y);
  return rst;
}
inline long16 operator&&(const double16& x, double y) {
  long16 rst = __CL_V16_LOP_S(x, &&, y);
  return rst;
}
inline long16 operator&&(double x, const double16& y) {
  long16 rst = __CL_S_LOP_V16(x, &&, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.g. LOGICAL - OR (||)
///////////////////////////////////////////////////////////
/* Logical or does not operate on doublen types. */

/* double2 logical or (||) */
inline long2 operator||(const double2& x, const double2& y) {
  long2 rst = __CL_V2_LOP_V2(x, ||, y);
  return rst;
}
inline long2 operator||(const double2& x, double y) {
  long2 rst = __CL_V2_LOP_S(x, ||, y);
  return rst;
}
inline long2 operator||(double x, const double2& y) {
  long2 rst = __CL_S_LOP_V2(x, ||, y);
  return rst;
}

/* double3 logical or (||) */
inline long3 operator||(const double3& x, const double3& y) {
  long3 rst = __CL_V3_LOP_V3(x, ||, y);
  return rst;
}
inline long3 operator||(const double3& x, double y) {
  long3 rst = __CL_V3_LOP_S(x, ||, y);
  return rst;
}
inline long3 operator||(double x, const double3& y) {
  long3 rst = __CL_S_LOP_V3(x, ||, y);
  return rst;
}

/* double4 logical or (||) */
inline long4 operator||(const double4& x, const double4& y) {
  long4 rst = __CL_V4_LOP_V4(x, ||, y);
  return rst;
}
inline long4 operator||(const double4& x, double y) {
  long4 rst = __CL_V4_LOP_S(x, ||, y);
  return rst;
}
inline long4 operator||(double x, const double4& y) {
  long4 rst = __CL_S_LOP_V4(x, ||, y);
  return rst;
}

/* double8 logical or (||) */
inline long8 operator||(const double8& x, const double8& y) {
  long8 rst = __CL_V8_LOP_V8(x, ||, y);
  return rst;
}
inline long8 operator||(const double8& x, double y) {
  long8 rst = __CL_V8_LOP_S(x, ||, y);
  return rst;
}
inline long8 operator||(double x, const double8& y) {
  long8 rst = __CL_S_LOP_V8(x, ||, y);
  return rst;
}

/* double16 logical or (||) */
inline long16 operator||(const double16& x, const double16& y) {
  long16 rst = __CL_V16_LOP_V16(x, ||, y);
  return rst;
}
inline long16 operator||(const double16& x, double y) {
  long16 rst = __CL_V16_LOP_S(x, ||, y);
  return rst;
}
inline long16 operator||(double x, const double16& y) {
  long16 rst = __CL_S_LOP_V16(x, ||, y);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.h. LOGICAL - NOT (!)
///////////////////////////////////////////////////////////

/* double2 logical not (!) */
inline long2 operator!(const double2& x) {
  long2 rst = __CL_LOP_V2(!, x);
  return rst;
}

/* double3 logical not (!) */
inline long3 operator!(const double3& x) {
  long3 rst = __CL_LOP_V3(!, x);
  return rst;
}
/* double4 logical not (!) */
inline long4 operator!(const double4& x) {
  long4 rst = __CL_LOP_V4(!, x);
  return rst;
}

/* double8 logical not (!) */
inline long8 operator!(const double8& x) {
  long8 rst = __CL_LOP_V8(!, x);
  return rst;
}

/* double16 logical not (!) */
inline long16 operator!(const double16& x) {
  long16 rst = __CL_LOP_V16(!, x);
  return rst;
}


///////////////////////////////////////////////////////////
/// 6.3.j. RIGHT-SHIFT (>>)
///////////////////////////////////////////////////////////
/* Right-shift does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.j. LEFT-SHIFT (<<)
///////////////////////////////////////////////////////////
/* Left-shift does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT ADD INTO (+=)
///////////////////////////////////////////////////////////

/* double2 assignment add into (+=) */
inline double2 operator+=(double2 &x, const double2& y) {
  __CL_V_OP_ASSIGN_V(double2, x, +, y);
}
inline double2 operator+=(double2 &x, double y) {
  __CL_V_OP_ASSIGN_V(double2, x, +, y);
}

/* double3 assignment add into (+=) */
inline double3 operator+=(double3 &x, const double3& y) {
  __CL_V_OP_ASSIGN_V(double3, x, +, y);
}
inline double3 operator+=(double3 &x, double y) {
  __CL_V_OP_ASSIGN_V(double3, x, +, y);
}

/* double4 assignment add into (+=) */
inline double4 operator+=(double4 &x, const double4& y) {
  __CL_V_OP_ASSIGN_V(double4, x, +, y);
}
inline double4 operator+=(double4 &x, double y) {
  __CL_V_OP_ASSIGN_V(double4, x, +, y);
}

/* double8 assignment add into (+=) */
inline double8 operator+=(double8 &x, const double8& y) {
  __CL_V_OP_ASSIGN_V(double8, x, +, y);
}
inline double8 operator+=(double8 &x, double y) {
  __CL_V_OP_ASSIGN_V(double8, x, +, y);
}

/* double16 assignment add into (+=) */
inline double16 operator+=(double16 &x, const double16& y) {
  __CL_V_OP_ASSIGN_V(double16, x, +, y);
}
inline double16 operator+=(double16 &x, double y) {
  __CL_V_OP_ASSIGN_V(double16, x, +, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT SUBTRACT FROM (-=)
///////////////////////////////////////////////////////////

/* double2 assignment subtract from (-=) */
inline double2 operator-=(double2 &x, const double2& y) {
  __CL_V_OP_ASSIGN_V(double2, x, -, y);
}
inline double2 operator-=(double2 &x, double y) {
  __CL_V_OP_ASSIGN_V(double2, x, -, y);
}
/* double3 assignment subtract from (-=) */
inline double3 operator-=(double3 &x, const double3& y) {
  __CL_V_OP_ASSIGN_V(double3, x, -, y);
}
inline double3 operator-=(double3 &x, double y) {
  __CL_V_OP_ASSIGN_V(double3, x, -, y);
}

/* double4 assignment subtract from (-=) */
inline double4 operator-=(double4 &x, const double4& y) {
  __CL_V_OP_ASSIGN_V(double4, x, -, y);
}
inline double4 operator-=(double4 &x, double y) {
  __CL_V_OP_ASSIGN_V(double4, x, -, y);
}

/* double8 assignment subtract from (-=) */
inline double8 operator-=(double8 &x, const double8& y) {
  __CL_V_OP_ASSIGN_V(double8, x, -, y);
}
inline double8 operator-=(double8 &x, double y) {
  __CL_V_OP_ASSIGN_V(double8, x, -, y);
}

/* double16 assignment subtract from (-=) */
inline double16 operator-=(double16 &x, const double16& y) {
  __CL_V_OP_ASSIGN_V(double16, x, -, y);
}
inline double16 operator-=(double16 &x, double y) {
  __CL_V_OP_ASSIGN_V(double16, x, -, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MULTIPLY INTO (*=)
///////////////////////////////////////////////////////////

/* double2 assignment multiply into (*=) */
inline double2 operator*=(double2 &x, const double2& y) {
  __CL_V_OP_ASSIGN_V(double2, x, *, y);
}
inline double2 operator*=(double2 &x, double y) {
  __CL_V_OP_ASSIGN_V(double2, x, *, y);
}

/* double3 assignment multiply into (*=) */
inline double3 operator*=(double3 &x, const double3& y) {
  __CL_V_OP_ASSIGN_V(double3, x, *, y);
}
inline double3 operator*=(double3 &x, double y) {
  __CL_V_OP_ASSIGN_V(double3, x, *, y);
}
/* double4 assignment multiply into (*=) */
inline double4 operator*=(double4 &x, const double4& y) {
  __CL_V_OP_ASSIGN_V(double4, x, *, y);
}
inline double4 operator*=(double4 &x, double y) {
  __CL_V_OP_ASSIGN_V(double4, x, *, y);
}

/* double8 assignment multiply into (*=) */
inline double8 operator*=(double8 &x, const double8& y) {
  __CL_V_OP_ASSIGN_V(double8, x, *, y);
}
inline double8 operator*=(double8 &x, double y) {
  __CL_V_OP_ASSIGN_V(double8, x, *, y);
}

/* double16 assignment multiply into (*=) */
inline double16 operator*=(double16 &x, const double16& y) {
  __CL_V_OP_ASSIGN_V(double16, x, *, y);
}
inline double16 operator*=(double16 &x, double y) {
  __CL_V_OP_ASSIGN_V(double16, x, *, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT DIVIDE INTO (/=)
///////////////////////////////////////////////////////////

/* double2 assignment divide into (/=) */
inline double2 operator/=(double2 &x, const double2& y) {
  __CL_V_OP_ASSIGN_V(double2, x, /, y);
}
inline double2 operator/=(double2 &x, double y) {
  __CL_V_OP_ASSIGN_V(double2, x, /, y);
}

/* double3 assignment divide into (/=) */
inline double3 operator/=(double3 &x, const double3& y) {
  __CL_V_OP_ASSIGN_V(double3, x, /, y);
}
inline double3 operator/=(double3 &x, double y) {
  __CL_V_OP_ASSIGN_V(double3, x, /, y);
}

/* double4 assignment divide into (/=) */
inline double4 operator/=(double4 &x, const double4& y) {
  __CL_V_OP_ASSIGN_V(double4, x, /, y);
}
inline double4 operator/=(double4 &x, double y) {
  __CL_V_OP_ASSIGN_V(double4, x, /, y);
}

/* double8 assignment divide into (/=) */
inline double8 operator/=(double8 &x, const double8& y) {
  __CL_V_OP_ASSIGN_V(double8, x, /, y);
}
inline double8 operator/=(double8 &x, double y) {
  __CL_V_OP_ASSIGN_V(double8, x, /, y);
}

/* double16 assignment divide into (/=) */
inline double16 operator/=(double16 &x, const double16& y) {
  __CL_V_OP_ASSIGN_V(double16, x, /, y);
}
inline double16 operator/=(double16 &x, double y) {
  __CL_V_OP_ASSIGN_V(double16, x, /, y);
}


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT MODULUS INTO (%=)
///////////////////////////////////////////////////////////
/* Does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT LEFT SHIFT BY (<<=)
///////////////////////////////////////////////////////////
/* Does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT RIGHT SHIFT BY (>>=)
///////////////////////////////////////////////////////////
/* Does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT AND INTO (&=)
///////////////////////////////////////////////////////////
/* Does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT INCLUSIVE OR INTO (|=)
///////////////////////////////////////////////////////////
/* Does not operate on doublen types. */


///////////////////////////////////////////////////////////
/// 6.3.o. ASSIGNMENT EXCLUSIVE OR INTO (^=)
///////////////////////////////////////////////////////////
/* Does not operate on doublen types. */


#endif //__CL_OPS_DOUBLEN_H


