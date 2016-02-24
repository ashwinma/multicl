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

#include <cl_cpu_ops.h>

// llong
llong rotate(llong v, llong i){
  i = i & 63;
  return (i == 0) ? v : ((ullong)v << i) | ((ullong)v >> (64 - i));
}
long2 rotate(long2 v, long2 i){
	long2 rst;
  i = i & 63;
  for (int k = 0; k < 2; k++)
    rst[k] = (i[k] == 0) ? v[k] : ((ullong)v[k] << i[k]) | ((ullong)v[k] >> (64 - i[k]));
  return rst;
}
long3 rotate(long3 v, long3 i){
	long3 rst;
  i = i & 63;
  for (int k = 0; k < 3; k++)
    rst[k] = (i[k] == 0) ? v[k] : ((ullong)v[k] << i[k]) | ((ullong)v[k] >> (64 - i[k]));
  return rst;
}
long4 rotate(long4 v, long4 i){
	long4 rst;
  i = i & 63;
  for (int k = 0; k < 4; k++)
    rst[k] = (i[k] == 0) ? v[k] : ((ullong)v[k] << i[k]) | ((ullong)v[k] >> (64 - i[k]));
  return rst;
}
long8 rotate(long8 v, long8 i){
	long8 rst;
  i = i & 63;
  for (int k = 0; k < 8; k++)
    rst[k] = (i[k] == 0) ? v[k] : ((ullong)v[k] << i[k]) | ((ullong)v[k] >> (64 - i[k]));
  return rst;
}
long16 rotate(long16 v, long16 i){
	long16 rst;
  i = i & 63;
  for (int k = 0; k < 16; k++)
    rst[k] = (i[k] == 0) ? v[k] : ((ullong)v[k] << i[k]) | ((ullong)v[k] >> (64 - i[k]));
  return rst;
}

