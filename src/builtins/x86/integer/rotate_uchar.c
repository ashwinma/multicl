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

// uchar
uchar rotate(uchar v, uchar i){
  i = i & 7;
  return (i == 0) ? v : (v << i) | (v >> (8 - i));
}
uchar2 rotate(uchar2 v, uchar2 i){
	uchar2 rst;
  i = i & 7;
  for (int k = 0; k < 2; k++)
    rst[k] = (i[k] == 0) ? v[k] : (v[k] << i[k]) | (v[k] >> (8 - i[k]));
  return rst;
}
uchar3 rotate(uchar3 v, uchar3 i){
	uchar3 rst;
  i = i & 7;
  for (int k = 0; k < 3; k++)
    rst[k] = (i[k] == 0) ? v[k] : (v[k] << i[k]) | (v[k] >> (8 - i[k]));
  return rst;
}
uchar4 rotate(uchar4 v, uchar4 i){
	uchar4 rst;
  i = i & 7;
  for (int k = 0; k < 4; k++)
    rst[k] = (i[k] == 0) ? v[k] : (v[k] << i[k]) | (v[k] >> (8 - i[k]));
  return rst;
}
uchar8 rotate(uchar8 v, uchar8 i){
	uchar8 rst;
  i = i & 7;
  for (int k = 0; k < 8; k++)
    rst[k] = (i[k] == 0) ? v[k] : (v[k] << i[k]) | (v[k] >> (8 - i[k]));
  return rst;
}
uchar16 rotate(uchar16 v, uchar16 i){
	uchar16 rst;
  i = i & 7;
  for (int k = 0; k < 16; k++)
    rst[k] = (i[k] == 0) ? v[k] : (v[k] << i[k]) | (v[k] >> (8 - i[k]));
  return rst;
}

