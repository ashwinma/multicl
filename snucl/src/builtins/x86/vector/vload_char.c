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

char2 vload2(size_t offset, const schar *p) {
  char2 rst;
  schar *p_offset = (schar *)p + (offset * 2);
  rst[0] = *(p_offset);
  rst[1] = *(p_offset + 1);
  return rst;
}

char3 vload3(size_t offset, const schar * p) {
  char3 rst;
  schar *p_offset = (schar *)p + (offset * 3);
  for (int i = 0; i < 3; i++)
    rst[i] = *(p_offset + i);
  return rst;
}

char4 vload4(size_t offset, const schar * p) {
  char4 rst;
  schar *p_offset = (schar *)p + (offset * 4);
  for (int i = 0; i < 4; i++)
    rst[i] = *(p_offset + i);
  return rst;
}

char8 vload8(size_t offset, const schar * p) {
  char8 rst;
  schar *p_offset = (schar *)p + (offset * 8);
  for (int i = 0; i < 8; i++)
    rst[i] = *(p_offset + i);
  return rst;
}

char16 vload16(size_t offset, const schar * p) {
  char16 rst;
  schar *p_offset = (schar *)p + (offset * 16);
  for (int i = 0; i < 16; i++)
    rst[i] = *(p_offset + i);
  return rst;
}

