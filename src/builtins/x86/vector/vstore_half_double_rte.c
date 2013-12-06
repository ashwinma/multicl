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
#include "half_util.h"

void vstore_half_rte(double data, size_t offset, half *p) {
  half *p_off = p + offset;
  *p_off = double2half_rte(data);
}

void vstore_half2_rte(double2 data, size_t offset, half *p) {
  half *p_off = p + (offset * 2);
  for (int i = 0; i < 2; i++)
    *(p_off + i) = double2half_rte(data[i]);
}

void vstore_half3_rte(double3 data, size_t offset, half *p) {
  half *p_off = p + (offset * 3);
  for (int i = 0; i < 3; i++)
    *(p_off + i) = double2half_rte(data[i]);
}

void vstore_half4_rte(double4 data, size_t offset, half *p) {
  half *p_off = p + (offset * 4);
  for (int i = 0; i < 4; i++)
    *(p_off + i) = double2half_rte(data[i]);
}

void vstore_half8_rte(double8 data, size_t offset, half *p) {
  half *p_off = p + (offset * 8);
  for (int i = 0; i < 8; i++)
    *(p_off + i) = double2half_rte(data[i]);
}

void vstore_half16_rte(double16 data, size_t offset, half *p) {
  half *p_off = p + (offset * 16);
  for (int i = 0; i < 16; i++)
    *(p_off + i) = double2half_rte(data[i]);
}

