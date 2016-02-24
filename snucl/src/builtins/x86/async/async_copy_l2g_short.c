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


event_t async_work_group_copy_l2g(short *dst, const short *src, size_t num_elements, event_t event) {
  for (int i = 0; i < num_elements; i++)
     dst[i] = src[i];
  return event;
}
event_t async_work_group_copy_l2g(short2 *dst, const short2 *src, size_t num_elements, event_t event) {
  for (int i = 0; i < num_elements; i++)
      dst[i] = src[i];
  return event;
}
event_t async_work_group_copy_l2g(short3 *dst, const short3 *src, size_t num_elements, event_t event) {
  for (int i = 0; i < num_elements; i++)
     dst[i] = src[i];
  return event;
}
event_t async_work_group_copy_l2g(short4 *dst, const short4 *src, size_t num_elements, event_t event) {
  for (int i = 0; i < num_elements; i++)
     dst[i] = src[i];
  return event;
}
event_t async_work_group_copy_l2g(short8 *dst, const short8 *src, size_t num_elements, event_t event) {
  for (int i = 0; i < num_elements; i++)
      dst[i] = src[i];
  return event;
}
event_t async_work_group_copy_l2g(short16 *dst, const short16 *src, size_t num_elements, event_t event) {
  for (int i = 0; i < num_elements; i++)
      dst[i] = src[i];
  return event;
}

