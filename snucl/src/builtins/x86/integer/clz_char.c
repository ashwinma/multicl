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
#include <math.h>
#include <stdio.h>


schar clz(schar y){
  schar rst=0;
  short  x=y;

  if (x == 0x0)
  {
    rst = 8;
  }
  else
  {
    if (x < 0)
    {
      x = x&0xFF;
    }
    for (rst = 0;  x < 0x80  ; rst++ )
    {
      x = (x<<1);
    }
  }

  return rst;
}

char2 clz(char2 x){
  char2 rst;
  rst[0]        = clz(x[0]);
  rst[1]        = clz(x[1]);
  return rst;
}
char3 clz(char3 x){
  char3 rst;
  rst[0]        = clz(x[0]);
  rst[1]        = clz(x[1]);
  rst[2]        = clz(x[2]);
  return rst;
}
char4 clz(char4 x){
  char4 rst;
  rst[0]        = clz(x[0]);
  rst[1]        = clz(x[1]);
  rst[2]        = clz(x[2]);
  rst[3]        = clz(x[3]);
  return rst;
}
char8 clz(char8 x){

  char8 rst;
  rst[0]        = clz(x[0]);
  rst[1]        = clz(x[1]);
  rst[2]        = clz(x[2]);
  rst[3]        = clz(x[3]);
  rst[4]        = clz(x[4]);
  rst[5]        = clz(x[5]);
  rst[6]        = clz(x[6]);
  rst[7]        = clz(x[7]);

  return rst;
}
char16 clz(char16 x){
  char16 rst;
  rst[0]        = clz(x[0]);
  rst[1]        = clz(x[1]);
  rst[2]        = clz(x[2]);
  rst[3]        = clz(x[3]);
  rst[4]        = clz(x[4]);
  rst[5]        = clz(x[5]);
  rst[6]        = clz(x[6]);
  rst[7]        = clz(x[7]);
  rst[8]        = clz(x[8]);
  rst[9]        = clz(x[9]);
  rst[10]        = clz(x[10]);
  rst[11]        = clz(x[11]);
  rst[12]        = clz(x[12]);
  rst[13]        = clz(x[13]);
  rst[14]        = clz(x[14]);
  rst[15]        = clz(x[15]);
  return rst;
}

