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


#ifndef INTERNAL_PRINTF_H_
#define INTERNAL_PRINTF_H_

#include <cl_cpu_types.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <string.h>
#include <vector>

enum _VTYPE {
  VCHAR = 1, VINT, VSHORT, VLONG, VFLOAT, VDOUBLE
};

void init_speci();
int find_end_of_format(const char * format, int start);
char * get_sub_string(const char** string, int start, int end);
char * get_sub_string(const char** string, int start);
bool has_string(std::vector<std::string>* vector, char * string);
int get_vector_type(char * format);
int getDim(char* subFormat, int start);
void delete_string(char* string, int start, int n);
bool is_vector(char* subFormat, int *type, int *dim);
bool is_double(char spec);
bool is_int(char spec);

//vint
int print_vector(char* subFormat, int16 val);
int print_vector(char* subFormat, int8 val);
int print_vector(char* subFormat, int4 val);
int print_vector(char* subFormat, int3 val);
int print_vector(char* subFormat, int2 val);
// vchar
int print_vector(char* subFormat, char16 val);
int print_vector(char* subFormat, char8 val);
int print_vector(char* subFormat, char4 val);
int print_vector(char* subFormat, char3 val);
int print_vector(char* subFormat, char2 val);
// vlong
int print_vector(char* subFormat, long16 val);
int print_vector(char* subFormat, long8 val);
int print_vector(char* subFormat, long4 val);
int print_vector(char* subFormat, long3 val);
int print_vector(char* subFormat, long2 val);
// vshort
int print_vector(char* subFormat, short16 val);
int print_vector(char* subFormat, short8 val);
int print_vector(char* subFormat, short4 val);
int print_vector(char* subFormat, short3 val);
int print_vector(char* subFormat, short2 val);
// vfloat
int print_vector(char* subFormat, float16 val);
int print_vector(char* subFormat, float8 val);
int print_vector(char* subFormat, float4 val);
int print_vector(char* subFormat, float3 val);
int print_vector(char* subFormat, float2 val);
// vdouble
int print_vector(char* subFormat, double16 val);
int print_vector(char* subFormat, double8 val);
int print_vector(char* subFormat, double4 val);
int print_vector(char* subFormat, double3 val);
int print_vector(char* subFormat, double2 val);

class Print {
  char* format;
  std::vector<void *> arglist;

  mutable int inx;

  public:
  Print(char* _format);

  template <typename T>
    Print& operator,(T arg) {
      T* _arg = (T *)malloc(sizeof(T));
      *_arg = arg;
      arglist.push_back(_arg);
      return *this;
    }

  Print& operator,(float arg);
  Print& operator,(int arg);
  Print& operator,(short arg);
  Print& operator,(char arg);
  
  void* get_arg() const;
  int print() const;

  ~Print();

};

int snu_printf(const Print& p);

#endif
