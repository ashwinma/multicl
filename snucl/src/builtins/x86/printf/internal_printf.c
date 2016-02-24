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



#include "internal_printf.h"

#include <stdio.h>
#include <stdlib.h>
#include <relational/cl_builtins_relational.h>

using namespace std;

int N_DIM = 5;
const char *DIM_FORMAT[] = {"2", "3", "4", "8", "16"};

string CONV_SPECI = "diouxXfFeEgGaAcsp%";
string INT_SPECI = "diouxXc";
string DOUB_SPECI = "fFeEgGaA";


void init_speci(vector<string>& VCHAR_SPECI, vector<string>& VINT_SPECI, std::vector<string>& VSHORT_SPECI, std::vector<string> &VLONG_SPECI, std::vector<string> &VFLOAT_SPECI, std::vector<string> &VDOUBLE_SPECI) { 
  VCHAR_SPECI.push_back("hhd"); VCHAR_SPECI.push_back("hhi"); VCHAR_SPECI.push_back("hho");
  VCHAR_SPECI.push_back("hhu"); VCHAR_SPECI.push_back("hhx"); VCHAR_SPECI.push_back("hhX");
  
  VSHORT_SPECI.push_back("hd"); VSHORT_SPECI.push_back("hi"); VSHORT_SPECI.push_back("ho"); 
  VSHORT_SPECI.push_back("hu"); VSHORT_SPECI.push_back("hx"); VSHORT_SPECI.push_back("hX"); 
  VSHORT_SPECI.push_back("hf"); VSHORT_SPECI.push_back("hF"); VSHORT_SPECI.push_back("he"); 
  VSHORT_SPECI.push_back("hE"); VSHORT_SPECI.push_back("hg"); VSHORT_SPECI.push_back("hG"); 
  VSHORT_SPECI.push_back("ha"); VSHORT_SPECI.push_back("hA");

  VINT_SPECI.push_back("hld"); VINT_SPECI.push_back("hli"); VINT_SPECI.push_back("hlo");
  VINT_SPECI.push_back("hlu"); VINT_SPECI.push_back("hlx"); VINT_SPECI.push_back("hlX");

  VLONG_SPECI.push_back("ld"); VLONG_SPECI.push_back("li"); VLONG_SPECI.push_back("lo");
  VLONG_SPECI.push_back("lu"); VLONG_SPECI.push_back("lx"); VLONG_SPECI.push_back("lX");

  VFLOAT_SPECI.push_back("hlf"); VFLOAT_SPECI.push_back("hlF"); VFLOAT_SPECI.push_back("hle"); 
  VFLOAT_SPECI.push_back("hlE"); VFLOAT_SPECI.push_back("hlg"); VFLOAT_SPECI.push_back("hlG"); 
  VFLOAT_SPECI.push_back("hla"); VFLOAT_SPECI.push_back("hlA");

  VDOUBLE_SPECI.push_back("lf"); VDOUBLE_SPECI.push_back("lF"); VDOUBLE_SPECI.push_back("le"); 
  VDOUBLE_SPECI.push_back("lE"); VDOUBLE_SPECI.push_back("lg"); VDOUBLE_SPECI.push_back("lG"); 
  VDOUBLE_SPECI.push_back("la"); VDOUBLE_SPECI.push_back("lA");  
}

int find_end_of_format(const char * format, int start) {
  int found;
  for( int i = start + 1; format[i] !='\0'; i++ ) {
    found = CONV_SPECI.find(format[i]);
    if( found != string::npos ) {
      return i;
    }
  }
  return -1;
}

char * get_sub_string(const char* string, int start, int end) {
  char *subString = (char *)malloc(64);
  int s = 0;

  for( int i = start; i <= end; i++, s++ ) {
    subString[s] = string[i];
  }
  subString[s] = '\0';
  return subString;
}

char * get_sub_string(const char* string, int start) {
  char *subString = (char *)malloc(64);
  int s = 0;
  int end = strlen(string)-1;

  return get_sub_string(string, start, end);
}


bool has_string(vector<string>& vector, char * string) {
  for( int i = 0; i < vector.size(); i++ ) {
    if( !(vector.at(i).compare(string)) ) {
      return true;
    }
  } 
  return false;
}

int get_vector_type(char * format, vector<string>& VCHAR_SPECI, vector<string>& VINT_SPECI, std::vector<string>& VSHORT_SPECI, std::vector<string> &VLONG_SPECI, std::vector<string> &VFLOAT_SPECI, std::vector<string> &VDOUBLE_SPECI)  {
  int type = -1;

  if( has_string(VINT_SPECI, format) ) {
    type = VINT;
  } else if( has_string(VCHAR_SPECI, format) ) {
    type = VCHAR;
  } else if( has_string(VSHORT_SPECI, format) ) {
    type = VSHORT;
  } else if( has_string(VLONG_SPECI, format) ) {
    type = VLONG;
  } else if( has_string(VFLOAT_SPECI, format) ) {
    type = VFLOAT;
  } else if( has_string(VDOUBLE_SPECI, format) ) {
    type = VDOUBLE;
  }
  return type;
}



void delete_string(char* string, int start, int n) {
  int i;

  for( i = start; string[i+n]!='\0'; i++ ) {
    string[i] = string[i+n];
  }
  string[i] = '\0';
}

int getDim(char* subFormat, int start) {
  int i, j, dim = 0;
  int s = start;
  bool find = false;
  const char *_dim;
  for( i = 0; i < N_DIM; i++ ) {
    _dim = DIM_FORMAT[i]; 
    int len = strlen(_dim);
    for( j = 0, s = start; j < len; j++, s++ ) {
      if( subFormat[s] != _dim[j] )
        find = false;
      else 
        find = true;
    }

    if( find ) {
      dim = atoi(_dim);
      if( dim == 16 ) 
        delete_string(subFormat, start-1, 3);
      else 
        delete_string(subFormat, start-1, 2);
      return dim;
    }
  }
  return 4;
}

bool is_vector(char* subFormat, int *type, int *dim, vector<string>& VCHAR_SPECI, vector<string>& VINT_SPECI, std::vector<string>& VSHORT_SPECI, std::vector<string> &VLONG_SPECI, std::vector<string> &VFLOAT_SPECI, std::vector<string> &VDOUBLE_SPECI) {
  char *vTypeFormat;
  
  for( int i = 0; subFormat[i] != '\0'; i++ ) {
     
    if( subFormat[i] == 'v' ) {

      // get vector type
      *dim = getDim(subFormat, i+1);

      vTypeFormat = get_sub_string(subFormat, i);
      *type = get_vector_type(vTypeFormat, VCHAR_SPECI, VINT_SPECI, VSHORT_SPECI, VLONG_SPECI, VFLOAT_SPECI, VDOUBLE_SPECI );
      free(vTypeFormat);

      if( *type == -1 ) {
        break;
      } else if( *type == VINT || *type == VFLOAT ) {
        delete_string(subFormat, i, 2);
      }

      return true;
    }
  }
  return false;
}

bool is_double(char spec) {
  int found;
  found = DOUB_SPECI.find(spec);
  if( found != string::npos ) {
      return true;
  }
  return false;
}

bool is_int(char spec) {
  int found;
  found = INT_SPECI.find(spec);
  if( found != string::npos ) {
      return true;
  }
  return false;
}

// vint
int print_vector(char* subFormat, int16 val) {
  int d = 0;
  for( d = 0; d < 15; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, int8 val) {
  int d = 0;
  for( d = 0; d < 7; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, int4 val) {
  int d = 0;
  for( d = 0; d < 3; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, int3 val) {
  int d = 0;
  for( d = 0; d < 2; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, int2 val) {
  printf(subFormat, val[0]);
  printf(",");
  return printf(subFormat, val[1]);
}
// vchar
int print_vector(char* subFormat, char16 val) {
  int d = 0;
  for( d = 0; d < 15; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, char8 val) {
  int d = 0;
  for( d = 0; d < 7; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, char4 val) {
  int d = 0;
  for( d = 0; d < 3; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, char3 val) {
  int d = 0;
  for( d = 0; d < 2; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, char2 val) {
  printf(subFormat, val[0]);
  printf(",");
  return printf(subFormat, val[1]);
}
// vlong
int print_vector(char* subFormat, long16 val) {
  int d = 0;
  for( d = 0; d < 15; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, long8 val) {
  int d = 0;
  for( d = 0; d < 7; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, long4 val) {
  int d = 0;
  for( d = 0; d < 3; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, long3 val) {
  int d = 0;
  for( d = 0; d < 2; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, long2 val) {
  printf(subFormat, val[0]);
  printf(",");
  return printf(subFormat, val[1]);
}
// vshort
int print_vector(char* subFormat, short16 val) {
  int d = 0;
  for( d = 0; d < 15; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, short8 val) {
  int d = 0;
  for( d = 0; d < 7; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, short4 val) {
  int d = 0;
  for( d = 0; d < 3; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, short3 val) {
  int d = 0;
  for( d = 0; d < 2; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, short2 val) {
  printf(subFormat, val[0]);
  printf(",");
  return printf(subFormat, val[1]);
}
// vfloat
int print_vector(char* subFormat, float16 val) {
  int d = 0;
  for( d = 0; d < 15; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, float8 val) {
  int d = 0;
  for( d = 0; d < 7; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, float4 val) {
  int d = 0;
  for( d = 0; d < 3; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, float3 val) {
  int d = 0;
  for( d = 0; d < 2; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, float2 val) {
  printf(subFormat, val[0]);
  printf(",");
  return printf(subFormat, val[1]);
}
// vdouble
int print_vector(char* subFormat, double16 val) {
  int d = 0;
  for( d = 0; d < 15; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, double8 val) {
  int d = 0;
  for( d = 0; d < 7; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, double4 val) {
  int d = 0;
  for( d = 0; d < 3; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, double3 val) {
  int d = 0;
  for( d = 0; d < 2; d++ ) {
    printf(subFormat, val[d]);
    printf(",");
  }
  return printf(subFormat, val[d]);
}
int print_vector(char* subFormat, double2 val) {
  printf(subFormat, val[0]);
  printf(",");
  return printf(subFormat, val[1]);
}


Print::Print(char* _format) {
  format = _format;
  inx = 0;
}


Print& Print::operator,(float arg) {
  double* _arg = (double *)malloc(sizeof(double));
  *_arg = (double)arg;
  arglist.push_back((void *)_arg);
  return *this;
}

Print& Print::operator,(int arg) {
  llong* _arg = (llong *)malloc(sizeof(llong));
  *_arg = (llong)arg;
  arglist.push_back((void *)_arg);
  return *this;
}

Print& Print::operator,(short arg) {
  llong* _arg = (llong *)malloc(sizeof(llong));
  *_arg = (llong)arg;
  arglist.push_back((void *)_arg);
  return *this;
}

Print& Print::operator,(char arg) {
  llong* _arg = (llong *)malloc(sizeof(llong));
  *_arg = (llong)arg;
  arglist.push_back((void *)_arg);
  return *this;
}

void* Print::get_arg() const {
  return arglist[inx++];
}

int Print::print() const {
  int ret;
  int start, end;
  char *subFormat;
  int type, dim;

  vector<string> VCHAR_SPECI;
  vector<string> VSHORT_SPECI;
  vector<string> VINT_SPECI;
  vector<string> VLONG_SPECI;
  vector<string> VFLOAT_SPECI;
  vector<string> VDOUBLE_SPECI;
  
  init_speci(VCHAR_SPECI, VINT_SPECI, VSHORT_SPECI, VLONG_SPECI, VFLOAT_SPECI, VDOUBLE_SPECI); 

  for( int i = 0; format[i] != '\0'; i++ ) {
    // if format is detected 
    if( format[i] == '%' ) 
    {
      start = i;
      end = find_end_of_format(format, start);

      // if undefined format
      if( end == -1 )
        return -1;

      // printf arg with sub format
      subFormat = get_sub_string(format, start, end);
      // 1) vector

      if( is_vector(subFormat, &type, &dim, VCHAR_SPECI, VINT_SPECI, VSHORT_SPECI, VLONG_SPECI, VFLOAT_SPECI, VDOUBLE_SPECI) ) {
        if( type == -1 ) 
          ret = -1; 
        if( type == VINT ) {
          if( dim == 16 ) {
            int16 *v16int = (int16 *)get_arg();
            ret = print_vector(subFormat, *v16int);
          } else if( dim == 8 ) {
            int8 *v8int = (int8 *)get_arg();
            ret = print_vector(subFormat, *v8int);
          } else if( dim == 4 ) {
            int4 *v4int = (int4 *)get_arg();
            ret = print_vector(subFormat, *v4int);
          } else if( dim == 3 ) {
            int3 *v3int = (int3 *)get_arg();
            ret = print_vector(subFormat, *v3int);
          } else if( dim == 2 ) {
            int2 *v2int = (int2 *)get_arg();
            ret = print_vector(subFormat, *v2int);
          }
        } else if( type == VCHAR ) {
          if( dim == 16 ) {
            char16 *v16char = (char16 *)get_arg();
            ret = print_vector(subFormat, *v16char);
          } else if( dim == 8 ) {
            char8 *v8char = (char8 *)get_arg();
            ret = print_vector(subFormat, *v8char);
          } else if( dim == 4 ) {
            char4 *v4char = (char4 *)get_arg();
            ret = print_vector(subFormat, *v4char);
          } else if( dim == 3 ) {
            char3 *v3char = (char3 *)get_arg();
            ret = print_vector(subFormat, *v3char);
          } else if( dim == 2 ) {
            char2 *v2char = (char2 *)get_arg();
            ret = print_vector(subFormat, *v2char);
          }
        } else if( type == VLONG ) {
          if( dim == 16 ) {
            long16 *v16long = (long16 *)get_arg();
            ret = print_vector(subFormat, *v16long);
          } else if( dim == 8 ) {
            long8 *v8long = (long8 *)get_arg();
            ret = print_vector(subFormat, *v8long);
          } else if( dim == 4 ) {
            long4 *v4long = (long4 *)get_arg();
            ret = print_vector(subFormat, *v4long);
          } else if( dim == 3 ) {
            long3 *v3long = (long3 *)get_arg();
            ret = print_vector(subFormat, *v3long);
          } else if( dim == 2 ) {
            long2 *v2long = (long2 *)get_arg();
            ret = print_vector(subFormat, *v2long);
          }
        } else if( type == VSHORT ) {
          if( dim == 16 ) {
            short16 *v16short = (short16 *)get_arg();
            ret = print_vector(subFormat, *v16short);
          } else if( dim == 8 ) {
            short8 *v8short = (short8 *)get_arg();
            ret = print_vector(subFormat, *v8short);
          } else if( dim == 4 ) {
            short4 *v4short = (short4 *)get_arg();
            ret = print_vector(subFormat, *v4short);
          } else if( dim == 3 ) {
            short3 *v3short = (short3 *)get_arg();
            ret = print_vector(subFormat, *v3short);
          } else if( dim == 2 ) {
            short2 *v2short = (short2 *)get_arg();
            ret = print_vector(subFormat, *v2short);
          }
        } else if( type == VFLOAT ) {
          if( dim == 16 ) {
            float16 *v16float = (float16 *)get_arg();
            ret = print_vector(subFormat, *v16float);
          } else if( dim == 8 ) {
            float8 *v8float = (float8 *)get_arg();
            ret = print_vector(subFormat, *v8float);
          } else if( dim == 4 ) {
            float4 *v4float = (float4 *)get_arg();
            ret = print_vector(subFormat, *v4float);
          } else if( dim == 3 ) {
            float3 *v3float = (float3 *)get_arg();
            ret = print_vector(subFormat, *v3float);
          } else if( dim == 2 ) {
            float2 *v2float = (float2 *)get_arg();
            ret = print_vector(subFormat, *v2float);
          }
        } else if( type == VDOUBLE ) {
          if( dim == 16 ) {
            double16 *v16double = (double16 *)get_arg();
            ret = print_vector(subFormat, *v16double);
          } else if( dim == 8 ) {
            double8 *v8double = (double8 *)get_arg();
            ret = print_vector(subFormat, *v8double);
          } else if( dim == 4 ) {
            double4 *v4double = (double4 *)get_arg();
            ret = print_vector(subFormat, *v4double);
          } else if( dim == 3 ) {
            double3 *v3double = (double3 *)get_arg();
            ret = print_vector(subFormat, *v3double);
          } else if( dim == 2 ) {
            double2 *v2double = (double2 *)get_arg();
            ret = print_vector(subFormat, *v2double);
          }
        }
      } else if( is_int(format[end]) ) {
        ret = printf(subFormat, *(llong *)get_arg());
      } else if( is_double(format[end]) ) {
        double *dval = (double *)get_arg();
        if( isnan(*dval)  )
          ret = printf("nan");
        else
          ret = printf(subFormat, *dval);
      } else if( format[end] == 's' ) {
        ret = printf(subFormat, *(char **)get_arg());
      } else if( format[end] == 'p' ) {
        void **lval = (void **)get_arg();
        ret = printf(subFormat, *lval);
      } else if( format[end] == '%' ) {
        ret = printf("%%");
      }

      // clean resource
      free(subFormat);

      // skip the format 
      i = end;
      continue;
    }

    putchar(format[i]);
  }
  if( ret >= 0 ) 
    ret = 0;
  return ret;
}

Print::~Print() {
  int size = arglist.size();
  for( int i = 0; i < size; i++ ) {
    void * arg = arglist[i];
    free(arg);
  }
}

int snu_printf(const Print& p) {
  return p.print();
}

