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

#define MSB_CHAR   0x8
#define MSB_SHORT   0x80
#define MSB_INT     0x8000
#define MSB_lONG   0x80000000L     

//schar
int any(schar x){
  schar tmp =( x>>4 ) & MSB_CHAR;
  if( tmp == MSB_CHAR )
   	return 1;
  else 
	return 0;  
}



int any(char2 x){
  char2 tmp = ( x>>4 ) & MSB_CHAR;
   if( tmp[0] == MSB_CHAR  || tmp[1]==MSB_CHAR)
        return 1;
  else
        return 0;
}

int any(char3 x){
  char3 tmp = ( x>>4 ) & MSB_CHAR;

   if( tmp[0] == MSB_CHAR  || tmp[1]==MSB_CHAR ||  tmp[2] == MSB_CHAR )
        return 1;
  else
        return 0;

}
int any(char4 x){
  char4 tmp = ( x>>4 ) & MSB_CHAR;

   if( tmp[0] == MSB_CHAR  || tmp[1]==MSB_CHAR ||  tmp[2] == MSB_CHAR  || tmp[3]==MSB_CHAR )
        return 1;
  else
        return 0;

}
int any(char8 x){
  char8 tmp = ( x>>4 ) & MSB_CHAR;

   if( tmp[0] == MSB_CHAR  || tmp[1]==MSB_CHAR ||  tmp[2] == MSB_CHAR  || tmp[3]==MSB_CHAR || 
		 tmp[4] == MSB_CHAR  || tmp[5]==MSB_CHAR ||  tmp[6] == MSB_CHAR  || tmp[7]==MSB_CHAR )
        return 1;
  else
        return 0;


}
int any(char16 x){
  char16 tmp =( x>>4 ) & MSB_CHAR;
   if( tmp[0] == MSB_CHAR  || tmp[1]==MSB_CHAR ||  tmp[2] == MSB_CHAR  || tmp[3]==MSB_CHAR || 
                 tmp[4] == MSB_CHAR  || tmp[5]==MSB_CHAR ||  tmp[6] == MSB_CHAR  || tmp[7]==MSB_CHAR || 
		 		 tmp[8] == MSB_CHAR  || tmp[9]==MSB_CHAR ||  tmp[10] == MSB_CHAR  || tmp[11]==MSB_CHAR || 
                 tmp[12] == MSB_CHAR  || tmp[13]==MSB_CHAR ||  tmp[14] == MSB_CHAR  || tmp[15]==MSB_CHAR 
			 )
	
        return 1;
  else
        return 0;



}

//short
int any(short x){
 
	short tmp =(x>>8) & MSB_SHORT;
  if( tmp == MSB_SHORT )
   	return 1;
  else 
	return 0;  
}
int any(short2 x){
  short2 tmp = (x>>8) & MSB_SHORT;
   if( tmp[0] == MSB_SHORT  || tmp[1]==MSB_SHORT)
        return 1;
  else
        return 0;
}
int any(short3 x){
  short3 tmp = (x>>8) & MSB_SHORT;

   if( tmp[0] == MSB_SHORT  || tmp[1]==MSB_SHORT ||  tmp[2] == MSB_SHORT )
        return 1;
  else
        return 0;
}
int any(short4 x){
  short4 tmp = (x>>8) & MSB_SHORT;

   if( tmp[0] == MSB_SHORT  || tmp[1]==MSB_SHORT ||  tmp[2] == MSB_SHORT  || tmp[3]==MSB_SHORT )
        return 1;
  else
        return 0;
}
int any(short8 x){
  short8 tmp = (x>>8) & MSB_SHORT;

   if( tmp[0] == MSB_SHORT  || tmp[1]==MSB_SHORT ||  tmp[2] == MSB_SHORT  || tmp[3]==MSB_SHORT || 
		 tmp[4] == MSB_SHORT  || tmp[5]==MSB_SHORT ||  tmp[6] == MSB_SHORT  || tmp[7]==MSB_SHORT )
        return 1;
  else
        return 0;
}
int any(short16 x){
  short16 tmp = (x>>8) & MSB_SHORT;
   if( tmp[0] == MSB_SHORT  || tmp[1]==MSB_SHORT ||  tmp[2] == MSB_SHORT  || tmp[3]==MSB_SHORT || 
                 tmp[4] == MSB_SHORT  || tmp[5]==MSB_SHORT ||  tmp[6] == MSB_SHORT  || tmp[7]==MSB_SHORT || 
		 		 tmp[8] == MSB_SHORT  || tmp[9]==MSB_SHORT ||  tmp[10] == MSB_SHORT  || tmp[11]==MSB_SHORT || 
                 tmp[12] == MSB_SHORT  || tmp[13]==MSB_SHORT ||  tmp[14] == MSB_SHORT  || tmp[15]==MSB_SHORT 
			 )
	
        return 1;
  else
        return 0;
}

//int
int any(int x){
 
	int  tmp =(x>>16) & MSB_INT;
  if( tmp == MSB_INT )
   	return 1;
  else 
	return 0;  
}

int any(int2 x){
  int2 tmp = (x>>16) & MSB_INT;
   if( tmp[0] == MSB_INT  || tmp[1]==MSB_INT)
        return 1;
  else
        return 0;
}
int any(int3 x){
  int3 tmp = (x>>16) & MSB_INT;

   if( tmp[0] == MSB_INT  || tmp[1]==MSB_INT ||  tmp[2] == MSB_INT  )
        return 1;
  else
        return 0;
}
int any(int4 x){
  int4 tmp = (x>>16) & MSB_INT;

   if( tmp[0] == MSB_INT  || tmp[1]==MSB_INT ||  tmp[2] == MSB_INT  || tmp[3]==MSB_INT )
        return 1;
  else
        return 0;
}
int any(int8 x){
  int8 tmp = (x>>16) & MSB_INT;

   if( tmp[0] == MSB_INT  || tmp[1]==MSB_INT ||  tmp[2] == MSB_INT  || tmp[3]==MSB_INT || 
		 tmp[4] == MSB_INT  || tmp[5]==MSB_INT ||  tmp[6] == MSB_INT  || tmp[7]==MSB_INT )
        return 1;
  else
        return 0;
}
int any(int16 x){
  int16 tmp = (x>>16) & MSB_INT;
   if( tmp[0] == MSB_INT  || tmp[1]==MSB_INT ||  tmp[2] == MSB_INT  || tmp[3]==MSB_INT || 
                 tmp[4] == MSB_INT  || tmp[5]==MSB_INT ||  tmp[6] == MSB_INT  || tmp[7]==MSB_INT || 
		 		 tmp[8] == MSB_INT  || tmp[9]==MSB_INT ||  tmp[10] == MSB_INT  || tmp[11]==MSB_INT || 
                 tmp[12] == MSB_INT  || tmp[13]==MSB_INT ||  tmp[14] == MSB_INT  || tmp[15]==MSB_INT 
			 )
	
        return 1;
  else
        return 0;
}


//llong
int any(llong x){
 	llong  tmp =(x>>32) & MSB_lONG;
  if( tmp == MSB_lONG )
   	return 1;
  else 
	return 0;  
}
int any(long2 x){
  long2 tmp = (x >> 32) & MSB_lONG;
  if( tmp[0] == MSB_lONG  || tmp[1]==MSB_lONG)
        return 1;
  else
	  return 0;
}
int any(long3 x){
  long3 tmp = (x >> 32) & MSB_lONG;
 if( tmp[0] == MSB_lONG  || tmp[1]==MSB_lONG ||  tmp[2] == MSB_lONG )
        return 1;
  else
        return 0;
}

int any(long4 x){
  long4 tmp = (x >> 32) & MSB_lONG;
 if( tmp[0] == MSB_lONG  || tmp[1]==MSB_lONG ||  tmp[2] == MSB_lONG  || tmp[3]==MSB_lONG )
        return 1;
  else
        return 0;
}
int any(long8 x){
  long8 tmp = (x >> 32) & MSB_lONG;
   if( tmp[0] == MSB_lONG  || tmp[1]==MSB_lONG ||  tmp[2] == MSB_lONG  || tmp[3]==MSB_lONG || 
		 tmp[4] == MSB_lONG  || tmp[5]==MSB_lONG ||  tmp[6] == MSB_lONG  || tmp[7]==MSB_lONG )
        return 1;
  else
        return 0;
}
int any(long16 x){
  long16 tmp = (x >> 32) & MSB_lONG;
   if( tmp[0] == MSB_lONG  || tmp[1]==MSB_lONG ||  tmp[2] == MSB_lONG  || tmp[3]==MSB_lONG || 
                 tmp[4] == MSB_lONG  || tmp[5]==MSB_lONG ||  tmp[6] == MSB_lONG  || tmp[7]==MSB_lONG || 
		 		 tmp[8] == MSB_lONG  || tmp[9]==MSB_lONG ||  tmp[10] == MSB_lONG  || tmp[11]==MSB_lONG || 
                 tmp[12] == MSB_lONG  || tmp[13]==MSB_lONG ||  tmp[14] == MSB_lONG  || tmp[15]==MSB_lONG 
			 )
	
        return 1;
  else
        return 0;
}

