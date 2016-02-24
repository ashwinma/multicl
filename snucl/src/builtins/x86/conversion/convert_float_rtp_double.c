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
#include "convert_util.h"

float convert_float_rtp(double x) {
	float ret;              
	__CHANGE_FP_MODE(_CL_FPMODE_RTP); 
	ret = (float)x;         
	__RESTORE_FP_MODE(_CL_FPMODE_RTP);
	return ret;            
}

float2 convert_float2_rtp(double2 x) {
	float2 ret;                            
	__CHANGE_FP_MODE(_CL_FPMODE_RTP);               
	ret = (float2){(float)x[0], (float)x[1]};
	__RESTORE_FP_MODE(_CL_FPMODE_RTP);              
	return ret;                          
}

float3 convert_float3_rtp(double3 x) {
	float3 ret;                                        
	__CHANGE_FP_MODE(_CL_FPMODE_RTP);                           
	ret = (float3){(float)x[0], (float)x[1], (float)x[2]};
	__RESTORE_FP_MODE(_CL_FPMODE_RTP);                          
	return ret;                                      
}

float4 convert_float4_rtp(double4 x) {
	float4 ret;                                                    
	__CHANGE_FP_MODE(_CL_FPMODE_RTP);                                       
	ret = (float4){(float)x[0], (float)x[1], (float)x[2], (float)x[3]};
	__RESTORE_FP_MODE(_CL_FPMODE_RTP);                                      
	return ret;                                                  
}

float8 convert_float8_rtp(double8 x) {
	float8 ret;                                                    
	__CHANGE_FP_MODE(_CL_FPMODE_RTP);                                       
	ret = (float8){(float)x[0], (float)x[1], (float)x[2], (float)x[3], 
	             (float)x[4], (float)x[5], (float)x[6], (float)x[7]};
	__RESTORE_FP_MODE(_CL_FPMODE_RTP);                                      
	return ret;                                                  
}

float16 convert_float16_rtp(double16 x) {
	float16 ret;                                                        
	__CHANGE_FP_MODE(_CL_FPMODE_RTP);                                           
	ret = (float16){(float)x[0], (float)x[1], (float)x[2], (float)x[3],     
	             (float)x[4], (float)x[5], (float)x[6], (float)x[7],     
	             (float)x[8], (float)x[9], (float)x[10], (float)x[11],   
	             (float)x[12], (float)x[13], (float)x[14], (float)x[15]};
	__RESTORE_FP_MODE(_CL_FPMODE_RTP);                                          
	return ret;                                                      
}

