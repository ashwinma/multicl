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

double convert_double_rte(llong x) {
	double ret;                                 
	__CHANGE_FP_MODE(_CL_FPMODE_RTE);                    
	ret = __SAFE_INT_ZERO_TO_FP_ZERO(x, double);
	__RESTORE_FP_MODE(_CL_FPMODE_RTE);                   
	return ret;                               
}

double2 convert_double2_rte(long2 x) {
	double2 ret;                                               
	__CHANGE_FP_MODE(_CL_FPMODE_RTE);                                  
	for (int i = 0; i < 2; i++)                             
		ret[i] = __SAFE_INT_ZERO_TO_FP_ZERO(x[i], double);
	__RESTORE_FP_MODE(_CL_FPMODE_RTE);                                 
	return ret;                                             
}

double3 convert_double3_rte(long3 x) {
	double3 ret;                                               
	__CHANGE_FP_MODE(_CL_FPMODE_RTE);                                  
	for (int i = 0; i < 3; i++)                             
		ret[i] = __SAFE_INT_ZERO_TO_FP_ZERO(x[i], double);
	__RESTORE_FP_MODE(_CL_FPMODE_RTE);                                 
	return ret;                                             
}

double4 convert_double4_rte(long4 x) {
	double4 ret;                                               
	__CHANGE_FP_MODE(_CL_FPMODE_RTE);                                  
	for (int i = 0; i < 4; i++)                             
		ret[i] = __SAFE_INT_ZERO_TO_FP_ZERO(x[i], double);
	__RESTORE_FP_MODE(_CL_FPMODE_RTE);                                 
	return ret;                                             
}

double8 convert_double8_rte(long8 x) {
	double8 ret;                                               
	__CHANGE_FP_MODE(_CL_FPMODE_RTE);                                  
	for (int i = 0; i < 8; i++)                             
		ret[i] = __SAFE_INT_ZERO_TO_FP_ZERO(x[i], double);
	__RESTORE_FP_MODE(_CL_FPMODE_RTE);                                 
	return ret;                                             
}

double16 convert_double16_rte(long16 x) {
	double16 ret;                                               
	__CHANGE_FP_MODE(_CL_FPMODE_RTE);                                  
	for (int i = 0; i < 16; i++)                            
		ret[i] = __SAFE_INT_ZERO_TO_FP_ZERO(x[i], double);
	__RESTORE_FP_MODE(_CL_FPMODE_RTE);                                 
	return ret;                                             
}

